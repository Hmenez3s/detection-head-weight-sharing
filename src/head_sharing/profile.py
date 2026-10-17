"""Reproducible storage, executed ConvMAC, latency, and peak-memory profiling."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

CONV_MAC_EXCLUSIONS = (
    "Non-Conv operations (normalization, activation, pooling, concatenation, decode, top-k, and copies) are excluded;",
    "one MAC means one multiply-accumulate (not two FLOPs). MACs are not latency or a VRAM-savings claim.",
)


def _unique_tensors(model):
    """Return unique parameter and persistent-buffer tensors by physical storage."""
    seen = set()
    for tensor in (*tuple(model.parameters()), *tuple(model.buffers())):
        storage = tensor.untyped_storage()
        key = (tensor.device.type, tensor.device.index, storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            yield tensor


def storage_inventory(model, checkpoint_path=None):
    """Count unique parameter elements/storage bytes; use stat for serialized size."""
    scope = getattr(model, "detect", model)
    params = {}
    for parameter in scope.parameters():
        params.setdefault(id(parameter), parameter)
    storages = set()
    storage_bytes = 0
    for tensor in _unique_tensors(scope):
        storage = tensor.untyped_storage()
        key = (tensor.device.type, tensor.device.index, storage.data_ptr(), storage.nbytes())
        if key not in storages:
            storages.add(key)
            storage_bytes += storage.nbytes()
    return {
        "unique_parameters": sum(p.numel() for p in params.values()),
        "unique_parameter_storage_bytes": sum(p.untyped_storage().nbytes() for p in params.values()),
        "unique_parameter_and_buffer_storage_bytes": storage_bytes,
        "serialized_file_bytes": Path(checkpoint_path).stat().st_size if checkpoint_path is not None else None,
    }


def measure_conv_macs(model, invoke):
    """Hook each unique Conv2d once and charge all executed calls, including aliases."""
    import torch
    from torch import nn

    calls = {"count": 0, "macs": 0}
    hooks = []
    convs = {id(module): module for module in model.modules() if isinstance(module, nn.Conv2d)}

    def count(module, inputs, output):
        x = inputs[0]
        if not isinstance(output, torch.Tensor) or x.ndim != 4 or output.ndim != 4:
            raise ValueError("ConvMAC hook requires NCHW Conv2d input/output")
        kh, kw = module.kernel_size
        calls["count"] += 1
        calls["macs"] += (output.shape[0] * output.shape[2] * output.shape[3] * output.shape[1]
                          * (x.shape[1] // module.groups) * kh * kw)

    for module in convs.values():
        hooks.append(module.register_forward_hook(count))
    try:
        invoke()
    finally:
        for hook in hooks:
            hook.remove()
    return {"unique_conv_modules": len(convs), "conv_invocations": calls["count"], "conv_macs": calls["macs"],
            "mac_convention": "B*Hout*Wout*Cout*(Cin/groups)*Kh*Kw; one MAC is one multiply-accumulate",
            "excluded_ops": list(CONV_MAC_EXCLUSIONS)}


def _sync(torch, device):
    torch.cuda.synchronize(device)


def _training_batch_size(requested, recipe_batch_size):
    if requested not in (1, recipe_batch_size):
        raise ValueError("training probe batch must be 1 or the frozen recipe batch_size")
    return requested


def _load_training_cache(cache_path, recipe, progress_path):
    from .cache import _sha, load_cache
    evidence=recipe["cache_evidence"].get("few",{})
    if not isinstance(evidence,dict) or _sha(Path(cache_path)/"metadata.json",progress_path=progress_path) != evidence.get("metadata_file_sha256"):
        raise ValueError("frozen few cache metadata FILE hash mismatch")
    return load_cache(cache_path,expected_producer=recipe["producer"],progress_path=progress_path)


def _fixed_probe_batch(source_batches, batch_size):
    """Use one full seeded batch for both steps; never label a partial tail full."""
    for selected in source_batches(0):
        if selected[0][0].shape[0] == batch_size:
            def batches(epoch):
                yield selected
            batches.steps = 1
            return batches
    raise ValueError("cache has no complete batch for the training probe")


def measure_latency(model, images, *, region, warmup=5, repetitions=20):
    """Synchronized matched FP32 timings. Full includes common-Q; head uses FP32 Q inputs."""
    import torch
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be nonnegative and repetitions positive")
    model.eval()
    if region == "full":
        call = lambda: model.forward_images(images)
        boundary = "FP32 frozen trunk -> FP16 roundtrip -> FP32 head/decode; Q included"
    elif region == "head":
        features = model.extract_features(images)
        call = lambda: model.forward_features(features)
        boundary = "FP32 Q feature tensors supplied; forward_features repeats FP32->FP16->FP32 boundary"
    else:
        raise ValueError("region must be full or head")
    with torch.inference_mode():
        for _ in range(warmup):
            call()
        if images.device.type == "cuda":
            _sync(torch, images.device)
        samples = []
        for _ in range(repetitions):
            if images.device.type == "cuda":
                _sync(torch, images.device)
            start = time.perf_counter()
            call()
            if images.device.type == "cuda":
                _sync(torch, images.device)
            samples.append((time.perf_counter() - start) * 1000)
    return {"region": region, "milliseconds": samples, "mean_ms": sum(samples) / len(samples),
            "warmup": warmup, "repetitions": repetitions, "dtype": "float32", "batch": images.shape[0],
            "resolution": list(images.shape[-2:]), "quantization_boundary": boundary,
            "timing_label": "CUDA synchronized" if images.device.type == "cuda" else "CPU non-study smoke only"}


def measure_cuda_peak(torch, device, operation):
    """Measure one isolated operation after recording baseline and resetting peak stats."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA peak measurement requires the explicitly selected CUDA device")
    torch.cuda.synchronize(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    operation()
    torch.cuda.synchronize(device)
    return {"baseline_allocated_bytes": baseline_allocated, "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--measurement", choices=("inference", "training"), default="inference")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trunk-artifact", required=True)
    parser.add_argument("--synthetic", action="store_true", help="CPU-only synthetic fixture")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--uuid"); parser.add_argument("--owned-root"); parser.add_argument("--ledger"); parser.add_argument("--reservation")
    parser.add_argument("--cache", help="few-cohort cache; required for training")
    parser.add_argument("--seed", type=int, default=11); parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1); parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20); parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.imgsz <= 0 or args.imgsz % 32 or args.batch_size < 1 or args.warmup < 0 or args.repetitions < 1:
        parser.error("invalid image size, batch, warmup, or repetitions")
    if args.measurement == "training" and (not args.cache or args.epochs < 3):
        parser.error("training requires --cache and epochs >= 3")
    if args.device == "cpu" and not args.synthetic: parser.error("CPU requires explicit synthetic fixture mode")
    if args.device == "cuda" and (args.synthetic or not all((args.uuid,args.owned_root,args.ledger,args.reservation))):
        parser.error("CUDA requires owned-root, ledger, reservation, UUID and non-synthetic checkpoint")
    ledger = progress_path = job_root = None
    if args.device == "cuda":
        from .runtime import AllocationLedger, _has_symlink_component, canonical_gpu_uuid
        from .supervisor import child_entry
        root, ledger_path = Path(args.owned_root), Path(args.ledger)
        if (_has_symlink_component(root) or not root.is_dir() or root.absolute()!=root.resolve(strict=True)
                or not (root/"STUDY_OWNER.txt").is_file() or (root/"STUDY_OWNER.txt").is_symlink()):
            raise RuntimeError("canonical marker-owned root required")
        if (_has_symlink_component(ledger_path) or not ledger_path.resolve(strict=True).is_relative_to(root.resolve(strict=True))
                or any(_has_symlink_component(ledger_path/n) for n in ("LEDGER_OWNER","ledger.json"))):
            raise RuntimeError("ledger must be link-free beneath owned root")
        ledger=AllocationLedger(ledger_path)
        if canonical_gpu_uuid(args.uuid)!=args.uuid: raise RuntimeError("device must be canonical UUID")
        child_entry(ledger,args.reservation,args.uuid); job=ledger.get_job(args.reservation)
        job_root,progress_path=Path(job["cwd"]),Path(job["progress_file"])
        if job["uuid"]!=args.uuid or _has_symlink_component(job_root) or not job_root.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
            raise RuntimeError("reservation identity or job confinement failure")
        if (_has_symlink_component(progress_path) or progress_path.absolute() != progress_path or
                progress_path.parent.resolve(strict=True) != job_root.resolve(strict=True) or
                (progress_path.exists() and not progress_path.is_file())):
            raise RuntimeError("supervisor progress path is not job-owned")
        if os.environ.get("CUDA_VISIBLE_DEVICES")!=args.uuid: raise RuntimeError("assigned UUID visibility mismatch")
        for key in ("YOLO_CONFIG_DIR","TORCH_HOME","MPLCONFIGDIR","TMPDIR","CUDA_CACHE_PATH","TORCH_EXTENSIONS_DIR","TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR"):
            value=os.environ.get(key)
            if not value or _has_symlink_component(Path(value)) or not Path(value).resolve(strict=True).is_relative_to(job_root.resolve(strict=True)):
                raise RuntimeError("native config/cache escaped supervisor job")
            if key=="YOLO_CONFIG_DIR" and not (Path(value)/"Ultralytics").is_dir(): raise RuntimeError("precreated Ultralytics config root required")
        for key in ("weights","config","trunk_artifact","protocol","output","cache"):
            value=getattr(args,key,None)
            if value is None: continue
            path=Path(value).absolute()
            if _has_symlink_component(path) or not path.resolve(strict=False).is_relative_to(root.resolve(strict=True)):
                raise RuntimeError("declared path escaped owned root: "+key)
            if key=="output" and (not path.resolve(strict=False).is_relative_to(job_root.resolve(strict=True)) or path.exists()):
                raise RuntimeError("output must be fresh beneath supervisor job")
        os.environ["NVIDIA_TF32_OVERRIDE"]="0"; os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    else:
        if args.output.exists(): parser.error("output must be fresh")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from .protocol import verify_frozen
    frozen=verify_frozen(args.protocol,synthetic=args.synthetic)
    recipe=frozen["recipe"]
    if recipe.get("imgsz")!=args.imgsz or recipe.get("quantization")!="fp32_fp16_fp32_v1" or not recipe.get("producer"):
        raise ValueError("profile settings or producer metadata differ from frozen recipe")
    if args.measurement == "training":
        _training_batch_size(args.batch_size,recipe["batch_size"])
    import hashlib
    import torch
    from .checkpoint import load_checkpoint
    device=torch.device("cuda" if args.device=="cuda" else "cpu")
    if device.type=="cuda":
        from .runtime import canonical_gpu_uuid
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("exactly one reserved CUDA GPU must be visible")
        actual=canonical_gpu_uuid(str(torch.cuda.get_device_properties(0).uuid))
        if actual!=args.uuid: raise RuntimeError("actual CUDA UUID differs from reservation")
        ledger.validate_device(args.reservation,actual)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=False
        torch.use_deterministic_algorithms(False)
    torch.set_num_threads(max(1,min(3,int(os.environ.get("HEAD_SHARING_CPU_BUDGET","1")))))
    files={"head_weights":args.weights,"head_config":args.config,"trunk_artifact":args.trunk_artifact}
    before={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in files.items()}
    artifact=torch.load(args.trunk_artifact,map_location="cpu",weights_only=True) if args.synthetic else args.trunk_artifact
    model=load_checkpoint(args.weights,args.config,trunk_artifact=artifact,synthetic_trunk=args.synthetic,map_location="cpu").to(device)
    if model.imgsz!=args.imgsz: raise ValueError("checkpoint resolution differs from profile")
    assigned_uuid=recipe["gpu_for_seed"].get(str(model.seed))
    if device.type=="cuda" and assigned_uuid!=args.uuid:
        raise RuntimeError("execution UUID differs from frozen seed assignment")
    sizes={"head":storage_inventory(model.detect,args.weights),"full_detector":storage_inventory(model.detector)}
    head_file_bytes=Path(args.weights).stat().st_size
    trunk_file_bytes=Path(args.trunk_artifact).stat().st_size
    result={"measurement":args.measurement,"arm":model.arm,"seed":model.seed,"imgsz":args.imgsz,
            "quantization":model.quantization,"numeric_policy":recipe["numeric_policy"],
            "model_identity":{"weights_sha256":before["head_weights"],"config_sha256":before["head_config"],
                              "trunk_sha256":recipe["trunk_sha256"],"recipe_sha256":frozen["recipe_sha256"]},
            "storage":sizes,"portable_weight_files":{"head_file_bytes":head_file_bytes,
                "common_trunk_file_bytes":trunk_file_bytes,"head_plus_trunk_file_bytes":head_file_bytes+trunk_file_bytes,
                "scope":"head and common trunk weights containers only; config/receipt bytes excluded"},
            "producer":recipe["producer"],"assigned_uuid":assigned_uuid,
            "execution_uuid":args.uuid if device.type=="cuda" else None,
            "measurement_inputs":{"batch_size":args.batch_size,"images":"zero-valued FP32 NCHW synthetic/inference tensor",
                                  "shape":[args.batch_size,3,args.imgsz,args.imgsz]}}
    if args.measurement=="inference":
        images=torch.zeros(args.batch_size,3,args.imgsz,args.imgsz,device=device,dtype=torch.float32)
        model.eval()
        with torch.inference_mode():
            common_features=model.extract_features(images)
            result["executed_conv_macs"]=measure_conv_macs(model,lambda:model.forward_images(images))
            result["head_executed_conv_macs"]=measure_conv_macs(model.detect,lambda:model.forward_features(common_features))
            del common_features  # Do not count an extra cached feature set in full inference VRAM.
            result["full_detector"]=measure_latency(model,images,region="full",warmup=args.warmup,repetitions=args.repetitions)
            result["head"]=measure_latency(model,images,region="head",warmup=args.warmup,repetitions=args.repetitions)
            result["inference_peak"]=measure_cuda_peak(torch,device,lambda:model.forward_images(images)) if device.type=="cuda" else None
    else:
        if device.type!="cuda": raise RuntimeError("native training profile is CUDA-only")
        from .train import cache_batches,train
        with _load_training_cache(args.cache,recipe,progress_path) as reader:
            metadata=reader.metadata
            if (metadata.get("cohort")!="few" or metadata.get("imgsz")!=args.imgsz or
                    metadata.get("batch_size")!=recipe["batch_size"] or metadata.get("quantization")!=recipe["quantization"] or
                    metadata.get("trunk_sha256")!=recipe["trunk_sha256"]):
                raise ValueError("few cache identity differs from frozen recipe")
            if args.seed!=model.seed:
                raise ValueError("training probe seed must match selected checkpoint seed")
            source_batches=cache_batches(reader,seed=args.seed,ids=metadata["cohort_ids"]["few"],batch_size=args.batch_size)
            batches=_fixed_probe_batch(source_batches,args.batch_size)
            warm=train(model,batches,epochs=args.epochs,seed=args.seed,device=device,lr=.01,stop_epoch=1)
            result["training_peak"]=measure_cuda_peak(torch,device,lambda:train(model,batches,epochs=args.epochs,seed=args.seed,device=device,lr=.01,start_epoch=1,optimizer=warm["optimizer"],criterion=warm["criterion"],stop_epoch=2))
        result["measurement_inputs"]={"batch_size":args.batch_size,
            "source":"frozen few cache; first full seeded epoch-0 batch, repeated in both probe steps"}
        result["training_protocol"]={"epochs":args.epochs,"cache_steps":1,"batch_size":args.batch_size,
            "input":"same complete seeded epoch-0 cache batch for warmup and measured step",
            "scope":"one warmup native loss/optimizer step plus measured second step; brief probe, not uninterrupted main-fit peak",
            "lr":.01,"optimizer":"native SGD","precision":"native BF16 head"}
    if device.type=="cuda":
        props=torch.cuda.get_device_properties(0)
        result["hardware"]={"name":props.name,"uuid":args.uuid,"torch_version":torch.__version__,"cuda_runtime_version":torch.version.cuda,"producer":recipe["producer"]}
    else: result["hardware"]={"name":"CPU synthetic smoke (not study evidence)","torch_version":torch.__version__}
    if before!={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in files.items()}: raise RuntimeError("checkpoint files changed during profiling")
    result["checkpoint_file_sha256"]=before
    result["checkpoint_files_unchanged"]=True
    if progress_path is not None:
        from .cache import _progress
        _progress(progress_path,"profile_complete",1,1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
