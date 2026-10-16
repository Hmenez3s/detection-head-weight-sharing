"""Canonical, immutable study recipe and protocol freeze checks."""
from __future__ import annotations
import argparse, hashlib, json, os, tempfile
from pathlib import Path

SEEDS=[11,22,33,44,55]
ARMS=["native","independent_k2","shared_k2"]
SELECTION={"cohort":"dev","metric":"AP50:95","tie":"earliest","training_length":"fixed_epochs"}
NUMERIC_POLICY={"trunk_dtype":"float32","feature_storage_dtype":"float16","head_input_dtype":"float32","head_train_autocast":"bfloat16","metric_head_dtype":"float32","NVIDIA_TF32_OVERRIDE":"0","CUBLAS_WORKSPACE_CONFIG":":4096:8","cuda_matmul_allow_tf32":False,"cudnn_allow_tf32":True,"cudnn_benchmark":False,"cudnn_deterministic":False,"deterministic_algorithms":False}
REQUIRED_RECIPE={"status","seed_list","arms","imgsz","batch_size","epochs","cadence","block_size","probe_size","optimizer","gains","quantization","producer","environment","trunk_sha256","split_sha256","cache_evidence","pilot_evidence","synthetic","gpu_for_seed","selection","numeric_policy"}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def validate_recipe(recipe, *, synthetic=False):
    if not isinstance(recipe,dict) or set(recipe)!=REQUIRED_RECIPE: raise ValueError("protocol recipe has unknown or missing fields")
    if recipe["status"]!="FROZEN": raise ValueError("protocol is not frozen")
    if recipe["seed_list"]!=SEEDS or recipe["arms"]!=ARMS: raise ValueError("main matrix must be exactly five paired seeds and three arms")
    for key in ("imgsz","batch_size","epochs","cadence","block_size","probe_size"):
        if type(recipe[key]) is not int or recipe[key] < 1: raise ValueError(f"invalid fixed {key}")
    geometry_ok = recipe["imgsz"] >= 64 and recipe["imgsz"] % 32 == 0 if synthetic else recipe["imgsz"] in (512,640)
    if not geometry_ok or recipe["batch_size"]<2: raise ValueError("invalid production geometry/batch")
    if recipe["block_size"] != recipe["batch_size"]: raise ValueError("physical block size must match extraction/training batch")
    if not synthetic and recipe["probe_size"] != 1000: raise ValueError("main clean probe must contain exactly 1000 images")
    if recipe["epochs"]<1 or recipe["cadence"]>recipe["epochs"]: raise ValueError("invalid fixed evaluation schedule")
    from .runtime import GPU_UUIDS
    assignment=recipe["gpu_for_seed"]
    if not isinstance(assignment,dict) or set(assignment)!={str(seed) for seed in SEEDS} or any(value not in GPU_UUIDS for value in assignment.values()):
        raise ValueError("each complete seed triplet requires one permitted physical GPU")
    if recipe["selection"] != SELECTION or recipe["numeric_policy"] != NUMERIC_POLICY:
        raise ValueError("selection or numerical policy differs from fixed study contract")
    if recipe["optimizer"]!={"name":"SGD","lr":.01,"momentum":.9,"weight_decay":.0005}: raise ValueError("optimizer proposal differs")
    if recipe["gains"]!={"box":7.5,"cls":.5,"dfl":1.5}: raise ValueError("native gains differ")
    if recipe["quantization"]!="fp32_fp16_fp32_v1": raise ValueError("common Q policy mismatch")
    for key in ("producer","environment","trunk_sha256","split_sha256","cache_evidence","pilot_evidence"):
        if recipe[key] in (None,"",{},[]): raise ValueError(f"missing feasibility/input evidence: {key}")
    if recipe["synthetic"] is not synthetic: raise ValueError("synthetic protocol labeling mismatch")
    for key in ("trunk_sha256","split_sha256"):
        value=recipe[key]
        if not isinstance(value,str) or len(value)!=64 or not set(value)<=set("0123456789abcdef"):
            raise ValueError(f"invalid artifact file digest: {key}")
    if not synthetic:
        evidence=recipe["cache_evidence"]
        if recipe["producer"] not in GPU_UUIDS or not isinstance(evidence,dict) or set(evidence)!={"few","dev","probe"}:
            raise ValueError("main cache producer/cohort evidence is incomplete")
        for row in evidence.values():
            value=row.get("metadata_file_sha256") if isinstance(row,dict) else None
            if not isinstance(value,str) or len(value)!=64 or not set(value)<=set("0123456789abcdef"):
                raise ValueError("main cache evidence requires metadata_file_sha256")
    return recipe


def freeze(recipe_path, output_path, *, synthetic=False):
    recipe=json.loads(Path(recipe_path).read_text(encoding="utf-8")); validate_recipe(recipe,synthetic=synthetic)
    output=Path(output_path)
    if output.exists() or output.is_symlink(): raise FileExistsError("immutable protocol freeze already exists")
    output.parent.mkdir(parents=True,exist_ok=True)
    payload={"format":1,"recipe":recipe,"recipe_sha256":hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(",",":")).encode()).hexdigest()}
    fd,name=tempfile.mkstemp(prefix=".freeze-",dir=output.parent)
    try:
        with os.fdopen(fd,"w") as f: json.dump(payload,f,sort_keys=True,indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.link(name,output)
    finally:
        if os.path.exists(name):os.unlink(name)
    return payload


def verify_frozen(path, *, synthetic=False):
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value,dict) or set(value)!={"format","recipe","recipe_sha256"} or value["format"]!=1: raise ValueError("invalid frozen protocol structure")
    recipe=validate_recipe(value["recipe"],synthetic=synthetic)
    actual=hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if actual!=value["recipe_sha256"]: raise ValueError("frozen protocol hash mismatch")
    return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    f=sub.add_parser("freeze"); f.add_argument("--recipe",required=True); f.add_argument("--output",required=True); f.add_argument("--synthetic",action="store_true")
    v=sub.add_parser("verify"); v.add_argument("--protocol",required=True); v.add_argument("--synthetic",action="store_true")
    a=p.parse_args(argv)
    result=freeze(a.recipe,a.output,synthetic=a.synthetic) if a.command=="freeze" else verify_frozen(a.protocol,synthetic=a.synthetic)
    print(json.dumps({"recipe_sha256":result["recipe_sha256"],"synthetic":result["recipe"]["synthetic"]},sort_keys=True)); return 0
if __name__=="__main__":raise SystemExit(main())
