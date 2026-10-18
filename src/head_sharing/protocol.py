"""Canonical, immutable study recipe and protocol freeze checks."""
from __future__ import annotations
import argparse, hashlib, json, os, tempfile
from pathlib import Path

SEEDS=[11,22,33,44,55]
ARMS=["native","independent_k2","shared_k2"]
from . import LOOPED_ARMS
LOOPED_MATRIX=["native", *LOOPED_ARMS]
OLD_RECIPE_SHA256="3db46e027e40eefde310b3dc35e5c0318e2613993f153ff7dfcd8533e37c9ff0"
OLD_PROTOCOL_FILE_SHA256="ffdff515abf7d92d6ffb7019764ffb7297b6bdb89519065c4b5c186929801955"
OLD_FREEZE_SHA256="70d7deb42d9e23d9efa6a3215f4e06841a3c336af860d208acb17d2152d359b3"
LOOPED_ARCHITECTURE={"loop_ab2":{"box_width":24,"class_width":96,"updates":2},"loop_a3":{"box_width":28,"class_width":108,"updates":3},"adapter":"depthwise_3x3_pointwise_1x1","normalization":"per_point_channel_layernorm","activation":"silu","alpha":0.1,"scale_ids":"learned_per_scale_channel_vector","readout_priors":"native_per_scale_bias"}
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
    env = recipe["environment"]
    refinement = isinstance(env, dict) and env.get("study_id") == "ab2_refinement_v1"
    shared_spatial = isinstance(env, dict) and env.get("study_id") == "shared_spatial_v1"
    if refinement:
        if recipe["seed_list"] != [11, 33, 55] or recipe["arms"] != ["loop_ab2"]:
            raise ValueError("refinement matrix must be exactly loop_ab2 and seeds 11,33,55")
    elif shared_spatial:
        if recipe["seed_list"] != SEEDS or recipe["arms"] != ["native"]:
            raise ValueError("shared spatial matrix must be exactly native and five paired seeds")
        from .head_spec import canonicalize_head_spec
        if "head_spec" not in env or canonicalize_head_spec(env["head_spec"]) != env["head_spec"]:
            raise ValueError("shared spatial recipe requires an explicit canonical v2 spec or None")
        spec = env["head_spec"]
        if spec is not None and spec.get("version") != 2:
            raise ValueError("shared spatial recipe requires a native v2 spec")
    elif recipe["seed_list"]!=SEEDS or recipe["arms"] not in (ARMS,LOOPED_MATRIX):
        raise ValueError("main matrix must be exactly five paired seeds and three arms")
    looped = recipe["arms"] == LOOPED_MATRIX
    for key in ("imgsz","batch_size","epochs","cadence","block_size","probe_size"):
        if type(recipe[key]) is not int or recipe[key] < 1: raise ValueError(f"invalid fixed {key}")
    geometry_ok = recipe["imgsz"] >= 64 and recipe["imgsz"] % 32 == 0 if synthetic else recipe["imgsz"] in (512,640)
    if not geometry_ok or recipe["batch_size"]<2: raise ValueError("invalid production geometry/batch")
    if shared_spatial:
        dose = (recipe["imgsz"], recipe["batch_size"], recipe["block_size"], recipe["epochs"], recipe["cadence"], recipe["probe_size"])
        if synthetic:
            if recipe["epochs"] > 40 or recipe["cadence"] > recipe["epochs"]:
                raise ValueError("synthetic shared spatial dose exceeds the fixed production target")
        elif dose != (512,32,32,40,5,1000):
            raise ValueError("shared spatial production dose must be exactly 512/B32/block32/40/cadence5/probe1000")
    if recipe["block_size"] != recipe["batch_size"]: raise ValueError("physical block size must match extraction/training batch")
    if not synthetic and recipe["probe_size"] != 1000: raise ValueError("main clean probe must contain exactly 1000 images")
    if recipe["epochs"]<1 or recipe["cadence"]>recipe["epochs"]: raise ValueError("invalid fixed evaluation schedule")
    from .runtime import GPU_UUIDS
    assignment=recipe["gpu_for_seed"]
    expected_seeds = [11, 33, 55] if refinement else SEEDS
    if not isinstance(assignment,dict) or set(assignment)!={str(seed) for seed in expected_seeds} or any(value not in GPU_UUIDS for value in assignment.values()):
        raise ValueError("each complete seed triplet requires one permitted physical GPU")
    if shared_spatial and assignment != {str(seed): GPU_UUIDS[1] for seed in SEEDS}:
        raise ValueError("shared spatial seeds must all use the assigned G1 GPU")
    if shared_spatial and not synthetic and recipe["producer"] != GPU_UUIDS[1]:
        raise ValueError("shared spatial production cache producer must be the assigned G1 GPU")
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
    if refinement:
        from .head_spec import canonicalize_head_spec
        if "head_spec" not in env:
            raise ValueError("refinement requires an explicit head_spec key")
        spec = env["head_spec"]
        if canonicalize_head_spec(spec) != spec:
            raise ValueError("refinement requires a canonical head_spec")
        reference = env.get("reference_baseline")
        if not isinstance(reference, dict) or set(reference) != {"protocol", "freeze"}:
            raise ValueError("refinement requires explicit baseline source bindings")
        for name, required in (("protocol", {"path", "sha256", "recipe_sha256"}), ("freeze", {"path", "sha256"})):
            item = reference.get(name)
            if not isinstance(item, dict) or set(item) != required:
                raise ValueError("baseline source bindings have an invalid schema")
            path = item["path"]
            from pathlib import PurePosixPath
            if (not isinstance(path, str) or "\\" in path or ":" in path or
                    str(PurePosixPath(path)) != path or PurePosixPath(path).is_absolute() or
                    any(part in ("", ".", "..") for part in path.split("/"))):
                raise ValueError("baseline source path must be normalized and relative")
            for key in required - {"path"}:
                value = item[key]
                if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                    raise ValueError("invalid baseline source digest")
        if not synthetic:
            if (reference["protocol"]["recipe_sha256"] != "11ad424d07404fd1687c897fb1db51ffc1287a331b2022b73854f14be752e3a2" or
                    reference["freeze"]["path"] != "looped-20261017T181804Z/selection/selected.json" or
                    reference["freeze"]["sha256"] != "df3d874822a1c8015c312ac6f0ce8c59b4d64c1494c6ca55f5ddd70c5d7c2a47"):
                raise ValueError("refinement baseline source identity differs from pinned artifacts")
            if (recipe["imgsz"], recipe["batch_size"], recipe["block_size"], recipe["epochs"], recipe["cadence"], recipe["probe_size"]) not in ((512,32,32,40,5,1000),(512,32,32,80,5,1000)):
                raise ValueError("refinement production scientific dose differs")
            if assignment != {str(seed): GPU_UUIDS[1] for seed in expected_seeds}:
                raise ValueError("refinement seeds must be assigned to the pinned baseline GPU")
    if looped:
        env=recipe["environment"]
        if not isinstance(env,dict) or env.get("study_id") != "looped_v1" or env.get("looped_architecture") != LOOPED_ARCHITECTURE:
            raise ValueError("looped study architecture contract mismatch")
        reference=env.get("reference_native")
        if not isinstance(reference,dict) or set(reference)!={"protocol","freeze"}:
            raise ValueError("looped study requires explicit native reference provenance")
        for name, required in (("protocol", {"path", "sha256", "recipe_sha256"}),
                               ("freeze", {"path", "sha256"})):
            item = reference.get(name)
            if not isinstance(item, dict) or set(item) != required:
                raise ValueError("native source bindings have an invalid schema")
            path = item["path"]
            from pathlib import PurePosixPath
            if (not isinstance(path, str) or "\\" in path or ":" in path or
                    str(PurePosixPath(path)) != path or PurePosixPath(path).is_absolute() or
                    any(part in ("", ".", "..") for part in path.split("/"))):
                raise ValueError("native source path must be normalized and relative")
            for key in required - {"path"}:
                value = item[key]
                if (not isinstance(value, str) or len(value) != 64 or
                        any(char not in "0123456789abcdef" for char in value)):
                    raise ValueError("invalid native source digest")
        if not synthetic:
            if (reference["protocol"]["sha256"] != OLD_PROTOCOL_FILE_SHA256 or
                    reference["protocol"]["recipe_sha256"] != OLD_RECIPE_SHA256 or
                    reference["freeze"]["sha256"] != OLD_FREEZE_SHA256):
                raise ValueError("looped native source must bind the pinned original artifacts")
            if (recipe["imgsz"],recipe["batch_size"],recipe["block_size"],recipe["epochs"],recipe["cadence"],recipe["probe_size"]) != (512,32,32,40,5,1000):
                raise ValueError("looped production scientific dose differs")
            expected={str(seed):GPU_UUIDS[1 if seed in (11,33,55) else 0] for seed in SEEDS}
            if assignment != expected: raise ValueError("looped GPU assignment differs from paired native controls")
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
