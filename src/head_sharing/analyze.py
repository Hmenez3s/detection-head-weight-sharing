"""Analyze explicitly frozen legacy or mixed-reference study evidence only.

All artifact paths are explicit, relative, link-free files under evidence_root.
Analysis performs no inference, scoring, or model loading. Synthetic evidence
is always labeled SYNTHETIC and must never be presented as study results.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

from .runtime import _has_symlink_component

SEEDS = (11, 22, 33, 44, 55)
ARMS = ("native", "independent_k2", "shared_k2")
LOOPED_ARMS = ("loop_ab2", "loop_a3")
LOOPED_MATRIX = ("native", *LOOPED_ARMS)
METRICS = ("AP50:95", "AP50", "AP75", "APs", "APm", "APl")
PREDICTION_POLICY = {"quantization": "fp32_fp16_fp32_v1", "conf_threshold": .001,
    "max_det": 300, "metric_head_dtype": "float32", "NVIDIA_TF32_OVERRIDE": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": True, "cudnn_benchmark": False,
    "cudnn_deterministic": False, "deterministic_algorithms": False}
PROFILE_SEEDS = (11, 22)
T95_DF4 = 2.7764451051977987


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _t_two_sided_p_df4(t_value):
    """Stable exact df=4 Student-t tail, expressed through its beta identity."""
    if not math.isfinite(t_value):
        raise ValueError("t statistic must be finite")
    radius = math.hypot(2.0, t_value)
    # 1-|t|/sqrt(t²+4), evaluated without subtracting nearly equal values.
    delta = (4.0 / radius) / (radius + abs(t_value))
    return max(0.0, min(1.0, 0.5 * delta * delta * (3.0-delta)))


def paired_stats(values):
    values = [float(x) for x in values]
    if len(values) != 5 or any(not math.isfinite(x) for x in values):
        raise ValueError("paired statistics require exactly five finite deltas")
    mean = sum(values) / 5
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / 4)
    half = T95_DF4 * sd / math.sqrt(5)
    if sd == 0:
        p_value = 1.0 if mean == 0 else 0.0
    else:
        p_value = _t_two_sided_p_df4(mean * math.sqrt(5) / sd)
    return {"deltas_pp": values, "mean_pp": mean, "sample_sd_pp": sd,
            "t95_df4_pp": [mean - half, mean + half], "t_df4": mean * math.sqrt(5) / sd if sd else None,
            "p_two_sided": p_value}


def holm_adjust(p_values):
    """Holm step-down adjusted p-values, preserving input order."""
    if not p_values or any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values):
        raise ValueError("Holm adjustment requires finite p-values in [0, 1]")
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(ordered)
    running = 0.0
    for rank, (index, p_value) in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * p_value)
        adjusted[index] = min(1.0, running)
    return adjusted


def _json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _file(root, ref):
    if not isinstance(ref, dict) or set(ref) - {"path", "sha256"} or not isinstance(ref.get("path"), str):
        raise ValueError("invalid explicit evidence file reference")
    rel = Path(ref["path"])
    if rel.is_absolute() or ".." in rel.parts or not ref["path"]:
        raise ValueError("evidence path must be relative beneath evidence_root")
    path = root / rel
    if (_has_symlink_component(path) or not path.is_file() or
            not path.resolve(strict=True).is_relative_to(root)):
        raise ValueError(f"missing, linked, or escaping evidence file: {ref['path']}")
    digest = sha(path)
    if not isinstance(ref.get("sha256"), str) or digest != ref["sha256"]:
        raise ValueError(f"evidence SHA256 mismatch: {ref['path']}")
    return path, digest


def _finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"non-finite numeric value: {label}")
    return float(value)


def _validate_synthetic_profile_producer(profile, descriptor):
    """Accept CPU smoke output without mistaking absent CUDA fields for data."""
    if profile.get("execution_uuid") is not None:
        raise ValueError("synthetic CPU profile claims a physical execution UUID")
    if profile.get("measurement") == "inference":
        if (profile.get("inference_peak") is not None or profile.get("branch_policy") != descriptor["branch_policy"] or
                any(profile.get(region, {}).get("timing_label") != "CPU non-study smoke only"
                    for region in ("full_detector", "head"))):
            raise ValueError("synthetic CPU inference profile producer contract mismatch")
    elif profile.get("training_peak") is not None:
        raise ValueError("synthetic CPU training profile claims CUDA peak evidence")


def _validate(manifest_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = _json(manifest_path)
    if set(manifest) != {"evidence_root", "protocol", "freeze", "members", "profile_seeds", "profiles"}:
        raise ValueError("manifest has missing or unknown fields")
    root_path = manifest_path.parent / manifest["evidence_root"]
    if _has_symlink_component(root_path):
        raise ValueError("evidence_root must be link-free")
    root = root_path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("evidence_root must be a directory")
    protocol_path, protocol_file_sha = _file(root, manifest["protocol"])
    protocol = _json(protocol_path)
    if protocol.get("format") != 1 or not isinstance(protocol.get("recipe"), dict):
        raise ValueError("invalid frozen protocol")
    recipe = protocol["recipe"]
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if protocol.get("recipe_sha256") != recipe_sha:
        raise ValueError("frozen protocol recipe hash mismatch")
    looped = recipe.get("arms") == list(LOOPED_MATRIX)
    arms = LOOPED_MATRIX if looped else ARMS
    expected = {(s, a) for s in SEEDS for a in arms}
    if (recipe.get("seed_list") != list(SEEDS) or recipe.get("arms") != list(arms) or
            recipe.get("epochs") != 40 or recipe.get("cadence") != 5 or
            recipe.get("quantization") != "fp32_fp16_fp32_v1"):
        raise ValueError("protocol differs from fixed study design")
    freeze_path, freeze_file_sha = _file(root, manifest["freeze"])
    freeze = _json(freeze_path)
    version = 2 if looped else 1
    if ((freeze.get("schema_version", 1) != version) or freeze.get("protocol_sha256") != recipe_sha or
            (looped and freeze.get("protocol") != manifest["protocol"].get("path")) or
            freeze.get("synthetic") is not recipe.get("synthetic") or
            freeze.get("trunk_sha256") != recipe.get("trunk_sha256") or recipe.get("batch_size") != 32):
        raise ValueError("freeze/protocol binding mismatch")
    freeze_rows = freeze.get("members")
    frozen_members = {(r.get("seed"), r.get("arm")): r for r in freeze_rows if isinstance(r, dict)} if isinstance(freeze_rows, list) else {}
    if set(frozen_members) != expected or len(freeze_rows or []) != 15:
        raise ValueError("freeze is not exactly the complete 15-member matrix")

    source_protocol = source_freeze = source_freeze_sha = source_recipe_sha = None
    if looped:
        from .evaluate import member_source_binding
        if not recipe.get("synthetic"):
            # Production reports consume the actual producer contract, not a
            # looser handwritten approximation used by synthetic fixtures.
            from .protocol import verify_frozen
            from .evaluate import verify_freeze
            verify_frozen(protocol_path, synthetic=False)
            verify_freeze(freeze_path, root)
        references = recipe.get("environment", {}).get("reference_native")
        if not isinstance(references, dict) or freeze.get("reference_native") != references:
            raise ValueError("freeze native reference differs from recipe")
        source_protocol_path, source_protocol_sha = _file(root, {"path": references.get("protocol", {}).get("path"),
                                                                      "sha256": references.get("protocol", {}).get("sha256")})
        if source_protocol_sha != references["protocol"].get("sha256"):
            raise ValueError("native source protocol file hash mismatch")
        source_protocol = _json(source_protocol_path)
        source_recipe = source_protocol.get("recipe")
        if not isinstance(source_recipe, dict):
            raise ValueError("native source protocol recipe is invalid")
        source_recipe_sha = hashlib.sha256(json.dumps(source_recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        inherited = ("seed_list", "imgsz", "batch_size", "block_size", "epochs", "cadence", "probe_size",
                     "optimizer", "gains", "quantization", "producer", "trunk_sha256", "split_sha256",
                     "cache_evidence", "synthetic", "gpu_for_seed", "selection", "numeric_policy")
        if (source_protocol.get("format") != 1 or source_protocol.get("recipe_sha256") != source_recipe_sha or
                source_recipe_sha != references["protocol"].get("recipe_sha256") or source_recipe.get("arms") != list(ARMS) or
                any(source_recipe.get(field) != recipe.get(field) for field in inherited)):
            raise ValueError("native source protocol identity or inherited recipe differs")
        source_freeze_path, source_freeze_sha = _file(root, {"path": references.get("freeze", {}).get("path"),
                                                                  "sha256": references.get("freeze", {}).get("sha256")})
        if source_freeze_sha != references["freeze"].get("sha256"):
            raise ValueError("native source freeze file hash mismatch")
        source_freeze = _json(source_freeze_path)
        source_protocol_ref = source_freeze.get("protocol")
        from .evaluate import _source_bundle_root
        source_bundle_root = _source_bundle_root(root, source_freeze_path, source_protocol_path, source_protocol_ref)
        source_rows = source_freeze.get("members", [])
        old_native = {(r.get("seed"), r.get("arm")): r for r in source_rows if isinstance(r, dict) and r.get("arm") == "native"}
        if (source_freeze.get("schema_version") != 1 or source_freeze.get("protocol_sha256") != source_recipe_sha or
                source_freeze.get("synthetic") is not source_recipe.get("synthetic") or
                source_freeze.get("trunk_sha256") != source_recipe.get("trunk_sha256") or
                source_freeze.get("split_sha256") != source_recipe.get("split_sha256") or
                source_freeze.get("imgsz") != source_recipe.get("imgsz") or
                source_freeze.get("batch_size") != source_recipe.get("batch_size") or
                source_freeze.get("quantization") != source_recipe.get("quantization") or
                source_freeze.get("producer") != source_recipe.get("producer") or
                source_freeze.get("selection") != source_recipe.get("selection") or
                source_freeze.get("policy") != PREDICTION_POLICY or
                len(source_rows) != 15 or set(old_native) != {(s, "native") for s in SEEDS}):
            raise ValueError("native source freeze header or controls are invalid")
        for seed in SEEDS:
            current = frozen_members[(seed, "native")]
            old = old_native[(seed, "native")]
            member_source_binding(freeze, current, freeze_file_sha)
            source_refs = {}
            for field in ("config", "weights"):
                ref = old.get(field)
                if (not isinstance(ref, str) or "\\" in ref or Path(ref).is_absolute() or
                        ".." in Path(ref).parts):
                    raise ValueError("native source checkpoint reference is not portable")
                target = source_bundle_root / ref
                if _has_symlink_component(target) or not target.absolute().is_relative_to(root):
                    raise ValueError("native source checkpoint reference escapes evidence root")
                source_refs[field] = target.relative_to(root).as_posix()
            expected_config, expected_weights = source_refs["config"], source_refs["weights"]
            if (current.get("source") != {"kind": "reused_native", "protocol_sha256": source_recipe_sha,
                                           "freeze_sha256": source_freeze_sha} or
                    current.get("config") != expected_config or current.get("weights") != expected_weights or
                    any(current.get(field) != old.get(field) for field in
                        ("selected_epoch", "selected_dev_ap", "config_sha256", "weights_sha256",
                         "run_record_sha256", "resume_sha256", "trunk_sha256"))):
                raise ValueError("reused native member differs from original frozen identity")
        source_binding_meta = {"protocol_sha256": source_recipe_sha, "freeze_sha256": source_freeze_sha}
    else:
        source_binding_meta = None

    rows = manifest["members"]
    if not isinstance(rows, list) or len(rows) != 15:
        raise ValueError("manifest must bind exactly fifteen members")
    members = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"seed", "arm", "training", "score", "prediction_receipt"}:
            raise ValueError("invalid member declaration")
        key = (row["seed"], row["arm"])
        if key not in expected or key in members:
            raise ValueError("duplicate or unexpected main matrix cell")
        frozen = frozen_members[key]
        reused = looped and key[1] == "native"
        expected_protocol_sha = source_recipe_sha if reused else recipe_sha
        expected_freeze_sha = source_freeze_sha if reused else freeze_file_sha
        if looped:
            source_binding = member_source_binding(freeze, frozen, freeze_file_sha)
            if source_binding != {"protocol_sha256": expected_protocol_sha, "freeze_sha256": expected_freeze_sha}:
                raise ValueError("frozen member source binding differs from declared provenance")
        train_path, train_sha = _file(root, row["training"])
        train = _json(train_path)
        if (train_sha != frozen.get("run_record_sha256") or (train.get("seed"), train.get("arm")) != key or
                train.get("protocol_sha256") != expected_protocol_sha or
                train.get("synthetic") is not recipe.get("synthetic") or train.get("epochs_completed") != 40 or
                train.get("target_epochs") != 40 or train.get("selected_epoch") != frozen.get("selected_epoch")):
            raise ValueError("training record is not bound to its frozen member")
        score_path, score_sha = _file(root, row["score"])
        score = _json(score_path)
        receipt_ref = row["prediction_receipt"]
        pred_path, pred_file_sha = _file(root, {"path": receipt_ref.get("path"), "sha256": receipt_ref.get("sha256")})
        receipt = _json(pred_path)
        for field, expected_value in (("seed", key[0]), ("arm", key[1]),
                                      ("selected_epoch", frozen.get("selected_epoch")),
                                      ("freeze_sha256", expected_freeze_sha),
                                      ("protocol_sha256", expected_protocol_sha)):
            if score.get(field) != expected_value or receipt.get(field) != expected_value:
                raise ValueError(f"score/receipt {field} differs from frozen member source")
        for field in ("prediction_sha256", "execution_uuid"):
            if score.get(field) != receipt.get(field):
                raise ValueError(f"score/receipt {field} mismatch")
        if receipt.get("config_sha256") != frozen.get("config_sha256") or receipt.get("weights_sha256") != frozen.get("weights_sha256"):
            raise ValueError("prediction receipt checkpoint identity mismatch")
        prediction_file_sha = None
        if receipt_ref.get("prediction_path"):
            _, actual = _file(root, {"path": receipt_ref["prediction_path"], "sha256": receipt_ref.get("prediction_sha256")})
            if actual != receipt.get("prediction_sha256"):
                raise ValueError("prediction byte hash mismatch")
            prediction_file_sha = actual
        for field in ("image_ids", "image_ids_sha256", "image_set_sha256", "annotations_sha256", "content_sha256"):
            if not receipt.get(field):
                raise ValueError(f"prediction receipt missing {field}")
        expected_uuid = "cpu" if recipe["synthetic"] else recipe["gpu_for_seed"][str(key[0])]
        if receipt.get("execution_uuid") != expected_uuid or receipt.get("policy") != PREDICTION_POLICY:
            raise ValueError("prediction assignment or policy mismatch")
        ids = receipt.get("image_ids")
        ids_sha = receipt.get("image_ids_sha256")
        if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or
                any(type(i) is not int for i in ids) or not isinstance(ids_sha, str) or len(ids_sha) != 64 or
                any(char not in "0123456789abcdef" for char in ids_sha)):
            raise ValueError("invalid evaluation image IDs or digest")
        members[key] = {"training": train, "score": score, "receipt": receipt,
                        "source": source_binding_meta if reused else {"protocol_sha256": recipe_sha, "freeze_sha256": freeze_file_sha},
                        "score_sha256": score_sha, "training_sha256": train_sha,
                        "receipt_sha256": pred_file_sha, "prediction_file_sha256": prediction_file_sha}
    if set(members) != expected:
        raise ValueError("main evidence matrix incomplete")
    cohort = {(tuple(sorted(v["receipt"]["image_ids"])), v["receipt"]["image_ids_sha256"],
               v["receipt"]["image_set_sha256"], v["receipt"]["annotations_sha256"]) for v in members.values()}
    if len(cohort) != 1:
        raise ValueError("prediction receipts disagree on evaluation cohort/annotations")

    profile_rows = manifest["profiles"]
    profile_arms = LOOPED_MATRIX if looped else ARMS
    expected_profile_count = 36 if looped else 24
    if manifest["profile_seeds"] != list(PROFILE_SEEDS) or not isinstance(profile_rows, list) or len(profile_rows) != expected_profile_count:
        raise ValueError(f"footprint scope must explicitly declare {expected_profile_count} seed-11/22 profiles")
    profiles, seen, matched_policies = [], set(), {}
    for item in profile_rows:
        expected_fields = {"seed", "arm", "measurement", "batch", "file", "branch_policy"} if looped else {"seed", "arm", "measurement", "batch", "file"}
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise ValueError("invalid profile declaration")
        seed, arm, measurement, batch = item["seed"], item["arm"], item["measurement"], item["batch"]
        policy = item.get("branch_policy", "legacy")
        key = (seed, arm, measurement, batch, policy) if looped else (seed, arm, measurement, batch)
        if (seed not in PROFILE_SEEDS or arm not in profile_arms or measurement not in ("inference", "training") or
                batch not in (1, recipe["batch_size"]) or key in seen or
                (looped and policy not in (("retained_dual", "fused_one2one") if measurement == "inference" else ("retained_dual",)))):
            raise ValueError("unexpected or duplicate declared profile")
        seen.add(key)
        _, profile_sha = _file(root, item["file"])
        profile = _json(root / item["file"]["path"])
        frozen = frozen_members[(seed, arm)]
        gpu = recipe["gpu_for_seed"][str(seed)]
        identity = profile.get("model_identity", {})
        expected_branch_policy = policy if measurement == "inference" else "retained_dual"
        if (profile.get("seed") != seed or profile.get("arm") != arm or profile.get("measurement") != measurement or
                profile.get("imgsz") != recipe["imgsz"] or profile.get("quantization") != recipe["quantization"] or
                profile.get("numeric_policy") != recipe["numeric_policy"] or profile.get("producer") != recipe["producer"] or
                profile.get("assigned_uuid") != recipe["gpu_for_seed"][str(seed)] or
                profile.get("checkpoint_files_unchanged") is not True or (looped and profile.get("branch_policy") != expected_branch_policy) or
                identity.get("config_sha256") != frozen.get("config_sha256") or identity.get("weights_sha256") != frozen.get("weights_sha256") or
                identity.get("trunk_sha256") != recipe["trunk_sha256"] or identity.get("recipe_sha256") != recipe_sha):
            raise ValueError("profile identity/policy differs from frozen assignment")
        if looped and recipe["synthetic"]:
            # The CPU producer intentionally emits no physical device or CUDA peaks.
            _validate_synthetic_profile_producer(profile, item)
        elif profile.get("execution_uuid") != gpu:
            raise ValueError("profile execution UUID differs from assigned device")
        inputs = profile.get("measurement_inputs", {})
        if inputs.get("batch_size") != batch:
            raise ValueError("profile batch differs from declared scope")
        if measurement == "inference":
            signature = []
            for region in ("full_detector", "head"):
                sample = profile.get(region, {})
                if (sample.get("batch") != batch or sample.get("dtype") != "float32" or
                        sample.get("warmup") != 5 or type(sample.get("warmup")) is not int or
                        sample.get("repetitions") != 20 or type(sample.get("repetitions")) is not int or
                        sample.get("resolution") != [recipe["imgsz"], recipe["imgsz"]] or
                        not isinstance(sample.get("quantization_boundary"), str) or
                        not isinstance(sample.get("mean_ms"), (int, float)) or not math.isfinite(sample["mean_ms"])):
                    raise ValueError("inference profile policy mismatch")
                signature.append((sample["warmup"], sample["repetitions"], sample["dtype"], tuple(sample["resolution"]), sample["quantization_boundary"]))
            if looped:
                peak = profile.get("inference_peak")
                if recipe["synthetic"]:
                    if peak is not None:
                        raise ValueError("synthetic CPU profile claims CUDA peak evidence")
                elif (not isinstance(peak, dict) or any(type(peak.get(field)) is not int or peak[field] < 0
                                                        for field in ("peak_allocated_bytes", "peak_reserved_bytes"))):
                    raise ValueError("inference allocated/reserved peak evidence missing")
                storage = profile.get("storage", {}).get("head", {})
                portable = profile.get("portable_weight_files", {})
                macs = profile.get("executed_conv_macs", {})
                if (any(type(storage.get(field)) is not int or storage[field] < 0 for field in
                        ("unique_parameters", "unique_parameter_storage_bytes", "unique_parameter_and_buffer_storage_bytes")) or
                        any(type(portable.get(field)) is not int or portable[field] < 0 for field in
                            ("head_file_bytes", "common_trunk_file_bytes", "head_plus_trunk_file_bytes")) or
                        type(macs.get("conv_macs")) is not int or macs["conv_macs"] <= 0):
                    raise ValueError("inference profile footprint fields are incomplete")
        else:
            tp = profile.get("training_protocol", {})
            if (tp.get("batch_size") != batch or tp.get("precision") != "native BF16 head" or
                    not tp.get("scope") or not tp.get("input")):
                raise ValueError("training probe policy mismatch")
            if looped:
                peak = profile.get("training_peak")
                if recipe["synthetic"]:
                    if peak is not None:
                        raise ValueError("synthetic CPU training profile claims CUDA peak evidence")
                elif (not isinstance(peak, dict) or any(type(peak.get(field)) is not int or peak[field] < 0
                                                        for field in ("peak_allocated_bytes", "peak_reserved_bytes"))):
                    raise ValueError("training allocated/reserved peak evidence missing")
            signature = (tp["batch_size"], tp["input"], tp["precision"], tp["scope"])
        policy_key = (gpu, measurement, batch, policy)
        if policy_key in matched_policies and matched_policies[policy_key] != signature:
            raise ValueError("profile arms are not policy-matched within GPU/batch/policy")
        matched_policies[policy_key] = signature
        profiles.append({"seed": seed, "arm": arm, "measurement": measurement, "batch": batch,
                         "gpu": gpu, "branch_policy": policy, "profile": profile, "sha256": profile_sha})
    if looped:
        wanted = {(s, a, "inference", b, p) for s in PROFILE_SEEDS for a in profile_arms for b in (1, recipe["batch_size"]) for p in ("retained_dual", "fused_one2one")}
        wanted |= {(s, a, "training", b, "retained_dual") for s in PROFILE_SEEDS for a in profile_arms for b in (1, recipe["batch_size"])}
    else:
        wanted = {(s, a, m, b) for s in PROFILE_SEEDS for a in ARMS for m in ("inference", "training") for b in (1, recipe["batch_size"])}
    if seen != wanted:
        raise ValueError("footprint matrix incomplete")
    provenance = {"manifest_sha256": sha(manifest_path), "protocol_file_sha256": protocol_file_sha,
                  "freeze_file_sha256": freeze_file_sha,
                  "source_protocol_file_sha256": sha(source_protocol_path) if looped else None,
                  "source_protocol_recipe_sha256": source_recipe_sha,
                  "source_freeze_file_sha256": source_freeze_sha,
                  "inputs": sorted([h for x in members.values() for h in (x["training_sha256"], x["score_sha256"], x["receipt_sha256"], x["prediction_file_sha256"]) if h] + [x["sha256"] for x in profiles])}
    return manifest, recipe, freeze, members, profiles, provenance


def analyze(input_path, output_path):
    manifest, recipe, freeze, members, profiles, provenance = _validate(input_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("output directory must be new/exclusive")
    looped = recipe["arms"] == list(LOOPED_MATRIX)
    arms = LOOPED_MATRIX if looped else ARMS
    metric_values, availability, score_rows = {}, {}, []
    for key, value in members.items():
        for name in METRICS:
            score = _finite(value["score"].get(name), f"{key} {name}")
            if name == "AP50:95" and score == -1:
                raise ValueError("primary AP50:95 must be available")
            if score != -1 and not 0 <= score <= 1:
                raise ValueError(f"{name} must be -1 (unavailable) or on the 0..1 scale")
            metric_values[(key[0], key[1], name)] = score
            availability[(key[0], key[1], name)] = score != -1
        source = value["source"]
        score_rows.append({"seed": key[0], "arm": key[1],
                           "source_kind": "reused_native" if looped and key[1] == "native" else "current",
                           "source_protocol_sha256": source["protocol_sha256"],
                           "source_freeze_sha256": source["freeze_sha256"],
                           **{name: metric_values[(key[0], key[1], name)] for name in METRICS},
                           **{f"{name}_available": availability[(key[0], key[1], name)] for name in METRICS}})

    if looped:
        contrast_specs = (("loop_ab2", "native", "primary_loop_ab2_minus_native"),
                          ("loop_a3", "native", "primary_loop_a3_minus_native"),
                          ("loop_a3", "loop_ab2", "secondary_loop_a3_minus_loop_ab2"))
    else:
        # Keep the historical contrast names and comparison semantics unchanged.
        contrast_specs = (("shared_k2", "independent_k2", "primary_shared_minus_independent"),
                          ("shared_k2", "native", "secondary_shared_minus_native"))
    contrasts = []
    for against_arm, baseline, label in contrast_specs:
        stats = paired_stats([(metric_values[(s, against_arm, "AP50:95")] - metric_values[(s, baseline, "AP50:95")]) * 100 for s in SEEDS])
        contrasts.append({"name": label, "comparison": f"{against_arm} - {baseline}",
                          "units": "percentage points (AP50:95 x 100)", **stats})
    primary = contrasts[:2] if looped else contrasts[:1]
    adjusted = holm_adjust([item["p_two_sided"] for item in primary])
    for item, p_adjusted in zip(primary, adjusted):
        item["holm_family"] = "two primary native-reference contrasts" if looped else "legacy primary contrast"
        item["p_holm"] = p_adjusted
    if looped:
        contrasts[2]["multiplicity"] = "secondary; outside two-test primary Holm family"

    curves = []
    for seed in SEEDS:
        for arm in arms:
            history = members[(seed, arm)]["training"].get("history")
            if not isinstance(history, list) or len(history) != 40:
                raise ValueError("training curve must contain all 40 epochs")
            for expected_epoch, row in enumerate(history, 1):
                if row.get("epoch") != expected_epoch:
                    raise ValueError("training curve epochs are incomplete or unordered")
                measured = expected_epoch % recipe["cadence"] == 0 or expected_epoch == recipe["epochs"]
                if measured != ("dev" in row and "probe" in row):
                    raise ValueError("dev/probe evaluation cadence mismatch")
                dev = row.get("dev", {}).get("AP50:95") if measured else None
                probe = row.get("probe", {}).get("AP50:95") if measured else None
                if (dev is None) != (probe is None):
                    raise ValueError("dev/probe evaluation cadence mismatch")
                dev_value = _finite(dev, "dev AP") if dev is not None else None
                probe_value = _finite(probe, "probe AP") if probe is not None else None
                curves.append({"seed": seed, "arm": arm, "epoch": expected_epoch,
                               "loss": _finite(row.get("loss"), "loss"),
                               "dev_ap50_95": dev_value if dev_value is not None else "",
                               "probe_ap50_95": probe_value if probe_value is not None else "",
                               "probe_minus_dev_ap50_95": probe_value - dev_value if dev_value is not None else ""})

    profile_table = []
    for item in profiles:
        p = item["profile"]
        inference = item["measurement"] == "inference"
        peak = p.get("inference_peak") if inference else p.get("training_peak")
        if peak is None:
            peak_allocated = peak_reserved = None
        else:
            peak_allocated = peak.get("peak_allocated_bytes")
            peak_reserved = peak.get("peak_reserved_bytes")
        row = {k: item[k] for k in ("seed", "arm", "measurement", "batch", "gpu", "branch_policy")}
        row.update({"head_unique_parameters": p.get("storage", {}).get("head", {}).get("unique_parameters"),
                    "head_unique_parameter_storage_bytes": p.get("storage", {}).get("head", {}).get("unique_parameter_storage_bytes"),
                    "head_unique_parameter_and_buffer_storage_bytes": p.get("storage", {}).get("head", {}).get("unique_parameter_and_buffer_storage_bytes"),
                    "full_unique_parameters": p.get("storage", {}).get("full_detector", {}).get("unique_parameters"),
                    "full_unique_parameter_storage_bytes": p.get("storage", {}).get("full_detector", {}).get("unique_parameter_storage_bytes"),
                    "full_unique_parameter_and_buffer_storage_bytes": p.get("storage", {}).get("full_detector", {}).get("unique_parameter_and_buffer_storage_bytes"),
                    "serialized_head_file_bytes": p.get("portable_weight_files", {}).get("head_file_bytes"),
                    "serialized_head_plus_trunk_bytes": p.get("portable_weight_files", {}).get("head_plus_trunk_file_bytes"),
                    "portable_file_scope": p.get("portable_weight_files", {}).get("scope"),
                    "conv_macs": p.get("executed_conv_macs", {}).get("conv_macs") if inference else None,
                    "head_conv_macs": p.get("head_executed_conv_macs", {}).get("conv_macs") if inference else None,
                    "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
                    "mean_full_ms": p.get("full_detector", {}).get("mean_ms") if inference else None,
                    "mean_head_ms": p.get("head", {}).get("mean_ms") if inference else None,
                    "timing_label": p.get("full_detector", {}).get("timing_label") if inference else None,
                    "warmup": p.get("full_detector", {}).get("warmup") if inference else None,
                    "repetitions": p.get("full_detector", {}).get("repetitions") if inference else None,
                    "training_probe_scope": p.get("training_protocol", {}).get("scope") if not inference else None})
        profile_table.append(row)
    matrix = {"seeds": list(SEEDS), "arms": list(arms), "epochs": 40, "cadence": 5}
    summary = {"label": "SYNTHETIC" if recipe["synthetic"] else "FROZEN_EVIDENCE",
               "study": "looped_v1" if looped else "legacy",
               "metric": "AP50:95", "metrics": list(METRICS),
               "metric_scale": "0..1; -1 denotes unavailable COCO size metric; paired contrasts are percentage points (x100)",
               "matrix": matrix, "values": score_rows, "contrasts": contrasts,
               "caveat": "One inherited frozen trunk; conditional head-seed comparisons; n=5 small sample. A confidence interval crossing zero is not evidence of equivalence.",
               "caveats": ["One inherited frozen trunk; head-seed comparisons are conditional on that trunk.",
                           *(["Reused val2017/native controls are not independent untouched confirmation and do not establish causal isolation of weight sharing."] if looped else []),
                           "n=5 paired seeds; uncertainty is substantial. A confidence interval crossing zero is not evidence of equivalence."],
               "footprint_scope": ("Fresh profiles on representative seeds 11 and 22; compare only matched GPU, batch, branch policy, precision, timing, and checkpoint scope. Training uses retained dual branch; fused inference is a separately prepared one2one policy. Portable checkpoint bytes, executable storage, ConvMAC, timing, and allocated/reserved memory are distinct measurements." if looped else
                   "Declared representative seeds 11 and 22, all legacy arms at each assigned GPU, inference/training batches 1 and frozen batch 32; not five independent footprint replications. Compare only matched GPU, batch and policy. The legacy 24 profiles do not establish fused one2one deployment measurements."),
               "provenance": provenance}
    if looped:
        summary["provenance"]["native_reference"] = {"protocol_sha256": provenance["source_protocol_recipe_sha256"],
                                                       "freeze_sha256": provenance["source_freeze_file_sha256"]}
    output.mkdir(parents=True)
    _write_json(output / "summary.json", summary)
    _write_csv(output / "scores.csv", ["seed", "arm", "source_kind", "source_protocol_sha256", "source_freeze_sha256",
                                         *METRICS, *[f"{name}_available" for name in METRICS]], score_rows)
    _write_csv(output / "paired_deltas.csv", ["contrast", "seed", "delta_pp"], [
        {"contrast": c["name"], "seed": s, "delta_pp": c["deltas_pp"][i]} for c in contrasts for i, s in enumerate(SEEDS)])
    _write_csv(output / "curves.csv", ["seed", "arm", "epoch", "loss", "dev_ap50_95", "probe_ap50_95", "probe_minus_dev_ap50_95"], curves)
    _write_csv(output / "profiles.csv", list(profile_table[0]), profile_table)
    _figures(output, contrasts, curves, arms)
    return summary


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _figures(output, contrasts, curves, arms=ARMS):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["svg.hashsalt"] = "head-sharing-analyze"
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(7, 2.5 * len(contrasts)), 4))
    for index, item in enumerate(contrasts):
        y = item["deltas_pp"]
        ax.scatter([index] * 5, y, label=item["name"])
        ax.errorbar(index, item["mean_pp"], yerr=[[item["mean_pp"] - item["t95_df4_pp"][0]], [item["t95_df4_pp"][1] - item["mean_pp"]]], fmt="_", capsize=4, color="black")
    ax.axhline(0, color="gray", linewidth=.8); ax.set_xticks(list(range(len(contrasts))), [c["name"] for c in contrasts]); ax.set_ylabel("Paired AP50:95 difference (percentage points)")
    fig.tight_layout(); fig.savefig(output / "paired_deltas.svg", metadata={"Date": None, "Creator": "head_sharing.analyze"}); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for arm in arms:
        rows = [r for r in curves if r["arm"] == arm]
        for axis, field, title in zip(axes, ("loss", "dev_ap50_95", "probe_ap50_95"), ("Training loss", "Dev AP50:95", "Probe AP50:95")):
            available = [(e, [r[field] for r in rows if r["epoch"] == e and r[field] != ""]) for e in range(1, 41)]
            points = [(e, sum(v) / len(v)) for e, v in available if v]
            axis.plot([p[0] for p in points], [p[1] for p in points], label=arm)
            axis.set_title(title); axis.set_xlabel("Epoch")
    axes[0].set_ylabel("Mean across five paired seeds"); axes[-1].legend(); fig.tight_layout()
    fig.savefig(output / "curves.svg", metadata={"Date": None, "Creator": "head_sharing.analyze"}); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="explicit manifest JSON")
    parser.add_argument("--output", required=True, help="new exclusive output directory")
    args = parser.parse_args(argv)
    analyze(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
