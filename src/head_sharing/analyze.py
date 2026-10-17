"""Analyze frozen study evidence only (no inference, scoring, or model loading).

CLI example::

    MPLBACKEND=Agg python -m head_sharing.analyze --input evidence/manifest.json --output results

The manifest is one JSON object rooted at ``evidence_root`` (relative to the
manifest file). It contains ``protocol`` and ``freeze`` objects with ``path``
and ``sha256``; ``members`` has exactly 15 objects with ``seed``, ``arm``,
``training``, ``score``, and ``prediction_receipt`` (each file reference is
``{"path": ..., "sha256": ...}``; receipt may additionally bind
``prediction_path`` and ``prediction_sha256``); and ``profiles`` has 24
objects with ``seed``, ``arm``, ``measurement`` (inference/training), ``batch``
and ``file`` reference. ``profile_seeds`` must be [11, 22]. Example member:

    {"seed":11,"arm":"native","training":{"path":"runs/11-native/training.json","sha256":"..."},
     "score":{"path":"scores/11-native.json","sha256":"..."},
     "prediction_receipt":{"path":"predictions/11-native.json","sha256":"...",
                            "prediction_path":"predictions/11-native.gz","prediction_sha256":"..."}}

All artifact paths are explicit, relative, link-free files under evidence_root.
Output is synthetic iff the frozen evidence is labelled synthetic; fixtures
must never be presented as study results.
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


def paired_stats(values):
    values = [float(x) for x in values]
    if len(values) != 5 or any(not math.isfinite(x) for x in values):
        raise ValueError("paired statistics require exactly five finite deltas")
    mean = sum(values) / 5
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / 4)
    half = T95_DF4 * sd / math.sqrt(5)
    return {"deltas_pp": values, "mean_pp": mean, "sample_sd_pp": sd,
            "t95_df4_pp": [mean - half, mean + half]}


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


def _validate(manifest_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = _json(manifest_path)
    if set(manifest) != {"evidence_root", "protocol", "freeze", "members", "profile_seeds", "profiles"}:
        raise ValueError("manifest has missing or unknown fields")
    base = manifest_path.parent
    root_path = base / manifest["evidence_root"]
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
    if recipe.get("seed_list") != list(SEEDS) or recipe.get("arms") != list(ARMS) or recipe.get("epochs") != 40 or recipe.get("cadence") != 5 or recipe.get("quantization") != "fp32_fp16_fp32_v1":
        raise ValueError("protocol differs from fixed study design")
    freeze_path, freeze_file_sha = _file(root, manifest["freeze"])
    freeze = _json(freeze_path)
    if (freeze.get("protocol_sha256") != recipe_sha or freeze.get("synthetic") is not recipe.get("synthetic") or
            freeze.get("trunk_sha256") != recipe.get("trunk_sha256") or recipe.get("batch_size") != 32):
        raise ValueError("freeze/protocol binding mismatch")
    frozen_members = {(r.get("seed"), r.get("arm")): r for r in freeze.get("members", []) if isinstance(r, dict)}
    expected = {(s, a) for s in SEEDS for a in ARMS}
    if set(frozen_members) != expected or len(freeze.get("members", [])) != 15:
        raise ValueError("freeze is not exactly the complete 15-member matrix")
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
        train_path, train_sha = _file(root, row["training"])
        train = _json(train_path)
        if train_sha != frozen.get("run_record_sha256") or (train.get("seed"), train.get("arm")) != key or train.get("protocol_sha256") != recipe_sha or train.get("synthetic") is not recipe.get("synthetic") or train.get("epochs_completed") != 40 or train.get("target_epochs") != 40 or train.get("selected_epoch") != frozen.get("selected_epoch"):
            raise ValueError("training record is not bound to its frozen member")
        score_path, score_sha = _file(root, row["score"])
        score = _json(score_path)
        receipt_ref = row["prediction_receipt"]
        pred_path, pred_file_sha = _file(root, {"path": receipt_ref.get("path"), "sha256": receipt_ref.get("sha256")})
        receipt = _json(pred_path)
        for field, expected_value in (("seed", key[0]), ("arm", key[1]), ("selected_epoch", frozen.get("selected_epoch")), ("freeze_sha256", freeze_file_sha), ("protocol_sha256", recipe_sha)):
            if score.get(field) != expected_value or receipt.get(field) != expected_value:
                raise ValueError(f"score/receipt {field} differs from frozen member")
        for field in ("prediction_sha256", "execution_uuid"):
            if score.get(field) != receipt.get(field):
                raise ValueError(f"score/receipt {field} mismatch")
        if receipt.get("config_sha256") != frozen.get("config_sha256") or receipt.get("weights_sha256") != frozen.get("weights_sha256"):
            raise ValueError("prediction receipt checkpoint identity mismatch")
        prediction_file_sha = None
        if receipt_ref.get("prediction_path"):
            pp, actual = _file(root, {"path": receipt_ref["prediction_path"], "sha256": receipt_ref.get("prediction_sha256")})
            if actual != receipt.get("prediction_sha256"):
                raise ValueError("prediction byte hash mismatch")
            prediction_file_sha = actual
        for field in ("image_ids", "image_ids_sha256", "image_set_sha256", "annotations_sha256", "content_sha256"):
            if not receipt.get(field):
                raise ValueError(f"prediction receipt missing {field}")
        policy = receipt.get("policy")
        expected_uuid = "cpu" if recipe["synthetic"] else recipe["gpu_for_seed"][str(key[0])]
        if receipt.get("execution_uuid") != expected_uuid or policy != PREDICTION_POLICY:
            raise ValueError("prediction assignment or policy mismatch")
        ids = receipt.get("image_ids")
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or any(type(i) is not int for i in ids):
            raise ValueError("invalid evaluation image IDs")
        members[key] = {"training": train, "score": score, "receipt": receipt, "score_sha256": score_sha, "training_sha256": train_sha, "receipt_sha256": pred_file_sha, "prediction_file_sha256": prediction_file_sha}
    if set(members) != expected:
        raise ValueError("main evidence matrix incomplete")
    # A single identical evaluation cohort and original annotation binding across all 15 receipts.
    cohort = {(tuple(sorted(v["receipt"]["image_ids"])), v["receipt"]["image_ids_sha256"], v["receipt"]["image_set_sha256"], v["receipt"]["annotations_sha256"]) for v in members.values()}
    if len(cohort) != 1:
        raise ValueError("prediction receipts disagree on evaluation cohort/annotations")
    profile_rows = manifest["profiles"]
    if manifest["profile_seeds"] != list(PROFILE_SEEDS) or not isinstance(profile_rows, list) or len(profile_rows) != 24:
        raise ValueError("footprint scope must explicitly declare 24 seed-11/22 profiles")
    profiles = []
    seen = set()
    matched_policies = {}
    for item in profile_rows:
        if not isinstance(item, dict) or set(item) != {"seed", "arm", "measurement", "batch", "file"}:
            raise ValueError("invalid profile declaration")
        seed, arm, measurement, batch = item["seed"], item["arm"], item["measurement"], item["batch"]
        key = (seed, arm, measurement, batch)
        if seed not in PROFILE_SEEDS or arm not in ARMS or measurement not in ("inference", "training") or batch not in (1, recipe["batch_size"]) or key in seen:
            raise ValueError("unexpected or duplicate declared profile")
        seen.add(key)
        fpath, digest = _file(root, item["file"])
        profile = _json(fpath)
        frozen = frozen_members[(seed, arm)]
        gpu = recipe["gpu_for_seed"][str(seed)]
        identity = profile.get("model_identity", {})
        if (profile.get("seed") != seed or profile.get("arm") != arm or profile.get("measurement") != measurement or profile.get("imgsz") != recipe["imgsz"] or profile.get("quantization") != recipe["quantization"] or profile.get("numeric_policy") != recipe["numeric_policy"] or profile.get("producer") != recipe["producer"] or profile.get("execution_uuid") != gpu or profile.get("assigned_uuid") != gpu or profile.get("checkpoint_files_unchanged") is not True or identity.get("config_sha256") != frozen.get("config_sha256") or identity.get("weights_sha256") != frozen.get("weights_sha256") or identity.get("trunk_sha256") != recipe["trunk_sha256"] or identity.get("recipe_sha256") != recipe_sha):
            raise ValueError("profile identity/policy differs from frozen assignment")
        inputs = profile.get("measurement_inputs", {})
        if inputs.get("batch_size") != batch:
            raise ValueError("profile batch differs from declared scope")
        if measurement == "inference":
            signature = []
            for region in ("full_detector", "head"):
                sample = profile.get(region, {})
                if (sample.get("batch") != batch or sample.get("dtype") != "float32" or
                        type(sample.get("warmup")) is not int or sample["warmup"] < 0 or
                        type(sample.get("repetitions")) is not int or sample["repetitions"] < 1 or
                        sample.get("resolution") != [recipe["imgsz"], recipe["imgsz"]] or
                        not isinstance(sample.get("quantization_boundary"), str)):
                    raise ValueError("inference profile policy mismatch")
                signature.append((sample["warmup"], sample["repetitions"], sample["dtype"], tuple(sample["resolution"]), sample["quantization_boundary"]))
        else:
            tp = profile.get("training_protocol", {})
            if (tp.get("batch_size") != batch or tp.get("precision") != "native BF16 head" or
                    not tp.get("scope") or not tp.get("input") or not profile.get("training_peak")):
                raise ValueError("training probe policy mismatch")
            signature = (tp["batch_size"], tp["input"], tp["precision"], tp["scope"])
        policy_key = (gpu, measurement, batch)
        if policy_key in matched_policies and matched_policies[policy_key] != signature:
            raise ValueError("profile arms are not policy-matched within GPU/batch")
        matched_policies[policy_key] = signature
        profiles.append({"seed": seed, "arm": arm, "measurement": measurement, "batch": batch, "gpu": gpu, "profile": profile, "sha256": digest})
    if len(seen) != 24 or any((s, a, m, b) not in seen for s in PROFILE_SEEDS for a in ARMS for m in ("inference", "training") for b in (1, recipe["batch_size"])):
        raise ValueError("footprint matrix incomplete")
    provenance = {"manifest_sha256": sha(manifest_path), "protocol_file_sha256": protocol_file_sha,
                  "freeze_file_sha256": freeze_file_sha, "inputs": sorted([h for x in members.values() for h in (x["training_sha256"], x["score_sha256"], x["receipt_sha256"], x["prediction_file_sha256"]) if h] + [x["sha256"] for x in profiles])}
    return manifest, recipe, freeze, members, profiles, provenance


def analyze(input_path, output_path):
    manifest, recipe, freeze, members, profiles, provenance = _validate(input_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("output directory must be new/exclusive")
    synthetic = recipe["synthetic"]
    metric = "AP50:95"
    for key, value in members.items():
        for name in METRICS:
            score = _finite(value["score"].get(name), f"{key} {name}")
            if not 0 <= score <= 1:
                raise ValueError(f"{name} must be on the 0..1 scale")
    values = {key: _finite(value["score"][metric], f"{key} {metric}") for key, value in members.items()}
    if any(not 0 <= x <= 1 for x in values.values()):
        raise ValueError("AP50:95 must be on the 0..1 scale")
    contrasts = []
    for against, label in (("independent_k2", "primary_shared_minus_independent"), ("native", "secondary_shared_minus_native")):
        contrasts.append({"name": label, "comparison": f"shared_k2 - {against}", "units": "percentage points (AP50:95 x 100)", **paired_stats([(values[(s, "shared_k2")] - values[(s, against)]) * 100 for s in SEEDS])})
    curves = []
    for seed in SEEDS:
        for arm in ARMS:
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
                curves.append({"seed": seed, "arm": arm, "epoch": expected_epoch,
                               "loss": _finite(row.get("loss"), "loss"),
                               "dev_ap50_95": _finite(dev, "dev AP") if dev is not None else "",
                               "probe_ap50_95": _finite(probe, "probe AP") if probe is not None else ""})
    profile_table = []
    for item in profiles:
        p = item["profile"]
        row = {k: item[k] for k in ("seed", "arm", "measurement", "batch", "gpu")}
        row.update({"serialized_head_plus_trunk_bytes": p.get("portable_weight_files", {}).get("head_plus_trunk_file_bytes"),
                    "conv_macs": p.get("executed_conv_macs", {}).get("conv_macs") if item["measurement"] == "inference" else None,
                    "peak_allocated_bytes": p.get("inference_peak", {}).get("peak_allocated_bytes") if item["measurement"] == "inference" and p.get("inference_peak") else p.get("training_peak", {}).get("peak_allocated_bytes") if p.get("training_peak") else None,
                    "mean_full_ms": p.get("full_detector", {}).get("mean_ms") if item["measurement"] == "inference" else None,
                    "training_probe_scope": p.get("training_protocol", {}).get("scope") if item["measurement"] == "training" else None})
        profile_table.append(row)
    summary = {"label": "SYNTHETIC" if synthetic else "FROZEN_EVIDENCE", "metric": metric,
               "metric_scale": "0..1; paired contrasts are percentage points (x100)",
               "matrix": {"seeds": list(SEEDS), "arms": list(ARMS), "epochs": 40, "cadence": 5},
               "values": [{"seed": s, "arm": a, "AP50:95": values[(s, a)]} for s in SEEDS for a in ARMS],
               "contrasts": contrasts,
               "caveat": "One frozen trunk; conditional head-seed comparisons; n=5 small sample. A confidence interval crossing zero is not evidence of equivalence.",
               "footprint_scope": "Declared representative seeds 11 and 22, all arms at each assigned GPU, inference/training batches 1 and frozen batch 32; not five independent footprint replications. Compare arms only within matched GPU, batch, and policy. Storage/file bytes/ConvMAC are not measured VRAM or speedup claims.",
               "provenance": provenance}
    output.mkdir(parents=True)
    _write_json(output / "summary.json", summary)
    _write_csv(output / "scores.csv", ["seed", "arm", metric], summary["values"])
    _write_csv(output / "paired_deltas.csv", ["contrast", "seed", "delta_pp"], [
        {"contrast": c["name"], "seed": s, "delta_pp": c["deltas_pp"][i]} for c in contrasts for i, s in enumerate(SEEDS)])
    _write_csv(output / "curves.csv", ["seed", "arm", "epoch", "loss", "dev_ap50_95", "probe_ap50_95"], curves)
    _write_csv(output / "profiles.csv", list(profile_table[0]), profile_table)
    _figures(output, contrasts, curves)
    return summary


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _figures(output, contrasts, curves):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["svg.hashsalt"] = "head-sharing-analyze"
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for index, item in enumerate(contrasts):
        y = item["deltas_pp"]
        ax.scatter([index] * 5, y, label=item["name"])
        ax.errorbar(index, item["mean_pp"], yerr=[[item["mean_pp"] - item["t95_df4_pp"][0]], [item["t95_df4_pp"][1] - item["mean_pp"]]], fmt="_", capsize=4, color="black")
    ax.axhline(0, color="gray", linewidth=.8); ax.set_xticks([0, 1], [c["name"] for c in contrasts]); ax.set_ylabel("Paired AP50:95 difference (percentage points)")
    fig.tight_layout(); fig.savefig(output / "paired_deltas.svg", metadata={"Date": None, "Creator": "head_sharing.analyze"}); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for arm in ARMS:
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
