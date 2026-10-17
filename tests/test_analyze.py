"""Synthetic-only analysis evidence fixtures; no study artifacts are read."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import pytest

from head_sharing.analyze import analyze, paired_stats, PREDICTION_POLICY

SEEDS = [11, 22, 33, 44, 55]
ARMS = ["native", "independent_k2", "shared_k2"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return {"path": name, "sha256": digest(path)}


def fixture(root):
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    gpu = {"11": "GPU-A", "22": "GPU-B", "33": "GPU-A", "44": "GPU-B", "55": "GPU-A"}
    recipe = {"seed_list": SEEDS, "arms": ARMS, "epochs": 40, "cadence": 5,
              "quantization": "fp32_fp16_fp32_v1", "imgsz": 64, "batch_size": 32,
              "gpu_for_seed": gpu, "trunk_sha256": "t" * 64, "producer": "synthetic-producer",
              "numeric_policy": {"metric_head_dtype": "float32"}, "synthetic": True}
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    protocol_ref = put(evidence, "protocol.json", {"format": 1, "recipe": recipe, "recipe_sha256": recipe_sha})
    freeze = {"protocol_sha256": recipe_sha, "synthetic": True, "trunk_sha256": recipe["trunk_sha256"], "members": []}
    for seed in SEEDS:
        for arm in ARMS:
            freeze["members"].append({"seed": seed, "arm": arm, "selected_epoch": 40,
                "run_record_sha256": "", "config_sha256": f"c{seed}{arm}",
                "weights_sha256": f"w{seed}{arm}", "trunk_sha256": recipe["trunk_sha256"]})
    # score and receipt hashes bind the actual serialized freeze bytes.
    freeze_ref = put(evidence, "freeze.json", freeze)
    freeze_sha = freeze_ref["sha256"]
    cohort_ids = [101, 202]
    ids_sha = hashlib.sha256(json.dumps(cohort_ids, separators=(",", ":")).encode()).hexdigest()
    receipt_refs = {}
    for seed in SEEDS:
        for arm in ARMS:
            hist = []
            for epoch in range(1, 41):
                row = {"epoch": epoch, "loss": 10 / epoch + seed / 1000}
                if epoch % 5 == 0:
                    row["dev"] = {"AP50:95": .2 + epoch / 1000}
                    row["probe"] = {"AP50:95": .18 + epoch / 1000}
                hist.append(row)
            training = {"seed": seed, "arm": arm, "protocol_sha256": recipe_sha,
                        "synthetic": True, "epochs_completed": 40, "target_epochs": 40,
                        "selected_epoch": 40, "history": hist}
            training_ref = put(evidence, f"runs/{seed}-{arm}/training.json", training)
            next(row for row in freeze["members"] if (row["seed"], row["arm"]) == (seed, arm))["run_record_sha256"] = training_ref["sha256"]
            prediction = evidence / f"predictions/{seed}-{arm}.gz"
            prediction.parent.mkdir(parents=True, exist_ok=True)
            prediction.write_bytes(gzip.compress(b'{"predictions":[],"synthetic":true}', mtime=0))
            receipt = {"schema_version": 1, "seed": seed, "arm": arm, "selected_epoch": 40,
                "freeze_sha256": freeze_sha, "protocol_sha256": recipe_sha,
                "prediction_sha256": digest(prediction), "execution_uuid": "cpu",
                "config_sha256": f"c{seed}{arm}", "weights_sha256": f"w{seed}{arm}",
                "image_ids": cohort_ids, "image_ids_sha256": ids_sha,
                "image_set_sha256": "image-set", "annotations_sha256": "annotations",
                "content_sha256": "content", "policy": PREDICTION_POLICY}
            receipt_ref = put(evidence, f"predictions/{seed}-{arm}.json", receipt)
            receipt_ref.update(prediction_path=f"predictions/{seed}-{arm}.gz", prediction_sha256=digest(prediction))
            score_ref = put(evidence, f"scores/{seed}-{arm}.json", {
                "seed": seed, "arm": arm, "selected_epoch": 40, "freeze_sha256": freeze_sha,
                "protocol_sha256": recipe_sha, "prediction_sha256": digest(prediction),
                "execution_uuid": "cpu", **{name: .4 + (seed - 11) / 1000 + ARMS.index(arm) / 100 for name in ("AP50:95", "AP50", "AP75", "APs", "APm", "APl")}})
            receipt_refs[(seed, arm)] = (training_ref, score_ref, receipt_ref)
    freeze_ref = put(evidence, "freeze.json", freeze)
    # Re-serialize receipts/scores with the final freeze hash (freeze itself includes training hashes).
    members = []
    for seed in SEEDS:
        for arm in ARMS:
            tr, sc, rr = receipt_refs[(seed, arm)]
            score_path = evidence / sc["path"]
            score = json.loads(score_path.read_text()); score["freeze_sha256"] = freeze_ref["sha256"]
            sc = put(evidence, sc["path"], score)
            rec_path = evidence / rr["path"]
            rec = json.loads(rec_path.read_text()); rec["freeze_sha256"] = freeze_ref["sha256"]
            rr = put(evidence, rr["path"], rec)
            rr.update(prediction_path=f"predictions/{seed}-{arm}.gz",
                      prediction_sha256=digest(evidence / f"predictions/{seed}-{arm}.gz"))
            members.append({"seed": seed, "arm": arm, "training": tr, "score": sc, "prediction_receipt": rr})
    profiles = []
    for seed in (11, 22):
        for arm in ARMS:
            frozen = next(r for r in freeze["members"] if (r["seed"], r["arm"]) == (seed, arm))
            for measurement in ("inference", "training"):
                for batch in (1, 32):
                    p = {"measurement": measurement, "seed": seed, "arm": arm, "imgsz": 64,
                         "quantization": recipe["quantization"], "numeric_policy": recipe["numeric_policy"],
                         "model_identity": {"config_sha256": frozen["config_sha256"], "weights_sha256": frozen["weights_sha256"],
                                            "trunk_sha256": recipe["trunk_sha256"], "recipe_sha256": recipe_sha},
                         "producer": recipe["producer"], "assigned_uuid": gpu[str(seed)], "execution_uuid": gpu[str(seed)],
                         "checkpoint_files_unchanged": True, "measurement_inputs": {"batch_size": batch},
                         "portable_weight_files": {"head_plus_trunk_file_bytes": 1000}}
                    if measurement == "inference":
                        p["executed_conv_macs"] = {"conv_macs": 123}
                        p["inference_peak"] = {"peak_allocated_bytes": 456}
                        for region in ("full_detector", "head"):
                            p[region] = {"batch": batch, "dtype": "float32", "warmup": 5, "repetitions": 20,
                                         "resolution": [64, 64], "quantization_boundary": "FP32->FP16->FP32", "mean_ms": 2.0}
                    else:
                        p["training_peak"] = {"peak_allocated_bytes": 456}
                        p["training_protocol"] = {"batch_size": batch, "precision": "native BF16 head", "scope": "synthetic probe", "input": "same frozen cached batch"}
                    ref = put(evidence, f"profiles/{seed}-{arm}-{measurement}-{batch}.json", p)
                    profiles.append({"seed": seed, "arm": arm, "measurement": measurement, "batch": batch, "file": ref})
    manifest = {"evidence_root": "evidence", "protocol": protocol_ref, "freeze": freeze_ref,
                "members": members, "profile_seeds": [11, 22], "profiles": profiles}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path, manifest


def test_paired_stats_hand_oracle():
    result = paired_stats([1, 2, 3, 4, 5])
    assert result["mean_pp"] == 3
    assert result["sample_sd_pp"] == pytest.approx(math.sqrt(2.5))
    half = 2.7764451051977987 * math.sqrt(2.5 / 5)
    assert result["t95_df4_pp"] == pytest.approx([3 - half, 3 + half])


def test_synthetic_complete_matrix_figures_and_deterministic_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    manifest, _ = fixture(tmp_path)
    out1, out2 = tmp_path / "result-1", tmp_path / "result-2"
    summary = analyze(manifest, out1)
    assert summary["label"] == "SYNTHETIC"
    assert [x["name"] for x in summary["contrasts"]] == ["primary_shared_minus_independent", "secondary_shared_minus_native"]
    assert "not evidence of equivalence" in summary["caveat"]
    assert len(list(csv.DictReader((out1 / "paired_deltas.csv").open()))) == 10
    assert len(list(csv.DictReader((out1 / "curves.csv").open()))) == 600
    assert (out1 / "paired_deltas.svg").read_text().startswith("<?xml")
    assert (out1 / "curves.svg").read_text().startswith("<?xml")
    analyze(manifest, out2)
    for name in ("summary.json", "scores.csv", "paired_deltas.csv", "curves.csv", "profiles.csv", "paired_deltas.svg", "curves.svg"):
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes(), name
    with pytest.raises(FileExistsError):
        analyze(manifest, out1)


def test_rejects_missing_duplicate_and_hash_mismatch(tmp_path):
    manifest_path, manifest = fixture(tmp_path)
    manifest["members"].pop()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        analyze(manifest_path, tmp_path / "missing-out")
    manifest_path, manifest = fixture(tmp_path / "other")
    manifest["members"].append(manifest["members"][0])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        analyze(manifest_path, tmp_path / "duplicate-out")
    manifest_path, manifest = fixture(tmp_path / "third")
    manifest["members"][0]["score"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        analyze(manifest_path, tmp_path / "hash-out")


def test_rejects_policy_and_cohort_mismatch(tmp_path):
    manifest_path, manifest = fixture(tmp_path)
    profile_ref = manifest["profiles"][0]["file"]
    profile_path = tmp_path / "evidence" / profile_ref["path"]
    profile = json.loads(profile_path.read_text()); profile["checkpoint_files_unchanged"] = False
    profile_path.write_text(json.dumps(profile))
    profile_ref["sha256"] = digest(profile_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="profile identity/policy"):
        analyze(manifest_path, tmp_path / "policy-out")

    manifest_path, manifest = fixture(tmp_path / "again")
    receipt_ref = manifest["members"][1]["prediction_receipt"]
    receipt_path = tmp_path / "again/evidence" / receipt_ref["path"]
    receipt = json.loads(receipt_path.read_text()); receipt["image_set_sha256"] = "different"
    receipt_path.write_text(json.dumps(receipt))
    receipt_ref["sha256"] = digest(receipt_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="prediction receipts disagree"):
        analyze(manifest_path, tmp_path / "cohort-out")


def test_rejects_prediction_bytes_not_bound_by_receipt(tmp_path):
    manifest_path, manifest = fixture(tmp_path)
    ref = manifest["members"][0]["prediction_receipt"]
    prediction_path = tmp_path / "evidence" / ref["prediction_path"]
    prediction_path.write_bytes(gzip.compress(b'{"predictions":[],"synthetic":false}', mtime=0))
    ref["prediction_sha256"] = digest(prediction_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="prediction byte hash mismatch"):
        analyze(manifest_path, tmp_path / "prediction-out")


def test_rejects_symlinked_evidence_root(tmp_path):
    manifest_path, manifest = fixture(tmp_path)
    (tmp_path / "evidence-link").symlink_to(tmp_path / "evidence", target_is_directory=True)
    manifest["evidence_root"] = "evidence-link"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="evidence_root"):
        analyze(manifest_path, tmp_path / "symlink-root-out")


def test_rejects_symlinked_evidence_ancestor(tmp_path):
    manifest_path, manifest = fixture(tmp_path)
    evidence = tmp_path / "evidence"
    original = evidence / "profiles"
    linked = evidence / "linked-profiles"
    linked.symlink_to(original, target_is_directory=True)
    ref = manifest["profiles"][0]["file"]
    ref["path"] = ref["path"].replace("profiles/", "linked-profiles/", 1)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="linked"):
        analyze(manifest_path, tmp_path / "symlink-out")
