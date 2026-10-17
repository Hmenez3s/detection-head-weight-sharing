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
METRICS = ("AP50:95", "AP50", "AP75", "APs", "APm", "APl")


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
                "run_record_sha256": "", "config": f"runs/{seed}-{arm}/selected.json",
                "weights": f"runs/{seed}-{arm}/selected.pt", "config_sha256": f"c{seed}{arm}",
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
    inference_rows = [r for r in csv.DictReader((out1 / "profiles.csv").open()) if r["measurement"] == "inference"]
    assert inference_rows and all(r["warmup"] == "5" and r["repetitions"] == "20" for r in inference_rows)
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


def test_rejects_noncanonical_inference_timing_after_hash_validation(tmp_path):
    manifest_path, manifest = fixture(tmp_path)
    # Rewrite every referenced inference profile and rehash it so validation
    # reaches the measurement-policy checks rather than stopping at integrity.
    for item in manifest["profiles"]:
        if item["measurement"] != "inference":
            continue
        path = tmp_path / "evidence" / item["file"]["path"]
        profile = json.loads(path.read_text())
        profile["full_detector"].update(warmup=0, repetitions=1)
        profile["head"].update(warmup=0, repetitions=1)
        item["file"]["sha256"] = put(tmp_path / "evidence", item["file"]["path"], profile)["sha256"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inference profile policy mismatch"):
        analyze(manifest_path, tmp_path / "rejected-noncanonical-out")


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


def mixed_fixture(root):
    """Derive an isolated synthetic v2 evidence matrix from generated legacy rows."""
    manifest_path, manifest = fixture(root)
    evidence = root / "evidence"
    old_protocol_ref = manifest["protocol"]
    old_protocol = json.loads((evidence / old_protocol_ref["path"]).read_text())
    old_recipe = old_protocol["recipe"]
    old_recipe_sha = old_protocol["recipe_sha256"]
    old_freeze_ref = manifest["freeze"]
    old_freeze_path = evidence / old_freeze_ref["path"]
    old_freeze = json.loads(old_freeze_path.read_text())
    old_freeze.update(schema_version=1, protocol="protocol/frozen.json", synthetic=old_recipe["synthetic"],
                      trunk_sha256=old_recipe["trunk_sha256"], split_sha256=old_recipe.get("split_sha256"),
                      imgsz=old_recipe["imgsz"], batch_size=old_recipe["batch_size"],
                      quantization=old_recipe["quantization"], producer=old_recipe["producer"],
                      selection=old_recipe.get("selection"), policy=PREDICTION_POLICY)
    old_freeze_ref = put(evidence, "source/selection/selected.json", old_freeze)
    old_protocol_ref = put(evidence, "source/protocol/frozen.json", old_protocol)
    refs = {"protocol": {"path": old_protocol_ref["path"], "sha256": old_protocol_ref["sha256"], "recipe_sha256": old_recipe_sha},
            "freeze": {"path": old_freeze_ref["path"], "sha256": old_freeze_ref["sha256"]}}

    recipe = dict(old_recipe)
    recipe["arms"] = ["native", "loop_ab2", "loop_a3"]
    recipe["environment"] = {"study_id": "looped_v1", "looped_architecture": {"synthetic": True},
                              "reference_native": refs}
    recipe["pilot_evidence"] = {"new_looped_technical_runs": True}
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    protocol_ref = put(evidence, "looped/protocol.json", {"format": 1, "recipe": recipe, "recipe_sha256": recipe_sha})

    old_rows = {(x["seed"], x["arm"]): x for x in old_freeze["members"]}
    new_rows = []
    for seed in SEEDS:
        native = dict(old_rows[(seed, "native")])
        native["source"] = {"kind": "reused_native", "protocol_sha256": old_recipe_sha,
                            "freeze_sha256": old_freeze_ref["sha256"]}
        native["config"] = "source/" + native["config"]
        native["weights"] = "source/" + native["weights"]
        new_rows.append(native)
        for arm, old_arm in (("loop_ab2", "independent_k2"), ("loop_a3", "shared_k2")):
            cell = dict(old_rows[(seed, old_arm)])
            cell.update(arm=arm, source={"kind": "current", "protocol_sha256": recipe_sha})
            new_rows.append(cell)
    freeze = {"schema_version": 2, "protocol_sha256": recipe_sha, "protocol": protocol_ref["path"],
              "synthetic": True, "trunk_sha256": recipe["trunk_sha256"], "members": new_rows,
              "reference_native": refs}
    # Rewrite only new-fit run records; reused native records keep their old protocol identity.
    row_refs = {(r["seed"], r["arm"]): r for r in manifest["members"]}
    for seed in SEEDS:
        for arm, old_arm in (("loop_ab2", "independent_k2"), ("loop_a3", "shared_k2")):
            row = row_refs[(seed, old_arm)]
            training_path = evidence / row["training"]["path"]
            training = json.loads(training_path.read_text()); training.update(arm=arm, protocol_sha256=recipe_sha)
            row["training"] = put(evidence, row["training"]["path"], training)
            frozen = next(m for m in new_rows if (m["seed"], m["arm"]) == (seed, arm))
            frozen["run_record_sha256"] = row["training"]["sha256"]
    freeze_ref = put(evidence, "looped/freeze.json", freeze)

    members = []
    for seed in SEEDS:
        for arm in ("native", "loop_ab2", "loop_a3"):
            from_arm = "independent_k2" if arm == "loop_ab2" else "shared_k2" if arm == "loop_a3" else arm
            row = dict(row_refs[(seed, from_arm)])
            row.update(seed=seed, arm=arm)
            source_sha, source_freeze_sha = (old_recipe_sha, old_freeze_ref["sha256"]) if arm == "native" else (recipe_sha, freeze_ref["sha256"])
            for field in ("score", "prediction_receipt"):
                path = evidence / row[field]["path"]
                value = json.loads(path.read_text()); value.update(seed=seed, arm=arm, protocol_sha256=source_sha, freeze_sha256=source_freeze_sha)
                if field == "score":
                    for metric in ("AP50:95", "AP50", "AP75", "APs", "APm", "APl"):
                        value[metric] = (-1 if seed == 11 and arm == "native" and metric == "APs" else
                                         .25 + (seed - 11) / 1000 + (0 if arm == "native" else .01 if arm == "loop_ab2" else .02))
                row[field] = put(evidence, row[field]["path"], value)
                if field == "prediction_receipt":
                    row[field].update(prediction_path=f"predictions/{seed}-{from_arm}.gz",
                                      prediction_sha256=digest(evidence / f"predictions/{seed}-{from_arm}.gz"))
            members.append(row)

    profiles = []
    old_profiles = {(p["seed"], p["arm"], p["measurement"], p["batch"]): p for p in manifest["profiles"]}
    for seed in (11, 22):
        for arm, old_arm in (("native", "native"), ("loop_ab2", "independent_k2"), ("loop_a3", "shared_k2")):
            frozen = next(x for x in new_rows if (x["seed"], x["arm"]) == (seed, arm))
            for measurement in ("inference", "training"):
                for batch in (1, 32):
                    source_item = old_profiles[(seed, old_arm, measurement, batch)]
                    source_path = evidence / source_item["file"]["path"]
                    profile = json.loads(source_path.read_text())
                    profile.update(seed=seed, arm=arm, measurement=measurement, execution_uuid=None,
                                   assigned_uuid=recipe["gpu_for_seed"][str(seed)], branch_policy="retained_dual")
                    profile["model_identity"].update(config_sha256=frozen["config_sha256"], weights_sha256=frozen["weights_sha256"], recipe_sha256=recipe_sha)
                    profile["inference_peak"] = None
                    profile["storage"] = {"head": {"unique_parameters": 100, "unique_parameter_storage_bytes": 400,
                        "unique_parameter_and_buffer_storage_bytes": 500}}
                    profile["portable_weight_files"] = {"head_file_bytes": 1000, "common_trunk_file_bytes": 2000,
                        "head_plus_trunk_file_bytes": 3000, "scope": "synthetic generated fixture"}
                    profile["executed_conv_macs"] = {"conv_macs": 1234}
                    if measurement == "inference":
                        profile["full_detector"]["timing_label"] = "CPU non-study smoke only"
                        profile["head"]["timing_label"] = "CPU non-study smoke only"
                    profile.pop("training_peak", None)
                    profile_ref = put(evidence, f"looped/profiles/{seed}-{arm}-{measurement}-{batch}-dual.json", profile)
                    descriptor = {"seed": seed, "arm": arm, "measurement": measurement, "batch": batch,
                                  "branch_policy": "retained_dual", "file": profile_ref}
                    profiles.append(descriptor)
                    if measurement == "inference":
                        fused = dict(profile); fused["branch_policy"] = "fused_one2one"
                        fused["storage"] = {**profile.get("storage", {}), "head": {**profile.get("storage", {}).get("head", {}), "unique_parameters": 10}}
                        fused_ref = put(evidence, f"looped/profiles/{seed}-{arm}-{measurement}-{batch}-fused.json", fused)
                        profiles.append({**descriptor, "branch_policy": "fused_one2one", "file": fused_ref})
    manifest.update(protocol=protocol_ref, freeze=freeze_ref, members=members, profiles=profiles)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path, manifest


def test_mixed_reference_matrix_outputs_all_metrics_curves_and_three_contrasts(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    manifest_path, manifest = mixed_fixture(tmp_path)
    out = tmp_path / "mixed-results"
    summary = analyze(manifest_path, out)
    assert summary["study"] == "looped_v1"
    assert summary["matrix"]["arms"] == ["native", "loop_ab2", "loop_a3"]
    assert [x["name"] for x in summary["contrasts"]] == ["primary_loop_ab2_minus_native", "primary_loop_a3_minus_native", "secondary_loop_a3_minus_loop_ab2"]
    for actual, expected in zip((x["deltas_pp"] for x in summary["contrasts"]), ([1.0] * 5, [2.0] * 5, [1.0] * 5)):
        assert actual == pytest.approx(expected)
    assert all(0 <= x['p_two_sided'] < 1e-50 for x in summary['contrasts'])
    assert all(0 <= x['p_holm'] < 1e-50 for x in summary['contrasts'][:2])
    assert len(summary["values"]) == 15 and all(set(METRICS) <= set(row) for row in summary["values"])
    unavailable = next(row for row in summary["values"] if row["seed"] == 11 and row["arm"] == "native")
    assert unavailable["APs"] == -1 and unavailable["APs_available"] is False
    assert len(list(csv.DictReader((out / "curves.csv").open()))) == 600
    curve = next(row for row in csv.DictReader((out / "curves.csv").open()) if row["epoch"] == "5")
    assert float(curve["probe_minus_dev_ap50_95"]) == pytest.approx(-.02)
    profiles = list(csv.DictReader((out / "profiles.csv").open()))
    assert len(profiles) == 36
    assert {x["branch_policy"] for x in profiles} == {"retained_dual", "fused_one2one"}
    assert {"peak_allocated_bytes", "peak_reserved_bytes", "head_conv_macs", "mean_head_ms", "mean_full_ms"} <= set(profiles[0])
    native = next(row for row in summary["values"] if row["arm"] == "native")
    current = next(row for row in summary["values"] if row["arm"] == "loop_ab2")
    assert native["source_kind"] == "reused_native" and current["source_kind"] == "current"
    assert native["source_protocol_sha256"] != current["source_protocol_sha256"]
    native_profile = next(p for p in manifest["profiles"] if p["seed"] == 11 and p["arm"] == "native" and
                          p["measurement"] == "inference" and p["branch_policy"] == "retained_dual")
    native_profile_json = json.loads((tmp_path / "evidence" / native_profile["file"]["path"]).read_text())
    new_recipe_sha = json.loads((tmp_path / "evidence" / manifest["protocol"]["path"]).read_text())["recipe_sha256"]
    assert native_profile_json["model_identity"]["recipe_sha256"] == new_recipe_sha
    frozen_native = next(row for row in json.loads((tmp_path / "evidence" / manifest["freeze"]["path"]).read_text())["members"]
                         if row["seed"] == 11 and row["arm"] == "native")
    assert native_profile_json["model_identity"]["weights_sha256"] == frozen_native["weights_sha256"]
    assert "not independent untouched confirmation" in " ".join(summary["caveats"])
    assert (out / "paired_deltas.svg").read_text().count("primary_loop") >= 1


def test_mixed_reference_tampering_rejected_before_output(tmp_path):
    for name in ("source", "cohort", "device", "policy", "count"):
        case = tmp_path / name
        path, manifest = mixed_fixture(case)
        evidence = case / "evidence"
        if name == "source":
            protocol_path = evidence / manifest["protocol"]["path"]
            protocol = json.loads(protocol_path.read_text())
            protocol["recipe"]["environment"]["reference_native"]["freeze"]["sha256"] = "0" * 64
            protocol["recipe_sha256"] = hashlib.sha256(json.dumps(protocol["recipe"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            protocol_path.write_text(json.dumps(protocol))
            manifest["protocol"]["sha256"] = digest(protocol_path)
        elif name == "cohort":
            ref = manifest["members"][1]["prediction_receipt"]
            receipt_path = evidence / ref["path"]
            receipt = json.loads(receipt_path.read_text()); receipt["image_set_sha256"] = "different-cohort"
            receipt_path.write_text(json.dumps(receipt)); ref["sha256"] = digest(receipt_path)
        elif name == "device":
            ref = manifest["profiles"][0]["file"]
            profile_path = evidence / ref["path"]
            profile = json.loads(profile_path.read_text()); profile["assigned_uuid"] = "wrong-device"
            profile_path.write_text(json.dumps(profile)); ref["sha256"] = digest(profile_path)
        elif name == "policy":
            manifest["profiles"][0]["branch_policy"] = "fused_one2one"
        else:
            manifest["profiles"].pop()
        path.write_text(json.dumps(manifest))
        output = case / "must-not-exist"
        with pytest.raises((ValueError, FileNotFoundError)):
            analyze(path, output)
        assert not output.exists()


def test_analysis_import_does_not_load_ml_runtime():
    import os
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c",
        "import sys; import head_sharing.analyze; assert not any(name == 'torch' or name.startswith('torch.') or name == 'ultralytics' or name.startswith('ultralytics.') for name in sys.modules)"],
        env=os.environ.copy(), check=False, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_df4_p_values_holm_zero_variance_and_tail_oracle():
    from head_sharing.analyze import holm_adjust, _t_two_sided_p_df4
    assert _t_two_sided_p_df4(1e8) == pytest.approx(6e-32, rel=1e-14)
    assert paired_stats([0] * 5)["p_two_sided"] == 1
    assert paired_stats([2] * 5)["p_two_sided"] == 0
    t95 = 2.7764451051977987
    values = [t95 + x * math.sqrt(2) for x in (-2, -1, 0, 1, 2)]
    assert paired_stats(values)["p_two_sided"] == pytest.approx(.05, abs=1e-14)
    assert paired_stats([-x for x in values])["p_two_sided"] == pytest.approx(.05, abs=1e-14)
    assert holm_adjust([.01, .04]) == pytest.approx([.02, .04])
    assert holm_adjust([.04, .01]) == pytest.approx([.04, .02])


def test_actual_looped_profile_cli_record_matches_synthetic_consumer_contract(tmp_path):
    import hashlib
    import torch
    from head_sharing.analyze import _validate_synthetic_profile_producer
    from head_sharing.checkpoint import save_checkpoint
    from head_sharing.model import HeadStudyModel
    from head_sharing.protocol import LOOPED_ARCHITECTURE, NUMERIC_POLICY, SELECTION, freeze
    from head_sharing.runtime import GPU_UUIDS

    model = HeadStudyModel("loop_a3", seed=11, imgsz=64)
    trunk_state = {k: v.detach().cpu().clone() for k, v in model.detector.state_dict().items()
                   if k.startswith("model.") and int(k.split(".")[1]) < 23}
    trunk = tmp_path / "trunk.pt"
    torch.save({"schema_version": 1, "architecture": "yolo26n", "upstream": "8.4.177",
                "excluded_head_index": 23, "state_dict": trunk_state}, trunk)
    trunk_sha = "f01ec41445fac9ef7078b48ef1b73473e6314bcb1a77e77c20666d6ecfcd9016"
    weights, config = tmp_path / "selected.pt", tmp_path / "selected.json"
    trunk_receipt = {"trunk_artifact_sha256": trunk_sha, "trunk_tensors": 468,
                     "trunk_bytes": 9109168, "head_index": 23, "input_channels": [64, 128, 256],
                     "source_variant": "U", "source_seed": 11, "source_epoch_zero_based": 79,
                     "source_training_epochs": 80, "pre_exposure": "synthetic generated topology fixture"}
    save_checkpoint(model, weights, config, trunk_receipt)

    root = tmp_path / "study"
    root.mkdir(); (root / "STUDY_OWNER.txt").write_text("generated synthetic fixture\n")
    source_recipe = {"status":"FROZEN", "seed_list":[11,22,33,44,55],
        "arms":["native","independent_k2","shared_k2"], "imgsz":64,"batch_size":2,"epochs":2,
        "cadence":1,"block_size":2,"probe_size":2,"optimizer":{"name":"SGD","lr":.01,"momentum":.9,"weight_decay":.0005},
        "gains":{"box":7.5,"cls":.5,"dfl":1.5},"quantization":"fp32_fp16_fp32_v1",
        "producer":GPU_UUIDS[1],"environment":{"device":"CPU synthetic"},"trunk_sha256":trunk_sha,
        "split_sha256":"a"*64,"cache_evidence":{"synthetic":True},"pilot_evidence":{"synthetic":True},
        "synthetic":True,"gpu_for_seed":{str(s):GPU_UUIDS[1] for s in (11,22,33,44,55)},
        "selection":SELECTION,"numeric_policy":NUMERIC_POLICY}
    source_recipe_sha = hashlib.sha256(json.dumps(source_recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source_protocol = root / "source-protocol.json"
    source_protocol.write_text(json.dumps({"format":1,"recipe":source_recipe,"recipe_sha256":source_recipe_sha}))
    source_freeze = root / "source-freeze.json"
    source_freeze.write_text(json.dumps({"schema_version":1,"protocol_sha256":source_recipe_sha}))
    refs = {"protocol":{"path":source_protocol.name,"sha256":hashlib.sha256(source_protocol.read_bytes()).hexdigest(),
                         "recipe_sha256":source_recipe_sha},
            "freeze":{"path":source_freeze.name,"sha256":hashlib.sha256(source_freeze.read_bytes()).hexdigest()}}
    recipe = dict(source_recipe); recipe["arms"] = ["native","loop_ab2","loop_a3"]
    recipe["environment"] = {"study_id":"looped_v1","looped_architecture":LOOPED_ARCHITECTURE,"reference_native":refs}
    recipe_path = tmp_path / "looped-recipe.json"; protocol_path = root / "looped-protocol.json"
    recipe_path.write_text(json.dumps(recipe)); freeze(recipe_path, protocol_path, synthetic=True)
    output = tmp_path / "profile.json"
    from head_sharing.profile import main
    assert main(["--device","cpu","--synthetic","--measurement","inference","--weights",str(weights),
        "--config",str(config),"--trunk-artifact",str(trunk),"--protocol",str(protocol_path),
        "--imgsz","64","--batch-size","1","--warmup","5","--repetitions","20","--output",str(output)]) == 0
    actual = json.loads(output.read_text())
    _validate_synthetic_profile_producer(actual, {"branch_policy":"retained_dual"})
    assert actual['arm'] == 'loop_a3'
    assert actual["execution_uuid"] is None and actual["inference_peak"] is None
    assert actual["assigned_uuid"] == GPU_UUIDS[1]
    assert actual["branch_policy"] == "retained_dual"
