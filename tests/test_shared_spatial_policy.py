import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "shared_spatial_private", ROOT / ".internal" / "shared_spatial.py")
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


def _old_recipe():
    from head_sharing.protocol import NUMERIC_POLICY, SELECTION
    return {"status": "FROZEN", "seed_list": [11, 22, 33, 44, 55],
            "arms": ["native", "independent_k2", "shared_k2"], "imgsz": 512, "batch_size": 32, "epochs": 40,
            "cadence": 5, "block_size": 32, "probe_size": 1000,
            "optimizer": {"name": "SGD", "lr": .01, "momentum": .9, "weight_decay": .0005},
            "gains": {"box": 7.5, "cls": .5, "dfl": 1.5},
            "quantization": "fp32_fp16_fp32_v1", "producer": policy.G1_UUID,
            "environment": {"torch_version": "fixture", "source_git_commit": "a09b8e49cc9ed7401aae3951b3dee0e632dbf1db",
                            "cache_source_git_commit": "749642b8063825c38cc6d64d26e30d3a313821e1",
                            "cumulative_gpu_cap_seconds_per_uuid": 172800,
                            "cpus_per_fit": 3, "job_allocation_seconds": {"11": 43200}},
            "trunk_sha256": policy.TRUNK_ARTIFACT_SHA256, "split_sha256": "b" * 64,
            "cache_evidence": {name: {"metadata_file_sha256": "c" * 64}
                               for name in ("few", "dev", "probe")},
            "pilot_evidence": {"source": "verified"}, "synthetic": False,
            "gpu_for_seed": {str(seed): policy.G1_UUID for seed in policy.SEEDS},
            "selection": SELECTION, "numeric_policy": NUMERIC_POLICY}


def _source_protocol(recipe):
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"path": "source/protocol.json", "sha256": "d" * 64, "recipe_sha256": recipe_sha}


def _recipes(old=None):
    old = _old_recipe() if old is None else old
    return policy.prepare_recipe(old, source_protocol=_source_protocol(old),
        source_freeze={"path": "source/freeze.json", "sha256": "f" * 64},
        source_git_commit="1234567890abcdef1234567890abcdef12345678")


def _receipt():
    return {"trunk_artifact_sha256": policy.TRUNK_ARTIFACT_SHA256, "trunk_tensors": 468,
            "trunk_bytes": 9_109_168, "head_index": 23, "input_channels": [64, 128, 256],
            "source_variant": "U", "source_seed": 11, "source_epoch_zero_based": 79,
            "source_training_epochs": 80, "pre_exposure": "synthetic fixture provenance"}


def _row(variant, seed, ap, *, probe=None):
    spec = None if variant == "native" else {"version": 2, "variant": variant}
    return {"variant": variant, "seed": seed, "epochs": 40, "head_spec": spec,
            "status": "completed", "closed": True, "dev_ap": ap,
            "probe_ap": ap - .01 if probe is None else probe, "selected_epoch": 35,
            "training": {"path": f"{variant}-{seed}/training.json", "sha256": "a" * 64},
            "selected_config": {"path": f"{variant}-{seed}/selected-e35.json", "sha256": "b" * 64},
            "selected_weights": {"path": f"{variant}-{seed}/selected-e35.pt", "sha256": "c" * 64},
            "recipe_sha256": "d" * 64}


def _screen(candidate_scores=None):
    candidate_scores = candidate_scores or {}
    rows = []
    for variant in policy.VARIANTS:
        for seed in policy.SCREEN_SEEDS:
            ap = .20 if variant == "native" else candidate_scores.get((variant, seed), .20)
            rows.append(_row(variant, seed, ap))
    return rows


def _failed(variant, seed, **changes):
    return {"variant": variant, "seed": seed, "status": "failed", "closed": True,
            "epochs": 40, "head_spec": None if variant == "native" else {"version": 2, "variant": variant},
            "dev_ap": None, "probe_ap": None, "selected_dev_ap": None,
            "epochs_completed": 9, "failure_reason": "synthetic failure",
            "failure_evidence": {"path": "owned/failure.json", "sha256": "e" * 64}} | changes


def test_prepares_six_full_recipes_without_mutating_or_inheriting_old_selection():
    source = _old_recipe()
    before = json.loads(json.dumps(source))
    recipes = _recipes(source)
    assert tuple(recipes) == policy.VARIANTS
    assert source == before
    for name, recipe in recipes.items():
        assert recipe["seed_list"] == list(policy.SEEDS) and recipe["arms"] == ["native"]
        assert (recipe["epochs"], recipe["cadence"], recipe["batch_size"], recipe["imgsz"]) == (40, 5, 32, 512)
        assert recipe["gpu_for_seed"] == {str(seed): policy.G1_UUID for seed in policy.SEEDS}
        assert recipe["environment"]["study_id"] == "shared_spatial_v1"
        assert recipe["environment"]["head_spec"] == (None if name == "native" else {"version": 2, "variant": name})
        assert recipe["environment"]["torch_version"] == "fixture"
        assert recipe["environment"]["source_git_commit"] == "1234567890abcdef1234567890abcdef12345678"
        assert recipe["environment"]["cache_source_git_commit"] == source["environment"]["cache_source_git_commit"]
        assert "cumulative_gpu_cap_seconds_per_uuid" not in recipe["environment"]
        assert recipe["environment"]["cpus_per_fit"] == source["environment"]["cpus_per_fit"]
    with pytest.raises(ValueError, match="source protocol recipe hash"):
        policy.prepare_recipe(source,
            source_protocol={"path": "source/protocol.json", "sha256": "d" * 64, "recipe_sha256": "e" * 64},
            source_freeze={"path": "source/freeze.json", "sha256": "f" * 64},
            source_git_commit="1234567890abcdef1234567890abcdef12345678")
    with pytest.raises(ValueError, match="normalized and relative"):
        policy.prepare_recipe(source,
            source_protocol={"path": "../protocol.json", "sha256": "d" * 64, "recipe_sha256": "e" * 64},
            source_freeze={"path": "source/freeze.json", "sha256": "f" * 64},
            source_git_commit="1234567890abcdef1234567890abcdef12345678")
    wrong_source = dict(source, producer="other")
    with pytest.raises(ValueError, match="G1"):
        policy.prepare_recipe(wrong_source,
            source_protocol=_source_protocol(wrong_source),
            source_freeze={"path": "f", "sha256": "f" * 64},
            source_git_commit="1234567890abcdef1234567890abcdef12345678")


def test_screen_requires_all_twelve_and_selects_only_positive_paired_candidates():
    with pytest.raises(ValueError, match="twelve"):
        policy.decide_stages(_screen()[:-1])
    scores = {("spatial_sepbn", 11): .202, ("spatial_sepbn", 33): .202,
              ("spatial_gn", 11): .202, ("spatial_gn", 33): .202,
              ("spatial_untied", 11): .9, ("spatial_untied", 33): .9}
    decision = policy.decide_stages(_screen(scores))
    assert decision["winner"] == "spatial_sepbn"  # fixed tie order; untied is descriptive only
    assert len(decision["screen"]) == 12 and len(decision["followup_cells"]) == 6
    assert decision["final_inference_authorized"] is False
    no_gain = policy.decide_stages(_screen())
    assert no_gain["winner"] is None and no_gain["followup_cells"] == []
    with pytest.raises(ValueError, match="unstarted"):
        policy.decide_stages(_screen()[:-1] + [{"variant": "spatial_sepbn_sharedpred", "seed": 33, "status": "unstarted"}])
    duplicate = _screen()
    duplicate[-1] = dict(duplicate[0])
    with pytest.raises(ValueError, match="duplicate"):
        policy.decide_stages(duplicate)
    failed_screen = _screen()
    failed_screen[1] = _failed("native", 33)
    assert policy.decide_stages(failed_screen)["winner"] is None


def test_followups_are_fixed_and_five_pair_stats_are_reused_without_final_authority():
    scores = {("spatial_gn", 11): .22, ("spatial_gn", 33): .22}
    initial = policy.decide_stages(_screen(scores))
    followups = [_row(name, seed, .21 if name == "native" else .22)
                 for seed in (22, 44, 55) for name in ("native", "spatial_gn")]
    complete = policy.decide_followups(initial, followups)
    assert complete["complete_five_pair_group"] is True
    assert complete["final_inference_authorized"] is False
    with pytest.raises(ValueError, match="non-winner"):
        policy.decide_followups(initial, followups[:-1] + [_row("spatial_sepbn", 55, .99)])
    report = policy.render_dev_report({"screens": _screen(scores)}, complete)
    assert "private development report" in report
    assert "+0.15 AP-point target was an aspiration" in report
    assert "descriptive t95(df=4) interval" in report
    assert "Final inference authority: NONE" in report and "Holdout status: CLOSED" in report
    failed55 = _failed("spatial_gn", 55, epochs_completed=40,
                       failure_reason="presentation failed after the final epoch",
                       failure_evidence={"path": "owned/seed55-failure.json", "sha256": "9" * 64})
    incomplete = policy.decide_followups(initial, followups[:-1] + [failed55])
    incomplete_report = policy.render_dev_report({"screens": _screen(scores)}, incomplete)
    assert "five-pair group is incomplete" in incomplete_report
    assert "Mean paired DEV delta" not in incomplete_report
    assert "t95(df=4)" not in incomplete_report
    assert "presentation failed after the final epoch" in incomplete_report
    assert "owned/seed55-failure.json" in incomplete_report and "9" * 64 in incomplete_report
    assert "native / seed 22: DEV AP50:95" in incomplete_report
    assert "spatial_gn / seed 22: DEV AP50:95" in incomplete_report
    assert "native / seed 44: DEV AP50:95" in incomplete_report
    assert "spatial_gn / seed 44: DEV AP50:95" in incomplete_report


def test_sealed_training_projection_binds_full_recipe_history_and_selected_files(tmp_path):
    recipe = _recipes()["native"]
    history = []
    for epoch in range(1, 41):
        row = {"epoch": epoch}
        if epoch % 5 == 0:
            score = .3 if epoch == 35 else .2
            row.update(dev={"AP50:95": score}, probe={"AP50:95": score - .01})
        history.append(row)
    config = {"format": 1, "architecture": "yolo26n", "arm": "native", "seed": 11,
              "imgsz": 512, "quantization": "fp32_fp16_fp32_v1", "trunk_receipt": _receipt()}
    config_path = tmp_path / "selected-e35.json"
    config_bytes = json.dumps(config, sort_keys=True).encode()
    config_path.write_bytes(config_bytes)
    weights_path = tmp_path / "selected-e35.pt"
    weights_path.write_bytes(b"synthetic placeholder bytes; not a tensor")
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    record = {"schema_version": 1, "purpose": "main", "synthetic": False, "arm": "native", "seed": 11,
              "selection_metric": "dev.AP50:95", "target_epochs": 40, "epochs_completed": 40,
              "protocol_sha256": recipe_sha, "recipe": recipe, "head_spec": None,
              "head_spec_sha256": policy.head_spec_digest(None),
              "history": history, "selected_epoch": 35, "selected_dev_ap": .3,
              "selected_config": "selected-e35.json", "head_updated": True,
              "hardware": {"device_uuid": policy.G1_UUID}, "frozen_trunk_digest": "f" * 64}
    record_path = tmp_path / "training.json"
    record_bytes = json.dumps(record, sort_keys=True).encode()
    record_path.write_bytes(record_bytes)
    projected = policy.project_training_record(record_path, hashlib.sha256(record_bytes).hexdigest(),
        recipe=recipe, seed=11,
        selected_config={"path": str(config_path), "sha256": hashlib.sha256(config_bytes).hexdigest()},
        selected_weights={"path": str(weights_path), "sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest()})
    assert projected["closed"] and projected["selected_epoch"] == 35 and projected["dev_ap"] == .3
    wrong = dict(record, selected_epoch=40)
    wrong_path = tmp_path / "wrong.json"
    wrong_bytes = json.dumps(wrong, sort_keys=True).encode()
    wrong_path.write_bytes(wrong_bytes)
    with pytest.raises(ValueError, match="earliest best"):
        policy.project_training_record(wrong_path, hashlib.sha256(wrong_bytes).hexdigest(),
            recipe=recipe, seed=11,
            selected_config={"path": str(config_path), "sha256": hashlib.sha256(config_bytes).hexdigest()},
            selected_weights={"path": str(weights_path), "sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest()})
    with pytest.raises(ValueError, match="selected artifact path"):
        policy.project_training_record(record_path, hashlib.sha256(record_bytes).hexdigest(), recipe=recipe, seed=11,
            selected_config={"path": str(tmp_path / "forged.json"), "sha256": "0" * 64},
            selected_weights={"path": str(weights_path), "sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest()})


def test_report_refuses_unstarted_or_fabricated_screen_decisions():
    rows = _screen()
    decision = policy.decide_stages(rows)
    rows[0] = rows[0] | {"status": "unstarted"}
    with pytest.raises(ValueError, match="selection-bound"):
        policy.render_dev_report({"screens": rows}, decision)
    with pytest.raises(ValueError, match="intact"):
        policy.render_dev_report({"screens": _screen()}, decision | {"final_inference_authorized": True})


def test_helper_has_no_native_runtime_or_live_operation_imports():
    source = (ROOT / ".internal" / "shared_spatial.py").read_text()
    assert "import torch" not in source and "from torch" not in source
    assert "subprocess" not in source and "socket" not in source


def test_v2_selected_config_projects_exact_spatial_spec_without_tensor_loading(tmp_path):
    recipe = _recipes()["spatial_gn"]
    spec = {"version": 2, "variant": "spatial_gn"}
    history = []
    for epoch in range(1, 41):
        row = {"epoch": epoch}
        if epoch % 5 == 0:
            row.update(dev={"AP50:95": .25}, probe={"AP50:95": .24})
        history.append(row)
    config = {"format": 2, "architecture": "yolo26n", "arm": "native", "seed": 33,
              "imgsz": 512, "quantization": "fp32_fp16_fp32_v1", "head_spec": spec,
              "trunk_receipt": _receipt()}
    config_path = tmp_path / "selected-e5.json"
    config_bytes = json.dumps(config, sort_keys=True).encode()
    config_path.write_bytes(config_bytes)
    weights_path = tmp_path / "selected-e5.pt"
    weights_path.write_bytes(b"synthetic placeholder bytes; not a tensor")
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    record = {"schema_version": 1, "purpose": "main", "synthetic": False, "arm": "native", "seed": 33,
              "selection_metric": "dev.AP50:95", "target_epochs": 40, "epochs_completed": 40,
              "protocol_sha256": recipe_sha, "recipe": recipe, "head_spec": spec,
              "head_spec_sha256": policy.head_spec_digest(spec), "history": history,
              "selected_epoch": 5, "selected_dev_ap": .25, "selected_config": "selected-e5.json",
              "head_updated": True, "hardware": {"device_uuid": policy.G1_UUID},
              "frozen_trunk_digest": "f" * 64}
    record_path = tmp_path / "training.json"
    record_bytes = json.dumps(record, sort_keys=True).encode()
    record_path.write_bytes(record_bytes)
    projected = policy.project_training_record(record_path, hashlib.sha256(record_bytes).hexdigest(),
        recipe=recipe, seed=33,
        selected_config={"path": str(config_path), "sha256": hashlib.sha256(config_bytes).hexdigest()},
        selected_weights={"path": str(weights_path), "sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest()})
    assert projected["variant"] == "spatial_gn" and projected["head_spec"] == spec
    assert projected["head_spec_sha256"] == policy.head_spec_digest(spec)
