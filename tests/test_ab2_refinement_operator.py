import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ab2_refinement_private", ROOT / ".internal" / "ab2_refinement.py")
operator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(operator)


def _screen(variant, score, macs):
    dose = 80 if variant == "baseline80" else 40
    head_spec = None if variant == "baseline80" else {
        "version": 1, "variant": variant, "class_inner": 96}
    return {"variant": variant, "head_spec": head_spec, "epochs": dose, "seed": 11,
            "dev_ap": score, "conv_macs": macs, "closed": True,
            "probe_ap": score - .01, "selected_epoch": dose,
            "training": {"sha256": "a" * 64}}


def _old_recipe():
    from head_sharing.protocol import NUMERIC_POLICY, SELECTION
    from head_sharing.runtime import GPU_UUIDS
    return {"status": "FROZEN", "seed_list": [11, 22, 33, 44, 55],
            "arms": ["native", "loop_ab2", "loop_a3"], "imgsz": 512,
            "batch_size": 32, "epochs": 40, "cadence": 5, "block_size": 32,
            "probe_size": 1000,
            "optimizer": {"name": "SGD", "lr": .01, "momentum": .9, "weight_decay": .0005},
            "gains": {"box": 7.5, "cls": .5, "dfl": 1.5},
            "quantization": "fp32_fp16_fp32_v1", "producer": GPU_UUIDS[1],
            "environment": {"study_id": "looped_v1", "looped_architecture": "old",
                "reference_native": {"fixture": True}, "torch_version": "2.9.1",
                "cuda_version": "fixture", "source_git_commit": "old-commit"},
            "trunk_sha256": "a" * 64, "split_sha256": "b" * 64,
            "cache_evidence": {name: {"metadata_file_sha256": "c" * 64} for name in ("few", "dev", "probe")},
            "pilot_evidence": {"fixture": "synthetic"}, "synthetic": False,
            "gpu_for_seed": {str(seed): GPU_UUIDS[1] for seed in (11, 22, 33, 44, 55)},
            "selection": SELECTION, "numeric_policy": NUMERIC_POLICY}


def test_raw_ap_selection_strict_improvement_and_stable_ties():
    rows = [_screen("baseline80", .17, 700), _screen("learned_alpha", .18, 600),
            _screen("step_norm", .18, 500), _screen("scale_gain", .16, 400),
            _screen("dw5", .15, 800)]
    selected = operator.decide_stages(rows)
    assert selected["winner"] == "step_norm"  # MAC precedes registered variant order.
    assert selected["permitted_confirmations"] == ["step_norm:seed33", "step_norm:seed55"]
    assert not selected["cost_admitted"]
    dose80 = [_screen(name, .2 if name == "baseline80" else .1, i + 1)
              for i, name in enumerate(operator.VARIANTS)]
    long_winner = operator.decide_stages(dose80)
    assert long_winner["permitted_confirmations"] == ["baseline80:seed33", "baseline80:seed55"]
    assert not long_winner["cost_admitted"]
    assert operator.decide_stages([_screen(name, operator.BASELINE11_AP, i + 1)
                                   for i, name in enumerate(operator.VARIANTS)])["selected_variant"] is None


def test_closed_screen_and_confirmation_matrix_are_enforced():
    rows = [_screen(name, score, index + 1) for index, (name, score) in enumerate(
        [("baseline80", .1), ("learned_alpha", .2), ("step_norm", .15), ("scale_gain", .14), ("dw5", .13)])]
    with pytest.raises(ValueError, match="five"):
        operator.decide_stages(rows[:4])
    with pytest.raises(ValueError, match="single-winner"):
        operator.decide_stages(rows, confirms=[{"variant": "dw5", "seed": 33, "epochs": 40,
                                                 "closed": True, "dev_ap": .2}])
    confirm = [{"variant": "learned_alpha", "head_spec": {"version": 1, "variant": "learned_alpha", "class_inner": 96},
                "seed": seed, "epochs": 40, "closed": True, "dev_ap": .19} for seed in (33, 55)]
    result = operator.decide_stages(rows, confirms=confirm, baseline40_mean=.15,
                                    fresh_main_seconds=6500, required_margin_seconds=100)
    assert result["cost_admitted"]
    blocked = operator.decide_stages(rows, confirms=confirm, baseline40_mean=.15,
                                     fresh_main_seconds=5499, required_margin_seconds=100)
    assert not blocked["cost_admitted"] and any("budget" in skip for skip in blocked["skips"])
    wrong_width = [_screen(row["variant"], row["dev_ap"], row["conv_macs"]) for row in rows]
    wrong_width[1]["head_spec"]["class_inner"] = 48
    with pytest.raises(ValueError, match="head_spec"):
        operator.decide_stages(wrong_width)
    wrong_ap = [_screen(row["variant"], row["dev_ap"], row["conv_macs"]) for row in rows]
    wrong_ap[1]["dev_ap"] = True
    with pytest.raises(ValueError, match="AP"):
        operator.decide_stages(wrong_ap)


def test_sealed_json_hash_and_report_are_honest(tmp_path):
    payload = {"fixture": "synthetic only", "score": .12}
    file = tmp_path / "fixture.json"
    content = json.dumps(payload, sort_keys=True).encode()
    file.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    assert operator.read_sealed_json(file, digest) == payload
    with pytest.raises(ValueError, match="SHA256"):
        operator.read_sealed_json(file, "0" * 64)

    screens = [_screen(name, .12 + i / 100, 100 + i) for i, name in enumerate(operator.VARIANTS)]
    decision = operator.decide_stages(screens)
    report = operator.render_dev_report({"screens": screens, "balances": {"main_seconds": 123},
                                         "throughput": {"fits_per_hour": 0.5},
                                         "diagnostics": [], "profiles": [], "source_refs": [],
                                         "cost": {"seed": 11, "class_inner": 48, "status": "SKIP"}}, decision)
    assert "development-only" in report
    assert "does not contain a complete n=3 candidate group" in report
    assert "All candidates used one frozen trunk; DEV data were reused" in report
    assert "No new COCO val2017 holdout evaluation was performed" in report
    assert "80-epoch baseline" in report
    assert "Separate exploratory class_inner48 cost-stage result" in report
    assert f"{operator.BASELINE11_AP:.17f}" in report
    followups = [_screen("dw5", .17, 104) | {"seed": seed} for seed in (33, 55)]
    selected = operator.decide_stages(screens, confirms=followups)
    complete_report = operator.render_dev_report({"screens": screens}, selected)
    assert "exploratory n=3 candidate study" in complete_report
    assert "not independent confirmation" in complete_report


def test_recipe_preparation_preserves_environment_and_does_not_mutate_input():
    old = _old_recipe()
    original = json.loads(json.dumps(old))
    recipe = operator.prepare_recipe(
        old, head_spec=None, epochs=80, source_protocol=operator.PINNED_PROTOCOL,
        source_freeze=operator.PINNED_FREEZE, producer=old["producer"],
        cache_evidence=old["cache_evidence"], pilot_evidence={"fixture": "synthetic"},
        environment_updates={"source_git_commit": "new-commit", "admission": {"slots": 1}})
    assert recipe["epochs"] == 80 and recipe["seed_list"] == [11, 33, 55]
    assert recipe["arms"] == ["loop_ab2"] and recipe["optimizer"] == old["optimizer"]
    assert recipe["environment"]["head_spec"] is None
    assert "looped_architecture" not in recipe["environment"]
    assert "reference_native" not in recipe["environment"]
    assert recipe["environment"]["torch_version"] == old["environment"]["torch_version"]
    assert recipe["environment"]["source_git_commit"] == "new-commit"
    assert recipe["environment"]["admission"] == {"slots": 1}
    assert old == original
    with pytest.raises(ValueError, match="protected"):
        operator.prepare_recipe(old, head_spec=None, epochs=40,
            source_protocol=operator.PINNED_PROTOCOL, source_freeze=operator.PINNED_FREEZE,
            producer=old["producer"], cache_evidence=old["cache_evidence"],
            pilot_evidence={"fixture": True}, environment_updates={"head_spec": {"bad": True}})


def _sealed_training_fixture(tmp_path, *, selected_epoch=35, incomplete=False):
    spec = {"version": 1, "variant": "learned_alpha", "class_inner": 96}
    old = _old_recipe()
    recipe = operator.prepare_recipe(old, head_spec=spec, epochs=40,
        source_protocol=operator.PINNED_PROTOCOL, source_freeze=operator.PINNED_FREEZE,
        producer=old["producer"], cache_evidence=old["cache_evidence"],
        pilot_evidence={"fixture": "metadata only"})
    history = []
    for epoch in range(1, 41):
        row = {"epoch": epoch, "loss": 1.0}
        if epoch % 5 == 0:
            # Producer-shaped cadence: no DEV/PROBE objects on non-evaluation epochs.
            score = .21 if epoch == 35 else .15 if epoch == 40 else .10 + epoch / 1000
            row["dev"] = {"AP50:95": score}
            row["probe"] = {"AP50:95": score - .01}
        history.append(row)
    if incomplete:
        history.pop()
    selected_name = f"selected-e{selected_epoch}.json"
    score = history[selected_epoch - 1]["dev"]["AP50:95"]
    config_path = tmp_path / selected_name
    config = {"format": 2, "arm": "loop_ab2", "seed": 11, "head_spec": spec}
    config_bytes = json.dumps(config, sort_keys=True).encode()
    config_path.write_bytes(config_bytes)
    weights_path = config_path.with_suffix(".pt")
    weights_path.write_bytes(b"synthetic tensor-file bytes; no tensor claim")
    record = {"purpose": "main", "synthetic": False, "arm": "loop_ab2", "seed": 11,
        "target_epochs": 40, "epochs_completed": 39 if incomplete else 40,
        "protocol_sha256": __import__("hashlib").sha256(json.dumps(
            recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "recipe": recipe, "head_spec": spec, "head_spec_sha256": operator.head_spec_digest(spec),
        "history": history, "selected_epoch": selected_epoch, "selected_dev_ap": score,
        "selected_config": selected_name, "head_updated": True,
        "hardware": {"device_uuid": operator.G1_UUID},
        "frozen_trunk_digest": operator.FROZEN_TRUNK_STATE_SHA256}
    record_path = tmp_path / "training.json"
    record_bytes = json.dumps(record, sort_keys=True).encode()
    record_path.write_bytes(record_bytes)
    return record_path, hashlib.sha256(record_bytes).hexdigest(), config_path, weights_path, record


def test_project_training_record_matches_real_cadence_and_selected_paths(tmp_path):
    training, seal, config, weights, record = _sealed_training_fixture(tmp_path)
    row = operator.project_training_record(training, seal,
        recipe_sha256=record["protocol_sha256"], head_spec=record["head_spec"],
        seed=11, epochs=40,
        selected_config={"path": str(config), "sha256": hashlib.sha256(config.read_bytes()).hexdigest()},
        selected_weights={"path": str(weights), "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()})
    assert row["closed"] is True and row["selected_epoch"] == 35 and row["dev_ap"] == .21
    last_config = tmp_path / "last.json"
    last_config.write_bytes(config.read_bytes())
    with pytest.raises(ValueError, match="paths differ"):
        operator.project_training_record(training, seal,
            recipe_sha256=record["protocol_sha256"], head_spec=record["head_spec"],
            seed=11, epochs=40,
            selected_config={"path": str(last_config), "sha256": hashlib.sha256(last_config.read_bytes()).hexdigest()},
            selected_weights={"path": str(weights), "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()})
    incomplete_dir = tmp_path / "incomplete"
    incomplete_dir.mkdir()
    incomplete, incomplete_seal, cfg, pt, bad_record = _sealed_training_fixture(incomplete_dir, incomplete=True)
    with pytest.raises(ValueError, match="incomplete"):
        operator.project_training_record(incomplete, incomplete_seal,
            recipe_sha256=bad_record["protocol_sha256"], head_spec=bad_record["head_spec"],
            seed=11, epochs=40,
            selected_config={"path": str(cfg), "sha256": hashlib.sha256(cfg.read_bytes()).hexdigest()},
            selected_weights={"path": str(pt), "sha256": hashlib.sha256(pt.read_bytes()).hexdigest()})
    for field, value, error in (("head_updated", False, "head updated"),
                                ("hardware", {"device_uuid": "other-gpu"}, "pinned G1"),
                                ("frozen_trunk_digest", "0" * 64, "trunk state digest")):
        invalid = dict(record)
        invalid[field] = value
        invalid_path = tmp_path / f"invalid-{field}.json"
        invalid_bytes = json.dumps(invalid, sort_keys=True).encode()
        invalid_path.write_bytes(invalid_bytes)
        with pytest.raises(ValueError, match=error):
            operator.project_training_record(invalid_path, hashlib.sha256(invalid_bytes).hexdigest(),
                recipe_sha256=record["protocol_sha256"], head_spec=record["head_spec"],
                seed=11, epochs=40,
                selected_config={"path": str(config), "sha256": hashlib.sha256(config.read_bytes()).hexdigest()},
                selected_weights={"path": str(weights), "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()})
    wrong_dir = tmp_path / "wrong-best"
    wrong_dir.mkdir()
    wrong, wrong_seal, cfg, pt, wrong_record = _sealed_training_fixture(wrong_dir, selected_epoch=40)
    with pytest.raises(ValueError, match="earliest best"):
        operator.project_training_record(wrong, wrong_seal,
            recipe_sha256=wrong_record["protocol_sha256"], head_spec=wrong_record["head_spec"],
            seed=11, epochs=40,
            selected_config={"path": str(cfg), "sha256": hashlib.sha256(cfg.read_bytes()).hexdigest()},
            selected_weights={"path": str(pt), "sha256": hashlib.sha256(pt.read_bytes()).hexdigest()})


def test_head_macs_map_and_optional_profile_use_head_scope(tmp_path):
    row = {"closed": True, "head_spec": {"version": 1, "variant": "learned_alpha", "class_inner": 96},
           "selected_weights": {"sha256": "a" * 64}, "selected_config": {"sha256": "b" * 64},
           "recipe_sha256": "c" * 64}
    row = operator.attach_head_conv_macs(row, 631357440)
    profile = {"measurement": "inference", "branch_policy": "retained_dual", "imgsz": 512,
        "measurement_inputs": {"batch_size": 1},
        "model_identity": {"weights_sha256": "a" * 64, "config_sha256": "b" * 64,
            "recipe_sha256": "c" * 64, "head_spec": row["head_spec"],
            "head_spec_sha256": operator.head_spec_digest(row["head_spec"]), "variant": "learned_alpha"},
        "executed_conv_macs": {"conv_macs": 999999999},
        "head_executed_conv_macs": {"conv_macs": 631357440}}
    profile_path = tmp_path / "profile.json"
    data = json.dumps(profile, sort_keys=True).encode()
    profile_path.write_bytes(data)
    ref = {"path": str(profile_path), "sha256": hashlib.sha256(data).hexdigest()}
    attached = operator.attach_profile(row, conv_macs=631357440, profile=ref)
    assert attached["conv_macs"] == 631357440 and attached["profile"]["sha256"] == ref["sha256"]
    profile["head_executed_conv_macs"]["conv_macs"] += 1
    profile_path.write_text(json.dumps(profile, sort_keys=True))
    bad_ref = {"path": str(profile_path), "sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="head ConvMAC"):
        operator.attach_profile(row, conv_macs=631357440, profile=bad_ref)


def test_module_is_native_runtime_free_at_import():
    source = (ROOT / ".internal" / "ab2_refinement.py").read_text()
    assert "import torch" not in source and "from torch" not in source
    assert "subprocess" not in source and "socket" not in source


def _failed(variant, done=3, **changes):
    dose = 80 if variant == "baseline80" else 40
    row = _screen(variant, .0, 1) | {
        "status": "failed", "dev_ap": None, "probe_ap": None, "epochs": dose, "epochs_completed": done,
        "failure_reason": "synthetic divergence", "failure_evidence": {"path": "x/failure.json", "sha256": "d" * 64}}
    del row["conv_macs"], row["selected_epoch"], row["training"]
    return row | changes


def test_failed_conditions_are_ineligible_and_reported_without_a_fake_score():
    rows = [_screen("baseline80", .16, 700), _failed("learned_alpha"), _screen("step_norm", .18, 500),
            _screen("scale_gain", .15, 400), _failed("dw5", done=0)]
    decision = operator.decide_stages(rows)
    assert decision["winner"] == "step_norm" and decision["selected_variant"] == "step_norm"
    assert [x["variant"] for x in decision["failed_conditions"]] == ["learned_alpha", "dw5"]
    assert decision["failed_conditions"][1]["epochs_completed"] == 0
    report = operator.render_dev_report({"screens": rows}, decision)
    assert "FAILED after 3 completed epoch(s)" in report and "no admissible full-dose DEV score" in report
    assert "synthetic divergence" in report and "d" * 64 in report
    assert "learned_alpha / seed 11 / 40 epochs: DEV AP" not in report and "None" not in report
    assert "learned_alpha / 40 epochs / n=" not in report
    confirms = [{"variant": "step_norm", "head_spec": rows[2]["head_spec"], "seed": seed, "epochs": 40,
                 "closed": True, "dev_ap": .17} for seed in (33, 55)]
    assert operator.decide_stages(rows, confirms=confirms)["confirmation_mean_dev_ap"] is not None
    for bad in (confirms[0] | {"dev_ap": None, "status": "failed"}, confirms[0] | {"status": "failed"},
                confirms[0] | {"dev_ap": float("nan")}):
        with pytest.raises(ValueError, match="single-winner"):
            operator.decide_stages(rows, confirms=[bad])


def test_all_failed_is_a_truthful_no_winner_outcome_with_partial_curve_not_ranked():
    rows = [_failed(name, done=i, history=[{"epoch": 1, "dev": {"AP50:95": .9}, "probe": {"AP50:95": float("nan")}}])
            for i, name in enumerate(operator.VARIANTS)]
    decision = operator.decide_stages(rows)
    assert decision["winner"] is None and decision["winner_dose"] is None and decision["winner_dev_ap"] is None
    assert decision["selected_variant"] is None and decision["permitted_confirmations"] == []
    assert not decision["cost_admitted"] and any("no completed" in skip for skip in decision["skips"])
    assert any(skip.startswith("confirmation:") for skip in decision["skips"])
    with pytest.raises(ValueError, match="prohibited"):
        operator.decide_stages(rows, confirms=[{"variant": "dw5", "seed": 33}])
    report = operator.render_dev_report({"screens": rows}, decision)
    assert "no winner" in report and "could not be assessed" in report and "INCOMPLETE partial curve" in report
    assert "not ranked" in report and "None" not in report and "nan" not in report.lower()


def test_malformed_failures_partial_ap_and_incomplete_projection_are_rejected():
    base = [_screen("baseline80", .16, 700), _screen("learned_alpha", .15, 1), _screen("step_norm", .14, 2),
            _screen("scale_gain", .13, 3), _failed("dw5")]
    assert operator.decide_stages(base)["winner"] == "baseline80"
    bad_changes = [{"dev_ap": 0.0}, {"dev_ap": .05}, {"probe_ap": .05}, {"failure_reason": " "},
                   {"failure_reason": None}, {"failure_evidence": {"path": "p"}},
                   {"failure_evidence": {"path": "", "sha256": "d" * 64}},
                   {"failure_evidence": {"path": "p", "sha256": "D" * 64}},
                   {"failure_evidence": {"path": "p", "sha256": "d" * 64, "x": 1}},
                   {"epochs_completed": 40}, {"epochs_completed": -1}, {"epochs_completed": True},
                   {"epochs_completed": 1.0}, {"seed": 33}, {"closed": False}, {"status": "running"},
                   {"head_spec": {"version": 1, "variant": "dw5", "class_inner": 48}}]
    for change in bad_changes:
        rows = base[:4] + [_failed("dw5", **change)]
        with pytest.raises(ValueError):
            operator.decide_stages(rows)
    with pytest.raises(ValueError, match="status"):
        operator.decide_stages(base[:4] + [base[4] | {"status": "missing"}])
    with pytest.raises(ValueError, match="status"):
        operator.decide_stages(base[:3] + [base[3] | {"status": None}] + base[4:])
    with pytest.raises(ValueError, match="five"):
        operator.decide_stages(base[:4])
    with pytest.raises(ValueError, match="closed"):
        operator.decide_stages(base[:4] + [_failed("dw5") | {"closed": False}])
    with pytest.raises(ValueError, match="AP"):
        operator.decide_stages(base[:4] + [_screen("dw5", .1, 1) | {"dev_ap": None}])
    decision = operator.decide_stages(base)
    with pytest.raises(ValueError, match="partial AP"):
        operator.render_dev_report({"screens": base[:4] + [_failed("dw5", dev_ap=.01)]}, decision)
    with pytest.raises(ValueError, match="failed condition cannot be the decision winner"):
        operator.render_dev_report({"screens": base}, decision | {"winner": "dw5"})
    with pytest.raises(ValueError, match="failed or partial-dose"):
        operator.decide_stages(base[:4] + [base[4] | {"status": "completed", "dev_ap": .12, "conv_macs": 4}])
    for done in (3, True, 40.0):
        with pytest.raises(ValueError, match="partial-dose"):
            operator.decide_stages(base[:3] + [base[3] | {"epochs_completed": done}] + base[4:])
