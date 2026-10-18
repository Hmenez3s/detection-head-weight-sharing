import builtins
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "shared_spatial_controller", ROOT / ".internal" / "run_shared_spatial_fits.py")
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)

SOURCE = "a" * 40
RECIPE_SHA = "b" * 64
UUID = controller.UUID


class FakeLedger:
    def __init__(self, root):
        self.root = Path(root)
        self.jobs = []
        self._guard = threading.RLock()
        self._next = 0
        self.policy = {"finite": True}
        self.policy_history = [{"audit": "synthetic"}]

    class Lock:
        def __init__(self, guard): self.guard = guard
        def __enter__(self): self.guard.acquire(); return self
        def __exit__(self, *_): self.guard.release()

    def _locked(self): return self.Lock(self._guard)

    def _load(self):
        return {"policy": self.policy, "policy_history": self.policy_history,
                "devices": {UUID: {"jobs": [dict(job) for job in self.jobs]}}}

    def reserve(self, uuid, seconds, **kwargs):
        assert uuid == UUID and type(seconds) is int and seconds == 90
        with self._guard:
            self._next += 1
            token = f"synthetic-reservation-{self._next}"
            self.jobs.append({"id": token, "uuid": UUID, "status": "reserved", "cpus": 3,
                              "headroom_pending_bytes": 4 * 1024**3})
            return token

    def get_job(self, token):
        with self._guard:
            found = next(job for job in self.jobs if job["id"] == token)
            return {**found, "actual_wall_elapsed_seconds": 12,
                    "headroom_summary": {"sample_count": 2}}

    def mark_complete(self, token): self.update_job(token, status="complete")

    def update_job(self, token, **updates):
        with self._guard:
            next(job for job in self.jobs if job["id"] == token).update(updates)


class FakePolicy:
    def verify_frozen(self, path):
        return {"recipe_sha256": RECIPE_SHA, "recipe": recipe_for("native")}

    def project_training_record(self, path, digest, **kwargs):
        return {"variant": "native", "head_spec": None, "seed": kwargs["seed"],
                "epochs": 40, "dev_ap": .2, "probe_ap": .1, "closed": True,
                "recipe_sha256": RECIPE_SHA, "training": {"sha256": digest}, "status": "completed"}

    def decide_stages(self, rows):
        return {"screen_closed": True, "winner": None, "decision_sha256": "c" * 64, "screen": rows}

    def decide_followups(self, decision, rows): return decision | {"followups": rows}


def recipe_for(variant):
    return {"epochs": 40, "cadence": 5, "seed_list": [11, 22, 33, 44, 55],
            "arms": ["native"], "gpu_for_seed": {str(seed): UUID for seed in (11, 22, 33, 44, 55)},
            "environment": {"study_id": "shared_spatial_v1", "head_spec": None,
                            "source_git_commit": SOURCE}}


class FakeHelpers:
    def available_ram(self): return 12 * 1024**3

    def claim_startup_file(self, path, deadline, stop):
        path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            if stop.is_set(): raise RuntimeError("synthetic startup wait stopped")
            try:
                return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if time.monotonic() >= deadline: raise TimeoutError("synthetic startup lock timeout")
                time.sleep(.002)


def make_controller(tmp_path, *, launch=None, policy=None, slots=2):
    root = tmp_path / "owned"
    root.mkdir()
    (root / "STUDY_OWNER.txt").write_text("synthetic owner\n")
    study = root / controller.STUDY_ID
    study.mkdir()
    (study / "SHARED_SPATIAL_OWNER.txt").write_text(controller.STUDY_ID + "\n")
    code = root / "code-snapshots" / SOURCE[:7]; code.mkdir(parents=True)
    (code / "SOURCE_COMMIT").write_text(SOURCE)
    py = root / "venv/Scripts/python.exe"; py.parent.mkdir(parents=True); py.write_text("synthetic executable placeholder")
    protocol = study / "recipes/native-11.json"
    protocol.parent.mkdir(); protocol.write_text("{}")
    ledger = FakeLedger(root / "allocation-ledger")
    services = {
        "uuid": UUID, "AllocationLedger": lambda path: ledger,
        "physical_memory_metrics": lambda uuid: {"uuid": uuid, "free_bytes": 15 * 1024**3,
                                                   "total_bytes": 16 * 1024**3},
        "verify_frozen": lambda path: {"recipe_sha256": RECIPE_SHA, "recipe": recipe_for("native")},
        "canonicalize_head_spec": lambda spec: spec,
        "launch": launch or (lambda *a, **k: 0), "inspect": inspect,
    }
    instance = controller.SharedSpatialController(root=root, source=SOURCE,
        study_subroot=controller.STUDY_ID, code=code, services=services,
        policy=policy or FakePolicy(), helper=FakeHelpers(), python=py,
        manifest={"files": "synthetic"}, controller_path=ROOT / ".internal/run_shared_spatial_fits.py", slots=slots)
    return instance, ledger, protocol


@pytest.mark.parametrize("slots", [0, 3, True])
def test_slots_cannot_expand_the_existing_two_fit_limit(tmp_path, slots):
    with pytest.raises(ValueError, match="slots must be 1 or 2"):
        make_controller(tmp_path, slots=slots)


def test_single_slot_uses_one_worker_for_unchanged_admitted_cells(tmp_path, monkeypatch):
    instance, _, protocol = make_controller(tmp_path, slots=1)
    instance._check_inputs = lambda *, pilot: None
    workers = []
    original_executor = controller.ThreadPoolExecutor

    class RecordingExecutor(original_executor):
        def __init__(self, *args, **kwargs):
            workers.append(kwargs["max_workers"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(controller, "ThreadPoolExecutor", RecordingExecutor)
    instance._run_job = lambda *, row: {"projected": {"variant": row["variant"], "seed": row["seed"]}}
    rows = [{"variant": "native", "seed": seed,
             "protocol": protocol.relative_to(instance.root).as_posix(),
             "recipe_sha256": RECIPE_SHA, "allocation_seconds": 90} for seed in (11, 33)]
    result = instance.run_scientific({"mode": "screen", "cells": rows})
    assert workers == [1]
    assert [row["projected"]["seed"] for row in result["completions"]] == [11, 33]
    assert result["decision"] is None and not result["screen_closed"]


def test_import_is_inert_and_does_not_import_native_or_launch_modules():
    source = (ROOT / ".internal/run_shared_spatial_fits.py").read_text()
    assert "import torch" not in source and "from torch" not in source
    assert "subprocess" not in source and "nvidia-smi" not in source
    assert "if __name__ == \"__main__\":" in source


def test_strict_cell_admission_rejects_incomplete_duplicate_and_unregistered_cells():
    policy = FakePolicy()
    cells = [{"variant": name, "seed": seed, "protocol": f"recipes/{name}-{seed}.json",
              "recipe_sha256": RECIPE_SHA, "allocation_seconds": 90}
             for name in controller.VARIANTS for seed in controller.SCREEN_SEEDS]
    rows, decision = controller.validate_scientific_plan({"mode": "screen", "cells": cells}, policy)
    assert len(rows) == 12 and decision is None
    subset, _ = controller.validate_scientific_plan({"mode": "screen", "cells": cells[:-1]}, policy)
    assert len(subset) == 11
    for bad in (cells + [cells[0]], [cells[0] | {"variant": "baseline80"}] + cells[1:],
                [cells[0] | {"seed": 22}] + cells[1:],
                [cells[0] | {"allocation_seconds": True}] + cells[1:],
                [cells[0] | {"protocol": "../other.json"}] + cells[1:]):
        with pytest.raises(ValueError):
            controller.validate_scientific_plan({"mode": "screen", "cells": bad}, policy)


def test_launch_lifecycle_pins_request_progress_completion_and_keeps_calls_mocked(tmp_path):
    observed = {}

    def fake_launch(ledger, token, command, *, cwd, log_path, progress_path,
                    first_progress_seconds, heartbeat_seconds, env, memory_guard, memory_poll_seconds):
        observed.update(command=command, env=env, token=token, cwd=cwd,
                        first=first_progress_seconds, heartbeat=heartbeat_seconds,
                        memory_poll=memory_poll_seconds, memory_guard=memory_guard)
        Path(progress_path).write_text(json.dumps({"phase": "trunk_loading", "done": 1}) + "\n" +
                                      json.dumps({"phase": "train_batch", "done": 1}) + "\n")
        output = Path(cwd) / "output"; output.mkdir()
        record = {"purpose": "main", "target_epochs": 40, "epochs_completed": 40,
                  "selected_epoch": 35, "selected_config": "selected-e35.json"}
        (output / "training.json").write_text(json.dumps(record))
        (output / "selected-e35.json").write_text("{}")
        (output / "selected-e35.pt").write_bytes(b"synthetic")
        ledger.mark_complete(token)
        return 0

    instance, _, _ = make_controller(tmp_path, launch=fake_launch)
    row = {"variant": "native", "seed": 11, "protocol": f"{controller.STUDY_ID}/recipes/native-11.json",
           "recipe_sha256": RECIPE_SHA, "allocation_seconds": 90}
    result = instance._run_job(row=row)
    assert result["closed"] and result["seed"] == 11 and result["uuid"] == UUID
    assert result["source_commit"] == SOURCE and result["recipe_sha256"] == RECIPE_SHA
    assert observed["first"] == 45 and observed["heartbeat"] == 180 and observed["memory_poll"] == 5
    assert observed["memory_guard"] is True and observed["env"]["PYTHONPATH"] == str(instance.code / "src")
    assert observed["command"][observed["command"].index("--seed") + 1] == "11"
    request = json.loads((instance.study / "runs/native/seed-11/request.json").read_text())
    completion = json.loads((instance.study / "runs/native/seed-11/completion.json").read_text())
    assert request["full_recipe"] == recipe_for("native") and request["uuid"] == UUID
    assert request["reservation"] == completion["reservation"]
    assert completion["reservation"].startswith("synthetic-reservation-")
    assert completion["training_record_sha256"] == hashlib.sha256(
        (instance.study / "runs/native/seed-11/output/training.json").read_bytes()).hexdigest()


def test_rejection_and_failed_worker_do_not_create_next_cell_or_fake_completion(tmp_path):
    called = []
    def invalid_launch(*args, **kwargs):
        called.append(True)
        raise RuntimeError("synthetic worker failure")
    instance, ledger, _ = make_controller(tmp_path, launch=invalid_launch)
    bad = {"variant": "native", "seed": 11, "protocol": "../outside.json",
           "recipe_sha256": RECIPE_SHA, "allocation_seconds": 90}
    with pytest.raises(ValueError): instance._frozen_cell(bad)
    assert not called and ledger.jobs == []
    row = {"variant": "native", "seed": 11, "protocol": f"{controller.STUDY_ID}/recipes/native-11.json",
           "recipe_sha256": RECIPE_SHA, "allocation_seconds": 90}
    with pytest.raises(RuntimeError, match="synthetic worker failure"): instance._run_job(row=row)
    assert called and instance.stop.is_set()
    job = instance.study / "runs/native/seed-11"
    assert (job / "failure.json").is_file() and not (job / "completion.json").exists()


def test_positive_train_batch_is_required_not_cache_progress(tmp_path):
    instance, _, _ = make_controller(tmp_path)
    progress = tmp_path / "progress.jsonl"
    progress.write_text(json.dumps({"phase": "cache_validation", "done": 5}) + "\n" +
                        json.dumps({"phase": "train_batch", "done": 0}) + "\n")
    assert not instance._progress_has_train_batch(progress)
    with progress.open("a") as stream:
        stream.write(json.dumps({"phase": "train_batch", "done": 1}) + "\n")
    assert instance._progress_has_train_batch(progress)


def test_technical_mode_runs_native_then_registered_candidate_without_ranking(tmp_path):
    calls = []
    def fake_launch(ledger, token, command, *, cwd, progress_path, **kwargs):
        calls.append(command)
        Path(progress_path).write_text(json.dumps({"phase": "train_batch", "done": 1}) + "\n")
        output = Path(cwd) / "output"; output.mkdir()
        head_spec = None
        if "--head-spec" in command:
            head_spec = json.loads(Path(command[command.index("--head-spec") + 1]).read_text())
        record = {"purpose": "pilot", "seed": 997, "target_epochs": 3, "epochs_completed": 3,
                  "head_spec": head_spec, "protocol_sha256": None, "hardware": {"device_uuid": UUID},
                  "head_updated": True, "history": [{"epoch": 1}], "selected_config": None}
        (output / "training.json").write_text(json.dumps(record))
        ledger.mark_complete(token)
        return 0
    instance, _, _ = make_controller(tmp_path, launch=fake_launch)
    instance._check_inputs = lambda *, pilot: None
    cache = instance.root / "cache/technical-pilot-512-v1"; cache.mkdir(parents=True)
    (cache / "metadata.json").write_text(json.dumps({"count": 512, "imgsz": 512, "batch_size": 32,
                                                       "producer": UUID}))
    manifest = instance.code / "manifests/technical-pilot.json"
    manifest.parent.mkdir(); manifest.write_text(json.dumps({"ids": {"few": list(range(512))}}))
    candidate = {"version": 2, "variant": "spatial_sepbn"}
    results = instance.run_pilot({"candidate_spec": candidate, "candidate_tag": "pilot-a",
                                  "native_allocation_seconds": 90, "candidate_allocation_seconds": 90})
    assert results["purpose"] == "technical_only_not_ranked" and len(results["records"]) == 2
    assert results["records"][0]["spec"] is None and results["records"][1]["spec"] == candidate
    assert calls[0][calls[0].index("--seed") + 1] == calls[1][calls[1].index("--seed") + 1] == "997"
    assert "--finite-diagnostics" in calls[0] and "--finite-diagnostics" in calls[1]
    assert not any("--dev-cache" in call or "--probe-cache" in call for call in calls)


def test_failed_prelaunch_resource_admission_does_not_reserve_or_create_job(tmp_path):
    instance, ledger, _ = make_controller(tmp_path)
    instance.s["physical_memory_metrics"] = lambda uuid: {"uuid": uuid, "free_bytes": 1,
                                                            "total_bytes": 16 * 1024**3}
    row = {"variant": "native", "seed": 11,
           "protocol": f"{controller.STUDY_ID}/recipes/native-11.json",
           "recipe_sha256": RECIPE_SHA, "allocation_seconds": 90}
    with pytest.raises(RuntimeError, match="VRAM admission"):
        instance._run_job(row=row)
    assert ledger.jobs == [] and not (instance.study / "runs/native/seed-11").exists()


def test_zero_train_progress_never_opens_the_startup_gate(tmp_path):
    def no_progress(ledger, token, command, *, progress_path, **kwargs):
        Path(progress_path).write_text(json.dumps({"phase": "train_batch", "done": 0}) + "\n")
        return 0
    instance, _, _ = make_controller(tmp_path, launch=no_progress)
    row = {"variant": "native", "seed": 11,
           "protocol": f"{controller.STUDY_ID}/recipes/native-11.json",
           "recipe_sha256": RECIPE_SHA, "allocation_seconds": 90}
    with pytest.raises(RuntimeError, match="positive train_batch progress"):
        instance._run_job(row=row)
    assert instance.stop.is_set()


def test_completed_record_cannot_escape_job_with_selected_artifact_path(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("synthetic external bytes")
    def escaped_record(ledger, token, command, *, cwd, progress_path, **kwargs):
        Path(progress_path).write_text(json.dumps({"phase": "train_batch", "done": 1}) + "\n")
        output = Path(cwd) / "output"; output.mkdir()
        (output / "training.json").write_text(json.dumps({"selected_epoch": 35,
            "selected_config": "../../../../outside.json", "purpose": "main"}))
        ledger.mark_complete(token)
        return 0
    instance, _, _ = make_controller(tmp_path, launch=escaped_record)
    row = {"variant": "native", "seed": 11,
           "protocol": f"{controller.STUDY_ID}/recipes/native-11.json",
           "recipe_sha256": RECIPE_SHA, "allocation_seconds": 90}
    with pytest.raises(ValueError, match="selected-checkpoint path"):
        instance._run_job(row=row)
    assert outside.read_text() == "synthetic external bytes"
    assert (instance.study / "runs/native/seed-11/failure.json").is_file()


def test_startup_mutex_releases_at_positive_batch_and_two_fits_overlap(tmp_path):
    entered = {11: threading.Event(), 33: threading.Event()}
    positive = {11: threading.Event(), 33: threading.Event()}
    release = {11: threading.Event(), 33: threading.Event()}
    ended = {11: threading.Event(), 33: threading.Event()}
    order = []; order_lock = threading.Lock()
    first_seed = {}
    first_positive = threading.Event()
    second_started = threading.Event()
    def launch(ledger, token, command, *, cwd, progress_path, **kwargs):
        seed = int(command[command.index("--seed") + 1])
        entered[seed].set()
        with order_lock:
            order.append(("launch", seed))
            ordinal = sum(event == "launch" for event, _ in order)
            if ordinal == 1: first_seed["seed"] = seed
            else: second_started.set()
        Path(progress_path).write_text(json.dumps({"phase": "train_batch", "done": 1}) + "\n")
        with order_lock: order.append(("positive", seed))
        if ordinal == 1: first_positive.set()
        positive[seed].set()
        assert release[seed].wait(5), "test did not release synthetic fit"
        _write_mock_completed_fit(ledger, token, cwd)
        ended[seed].set()
        return 0
    instance, _, _ = make_controller(tmp_path, launch=launch)
    instance._check_inputs = lambda *, pilot: None
    result_box = []
    runner = threading.Thread(target=lambda: result_box.append(instance.run_scientific(_two_native_cells(instance))))
    runner.start()
    try:
        assert first_positive.wait(3)
        assert second_started.wait(3), "second launch did not begin after first positive batch"
        second_seed = 33 if first_seed["seed"] == 11 else 11
        assert positive[second_seed].wait(3), "second launch did not emit positive training progress"
        assert not ended[first_seed["seed"]].is_set(), "first fit ended before second started"
        assert order[:4] == [("launch", first_seed["seed"]), ("positive", first_seed["seed"]),
                             ("launch", second_seed), ("positive", second_seed)]
    finally:
        release[11].set(); release[33].set(); runner.join(5)
    assert not runner.is_alive() and len(result_box) == 1
    assert len(result_box[0]["completions"]) == 2 and ended[11].is_set() and ended[33].is_set()


def _two_native_cells(instance):
    second = instance.study / "recipes/native-33.json"
    second.write_text("synthetic frozen protocol reference")
    return {"mode": "screen", "cells": [
        {"variant": "native", "seed": seed,
         "protocol": f"{controller.STUDY_ID}/recipes/native-{seed}.json",
         "recipe_sha256": RECIPE_SHA, "allocation_seconds": 90}
        for seed in (11, 33)]}


def _write_mock_completed_fit(ledger, token, cwd):
    output = Path(cwd) / "output"; output.mkdir()
    (output / "training.json").write_text(json.dumps({"selected_epoch": 35,
        "selected_config": "selected-e35.json", "purpose": "main"}))
    (output / "selected-e35.json").write_text("{}")
    (output / "selected-e35.pt").write_bytes(b"synthetic")
    ledger.mark_complete(token)


def test_preprogress_init_failure_stops_queued_gpu_launch_and_leaves_cell_unstarted(tmp_path, capsys):
    launched = []
    def launch(ledger, token, command, *, cwd, progress_path, **kwargs):
        seed = int(command[command.index("--seed") + 1])
        launched.append(seed)
        if len(launched) == 1:
            return 1
        Path(progress_path).write_text(json.dumps({"phase": "train_batch", "done": 1}) + "\n")
        _write_mock_completed_fit(ledger, token, cwd)
        return 0
    instance, _, _ = make_controller(tmp_path, launch=launch)
    instance._check_inputs = lambda *, pilot: None
    with pytest.raises(RuntimeError, match="stage paused"):
        instance.run_scientific(_two_native_cells(instance))
    assert len(launched) == 1
    unstarted_seed = 33 if launched[0] == 11 else 11
    assert not (instance.study / f"runs/native/seed-{unstarted_seed}").exists()
    output = capsys.readouterr().out
    assert f'"seed": {unstarted_seed}' in output and '"unstarted_cells"' in output


def test_wire_validates_root_source_and_manifest_before_import_callbacks(tmp_path, monkeypatch):
    instance, _, _ = make_controller(tmp_path)
    calls = []
    monkeypatch.setattr(controller, "_manifest", lambda *a: calls.append("manifest") or (_ for _ in ()).throw(AssertionError()))
    with pytest.raises(ValueError, match="full immutable source SHA"):
        controller._wire(instance.root, "short", controller.STUDY_ID, instance.controller_path)
    assert calls == []
    (instance.code / "SOURCE_COMMIT").write_text("0" * 40)
    with pytest.raises(ValueError, match="SOURCE_COMMIT mismatch"):
        controller._wire(instance.root, SOURCE, controller.STUDY_ID, instance.controller_path)
    assert calls == []
    (instance.code / "SOURCE_COMMIT").write_text(SOURCE)
    escaped = tmp_path / "escaped-root"
    escaped.symlink_to(instance.root, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical marker-owned root"):
        controller._wire(escaped, SOURCE, controller.STUDY_ID, instance.controller_path)
    assert calls == []

    monkeypatch.undo()
    code = instance.code
    (code / ".internal").mkdir()
    (code / "src").mkdir()
    (code / ".internal/shared_spatial.py").write_bytes(b"synthetic policy bytes")
    (code / ".internal/run_ab2_fits.py").write_bytes(b"synthetic helper bytes")
    (code / ".internal/run_shared_spatial_fits.py").write_bytes(instance.controller_path.read_bytes())
    (code / "shared-spatial-source-manifest.json").write_text(json.dumps({
        "source_git_commit": SOURCE,
        "files": {
            "controller": {"path": ".internal/run_shared_spatial_fits.py",
                          "sha256": hashlib.sha256(instance.controller_path.read_bytes()).hexdigest()},
            "policy": {"path": ".internal/shared_spatial.py", "sha256": "0" * 64},
            "ram_startup_helper": {"path": ".internal/run_ab2_fits.py",
                                    "sha256": hashlib.sha256((code / ".internal/run_ab2_fits.py").read_bytes()).hexdigest()},
        }}))
    helper_calls = []
    imported_private = []
    original_import = builtins.__import__
    def import_spy(name, *args, **kwargs):
        if name.startswith("head_sharing"):
            imported_private.append(name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(controller, "_load_helper", lambda *a: helper_calls.append("helper"))
    monkeypatch.setattr(builtins, "__import__", import_spy)
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        controller._wire(instance.root, SOURCE, controller.STUDY_ID, instance.controller_path)
    assert helper_calls == [] and imported_private == []


def test_actual_policy_projection_status_enters_selector_and_followup_subsets_stay_sealed(tmp_path):
    policy_spec = importlib.util.spec_from_file_location(
        "shared_spatial_real_policy_for_controller_test", ROOT / ".internal/shared_spatial.py")
    real_policy = importlib.util.module_from_spec(policy_spec); policy_spec.loader.exec_module(real_policy)
    from head_sharing.protocol import NUMERIC_POLICY, SELECTION

    old_recipe = {"status": "FROZEN", "seed_list": [11, 22, 33, 44, 55],
        "arms": ["native", "independent_k2", "shared_k2"], "imgsz": 512, "batch_size": 32,
        "epochs": 40, "cadence": 5, "block_size": 32, "probe_size": 1000,
        "optimizer": {"name": "SGD", "lr": .01, "momentum": .9, "weight_decay": .0005},
        "gains": {"box": 7.5, "cls": .5, "dfl": 1.5}, "quantization": "fp32_fp16_fp32_v1",
        "producer": real_policy.G1_UUID, "environment": {"torch_version": "fixture"},
        "trunk_sha256": real_policy.TRUNK_ARTIFACT_SHA256, "split_sha256": "b" * 64,
        "cache_evidence": {name: {"metadata_file_sha256": "c" * 64} for name in ("few", "dev", "probe")},
        "pilot_evidence": {"source": "synthetic"}, "synthetic": False,
        "gpu_for_seed": {str(seed): real_policy.G1_UUID for seed in real_policy.SEEDS},
        "selection": SELECTION, "numeric_policy": NUMERIC_POLICY}
    old_sha = hashlib.sha256(json.dumps(old_recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    recipes = real_policy.prepare_recipe(old_recipe,
        source_protocol={"path": "source/protocol.json", "sha256": "d" * 64, "recipe_sha256": old_sha},
        source_freeze={"path": "source/freeze.json", "sha256": "f" * 64}, source_git_commit=SOURCE)
    recipe = recipes["native"]
    receipt = {"trunk_artifact_sha256": real_policy.TRUNK_ARTIFACT_SHA256, "trunk_tensors": 468,
        "trunk_bytes": 9_109_168, "head_index": 23, "input_channels": [64, 128, 256],
        "source_variant": "U", "source_seed": 11, "source_epoch_zero_based": 79,
        "source_training_epochs": 80, "pre_exposure": "synthetic fixture provenance"}
    config = {"format": 1, "architecture": "yolo26n", "arm": "native", "seed": 11,
        "imgsz": 512, "quantization": "fp32_fp16_fp32_v1", "trunk_receipt": receipt}
    config_path = tmp_path / "selected-e35.json"
    config_bytes = json.dumps(config, sort_keys=True).encode(); config_path.write_bytes(config_bytes)
    weights_path = tmp_path / "selected-e35.pt"; weights_path.write_bytes(b"synthetic bytes, not a tensor")
    history = []
    for epoch in range(1, 41):
        row = {"epoch": epoch}
        if epoch % 5 == 0:
            score = .30 if epoch == 35 else .20
            row.update(dev={"AP50:95": score}, probe={"AP50:95": score - .01})
        history.append(row)
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    record = {"schema_version": 1, "purpose": "main", "synthetic": False, "arm": "native", "seed": 11,
        "selection_metric": "dev.AP50:95", "target_epochs": 40, "epochs_completed": 40,
        "protocol_sha256": recipe_sha, "recipe": recipe, "head_spec": None,
        "head_spec_sha256": real_policy.head_spec_digest(None), "history": history,
        "selected_epoch": 35, "selected_dev_ap": .30, "selected_config": "selected-e35.json",
        "head_updated": True, "hardware": {"device_uuid": real_policy.G1_UUID},
        "frozen_trunk_digest": "e" * 64}
    record_path = tmp_path / "training.json"
    record_bytes = json.dumps(record, sort_keys=True).encode(); record_path.write_bytes(record_bytes)
    projected = real_policy.project_training_record(record_path, hashlib.sha256(record_bytes).hexdigest(),
        recipe=recipe, seed=11,
        selected_config={"path": str(config_path), "sha256": hashlib.sha256(config_bytes).hexdigest()},
        selected_weights={"path": str(weights_path), "sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest()})
    assert projected["status"] == "completed"
    screen = []
    for variant in real_policy.VARIANTS:
        for seed in real_policy.SCREEN_SEEDS:
            if (variant, seed) == ("native", 11):
                screen.append(projected)
            else:
                spec = None if variant == "native" else {"version": 2, "variant": variant}
                score = .20 if variant != "spatial_sepbn" else .40
                screen.append({"variant": variant, "seed": seed, "status": "completed", "closed": True,
                    "epochs": 40, "head_spec": spec, "dev_ap": score, "probe_ap": score - .01,
                    "training": {"sha256": "a" * 64}})
    decision = real_policy.decide_stages(screen)
    assert decision["winner"] == "spatial_sepbn"
    partial = [{"variant": "native", "seed": 22, "protocol": f"{controller.STUDY_ID}/recipes/native-22.json",
                "recipe_sha256": recipe_sha, "allocation_seconds": 90}]
    rows, returned_decision = controller.validate_scientific_plan(
        {"mode": "followup", "decision": decision, "cells": partial}, real_policy)
    assert len(rows) == 1 and returned_decision is decision

    call_root = tmp_path / "controller-call"; call_root.mkdir()
    instance, _, _ = make_controller(call_root, policy=real_policy)
    instance._check_inputs = lambda *, pilot: None
    instance.s["verify_frozen"] = lambda path: {"recipe_sha256": recipe_sha, "recipe": recipe}
    (instance.study / "recipes/native-22.json").write_text("synthetic frozen protocol")
    projected22 = {"variant": "native", "seed": 22, "status": "completed", "closed": True,
                   "epochs": 40, "head_spec": None, "dev_ap": .2, "probe_ap": .19,
                   "training": {"sha256": "a" * 64}}
    instance._run_job = lambda *, row: {"projected": projected22}
    decision_calls = []
    original_decide_followups = real_policy.decide_followups
    real_policy.decide_followups = lambda *args: decision_calls.append(args)
    try:
        result = instance.run_scientific({"mode": "followup", "decision": decision, "cells": partial})
    finally:
        real_policy.decide_followups = original_decide_followups
    assert result["decision"] is decision and result["followups_closed"] is False
    assert not decision_calls
