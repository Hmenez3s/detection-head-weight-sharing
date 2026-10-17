import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from head_sharing.runtime import AllocationLedger, GPU_UUIDS, initialize
from head_sharing.supervisor import _launch_deadline, child_entry, launch, windows_detached_flags


def setup(tmp_path):
    study = tmp_path / "study"
    study.mkdir(); (study / "STUDY_OWNER.txt").touch()
    root = initialize(study, "ledger")
    ledger = AllocationLedger(root)
    token = ledger.reserve(GPU_UUIDS[0], 20, phase="main", cpus=1)
    jobdir = study / "job"
    jobdir.mkdir()
    return ledger, token, jobdir


def test_progress_heartbeat_exit_and_only_owned_child_reaped(tmp_path):
    ledger, token, cwd = setup(tmp_path)
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    script = ("import os,pathlib,time; root=pathlib.Path.cwd(); "
              "assert os.environ['YOLO_OFFLINE']=='true'; "
              "assert os.environ['HEAD_SHARING_CPU_BUDGET']=='1'; "
              "assert pathlib.Path(os.environ['HEAD_SHARING_LAUNCH_GATE']).is_file(); "
              "assert all(os.environ[k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')); "
              "assert os.environ['CUDA_VISIBLE_DEVICES']=='" + GPU_UUIDS[0] + "'; "
              "assert all(pathlib.Path(os.environ[k]).resolve().is_relative_to(root) for k in "
              "('YOLO_CONFIG_DIR','TORCH_HOME','MPLCONFIGDIR','TMPDIR','TEMP','TMP','CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')); "
              "p=pathlib.Path('progress'); p.write_text('one'); time.sleep(.15); p.write_text('two')")
    try:
        code = launch(ledger, token, [sys.executable, "-c", script], cwd=cwd,
                      log_path=cwd/"job.log", progress_path=cwd/"progress", first_progress_seconds=2)
        assert code == 0
        job = ledger.get_job(token)
        assert job["status"] == "complete" and job["pid"] and job["started"] and job["ended"]
        assert job["actual_wall_elapsed_seconds"] >= 0 and job["seconds"] == 20
        assert sentinel.poll() is None
    finally:
        sentinel.terminate(); sentinel.wait(timeout=3)


@pytest.mark.parametrize("poll", [float("nan"), float("inf"), -float("inf"), True, "1"])
def test_invalid_guard_polling_is_rejected_before_metrics_or_child_launch(tmp_path, poll):
    ledger, token, cwd = setup(tmp_path)
    metric_calls = []
    child_calls = []
    with pytest.raises(Exception, match="finite"):
        launch(ledger, token, [sys.executable, "-c", "open('started','w').write('yes')"],
               cwd=cwd, log_path=cwd/"job.log", progress_path=cwd/"progress",
               memory_guard=True, memory_poll_seconds=poll,
               memory_metrics=lambda _uuid: metric_calls.append(_uuid),
               popen_factory=lambda *args, **kwargs: child_calls.append(args))
    assert metric_calls == child_calls == []
    assert list(cwd.iterdir()) == []


def test_guard_query_cost_is_reserved_without_changing_legacy_deadline():
    started = 100.0
    legacy = _launch_deadline(started, 20, memory_guard=False)
    guarded = _launch_deadline(started, 20, memory_guard=True)
    assert legacy == 112.0 and guarded == 111.0
    assert legacy - guarded == 1.0


def test_frequent_progress_keeps_watchdog_live_without_fsyncing_each_poll(tmp_path, monkeypatch):
    ledger, token, cwd = setup(tmp_path)
    original = ledger.update_job
    persisted = []

    def observed(token, **updates):
        if updates.get('progress') is not None:
            persisted.append(updates['progress'])
        return original(token, **updates)

    monkeypatch.setattr(ledger, 'update_job', observed)
    child = "import pathlib,time\np=pathlib.Path('progress')\nfor n in range(20):\n p.write_text(str(n)); time.sleep(.05)\n"
    assert launch(ledger, token, [sys.executable, '-c', child], cwd=cwd,
                  log_path=cwd/'job.log', progress_path=cwd/'progress',
                  first_progress_seconds=2, heartbeat_seconds=2) == 0
    record = ledger.get_job(token)
    assert record['status'] == 'complete'
    assert 1 <= len(persisted) <= 2  # first snapshot + terminal; no five-second boundary
    assert record['progress'] == (cwd/'progress').stat().st_mtime


def test_guarded_launch_records_mocked_admission_and_runs_child(tmp_path):
    ledger, token, cwd = setup(tmp_path)
    ledger.authorize_extension(reason="approved", evidence="receipt")
    ledger.update_job(token, peak_memory_bytes=100, headroom_pending_bytes=100)
    samples = []
    def metrics(device_uuid):
        samples.append(device_uuid)
        return {"uuid": device_uuid, "free_bytes": 900, "total_bytes": 1_000,
                "sampled_at": time.time()}
    code = launch(ledger, token, [sys.executable, "-c", "open('progress','w').write('ok')"],
                  cwd=cwd, log_path=cwd/"job.log", progress_path=cwd/"progress",
                  first_progress_seconds=2, memory_guard=True, memory_metrics=metrics)
    assert code == 0 and samples == [GPU_UUIDS[0]]
    job = ledger.get_job(token)
    assert job["status"] == "complete"
    assert job["headroom_receipts"][0]["admission"] is True
    assert job["headroom_receipts"][0]["required_free_bytes"] == 200


def test_guarded_low_headroom_reaps_only_owned_child_and_keeps_charge(tmp_path):
    ledger, token, cwd = setup(tmp_path)
    ledger.authorize_extension(reason="approved", evidence="receipt")
    ledger.update_job(token, peak_memory_bytes=100, headroom_pending_bytes=100)
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    samples = iter([
        {"uuid": GPU_UUIDS[0], "free_bytes": 900, "total_bytes": 1_000, "sampled_at": time.time()},
        {"uuid": GPU_UUIDS[0], "free_bytes": 50, "total_bytes": 1_000, "sampled_at": time.time()},
    ])
    try:
        with pytest.raises(Exception, match="headroom"):
            launch(ledger, token, [sys.executable, "-c", "import time; time.sleep(10)"],
                   cwd=cwd, log_path=cwd/"job.log", progress_path=cwd/"progress",
                   memory_guard=True, memory_metrics=lambda _uuid: next(samples), memory_poll_seconds=.01)
        job = ledger.get_job(token)
        assert job["status"] == "failed" and job["seconds"] == 20
        assert len(job["headroom_receipts"]) == 2
        assert sentinel.poll() is None
    finally:
        sentinel.terminate(); sentinel.wait(timeout=3)


def test_child_environment_uses_owned_ledger_and_trusts_supervisor_identity(tmp_path):
    ledger, token, cwd = setup(tmp_path)
    forged = tmp_path / "foreign-ledger"
    script = ("import os,pathlib; "
              f"assert os.environ['HEAD_SHARING_LEDGER']=={str(ledger.root)!r}; "
              f"assert os.environ['HEAD_SHARING_RESERVATION']=={token!r}; "
              f"assert os.environ['CUDA_VISIBLE_DEVICES']=={GPU_UUIDS[0]!r}; "
              "assert pathlib.Path(os.environ['HEAD_SHARING_LAUNCH_GATE']).is_file(); "
              "pathlib.Path('progress').write_text('ok')")
    assert launch(ledger, token, [sys.executable, "-c", script], cwd=cwd,
                  log_path=cwd/"job.log", progress_path=cwd/"progress", first_progress_seconds=2,
                  env={"HEAD_SHARING_LEDGER": str(forged), "CUDA_VISIBLE_DEVICES": "forged"}) == 0
    assert ledger.get_job(token)["status"] == "complete"


def test_stalled_child_is_reaped_and_failure_charged(tmp_path):
    ledger, token, cwd = setup(tmp_path)
    with pytest.raises(TimeoutError, match="first real progress"):
        launch(ledger, token, [sys.executable, "-c", "import time; time.sleep(10)"],
               cwd=cwd, log_path=cwd/"job.log", progress_path=cwd/"progress", first_progress_seconds=.25)
    assert ledger.get_job(token)["status"] == "failed"
    # Failure remains charged (the 20-second reservation is not refunded).
    state = __import__("json").loads((ledger.root / "ledger.json").read_text())
    assert state["devices"][GPU_UUIDS[0]]["main"] == 20


def test_child_guard_waits_for_containment_and_checks_exact_uuid(tmp_path, monkeypatch):
    ledger, token, cwd = setup(tmp_path)
    ledger.update_job(token, status="launching")
    gate = cwd / "ready"
    monkeypatch.setenv("HEAD_SHARING_LAUNCH_GATE", str(gate))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUIDS[0])
    with pytest.raises(Exception, match="containment"):
        child_entry(ledger, token, GPU_UUIDS[0], gate_timeout=0)
    gate.touch()
    assert child_entry(ledger, token, GPU_UUIDS[0]) is None
    with pytest.raises(Exception, match="device"):
        child_entry(ledger, token, GPU_UUIDS[0], GPU_UUIDS[1])


def test_crashed_child_records_exit_and_windows_flags_are_exact(tmp_path):
    ledger, token, cwd = setup(tmp_path)
    assert launch(ledger, token, [sys.executable, "-c", "raise SystemExit(17)"], cwd=cwd,
                  log_path=cwd/"job.log", progress_path=cwd/"progress") == 17
    assert ledger.get_job(token)["exit_code"] == 17
    assert windows_detached_flags() == 0x01000208


def test_normal_parent_exit_reaps_own_descendant_but_not_unrelated_sentinel(tmp_path):
    ledger, token, cwd = setup(tmp_path)
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"], start_new_session=True)
    child_pid_file = cwd / "descendant.pid"
    child_ready = cwd / "descendant.ready"
    child_group = cwd / "descendant.group"
    child_marker = cwd / "descendant-survived"
    child = ("import os,signal,time,pathlib\n"
             "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
             f"pathlib.Path({str(child_ready)!r}).touch()\n"
             f"pathlib.Path({str(child_group)!r}).write_text(str(os.getpgrp()))\n"
             f"time.sleep(10); pathlib.Path({str(child_marker)!r}).touch()\n")
    parent = ("import pathlib,subprocess,sys,time\n"
              "p=subprocess.Popen([sys.executable,'-c'," + repr(child) + "])\n"
              f"pathlib.Path({str(child_pid_file)!r}).write_text(str(p.pid))\n"
              "deadline=time.time()+2\n"
              "while not pathlib.Path('descendant.ready').exists() and time.time()<deadline:\n time.sleep(.01)\n"
              "pathlib.Path('progress').touch()\n")
    try:
        assert launch(ledger, token, [sys.executable, "-c", parent], cwd=cwd,
                      log_path=cwd/"job.log", progress_path=cwd/"progress", first_progress_seconds=2) == 0
        assert child_group.read_text() == str(ledger.get_job(token)["pid"])
        time.sleep(1.1)
        assert not child_marker.exists()
        assert sentinel.poll() is None
        assert ledger.get_job(token)["status"] == "complete"
    finally:
        sentinel.terminate(); sentinel.wait(timeout=3)


def test_rejects_unowned_or_linked_cwd_before_creating_anything(tmp_path):
    ledger, token, cwd = setup(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(Exception, match="owned study root"):
        launch(ledger, token, [sys.executable, "-c", "pass"], cwd=outside,
               log_path=outside/"log", progress_path=outside/"progress")
    assert list(outside.iterdir()) == []


def test_lexical_dotdot_cwd_cannot_escape_and_writes_nothing(tmp_path):
    ledger, token, cwd = setup(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    escape = ledger.root.parent / ".." / "outside"
    with pytest.raises(Exception, match="owned study root"):
        launch(ledger, token, [sys.executable, "-c", "pass"], cwd=escape,
               log_path=escape/"log", progress_path=escape/"progress")
    assert list(outside.iterdir()) == []


def test_timeout_stops_early_and_counts_cleanup_within_reserved_limit(tmp_path):
    ledger, token, cwd = setup(tmp_path)
    token = ledger.reserve(GPU_UUIDS[1], 10, phase="main", cpus=1)
    script = ("import signal,time,pathlib; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
              "pathlib.Path('progress').touch(); time.sleep(10)")
    with pytest.raises(TimeoutError, match="reserved allocation deadline"):
        launch(ledger, token, [sys.executable, "-c", script], cwd=cwd,
               log_path=cwd/"timeout.log", progress_path=cwd/"progress", first_progress_seconds=5)
    job = ledger.get_job(token)
    assert job["actual_wall_elapsed_seconds"] <= job["seconds"]
    assert job["actual_wall_elapsed_seconds"] >= 2


def test_unknown_reservation_and_unowned_log_path_rejected(tmp_path):
    ledger, token, cwd = setup(tmp_path)
    with pytest.raises(Exception):
        launch(ledger, "unknown", ["x"], cwd=cwd, log_path=cwd/"a", progress_path=cwd/"b")
    outside = tmp_path / "outside.log"
    with pytest.raises(Exception, match="logs/progress"):
        launch(ledger, token, ["x"], cwd=cwd, log_path=outside, progress_path=cwd/"p")


def test_triton_cache_cannot_be_imported_as_optional_package(tmp_path):
    ledger, token, cwd = setup(tmp_path)
    script = ("import importlib.util,pathlib,os; root=pathlib.Path.cwd().resolve(); "
              "assert pathlib.Path(os.environ['TRITON_CACHE_DIR']).name=='triton-cache'; "
              "spec=importlib.util.find_spec('triton'); "
              "assert spec is None or not any(pathlib.Path(p).resolve().is_relative_to(root) "
              "for p in (spec.submodule_search_locations or [])); "
              "pathlib.Path('progress').write_text('optional-import-safe')")
    assert launch(ledger, token, [sys.executable, '-c', script], cwd=cwd,
                  log_path=cwd/'job.log', progress_path=cwd/'progress', first_progress_seconds=2) == 0
    assert (cwd/'progress').read_text() == 'optional-import-safe'
