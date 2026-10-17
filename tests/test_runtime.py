import json
import os
import threading
import time

import pytest

from head_sharing.runtime import AllocationLedger, GPU_UUIDS, RuntimeErrorSafe, initialize


def owned(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    (root / "STUDY_OWNER.txt").write_text("operator-owned\n")
    return initialize(root, "ledger")


def test_independent_uuid_caps_held_reserves_and_failed_charge(tmp_path):
    ledger = AllocationLedger(owned(tmp_path))
    first = ledger.reserve(GPU_UUIDS[0], 46_800, phase="main")
    with pytest.raises(RuntimeErrorSafe):
        ledger.reserve(GPU_UUIDS[0], 9, phase="main")
    ledger.update_job(first, status="failed", exit_code=-9)
    final = ledger.reserve(GPU_UUIDS[0], 7_200, phase="final")
    contingency = ledger.reserve(GPU_UUIDS[1], 3_600, phase="contingency")
    independent = ledger.reserve(GPU_UUIDS[1], 57_600, phase="main") if False else None
    assert ledger.get_job(first)["status"] == "failed"
    assert ledger.get_job(final)["phase"] == "final"
    assert ledger.get_job(contingency)["phase"] == "contingency"
    assert json.loads((ledger.root / "ledger.json").read_text())["devices"][GPU_UUIDS[0]]["main"] == 46_800


def test_live_lock_contention_waits_for_release(tmp_path):
    ledger = AllocationLedger(owned(tmp_path))
    ready = threading.Event()
    def writer():
        with ledger._locked():
            ready.set()
            time.sleep(0.2)
    thread = threading.Thread(target=writer)
    thread.start()
    assert ready.wait(2)
    try:
        token = ledger.reserve(GPU_UUIDS[0], 10, phase="main", cpus=1)
    finally:
        thread.join(2)
    assert ledger.get_job(token)["status"] == "reserved"


def test_lock_unknown_stale_and_double_reservation_fail_closed(tmp_path):
    root = owned(tmp_path)
    ledger = AllocationLedger(root)
    (root / "ledger.lock").write_text('{"pid":999999}')
    with pytest.raises(RuntimeErrorSafe, match="lock persists"):
        ledger.reserve(GPU_UUIDS[0], 9, phase="main")
    (root / "ledger.lock").unlink()
    token = ledger.reserve(GPU_UUIDS[0], 9, phase="main")
    with pytest.raises(RuntimeErrorSafe, match="already has"):
        ledger.reserve(GPU_UUIDS[0], 9, phase="main")
    with pytest.raises(RuntimeErrorSafe, match="unknown UUID"):
        ledger.reserve("GPU-unknown", 1, phase="main")
    assert ledger.get_job(token)["status"] == "reserved"


def test_launch_claim_is_atomic_and_nonrepeatable(tmp_path):
    ledger = AllocationLedger(owned(tmp_path))
    token = ledger.reserve(GPU_UUIDS[0], 9, phase="main")
    claimed = []
    def claim():
        try: claimed.append(ledger.claim_launch(token)["status"])
        except RuntimeErrorSafe: claimed.append("denied")
    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert sorted(claimed) == ["denied", "launching"]


def test_concurrent_reservations_are_serialized(tmp_path):
    ledger = AllocationLedger(owned(tmp_path))
    outcomes = []
    def reserve():
        try:
            outcomes.append(ledger.reserve(GPU_UUIDS[0], 9, phase="main"))
        except RuntimeErrorSafe:
            outcomes.append("denied")
    threads = [threading.Thread(target=reserve) for _ in range(8)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert sum(item != "denied" for item in outcomes) == 1
    assert sum(item == "denied" for item in outcomes) == 7


def test_active_jobs_keep_cpu_allocation_reserved(tmp_path):
    ledger = AllocationLedger(owned(tmp_path))
    first = ledger.reserve(GPU_UUIDS[0], 10, phase="main", cpus=12)
    ledger.update_job(first, status="running")
    with pytest.raises(RuntimeErrorSafe, match="aggregate CPU"):
        ledger.reserve(GPU_UUIDS[1], 10, phase="main", cpus=1)


def test_cpu_and_balance_limits_and_exact_device(tmp_path):
    ledger = AllocationLedger(owned(tmp_path))
    with pytest.raises(RuntimeErrorSafe, match="CPU"):
        ledger.reserve(GPU_UUIDS[0], 9, phase="main", cpus=13)
    token = ledger.reserve(GPU_UUIDS[0], 9, phase="main")
    with pytest.raises(RuntimeErrorSafe, match="device"): 
        ledger.validate_device(token, GPU_UUIDS[1])
    ledger.update_job(token, status="running")
    assert ledger.validate_device(token, GPU_UUIDS[0])
    with pytest.raises(RuntimeErrorSafe, match="already has"):
        ledger.reserve(GPU_UUIDS[0], 9, phase="main")
    with pytest.raises(RuntimeErrorSafe):
        ledger.reserve(GPU_UUIDS[0], 46_801, phase="main")


def test_ledger_rejects_noncanonical_and_linked_roots(tmp_path):
    study = tmp_path / "study"
    study.mkdir(); (study / "STUDY_OWNER.txt").touch()
    root = initialize(study, "ledger")
    with pytest.raises(RuntimeErrorSafe, match="noncanonical/linked"):
        AllocationLedger(root.parent / "other" / ".." / root.name)
    alias = tmp_path / "ledger-alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(RuntimeErrorSafe, match="noncanonical/linked"):
        AllocationLedger(alias)


def test_initializer_refuses_symlinked_owned_root(tmp_path):
    real = tmp_path / "real"
    real.mkdir(); (real / "STUDY_OWNER.txt").touch()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeErrorSafe, match="links"):
        initialize(alias, "ledger")


def test_initializer_refuses_existing_subroot_and_missing_owner(tmp_path):
    with pytest.raises(RuntimeErrorSafe, match="marker"):
        initialize(tmp_path, "ledger")
    root = tmp_path / "owned"
    root.mkdir(); (root / "STUDY_OWNER.txt").touch()
    initialize(root, "ledger")
    with pytest.raises(RuntimeErrorSafe, match="already exists"):
        initialize(root, "ledger")


def test_torch_bare_uuid_matches_nvml_identity_but_not_other_device(tmp_path):
    from head_sharing.runtime import canonical_gpu_uuid
    ledger = AllocationLedger(owned(tmp_path))
    token = ledger.reserve(GPU_UUIDS[1], 120, phase='main', cpus=1)
    ledger.update_job(token, status='running')
    assert canonical_gpu_uuid(GPU_UUIDS[1][4:]) == GPU_UUIDS[1]
    assert canonical_gpu_uuid(GPU_UUIDS[1]) == GPU_UUIDS[1]
    assert ledger.validate_device(token, GPU_UUIDS[1][4:])
    with pytest.raises(RuntimeErrorSafe, match='device'):
        ledger.validate_device(token, GPU_UUIDS[0][4:])
    for invalid in ('', 'GPU-unknown', GPU_UUIDS[1].replace('-', ''), None):
        with pytest.raises(RuntimeErrorSafe, match='UUID'):
            canonical_gpu_uuid(invalid)


def test_transient_windows_sharing_denial_retries_without_losing_charge(tmp_path, monkeypatch):
    import head_sharing.runtime as runtime
    ledger = AllocationLedger(owned(tmp_path))
    replace = runtime.os.replace
    attempts = []
    def briefly_busy(source, target):
        attempts.append(str(target))
        if len(attempts) == 1:
            exc = PermissionError('owned file temporarily open without delete sharing')
            exc.winerror = 5
            raise exc
        return replace(source, target)
    monkeypatch.setattr(runtime.os, 'replace', briefly_busy)
    token = ledger.reserve(GPU_UUIDS[0], 10, phase='main', cpus=1)
    assert len(attempts) == 2 and ledger.get_job(token)['seconds'] == 10
    assert ledger._load()['devices'][GPU_UUIDS[0]]['main'] == 10


def test_persistent_sharing_denial_is_bounded_and_keeps_previous_ledger(tmp_path, monkeypatch):
    import head_sharing.runtime as runtime
    ledger = AllocationLedger(owned(tmp_path))
    before = (ledger.root/'ledger.json').read_bytes()
    calls = []
    def always_busy(source, target):
        calls.append(target)
        exc = PermissionError('persistent denial')
        exc.winerror = 32
        raise exc
    clock = iter([0.0, 2.0])
    monkeypatch.setattr(runtime.os, 'replace', always_busy)
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: next(clock))
    with pytest.raises(PermissionError):
        runtime._atomic_json(ledger.root/'ledger.json', {'must_not_replace': True})
    assert len(calls) == 1 and (ledger.root/'ledger.json').read_bytes() == before
    assert not list(ledger.root.glob('.ledger-*'))


def test_explicit_budget_extension_preserves_old_state_and_is_idempotent(tmp_path):
    ledger = AllocationLedger(owned(tmp_path))
    token = ledger.reserve(GPU_UUIDS[0], 46_800, phase="main", cpus=2)
    ledger.update_job(token, status="failed")
    before = ledger._load()
    assert ledger.authorize_extension(reason="approved study", evidence="authorization-01")
    after = ledger._load()
    assert after["devices"] == before["devices"]
    assert after["policy"]["total_seconds"] == 172_800
    assert after["policy"]["main_seconds"] == 140_400
    assert after["policy"]["final_seconds"] == 21_600
    assert after["policy"]["contingency_seconds"] == 10_800
    assert len(after["policy_history"]) == 1
    assert not ledger.authorize_extension(reason="approved study", evidence="authorization-01")
    assert len(ledger._load()["policy_history"]) == 1
    assert ledger.get_job(token)["seconds"] == 46_800
    first = ledger.reserve(GPU_UUIDS[0], 9, phase="main", cpus=2)
    second = ledger.reserve(GPU_UUIDS[0], 9, phase="main", cpus=2)
    with pytest.raises(RuntimeErrorSafe, match="maximum"):
        ledger.reserve(GPU_UUIDS[0], 9, phase="main", cpus=2)
    assert ledger.get_job(first)["seconds"] == ledger.get_job(second)["seconds"] == 9


def test_extension_validation_and_concurrent_active_unknown_capacity(tmp_path):
    ledger = AllocationLedger(owned(tmp_path))
    with pytest.raises(RuntimeErrorSafe, match="required"):
        ledger.authorize_extension(reason=" ", evidence="proof")
    ledger.authorize_extension(reason="approved", evidence="receipt")
    first = ledger.reserve(GPU_UUIDS[0], 140_400, phase="main", cpus=6)
    second = ledger.reserve(GPU_UUIDS[0], 21_600, phase="final", cpus=6)
    ledger.update_job(first, status="active-unknown")
    ledger.update_job(second, status="failed")
    with pytest.raises(RuntimeErrorSafe, match="phase balance"):
        ledger.reserve(GPU_UUIDS[0], 9, phase="final", cpus=1)
    with pytest.raises(RuntimeErrorSafe, match="aggregate CPU"):
        ledger.reserve(GPU_UUIDS[1], 9, phase="main", cpus=7)
    assert ledger.get_job(second)["status"] == "failed"


def test_authorized_same_uuid_concurrent_reservations_admit_exactly_two(tmp_path):
    ledger = AllocationLedger(owned(tmp_path))
    ledger.authorize_extension(reason="approved", evidence="receipt")
    outcomes = []
    def reserve():
        try:
            outcomes.append(ledger.reserve(GPU_UUIDS[0], 9, phase="main", cpus=1))
        except RuntimeErrorSafe:
            outcomes.append("denied")
    threads = [threading.Thread(target=reserve) for _ in range(8)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert sum(result != "denied" for result in outcomes) == 2
    assert sum(result == "denied" for result in outcomes) == 6
    state = ledger._load()
    assert sum(job["seconds"] for job in state["devices"][GPU_UUIDS[0]]["jobs"]) == 18


def test_headroom_guard_serializes_pending_peak_commitments_and_fails_closed(tmp_path):
    ledger = AllocationLedger(owned(tmp_path))
    ledger.authorize_extension(reason="approved", evidence="receipt")
    first = ledger.reserve(GPU_UUIDS[0], 10, phase="main", cpus=1, peak_memory_bytes=200)
    second = ledger.reserve(GPU_UUIDS[0], 10, phase="main", cpus=1, peak_memory_bytes=200)
    sample = {"uuid": GPU_UUIDS[0], "free_bytes": 1_500, "total_bytes": 10_000,
              "sampled_at": time.time()}
    assert ledger.guard_headroom(first, sample, admission=True)["required_free_bytes"] == 1_400
    with pytest.raises(RuntimeErrorSafe, match="headroom"):
        ledger.guard_headroom(second, {**sample, "free_bytes": 1_300}, admission=True)
    stale = {**sample, "sampled_at": time.time()-30}
    with pytest.raises(RuntimeErrorSafe, match="stale"):
        ledger.guard_headroom(first, stale, admission=False)
    wrong = {**sample, "uuid": GPU_UUIDS[1]}
    with pytest.raises(RuntimeErrorSafe, match="mismatch"):
        ledger.guard_headroom(first, wrong, admission=False)
    assert len(ledger.get_job(second)["headroom_receipts"]) == 1


def test_headroom_receipts_are_bounded_but_summary_covers_every_sample(tmp_path):
    ledger = AllocationLedger(owned(tmp_path))
    ledger.authorize_extension(reason="approved", evidence="receipt")
    token = ledger.reserve(GPU_UUIDS[0], 20, phase="main", cpus=1, peak_memory_bytes=50)
    failure_time = None
    for index in range(128):
        free = 99 if index == 83 else 500
        sampled_at = time.time()
        if index == 83:
            failure_time = sampled_at
        sample = {"uuid": GPU_UUIDS[0], "free_bytes": free, "total_bytes": 1_000,
                  "sampled_at": sampled_at}
        try:
            ledger.guard_headroom(token, sample, admission=(index == 0))
        except RuntimeErrorSafe:
            assert index == 83
    job = ledger.get_job(token)
    summary = job["headroom_summary"]
    assert len(job["headroom_receipts"]) == 32
    assert job["headroom_receipts"][0]["admission"] is True
    assert any(not receipt["passed"] for receipt in job["headroom_receipts"])
    assert summary["sample_count"] == 128
    assert summary["min_free_fraction"] == 0.099
    assert summary["min_free_fraction_sample"] == {
        "free_bytes": 99, "total_bytes": 1_000, "sampled_at": failure_time}
    assert summary["min_free_bytes"] == 99 and summary["min_total_bytes"] == 1_000


def test_exclusive_lock_creation_retries_transient_access_denial(tmp_path, monkeypatch):
    import head_sharing.runtime as runtime
    ledger = AllocationLedger(owned(tmp_path))
    real_open = runtime.os.open
    calls = []
    def temporarily_pending(path, flags, *args, **kwargs):
        if os.fspath(path) == os.fspath(ledger.root / 'ledger.lock'):
            calls.append(path)
            if len(calls) <= 2:
                raise PermissionError(13, 'exclusive create temporarily denied')
        return real_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(runtime.os, 'open', temporarily_pending)
    token = ledger.reserve(GPU_UUIDS[1], 10, phase='main', cpus=1)
    assert len(calls) == 3
    assert ledger.get_job(token)['seconds'] == 10
    assert ledger._load()['devices'][GPU_UUIDS[1]]['main'] == 10
    assert not (ledger.root / 'ledger.lock').exists()


def test_persistent_exclusive_lock_denial_is_bounded_and_preserves_ledger(tmp_path, monkeypatch):
    import head_sharing.runtime as runtime
    ledger = AllocationLedger(owned(tmp_path))
    before = (ledger.root / 'ledger.json').read_bytes()
    real_open = runtime.os.open
    def inaccessible(path, flags, *args, **kwargs):
        if os.fspath(path) == os.fspath(ledger.root / 'ledger.lock'):
            raise PermissionError(13, 'persistent access denial')
        return real_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(runtime.os, 'open', inaccessible)
    monkeypatch.setattr(runtime, 'LEDGER_LOCK_WAIT_SECONDS', .02)
    with pytest.raises(RuntimeErrorSafe, match='lock'):
        ledger.reserve(GPU_UUIDS[1], 10, phase='main', cpus=1)
    assert (ledger.root / 'ledger.json').read_bytes() == before
    assert not (ledger.root / 'ledger.lock').exists()


def test_job_reader_waits_for_cooperating_writer(tmp_path):
    ledger = AllocationLedger(owned(tmp_path))
    token = ledger.reserve(GPU_UUIDS[0], 10, phase='main', cpus=1)
    ready = threading.Event()
    def writer():
        with ledger._locked():
            ready.set()
            time.sleep(.1)
    thread = threading.Thread(target=writer)
    thread.start()
    assert ready.wait(2)
    try:
        assert ledger.get_job(token)['id'] == token
    finally:
        thread.join(2)
