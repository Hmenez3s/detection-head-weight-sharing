import json

import pytest

from head_sharing.runtime import AllocationLedger, GPU_UUIDS, RuntimeErrorSafe, initialize


def owned(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    (root / "STUDY_OWNER.txt").write_text("synthetic-test-ledger\n")
    return initialize(root, "ledger")


def extended(ledger):
    ledger.authorize_extension(reason="approved study", evidence="authorization-01")


def amend(ledger, *, total=180_000, main=147_600, final=21_600, contingency=10_800,
          reason="approved finite tranche", evidence="authorization-02"):
    return ledger.authorize_amendment(total_seconds=total, main_seconds=main,
                                      final_seconds=final, contingency_seconds=contingency,
                                      reason=reason, evidence=evidence)


def test_amendment_appends_chain_and_preserves_debits_jobs_and_first_extension_replay(tmp_path):
    ledger = AllocationLedger(owned(tmp_path))
    old = ledger.reserve(GPU_UUIDS[0], 46_800, phase="main", cpus=2)
    ledger.update_job(old, status="failed", exit_code=-9)
    extended(ledger)
    intervening = ledger.reserve(GPU_UUIDS[1], 12, phase="main", cpus=1)
    ledger.update_job(intervening, status="failed", exit_code=-9)
    before = ledger._load()
    assert amend(ledger)
    after = ledger._load()

    assert after["policy"] == {
        "total_seconds": 180_000, "main_seconds": 147_600,
        "final_seconds": 21_600, "contingency_seconds": 10_800,
        "max_active_jobs_per_uuid": 2,
    }
    assert len(after["policy_history"]) == 2
    assert after["policy_history"][1]["before_caps"] == before["policy"]
    assert after["policy_history"][1]["before_balances"] == after["policy_history"][1]["after_balances"]
    assert after["devices"] == before["devices"]
    assert ledger.get_job(old)["seconds"] == 46_800

    # Historical extension replay stays a no-op and never rolls caps back.
    replay_before = (ledger.root / "ledger.json").read_bytes()
    assert not ledger.authorize_extension(reason="approved study", evidence="authorization-01")
    assert (ledger.root / "ledger.json").read_bytes() == replay_before
    assert ledger._load()["policy"] == after["policy"]
    assert not amend(ledger)
    assert (ledger.root / "ledger.json").read_bytes() == replay_before
    token = ledger.reserve(GPU_UUIDS[0], 100_000, phase="main", cpus=2)
    assert ledger.get_job(token)["seconds"] == 100_000


def test_legacy_unextended_and_first_extension_states_remain_valid(tmp_path):
    ledger = AllocationLedger(owned(tmp_path))
    assert ledger._load().get("policy") is None
    token = ledger.reserve(GPU_UUIDS[0], 9, phase="main", cpus=1)
    assert ledger.get_job(token)["status"] == "reserved"
    assert ledger.authorize_extension(reason="approved", evidence="receipt")
    assert ledger._load()["policy"]["total_seconds"] == 172_800
    assert ledger.reserve(GPU_UUIDS[0], 9, phase="main", cpus=1)


def test_rejects_invalid_caps_decreases_and_conflicting_replay_without_write(tmp_path):
    ledger = AllocationLedger(owned(tmp_path))
    extended(ledger)
    bad_inputs = [
        {"total": True, "main": 147_601, "final": 21_599, "contingency": 10_800},
        {"total": 180_000, "main": 147_599, "final": 21_600, "contingency": 10_800},
        {"total": 180_000, "main": 147_600.0, "final": 21_600, "contingency": 10_800},
        {"total": 180_000, "main": 147_600, "final": 21_600, "contingency": 10_801},
        {"total": 172_799, "main": 140_399, "final": 21_600, "contingency": 10_800},
    ]
    for caps in bad_inputs:
        before = (ledger.root / "ledger.json").read_bytes()
        with pytest.raises(RuntimeErrorSafe):
            ledger.authorize_amendment(total_seconds=caps["total"], main_seconds=caps["main"],
                final_seconds=caps["final"], contingency_seconds=caps["contingency"],
                reason="approved", evidence="receipt")
        assert (ledger.root / "ledger.json").read_bytes() == before

    assert amend(ledger)
    before = (ledger.root / "ledger.json").read_bytes()
    with pytest.raises(RuntimeErrorSafe, match="monotonic"):
        amend(ledger, total=180_001, main=147_599, final=21_601,
              contingency=10_801, evidence="different-receipt")
    assert (ledger.root / "ledger.json").read_bytes() == before


def test_amendment_requires_all_uuids_idle_and_rejected_attempt_is_byte_identical(tmp_path):
    ledger = AllocationLedger(owned(tmp_path))
    extended(ledger)
    token = ledger.reserve(GPU_UUIDS[1], 9, phase="main", cpus=1)
    for status in ("reserved", "launching", "running", "active-unknown"):
        ledger.update_job(token, status=status)
        before = (ledger.root / "ledger.json").read_bytes()
        with pytest.raises(RuntimeErrorSafe, match="active reservations"):
            amend(ledger)
        assert (ledger.root / "ledger.json").read_bytes() == before
    ledger.update_job(token, status="failed")
    assert amend(ledger)


def test_history_validation_rejects_tampered_earlier_link_and_balance(tmp_path):
    ledger = AllocationLedger(owned(tmp_path))
    extended(ledger)
    amend(ledger)
    path = ledger.root / "ledger.json"
    state = json.loads(path.read_text())
    state["policy_history"][0]["after_caps"]["main_seconds"] += 1
    path.write_text(json.dumps(state))
    with pytest.raises(RuntimeErrorSafe, match="history"):
        ledger._load()

    state = json.loads(path.read_text())
    state["policy_history"][0]["after_caps"]["main_seconds"] -= 1
    state["policy_history"][0]["after_balances"][GPU_UUIDS[0]]["main"] += 1
    path.write_text(json.dumps(state))
    with pytest.raises(RuntimeErrorSafe, match="history"):
        ledger._load()


def test_cli_amendment_is_explicit_and_reports_idempotent_replay(tmp_path, capsys):
    from head_sharing.runtime import main

    root = owned(tmp_path)
    ledger = AllocationLedger(root)
    extended(ledger)
    args = ["authorize-amendment", "--root", str(root), "--total-seconds", "180000",
            "--main-seconds", "147600", "--final-seconds", "21600",
            "--contingency-seconds", "10800", "--reason", "approved", "--evidence", "receipt"]
    assert main(args) == 0
    assert capsys.readouterr().out.strip() == "authorized"
    assert main(args) == 0
    assert capsys.readouterr().out.strip() == "already-authorized"
    assert AllocationLedger(root)._load()["policy"]["total_seconds"] == 180_000
