"""Fail-closed UUID allocation ledger. No CUDA module is imported here."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid

GPU_UUIDS = (
    "GPU-13916c7e-aa3c-fa5e-48ef-7b8cca10ea99",
    "GPU-08055392-87d6-8287-947d-0bf94c711445",
)
CAP_SECONDS = 57_600
FINAL_SECONDS = 7_200
CONTINGENCY_SECONDS = 3_600
EXTENDED_CAP_SECONDS = 172_800
EXTENDED_FINAL_SECONDS = 21_600
EXTENDED_CONTINGENCY_SECONDS = 10_800
MAX_ACTIVE_JOBS_PER_UUID = 2
MAX_CPUS = 12
ACTIVE_STATUSES = {"reserved", "launching", "running", "active-unknown"}

LEDGER_LOCK_WAIT_SECONDS = 10.0
CLEANUP_SAFETY_SECONDS = 8
MIN_ALLOCATION_SECONDS = CLEANUP_SAFETY_SECONDS + 1


class RuntimeErrorSafe(RuntimeError):
    """An allocation or ownership invariant failed; callers must stop."""


def canonical_gpu_uuid(value: str) -> str:
    """Unify PyTorch's bare UUID and NVML's GPU-prefixed physical identity."""
    if not isinstance(value, str):
        raise RuntimeErrorSafe("invalid GPU UUID representation")
    raw = value.removeprefix("GPU-")
    if len(raw) != 36 or any(raw[i] != "-" for i in (8, 13, 18, 23)):
        raise RuntimeErrorSafe("invalid GPU UUID representation")
    try:
        return "GPU-" + str(uuid.UUID(raw))
    except ValueError as exc:
        raise RuntimeErrorSafe("invalid GPU UUID representation") from exc


def _is_link_or_junction(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    return any(_is_link_or_junction(part) for part in (absolute, *absolute.parents))


def _safe_child(parent: Path, name: str) -> Path:
    if not name or name in (".", "..") or Path(name).name != name:
        raise RuntimeErrorSafe("invalid ledger subroot")
    if _has_symlink_component(parent):
        raise RuntimeErrorSafe("linked ownership root")
    resolved_parent = parent.resolve(strict=True)
    if parent.absolute() != resolved_parent:
        raise RuntimeErrorSafe("owned study root must be canonical")
    parent = resolved_parent
    child = parent / name
    if child.exists() or _is_link_or_junction(child):
        raise RuntimeErrorSafe("ledger subroot already exists")
    if not (parent / "STUDY_OWNER.txt").is_file():
        raise RuntimeErrorSafe("owned study root marker missing")
    return child


def initialize(owned_study_root: str | Path, subroot: str) -> Path:
    """Create exactly one new ledger directory below an already-owned root."""
    parent = Path(owned_study_root)
    if _has_symlink_component(parent) or not parent.is_dir():
        raise RuntimeErrorSafe("owned study root must be an existing real directory without links")
    path = _safe_child(parent, subroot)
    path.mkdir(mode=0o700)
    (path / "LEDGER_OWNER").write_text("head-sharing-runtime-v1\n", encoding="utf-8")
    state = {"version": 1, "devices": {u: {"main": 0, "final": 0, "contingency": 0, "jobs": []} for u in GPU_UUIDS}}
    _atomic_json(path / "ledger.json", state)
    return path


def _atomic_json(path: Path, value: object) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=".ledger-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows readers/AV can briefly deny replacement even for an owned
        # writable file. Keep the durable temp and lock; never rewrite in place.
        deadline = time.monotonic() + 1.0
        while True:
            try:
                os.replace(temp_name, path)
                break
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in (5, 32, 33) or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class AllocationLedger:
    """Persistent, exclusive allocations charged in full at reservation time."""
    def __init__(self, ledger_root: str | Path, *, clock=time.time):
        self.root = Path(ledger_root)
        if (_has_symlink_component(self.root) or not self.root.is_dir() or
                self.root.absolute() != self.root.resolve(strict=True) or
                not (self.root / "LEDGER_OWNER").is_file()):
            raise RuntimeErrorSafe("ledger ownership proof missing or path is noncanonical/linked")
        self.clock = clock
        self._thread_lock = threading.Lock()

    @contextmanager
    def _locked(self):
        lock = self.root / "ledger.lock"
        deadline = time.monotonic() + LEDGER_LOCK_WAIT_SECONDS
        # Supervisors sharing this instance must queue locally instead of
        # repeatedly racing each other for the cross-process Windows lock.
        if not self._thread_lock.acquire(timeout=LEDGER_LOCK_WAIT_SECONDS):
            raise RuntimeErrorSafe("ledger thread lock persists; operator review required")
        try:
            while True:
                try:
                    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    break
                except (FileExistsError, PermissionError) as exc:
                    # Never remove or bypass another process's lock. Both
                    # local queueing and file acquisition share one deadline.
                    if time.monotonic() >= deadline:
                        raise RuntimeErrorSafe("ledger lock persists or is inaccessible; operator review required") from exc
                    time.sleep(min(0.05, max(0.005, deadline - time.monotonic())))
            try:
                try:
                    # This file is an ephemeral mutex, not a durable budget
                    # record. Flushing it made every READ wait a disk rotation;
                    # durable allocation writes still use _atomic_json/fsync.
                    os.write(fd, json.dumps({"pid": os.getpid(), "created": self.clock()}).encode())
                finally:
                    os.close(fd)
                yield
            finally:
                try:
                    lock.unlink()
                except FileNotFoundError:
                    pass
        finally:
            self._thread_lock.release()

    @staticmethod
    def _base_caps():
        return {"total_seconds": CAP_SECONDS,
                "main_seconds": CAP_SECONDS-FINAL_SECONDS-CONTINGENCY_SECONDS,
                "final_seconds": FINAL_SECONDS, "contingency_seconds": CONTINGENCY_SECONDS,
                "max_active_jobs_per_uuid": 1}

    @staticmethod
    def _extended_caps():
        return {"total_seconds": EXTENDED_CAP_SECONDS,
                "main_seconds": EXTENDED_CAP_SECONDS-EXTENDED_FINAL_SECONDS-EXTENDED_CONTINGENCY_SECONDS,
                "final_seconds": EXTENDED_FINAL_SECONDS,
                "contingency_seconds": EXTENDED_CONTINGENCY_SECONDS,
                "max_active_jobs_per_uuid": MAX_ACTIVE_JOBS_PER_UUID}

    @staticmethod
    def _valid_caps(caps):
        keys = {"total_seconds", "main_seconds", "final_seconds", "contingency_seconds",
                "max_active_jobs_per_uuid"}
        return (isinstance(caps, dict) and set(caps) == keys and
                all(type(caps[key]) is int and caps[key] >= 0 for key in keys) and
                caps["total_seconds"] == caps["main_seconds"] + caps["final_seconds"] + caps["contingency_seconds"] and
                caps["max_active_jobs_per_uuid"] in (1, MAX_ACTIVE_JOBS_PER_UUID))

    @staticmethod
    def _valid_balances(balances):
        phases = {"main", "final", "contingency"}
        return (isinstance(balances, dict) and set(balances) == set(GPU_UUIDS) and
                all(isinstance(balances[device], dict) and set(balances[device]) == phases and
                    all(type(value) is int and value >= 0 for value in balances[device].values())
                    for device in GPU_UUIDS))

    @classmethod
    def _validate_policy_history(cls, policy, history):
        if not isinstance(history, list):
            raise RuntimeErrorSafe("invalid policy history")
        if policy is None:
            if history:
                raise RuntimeErrorSafe("policy history exists without authorized policy")
            return
        if not history:
            raise RuntimeErrorSafe("authorized policy has no history")
        prior_caps = cls._base_caps()
        for index, record in enumerate(history):
            if not isinstance(record, dict):
                raise RuntimeErrorSafe("invalid policy history")
            before, after = record.get("before_caps"), record.get("after_caps")
            before_balances, after_balances = record.get("before_balances"), record.get("after_balances")
            authorization = record.get("authorization")
            if (not cls._valid_caps(before) or not cls._valid_caps(after) or before != prior_caps or
                    not cls._valid_balances(before_balances) or not cls._valid_balances(after_balances) or
                    before_balances != after_balances or
                    not isinstance(authorization, dict) or
                    set(authorization) != {"reason", "evidence", *after.keys()} or
                    not isinstance(authorization.get("reason"), str) or not authorization["reason"].strip() or
                    not isinstance(authorization.get("evidence"), str) or not authorization["evidence"].strip() or
                    {key: authorization.get(key) for key in after} != after or
                    type(record.get("created")) not in (int, float) or
                    (type(record.get("created")) is float and not math.isfinite(record["created"])) or
                    record["created"] < 0):
                raise RuntimeErrorSafe("invalid or unauthorised ledger policy history")
            if index == 0:
                if after != cls._extended_caps():
                    raise RuntimeErrorSafe("invalid historical budget extension")
            elif (before["max_active_jobs_per_uuid"] != MAX_ACTIVE_JOBS_PER_UUID or
                  after["max_active_jobs_per_uuid"] != MAX_ACTIVE_JOBS_PER_UUID or
                  any(after[key] < before[key] for key in
                      ("total_seconds", "main_seconds", "final_seconds", "contingency_seconds")) or
                  not any(after[key] > before[key] for key in
                          ("total_seconds", "main_seconds", "final_seconds", "contingency_seconds"))):
                raise RuntimeErrorSafe("budget amendment must be monotonic and increase a cap")
            prior_caps = after
        if policy != prior_caps:
            raise RuntimeErrorSafe("effective policy does not match authorization history")

    def _load(self):
        try:
            state = json.loads((self.root / "ledger.json").read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeErrorSafe("ledger unreadable; refusing allocation") from exc
        if state.get("version") != 1 or set(state.get("devices", {})) != set(GPU_UUIDS):
            raise RuntimeErrorSafe("ledger UUID/config mismatch")
        self._validate_policy_history(state.get("policy"), state.get("policy_history", []))
        return state

    def authorize_extension(self, *, reason: str, evidence: str) -> bool:
        """Apply the historical first extension, idempotent even after amendments."""
        if not isinstance(reason, str) or not reason.strip() or not isinstance(evidence, str) or not evidence.strip():
            raise RuntimeErrorSafe("authorization reason and evidence are required")
        authorization = {"reason": reason.strip(), "evidence": evidence.strip(), **self._extended_caps()}
        with self._locked():
            state = self._load()
            history = state.get("policy_history", [])
            if state.get("policy") is not None:
                if history[0]["authorization"] == authorization:
                    return False
                raise RuntimeErrorSafe("policy is already authorized; refusing policy mutation")
            before = {u: {k: state["devices"][u][k] for k in ("main", "final", "contingency")} for u in GPU_UUIDS}
            after_caps = self._extended_caps()
            state["policy"] = after_caps
            state.setdefault("policy_history", []).append({"authorization": authorization,
                "before_caps": self._base_caps(), "after_caps": after_caps, "before_balances": before,
                "after_balances": {u: dict(before[u]) for u in GPU_UUIDS}, "created": self.clock()})
            _atomic_json(self.root / "ledger.json", state)
            return True

    def authorize_amendment(self, *, total_seconds: int, main_seconds: int, final_seconds: int,
                            contingency_seconds: int, reason: str, evidence: str) -> bool:
        """Append a finite, monotonic budget tranche without changing old debits."""
        caps = {"total_seconds": total_seconds, "main_seconds": main_seconds,
                "final_seconds": final_seconds, "contingency_seconds": contingency_seconds,
                "max_active_jobs_per_uuid": MAX_ACTIVE_JOBS_PER_UUID}
        if (not self._valid_caps(caps) or not isinstance(reason, str) or not reason.strip() or
                not isinstance(evidence, str) or not evidence.strip()):
            raise RuntimeErrorSafe("finite integer caps, reason, and evidence are required")
        authorization = {"reason": reason.strip(), "evidence": evidence.strip(), **caps}
        with self._locked():
            state = self._load()
            history = state.get("policy_history", [])
            if state.get("policy") is None:
                raise RuntimeErrorSafe("apply the historical extension before a budget amendment")
            if any(record["authorization"] == authorization for record in history):
                return False
            before_caps = state["policy"]
            if (any(caps[key] < before_caps[key] for key in
                    ("total_seconds", "main_seconds", "final_seconds", "contingency_seconds")) or
                    not any(caps[key] > before_caps[key] for key in
                            ("total_seconds", "main_seconds", "final_seconds", "contingency_seconds"))):
                raise RuntimeErrorSafe("budget amendment must be monotonic and increase a cap")
            if any(job["status"] in ACTIVE_STATUSES for device in state["devices"].values()
                   for job in device["jobs"]):
                raise RuntimeErrorSafe("active reservations block budget amendment")
            balances = {u: {k: state["devices"][u][k] for k in ("main", "final", "contingency")}
                        for u in GPU_UUIDS}
            state["policy"] = caps
            history.append({"authorization": authorization, "before_caps": before_caps,
                            "after_caps": caps, "before_balances": balances,
                            "after_balances": {u: dict(balances[u]) for u in GPU_UUIDS},
                            "created": self.clock()})
            state["policy_history"] = history
            _atomic_json(self.root / "ledger.json", state)
            return True

    def reserve(self, device_uuid: str, seconds: int, *, phase: str, cpus: int = 6, command: list[str] | None = None,
                peak_memory_bytes: int | None = None) -> str:
        if device_uuid not in GPU_UUIDS or phase not in ("main", "final", "contingency"):
            raise RuntimeErrorSafe("unknown UUID or allocation phase")
        if type(seconds) is not int or seconds < MIN_ALLOCATION_SECONDS or type(cpus) is not int or cpus <= 0 or cpus > 12:
            raise RuntimeErrorSafe("invalid duration or CPU allocation")
        with self._locked():
            state = self._load()
            device = state["devices"][device_uuid]
            active = ACTIVE_STATUSES
            policy = state.get("policy")
            max_slots = 1
            limits = {"main": CAP_SECONDS-FINAL_SECONDS-CONTINGENCY_SECONDS, "final": FINAL_SECONDS, "contingency": CONTINGENCY_SECONDS}
            if policy is not None:
                limits = {"main": policy["main_seconds"], "final": policy["final_seconds"],
                          "contingency": policy["contingency_seconds"]}
                max_slots = policy["max_active_jobs_per_uuid"]
            if peak_memory_bytes is not None and (type(peak_memory_bytes) is not int or peak_memory_bytes <= 0):
                raise RuntimeErrorSafe("peak memory commitment must be a positive integer")
            if sum(j["status"] in active for j in device["jobs"]) >= max_slots:
                raise RuntimeErrorSafe("physical UUID already has the maximum reserved jobs")
            if device[phase] + seconds > limits[phase]:
                raise RuntimeErrorSafe("allocation exceeds phase balance")
            if sum(cpus for d in state["devices"].values() for j in d["jobs"] if j["status"] in {"reserved", "launching", "running", "active-unknown"} for cpus in [j["cpus"]]) + cpus > MAX_CPUS:
                raise RuntimeErrorSafe("aggregate CPU allocation exceeds 12")
            token = str(uuid.uuid4())
            device[phase] += seconds
            job = {"id": token, "uuid": device_uuid, "phase": phase, "seconds": seconds, "cpus": cpus,
                   "command": command or [], "created": self.clock(), "pid": None, "pid_identity": None,
                   "started": None, "ended": None, "exit_code": None, "progress": None, "status": "reserved"}
            if peak_memory_bytes is not None:
                job.update(peak_memory_bytes=peak_memory_bytes, headroom_pending_bytes=peak_memory_bytes)
            device["jobs"].append(job)
            _atomic_json(self.root / "ledger.json", state)
            return token

    def guard_headroom(self, token: str, sample: dict, *, admission: bool) -> dict:
        """Validate and record mocked/injected physical-memory evidence under lock."""
        with self._locked():
            state = self._load()
            if state.get("policy") is None:
                raise RuntimeErrorSafe("headroom guard requires explicit authorized policy")
            found = None
            device = None
            for candidate in state["devices"].values():
                for job in candidate["jobs"]:
                    if job["id"] == token:
                        found, device = job, candidate
                        break
            if found is None or found["status"] not in ACTIVE_STATUSES:
                raise RuntimeErrorSafe("headroom guard requires an active reservation")
            if not isinstance(sample, dict) or sample.get("uuid") != found["uuid"]:
                raise RuntimeErrorSafe("physical-memory UUID mismatch")
            free, total, sampled_at = sample.get("free_bytes"), sample.get("total_bytes"), sample.get("sampled_at")
            if (type(free) is not int or type(total) is not int or free < 0 or total <= 0 or free > total or
                    type(sampled_at) not in (int, float) or not (0 <= self.clock() - sampled_at <= 15)):
                raise RuntimeErrorSafe("physical-memory sample is malformed or stale")
            if admission:
                peak = found.get("peak_memory_bytes")
                if type(peak) is not int or peak <= 0:
                    raise RuntimeErrorSafe("guarded launch requires a declared positive peak memory commitment")
                pending = sum(j.get("headroom_pending_bytes") or 0 for j in device["jobs"]
                              if j is not found and j["status"] in ACTIVE_STATUSES)
                required = (total + 9) // 10 + pending + peak
            else:
                required = (total + 9) // 10
            receipt = {"uuid": found["uuid"], "free_bytes": free, "total_bytes": total,
                       "sampled_at": sampled_at, "required_free_bytes": required,
                       "admission": bool(admission), "passed": free >= required}
            summary = found.setdefault("headroom_summary", {
                "sample_count": 0, "min_free_fraction": None, "min_free_bytes": None,
                "min_total_bytes": None, "min_free_fraction_sample": None,
                "min_free_bytes_sampled_at": None, "min_total_bytes_sampled_at": None,
                "first_sampled_at": sampled_at, "last_sampled_at": sampled_at})
            summary["sample_count"] += 1
            summary["last_sampled_at"] = sampled_at
            fraction = free / total
            if summary["min_free_fraction"] is None or fraction < summary["min_free_fraction"]:
                summary["min_free_fraction"] = fraction
                summary["min_free_fraction_sample"] = {"free_bytes": free, "total_bytes": total,
                                                        "sampled_at": sampled_at}
            if summary["min_free_bytes"] is None or free < summary["min_free_bytes"]:
                summary["min_free_bytes"] = free
                summary["min_free_bytes_sampled_at"] = sampled_at
            if summary["min_total_bytes"] is None or total < summary["min_total_bytes"]:
                summary["min_total_bytes"] = total
                summary["min_total_bytes_sampled_at"] = sampled_at
            receipts = found.setdefault("headroom_receipts", [])
            receipts.append(receipt)
            if len(receipts) > 32:
                indexed = list(enumerate(receipts))
                admission_indexes = [i for i, entry in indexed if entry.get("admission")]
                failed_indexes = [i for i, entry in indexed if not entry.get("passed")]
                keep = set(admission_indexes[:1])
                for index in reversed(failed_indexes):
                    if len(keep) == 32:
                        break
                    keep.add(index)
                for index in range(len(receipts)-1, -1, -1):
                    if len(keep) == 32:
                        break
                    keep.add(index)
                found["headroom_receipts"] = [entry for i, entry in indexed if i in keep]
            if not receipt["passed"]:
                _atomic_json(self.root / "ledger.json", state)
                raise RuntimeErrorSafe("physical-memory headroom below required margin")
            _atomic_json(self.root / "ledger.json", state)
            return receipt

    def validate_device(self, token: str, actual_uuid: str, *, cuda_module=None):
        """Validate the exact reserved physical UUID from a guarded child."""
        job = self.get_job(token)
        if job["status"] not in {"launching", "running"} or canonical_gpu_uuid(actual_uuid) != job["uuid"]:
            raise RuntimeErrorSafe("device does not match active reservation")
        return True

    def claim_launch(self, token: str):
        """Atomically claim a reservation once before any filesystem/process mutation."""
        with self._locked():
            state = self._load()
            for device in state["devices"].values():
                for job in device["jobs"]:
                    if job["id"] == token:
                        if job["status"] != "reserved":
                            raise RuntimeErrorSafe("reservation is not launchable")
                        job["status"] = "launching"
                        _atomic_json(self.root / "ledger.json", state)
                        return dict(job)
        raise RuntimeErrorSafe("unknown reservation")

    def get_job(self, token: str):
        # Cooperative reads must not hold a Windows handle across a writer's
        # atomic replacement. Noncooperative readers are covered by bounded retry.
        with self._locked():
            state = self._load()
        for device in state["devices"].values():
            for job in device["jobs"]:
                if job["id"] == token:
                    return dict(job)
        raise RuntimeErrorSafe("unknown reservation")

    def update_job(self, token: str, **updates):
        with self._locked():
            state = self._load()
            for device in state["devices"].values():
                for job in device["jobs"]:
                    if job["id"] == token:
                        job.update(updates)
                        _atomic_json(self.root / "ledger.json", state)
                        return dict(job)
        raise RuntimeErrorSafe("unknown reservation")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init", help="initialize a new ledger subroot")
    init.add_argument("--owned-study-root", required=True)
    init.add_argument("--subroot", required=True)
    status = sub.add_parser("status", help="show UUID balances (no paths)")
    status.add_argument("--root", required=True)
    reserve = sub.add_parser("reserve", help="reserve a bounded allocation")
    reserve.add_argument("--root", required=True); reserve.add_argument("--uuid", required=True)
    reserve.add_argument("--seconds", type=int, required=True); reserve.add_argument("--phase", choices=("main", "final", "contingency"), required=True)
    reserve.add_argument("--cpus", type=int, default=6)
    reserve.add_argument("--peak-memory-bytes", type=int)
    extend = sub.add_parser("authorize-extension", help="apply the historical first cumulative-budget extension")
    extend.add_argument("--root", required=True); extend.add_argument("--reason", required=True); extend.add_argument("--evidence", required=True)
    amend = sub.add_parser("authorize-amendment", help="append a finite cumulative-budget amendment")
    amend.add_argument("--root", required=True)
    amend.add_argument("--total-seconds", type=int, required=True)
    amend.add_argument("--main-seconds", type=int, required=True)
    amend.add_argument("--final-seconds", type=int, required=True)
    amend.add_argument("--contingency-seconds", type=int, required=True)
    amend.add_argument("--reason", required=True); amend.add_argument("--evidence", required=True)
    args = parser.parse_args(argv)
    if args.action == "init":
        initialize(args.owned_study_root, args.subroot); return 0
    ledger = AllocationLedger(args.root)
    if args.action == "authorize-extension":
        print("authorized" if ledger.authorize_extension(reason=args.reason, evidence=args.evidence) else "already-authorized")
        return 0
    if args.action == "authorize-amendment":
        applied = ledger.authorize_amendment(total_seconds=args.total_seconds, main_seconds=args.main_seconds,
            final_seconds=args.final_seconds, contingency_seconds=args.contingency_seconds,
            reason=args.reason, evidence=args.evidence)
        print("authorized" if applied else "already-authorized")
        return 0
    if args.action == "status":
        with ledger._locked():
            state = ledger._load()
        public = {device_uuid: {"main": device["main"], "final": device["final"],
                                "contingency": device["contingency"],
                                "effective_policy": state.get("policy", AllocationLedger._base_caps()),
                                "jobs": [{key: job.get(key) for key in ("id", "phase", "seconds", "cpus", "status", "pid", "started", "ended", "exit_code", "progress")}
                                         for job in device["jobs"]]}
                  for device_uuid, device in state["devices"].items()}
        print(json.dumps(public, sort_keys=True)); return 0
    print(ledger.reserve(args.uuid, args.seconds, phase=args.phase, cpus=args.cpus,
                         peak_memory_bytes=args.peak_memory_bytes)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
