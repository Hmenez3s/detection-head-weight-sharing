"""Fail-closed UUID allocation ledger. No CUDA module is imported here."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

GPU_UUIDS = (
    "GPU-13916c7e-aa3c-fa5e-48ef-7b8cca10ea99",
    "GPU-08055392-87d6-8287-947d-0bf94c711445",
)
CAP_SECONDS = 57_600
FINAL_SECONDS = 7_200
CONTINGENCY_SECONDS = 3_600
MAX_CPUS = 12
LEDGER_LOCK_WAIT_SECONDS = 2.0
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
        os.replace(temp_name, path)
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

    @contextmanager
    def _locked(self):
        lock = self.root / "ledger.lock"
        deadline = time.monotonic() + LEDGER_LOCK_WAIT_SECONDS
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                break
            except FileExistsError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeErrorSafe("ledger lock persists; unknown/stale owner requires operator review") from exc
                time.sleep(min(0.05, max(0.005, deadline - time.monotonic())))
        try:
            os.write(fd, json.dumps({"pid": os.getpid(), "created": self.clock()}).encode())
            os.fsync(fd)
            os.close(fd)
            yield
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass

    def _load(self):
        try:
            state = json.loads((self.root / "ledger.json").read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeErrorSafe("ledger unreadable; refusing allocation") from exc
        if state.get("version") != 1 or set(state.get("devices", {})) != set(GPU_UUIDS):
            raise RuntimeErrorSafe("ledger UUID/config mismatch")
        return state

    def reserve(self, device_uuid: str, seconds: int, *, phase: str, cpus: int = 6, command: list[str] | None = None) -> str:
        if device_uuid not in GPU_UUIDS or phase not in ("main", "final", "contingency"):
            raise RuntimeErrorSafe("unknown UUID or allocation phase")
        if type(seconds) is not int or seconds < MIN_ALLOCATION_SECONDS or type(cpus) is not int or cpus <= 0 or cpus > 12:
            raise RuntimeErrorSafe("invalid duration or CPU allocation")
        with self._locked():
            state = self._load()
            device = state["devices"][device_uuid]
            active = {"reserved", "launching", "running", "active-unknown"}
            if any(j["status"] in active for j in device["jobs"]):
                raise RuntimeErrorSafe("physical UUID already has a reserved job")
            limits = {"main": CAP_SECONDS-FINAL_SECONDS-CONTINGENCY_SECONDS, "final": FINAL_SECONDS, "contingency": CONTINGENCY_SECONDS}
            if device[phase] + seconds > limits[phase]:
                raise RuntimeErrorSafe("allocation exceeds phase balance")
            if sum(cpus for d in state["devices"].values() for j in d["jobs"] if j["status"] in {"reserved", "launching", "running", "active-unknown"} for cpus in [j["cpus"]]) + cpus > MAX_CPUS:
                raise RuntimeErrorSafe("aggregate CPU allocation exceeds 12")
            token = str(uuid.uuid4())
            device[phase] += seconds
            device["jobs"].append({"id": token, "uuid": device_uuid, "phase": phase, "seconds": seconds, "cpus": cpus, "command": command or [], "created": self.clock(), "pid": None, "pid_identity": None, "started": None, "ended": None, "exit_code": None, "progress": None, "status": "reserved"})
            _atomic_json(self.root / "ledger.json", state)
            return token

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
        for device in self._load()["devices"].values():
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
    args = parser.parse_args(argv)
    if args.action == "init":
        initialize(args.owned_study_root, args.subroot); return 0
    ledger = AllocationLedger(args.root)
    if args.action == "status":
        state = ledger._load()
        public = {device_uuid: {"main": device["main"], "final": device["final"],
                                "contingency": device["contingency"],
                                "jobs": [{key: job.get(key) for key in ("id", "phase", "seconds", "cpus", "status", "pid", "started", "ended", "exit_code", "progress")}
                                         for job in device["jobs"]]}
                  for device_uuid, device in state["devices"].items()}
        print(json.dumps(public, sort_keys=True)); return 0
    print(ledger.reserve(args.uuid, args.seconds, phase=args.phase, cpus=args.cpus)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
