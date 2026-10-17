"""Bounded synthetic/worker process supervision tied to ledger reservations."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .runtime import (AllocationLedger, CLEANUP_SAFETY_SECONDS,
                     MIN_ALLOCATION_SECONDS, RuntimeErrorSafe, _has_symlink_component)

# Fine-grained progress lives in the job's append-only file; the durable
# allocation ledger only needs a periodic snapshot plus terminal status.
LEDGER_PROGRESS_INTERVAL_SECONDS = 5.0


def windows_detached_flags() -> int:
    """Exact flags proven to survive the SSH job boundary on supported Windows."""
    # Values are Windows API creation flags; Python exposes them only on Windows.
    return (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200) |
            getattr(subprocess, "DETACHED_PROCESS", 0x00000008) |
            getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000))


def child_entry(ledger: AllocationLedger, token: str, expected_uuid: str, actual_uuid: str | None = None,
                *, gate_timeout: float = 60):
    """Wait for containment setup, then validate UUID before native imports."""
    gate_text = os.environ.get("HEAD_SHARING_LAUNCH_GATE")
    if not gate_text:
        raise RuntimeErrorSafe("guarded launch gate is missing")
    gate = Path(gate_text)
    deadline = time.monotonic() + gate_timeout
    while not gate.is_file():
        if time.monotonic() >= deadline:
            raise RuntimeErrorSafe("supervisor did not establish child containment")
        time.sleep(0.05)
    job = ledger.get_job(token)
    if (job["status"] not in {"launching", "running"} or job["uuid"] != expected_uuid or
            os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid):
        raise RuntimeErrorSafe("child reservation/assigned UUID mismatch")
    if actual_uuid is not None:
        ledger.validate_device(token, actual_uuid)


def _pid_identity(pid: int) -> str:
    # Process creation identity prevents signaling a reused PID.
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        return stat.read_text().split()[21]
    if sys.platform == "darwin":
        result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], check=False,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class FILETIME(ctypes.Structure):
            _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME),
                                           ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME)]
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            raise RuntimeErrorSafe("cannot verify child process identity")
        try:
            created, exited, kernel_time, user_time = FILETIME(), FILETIME(), FILETIME(), FILETIME()
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time)):
                raise RuntimeErrorSafe("cannot read child process creation identity")
            return str((created.high << 32) | created.low)
        finally:
            kernel.CloseHandle(handle)
    raise RuntimeErrorSafe("process creation identity unavailable")


class _WindowsJob:
    """Documented Windows Job Object with kill-on-close containment."""
    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ctypes, self.wintypes = ctypes, wintypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise RuntimeErrorSafe("cannot create owned Windows process job")
        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [("ReadOperationCount", ctypes.c_ulonglong), ("WriteOperationCount", ctypes.c_ulonglong),
                        ("OtherOperationCount", ctypes.c_ulonglong), ("ReadTransferCount", ctypes.c_ulonglong),
                        ("WriteTransferCount", ctypes.c_ulonglong), ("OtherTransferCount", ctypes.c_ulonglong)]
        class EXT(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]
        info = EXT(); info.BasicLimitInformation.LimitFlags = 0x2000
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.close(); raise RuntimeErrorSafe("cannot enable Windows kill-on-close")

    def assign(self, pid: int):
        self.kernel.OpenProcess.argtypes = [self.wintypes.DWORD, self.wintypes.BOOL, self.wintypes.DWORD]
        self.kernel.OpenProcess.restype = self.wintypes.HANDLE
        process = self.kernel.OpenProcess(0x0100 | 0x0001, False, pid)
        if not process:
            raise RuntimeErrorSafe("cannot open child for Windows job assignment")
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle, process):
                raise RuntimeErrorSafe("cannot contain child in Windows job")
        finally:
            self.kernel.CloseHandle(process)

    def close(self):
        if getattr(self, "handle", None):
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    if sys.platform == "darwin":
        result = subprocess.run(["ps", "-axo", "pgid=,stat="], check=False,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True)
        if result.returncode == 0:
            for row in result.stdout.splitlines():
                fields = row.split()
                if len(fields) >= 2 and fields[0] == str(pgid) and not fields[1].startswith("Z"):
                    return True
            return False
    return True


def _terminate_owned(proc: subprocess.Popen, identity: str, *, grace: float = 2.0, windows_job=None):
    if os.name == "nt":
        if windows_job is None:
            raise RuntimeErrorSafe("Windows descendant containment unavailable")
        windows_job.close()
        try: proc.wait(timeout=grace)
        except subprocess.TimeoutExpired: raise RuntimeErrorSafe("Windows job did not reap")
        return
    if proc.poll() is None and _pid_identity(proc.pid) != identity:
        raise RuntimeErrorSafe("PID identity changed; refusing to terminate")
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        # The Popen handle still identifies our leader; never guess at another PID.
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=grace)
            except subprocess.TimeoutExpired: pass
    deadline = time.monotonic() + grace
    while _group_alive(proc.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _group_alive(proc.pid):
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    deadline = time.monotonic() + grace
    while _group_alive(proc.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _group_alive(proc.pid):
        raise RuntimeErrorSafe("owned process group remains after termination")
    try: proc.wait(timeout=grace)
    except subprocess.TimeoutExpired: raise RuntimeErrorSafe("owned process did not reap")


class _LaunchWatchdog:
    """Ledger-free deadline/progress enforcement for one contained child tree."""
    def __init__(self, proc: subprocess.Popen, identity: str, windows_job, *, deadline: float,
                 started: float, progress: Path, first_progress_seconds: float,
                 heartbeat_seconds: float):
        self.proc, self.identity, self.windows_job = proc, identity, windows_job
        self.deadline = deadline
        self.started = started
        self.progress = progress
        self.first_progress_seconds = first_progress_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.reason: str | None = None
        self.cleanup_error: BaseException | None = None
        self._stop_attempted = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._finished = threading.Event()
        self._thread = threading.Thread(target=self._watch, name="head-sharing-watchdog", daemon=False)
        self._thread.start()

    def _watch(self):
        try:
            while not self._stop_event.is_set() and self.proc.poll() is None:
                now = time.time()
                if now >= self.deadline:
                    self._stop("reserved allocation deadline exceeded")
                    return
                try:
                    stamp = self.progress.stat().st_mtime
                except FileNotFoundError:
                    stamp = None
                if stamp is None or stamp <= self.started:
                    if now - self.started > self.first_progress_seconds:
                        self._stop("first real progress deadline exceeded")
                        return
                elif now - stamp > self.heartbeat_seconds:
                    self._stop("progress heartbeat stalled")
                    return
                self._stop_event.wait(min(0.05, max(0.005, self.deadline-now)))
        except BaseException as exc:
            # A watchdog failure must not silently leave the owned child running.
            try:
                self._stop(f"watchdog enforcement failed ({type(exc).__name__})")
            except BaseException:
                pass
        finally:
            self._finished.set()

    def _stop(self, reason: str | None):
        with self._lock:
            if reason is not None and self.reason is None:
                self.reason = reason
            if self._stop_attempted:
                if self.cleanup_error is not None:
                    raise self.cleanup_error
                return
            self._stop_attempted = True
            try:
                _terminate_owned(self.proc, self.identity, windows_job=self.windows_job)
            except BaseException as exc:
                self.cleanup_error = exc
                raise
            finally:
                self._stop_event.set()

    def stop_tree(self):
        self._stop(None)

    def raise_if_fired(self):
        if self.reason is None:
            return
        if self.cleanup_error is not None:
            raise RuntimeErrorSafe(
                f"owned process tree cleanup unverified ({type(self.cleanup_error).__name__}: {self.cleanup_error}); allocation remains active"
            )
        raise TimeoutError(self.reason)

    def stop(self):
        self._stop_event.set()
        self._thread.join()


def physical_memory_metrics(device_uuid: str) -> dict:
    """Query bounded nvidia-smi physical-memory data for one exact UUID."""
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.free,memory.total",
                                 "--format=csv,noheader,nounits"], check=True, timeout=1.0,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True)
        rows = [line.split(",") for line in result.stdout.splitlines() if line.strip()]
        matches = [row for row in rows if len(row) == 3 and row[0].strip() == device_uuid]
        if len(matches) != 1:
            raise RuntimeErrorSafe("physical-memory query UUID missing or ambiguous")
        free_mib, total_mib = int(matches[0][1].strip()), int(matches[0][2].strip())
        return {"uuid": device_uuid, "free_bytes": free_mib * 1024 * 1024,
                "total_bytes": total_mib * 1024 * 1024, "sampled_at": time.time()}
    except RuntimeErrorSafe:
        raise
    except Exception as exc:
        raise RuntimeErrorSafe("bounded physical-memory query failed") from exc


def _launch_deadline(started: float, allocation_seconds: int, *, memory_guard: bool) -> float:
    """Keep the legacy cleanup window, reserving one query second only when guarded."""
    return started + allocation_seconds - CLEANUP_SAFETY_SECONDS - (1 if memory_guard else 0)


def launch(ledger: AllocationLedger, token: str, command: list[str], *, cwd: str | Path,
           log_path: str | Path, progress_path: str | Path, first_progress_seconds: float = 300,
           heartbeat_seconds: float = 120, env: dict[str, str] | None = None,
           popen_factory=subprocess.Popen, memory_guard: bool = False,
           memory_metrics=physical_memory_metrics, memory_poll_seconds: float = 5.0) -> int:
    if memory_guard and (isinstance(memory_poll_seconds, bool) or
                         not isinstance(memory_poll_seconds, (int, float)) or
                         not 0 < memory_poll_seconds <= 15 or
                         not math.isfinite(memory_poll_seconds)):
        raise RuntimeErrorSafe("memory polling interval must be finite and in (0, 15]")
    job = ledger.get_job(token)
    owner_root = ledger.root.parent
    requested_cwd = Path(cwd).absolute()
    if (_has_symlink_component(owner_root) or not (owner_root / "STUDY_OWNER.txt").is_file() or
            owner_root.absolute() != owner_root.resolve(strict=True) or
            _has_symlink_component(requested_cwd) or not requested_cwd.is_dir()):
        raise RuntimeErrorSafe("cwd and ledger owner root must be canonical, link-free directories")
    resolved_owner = owner_root.resolve(strict=True)
    job_root = requested_cwd.resolve(strict=True)
    if not job_root.is_relative_to(resolved_owner):
        raise RuntimeErrorSafe("cwd must resolve beneath the verified owned study root")
    if job["status"] != "reserved" or not command:
        raise RuntimeErrorSafe("launch requires a live reservation and explicit command")
    log, progress = Path(log_path).absolute(), Path(progress_path).absolute()
    if (_has_symlink_component(log) or _has_symlink_component(progress) or log.exists() or progress.exists() or
            log.parent != requested_cwd or progress.parent != requested_cwd):
        raise RuntimeErrorSafe("logs/progress must be new files owned by the verified job cwd")
    minimum_seconds = MIN_ALLOCATION_SECONDS + (1 if memory_guard else 0)
    if job["seconds"] < minimum_seconds:
        raise RuntimeErrorSafe("allocation is too short for bounded cleanup safety allowance")
    if memory_guard:
        try:
            ledger.guard_headroom(token, memory_metrics(job["uuid"]), admission=True)
        except BaseException as exc:
            ledger.update_job(token, status="failed", ended=time.time(),
                              failure=f"headroom-admission:{type(exc).__name__}",
                              headroom_failure=type(exc).__name__, headroom_pending_bytes=0)
            raise
    gate_path = job_root / ".supervisor-ready"
    if gate_path.exists() or _has_symlink_component(gate_path):
        raise RuntimeErrorSafe("launch gate path is not new and owned")
    started = time.time()
    job = ledger.claim_launch(token)
    cache_dirs = {
        "YOLO_CONFIG_DIR": job_root / "yolo",
        "TORCH_HOME": job_root / "torch",
        "MPLCONFIGDIR": job_root / "mpl",
        "TMPDIR": job_root / "tmp",
        "TEMP": job_root / "tmp",
        "TMP": job_root / "tmp",
        "CUDA_CACHE_PATH": job_root / "cuda-cache",
        "TORCH_EXTENSIONS_DIR": job_root / "torch-extensions",
        "TORCHINDUCTOR_CACHE_DIR": job_root / "torchinductor",
        # An importable cwd/triton directory shadows the absent optional package
        # as a namespace module and breaks torch.optim's Dynamo initialization.
        "TRITON_CACHE_DIR": job_root / "triton-cache",
    }
    for path in set(cache_dirs.values()):
        if _has_symlink_component(path) or not path.absolute().is_relative_to(job_root):
            raise RuntimeErrorSafe("job cache path escaped its owned root")
        path.mkdir(exist_ok=True)
    (cache_dirs["YOLO_CONFIG_DIR"] / "Ultralytics").mkdir(exist_ok=True)
    child_env = {**os.environ, **(env or {}), "HEAD_SHARING_RESERVATION": token,
                 "HEAD_SHARING_LAUNCH_GATE": str(gate_path), "HEAD_SHARING_LEDGER": str(ledger.root),
                 "HEAD_SHARING_CPU_BUDGET": str(job["cpus"]),
                 "CUDA_VISIBLE_DEVICES": job["uuid"], "YOLO_OFFLINE": "true"}
    child_env.update({key: str(path) for key, path in cache_dirs.items()})
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        child_env[key] = str(job["cpus"])
    kwargs = {"cwd": str(job_root), "stdin": subprocess.DEVNULL,
              "stdout": open(log, "xb"), "stderr": subprocess.STDOUT,
              "env": child_env, "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = windows_detached_flags()
    else:
        kwargs["start_new_session"] = True
    proc = None
    identity = None
    windows_job = None
    watchdog = None
    tree_stopped = False
    headroom_failure = None
    deadline = _launch_deadline(started, job["seconds"], memory_guard=memory_guard)
    try:
        if os.name == "nt":
            windows_job = _WindowsJob()
        proc = popen_factory(command, **kwargs)
        identity = (_pid_identity(proc.pid) if proc.poll() is None
                    else f"exited-owned-popen:{proc.pid}:{started}")
        if windows_job is not None:
            windows_job.assign(proc.pid)
        watchdog = _LaunchWatchdog(proc, identity, windows_job, deadline=deadline, started=started,
                                   progress=progress, first_progress_seconds=first_progress_seconds,
                                   heartbeat_seconds=heartbeat_seconds)
        ledger.update_job(token, pid=proc.pid, pid_identity=identity, started=started, status="running",
                          log=str(log), progress_file=str(progress), cwd=str(job_root))
        watchdog.raise_if_fired()
        gate_path.touch(exist_ok=False)
        watchdog.raise_if_fired()
        last_progress_stamp = None
        last_progress_commit = None
        last_memory_check = started
        while proc.poll() is None:
            now = time.time()
            if now >= deadline:
                raise TimeoutError("reserved allocation deadline exceeded")
            if memory_guard and now - last_memory_check >= memory_poll_seconds:
                try:
                    ledger.guard_headroom(token, memory_metrics(job["uuid"]), admission=False)
                except BaseException as exc:
                    headroom_failure = type(exc).__name__
                    raise
                last_memory_check = now
            if progress.exists():
                stamp = progress.stat().st_mtime
                if stamp > started:
                    if stamp != last_progress_stamp:
                        last_progress_stamp = stamp
                        if last_progress_commit is None or now-last_progress_commit >= LEDGER_PROGRESS_INTERVAL_SECONDS:
                            ledger.update_job(token, progress=stamp)
                            last_progress_commit = now
                    if now - stamp > heartbeat_seconds:
                        raise TimeoutError("progress heartbeat stalled")
                elif now - started > first_progress_seconds:
                    raise TimeoutError("first real progress deadline exceeded")
            elif now - started > first_progress_seconds:
                raise TimeoutError("first real progress deadline exceeded")
            time.sleep(min(0.25, max(0.01, deadline-now)))
        watchdog.raise_if_fired()
        code = proc.returncode
        watchdog.stop_tree()
        tree_stopped = True
        watchdog.stop()
        watchdog.raise_if_fired()
        ended = time.time()
        if progress.exists() and progress.stat().st_mtime > started:
            last_progress_stamp = progress.stat().st_mtime
        ledger.update_job(token, ended=ended, actual_wall_elapsed_seconds=ended-started,
                          progress=last_progress_stamp,
                          exit_code=code, status="complete" if code == 0 else "failed")
        return code
    except BaseException as exc:
        if proc is not None and not tree_stopped:
            try:
                if watchdog is not None and watchdog.reason is not None:
                    watchdog.stop()
                    if watchdog.cleanup_error is not None:
                        raise watchdog.cleanup_error
                    tree_stopped = True
                elif watchdog is not None:
                    watchdog.stop_tree()
                    watchdog.stop()
                    tree_stopped = True
                else:
                    _terminate_owned(proc, identity or _pid_identity(proc.pid), windows_job=windows_job)
                    tree_stopped = True
            except BaseException as cleanup_exc:
                ended = time.time()
                cleanup_updates = ({"headroom_failure": headroom_failure}
                                   if headroom_failure is not None else {})
                ledger.update_job(token, ended=ended, actual_wall_elapsed_seconds=ended-started,
                                  status="active-unknown", failure="cleanup-unverified", **cleanup_updates)
                raise RuntimeErrorSafe(f"owned process tree cleanup unverified ({type(cleanup_exc).__name__}: {cleanup_exc}); allocation remains active") from exc
        if windows_job is not None and not tree_stopped:
            windows_job.close()
        ended = time.time()
        failure_updates = {}
        if headroom_failure is not None:
            failure_updates["headroom_failure"] = headroom_failure
        ledger.update_job(token, ended=ended, actual_wall_elapsed_seconds=ended-started,
                          exit_code=proc.poll() if proc else None, status="failed", failure=type(exc).__name__,
                          **failure_updates)
        raise
    finally:
        if watchdog is not None:
            watchdog.stop()
        kwargs["stdout"].close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="explicit initialized ledger root")
    parser.add_argument("--reservation", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--first-progress-seconds", type=float, default=300)
    parser.add_argument("--heartbeat-seconds", type=float, default=120)
    parser.add_argument("--memory-guard", action="store_true",
                        help="require exact-UUID physical-memory headroom before and during launch")
    parser.add_argument("--memory-poll-seconds", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="child command after --")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("provide child command after --")
    return launch(AllocationLedger(args.root), args.reservation, command, cwd=args.cwd,
                  log_path=args.log, progress_path=args.progress,
                  first_progress_seconds=args.first_progress_seconds, heartbeat_seconds=args.heartbeat_seconds,
                  memory_guard=args.memory_guard, memory_poll_seconds=args.memory_poll_seconds)

if __name__ == "__main__":
    raise SystemExit(main())
