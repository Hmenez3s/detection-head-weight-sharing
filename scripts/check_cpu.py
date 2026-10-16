#!/usr/bin/env python3
"""Run focused pytest checks in an isolated, disposable CPU environment."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time


def _reap_group(proc: subprocess.Popen[bytes], grace: float = 0.5) -> None:
    """Terminate only this runner's session/process group and reap its leader."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    deadline = time.monotonic() + grace
    while True:
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            break
        if time.monotonic() >= deadline:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            break
        time.sleep(0.02)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.seconds <= 0 or not args.pytest_args or args.pytest_args[0] != "--":
        parser.error("provide --seconds N -- <pytest paths>")
    paths = args.pytest_args[1:]
    if not paths:
        parser.error("at least one pytest path is required")

    started = time.monotonic()
    parent = Path(tempfile.mkdtemp(prefix=f"head-sharing-{os.getpid()}-"))
    root = parent.resolve(strict=True)
    fallback = Path("/tmp/Ultralytics")

    def inventory() -> tuple[tuple[str, int, int], ...] | None:
        if not fallback.exists():
            return None
        return tuple(sorted((str(p.relative_to(fallback)), hashlib.sha256(p.read_bytes()).hexdigest())
                            for p in fallback.rglob("*") if p.is_file()))

    before = inventory()
    proc = None
    try:
        env = os.environ.copy()
        owned = {
            "YOLO_CONFIG_DIR": root / "yolo",
            "TORCH_HOME": root / "torch",
            "MPLCONFIGDIR": root / "mpl",
            "XDG_CACHE_HOME": root / "cache",
            "TMPDIR": root / "tmp",
        }
        for key, value in owned.items():
            value.mkdir(parents=True, exist_ok=True)
            env[key] = str(value)
        (root / "yolo" / "Ultralytics").mkdir()
        env.update(CUDA_VISIBLE_DEVICES="", YOLO_OFFLINE="true", PYTHONPATH="src",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                   PYTHONDONTWRITEBYTECODE="1", HEAD_SHARING_TEST_ROOT=str(root))
        guard = r'''
import socket

def denied(*args, **kwargs):
    raise RuntimeError("network access denied by check_cpu")

socket.create_connection = denied
socket.getaddrinfo = denied
socket.socket.connect = denied
import pytest
raise SystemExit(pytest.main(["-p", "no:cacheprovider", *%r]))
''' % paths
        proc = subprocess.Popen([sys.executable, "-c", guard], env=env,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            print("pytest timed out; owned process group reaped", file=sys.stderr)
            return 124
        if inventory() != before:
            raise RuntimeError("frozen /tmp/Ultralytics inventory changed")
        return code
    finally:
        if proc is not None:
            _reap_group(proc)
        if inventory() != before:
            raise RuntimeError("frozen /tmp/Ultralytics changed during cleanup; evidence retained")
        shutil.rmtree(root)
        try:
            parent.rmdir()
        except OSError:
            pass
        print(f"walltime_seconds={time.monotonic() - started:.3f} workers=1 cpu_threads<=3")


if __name__ == "__main__":
    raise SystemExit(main())
