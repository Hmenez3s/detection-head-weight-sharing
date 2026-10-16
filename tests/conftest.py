"""Fail closed unless tests run inside check_cpu's owned sandbox."""
import os
from pathlib import Path
import socket

root_text = os.environ.get("HEAD_SHARING_TEST_ROOT")
if not root_text:
    raise RuntimeError("tests must be launched through scripts/check_cpu.py")
root = Path(root_text).resolve(strict=True)
for key in ("YOLO_CONFIG_DIR", "TORCH_HOME", "MPLCONFIGDIR", "XDG_CACHE_HOME", "TMPDIR"):
    value = Path(os.environ[key]).resolve(strict=True)
    if not value.is_relative_to(root):
        raise RuntimeError(f"{key} is outside owned test root")
if not (Path(os.environ["YOLO_CONFIG_DIR"]) / "Ultralytics").is_dir():
    raise RuntimeError("Ultralytics config child was not precreated")
if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("YOLO_OFFLINE") != "true":
    raise RuntimeError("CPU/offline test environment is not active")

def _deny_network(*args, **kwargs):
    raise RuntimeError("network access denied by tests/conftest.py")

socket.create_connection = _deny_network
socket.getaddrinfo = _deny_network
socket.socket.connect = _deny_network

from ultralytics.utils import USER_CONFIG_DIR

expected_config = (Path(os.environ["YOLO_CONFIG_DIR"]) / "Ultralytics").resolve(strict=True)
if USER_CONFIG_DIR.resolve(strict=True) != expected_config:
    raise RuntimeError(f"Ultralytics config escaped owned root: {USER_CONFIG_DIR}")

import torch

torch.set_num_threads(1)
