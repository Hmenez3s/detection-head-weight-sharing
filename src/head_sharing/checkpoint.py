"""Portable tensor-only checkpoints for head-study detectors."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import torch

from .model import HeadStudyModel

CONFIG_KEYS = {"format", "architecture", "arm", "seed", "imgsz", "quantization", "trunk_receipt"}
RECEIPT_KEYS = {
    "trunk_artifact_sha256", "trunk_tensors", "trunk_bytes", "head_index", "input_channels", "source_variant",
    "source_seed", "source_epoch_zero_based", "source_training_epochs", "pre_exposure",
}


def _read_verified_trunk_artifact(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    try:
        content = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError("cannot read sanitized trunk artifact") from exc
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("trunk artifact SHA256 does not match receipt")
    try:
        artifact = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as exc:
        raise ValueError("invalid sanitized trunk artifact") from exc
    if not isinstance(artifact, dict):
        raise ValueError("trunk artifact must be a tensor container")
    return artifact


def load_trunk_artifact(model: HeadStudyModel, artifact: dict[str, Any] | str | Path, *,
                        receipt: dict[str, Any], synthetic: bool = False) -> None:
    """Load a hash-verified production path; in-memory containers require synthetic=True."""
    validate_trunk_receipt(receipt)
    if isinstance(artifact, (str, Path)):
        artifact = _read_verified_trunk_artifact(artifact, receipt["trunk_artifact_sha256"])
    elif not synthetic:
        raise ValueError("in-memory trunk containers are synthetic test fixtures only")
    expected_fields = {"state_dict", "schema_version", "architecture", "upstream", "excluded_head_index"}
    if not isinstance(artifact, dict) or set(artifact) != expected_fields:
        raise ValueError("trunk artifact has unknown or missing fields")
    metadata = (artifact["schema_version"], artifact["architecture"], artifact["upstream"], artifact["excluded_head_index"])
    if metadata != (1, "yolo26n", "8.4.177", 23):
        raise ValueError("trunk artifact metadata mismatch")
    supplied = artifact["state_dict"]
    if not isinstance(supplied, dict) or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in supplied.items()):
        raise ValueError("trunk artifact must contain named tensors only")
    expected = model.state_dict()
    target_names = {"detector." + name for name in model.detector.state_dict()
                    if name.startswith("model.") and name.split(".")[1].isdigit() and int(name.split(".")[1]) < 23}
    mapped = {"detector." + name: tensor for name, tensor in supplied.items()}
    if set(mapped) != target_names:
        raise ValueError("trunk artifact tensor topology mismatch or excluded head content")
    for name, value in mapped.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f"trunk artifact tensor shape/dtype mismatch: {name}")
        if value.dtype not in (torch.float32, torch.int64):
            raise ValueError(f"trunk artifact tensor dtype is not allowlisted: {name}")
    model.load_state_dict(mapped, strict=False)
    for parameter in model.detector.model[:-1].parameters():
        parameter.requires_grad_(False)
    model.detector.model[:-1].eval()


def validate_trunk_receipt(receipt: dict[str, Any]) -> None:
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_KEYS:
        raise ValueError("trunk receipt has unknown or missing fields")
    sha = receipt["trunk_artifact_sha256"]
    if sha != "f01ec41445fac9ef7078b48ef1b73473e6314bcb1a77e77c20666d6ecfcd9016":
        raise ValueError("trunk artifact SHA256 does not match the allowlisted source export")
    if any(type(receipt[key]) is not int for key in ("trunk_tensors", "trunk_bytes", "head_index")):
        raise ValueError("trunk receipt counts must be integers")
    if receipt["trunk_tensors"] != 468 or receipt["trunk_bytes"] != 9_109_168 or receipt["head_index"] != 23:
        raise ValueError("trunk receipt does not identify the allowlisted YOLO26n trunk")
    if receipt["input_channels"] != [64, 128, 256]:
        raise ValueError("unexpected trunk input channels")
    if receipt["source_variant"] != "U" or receipt["source_seed"] != 11:
        raise ValueError("unexpected source provenance")
    if receipt["source_epoch_zero_based"] != 79 or receipt["source_training_epochs"] != 80:
        raise ValueError("unexpected source epoch provenance")
    if not isinstance(receipt["pre_exposure"], str) or not receipt["pre_exposure"]:
        raise ValueError("pre-exposure disclosure is required")


def _alias_groups(model: HeadStudyModel) -> list[list[str]]:
    by_storage: dict[tuple[int, int], list[str]] = {}
    for name, tensor in model.state_dict().items():
        if tensor.numel():
            by_storage.setdefault((tensor.untyped_storage().data_ptr(), tensor.storage_offset()), []).append(name)
    return [names for names in by_storage.values() if len(names) > 1]


def _cpu_head_state(model: HeadStudyModel) -> dict[str, torch.Tensor]:
    """Copy head state once per exact physical tensor, preserving serialized aliases."""
    copies: dict[tuple[Any, ...], torch.Tensor] = {}
    state: dict[str, torch.Tensor] = {}
    for name, value in model.detect.state_dict().items():
        key = (value.device.type, value.device.index, value.untyped_storage().data_ptr(), value.storage_offset(),
               tuple(value.shape), tuple(value.stride()), value.dtype)
        if key not in copies:
            copies[key] = value.detach().cpu().contiguous()
        state[name] = copies[key]
    return state


def save_checkpoint(model: HeadStudyModel, weights_path: str | Path, config_path: str | Path,
                    trunk_receipt: dict[str, Any]) -> None:
    validate_trunk_receipt(trunk_receipt)
    weights_path, config_path = Path(weights_path), Path(config_path)
    state = _cpu_head_state(model)
    torch.save(state, weights_path)
    config = {"format": 1, "architecture": model.architecture, "arm": model.arm, "seed": model.seed,
              "imgsz": model.imgsz, "quantization": model.quantization, "trunk_receipt": trunk_receipt}
    config_path.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _read_config(path: str | Path) -> dict[str, Any]:
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid checkpoint config JSON") from exc
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError("checkpoint config has unknown or missing fields")
    if type(config["format"]) is not int or config["format"] != 1 or config["architecture"] != "yolo26n":
        raise ValueError("unsupported checkpoint format or architecture")
    if config["arm"] not in ("native", "independent_k2", "shared_k2"):
        raise ValueError("unknown checkpoint arm")
    if type(config["seed"]) is not int or type(config["imgsz"]) is not int:
        raise ValueError("checkpoint seed and image size must be integers")
    if config["quantization"] != HeadStudyModel.quantization:
        raise ValueError("checkpoint quantization policy mismatch")
    validate_trunk_receipt(config["trunk_receipt"])
    return config


def load_checkpoint(weights_path: str | Path, config_path: str | Path, *, trunk_artifact: dict[str, Any] | str | Path,
                    synthetic_trunk: bool = False, map_location: str = "cpu") -> HeadStudyModel:
    config = _read_config(config_path)
    model = HeadStudyModel(config["arm"], config["seed"], config["imgsz"])
    try:
        state = torch.load(weights_path, map_location=map_location, weights_only=True)
    except (OSError, RuntimeError, EOFError) as exc:
        raise ValueError("invalid tensor checkpoint") from exc
    if not isinstance(state, dict) or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("checkpoint must contain only named tensors")
    expected = model.detect.state_dict()
    if set(state) != set(expected):
        raise ValueError("checkpoint tensor topology does not match constructed head")
    for key, value in state.items():
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError(f"checkpoint tensor shape/dtype mismatch: {key}")
    head_alias_groups = [[name.removeprefix("detector.model.23.") for name in group
                          if name.startswith("detector.model.23.")]
                         for group in _alias_groups(model)]
    for group in (group for group in head_alias_groups if len(group) > 1):
        reference = state[group[0]]
        if any(not torch.equal(reference, state[key]) for key in group[1:]):
            raise ValueError("conflicting duplicate tensors for aliased module state")
    load_trunk_artifact(model, trunk_artifact, receipt=config["trunk_receipt"], synthetic=synthetic_trunk)
    model.detect.load_state_dict(state, strict=True)
    model.eval()
    return model


def save_resume(model: HeadStudyModel, optimizer: torch.optim.Optimizer, path: str | Path, *, epoch: int) -> None:
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    torch.save({"model": _cpu_head_state(model), "optimizer": optimizer.state_dict(), "epoch": epoch}, path)


def load_resume(model: HeadStudyModel, optimizer: torch.optim.Optimizer, path: str | Path) -> int:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as exc:
        raise ValueError("invalid resume checkpoint") from exc
    if not isinstance(payload, dict) or set(payload) != {"model", "optimizer", "epoch"}:
        raise ValueError("invalid resume checkpoint structure")
    epoch = payload["epoch"]
    if type(epoch) is not int or epoch < 0:
        raise ValueError("invalid resume epoch")
    optimizer_state = payload["optimizer"]
    if not isinstance(optimizer_state, dict) or set(optimizer_state) != {"state", "param_groups"}:
        raise ValueError("invalid resume optimizer state")
    state = payload["model"]
    expected = model.detect.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("resume head topology mismatch")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError(f"invalid resume tensor: {key}")
    head_alias_groups = [[name.removeprefix("detector.model.23.") for name in group
                          if name.startswith("detector.model.23.")]
                         for group in _alias_groups(model)]
    for group in (group for group in head_alias_groups if len(group) > 1):
        if any(not torch.equal(state[group[0]], state[k]) for k in group[1:]):
            raise ValueError("conflicting duplicate tensors for aliased module state")
    optimizer.load_state_dict(optimizer_state)
    model.detect.load_state_dict(state, strict=True)
    return epoch
