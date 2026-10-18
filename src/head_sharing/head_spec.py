"""Canonical, dependency-free identity for optional AB2 head designs."""
from __future__ import annotations

import hashlib
import json

_FIELDS = {"version", "variant", "class_inner"}
_VARIANTS = {"baseline", "learned_alpha", "step_norm", "scale_gain", "dw5"}
_WIDTHS = {96, 48}


def canonicalize_head_spec(spec):
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) != _FIELDS:
        raise ValueError("head_spec must contain exactly version, variant, and class_inner")
    version, variant, width = spec["version"], spec["variant"], spec["class_inner"]
    if type(version) is not int or version != 1:
        raise ValueError("unsupported head_spec version")
    if type(variant) is not str or variant not in _VARIANTS:
        raise ValueError("unsupported head_spec variant")
    if type(width) is not int or width not in _WIDTHS:
        raise ValueError("unsupported head_spec class_inner")
    if variant == "baseline" and width == 96:
        return None
    return {"version": 1, "variant": variant, "class_inner": width}


def head_spec_digest(spec) -> str:
    canonical = canonicalize_head_spec(spec)
    if canonical is None:
        canonical = {"version": 1, "variant": "baseline", "class_inner": 96}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
