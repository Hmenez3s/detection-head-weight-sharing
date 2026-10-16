"""Explicit, path-free study manifests and paired physical block ordering."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path, PureWindowsPath
from typing import Iterator

PROBE_SEED_XOR = 0xC1EA
PROBE_SIZE = 1000


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compact_manifest(source: dict, image_root: str | Path, probe_ids: list[int] | None = None,
                     probe_seed: int | None = None) -> dict:
    """Validate source-shaped metadata and emit only IDs, identities and digests.

    Unless explicit probe IDs/seed are supplied, choose the clean-train probe as
    Random(20261016 XOR 0xC1EA).sample(sorted(few), 1000), before outcomes.
    """
    ids = source.get("ids")
    if not isinstance(ids, dict) or not all(k in ids for k in ("few", "dev", "full")):
        raise ValueError("manifest requires ids.few/dev/full")
    few, dev, full = (list(ids[k]) for k in ("few", "dev", "full"))
    for name, values in (("few", few), ("dev", dev), ("full", full)):
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate IDs in {name}")
    if set(few) & set(dev):
        raise ValueError("few and dev IDs overlap")
    rows = source.get("images")
    if not isinstance(rows, list):
        raise ValueError("images must be a list")
    root = Path(image_root).resolve(strict=True)
    by_id = {}
    resolved_paths = {}
    for row in rows:
        if not all(k in row for k in ("file_name", "image_id", "sha256")):
            raise ValueError("image rows require file_name, image_id, sha256")
        if row["image_id"] in by_id:
            raise ValueError("duplicate image rows")
        name = row["file_name"]
        if not isinstance(name, str) or not name or "\x00" in name:
            raise ValueError("invalid image filename")
        relative = Path(name)
        windows = PureWindowsPath(name)
        if relative.is_absolute() or windows.is_absolute() or windows.drive or any(part == ".." for part in relative.parts + windows.parts):
            raise ValueError(f"image filename must remain relative to image root: {name}")
        path = (root / relative).resolve(strict=False)
        if not path.is_relative_to(root):
            raise ValueError(f"image path escapes image root: {name}")
        by_id[row["image_id"]] = row
        resolved_paths[row["image_id"]] = path
    needed = set(few) | set(dev)
    if not needed <= set(by_id):
        raise ValueError("missing image identity row")
    for image_id in needed:
        if not resolved_paths[image_id].is_file():
            raise ValueError(f"missing image: {by_id[image_id]['file_name']}")
    identities = []
    for image_id in sorted(needed):
        row = by_id[image_id]
        actual = hashlib.sha256(resolved_paths[image_id].read_bytes()).hexdigest()
        if actual != row["sha256"]:
            raise ValueError(f"image digest mismatch: {row['file_name']}")
        identities.append({"file_name": row["file_name"], "image_id": image_id, "sha256": actual})
    if probe_ids is not None and probe_seed is not None:
        raise ValueError("choose explicit probe IDs or probe seed, not both")
    if probe_ids is not None:
        probe = sorted(probe_ids)
        if len(probe) != len(set(probe)) or not set(probe) <= set(few) or set(probe) & set(dev):
            raise ValueError("probe IDs must be unique members of few and disjoint from dev")
    else:
        seed = 20261016 if probe_seed is None else probe_seed
        probe = sorted(random.Random(seed ^ PROBE_SEED_XOR).sample(sorted(few), min(PROBE_SIZE, len(few))))
    # The unused source full-data cohort is not part of this study's public inputs.
    return {"ids": {"few": sorted(few), "dev": sorted(dev)},
            "counts": {"few": len(few), "dev": len(dev)},
            "categories": source["categories"], "images": identities,
            "manifest_sha256": source["manifest_sha256"], "source_annotation_sha256": source["source_annotation_sha256"],
            "probe_ids": probe}


def fixed_blocks(ids: list[int], batch_size: int, seed: int = 20261016) -> dict:
    if batch_size < 1 or len(ids) != len(set(ids)):
        raise ValueError("batch_size must be positive and IDs unique")
    order = list(ids)
    random.Random(seed).shuffle(order)
    blocks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    return {"ids": order, "batch_size": batch_size, "tail_policy": "keep_partial",
            "blocks": blocks, "sha256": _digest({"ids": order, "batch_size": batch_size, "tail_policy": "keep_partial"})}


def epoch_batches(layout: dict, seed: int, epoch: int) -> Iterator[tuple[list[int], bool]]:
    """Shuffle fixed blocks and choose one shared original/flip view per batch."""
    rng = random.Random((seed << 32) ^ epoch)
    order = list(range(len(layout["blocks"])))
    rng.shuffle(order)
    for block_i in order:
        yield layout["blocks"][block_i], bool(rng.getrandbits(1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="validate explicit source-shaped manifest and image root")
    p.add_argument("--manifest", required=True); p.add_argument("--image-root", required=True); p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    source = json.loads(Path(args.manifest).read_text())
    result = compact_manifest(source, args.image_root)
    Path(args.output).write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
