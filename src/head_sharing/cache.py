"""Contiguous, provenance-bound FP16 feature cache for the common Q predictor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

FORMAT = 2
BATCH_SIZE = 32
QUANTIZATION = "fp32_fp16_fp32_v1"
CHANNELS_STRIDES = ((64, 8), (128, 16), (256, 32))
VIEWS = ("original", "pixel_hflip")


def _sha(path: Path, *, progress_path: str | Path | None = None) -> str:
    digest = hashlib.sha256(); count = 0; last_report = time.monotonic()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block); count += len(block)
            if progress_path is not None and (count % (64 * 1024 * 1024) < len(block) or time.monotonic()-last_report >= 20):
                _progress(progress_path, "hashed_bytes", count, path.stat().st_size); last_report = time.monotonic()
    if progress_path is not None: _progress(progress_path, "hashed_bytes", count, count)
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".cache-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid cache metadata: {path.name}") from exc
    if not isinstance(value, dict): raise ValueError("cache metadata must be an object")
    return value


class CacheReader:
    """Validated immutable cache handle; hashes once at open, then reuses read-only maps."""
    def __init__(self, root: str | Path, *, expected_producer: str | None = None,
                 expected_identity: dict | None = None, progress_path: str | Path | None = None):
        self.root = Path(root)
        metadata_path = self.root / "metadata.json"
        if self.root.is_symlink() or metadata_path.is_symlink() or not metadata_path.is_file(): raise ValueError("cache metadata path is missing or linked")
        self.metadata = _read_json(metadata_path)
        m = self.metadata
        if m.get("format") != FORMAT or m.get("complete") is not True: raise ValueError("cache is incomplete or unsupported")
        ident_keys = ("trunk_sha256", "manifest_sha256", "imgsz", "quantization", "view_policy", "producer", "cohort", "cohort_ids", "block_order", "fixed_layout", "batch_size")
        identity = {k: m.get(k) for k in ident_keys}
        if m.get("identity_sha256") != _json_hash(identity): raise ValueError("cache identity hash mismatch")
        content = dict(m); recorded_content = content.pop("metadata_sha256", None)
        if recorded_content != _json_hash(content): raise ValueError("cache metadata integrity mismatch")
        if expected_producer is not None and m.get("producer") != expected_producer: raise ValueError("cache producer identity mismatch")
        if expected_identity is not None and any(m.get(k) != v for k, v in expected_identity.items()): raise ValueError("cache identity mismatch")
        if m.get("quantization") != QUANTIZATION or m.get("view_policy") != (list(VIEWS) if m.get("cohort") == "few" else ["original"]): raise ValueError("cache Q/view policy mismatch")
        if m.get("cohort") not in ("few", "dev", "probe") or not isinstance(m.get("imgsz"), int) or m["imgsz"] <= 0 or m["imgsz"] % 32: raise ValueError("cache geometry/cohort layout mismatch")
        if m.get("fixed_layout", {}).get("ids") != m.get("block_order", {}).get(m.get("cohort")): raise ValueError("fixed block layout mismatch")
        self.maps = []
        known = {f"features-{i}.f16" for i in range(3)}
        try:
            if len(m.get("feature_files", [])) != 3 or len(m.get("images", [])) != m.get("count"): raise ValueError("cache layout is incomplete")
            for i, item in enumerate(m["feature_files"]):
                name = item.get("file")
                if name not in known or Path(name).name != name: raise ValueError("invalid cache shard filename")
                path = self.root / name
                if path.is_symlink() or not path.is_file() or _sha(path, progress_path=progress_path) != item.get("sha256"): raise ValueError("cache feature shard hash mismatch")
                expected_views = 2 if m.get("cohort") == "few" else 1
                channels, stride = CHANNELS_STRIDES[i]
                expected_shape = [expected_views, m["count"], channels, m["imgsz"]//stride, m["imgsz"]//stride]
                if item.get("dtype") != "float16" or item.get("shape") != expected_shape: raise ValueError("cache shard layout mismatch")
                nbytes = int(np.prod(item["shape"])) * 2
                if path.stat().st_size != nbytes: raise ValueError("cache feature shard size mismatch")
                self.maps.append(np.memmap(path, mode="r", dtype="float16", shape=tuple(item["shape"])))
            if sorted(known) != sorted(item["file"] for item in m["feature_files"]): raise ValueError("cache shards incomplete")
        except BaseException:
            self.close()
            raise
        self.read_bytes = 0
        self.read_seconds = 0.0

    def close(self):
        for array in getattr(self, "maps", []):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None: mmap.close()
        self.maps = []

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def load_cache(root: str | Path, *, expected_producer: str | None = None, expected_identity: dict | None = None,
               progress_path: str | Path | None = None):
    """Open once for all epochs; hashing reports real progress to the owned watchdog."""
    return CacheReader(root, expected_producer=expected_producer, expected_identity=expected_identity,
                       progress_path=progress_path)


def read_blocks(reader: CacheReader, cohort: str, ids: list[int], *, flipped: bool = False, batch_size: int = BATCH_SIZE):
    """Read contiguous view spans in physical order; preserve partial tails."""
    import torch
    if batch_size < 1: raise ValueError("batch_size must be positive")
    if cohort not in ("few", "dev", "probe"): raise ValueError("unknown cohort")
    m = reader.metadata
    if cohort != m["cohort"]: raise ValueError("reader is bound to a different cohort")
    if flipped and cohort != "few": raise ValueError("pixel-flip view is not available for this cohort")
    if len(ids) != len(set(ids)): raise ValueError("requested IDs must be unique")
    cohort_ids = set(m["cohort_ids"][cohort])
    if not set(ids) <= cohort_ids: raise ValueError(f"IDs are not all in {cohort} cohort")
    index = {row["image_id"]: i for i, row in enumerate(m["images"])}
    requested = set(ids)
    ordered = [i for i in m["block_order"][cohort] if i in requested]
    if len(ordered) != len(ids) or ordered != ids: raise ValueError("IDs must be in common physical block order")
    selected = [index[i] for i in ordered]
    for start in range(0, len(selected), batch_size):
        positions = selected[start:start + batch_size]
        if positions != list(range(positions[0], positions[0] + len(positions))): raise ValueError("requested IDs must follow physical block order")
        view = int(flipped)
        started = time.monotonic()
        tensors = [torch.from_numpy(np.array(a[view, positions[0]:positions[-1]+1], copy=True)).float() for a in reader.maps]
        reader.read_seconds += time.monotonic() - started
        reader.read_bytes += sum(t.numel() * 2 for t in tensors)
        yield tensors, [m["images"][p] for p in positions]


def _metadata_targets(rows: list[dict], cat_to_class: dict[int, int], geometry) -> list[list[float]]:
    from .geometry import targets_xywhn
    boxes, classes = [], []
    for row in rows:
        x, y, w, h = row["bbox"]
        boxes.append([x, y, x + w, y + h]); classes.append(cat_to_class[row["category_id"]])
    return targets_xywhn(np.asarray(boxes).reshape(-1, 4), np.asarray(classes, dtype=np.int64), geometry).tolist()


def _categories(annotations: dict) -> list[dict]:
    rows = annotations.get("categories")
    if not isinstance(rows, list) or len(rows) != 80: raise ValueError("annotations require the 80 COCO categories")
    ordered = sorted(rows, key=lambda row: row["id"])
    if any(set(row) < {"id", "name"} for row in ordered): raise ValueError("invalid COCO category")
    return [{"category_id": row["id"], "class_id": i, "name": row["name"]} for i, row in enumerate(ordered)]


def build_cache(*, image_root: str | Path, annotations_path: str | Path, manifest_path: str | Path,
                receipt_path: str | Path, trunk_artifact: str | Path | dict, output: str | Path,
                model, producer: str, cohort: str = "few", batch_size: int = BATCH_SIZE,
                progress_path: str | Path | None = None, synthetic: bool = False):
    """Build a cohort-specific cache only after strict trunk and source validation."""
    import cv2
    import torch
    from .checkpoint import load_trunk_artifact, validate_trunk_receipt
    from .geometry import letterbox
    from .data import fixed_blocks
    if cohort not in ("few", "dev", "probe"): raise ValueError("cohort must be few, dev, or probe")
    if batch_size < 1: raise ValueError("batch_size must be positive")
    root = Path(output)
    if root.exists() and (root.is_symlink() or not root.is_dir() or any(root.iterdir())): raise ValueError("cache output must be a new or empty directory")
    manifest = _read_json(Path(manifest_path)); receipt = _read_json(Path(receipt_path))
    validate_trunk_receipt(receipt)
    # Strict artifact verification precedes output directory creation or shard mutation.
    load_trunk_artifact(model, trunk_artifact, receipt=receipt, synthetic=synthetic)
    annotation_path = Path(annotations_path)
    if _sha(annotation_path, progress_path=progress_path) != manifest.get("source_annotation_sha256"): raise ValueError("source annotation SHA256 mismatch")
    annotations = _read_json(annotation_path)
    required = ("ids", "probe_ids", "counts", "manifest_sha256", "source_annotation_sha256", "images", "categories")
    if any(k not in manifest for k in required) or set(manifest["ids"]) != {"few", "dev"}: raise ValueError("invalid compact manifest")
    cohort_ids = {"few": list(manifest["ids"]["few"]), "dev": list(manifest["ids"]["dev"]), "probe": list(manifest["probe_ids"])}
    if manifest["counts"] != {"few": len(cohort_ids["few"]), "dev": len(cohort_ids["dev"])}: raise ValueError("manifest cohort counts mismatch")
    ids = cohort_ids[cohort]
    if not ids or len(ids) != len(set(ids)): raise ValueError("selected cohort must contain unique IDs")
    if cohort == "probe" and not set(ids) <= set(cohort_ids["few"]): raise ValueError("probe IDs must be few members")
    if cohort_ids["few"] and cohort_ids["dev"] and set(cohort_ids["few"]) & set(cohort_ids["dev"]): raise ValueError("few/dev cohorts overlap")
    identities = {row["image_id"]: row for row in manifest["images"]}
    image_rows = {row["id"]: row for row in annotations.get("images", [])}
    if len(image_rows) != len(annotations.get("images", [])): raise ValueError("duplicate COCO image metadata")
    if not set(ids) <= image_rows.keys() or not set(ids) <= identities.keys(): raise ValueError("image metadata/annotations do not cover cohort")
    categories = _categories(annotations)
    if categories != manifest["categories"]: raise ValueError("category mapping mismatch")
    cat_to_class = {row["category_id"]: row["class_id"] for row in categories}
    indexed = {image_id: [] for image_id in ids}
    for ann in annotations.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id in indexed and ann.get("iscrowd", 0) != 1:
            if ann.get("category_id") not in cat_to_class: raise ValueError("annotation category is unknown")
            indexed[image_id].append(ann)
    for image_id in ids:
        source, manifest_image = image_rows[image_id], identities[image_id]
        if (source.get("file_name") != manifest_image.get("file_name") or
            [source.get("width"), source.get("height")] != [manifest_image.get("width", source.get("width")), manifest_image.get("height", source.get("height"))]):
            raise ValueError("COCO image identity/dimensions mismatch")
    layout = fixed_blocks(ids, batch_size)
    ordered = layout["ids"]
    started = time.monotonic()
    if not root.exists(): root.mkdir(parents=True)
    imgsz = int(model.imgsz)
    nviews = 2 if cohort == "few" else 1
    shapes = [(nviews, len(ids), c, imgsz // stride, imgsz // stride) for c, stride in CHANNELS_STRIDES]
    paths = [root / f"features-{i}.f16" for i in range(3)]
    arrays = [np.memmap(p, mode="w+", dtype=np.float16, shape=shape) for p, shape in zip(paths, shapes, strict=True)]
    meta_by_id = {}
    image_root = Path(image_root).resolve(strict=True)
    try:
        for start in range(0, len(ordered), batch_size):
            batch_ids = ordered[start:start + batch_size]
            for view, flipped in enumerate((False, True) if cohort == "few" else (False,)):
                pixels, batch_meta = [], []
                for image_id in batch_ids:
                    row, ident = image_rows[image_id], identities[image_id]
                    rel = Path(ident["file_name"]); path = (image_root / rel).resolve(strict=True)
                    if not path.is_relative_to(image_root) or path.is_symlink() or _sha(path, progress_path=progress_path) != ident["sha256"]: raise ValueError("image root escape or digest mismatch")
                    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if bgr is None: raise ValueError("cannot decode image")
                    if [row.get("width"), row.get("height")] != [bgr.shape[1], bgr.shape[0]]: raise ValueError("COCO image dimensions mismatch")
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); geo = letterbox(rgb, imgsz, flipped=flipped)
                    pixels.append(np.ascontiguousarray(geo.image.transpose(2, 0, 1))); batch_meta.append((image_id, geo, rgb))
                x = torch.from_numpy(np.stack(pixels)).to(device=next(model.parameters()).device, dtype=torch.float32) / 255.
                with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
                    features = model.extract_features(x)
                for dst, feature in zip(arrays, features, strict=True):
                    dst[view, start:start+len(batch_ids)] = feature.detach().cpu().numpy().astype(np.float16)
                if view == 0:
                    # Keep clean targets paired with clean features; the second extraction
                    # pass must not overwrite them with the horizontally flipped boxes.
                    for image_id, geo, rgb in batch_meta:
                        flipped_geo = letterbox(rgb, imgsz, flipped=True)
                        ident = identities[image_id]
                        meta_by_id[image_id] = {"image_id": image_id, "file_name": ident["file_name"], "sha256": ident["sha256"],
                            "original_shape": list(geo.original_shape), "ratio": list(geo.ratio), "pad": list(geo.pad),
                            "targets_xywhn": _metadata_targets(indexed[image_id], cat_to_class, geo),
                            "flipped": {"original_shape": list(flipped_geo.original_shape), "ratio": list(flipped_geo.ratio), "pad": list(flipped_geo.pad), "targets_xywhn": _metadata_targets(indexed[image_id], cat_to_class, flipped_geo)}}
                _progress(progress_path, "extracted_blocks", min(start+len(batch_ids), len(ordered)), len(ordered))
        for array in arrays: array.flush()
        feature_files = [{"file": p.name, "shape": list(a.shape), "dtype": "float16", "sha256": _sha(p, progress_path=progress_path)} for p, a in zip(paths, arrays, strict=True)]
        _progress(progress_path, "hashed_complete", len(ordered), len(ordered))
    finally:
        for array in arrays:
            if getattr(array, "_mmap", None): array._mmap.close()
    ids_by_cohort = {"few": cohort_ids["few"], "dev": cohort_ids["dev"], "probe": cohort_ids["probe"]}
    block_orders = {name: fixed_blocks(values, batch_size)["ids"] for name, values in ids_by_cohort.items()}
    identity = {"trunk_sha256": receipt["trunk_artifact_sha256"], "manifest_sha256": manifest["manifest_sha256"], "imgsz": imgsz,
                "quantization": QUANTIZATION, "view_policy": list(VIEWS if cohort == "few" else ("original",)), "producer": producer,
                "cohort": cohort, "cohort_ids": ids_by_cohort, "block_order": {cohort: block_orders[cohort]}, "fixed_layout": layout, "batch_size": batch_size}
    metadata = {"format": FORMAT, "complete": True, **identity, "identity_sha256": _json_hash(identity),
        "tail_policy": "keep_partial", "count": len(ordered), "ids": manifest["ids"], "probe_ids": manifest["probe_ids"],
        "counts": manifest["counts"], "source_annotation_sha256": manifest["source_annotation_sha256"], "categories": categories,
        "images": [meta_by_id[i] for i in ordered], "feature_files": feature_files, "synthetic": bool(synthetic),
        "torch_version": torch.__version__, "backend": {"device": producer, "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "cudnn_deterministic": torch.backends.cudnn.deterministic},
        "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE")}, "write_seconds": time.monotonic()-started,
        "bytes_written": sum(p.stat().st_size for p in paths)}
    metadata["metadata_sha256"] = _json_hash(metadata)
    _atomic_json(root / "metadata.json", metadata)
    _progress(progress_path, "cache_complete", len(ordered), len(ordered))
    return metadata


def _progress(path: str | Path | None, phase: str, done: int, total: int):
    if path is None: return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"phase": phase, "done": done, "total": total, "time": time.time()}) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def compare_cache(model, images, features: list, *, atol: float, rtol: float, geometry_max_abs: float = 0.0):
    """Compare matched batches across Q features, raw native head, decoded output and geometry."""
    import torch
    if atol < 0 or rtol < 0: raise ValueError("comparison tolerances must be nonnegative")
    with torch.no_grad():
        expected = [x.to(device=images.device, dtype=torch.float16).float() for x in model.extract_features(images)]
        stored = [x.to(device=images.device, dtype=torch.float32) for x in features]
        for a, b in zip(expected, stored, strict=True): torch.testing.assert_close(a, b, atol=0, rtol=0)
        online = model.forward_features(expected)
        cached = model.forward_features(stored)
    errors = {}
    for branch in ("one2many", "one2one"):
        for key in ("boxes", "scores"):
            a, b = online[1][branch][key], cached[1][branch][key]
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
            errors[f"{branch}_{key}_max_abs"] = float((a-b).abs().max())
    torch.testing.assert_close(online[0], cached[0], atol=atol, rtol=rtol)
    return {"decoded_max_abs": float((online[0]-cached[0]).abs().max()), "feature_max_abs": 0.0,
            "geometry_max_abs": float(geometry_max_abs), "atol": atol, "rtol": rtol, **errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="stream an explicit cohort into an immutable feature cache")
    for key in ("image-root", "annotations", "receipt", "trunk-artifact", "manifest", "output"): build.add_argument(f"--{key}", required=True)
    build.add_argument("--device", required=True, help="cpu or reserved GPU UUID")
    build.add_argument("--cohort", choices=("few", "dev", "probe"), default="few")
    build.add_argument("--imgsz", type=int, choices=(512, 640), default=640)
    build.add_argument("--owned-root"); build.add_argument("--ledger"); build.add_argument("--reservation")
    read = sub.add_parser("read", help="validate cache and print metadata")
    read.add_argument("--cache", required=True); read.add_argument("--expected-producer")
    compare = sub.add_parser("compare", help="compare matched online-Q/cache batch groups")
    compare.add_argument("--cache", required=True); compare.add_argument("--expected-producer", required=True)
    for key in ("image-root", "receipt", "trunk-artifact"): compare.add_argument(f"--{key}", required=True)
    compare.add_argument("--probe-id", type=int, required=True); compare.add_argument("--imgsz", type=int, choices=(512,640), default=640)
    compare.add_argument("--device", required=True, help="cpu or reserved producer UUID")
    compare.add_argument("--owned-root"); compare.add_argument("--ledger"); compare.add_argument("--reservation")
    compare.add_argument("--atol", type=float, required=True); compare.add_argument("--rtol", type=float, required=True)
    args = parser.parse_args(argv)
    if args.command == "read":
        with load_cache(args.cache, expected_producer=args.expected_producer) as reader: print(json.dumps(reader.metadata, sort_keys=True))
        return 0
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; producer = "cpu"; progress_path = None
    else:
        if not all((args.owned_root, args.ledger, args.reservation)): parser.error("GPU mode requires --owned-root, --ledger, and --reservation")
        from .runtime import AllocationLedger, _has_symlink_component, canonical_gpu_uuid
        from .supervisor import child_entry
        owned_root = Path(args.owned_root)
        if (_has_symlink_component(owned_root) or not owned_root.is_dir() or owned_root.absolute() != owned_root.resolve(strict=True) or not (owned_root/"STUDY_OWNER.txt").is_file()): raise RuntimeError("GPU mode requires canonical owned study root")
        ledger_root = Path(args.ledger)
        if (_has_symlink_component(ledger_root) or not ledger_root.is_dir() or not ledger_root.resolve(strict=True).is_relative_to(owned_root.resolve(strict=True))): raise RuntimeError("GPU ledger must be within owned root")
        ledger = AllocationLedger(ledger_root); child_entry(ledger, args.reservation, args.device)
        job = ledger.get_job(args.reservation); progress_path = Path(job["progress_file"])
        if progress_path.parent.resolve(strict=True) != Path(job["cwd"]).resolve(strict=True) or progress_path.is_symlink(): raise RuntimeError("progress file is not supervisor-owned")
        _progress(progress_path, "ownership_verified", 0, 1)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != args.device: raise RuntimeError("GPU visibility differs from reservation")
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("reserved CUDA device unavailable")
        actual_uuid = canonical_gpu_uuid(str(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "uuid", "")))
        if actual_uuid != args.device: raise RuntimeError("actual torch CUDA UUID differs from reservation")
        ledger.validate_device(args.reservation, actual_uuid); producer = actual_uuid
        _progress(progress_path, "native_import_complete", 1, 1)
    from .model import HeadStudyModel
    import torch
    if args.command == "compare" and producer != args.expected_producer: raise ValueError("comparison must run on cache producer")
    model = HeadStudyModel(imgsz=args.imgsz).to("cpu" if args.device == "cpu" else "cuda").eval()
    receipt = _read_json(Path(args.receipt))
    if args.command == "build":
        output = Path(args.output).absolute()
        if args.device != "cpu":
            if _has_symlink_component(output) or not output.resolve(strict=False).is_relative_to(owned_root.resolve(strict=True)): raise RuntimeError("cache output must be link-free under owned root")
        metadata = build_cache(image_root=args.image_root, annotations_path=args.annotations, manifest_path=args.manifest, receipt_path=args.receipt,
            trunk_artifact=args.trunk_artifact, output=output, model=model, producer=producer, cohort=args.cohort, progress_path=progress_path)
        print(json.dumps({"producer": producer, "cohort": args.cohort, "count": metadata["count"], "bytes_written": metadata["bytes_written"], "write_seconds": metadata["write_seconds"]})); return 0
    from .checkpoint import load_trunk_artifact
    load_trunk_artifact(model, args.trunk_artifact, receipt=receipt)
    import cv2
    from .geometry import letterbox
    with load_cache(args.cache, expected_producer=producer, progress_path=progress_path) as reader:
        m = reader.metadata
        if args.imgsz != m["imgsz"]: raise ValueError("comparison resolution differs from cache")
        if args.probe_id not in m["cohort_ids"]["probe"]: raise ValueError("comparison ID is not a declared probe")
        ordered = m["block_order"][m["cohort"]]; probe_pos = ordered.index(args.probe_id)
        start = (probe_pos // m["batch_size"]) * m["batch_size"]; stop = min(start+m["batch_size"],len(ordered)); group_ids=ordered[start:stop]
        rows = {x["image_id"]:x for x in m["images"]}; image_root = Path(args.image_root).resolve(strict=True)
        for flipped in (False, True) if m["cohort"] == "few" else (False,):
            pixels=[]; geometry_error=0.0
            for image_id in group_ids:
                row=rows[image_id]; path=(image_root/row["file_name"]).resolve(strict=True)
                if not path.is_relative_to(image_root) or _sha(path)!=row["sha256"]: raise ValueError("probe group image identity mismatch")
                bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
                if bgr is None: raise ValueError("cannot decode probe group image")
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); geo=letterbox(rgb,args.imgsz,flipped=flipped)
                stored=row["flipped"] if flipped else row
                for actual, recorded in zip((geo.original_shape,geo.ratio,geo.pad),(stored["original_shape"],stored["ratio"],stored["pad"]),strict=True):
                    geometry_error=max(geometry_error,max(abs(float(a)-float(b)) for a,b in zip(actual,recorded,strict=True)))
                pixels.append(geo.image.transpose(2,0,1).copy())
            images=torch.from_numpy(np.stack(pixels)).float().to(next(model.parameters()).device)/255
            view=int(flipped)
            features=[torch.from_numpy(np.array(a[view,start:stop],copy=True)).float().to(images.device) for a in reader.maps]
            result=compare_cache(model,images,features,atol=args.atol,rtol=args.rtol,geometry_max_abs=geometry_error)
            print(json.dumps({"producer":producer,"cohort":m["cohort"],"build_batch_size":m["batch_size"],"comparison_batch_ids":group_ids,"mode":"matched_build_group","view":"pixel_hflip" if flipped else "original",**result},sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
