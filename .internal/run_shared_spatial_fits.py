"""Parent-invoked, inert launcher for the shared-spatial DEV study and seed-997 pilots.

This module has no import-time host, filesystem, ledger, process, or GPU effects.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

UUID = "GPU-08055392-87d6-8287-947d-0bf94c711445"
STUDY_ID = "shared-spatial-20261018T210955Z"
POLICY_STUDY_ID = "shared_spatial_v1"
VARIANTS = ("native", "spatial_untied", "spatial_sepbn", "spatial_gn",
            "spatial_sepbn_k5", "spatial_sepbn_sharedpred")
CANDIDATES = VARIANTS[2:]
SEEDS = (11, 22, 33, 44, 55)
SCREEN_SEEDS = (11, 33)
PILOT_SEED = 997


class _UnstartedCell(RuntimeError):
    """A queued cell suppressed before its owned job tree/reservation was created."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _full_sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _relative(value: object, label: str) -> PurePosixPath:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value):
        raise ValueError(f"{label} must be a normalized relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError(f"{label} must be a normalized relative path")
    return path


def _no_links(path: Path, *, include_missing: bool = False) -> bool:
    """Check every extant component, including Windows junctions, without resolving aliases."""
    absolute = path.absolute()
    for part in (absolute, *absolute.parents):
        if part.exists() or part.is_symlink():
            if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
                return False
        elif not include_missing:
            continue
    return True


def _inside(path: Path, root: Path, *, strict: bool = True) -> Path:
    if not _no_links(path, include_missing=not strict):
        raise ValueError(f"linked path rejected: {path}")
    resolved = path.resolve(strict=strict)
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escaped owned root: {path}")
    return resolved


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_bootstrap(root: Path, source: str, study_subroot: str) -> Path:
    """Pure filesystem authority gate; run before reading manifests or importing private code."""
    if not _full_sha(source) or study_subroot != STUDY_ID:
        raise ValueError("full immutable source SHA and exact shared-spatial namespace are required")
    if (not root.is_absolute() or not root.is_dir() or root.absolute() != root.resolve(strict=True) or
            not _no_links(root) or not (root / "STUDY_OWNER.txt").is_file() or
            (root / "STUDY_OWNER.txt").is_symlink()):
        raise ValueError("canonical marker-owned root required before source inspection")
    study = root / study_subroot
    owner = study / "SHARED_SPATIAL_OWNER.txt"
    if (not study.is_dir() or not _no_links(study) or not study.resolve(strict=True).is_relative_to(root) or
            not owner.is_file() or owner.is_symlink() or owner.read_text(encoding="utf-8").strip() != STUDY_ID):
        raise ValueError("canonical owned shared-spatial namespace required before source inspection")
    code = root / "code-snapshots" / source[:7]
    marker = code / "SOURCE_COMMIT"
    if (not code.is_dir() or not _no_links(code) or not _no_links(marker) or not marker.is_file() or
            marker.read_text(encoding="utf-8").strip() != source):
        raise ValueError("immutable source directory/SOURCE_COMMIT mismatch")
    return code


def _manifest(root: Path, source: str, controller_file: Path) -> tuple[Path, dict]:
    code = _validate_bootstrap(root, source, STUDY_ID)
    source_marker = code / "SOURCE_COMMIT"
    manifest_path = code / "shared-spatial-source-manifest.json"
    if not _no_links(source_marker) or not _no_links(manifest_path):
        raise ValueError("immutable source marker/manifest is linked")
    manifest = _read_json(manifest_path)
    expected_paths = {"controller": ".internal/run_shared_spatial_fits.py",
                      "policy": ".internal/shared_spatial.py",
                      "ram_startup_helper": ".internal/run_ab2_fits.py"}
    if manifest.get("source_git_commit") != source or manifest.get("files") is None:
        raise ValueError("immutable source manifest does not bind the requested commit")
    if set(manifest["files"]) != set(expected_paths):
        raise ValueError("immutable source manifest file set differs")
    for key, relative in expected_paths.items():
        path = code / relative
        if not path.is_file() or not _no_links(path):
            raise ValueError(f"immutable source file missing or linked: {relative}")
        actual = _sha(path.read_bytes())
        if manifest["files"][key] != {"path": relative, "sha256": actual}:
            raise ValueError(f"immutable source manifest hash mismatch: {relative}")
    if _sha(controller_file.read_bytes()) != manifest["files"]["controller"]["sha256"]:
        raise ValueError("invoked controller differs from immutable manifest")
    return code, manifest


def _load_helper(path: Path):
    """Load only the two already-existing resource helpers; never execute its main."""
    spec = importlib.util.spec_from_file_location("_shared_spatial_staged_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load immutable RAM/startup helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "available_ram", None)) or not callable(getattr(module, "claim_startup_file", None)):
        raise ValueError("immutable helper lacks required RAM/startup functions")
    return module


def _cell_identity(variant: object, seed: object) -> tuple[str, int]:
    if variant not in VARIANTS or type(seed) is not int or seed not in SEEDS:
        raise ValueError("cell is outside the frozen shared-spatial family")
    return variant, seed


def validate_scientific_plan(plan: object, policy) -> tuple[list[dict], dict | None]:
    """Validate screen/follow-up authority exclusively through the committed pure policy."""
    if not isinstance(plan, dict) or set(plan) not in ({"mode", "cells"}, {"mode", "cells", "decision"}):
        raise ValueError("scientific plan has unknown or missing fields")
    mode, cells = plan["mode"], plan["cells"]
    if mode not in ("screen", "followup") or not isinstance(cells, list) or not cells:
        raise ValueError("scientific plan mode/cells invalid")
    if mode == "screen":
        if "decision" in plan:
            raise ValueError("screen plan cannot carry a prior selection decision")
        expected = {(variant, seed) for variant in VARIANTS for seed in SCREEN_SEEDS}
        decision = None
    else:
        decision = plan.get("decision")
        if not isinstance(decision, dict) or decision.get("winner") not in CANDIDATES:
            raise ValueError("follow-up requires the sole closed positive DEV winner")
        expected = {(variant, seed) for variant in ("native", decision["winner"]) for seed in (22, 44, 55)}
        if not isinstance(decision.get("decision_sha256"), str):
            raise ValueError("follow-up decision seal missing")
        # Recompute the sealed decision so caller-supplied winner data is never authority.
        if decision.get("screen_closed") is not True:
            raise ValueError("follow-up decision screen is not closed")
        rebuilt = policy.decide_stages(decision.get("screen"))
        if (rebuilt.get("winner") != decision.get("winner") or
                rebuilt.get("decision_sha256") != decision.get("decision_sha256")):
            raise ValueError("follow-up decision is not the pure policy's closed positive winner")
    rows = []
    seen = set()
    for row in cells:
        required = {"variant", "seed", "protocol", "recipe_sha256", "allocation_seconds"}
        if not isinstance(row, dict) or set(row) not in (required, required | {"attempt"}):
            raise ValueError("plan cell fields differ from the frozen launch schema")
        key = _cell_identity(row["variant"], row["seed"])
        if key not in expected or key in seen:
            raise ValueError("duplicate or unadmitted scientific cell")
        if type(row["allocation_seconds"]) is not int or row["allocation_seconds"] < 10:
            raise ValueError("each scientific cell needs an explicit finite allocation")
        if "attempt" in row and (not isinstance(row["attempt"], str) or
                                 not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", row["attempt"])):
            raise ValueError("fresh recovery attempt needs a strict explicit identifier")
        if not isinstance(row["recipe_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["recipe_sha256"]):
            raise ValueError("invalid frozen recipe SHA256")
        _relative(row["protocol"], "protocol")
        seen.add(key); rows.append(dict(row))
    if not seen:
        raise ValueError("plan does not contain an admitted nonempty stage")
    return rows, decision


class SharedSpatialController:
    """Injectable controller core; production services are wired only by explicit main()."""
    def __init__(self, *, root: Path, source: str, study_subroot: str, code: Path,
                 services: dict, policy, helper, python: Path, manifest: dict,
                 controller_path: Path, slots: int = 2):
        self.root, self.source, self.study_subroot = root, source, study_subroot
        self.study = root / study_subroot
        self.code, self.s = code, services
        self.policy, self.helper, self.python = policy, helper, python
        self.manifest, self.controller_path = manifest, controller_path
        if type(slots) is not int or slots not in (1, 2):
            raise ValueError("scientific slots must be 1 or 2")
        self.slots = slots
        self.uuid = self.s["uuid"]
        self.stop = threading.Event()
        self._launch_lock = threading.RLock()
        self.startup = self.study / "operator" / "main-startup.lock"

    def _request_stop(self):
        with self._launch_lock:
            self.stop.set()

    def _validate_roots(self):
        if (not self.root.is_absolute() or not self.root.is_dir() or
                self.root.absolute() != self.root.resolve(strict=True) or not _no_links(self.root) or
                not (self.root / "STUDY_OWNER.txt").is_file() or
                (self.root / "STUDY_OWNER.txt").is_symlink()):
            raise ValueError("--root must be a canonical link-free marker-owned directory")
        if self.study_subroot != STUDY_ID:
            raise ValueError("only the assigned shared-spatial study namespace is permitted")
        owner = self.study / "SHARED_SPATIAL_OWNER.txt"
        if (not self.study.is_dir() or not _no_links(self.study) or
                not self.study.resolve(strict=True).is_relative_to(self.root) or
                not owner.is_file() or owner.is_symlink() or
                owner.read_text(encoding="utf-8").strip() != STUDY_ID):
            raise ValueError("shared-spatial namespace must be canonical, owned and confined")
        expected_code = self.root / "code-snapshots" / self.source[:7]
        source_marker = self.code / "SOURCE_COMMIT"
        if (not _full_sha(self.source) or self.code.absolute() != expected_code.absolute() or
                not _no_links(self.code) or not _no_links(source_marker) or not source_marker.is_file() or
                source_marker.read_text(encoding="utf-8").strip() != self.source):
            raise ValueError("callable source directory differs from the immutable requested Git SHA")
        expected_python = self.root / "venv/Scripts/python.exe"
        if (self.python.absolute() != expected_python.absolute() or not self.python.is_file() or
                not _no_links(self.python)):
            raise ValueError("isolated study Python is missing, linked, or outside the owned root")

    def _frozen_cell(self, row):
        path = self.root / _relative(row["protocol"], "protocol")
        _inside(path, self.study)
        if not path.is_file():
            raise ValueError("frozen protocol is missing")
        frozen = self.s["verify_frozen"](path)
        recipe = frozen["recipe"]
        env = recipe["environment"]
        variant, seed = row["variant"], row["seed"]
        expected_spec = None if variant == "native" else {"version": 2, "variant": variant}
        if (frozen["recipe_sha256"] != row["recipe_sha256"] or recipe["epochs"] != 40 or
                recipe["cadence"] != 5 or recipe["seed_list"] != list(SEEDS) or
                recipe["arms"] != ["native"] or recipe["gpu_for_seed"] != {str(s): self.uuid for s in SEEDS} or
                env.get("study_id") != POLICY_STUDY_ID or env.get("head_spec") != expected_spec or
                env.get("source_git_commit") != self.source):
            raise ValueError("frozen full recipe/source/spec/seed assignment differs from the admitted cell")
        return path, frozen, recipe, expected_spec

    def _pilot_input(self, spec):
        allowed = {"candidate_spec", "native_allocation_seconds", "candidate_allocation_seconds", "candidate_tag"}
        if not isinstance(spec, dict) or set(spec) != allowed:
            raise ValueError("technical pilot input fields differ")
        candidate = self.s["canonicalize_head_spec"](spec["candidate_spec"])
        if (candidate != {"version": 2, "variant": candidate.get("variant")} or
                candidate["variant"] not in CANDIDATES):
            raise ValueError("technical v2 pilot must name one registered candidate")
        for key in ("native_allocation_seconds", "candidate_allocation_seconds"):
            if type(spec[key]) is not int or spec[key] < 10:
                raise ValueError("each technical pilot needs an explicitly declared finite allocation")
        tag = spec["candidate_tag"]
        if not isinstance(tag, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", tag):
            raise ValueError("technical pilot tag is invalid")
        return candidate, tag

    def _check_inputs(self, *, pilot):
        paths = ([self.root / "cache/technical-pilot-512-v1", self.code / "manifests/technical-pilot.json"]
                 if pilot else [self.root / "cache/main-few-512-v2", self.root / "cache/main-dev-512-v2",
                                self.root / "cache/main-probe-512-v2", self.code / "manifests/coco-few-dev.json",
                                self.root / "data/annotations/instances_train2017.json"])
        paths += [self.code / "models/trunk-receipt.json", self.root / "sources/trunk-u11.pt"]
        for path in paths:
            _inside(path, self.root)
            if not path.exists():
                raise ValueError(f"required immutable/protected input missing: {path}")
        caches = ([self.root / "cache/technical-pilot-512-v1"] if pilot else
                  [self.root / f"cache/main-{cohort}-512-v2" for cohort in ("few", "dev", "probe")])
        for cache in caches:
            metadata = cache / "metadata.json"
            _inside(metadata, self.root)
            if not metadata.is_file():
                raise ValueError(f"frozen cache metadata missing: {metadata}")
        if not (self.root / "venv/Scripts/python.exe").is_file():
            raise ValueError("isolated study interpreter missing")

    def _admit_resources(self, ledger, allocation_seconds):
        ram = self.helper.available_ram()
        sample = self.s["physical_memory_metrics"](self.uuid)
        if type(ram) is not int or ram < 9 * 1024**3:
            raise RuntimeError("host available RAM below 9 GiB before heavy initialization")
        if (not isinstance(sample, dict) or sample.get("uuid") != self.uuid or
                type(sample.get("free_bytes")) is not int or type(sample.get("total_bytes")) is not int or
                sample["total_bytes"] <= 0 or sample["free_bytes"] < (sample["total_bytes"] + 9) // 10 + 4 * 1024**3):
            raise RuntimeError("fresh G1 VRAM admission margin below 10% plus 4 GiB")
        with ledger._locked():
            state = ledger._load()
        active = [j for d in state["devices"].values() for j in d["jobs"]
                  if j["status"] in ("reserved", "launching", "running", "active-unknown")]
        mine = [j for j in active if j["uuid"] == self.uuid]
        if len(mine) >= 2 or sum(j["cpus"] for j in active) + 3 > 12:
            raise RuntimeError("global G1/host capacity is already occupied")
        pending = sum(j.get("headroom_pending_bytes") or 0 for j in mine)
        if sample["free_bytes"] < (sample["total_bytes"] + 9) // 10 + pending + 4 * 1024**3:
            raise RuntimeError("VRAM margin excludes existing pending commitments")
        return {"available_ram_bytes": ram, "vram": sample, "active_g1": len(mine),
                "active_host_cpus": sum(j["cpus"] for j in active),
                "pending_g1_bytes": pending, "allocation_seconds": allocation_seconds}

    def _job_tree(self, relative: str):
        rel = _relative(relative, "job output")
        path = self.root / rel
        _inside(path, self.study, strict=False)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"job path is not fresh: {relative}")
        return path

    def _command(self, *, pilot, token, row, protocol_path, recipe, spec, output, job):
        py = str(self.python)
        command = [py, "-B", "-u", "-m", "head_sharing.train", "pilot" if pilot else "run",
                   "--device", self.uuid, "--owned-root", str(self.root), "--ledger", str(self.root / "allocation-ledger"),
                   "--reservation", token, "--arm", "native", "--seed", str(PILOT_SEED if pilot else row["seed"])]
        cache = self.root / ("cache/technical-pilot-512-v1" if pilot else "cache/main-few-512-v2")
        manifest = self.code / ("manifests/technical-pilot.json" if pilot else "manifests/coco-few-dev.json")
        command += ["--cache", str(cache), "--manifest", str(manifest), "--receipt", str(self.code / "models/trunk-receipt.json"),
                    "--trunk-artifact", str(self.root / "sources/trunk-u11.pt"), "--output", str(output)]
        if pilot:
            command += ["--imgsz", "512", "--epochs", "3", "--batch-size", "32", "--producer", self.uuid]
            command.append("--finite-diagnostics")
            if spec is not None:
                spec_path = job / "head-spec.json"
                spec_path.write_text(json.dumps(spec, sort_keys=True) + "\n", encoding="utf-8")
                command += ["--head-spec", str(spec_path)]
        else:
            command += ["--protocol", str(protocol_path), "--dev-cache", str(self.root / "cache/main-dev-512-v2"),
                        "--probe-cache", str(self.root / "cache/main-probe-512-v2"),
                        "--annotations", str(self.root / "data/annotations/instances_train2017.json")]
        return command

    def _progress_has_train_batch(self, path: Path) -> bool:
        if not _no_links(path, include_missing=True):
            raise ValueError("progress path is linked")
        if not path.is_file():
            return False
        # Progress is append-only. Ignore only a torn final line; do not trust cache or process events.
        lines = path.read_bytes().splitlines()
        for line in lines:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if (isinstance(row, dict) and row.get("phase") == "train_batch" and
                    type(row.get("done")) is int and row["done"] > 0):
                return True
        return False

    def _run_job(self, *, row, pilot=False, pilot_spec=None):
        ledger = self.s["AllocationLedger"](self.root / "allocation-ledger")
        pilot_evidence = None
        if pilot:
            spec, tag, pilot_evidence = pilot_spec
            variant = "native" if spec is None else spec["variant"]
            allocation = row["allocation_seconds"]
            relative = f"{self.study_subroot}/pilot/{tag}-{variant}"
            protocol_path = None; recipe = None
        else:
            variant = row["variant"]
            allocation = row["allocation_seconds"]
            suffix = f"/attempt-{row['attempt']}" if row.get("attempt") else ""
            relative = f"{self.study_subroot}/runs/{variant}/seed-{row['seed']}{suffix}"
            protocol_path, frozen, recipe, spec = self._frozen_cell(row)
        job = self._job_tree(relative)
        out = job / "output"
        if out.exists() or out.is_symlink():
            raise FileExistsError("output directory is not fresh")
        with ledger._locked():
            state = ledger._load()
            if state.get("policy") is None or not state.get("policy_history"):
                raise RuntimeError("no explicitly authorized finite allocation policy")
        if self.stop.is_set():
            raise _UnstartedCell("queued cell suppressed after an owned failure")
        try:
            initial_admission = self._admit_resources(ledger, allocation)
        except BaseException:
            self._request_stop()
            raise

        job_created = False
        token = None
        executor = None
        future = None
        status = None
        try:
            executor = ThreadPoolExecutor(max_workers=1)
            with self._startup_slot(allocation):
                try:
                    if self.stop.is_set():
                        raise _UnstartedCell("queued cell suppressed before reservation")
                    if job.exists() or job.is_symlink():
                        raise FileExistsError(f"job path is not fresh: {relative}")
                    # Re-sample while serialized, immediately before heavy initialization.
                    admission = self._admit_resources(ledger, allocation)
                    self._launch_lock.acquire()
                    launch_lock_held = True
                    if self.stop.is_set():
                        raise _UnstartedCell("queued cell suppressed before reservation")
                    job.parent.mkdir(parents=True, exist_ok=True)
                    job.mkdir(exist_ok=False)
                    job_created = True
                    out = job / "output"
                    token = ledger.reserve(self.uuid, allocation, phase="main", cpus=3,
                        peak_memory_bytes=4 * 1024**3,
                        command=["shared_spatial_pilot" if pilot else "shared_spatial_fit",
                                 variant, str(PILOT_SEED if pilot else row["seed"]), self.source])
                    command = self._command(pilot=pilot, token=token, row=row, protocol_path=protocol_path,
                                            recipe=recipe, spec=spec, output=out, job=job)
                    progress = job / "progress.jsonl"
                    env = {"PYTHONPATH": str(self.code / "src"), "NVIDIA_TF32_OVERRIDE": "0",
                           "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONDONTWRITEBYTECODE": "1",
                           "OPENCV_FOR_THREADS_NUM": "3"}
                    launch_kwargs = {"cwd": job, "log_path": job / "worker.log", "progress_path": progress,
                                     "first_progress_seconds": 45, "heartbeat_seconds": 180, "env": env,
                                     "memory_guard": True, "memory_poll_seconds": 5.0}
                    self.s["inspect"].signature(self.s["launch"]).bind(ledger, token, command, **launch_kwargs)
                    request = {"study_id": STUDY_ID,
                        "purpose": "technical_pilot_not_ranked" if pilot else "main_dev_fit",
                        "source_commit": self.source, "source_manifest": self.manifest,
                        "full_recipe": recipe, "recipe_sha256": None if pilot else row["recipe_sha256"],
                        "technical_input": pilot_evidence, "head_spec": spec, "variant": variant,
                        "seed": PILOT_SEED if pilot else row["seed"], "uuid": self.uuid,
                        "reservation": token, "allocation_seconds": allocation, "job": relative,
                        "python": str(self.python), "host_available_before_heavy_init": admission["available_ram_bytes"],
                        "initial_admission": initial_admission, "admission": admission,
                        "first_progress_seconds": 45, "heartbeat_seconds": 180,
                        "memory_poll_seconds": 5.0, "cpus": 3, "controller_slots": self.slots, "command": command,
                        "environment": env, "holdout_evaluated": False}
                    (job / "request.json").write_text(json.dumps(request, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                    print(json.dumps({"event": "shared_spatial_reserved", "request": request}, sort_keys=True), flush=True)
                    future = executor.submit(self.s["launch"], ledger, token, command, **launch_kwargs)
                    self._launch_lock.release()
                    launch_lock_held = False
                    progress_seen = False
                    while not future.done():
                        if self._progress_has_train_batch(progress):
                            progress_seen = True
                            print(json.dumps({"event": "shared_spatial_train_batch_started", "job": relative}, sort_keys=True), flush=True)
                            break
                        time.sleep(.05)
                    if future.done():
                        status = future.result()
                    if not progress_seen and self._progress_has_train_batch(progress):
                        progress_seen = True
                    if not progress_seen:
                        raise RuntimeError("supervisor finished before positive train_batch progress")
                except BaseException:
                    # Set stop before releasing the mutex on every pre-progress failure.
                    self._request_stop()
                    raise
                finally:
                    if locals().get("launch_lock_held", False):
                        self._launch_lock.release()
            # Do not let executor shutdown/context exit hold the startup mutex through the fit.
            if status is None:
                status = future.result()
            state = ledger.get_job(token)
            if status != 0 or state.get("status") != "complete":
                raise RuntimeError(f"owned worker did not complete: {token}/{state.get('status')}")
            training_path = out / "training.json"
            _inside(training_path, job)
            record_bytes = training_path.read_bytes()
            record = json.loads(record_bytes)
            if pilot:
                if (record.get("purpose") != "pilot" or record.get("seed") != PILOT_SEED or
                        record.get("epochs_completed") != 3 or record.get("target_epochs") != 3 or
                        record.get("head_spec") != spec or record.get("protocol_sha256") is not None or
                        record.get("hardware", {}).get("device_uuid") != self.uuid or
                        not record.get("head_updated") or any("dev" in r or "probe" in r for r in record.get("history", []))):
                    raise ValueError("technical pilot completion identity mismatch")
            else:
                selected_epoch = record.get("selected_epoch")
                if (type(selected_epoch) is not int or not 1 <= selected_epoch <= 40 or
                        record.get("selected_config") != f"selected-e{selected_epoch}.json"):
                    raise ValueError("completed record selected-checkpoint path is invalid")
                config_path = out / f"selected-e{selected_epoch}.json"
                weights_path = out / f"selected-e{selected_epoch}.pt"
                _inside(config_path, job); _inside(weights_path, job)
                projected = self.policy.project_training_record(
                    training_path, _sha(record_bytes), recipe=recipe, seed=row["seed"],
                    selected_config={"path": str(config_path), "sha256": _sha(config_path.read_bytes())},
                    selected_weights={"path": str(weights_path), "sha256": _sha(weights_path.read_bytes())})
                if projected["variant"] != variant or projected["head_spec"] != spec:
                    raise ValueError("projected completion differs from frozen cell")
            completion = {"request_sha256": _sha((job / "request.json").read_bytes()),
                "source_commit": self.source, "recipe_sha256": None if pilot else row["recipe_sha256"],
                "spec": spec, "seed": PILOT_SEED if pilot else row["seed"], "uuid": self.uuid,
                "reservation": token, "job": relative,
                "training_record": training_path.relative_to(self.root).as_posix(),
                "training_record_sha256": _sha(record_bytes),
                "actual_wall_seconds": state.get("actual_wall_elapsed_seconds"),
                "headroom_summary": state.get("headroom_summary"), "closed": True,
                "holdout_evaluated": False}
            if not pilot:
                completion["projected"] = projected
            with (job / "completion.json").open("x", encoding="utf-8") as stream:
                json.dump(completion, stream, sort_keys=True, indent=2); stream.write("\n")
            print(json.dumps({"event": "shared_spatial_complete", "completion": completion}, sort_keys=True), flush=True)
            return completion
        except _UnstartedCell:
            raise
        except BaseException as exc:
            self._request_stop()
            if job_created:
                try:
                    if token is not None:
                        current = ledger.get_job(token)
                        if current.get("status") == "reserved" and current.get("pid") is None:
                            ledger.update_job(token, status="failed", failure=type(exc).__name__, ended=time.time())
                except BaseException:
                    pass
                failure = {"source_commit": self.source, "job": relative,
                           "variant": variant, "seed": PILOT_SEED if pilot else row["seed"],
                           "error_type": type(exc).__name__, "error": str(exc), "stopped_new_launches": True}
                try:
                    with (job / "failure.json").open("x", encoding="utf-8") as stream:
                        json.dump(failure, stream, sort_keys=True, indent=2); stream.write("\n")
                except FileExistsError:
                    pass
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)

    class _StartupSlot:
        def __init__(self, owner, wait_seconds): self.owner = owner; self.wait_seconds = wait_seconds; self.fd = None
        def __enter__(self):
            if self.owner.stop.is_set():
                raise _UnstartedCell("queued cell suppressed before startup lock")
            self.owner.startup.parent.mkdir(parents=True, exist_ok=True)
            if (not _no_links(self.owner.startup.parent) or
                    not _no_links(self.owner.startup, include_missing=True)):
                raise ValueError("startup lock path is linked")
            try:
                self.fd = self.owner.helper.claim_startup_file(
                    self.owner.startup, time.monotonic() + self.wait_seconds, self.owner.stop)
            except BaseException as exc:
                if self.owner.stop.is_set():
                    raise _UnstartedCell("queued cell suppressed while waiting for startup lock") from exc
                raise
            os.write(self.fd, json.dumps({"pid": os.getpid(), "study_id": STUDY_ID}).encode()); os.close(self.fd)
            self.fd = None
        def __exit__(self, *_):
            self.owner.startup.unlink()

    def _startup_slot(self, wait_seconds):
        return self._StartupSlot(self, wait_seconds)

    def run_pilot(self, pilot_input):
        self._validate_roots()
        candidate, tag = self._pilot_input(pilot_input)
        self._check_inputs(pilot=True)
        metadata_path = self.root / "cache/technical-pilot-512-v1/metadata.json"
        manifest_path = self.code / "manifests/technical-pilot.json"
        metadata_bytes, manifest_bytes = metadata_path.read_bytes(), manifest_path.read_bytes()
        metadata, pilot_manifest = json.loads(metadata_bytes), json.loads(manifest_bytes)
        if (metadata.get("count") != 512 or metadata.get("imgsz") != 512 or metadata.get("batch_size") != 32 or
                metadata.get("producer") != self.uuid or len(pilot_manifest.get("ids", {}).get("few", [])) != 512):
            raise ValueError("seed-997 pilot requires the separate registered 512-image cache/manifest")
        records = []
        for spec, key in ((None, "native_allocation_seconds"), (candidate, "candidate_allocation_seconds")):
            if self.stop.is_set():
                raise RuntimeError("technical pilot stopped after owned failure; remaining task stays unstarted")
            evidence = {"cache_metadata_sha256": _sha(metadata_bytes),
                        "pilot_manifest_sha256": _sha(manifest_bytes), "cache_images": 512,
                        "imgsz": 512, "batch_size": 32}
            records.append(self._run_job(row={"allocation_seconds": pilot_input[key]}, pilot=True,
                                         pilot_spec=(spec, tag + ("-native" if spec is None else "-candidate"), evidence)))
        return {"purpose": "technical_only_not_ranked", "seed": PILOT_SEED, "records": records}

    def run_scientific(self, plan):
        self._validate_roots()
        self._check_inputs(pilot=False)
        rows, decision = validate_scientific_plan(plan, self.policy)
        checked = []
        # Verify every protocol, full recipe, root, output freshness and ledger policy before mutation.
        for row in rows:
            path, frozen, recipe, spec = self._frozen_cell(row)
            suffix = f"/attempt-{row['attempt']}" if row.get("attempt") else ""
            job_rel = f"{self.study_subroot}/runs/{row['variant']}/seed-{row['seed']}{suffix}"
            self._job_tree(job_rel)
            checked.append((row, path, frozen, recipe, spec))
        completed = {}
        failures = []
        unstarted = []
        next_index = 0
        running = {}
        pool = ThreadPoolExecutor(max_workers=min(self.slots, len(checked)))
        try:
            while next_index < len(checked) and len(running) < self.slots and not self.stop.is_set():
                row = checked[next_index][0]
                running[pool.submit(self._run_job, row=row)] = next_index
                next_index += 1
            while running:
                done, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
                for future in done:
                    index = running.pop(future)
                    row = checked[index][0]
                    try:
                        completed[index] = future.result()
                    except _UnstartedCell:
                        unstarted.append(dict(row))
                    except BaseException as exc:
                        failures.append({"type": type(exc).__name__, "message": str(exc),
                                         "variant": row["variant"], "seed": row["seed"],
                                         "attempt": row.get("attempt")})
                        self._request_stop()
                while not failures and next_index < len(checked) and len(running) < self.slots and not self.stop.is_set():
                    row = checked[next_index][0]
                    running[pool.submit(self._run_job, row=row)] = next_index
                    next_index += 1
            unstarted.extend(dict(checked[i][0]) for i in range(next_index, len(checked)))
        finally:
            pool.shutdown(wait=True, cancel_futures=False)
        results = [completed[i] for i in sorted(completed)]
        if failures or unstarted:
            paused = {"mode": plan["mode"], "status": "paused_after_owned_failure",
                      "screen_closed": False, "decision": decision,
                      "completions": results, "failures": failures,
                      "unstarted_cells": unstarted}
            print(json.dumps({"event": "shared_spatial_paused", "result": paused}, sort_keys=True), flush=True)
            raise RuntimeError("shared-spatial stage paused; unstarted cells remain open and unscored")
        if plan["mode"] == "screen" and len(rows) == 12:
            decision = self.policy.decide_stages([r["projected"] for r in results])
        elif plan["mode"] == "screen":
            decision = None
        elif len(rows) == 6:
            decision = self.policy.decide_followups(decision, [r["projected"] for r in results])
        # A partial continuation returns the original sealed winner decision unchanged.
        return {"mode": plan["mode"], "screen_closed": bool(plan["mode"] == "screen" and decision),
                "followups_closed": bool(plan["mode"] == "followup" and len(rows) == 6),
                "decision": decision, "completions": results, "unstarted_cells": []}


def _wire(root: Path, source: str, study_subroot: str, controller_path: Path):
    _validate_bootstrap(root, source, study_subroot)
    code, manifest = _manifest(root, source, controller_path)
    if os.name != "nt":
        raise RuntimeError("live controller invocation is supported only on the verified Windows study host")
    sys.path.insert(0, str(code / "src"))
    from head_sharing.runtime import AllocationLedger, GPU_UUIDS
    from head_sharing.supervisor import launch, physical_memory_metrics
    from head_sharing.protocol import verify_frozen
    from head_sharing.head_spec import canonicalize_head_spec
    policy_path = code / ".internal/shared_spatial.py"
    pspec = importlib.util.spec_from_file_location("_shared_spatial_pinned_policy", policy_path)
    policy_module = importlib.util.module_from_spec(pspec); pspec.loader.exec_module(policy_module)
    helper = _load_helper(code / ".internal/run_ab2_fits.py")
    services = {"AllocationLedger": AllocationLedger, "uuid": GPU_UUIDS[1],
                "physical_memory_metrics": physical_memory_metrics, "launch": launch,
                "verify_frozen": verify_frozen, "canonicalize_head_spec": canonicalize_head_spec,
                "inspect": __import__("inspect")}
    return code, manifest, policy_module, helper, services


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source", required=True, help="full immutable Git SHA")
    parser.add_argument("--study-subroot", required=True)
    parser.add_argument("--mode", choices=("technical", "screen", "followup"), required=True)
    parser.add_argument("--plan", required=True, help="root-relative, predeclared JSON launch input")
    parser.add_argument("--slots", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    root = Path(args.root)
    code, manifest, policy, helper, services = _wire(root, args.source, args.study_subroot, Path(__file__).resolve())
    controller = SharedSpatialController(root=root, source=args.source, study_subroot=args.study_subroot,
        code=code, services=services, policy=policy, helper=helper,
        python=root / "venv/Scripts/python.exe", manifest=manifest,
        controller_path=Path(__file__).resolve(), slots=args.slots)
    controller._validate_roots()
    plan_path = root / _relative(args.plan, "plan")
    _inside(plan_path, controller.study)
    plan = _read_json(plan_path)
    if args.mode == "technical":
        if set(plan) != {"candidate_spec", "native_allocation_seconds", "candidate_allocation_seconds", "candidate_tag"}:
            raise ValueError("technical plan must declare native and candidate finite seconds and one registered spec")
        result = controller.run_pilot(plan)
    else:
        if plan.get("mode") != args.mode:
            raise ValueError("CLI mode differs from sealed scientific plan")
        result = controller.run_scientific(plan)
    print(json.dumps({"event": "shared_spatial_stage_complete", "result": result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
