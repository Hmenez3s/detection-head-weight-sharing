"""Pure recipe, sealed-evidence, and DEV-only selection policy for shared spatial."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from head_sharing.head_spec import canonicalize_head_spec, head_spec_digest
from head_sharing.runtime import GPU_UUIDS

STUDY_ID = "shared_spatial_v1"
G1_UUID = "GPU-08055392-87d6-8287-947d-0bf94c711445"
SEEDS = (11, 22, 33, 44, 55)
SCREEN_SEEDS = (11, 33)
VARIANTS = ("native", "spatial_untied", "spatial_sepbn", "spatial_gn",
            "spatial_sepbn_k5", "spatial_sepbn_sharedpred")
CANDIDATES = ("spatial_sepbn", "spatial_gn", "spatial_sepbn_k5", "spatial_sepbn_sharedpred")
TIE_ORDER = CANDIDATES
TRUNK_ARTIFACT_SHA256 = "f01ec41445fac9ef7078b48ef1b73473e6314bcb1a77e77c20666d6ecfcd9016"
TRUNK_RECEIPT_FIELDS = {"trunk_artifact_sha256", "trunk_tensors", "trunk_bytes", "head_index",
                        "input_channels", "source_variant", "source_seed", "source_epoch_zero_based",
                        "source_training_epochs", "pre_exposure"}


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _recipe_sha(recipe):
    return _digest(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode())


def _spec(variant):
    return None if variant == "native" else {"version": 2, "variant": variant}


def _check_reference(reference, fields, name):
    if not isinstance(reference, dict) or set(reference) != fields:
        raise ValueError(f"{name} source binding has unknown or missing fields")
    path = reference.get("path")
    if (not isinstance(path, str) or not path or "\\" in path or ":" in path or
            str(PurePosixPath(path)) != path or PurePosixPath(path).is_absolute() or
            any(part in ("", ".", "..") for part in path.split("/"))):
        raise ValueError(f"{name} source path must be normalized and relative")
    if any(not _sha(reference.get(field)) for field in fields - {"path"}):
        raise ValueError(f"{name} source digest is invalid")
    return dict(reference)


def prepare_recipe(old_recipe, *, source_protocol, source_freeze, source_git_commit):
    """Copy verified shared inputs and return all six complete immutable recipe values."""
    if not isinstance(old_recipe, dict):
        raise ValueError("source recipe must be an object")
    if old_recipe.get("producer") != G1_UUID:
        raise ValueError("shared-spatial recipes are restricted to the assigned G1 producer")
    from head_sharing.protocol import validate_recipe
    validate_recipe(old_recipe, synthetic=False)
    protocol_ref = _check_reference(source_protocol, {"path", "sha256", "recipe_sha256"}, "protocol")
    freeze_ref = _check_reference(source_freeze, {"path", "sha256"}, "freeze")
    if not isinstance(source_git_commit, str) or len(source_git_commit) != 40 or any(
            char not in "0123456789abcdef" for char in source_git_commit):
        raise ValueError("new immutable source_git_commit must be a lowercase full Git SHA")
    if protocol_ref["recipe_sha256"] != _recipe_sha(old_recipe):
        raise ValueError("source protocol recipe hash does not match the supplied frozen recipe")
    recipes = {}
    for variant in VARIANTS:
        spec = _spec(variant)
        recipe = copy.deepcopy(old_recipe)
        recipe.update(seed_list=list(SEEDS), arms=["native"], epochs=40, cadence=5,
                      imgsz=512, batch_size=32, block_size=32, probe_size=1000,
                      gpu_for_seed={str(seed): G1_UUID for seed in SEEDS})
        environment = copy.deepcopy(old_recipe.get("environment"))
        if not isinstance(environment, dict):
            raise ValueError("source recipe environment must be an object")
        for stale in ("reference_native", "reference_baseline", "looped_architecture"):
            environment.pop(stale, None)
        environment.pop("cumulative_gpu_cap_seconds_per_uuid", None)
        environment.update(study_id=STUDY_ID, head_spec=spec,
                           source_protocol=protocol_ref, source_freeze=freeze_ref,
                           source_git_commit=source_git_commit)
        recipe["environment"] = environment
        validate_recipe(recipe, synthetic=False)
        recipes[variant] = recipe
    return recipes


def read_sealed_json(path, expected_sha256):
    if not _sha(expected_sha256):
        raise ValueError("expected SHA256 must be lowercase hexadecimal")
    content = Path(path).read_bytes()
    if _digest(content) != expected_sha256:
        raise ValueError("sealed JSON SHA256 mismatch")
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("sealed file is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("sealed JSON root must be an object")
    return value


def _file_ref(reference, expected_path=None):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("artifact reference must contain exactly path and sha256")
    if not isinstance(reference["path"], str) or not reference["path"] or not _sha(reference["sha256"]):
        raise ValueError("artifact path or SHA256 is invalid")
    path = Path(reference["path"])
    expected = Path(expected_path) if expected_path is not None else None
    if (expected is not None and path.absolute() != expected.absolute()) or path.is_symlink():
        raise ValueError("selected artifact path differs from the producer-selected path")
    actual = _digest(path.read_bytes())
    if actual != reference["sha256"]:
        raise ValueError("artifact SHA256 mismatch")
    return actual


def project_training_record(path, expected_sha256, *, recipe, seed, selected_config, selected_weights):
    """Project a complete, sealed producer record without loading model/tensor bytes."""
    record = read_sealed_json(path, expected_sha256)
    if not isinstance(recipe, dict):
        raise ValueError("expected full frozen recipe is required")
    from head_sharing.protocol import validate_recipe
    validate_recipe(recipe, synthetic=False)
    env = recipe["environment"]
    spec = canonicalize_head_spec(env.get("head_spec"))
    if env.get("study_id") != STUDY_ID or spec != env.get("head_spec"):
        raise ValueError("expected recipe is not canonical shared-spatial identity")
    recipe_sha = _recipe_sha(recipe)
    if (record.get("schema_version") != 1 or record.get("purpose") != "main" or
            record.get("synthetic") is not False or record.get("arm") != "native" or
            record.get("selection_metric") != "dev.AP50:95" or type(seed) is not int or seed not in SEEDS or
            record.get("seed") != seed or record.get("target_epochs") != 40 or
            record.get("epochs_completed") != 40 or record.get("protocol_sha256") != recipe_sha or
            record.get("recipe") != recipe or record.get("head_spec") != spec or
            record.get("head_spec_sha256") != head_spec_digest(spec)):
        raise ValueError("training record is incomplete or bound to another study/recipe/spec/seed")
    history = record.get("history")
    if not isinstance(history, list) or [row.get("epoch") if isinstance(row, dict) else None for row in history] != list(range(1, 41)):
        raise ValueError("training history must contain exactly 40 contiguous epochs")
    cadence = [epoch for epoch in range(1, 41) if epoch % 5 == 0 or epoch == 40]
    scores = {}
    for epoch in cadence:
        row = history[epoch - 1]
        for cohort in ("dev", "probe"):
            metrics = row.get(cohort)
            value = metrics.get("AP50:95") if isinstance(metrics, dict) else None
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"invalid scheduled {cohort.upper()} AP")
            if cohort == "dev":
                scores[epoch] = float(value)
    if any("dev" in history[e - 1] or "probe" in history[e - 1] for e in range(1, 41) if e not in cadence):
        raise ValueError("DEV/PROBE metrics occur outside the frozen cadence")
    selected_epoch = record.get("selected_epoch")
    if (type(selected_epoch) is not int or selected_epoch not in cadence or
            selected_epoch != max(cadence, key=lambda epoch: scores[epoch])):
        raise ValueError("selected checkpoint is not the earliest best cadence DEV epoch")
    selected_name = f"selected-e{selected_epoch}.json"
    if record.get("selected_config") != selected_name:
        raise ValueError("producer selected-config path is not the selected epoch")
    record_dir = Path(path).resolve().parent
    config_path, weights_path = record_dir / selected_name, record_dir / f"selected-e{selected_epoch}.pt"
    config_sha = _file_ref(selected_config, config_path)
    weights_sha = _file_ref(selected_weights, weights_path)
    config = read_sealed_json(selected_config["path"], config_sha)
    expected_config_keys = {"format", "architecture", "arm", "seed", "imgsz", "quantization", "trunk_receipt"}
    if spec is not None:
        expected_config_keys.add("head_spec")
    receipt = config.get("trunk_receipt")
    valid_receipt = (isinstance(receipt, dict) and set(receipt) == TRUNK_RECEIPT_FIELDS and
                     receipt.get("trunk_artifact_sha256") == TRUNK_ARTIFACT_SHA256 == recipe["trunk_sha256"] and
                     receipt.get("trunk_tensors") == 468 and receipt.get("trunk_bytes") == 9_109_168 and
                     receipt.get("head_index") == 23 and receipt.get("input_channels") == [64, 128, 256] and
                     receipt.get("source_variant") == "U" and receipt.get("source_seed") == 11 and
                     receipt.get("source_epoch_zero_based") == 79 and
                     receipt.get("source_training_epochs") == 80 and
                     isinstance(receipt.get("pre_exposure"), str) and bool(receipt.get("pre_exposure")))
    if (set(config) != expected_config_keys or config.get("architecture") != "yolo26n" or
            config.get("arm") != "native" or config.get("seed") != seed or
            config.get("imgsz") != recipe["imgsz"] or config.get("quantization") != recipe["quantization"] or
            config.get("head_spec") != spec or config.get("format") != (1 if spec is None else 2) or
            not valid_receipt):
        raise ValueError("selected config has wrong producer architecture/arm/seed/spec/format/receipt identity")
    if record.get("selected_dev_ap") != scores[selected_epoch] or record.get("head_updated") is not True:
        raise ValueError("record selected score or head-update evidence is invalid")
    hardware = record.get("hardware")
    if not isinstance(hardware, dict) or hardware.get("device_uuid") != G1_UUID:
        raise ValueError("training record is not bound to G1")
    trunk = record.get("frozen_trunk_digest")
    if not _sha(trunk):
        raise ValueError("training record lacks its frozen trunk binding")
    return {"variant": "native" if spec is None else spec["variant"], "head_spec": spec,
            "seed": seed, "epochs": 40, "dev_ap": scores[selected_epoch],
            "probe_ap": history[selected_epoch - 1]["probe"]["AP50:95"],
            "history": history, "selected_epoch": selected_epoch,
            "training": {"path": str(path), "sha256": expected_sha256},
            "selected_config": dict(selected_config), "selected_weights": dict(selected_weights),
            "recipe_sha256": recipe_sha, "head_spec_sha256": head_spec_digest(spec),
            "trunk_sha256": recipe["trunk_sha256"], "frozen_trunk_digest": trunk,
            "source": copy.deepcopy(env.get("source_protocol")), "closed": True}


def _row_key(row):
    return row.get("variant"), row.get("seed")


def _validate_failed(row, variant, seed):
    if row.get("status") != "failed" or row.get("closed") is not True:
        raise ValueError("failed disposition must be explicitly closed")
    if (row.get("variant") != variant or type(row.get("seed")) is not int or row.get("seed") != seed or
            row.get("dev_ap") is not None or row.get("probe_ap") is not None or
            row.get("selected_dev_ap") is not None):
        raise ValueError("failed disposition identity/score is invalid")
    done, reason, evidence = row.get("epochs_completed"), row.get("failure_reason"), row.get("failure_evidence")
    if type(done) is not int or not 0 <= done <= 40 or not isinstance(reason, str) or not reason.strip():
        raise ValueError("failed disposition needs progress within the target dose and a reason")
    if not isinstance(evidence, dict) or set(evidence) != {"path", "sha256"} or not evidence.get("path") or not _sha(evidence.get("sha256")):
        raise ValueError("failed disposition needs sealed failure evidence")


def _validate_projected(row, variant, seed):
    spec = _spec(variant)
    if (row.get("closed") is not True or row.get("status", "completed") != "completed" or
            row.get("variant") != variant or type(row.get("seed")) is not int or row.get("seed") != seed or
            row.get("epochs") != 40 or row.get("head_spec") != spec or
            type(row.get("dev_ap")) not in (int, float) or not math.isfinite(row["dev_ap"]) or
            not 0 <= row["dev_ap"] <= 1):
        raise ValueError("completed disposition does not match its sealed full-dose cell")
    if not isinstance(row.get("training"), dict) or not _sha(row["training"].get("sha256")):
        raise ValueError("completed disposition lacks sealed training provenance")
    return row


def _paired(rows, candidate):
    by_key = {_row_key(row): row for row in rows if row.get("status", "completed") == "completed"}
    if any((name, seed) not in by_key for name in ("native", candidate) for seed in SCREEN_SEEDS):
        return None
    return [100.0 * (by_key[(candidate, seed)]["dev_ap"] - by_key[("native", seed)]["dev_ap"])
            for seed in SCREEN_SEEDS]


def decide_stages(screens, *, followups=()):
    """Close all twelve screen cells, select at most one positive paired winner, and bind follow-ups."""
    expected = {(variant, seed) for variant in VARIANTS for seed in SCREEN_SEEDS}
    if not isinstance(screens, list) or len(screens) != 12:
        raise ValueError("selection requires all twelve screen dispositions")
    by_key = {}
    for row in screens:
        if not isinstance(row, dict) or row.get("status") not in ("completed", "failed"):
            raise ValueError("every screen disposition must be completed or failed; unstarted is not closed")
        key = _row_key(row)
        if key not in expected or key in by_key:
            raise ValueError("screen contains duplicate or unexpected condition/seed")
        if row["status"] == "failed":
            _validate_failed(row, *key)
        else:
            _validate_projected(row, *key)
        by_key[key] = row
    if set(by_key) != expected:
        raise ValueError("all six conditions x seeds 11/33 must be closed")
    candidates = []
    for variant in CANDIDATES:
        deltas = _paired(screens, variant)
        if deltas is not None:
            mean = sum(deltas) / 2
            if mean > 0:
                candidates.append((mean, TIE_ORDER.index(variant), variant, deltas))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    winner = candidates[0] if candidates else None
    selected = winner[2] if winner else None
    if list(followups):
        raise ValueError("follow-ups require the immutable decision; use decide_followups")
    refs = copy.deepcopy(screens)
    decision = {"study_id": STUDY_ID, "screen": refs, "screen_closed": True,
            "winner": selected, "winner_mean_paired_dev_delta_pp": winner[0] if winner else None,
            "winner_screen_deltas_pp": winner[3] if winner else None,
            "followup_cells": ([{"variant": "native", "seed": s} for s in (22, 44, 55)] +
                               [{"variant": selected, "seed": s} for s in (22, 44, 55)]) if selected else [],
            "followups": [], "final_inference_authorized": False,
            "complete_five_pair_group": False}
    decision["decision_sha256"] = _decision_digest(decision)
    return decision


def _decision_digest(decision):
    payload = {key: value for key, value in decision.items() if key != "decision_sha256"}
    return _digest(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def decide_followups(decision, followups):
    """Attach only the frozen winner plus fresh native 22/44/55 outcomes; never rerank."""
    if (not isinstance(decision, dict) or decision.get("study_id") != STUDY_ID or
            decision.get("screen_closed") is not True or
            decision.get("decision_sha256") != _decision_digest(decision)):
        raise ValueError("follow-up decision is not an intact closed shared-spatial screen")
    if decision.get("winner") is None:
        if followups:
            raise ValueError("follow-ups are prohibited when no candidate won")
        return {**decision, "followups": [], "complete_five_pair_group": False,
                "final_inference_authorized": False}
    variant = decision["winner"]
    expected = {(name, seed) for name in ("native", variant) for seed in (22, 44, 55)}
    if not isinstance(followups, list) or len(followups) != 6:
        raise ValueError("exactly six winner/native follow-up dispositions are required")
    seen = set()
    for row in followups:
        if not isinstance(row, dict) or row.get("status") not in ("completed", "failed"):
            raise ValueError("follow-up must be a closed completed or failed disposition")
        key = _row_key(row)
        if key not in expected or key in seen:
            raise ValueError("follow-up contains duplicate or non-winner condition")
        if row["status"] == "failed":
            _validate_failed(row, *key)
        else:
            _validate_projected(row, *key)
        seen.add(key)
    if seen != expected:
        raise ValueError("fresh native and winner follow-ups are incomplete")
    result = {**decision, "followups": copy.deepcopy(followups)}
    result.pop("decision_sha256", None)
    result["complete_five_pair_group"] = all(
        any(row.get("variant") == name and row.get("seed") == seed and row.get("status") == "completed"
            for row in decision["screen"] + followups)
        for name in ("native", variant) for seed in SEEDS)
    result["final_inference_authorized"] = False
    result["decision_sha256"] = _decision_digest(result)
    return result


def _stats(deltas):
    from head_sharing.analyze import paired_stats
    result = paired_stats(deltas)
    return {"deltas_pp": result["deltas_pp"], "mean_pp": result["mean_pp"],
            "sample_sd_pp": result["sample_sd_pp"], "t95_df4_pp": result["t95_df4_pp"]}


def render_dev_report(study, decision):
    """Render a private English DEV report; never an inference or promotion artifact."""
    if (not isinstance(study, dict) or not isinstance(decision, dict) or
            decision.get("study_id") != STUDY_ID or
            decision.get("decision_sha256") != _decision_digest(decision)):
        raise ValueError("report requires an intact shared-spatial study decision")
    screens, followups = study.get("screens"), decision.get("followups", [])
    if not isinstance(screens, list) or len(screens) != 12 or decision.get("screen_closed") is not True:
        raise ValueError("report requires the complete twelve-cell screen")
    observed = {(row.get("variant"), row.get("seed")): row for row in screens + followups
                if isinstance(row, dict)}
    if len(observed) != len(screens) + len(followups):
        raise ValueError("report contains duplicate outcome identities")
    if decision.get("winner") not in (None, *CANDIDATES):
        raise ValueError("decision names an unregistered winner")
    if decision.get("final_inference_authorized") is not False:
        raise ValueError("shared-spatial DEV evidence never authorizes final inference")
    if screens != decision.get("screen"):
        raise ValueError("report screen differs from the selection-bound dispositions")
    lines = ["# Shared-spatial — private development report", "",
             "DEV-only exploratory results; no holdout evaluation, final inference, publication, or promotion is authorized.",
             "The +0.15 AP-point target was an aspiration, not a guaranteed outcome. PROBE is in-sample descriptive evidence.",
             "All conditions inherit one frozen trunk; DEV was reused for adaptive selection, so the t95 interval is descriptive after selection and is not a significance gate.",
             "Matched initialization controls initialization, not all architectural confounding. GN is GroupNorm (not LayerNorm); storage savings do not imply speed gains.", "",
             "## Screen dispositions"]
    for variant in VARIANTS:
        for seed in SCREEN_SEEDS:
            row = observed.get((variant, seed))
            if row is None:
                raise ValueError("screen report is missing a disposition")
            if row.get("status") == "failed":
                _validate_failed(row, variant, seed)
                lines.append(f"- {variant} / seed {seed}: FAILED after {row['epochs_completed']} epochs; no full-dose score; {row['failure_reason']} (sealed {row['failure_evidence']['path']} / {row['failure_evidence']['sha256']}).")
            elif row.get("status") == "completed":
                _validate_projected(row, variant, seed)
                lines.append(f"- {variant} / seed {seed}: DEV AP50:95 {row['dev_ap']:.8f}; PROBE {row.get('probe_ap', 'not reported')}; selected epoch {row.get('selected_epoch', 'unknown')}; training SHA256 {row['training']['sha256']}.")
            else:
                raise ValueError("report cannot treat an unstarted condition as closed")
    winner = decision.get("winner")
    if winner is not None and followups:
        lines.extend(["", "## Fresh native and winner follow-ups"])
        for seed in (22, 44, 55):
            for variant in ("native", winner):
                row = observed.get((variant, seed))
                if row is None:
                    raise ValueError("report is missing an admitted follow-up disposition")
                if row.get("status") == "failed":
                    _validate_failed(row, variant, seed)
                    lines.append(f"- {variant} / seed {seed}: FAILED after {row['epochs_completed']} epochs; no admissible full-dose score; reason: {row['failure_reason']}; sealed failure reference: {row['failure_evidence']['path']} (SHA256 {row['failure_evidence']['sha256']}).")
                elif row.get("status") == "completed":
                    _validate_projected(row, variant, seed)
                    lines.append(f"- {variant} / seed {seed}: DEV AP50:95 {row['dev_ap']:.8f}; PROBE {row.get('probe_ap', 'not reported')}; selected epoch {row.get('selected_epoch', 'unknown')}; training SHA256 {row['training']['sha256']}.")
                else:
                    raise ValueError("report cannot treat an unstarted follow-up as closed")
    winner = decision.get("winner")
    lines.extend(["", "## Selection"])
    if winner is None:
        lines.append("No candidate had both complete native/candidate pairs with a strictly positive paired DEV mean; no follow-ups were admitted.")
    else:
        lines.append(f"Selected at most one candidate: {winner}; screen paired mean {decision['winner_mean_paired_dev_delta_pp']:.4f} AP points. The adaptive screen alone is not a five-seed estimate.")
    lines.append("The untied matched-initialization condition is descriptive only and is never eligible to win.")
    lines.extend(["", "## Five-seed paired DEV outcomes"])
    complete = decision.get("complete_five_pair_group") is True
    if winner and complete:
        for seed in SEEDS:
            native, candidate = observed.get(("native", seed)), observed.get((winner, seed))
            if not native or not candidate or native.get("status") != "completed" or candidate.get("status") != "completed":
                raise ValueError("complete group is missing a finite native/candidate pair")
            delta = 100 * (candidate["dev_ap"] - native["dev_ap"])
            lines.append(f"- seed {seed}: native {native['dev_ap']:.8f}, {winner} {candidate['dev_ap']:.8f}, paired delta {delta:.4f} AP points.")
        stats = _stats([100 * (observed[(winner, seed)]["dev_ap"] - observed[("native", seed)]["dev_ap"])
                        for seed in SEEDS])
        lines.append(f"Mean paired DEV delta: {stats['mean_pp']:.4f} AP points; sample SD {stats['sample_sd_pp']:.4f}; descriptive t95(df=4) interval [{stats['t95_df4_pp'][0]:.4f}, {stats['t95_df4_pp'][1]:.4f}]. This interval is exploratory after adaptive selection, on one inherited trunk; no p-value or final-inference claim is made.")
    else:
        lines.append("The five-pair group is incomplete; no five-seed mean, SD, or interval is reported or imputed.")
    lines.extend(["", "Final inference authority: NONE. Holdout status: CLOSED."])
    return "\n".join(lines) + "\n"
