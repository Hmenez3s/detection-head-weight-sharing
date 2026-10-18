"""Inert bookkeeping for the private AB2 refinement study; parent owns execution."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from head_sharing.head_spec import canonicalize_head_spec, head_spec_digest

STUDY_ID = "ab2_refinement_v1"
VARIANTS = ("baseline80", "learned_alpha", "step_norm", "scale_gain", "dw5")
VARIANT_ORDER = ("baseline80", "learned_alpha", "step_norm", "scale_gain", "dw5")
BASELINE11_AP = 0.14701567894683693
G1_UUID = "GPU-08055392-87d6-8287-947d-0bf94c711445"
FROZEN_TRUNK_STATE_SHA256 = "5defe62306ced17972e0f20d7c02627339d54b7460083fa2375dc88f08ffb2e2"
PINNED_PROTOCOL = {"path": "looped-20261017T181804Z/protocol/frozen.json",
                   "sha256": "25ae058950b690e89659ed80f6175d3d4f78409f00ce1735535197ecb6905a4b",
                   "recipe_sha256": "11ad424d07404fd1687c897fb1db51ffc1287a331b2022b73854f14be752e3a2"}
PINNED_FREEZE = {"path": "looped-20261017T181804Z/selection/selected.json",
                 "sha256": "df3d874822a1c8015c312ac6f0ce8c59b4d64c1494c6ca55f5ddd70c5d7c2a47"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def read_sealed_json(path, expected_sha256):
    """Read an explicitly named JSON file only if its exact bytes match the seal."""
    if not _is_sha(expected_sha256):
        raise ValueError("expected SHA256 must be lowercase hexadecimal")
    content = Path(path).read_bytes()
    if _sha256(content) != expected_sha256:
        raise ValueError("sealed JSON file SHA256 mismatch")
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("sealed file is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("sealed JSON root must be an object")
    return value


def verify_file_ref(reference):
    """Verify one explicit {path, sha256} artifact binding and return its hash."""
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("artifact reference must contain exactly path and sha256")
    if not isinstance(reference["path"], str) or not reference["path"] or not _is_sha(reference["sha256"]):
        raise ValueError("invalid artifact reference")
    actual = _sha256(Path(reference["path"]).read_bytes())
    if actual != reference["sha256"]:
        raise ValueError("artifact SHA256 mismatch")
    return actual


def prepare_recipe(old_recipe, *, head_spec, epochs, source_protocol, source_freeze,
                   producer, cache_evidence, pilot_evidence, environment_updates=None):
    """Return an unfrozen new-study recipe, copying common values from the old recipe.

    Parent supplies freshly verified feasibility/cache evidence. No recipe is frozen here.
    """
    if not isinstance(old_recipe, dict):
        raise ValueError("old recipe must be an object")
    if type(epochs) is not int or epochs not in (40, 80):
        raise ValueError("refinement dose must be exactly 40 or 80")
    spec = canonicalize_head_spec(head_spec)
    if spec != head_spec:
        raise ValueError("head_spec must already be canonical")
    for reference, fields in ((source_protocol, {"path", "sha256", "recipe_sha256"}),
                              (source_freeze, {"path", "sha256"})):
        if not isinstance(reference, dict) or set(reference) != fields:
            raise ValueError("source binding has unknown or missing fields")
        if not isinstance(reference["path"], str) or not reference["path"] or any(
                not _is_sha(reference[key]) for key in fields - {"path"}):
            raise ValueError("source binding path/hash is invalid")
    if source_protocol != PINNED_PROTOCOL or source_freeze != PINNED_FREEZE:
        raise ValueError("refinement source identity differs from pinned G1 artifacts")
    updates = {} if environment_updates is None else environment_updates
    protected = {"study_id", "head_spec", "reference_baseline", "looped_architecture", "reference_native"}
    if not isinstance(updates, dict) or protected.intersection(updates):
        raise ValueError("environment updates cannot override protected study identity")
    recipe = copy.deepcopy(old_recipe)
    recipe.update(seed_list=[11, 33, 55], arms=["loop_ab2"], epochs=epochs,
                  producer=producer, cache_evidence=copy.deepcopy(cache_evidence),
                  pilot_evidence=copy.deepcopy(pilot_evidence),
                  gpu_for_seed={str(seed): old_recipe["gpu_for_seed"][str(seed)] for seed in (11, 33, 55)})
    environment = copy.deepcopy(old_recipe.get("environment", {}))
    if not isinstance(environment, dict):
        raise ValueError("old recipe environment must be an object")
    environment.pop("looped_architecture", None)
    environment.pop("reference_native", None)
    environment.update(updates)
    environment.update(study_id=STUDY_ID, head_spec=spec,
                       reference_baseline={"protocol": dict(source_protocol), "freeze": dict(source_freeze)})
    recipe["environment"] = environment
    # protocol.validate_recipe is the authority for the complete producer recipe ABI.
    from head_sharing.protocol import validate_recipe
    validate_recipe(recipe, synthetic=False)
    return recipe


def project_training_record(path, expected_sha256, *, recipe_sha256, head_spec, seed, epochs,
                            selected_config, selected_weights):
    """Verify a sealed completed training JSON and its separately sealed selected artifacts."""
    record = read_sealed_json(path, expected_sha256)
    spec = canonicalize_head_spec(head_spec)
    if spec != head_spec or type(seed) is not int or seed not in (11, 33, 55):
        raise ValueError("expected canonical identity/seed is invalid")
    if type(epochs) is not int or epochs not in (40, 80) or not _is_sha(recipe_sha256):
        raise ValueError("expected dose/recipe hash is invalid")
    recipe = record.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("training record lacks full recipe identity")
    from head_sharing.protocol import validate_recipe
    validate_recipe(recipe, synthetic=False)
    env = recipe["environment"]
    calculated_recipe_sha = _sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode())
    if (record.get("purpose") != "main" or record.get("synthetic") is not False or
            record.get("arm") != "loop_ab2" or record.get("seed") != seed or
            record.get("target_epochs") != epochs or record.get("epochs_completed") != epochs or
            record.get("protocol_sha256") != recipe_sha256 or calculated_recipe_sha != recipe_sha256 or
            recipe["epochs"] != epochs or env.get("study_id") != STUDY_ID or env.get("head_spec") != spec or
            record.get("head_spec") != spec or record.get("head_spec_sha256") != head_spec_digest(spec)):
        raise ValueError("training record is incomplete or has wrong study identity/dose")
    history = record.get("history")
    if not isinstance(history, list) or not history or any(
            not isinstance(row, dict) or type(row.get("epoch")) is not int for row in history):
        raise ValueError("training record lacks a valid completion curve")
    if [row["epoch"] for row in history] != list(range(1, epochs + 1)):
        raise ValueError("training history does not close the immutable target")
    cadence = recipe["cadence"]
    eval_epochs = [epoch for epoch in range(1, epochs + 1) if epoch % cadence == 0 or epoch == epochs]
    scores = {}
    for epoch in eval_epochs:
        dev = history[epoch - 1].get("dev")
        probe = history[epoch - 1].get("probe")
        if not isinstance(dev, dict) or not isinstance(probe, dict):
            raise ValueError("scheduled evaluation epoch lacks DEV/PROBE metrics")
        for label, metrics in (("DEV", dev), ("PROBE", probe)):
            score = metrics.get("AP50:95")
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"scheduled {label} AP50:95 is invalid")
        scores[epoch] = dev["AP50:95"]
    if any("dev" in history[epoch - 1] or "probe" in history[epoch - 1]
           for epoch in range(1, epochs + 1) if epoch not in eval_epochs):
        raise ValueError("DEV/PROBE metrics appear outside the producer cadence")
    selected_epoch = record.get("selected_epoch")
    if type(selected_epoch) is not int or selected_epoch not in eval_epochs:
        raise ValueError("training record has no selected dev checkpoint on the evaluation cadence")
    if selected_epoch != max(eval_epochs, key=lambda epoch: scores[epoch]):
        raise ValueError("selected epoch is not the earliest best evaluated DEV epoch")
    selected = history[selected_epoch - 1]
    dev = selected["dev"]
    if record.get("selected_dev_ap") != dev["AP50:95"]:
        raise ValueError("selected dev score disagrees with history")
    if record.get("head_updated") is not True:
        raise ValueError("training record does not prove the head updated")
    hardware = record.get("hardware")
    if not isinstance(hardware, dict) or hardware.get("device_uuid") != G1_UUID:
        raise ValueError("training record is not bound to the pinned G1 device")
    if record.get("frozen_trunk_digest") != FROZEN_TRUNK_STATE_SHA256:
        raise ValueError("training record frozen trunk state digest mismatch")
    selected_name = f"selected-e{selected_epoch}.json"
    if record.get("selected_config") != selected_name:
        raise ValueError("record selected_config does not name the selected checkpoint")
    record_dir = Path(path).resolve().parent
    expected_config = (record_dir / selected_name).resolve()
    expected_weights = expected_config.with_suffix(".pt")
    if (Path(selected_config["path"]).resolve() != expected_config or
            Path(selected_weights["path"]).resolve() != expected_weights):
        raise ValueError("selected config/weights paths differ from the training record selection")
    config_hash = verify_file_ref(selected_config)
    weights_hash = verify_file_ref(selected_weights)
    config = read_sealed_json(selected_config["path"], config_hash)
    if (config.get("arm") != "loop_ab2" or config.get("seed") != seed or
            config.get("head_spec") != spec or config.get("format") != (1 if spec is None else 2)):
        raise ValueError("selected checkpoint config identity differs from the sealed training record")
    return {"variant": "baseline80" if spec is None else spec["variant"], "head_spec": spec,
            "seed": seed, "epochs": epochs, "dev_ap": float(dev["AP50:95"]),
            "probe_ap": _metric(history, selected_epoch, "probe", "AP50:95"),
            "history": history, "training": {"path": str(path), "sha256": expected_sha256},
            "selected_config": {"path": selected_config["path"], "sha256": config_hash},
            "selected_weights": {"path": selected_weights["path"], "sha256": weights_hash},
            "recipe_sha256": recipe_sha256, "selected_epoch": selected_epoch,
            "closed": True}


def _metric(history, epoch, block, metric):
    value = history[epoch - 1].get(block)
    return value.get(metric) if isinstance(value, dict) else None


def attach_head_conv_macs(row, head_conv_macs):
    """Attach a parent's CPU-executed B1/512 retained-dual head MAC count."""
    if not isinstance(row, dict) or row.get("closed") is not True:
        raise ValueError("ConvMAC evidence requires a closed projected run")
    if type(head_conv_macs) is not int or head_conv_macs < 1:
        raise ValueError("executed head ConvMAC count must be a positive integer")
    result = dict(row)
    result["conv_macs"] = head_conv_macs
    return result


def attach_profile(row, *, conv_macs, profile):
    """Optionally bind a sealed retained-dual B1/512 profile to a projected run."""
    if not isinstance(row, dict) or row.get("closed") is not True:
        raise ValueError("profile requires a closed projected run")
    if type(conv_macs) is not int or conv_macs < 1 or row.get("conv_macs") != conv_macs:
        raise ValueError("profile head ConvMAC must match the selected screen evidence")
    profile_json = read_sealed_json(profile["path"], profile["sha256"])
    identity = profile_json.get("model_identity")
    expected_variant = "baseline" if row["head_spec"] is None else row["head_spec"]["variant"]
    macs = profile_json.get("head_executed_conv_macs")
    inputs = profile_json.get("measurement_inputs")
    if (profile_json.get("measurement") != "inference" or profile_json.get("branch_policy") != "retained_dual" or
            profile_json.get("imgsz") != 512 or not isinstance(inputs, dict) or inputs.get("batch_size") != 1 or
            not isinstance(identity, dict) or identity.get("weights_sha256") != row["selected_weights"]["sha256"] or
            identity.get("config_sha256") != row["selected_config"]["sha256"] or
            identity.get("recipe_sha256") != row["recipe_sha256"] or identity.get("variant") != expected_variant or
            identity.get("head_spec") != row["head_spec"] or
            identity.get("head_spec_sha256") != head_spec_digest(row["head_spec"]) or
            not isinstance(macs, dict) or macs.get("conv_macs") != conv_macs):
        raise ValueError("profile identity, scope, or executed head ConvMAC does not match selected run")
    result = dict(row)
    result.update(conv_macs=conv_macs, profile={"path": profile["path"], "sha256": profile["sha256"]})
    return result


def decide_stages(screens, *, baseline40_ap=BASELINE11_AP, confirms=(), fresh_main_seconds=None,
                  required_margin_seconds=None, baseline40_mean=None, cost_seed_seconds=5400):
    """Choose one closed screen winner; admit only its confirmation/cost stages."""
    if not isinstance(screens, list) or len(screens) != 5:
        raise ValueError("selection requires all five sealed screen records")
    if type(baseline40_ap) not in (int, float) or not math.isfinite(baseline40_ap) or not 0 <= baseline40_ap <= 1:
        raise ValueError("raw baseline40 DEV AP must be a finite fraction")
    by_variant = {}
    for row in screens:
        if not isinstance(row, dict) or row.get("closed") is not True:
            raise ValueError("every screen must be closed and sealed")
        variant, dose = row.get("variant"), row.get("epochs")
        if variant not in VARIANTS or variant in by_variant or dose != (80 if variant == "baseline80" else 40):
            raise ValueError("screen variants/doses are incomplete or invalid")
        expected_spec = None if variant == "baseline80" else {
            "version": 1, "variant": variant, "class_inner": 96}
        try:
            canonical_spec = canonicalize_head_spec(row.get("head_spec"))
        except ValueError as exc:
            raise ValueError("screen head_spec is not canonical") from exc
        if row.get("head_spec") != expected_spec or canonical_spec != expected_spec:
            raise ValueError("screen head_spec does not match its registered variant/dose")
        if (type(row.get("seed")) is not int or row["seed"] != 11 or
                type(row.get("dev_ap")) not in (int, float) or not math.isfinite(row["dev_ap"]) or
                not 0 <= row["dev_ap"] <= 1):
            raise ValueError("screens require finite raw seed-11 DEV AP fractions")
        if type(row.get("conv_macs")) is not int or row["conv_macs"] < 1:
            raise ValueError("screen is missing actual ConvMAC tie-break evidence")
        by_variant[variant] = row
    if set(by_variant) != set(VARIANTS):
        raise ValueError("all four primary40 and baseline80 screens are required")
    primary = [by_variant[name] for name in VARIANT_ORDER[1:]]
    winner = min(primary + [by_variant["baseline80"]],
                 key=lambda row: (-row["dev_ap"], row["conv_macs"], VARIANT_ORDER.index(row["variant"])))
    improves = winner["dev_ap"] > baseline40_ap
    selected = winner["variant"] if improves else None
    permitted_confirms = []
    skipped = []
    if selected is None:
        skipped.extend(["confirmation: no strict seed-11 DEV improvement over raw baseline40",
                        "cost: no quality winner"])
    else:
        permitted_confirms = [f"{selected}:seed33", f"{selected}:seed55"]
        if winner["epochs"] == 80:
            skipped.append("cost: class_inner48 is excluded for an 80-epoch winner")
    confirmations = list(confirms)
    mean = None
    if type(cost_seed_seconds) is not int or cost_seed_seconds != 5400:
        raise ValueError("cost stage seed reserve is fixed at 5400 MAIN seconds")
    if baseline40_mean is not None and (type(baseline40_mean) not in (int, float) or
                                         not math.isfinite(baseline40_mean) or not 0 <= baseline40_mean <= 1):
        raise ValueError("matched baseline40 mean DEV AP must be a finite fraction")
    if required_margin_seconds is not None and (type(required_margin_seconds) not in (int, float) or
                                                 not math.isfinite(required_margin_seconds) or required_margin_seconds < 0):
        raise ValueError("parent required margin must be a finite nonnegative number")
    if fresh_main_seconds is not None and (type(fresh_main_seconds) not in (int, float) or
                                           not math.isfinite(fresh_main_seconds) or fresh_main_seconds < 0):
        raise ValueError("fresh MAIN seconds must be finite and nonnegative")
    if selected is None and confirmations:
        raise ValueError("confirmation runs are prohibited when seed11 does not improve baseline40")
    if selected is not None:
        allowed = set(permitted_confirms)
        for row in confirmations:
            key = f"{row.get('variant')}:seed{row.get('seed')}"
            if (key not in allowed or row.get("closed") is not True or
                    type(row.get("seed")) is not int or row.get("epochs") != winner["epochs"] or
                    row.get("head_spec") != winner["head_spec"] or
                    canonicalize_head_spec(row.get("head_spec")) != winner["head_spec"] or
                    type(row.get("dev_ap")) not in (int, float) or not math.isfinite(row["dev_ap"]) or
                    not 0 <= row["dev_ap"] <= 1):
                raise ValueError("confirmation is outside the single-winner seed33/55 matrix")
        if len({f"{x.get('variant')}:seed{x.get('seed')}" for x in confirmations}) != len(confirmations):
            raise ValueError("duplicate confirmation")
        if winner["epochs"] == 80:
            pass  # Confirmation remains allowed, but the 40-only cost stage never does.
        elif len(confirmations) < 2:
            skipped.append("cost: requires closed winner seeds 11, 33, and 55")
        else:
            values = [winner["dev_ap"]] + [float(row["dev_ap"]) for row in confirmations]
            mean = sum(values) / 3
            cost_ok = (baseline40_mean is not None and mean > baseline40_mean and
                       type(fresh_main_seconds) in (int, float) and type(required_margin_seconds) in (int, float) and
                       fresh_main_seconds >= cost_seed_seconds + required_margin_seconds)
            if not cost_ok:
                skipped.append("cost: three-seed DEV mean or fresh MAIN budget plus explicit margin is insufficient")
    else:
        mean = None
    cost = selected is not None and winner["epochs"] == 40 and len(confirmations) == 2 and mean is not None and \
        baseline40_mean is not None and mean > baseline40_mean and type(fresh_main_seconds) in (int, float) and \
        type(required_margin_seconds) in (int, float) and fresh_main_seconds >= cost_seed_seconds + required_margin_seconds
    if selected is not None and winner["epochs"] == 40 and not cost and not any(x.startswith("cost:") for x in skipped):
        skipped.append("cost: stage not admitted")
    return {"winner": winner["variant"], "winner_dose": winner["epochs"], "winner_dev_ap": winner["dev_ap"],
            "improves_baseline40_seed11": improves, "selected_variant": selected,
            "permitted_confirmations": permitted_confirms, "confirmations": confirmations,
            "confirmation_mean_dev_ap": mean, "cost_admitted": bool(cost), "skips": skipped,
            "screen_variants": [row["variant"] for row in screens]}


def render_dev_report(study, decision):
    """Render a candid English-only development report from parent-supplied observations."""
    if not isinstance(study, dict) or not isinstance(decision, dict):
        raise ValueError("report inputs must be objects")
    screens = study.get("screens", [])
    if len(screens) != 5 or any(row.get("closed") is not True for row in screens):
        raise ValueError("report requires all five closed screens")
    confirmations = list(decision.get("confirmations", []))
    complete_followups = ({row.get("seed") for row in confirmations} == {33, 55} and
                          decision.get("selected_variant") is not None)
    if complete_followups:
        study_qualification = ("This was an adaptive, exploratory n=3 candidate study. Seed 11 selected the candidate; "
                              "seeds 33 and 55 are same-study follow-ups, not independent confirmation.")
    else:
        study_qualification = (f"This report contains the five-arm seed11 adaptive screen and {len(confirmations)} "
                               "closed winner follow-up(s); it does not contain a complete n=3 candidate group.")
    lines = ["# AB2 refinement — development-only report", "", study_qualification,
             "No significance or held-out improvement claim is made. COCO evaluation was not performed. "
             "All candidates used one frozen trunk; DEV data were reused for adaptive selection.",
             "", f"Top seed11 screen result: {decision['winner']} at {decision['winner_dose']} epochs "
             f"(raw DEV AP50:95 {decision['winner_dev_ap']:.8f}); admitted candidate: "
             f"{decision.get('selected_variant') or 'none'}.",
             "The 80-epoch baseline is a distinct dose family and cannot isolate an architecture effect from training duration.", "",
             "## Seed-level development and probe results"]
    for row in screens + list(decision.get("confirmations", [])):
        probe = row.get("probe_ap")
        lines.append(f"- {row['variant']} / seed {row['seed']} / {row['epochs']} epochs: DEV AP50:95 {row['dev_ap']:.8f}; PROBE AP50:95 {probe if probe is not None else 'not reported'}; selected epoch {row.get('selected_epoch', 'unknown')}; training SHA256 {row.get('training', {}).get('sha256', 'not supplied')}.")
    lines.extend(["", "## Selection and stages"])
    lines.append(f"Strict improvement over raw seed-11 baseline40 DEV AP {BASELINE11_AP:.17f} was " + ("observed." if decision["improves_baseline40_seed11"] else "not observed; no adaptive follow-ups were admitted."))
    lines.append("The admitted candidate is a single adaptively chosen trunk variant; this does not show that sharing caused a gain." if decision.get("selected_variant") is not None else "No candidate was admitted by the strict-improvement rule; this screen cannot establish a sharing benefit.")
    lines.append("Cost-stage admission: " + ("admitted under the supplied fresh-MAIN budget and explicit margin." if decision["cost_admitted"] else "skipped or not admitted; cost results are not combined with the quality comparison."))
    lines.append("Skips: " + ("; ".join(decision.get("skips", [])) if decision.get("skips") else "none"))
    groups = {}
    for row in screens + list(decision.get("confirmations", [])):
        groups.setdefault((row["variant"], row["epochs"]), []).append(float(row["dev_ap"]))
    lines.extend(["", "## Available-group DEV summary (population standard deviation)"])
    for (variant, dose), values in sorted(groups.items()):
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        lines.append(f"- {variant} / {dose} epochs / n={len(values)}: mean {mean:.8f}, SD {sd:.8f}.")
    lines.extend(["", "## Per-seed training curves and deltas"])
    for row in screens + list(decision.get("confirmations", [])):
        history = row.get("history", [])
        compact = [{"epoch": item.get("epoch"), "dev_ap": _metric(history, index + 1, "dev", "AP50:95"),
                    "probe_ap": _metric(history, index + 1, "probe", "AP50:95"),
                    "dev_delta_from_matched_baseline": (
                        _metric(history, index + 1, "dev", "AP50:95") -
                        (BASELINE11_AP if row.get("seed") == 11 else row["matched_baseline_dev_ap"])
                        if isinstance(_metric(history, index + 1, "dev", "AP50:95"), (int, float)) and
                           (row.get("seed") == 11 or isinstance(row.get("matched_baseline_dev_ap"), (int, float)))
                        else None)}
                   for index, item in enumerate(history)]
        lines.append(f"- {row['variant']} / seed {row['seed']} curve: " + json.dumps(compact, allow_nan=False))
    if "cost" in study:
        lines.extend(["", "## Separate exploratory class_inner48 cost-stage result",
                      "This seed11 efficiency check is reported separately and is not combined with the quality comparison.",
                      "```json\n" + json.dumps(study["cost"], sort_keys=True, indent=2, allow_nan=False) + "\n```"])
    for section, title in (("balances", "Actual ledger balances"), ("throughput", "Actual throughput"),
                           ("diagnostics", "Read-only update diagnostics"), ("profiles", "Raw profile references"),
                           ("source_refs", "Source/configuration/checkpoint references")):
        lines.extend(["", "## " + title])
        value = study.get(section, "not supplied")
        lines.append("```json\n" + json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n```")
    return "\n".join(lines) + "\n"
