# AB2 refinement private operator notes

Parent-only API guide for the development study. This module is bookkeeping, not an experiment runner. It imports no Torch/native runtime and does not freeze, launch, train, profile, allocate, inspect a ledger, or write output. Parent owns every lifecycle action and all fresh output creation.

## Scope and provenance

Use only the fresh namespace `ROOT/ab2-refinement-20261018T041300Z` and the SAME sole allocator ledger already used by the parent. Do not reset/refund, open another Goal/session/board/service, reuse or overwrite a result namespace, publish, or run any operation from this manual in the builder. Parent must stop its own processes at completion and report the final count and wall time.

Read source evidence explicitly from ROOT and verify exact bytes before use. The immutable reference identities are `.internal/ab2-reference-identity.json` (source protocol path/file SHA256 `25ae058950b690e89659ed80f6175d3d4f78409f00ce1735535197ecb6905a4b` and canonical recipe SHA256 `11ad424d07404fd1687c897fb1db51ffc1287a331b2022b73854f14be752e3a2`) and `.internal/ab2-reference-frozen.json` (selected freeze SHA256 `df3d874822a1c8015c312ac6f0ce8c59b4d64c1494c6ca55f5ddd70c5d7c2a47`). Historical G1 baseline40 seed11 raw DEV AP50:95 is `0.14701567894683693`, not rounded AP points. The old 40-epoch baseline records remain in their old identity and are not given new-study fields. The baseline80 record is new-study format 1, carries its full recipe and target dose 80, and is not relabeled as the old baseline.

## APIs and examples

Import the private source with ROOT on `sys.path` and `src` on `PYTHONPATH`. `prepare_recipe(old_recipe, *, head_spec, epochs, source_protocol, source_freeze, producer, cache_evidence, pilot_evidence, environment_updates=None)` deep-copies the common recipe/environment, preserving runtime and numeric provenance. It removes only obsolete `looped_architecture` and `reference_native`, then replaces study identity/spec/reference and caller-supplied updates. Updates cannot override protected identity keys. It validates with `head_sharing.protocol.validate_recipe` and returns an unfrozen object. `head_spec` must be canonical; baseline uses `None`. Caller supplies actual provenance/evidence.

```python
recipe = prepare_recipe(
    old_recipe, head_spec=canonical_spec, epochs=40,
    source_protocol=verified_protocol_binding, source_freeze=verified_freeze_binding,
    producer=actual_producer, cache_evidence=actual_cache_evidence,
    pilot_evidence=actual_pilot_receipt,
    environment_updates={"source_git_commit": actual_commit, "admission": actual_receipt},
)
# Parent alone freezes the returned recipe into a fresh protocol path.
```

`read_sealed_json(path, expected_sha256)` checks exact file bytes and parses an object. `project_training_record(path, expected_sha256, *, recipe_sha256, head_spec, seed, epochs, selected_config, selected_weights)` validates a real completed `main` run, pinned G1 UUID, frozen trunk state digest, changed head, full contiguous training history and the producer's cadence-only DEV/PROBE evaluations. It selects the earliest best evaluated DEV epoch. It requires the config path to be the record's exact `selected_config` beside `training.json`, and weights to have the same basename with `.pt`; both are hash-checked. No tensors are loaded. `attach_head_conv_macs(row, head_conv_macs)` adds the parent's premeasured CPU-executed **head-only** ConvMAC count (B1, 512, retained dual) to each seed11 screen, including losers, without a profile. The count comes from `profile.measure_conv_macs(model.detect, ...)`/the producer's `head_executed_conv_macs`, not full-network `executed_conv_macs`.

```python
row = project_training_record(
    training_json, training_json_sha256, recipe_sha256=frozen_recipe_sha256,
    head_spec=canonical_spec, seed=11, epochs=40,
    selected_config={"path": selected_json, "sha256": selected_json_sha256},
    selected_weights={"path": selected_pt, "sha256": selected_pt_sha256},
)
row = attach_head_conv_macs(row, cpu_head_conv_macs)
# P8 only: optional measured profile binding for selected baseline/winner.
row = attach_profile(row, conv_macs=cpu_head_conv_macs,
                     profile={"path": profile_json, "sha256": profile_sha256})
```

`attach_profile` accepts only a sealed retained-dual inference profile at B1/imgsz512 whose `head_executed_conv_macs` equals the already attached screen count and whose model/config/recipe identity matches. `decide_stages(screens, *, baseline40_ap=0.14701567894683693, confirms=(), fresh_main_seconds=None, required_margin_seconds=None, baseline40_mean=None)` requires four canonical variant/class_inner96 primary40 seed11 rows plus baseline80 seed11 with `head_spec=None`; it rejects mismatched identity, dose or non-fraction AP. It selects on raw DEV AP; ties use head ConvMAC then `[baseline80, learned_alpha, step_norm, scale_gain, dw5]`. Only a strict seed11 improvement admits that same variant/spec/dose at seeds33 and55. It never searches again. Cost class_inner48 is only eligible after a complete 40-epoch winner's 11/33/55 DEV mean beats the matched old G1 baseline40 mean and fresh MAIN covers 5400 seconds plus the parent's explicit required margin. It is always skipped for an 80-epoch winner.

`render_dev_report(study, decision)` returns English Markdown only, qualifies n based on actual closed follow-ups, explicitly discloses one frozen trunk and reused DEV selection, and includes a separate exploratory cost section when `study["cost"]` is supplied. It does not call a no-follow-up screen an n=3 study. No held-out/COCO improvement, independent confirmation, significance, or causal-sharing claim. Include actual negative outcomes and skips.

## Actual producer APIs / diagnostic discipline

- Identity: `head_sharing.head_spec.canonicalize_head_spec` and `head_spec_digest`.
- Recipe ABI: `head_sharing.protocol.validate_recipe`; parent alone calls `freeze` / `verify_frozen`.
- Main run: existing `head_sharing.train.experiment(purpose="main", arm="loop_ab2", seed=..., recipe=..., protocol_sha256=..., cache_paths=..., manifest_path=..., annotations_path=..., trunk_artifact=..., receipt=..., output=..., device=...)`. Main identity is frozen in recipe; do not override `head_spec`.
- Selected checkpoint integrity: existing `head_sharing.checkpoint` load/save/resume APIs owned by producer; this helper verifies JSON/file hashes and does not load tensors.
- Profile: existing `head_sharing.profile` API and its raw `model_identity` bindings. `executed_conv_macs` is the full-network count; selection uses `head_executed_conv_macs.conv_macs` from the retained-dual B1/imgsz512 head invocation. CPU-executed head MAC maps are available to all five screens before P8; raw GPU profiles are optional for the selected baseline/winner only.
- Read-only diagnostics: `head_sharing.model.diagnose_ab2_updates(model, features)`. Parent loads actual selected checkpoint through existing checkpoint/profile APIs and obtains a fixed full TRAIN batch from the actual `few` cache. Collect diagnostics before/after selection as appropriate. Do not collect on DEV/PROBE/COCO or add work inside the training loop. The function returns four towers, each with two update entries containing `h_l2`, `alphaF_l2`, zero-safe `ratio`, and `alpha`. Preserve raw values and checkpoint/profile/cache hashes.

The `None` spec for baseline80 remains a valid new-study format-1 checkpoint: the full frozen recipe and `target_epochs=80` distinguish it from old baseline40. Old baseline records/profiles retain original recipe identity. Profile identity must include canonical head spec/digest/variant for all new-study profiles, including baseline80.

## Failure and stop behavior

Invalid or missing JSON/hash, incomplete history, mismatched spec/seed/dose/source, missing actual ConvMAC, or profile artifact mismatch raises `ValueError`: stop that projection and reconcile producer evidence; do not guess a field or fabricate proof. A legitimate non-improvement is a completed negative screen with named follow-up SKIPs, not a failure. Parent retains sole ownership of actual freeze, training, sealing, diagnostics, profiling, namespace writes and ledger/process cleanup. No real inputs/cache/books/models are read by these synthetic unit tests.
