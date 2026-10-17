# Looped head study protocol

## Scope and recipe

The new ordered matrix is `native`, `loop_ab2`, `loop_a3` over seeds
11, 22, 33, 44, 55: ten new fits plus five reused native controls.
`LOOPED_ARMS` contains only the two new model arms; `LOOPED_MATRIX` adds the
native reference. Historical `native/independent_k2/shared_k2` recipes remain
valid and retain their original protocol identity.

A looped recipe retains the existing top-level keys and declares
`environment.study_id="looped_v1"`, the fixed
`environment.looped_architecture`, and portable relative source bindings in
`environment.reference_native`. The protocol source object is exactly
`{path, sha256, recipe_sha256}`; the freeze source object is exactly
`{path, sha256}`. Paths are normalized relative POSIX paths, resolved under
the marker-owned study root. Production pins source recipe SHA
`3db46e027e40eefde310b3dc35e5c0318e2613993f153ff7dfcd8533e37c9ff0`, source
protocol file SHA
`ffdff515abf7d92d6ffb7019764ffb7297b6bdb89519065c4b5c186929801955`, and
source freeze file SHA
`70d7deb42d9e23d9efa6a3215f4e06841a3c336af860d208acb17d2152d359b3`.
Production also enforces 512px, batch/block 32, 40 epochs, cadence 5, probe
1000 and the original seed-to-GPU pairing. Inherited scientific fields are
checked against the referenced source recipe. Synthetic fixtures use their
own generated source bindings and may reduce dose/geometry while remaining
explicitly synthetic.

The architectures use box/class widths 24/96 with two residual updates
(`loop_ab2`) and 28/108 with three (`loop_a3`). Both use one depthwise 3x3
then pointwise 1x1 adapter stage, channel-only per-point LayerNorm, SiLU,
residual alpha 0.1, learned per-scale channel IDs and native per-scale readout
priors. The trunk, quantization boundary, optimizer, losses, data order,
caches, selection and evaluation geometry remain unchanged. Reused AP is not
an independent untouched confirmation benchmark.

## Cache I/O and execution-source provenance

Batch reads use buffered, contiguous reads into owned FP16 arrays before the
unchanged FP16-to-FP32 conversion. Read-only whole-file maps remain available
for explicit diagnostics, but batch iteration does not touch their payloads.
This prevents training from keeping the entire two-view cache resident in each
process working set; it does not change the cache bytes, image order, block
flips, targets, quantization boundary or model arithmetic.

The first two new fits completed all 40 epochs with source `a09b8e4`; subsequent
fits use the bounded reader introduced in `91209cf`. Neither initial fit was
interrupted for this change. CPU checks establish exact feature, native-loss,
gradient and SGD-update equivalence for both loop arms. A separate disposable
Windows observation confirms reduced mapped residency; it is not an accuracy
or training-throughput benchmark. Execution-source commits are recorded per fit
rather than attributing every fit to the final evaluator commit. The frozen
scientific recipe and cumulative resource limits remain unchanged.

## Freeze serialization and validation

Legacy freeze schema version 1 remains unchanged. Its top-level fields are
`schema_version`, `protocol_sha256`, `protocol`, `trunk_sha256`,
`split_sha256`, `imgsz`, `batch_size`, `quantization`, `producer`,
`synthetic`, `selection`, `policy`, and `members`. Each member contains
`seed`, `arm`, `selected_epoch`, `selected_dev_ap`, `config`, `weights`,
`config_sha256`, `weights_sha256`, `run_record_sha256`, `resume_sha256`, and
`trunk_sha256`.

A mixed freeze uses schema version 2 and the same common fields and 15-member
ordered matrix, plus top-level `reference_native` exactly equal to the recipe
reference declaration. Every member adds `source`:

- New fits: `{"kind":"current","protocol_sha256":"<new recipe SHA>"}`.
- Reused controls: `{"kind":"reused_native","protocol_sha256":"<old recipe SHA>","freeze_sha256":"<original freeze file SHA>"}`.

The new freeze does not embed its own file SHA (which would be circular).
The reused native member rows point into the portable bundle and preserve the
original selected config, weights, training record and resume bytes. The
source freeze still declares its historical 15 rows, but only its native five
artifact closures are required; nonnative historical artifacts are not
imported. Freeze creation validates all ten new complete runs and all five
native runs against their respective recipes, including authoritative tensor
resume/history, cadence and earliest dev-best selection. Subsequent verification
rechecks the exact frozen config, weights, training and resume byte hashes and
JSON identities without deserializing tensors or importing the inference stack.
The complete 15-member freeze is verified before any holdout annotations, IDs,
or images are opened.

`member_source_binding(frozen, member, current_freeze_file_sha)` is a
stdlib-only helper suitable for analysis imports. For legacy/current rows it
returns the current protocol and freeze hashes; for reused native it returns
the source protocol and source freeze hashes. It imports no torch or model
code. Prediction of a reused native row is rejected: retain its original
prediction/receipt/score rather than minting a new inference identity. New-arm
predictions bind to the current freeze; score replay resolves each member's
source identity and preserves original native receipts and scores. Cohort,
annotation/image bytes, selected config/weights, numerical policy, and
assigned execution device remain checked.

## API and CLI

Producer APIs are `freeze_selected(protocol_path, run_dirs, output,
owned_root)`, `verify_freeze(freeze_path, owned_root)`, `predict(...)`,
`score_prediction(...)`, and the stdlib-only source helper above. The existing
CLI remains the source of truth and takes references from the frozen recipe:

- `freeze-selected --protocol ... --owned-root ... --output ... --run ...`
  (repeat `--run`; ten new directories for looped studies, fifteen for legacy).
- `predict --freeze ... --owned-root ... --annotations ... --image-root ...
  --image-ids ... --trunk-artifact ... --receipt ... --output ... --seed ...
  --arm ... [--device ...] [--synthetic]`.
- `score --freeze ... --owned-root ... --annotations ... --predictions ...`.

No additional reference flags or sidecar source of truth are needed. Analysis
consumers must use member source bindings, not infer provenance from the
matrix-level current protocol. Do not report results until actual new fits,
freeze, predictions, scoring and analysis exist; synthetic tests establish
plumbing only, not scientific results.
