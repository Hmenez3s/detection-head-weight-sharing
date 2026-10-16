# Allocation runtime

This runtime is an operator utility, not a service. It never discovers devices, selects a fallback, launches a scheduler, or initializes CUDA. The two permitted physical UUIDs and per-device 57,600-second ceiling are fixed in `head_sharing.runtime`. Main reservations are capped at 46,800 seconds, preserving 7,200 seconds for final work and 3,600 for contingency; each reservation is charged in full when granted, including failed or killed jobs. Balances are per UUID and phase: no borrowing, resetting, or automatic stale-job reclaim. A job reservation excludes another job on that physical UUID. The aggregate logical-CPU ceiling is 12; default reservation is six CPUs/job.

## Initialize exactly once

The operator first creates/owns the remote study workspace and its `STUDY_OWNER.txt` marker. From that already-owned root, initialize a **new, unique child directory** for each fresh ledger workspace:

```sh
python -m head_sharing.runtime init --owned-study-root "$STUDY_ROOT" --subroot "ledger-<unique-run-name>"
```

`STUDY_ROOT` must already be a real directory, contain `STUDY_OWNER.txt`, and have no symbolic-link, junction, or reparse-point path component. The requested child must not exist. Initialization creates its private ledger-owner marker and initial ledger; it does not clear, replace, or migrate anything. Never point initialization at an existing ledger subroot. Once initialized, persist and reuse that exact explicit `LEDGER_ROOT`; `status` and `reserve` do not initialize or reset state. Writers retry ordinary lock contention for at most two seconds, waiting for the current lock owner to finish; they never delete or reclaim its lock. A lock still present after that bounded wait is ambiguous/stale and blocks writes until the operator establishes ownership and resolves it with evidence. Do not delete locks or edit balances based on guesswork.

```sh
python -m head_sharing.runtime status --root "$LEDGER_ROOT"
python -m head_sharing.runtime reserve --root "$LEDGER_ROOT" --uuid GPU-13916c7e-aa3c-fa5e-48ef-7b8cca10ea99 --phase main --seconds 3600 --cpus 6
```

The reservation token is the supervisor's required `--reservation` value. Paths/host identity must be supplied by the operator and are not emitted in public diagnostics. The runtime API must reserve before any CUDA/native import and sets `CUDA_VISIBLE_DEVICES` to exactly the assigned UUID. `child_entry(...)` waits for the containment gate and validates reservation state plus that exact environment value before native imports; after CUDA initialization, query and pass the actual physical UUID to `validate_device(...)` before doing work. Device validation never accepts an index or CPU fallback. Integration with a CUDA library's UUID query is an operator-owned adapter and remains unverified by CPU tests here.

## Bounded supervisor and detached operator launch

The CLI example below runs the supervisor in the foreground; it is not SSH-disconnect-safe by itself. On Windows, the operator must launch the supervisor process itself outside the SSH job using the already-proven detached `Popen` flags (`CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS | CREATE_BREAKAWAY_FROM_JOB`), `stdin=DEVNULL`, explicit cwd, and operator-owned log. That detached supervisor then launches its own guarded child tree. The builder does not launch it remotely.

The supervisor CLI takes an explicit initialized ledger, reservation, private job working directory beneath the verified owner root, and new log/progress files:

```sh
python -m head_sharing.supervisor --root "$LEDGER_ROOT" --reservation "$TOKEN" --cwd "$JOB_DIR" --log "$JOB_DIR/job.log" --progress "$JOB_DIR/progress.json" --first-progress-seconds 300 --heartbeat-seconds 120 -- python -m your_guarded_child
```

The child must create/update progress only after real work begins and continue heartbeat updates; mere process start is not progress. Supervisor enforces the child deadline eight seconds before the reservation ends, records PID/start identity/start/end/exit/progress and `actual_wall_elapsed_seconds` (including termination/cleanup grace, separately from the conservative full reservation debit), and kills only its own process group/tree. Allocations shorter than nine seconds are rejected before launch. The eight-second safety allowance covers the bounded two-second TERM grace, two-second KILL/reap bound, lock contention allowance, and polling overhead; a failed cleanup stays active/unknown and charged. The job cwd must be a link-/junction-free directory beneath the verified `STUDY_OWNER.txt` root; cwd/log/cache writes outside that ownership root fail before mutation. The launch claim is atomic and only one caller can start a reservation. Children must call `supervisor.child_entry(...)` before importing CUDA/native modules; it waits for the supervisor's containment gate, then validates that the reservation is already launching/running and the actual UUID exactly matches. Logs and progress must be new and confined to the job's own cwd. Before launch, it confines `YOLO_CONFIG_DIR` (including its precreated `Ultralytics` child), `TORCH_HOME`, `MPLCONFIGDIR`, `TMPDIR`, Windows `TEMP`/`TMP`, `CUDA_CACHE_PATH`, `TORCH_EXTENSIONS_DIR`, `TORCHINDUCTOR_CACHE_DIR`, and `TRITON_CACHE_DIR` beneath the owned job cwd, and sets `YOLO_OFFLINE=true`. These are child-process environment settings only; drivers and system settings are untouched. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` are set to the reserved CPU allocation and `HEAD_SHARING_CPU_BUDGET` exposes that allocation; the child divides it between its own torch/decode workers. On Windows the Popen flags are exactly `CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS | CREATE_BREAKAWAY_FROM_JOB`, with `stdin=DEVNULL`, explicit cwd, and owned logs. A documented Windows Job Object uses `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` for descendant cleanup rather than `taskkill`; child code must use the guarded entry before native imports. Those flags are recorded from prior operator evidence; structural CPU tests are not proof of future SSH disconnect or actual hardware execution. No Task Scheduler, shared service, network launcher, or global workaround is provided.

A crashed or interrupted allocation remains fully charged. Review recorded PID identity, process state, exit/progress evidence, and ledger before any operator reconciliation; PID reuse, missing heartbeat, unknown lock owner, UUID mismatch, and inconsistent state are stop conditions. This package does not implement automatic reclaim or balance transfer. CPU tests use fake UUIDs and disposable synthetic children only; they do not establish real UUID querying, Windows survival, VRAM/RAM/disk margins, or CUDA execution.
