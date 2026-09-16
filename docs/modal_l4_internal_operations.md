# Internal replacement programme operations

The user authorized a separate additional **$20** on 2026-09-11 for the internal replacement plan. Previous `results/modal-l4-runner` is closed and must never be reset, resumed, merged into this ledger or overwritten. Its allocation/spending snapshot and hash are recorded in the new manifest.

Use `/Users/w3joe/.local/share/hypercast4d/runtime/bin/python` and the sibling `hypercast` and `modal` commands. The new root is `results/modal-l4-internal`, state `controller/controller_state.json`, logs `controller/controller.log`. Credentials already reside in the configured local Modal CLI environment; do not copy them into the project or logs.

## Start and supervise

The prepared pilot contains 24 fits (three backbones, eight variants), 32/5, seed 907, 5 epochs. The first job executes the full bounded CUDA preflight before fitting native DLinear; all others depend on it. Each subsequent pilot job has a 300-second timeout; the first allows 900 seconds for the numerical gate. No automatic blind retries: the AI diagnoses failures, records their cost and authorizes a bounded repair within this same allocation.

The LaunchAgent is `com.hypercast4d.modal-l4-internal-controller` under `~/Library/LaunchAgents`. It runs `caffeinate -is` and the existing generic controller CLI with the new results root, `--watch --poll-seconds 45`. It restarts after unexpected failure, exits normally when a stage ends, and keeps the Mac awake while running. Keep the Mac powered, lid open and network connected.

```sh
/Users/w3joe/.local/share/hypercast4d/runtime/bin/python scripts/run_modal_l4_controller.py --results-root results/modal-l4-internal --watch --poll-seconds 45
launchctl print gui/$(id -u)/com.hypercast4d.modal-l4-internal-controller
```

Do not run the manual command concurrently with the service. Before restarting, verify actual process arguments and cloud tasks, not just a PID file. The controller lock prevents duplicates within the root; its global Modal app check prevents overlapping experiment GPU jobs. Leave the old programme's service stopped.

The existing Codex heartbeat `modal-l4-pilot-monitor` is repointed to this programme every 20 minutes. It is responsible for intelligent stage transitions, interpreting results, repairing operational issues and deciding/documenting informative follow-ups. It must inspect actual processes and artifacts, not report a heartbeat as a continuously executing AI process. The heartbeat tool provides no per-run model override; do not claim Spark is selected without verifying a supported setting.

## Current priority: remaining twelve backbones

The original three-model development completed all 48 fits. Its previously planned robustness stage is superseded by the user's request to cover the other twelve TSLib backbones without repeating these three. **Do not use the old `--advance` command after development.**

The new versioned protocol is `remaining-internal-v1`. It covers eleven ordinary dense/pointwise sites and a separately labelled FiLM spectral intervention. Details and parameter-budget exceptions are in `docs/modal_l4_remaining_models_plan.md`.

```sh
OMP_NUM_THREADS=2 /Users/w3joe/.local/share/hypercast4d/runtime/bin/python scripts/prepare_modal_l4_remaining.py --verify
/Users/w3joe/.local/share/hypercast4d/runtime/bin/python scripts/prepare_modal_l4_remaining.py --prepare
launchctl kickstart gui/$(id -u)/com.hypercast4d.modal-l4-internal-controller
```

Preparation appends only twelve native calibrations, with a 600-second timeout each and all eight variant GPU gates before each native fit. Dependency chaining stops the stage after a failed calibration for AI diagnosis. The remaining 84 variant fits require a separate complete-stage cost decision after reviewing these calibration timings and numerical reports. Keep the same cumulative ledger. Do not interpret five-epoch scores as effectiveness evidence or use them to drop controls.

The supervisor must verify all twelve calibration reports, actual NVIDIA L4 runtime records, paired initialization audits, nested split/scaler records and finite curves. Estimate every one of the remaining 84 fits, including variant overhead, queue/CPU/memory costs, 3× uncertainty and the $1 reserve. If the full matrix is unaffordable, record and disclose the limitation before collecting an unbalanced subset. The current preparation script intentionally stops at calibration; it cannot silently append the old robustness stage.

## Historical advance procedure (superseded after development)

```sh
/Users/w3joe/.local/share/hypercast4d/runtime/bin/python scripts/prepare_modal_l4_internal.py --advance
launchctl kickstart gui/$(id -u)/com.hypercast4d.modal-l4-internal-controller
```

The advance command requires every scheduled job complete, no active cloud job, valid initialization/split artifacts, matching frozen inputs and fresh billing. It preserves the entire new-study cost ledger and source history. It appends a frozen decision record before proceeding.

Pilot → development: 24 arms × two rates (0.0003, 0.001), seed 701, max 150 epochs; no arm selection from pilot accuracy. Development → robustness: freeze each arm's outer development MAE/persistence choice, lower rate for relative ties within 0.1%; all 24 arms × three chronological folds × seeds 401/503/601. Reduce only the number of robustness seeds, symmetrically, if the measured conservative estimate plus queue margin cannot fit. A budget rejection is a stop for review, not permission to raise the cap. After robustness the preparation command stops: the AI must analyze paired results and preregister any further placements, sensitivity, or random-tying controls before launching them.

There is no independent final stage on the already-seen Copper tail. The worker explicitly rejects the old final-test endpoint for this initialization/stopping protocol. New independent data and a separately validated final protocol are required. These historical Copper results are exploratory.

## Spending and provenance

Cap $20; keep $1 safety reserve. Per-job admission reserves its full timeout plus 120 seconds, prices L4 + two CPU cores + 4 GiB memory at refreshed rates, applies a 3× multiplier and startup allowance. The ledger charges all attempts, including failed/interrupted jobs. Accounted spend is the larger of conservative ledger or monotonic metered increase from this allocation's fresh baseline. Monthly billed costs may lag or be offset by credits; never use credits to enlarge the allocation. The cap is an application guard, not a provider-enforced billing limit.

CPU numerical report: `controller/cpu-preflight.json`. GPU numerical report: first job's `preflight.json`. Every fit returns `initialization.json`, `split_audit.json`, `learning_curves.csv`, `runs.csv`, predictions and runtime/package/GPU metadata. Frozen source, data, specs, plan and operations are archived in `controller/provenance/source-data-specs.tar.gz`, with SHA-256 manifest. Never change frozen source while jobs run; any necessary repair requires stopping/reconciling activity and versioning its provenance first.
