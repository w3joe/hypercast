# Modal L4 operations

Repaired 10 September 2026. The 27-configuration pilot is complete. The next stage is **controlled development**, specified in `modal_l4_controlled_stage.md`: CUDA preflight, then all 27 configurations at two learning rates for 150 epochs each. This is still development, not confirmatory evidence. Thirteen unique configurations had completed before repair, with one additional duplicate TSMixer complex job. Preserve all 14 job directories; the first successful matching job is the primary pilot artifact. Historical runs were not frozen before launch.

## Runtime and service

Durable Python/CLI environment: `/Users/w3joe/.local/share/hypercast4d/runtime/bin`.

Controller: `scripts/run_modal_l4_controller.py`.
State and log: `results/modal-l4-runner/controller/controller_state.json` and `controller.log`.
Backend discovery and log: `results/modal-l4-runner/.runtime/`.
LaunchAgent label: `com.hypercast4d.modal-l4-controller`.
The service runs the controller under `caffeinate -is`, polls every 45 seconds, and restarts after unsuccessful exits. Successful completion stops the process. Keep the Mac powered, lid open, and Codex open for AI supervision.

Check the service:

```sh
launchctl print gui/501/com.hypercast4d.modal-l4-controller
```

Restart a stopped service (do not use `-k` to kill a healthy process):

```sh
launchctl kickstart gui/501/com.hypercast4d.modal-l4-controller
```

Repair/reconcile state without submitting, **only while the controller is stopped**:

```sh
/Users/w3joe/.local/share/hypercast4d/runtime/bin/python scripts/run_modal_l4_controller.py --results-root results/modal-l4-runner --reconcile-only
```

The file lock prevents duplicate controllers. The PID file alone is not proof of liveness. The controller checks every job in its workspace and the Modal app list before a new submission. Missing submission records block scheduling until investigated. A crash after POST is recovered through normalized scientific configuration matching. Completed jobs are never retried merely because an old manifest hash differed.

## Budget and recovery

The programme has one cumulative **$20 total limit** and **$1 safety reserve**. Never reset the original billing baseline or cost ledger for a new stage. The ledger includes duplicate and failed jobs. Spending used for admission is the larger of cumulative metered billing since the original baseline and conservative per-attempt estimates. Estimates use full job wall time plus startup allowance, with a 3x factor; they are not invoices. Reserve each candidate’s entire remote timeout plus 120 seconds overhead before submission (pilot: one hour; controlled preflight: 600 seconds; controlled development: 900 seconds). Modal enforces the timeout even when the Mac is unavailable, and automatic Modal retries are disabled. Billing/cloud query failures pause submissions and retry on the next cycle. Local interruptions/cancellations retain conservative costs; verify remote shutdown before retrying.

Only one L4 may be active. Do not manually fan out jobs. Runtime and budget cancellation requests are verified with subsequent job/cloud checks before another submission. Unrelated Modal applications must not be stopped by this supervisor.

Credentials remain in the user's Modal CLI configuration. Do not print or copy token values.

## AI supervision and subsequent stages

The existing `modal-l4-pilot-monitor` heartbeat checks every 20 minutes. Its prompt authorizes repair/restart and active scientific decisions within the existing user authorization, rather than merely printing restart instructions. Heartbeats use the target thread's model; the automation tool has no per-heartbeat model override. The user requested GPT-5.3-Codex-Spark; select that model for the supervisor thread in Codex if it is not already selected. Do not claim Spark is configured based only on prompt text.

After the pilot, the Python controller exits successfully and records `pilot_complete`; it does **not** silently promote pilot results to robust confirmation. The AI supervisor must:

1. Verify the full balanced pilot, failure modes, CUDA device, artifacts, timings and actual budget reconciliation.
2. Implement and test the preparation in the main plan: matched effective initialization, paired shared-module RNG/state, CUDA timing/memory instrumentation, and source/data/dependency/spec snapshots.
3. Estimate an affordable balanced tuning and validation matrix from measured runtime. Preserve real dense, original, lift-only, low-rank controls and all 2D/4D/8D families across the three backbones. Document any reduction from the proposed 9,450 fits before execution.
4. Extend the resumable manifest/controller while preserving all prior costs and provenance, then start the next stage. Do not ask for new authorization unless the $20 limit must increase or the task scope materially changes.
5. After required comparisons, choose follow-ups with written hypotheses, controls, cost estimates and stopping rules. Avoid test-driven tuning. Stop when the budget cannot admit another safe job or the scientific objectives are addressed, and report inconclusive results honestly.

Source, dataset checksums and dependency versions for the repaired continuation are stored under `results/modal-l4-runner/controller/repair-provenance/`. Earlier artifacts must not be represented as having this provenance retrospectively.

The controlled-stage manifest appends candidates to the original state rather than replacing it. Controlled source/data/spec hashes are verified before every new submission; a changed frozen input pauses the queue for review. Check `initialization.json`, `learning_curves.csv`, `runtime.json` and `preflight.json` alongside normal job artifacts.

After controlled development, use `python scripts/report_modal_l4_development.py` for a read-only summary. Its `--freeze` option records learning-rate choices only after all 54 arms and controller accounting are complete. It also estimates the provisional 162-fit robust follow-up; review the scientific protocol and remaining budget before appending that stage.

## Controlled robustness continuation

Development completed all 54 arms at 23:06 UTC on 10 September. Rates are frozen in `controller/development_selection.json`; full development analysis is in `controller/development_analysis.md`. The next authorized stage is specified in `docs/modal_l4_robustness_stage.md`: all 27 configurations, window 20/horizon 5, seeds 101/211 and three robust folds, 150 epochs, six fits per job. The final test remains closed. Use `scripts/prepare_modal_l4_robustness.py` once while the service is stopped to append the stage; then restart the existing service with the documented kickstart command. Do not rerun preparation if robustness candidates already exist.

Each robustness job has a 1,200-second remote timeout and a full-timeout-plus-120-second reservation (approximately $1.07). All original cumulative accounting remains. The unchanged controller supports candidate-specific evaluations and validates six result rows per completed robustness job. `controlled-robustness_complete` means all candidates have reconciled; then review all 162 fits, convergence, paired effects and actual remaining budget before choosing the next stage. Saved artifacts should have 900 learning-curve rows and six initialization records per job. New stage provenance is in `controller/robustness-provenance/`, extending the unchanged training-source snapshot and hashes from controlled development.

## Locked final continuation

Robustness completed all 162 fits at 01:23 UTC on 11 September. Full paired analysis and figures are under `controller/robustness-analysis/`. The next frozen stage is `locked-final`, documented in `docs/modal_l4_final_stage.md`: all 27 parents, both seeds 101/211, 54 fits at 150 epochs on the first 85%, evaluated once on the final 15%. Treat this as a locked retrospective comparison. Do not use partial test results to modify the remaining jobs or tune follow-ups.

`scripts/prepare_modal_l4_final.py` prepares the stage once; subsequent heartbeats must resume the existing service rather than prepare duplicates. Each final job has a 300-second remote timeout and approximately $0.37431 reservation including 120 seconds overhead. All prior charges remain, final automatic retries are disabled, and the normal cumulative $20 guard with $1 reserve remains authoritative. Expected stage cost is about $4.00; overruns may leave the balanced matrix incomplete and must be reported as such.

The controller now distinguishes final and validation hashes and expects two result rows per completed final job. Final requests reference exact robustness parents and pass only a timeout override, preserving the test-once guard and inherited scientific settings. Check two fold-0 initialization records and 300 learning-curve rows per final job. The epoch `validation_loss` field is computed on the training prefix during final refitting; it is not test loss. The idle backend was restarted for the API timeout support before launch. Versioned source and previous hashes are in `controller/final-provenance/`.

On `locked-final_complete`, validate all 54 fits and prepare the final report using the preregistered analysis, keeping robustness and final scores separate. The robustness analysis script requires its earlier completed state and should not be rerun against final-stage state or overwrite frozen robustness artifacts. On a budget stop or failed final job, retain every result and report incompleteness; do not discard a losing family or override the safety reserve to finish. No new recipe may be tuned on test results.

## Programme stopped at budget guard — 11 September 2026

The service exited cleanly at 03:02 UTC with 50/54 final fits complete. iTransformer rank-4 and rank-2 final candidates remain `blocked_by_budget`; neither was submitted. All 162 robustness fits and all 54 development fits are complete. There were no launched programme training failures. Conservative cumulative accounting is $18.793498; the $1 reserve leaves $0.206502 available, below the $0.37431 full final-job reservation. Do not lower the reserve, reset estimates, change timeouts or resubmit under new hashes to bypass this stop.

The final report is `docs/modal_l4_results.md`; available final results and missing contrasts are in `controller/final-analysis/`. The controller state retains `needs_review` and the original per-candidate statuses, with a separate `programme_status=stopped_budget` annotation. The recurring monitor is paused after final reporting; the LaunchAgent is installed but idle. No restart is necessary under the existing authorization. Preserve all data and source snapshots for any user-requested review or separately authorized future programme. The completed-stage analysis is conditional and incomplete at final testing, not a claim of universal hypercomplex superiority or inferiority.
