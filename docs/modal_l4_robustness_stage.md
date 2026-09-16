# Controlled robustness — 11 September 2026 (London)

Development completed all 54 arms without failures at 23:06 UTC on 10 September. Every arm has 150 finite epoch records and complete metrics; shared-module initialization hashes agree and the 90 frozen inputs are unchanged. The programme's conservative cumulative ledger is $8.005151, with $1 retained as safety reserve. Modal metering is delayed and approximately $0.89 above the original baseline; neither figure is an invoice.

## Scientific decision

Continue the previously proposed balanced seed/period robustness stage. Hypothesis: any frontend benefit or degradation observed during development should persist across training seeds and chronological regimes. This distinguishes an isolated optimization result from a consistent result on this dataset. All nine variants of DLinear, TSMixer and iTransformer remain, including original, lift-only, real dense, rank-8/4/2 and complex/quaternion/octonion.

At their individually selected rates, all nine hypercomplex configurations still lose to their original backbone and persistence on the development period. Some beat added real dense; that does not establish useful forecasting performance. The original models also lose to persistence. Training/validation divergence and sensitivity to the final epoch remain concerns. Keep the fixed 150-epoch rule for this replication; a different checkpoint policy would be a separately documented exploratory protocol with equal treatment of all arms. Do not change it midway or retrospectively select favorable epochs.

The rates in `results/modal-l4-runner/controller/development_selection.json` are frozen using the earlier minimum MAE/persistence rule, with relative differences within 0.1% favoring the lower rate. All 54 completed artifacts were checked before freezing. New seeds assess optimization stability, but these historical validation dates overlap development exposure; this is robustness analysis, not an independent holdout or a universal claim about hypercomplex layers.

## Frozen execution

- All 27 configurations; one job per configuration with six sequential fits.
- Window 20, horizon 5; Copper target and the same four inputs.
- Seeds 101 and 211, shared across every configuration.
- Built-in robust folds: train first 55%, validate 55–65%; train first 65%, validate 65–75%; train first 75%, validate 75–85%. Training-only preprocessing; full targets within each split.
- 150 fixed epochs, batch 32, Adam/MSE, frozen per-configuration rate; no early stopping or best-checkpoint restoration.
- Existing matched-v1 initialization, paired named RNG streams, benchmark instrumentation and pinned Modal image remain unchanged. Every job must save six initialization records, six result rows, 900 epoch records, predictions and runtime metadata.
- One L4 job at a time; 1,200-second remote timeout per six-fit job, no Modal retries. The controller allows at most one additional attempt, with every attempt charged to the cumulative ledger.
- Total: 27 jobs, 162 fits. This stage never invokes final-test. The last 15% stays closed.

## Analysis specification

Primary score is MAE divided by persistence MAE, averaged equally across seeds and folds. Report each fold and backbone separately, paired relative effects against real dense, the corresponding real low-rank map and persistence, and direct dimension comparisons. Original/lift-only are explanatory controls. Report actual MAE/MSE, per-lead errors, parameter counts, synchronized training and inference times, peak GPU allocation, failures and seed dispersion. Neither two seeds nor overlapping forecast origins count as independent datasets.

For temporal uncertainty, average five lead absolute errors at each origin and average seeds before resampling. Use paired circular moving-block bootstrap, 10,000 resamples, RNG seed 20260911; resample identical origin indices jointly for all models within each fold, with folds resampled separately and then equally averaged. Compute relative MAE effects from the resampled errors, not averages of pointwise error ratios. The family comprises 36 contrasts: per backbone three hyper-versus-real, three hyper-versus-low-rank, three dimension pairs and three hyper-versus-persistence. Report pointwise 95% percentile intervals and Bonferroni family-wise 95% percentile intervals (tail probability 0.05/(2*36)); warn that extreme tails have limited Monte Carlo precision. The practical margin is 2%, with a 5% sensitivity threshold. Unadjusted apparent wins do not establish the family-wise benefit criterion.

Block length is 60 origins, with 30/120 sensitivity. This was chosen solely from selected development predictions: for each of the 36 origin-level absolute-loss differences, search lags 5–60 for the first five consecutive absolute biased autocorrelations below 0.1; use 60 when none qualifies, then take the maximum across contrasts. The development period contains 297 origins; the resulting contrast lags ranged from 5 to the cap of 60. Persistent dependence reaches the search cap, so this is a sensitivity-based descriptive interval, not proof that blocks are independent. Report origin counts and origins/block-length for each fold; short effective block counts or sensitivity-dependent results require an inconclusive uncertainty assessment. Seed variation is reported separately; two seeds do not support precise seed-population inference.

Do not claim equivalence merely because an interval crosses zero. Keep all failed arms in reliability reporting and do not compute a balanced aggregate from incomplete arms. Since validation periods were previously exposed, formal intervals quantify conditional uncertainty rather than removing model-selection bias. Any locked retrospective final test requires a separate frozen protocol and measured budget review after this full stage. No test scores informed this decision.

## Budget, stopping and provenance

Measured selected-development training time, multiplied by six and scaled by mean robust training fraction 0.65/0.70, plus 30 seconds overhead per job, the established 3x cost factor and $0.05/job allowance yields an expected conservative stage cost of $6.76. This fits the $10.99 available after the $1 reserve. It is a planning estimate: longer training, startup or failures can force a budget stop before all fits finish. Admission still reserves each job's full 1,200 seconds plus 120 seconds overhead, approximately $1.07, and uses the larger of cumulative metering and cumulative estimated charges. Earlier pilot, duplicate, preflight and development charges remain intact.

Stop submissions on unavailable cloud/billing status, changed frozen inputs, unresolved jobs, exhausted retries or insufficient full-job reservation. Cancel over-limit active work and verify its termination before another launch. Complete the stage only after all 162 fit artifacts reconcile. At completion assess whether the remaining budget supports a balanced follow-up or final comparison; never drop losing families to fund a favorable subset.

`scripts/prepare_modal_l4_robustness.py` appends this stage under the existing controller lock, rechecks cloud/billing, validates the frozen rate selection against development artifacts, and preserves the old state and cumulative ledger. The training/controller source and original 90 hashes stay unchanged; the new preparation script, selection report, frozen selections and this protocol are added to the frozen set. `controller/robustness-provenance/` stores the preparation state, new manifest, hashes and new files; the exact unchanged training source remains in `controller/controlled-provenance/working-tree.tar.gz`.
