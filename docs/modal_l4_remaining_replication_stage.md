# Fixed-rate, longer-training replication

Frozen before any follow-up scoring on 2026-09-11. All 192 tuning fits across twelve remaining models passed artifact, initialization, nested split/scaler, checkpoint-selection and prediction-metric audits. The original excluded DLinear, TSMixer and iTransformer are not part of this stage. There were no tuning failures; the earlier SCINet calibration failure remains charged in the same ledger.

## Question and equal controls

Do the model-dependent hypercomplex/low-rank trade-offs survive a longer training allowance and a fresh initialization seed after learning rates have been fixed? Preserve all twelve backbones and all eight variants: native, controlled real, 2D, 4D, 8D and their three low-rank budget controls. Keep the same four observed features, internal sites, external head, surrounding architecture, optimizer, data, context 32/horizon 5 and batch 32. Preserve the separate FiLM spectral interpretation and SegRNN's disclosed low-rank parameter excess. SCINet retains equal IEEE FP32 convolution/matmul precision.

Freeze each arm's learning rate from the complete development grid: choose the minimum outer development MAE/persistence; within a relative .1% tie choose the lower rate. Save all 96 choices, including development job and trial IDs, before submission. Use one fresh paired seed, **401**, a uniform **50-epoch maximum**, patience 20, relative checkpoint threshold .001, chronological inner-only checkpoint selection and restored best weights. Scale only on actual training rows. Eight independent child fits per model yield **96 fits in twelve sequential L4 jobs**.

This is deliberately a different evaluation setting from the 25-epoch seed-701 tuning stage. Do not attribute a difference between those stages solely to seed or epoch count, pool them as identically trained replicates, or call reused Copper dates held-out data. This is limited replication, not independent confirmation. No model or dimension is removed based on earlier accuracy.

## Budget and reduction from the draft proposal

Conservative completed spend is $11.379564; $7.620436 remains after the $1 reserve. The proposed two fresh seeds at 50 epochs would require about $10.42 in full reservations and does not fit. Even two seeds at 35 epochs require about $8.22. We therefore reduce uniformly to one fresh seed and retain 50 epochs, prioritizing convergence after **91/192 tuning fits selected the final epoch**. Fifty epochs may still be insufficient; inspect the new curves rather than assume convergence.

For each selected arm, use its measured training seconds per executed epoch, multiply by 50, sum the eight arms, and allow 20% additional runtime for the fresh seed. Add 120 seconds per batch for non-training work, round timeouts upward to 30 seconds with a 300-second minimum, then reserve full timeout plus 120 seconds at the existing L4 + two CPU cores + 4 GiB rates, 3x uncertainty and $0.05 startup. The complete-stage full reservations are approximately **$6.81**, plus a separate **$0.50 planning margin**. Admission must recompute this against fresh billing and current rates; the $20 cap and $1 reserve never change.

All twelve jobs must fit as a complete matrix before any submission. Run only one L4, chain batches so a failure stops the queue, and allow no blind retry. Preserve partial child artifacts and all attempt costs. A hard container loss can prevent transfer and requires explicit unresolved-artifact reporting; do not fabricate or silently repeat completed children. After this stage, inspect actual remaining funds before further work.

## Execution and validation

The separate `remaining_replication` evaluation field holds the exact eight-arm rate map. It is incompatible with `remaining_tuning` and requires a native template, the remaining-model initialization protocol, one cell/fold and no repeated numerical preflight. Child trials contain one variant, one frozen learning rate and one seed. Source changes preserve the original two-rate tuning semantics and all historical candidate hashes. CLI/worker/controller count eight fits for seed 401. The code also tests multi-seed pairing but this executable stage uses only seed 401.

Every fit returns labelled curves, predictions, parameters, selected epoch, initialization and split audits. Batch manifests and archives preserve each child, including partial failures. Pair untouched weights within each seed; verify that different seeds produce different reference states. CPU remote-payload tests exercise one- and two-seed replication, rate maps, labels and artifact counts, in addition to prior tuning isolation and partial-failure tests. Actual CLI dry runs cover all twelve batch candidates and all 96 child settings.

## Preregistered uncertainty and interpretation

Freeze 180 primary contrasts: for each of twelve models, each of 2D/4D/8D versus controlled real, native, persistence and corresponding low-rank control (twelve), plus the three pairwise dimension comparisons. Compare mean absolute errors across the five leads at each forecast origin. Use identical origins across arms and models; overlapping leads and seeds are not independent observations.

Development-only block selection examined all 180 chosen-rate origin-loss differences. For each contrast, find the first lag at least five followed by five absolute autocorrelations below 1.96/sqrt(297), with a cap of sixty. The maximum selected lag is **60**, with 23 contrasts reaching the cap. Freeze circular moving-block resampling at length 60, and sensitivity lengths 30 and 120. There are only about 4.95 effective blocks at length 60, so intervals are exploratory and their coverage is uncertain.

Use 10,000 resamples, RNG seed 20260911, the same resampled origins for all contrasts, and a family-wise 95% max-absolute centered/studentized bootstrap interval. Negative mean absolute-loss difference favours the left arm. Report unadjusted point differences and relative error changes alongside simultaneous intervals; retain degenerate-variance flags. Do not interpret nonsignificance as equivalence, nor an interval excluding zero as independent confirmation. Do not pool the differently trained development and replication seeds to manufacture replication counts. Learning-rate selection on these same dates remains a source of optimism.

The exact implementation is frozen in `scripts/analyze_remaining_replication.py`; it refuses to analyze an incomplete stage. Its tests verify temporal dependence, shared opposite contrasts, reproducibility and degenerate/nonfinite behavior. The selection diagnostics live in `controller/replication-provenance/analysis-plan.json`. Audit all 96 fits and convergence before running the final analysis; source, fixed rates, CLI manifests and this document are archived before launch.
