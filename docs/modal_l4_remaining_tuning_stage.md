# Remaining-model balanced tuning stage

Prepared 2026-09-11 after all twelve native calibrations passed their eight-variant GPU gates, initialization pairing, nested split/scaler and restored-checkpoint audits. The SCINet precision repair passed the original CUDA tolerance; its failed attempt remains charged. Closed original-study results and the excluded DLinear, TSMixer and iTransformer backbones are preserved.

## Frozen comparison

Twelve backbones, eight variants each, two learning rates (.0003 and .001), one paired seed (701), context 32/horizon 5, batch 32, maximum 25 epochs, patience 20 and relative checkpoint threshold .001. Restore the checkpoint selected by the chronological inner split. Scale on actual training rows only. Keep all surrounding architecture, observed features, activation, dropout, optimizer settings and external forecast head fixed. SCINet uses IEEE FP32 for both convolution and matrix multiplication in every arm. FiLM remains a separately labelled spectral intervention with a native already-complex operator; its native initialization differs in scale from the controlled real baseline. Preserve the disclosed approximate low-rank budgets, including SegRNN's eight-parameter excess for the 8D control.

This gives **192 fits** with equal tuning opportunities. Each model is one sequential batch of sixteen fits, in the frozen variant order native, real, complex, quaternion, octonion, lowrank2, lowrank4, lowrank8, and ascending learning rate within each variant. Every fit initializes independently with the same paired stream and training seed. No state, optimizer or learned weight transfers between fits. The original three backbones are not repeated. The older 84-fit, five-epoch expansion proposal is superseded by the user's tuning priority.

The maximum is reduced from the earlier three-model 150 epochs to 25 uniformly before tuning scores are collected. Runtime feasibility—not pilot accuracy—determines this reduction. Eleven of twelve native calibrations selected epoch five, so undertraining remains a substantial possibility. A 25-epoch comparison is exploratory development, not a completed robustness study. Do not select or exclude dimensions or backbones based on five-epoch accuracy.

## Full-stage admission and stopping

Conservative spend at preparation is $7.216611 from the same additional $20 ledger; delayed metering is about $0.70. Remaining headroom after the $1 reserve is $11.783389.

The twelve native five-epoch fits used 52.590970365 training seconds in total. For each backbone, estimate sixteen fits at five times that duration for 25 epochs, then multiply by two to allow slower variants. Add 120 seconds per batch for initialization, predictions and artifact work; round the timeout up to a 30-second boundary, with a minimum of 300 seconds. The controller reserves each full timeout plus another 120 seconds, prices L4 plus two CPU cores and 4 GiB memory, applies 3x uncertainty and adds $0.05 startup per job.

The resulting **complete-stage timeout reservations total $9.472195**, with an additional $1.50 planning margin. Both fit inside $11.783389. Batch timeouts range from 330 to 1650 seconds. This is a conservative reservation calculation, not a forecast of actual billing. At the same assumptions, 50/100/150 epoch grids do not fit this remaining budget. These choices are frozen before new scores are observed.

Use one L4 and one job at a time. Each batch depends on the previous batch succeeding. No automatic retries. A failed, missing or incomplete batch stops further submissions for diagnosis, preserves all attempts and is never treated as a complete comparison. A diagnosed retry must account for earlier completed fits, avoid silently duplicating them, and be separately costed; this initial batch runner intentionally does not provide automatic resume. All twelve current calibrations remain complete and are not rerun.

## Artifact and failure handling

`remaining_tuning: true` is an opt-in normalized evaluation field, restricted to the remaining-model protocol, a native template, one cell/seed/fold and completed prior numerical gates. It expands to the fixed sixteen independent child evaluations and changes the candidate hash. CLI, remote worker and controller all expect sixteen fits per job. Historical candidate hashes must remain identical.

Each child keeps its own request, status, curves, predictions, initialization and split audits. Root CSV/JSON tables label trial ID, backbone, variant and learning rate; CSV tables additionally label trial status. Summary tables retain individual arms and rates. They must not be interpreted as one averaged model. `batch-manifest.json` records completed, failed and pending trials. `batch-artifacts.tar.gz` preserves the original child directories and is replaced atomically after every attempted fit.

A soft deadline ninety seconds before the remote timeout is checked before each fit and after each training epoch, allowing partial curves and prior completed results to return through the normal Modal artifact channel. A normal child exception returns those artifacts and an accurate partial completion count. A catastrophic process/container loss can still prevent transfer; preserve the full failed attempt cost and mark its artifacts unresolved rather than manufacturing results. The hard Modal timeout is the final spending bound.

Validation includes the complete 16-fit synthetic remote-payload path, exact agreement of a late batched fit with its isolated counterpart, all trial labels and paired untouched-state digests, injected partial failure, soft-deadline artifact return and historical controller/worker/CLI regressions. Actual CLI dry runs cover all twelve batch candidates and their 192 child specifications. First GPU batch completion must be audited before drawing findings.

## Analysis and next decision

Review all 192 trial artifacts, numerical finiteness, initialization pairing, identical splits/scalers, selected inner checkpoint and convergence. Report the fraction of fits still selecting the final epoch. For each arm choose learning rate by development outer MAE/persistence, using the lower rate for relative ties within .1%; also show both-rate results so tuning sensitivity stays visible. Equal search budgets do not remove selection optimism on this already-seen period.

Report each model/site separately, with persistence, native and controlled real baselines, corresponding low-rank controls, whole-model parameter counts and measured timings. The possible contrasts remain hyper versus real/native/persistence/low-rank and pairwise dimensions: fifteen per backbone, 180 over twelve backbones. Do not turn a best development score into a significance or equivalence claim. Any later robust comparison must freeze chosen rates, independent paired seeds and temporal evaluation protocol before scoring; dependence-aware resampling and multiplicity correction are required. Copper remains exploratory and independent confirmation requires separately frozen new data.

After this stage, inspect actual remaining funds and convergence before choosing any follow-up. Do not automatically repeat a complete grid, increase the cap, restore the superseded original-three robustness stage or claim a universal winning algebra.

Preparation commands: `python scripts/prepare_modal_l4_tuning.py --verify`, then `--prepare`, then restart only `com.hypercast4d.modal-l4-internal-controller`. Decisions and source archives live under `results/modal-l4-internal/controller/tuning-provenance`. The supervisor remains periodic AI monitoring, not continuously executing AI.
