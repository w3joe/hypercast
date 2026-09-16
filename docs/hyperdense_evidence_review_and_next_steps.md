# HyperDense: consolidated evidence and next decision

Reviewed 11 September 2026 after the authorized representation pilot. This review uses the completed reports, underlying score tables, all twelve pilot prediction archives, checkpoint contents, inference measurements and convergence review. New calculations use only already-saved development predictions. No new periods were scored, models trained or paid jobs launched.

**Recommendation: validate transfer of the representation package before funding the 128-fit HyperDense tuning stage.** The evidence supports a useful implementation optimization and a promising native-model representation change. It does not yet establish practical forecasting superiority from hypercomplex structure.

## What the completed programmes establish

The original programme recorded 323 fits, including duplicates and a smoke fit. The separate internal programme recorded 372 successful fits, including short calibrations and development tuning. The latest pilot added twelve fits. These 707 recorded fits are not independent replications: datasets, periods, tuning outcomes and training ceilings differ.

| Evidence | Main conclusion | Important limit |
|---|---|---|
| Input frontend: DLinear, TSMixer, iTransformer | Complete robustness comparison favours native models and persistence over every hypercomplex frontend | Added real frontend is itself weak; final matrix is missing two low-rank configurations |
| Original-three internal study: 24 pilot + 48 development fits | Correctness and paired initialization passed; selected-rate development scores remained above persistence | Planned robustness was superseded; no new runs of these backbones are recommended |
| Twelve additional backbones: 192 tuning + 96 follow-up fits | No universal dimension wins; all 96 follow-up arms lose to persistence | One follow-up seed, reused evaluation dates, limited learning-rate search |
| Data diagnostics | Alignment, targets, scaling and persistence passed across 142,560 forecasts; large level shift is plausible | Descriptive distribution shift is not proof of the cause of all errors |
| Native representation pilot: twelve fits | Relative/residual beats direct output in all six paired seeds | Same inner data selects checkpoints and supplies headline scores; no HyperDense arm was trained |
| Matched implementation checks on L4 | Caching and dense export reduce overhead while preserving the tested maps | Fresh weights; comparison is among implementations of hypercomplex models, not against native/real/low-rank models in the same timing run |

Sources: [original programme](modal_l4_results.md), [internal programme](modal_l4_remaining_results.md), [diagnostics](hyperdense_followup_results.md), [latest pilot](hyperdense_representation_pilot_results.md). The first programme's historical final 15% was already scored. There is no basis for calling that Copper tail an untouched confirmation set.

## Accuracy and compression need separate decisions

**MICN 8D:** the strongest adjusted hyper-versus-controlled-real result was a 10.87% MAE reduction. It nevertheless lost to native MICN by 4.78% and persistence by 17.01%. Its whole-model parameter saving is only about 1.10%. This particular small internal site is an accuracy hypothesis, not a plausible route to 25% model compression.

**FiLM 4D:** 37.5% fewer parameters than controlled real with 0.56% higher MAE remains a compression signal, not established noninferiority. Against already-complex native FiLM, the saving is only about 16.7%. The comparison baseline must be stated explicitly.

The exact-budget FiLM low-rank-4 control has the same 5,243,537 parameters as FiLM 4D, but lower MAE/persistence (1.1666 versus 1.1815) and lower recorded batch-32 latency (12.14 versus 16.60 ms). Thus HyperDense 4D was approximately 1.28% worse in MAE and 37% slower than that control in the historical run. Both still lost to persistence. Across the twelve backbones, the best low-rank point score beats the best hypercomplex point score on eight; that post-hoc envelope comparison is descriptive, not a corrected significance test.

Retain native, controlled real, 2D, 4D, 8D and all three corresponding low-rank arms if tuning proceeds. Do not pick one algebra from point rankings. Also do not discard MICN or FiLM after seeing disagreement and then call the resulting study the originally balanced comparison.

## What closer inspection adds to the native pilot

The median paired development-MAE reductions are 11.52% for MICN and 2.08% for FiLM. Mean MSE across seeds also falls (MICN 0.004168 to 0.003549; FiLM 0.003586 to 0.003356), so the aggregate improvement is not confined to MAE alone. None of this evaluates new dates.

I partitioned each saved 207-origin development sequence into three consecutive groups of 69 origins, retaining all five leads. This is a newly chosen, post-hoc diagnostic; no statistical confidence or causal interpretation is attached to it.

| Model | Seed | Relative MAE reduction: first third | Middle third | Last third |
|---|---:|---:|---:|---:|
| MICN | 1101 | 13.43% | 14.58% | 14.87% |
| MICN | 1102 | 11.63% | 3.99% | 19.57% |
| MICN | 1103 | 12.10% | 8.11% | 11.41% |
| FiLM | 1101 | 4.85% | 10.86% | **−13.68%** |
| FiLM | 1102 | 7.01% | 7.32% | **−6.68%** |
| FiLM | 1103 | 5.28% | 5.24% | **−5.28%** |

Positive values favour relative/residual. MICN improves in all nine seed/time cells and on roughly 64–68% of forecast origins. FiLM loses in the last third for all three seeds and improves on only roughly 50–55% of origins. Overlapping leads and shared dates make these cells dependent. This does not overturn the aggregate paired-score result; it makes transfer of a common representation package uncertain.

Relative/residual also fails to beat persistence at lead 1 for all three FiLM seeds; its ratios range from 1.004 to 1.045. MICN is approximately tied at lead 1 and is strongest against persistence at lead 5. A single aggregate error hides this horizon dependence.

The direct FiLM seed 1103 checkpoint was selected at epoch 88, with training stopped by the ceiling at 100. Twelve stale epochs do not exhaust patience 20. The score criterion passed, but the full integrity/convergence advancement gate remains uncleared. The saved checkpoints contain model state and metadata, **not Adam state, RNG state or sampler state**. They support inference and warm starts; they do not support exact continuation of the original optimization trajectory. Extending from the restored best weights would be a new training protocol.

Reproducible new diagnostics: [existing-results review](../results/hyperdense-evidence-review/existing-results-review.json), generated by `scripts/review_hyperdense_evidence.py`.

## What to do next, in order

### 1. Use the saved checkpoints for a no-training transfer stress test

Before another paid training programme, freeze a separate diagnostic manifest and evaluate **all twelve unchanged pilot checkpoints** on the already-exposed outer Copper period, 2020-08-11 through 2021-10-19. Use the original training scaler, context 32, horizon 5, target alignment and persistence, with CPU inference. Report all seeds, leads, bias and predeclared chronological/regime summaries; do not select or retrain checkpoints on this period.

This specifically tests the remaining question: does the package help in the period where roughly 76% of Copper prices exceed the training maximum? It costs no additional model fits or L4 time. It is an explicitly retrospective development stress test, **not completion of the original pilot's preregistered gate or independent confirmation**. A poor result would argue for investigating representation and baselines before spending on 128 hypercomplex tuning fits. A favourable result would justify further development, not a success claim.

This review has not performed that new-period inference. Preserve the original pilot's inner-only report and store any stress-test results separately.

### 2. Freeze new data and a real separation between development and confirmation

If the stress test is useful, choose the next dataset based on the question rather than the strongest Copper result. For general forecasting, ETTh1 is a reasonable first nonfinancial benchmark: the authors provide hourly transformer-load/oil-temperature data with six load series and one target. Preserve all seven numeric channels; do not choose four retrospectively to fit the current wrapper. The [dataset author's repository](https://github.com/zhouhaoyi/ETDataset/blob/main/README.md) documents the variables and daily/weekly structure. The [official Time-Series-Library](https://github.com/thuml/Time-Series-Library) also supports ETT with seven-channel configurations.

ETTh1 is only a proposal, not a frozen or downloaded dataset here. A new dataset is not automatically an independent confirmation set: audit prior project use, freeze a source revision/hash, target, input columns, dates and a final test partition before any new tuning. The current four-feature preparation and hard-coded model construction require validation for seven channels. Preserve feature count equally across every algebra; dimensions describe the replaced latent operator, not a reason to add or remove observed channels. Choose context/horizon and seasonal-persistence baselines for hourly data before scoring, and recalibrate compute. Retaining 32/5 would be a labelled custom short-horizon experiment, not a standard long-horizon benchmark reproduction.

Use distinct training, checkpoint-selection, development-ranking and final-test roles. Do not keep using the same 207 inner origins to select checkpoints, rank hyperparameters, choose a representation and claim confirmation. More seeds on the same data do not fix this.

### 3. Resolve representation stability and convergence before the large grid

Keep both MICN and FiLM. On the frozen development design, compare the same two packages with three paired seeds and a common, predeclared sufficient ceiling within each comparison. A uniform 150-epoch ceiling is a candidate to calibrate, not a guarantee of convergence. A clean replay from seeded initialization is needed for a longer-trajectory comparison because exact optimizer state was not archived. Do not extend only a favourable arm.

The current package changes centering, persistence anchoring and output initialization together. A factorial ablation is optional if the scientific aim is to identify which change helps; it is not necessary to pretend this pilot isolated centering alone. In either case, save the latest resumable state as well as the best inference checkpoint, including optimizer, RNG, data-order and scheduler state where applicable.

### 4. Only then test the HyperDense hypothesis with equal opportunity

The proposed 128-fit search remains conditional: two models × eight arms × eight optimizer/initialization settings, followed by 80 fits for five paired confirmation seeds on a frozen dataset. Implement the AdamW/initialization grid, effective-variance controls for complex and factorized operators, and checkpoint/preprocessing reconstruction tests before admission. Do not silently reuse the current Adam-only pilot runner as if it implements this grid.

Keep accuracy and compression success criteria distinct. Accuracy needs useful improvement over native and controlled real, with persistence a practical benchmark. Compression needs the predeclared noninferiority margin and explicit savings against native as well as controlled real; matched low-rank performance remains essential. Use paired seeds and blocked time uncertainty with correction for the declared contrast family. Few effective blocks should limit claims.

## Implementation work can proceed independently of accuracy research

L4 cached constants achieved a median 1.25× isolated-layer speedup; dense export 11.23×. Dense-export whole-model latency fell about 11–13% for MICN and 31–37% for FiLM relative to their original hypercomplex implementations. These are useful engineering findings.

Keep the compact model for training/checkpoint storage and test cached inference where compact deployment matters. Treat dense export as a deployment option that expands resident weights and must be regenerated after learning. The existing prototypes freeze parameters and are inference-only; do not put them directly into training. Before claiming superiority over native or low-rank execution, benchmark all relevant trained arms in the same job, precision, batch shapes and input-residency conditions. Do not multiply speed ratios from different studies to imply an unmeasured native-baseline speedup.

## Spending and scope

The original $20, internal $20 and latest $6 allocations are closed at conservative accounted totals of approximately $18.79, $15.51 and $3.48 respectively. These are separate ledgers, not reusable balances or final invoices. No new spending cap is authorized by asking for this review.

The earlier $48 proposal was tied to the historical workload and included stages now completed. It is not a current launch quote. New feature counts, horizons, dates, epoch ceilings and the full eight-arm implementation require fresh calibration and full-stage admission. Parallel execution reduces elapsed time, not necessarily GPU-seconds or billed cost. Maintain the latest operational preference for at most two L4 jobs concurrently unless changed explicitly; retain all failed-attempt charges and the protected reserve.

The immediate recommendation is the checkpoint-only retrospective stress test, followed by a separately costed data/representation development stage. The full HyperDense tuning programme should wait for that evidence.
