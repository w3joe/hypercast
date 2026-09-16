# Internal dense replacement experiment

Status: **execution authorized on 2026-09-11**. The user requested that these experiments start with **another $20**, a separate allocation from the previous programme. Run one Modal L4 at a time. Preserve the previous $18.793498 conservative ledger and results. Operational manifests, numerical reports and the new cumulative ledger live in `results/modal-l4-internal/`; see `docs/modal_l4_internal_operations.md`.

## Question and architectural constraint

Does replacing an existing internal real dense map with a complex (2D), quaternion (4D) or octonion (8D) map improve accuracy, parameter efficiency or measured runtime?

Keep the original four observed input features. Preserve every surrounding operation: input handling, normalization, activation, residual connection, dropout, attention head count, layer width, output dimension and forecast head. Do not add a learned input expansion, an extra activation or an additional dense layer. Only the selected existing operator changes.

The primary comparison replaces **one internal map at a time**, chosen before inspecting results. The secondary comparison replaces a declared group of maps. This separates the effect of a single replacement from the cumulative effects of changing several sites. These are experiments on the repository's wrapped TSLib models, not a claim to reproduce each published model's original forecasting protocol.

## Verified primary sites

The CLI expanded the existing original presets at window 32/horizon 5. These are actual module paths and node IDs from the resulting listings, not hypothetical names.

| Model | Replace first | Real map | What it mixes |
|---|---|---|---|
| DLinear | Seasonal branch `Linear_Seasonal` | 32 → 32 | Temporal coordinates, separately for each observed channel |
| TSMixer | First block `model.0.temporal.0` | 32 → 32 | Temporal coordinates before the existing ReLU |
| iTransformer | First attention block `attention.out_projection` | 32 → 32 | Attention output embedding coordinates |

Exact expanded IDs:

- DLinear: `layers_0_model_linear_seasonal`.
- TSMixer: `layers_0_model_model_0_temporal_0`.
- iTransformer: `layers_0_model_encoder_attn_layers_0_attention_out_projection`.

Primary context/horizon: **32/5**. This uses the wrapper's existing 32-step internal representation without adding left padding. Keep `d_model=32`, one block and the original four channels. The external forecast head remains unchanged. Because the wrapper sets internal prediction length equal to its padded context, DLinear's selected map is 32→32 even though the external forecast horizon is 5.

The algebras group coordinates along the replaced map's last axis: time for the DLinear/TSMixer sites and embedding coordinates for iTransformer. They are not automatically modelling groups of two/four/eight physical variables. Report each site/model separately rather than treating them as identical operations.

### Important incompatibilities

TSMixer's channel MLP uses 4→32 and 32→4 maps. A direct 8D replacement cannot preserve width 4. Leave those maps unchanged in the primary three-dimension comparison. Do not pad or lift four observed channels to eight to make 8D fit. A separately labelled 2D/4D-only channel experiment is possible later, but cannot answer the balanced 2D/4D/8D question.

iTransformer's feed-forward maps are implemented as kernel-1 Conv1d layers, 32→64 and 64→32. These are pointwise linear maps across embedding channels, but the current CLI dense replacement does not directly accept them. A later feed-forward experiment requires an explicit transpose/linear/transpose wrapper and a numerical equivalence check before substituting HyperDense. It must preserve the token axis, GELU, dropout, normalization and residual paths.

## Primary controls: eight configurations per model

| Arm | Selected operator | Parameters at 32→32, including output bias |
|---|---|---:|
| Native original | Existing operator and native initialization | 1,056 |
| Real replacement control | Replace the same graph node with real dense; controlled initialization | 1,056 |
| 2D | Complex map, 16 hypercomplex input/output units | 544 |
| 4D | Quaternion map, 8 hypercomplex input/output units | 288 |
| 8D | Octonion map, 4 hypercomplex input/output units | 160 |
| Real rank-8 | 32→8→32, no intervening activation | 544 |
| Real rank-4 | 32→4→32, no intervening activation | 288 |
| Real rank-2 | 32→2→32, no intervening activation | 160 |

For exact low-rank parameter matching, the first factor has **no bias** and the second has the same 32-dimensional bias as the other arms. These factorizations match the parameter budget, not the rank or expressivity of the hypercomplex map. Report whole-model parameters and effective rank separately.

This makes **24 configurations** across the three backbones. Persistence is a prediction baseline and needs no training. The native arm matters particularly for DLinear: its temporal weights start at the constant averaging matrix 1/32. Replacing that with a random initializer changes behavior even for real dense. Keep native and controlled-real results separate, and never attribute their difference to algebra.

An additional zero-cost implementation check replaces a real layer with an identically weighted real layer and compares it with the native model. That is an equivalence gate, not another tuned experimental arm.

## Preparation required before training

The original saved CLI prototypes validate graph shape only. Execution now uses the separate opt-in `internal-matched-v1` protocol and the generated 24-arm manifest after passing numerical and split gates. Existing `matched-v1` still explicitly rejects schema-version-2 graphs; it is not weakened or silently bypassed.

1. Implement a versioned graph/internal initialization protocol. Build one reference model per seed/cell/fold, copy all untouched weights and buffers into every arm by stable source/module identity, then initialize only the replacement separately. Assert identical digests for every untouched tensor, including the forecast head. Use a stable named RNG stream per replaced site and pair data order/dropout streams independently of constructor draw counts.
2. At a controlled replacement, match expected effective real weight variance across real/2D/4D/8D, zero biases and use the same activation-gain convention. Width 32 gives variance 1/32 under the previous convention. A structured matrix cannot have every entry independently matched to a dense matrix; document the remaining correlations. Initialize low-rank products to the same effective variance in expectation. The native-original arm retains the native tested weights while sharing unchanged weights with the other arms.
3. Verify expansion and real-to-identical-real replacement preserve forward output, input gradients, untouched parameter gradients, optimizer behavior, parameter count and graph connectivity. Test CPU FP64, CPU FP32 and CUDA FP32. For dropout, use evaluation mode for strict graph equivalence and reset paired RNG streams for training-mode checks.
4. Verify each hypercomplex forward and gradient against an independently assembled real block matrix, including component packing and non-square cases needed for later FFN replacement. Check empirical initialization scale. Confirm the selected layer is actually executed and receives gradients.
5. Implement the exact-budget low-rank graph replacement without an added nonlinearity. Verify only the declared site changes and shared call sites are preserved intentionally; CLI replacement currently detaches a selected shared call. The three selected primary sites each have a single observed call.
6. Add the nested early-stopping data split described below, and record every training/inner-validation epoch plus the restored checkpoint epoch. Freeze source, data, manifests, analysis code and dependency versions before cloud execution.

Run small synthetic controls before real-data fits: recover a teacher map that is inside the chosen algebra's hypothesis class, and compare on an unrestricted real teacher map. These are implementation/capacity diagnostics with held-out synthetic examples, not evidence that one algebra is superior on real forecasting data.

Execution detail: the initial synthetic gate uses exact orthogonal projection over the full matrix basis (equivalent to recovery on a spanning input basis), with independent random inputs for the forward/gradient check. It separates representability from optimization. This is a closed-form capacity diagnostic, not a learned-teacher recovery claim; iterative teacher-training experiments remain an optional follow-up. The GPU gate runs inside the first native pilot job before that job trains. A gate failure stops the dependent queue. Full-model identical-real equivalence covers FP32/FP64, evaluation/training modes with paired dropout, input/parameter gradients and one Adam update. All 24 graphs check selected-site gradient participation and exact parameter budgets.

## Training and data protocol

Use Adam/MSE, batch 32, full precision and at most **150 epochs** for every arm. Allocate the same learning-rate grid, initially **0.0003 and 0.001**. Optimize neither the number of learning rates nor the epoch budget only for an apparent winner.

Unlike the previous fixed-final-epoch runs, use **inner-validation checkpoint selection**: patience 20 epochs, minimum relative inner-MSE improvement 0.1%, restore the best inner-validation checkpoint. The outer evaluation period is never used to choose the epoch. Apply this rule to native, real, hypercomplex and low-rank arms identically. Add a final-epoch sensitivity score from the same training trajectory where feasible, labelled secondary; do not choose between reporting policies after seeing the outer result.

For each nominal robust training prefix (55%, 65%, 75%), allocate its final 15% to inner validation and train on its first 85%. For example, fold 1 is train 0–46.75%, inner validation 46.75–55%, outer evaluation 55–65%. The other outer evaluations are 65–75% and 75–85%. Fit preprocessing on the actual training subset, not inner validation or outer data. Require every forecast target to lie completely within its split. Preserve these exact boundaries across all arms.

The previous Copper dataset has already been used for development, robustness and final scoring. Any new internal-replacement results on it are **exploratory**, even if new seeds or cells are used. Its old final 15% cannot be relabelled untouched. Do not select these replacements using the previous final-period scores: the sites above are chosen for direct replaceability and architectural coverage.

For independent confirmation, reserve new chronological observations or a separately frozen dataset not previously used for these decisions. Freeze the file hash, four input columns, target, dates and splits before scoring. Until that data is specified and available, the independent final stage is conditional and must not launch. Independent datasets are required for a broader claim; more seeds on Copper do not substitute for them.

## Stages and fit counts

| Stage | Matrix | Fits |
|---|---|---:|
| Numerical preparation | Synthetic CPU/CUDA gates above | No real-data tuning |
| Runtime pilot | 24 arms × one seed × one fold × 5 epochs | 24 |
| Equal development tuning | 24 arms × two rates × seed 701 × 32/5 × one outer development period | 48 |
| Primary robustness | 24 arms × seeds 401/503/601 × 32/5 × three outer periods | 216 |
| Independent final, conditional on new data and budget | 24 frozen arms × three frozen seeds × one untouched period | 72 |

Choose the development rate using outer development MAE/persistence after inner-only epoch selection; relative differences within 0.1% choose the lower rate. Freeze one rate per configuration before primary robustness. Pilot seed 907 is for implementation/runtime only; do not select a winner from it. The development period uses nominal train prefix 70%, inner split as above, and outer 70–85%. These historical periods overlap robustness, so the primary robustness stage measures seed/regime stability rather than independent confirmation.

Maximum planned fits are 360, including the 24 short pilot fits. Actual runtime depends on checkpoint stopping, graph execution and replacement sites. Do not extrapolate CUDA speed from parameter counts. If this is too costly after the pilot, reduce cells or seed replication symmetrically and document the narrower question before proceeding; keep all dimensions and real controls. Do not start a final stage without enough conservative allocation to admit the complete balanced matrix, including end-of-queue reservations.

## Secondary location experiments

Only after the primary comparison is complete and if budget remains, freeze a separate placement matrix:

- DLinear: seasonal only, trend only, both branches.
- TSMixer: first temporal map only, second temporal map only, both temporal maps in the first block.
- iTransformer: attention output only, Q/K/V only, all four attention projections. Separately test the FFN pair after implementing and validating the Conv1d-equivalent wrapper.

For each selected placement compare real, 2D, 4D and 8D with the same surrounding graph, and retain the relevant parameter-budget controls. Do not combine placement search and a dimension ranking into a single uncorrected winner. A prespecified 32/10 sensitivity tests horizon transfer without changing the internal 32-wide map. Context 64 changes some temporal dimensions and requires distinct validated templates rather than reusing fixed-32 replacements.

To distinguish algebra-specific structure from generic compression, a later control should use random signed weight-tying maps with the same free-parameter counts and matched scale, over multiple frozen pattern seeds. Low-rank controls alone cannot establish that the algebra multiplication table is responsible for a benefit.

## Outcomes and decision rules

Primary: equal-seed/equal-fold MAE relative to persistence. Report each backbone/site separately. A practical accuracy benefit requires more than 2% improvement against the controlled real replacement with a multiplicity-adjusted interval, and usefulness must also be assessed against native original and persistence. Compression benefit requires noninferiority within 2% plus lower measured whole-model parameters; speed/memory benefit requires direct measurements.

Preregister 45 primary contrasts: per backbone three hyper-versus-controlled-real, three hyper-versus-native-original, three hyper-versus-corresponding-low-rank, three hyper-versus-persistence and three pairwise dimension comparisons. Use paired temporal block resampling of origin-level errors, average seed losses before treating dates as the sampling unit, choose block length from development only (at least horizon), and report half/double sensitivity and effective block counts. Use 10,000 resamples and family-wise correction; do not treat overlapping forecast origins or seeds as independent datasets. Keep failure rates and missing arms visible. No nonsignificant result should be labelled equivalent.

Also report absolute MAE/MSE, per-lead errors, selected epoch, train/inner-validation curves, seed dispersion, total parameters, effective rank of selected operators, synchronized training duration, full job duration, batch-32 device-resident inference latency and peak GPU memory. Report accuracy, compression and runtime conclusions separately.

## CLI prototype workflow and budget

The files under `plans/internal-replacement/` contain three expanded originals, their actual layer listings, and real/2D/4D/8D single-site prototypes. The CLI validates the prototypes at 32/5 and 32/10. These files establish topology and dimensions only; the initialization, nested stopping and low-rank controls above remain implementation work.

Example using the verified TSMixer node:

```sh
hypercast model expand original-tsmixer.yaml --out tsmixer-graph.yaml --cells 32/5,32/10
hypercast layer list tsmixer-graph.yaml --cells 32/5,32/10
hypercast layer replace tsmixer-graph.yaml layers_0_model_model_0_temporal_0 \
  --type hyper_dense --set algebra=quaternion --cells 32/5,32/10
hypercast model validate tsmixer-graph.yaml --cells 32/5,32/10
```

Run one Modal L4 at a time if later authorized. Obtain prices through the CLI and estimate stage cost from the new pilot, including GPU, CPU, memory, startup, failed jobs and the established uncertainty factor. Keep a safety reserve and reserve each full timeout before submission. The previous monitor stays paused; this plan does not restart it or spend the old remaining balance. A costed runnable manifest is the output of preparation and pilot, not an assumption in this plan.
