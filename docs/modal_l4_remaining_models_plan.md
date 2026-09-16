# Next experiment: remaining model architectures

User steering recorded 2026-09-11: run the next experiment across the other models; do not repeat DLinear, TSMixer or iTransformer.

## Priority and scope

Finish the currently scheduled internal-development stage. Then prioritize breadth across the **12 remaining pinned TSLib architectures**, rather than automatically appending the previously planned robustness stage on the original three. Preserve completed work and report that their robustness stage was superseded by the user's new priority, not completed.

Exclude `tslib-dlinear`, `tslib-tsmixer`, `tslib-itransformer`, and their inspired aliases from every new-stage candidate. The relevant comparable model catalogue has 15 TSLib backbones. Paper CNN/LSTM, residual TCN and pre-existing hypercomplex hybrid presets are separate architecture families; they are not counted as additional TSLib backbones or silently represented as covered by this matrix. This scope interprets “all models” as all remaining backbones in the same catalogue used for the current comparison.

No additional funding was requested or authorized with this instruction. Use the **remaining balance of the existing additional $20 allocation**, including current tuning costs, with its $1 reserve and one-L4 limit. Do not create a fresh $20 ledger. A complete broad pilot and full tuning/robustness on every backbone may not fit; measure runtime and disclose any incomplete coverage rather than silently dropping models or controls.

## Why only three were tested first

DLinear, TSMixer and iTransformer supplied three different mechanisms with simple, executed 32→32 replacement sites. They were a bounded initial comparison, not an exhaustive survey. The next experiment expands architectural coverage as requested.

## Verified inventory

Baseline specs are saved under `plans/internal-expansion/`. `executed-layer-inventory.json` records CPU forward/backward checks at context 32, horizon 5 for all 12 backbones. All 12 baseline checks passed. Forward hooks record actual execution, input/output shapes, call counts and weight-gradient presence. This is a baseline inventory, not validation of hypercomplex replacements.

Proposed primary sites below are selected from structure and execution evidence, without looking at their forecasting accuracy. Paths are relative to the upstream model inside `layers.0.model`.

| Backbone | Proposed site | Real dimensions | Preparation |
|---|---|---|---|
| PatchTST | `encoder.attn_layers.0.attention.out_projection` | 32→32 | Direct dense replacement |
| TimesNet | `predict_linear` | 32→64 | Direct temporal dense replacement |
| Crossformer | `encoder.encode_blocks.0.encode_layers.0.time_attention.out_projection` | 32→32 | Direct attention projection replacement |
| FiLM | Forecast path has `mlp` 3→1; no compatible ordinary dense map | 3→1 | Special case below |
| FreTS | `fc.0` | 4096→256 | Replace existing forecast MLP map; retain native complex frequency processing |
| LightTS | `layer_3.spatial_proj.0` | 16→16 | Direct dense replacement |
| MICN | `regression` | 32→32 | Direct temporal dense replacement |
| MSGNet | `seq2pred.seq2pred` | 32→32 | Single-call temporal map; avoids detaching a shared attention call |
| SCINet | `projection_1` | Conv1d 32→64, kernel 1, no bias | Equivalent transpose/linear/transpose wrapper required |
| SegRNN | `predict.1` | 32→8 | Direct dense replacement |
| TimeMixer | `pdm_blocks.0.mixing_multi_scale_trend.up_sampling_layers.0.2` | 32→32 | Single-call dense map; preserve surrounding shared modules |
| TimeXer | `encoder.layers.0.self_attention.out_projection` | 32→32 | Direct attention projection replacement |

FiLM cannot receive a shape-preserving 2D/4D/8D HyperDense at its 3→1 forecast fusion layer. Do not pad/expand that layer or replace the common external head merely to mark FiLM covered. Investigate its existing learned spectral operators for an equivalent real-block representation and separately labelled structured spectral replacement. That would be a different intervention requiring native-operator equivalence, component-layout and complex-output checks. Until validated, report FiLM as a baseline with an explicit replacement-compatibility limitation, not a completed HyperDense experiment. Keep it visible in the coverage report.

SCINet's kernel-1 projection is a dense map across the temporal-channel axis. Its 32→64 dimensions support all three algebras, but the no-bias setting and tensor axis must be preserved. Validate the equivalent real wrapper before running any replacement.

## Experimental design

For each compatible backbone, keep native original, controlled real replacement, 2D, 4D, 8D and three real low-rank controls. Preserve all observed inputs, context, output head, activation, normalization, residual paths and dropout. FreTS's existing complex processing remains in every arm; its experiment tests its forecast dense map, not the effect of introducing FFT.

The target is eight arms for each of 12 backbones (96 configurations), conditional on a scientifically valid FiLM intervention. Eleven compatible backbones would give 88 replacement configurations, with FiLM separately marked as pending/incompatible rather than omitted. Do not claim 96 runnable configurations before implementation and verification.

For non-square sites, a low-rank factorization does not necessarily match the hypercomplex parameter count exactly. Choose the largest integer rank that does not exceed the corresponding hypercomplex map budget, omit the first-factor bias, and preserve the original output bias setting. Record the resulting gap. Use separate parameter-matched width comparisons only if explicitly designed; do not mislabel approximate budgets as exact.

Prototype audit exception: SegRNN's 32→8 map has an 8D budget of 40 parameters including bias; even a standard rank-one real factorization needs 48. Retain rank one as the nearest positive-rank control and disclose its eight-parameter excess. Do not silently omit the control, invent a rank-zero learned map or widen the model to manufacture an exact match.

1. Implement a new versioned initialization/replacement protocol for these shapes and architectures. The current `internal-matched-v1` supports only its original three sites and assumes width 32; it cannot be reused by changing labels. Pair every untouched parameter/buffer and initialize selected maps with matched effective variance. Preserve shared module calls and all data-dependent forward paths; avoid graph tracing that freezes input-dependent branch choices.
2. Require actual CPU and CUDA forward/gradient checks, selected-site gradient participation, identical-real output/input/parameter-gradient and optimizer-step equivalence, shape and whole-model parameter audits. For rectangular fixed-algebra maps, compare with independent real block matrices.
3. Run a bounded breadth pilot across the remaining backbones, initially five epochs and one frozen seed, before selecting a longer affordable matrix. Use nested chronological stopping and train-only scaling as in the current study. Five-epoch scores are runtime/implementation diagnostics, not evidence of effectiveness.
4. Estimate the complete balanced next stage from measured timings and remaining cumulative funds before launching it. Prefer coverage of every compatible backbone over additional rounds on the original three. Any reduction in rates, epochs, folds or seeds must be symmetric and documented before scoring; never hide omitted controls or incompatible models.
5. Analyze each backbone/site separately. Retain persistence and native-original comparisons. More architectural coverage does not create independent data: the Copper periods remain exploratory. Freeze updated contrasts and multiplicity correction for the expanded matrix before robustness scoring.

## Supervisor action after current tuning completes

Read this document and `docs/hyperdense_literature_notes.md` before choosing the next stage. **Do not call `scripts/prepare_modal_l4_internal.py --advance` after internal-development completes**, because that command would append the now-superseded robustness stage for the original three. Instead prepare and verify a new expansion stage, append it to the same cumulative ledger, and resume only the existing `com.hypercast4d.modal-l4-internal-controller` service.

The new protocol, any CLI compatibility work and candidate manifest must be completed before cloud submission. If source changes are needed, wait for current jobs to stop, reconcile billing, and version the frozen provenance. The online literature is guidance for future hypotheses, not permission to alter a running comparison.

## Offline preparation completed during the 06:39 UTC monitor review

`plans/internal-expansion/controls_prototype.py` now implements eager module replacement for the eleven compatible backbones, keeping their original forward methods and data-dependent branches. It is outside the active worker package and is not integrated into the running experiment. Run its CPU checks with the durable Python CLI. Evidence is saved in `plans/internal-expansion/replacement-prototype-preflight.json`.

All 88 prototype configurations passed CPU forward/backward, selected-site participation, untouched-state pairing and RNG-preservation checks. Identically weighted real replacements passed full-model output, input-gradient and parameter-gradient comparisons for all eleven backbones, including SCINet's kernel-one projection. SegRNN required preserving contiguous dense-output memory layout before its existing `view()` operation; the prototype handles this without changing upstream code.

The report explicitly marks `cloud_ready: false`. Remaining gates are CUDA validation, training-mode/optimizer equivalence, independent rectangular algebra checks, and worker/CLI integration with a new versioned protocol after the active stage stops. FiLM's spectral intervention remains a separate feasibility task. Whole-model counts and low-rank budget gaps are recorded per configuration. No GPU jobs or new stage candidates were submitted by this offline preparation.

Integration must use an explicit allowlisted backbone/variant evaluation setting so candidate hashes distinguish each intervention. Rebuild the native reference from a stable named seed, replace only the declared module in the worker model and copy the complete paired state. Preserve the generic runner's nested stopping/scaler audit and recorded initialization report. The current graph-only initializer must retain its original semantics and historical hashes.

## Further offline validation during the 07:04 UTC monitor review

The eleven-backbone prototype now passes both evaluation- and training-mode identical-real comparisons with paired dropout and one Adam update, including input, untouched-parameter and selected-parameter gradients. Eighteen independent FP64 algebra reference checks cover all selected rectangular shapes, including no-bias SCINet and the large FreTS map. The refreshed `replacement-prototype-preflight.json` records these checks; CUDA and production integration remain pending.

FiLM now has a **separate spectral prototype**, `plans/internal-expansion/film_spectral_prototype.py`, with results in `film-spectral-preflight.json`. It replaces only `layers.0.model.spec_conv_1.0`, the first existing spectral operator. Each of its sixteen frequency-specific complex 256→256 maps is represented as a real 512→512 map, preserving FFT, frequency mask, inverse FFT and no bias. The 3→1 fusion and other scales remain untouched. Component layout is all 256 real coordinates followed by all 256 imaginary coordinates; 4D/8D further group those coordinates. This is not the four-STFT-window grouping in FIA-Net.

All eight FiLM variants pass CPU forward/backward and untouched-state pairing. An exact real-block implementation that preserves the native complex weight ties also passes full-model forward, input/parameter-gradient and Adam-step equivalence in both modes. The unconstrained real arm is deliberately more expressive; it is not expected to have the same training update as the native tied complex operator. Native FiLM is already complex-valued, so this intervention must be reported separately from ordinary dense replacements.

These results establish 96 CPU-tested prototypes across all twelve remaining backbones, not 96 cloud-ready runs. Before promotion, FiLM still needs independent per-frequency structured-map checks, CUDA checks and seed/cell/fold initialization integration. Its native initializer has a much smaller scale than the proposed controlled variance; keep native and controlled-real comparisons separate, inspect numerical stability, and never attribute their difference solely to algebra. Preserve this distinction in any future analysis.

## Production calibration protocol, 2026-09-11

The original development stage completed all 48 fits and passed initialization, split and restored-checkpoint review. Source and the cumulative state were archived under `controller/expansion-provenance` before integrating `remaining-internal-v1`. This protocol uses explicit allowlisted backbone/variant evaluation fields, deterministic seed/cell/fold streams, eager native forwards and exactly one selected internal site. All prior candidate hashes must remain unchanged.

FiLM now includes independent per-frequency real-block forward and gradient references for 2D/4D/8D, plus seed/cell/fold initialization. Its spectral intervention remains separate from the eleven ordinary dense/pointwise interventions. Native complex weights and the controlled real initializer differ in both scale and constraints; algebra contrasts use the controlled real arm. SegRNN's 8D low-rank control retains the disclosed eight-parameter excess.

The next stage is `remaining-calibration`: twelve native fits, five epochs, seed 907, context 32, horizon 5, learning rate .001, batch 32, full precision, chronological inner stopping and train-only scaling. Each job first runs all eight configurations through GPU forward/backward and parameter audits, native-to-identically-weighted-real checks in both modes including one Adam update, and independent algebra reference checks. SegRNN's evaluation-mode gradient diagnostic disables cuDNN because cuDNN GRU backward requires training mode; its training-mode diagnostics and measured fits use the normal backend. All-variant gradient gates exercise training mode.

Each calibration has a 600-second remote timeout with no automatic retry. At the current rates, all twelve full reservations including 120 seconds overhead, CPU/memory, 3× uncertainty and startup allowance total approximately $7.27. Conservative spend before this stage is $6.252325, leaving $12.747675 after the $1 reserve. These are maximum admission reservations, not predicted charges. The controller remains sequential on one L4.

The frozen breadth target is 96 configurations: twelve calibration natives plus seven additional arms for each backbone. Do not rerun the calibration natives in the five-epoch pilot. After successful calibration, cost the entire remaining 84-fit stage from observed timings with allowances for slower variants before appending it. The full 96-fit programme has conditional affordability; calibration is authorized as a bounded feasibility stage, not a promise that all subsequent comparisons fit. No selection on pilot accuracy, no excluded original-three backbones, and no hidden omission of failed/unsupported models.

Hypotheses for later, adequately trained comparisons: fixed algebra may improve error/parameter trade-offs at some sites; dimension need not improve accuracy monotonically; ordinary low-rank compression may explain similar gains. Keep native, controlled real and persistence baselines and compare each dimension with its corresponding low-rank budget. For a future robustness stage, freeze data splits, learning-rate choices, paired temporal error contrasts and multiplicity correction before scoring; use dependence-aware resampling over forecast origins rather than treating overlapping horizon errors as independent. This runtime stage cannot establish those hypotheses.

Verification evidence is `plans/internal-expansion/integrated-verification.json`; it records all twelve production CPU gates and all 96 actual CLI dry runs. GPU results appear only after calibration executes. `scripts/prepare_modal_l4_remaining.py --prepare` verifies the old archive, source hashes, prior allocation, completed review, cloud inactivity and fresh billing before appending this stage and freezing a new archive. The old three-model robustness stage remains explicitly superseded.

## Latest steering after seven calibrations

Read `docs/modal_l4_remaining_tuning_note.md` before advancing. The user requests hyperparameter tuning. Prioritize costing a balanced, longer two-learning-rate stage across all remaining models over automatically appending the 84 short pilot fits. Keep equal opportunities for native, real, hypercomplex and low-rank arms. Preserve all excluded backbones and the same budget. Calibrations and GPU gates must pass first; the full tuning stage is conditional on a recorded complete-stage cost decision.

SCINet's initial CUDA equivalence gate stopped before training with a maximum output difference of 0.0003024. Its repair uses equal IEEE FP32 precision for convolution and matrix multiplication in both diagnostics and every subsequent SCINet fit. No tolerances are relaxed. The prior source, failed attempt and cumulative cost remain preserved in `controller/scinet-repair-provenance`; the original integration report describes the first version, while the repair record tracks its changed inputs and targeted validation. Only one diagnosed retry is admitted, within the existing cap, before further review.
