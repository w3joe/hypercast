# All 15 TSLib backbones: seven-hour dense versus HyperDense study

Prepared 15 September 2026. Planning artifact only; no new paid jobs launched.

## Recommendation

Target a controlled, single-dataset comparison with all 15 backbones, eight arms,
equal two-rate tuning, and three fresh paired evaluation seeds. Budget **$80**
gross, with at most **10 concurrent L4 workers** and an absolute seven-hour
deadline including preparation and reporting. Expected planning envelope:
**40–60 worker-hours, approximately $37–$56 compute**, or roughly **$45–$65**
including operational overhead. These are provisional estimates, not a measured
all-model runtime forecast. Admit the full matrix only after calibration.

This can support useful within-task effect estimates. It cannot establish
universal superiority across datasets or a standard long-horizon TSLib result.
Three seeds are a practical minimum; small effects may remain inconclusive.

## Scope and current readiness

Models: DLinear, TSMixer, iTransformer, PatchTST, TimesNet, Crossformer, FreTS,
LightTS, MICN, MSGNet, SCINet, SegRNN, TimeMixer, TimeXer and FiLM.

Use **internal selected-site replacement**, preserving the surrounding backbone,
input features and external forecast head. Record each exact module path. This
tests the selected replacement site, not replacement of every dense layer.
An equivalent dense export of HyperDense weights is an inference optimization,
not an independently trained dense accuracy control.

The repository already has historical replacement implementations for all 15.
However, `native_transfer.py` and the pinned seven-channel HyperDense runner
currently admit only MICN and FiLM. Extending and checking the other 13 is real
work. Reuse existing eager module paths and control checks; freeze a new version
without changing historical experiment packages. The seven-hour target is
conditional on this integration passing in the first hour. If it does not, report
the readiness shortfall rather than claim a completed all-15 study.

FiLM's native operator is already complex. Its intervention is a separately
labelled spectral replacement, with controlled real and native reported
separately. Preserve its canonical Legendre buffers. For rectangular operators,
record the actual low-rank budget gap; the old exact-budget MICN/FiLM helper
cannot simply be applied to every shape. Preserve SegRNN's disclosed rank-one
exception where its 8D budget is below the minimum positive-rank real map.

## Data and comparison

Default to the existing **ETTh1, seven channels, OT target, context 32 hours,
horizon 5 hours**, relative-input/persistence-residual task. Retain the existing
chronological roles: train rows [0,6480), checkpoint selection [6480,8640),
development ranking [8640,11520), final test [11520,14400). Verify target windows
stay inside their assigned partitions and that all arms use the same origins.
This is a custom short-horizon experiment using TSLib-core adaptations.

Audit final-test exposure before launch; earlier reports say it was withheld,
but that must be checked against current artifacts. If exposed, label the
evaluation retrospective; do not silently invent an untouched-test claim.
Fit normalization on training data only. Supply no final-test labels to tuning
workers. Freeze all settings and the analysis before releasing the test bundle.
Do not merge old Copper or MICN/FiLM search scores into new balanced replicates.

Eight arms per backbone:

1. Native operator and initialization.
2. Controlled real dense replacement.
3. Complex HyperDense (2D).
4. Quaternion HyperDense (4D).
5. Octonion HyperDense (8D).
6. Real low-rank control for the 2D parameter budget.
7. Real low-rank control for the 4D parameter budget.
8. Real low-rank control for the 8D parameter budget.

Pair untouched initial weights, seed, minibatch order, data, forecast formulation,
precision and stopping policy. Match effective selected-map initialization
variance across controlled arms, while documenting native differences. Run
identically weighted dense forward/backward/optimizer equivalence checks,
independent algebra references, selected-site gradient checks, seven-channel
participation checks, finite-state checks and checkpoint reconstruction.

## Matrix and training

| Stage | Matrix | Fits |
|---|---|---:|
| Correctness/timing calibration | 15 models × 8 arms × 3 epochs, isolated seed | 120 short fits |
| Equal development tuning | 15 × 8 × 2 learning rates × 1 seed | 240 |
| Fresh paired evaluation | 15 × 8 × 3 new seeds, selected rate frozen | 360 |
| Total | | 600 full fits + 120 short fits |

Use Adam, MAE training, batch 32, fixed selected-weight amplitude 1, constant
schedule, learning rates {0.0003, 0.001}, maximum 150 epochs, patience 20, and
restore the best inner-MAE checkpoint. Keep optimizer details and IEEE FP32
identical to the pinned implementation. Select each arm's rate by development
MAE, with a frozen deterministic tie rule. This is equal limited tuning, not a
claim of optimal hyperparameters. Native gains are not rescaled for this grid.

Evaluation seeds are not tuning seeds. Each evaluation fit trains on the same
training partition, selects its checkpoint on inner data, and is scored on the
held-out test once after settings are frozen. Save predictions, curves, best
weights and resumable latest state including optimizer/RNG/sampler state.
Flag ceiling-hit and time-interrupted fits; do not present them as converged.

## Seven-hour schedule

| Elapsed time | Work |
|---|---|
| 00:00–00:45 | Freeze protocol/data/source; extend wrappers; local correctness checks; image/data preparation |
| 00:45–01:15 | L4 checks and timing for all 120 model/arm combinations; full-stage admission decision |
| 01:15–03:00 | Balanced two-rate development tuning; rank settings and freeze choices |
| 03:00–06:00 | Three-seed evaluation; archive and audit continuously |
| 06:00–06:30 | Finish admitted work, final scoring and matched inference measurements |
| 06:30–07:00 | Audit, uncertainty analysis, tables, report and verify workers stopped |

Use a global queue of ten single-L4 workers, one training fit per worker. Reuse
the image and staged data, cap CPU threads, and schedule predicted slow jobs
first within each stage. Do not assign one permanent GPU per model. Save each
fit independently and cap aggregate concurrency across every app involved.
Check existing Modal activity first; other GPU tasks reduce available slots.

At calibration, estimate each arm's ceiling from measured epoch/checkpoint time
with startup and runtime variability allowances. Simulate the ten-worker queue;
check both aggregate GPU-hours and the longest chain of dependent jobs.
The 240 tuning fits need approximately <=17.5 worker-hours and 360 evaluation
fits <=30 worker-hours to meet the nominal schedule. Those allowances correspond
to mean fit durations of 4.4 and 5 minutes, respectively, before extra slack.

If the full plan does not fit, prospectively choose the balanced fixed-rate
fallback **before scoring development outcomes**: use 0.001 for every arm and
run 360 three-seed evaluation fits, preserving all models and controls. Label it
fixed-hyperparameter evaluation. Do not drop slow models or inconvenient arms,
or use performance-based elimination. If even the fallback cannot fit, a
complete, adequately trained all-15 result cannot honestly be promised in seven
hours. Deliver an explicit incomplete coverage table at the deadline.

Reserve enough runtime and money for all remaining required fits before optional
work. Stop admitting jobs that cannot finish and return artifacts by 06:30;
use cooperative checkpointing before hard cancellation. Count failed work in
costs. Retry only a diagnosed transient failure when its replacement fits the
remaining time/budget. A partial seed block is reported as incomplete.

## Accuracy and reporting

Primary contrast: percentage MAE reduction of each HyperDense dimension versus
controlled real, per backbone: `100 × (MAE_real − MAE_hyper) / MAE_real`.
Positive values mean improvement. Report raw MAE, RMSE, per-lead errors, bias,
native comparison, persistence and 24-hour seasonal persistence. Report all
dimensions rather than select the best test-set dimension.

For the 45 primary contrasts, use paired seed and moving-block time resampling,
sharing sampled seeds and forecast origins across arms. Choose block length from
development error dependence, with sensitivity checks at half/double the chosen
length. Use simultaneous 95% intervals for the primary family. Mark secondary
native/low-rank contrasts separately and correct their declared family if making
inferential claims. Overlapping horizons are not independent samples, and three
seeds limit seed-uncertainty precision.

Predeclare a 2% relative MAE improvement as the practical accuracy threshold;
distinguish its point estimate from confidence in exceeding that threshold.
Nonsignificance is not equivalence. Show model-specific differences rather than
pool raw errors or claim a universal winning algebra.

Also report whole-model and selected-site parameters, checkpoint bytes, training
seconds, peak GPU memory, and synchronized batch-1/batch-32 inference timings.
Benchmark dense, HyperDense and low-rank on the same L4/precision/input residency.
Compression does not imply faster execution. Dense-export inference, if timed,
gets its own label and expanded memory count.

Deliver: 15-model coverage table, all 120 arm summaries, all seed-level results,
paired differences and uncertainty, convergence flags, predictions/checkpoints,
frozen manifest/source hashes, and resource ledger. The fixed-rate fallback omits
tuning results and clearly says so.

## Budget and evidence

Modal public standard task prices checked 15 September 2026:
<https://modal.com/pricing>. One L4 is $0.000222/second ($0.7992/hour), CPU is
$0.0000131/physical-core/second, and memory $0.00000222/GiB/second. With two
physical CPU cores and 4 GiB per worker, total is approximately **$0.9255/hour**.
Verify the actual CPU allocation semantics and billed configuration at launch.

| Resource envelope | Gross compute |
|---|---:|
| 40 worker-hours | $37.02 |
| 50 worker-hours | $46.27 |
| 60 worker-hours | $55.53 |
| All 10 workers allocated for all 7 hours | $64.78 |
| Recommended allocation including contingency | **$80** |

The $80 figure is a proposed budget, not a provider-enforced spend limit. The
controller must enforce elapsed time, allocations, outstanding reservations and
cleanup. Extra CPU/memory consumption, orchestration, storage and transfer can
add cost; use standard unpinned-region tasks and verify rates before launch.
Do not subtract promotional credits unless their remaining balance is verified.
Past allocation caps and conservative reservations are not new-run invoices.

Measured evidence: the completed ETTh1 search ledger contains 112 MICN fits
(mean 89.7 s, median 87.1 s, maximum 163.6 s) and 112 FiLM fits (mean 438.7 s,
median 414.5 s, maximum 1,118.5 s), totalling 16.44 worker-hours. It is marked
complete with $65.32 conservative accounted spend; that accounting includes
per-fit allowances and is not the measured worker-time charge. These settings
and two models cannot price the other 13 without fresh calibration.

Evidence files: `results/hyperdense-etth1-v2/search-001/ledger.json`,
`docs/hyperdense_etth1_calibration_results.md`,
`docs/etth1_native_development_results.md`, and
`docs/modal_l4_remaining_results.md`.
