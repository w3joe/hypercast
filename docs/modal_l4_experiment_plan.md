# HyperDense 2D / 4D / 8D experiment plan

Status: programme stopped at the conservative budget guard on 11 September 2026. Pilot, 54 controlled development fits and 162 robustness fits completed; 50/54 locked retrospective final fits completed, with two iTransformer low-rank controls blocked before submission. See `modal_l4_results.md` for conclusions and limitations, and `modal_l4_operations.md` for the preserved state. The proposal below records the original scope; later frozen stage documents supersede its run counts under the $20 limit.

## Authorized operating constraints

Latest user instruction: total Modal budget reduced to USD 20; use GPT-5.3-Codex-Spark for setup and ongoing supervision. This supersedes the earlier USD 100 allowance.

The user has selected this Mac as the controller and will keep it awake. The total Modal budget for the entire programme is **USD 20**, including pilot, development/tuning, confirmation, retries, follow-up experiments, CPU, memory, storage and other Modal charges. This is not a per-stage budget. Codex usage is separate from this Modal allowance.

The agent may select and run scientifically justified follow-up experiments after completing the necessary planned comparisons, within the remaining budget and the one-L4 concurrency limit. It should prioritize unresolved questions, robustness checks and independent replication over simply adding more trials. Record the hypothesis, control, expected information gain, estimated cost and stopping criterion before each follow-up. Preserve negative and inconclusive results. Do not tune against the locked test set or silently alter completed confirmatory comparisons; follow-ups using already-seen outcomes are exploratory until independently confirmed.

The initial 9,450-fit matrix is a proposal subject to the pilot's cost estimate. If it will not fit within USD 20, the agent should revise and document a balanced design before confirmation, preserving all three dimensions and essential real-valued controls. Spend is a hard constraint; exhausting the budget is a stopping condition even if scientifically useful experiments remain.

Implement budget enforcement before cloud execution: persist cumulative incurred/estimated charges and reserve the conservative maximum cost of each in-flight job, including retries and shutdown overhead. Do not launch a job whose reservation would exceed the remaining allocation. Use conservative per-job runtime limits and leave a billing uncertainty reserve; reconcile with available Modal billing records. Delayed billing means a local estimate alone cannot guarantee a dollar-exact cap. Stop submissions and cancel work at the conservative safety threshold. An additional budget requires explicit user authorization.

Credential setup is local to the Mac: authenticate with the Modal CLI, which stores credentials in the user's home configuration (`~/.modal.toml`). Never put token values in this plan, experiment YAMLs, the repository, logs, or chat. Monitoring should start after a resumable runner and budget enforcement are ready; no active monitor is implied by this document.

## Questions and scope

Measure separately whether HyperDense improves forecasting accuracy, reduces parameters at comparable accuracy, or improves measured GPU time/memory. A parameter reduction is not evidence of faster computation. Compare 2D complex, 4D quaternion, and 8D octonion without changing their real input/output widths.

The primary claim is about the existing four-series dataset and these architectures. Neither more random seeds nor more forecasting windows establishes general superiority across datasets. A broad claim needs a separately preregistered replication on independent datasets.

The experiment must allow four conclusions: practically better, practically equivalent, worse, or inconclusive. It cannot promise an exact universal answer.

## Controlled model matrix

Use three existing CLI presets: `tslib-dlinear`, `tslib-tsmixer`, and `tslib-itransformer`. Keep each backbone and forecast head unchanged within its comparison. Test the same nine variants on each backbone: 27 configurations.

For the primary comparison, use a shared frontend:

`4 inputs → Dense(32) → ReLU → tested 32-to-32 map → ReLU → backbone → existing head`

The learned lift makes 8D possible without adding information. Apply it to every compared map. The ReLU between lift and map prevents two adjacent linear layers collapsing into one linear map. This is an experiment on algebra structure in a learned representation, not eight observed physical variables.

| Variant | Tested frontend | Purpose |
|---|---|---|
| Original | No frontend | Whether the modification helps the existing model |
| Lift only | Dense(32), ReLU | Whether lifting alone explains improvement |
| Real dense | Dense(32), ReLU, Dense(32), ReLU | Same real width and nonlinear depth |
| Complex | Shared frontend with complex `units=16` | 2D |
| Quaternion | Shared frontend with quaternion `units=8` | 4D |
| Octonion | Shared frontend with octonion `units=4` | 8D |
| Real rank-8 | Shared frontend; tested map Dense(8) → Dense(32) | Approximate 2D parameter budget |
| Real rank-4 | Shared frontend; tested map Dense(4) → Dense(32) | Approximate 4D parameter budget |
| Real rank-2 | Shared frontend; tested map Dense(2) → Dense(32) | Approximate 8D parameter budget |

Do not insert an activation between the two low-rank factors. Use the same final ReLU as the other tested maps. These are alternative compression controls, not mathematically equivalent models.

At width 32, the tested map has these parameter counts, including biases:

| Map | Parameters |
|---|---:|
| Real dense | 1,056 |
| Complex | 544 |
| Quaternion | 288 |
| Octonion | 160 |
| Real rank-8 | 552 |
| Real rank-4 | 292 |
| Real rank-2 | 162 |

The low-rank comparisons are within 1.5% of the corresponding layer budget using existing bias-enabled CLI layers. Report the mismatch rather than calling them exact matches. Report total-model parameters as well: a large backbone/head can make frontend savings small.

This fixed-width comparison deliberately changes parameter count with dimension. It does not isolate dimension independently of compression. The compressed real controls and width sensitivity experiments are necessary to interpret that tradeoff.

## Required preparation before confirmatory runs

1. Freeze the actual working tree, including currently uncommitted code, dependency versions, dataset checksum, Modal image, all YAMLs, and the analysis specification. A Git SHA alone would not identify the current source.
2. Add a configurable controlled initialization for the experiment. Current HyperDense uses component-wise Glorot normal; real Dense uses Glorot uniform. For equal real widths, the HyperDense effective matrix variance grows with component count. Use an explicitly matched effective weight distribution/variance, zero biases, and a common activation-gain convention across all four maps. Scaling current component weights by `1/sqrt(d)` addresses variance but does not by itself match uniform versus normal distributions. Keep the released initialization available for a separate sensitivity run; do not silently change the paper presets.
3. Pair initialization of shared lifts, backbones, and heads across compared candidates. Identical global seeds are insufficient because different layer constructors consume different numbers of random values. Use independent named RNG streams or copy the common initial state, and pair minibatch ordering. Different tested maps need not have identical weights.
4. Validate algebra forward/gradient behavior, component-major packing, every planned shape, and measured initialization variance. Check both CPU and CUDA on a small preflight; no extra tuning from test data.
5. Record CUDA-synchronized training/inference timing and peak allocated GPU memory if these are not already exported. Existing process RSS is not peak GPU memory. Benchmark inference at fixed batch sizes with warm-up, recording both training time and end-to-end billed job duration.

The opt-in `--eval initialization=matched-v1` protocol now implements items 2–3 for the sequential experiment architecture; `--eval benchmark=true` enables item 5. CPU checks cover all 162 planned architecture/cell combinations; a bounded CUDA preflight gates controlled development. See `modal_l4_controlled_stage.md` for exact conventions and provenance. Earlier pilot runs retain their original initialization and interpretation.

## Data and training protocol

- Primary target: Copper; four observed input columns, identical ordering and preprocessing for all variants.
- Use the existing `robust` chronological folds: train 0–55%, validate 55–65%; train 0–65%, validate 65–75%; train 0–75%, validate 75–85%. Fit normalization on each training prefix. Require complete forecast targets inside each split.
- Reserve the last 15% for a single locked final evaluation. The existing final-test command refits on the first 85%. Audit previous experiment exposure: this workspace already contains results, so this tail must not be described as historically unseen without checking. If it has informed prior decisions, call this a locked retrospective test and obtain new data for prospective confirmation.
- Primary six window/horizon cells: `10/1,20/1,20/5,40/5,60/10,60/20`. These span short/long context and near/far forecasts. Additional cells are sensitivity analyses, not additional independent datasets.
- Adam, MSE loss, batch size 32, full precision, identical shuffle policy, no early stopping. Use 150 epochs as the planned fixed training budget. Inspect development-only convergence curves; if 150 is insufficient, revise the common budget and cost estimate before confirmation, never only for the apparent winner.
- Equal tuning budget: learning rates `0.0003,0.001,0.003`, three development seeds `7,19,31`, four development cells `10/1,20/5,60/10,60/20`, all three folds. Select one learning rate per configuration by equal-cell mean MAE/persistence. Break ties within 0.1% by choosing the lower rate.
- Confirmation seeds: `101,211,307,401,503,601,701,809,907,1009`. Freeze them in advance. Reusing validation dates with new seeds assesses optimization stability; it does not create an independent temporal holdout.
- Never remove an algebra family because its early results look weak. Failed/nonfinite fits are recorded, investigated, and included in reliability reporting; they are not silently replaced with favorable seeds.

## Execution stages on one L4

| Stage | Fits | Purpose |
|---|---:|---|
| Pilot: 27 configurations × 2 cells × 1 seed × 1 fold, 10 epochs | 54 | CUDA, artifacts, runtime and resource validation |
| Equal tuning: 27 × 3 rates × 4 cells × 3 seeds × 3 folds | 2,916 | Freeze rate per configuration |
| Robust validation: 27 × 6 cells × 10 seeds × 3 folds | 4,860 | Stability, regime and horizon comparisons |
| Locked final test: 27 × 6 cells × 10 seeds | 1,620 | Confirm all preregistered controls and dimensions |
| Base programme total | 9,450 | Excludes preparation and sensitivity runs |

All 27 configurations enter the final comparison with their frozen tuning choices; do not select only a winning algebra and then omit its controls. Final-test jobs are invoked once per frozen candidate. The CLI's test-once guard is per candidate hash, so procedural discipline must also prevent repeated test-driven modifications under new hashes.

Run exactly one Modal L4 job at a time, using blocking CLI submissions. Do not use concurrent shells or detached fan-out. Trial execution is serial in the current Modal worker. Use a manifest-driven wrapper around individual CLI commands; the CLI does not provide a sweep command. The wrapper should record job IDs, wait for terminal status, export artifacts, and resume by skipping verified completed entries.

Use bounded batches of seeds if a job could approach the worker's 24-hour timeout. Account for additional container starts. Verify cancellation of an old job before retrying or submitting the next one.

The authorized currency budget is USD 20. Do not quote expected GPU-hours or promise completion of the entire matrix before the pilot. Estimate each stage from observed seconds/epoch for each backbone, cell and variant, including container startup, data transfer and CPU/memory charges. Report an uncertainty range and compare actual cumulative usage with the estimate. Verify current Modal prices and configure conservative runtime and spending limits before the pilot.

## CLI workflow

The following commands describe future execution. Planning verified `model list`, `layer types`, and dry-runs for all three HyperDense dimensions on all six cells. Each ten-seed robust dry-run reported 180 trials and CUDA/L4 execution. No Modal call was made.

The working executable in this session is `/tmp/hypercast4d-runtime/bin/hypercast`; the workspace `.venv` is broken. Before cloud execution, create a durable environment with the optional Modal dependency and authenticate it. Use `hypercast` below once that environment is active.

Example construction, repeated for each backbone and candidate:

```sh
hypercast model create tslib-tsmixer --out tsmixer-4d.yaml
hypercast layer insert tsmixer-4d.yaml --before core --type dense --id input_lift --set units=32
hypercast layer insert tsmixer-4d.yaml --before core --type activation --id lift_activation --set kind=relu
hypercast layer insert tsmixer-4d.yaml --before core --type hyper_dense --id test_layer --set algebra=quaternion --set units=8
hypercast layer insert tsmixer-4d.yaml --before core --type activation --id test_activation --set kind=relu
hypercast model validate tsmixer-4d.yaml --cells 10/1,20/1,20/5,40/5,60/10,60/20
```

For 2D use `complex` with 16 units; for 8D use `octonion` with 4 units. Use distinct filenames. Apply the controlled initialization settings after their planned implementation; the commands above alone retain current initialization.

Preflight a frozen confirmation request:

```sh
hypercast experiment run tsmixer-4d.yaml \
  --preset robust --epochs 150 --batch-size 32 --learning-rate 0.001 \
  --cells 10/1,20/1,20/5,40/5,60/10,60/20 \
  --seeds 101,211,307,401,503,601,701,809,907,1009 \
  --target modal --gpu L4 --dry-run --json
```

Here `0.001` is an example; use the frozen rate selected for that configuration. To execute later, replace `--dry-run` with `--yes`. Omit `--gpu-count`: the Modal CLI path rejects that setting, even when it is 1; `--gpu L4` selects one GPU.

```sh
hypercast job status JOB_ID --follow
hypercast job logs JOB_ID --follow
hypercast job results JOB_ID --out exported-results/JOB_ID
hypercast experiment final-test VALIDATION_JOB_ID --yes
```

Final-test inherits the parent's execution target. Validate it remains Modal/L4. Custom folds are rejected; use the built-in `robust` preset. Supply an experiment-specific `--results-root` consistently for submission and inspection. Export requests, status, runs, predictions, diagnostics, logs and compute metadata for every job.

## Analysis and decision rules

Primary outcome: equal-cell average of MAE divided by persistence MAE. Compare models within the same backbone, cell, fold, seed and forecast origin. Report absolute MAE, MSE, per-lead error, failure rate, total parameters, peak GPU memory, training time and inference latency separately. Report each backbone separately before any equal-backbone aggregate.

Define relative improvement as `100 × (control MAE − candidate MAE) / control MAE`. Preregister 2% as the minimum practically meaningful accuracy difference, with a 5% sensitivity threshold. These are proposed utility thresholds, not properties of the dataset.

- Accuracy benefit: the multiplicity-adjusted interval supports more than 2% improvement against the same-width real map. A useful forecasting model should also improve on persistence.
- Compression benefit: use a one-sided noninferiority comparison with a 2% margin against the same-width real map, alongside measured whole-model savings. Report whether it also beats the corresponding compressed real control.
- Dimension difference: compare complex/quaternion/octonion directly, with the same criteria, and disclose their different parameter budgets.
- Practical equivalence: the entire adjusted interval lies inside ±2%. A nonsignificant difference is not equivalence.
- Otherwise report inconclusive or a supported degradation; do not rank tiny noisy differences as wins.

Use saved origin-level predictions for paired temporal block resampling, averaging repeated-seed losses rather than treating every seed as new market data. Resample the same dates jointly across compared models and cells; keep folds separate for validation summaries. Choose block lengths from development loss dependence, with a lower bound at least the largest horizon, and preregister half/double-length sensitivity. Use 10,000 resamples. Report seed variability separately and optionally resample seeds jointly as another uncertainty component.

Do not perform an ordinary independent-sample t-test over thousands of overlapping forecasts or count seeds × cells × folds as independent replications. Block-bootstrap intervals depend on temporal assumptions and may be unstable with short or shifting regimes; report effective block counts, regime-specific results and inconclusive intervals where appropriate. See [Politis and Romano, The Stationary Bootstrap](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870).

Preregister the confirmatory family: three hyper-versus-real, three hyper-versus-corresponding-low-rank, and three pairwise dimension comparisons per backbone, plus the three hyper-versus-persistence comparisons per backbone: 36 contrasts. Use simultaneous bootstrap intervals or a prespecified family-wise correction; report unadjusted effect estimates too. Original/lift-only comparisons are explanatory unless separately added to this family before test access.

## Sensitivity and broader mechanism checks

Run these after development results and before unlocking the final test if they will affect confirmatory claims. Their counts and hypotheses require an explicit extension of the frozen manifest and budget.

1. **Width sensitivity:** repeat real/2D/4D/8D at widths 16 and 64. Keep real widths equal within each comparison. This distinguishes an effect at width 32 from a consistent accuracy/size curve.
2. **Internal replacement:** in TSMixer, expand the model and replace one declared 32→32 internal channel Dense with real/2D/4D/8D maps, preserving every surrounding operation. All four arms use the same expanded graph. This tests whether a result transfers beyond an added input adapter. Obtain actual node IDs through `model expand` and `layer list`; do not invent them.
3. **Initialization sensitivity:** compare current released initialization with the controlled initialization. If an advantage disappears after matching scale, attribute it accordingly.
4. **Algebra versus generic tying:** compare against random signed weight-tying patterns with equal parameter count, fan-in and scale. This control needs implementation and multiple frozen pattern seeds. Low-rank controls alone cannot establish that the multiplication table itself is special.
5. **Alternative multiplication tables:** `split_complex`, `coquaternion`, and `cl11` at the same widths, as secondary algebra comparisons. Do not pool them into a dimensional winner after inspecting results.
6. **Input-order sensitivity:** freeze several raw-feature permutations, keeping the target consistent, and verify component-major layouts. Use identical permutations for every candidate. Learned lifting reduces the direct interpretation of raw grouping; report that limitation.
7. **Other targets:** repeat a reduced frozen matrix on FCX, CLP and SCCO using `--eval target_column=...` after verifying exact workbook column names. These are correlated targets from the same dataset, not independent replications.
8. **Independent datasets and synthetic controls:** add at least two different real-world domains, using predefined four-channel inputs under the current loader, plus controlled synthetic processes with and without cross-channel dependence. Full multivariate or native eight-observed-channel claims require extending the four-column loader. Train-only preprocessing, chronological holdouts, and identical tuning budgets remain mandatory.

The final report should contain paired effect sizes with uncertainty, accuracy-versus-total-parameters and accuracy-versus-measured-cost plots, fold/horizon breakdowns, convergence and failure diagnostics, and a direct answer for each dimension. Distinguish evidence for the current implementation, parameter sharing, the specific algebra, and transfer across datasets.
