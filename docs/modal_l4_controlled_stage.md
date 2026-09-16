# Controlled development decision — 10 September 2026

This decision supersedes the initial proposal's development matrix for the authorized $20 programme. The 27-configuration implementation pilot completed all 54 intended fits plus one duplicate two-fit job. At this decision point the controller conservatively accounts for $2.211296; the delayed Modal meter has risen $0.13 since the original baseline. Those are distinct quantities. Preserve the original ledger and billing baseline.

## Question and scope

Does a matched-initialization HyperDense frontend improve accuracy or compression relative to the same-width real map and the corresponding low-rank control, under an equally allocated training recipe? Keep original and lift-only baselines to reveal the cost of adding the frontend itself. The pilot's apparent wins do not select which families continue: all 27 configurations remain.

This stage is exploratory development on the existing Copper dataset. No final-test command is authorized by this manifest. The previous pilot already exposed the standard validation period. The final 15% remains closed to this controller; its historical exposure in other workspace studies must be audited before it is described as unseen.

## Frozen next-stage matrix

- Three backbones: DLinear, TSMixer, iTransformer.
- Nine variants per backbone: original, lift only, real dense, complex 2D, quaternion 4D, octonion 8D, real rank-8, rank-4, rank-2.
- Equal learning-rate grid: 0.0003 and 0.001. The previously proposed 0.003 arm is omitted for every configuration to preserve budget; the pilot showed optimization instability at 0.001, so a lower rate is worth testing.
- One development cell: window 20, horizon 5. This retains the measured 20-step context while adding a five-step forecast, rather than spending all development budget on one-step prediction.
- One development seed: 19, distinct from pilot seed 7.
- Standard chronological fold: train first 70%, validate next 15%; preprocessing fitted on training only.
- 150 fixed epochs, batch 32, Adam/MSE, full precision, shuffled training order paired by seed. No early stopping and no best-checkpoint restore. Save every epoch's losses; compare final-epoch performance consistently.
- Total: **54 development fits in 54 sequential jobs**. No candidate filtering.
- Preceded by one bounded CUDA preflight job: synthetic forward/gradient/variance and all 162 architecture/cell shape checks (27 configurations × six planned cells), paired shared-state verification, then one 1-epoch DLinear smoke fit. Development jobs depend on verified preflight completion.

Choose one rate per configuration by validation MAE/persistence in this common cell. If relative scores differ by at most 0.1%, choose the lower rate. Do not use test data or select different epochs per apparent winner. If any arm fails, retain the failure and diagnose it; at most one automatic retry is allowed, and all attempts cost budget. Do not interpret an incomplete matrix as balanced confirmation.

## Implemented controls

`--eval initialization=matched-v1` enables a separate initialization protocol. Default and released/paper initialization remain available and unchanged when the option is absent.

Real dense and HyperDense tested maps use zero biases and normal entries with expected effective variance `2/(real_input_width + real_output_width)` and gain 1. For the 32-to-32 comparison that variance is 1/32 for all four map types. The two linear low-rank factors use symmetric standard deviation `(1/(32*rank))**0.25`, matching the product's effective entry variance in expectation. The product distribution is not normal, and structured weight correlations necessarily differ between maps.

Independent SHA-256-derived CPU RNG streams use protocol, seed, window, horizon, fold and semantic layer role. Shared input lifts, native backbone blocks and forecast heads have identical starting state across variants when their shapes match, verified by saved state digests. Native upstream initialization is reconstructed under its own stream; it is not replaced with a different initializer. Original four-channel backbones/heads cannot be elementwise paired with the lifted 32-channel versions. Model construction does not perturb the controlled streams, and the data loader uses a separate seeded generator.

`--eval benchmark=true` records synchronized training wall time, peak allocated GPU memory, and inference latency at batch 32 with 10 warmups and 30 repetitions. Inference input is resident on the device, so latency excludes transfer. These timing fields are not Modal billed duration. Full job wall time, returned remote elapsed time, actual GPU, image ID and package versions are recorded separately. Nonfinite losses fail visibly.

Save `initialization.json`, `learning_curves.csv`, `runtime.json`, and normal predictions/results for every job. The preflight also saves `preflight.json`. Matched initialization plus changed seed/epochs is a controlled new protocol; differences from the old pilot cannot be attributed solely to initialization.

## Cost and stopping rules

Extrapolating each configuration's window-20 pilot training time from 10 to 150 epochs gives about 2,142 training seconds over both rates. At the verified L4+2 CPU+4 GiB rates, with a 3x factor, 30 seconds of overhead per job and an extra $0.05/job allowance, the development estimate is **$5.61**. Doubling the extrapolated training duration gives **$7.26** under the same conservative accounting. Neither estimate is a guaranteed invoice or duration forecast for the new five-step horizon.

The preflight remote timeout is 600 seconds; development timeouts are 900 seconds per fit. Admission reserves each job's full timeout plus 120 seconds of overhead at the 3x accounting factor. The global limit remains $20 with $1 held back; the controller never resets prior spending. Remote retries are disabled. Billing uncertainty, a changed frozen source/data/spec input, unresolved orphaned jobs, or preflight failure prevents new submissions. A budget stop takes precedence over completing the matrix.

## Subsequent stage, contingent on measured cost

After all development arms finish, review failures/convergence and freeze the learning-rate choices. A provisional affordable robustness matrix is all 27 configurations × the same 20/5 cell × seeds 101 and 211 × all three built-in robust chronological folds: **162 fits at 150 epochs**. This tests seed and temporal-regime stability but narrows the original multi-horizon claim. Cost this matrix using measured development timings before committing it; preserve all dimensions and controls.

If the budget supports it, perform the locked retrospective final comparison for all 27 configurations and both frozen seeds (54 fits) only after the full validation protocol and analysis family are frozen and historical test exposure is audited. Add further horizons or internal replacement as explicitly documented exploratory follow-ups only if remaining budget permits. Otherwise report limited scope and inconclusive evidence rather than overspending or discarding losing families.

## Provenance

`results/modal-l4-runner/controller/controlled-provenance/` stores the actual source snapshot (including uncommitted code), dataset, exact specs, plan, local dependency versions, CPU preflight and SHA-256 manifest. Every submission verifies the frozen input hashes. Remote Python is 3.12 and image dependencies are pinned to exact versions; each returned runtime records actual versions and the Modal status records image ID. Historical pilot artifacts are preserved with their original requests and are not relabeled as having controlled provenance.
