# Paper-informed ETTh1 HyperDense experiment

Prepared 12 September 2026. Status: research plan, not an authorized or launch-ready job queue. The completed $19 native allocation remains closed. This document supersedes the old eight-setting/100-epoch proposal for this prospective ETTh1 study; historical protocols and results remain unchanged.

**Question:** with the successful relative/residual representation held fixed, do selected 2D, 4D or 8D internal replacements improve forecasting or the accuracy–size–latency trade-off over equally tuned native, real and low-rank models?

## What the papers actually suggest

I reviewed the methods and experimental sections below. Their settings are evidence for choices to investigate, not proven optimal settings for our custom ETTh1 task.

| Primary paper and location | Optimization or design evidence | Decision for this experiment |
|---|---|---|
| [Tay et al., Quaternion Networks, ACL 2019](https://aclanthology.org/P19-1145.pdf), §§4.1, 4.3 | Their attention experiment tunes Adam learning rates 0.001/0.0003 and batches 32/64. Translation compares partial and full replacement under common training settings; full replacement is not consistently better. | Select one internal site per backbone; tune both rates for every arm. Fix batch 32 to preserve the pilot workload. These attention settings are not presented as a forecasting recipe. |
| [Trabelsi et al., Deep Complex Networks, ICLR 2018](https://arxiv.org/pdf/1705.09792), §3.6 and speech-spectrum experiment | Derives initialization from complex weight variance; experiments use scaled semi-unitary initialization and analogous orthogonal real initialization. Speech prediction uses Adam at 0.0001. | Add 0.0001 to the common search. Control the expanded real operator's scale and log activation/gradient statistics. Our Gaussian variance control is an adaptation, not a reproduction of their semi-unitary procedure. |
| [Gaudet & Maida, Deep Quaternion Networks](https://arxiv.org/pdf/1712.04604), §III.E and §IV.A | Derives quaternion variance as 4σ². Image experiments use Nesterov SGD, gradient clipping and a scheduled learning rate; quaternion batch normalization adds training overhead. | Make component-count scaling explicit and measure actual runtime. Do not transplant image-specific SGD schedules, clipping or quaternion normalization into only the hypercomplex arms. |
| [Yakir et al., FIA-Net, 2025 preprint v1](https://arxiv.org/html/2502.19983v1), §§4.2, 5.1, B.3, D.4 | Uses Adam, initial rate 0.001, exponential decay and MSE; octonions aggregate four complex STFT windows. Appendix settings include short ten-epoch runs; dimension experiments also change window count. | Include a decaying-rate option and specify component grouping. Retain our demonstrated 150-epoch ceiling and MAE objective. Do not change window count or claim this internal-replacement study reproduces FIA-Net. |
| [Yi et al., FreTS, NeurIPS 2023](https://proceedings.nips.cc/paper_files/paper/2023/file/f1d16af76939f476b5f040fd1398c0a3-Paper-Conference.pdf), Appendix B.3–B.5 | Tunes batch size and learning rate on validation data; operates on Fourier real/imaginary components. Its NLinear ablation retains subtraction/addition of the latest value while changing the internal learner. | Keep relative/residual and observed information fixed. Separate checkpoint selection, setting selection and final evaluation. No repeat of DLinear or the other original three backbones. |
| [Zhou et al., FiLM, NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/file/524ef58c2bd075775861234266e5e020-Paper-Conference.pdf), §3 and Table 3 | Frequency-enhanced layers use Fourier processing and investigate low-rank approximation. Its projection ablations also show instability when changing the memory projection. | Preserve Legendre projections, Fourier masks and other scales. Retain low-rank baselines; our two-factor exact-budget control is not the paper's three-factor approximation. |
| [Zhang et al., PHM, ICLR 2021](https://arxiv.org/pdf/2102.08597), §3 and Appendix B | Learns component interaction rules rather than fixing them; NLI uses Adam 0.0004, and translation examines several dimensions. | Keep PHM outside the main eight arms. It is a separately costed follow-up if we want to distinguish fixed algebra from learned weight sharing. |

The literature does not establish a universally winning optimizer, dimension or grouping. None of the reviewed evidence warrants extra tuning opportunities only for HyperDense.

## Fixed task and comparison

Retain the frozen ETTh1 source and partitions in `results/etth1-transfer/prepared-v3/manifest.json`: seven channels, OT target, 32 hourly inputs, five hourly outputs, train-only scaling. Inputs are relative to each feature's latest value; the output adds the latest OT value to a zero-initialized learned correction. This is a package, not a claim about centering alone.

| Role | Raw row interval, end exclusive | Use |
|---|---|---|
| Training | [0,6480) | Gradient updates; fit preprocessing |
| Inner | [6480,8640) | Select checkpoints only |
| Development | [8640,11520) | Rank settings, assess stopping, freeze decisions |
| Final test | [11520,14400) | Once-only confirmation after all decisions |

Every target horizon must remain entirely inside its assigned partition. Historical observations before a boundary may supply context; future observations may not. The unused tail stays unused. Development has already been inspected; it is not confirmation. Final-test tensors remain absent from calibration and tuning packages. Retain the original train/inner split during confirmation rather than silently expanding the training set.

Both MICN and FiLM receive eight arms: **native, controlled real, complex 2D, quaternion 4D, octonion 8D, low-rank-2, low-rank-4 and low-rank-8**. The numbers on low-rank arms identify their matched HyperDense budgets, not their numerical ranks.

- MICN: replace only `layers.0.model.regression`, a 32→32 temporal map. Exact low-rank ranks are 8, 4 and 2. Keep all seven observed channels.
- FiLM: replace only `layers.0.model.spec_conv_1.0`, sixteen existing frequency maps with 256 complex input/output coordinates, represented as 512 real coordinates. Exact real-factor ranks are 128, 64 and 32. Keep Fourier operations, masks, Legendre memory, other scales and surrounding projections fixed. Native FiLM is already complex; controlled real is a larger unconstrained model.

For MICN, define components as equal chronological blocks of the 32 temporal coordinates. For FiLM, retain real/imaginary pairs: split the 256 latent indices into q/2 contiguous blocks and order components `[Re(block0), Im(block0), …]` for q=2/4/8; invert the output permutation. This differs from blindly interpreting the existing 512-vector as q consecutive blocks. Freeze these deterministic maps before any fitting and document the algebra multiplication orientation and basis order. They give an explicit latent grouping, not the semantic STFT windows of FIA-Net. All arms still receive identical information and return the same physical coordinates. Permuted grouping is a possible later mechanism study, not an extra setting secretly selected for a winning algebra.

Pair untouched parameter tensors, data order and dropout RNG streams across arms using named seeds. Do not equate identical random seed integers with verified tensor pairing. Native is the upstream parameterization; controlled real is the independently initialized surgery control. Include an identically weighted real-conversion equivalence gate, but do not mislabel that numerical gate as an additional fitted arm.

## Optimization protocol

Use **12 settings per arm**: initial learning rate `{0.0001, 0.0003, 0.001}` × selected-weight amplitude `{0.5, 1.0}` × schedule `{constant, exponential}`. Adam uses betas `(0.9,0.999)`, epsilon `1e-7`, zero weight decay, batch 32 and MAE loss. Exponential means `lr(epoch) = initial_lr × 0.98^(epoch−1)`, stepped after each completed epoch; gamma 0.98 is our prospective choice, not a value verified from FIA-Net. Keep dropout and architecture defaults fixed; no clipping or mixed precision in the main protocol.

This replaces the previous proposed AdamW/weight-decay axis with a schedule axis and adds the lower rate. It retains the pilot's constant-rate Adam recipe as an option. Weight decay on tied weights and on low-rank factors induces different effective regularization; a common numeric decay value does not solve that problem. MSE training and polar/unitary initialization remain separate possible matched ablations, not additional uncounted searches. Report MSE as a secondary metric regardless.

For the controlled real/hypercomplex/low-rank family, let `v = 2/(Nin+Nout)` for the expanded real map and `g` be the amplitude multiplier. Initialize controlled real and each learned hypercomplex component with zero-mean Gaussian entries of standard deviation `g√v`. An expanded HyperDense matrix entry is a signed copy of one learned entry; do not apply component-sized Xavier and inadvertently increase expanded variance with dimension. For two independent rank-r factors use standard deviation `√g × (v/r)^(1/4)` per factor: their product entries then have variance `g²v`. Biases are zero in this family and are not scaled.

Native uses its actual upstream initialization, with its selected weight tensor(s) multiplied by g and its native biases retained. In particular, g=1 preserves the native initialization rather than normalizing it away. Thus **expanded variance is matched within the controlled replacement family, not asserted equal to native**. Report native scale explicitly. All untouched weights and the residual head remain paired. For FiLM, apply this per frequency map and account for complex real/imaginary components explicitly.

Use a common **150-epoch maximum and patience 20**, strictly lower inner MAE selecting the checkpoint; restore it before development scoring. Save LR, training/inner curves, selected epoch and stale count. If any selected finalist reaches the ceiling without exhausting patience, stop before confirmation: prospectively recost an equal extension protocol for affected comparison groups rather than extending only a desired winner. A hard timeout is a failed/censored attempt, never an ordinary poor accuracy score. No automatic retries.

The zero residual head initially blocks backbone gradients. Verify exact persistence at initialization and recovery of nonzero selected-operator gradients after head updates; a finite zero gradient on the first batch is insufficient evidence that a replacement is being trained.

## Stages and fit counts

| Stage | Scope | Fits |
|---|---|---:|
| Local gates | Seven-channel surgery, variance, gradients, split isolation, checkpoint/scheduler continuation | No paid fits |
| L4 gate and calibration | One numerical gate job, then each of 16 model/arm combinations for 3 full training epochs, seed 2300; time inner forward without inner labels or scores | 16 short fits |
| Balanced search | 2 models × 8 arms × 12 settings, seed 2301 | 192 |
| Seed check / reranking | Best two settings within every model/arm, each trained with seed 2302 | 32 |
| Final confirmation | One frozen setting per model/arm × five fresh paired seeds 2401–2405 | 80 |
| Trained inference profile | All 16 selected models from predeclared seed 2401, batch 1 and 32; two bounded jobs, one per backbone | No training |

Main search plus confirmation is **304 full fits**, not the earlier 208-fit proposal. Use development MAE to pick each arm's top two settings after stage one. Break exact ties by the fixed grid order in the machine-readable proposal. After stage two, select the setting with lowest mean development MAE over seeds 2301 and 2302, again using fixed-order ties. This two-seed reranking is a budget compromise; it is not a full two-seed grid. Retain every attempted configuration and both seed scores.

Do not eliminate dimensions or one backbone from the final matrix based on tuning rankings. Freeze all sixteen selected configurations, source hashes, test scorer and contrast family before confirmation. Complete and audit all 80 training artifacts before releasing any test scores. Do not let earlier test returns alter remaining fits. If no hypercomplex arm in either backbone shows at least 2% mean development-MAE improvement versus both its selected native and real baselines, or an MAE-within-2% parameter trade-off worth studying, stop and report the null development result. This advancement rule is exploratory; it does not certify success.

## Accuracy, compression and speed decisions

Primary accuracy contrasts: each of six model/algebra combinations against its own native, controlled real, persistence and matched low-rank baseline. A useful accuracy success requires final MAE at least 2% below native and real, and below persistence and matched low rank. Average losses across the five seeds; do not pick the best seed. Retain all leads together within each origin, report every paired seed and lead, and explicitly report the one-hour persistence comparison that native FiLM previously failed. Daily seasonal persistence remains a secondary baseline.

Implement paired hierarchical moving-block resampling: resample the five seed pairs and common contiguous forecast-origin blocks, preserving all models and all five leads within each sampled block. Use 168-origin blocks as the proposed primary length, with 24 and 336 as declared sensitivity analyses; check dependence using development residuals before freezing the scorer, without inspecting test residuals. Use 10,000 replicates and seed 2501. Adjust the 24 primary directional contrasts with Bonferroni one-sided bounds (family alpha 0.05); compare ratio upper bounds to 0.98 for native/real and 1 for persistence/low rank. Validate the analysis on synthetic null and shifted-loss arrays. Approximately seventeen weekly test blocks and five seeds limit precision; if bounds are inconclusive, report that. Bootstrap intervals are approximate, not a guarantee against nonstationarity.

Report actual whole-model parameter counts and noninferiority trade-offs (MAE ratio upper bound ≤1.02 against each baseline), using the same corrected ratio bounds from the 24-contrast family rather than adding uncorrected claims. **Retain the previous ≥25% native-relative saving threshold; no current single-site arm can meet it.** MICN 8D saves 1.0843%; FiLM 4D saves 16.6636%; FiLM 8D saves 24.99546%. These can be informative Pareto points but not a 25% success. Savings versus expanded-real FiLM must be labelled separately. Do not add sites retrospectively to meet the threshold.

Profile trained checkpoints, not fresh random weights. Separate compact original execution, cached-constant inference and equivalent dense export; never train with inference-only frozen prototypes. Compare each with native/real/low-rank under the same precision and input residency. Predeclare batch-1 full-model latency as primary and batch-32 as secondary, with at least 100 warm-ups and 1,000 synchronized iterations in randomized model order over five repeated sessions. Report median/p95, device-only and host-to-device-plus-forward timings, peak allocation, resident weights and checkpoint bytes. A proposed practical speed signal is ≥10% lower batch-1 end-to-end median latency than native and matched low rank across repeated sessions while meeting the accuracy noninferiority margin. Dense export cannot claim compact resident storage. This profile covers one predeclared training seed, not a five-seed speed study.

## Implementation gates before any launch

Create versioned modules and runner rather than changing frozen `remaining_controls.py`, `remaining_spectral.py` or the completed native trainer. Required checks:

1. Native seven-channel reconstruction; identically weighted real surgery forward/input/parameter gradients and one optimizer step; independent 2D/4D/8D matrix references, including FiLM FFT boundaries and inverse permutations.
2. Exact parameter/rank budgets, untouched-tensor hashes, actual selected-site execution, finite model/optimizer state and nonzero gradients after residual-head updates. Check effective matrix variance analytically and empirically across initialization seeds, not just tiny 8D tensors once.
3. Resume equivalence including scheduler, Adam, RNG and sampler; best-checkpoint replay; batch-order pairing; intentional train/inner/development target perturbation tests that verify their different roles.
4. Explicit IEEE FP32 convolution/matmul policy, recorded library/device versions and no autocast. The previous CPU/L4 MICN discrepancies motivate a gate with a frozen tolerance, not silently widening it. Preserve the native pilot as historical evidence; new precision settings are equal across the new comparison.
5. Final-test packaging disabled until configurations and scorer are frozen; strict refusal to overwrite allocations; entire-stage budget admission, at most two L4s, no automatic retry, and artifact return for completed epochs where possible.

## Budget and operational boundary

All previous caps are closed; planning authorizes no spending. Current public [Modal pricing](https://modal.com/pricing) lists approximately $0.80/L4-hour, with CPU and RAM charged separately. For an initial conservative calculation use the prior CLI combined rate $0.9266/hour for one L4, two CPU cores and 4 GiB RAM, then refresh CLI rates before admission.

A **provisional $14 calibration cap** would cover one gate plus sixteen calibration attempts, each hard-bounded to 600 seconds: reservation per attempt `(600+120)/3600 × 0.9266 × 3 + 0.05 = $0.60596`; seventeen reservations total $10.30132; 20% margin plus $1 reserve requires $13.361584, rounded up. This bounds exposure, not a promise all gates fit ten minutes. It is not a request to launch unfinished code; revise before authorization if local preflight requires more resources/time.

After calibration, for each backbone use the slowest measured full epoch across its eight arms, including validation and checkpoint I/O, and take the maximum across variants so both arms in a comparison have sufficient common ceilings. Set timeout to `ceil_to_30s(150 × 1.5 × max_epoch_seconds + 120)`. Reserve `(timeout+120)/3600 × refreshed_hourly_rate × 3 + 0.05` per attempt, then add 20% stage margin and $1 reserve. Include the separately bounded inference-profile jobs. All failed attempts stay charged; no credit is assumed for early completion or unused old caps.

For scale only, reusing the old native timeouts for 152 MICN plus 152 FiLM fits yields $360.27 reservations, or roughly **$434 with margin/reserve**, excluding calibration and profiling. This is an uncalibrated scenario, not a quote or a bound: hypercomplex execution, stricter precision and checkpoint I/O can change it. The old $48 proposal is inapplicable. Calibrate first, then present separate development and confirmation caps. Do not start a partial grid because only some of it fits the budget.

No broader dataset, longer horizon, PHM model, extra replacement site or repeated original-three backbone is included. If this task produces a useful result, a separately frozen second dataset or longer-horizon study is needed before a general forecasting claim.

## Local evidence and proposed machine-readable scope

- [Completed native experiment](etth1_native_development_results.md)
- [Native formulation decision](../results/etth1-transfer/development-001/development-decision.json)
- [Analytic parameter inventory](../results/etth1-transfer/development-001/next-control-site-inventory.json)
- [Prospective scope and cost arithmetic](../plans/hyperdense-etth1-v2/proposal.json)

The immediate next action is local implementation and testing of these controls, followed by review of the concrete calibration launcher. No new paid work was launched while preparing this plan.
