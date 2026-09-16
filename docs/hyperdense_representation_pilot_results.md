# Authorized HyperDense pilot results

Completed on Modal L4 after authorization of a new $6 cap and parallel execution. The inference gate ran first; the six paired training jobs ran in waves of two. All seven jobs completed without retries. All twelve restored checkpoints, curves and development forecasts were archived and passed the returned-artifact audit. No outer or final-test predictions were generated.

## Representation package

Native MICN and native FiLM each received seeds 1101–1103, a levels/direct arm and a relative/residual arm. The latter changes input centering, output formulation and external-head initialization together. Both use MAE training and inner-MAE checkpoint selection, the same Adam settings, up to 100 epochs and patience 20. FiLM native is already complex.

| Model | Seed | Direct MAE / persistence | Relative MAE / persistence | Relative MAE reduction | Direct best / ran epochs | Relative best / ran epochs |
|---|---:|---:|---:|---:|---:|---:|
| MICN | 1101 | 1.1382 | 0.9744 | 14.39% | 38 / 58 | 29 / 49 |
| MICN | 1102 | 1.0889 | 0.9634 | 11.52% | 58 / 78 | 21 / 41 |
| MICN | 1103 | 1.0761 | 0.9659 | 10.25% | 28 / 48 | 42 / 62 |
| FILM | 1101 | 0.9885 | 0.9679 | 2.08% | 64 / 84 | 15 / 35 |
| FILM | 1102 | 0.9973 | 0.9693 | 2.81% | 53 / 73 | 21 / 41 |
| FILM | 1103 | 0.9893 | 0.9708 | 1.88% | 88 / 100 | 12 / 32 |

The prespecified paired development **score criterion** passed for the two-model package. This is a development screening rule, not a significance test or an accuracy claim on independent data. The full advancement gate is not yet cleared because a ceiling fit requires convergence review.

- MICN: median paired MAE reduction 11.52% across all three seeds.
- FILM: median paired MAE reduction 2.08% across all three seeds.

Training reached the 100-epoch ceiling in 1 fit(s): film/1103/levels_direct. Final-executed-epoch checkpoint flags: 0. Patience stopping is evidence of a plateau under this protocol, not proof of global convergence.

The 207 development origins span targets from 2019-10-09 through 2020-08-10 and were already inspected in earlier work. Their persistence MAE is about 0.04339. These scores also selected checkpoints. They cannot confirm an outer-period forecasting gain or establish that HyperDense improves accuracy; only native backbones were trained in this pilot.

## L4 inference

All 63 layer cases and four full-model cases passed float32 numerical equivalence. Cached constants achieved a median 1.25× layer speedup; equivalent dense export achieved 11.23×. These are medians of casewise timing ratios, not throughput estimates for an application.

| Full model | Batch | Original ms | Cached ms | Dense export ms | Dense latency reduction |
|---|---:|---:|---:|---:|---:|
| MICN | 1 | 4.110 | 3.889 | 3.639 | 11.5% |
| MICN | 32 | 4.178 | 3.894 | 3.649 | 12.7% |
| FILM | 1 | 18.647 | 17.227 | 11.692 | 37.3% |
| FILM | 32 | 17.214 | 16.338 | 11.862 | 31.1% |

The inference job uses matched fresh weights for MICN 8D and FiLM 4D, synchronized repeated timings and separate CUDA operator traces. It precedes pilot training and does not profile the trained native checkpoints. Dense export expands resident weight storage; cached constants preserve compact weights. Peak allocator increments are reported with all implementations resident and exclude external process memory. These trials do not establish an end-to-end deployment speedup or performance uncertainty across independent machines.

## Budget and next decision

Conservative accounted reservations total **$3.48 of $6**, including startup/runtime allowances. The protected $1 reserve remains intact, and $2.52 is unspent against this conservative accounting. This is not a final Modal invoice. Prior allocation ledgers and frozen source archives remain unchanged.

The pilot allocation ends here. The representation evidence can inform a separately frozen balanced eight-arm study, but this run authorizes neither that larger study nor new datasets. Independent data, optimizer/initialization grid controls, recosting and an explicit new cap are still needed before confirmation.

## Artifacts

- [Full scores, paired gate, per-lead errors and audit](../results/hyperdense-representation-pilot/allocation-001/analysis.json)
- [New allocation ledger and call IDs](../results/hyperdense-representation-pilot/allocation-001/ledger.json)
- [Frozen parallel manifest](../results/hyperdense-representation-pilot/prepared-v2-parallel/manifest.json)
- [User authorization record](../results/hyperdense-representation-pilot/authorization.json)
- `allocation-001/job-00.tar.gz` contains inference results and CUDA traces; `job-01.tar.gz` through `job-06.tar.gz` contain the twelve trained checkpoints and diagnostics.
- `prepared-v1` and `prepared-v2-parallel` retain their separate frozen source archives.
