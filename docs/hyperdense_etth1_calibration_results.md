# HyperDense ETTh1 calibration results — 12 September 2026

The corrected calibration completed on **two Modal L4s in parallel**. All 16 configurations passed their three-epoch training, finite-state and exact same-device checkpoint-replay checks. Both independent GPU gates passed; untouched-state hashes match across workers. This establishes implementation readiness and timing evidence, **not forecasting superiority**. Inner evaluation inputs were forwarded for timing, but inner, development and test labels were absent from the calibration package.

## Scope and controls

MICN and FiLM each used native, controlled real, 2D, 4D and 8D HyperDense, and three corresponding parameter-matched low-rank arms. All use seven ETTh1 channels, OT target, context 32, horizon 5, relative/residual formulation, seed 2300, full selected-weight amplitude, Adam learning rate 0.001 and exponential decay 0.98. Three epochs per fit; no accuracy-based calibration selection. Historical backbones were not repeated.

The GPU gates covered real replacement forward/backward/Adam equivalence, independent expanded-matrix references for every algebra, spectral grouping, persistence initialization, gradient recovery, parameter counts, untouched-state pairing and CPU/CUDA numerical agreement. Continuation equality was checked on a synthetic dropout model; exact full-backbone cross-device resume trajectories are not established. The local control suites passed 17 tests, including development-target isolation and two-seed ranking logic.

## Integrity issue and repair

The initial gate and 12 short fits returned successfully. The archive audit then found a FiLM real/native untouched-state hash mismatch. Reconstructing initial states isolated this to host-dependent rounding in fixed SciPy-generated Legendre matrices (maximum absolute discrepancy about 5.96e-8), not learned-weight initialization. Training was stopped before two reserved jobs were submitted.

Canonical fixed buffers from the initial native checkpoint were frozen in `src/hypercast4d/_constants/film_legendre_v1.pt` (SHA256 `643b6e3339570a3668ac8c0f2d0ad95fff9ba9b0b61fc330ffe7b948b76ad491`). They are non-trained buffers. New pinned modules preserve the original frozen implementation. Two explicitly bounded repair jobs each ran a full gate and eight fits. Their returned archives passed the complete audit, including exact canonical-buffer and cross-worker hash checks. The 12 earlier fits remain archived and marked superseded; they are not part of the corrected 16-fit result.

## Measurements and accounting

The table reports the maximum combined training, inner-input forward and checkpoint-write time among each corrected fit's three epochs. These are short training calibration measurements, not standalone inference benchmarks.

| Backbone | Arm | Whole-model parameters | Max epoch seconds |
|---|---|---:|---:|
| film | native | 6,292,599 | 6.579 |
| film | real | 8,389,751 | 8.612 |
| film | complex | 6,292,599 | 10.557 |
| film | quaternion | 5,244,023 | 10.526 |
| film | octonion | 4,719,735 | 15.303 |
| film | lowrank2 | 6,292,599 | 12.497 |
| film | lowrank4 | 5,244,023 | 12.366 |
| film | lowrank8 | 4,719,735 | 12.474 |
| micn | native | 82,636 | 1.945 |
| micn | real | 82,636 | 1.947 |
| micn | complex | 82,124 | 1.905 |
| micn | quaternion | 81,868 | 1.950 |
| micn | octonion | 81,740 | 2.881 |
| micn | lowrank2 | 82,124 | 2.511 |
| micn | lowrank4 | 81,868 | 2.511 |
| micn | lowrank8 | 81,740 | 2.505 |

Both corrected workers returned successfully (145.24 and 212.38 seconds inside the worker functions). CLI verification confirmed the app stopped with zero tasks.

Conservative accounting is **$10.30132 against the $14 cap**, retaining all 17 reservations: the original gate, 12 superseded fits, two unsubmitted reservations and two corrected batches. This is conservative resource accounting, not a final provider invoice. With the planned 20% margin and $1 reserve, admission totals $13.361584. The allocation is closed; unused capacity is not new authorization. All earlier allocations, failed-attempt costs and source hashes remain preserved.

## Concrete next stage — prepared, not launched

The frozen package `results/hyperdense-etth1-v2/search-prepared-v1` contains a tested launcher, 76 frozen source/artifact files, a label-controlled development bundle, and 224 job templates:

- 192 fits: both backbones × eight arms × 12 equal settings, seed 2301.
- 32 fits: the top two settings for every backbone/arm repeated at seed 2302; mean development MAE across both seeds selects one setting, ties resolved by setting ID.
- Grid: learning rate {0.0001, 0.0003, 0.001}, selected-weight amplitude {0.5, 1}, constant/exponential schedule. Adam, MAE, batch 32, at most 150 epochs, inner-MAE patience 20, explicit IEEE FP32.
- At most two L4s, no automatic retries, reservation before submission, drain both workers and stop future waves on failure. Best/latest checkpoints, optimizer, scheduler, RNG, curves and development forecasts are archived. Test data is absent.

**The required new cap is $547 for search and reranking.** This is a conservative admission ceiling, not an expected invoice. It uses $0.9266/hour for one L4 plus two CPUs and 4 GiB, the slowest observed epoch per backbone (including superseded timing observations to retain host variability), a 1.5× epoch allowance, 120-second overhead, 3× resource reservation, then 20% stage margin and $1 reserve. Common per-fit timeouts are MICN 1,320 seconds and FiLM 3,570 seconds. Rates and frozen hashes must be rechecked at launch.

A later 80-fit five-seed confirmation stage is provisionally **$196**, subject to recosting after search. Confirmation scoring and trained-checkpoint inference profiling remain separate work; their complete implementation and inference budget are not included in this launch package. Neither stage is authorized by the calibration cap. No accuracy claim or final-test selection is justified by this calibration.

## Audit artifacts

- `results/hyperdense-etth1-v2/calibration-001/ledger.json`: all original and corrected reservations.
- `ledger-before-manual-repair.json` and `integrity-issue-and-manual-repair.json`: preserved stopped state and diagnosis.
- `job-15.tar.gz`, `job-16.tar.gz`: corrected gates, 16 trained checkpoints and timing records.
- `pinned-analysis.json`: full archive/source/control audit and conservative costing.
- `completion.json`: closed allocation and next-package hashes.
- `results/hyperdense-etth1-v2/prepared-calibration-v2-pinned`: corrected frozen calibration bundle.
- `results/hyperdense-etth1-v2/search-prepared-v1`: next-stage frozen source, data and plan.

The original paper-informed plan remains at `docs/hyperdense_etth1_paper_informed_plan.md`. This report records the implementation repair and measured costs without rewriting that historical proposal.
