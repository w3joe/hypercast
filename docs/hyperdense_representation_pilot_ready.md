# Native representation pilot: original preparation and authorized update

> Completed: [pilot results](hyperdense_representation_pilot_results.md). All seven jobs completed, with no failures.

> Update: the user authorized the new $6 cap and requested parallel execution. The executed bundle is `prepared-v2-parallel`, retaining v1. After the inference gate, training jobs run in waves of at most two L4s. This supersedes the one-L4/sequential launch proposal below. Source snapshots and the authorization record are retained; no prior allocation was reopened.

Prepared 11 September 2026. No paid jobs were submitted. This is a new allocation proposal; both historical allocations remain closed. The old controller is not resumed. Read-only Modal CLI checks showed two unrelated deployed apps with zero tasks and no Hypercast job. All 159 historical frozen-source hashes matched; the historical controller state SHA256 was `861e367ce9612196bfbc4f5636c46fbea0c3b3faad77267860956950dcfff3e3`.

## Concrete scope

Seven sequential jobs on at most one L4, with two CPU cores and 4 GiB RAM:

| Jobs | Scope | Timeout each | Full conservative reservation |
|---|---|---:|---:|
| 1 | Inference equivalence, latency, memory and operator attribution | 600 s | $0.6060 |
| 3 | Native MICN, two formulations per seed | 300 s | $1.1229 total |
| 3 | Native FiLM, two formulations per seed | 570 s | $1.7484 total |

Inference covers the seven historical shape/bias cases, three algebras, batches 1/32/256, and full MICN 8D and FiLM 4D graphs at batches 1/32. Original, cached-constant and dense-export paths share weights and resident inputs, use synchronized randomized repeated timings, and must pass numerical equivalence. CUDA operator traces run separately from timing. Allocator measurements report incremental live-allocation peaks with all three implementations resident; these are not isolated process-memory measurements. Compact and expanded registered storage are reported separately. Weights are fresh, not previously trained checkpoints.

The 12 training fits are MICN and FiLM × seeds 1101–1103 × levels/direct and relative/residual. Native FiLM remains explicitly labelled already complex. Shared settings: context 32, horizon 5, batch 32, Adam at 0.001, betas 0.9/0.999, epsilon 1e-7, zero weight decay, shuffled paired minibatches, at most 100 epochs, patience 20. Training and checkpoint selection both use MAE. The minimum strictly lower inner MAE selects the checkpoint (zero relative threshold). These choices resolve previously unspecified pilot settings and apply equally to both formulations; they do not reproduce the old MSE-selected scores.

The relative arm subtracts each feature's latest scaled observation from its history, retains the original target anchor, and zero-initializes only the external correction head. Untouched backbone weights and head dimensions are paired. This tests a combined representation/output/initialization package. It does not isolate these three changes causally.

## Frozen data and artifacts

[Prepared manifest](../results/hyperdense-representation-pilot/prepared-v1/manifest.json) and `development.npz` contain 1,158 training windows (targets 2015-01-02–2019-10-08, with initial context excluded) and 207 inner-development windows (2019-10-09–2020-08-10). The scaler uses training rows only. All five targets remain inside their assigned partition. The bundle has no outer or final-test tensors. Source input hashes, scaler values, feature order, dates, row bounds, source hashes and a source archive are retained. These Copper development dates were already inspected and remain exploratory.

Every successful fit archives the restored trained state dictionary, architecture and actual representation/head metadata, seed, split/scaler/source manifest, selected epoch, curves, predictions and persistence. Raw-unit MAE, MSE, bias and per-lead errors are saved. Point predictions provide no probabilistic calibration claim. Checkpoint reload predictions must match, and recomputed inner MAE must match the selected checkpoint score.

Advancement requires complete paired evidence and improvements in at least two of three seeds for **both** models, with negative median paired MAE change. Fits selecting their final executed epoch are flagged for convergence review. Numerical integrity and convergence review precede any advancement decision; no larger stage launches automatically. A disagreement between models requires revising the hypothesis, not dropping one. Independent confirmation, balanced eight-arm tuning and its AdamW/initialization grid are outside this pilot and still require separate preparation and authorization.

## Verification and remaining runtime uncertainty

15 focused tests passed: eight new pilot tests plus the seven existing optimization tests. They cover full MICN/FiLM native equivalence, paired parameter counts and initialization, exact initial persistence, relative inputs, shift equivariance, finite gradients and backbone gradient recovery after a head update, two-epoch training/checkpoint round trips, data tampering, chronological targets and scaler fitting, complete-stage budget admission, and partial worker artifacts on failure. CLI help and compilation checks passed. No research result was selected from these synthetic smoke fits.

The Modal launch path has not been executed. Local tests cover worker failure packaging and CPU fit/checkpoint logic; CUDA equivalence, CUDA training and remote artifact transport remain to be verified by the proposed paid jobs. Hard platform termination may prevent partial artifact return; its reservation remains charged regardless. No automatic retries are allowed.

## Budget and launch

Read-only `modal billing rates --json` rechecked L4 $0.8000/hour, CPU $0.0473/core-hour and RAM $0.008/GiB-hour: **$0.9266/hour** for this configuration. Historical workload estimates are unchanged. Each job reserves `(timeout + 120 seconds) × 3 × hourly rate + $0.05`, including failed attempts. The seven reservations sum to **$3.477275**; 20% margin plus a protected $1 reserve requires **$5.17273**, rounded up to a proposed **new $6 cap**. These are conservative reservations, not guaranteed invoice totals. Launch reprices and rejects a stage that no longer fits. Different or larger data requires recalibration and a new bundle/cost proposal.

After explicit authorization only:

```sh
/Users/w3joe/.local/share/hypercast4d/runtime/bin/python scripts/launch_representation_pilot.py \
  --bundle results/hyperdense-representation-pilot/prepared-v1 \
  --allocation results/hyperdense-representation-pilot/allocation-001 \
  --authorized-cap-usd 6
```

The CLI refuses an existing allocation directory, verifies frozen source/data, checks active Modal tasks, locks against a second pilot launcher, admits the full stage, and charges each reservation before starting that attempt. It stops on any failure and retains successful or caught-failure artifacts in per-job tarballs. The prior ledgers, failed costs, optimization prototypes and unrelated working-tree edits are unchanged. The periodic monitor remains untouched; no Spark configuration claim is made.
