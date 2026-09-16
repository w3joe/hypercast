# ETTh1 calibration results and prepared native development stage

The authorized single L4 calibration completed successfully. All four configurations—MICN and FiLM, each with levels/direct and relative/residual—ran three training epochs on 6,444 windows. The inner forward pass was timed without using inner targets or producing selection scores. Development and final-test targets were not scored. Calibration seed2100 is excluded from the proposed development comparisons.

All four checkpoints were downloaded and audited for finite model tensors, expected architecture/data metadata, optimizer state, sampler state, and Python/NumPy/Torch/CUDA RNG state. The worker stopped normally, and a subsequent Modal CLI check showed no active tasks. The new $2 allocation conservatively accounts for **$0.60596**; $1.39404 remains against that accounting, including the protected $1 reserve. This allocation is complete; it will not fund the next stage. Accounted reservations are not final invoice totals.

## Measured timing

| Model | Formulation | Training seconds/epoch (3 measurements) | Inner forward seconds/epoch |
|---|---|---|---|
| MICN | Levels/direct | 3.603, 1.935, 1.963 | 0.176, 0.172, 0.176 |
| MICN | Relative/residual | 2.001, 1.861, 1.998 | 0.175, 0.195, 0.177 |
| FiLM | Levels/direct | 6.391, 6.174, 6.153 | 0.556, 0.529, 0.530 |
| FiLM | Relative/residual | 6.153, 5.995, 6.088 | 0.552, 0.555, 0.556 |

MICN has 82,636 parameters; native FiLM has 6,292,599. The two formulations have identical parameter counts within each backbone. Maximum measured GPU allocation was about 28.3 MiB for MICN and 191.6 MiB for FiLM; this excludes process memory, other GPU allocations and serialization buffers. These are calibration timings, not accuracy results.

## Concrete next launch scope

Twelve native development fits: two models × two formulations × seeds2201–2203. All seven observed ETTh1 channels remain present, OT is the target, context is32 hours and horizon is5 hours. This is a custom short-horizon task. Training uses Adam at0.001, betas0.9/0.999, epsilon1e-7, no weight decay, batch32, MAE loss, maximum150 epochs, patience20 and strictly lower inner-MAE checkpoint selection. Development MAE compares the packages only after checkpoint selection; persistence and24-hour seasonal persistence are both reported. Final-test tensors are absent from the supplied bundle.

Jobs run in waves of two L4s, with a common timeout for both formulations within each model. There are no automatic retries or automatic larger-stage launches. A failure stops future waves after collecting the other active result. A soft deadline preserves completed-epoch state where possible; a hard platform termination can still prevent partial artifact return and remains charged.

The new training implementation saves an atomic latest resumable checkpoint after every completed epoch, including model, Adam, RNG, sampler, best-state and stopping history. A separate best inference checkpoint is restored and its inner-MAE score recomputed before development scoring. Every result includes predictions, both baselines, MAE/MSE/bias/per-lead errors, selected/executed epochs, stopping status, parameter count and checkpoint hashes.

Local verification passed eleven focused tests across the native wrapper and transfer trainer. Tests include exact four-channel graph/input-gradient equivalence, all-seven-channel gradients, persistence initialization, split boundaries and daily-baseline causality, calibration checkpoint serialization, exact CPU dropout-training continuation across a checkpoint, rejection of changed resume metadata/nonfinite loss, and invariance of trained/selected weights when development labels change. Three transfer-training tests were rerun after adding the full stage plan to checkpoint metadata. CLI help passed. CUDA training was exercised by calibration; the new full development runner's remote transport and exact CUDA-resume equivalence remain untested. The prior Copper frozen implementation was not changed.

## Admission estimate

Use the maximum observed training-plus-inner-forward epoch across both formulations and all three calibration epochs, including warm-up overhead. Multiply by150 epochs and1.5 for runtime uncertainty, add120 seconds per-fit overhead and round timeout upward to30 seconds. Reserve timeout plus120 seconds at three times the verified combined resource rate ($0.9266/hour), plus$0.05 per attempt. The allowance also needs to cover checkpoint I/O and development scoring, which the timing calibration did not directly benchmark.

| Model | Max measured combined epoch | Timeout per fit | Fits | Reservation per fit | Total |
|---|---:|---:|---:|---:|---:|
| MICN | 3.779 s | 990 s | 6 | $0.907105 | $5.442630 |
| FiLM | 6.947 s | 1710 s | 6 | $1.463065 | $8.778390 |
| Total | | | 12 | | **$14.221020** |

With20% margin and a protected$1 reserve, admission requires **$18.065224**, rounded up to a proposed **new $19 cap**. The12 configured timeouts sum to4.5 GPU-hours; at two concurrent jobs, their timeout-only wall-clock bound is2.25 hours, excluding setup/return time. Early stopping may shorten execution. This is an estimate for native ETTh1 development only; it does not price the128-fit HyperDense grid, five-seed confirmation, or a different context/dataset.

A separately frozen [development plan](../results/etth1-transfer/development-prepared-v1/plan.json) and source archive are ready. The CLI reprices at launch and refuses a complete stage that does not fit the new cap, an existing allocation directory, changed source/data or conflicting Modal activity.

Only after authorization of the new development cap:

```sh
/Users/w3joe/.local/share/hypercast4d/runtime/bin/python scripts/run_etth1_development.py \
  --bundle results/etth1-transfer/prepared-v3 \
  --plan results/etth1-transfer/development-prepared-v1/plan.json \
  --allocation results/etth1-transfer/development-001 \
  --authorized-cap-usd 19
```

## Artifacts and decision boundary

- [Calibration timings](../results/etth1-transfer/calibration-001/calibration.json)
- [Checkpoint audit and complete cost calculation](../results/etth1-transfer/calibration-001/audit-and-cost-proposal.json)
- [Calibration ledger](../results/etth1-transfer/calibration-001/ledger.json)
- [Frozen dataset manifest](../results/etth1-transfer/prepared-v3/manifest.json)
- [Prior Copper stress test and data preparation](hyperdense_outer_stress_and_etth1_preparation.md)

The calibration establishes feasibility and provides timings. It does not establish forecasting quality. Retain both models and all paired seeds in development, review convergence and both practical baselines, and report any disagreement before choosing a representation. Neither final-test scoring nor HyperDense tuning is authorized by this proposal.
