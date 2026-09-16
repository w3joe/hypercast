# Outer Copper stress test and next-stage preparation

Completed locally after the user authorized the checkpoint stress test and asked to proceed to the next step. No new paid jobs or real-data training were launched. All historical ledgers and frozen sources remain unchanged.

## Retrospective transfer results

Evaluated all twelve unchanged native-model checkpoints on 297 forecast origins (five leads each), covering outer target dates 2020-08-11–2021-10-19. The stress-test manifest was written before inference. The original training scaler, feature order and context32/horizon5 were preserved. No checkpoint was reselected, warm-started or retrained, and the later Copper tail was not scored.

| Model | Seed | Direct MAE / persistence | Relative/residual MAE / persistence | MAE reduction versus direct |
|---|---:|---:|---:|---:|
| MICN | 1101 | 1.1679 | 0.9916 | 15.09% |
| MICN | 1102 | 1.2609 | 0.9774 | 22.48% |
| MICN | 1103 | 1.3154 | 0.9863 | 25.01% |
| FiLM | 1101 | 1.0581 | 1.0076 | 4.77% |
| FiLM | 1102 | 1.1128 | 1.0096 | 9.28% |
| FiLM | 1103 | 1.1703 | 1.0097 | 13.72% |

The package improves over direct output in all six pairs, including the previously concerning FiLM model. MICN's median paired improvement over direct is 22.48%, but its median benefit over persistence is only 1.37%. All three MICN relative/residual runs beat persistence by point MAE (0.84–2.26%); all three FiLM runs remain 0.76–0.97% worse. Large gains over direct do not imply equally large practical forecasting gains.

For origins above the training maximum, MICN relative/residual MAE/persistence is 1.0001, 0.9895 and 0.9963; FiLM is 1.0134, 1.0168 and 1.0148. The package greatly reduces the original direct-model weakness but does not establish that the high-price regime is solved. MICN improves over direct in every chronological third for every seed. FiLM improves in eight of nine such cells; seed1101's first third is 1.09% worse.

Before outer scoring, each CPU model replayed its archived inner forecasts. Maximum absolute CPU-versus-L4 replay discrepancy was 0.000240 raw price units, within the declared `atol=2e-5, rtol=2e-4` check. Target reconstruction passed at absolute tolerance 5e-7. All 17,820 outer forecast values were finite and retain source target-row indices. CPU and GPU arithmetic need not be bit-identical.

These are already-exposed retrospective development dates. This result is not independent confirmation and does not clear the original FiLM convergence issue. It strengthens the case for testing the package on a new dataset while retaining both models. No HyperDense arm was trained or evaluated in this stress test.

Artifacts: [frozen stress manifest](../results/hyperdense-outer-stress-v1/manifest.json), [all scores, leads, bias and regimes](../results/hyperdense-outer-stress-v1/scores.json), [paired summary](../results/hyperdense-outer-stress-v1/summary.json). Reproduction script: `scripts/stress_test_representation.py`; it refuses to overwrite the completed directory.

## Next step completed locally: ETTh1 preparation

Downloaded the author's ETTh1 file at immutable commit `1d16c8f4f943005d613b5bc962e9eeb06058cf07`. Source SHA256: `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`. The repository audit found no prior ETTh1 runs, only proposal references. This is a new dataset for the project, not a prospective collection or a claim of universal freedom from public-benchmark exposure.

The active frozen bundle is **`results/etth1-transfer/prepared-v3`**. Earlier preparation snapshots remain preserved; v3 adds checkpoint/optimizer/RNG archiving to the calibration path. No versions have been trained on real data.

All seven numeric channels are retained, ordered `OT, HUFL, HULL, MUFL, MULL, LUFL, LULL`; OT is the target. Context is 32 hours and horizon five hours. This deliberately preserves the existing study's latent shape and is a custom short-horizon task, not a standard long-horizon benchmark reproduction. Persistence and a 24-hour seasonal baseline use the same observed 32-hour history.

| Role | Half-open source rows | Forecast origins | Use |
|---|---|---:|---|
| Train | 0–6480 | 6444 | Fit scaler and weights |
| Inner | 6480–8640 | 2156 | Choose checkpoint |
| Development | 8640–11520 | 2876 | Compare packages/settings |
| Final test | 11520–14400 | 2876 | Withheld until model/protocol freeze |
| Remaining tail | 14400–17420 | — | Unused |

All targets stay within their own role; histories can use prior observed rows. Final-test targets span 2017-10-24–2018-02-20. The final-test tensors are absent from the development bundle, no test forecasts or test-derived scaling statistics were computed, and no baseline/test metrics have been inspected. Exact dates, hashes and scaler values are recorded in the [manifest](../results/etth1-transfer/prepared-v3/manifest.json).

`src/hypercast4d/native_transfer.py` is an isolated seven-channel native wrapper. The frozen Copper implementation and app's four-feature schema were not changed. Eight focused tests passed: four-channel forward/input-gradient equivalence against historical models, paired seven-channel initialization, active gradients through all input channels, persistence initialization and gradient recovery, target boundaries and seasonal-baseline causality, and a train-only timing smoke run with checkpoint/optimizer/RNG serialization. The real development bundle also passed target reconstruction and training-only scaler checks. These CPU gates do not establish CUDA correctness or forecasting quality on ETTh1.

## Next paid action: one bounded calibration, not the full study

Prepared `scripts/calibrate_etth1_transfer.py` for **one L4 job, maximum 600 seconds**, two CPU cores and 4 GiB memory, no retries. It exercises MICN and FiLM × both packages for three training epochs, seed2100, and measures inner forward time without accessing inner targets or calculating selection scores. Development/test targets are not scored. Checkpoints include model, optimizer, sampler and RNG state for subsequent resume validation; the calibration seed is excluded from development comparisons.

The resulting timing will cost the proposed twelve-fit development comparison (three paired seeds2201–2203, up to150 epochs, inner-MAE selection and development-MAE ranking). Neither those twelve fits nor the 128-fit HyperDense grid is submitted by the calibration CLI. A full resumable development runner and equal eight-arm tuning controls remain separate work.

At the most recently verified resource rates, the calibration reservation is **$0.60596**. With 20% margin and a protected $1 reserve, admission requires **$1.727152**, rounded to a proposed **new $2 cap**. The CLI refreshes pricing and rejects an insufficient cap, refuses existing allocation directories, checks active Modal tasks, acquires the pilot lock and records the full reservation before submission. Failed attempts remain charged. Prior allocations stay closed; asking to proceed did not specify a new dollar cap.

Only after explicit new spending authorization:

```sh
/Users/w3joe/.local/share/hypercast4d/runtime/bin/python scripts/calibrate_etth1_transfer.py \
  --bundle results/etth1-transfer/prepared-v3 \
  --allocation results/etth1-transfer/calibration-001 \
  --authorized-cap-usd 2
```

The default development preference remains two L4 jobs in parallel; the calibration itself needs only one. No larger-study budget is quoted until this workload has been measured.
