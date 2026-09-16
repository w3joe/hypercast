# HyperDense experiment results — 11 September 2026

## Outcome

The experiments do not establish a useful forecasting advantage for this HyperDense frontend on the Copper dataset. The complete robustness comparison favors the original backbones and persistence. Some hypercomplex maps outperform the added real dense frontend, but that frontend is itself a weak comparator for practical forecasting. Results depend on backbone and period; no universal 2D/4D/8D winner is established.

The final comparison is incomplete: iTransformer rank-4 and rank-2 controls did not run because the conservative budget guard stopped the queue. All completed results are retained. No aggregate final winner or complete confirmatory result is declared.

## What ran

| Stage | Completed fits | Scope |
|---|---:|---|
| Pilot | 54 intended, plus 2 duplicate fits | 27 configurations, two one-step windows, seed 7, 10 epochs |
| CUDA preflight | 1 smoke fit plus numerical checks | CPU/CUDA forward and gradient references, variance, packing and shape checks |
| Controlled development | 54/54 | All 27 configurations at two rates, seed 19, window 20/horizon 5, 150 epochs |
| Robustness | 162/162 | All 27 configurations, seeds 101/211, three chronological folds, 150 epochs |
| Locked retrospective final | 50/54 | 25/27 configurations, both seeds, first 85% train/final 15% test, 150 epochs |

There were 135 cloud jobs and 323 fitted runs including the duplicate and smoke fit. No launched programme job failed. Two final configurations were blocked before submission. Historical unrelated interrupted jobs were not rerun by this programme. The pilot was exploratory and had initialization confounds; controlled results use matched effective variance and paired shared-layer initialization. Different dimensions have different parameter budgets at the same real width, so this is not a dimension-only causal comparison.

## Complete robustness comparison

Scores are equal-fold, equal-seed mean MAE divided by persistence MAE. Lower is better; 1 means matching the last-observed-value forecast.

| Backbone | Original | Added real dense | 2D | 4D | 8D |
|---|---:|---:|---:|---:|---:|
| DLinear | 1.151 | 1.544 | 1.568 | 1.572 | 1.433 |
| TSMixer | 1.324 | 1.722 | 1.811 | 1.611 | 1.655 |
| iTransformer | 1.137 | 2.572 | 2.601 | 2.334 | 2.052 |

All nine hypercomplex configurations have higher mean error than their original backbone and persistence. Added real dense and lift-only controls also generally worsen the original models, implicating the frontend architecture and generalization behavior rather than isolating hypercomplex arithmetic as the cause. Training/validation divergence persists after 150 epochs. More epochs alone are not supported as a remedy.

Against added real dense, the 8D point estimate improves by 7.2%, 3.9% and 20.2% for the three backbones. However, none of the nine hyper-versus-real family-wise intervals at the prespecified block length establishes the required >2% practical benefit. TSMixer 4D has an adjusted interval above zero, but its lower bound is only about 0.5%, below that threshold. This is not evidence of equivalence for the other comparisons.

Whole-model parameter savings against added real dense range from 3.4–7.8% for 2D, 5.1–11.8% for 4D and 5.9–13.7% for 8D. All these adapted models remain larger than their original backbone. Inference was generally slower in the measured device-resident batch-32 benchmark; parameter savings do not imply GPU speed. See the full resource table and accuracy-versus-parameter figure in the robustness report.

## Incomplete locked retrospective final comparison

These scores average the two frozen seeds on the final period; do not pool them with robustness validation scores.

| Backbone | Original | Added real dense | 2D | 4D | 8D |
|---|---:|---:|---:|---:|---:|
| DLinear | 1.097 | 1.329 | 1.211 | 1.324 | 1.140 |
| TSMixer | 1.177 | 1.263 | 1.376 | 1.289 | 1.166 |
| iTransformer | 1.173 | 2.564 | 2.481 | 1.334 | 1.417 |

All hypercomplex mean scores still lose to persistence. TSMixer 8D is about 0.9% better than its original backbone on this final period, a descriptive difference below the 2% practical margin; it also loses to lift-only. Thus it would be inaccurate to say every variant loses to the original in every comparison.

iTransformer 4D and 8D improve substantially against added real dense: about 48.0% and 44.7%. Their adjusted block-length-60 intervals are approximately [38.6%, 58.9%] and [29.4%, 60.7%]. These are meaningful relative signals against that particular frontend control, while both models still have higher mean error than the original and persistence. The missing iTransformer rank-4/rank-2 controls prevent their final matched-compression comparisons. The full 36-comparison correction is retained for available contrasts; missing contrasts are explicitly marked unavailable.

## Limits and direct answers

- **2D:** no consistent practical forecasting benefit established across these backbones and periods.
- **4D:** mixed validation results and a strong final-period improvement over iTransformer's added real dense frontend, but no demonstrated practical superiority over the useful baselines.
- **8D:** favorable relative error and compression signals in several comparisons, but no universal winner or robust advantage over persistence. Its behavior depends on backbone and period.

The claim is restricted to a learned 4-to-32 input frontend, this dataset/target, window 20/horizon 5 and the frozen training recipe. It says little about replacing internal layers, native eight-channel observations, other widths, unrelated datasets or algebra structure versus generic parameter sharing. Two seeds and three periods do not create independent datasets.

Uncertainty uses the preregistered paired circular block bootstrap, 10,000 resamples, 36-contrast Bonferroni correction, block length 60 and 30/120 sensitivity. Robust folds have only 196–197 origins, about three effective blocks at the primary length; the final period has 298 origins, about five. Strong time dependence, few seeds and extreme-tail Monte Carlo precision limit interpretation. Validation periods overlap earlier development exposure, and historical final-test exposure cannot be ruled out. Call the last stage locked retrospective, not prospective confirmation.

## Budget stop and retained next questions

The controller stopped cleanly at 03:02 UTC. Conservative whole-programme accounting is **$18.793498 of $20**, including all prior stages, the duplicate, startup allowances, and the established 3x factor. With the $1 safety reserve, $0.206502 remains available; another final job requires a $0.37431 reservation. This is an accounting stop, not a claim that Modal invoiced $18.79. The latest delayed metered increase is about $3.44 and billed cost is zero after credits. No experiment GPU or controller process is active.

No further cloud experiment fits the existing safe reservation, so follow-ups stop here. The original plan already identified internal replacement, width sensitivity, generic weight-tying controls and independent datasets as unresolved questions; they remain unrun. Any future programme needs a new budget and frozen design, with new data for prospective claims. Do not tune a new recipe against these final test outcomes.

## Artifacts

- Full robustness results, fold/seed summaries, intervals, per-lead errors, resources and figures: `results/modal-l4-runner/controller/robustness-analysis/`.
- Incomplete final results, all available/missing contrasts, seed/resource summaries and per-lead errors: `results/modal-l4-runner/controller/final-analysis/`.
- Exact source/job/selection provenance: `results/modal-l4-runner/controller/{controlled,robustness,final}-provenance/` and each job's request, runtime and prediction files.
- Reproducible analysis entry points: `scripts/analyze_modal_l4_robustness.py` and `scripts/analyze_modal_l4_final.py`. The robustness script requires its completed-stage state; use the preserved pre-final state for reproduction, not the current final-stage state. Preserve frozen outputs when reproducing.
