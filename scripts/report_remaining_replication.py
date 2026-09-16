"""Render the completed study's existing results; performs no training or selection."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/modal-l4-internal/controller"
analysis = json.loads((DATA / "remaining-replication-analysis.json").read_text())
state = json.loads((DATA / "controller_state.json").read_text())
audit = json.loads((DATA / "remaining-replication-review.json").read_text())
assert audit["passed"] and len(analysis["scores"]) == 96
names = dict(patchtst="PatchTST", timesnet="TimesNet", crossformer="Crossformer",
             frets="FreTS", lightts="LightTS", micn="MICN", msgnet="MSGNet",
             scinet="SCINet", segrnn="SegRNN", timemixer="TimeMixer",
             timexer="TimeXer", film="FiLM (spectral)")
variants = ["complex", "quaternion", "octonion"]
dims = dict(zip(variants, ["2D", "4D", "8D"]))
scores = {(r["backbone"], r["variant"]): r for r in analysis["scores"]}
delta = np.array([[100 * (scores[b, v]["mae"] / scores[b, "real"]["mae"] - 1)
                   for v in variants] for b in names])
fig, ax = plt.subplots(figsize=(8.4, 8.2), layout="constrained")
limit = max(abs(delta.min()), abs(delta.max()))
im = ax.imshow(delta, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
ax.set_xticks(range(3), ["2D · complex", "4D · quaternion", "8D · octonion"])
ax.set_yticks(range(12), list(names.values()))
ax.tick_params(length=0, pad=9)
for i in range(12):
    for j in range(3):
        ax.text(j, i, f"{delta[i,j]:+.1f}%", ha="center", va="center",
                color="white" if abs(delta[i,j]) > .65 * limit else "#18232b", fontsize=11)
ax.set_title("Hypercomplex internal replacements: mixed accuracy effects\n"
             "MAE change versus each model's controlled real replacement", loc="left", pad=20, fontsize=13)
fig.colorbar(im, ax=ax, shrink=.7, label="MAE change (%) · negative is better")
fig.supxlabel("Fixed tuned rates · fresh seed · up to 50 epochs · same Copper period\n"
               "Point estimates only. All 36 hypercomplex arms have higher MAE than persistence.", fontsize=10)
out = ROOT / "docs/figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "remaining-replication.png", dpi=180)
fig.savefig(out / "remaining-replication.pdf")
plt.close(fig)

lines = ["# Internal HyperDense replacement: completed remaining-model study",
         "", "Results recorded 11 September 2026. **No consistent accuracy advantage or universal winning dimension emerged.** Some replacements improved the controlled real arm, but every hypercomplex follow-up arm lost to persistence on mean absolute error. Parameter compression is measurable; it did not generally produce faster inference.",
         "", "## Scope and tuning", "",
         "The expansion covered twelve remaining pinned TSLib backbones. DLinear, TSMixer and iTransformer were excluded as requested. This does not cover every other model family or hybrid preset in the application. These are selected internal-layer replacements, not an extra hypercomplex input layer, and not a replacement of every dense layer. FiLM is a separately labelled spectral intervention; its native operator is already complex.",
         "", "Each backbone had eight arms: native, controlled real replacement, 2D/4D/8D HyperDense and three corresponding approximate parameter-budget low-rank controls. Untouched weights were paired, with the same features, chronological splits, scaler fitting, surrounding architecture and external head.",
         "", "**192 tuning fits:** 12 backbones × 8 arms × learning rates 0.0003 and 0.001, seed 701, maximum 25 epochs. Each arm received the same search opportunity. The best development MAE selected its rate, with the lower rate preferred for relative ties within 0.1%.",
         "", "**96 follow-up fits:** all arms retained, rates frozen, fresh seed 401, maximum 50 epochs. Context 32, horizon 5, batch 32, patience 20 and relative checkpoint threshold 0.001 were held fixed. Checkpoints were selected on a chronological inner split and restored before scoring. All 96 fits passed artifact, initialization, split/scaler, checkpoint and prediction-metric audits. Ten chose their final executed epoch; 25 stopped before epoch 50. This does not establish full convergence.",
         "", "Only learning rate was searched. Width, dropout, weight decay, initialization scaling and other hyperparameters were not broadly tuned. Equal tuning is necessary for a fair comparison, but this small grid does not establish each arm's achievable optimum.",
         "", "## Full dimension comparison", "",
         "![MAE change versus the controlled real arm](figures/remaining-replication.png)", "",
         "MAE divided by persistence MAE; **lower is better and below 1 would beat persistence**. The controlled real arm is the matched replacement baseline; native preserves the model's original operator and initialization. They answer different comparisons.", "",
         "| Backbone | Native | Real | 2D | 4D | 8D | Low-rank 2 | Low-rank 4 | Low-rank 8 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for b, label in names.items():
    lines.append("| " + label + " | " + " | ".join(f'{scores[b,v]["mae_ratio"]:.3f}' for v in ["native", "real", *variants, "lowrank2", "lowrank4", "lowrank8"]) + " |")
lines += ["", "Among hypercomplex arms, 2D has the lowest point MAE on six backbones, 4D on three and 8D on three. At least one dimension beats controlled real on seven of twelve backbones by point estimate. These winner counts are descriptive and do not establish a statistically superior dimension.",
          "", "## What survived the follow-up", "",
          "MICN 8D reduced MAE by **10.87% versus controlled real**. Its simultaneous interval excluded zero at each tested block length (30, 60 and 120), the clearest such signal against that control. However, it was **4.78% worse than native MICN** and **17.01% worse than persistence**. Thus it is evidence for a particular replacement comparison, not a practical forecasting win.",
          "", "PatchTST 2D improved controlled real by 3.54%; its adjusted interval excluded zero at the primary block length 60, but not at length 30. PatchTST 4D had a better point score than 2D, illustrating why point winners and uncertainty conclusions must be separated. No hypercomplex arm beat persistence on point MAE, let alone a favourable adjusted interval. In fact all 96 follow-up arms lost to persistence; the closest was SegRNN controlled real at 1.027 times persistence MAE.",
          "", "Some large development gains did not survive: FreTS 4D changed from roughly 9.7% better than controlled real to 17.6% worse, while SegRNN 8D changed from 29.9% better to 10.9% worse. Because both seed and maximum epochs changed, this cannot be attributed to either alone. It shows why short-run rankings were insufficient.",
          "", "## Compression and measured inference", "",
          "The following selects each backbone's lowest-MAE hypercomplex arm descriptively. Negative error changes favour that arm. Parameter reductions refer to the entire model, not just the replaced layer. Inference ratios above 1 mean slower than controlled real.", "",
          "| Backbone | Best dimension | MAE change vs real | MAE change vs native | Parameters saved vs real | Inference time / real |",
          "|---|---|---:|---:|---:|---:|"]
for b, label in names.items():
    v = min(variants, key=lambda v: scores[b,v]["mae"])
    r, real, native = scores[b,v], scores[b,"real"], scores[b,"native"]
    lines.append(f'| {label} | {dims[v]} | {100*(r["mae"]/real["mae"]-1):+.2f}% | {100*(r["mae"]/native["mae"]-1):+.2f}% | {100*(1-r["parameters"]/real["parameters"]):.2f}% | {r["inference_ms_per_batch"]/real["inference_ms_per_batch"]:.2f}× |')
lines += ["", "FiLM 4D used 37.5% fewer whole-model parameters than controlled real, with 0.56% higher point MAE and roughly 43% slower measured inference. Against already-complex native FiLM, the parameter reduction is about 16.7%. This is a possible compression trade-off, not established accuracy equivalence. Overall, 34 of 36 hypercomplex arms had slower measured batch-32 inference than controlled real. Timings are one benchmark per fit with warm-up and repeats, not independent repeated performance trials.",
          "", "## Uncertainty and limits", "",
          "The preregistered analysis compared 180 contrasts using 10,000 circular moving-block bootstrap resamples and family-wise 95% max-absolute centered/studentized intervals. Each observation is one forecast origin's mean absolute error over five leads; the same resampled origins are shared across comparisons. Primary block length 60 was selected using development data only, with sensitivity checks at 30 and 120. There are only 297 origins and about 4.95 effective blocks at length 60, so interval coverage is uncertain. At the primary length, 11 contrasts favoured the left arm and 70 favoured the right; only two favourable comparisons were hypercomplex versus controlled real. Seven favourable contrasts survived all three lengths, many against other compressed controls.",
          "", "The Copper evaluation dates were already used for rate selection. The follow-up has only one fresh seed and a different epoch allowance. It is limited replication on reused data, not independent confirmation; do not pool the two stages as identically trained replicates. Nonsignificance is not equivalence. This study cannot show that hypercomplex layers never help, nor rank algebras universally. It tests these specific internal sites and training settings.",
          "", "## Completion and budget", "",
          "The additional $20 internal allocation contains 372 successful fits: 24 original-three internal pilot, 48 development, 12 remaining-model calibrations, 192 tuning and 96 follow-up. There were 109 submitted jobs, including one failed SCINet calibration attempt whose cost remains charged. The expansion alone comprises 300 successful fits. No tuning or follow-up batch failed. The original separate allocation remains closed and unchanged.", "",
          f'Conservative accounted spend is **${state["budget"]["estimated_spend_usd"]:.2f} of $20**; delayed metered allocation usage is **${state["budget"]["confirmed_spend_usd"]:.2f}**, not a final bill. The conservative ledger includes uncertainty and startup allowances. It leaves ${20-state["budget"]["estimated_spend_usd"]:.2f} against the cap, including the protected $1 reserve. The internal programme ran from approximately 05:25 to 11:51 UTC on 11 September (about 6 hours 26 minutes elapsed, not continuous billed GPU time).',
          "", "The planned matrix and analysis are complete. No further useful balanced stage under the current protocol fits the remaining conservative headroom: even twelve minimum 300-second job reservations plus overhead total about $4.49, above the $3.49 available after the reserve. This is not a claim that every conceivable small experiment is impossible. Further short or selectively chosen runs on the same data would not resolve the central limitations, so training ends with funds unspent.",
          "", "A future study should preregister independent data and multiple paired seeds, allocate equal tuning budgets for learning rate, regularization and initialization, and train enough to assess convergence. A focused representation-aware hypothesis should be chosen before scoring; extra tuning should not target hypercomplex arms alone. This is a recommendation for a separately scoped study, not a new launch.",
          "", "## Reproducible evidence", "",
          "- [Frozen tuning protocol](modal_l4_remaining_tuning_stage.md)",
          "- [Frozen follow-up and uncertainty protocol](modal_l4_remaining_replication_stage.md)",
          "- [Analysis with all 96 scores and 180 intervals](../results/modal-l4-internal/controller/remaining-replication-analysis.json)",
          "- [Full 96-fit audit](../results/modal-l4-internal/controller/remaining-replication-review.json)",
          "- [192-fit tuning audit](../results/modal-l4-internal/controller/remaining-tuning-review.json)",
          "- [Existing literature notes](hyperdense_literature_notes.md)", ""]
(ROOT / "docs/modal_l4_remaining_results.md").write_text("\n".join(lines))
print("Saved docs/modal_l4_remaining_results.md and docs/figures/remaining-replication.{png,pdf}")
