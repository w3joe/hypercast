#!/usr/bin/env python3
"""Stage and publish HyperCAST benchmark results + visual charts to Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from huggingface_hub import (
    HfApi,
    add_collection_item,
    create_collection,
    list_collections,
)

ROOT = Path(__file__).resolve().parents[1]

DATASET_LABELS: dict[str, str] = {
    "ili": "ILI (7ch · W)",
    "nyc_taxi": "NYC Taxi (8ch · H)",
    "exchange": "Exchange (8ch · D)",
    "etth2": "ETTh2 (7ch · H)",
    "ettm1": "ETTm1 (7ch · 15m)",
    "ettm2": "ETTm2 (7ch · 15m)",
    "weather": "Weather (21ch · 10m)",
    "electricity": "Electricity (321ch · H)",
    "traffic": "Traffic (862ch · H)",
}

BACKBONE_LABELS: dict[str, str] = {
    "crossformer": "Crossformer",
    "dlinear": "DLinear",
    "film": "FiLM",
    "frets": "FreTS",
    "itransformer": "iTransformer",
    "lightts": "LightTS",
    "micn": "MICN",
    "msgnet": "MSGNet",
    "patchtst": "PatchTST",
    "scinet": "SCINet",
    "segrnn": "SegRNN",
    "timemixer": "TimeMixer",
    "timesnet": "TimesNet",
    "timexer": "TimeXer",
    "tsmixer": "TSMixer",
}

PALETTE = {
    "hd": "#4F46E5",       # Indigo 600
    "hd_alt": "#0D9488",   # Teal 600
    "lr": "#D97706",       # Amber 600
    "real": "#475569",     # Slate 600
    "native": "#0F172A",   # Slate 900
    "neg": "#E11D48",      # Rose 600
    "complex": "#2563EB",  # Blue 600
    "quaternion": "#7C3AED",  # Violet 600
    "octonion": "#0D9488",    # Teal 600
    "grid": "#E2E8F0",
    "bg": "#FFFFFF",
    "panel": "#F8FAFC",
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["panel"],
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.85,
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": True,
            "legend.facecolor": "#FFFFFF",
            "legend.edgecolor": "#E2E8F0",
            "legend.fontsize": 8.8,
        }
    )


def generate_figures(run_dir: Path, assets_dir: Path) -> None:
    """Render 5 high-DPI publication figures into ``assets_dir``."""
    _apply_style()
    assets_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(run_dir / "validation-selected-cells.csv")
    cfgs = pd.read_csv(run_dir / "all-datasets-summary.csv")

    # ------------------------------------------------------------------
    # Figure 1: Hero 2-Panel Overview (Dataset Win Rates & Regime Split)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.3, 1.0]}
    )
    ds_order = list(DATASET_LABELS.keys())
    ds_stats = []
    for ds in ds_order:
        sub = cells[cells["dataset"] == ds]
        n = len(sub)
        ds_stats.append(
            {
                "ds": ds,
                "label": DATASET_LABELS[ds],
                "wr_real": (sub["val_hd_vs_real_pct"] > 0).sum() / n * 100.0,
                "wr_lr": (sub["val_hd_vs_val_lr_pct"] > 0).sum() / n * 100.0,
                "wins_real": f"{(sub['val_hd_vs_real_pct'] > 0).sum()}/{n}",
                "med_real": float(sub["val_hd_vs_real_pct"].median()),
            }
        )
    ds_df = pd.DataFrame(ds_stats).sort_values("wr_real", ascending=True)
    y = np.arange(len(ds_df))
    h = 0.36
    bars_r = ax1.barh(
        y + h / 2,
        ds_df["wr_real"],
        height=h,
        color=PALETTE["hd"],
        label="Val-HyperDense beats Real",
        zorder=3,
    )
    ax1.barh(
        y - h / 2,
        ds_df["wr_lr"],
        height=h,
        color=PALETTE["lr"],
        alpha=0.88,
        label="Val-HyperDense beats Val-LowRank",
        zorder=3,
    )
    ax1.axvline(50.0, color="#64748B", linestyle="--", linewidth=1.2, zorder=2, label="50% parity")
    for bar, (_, r) in zip(bars_r, ds_df.iterrows()):
        ax1.text(
            bar.get_width() + 1.3,
            bar.get_y() + bar.get_height() / 2,
            f"{r['wins_real']} ({r['med_real']:+.2f}%)",
            va="center",
            ha="left",
            fontsize=8.3,
            color="#1E293B",
            fontweight="bold",
        )
    ax1.set_yticks(y)
    ax1.set_yticklabels(ds_df["label"])
    ax1.set_xlim(0, 114)
    ax1.set_xlabel("Backbone Win Rate (%) — Zero-Leakage Validation Selection")
    ax1.set_title("A. Per-Dataset Win Rate Across 15 TSLib Backbones")
    ax1.legend(loc="lower right")

    low_ch = cells[cells["dataset"].isin(ds_order[:7])]
    high_ch = cells[cells["dataset"].isin(ds_order[7:])]
    regimes = ["Low / Medium Channels\n(7–21 ch · 105 cells)", "High-Dimensional Channels\n(321–862 ch · 26 cells)"]
    hd_meds = [float(low_ch["val_hd_vs_real_pct"].median()), float(high_ch["val_hd_vs_real_pct"].median())]
    hd_means = [float(low_ch["val_hd_vs_real_pct"].mean()), float(high_ch["val_hd_vs_real_pct"].mean())]
    lr_meds = [float(low_ch["val_lr_vs_real_pct"].median()), float(high_ch["val_lr_vs_real_pct"].median())]

    x = np.arange(len(regimes))
    w = 0.25
    b1 = ax2.bar(x - w, hd_meds, width=w, color=PALETTE["hd"], label="Val-HyperDense (Median %)", zorder=3)
    b2 = ax2.bar(x, hd_means, width=w, color=PALETTE["hd_alt"], label="Val-HyperDense (Mean %)", zorder=3)
    b3 = ax2.bar(x + w, lr_meds, width=w, color=PALETTE["lr"], label="Val-LowRank (Median %)", zorder=3)
    ax2.axhline(0.0, color="#64748B", linewidth=1.0, zorder=2)
    for bars in (b1, b2, b3):
        for b in bars:
            val = b.get_height()
            ax2.text(
                b.get_x() + b.get_width() / 2,
                val + (0.12 if val >= 0 else -0.28),
                f"{val:+.2f}%",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8.5,
                fontweight="bold",
                color="#0F172A",
            )
    ax2.set_xticks(x)
    ax2.set_xticklabels(regimes)
    ax2.set_ylim(-0.6, max(hd_means + lr_meds) + 1.0)
    ax2.set_ylabel("Test MAE Improvement vs. Real (%)")
    ax2.set_title("B. Channel-Dimensionality Regime Effect")
    ax2.legend(loc="upper left")

    fig.suptitle(
        "HyperCAST Multi-Dataset Benchmark (3,157 GPU Fits · 9 Datasets × 15 Backbones × 8 Arms)",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(assets_dir / "fig1_hero_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Backbone Architecture Ranking (15 TSLib Backbones)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    b_rows = []
    for b, g in cells.groupby("backbone"):
        n = len(g)
        b_rows.append(
            {
                "backbone": BACKBONE_LABELS.get(b, b),
                "med_vs_lr": float(g["val_hd_vs_val_lr_pct"].median()),
                "med_vs_real": float(g["val_hd_vs_real_pct"].median()),
                "wins_lr": f"{(g['val_hd_vs_val_lr_pct'] > 0).sum()}/{n}",
                "wins_real": f"{(g['val_hd_vs_real_pct'] > 0).sum()}/{n}",
            }
        )
    bdf = pd.DataFrame(b_rows).sort_values("med_vs_lr", ascending=True)
    yb = np.arange(len(bdf))
    hb = 0.37

    colors_lr = [PALETTE["hd"] if v >= 0 else PALETTE["neg"] for v in bdf["med_vs_lr"]]
    bars_lr = ax.barh(
        yb + hb / 2,
        bdf["med_vs_lr"],
        height=hb,
        color=colors_lr,
        label="Val-HyperDense vs. Parameter-Matched Low-Rank (Median %)",
        zorder=3,
    )
    ax.barh(
        yb - hb / 2,
        bdf["med_vs_real"],
        height=hb,
        color="#94A3B8",
        alpha=0.85,
        label="Val-HyperDense vs. Unconstrained Real (Median %)",
        zorder=3,
    )
    ax.axvline(0.0, color="#334155", linewidth=1.1, zorder=2)

    for bar, (_, r) in zip(bars_lr, bdf.iterrows()):
        v = r["med_vs_lr"]
        offset = 0.06 if v >= 0 else -0.06
        ha = "left" if v >= 0 else "right"
        ax.text(
            v + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{v:+.2f}% ({r['wins_lr']} wins)",
            va="center",
            ha=ha,
            fontsize=8.3,
            fontweight="bold",
            color="#0F172A",
        )

    ax.set_yticks(yb)
    ax.set_yticklabels(bdf["backbone"], fontweight="bold")
    ax.set_xlim(-2.95, 2.55)
    ax.set_xlabel("Median Held-Out Test MAE Improvement (%) Across Datasets (Positive = HyperDense Better)")
    ax.set_title(
        "Which Architectures Benefit From Hypercomplex Structure? (Multi-Scale / Spatial Mixers vs. Channel-Independent)"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(assets_dir / "fig2_backbone_architecture_ranking.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: 15 × 9 Annotated Matrix Heatmap
    # ------------------------------------------------------------------
    pivot_real = cells.pivot(index="backbone", columns="dataset", values="val_hd_vs_real_pct")
    pivot_lr = cells.pivot(index="backbone", columns="dataset", values="val_hd_vs_val_lr_pct")

    ordered_backbones = [
        k for k in sorted(BACKBONE_LABELS.keys(), key=lambda b: float(cells[cells["backbone"] == b]["val_hd_vs_real_pct"].median()), reverse=True)
    ]
    pivot_real = pivot_real.reindex(index=ordered_backbones, columns=ds_order)
    pivot_lr = pivot_lr.reindex(index=ordered_backbones, columns=ds_order)

    fig, ax = plt.subplots(figsize=(12.2, 7.2))
    mat = pivot_real.to_numpy(dtype=float)
    clipped = np.clip(np.nan_to_num(mat, nan=0.0), -6.0, 6.0)
    cmap = plt.cm.RdBu  # Red=negative, Blue=positive
    norm = TwoSlopeNorm(vmin=-6.0, vcenter=0.0, vmax=6.0)
    im = ax.imshow(clipped, cmap=cmap, norm=norm, aspect="auto")
    ax.grid(False)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#E2E8F0", zorder=2))
                ax.text(j, i, "skip", ha="center", va="center", fontsize=7.8, color="#64748B")
                continue
            lr_win = pivot_lr.iloc[i, j] > 0
            star = "★" if lr_win else ""
            txt_color = "#FFFFFF" if abs(clipped[i, j]) > 3.3 else "#0F172A"
            ax.text(
                j,
                i,
                f"{val:+.1f}%{star}",
                ha="center",
                va="center",
                fontsize=8.0,
                fontweight="bold" if lr_win else "normal",
                color=txt_color,
            )

    ax.set_xticks(np.arange(len(ds_order)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in ds_order], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(ordered_backbones)))
    ax.set_yticklabels([BACKBONE_LABELS[b] for b in ordered_backbones], fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
    cbar.set_label("Val-Selected HyperDense vs. Real Test MAE Gain (%, clipped ±6%)")
    ax.set_title(
        "Complete 9-Dataset × 15-Backbone Matrix  (★ = HyperDense Also Beats Parameter-Matched Low-Rank)"
    )
    fig.tight_layout()
    fig.savefig(assets_dir / "fig3_heatmap_9x15.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4: Multi-Step Forecast Horizon Line Curves (Lead h = 1..5)
    # ------------------------------------------------------------------
    showcases = [
        ("ili", "timesnet", "A. ILI · TimesNet (Weekly Epidemiology)"),
        ("electricity", "timesnet", "B. Electricity · TimesNet (321-Ch Grid)"),
        ("traffic", "patchtst", "C. Traffic · PatchTST (862-Ch Sensors)"),
        ("traffic", "tsmixer", "D. Traffic · TSMixer (862-Ch Mixer)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.6))
    leads = np.arange(1, 6)
    arm_styles: list[tuple[str, str, str, str, float]] = [
        ("native", "Native TSLib", PALETTE["native"], "--", 1.6),
        ("real", "Real (Dense Control)", PALETTE["real"], "-.", 1.7),
        ("lowrank4", "Low-Rank (4× Comp.)", PALETTE["lr"], ":", 1.9),
        ("complex", "Complex (2D · 2× Comp.)", PALETTE["complex"], "-", 1.9),
        ("quaternion", "Quaternion (4D · 4× Comp.)", PALETTE["quaternion"], "-", 2.1),
        ("octonion", "Octonion (8D · 8× Comp.)", PALETTE["octonion"], "-", 2.1),
    ]

    for ax, (ds_id, bkbn, title) in zip(axes.flat, showcases):
        for arm_key, label, color, ls, lw in arm_styles:
            lead_curves = []
            for seed in (3301, 3302, 3303):
                rfile = run_dir / "trials" / ds_id / f"{bkbn}__{arm_key}__seed{seed}" / "result.json"
                if rfile.is_file():
                    rec = json.loads(rfile.read_text(encoding="utf-8"))
                    lead_curves.append(rec["per_lead_mae"])
            if not lead_curves:
                continue
            arr = np.asarray(lead_curves, dtype=float)
            mean_c = arr.mean(axis=0)
            sd_c = arr.std(axis=0)
            ax.plot(leads, mean_c, label=label, color=color, linestyle=ls, linewidth=lw, marker="o", markersize=4.5)
            if arr.shape[0] > 1 and arm_key in ("complex", "quaternion", "octonion", "real"):
                ax.fill_between(leads, mean_c - sd_c, mean_c + sd_c, color=color, alpha=0.10)

        ax.set_xticks(leads)
        ax.set_xticklabels([f"t+{h}" for h in leads])
        ax.set_xlabel("Forecast Horizon Step")
        ax.set_ylabel("Held-Out Test MAE (Original Units)")
        ax.set_title(title)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03), fontsize=9.2)
    fig.suptitle("Multi-Step Horizon Error Trajectories (t+1 → t+5, Mean ± 1 SD Across 3 Seeds)", fontsize=13, fontweight="bold", y=1.08)
    fig.tight_layout()
    fig.savefig(assets_dir / "fig4_lead_horizon_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 5: Training Convergence Line Graph & Compression Pareto Curve
    # ------------------------------------------------------------------
    fig, (ax_conv, ax_pareto) = plt.subplots(1, 2, figsize=(13.2, 5.0))

    # Left: Validation MAE convergence across epochs on ILI · TimesNet
    conv_arms = [
        ("real", "Real (100% site params)", PALETTE["real"], "-."),
        ("lowrank8", "Low-Rank 8× (12.5% params)", PALETTE["lr"], ":"),
        ("complex", "Complex 2D (50% params)", PALETTE["complex"], "-"),
        ("quaternion", "Quaternion 4D (25% params)", PALETTE["quaternion"], "-"),
        ("octonion", "Octonion 8D (12.5% params)", PALETTE["octonion"], "-"),
    ]
    for arm_key, label, color, ls in conv_arms:
        curves = []
        for seed in (3301, 3302, 3303):
            p = run_dir / "trials" / "ili" / f"timesnet__{arm_key}__seed{seed}" / "result.json"
            if p.is_file():
                hist = json.loads(p.read_text(encoding="utf-8"))["history"]
                # Running best inner-validation MAE (reflects restored early-stopping checkpoint)
                vals = np.minimum.accumulate([h["inner_mae"] for h in hist])
                curves.append(vals.tolist())
        if not curves:
            continue
        max_len = min(75, max(len(c) for c in curves))
        padded = np.array([c[:max_len] + [c[-1]] * max(0, max_len - len(c)) for c in curves], dtype=float)
        ep_x = np.arange(1, max_len + 1)
        m_curve = padded.mean(axis=0)
        sd_curve = padded.std(axis=0)
        ax_conv.plot(ep_x, m_curve, label=label, color=color, linestyle=ls, linewidth=2.1)
        if arm_key in ("octonion", "real", "lowrank8"):
            ax_conv.fill_between(ep_x, m_curve - sd_curve, m_curve + sd_curve, color=color, alpha=0.10)

    ax_conv.set_xlabel("Training Epoch")
    ax_conv.set_ylabel("Best-So-Far Validation MAE (Scaled)")
    ax_conv.set_title("A. Validation Convergence Trajectory (ILI · TimesNet)")
    ax_conv.legend(loc="upper right")

    # Right: Log2-spaced Compression Factor (1x -> 2x -> 4x -> 8x) on Spatial/Multi-Scale Mixers
    mixer_set = {"timesnet", "lightts", "micn", "msgnet", "crossformer"}
    sub_cfg = cfgs[cfgs["backbone"].isin(mixer_set)].copy()
    real_lookup = sub_cfg[sub_cfg["arm"] == "real"].set_index(["dataset", "backbone"])["mae"]

    def _rel_gain(arm_name: str) -> tuple[float, float]:
        s = sub_cfg[sub_cfg["arm"] == arm_name].set_index(["dataset", "backbone"])["mae"]
        common = s.index.intersection(real_lookup.index)
        gains = (1.0 - s.loc[common] / real_lookup.loc[common]) * 100.0
        return float(gains.median()), float(gains.mean())

    pos_x = np.array([0.0, 1.0, 2.0, 3.0])
    hd_y = [0.0, _rel_gain("complex")[0], _rel_gain("quaternion")[0], _rel_gain("octonion")[0]]
    lr_y = [0.0, _rel_gain("lowrank2")[0], _rel_gain("lowrank4")[0], _rel_gain("lowrank8")[0]]

    ax_pareto.plot(
        pos_x, hd_y, color=PALETTE["hd"], marker="o", markersize=8.0, linewidth=2.4,
        label="HyperDense (Complex 2D / Quaternion 4D / Octonion 8D)", zorder=4
    )
    ax_pareto.plot(
        pos_x, lr_y, color=PALETTE["lr"], marker="s", markersize=7.5, linestyle="--", linewidth=2.1,
        label="Parameter-Matched Low-Rank (Rank-1/2 / 1/4 / 1/8)", zorder=3
    )
    for px, hy, ly in zip(pos_x, hd_y, lr_y):
        hd_up = hy >= ly
        ax_pareto.annotate(
            f"{hy:+.2f}%",
            (px, hy),
            textcoords="offset points",
            xytext=(0, 10 if hd_up else -15),
            ha="center",
            fontsize=8.6,
            fontweight="bold",
            color=PALETTE["hd"],
        )
        if px > 0:
            ax_pareto.annotate(
                f"{ly:+.2f}%",
                (px, ly),
                textcoords="offset points",
                xytext=(0, -15 if hd_up else 10),
                ha="center",
                fontsize=8.3,
                fontweight="bold",
                color=PALETTE["lr"],
            )
    ax_pareto.axhline(0.0, color="#64748B", linewidth=1.0)
    ax_pareto.set_xticks(pos_x)
    ax_pareto.set_xticklabels([
        "100% Budget\n(1× Dense Real)",
        "50% Budget\n(2× Compression)",
        "25% Budget\n(4× Compression)",
        "12.5% Budget\n(8× Compression)",
    ])
    ax_pareto.set_xlim(-0.25, 3.25)
    ax_pareto.set_ylim(min(hd_y + lr_y) - 0.22, max(hd_y + lr_y) + 0.22)
    ax_pareto.set_xlabel("Replaced Layer Parameter Budget (% of Unconstrained Real)")
    ax_pareto.set_ylabel("Median Test MAE Gain vs. Real (%)")
    ax_pareto.set_title("B. Compression–Accuracy Pareto on Spatial / Multi-Scale Mixers")
    ax_pareto.legend(loc="lower left")

    fig.suptitle(
        "Optimization Convergence & Parameter–Accuracy Tradeoff",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(assets_dir / "fig5_pareto_and_convergence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_dataset_card(repo_id: str, run_dir: Path) -> str:
    """Render a visually rich Hugging Face Dataset Card with embedded charts & tables."""
    stats_data = json.loads((run_dir / "statistical-tests.json").read_text(encoding="utf-8"))
    comp = stats_data["comparisons"]
    n_cells = stats_data["complete_cells"]
    n_fits = stats_data["total_completed_fits"]

    cells = pd.read_csv(run_dir / "validation-selected-cells.csv")
    cfgs = pd.read_csv(run_dir / "all-datasets-summary.csv")

    def _fmt(key: str) -> str:
        c = comp[key]
        return (
            f"**{c['wins']} / {c['total']} ({c['win_rate']*100:.1f}%)** | "
            f"`{c['median_pct']:+.2f}%` | `{c['mean_pct']:+.2f}%` | "
            f"`{c['wilcoxon_p']:.3g}` | `{c['sign_test_p']:.3g}`"
        )

    ds_table_rows = []
    for ds_id, label in DATASET_LABELS.items():
        sub_c = cells[cells["dataset"] == ds_id]
        sub_g = cfgs[cfgs["dataset"] == ds_id]
        best = sub_g.loc[sub_g["mae"].idxmin()]
        n = len(sub_c)
        wr_real = f"{(sub_c['val_hd_vs_real_pct'] > 0).sum()}/{n}"
        wr_lr = f"{(sub_c['val_hd_vs_val_lr_pct'] > 0).sum()}/{n}"
        med_r = float(sub_c["val_hd_vs_real_pct"].median())
        ds_table_rows.append(
            f"| **{label}** | `{wr_real}` (`{med_r:+.2f}%`) | `{wr_lr}` | "
            f"`{BACKBONE_LABELS.get(best['backbone'], best['backbone'])} + {best['arm']}` | "
            f"`{best['mae']:.4f}` (`{best['mae_over_persistence']:.3f}×` pers.) |"
        )
    ds_table_md = "\n".join(ds_table_rows)

    return f"""---
pretty_name: "HyperCAST TSLib-15 Multi-Dataset HyperDense Benchmark Results"
license: apache-2.0
task_categories:
  - time-series-forecasting
tags:
  - time-series
  - forecasting
  - hypercomplex
  - quaternion
  - octonion
  - tslib
  - benchmark
  - hypercast
size_categories:
  - 1K<n<10K
configs:
  - config_name: fits
    default: true
    data_files:
      - split: test
        path: data/fits.parquet
  - config_name: configs
    data_files:
      - split: test
        path: data/configs.parquet
  - config_name: validation_cells
    data_files:
      - split: test
        path: data/validation_cells.parquet
---

# HyperCAST: Multi-Dataset TSLib-15 × HyperDense Benchmark (`3,157` GPU Fits)

[![GitHub Repo](https://img.shields.io/badge/GitHub-w3joe%2Fhypercast-181717?logo=github)](https://github.com/w3joe/hypercast)
[![Completed Fits](https://img.shields.io/badge/GPU_Fits-3%2C157_%2F_3%2C168-4F46E5)](#)
[![Datasets](https://img.shields.io/badge/Datasets-9_Domains_(7--862_Channels)-0D9488)](#per-dataset-breakdown)
[![Backbones](https://img.shields.io/badge/TSLib_Backbones-15_Architectures-D97706)](#architecture-family-sensitivity)
[![Hardware](https://img.shields.io/badge/Compute-16%C3%97_NVIDIA_L4_(IEEE_FP32)-334155)](#)

Complete experimental records, validation-selected statistical tests, and per-epoch training trajectories for the **HyperCAST 9-Dataset × 15-Backbone × 8-Arm (`tslib15-multi-20261119`)** benchmark evaluating structured hypercomplex projections (**Complex 2D**, **Quaternion 4D**, **Octonion 8D**) against unmodified **TSLib (`native`)**, variance-matched **Dense (`real`)**, and parameter-matched **Low-Rank (`lowrank2`, `lowrank4`, `lowrank8`)** controls.

![Executive Benchmark Overview](assets/fig1_hero_overview.png)

---

## Key Takeaways at a Glance

1. **Lossless 2×–8× Layer Compression Across Diverse Domains:** Selecting the hypercomplex algebra strictly on inner validation MAE (**zero test leakage**) beats variance-matched dense `real` on **79 / 131 cells (`60.3%`, Sign *p* = `0.023`, Wilcoxon *p* = `0.048`)** while matching unmodified `native` TSLib (`51.9%`, *p* = `0.57`) at **12.5%–50%** of the replaced site's parameter budget.
2. **Sharp High-Dimensional Regime Gain (`321–862` Channels):** On wide multivariate sensor grids (`electricity` 321ch and `traffic` 862ch), structured compression yields **`+3.66%` median (`+4.25%` mean)** test MAE improvement over `real` (`+4.46%` on `traffic`, beating both `native` and `lowrank`).
3. **Clear Architectural Split (Spatial/Multi-Scale Mixers vs. Channel-Independent):** Single-site HyperDense ties parameter-matched `Low-Rank` overall (`48.9%` vs. `51.1%`, *p* = `0.74`), because 12 of 15 default TSLib sites act purely along single-channel time steps. On **multi-scale & spatial mixers**, HyperDense consistently outperforms parameter-matched `Low-Rank` (**`TimesNet` `+1.73%`**, **`LightTS` `+0.84%` [7/9 wins]**, **`MICN` `+0.68%`**, **`Crossformer` `+0.36%`**, **`MSGNet` `+0.33%`**).

---

## Pooled Zero-Leakage Statistical Summary (`N = {n_cells}` Complete `8-Arm × 3-Seed` Cells)

Algebra selection (`complex`, `quaternion`, `octonion`) and rank selection (`lowrank2`, `lowrank4`, `lowrank8`) are performed strictly on `inner` validation MAE before evaluating on held-out chronological `test` origins:

| Comparison | Cell Win Rate | Median Gain | Mean Gain | Wilcoxon *p* | Sign *p* |
|---|---:|---:|---:|---:|---:|
| **Val-HyperDense vs. `real` (Dense)** | {_fmt('val_hd_vs_real')} |
| **Val-LowRank vs. `real` (Dense)** | {_fmt('val_lr_vs_real')} |
| **Val-HyperDense vs. `native` (TSLib)** | {_fmt('val_hd_vs_native')} |
| **Val-HyperDense vs. Val-LowRank** | {_fmt('val_hd_vs_val_lr')} |
| **Oracle 3v3 HyperDense vs. LowRank** | {_fmt('oracle_hd_vs_oracle_lr')} |

---

## Architecture Family Sensitivity

![Backbone Architecture Sensitivity](assets/fig2_backbone_architecture_ranking.png)

---

## Complete `9 × 15` Benchmark Matrix

Every cell shows held-out test MAE gain (%) of **Validation-Selected HyperDense vs. `real`** averaged over 3 seeds (`3301, 3302, 3303`). Cells marked with **★** also outperform the validation-selected **parameter-matched Low-Rank** control:

![9x15 Matrix Heatmap](assets/fig3_heatmap_9x15.png)

---

## Multi-Step Horizon Error Curves (`t+1 → t+5`) & Optimization Pareto

![Multi-Step Horizon Error Curves](assets/fig4_lead_horizon_curves.png)

![Convergence and Compression-Accuracy Pareto](assets/fig5_pareto_and_convergence.png)

---

## Per-Dataset Breakdown

All datasets use a strict `45% train / 15% inner-val / 20% development / 20% test` chronological split (`context=32`, `horizon=5`, `epochs=100`, `patience=20`, `batch_size=32`, `lr=0.001`, training-only min–max scaling, and a 14,400-row chronological prefix cap on >14.4k datasets matching ETTh1):

| Dataset (Channels · Freq) | Val-HD > `real` (Median) | Val-HD > `lowrank` | #1 Overall Config (120 Arms) | Best Test MAE (`× Persistence`) |
|---|---:|---:|---|---:|
{ds_table_md}

---

## Quick Load in Python

```python
from datasets import load_dataset

# 1. Load all 3,157 individual GPU seed fits
fits = load_dataset("{repo_id}", "fits", split="test").to_pandas()

# 2. Load 1,054 seed-aggregated (dataset, backbone, arm) summaries
configs = load_dataset("{repo_id}", "configs", split="test").to_pandas()

# 3. Load 131 zero-leakage validation-selected (dataset, backbone) comparisons
cells = load_dataset("{repo_id}", "validation_cells", split="test").to_pandas()
print(cells[["dataset", "backbone", "val_selected_hd_arm", "val_hd_vs_real_pct", "val_hd_vs_val_lr_pct"]].head(10))
```

## Repository Files

| Path | Description |
|---|---|
| `data/fits.parquet` | Default HF Dataset Viewer table (`{n_fits}` rows — every seed fit, metrics, runtime, VRAM, SHA-256) |
| `data/configs.parquet` | Seed-aggregated `(dataset, backbone, arm)` summary table (`1,054` rows) |
| `data/validation_cells.parquet` | Leak-free validation-selected cells (`{n_cells}` rows) |
| `assets/fig1..fig5_*.png` | High-resolution benchmark visualizations (220 DPI) |
| `csv/` | Plain CSV mirrors (`raw-all-3157-fits.csv`, `all-datasets-summary.csv`, `validation-selected-cells.csv`, per-dataset `summary.csv`) |
| `statistical-tests.json` | Exact Wilcoxon signed-rank and binomial sign-test JSON outputs |
| `archives/tslib15-multi-20261119-results.zip` | Complete raw trial archive (`{n_fits}` `result.json` files with epoch-by-epoch curves and per-lead metrics) |
"""


def stage_hf_repo(run_dir: Path, full_zip: Path, stage_dir: Path, repo_id: str) -> None:
    """Populate ``stage_dir`` with Parquet splits, figures, CSVs, JSONs, and full archive."""
    data_dir = stage_dir / "data"
    csv_dir = stage_dir / "csv"
    arch_dir = stage_dir / "archives"
    assets_dir = stage_dir / "assets"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    arch_dir.mkdir(parents=True, exist_ok=True)

    generate_figures(run_dir, assets_dir)

    fits_df = pd.read_csv(run_dir / "raw-all-3157-fits.csv")
    cfgs_df = pd.read_csv(run_dir / "all-datasets-summary.csv")
    cells_df = pd.read_csv(run_dir / "validation-selected-cells.csv")

    fits_df.to_parquet(data_dir / "fits.parquet", index=False)
    cfgs_df.to_parquet(data_dir / "configs.parquet", index=False)
    cells_df.to_parquet(data_dir / "validation_cells.parquet", index=False)

    for fname in (
        "raw-all-3157-fits.csv",
        "all-datasets-summary.csv",
        "validation-selected-cells.csv",
    ):
        shutil.copy2(run_dir / fname, csv_dir / fname)

    for ds_csv in sorted((run_dir / "datasets").glob("*/summary.csv")):
        dst = csv_dir / "datasets" / ds_csv.parent.name / "summary.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ds_csv, dst)

    shutil.copy2(run_dir / "statistical-tests.json", stage_dir / "statistical-tests.json")
    if full_zip.is_file():
        shutil.copy2(full_zip, arch_dir / full_zip.name)

    (stage_dir / "README.md").write_text(
        build_dataset_card(repo_id, run_dir), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "results" / "tslib15-multi-20261119",
    )
    parser.add_argument(
        "--full-zip",
        type=Path,
        default=ROOT.parent / "tslib15-multi-20261119-results.zip",
    )
    parser.add_argument(
        "--namespace",
        default="",
        help="HF username or org (defaults to authenticated token owner)",
    )
    parser.add_argument(
        "--dataset-name",
        default="hypercast-tslib15-multi-benchmark",
    )
    parser.add_argument(
        "--collection-title",
        default="HyperCAST: Hypercomplex Time-Series Forecasting",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    full_zip = args.full_zip.resolve()

    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        generate_figures(run_dir, args.preview_dir)
        print(f"Rendered preview figures in {args.preview_dir}")
        return

    if args.dry_run:
        ns = args.namespace or "preview-user"
        repo_id = f"{ns}/{args.dataset_name}"
        with tempfile.TemporaryDirectory(prefix="hf-stage-") as tmp:
            st = Path(tmp)
            stage_hf_repo(run_dir, full_zip, st, repo_id)
            files = sorted(str(p.relative_to(st)) for p in st.rglob("*") if p.is_file())
            total_mb = sum(p.stat().st_size for p in st.rglob("*") if p.is_file()) / (1024 * 1024)
            print(f"Dry-run staged {len(files)} files ({total_mb:.2f} MB) for {repo_id}:")
            for f in files:
                print(" ", f)
        return

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    who = api.whoami()
    namespace = args.namespace or who["name"]
    repo_id = f"{namespace}/{args.dataset_name}"

    print(f"Authenticated as {who['name']}; creating/updating dataset {repo_id} (private={args.private})...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(prefix="hf-stage-") as tmp:
        st = Path(tmp)
        stage_hf_repo(run_dir, full_zip, st, repo_id)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(st),
            commit_message="Add benchmark comparison charts and enhanced visual Dataset Card",
        )

    dataset_url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Dataset uploaded: {dataset_url}")

    existing_col: Any = None
    for col in list_collections(owner=namespace, token=token):
        if col.title == args.collection_title:
            existing_col = col
            break

    if existing_col is None:
        col = create_collection(
            title=args.collection_title,
            namespace=namespace,
            description=(
                "Benchmark datasets, multi-dataset TSLib ablation archives, and "
                "hypercomplex forecasting artifacts for HyperCAST (github.com/w3joe/hypercast)."
            ),
            private=args.private,
            token=token,
        )
    else:
        col = existing_col

    add_collection_item(
        collection_slug=col.slug,
        item_id=repo_id,
        item_type="dataset",
        note="9-dataset x 15 TSLib backbones x 8 arms x 3 seeds benchmark (3,157 GPU fits on 16x L4)",
        exists_ok=True,
        token=token,
    )
    print(f"Collection ready: https://huggingface.co/collections/{col.slug}")


if __name__ == "__main__":
    main()
