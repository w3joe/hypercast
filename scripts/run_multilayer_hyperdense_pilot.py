#!/usr/bin/env python3
"""Leakage-safe development pilot runner for multi-layer cross-channel HyperDense."""

from __future__ import annotations

import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypercast4d.hyperdense_transfer import ieee_precision  # noqa: E402
from hypercast4d.multilayer_hyperdense import (  # noqa: E402
    MULTILAYER_MODELS,
    build_multilayer_model,
)
from hypercast4d.training import fit_model, predict, seed_everything  # noqa: E402
from hypercast4d.tslib15_multi import (  # noqa: E402
    SlidingWindowDataset,
    prepare_dataset_bundle,
)

DEFAULT_PILOT_ARMS: tuple[str, ...] = (
    "real",
    "complex",
    "split_complex",
    "quaternion",
    "coquaternion",
    "octonion",
    "lowrank2",
    "lowrank4",
    "lowrank8",
    "blockdiag2",
    "blockdiag4",
    "blockdiag8",
)
PILOT_HD_ARMS: tuple[str, ...] = (
    "complex",
    "split_complex",
    "quaternion",
    "coquaternion",
    "cl11",
    "octonion",
)
PILOT_LR_ARMS: tuple[str, ...] = ("lowrank2", "lowrank4", "lowrank8")
PILOT_BD_ARMS: tuple[str, ...] = ("blockdiag2", "blockdiag4", "blockdiag8")


def run_single_fit(
    manifest: dict[str, Any],
    bundle_bytes: bytes,
    *,
    model_name: str,
    arm: str,
    seed: int,
    budget_mode: str = "compress",
    base_width: int = 64,
    num_layers: int = 3,
    epochs: int = 50,
    patience: int = 15,
    learning_rate: float = 1e-3,
    eval_split: str = "development",
    device: str = "cpu",
) -> dict[str, Any]:
    """Train on ``train``, early-stop on ``inner``, and score on ``eval_split``."""
    if eval_split == "test" and not manifest.get("include_test", False):
        raise ValueError("Bundle excludes held-out test split (include_test=False)")

    with np.load(io.BytesIO(bundle_bytes), allow_pickle=False) as arrays:
        values = arrays["values"]
        train_starts = arrays["train_starts"]
        inner_starts = arrays["inner_starts"]
        eval_starts = arrays[f"{eval_split}_starts"]

    window = int(manifest["context"])
    horizon = int(manifest["horizon"])
    features = int(manifest["features"])

    train_ds = SlidingWindowDataset(values, train_starts, window=window, horizon=horizon)
    inner_ds = SlidingWindowDataset(values, inner_starts, window=window, horizon=horizon)
    eval_ds = SlidingWindowDataset(values, eval_starts, window=window, horizon=horizon)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(seed)

    model, report = build_multilayer_model(
        model_name,
        arm,
        seed,
        features=features,
        dataset_id=manifest["dataset"],
        window=window,
        horizon=horizon,
        base_width=base_width,
        num_layers=num_layers,
        budget_mode=budget_mode,
    )

    t0 = time.monotonic()
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    history: list[dict[str, float | int]] = []

    def _epoch_cb(epoch: int, train_loss: float, val_loss: float) -> None:
        history.append(
            {
                "epoch": int(epoch),
                "train_mae": float(train_loss),
                "inner_mae": float(val_loss),
            }
        )

    with ieee_precision():
        fit = fit_model(
            model,
            train_ds,
            inner_ds,
            seed=seed,
            epochs=epochs,
            batch_size=32,
            learning_rate=learning_rate,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            adam_amsgrad=False,
            loss_name="mae",
            shuffle=True,
            early_stopping_patience=patience,
            early_stopping_min_delta=0.0,
            early_stopping_relative_delta=0.0,
            restore_best_weights=True,
            device=dev,
            epoch_callback=_epoch_cb,
        )
        preds_scaled = predict(model, eval_ds, dev, batch_size=64)
    elapsed = time.monotonic() - t0
    peak_gpu_bytes = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0

    span = float(manifest["scaler_span"][0])
    minimum = float(manifest["scaler_minimum"][0])
    actual_scaled = np.stack([values[t : t + horizon, 0] for t in eval_starts], axis=0)
    pers_scaled = np.repeat(values[eval_starts - 1, 0:1], horizon, axis=1)

    preds = preds_scaled * span + minimum
    actual = actual_scaled * span + minimum
    persistence = pers_scaled * span + minimum
    err = preds - actual

    return {
        "dataset": manifest["dataset"],
        "eval_split": eval_split,
        "model_name": model_name,
        "arm": arm,
        "budget_mode": budget_mode,
        "seed": seed,
        "base_width": int(base_width),
        "num_layers": int(num_layers),
        "inner_val_mae_scaled": float(fit.best_validation_loss),
        "mae": float(np.abs(err).mean()),
        "mse": float(np.square(err).mean()),
        "rmse": float(np.sqrt(np.square(err).mean())),
        "bias": float(err.mean()),
        "per_lead_mae": np.abs(err).mean(axis=0).tolist(),
        "per_lead_mse": np.square(err).mean(axis=0).tolist(),
        "per_lead_bias": err.mean(axis=0).tolist(),
        "persistence_mae": float(np.abs(persistence - actual).mean()),
        "epochs_ran": int(fit.epochs_ran),
        "best_epoch": int(fit.best_epoch),
        "train_seconds": round(elapsed, 2),
        "peak_gpu_bytes": peak_gpu_bytes,
        "parameters": int(report["parameters"]),
        "selected_parameters": int(report["selected_parameters"]),
        "selected_parameter_share": float(report["selected_parameter_share"]),
        "replaced_site_count": int(report["replaced_site_count"]),
        "history": history,
    }


def summarize_pilot(records: list[dict[str, Any]], out_dir: Path) -> None:
    """Aggregate pilot runs and evaluate leakage-free advancement gates."""
    flat_records = [
        {k: v for k, v in r.items() if k not in {"history", "per_lead_mae", "per_lead_mse", "per_lead_bias"}}
        for r in records
    ]
    df = pd.DataFrame(flat_records)
    df.to_csv(out_dir / "raw-fits.csv", index=False)

    cfg = df.groupby(["dataset", "model_name", "budget_mode", "arm"], as_index=False).agg(
        seeds=("mae", "count"),
        inner_val_mae=("inner_val_mae_scaled", "mean"),
        eval_mae=("mae", "mean"),
        eval_mae_sd=("mae", "std"),
        persistence_mae=("persistence_mae", "mean"),
        parameters=("parameters", "first"),
        selected_parameters=("selected_parameters", "first"),
        selected_share=("selected_parameter_share", "first"),
        mean_epochs=("epochs_ran", "mean"),
        mean_seconds=("train_seconds", "mean"),
    )
    cfg.to_csv(out_dir / "summary.csv", index=False)

    cell_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], pd.DataFrame] = defaultdict(pd.DataFrame)
    for key, sub in cfg.groupby(["dataset", "model_name", "budget_mode"]):
        grouped[key] = sub.set_index("arm")

    for (ds, mdl, bmode), m in sorted(grouped.items()):
        avail_hd = [a for a in PILOT_HD_ARMS if a in m.index]
        avail_lr = [a for a in PILOT_LR_ARMS if a in m.index]
        avail_bd = [a for a in PILOT_BD_ARMS if a in m.index]
        if not ("real" in m.index and avail_hd and avail_lr and avail_bd):
            continue
        best_hd = m.loc[avail_hd, "inner_val_mae"].idxmin()
        best_lr = m.loc[avail_lr, "inner_val_mae"].idxmin()
        best_bd = m.loc[avail_bd, "inner_val_mae"].idxmin()

        hd_mae = float(m.loc[best_hd, "eval_mae"])
        lr_mae = float(m.loc[best_lr, "eval_mae"])
        bd_mae = float(m.loc[best_bd, "eval_mae"])
        real_mae = float(m.loc["real", "eval_mae"])

        cell_rows.append(
            {
                "dataset": ds,
                "model_name": mdl,
                "budget_mode": bmode,
                "selected_hd": best_hd,
                "selected_lr": best_lr,
                "selected_bd": best_bd,
                "hd_vs_real_pct": (1.0 - hd_mae / real_mae) * 100.0,
                "hd_vs_lr_pct": (1.0 - hd_mae / lr_mae) * 100.0,
                "hd_vs_bd_pct": (1.0 - hd_mae / bd_mae) * 100.0,
            }
        )

    if not cell_rows:
        return
    cdf = pd.DataFrame(cell_rows)
    cdf.to_csv(out_dir / "gate-cells.csv", index=False)
    n = len(cdf)
    print(f"\n=== Multi-Layer Cross-Channel Pilot Gate ({n} cells, split=development) ===")
    for label, col in (
        ("Inner-Selected HD vs Real     ", "hd_vs_real_pct"),
        ("Inner-Selected HD vs LowRank  ", "hd_vs_lr_pct"),
        ("Inner-Selected HD vs BlockDiag", "hd_vs_bd_pct"),
    ):
        vals = cdf[col].to_numpy()
        wins = int((vals > 0).sum())
        p_w = float(stats.wilcoxon(vals).pvalue) if n >= 5 and np.any(vals != 0) else float("nan")
        print(
            f"  {label}: wins={wins}/{n} ({wins/n*100:5.1f}%) "
            f"median={np.median(vals):+6.2f}% mean={np.mean(vals):+6.2f}% Wilcoxon p={p_w:.3g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="ili,etth2,weather,electricity")
    parser.add_argument("--models", default=",".join(MULTILAYER_MODELS))
    parser.add_argument("--arms", default=",".join(DEFAULT_PILOT_ARMS))
    parser.add_argument("--budget-modes", default="compress,isoparam")
    parser.add_argument("--seeds", default="4101,4102,4103")
    parser.add_argument("--base-width", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--max-rows", type=int, default=14400)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "multilayer-hyperdense-pilot",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    budget_modes = [b.strip() for b in args.budget_modes.split(",") if b.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for ds_id in datasets:
        blob, manifest = prepare_dataset_bundle(
            ds_id,
            ROOT,
            include_test=False,
            max_rows=args.max_rows if args.max_rows > 0 else None,
        )
        for bmode in budget_modes:
            for mdl in models:
                for arm in arms:
                    for seed in seeds:
                        rec = run_single_fit(
                            manifest,
                            blob,
                            model_name=mdl,
                            arm=arm,
                            seed=seed,
                            budget_mode=bmode,
                            base_width=args.base_width,
                            num_layers=args.num_layers,
                            epochs=args.epochs,
                            patience=args.patience,
                            eval_split="development",
                            device=args.device,
                        )
                        records.append(rec)
                        print(
                            f"[{ds_id:11s} | {bmode:8s} | {mdl:20s} | {arm:13s} | s={seed}] "
                            f"dev_mae={rec['mae']:.4f} (pers={rec['persistence_mae']:.4f}) "
                            f"params={rec['parameters']:,d} ({rec['selected_parameter_share']*100:.1f}% site) "
                            f"ep={rec['epochs_ran']} ({rec['train_seconds']:.1f}s)",
                            flush=True,
                        )

    summarize_pilot(records, args.out_dir)


if __name__ == "__main__":
    main()
