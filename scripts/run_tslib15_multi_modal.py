#!/usr/bin/env python3
"""Modal GPU calibration & full-study runner for the 15-model × 8-arm benchmark across datasets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypercast4d.tslib15 import ARMS, BACKBONES  # noqa: E402
from hypercast4d.tslib15_multi import (  # noqa: E402
    DEFAULT_EVALUATION_SEEDS,
    NEW_BENCHMARK_DATASETS,
    prepare_dataset_bundle,
)


def _remote_calibrate_dataset_backbone(
    manifest: dict[str, Any],
    bundle_bytes: bytes,
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run short calibration jobs for one (dataset, backbone) inside a Modal GPU container."""
    import torch
    from hypercast4d.tslib15_multi import run_trial

    torch.set_num_threads(2)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for job in jobs:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                res = run_trial(manifest, bundle_bytes, job, Path(tmp), device="cuda")
                results.append(res)
            except Exception as err:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                errors.append(
                    {
                        "job": job,
                        "error_type": type(err).__name__,
                        "error": str(err),
                        "traceback": traceback.format_exc()[-1200:],
                    }
                )
    return {"gpu": gpu_name, "results": results, "errors": errors}


def _remote_evaluate_trial(
    manifest: dict[str, Any],
    bundle_bytes: bytes,
    job: dict[str, Any],
) -> dict[str, Any]:
    """Run one full test evaluation trial inside a Modal GPU container."""
    import torch
    from hypercast4d.tslib15_multi import run_trial

    torch.set_num_threads(2)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            res = run_trial(manifest, bundle_bytes, job, tmp_path, device="cuda")
            pred_bytes = (
                (tmp_path / "predictions.npz").read_bytes()
                if (tmp_path / "predictions.npz").is_file()
                else None
            )
            return {"ok": True, "gpu": gpu_name, "result": res, "predictions": pred_bytes}
        except Exception as err:
            return {
                "ok": False,
                "gpu": gpu_name,
                "job": job,
                "error_type": type(err).__name__,
                "error": str(err),
                "traceback": traceback.format_exc()[-1500:],
            }


def _modal_hourly_rate(gpu: str, cpu: int = 2, mem_gib: int = 8) -> float:
    cli = str(Path(sys.executable).with_name("modal"))
    rates = json.loads(
        subprocess.run(
            [cli, "billing", "rates", "--json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    )
    gpu_key = {
        "L4": "gpu_hour_cost_l4",
        "A10G": "gpu_hour_cost_a10g",
        "L40S": "gpu_hour_cost_l40s",
        "A100-40GB": "gpu_hour_cost_a100_40gb",
        "A100-80GB": "gpu_hour_cost_a100_80gb",
        "H100": "gpu_hour_cost_h100",
    }[gpu]
    return (
        float(rates[gpu_key])
        + cpu * float(rates["cpu_hour_cost"])
        + mem_gib * float(rates["mem_gib_hour_cost"])
    )


def _build_modal_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "einops==0.8.2",
            "scipy==1.18.1",
            "numpy==2.5.3",
            "openpyxl==3.1.5",
            "pandas==3.0.6",
            "pyarrow==26.0.1",
            "torch==2.14.0",
        )
        .env({"PYTHONPATH": "/opt/hypercast/src", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        .add_local_dir(
            ROOT / "src",
            remote_path="/opt/hypercast/src",
            ignore=["**/__pycache__/**", "**/web_dist/**", "**/*.egg-info/**"],
        )
    )


def run_calibration(
    datasets: list[str],
    output_dir: Path,
    *,
    gpu: str = "L4",
    max_containers: int = 10,
    epochs: int = 2,
    timeout_seconds: int = 420,
) -> dict[str, Any]:
    """Run a 2-epoch label-free calibration across ``datasets`` and project full-study costs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hourly_rate = _modal_hourly_rate(gpu, cpu=2, mem_gib=8)
    print(f"Preparing {len(datasets)} datasets for {gpu} calibration (${hourly_rate:.4f}/worker-hr)...", flush=True)

    bundles: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for ds_id in datasets:
        blob, manifest = prepare_dataset_bundle(ds_id, ROOT, include_test=False)
        bundles[ds_id] = (blob, manifest)
        print(
            f"  {ds_id:12s}: rows={manifest['rows']:6d} features={manifest['features']:3d} "
            f"train_windows={manifest['partitions']['train']['origins']:5d} bundle={len(blob)/1e6:.2f}MB",
            flush=True,
        )

    image = _build_modal_image()
    app = modal.App("hypercast4d-tslib15-multi-calibration")
    remote_fn = app.function(
        image=image,
        gpu=gpu,
        cpu=(2, 2),
        memory=(8192, 8192),
        retries=0,
        timeout=timeout_seconds + 120,
        startup_timeout=180,
        max_containers=max_containers,
        scaledown_window=60,
        serialized=True,
    )(_remote_calibrate_dataset_backbone)

    tasks: list[tuple[str, str, list[dict[str, Any]]]] = []
    for ds_id in datasets:
        for backbone in BACKBONES:
            # Time quaternion on every backbone, plus native on FiLM in separate containers
            # so wide-channel FiLM runs stay isolated and never share timeouts.
            arms = ("native", "quaternion") if backbone == "film" else ("quaternion",)
            for arm in arms:
                tasks.append(
                    (
                        ds_id,
                        backbone,
                        [
                            {
                                "dataset": ds_id,
                                "phase": "calibration",
                                "backbone": backbone,
                                "arm": arm,
                                "seed": 3102,
                                "epochs": epochs,
                                "learning_rate": 0.001,
                                "timeout_seconds": timeout_seconds,
                            }
                        ],
                    )
                )

    print(f"Launching {len(tasks)} calibration containers (max {max_containers} concurrent {gpu}s)...", flush=True)
    started = time.perf_counter()
    raw_results: list[dict[str, Any]] = []
    raw_errors: list[dict[str, Any]] = []

    with modal.enable_output(), app.run():
        def invoke(item: tuple[str, str, list[dict[str, Any]]]) -> tuple[str, str, dict[str, Any]]:
            ds_id, backbone, jobs = item
            blob, manifest = bundles[ds_id]
            out = remote_fn.remote(manifest, blob, jobs)
            return ds_id, backbone, out

        with ThreadPoolExecutor(max_workers=max_containers) as pool:
            futures = [pool.submit(invoke, item) for item in tasks]
            completed = 0
            for fut in as_completed(futures):
                completed += 1
                try:
                    ds_id, backbone, out = fut.result()
                    raw_results.extend(out["results"])
                    raw_errors.extend(out["errors"])
                    status_tag = "OK" if not out["errors"] else f"ERR({len(out['errors'])})"
                    sec_list = [
                        round(r["timings"][-1]["total_seconds"], 2)
                        for r in out["results"]
                    ]
                    vram_mb = [
                        round((r.get("peak_gpu_bytes") or 0) / (1024**2), 1)
                        for r in out["results"]
                    ]
                    print(
                        f"  [{completed:3d}/{len(tasks)}] {ds_id:12s} {backbone:13s} {status_tag:6s} "
                        f"max_epoch_s={sec_list} peak_vram_mb={vram_mb}",
                        flush=True,
                    )
                except Exception as exc:
                    raw_errors.append({"error": str(exc)})
                    print(f"  [{completed:3d}/{len(tasks)}] CONTAINER ERROR: {exc}", flush=True)

    wall_seconds = time.perf_counter() - started
    # Summarize per-dataset projections for full 360-fit (15 x 8 x 3) study at 100-epoch ceiling and ~45-epoch realistic average.
    dataset_summaries: list[dict[str, Any]] = []
    for ds_id in datasets:
        manifest = bundles[ds_id][1]
        ds_res = [r for r in raw_results if r["job"]["dataset"] == ds_id]
        ds_err = [e for e in raw_errors if e.get("job", {}).get("dataset") == ds_id]
        per_backbone: dict[str, dict[str, Any]] = {}
        for r in ds_res:
            b = r["job"]["backbone"]
            warmup_s = r["timings"][0]["total_seconds"]
            steady_s = (
                r["timings"][-1]["total_seconds"]
                if len(r["timings"]) >= 2
                else warmup_s
            )
            vram_mb = (r.get("peak_gpu_bytes") or 0) / (1024**2)
            prev = per_backbone.get(b)
            if prev is None or steady_s > prev["sec_per_epoch"]:
                per_backbone[b] = {
                    "warmup_epoch_s": round(warmup_s, 4),
                    "sec_per_epoch": round(steady_s, 4),
                    "peak_vram_mb": round(vram_mb, 1),
                    "parameters": r["control"]["parameters"],
                    "arm_sampled": r["job"]["arm"],
                }
        # 8 arms * 3 seeds = 24 fits per backbone; account for 1 warmup epoch + (E-1) steady epochs + 1s overhead.
        gpu_hours_100 = (
            sum((v["warmup_epoch_s"] + v["sec_per_epoch"] * 99 + 1.0) * 24 for v in per_backbone.values())
            / 3600.0
        )
        gpu_hours_45 = (
            sum((v["warmup_epoch_s"] + v["sec_per_epoch"] * 44 + 1.0) * 24 for v in per_backbone.values())
            / 3600.0
        )
        slowest = sorted(
            per_backbone.items(), key=lambda kv: -kv[1]["sec_per_epoch"]
        )[:3]
        dataset_summaries.append(
            {
                "dataset": ds_id,
                "rows": manifest["rows"],
                "features": manifest["features"],
                "train_windows": manifest["partitions"]["train"]["origins"],
                "backbones_ok": len(per_backbone),
                "errors": ds_err,
                "max_vram_mb": max((v["peak_vram_mb"] for v in per_backbone.values()), default=0.0),
                "projected_gpu_hours_100ep": round(gpu_hours_100, 2),
                "projected_gpu_hours_45ep": round(gpu_hours_45, 2),
                "projected_cost_100ep_usd": round(gpu_hours_100 * hourly_rate, 2),
                "projected_cost_45ep_usd": round(gpu_hours_45 * hourly_rate, 2),
                "projected_wall_10gpu_45ep_hr": round(gpu_hours_45 / 10.0, 2),
                "slowest_backbones": [
                    {"backbone": b, **info} for b, info in slowest
                ],
                "per_backbone": per_backbone,
            }
        )

    summary = {
        "gpu": gpu,
        "hourly_rate_usd": hourly_rate,
        "calibration_epochs": epochs,
        "calibration_wall_seconds": round(wall_seconds, 1),
        "approx_calibration_cost_usd": round(
            sum(r["training_elapsed_seconds"] for r in raw_results) / 3600.0 * hourly_rate, 2
        ),
        "datasets": dataset_summaries,
        "total_errors": len(raw_errors),
    }
    (output_dir / "calibration-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "calibration-raw.json").write_text(
        json.dumps({"results": raw_results, "errors": raw_errors}, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["calibrate"], default="calibrate"
    )
    parser.add_argument(
        "--datasets",
        default=",".join(NEW_BENCHMARK_DATASETS),
        help="Comma-separated registered dataset IDs",
    )
    parser.add_argument("--gpu", default="L4", choices=["L4", "A10G", "L40S", "A100-40GB", "A100-80GB", "H100"])
    parser.add_argument("--max-containers", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=420)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "tslib15-multi-calibration",
    )
    args = parser.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    summary = run_calibration(
        datasets,
        args.output_dir,
        gpu=args.gpu,
        max_containers=args.max_containers,
        epochs=args.epochs,
        timeout_seconds=args.timeout_seconds,
    )
    print("\n=== CALIBRATION SUMMARY ===", flush=True)
    for row in summary["datasets"]:
        print(
            f"{row['dataset']:12s} ok={row['backbones_ok']:2d}/15 err={len(row['errors'])} "
            f"vram={row['max_vram_mb']:7.1f}MB "
            f"45ep={row['projected_gpu_hours_45ep']:6.2f}h (${row['projected_cost_45ep_usd']:6.2f}) "
            f"100ep={row['projected_gpu_hours_100ep']:6.2f}h (${row['projected_cost_100ep_usd']:6.2f})",
            flush=True,
        )
    print(f"Calibration wall time: {summary['calibration_wall_seconds']}s, approx GPU cost: ${summary['approx_calibration_cost_usd']}", flush=True)


if __name__ == "__main__":
    main()
