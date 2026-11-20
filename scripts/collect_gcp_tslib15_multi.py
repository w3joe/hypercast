#!/usr/bin/env python3
"""Check live progress and collect completed multi-dataset GCP trial results from GCS."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _gcloud(account: str, *args: str, check: bool = False, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gcloud", f"--account={account}", *args, "--quiet"],
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Sync completed trials from GCS and build per-dataset summaries")
    parser.add_argument("--include-predictions", action="store_true", help="Also download predictions.npz files (default skips them for fast result.json sync)")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    launch = json.loads((run_dir / "launch.json").read_text(encoding="utf-8"))
    account = launch["account"]
    project = launch["project"]
    prefix = launch["gcs_prefix"]

    print(f"=== Run {launch['run_id']} ({prefix}) ===")
    for vm in launch["vms"]:
        st = _gcloud(account, "compute", "instances", "describe", vm["name"], f"--project={project}", f"--zone={vm['zone']}", "--format=value(status)")
        vm_status = st.stdout.strip() if st.returncode == 0 else "DELETED/GONE"
        prog = _gcloud(account, "storage", "cat", f"{prefix}/vm-{vm['shard']}-progress.json")
        if prog.returncode == 0:
            p = json.loads(prog.stdout)
            print(
                f"  {vm['name']} ({vm['zone']}, {vm_status}): state={p.get('state')} "
                f"ok={p.get('completed', 0)}/{p.get('assigned', 0)} fail={p.get('failed', 0)} "
                f"updated={p.get('updated_at')}"
            )
        else:
            print(f"  {vm['name']} ({vm['zone']}, {vm_status}): bootstrapping (no progress.json yet)")

    if not args.download:
        return

    trials_dir = run_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    print(f"Syncing {prefix}/trials -> {trials_dir} ...")
    rsync_args = ["storage", "rsync", "--recursive"]
    if not args.include_predictions:
        rsync_args.append(r"--exclude=.*predictions\.npz$")
    rsync_args.extend([f"{prefix}/trials", str(trials_dir)])
    _gcloud(account, *rsync_args, check=True, timeout=900)

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for res_file in sorted(trials_dir.rglob("result.json")):
        rec = json.loads(res_file.read_text(encoding="utf-8"))
        by_dataset[rec["job"]["dataset"]].append(rec)

    all_summary_rows: list[dict[str, Any]] = []
    for ds_id, records in sorted(by_dataset.items()):
        ds_out = run_dir / "datasets" / ds_id
        ds_out.mkdir(parents=True, exist_ok=True)
        (ds_out / "test-results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for rec in records:
            grouped[(rec["job"]["backbone"], rec["job"]["arm"])].append(rec)

        real_mae_by_backbone = {
            b: float(np.mean([r["mae"] for r in recs]))
            for (b, a), recs in grouped.items()
            if a == "real"
        }
        rows = []
        for (backbone, arm), recs in sorted(grouped.items()):
            maes = [float(r["mae"]) for r in recs]
            pers = [float(r["persistence_mae"]) for r in recs]
            mean_mae = float(np.mean(maes))
            real_mae = real_mae_by_backbone.get(backbone)
            imp_pct = (1.0 - mean_mae / real_mae) * 100.0 if real_mae and real_mae > 0 else None
            row = {
                "dataset": ds_id,
                "backbone": backbone,
                "arm": arm,
                "seeds": len(recs),
                "mae": mean_mae,
                "mae_sd": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
                "persistence_mae": float(np.mean(pers)),
                "mae_over_persistence": mean_mae / float(np.mean(pers)),
                "improvement_vs_real_pct": imp_pct,
                "parameters": int(recs[0]["control"]["parameters"]),
                "mean_epochs": float(np.mean([r["epochs_ran"] for r in recs])),
                "mean_train_seconds": float(np.mean([r["training_elapsed_seconds"] for r in recs])),
            }
            rows.append(row)
            all_summary_rows.append(row)
        pd.DataFrame(rows).to_csv(ds_out / "summary.csv", index=False)
        print(f"  {ds_id:12s}: {len(records):3d} fits -> {len(rows):3d} model/arm configs saved to {ds_out / 'summary.csv'}")

    if all_summary_rows:
        pd.DataFrame(all_summary_rows).to_csv(run_dir / "all-datasets-summary.csv", index=False)
        print(f"Combined summary written to {run_dir / 'all-datasets-summary.csv'} ({len(all_summary_rows)} configs)")


if __name__ == "__main__":
    main()
