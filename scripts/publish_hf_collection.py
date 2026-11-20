#!/usr/bin/env python3
"""Stage and publish HyperCAST benchmark results to a Hugging Face Dataset & Collection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd
from huggingface_hub import (
     HfApi,
    add_collection_item,
    create_collection,
    list_collections,
)

ROOT = Path(__file__).resolve().parents[1]


def build_dataset_card(repo_id: str, run_dir: Path) -> str:
    """Render a Hugging Face Dataset Card with YAML frontmatter for Dataset Viewer."""
    stats_data = json.loads((run_dir / "statistical-tests.json").read_text(encoding="utf-8"))
    comp = stats_data["comparisons"]
    n_cells = stats_data["complete_cells"]
    n_fits = stats_data["total_completed_fits"]

    def _fmt(key: str) -> str:
        c = comp[key]
        return (
            f"{c['wins']} / {c['total']} ({c['win_rate']*100:.1f}%) | "
            f"{c['median_pct']:+.2f}% | {c['mean_pct']:+.2f}% | "
            f"`{c['wilcoxon_p']:.3g}` | `{c['sign_test_p']:.3g}`"
        )

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

# {repo_id.split('/')[-1]}

Complete experimental artifacts and statistical summaries from the **HyperCAST 9-Dataset × 15-Backbone × 8-Arm (`tslib15-multi-20261119`)** study evaluating hypercomplex linear projections (`HyperDense`) against unmodified TSLib (`native`), variance-matched dense (`real`), and parameter-matched low-rank (`lowrank2`, `lowrank4`, `lowrank8`) controls.

- **Codebase:** [github.com/w3joe/hypercast](https://github.com/w3joe/hypercast)
- **Protocol:** `tslib15-multi-v1-deterministic-cudnn` (`context=32`, `horizon=5`, `epochs=100`, `patience=20`, `lr=0.001`, 14,400-row chronological prefix cap matching ETTh1; strict `45% / 15% / 20% / 20%` chronological split)
- **Compute:** 2 × GCP `g2-standard-96` (16 × NVIDIA L4 24GB GPUs, IEEE FP32 deterministic cuDNN)
- **Scope:** `{n_fits} / 3,168` completed fits across **9 datasets** (`ili`, `nyc_taxi`, `exchange`, `etth2`, `ettm1`, `ettm2`, `weather`, `electricity`, `traffic`), **15 TSLib backbones**, **8 arms**, and **3 seeds** (`3301, 3302, 3303`).

---

## Pooled Zero-Leakage Statistical Summary (`N = {n_cells}` complete `(dataset, backbone)` cells)

Algebra selection (`complex`, `quaternion`, `octonion`) and rank selection (`lowrank2`, `lowrank4`, `lowrank8`) are performed strictly on inner validation MAE before scoring on held-out test origins:

| Comparison | Win Rate | Median Gain | Mean Gain | Wilcoxon *p* | Sign *p* |
|---|---:|---:|---:|---:|---:|
| **Val-HyperDense vs. `real`** | {_fmt('val_hd_vs_real')} |
| **Val-LowRank vs. `real`** | {_fmt('val_lr_vs_real')} |
| **Val-HyperDense vs. `native`** | {_fmt('val_hd_vs_native')} |
| **Val-HyperDense vs. Val-LowRank** | {_fmt('val_hd_vs_val_lr')} |
| **Oracle 3v3 HyperDense vs. LowRank** | {_fmt('oracle_hd_vs_oracle_lr')} |

---

## Repository Structure

- **`data/fits.parquet`** (`{n_fits}` rows, default viewer config) — Every completed seed fit with test MAE/RMSE/MSE/bias, inner validation MAE, persistence MAE, epochs, runtime, GPU memory, parameter counts, and checkpoint SHA-256 digests.
- **`data/configs.parquet`** (`1,054` rows) — Seed-aggregated `(dataset, backbone, arm)` summary table.
- **`data/validation_cells.parquet`** (`{n_cells}` rows) — Leak-free validation-selected comparisons for every complete `(dataset, backbone)` cell.
- **`csv/`** — CSV mirrors (`raw-all-3157-fits.csv`, `all-datasets-summary.csv`, `validation-selected-cells.csv`, and per-dataset `datasets/<dataset>/summary.csv`).
- **`statistical-tests.json`** — Exact Wilcoxon signed-rank and binomial sign-test outputs.
- **`archives/tslib15-multi-20261119-results.zip`** — Full archive containing all `{n_fits}` raw `trials/<dataset>/<backbone>__<arm>__seed<seed>/result.json` files (including per-epoch training/validation curves and per-lead metrics) plus GCP run metadata.

## Quick Load in Python

```python
from datasets import load_dataset

fits = load_dataset("{repo_id}", "fits", split="test").to_pandas()
cells = load_dataset("{repo_id}", "validation_cells", split="test").to_pandas()
print(cells[["dataset", "backbone", "val_selected_hd_arm", "val_hd_vs_real_pct", "val_hd_vs_val_lr_pct"]].head())
```
"""


def stage_hf_repo(run_dir: Path, full_zip: Path, stage_dir: Path, repo_id: str) -> None:
    """Populate ``stage_dir`` with Parquet viewer splits, CSVs, JSONs, and full archive."""
    data_dir = stage_dir / "data"
    csv_dir = stage_dir / "csv"
    arch_dir = stage_dir / "archives"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    arch_dir.mkdir(parents=True, exist_ok=True)

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
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    full_zip = args.full_zip.resolve()

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
            commit_message="Upload HyperCAST 9-dataset TSLib-15 x 8-arm benchmark results (3,157 fits)",
        )

    dataset_url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Dataset uploaded: {dataset_url}")

    # Find existing collection or create a new one
    existing_col = None
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
