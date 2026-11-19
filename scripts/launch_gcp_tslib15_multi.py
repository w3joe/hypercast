#!/usr/bin/env python3
"""Prepare 9-dataset 15x8x3 benchmark bundles, upload to GCS, and launch 2x g2-standard-96 (16x L4) VMs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypercast4d.tslib15 import ARMS, BACKBONES  # noqa: E402
from hypercast4d.tslib15_multi import (  # noqa: E402
    DEFAULT_EVALUATION_SEEDS,
    NEW_BENCHMARK_DATASETS,
    prepare_dataset_bundle,
)

# Relative per-dataset/backbone cost weights (used only to snake-balance VM 0 vs VM 1 evenly).
SKIP_OUTLIERS: set[tuple[str, str]] = {
    ("electricity", "film"),
    ("traffic", "film"),
    ("traffic", "scinet"),
}

IMAGE_NAME = "pytorch-2-9-cu129-ubuntu-2404-nvidia-580-v20261111"
IMAGE_PROJECT = "deeplearning-platform-release"
FALLBACK_ZONES: tuple[str, ...] = (
    # North America (US + Canada)
    "us-west4-a",
    "us-west4-c",
    "us-west1-c",
    "northamerica-northeast1-b",
    "northamerica-northeast1-c",
    "northamerica-northeast2-a",
    "northamerica-northeast2-b",
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-east1-b",
    "us-east1-c",
    "us-east1-d",
    "us-west1-a",
    "us-west1-b",
    "us-east4-a",
    "us-east4-c",
    # Europe
    "europe-west4-a",
    "europe-west4-b",
    "europe-west4-c",
    "europe-west1-b",
    "europe-west1-c",
    "europe-west2-a",
    "europe-west2-b",
    "europe-west3-a",
    "europe-west3-b",
    "europe-west6-b",
    "europe-west6-c",
    # Asia & Middle East
    "asia-southeast1-a",
    "asia-southeast1-b",
    "asia-southeast1-c",
    "asia-east1-a",
    "asia-east1-b",
    "asia-east1-c",
    "asia-northeast1-a",
    "asia-northeast1-b",
    "asia-northeast1-c",
    "asia-northeast3-a",
    "asia-northeast3-b",
    "asia-south1-a",
    "asia-south1-b",
    "asia-south1-c",
    "me-central2-a",
    "me-central2-c",
)

# Rough dataset ordering from fastest to slowest so quick datasets (ili, nyc_taxi, exchange, etth2)
# stream results to GCS first while still balancing both VMs equally via alternating assignment.
DATASET_ORDER: tuple[str, ...] = (
    "ili",
    "nyc_taxi",
    "exchange",
    "etth2",
    "ettm1",
    "ettm2",
    "weather",
    "electricity",
    "traffic",
)


def _gcloud(account: str, *args: str, check: bool = True, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gcloud", f"--account={account}", *args, "--quiet"],
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
    )


def _startup_script(
    gcs_prefix: str,
    shard_index: int,
    shard_count: int,
    gpu_count: int,
    project: str,
) -> str:
    q_prefix = shlex.quote(gcs_prefix)
    q_proj = shlex.quote(project)
    return f"""#!/bin/bash
set -euo pipefail
mkdir -p /opt/hypercast-run
cd /opt/hypercast-run
exec > >(tee /opt/hypercast-run/bootstrap.log) 2>&1

cleanup() {{
  code=$?
  trap - EXIT
  set +e
  echo "$code" > /opt/hypercast-run/exit-code
  gcloud storage cp /opt/hypercast-run/bootstrap.log {q_prefix}/vm-{shard_index}-bootstrap.log --quiet || true
  gcloud storage cp /opt/hypercast-run/exit-code {q_prefix}/vm-{shard_index}-exit-code --quiet || true
  # Self-delete this ephemeral VM once work (or fatal error) concludes so billing stops immediately.
  ZONE=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{{print $NF}}')
  NAME=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
  if [ -n "$ZONE" ] && [ -n "$NAME" ]; then
    gcloud compute instances delete "$NAME" --project={q_proj} --zone="$ZONE" --quiet || true
  fi
}}
trap cleanup EXIT

echo "=== HyperCAST GCP 8xL4 Worker (shard {shard_index}/{shard_count}) ==="
date -u
nvidia-smi --query-gpu=index,name,memory.total --format=csv

PY=$(command -v python3 || echo /opt/conda/bin/python3)
if [ -x /opt/conda/bin/python ]; then
  PY=/opt/conda/bin/python
fi
echo "Using Python: $PY"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count())"

# Install lightweight missing pure-Python deps only (torch/numpy/pandas/scipy are preinstalled in DLVM image).
$PY -c "import einops, yaml, scipy, pandas, sklearn, torch" 2>/dev/null || $PY -m pip install -q --break-system-packages einops pyyaml scipy pandas scikit-learn

gcloud storage cp {q_prefix}/payload.tar.gz payload.tar.gz --quiet
tar -xzf payload.tar.gz

$PY /opt/hypercast-run/scripts/gcp_tslib15_worker.py \\
  --work-root /opt/hypercast-run \\
  --gcs-prefix {q_prefix} \\
  --shard-index {shard_index} \\
  --shard-count {shard_count} \\
  --gpu-count {gpu_count}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", default="onflowai.general@gmail.com")
    parser.add_argument("--project", default="onflow-30k")
    parser.add_argument("--bucket", default="onflow-30k-media")
    parser.add_argument("--datasets", default=",".join(DATASET_ORDER))
    parser.add_argument("--max-rows", type=int, default=14400)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--vm-count", type=int, default=2)
    parser.add_argument("--gpus-per-vm", type=int, default=8)
    parser.add_argument("--machine-type", default="g2-standard-96")
    parser.add_argument("--max-hours", type=int, default=14)
    parser.add_argument("--run-id", default=f"tslib15-multi-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--include-outliers", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-upload", action="store_true", help="Reuse already-uploaded payload.tar.gz/plan.json in GCS")
    parser.add_argument("--shards", default="", help="Comma-separated subset of shard indices to launch (default: all)")
    parser.add_argument("--zones", default="", help="Comma-separated override list of GCE zones to try")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    max_rows = args.max_rows if args.max_rows > 0 else None
    run_dir = ROOT / "results" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    gcs_prefix = f"gs://{args.bucket}/hypercast4d/{args.run_id}"

    print(f"Preparing run {args.run_id} (max_rows={max_rows}, epochs={args.epochs}, {args.vm_count}x {args.machine_type})...", flush=True)
    jobs: list[dict[str, Any]] = []
    manifests: dict[str, dict[str, Any]] = {}
    skipped_outliers: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="hypercast-gcp-pkg-") as tmp:
        stage = Path(tmp)
        bundles_dir = stage / "bundles"
        bundles_dir.mkdir(parents=True)

        for ds_id in datasets:
            blob, manifest = prepare_dataset_bundle(
                ds_id, ROOT, include_test=True, max_rows=max_rows
            )
            manifests[ds_id] = manifest
            (bundles_dir / f"{ds_id}.npz").write_bytes(blob)
            (bundles_dir / f"{ds_id}.manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            print(
                f"  bundle {ds_id:12s}: rows={manifest['rows']:5d}/{manifest['file_rows']:5d} "
                f"ch={manifest['features']:3d} test_origins={manifest['partitions']['test']['origins']:4d} "
                f"size={len(blob)/1e6:.2f}MB",
                flush=True,
            )

        # Build job list ordered by dataset (fastest first) and alternating shards so both VMs
        # receive an identical workload distribution across every dataset and backbone!
        idx = 0
        for ds_id in datasets:
            for backbone in BACKBONES:
                if (not args.include_outliers) and (ds_id, backbone) in SKIP_OUTLIERS:
                    skipped_outliers.append({"dataset": ds_id, "backbone": backbone, "reason": ">200s/epoch on A10G/L4"})
                    continue
                for arm in ARMS:
                    for seed in DEFAULT_EVALUATION_SEEDS:
                        jobs.append(
                            {
                                "index": idx,
                                "shard": idx % args.vm_count,
                                "dataset": ds_id,
                                "phase": "test",
                                "backbone": backbone,
                                "arm": arm,
                                "seed": seed,
                                "epochs": args.epochs,
                                "learning_rate": 0.001,
                                "timeout_seconds": 3600,
                            }
                        )
                        idx += 1

        plan = {
            "run_id": args.run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "gcs_prefix": gcs_prefix,
            "project": args.project,
            "account": args.account,
            "machine_type": args.machine_type,
            "vm_count": args.vm_count,
            "gpus_per_vm": args.gpus_per_vm,
            "max_rows": max_rows,
            "epochs": args.epochs,
            "datasets": datasets,
            "total_jobs": len(jobs),
            "skipped_outliers": skipped_outliers,
            "manifests": manifests,
            "jobs": jobs,
        }
        plan_path = run_dir / "plan.json"
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

        payload_path = stage / "payload.tar.gz"
        with tarfile.open(payload_path, "w:gz") as tar:
            tar.add(plan_path, arcname="plan.json")
            for p in (ROOT / "src" / "hypercast4d").rglob("*"):
                if (
                    p.is_file()
                    and p.suffix in {".py", ".json", ".pt"}
                    and "web_dist" not in p.parts
                    and "__pycache__" not in p.parts
                ):
                    tar.add(p, arcname=str(p.relative_to(ROOT)))
            tar.add(ROOT / "scripts" / "gcp_tslib15_worker.py", arcname="scripts/gcp_tslib15_worker.py")
            for p in bundles_dir.iterdir():
                tar.add(p, arcname=f"bundles/{p.name}")

        print(
            f"Prepared {len(jobs)} total trials across {len(datasets)} datasets "
            f"(skipped {len(skipped_outliers)} heavy outliers); payload={payload_path.stat().st_size/1e6:.2f} MB",
            flush=True,
        )
        if args.dry_run:
            print("Dry run complete; no GCS upload or VMs created.")
            return

        if not args.skip_upload:
            print(f"Uploading payload.tar.gz & plan.json to {gcs_prefix} ...", flush=True)
            _gcloud(args.account, "storage", "cp", str(payload_path), f"{gcs_prefix}/payload.tar.gz", timeout=600)
            _gcloud(args.account, "storage", "cp", str(plan_path), f"{gcs_prefix}/plan.json", timeout=120)
        else:
            print(f"Skipping GCS upload; reusing {gcs_prefix}/payload.tar.gz", flush=True)

        existing_launch_path = run_dir / "launch.json"
        launched: list[dict[str, str]] = []
        if existing_launch_path.is_file():
            try:
                launched = list(json.loads(existing_launch_path.read_text(encoding="utf-8")).get("vms", []))
            except Exception:
                launched = []

        target_shards = (
            [int(s.strip()) for s in args.shards.split(",") if s.strip()]
            if args.shards.strip()
            else list(range(args.vm_count))
        )
        candidate_zones = (
            tuple(z.strip() for z in args.zones.split(",") if z.strip())
            if args.zones.strip()
            else FALLBACK_ZONES
        )
        exhausted_zones: set[str] = set()
        slug = "".join(c for c in args.run_id.lower() if c.isalnum())[-12:]
        label_run = "".join(c if c.isalnum() or c in "-_" else "-" for c in args.run_id.lower())[:60]
        try:
            for shard in target_shards:
                vm_name = f"hypercast-{slug}-vm{shard}"
                script_path = stage / f"startup-{shard}.sh"
                script_path.write_text(
                    _startup_script(gcs_prefix, shard, args.vm_count, args.gpus_per_vm, args.project),
                    encoding="utf-8",
                )
                created_zone = None
                last_err = ""
                for zone in candidate_zones:
                    if zone in exhausted_zones:
                        continue
                    print(f"Creating {vm_name} ({args.machine_type}) in {zone} ...", flush=True)
                    res = _gcloud(
                        args.account,
                        "compute",
                        "instances",
                        "create",
                        vm_name,
                        f"--project={args.project}",
                        f"--zone={zone}",
                        f"--machine-type={args.machine_type}",
                        f"--image={IMAGE_NAME}",
                        f"--image-project={IMAGE_PROJECT}",
                        "--scopes=cloud-platform",
                        "--boot-disk-size=100GB",
                        "--boot-disk-type=pd-balanced",
                        "--boot-disk-auto-delete",
                        "--maintenance-policy=TERMINATE",
                        "--no-restart-on-failure",
                        f"--max-run-duration={args.max_hours}h",
                        "--instance-termination-action=DELETE",
                        f"--metadata-from-file=startup-script={script_path}",
                        f"--labels=app=hypercast4d,run={label_run}",
                        check=False,
                        timeout=300,
                    )
                    if res.returncode == 0:
                        created_zone = zone
                        print(f"  LAUNCHED {vm_name} in {zone}!", flush=True)
                        break
                    last_err = (res.stderr or res.stdout).strip()
                    if (
                        "ZONE_RESOURCE_POOL_EXHAUSTED" in last_err
                        or "does not have enough resources" in last_err
                        or "QUOTA_EXCEEDED" in last_err
                        or "was not found" in last_err
                    ):
                        exhausted_zones.add(zone)
                        print(f"  {zone} unavailable ({last_err.splitlines()[-1][:120]}), trying next zone...", flush=True)
                        continue
                    raise RuntimeError(f"Failed creating {vm_name} in {zone}: {last_err}")
                if not created_zone:
                    raise RuntimeError(f"Could not place {vm_name} in any zone: {last_err}")
                launched = [v for v in launched if v.get("shard") != str(shard)]
                launched.append({"name": vm_name, "zone": created_zone, "shard": str(shard)})
        finally:
            if launched:
                launch_record = {
                    "run_id": args.run_id,
                    "gcs_prefix": gcs_prefix,
                    "project": args.project,
                    "account": args.account,
                    "vms": sorted(launched, key=lambda x: int(x["shard"])),
                    "total_jobs": len(jobs),
                    "launched_utc": datetime.now(timezone.utc).isoformat(),
                }
                existing_launch_path.write_text(json.dumps(launch_record, indent=2), encoding="utf-8")

        print("\nLaunch summary saved to:", run_dir / "launch.json", flush=True)
        print(json.dumps(launch_record, indent=2), flush=True)


if __name__ == "__main__":
    main()
