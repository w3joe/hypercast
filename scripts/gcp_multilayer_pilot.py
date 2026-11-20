#!/usr/bin/env python3
"""Multi-GPU GCP launcher, worker, and collector for the Phase 2 Multi-Layer HyperDense pilot."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

IMAGE_NAME = "pytorch-2-9-cu129-ubuntu-2404-nvidia-580-v20261111"
IMAGE_PROJECT = "deeplearning-platform-release"
FALLBACK_ZONES: tuple[str, ...] = (
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-east1-b",
    "us-east1-c",
    "us-east1-d",
    "us-west1-a",
    "us-west1-b",
    "us-west1-c",
    "us-east4-a",
    "us-east4-c",
    "us-west4-a",
    "us-west4-c",
    "northamerica-northeast1-b",
    "northamerica-northeast1-c",
    "europe-west4-a",
    "europe-west4-b",
    "europe-west4-c",
    "europe-west1-b",
    "europe-west1-c",
    "europe-west6-c",
    "asia-southeast1-c",
    "asia-southeast1-b",
    "asia-southeast1-a",
    "asia-east1-a",
    "asia-east1-b",
    "asia-east1-c",
    "asia-northeast1-a",
    "asia-northeast1-b",
)

PILOT_DATASETS = ("ili", "etth2", "weather", "electricity")
PILOT_MODELS = ("hyper_channel_mixer", "tsmixer_cc", "itransformer_cc", "crossformer_cc")
PILOT_ARMS = (
    "real",
    "complex",
    "split_complex",
    "quaternion",
    "coquaternion",
    "cl11",
    "octonion",
    "lowrank2",
    "lowrank4",
    "lowrank8",
    "blockdiag2",
    "blockdiag4",
    "blockdiag8",
)
PILOT_BUDGET_MODES = ("compress", "isoparam")
PILOT_SEEDS = (4101, 4102, 4103)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gcloud(
    account: str | None, *args: str, check: bool = True, timeout: int = 300
) -> subprocess.CompletedProcess[str]:
    cmd = ["gcloud"]
    if account:
        cmd.append(f"--account={account}")
    cmd.extend([*args, "--quiet"])
    return subprocess.run(cmd, capture_output=True, text=True, check=check, timeout=timeout)


def _startup_script(
    gcs_prefix: str, shard_index: int, shard_count: int, gpu_count: int, project: str
) -> str:
    q_prefix = shlex.quote(gcs_prefix)
    q_proj = shlex.quote(project)
    return f"""#!/bin/bash
set -euo pipefail
mkdir -p /opt/hypercast-pilot
cd /opt/hypercast-pilot
exec > >(tee /opt/hypercast-pilot/bootstrap.log) 2>&1

cleanup() {{
  code=$?
  trap - EXIT
  set +e
  echo "$code" > /opt/hypercast-pilot/exit-code
  gcloud storage cp /opt/hypercast-pilot/bootstrap.log {q_prefix}/vm-{shard_index}-bootstrap.log --quiet || true
  gcloud storage cp /opt/hypercast-pilot/exit-code {q_prefix}/vm-{shard_index}-exit-code --quiet || true
  ZONE=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{{print $NF}}')
  NAME=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
  if [ -n "$ZONE" ] && [ -n "$NAME" ]; then
    gcloud compute instances delete "$NAME" --project={q_proj} --zone="$ZONE" --quiet || true
  fi
}}
trap cleanup EXIT

echo "=== HyperCAST Multi-Layer Cross-Channel Pilot Worker (shard {shard_index}/{shard_count}) ==="
date -u
nvidia-smi --query-gpu=index,name,memory.total --format=csv

PY=$(command -v python3 || echo /opt/conda/bin/python3)
if [ -x /opt/conda/bin/python ]; then
  PY=/opt/conda/bin/python
fi
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count())"
$PY -c "import einops, yaml, scipy, pandas, sklearn, torch" 2>/dev/null || $PY -m pip install -q --break-system-packages einops pyyaml scipy pandas scikit-learn

gcloud storage cp {q_prefix}/payload.tar.gz payload.tar.gz --quiet
tar -xzf payload.tar.gz

$PY /opt/hypercast-pilot/scripts/gcp_multilayer_pilot.py worker \\
  --work-root /opt/hypercast-pilot \\
  --gcs-prefix {q_prefix} \\
  --shard-index {shard_index} \\
  --shard-count {shard_count} \\
  --gpu-count {gpu_count}
"""


def cmd_launch(args: argparse.Namespace) -> None:
    """Package development-only dataset bundles and launch GCE GPU VM(s)."""
    from hypercast4d.tslib15_multi import prepare_dataset_bundle

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    budget_modes = [b.strip() for b in args.budget_modes.split(",") if b.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    max_rows = args.max_rows if args.max_rows > 0 else None

    run_dir = ROOT / "results" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    gcs_prefix = f"gs://{args.bucket}/hypercast4d/{args.run_id}"

    print(
        f"Preparing Phase 2 development pilot {args.run_id} "
        f"(include_test=False, {args.vm_count}x {args.machine_type})...",
        flush=True,
    )
    jobs: list[dict[str, Any]] = []
    manifests: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="hypercast-mlpilot-") as tmp:
        stage = Path(tmp)
        bundles_dir = stage / "bundles"
        bundles_dir.mkdir(parents=True)

        for ds_id in datasets:
            # Strictly exclude held-out test rows from Phase 2 pilot bundles!
            blob, manifest = prepare_dataset_bundle(
                ds_id, ROOT, include_test=False, max_rows=max_rows
            )
            manifests[ds_id] = manifest
            (bundles_dir / f"{ds_id}.npz").write_bytes(blob)
            (bundles_dir / f"{ds_id}.manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            print(
                f"  bundle {ds_id:12s}: rows={manifest['rows']:5d} ch={manifest['features']:3d} "
                f"dev_origins={manifest['partitions']['development']['origins']:4d} "
                f"include_test={manifest['include_test']} ({len(blob)/1e6:.2f}MB)",
                flush=True,
            )

        idx = 0
        for ds_id in datasets:
            for mdl in models:
                for bmode in budget_modes:
                    for arm in arms:
                        # Skip duplicate 'real' in isoparam mode? Keep explicit for self-contained cells.
                        for seed in seeds:
                            jobs.append(
                                {
                                    "index": idx,
                                    "shard": idx % args.vm_count,
                                    "dataset": ds_id,
                                    "model_name": mdl,
                                    "budget_mode": bmode,
                                    "arm": arm,
                                    "seed": seed,
                                    "base_width": args.base_width,
                                    "num_layers": args.num_layers,
                                    "epochs": args.epochs,
                                    "patience": args.patience,
                                    "learning_rate": 1e-3,
                                    "eval_split": "development",
                                    "timeout_seconds": 1800,
                                }
                            )
                            idx += 1

        plan = {
            "run_id": args.run_id,
            "created_utc": _utc_now(),
            "gcs_prefix": gcs_prefix,
            "project": args.project,
            "account": args.account,
            "machine_type": args.machine_type,
            "vm_count": args.vm_count,
            "gpus_per_vm": args.gpus_per_vm,
            "include_test": False,
            "eval_split": "development",
            "datasets": datasets,
            "models": models,
            "arms": arms,
            "budget_modes": budget_modes,
            "seeds": seeds,
            "total_jobs": len(jobs),
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
            tar.add(
                ROOT / "scripts" / "run_multilayer_hyperdense_pilot.py",
                arcname="scripts/run_multilayer_hyperdense_pilot.py",
            )
            tar.add(
                ROOT / "scripts" / "gcp_multilayer_pilot.py",
                arcname="scripts/gcp_multilayer_pilot.py",
            )
            for p in bundles_dir.iterdir():
                tar.add(p, arcname=f"bundles/{p.name}")

        print(
            f"Prepared {len(jobs)} Phase 2 pilot trials across {len(datasets)} datasets; "
            f"payload={payload_path.stat().st_size/1e6:.2f} MB",
            flush=True,
        )
        if args.dry_run:
            print("Dry run complete; no GCS upload or VMs created.")
            return

        if not args.skip_upload:
            print(f"Uploading payload.tar.gz & plan.json to {gcs_prefix} ...", flush=True)
            _gcloud(args.account, "storage", "cp", str(payload_path), f"{gcs_prefix}/payload.tar.gz", timeout=600)
            _gcloud(args.account, "storage", "cp", str(plan_path), f"{gcs_prefix}/plan.json", timeout=120)

        candidate_zones = (
            tuple(z.strip() for z in args.zones.split(",") if z.strip())
            if args.zones.strip()
            else FALLBACK_ZONES
        )
        launched: list[dict[str, str]] = []
        exhausted: set[str] = set()
        slug = "".join(c for c in args.run_id.lower() if c.isalnum())[-12:]
        label_run = "".join(c if c.isalnum() or c in "-_" else "-" for c in args.run_id.lower())[:60]

        try:
            for shard in range(args.vm_count):
                vm_name = f"hypercast-{slug}-vm{shard}"
                script_path = stage / f"startup-{shard}.sh"
                script_path.write_text(
                    _startup_script(gcs_prefix, shard, args.vm_count, args.gpus_per_vm, args.project),
                    encoding="utf-8",
                )
                created_zone = None
                last_err = ""
                for zone in candidate_zones:
                    if zone in exhausted:
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
                        exhausted.add(zone)
                        print(f"  {zone} unavailable, trying next zone...", flush=True)
                        continue
                    raise RuntimeError(f"Failed creating {vm_name} in {zone}: {last_err}")
                if not created_zone:
                    raise RuntimeError(f"Could not place {vm_name} in any zone: {last_err}")
                launched.append({"name": vm_name, "zone": created_zone, "shard": str(shard)})
        finally:
            if launched:
                rec = {
                    "run_id": args.run_id,
                    "gcs_prefix": gcs_prefix,
                    "project": args.project,
                    "account": args.account,
                    "vms": launched,
                    "total_jobs": len(jobs),
                    "launched_utc": _utc_now(),
                }
                (run_dir / "launch.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
                print(json.dumps(rec, indent=2), flush=True)


def _trial_key(job: dict[str, Any]) -> str:
    return f"{job['dataset']}/{job['model_name']}__{job['budget_mode']}__{job['arm']}__seed{job['seed']}"


def cmd_single_trial(args: argparse.Namespace) -> None:
    """Execute one isolated trial subprocess pinned to a single GPU."""
    import torch

    sys.path.insert(0, str(args.work_root / "scripts"))
    from run_multilayer_hyperdense_pilot import run_single_fit

    torch.set_num_threads(4)
    job = json.loads(args.job_file.read_text(encoding="utf-8"))
    ds_id = job["dataset"]
    manifest = json.loads(
        (args.work_root / "bundles" / f"{ds_id}.manifest.json").read_text(encoding="utf-8")
    )
    bundle_bytes = (args.work_root / "bundles" / f"{ds_id}.npz").read_bytes()
    res = run_single_fit(
        manifest,
        bundle_bytes,
        model_name=job["model_name"],
        arm=job["arm"],
        seed=int(job["seed"]),
        budget_mode=job["budget_mode"],
        base_width=int(job["base_width"]),
        num_layers=int(job["num_layers"]),
        epochs=int(job["epochs"]),
        patience=int(job["patience"]),
        learning_rate=float(job["learning_rate"]),
        eval_split=job["eval_split"],
        device="cuda",
    )
    (args.out_dir / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


def cmd_worker(args: argparse.Namespace) -> None:
    """Dynamic multi-GPU worker streaming per-trial JSON results to GCS."""
    import torch

    work_root = args.work_root.resolve()
    gcs_prefix = args.gcs_prefix.rstrip("/")
    detected = torch.cuda.device_count()
    if detected < args.gpu_count:
        raise RuntimeError(f"Requested {args.gpu_count} GPUs, CUDA detected {detected}")

    plan = json.loads((work_root / "plan.json").read_text(encoding="utf-8"))
    shard_jobs = [j for j in plan["jobs"] if j["shard"] == args.shard_index]

    ls_proc = _gcloud(None, "storage", "ls", f"{gcs_prefix}/trials/**", check=False, timeout=120)
    existing = {line.strip() for line in ls_proc.stdout.splitlines() if line.strip()} if ls_proc.returncode == 0 else set()

    pending: queue.Queue[dict[str, Any]] = queue.Queue()
    skipped = 0
    for job in shard_jobs:
        uri = f"{gcs_prefix}/trials/{_trial_key(job)}.json"
        if uri in existing:
            skipped += 1
        else:
            pending.put(job)

    lock = threading.Lock()
    stats: dict[str, Any] = {
        "shard": args.shard_index,
        "gpu_count": args.gpu_count,
        "gpu_name": torch.cuda.get_device_name(0),
        "assigned": len(shard_jobs),
        "skipped_existing": skipped,
        "completed": 0,
        "failed": 0,
        "failures": [],
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "state": "running",
    }
    prog_file = work_root / f"vm-{args.shard_index}-progress.json"

    def upload_progress() -> None:
        prog_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        _gcloud(None, "storage", "cp", str(prog_file), f"{gcs_prefix}/vm-{args.shard_index}-progress.json", check=False, timeout=60)

    upload_progress()
    last_up = [time.monotonic()]

    def gpu_loop(gpu_id: int) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            tkey = _trial_key(job)
            t0 = time.monotonic()
            with tempfile.TemporaryDirectory(prefix=f"pilot-g{gpu_id}-") as tmp:
                out_dir = Path(tmp)
                job_file = out_dir / "job.json"
                job_file.write_text(json.dumps(job), encoding="utf-8")
                env = {
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": str(gpu_id),
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "PYTHONPATH": str(work_root / "src"),
                }
                try:
                    proc = subprocess.run(
                        [
                            sys.executable,
                            str(work_root / "scripts" / "gcp_multilayer_pilot.py"),
                            "single-trial",
                            "--work-root",
                            str(work_root),
                            "--job-file",
                            str(job_file),
                            "--out-dir",
                            str(out_dir),
                        ],
                        env=env,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=int(job.get("timeout_seconds", 1800)),
                    )
                    ok = proc.returncode == 0 and (out_dir / "result.json").is_file()
                    err = "" if ok else ((proc.stderr or "") + "\n" + (proc.stdout or ""))[-1200:]
                except Exception as exc:
                    ok, err = False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}"

                elapsed = round(time.monotonic() - t0, 1)
                if ok:
                    res_path = out_dir / "result.json"
                    _gcloud(None, "storage", "cp", str(res_path), f"{gcs_prefix}/trials/{tkey}.json", timeout=120)
                    res = json.loads(res_path.read_text(encoding="utf-8"))
                    with lock:
                        stats["completed"] += 1
                        stats["updated_at"] = _utc_now()
                        done = stats["completed"] + stats["failed"] + skipped
                        print(f"[g{gpu_id}] OK   ({done}/{len(shard_jobs)}) {tkey:62s} mae={res['mae']:.4f} ({elapsed:.1f}s)", flush=True)
                else:
                    err_path = out_dir / "error.json"
                    err_path.write_text(json.dumps({"trial": tkey, "error": err}), encoding="utf-8")
                    _gcloud(None, "storage", "cp", str(err_path), f"{gcs_prefix}/failures/{tkey.replace('/', '__')}.json", check=False, timeout=120)
                    with lock:
                        stats["failed"] += 1
                        stats["failures"].append({"trial": tkey, "error": err[:300]})
                        stats["updated_at"] = _utc_now()
                        done = stats["completed"] + stats["failed"] + skipped
                        print(f"[g{gpu_id}] FAIL ({done}/{len(shard_jobs)}) {tkey} ({elapsed:.1f}s)", flush=True)

                with lock:
                    if time.monotonic() - last_up[0] >= 25.0 or pending.empty():
                        last_up[0] = time.monotonic()
                        upload_progress()

    with ThreadPoolExecutor(max_workers=args.gpu_count) as pool:
        for fut in [pool.submit(gpu_loop, g) for g in range(args.gpu_count)]:
            fut.result()

    stats["state"] = "complete"
    stats["finished_at"] = _utc_now()
    stats["updated_at"] = stats["finished_at"]
    upload_progress()


def cmd_collect(args: argparse.Namespace) -> None:
    """Check VM/GCS status and optionally download + evaluate Phase 2 advancement gates."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_multilayer_hyperdense_pilot import summarize_pilot

    run_dir = args.run_dir.resolve()
    launch = json.loads((run_dir / "launch.json").read_text(encoding="utf-8"))
    account = launch["account"]
    project = launch["project"]
    prefix = launch["gcs_prefix"]

    print(f"=== Multi-Layer Pilot {launch['run_id']} ({prefix}) ===")
    for vm in launch["vms"]:
        st = _gcloud(account, "compute", "instances", "describe", vm["name"], f"--project={project}", f"--zone={vm['zone']}", "--format=value(status)", check=False)
        vm_status = st.stdout.strip() if st.returncode == 0 else "DELETED/GONE"
        prog = _gcloud(account, "storage", "cat", f"{prefix}/vm-{vm['shard']}-progress.json", check=False)
        if prog.returncode == 0:
            p = json.loads(prog.stdout)
            print(
                f"  {vm['name']} ({vm['zone']}, {vm_status}): state={p.get('state')} "
                f"ok={p.get('completed', 0)}/{p.get('assigned', 0)} fail={p.get('failed', 0)} "
                f"updated={p.get('updated_at')}"
            )
        else:
            print(f"  {vm['name']} ({vm['zone']}, {vm_status}): bootstrapping")

    if not args.download:
        return

    trials_dir = run_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    _gcloud(account, "storage", "rsync", "--recursive", f"{prefix}/trials", str(trials_dir), timeout=600)
    records = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(trials_dir.rglob("*.json"))
    ]
    if records:
        summarize_pilot(records, run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_launch = sub.add_parser("launch")
    p_launch.add_argument("--account", default="onflowai.general@gmail.com")
    p_launch.add_argument("--project", default="onflow-30k")
    p_launch.add_argument("--bucket", default="onflow-30k-media")
    p_launch.add_argument("--datasets", default=",".join(PILOT_DATASETS))
    p_launch.add_argument("--models", default=",".join(PILOT_MODELS))
    p_launch.add_argument("--arms", default=",".join(PILOT_ARMS))
    p_launch.add_argument("--budget-modes", default=",".join(PILOT_BUDGET_MODES))
    p_launch.add_argument("--seeds", default=",".join(str(s) for s in PILOT_SEEDS))
    p_launch.add_argument("--base-width", type=int, default=64)
    p_launch.add_argument("--num-layers", type=int, default=3)
    p_launch.add_argument("--epochs", type=int, default=60)
    p_launch.add_argument("--patience", type=int, default=15)
    p_launch.add_argument("--max-rows", type=int, default=14400)
    p_launch.add_argument("--vm-count", type=int, default=1)
    p_launch.add_argument("--gpus-per-vm", type=int, default=8)
    p_launch.add_argument("--machine-type", default="g2-standard-96")
    p_launch.add_argument("--max-hours", type=int, default=5)
    p_launch.add_argument("--zones", default="")
    p_launch.add_argument("--skip-upload", action="store_true")
    p_launch.add_argument("--dry-run", action="store_true")
    p_launch.add_argument(
        "--run-id",
        default=f"multilayer-pilot-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
    )

    p_worker = sub.add_parser("worker")
    p_worker.add_argument("--work-root", type=Path, required=True)
    p_worker.add_argument("--gcs-prefix", required=True)
    p_worker.add_argument("--shard-index", type=int, default=0)
    p_worker.add_argument("--shard-count", type=int, default=1)
    p_worker.add_argument("--gpu-count", type=int, default=8)

    p_single = sub.add_parser("single-trial")
    p_single.add_argument("--work-root", type=Path, required=True)
    p_single.add_argument("--job-file", type=Path, required=True)
    p_single.add_argument("--out-dir", type=Path, required=True)

    p_collect = sub.add_parser("collect")
    p_collect.add_argument("--run-dir", type=Path, required=True)
    p_collect.add_argument("--download", action="store_true")

    args = parser.parse_args()
    if args.cmd == "launch":
        cmd_launch(args)
    elif args.cmd == "worker":
        cmd_worker(args)
    elif args.cmd == "single-trial":
        cmd_single_trial(args)
    elif args.cmd == "collect":
        cmd_collect(args)


if __name__ == "__main__":
    main()
