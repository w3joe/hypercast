#!/usr/bin/env python3
"""VM-side 8-GPU dynamic trial runner with per-trial GCS checkpointing."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gsutil_ls(prefix: str) -> set[str]:
    proc = subprocess.run(
        ["gcloud", "storage", "ls", f"{prefix.rstrip('/')}/**", "--quiet"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if proc.returncode != 0:
        return set()
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def _upload_file(local_path: Path, remote_uri: str) -> None:
    subprocess.run(
        ["gcloud", "storage", "cp", str(local_path), remote_uri, "--quiet"],
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )


def _upload_json(payload: dict[str, Any], remote_uri: str, scratch: Path) -> None:
    scratch.parent.mkdir(parents=True, exist_ok=True)
    tmp = scratch.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(scratch)
    _upload_file(scratch, remote_uri)


def _run_single_trial_subprocess(
    work_root: Path,
    job: dict[str, Any],
    gpu_id: int,
    out_dir: Path,
) -> tuple[bool, str]:
    """Run one trial in an isolated Python subprocess pinned to ``CUDA_VISIBLE_DEVICES=gpu_id``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    job_file = out_dir / "job.json"
    job_file.write_text(json.dumps(job), encoding="utf-8")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTHONPATH": str(work_root / "src"),
    }
    cmd = [
        sys.executable,
        str(work_root / "scripts" / "gcp_tslib15_worker.py"),
        "--single-trial",
        "--work-root",
        str(work_root),
        "--job-file",
        str(job_file),
        "--out-dir",
        str(out_dir),
    ]
    timeout = int(job.get("timeout_seconds", 3600)) + 120
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode == 0 and (out_dir / "result.json").is_file():
        return True, ""
    err_tail = ((proc.stderr or "") + "\n" + (proc.stdout or ""))[-1500:]
    return False, f"returncode={proc.returncode}\n{err_tail}"


def _execute_single_trial(work_root: Path, job_file: Path, out_dir: Path) -> None:
    import torch
    from hypercast4d.tslib15_multi import run_trial

    torch.set_num_threads(4)
    job = json.loads(job_file.read_text(encoding="utf-8"))
    ds_id = job["dataset"]
    manifest = json.loads(
        (work_root / "bundles" / f"{ds_id}.manifest.json").read_text(encoding="utf-8")
    )
    bundle_bytes = (work_root / "bundles" / f"{ds_id}.npz").read_bytes()
    run_trial(manifest, bundle_bytes, job, out_dir, device="cuda")


def run_shard(
    work_root: Path,
    gcs_prefix: str,
    shard_index: int,
    shard_count: int,
    gpu_count: int,
) -> None:
    import torch

    detected = torch.cuda.device_count()
    if detected < gpu_count:
        raise RuntimeError(f"Requested {gpu_count} GPUs, CUDA detected {detected}")

    plan = json.loads((work_root / "plan.json").read_text(encoding="utf-8"))
    all_jobs: list[dict[str, Any]] = plan["jobs"]
    shard_jobs = [job for job in all_jobs if job["shard"] == shard_index]
    print(
        f"[VM shard {shard_index}/{shard_count}] {len(shard_jobs)} assigned trials across {gpu_count} GPUs",
        flush=True,
    )

    existing = _gsutil_ls(f"{gcs_prefix}/trials")
    pending: queue.Queue[dict[str, Any]] = queue.Queue()
    skipped = 0
    for job in shard_jobs:
        trial_key = f"{job['dataset']}/{job['backbone']}__{job['arm']}__seed{job['seed']}"
        expected_uri = f"{gcs_prefix}/trials/{trial_key}/result.json"
        if expected_uri in existing:
            skipped += 1
            continue
        pending.put(job)

    print(
        f"[VM shard {shard_index}] Skipping {skipped} already-uploaded trials; {pending.qsize()} remaining",
        flush=True,
    )

    lock = threading.Lock()
    stats: dict[str, Any] = {
        "shard": shard_index,
        "shard_count": shard_count,
        "gpu_count": gpu_count,
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
    progress_local = work_root / f"vm-{shard_index}-progress.json"
    _upload_json(stats, f"{gcs_prefix}/vm-{shard_index}-progress.json", progress_local)
    last_upload = [time.monotonic()]

    def gpu_loop(gpu_id: int) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            trial_key = f"{job['dataset']}/{job['backbone']}__{job['arm']}__seed{job['seed']}"
            t0 = time.monotonic()
            with tempfile.TemporaryDirectory(prefix=f"trial-g{gpu_id}-") as tmp:
                out_dir = Path(tmp)
                try:
                    ok, err = _run_single_trial_subprocess(work_root, job, gpu_id, out_dir)
                except Exception as exc:
                    ok, err = False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}"

                elapsed = round(time.monotonic() - t0, 1)
                if ok:
                    res_path = out_dir / "result.json"
                    pred_path = out_dir / "predictions.npz"
                    _upload_file(res_path, f"{gcs_prefix}/trials/{trial_key}/result.json")
                    if pred_path.is_file():
                        _upload_file(pred_path, f"{gcs_prefix}/trials/{trial_key}/predictions.npz")
                    res = json.loads(res_path.read_text(encoding="utf-8"))
                    with lock:
                        stats["completed"] += 1
                        stats["updated_at"] = _utc_now()
                        done_total = stats["completed"] + stats["failed"] + skipped
                        print(
                            f"[g{gpu_id}] OK   ({done_total}/{len(shard_jobs)}) {trial_key:46s} "
                            f"ep={res['epochs_ran']:3d} mae={res['mae']:.4f} ({elapsed:.1f}s)",
                            flush=True,
                        )
                else:
                    fail_record = {
                        "trial": trial_key,
                        "job": job,
                        "gpu_id": gpu_id,
                        "elapsed_s": elapsed,
                        "error": err,
                    }
                    err_path = out_dir / "error.json"
                    err_path.write_text(json.dumps(fail_record, indent=2), encoding="utf-8")
                    _upload_file(err_path, f"{gcs_prefix}/failures/{trial_key.replace('/', '__')}.json")
                    with lock:
                        stats["failed"] += 1
                        stats["failures"].append({"trial": trial_key, "error": err[:300]})
                        stats["updated_at"] = _utc_now()
                        done_total = stats["completed"] + stats["failed"] + skipped
                        print(
                            f"[g{gpu_id}] FAIL ({done_total}/{len(shard_jobs)}) {trial_key:46s} ({elapsed:.1f}s)",
                            flush=True,
                        )

                with lock:
                    now = time.monotonic()
                    if now - last_upload[0] >= 30.0 or pending.empty():
                        last_upload[0] = now
                        try:
                            _upload_json(
                                stats,
                                f"{gcs_prefix}/vm-{shard_index}-progress.json",
                                progress_local,
                            )
                        except Exception:
                            pass

    with ThreadPoolExecutor(max_workers=gpu_count) as pool:
        futures = [pool.submit(gpu_loop, g) for g in range(gpu_count)]
        for fut in futures:
            fut.result()

    stats["state"] = "complete"
    stats["finished_at"] = _utc_now()
    stats["updated_at"] = stats["finished_at"]
    _upload_json(stats, f"{gcs_prefix}/vm-{shard_index}-progress.json", progress_local)
    _upload_json(stats, f"{gcs_prefix}/vm-{shard_index}-DONE.json", work_root / f"vm-{shard_index}-DONE.json")
    print(f"[VM shard {shard_index}] ALL DONE: {stats['completed']} ok, {stats['failed']} failed, {skipped} skipped", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-trial", action="store_true")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--job-file", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--gcs-prefix")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--gpu-count", type=int, default=8)
    args = parser.parse_args()

    if args.single_trial:
        assert args.job_file and args.out_dir
        _execute_single_trial(args.work_root, args.job_file, args.out_dir)
        return

    assert args.gcs_prefix
    run_shard(
        args.work_root.resolve(),
        args.gcs_prefix.rstrip("/"),
        args.shard_index,
        args.shard_count,
        args.gpu_count,
    )


if __name__ == "__main__":
    main()
