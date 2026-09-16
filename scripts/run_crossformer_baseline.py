"""Prepare and run the staged Crossformer baseline search on at most ten L4s."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time

import numpy as np

from run_crossformer_baseline_worker import worker


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_UPPER_USD = 21.508189322325403
TOTAL_CAP_USD = 80.0
MAX_L4 = 10


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path); temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)); temp.replace(path)


def prepare(run):
    import pandas as pd
    prepared = run/"prepared-v1"
    if prepared.exists():
        return prepared, json.loads((prepared/"plan.json").read_text())
    prepared.mkdir(parents=True)
    source = ROOT/"data/external/etth1-1d16c8f/ETTh1.csv"
    columns = ["OT", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
    frame = pd.read_csv(source, nrows=14400, usecols=columns)
    raw = frame[columns].to_numpy(dtype=np.float64)
    minimum = raw[:8640].min(0); span = raw[:8640].max(0)-minimum; span[span == 0] = 1
    values = ((raw-minimum)/span).astype("float32")
    np.savez_compressed(prepared/"development.npz", values=values)
    frozen = prepared/"frozen"
    shutil.copytree(ROOT/"src", frozen/"src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (frozen/"scripts").mkdir()
    shutil.copy2(ROOT/"scripts/run_crossformer_baseline_worker.py", frozen/"scripts/run_crossformer_baseline_worker.py")
    plan = dict(protocol="crossformer-baseline-search-v1", created_epoch=time.time(),
                previous_conservative_upper_usd=PREVIOUS_UPPER_USD, total_cap_usd=TOTAL_CAP_USD,
                max_l4=MAX_L4, dataset="ETTh1", source_sha256=sha(source), source_rows=17420,
                supplied_rows=[0,14400], untouched_tail_rows=[14400,17420],
                roles=dict(train=[0,8640], inner=[8640,11520], development=[11520,14400]),
                scaler_rows=[0,8640], scaler_minimum=minimum.tolist(), scaler_span=span.tolist(),
                context_candidates=[32,96,192], modes=["relative_residual","levels_direct"],
                learning_rates=[.0001,.0003,.001], schedules=["constant","exponential"],
                d_models=[32,64], dropouts=[0.,.1], screening_seed=4101,
                refinement_seed=4201, confirmation_seeds=[4301,4302,4303],
                selection="Minimum mean development MAE across three confirmation seeds; deterministic lexical tie break",
                note="No rows from the untouched tail are loaded, bundled, scored, or used for selection")
    plan["development_sha256"] = sha(prepared/"development.npz")
    atomic_json(prepared/"plan.json", plan)
    return prepared, plan


def signature(job):
    return tuple(job[k] for k in ("window","mode","learning_rate","schedule","d_model","dropout"))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(); run = args.run_dir.resolve(); run.mkdir(parents=True, exist_ok=False)
    prepared, plan = prepare(run)
    import fcntl, modal
    lock = (run/"controller.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    global_lock = (ROOT/"results/.representation-pilot.lock").open("a")
    fcntl.flock(global_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    cli = str(Path(sys.executable).with_name("modal"))
    def modal_json(*parts):
        return json.loads(subprocess.run([cli,*parts,"--json"], capture_output=True, text=True,
                                         check=True, timeout=30).stdout)
    active = [x for x in modal_json("app","list") if int(x.get("tasks",0))]
    if active: raise RuntimeError(f"Other Modal tasks active: {active}")
    rates = modal_json("billing","rates")
    hourly = float(rates["gpu_hour_cost_l4"])+2*float(rates["cpu_hour_cost"])+4*float(rates["mem_gib_hour_cost"])
    deadline = time.time()+4*3600
    ledger = dict(status="starting", attempts=[], app_id=None, start_epoch=time.time(), deadline_epoch=deadline,
                  previous_conservative_upper_usd=PREVIOUS_UPPER_USD, total_cap_usd=TOTAL_CAP_USD,
                  max_l4=MAX_L4, hourly_rate=hourly, conservative_total_upper_usd=PREVIOUS_UPPER_USD+3)
    mutex = threading.RLock()
    def save():
        with mutex:
            elapsed = time.time()-ledger["start_epoch"]
            ledger["conservative_total_upper_usd"] = PREVIOUS_UPPER_USD+3+elapsed*MAX_L4*hourly*1.1/3600
            atomic_json(run/"ledger.json", ledger)
    save()
    image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("numpy==2.5.3","torch==2.14.0","einops==0.8.2","scipy==1.18.1")
             .env({"PYTHONPATH":"/opt/pilot/src:/opt/pilot/scripts","OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2"})
             .add_local_dir(prepared/"frozen", remote_path="/opt/pilot", ignore=["**/__pycache__/**"]))
    app = modal.App("hypercast4d-crossformer-baseline")
    remote = app.function(image=image, gpu="L4", cpu=(2,2), memory=(4096,4096), retries=0,
                          timeout=3600, startup_timeout=180, max_containers=MAX_L4,
                          scaledown_window=60, serialized=True)(worker)
    data = (prepared/"development.npz").read_bytes()
    if sha(prepared/"development.npz") != plan["development_sha256"]: raise ValueError("Data changed")
    predictions = run/"predictions"; predictions.mkdir()
    def execute(jobs, stage):
        ledger["status"] = stage+"_running"; save(); rows=[]
        def task(job):
            with mutex:
                save()
                if ledger["conservative_total_upper_usd"] >= 72 or time.time()+job["timeout_seconds"]+180 >= deadline:
                    raise RuntimeError("Budget/deadline admission stop")
                index = len(ledger["attempts"])
                attempt = dict(index=index, stage=stage, job=job, status="submitting", submitted_epoch=time.time())
                ledger["attempts"].append(attempt); save()
            call = remote.spawn(plan, data, job)
            with mutex: attempt.update(status="running", call_id=call.object_id); save()
            try:
                status, blob = call.get(timeout=job["timeout_seconds"]+180)
                archive_path = run/f"job-{index:03d}.tar.gz"; archive_path.write_bytes(blob)
                if not status["ok"]: raise RuntimeError(status["error"])
                with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
                    result = json.load(archive.extractfile("trial/result.json"))
                    (predictions/f"{index:03d}.npz").write_bytes(archive.extractfile("trial/development_predictions.npz").read())
                if not result["checkpoint_replay_passed"]: raise RuntimeError("Checkpoint replay failed")
                with mutex: attempt.update(status="complete", ended_epoch=time.time(), archive_sha256=sha(archive_path)); save()
                return result
            except BaseException as exc:
                try: call.cancel()
                except Exception: pass
                with mutex: attempt.update(status="failed", ended_epoch=time.time(), error=repr(exc)); save()
                raise
        with ThreadPoolExecutor(max_workers=MAX_L4) as pool:
            futures = [pool.submit(task, j) for j in sorted(jobs, key=lambda x:x["timeout_seconds"], reverse=True)]
            for future in as_completed(futures):
                rows.append(future.result()); atomic_json(run/f"{stage}-results.json", rows); save()
                print(json.dumps(dict(stage=stage, returned=len(rows), total=len(jobs),
                                      upper_usd=ledger["conservative_total_upper_usd"])), flush=True)
        return rows
    def job(stage, seed, window, mode, learning_rate, schedule="constant", d_model=32, dropout=.1,
            epochs=100, patience=15):
        timeout = {32:1200,96:2100,192:3300}[window]
        return dict(phase=stage, seed=seed, window=window, mode=mode, learning_rate=learning_rate,
                    schedule=schedule, d_model=d_model, dropout=dropout, epochs=epochs,
                    patience=patience, timeout_seconds=timeout)
    try:
        with app.run():
            ledger["app_id"] = app.app_id; save()
            screen_jobs = [job("screen",4101,w,m,lr) for w in plan["context_candidates"]
                           for m in plan["modes"] for lr in plan["learning_rates"]]
            screen = execute(screen_jobs,"screen")
            best_screen = min(screen, key=lambda r:(r["development_mae"],signature(r["job"])))
            w,m = best_screen["job"]["window"],best_screen["job"]["mode"]
            atomic_json(run/"screen-selection.json",dict(selected_window=w,selected_mode=m,best=best_screen))
            refine_jobs = [job("refine",4201,w,m,lr,schedule,dmodel,dropout)
                           for lr in plan["learning_rates"] for schedule in plan["schedules"]
                           for dmodel in plan["d_models"] for dropout in plan["dropouts"]]
            refine = execute(refine_jobs,"refine")
            ranked = sorted(refine,key=lambda r:(r["development_mae"],signature(r["job"])))
            top = ranked[:3]
            atomic_json(run/"refinement-selection.json",dict(candidates=[r["job"] for r in top]))
            confirm_jobs=[]
            for candidate in top:
                for seed in plan["confirmation_seeds"]:
                    j={**candidate["job"],"phase":"confirmation","seed":seed,"epochs":150,"patience":20}
                    confirm_jobs.append(j)
            confirmation=execute(confirm_jobs,"confirmation")
            groups={signature(c["job"]):[] for c in top}
            for result in confirmation: groups[signature(result["job"])].append(result)
            summaries=[]
            for sig,rr in groups.items():
                summaries.append(dict(settings=rr[0]["job"], mean_development_mae=float(np.mean([x["development_mae"] for x in rr])),
                                      seed_development_mae=[x["development_mae"] for x in sorted(rr,key=lambda x:x["job"]["seed"])],
                                      persistence_mae=rr[0]["persistence_mae"], beats_persistence_all_seeds=all(x["beats_persistence"] for x in rr),
                                      mean_epochs=float(np.mean([x["epochs_ran"] for x in rr])), parameters=rr[0]["parameters"]))
            selected=min(summaries,key=lambda x:(x["mean_development_mae"],signature(x["settings"])))
            atomic_json(run/"selected-baseline.json",selected)
            with (run/"confirmation-summary.csv").open("w",newline="") as f:
                fields=["window","mode","learning_rate","schedule","d_model","dropout","mean_development_mae","persistence_mae","improvement_vs_persistence_pct","mean_epochs","parameters","seed_development_mae"]
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                for x in sorted(summaries,key=lambda x:x["mean_development_mae"]):
                    settings=x["settings"]; writer.writerow({**{k:settings[k] for k in fields[:6]},
                        "mean_development_mae":x["mean_development_mae"],"persistence_mae":x["persistence_mae"],
                        "improvement_vs_persistence_pct":100*(x["persistence_mae"]-x["mean_development_mae"])/x["persistence_mae"],
                        "mean_epochs":x["mean_epochs"],"parameters":x["parameters"],"seed_development_mae":json.dumps(x["seed_development_mae"])})
            improvement=100*(selected["persistence_mae"]-selected["mean_development_mae"])/selected["persistence_mae"]
            report=["# Crossformer baseline tuning", "", "The ETTh1 tail at rows [14400,17420) remained sealed.", "",
                    f"Selected configuration: context {selected['settings']['window']}, {selected['settings']['mode']}, d_model {selected['settings']['d_model']}, dropout {selected['settings']['dropout']}, {selected['settings']['schedule']} schedule, learning rate {selected['settings']['learning_rate']}.", "",
                    f"Mean development MAE across three fresh seeds: {selected['mean_development_mae']:.5f}; persistence MAE: {selected['persistence_mae']:.5f}; improvement: {improvement:.2f}%.", "",
                    "This is a development-selected baseline for the upcoming layer-location sweep, not a final held-out result.", ""]
            (run/"report.md").write_text("\n".join(report))
            ledger["status"]="complete"; ledger["selected_baseline"]=selected; ledger["end_epoch"]=time.time(); save()
    except BaseException as exc:
        ledger["status"]="stopped_for_review"; ledger["error"]=repr(exc); ledger["end_epoch"]=time.time(); save(); raise
    finally:
        ledger["end_epoch"]=time.time(); save()
        print(json.dumps(dict(status=ledger["status"],upper_usd=ledger["conservative_total_upper_usd"])),flush=True)


if __name__ == "__main__": main()
