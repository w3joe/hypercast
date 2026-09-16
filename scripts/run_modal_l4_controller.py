#!/usr/bin/env python3
"""Controller for long-running, resumable HyperDense trials on Modal L4.

The controller keeps explicit manifest state in a JSON file and resumes safely
across interruptions.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hypercast4d.architecture import architecture_hash, presets as load_presets
from hypercast4d.compute import normalize_execution
from hypercast4d.model_editing import checked as checked_spec, load_model, write_model
from hypercast4d.workspace_runtime import ensure_backend, request
from hypercast4d.playground_runner import normalize_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "modal-l4-runner"
STATE_FILENAME = "controller_state.json"
MODAL_RATE_COMMAND = [
    str(Path(sys.executable).with_name("modal")),
    "billing",
    "rates",
    "--json",
]
MODAL_SUMMARY_COMMAND = [
    str(Path(sys.executable).with_name("modal")),
    "billing",
    "summary",
    "--json",
    "--for",
    "this month",
]


@dataclass(frozen=True)
class Variant:
    key: str
    name: str
    layers: list[dict[str, Any]]


BACKBONES = ("tslib-dlinear", "tslib-tsmixer", "tslib-itransformer")
VARIANTS = [
    Variant(
        key="original",
        name="Original",
        layers=[],
    ),
    Variant(
        key="lift_only",
        name="Lift only",
        layers=[
            {"id": "input_lift", "type": "dense", "params": {"units": 32}},
            {"id": "lift_activation", "type": "activation", "params": {"kind": "relu"}},
        ],
    ),
    Variant(
        key="real_dense",
        name="Real dense",
        layers=[
            {"id": "input_lift", "type": "dense", "params": {"units": 32}},
            {"id": "lift_activation", "type": "activation", "params": {"kind": "relu"}},
            {"id": "test_layer", "type": "dense", "params": {"units": 32}},
            {"id": "test_activation", "type": "activation", "params": {"kind": "relu"}},
        ],
    ),
    Variant(
        key="complex",
        name="Complex 2D",
        layers=[
            {"id": "input_lift", "type": "dense", "params": {"units": 32}},
            {"id": "lift_activation", "type": "activation", "params": {"kind": "relu"}},
            {"id": "test_layer", "type": "hyper_dense", "params": {"units": 16, "algebra": "complex"}},
            {"id": "test_activation", "type": "activation", "params": {"kind": "relu"}},
        ],
    ),
    Variant(
        key="quaternion",
        name="Quaternion 4D",
        layers=[
            {"id": "input_lift", "type": "dense", "params": {"units": 32}},
            {"id": "lift_activation", "type": "activation", "params": {"kind": "relu"}},
            {"id": "test_layer", "type": "hyper_dense", "params": {"units": 8, "algebra": "quaternion"}},
            {"id": "test_activation", "type": "activation", "params": {"kind": "relu"}},
        ],
    ),
    Variant(
        key="octonion",
        name="Octonion 8D",
        layers=[
            {"id": "input_lift", "type": "dense", "params": {"units": 32}},
            {"id": "lift_activation", "type": "activation", "params": {"kind": "relu"}},
            {"id": "test_layer", "type": "hyper_dense", "params": {"units": 4, "algebra": "octonion"}},
            {"id": "test_activation", "type": "activation", "params": {"kind": "relu"}},
        ],
    ),
    Variant(
        key="real_rank8",
        name="Real rank-8",
        layers=[
            {"id": "input_lift", "type": "dense", "params": {"units": 32}},
            {"id": "lift_activation", "type": "activation", "params": {"kind": "relu"}},
            {"id": "test_layer", "type": "dense", "params": {"units": 8}},
            {"id": "test_layer_expand", "type": "dense", "params": {"units": 32}},
            {"id": "test_activation", "type": "activation", "params": {"kind": "relu"}},
        ],
    ),
    Variant(
        key="real_rank4",
        name="Real rank-4",
        layers=[
            {"id": "input_lift", "type": "dense", "params": {"units": 32}},
            {"id": "lift_activation", "type": "activation", "params": {"kind": "relu"}},
            {"id": "test_layer", "type": "dense", "params": {"units": 4}},
            {"id": "test_layer_expand", "type": "dense", "params": {"units": 32}},
            {"id": "test_activation", "type": "activation", "params": {"kind": "relu"}},
        ],
    ),
    Variant(
        key="real_rank2",
        name="Real rank-2",
        layers=[
            {"id": "input_lift", "type": "dense", "params": {"units": 32}},
            {"id": "lift_activation", "type": "activation", "params": {"kind": "relu"}},
            {"id": "test_layer", "type": "dense", "params": {"units": 2}},
            {"id": "test_layer_expand", "type": "dense", "params": {"units": 32}},
            {"id": "test_activation", "type": "activation", "params": {"kind": "relu"}},
        ],
    ),
]


STAGES = {
    "pilot": {
        "preset": "standard",
        "cells": "10/1,20/1",
        "seeds": [7],
        "folds": 1,
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "target": "modal",
        "gpu": "L4",
    },
}

RUNNING_STATES = {"queued", "starting", "running"}
TERMINAL_STATES = {"complete", "failed", "cancelled", "interrupted"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_json_command(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")
    try:
        return json.loads((result.stdout or "{}"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Could not parse JSON from {' '.join(command)}") from error


def load_modal_rates() -> dict[str, float]:
    rates = _run_json_command(MODAL_RATE_COMMAND)
    parsed = {
        "gpu_l4_per_hour": float(rates.get("gpu_hour_cost_l4", 0.0)),
        "cpu_core_per_hour": float(rates.get("cpu_hour_cost", 0.0)),
        "mem_gib_hour_cost": float(rates.get("mem_gib_hour_cost", 0.0)),
    }
    if any(not math.isfinite(value) or value <= 0 for value in parsed.values()):
        raise RuntimeError("Modal returned invalid resource rates; submissions paused")
    return parsed


def parse_monthly_summary() -> dict[str, float]:
    summary = _run_json_command(MODAL_SUMMARY_COMMAND)
    parsed = {
        "metered_cost": float(summary.get("metered_cost", "0")),
        "billed_cost": float(summary.get("billed_cost", "0")),
    }
    if "metered_cost" not in summary or any(not math.isfinite(v) or v < 0 for v in parsed.values()):
        raise RuntimeError("Modal returned invalid billing totals; submissions paused")
    return parsed


def normalize_cells(raw: str) -> list[dict[str, int]]:
    cells: list[dict[str, int]] = []
    seen = set()
    for pair in raw.split(","):
        window, horizon = pair.strip().split("/")
        cell = {"window": int(window), "horizon": int(horizon)}
        key = (cell["window"], cell["horizon"])
        if key in seen:
            raise ValueError(f"Duplicate cell {cell['window']}/{cell['horizon']}")
        seen.add(key)
        cells.append(cell)
    if not cells:
        raise ValueError("At least one cell is required")
    return cells


def create_candidate_list(results_root: Path, evaluation: dict[str, Any], execution: dict[str, Any], specs_dir: Path) -> list[dict[str, Any]]:
    presets = {preset["preset_id"]: preset for preset in load_presets() if preset.get("preset_id") in BACKBONES}
    candidates: list[dict[str, Any]] = []

    for backbone in BACKBONES:
        base = presets[backbone]
        for variant in VARIANTS:
            variant_layers = deepcopy(variant.layers)
            spec = deepcopy(base)
            spec.pop("preset_id", None)
            spec["locked"] = False
            spec["name"] = f"{base['name']} + {variant.name}"
            if variant_layers:
                spec["layers"] = variant_layers + spec["layers"]
            checked_spec(spec, evaluation["cells"])

            candidate_hash = architecture_hash(
                spec,
                {**evaluation, "execution": execution},
            )
            slug = f"{backbone}-{variant.key}"
            path = specs_dir / f"{slug}.yaml"
            write_model(path, spec, overwrite=True)
            candidates.append(
                {
                    "id": slug,
                    "backbone": backbone,
                    "variant": variant.key,
                    "variant_name": variant.name,
                    "architecture_file": str(path.relative_to(results_root)),
                    "candidate_hash": candidate_hash,
                    "status": "pending",
                    "job_id": None,
                    "attempts": 0,
                    "started_at": None,
                    "finished_at": None,
                    "reserved_usd": 0.0,
                    "estimated_cost_usd": 0.0,
                    "observed_cost_usd": 0.0,
                    "status_message": "",
                }
            )
    return candidates


def estimate_job_cost_usd(num_trials: int, rates: dict[str, float], avg_seconds_per_trial: float, uncertainty_factor: float, startup_usd: float) -> float:
    cpu = rates["cpu_core_per_hour"] * 2.0
    mem = rates["mem_gib_hour_cost"] * 4.0
    gpu = rates["gpu_l4_per_hour"]
    per_second = (cpu + mem + gpu) / 3600.0
    return round(max(1.0, avg_seconds_per_trial) * max(1, int(num_trials)) * per_second * uncertainty_factor + startup_usd, 6)


def estimate_trials(stage: dict[str, Any], total_folds_override: int | None = None) -> int:
    folds = int(total_folds_override) if total_folds_override is not None else max(1, len(stage.get("folds", [])))
    return max(1, len(stage["cells"])) * max(1, len(stage.get("seeds", []))) * max(1, folds) * (8 if stage.get('remaining_replication') else 16 if stage.get('remaining_tuning') else 1)


def canonical_hash(architecture: dict, evaluation: dict, execution: dict) -> str:
    # Runtime limits do not change the scientific configuration. Match both old
    # requests and new bounded requests using the backend's normalized defaults.
    execution = normalize_execution(execution)
    execution.pop("timeout_seconds", None)
    evaluation = normalize_evaluation({**evaluation, "device": "cuda"})
    return architecture_hash(architecture, {**evaluation, "execution": execution})


def migrate_state(state: dict) -> None:
    if float(state["budget"]["total_usd"]) > 20:
        raise RuntimeError("This programme is authorized for at most $20")
    if normalize_execution(state["stage"]["execution"]) != {"target": "modal", "gpu": "L4"}:
        raise RuntimeError("Controller requires exactly one Modal L4")
    for candidate in state["candidates"]:
        architecture = load_model(Path(state["results_root"]) / candidate["architecture_file"])
        candidate["candidate_hash"] = canonical_hash(
            architecture, candidate_evaluation(state, candidate), state["stage"]["execution"])
        if candidate.get("phase") == "final_test":
            candidate["candidate_hash"] = "final-test:" + candidate["candidate_hash"]
    state["schema_version"] = 2


def candidate_evaluation(state: dict, candidate: dict) -> dict:
    return candidate.get("evaluation", state["stage"]["evaluation"])


def verify_frozen_inputs(state: dict) -> None:
    for relative, expected in state.get("frozen_inputs", {}).items():
        path = PROJECT_ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen experiment input changed: {relative}; review and version a new stage")


def prepare_controlled_development(state: dict) -> None:
    """Append a balanced development stage; never reset original jobs or costs."""
    if any(c.get("stage_name") == "controlled-development" for c in state["candidates"]):
        return
    if not all(c["status"] == "complete" for c in state["candidates"]):
        raise RuntimeError("Finish and reconcile the pilot before preparing development")
    originals = list(state["candidates"])
    for c in originals:
        c.setdefault("stage_name", "pilot")
    base_eval = {"preset": "standard", "cells": [{"window": 20, "horizon": 5}],
                 "seeds": [19], "epochs": 150, "batch_size": 32,
                 "initialization": "matched-v1", "benchmark": True,
                 "early_stopping_patience": None, "restore_best_weights": False}

    def fresh(original, identifier, stage, evaluation, timeout):
        row = {key: original[key] for key in ("backbone", "variant", "variant_name", "architecture_file")}
        row.update(id=identifier, stage_name=stage, evaluation=evaluation, timeout_seconds=timeout,
                   status="pending", job_id=None, attempts=0, reserved_usd=0., estimated_cost_usd=0.,
                   started_at=None, finished_at=None, status_message="")
        return row

    preflight = fresh(originals[0], "controlled-preflight", "controlled-preflight",
        {**base_eval, "epochs": 1, "seeds": [7], "learning_rate": .0003, "numerical_preflight": True}, 600)
    state["candidates"].append(preflight)
    # Finish every family at one rate before the next rate. No winner filtering.
    for rate in (.0003, .001):
        for original in originals:
            row = fresh(original, f"{original['id']}-matched-lr{rate:g}", "controlled-development",
                        {**base_eval, "learning_rate": rate}, 900)
            row["depends_on"] = [preflight["id"]]
            state["candidates"].append(row)
    state["stage_name"] = "controlled-development"
    state["schedule_status"] = "prepared"
    state.setdefault("stage_history", []).append({"stage": "pilot", "completed": 27,
        "finished_at": state.get("updated_at"), "accounted_spend_usd": accounted_spend(state)})
    migrate_state(state)


def _jobs_by_hash(jobs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets = defaultdict(list)
    for job in jobs:
        payload = job["request"]
        if payload.get("phase") not in {"validation", "final_test"}:
            continue
        key = canonical_hash(payload["architecture"], payload["evaluation"], payload["execution"])
        if payload.get("phase") == "final_test":
            key = "final-test:" + key
        buckets[key].append(job)
    return buckets


def load_state(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("Corrupt controller state")
    return data


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def ensure_single_controller(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)

    @contextlib.contextmanager
    def manager():
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(lock_fd)
            raise RuntimeError("Another controller instance is already active") from error
        try:
            os.ftruncate(lock_fd, 0)
            os.lseek(lock_fd, 0, os.SEEK_SET)
            os.write(lock_fd, f"{os.getpid()}\n".encode())
            os.lseek(lock_fd, 0, os.SEEK_SET)
            yield
        finally:
            try:
                os.ftruncate(lock_fd, 0)
                os.close(lock_fd)
            except OSError:
                pass

    return manager()


def parse_iso_time(raw: str | None) -> float:
    if not raw:
        return time.time()
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return time.time()


def collect_job_metrics(results_root: str | Path, job_id: str) -> float:
    runs_path = Path(results_root) / "jobs" / job_id / "runs.csv"
    if not runs_path.exists():
        return 0.0

    values: list[float] = []
    with runs_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_value = row.get("train_seconds")
            if raw_value is None:
                continue
            try:
                values.append(float(raw_value))
            except (TypeError, ValueError):
                continue
    if not values:
        return 0.0
    return sum(values) / len(values)


def reconcile_candidate_state(state, jobs, now, rates, stages):
    by_hash = _jobs_by_hash(jobs)
    # Unknown or orphaned jobs must also occupy the single GPU slot.
    has_active = any(j["status"].get("state") not in TERMINAL_STATES for j in jobs)
    for candidate in state["candidates"]:
        rows = by_hash.get(candidate["candidate_hash"], [])
        if not rows:
            if candidate.get("attempts", 0) or candidate.get("job_id"):
                candidate["status"] = "missing"
                candidate["status_message"] = "Unresolved submission; retain reservation and investigate before retry"
                has_active = True
            continue
        candidate["job_ids"] = sorted(row["id"] for row in rows)
        candidate["attempts"] = max(candidate.get("attempts", 0), len(rows))
        active = [r for r in rows if r["status"].get("state") not in TERMINAL_STATES]
        complete = [r for r in rows if r["status"].get("state") == "complete"]
        # Preserve the first successful replicate; duplicate artifacts stay intact.
        tracked = sorted(active or complete or rows, key=lambda r: r["id"])[0 if (active or complete) else -1]
        status = tracked["status"]
        name = status.get("state", "unknown")
        candidate["job_id"] = tracked["id"]
        candidate["started_at"] = status.get("created_at")
        candidate["finished_at"] = status.get("updated_at") if name in TERMINAL_STATES else None
        candidate["status"] = name if name in TERMINAL_STATES else "running"
        candidate["status_message"] = status.get("error") or f"state={name}, completed={status.get('completed', 0)}/{status.get('total', 0)}"
        if name == "complete":
            expected = estimate_trials(normalize_evaluation(candidate_evaluation(state, candidate)),
                total_folds_override=1 if candidate.get("phase") == "final_test" else None)
            if int(status.get("completed", 0)) != expected or len(tracked.get("runs", [])) != expected:
                candidate["status"] = "invalid_artifacts"
                candidate["status_message"] = "Complete status has missing trial artifacts; investigate before retry"
        if name in TERMINAL_STATES:
            candidate["reserved_usd"] = 0.0
        else:
            candidate["reserved_usd"] = max(candidate.get("reserved_usd", 0), expected_cost_for_candidate(state, candidate, 1))
    return True, has_active


def update_cost_ledger(state: dict, jobs: list[dict]) -> None:
    # Include every attempt (including duplicates/failures), and container/startup
    # time, not merely training seconds. Keep estimates across backend outages.
    ledger = state.setdefault("cost_ledger", {})
    budget = state["budget"]
    for job in jobs:
        status = job["status"]
        if status.get("state") not in TERMINAL_STATES:
            continue
        old = ledger.get(job["id"], {})
        if old.get("state") == status["state"]:
            continue
        start, end = status.get("created_at"), status.get("updated_at")
        seconds = max(1.0, parse_iso_time(end) - parse_iso_time(start)) if start and end else budget["max_job_seconds"]
        # Interrupted/cancelled jobs can outlive local status. Retain the full
        # remote timeout reservation unless remote completion is confirmed.
        if status["state"] in {"cancelled", "interrupted"}:
            seconds = max(seconds, job["request"].get("execution", {}).get("timeout_seconds", 86400))
        cost = estimate_job_cost_usd(1, state["rates"], seconds,
            budget["uncertainty_factor"], budget["startup_cost_usd"])
        ledger[job["id"]] = {"state": status["state"], "seconds": seconds, "estimated_usd": cost}
    budget["estimated_spend_usd"] = round(sum(row["estimated_usd"] for row in ledger.values()), 6)
    budget["jobs_submitted"] = max(budget.get("jobs_submitted", 0), len(jobs))


def accounted_spend(state: dict) -> float:
    budget = state["budget"]
    return max(float(budget.get("confirmed_spend_usd", 0)), float(budget.get("estimated_spend_usd", 0)))


def remote_apps_active() -> bool:
    apps = _run_json_command([str(Path(sys.executable).with_name("modal")), "app", "list", "--json"])
    return any(app.get("description") == "hypercast4d-playground-jobs"
        and (app.get("state") != "stopped" or int(app.get("tasks", 0)) > 0) for app in apps)


def stage_cost_breakdown(state: dict[str, Any], rates: dict[str, float], billing: dict[str, float]) -> None:
    running = [row for row in state["candidates"] if row["status"] in RUNNING_STATES]
    budget = state["budget"]
    in_flight = sum(float(row.get("reserved_usd", 0.0)) for row in running)
    confirmed = accounted_spend(state)
    estimated = sum(float(row.get("estimated_cost_usd", 0.0)) for row in state["candidates"] if row["status"] in RUNNING_STATES | {"pending", "failed", "interrupted", "blocked_by_budget"})
    available = max(0.0, float(budget["total_usd"]) - confirmed - in_flight - budget["safety_reserve_usd"])
    summary = {
        "candidates_pending": len([row for row in state["candidates"] if row["status"] in {"pending", "interrupted", "failed", "blocked_by_budget"}]),
        "candidates_running": len(running),
        "candidates_complete": len([row for row in state["candidates"] if row["status"] == "complete"]),
        "candidates_exhausted": len([row for row in state["candidates"] if row["status"] == "exhausted"]),
        "confirmed_spend_usd": budget["confirmed_spend_usd"],
        "conservative_accounted_spend_usd": round(confirmed, 6),
        "inflight_reservation_usd": round(in_flight, 6),
        "estimated_run_cost_reserve_usd": round(estimated, 6),
        "available_after_reserve_and_safety_usd": round(available, 6),
        "billing_delta_metered_usd": max(0.0, billing["metered_cost"] - float(state["billing"]["baseline_metered_usd"])) ,
    }
    print(json.dumps(summary, sort_keys=True))


def build_state(results_root: Path, stage_name: str) -> dict[str, Any]:
    rates = load_modal_rates()
    baseline = parse_monthly_summary()
    stage = STAGES[stage_name]
    execution = normalize_execution({"target": stage["target"], "gpu": stage["gpu"]})
    evaluation = {
        "preset": stage["preset"],
        "cells": normalize_cells(stage["cells"]),
        "seeds": stage["seeds"],
        "folds": [{"train_fraction": 0.70, "validation_fraction": 0.15}] * int(stage.get("folds", 1)),
        "epochs": stage["epochs"],
        "batch_size": stage["batch_size"],
        "learning_rate": stage["learning_rate"],
    }
    specs_dir = results_root / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    candidates = create_candidate_list(results_root, evaluation, execution, specs_dir)
    return {
        "experiment_id": "modal-l4-pilot-2026-09-10",
        "created_at": _utcnow(),
        "updated_at": _utcnow(),
        "project_root": str(PROJECT_ROOT),
        "results_root": str(results_root),
        "stage_name": stage_name,
        "stage": {
            "evaluation": evaluation,
            "execution": execution,
        },
        "rates": rates,
        "budget": {
            "total_usd": 20.0,
            "safety_reserve_usd": 1.0,
            "max_job_seconds": 3600,
            "avg_fit_seconds": 90.0,
            "jobs_submitted": 0,
            "confirmed_spend_usd": 0.0,
            "estimated_spend_usd": 0.0,
            "max_retries": 1,
            "uncertainty_factor": 3.0,
            "startup_cost_usd": 0.05,
        },
        "billing": {
            "baseline_metered_usd": baseline["metered_cost"],
            "baseline_billed_usd": baseline["billed_cost"],
        },
        "candidates": candidates,
    }


def initialize_state(state_path: Path, results_root: Path, stage_name: str) -> dict[str, Any]:
    if not state_path.exists():
        state = build_state(results_root, stage_name)
        save_state(state_path, state)
        return state

    state = load_state(state_path)
    if state.get("stage_name") != stage_name:
        raise RuntimeError("Stage changes must preserve the existing budget ledger")
    return state


def expected_cost_for_candidate(state: dict[str, Any], candidate: dict[str, Any], num_trials: int) -> float:
    budget = state["budget"]
    # Reserve the remote timeout plus cancellation/startup overhead. Pilot mean
    # duration is useful for planning, but is not a safe maximum reservation.
    return estimate_job_cost_usd(1, state["rates"], float(candidate.get("timeout_seconds", budget["max_job_seconds"])) + 120,
        float(budget["uncertainty_factor"]), float(budget["startup_cost_usd"]))


def can_submit_candidate(state: dict[str, Any], candidate: dict[str, Any], num_trials: int) -> bool:
    budget = state["budget"]
    inflight = sum(float(row.get("reserved_usd", 0.0)) for row in state["candidates"] if row["status"] in RUNNING_STATES)
    confirmed = accounted_spend(state)
    safety = float(budget["safety_reserve_usd"])
    reservation = expected_cost_for_candidate(state, candidate, num_trials)
    available = float(budget["total_usd"]) - confirmed - inflight
    candidate["estimated_cost_usd"] = reservation
    return available >= safety + reservation and reservation > 0


def submit_candidate(base_url: str, state: dict[str, Any], candidate: dict[str, Any], num_trials: int) -> None:
    candidate_path = Path(state["results_root"]) / candidate["architecture_file"]
    architecture = load_model(candidate_path)
    request_payload = {
        "architecture": architecture,
        "evaluation": candidate_evaluation(state, candidate),
        "execution": {**state["stage"]["execution"], "timeout_seconds": int(candidate.get("timeout_seconds", state["budget"]["max_job_seconds"]))},
    }
    if candidate.get("phase") == "final_test":
        job = request(base_url, f"/api/v1/jobs/{candidate['parent_job_id']}/final-test",
                      {"timeout_seconds": request_payload["execution"]["timeout_seconds"]})
    else:
        job = request(base_url, "/api/v1/jobs", request_payload)
    if not isinstance(job, dict) or not job.get("id"):
        raise RuntimeError("Job submission response missing id")

    reservation = expected_cost_for_candidate(state, candidate, num_trials)
    candidate.update(
        {
            "job_id": job["id"],
            "status": "running",
            "attempts": candidate.get("attempts", 0),
            "started_at": _utcnow(),
            "finished_at": None,
            "reserved_usd": reservation,
            "estimated_cost_usd": reservation,
            "status_message": f"submitted: {job['id']}",
        }
    )
    state["budget"]["jobs_submitted"] = state["budget"].get("jobs_submitted", 0) + 1


def choose_next_candidate(state: dict[str, Any]) -> dict[str, Any] | None:
    status_by_id = {c["id"]: c["status"] for c in state["candidates"]}
    for candidate in state["candidates"]:
        if candidate["status"] in {"pending", "failed", "interrupted"}:
            if not all(status_by_id.get(key) == "complete" for key in candidate.get("depends_on", [])):
                continue
            if candidate.get("attempts", 0) >= 1 + candidate.get("max_retries", state["budget"].get("max_retries", 1)):
                candidate["status"] = "exhausted"
            else:
                return candidate
    return None


def enforce_runtime_guards(state: dict[str, Any], base_url: str, now: str) -> bool:
    changed = False
    for candidate in state["candidates"]:
        if candidate["status"] not in RUNNING_STATES:
            continue
        started = parse_iso_time(candidate.get("started_at"))
        max_seconds = float(candidate.get("timeout_seconds", state["budget"].get("max_job_seconds", 3600.0)))
        over_budget = accounted_spend(state) + float(candidate.get("reserved_usd", 0)) > float(state["budget"]["total_usd"]) - float(state["budget"]["safety_reserve_usd"])
        if (time.time() - started) <= max_seconds and not over_budget:
            continue

        try:
            request(base_url, f"/api/v1/jobs/{candidate['job_id']}/cancel", {})
            candidate["status"] = "running"
            candidate["finished_at"] = None
            candidate["status_message"] = "Cancellation requested by budget guard" if over_budget else "Cancellation requested by runtime guard"
            changed = True
        except Exception as error:
            candidate["status_message"] = f"runtime guard: cancel failed: {error}"
            changed = True
    return changed


def refresh_billing(state: dict[str, Any], billing: dict[str, float]) -> None:
    state["latest_billing_metered_usd"] = billing["metered_cost"]
    state["latest_billing_billed_usd"] = billing["billed_cost"]
    baseline = float(state["billing"]["baseline_metered_usd"])
    delta = max(0.0, billing["metered_cost"] - baseline)
    state["budget"]["confirmed_spend_usd"] = round(max(float(state["budget"].get("confirmed_spend_usd", 0.0)), delta), 6)


def run_cycle(state_path, results_root, lock_path, watch, poll_interval, max_steps, reconcile_only=False):
    with ensure_single_controller(lock_path):
        stage_name = load_state(state_path).get("stage_name", "pilot") if state_path.exists() else "pilot"
        state = initialize_state(state_path, results_root, stage_name)
        migrate_state(state)
        state["controller_pid"] = os.getpid()
        step = 0
        while True:
            try:
                backend = ensure_backend(PROJECT_ROOT, results_root)
                base_url = str(backend["url"])
                jobs = request(base_url, "/api/v1/jobs")
                _, has_running = reconcile_candidate_state(state, jobs, _utcnow(), state["rates"], state["stage"]["evaluation"])
                update_cost_ledger(state, jobs)
                # Billing or cloud-state failure prevents submission, but the
                # runtime guard still executes and the daemon retries next cycle.
                enforce_runtime_guards(state, base_url, _utcnow())
                rates = load_modal_rates()
                state["rates"] = {key: max(state["rates"][key], value) for key, value in rates.items()}
                billing = parse_monthly_summary()
                refresh_billing(state, billing)
                enforce_runtime_guards(state, base_url, _utcnow())
                cloud_active = remote_apps_active()
                state["remote_app_active"] = cloud_active
                state.pop("last_error", None)
                if not has_running and not cloud_active and not reconcile_only:
                    candidate = choose_next_candidate(state)
                    if candidate is not None:
                        verify_frozen_inputs(state)
                        trials = estimate_trials(normalize_evaluation(candidate_evaluation(state, candidate)),
                            total_folds_override=1 if candidate.get("phase") == "final_test" else None)
                        if can_submit_candidate(state, candidate, trials):
                            # Persist intent first: even a lost HTTP response or
                            # process crash must not cause an automatic duplicate.
                            candidate.update(status="submitting", attempts=candidate.get("attempts", 0) + 1,
                                started_at=_utcnow(), reserved_usd=candidate["estimated_cost_usd"])
                            save_state(state_path, state)
                            submit_candidate(base_url, state, candidate, trials)
                            state["schedule_status"] = "running"
                        else:
                            candidate["status"] = "blocked_by_budget"
                state["updated_at"] = _utcnow()
                save_state(state_path, state)
                stage_cost_breakdown(state, state["rates"], billing)
            except Exception as error:
                state["last_error"] = f"{type(error).__name__}: {error}"
                state["updated_at"] = _utcnow()
                save_state(state_path, state)
                print(f"Controller paused submissions; retrying: {state['last_error']}", flush=True)
                if not watch:
                    raise
            step += 1
            if not watch or reconcile_only or (max_steps is not None and step >= max_steps):
                return
            if not state.get("last_error"):
                runnable = any(c["status"] in RUNNING_STATES | {"submitting", "missing"} for c in state["candidates"])
                if not runnable:
                    runnable = choose_next_candidate(state) is not None
                if not runnable and not state.get("remote_app_active"):
                    state["schedule_status"] = f"{state['stage_name']}_complete" if all(c["status"] == "complete" for c in state["candidates"]) else "needs_review"
                    save_state(state_path, state)
                    print(f"Schedule stopped: {state['schedule_status']}. AI supervisor should review the next stage.", flush=True)
                    return
            time.sleep(max(1, poll_interval))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Workspace-specific results root for this controller (separate from other runs)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep the controller active and run multiple scheduling cycles until budget or completion.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=45,
        help="Polling interval when watching",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Maximum watch cycles before exiting",
    )
    parser.add_argument(
        "--stage",
        default="pilot",
        choices=tuple(STAGES),
        help="Execution stage from the pre-approved plan",
    )
    parser.add_argument("--reconcile-only", action="store_true", help="Repair state and costs without submitting jobs")
    parser.add_argument("--prepare-controlled-development", action="store_true", help="Append the documented controlled 150-epoch development stage without submitting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    if args.stage != "pilot":
        raise ValueError("Only pilot is prepared in this setup run")

    state_root = results_root / "controller"
    state_path = state_root / STATE_FILENAME
    lock_path = state_root / ".controller.lock"

    if args.prepare_controlled_development:
        with ensure_single_controller(lock_path):
            state = load_state(state_path)
            prepare_controlled_development(state)
            save_state(state_path, state)
            print("Controlled development prepared; existing budget and pilot artifacts preserved.")
        return

    run_cycle(
        state_path=state_path,
        results_root=results_root,
        lock_path=lock_path,
        watch=args.watch,
        poll_interval=args.poll_seconds,
        max_steps=args.steps,
        reconcile_only=args.reconcile_only,
    )


if __name__ == "__main__":
    main()
