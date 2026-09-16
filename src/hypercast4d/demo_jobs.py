"""Asynchronous demo jobs: durable reservations, private artifacts, bounded workers."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import uuid
from datetime import datetime, UTC

from . import demo_policy as policy

ARTIFACTS = {"summary.json", "runs.csv", "per_lead.csv", "predictions.csv", "weights.json", "learning_curves.csv"}


def now():
    return datetime.now(UTC).isoformat()


def csv_rows(data):
    rows = []
    for raw in csv.DictReader(io.StringIO(data.decode())):
        row = {}
        for key, value in raw.items():
            try:
                number = float(value)
                row[key] = (int(number) if number.is_integer() else number) if math.isfinite(number) else None
            except (TypeError, ValueError):
                row[key] = value or None
        rows.append(row)
    return rows


class DemoJobs:
    def __init__(self, store, gateway):
        self.store, self.gateway = store, gateway

    def submit(self, owner, payload):
        spec = policy.architecture(payload["architecture"])
        evaluation = policy.evaluation(payload.get("evaluation"))
        execution = policy.execution(payload.get("execution"))
        request = {"architecture": spec, "evaluation": evaluation, "execution": execution,
                   "phase": "validation", "demo": True}
        candidate_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        job_id = uuid.uuid4().hex
        request.update(job_id=job_id, candidate_hash=candidate_hash)
        record = {"id": job_id, "request": request, "summary": [], "runs": [], "per_lead": [],
                  "status": {"id": job_id, "state": "queued", "phase": "validation",
                    "created_at": now(), "updated_at": now(), "completed": 0, "total": 1,
                    "architecture_name": spec["name"], "candidate_hash": candidate_hash,
                    "preset": "quick", "protocol": "chronological-v1",
                    "execution_target": "modal", "gpu": "L4", "demo": True,
                    "error": None}}
        # Refresh finished global jobs before checking shared queue capacity.
        self.refresh(None)
        self.store.reserve_run(owner, record)
        try:
            call_id = self.gateway.spawn(request)
            self.store.update_job(job_id, record, call_id)
        except Exception:
            record["status"].update(state="failed", error="The demo worker could not be started. Please try later.")
            self.store.update_job(job_id, record)
        return record

    def refresh(self, owner):
        # One API process; serialize result materialization and cancel races.
        with self.store.lock:
            for row in self.store.jobs(owner, active_only=True):
                record, job_id = row["record"], row["id"]
                if not row["call_id"]:
                    # A crash between reservation and dispatch must not refund or rerun.
                    if self.store.clock() - row["created"] > 60:
                        record["status"].update(state="failed", error="Demo submission was interrupted.")
                        self.store.update_job(job_id, record)
                    continue
                if self.store.clock() - row["created"] > 3600:
                    self.gateway.cancel(row["call_id"])
                    record["status"].update(state="failed", error="This demo run expired in the queue.")
                    self.store.update_job(job_id, record)
                    continue
                result = self.gateway.poll(row["call_id"])
                if result is None:
                    continue
                files = result.get("files", {})
                if sum(len(value) for value in files.values()) > 16_000_000:
                    result = {"ok": False, "error": "Demo output exceeded the size limit."}
                    files = {}
                root = self.store.root / "artifacts" / job_id
                root.mkdir(parents=True, exist_ok=True)
                for name, data in files.items():
                    if name in ARTIFACTS:
                        (root / name).write_bytes(data)
                (root / "training.log").write_text(result.get("log", "")[-100_000:])
                if "summary.json" in files:
                    record["summary"] = json.loads(files["summary.json"])
                for key in ("runs", "per_lead"):
                    if key + ".csv" in files:
                        record[key] = csv_rows(files[key + ".csv"])
                complete = result.get("ok", False)
                record["status"].update(state="complete" if complete else "failed",
                    completed=1 if complete else 0, updated_at=now(), error=result.get("error"))
                self.store.update_job(job_id, record)

    def cancel(self, owner, job_id):
        with self.store.lock:
            row = self.store.job(owner, job_id)
            record = row["record"]
            if record["status"]["state"] in {"queued", "starting", "running"}:
                if row["call_id"]:
                    self.gateway.cancel(row["call_id"])
                record["status"].update(state="cancelled", updated_at=now())
                self.store.update_job(job_id, record)
            return record

    def artifact(self, owner, job_id, filename):
        self.store.job(owner, job_id)  # Ownership check before looking at any filesystem path.
        if filename not in ARTIFACTS | {"training.log"}:
            raise KeyError(filename)
        path = self.store.root / "artifacts" / job_id / filename
        if not path.is_file():
            raise KeyError(filename)
        return path


def train(request, *, require_cuda=True):
    """Worker-side checks too: callers cannot bypass policy by skipping the API."""
    from .modal_runner import _execute_payload
    spec = policy.architecture(request["architecture"])
    evaluation = policy.evaluation(request["evaluation"])
    execution = policy.execution(request["execution"])
    if request.get("phase") != "validation":
        raise ValueError("Held-out final tests are unavailable in the public demo.")
    cell = evaluation["cells"][0]
    policy.inspect("validate", {"architecture": spec, **cell})
    if not require_cuda:  # Local verification only; production entrypoint always requires CUDA.
        evaluation["device"] = "cpu"
    clean = {"job_id": request["job_id"], "candidate_hash": request["candidate_hash"],
             "phase": "validation", "architecture": spec, "evaluation": evaluation,
             "execution": execution, "demo": True}
    result = _execute_payload(json.dumps(clean), policy.dataset(), None, require_cuda=require_cuda)
    result["files"] = {key: value for key, value in result["files"].items() if key in ARTIFACTS}
    return result
