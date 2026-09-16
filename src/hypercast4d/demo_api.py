"""Public API with email-code sessions. The local research API stays private."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request as URLRequest, urlopen

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from . import demo_policy as policy
from .demo_jobs import DemoJobs
from .demo_store import DemoStore, LimitError

COOKIE = "__Host-hypercast-demo"


def send_code(email, code):
    """Send only the requested sign-in code, never marketing or account passwords."""
    body = {"from": os.environ["DEMO_EMAIL_FROM"], "to": [email],
            "subject": "Your Hypercast demo code",
            "text": f"Your Hypercast code is {code}. It expires in 10 minutes.\n\n"
                    "Use it to try the demo without creating an account. "
                    "If you did not request this code, ignore this email."}
    request = URLRequest("https://api.resend.com/emails", data=json.dumps(body).encode(),
                         headers={"Authorization": "Bearer " + os.environ["RESEND_API_KEY"],
                                  "Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError("Verification email could not be delivered.")


def create_demo_app(root: Path, gateway, *, origin: str, pepper: str,
                    mailer=send_code, persist=lambda: None):
    parsed = urlparse(origin)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("DEMO_PUBLIC_ORIGIN must be the exact HTTPS frontend origin.")
    origin = origin.rstrip("/")
    store = DemoStore(root, pepper, persist=persist)
    store.cleanup()
    last_cleanup = [store.clock()]
    jobs = DemoJobs(store, gateway)
    app = FastAPI(title="Hypercast Demo", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.jobs = store, jobs
    public = {"/api/v1/demo/session", "/api/v1/demo/request-code", "/api/v1/demo/verify"}

    @app.middleware("http")
    async def protect(request: Request, call_next):
        if request.method not in {"GET", "POST"}:
            return JSONResponse({"detail": "Method unavailable."}, status_code=405)
        if request.method == "POST":
            if request.headers.get("origin") != origin:
                return JSONResponse({"detail": "Request must originate from the demo website."}, status_code=403)
            if request.headers.get("content-type", "").split(";")[0] != "application/json":
                return JSONResponse({"detail": "JSON requests are required."}, status_code=415)
            chunks, size = [], 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > policy.MAX_BODY:
                    return JSONResponse({"detail": "Demo request is too large."}, status_code=413)
                chunks.append(chunk)
            request._body = b"".join(chunks)
        owner = store.owner(request.cookies.get(COOKIE))
        request.state.owner = owner
        if not owner and request.url.path not in public:
            return JSONResponse({"detail": "Verify your email to try the demo."}, status_code=401,
                                headers={"Cache-Control": "no-store"})
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(LimitError)
    async def limited(_, error):
        return JSONResponse({"detail": str(error)}, status_code=429)

    @app.exception_handler(ValueError)
    async def invalid(_, error):
        return JSONResponse({"detail": str(error)}, status_code=422)

    @app.exception_handler(KeyError)
    async def missing(_, error):
        return JSONResponse({"detail": "The requested demo item was not found."}, status_code=404)

    @app.get("/api/v1/demo/session")
    def session(request: Request):
        if store.clock() - last_cleanup[0] > 3600:
            store.cleanup()
            last_cleanup[0] = store.clock()
        owner = request.state.owner
        return {"authenticated": bool(owner), "limits": policy.LIMITS,
                **(store.quota(owner) if owner else {"remaining_runs": 0, "demo_available": True})}

    @app.post("/api/v1/demo/request-code")
    def request_code(payload: dict):
        challenge_id, code, email = store.challenge(str(payload.get("email", "")))
        try:
            mailer(email, code)
        except Exception as error:
            raise HTTPException(503, "Email delivery is unavailable. Please try again later.") from error
        return {"challenge_id": challenge_id}

    @app.post("/api/v1/demo/verify")
    def verify(payload: dict):
        token = store.verify(str(payload.get("challenge_id", "")), str(payload.get("code", "")))
        response = JSONResponse({"authenticated": True})
        response.set_cookie(COOKIE, token, max_age=7 * 86400, httponly=True, secure=True, samesite="lax", path="/")
        return response

    @app.post("/api/v1/demo/logout")
    def logout(request: Request):
        store.logout(request.cookies[COOKIE])
        response = JSONResponse({"authenticated": False})
        response.delete_cookie(COOKIE, secure=True, httponly=True, samesite="lax", path="/")
        return response

    @app.get("/api/v1/catalog")
    def catalog():
        from .architecture import layer_catalog, presets
        from .method_collection import method_collection
        demo_eval = policy.evaluation({})
        return {**layer_catalog(), "presets": presets(), "method_collection": method_collection(),
                "demo": policy.LIMITS,
                "evaluation_presets": {"quick": demo_eval}, "evaluation_defaults": demo_eval}

    @app.get("/api/v1/compute")
    def compute():
        return {"local": {"available": False}, "gcp": {"available": False},
                "modal": {"available": True, "sdk_installed": True, "authenticated": True,
                          "setup_command": "", "gpus": [{"id": "L4", "label": "L4", "description": "Demo worker"}]}}

    @app.post("/api/v1/architectures/{operation}")
    def inspect(operation: str, payload: dict, request: Request):
        if operation not in {"validate", "convert", "describe", "weights", "internal-graph", "edit"}:
            raise HTTPException(404, "Operation unavailable in the demo.")
        policy.architecture(payload["architecture"])
        if "cells" in payload:
            if not isinstance(payload["cells"], list) or len(payload["cells"]) != 1:
                raise ValueError("Demo edits support one evaluation cell.")
            payload = {**payload, **payload["cells"][0]}
        policy.cell(payload.get("window", 10), payload.get("horizon", 1))
        store.reserve_inspection(request.state.owner)
        return gateway.inspect(operation, payload)

    @app.get("/api/v1/architectures")
    def architectures(request: Request):
        return store.architectures(request.state.owner)

    @app.post("/api/v1/architectures")
    def save_architecture(payload: dict, request: Request):
        spec = policy.architecture(payload)
        return store.save_architecture(request.state.owner, spec, payload.get("view"), payload.get("id"))

    @app.get("/api/v1/architectures/{architecture_id}")
    def architecture(architecture_id: str, request: Request):
        for record in store.architectures(request.state.owner):
            if record["id"] == architecture_id:
                return record
        raise KeyError(architecture_id)

    @app.get("/api/v1/jobs")
    def list_jobs(request: Request):
        jobs.refresh(request.state.owner)
        return [row["record"] for row in store.jobs(request.state.owner)]

    @app.post("/api/v1/jobs")
    def submit(payload: dict, request: Request):
        return jobs.submit(request.state.owner, payload)

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str, request: Request):
        store.job(request.state.owner, job_id)
        jobs.refresh(request.state.owner)
        return store.job(request.state.owner, job_id)["record"]

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel(job_id: str, request: Request):
        return jobs.cancel(request.state.owner, job_id)

    @app.post("/api/v1/jobs/{job_id}/final-test")
    def final_test(job_id: str):
        raise HTTPException(403, "Final tests are unavailable in the public demo. Export your architecture for full training.")

    @app.get("/api/v1/jobs/{job_id}/log")
    def log(job_id: str, request: Request):
        store.job(request.state.owner, job_id)
        try:
            return PlainTextResponse(jobs.artifact(request.state.owner, job_id, "training.log").read_text())
        except KeyError:
            return PlainTextResponse("Waiting for the demo worker. Logs and results arrive when the run finishes.")

    @app.get("/api/v1/jobs/{job_id}/weights")
    def weights(job_id: str, request: Request):
        return json.loads(jobs.artifact(request.state.owner, job_id, "weights.json").read_bytes())

    @app.get("/api/v1/jobs/{job_id}/predictions.csv")
    def predictions(job_id: str, request: Request):
        return FileResponse(jobs.artifact(request.state.owner, job_id, "predictions.csv"),
                            filename="demo-predictions.csv", media_type="text/csv")

    @app.get("/api/v1/jobs/{job_id}/forecasts")
    def forecasts(job_id: str, request: Request, window: int, horizon: int, seed: int, fold: int, lead: int = 1):
        from .visualizations import read_forecasts
        policy.cell(window, horizon)
        policy.bounded_int(lead, "forecast lead", 1, horizon)
        return read_forecasts(jobs.artifact(request.state.owner, job_id, "predictions.csv"),
                             window=window, horizon=horizon, seed=seed, fold=fold, lead=lead)

    @app.get("/api/runs")
    @app.get("/api/v1/comparison-archives")
    def empty_research_archive():
        return []

    return app
