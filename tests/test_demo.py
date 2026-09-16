import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hypercast4d import demo_policy as policy
from hypercast4d.architecture import presets
from hypercast4d.demo_api import COOKIE, create_demo_app
from hypercast4d.demo_jobs import train
from hypercast4d.demo_store import DemoStore, LimitError

ORIGIN = "https://hypercast.example"
PEPPER = "testing-only-secret-" * 3


def preset(name="tslib-tsmixer"):
    return next(p for p in presets() if p["preset_id"] == name)


class Gateway:
    def __init__(self):
        self.requests = []
        self.results = {}
        self.cancelled = []

    def spawn(self, request):
        self.requests.append(copy.deepcopy(request))
        return request["job_id"]

    def poll(self, call_id):
        return self.results.get(call_id)

    def cancel(self, call_id):
        self.cancelled.append(call_id)

    def inspect(self, operation, payload):
        return policy.inspect(operation, payload)


@pytest.fixture
def demo(tmp_path):
    gateway, sent = Gateway(), []
    app = create_demo_app(tmp_path, gateway, origin=ORIGIN, pepper=PEPPER,
                          mailer=lambda email, code: sent.append((email, code)))
    with TestClient(app, base_url=ORIGIN, headers={"Origin": ORIGIN}) as client:
        yield client, app, gateway, sent


def verify(client, sent, email="tester@example.com"):
    response = client.post("/api/v1/demo/request-code", json={"email": email})
    assert response.status_code == 200, response.text
    challenge = response.json()["challenge_id"]
    response = client.post("/api/v1/demo/verify", json={"challenge_id": challenge, "code": sent[-1][1]})
    assert response.status_code == 200, response.text
    return challenge, response


def submit(client, **overrides):
    return client.post("/api/v1/jobs", json={"architecture": preset(),
        "evaluation": {"epochs": 1}, "execution": {"target": "modal", "gpu": "L4"}, **overrides})


def test_email_code_session_is_single_use_secure_and_revocable(demo):
    client, app, gateway, sent = demo
    assert client.get("/api/v1/demo/session").json()["authenticated"] is False
    for path in ("/api/v1/catalog", "/api/v1/jobs", "/api/v1/architectures", "/api/runs", "/api/v1/identity"):
        assert client.get(path).status_code == 401
    challenge, response = verify(client, sent)
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie and "Secure" in cookie and "SameSite=lax" in cookie
    assert client.get("/api/v1/demo/session").json()["remaining_runs"] == 3
    assert client.post("/api/v1/demo/verify", json={"challenge_id": challenge, "code": sent[-1][1]}).status_code == 422
    old_cookie = client.cookies.get(COOKIE)
    assert client.post("/api/v1/demo/logout", json={}).status_code == 200
    client.cookies.set(COOKIE, old_cookie)
    assert client.get("/api/v1/jobs").status_code == 401


def test_wrong_code_attempts_survive_restart_and_email_is_not_stored(tmp_path):
    store = DemoStore(tmp_path, PEPPER)
    challenge, code, _ = store.challenge("Private.Address@example.com")
    for _ in range(5):
        with pytest.raises(ValueError, match="invalid or expired"):
            store.verify(challenge, "wrong")
    restored = DemoStore(tmp_path, PEPPER)
    with pytest.raises(ValueError):
        restored.verify(challenge, code)
    assert b"private.address@example.com" not in (tmp_path / "demo.sqlite").read_bytes()
    assert code.encode() not in (tmp_path / "demo.sqlite").read_bytes()


def test_expired_code_and_session(tmp_path):
    clock = [1_000_000.0]
    store = DemoStore(tmp_path, PEPPER, clock=lambda: clock[0])
    challenge, code, _ = store.challenge("a@example.com")
    clock[0] += 601
    with pytest.raises(ValueError):
        store.verify(challenge, code)
    challenge, code, _ = store.challenge("b@example.com")
    token = store.verify(challenge, code)
    assert store.owner(token)
    clock[0] += 7 * 86400 + 1
    assert store.owner(token) is None


def test_email_throttle_and_send_failure_do_not_grant_session(demo):
    client, app, gateway, sent = demo
    for _ in range(3):
        assert client.post("/api/v1/demo/request-code", json={"email": "a@example.com"}).status_code == 200
    assert client.post("/api/v1/demo/request-code", json={"email": "A@example.com"}).status_code == 429
    assert len(sent) == 3
    assert not client.get("/api/v1/demo/session").json()["authenticated"]


def test_delivery_failure_fails_closed(tmp_path):
    def fail(email, code):
        raise RuntimeError("provider unavailable")
    app = create_demo_app(tmp_path, Gateway(), origin=ORIGIN, pepper=PEPPER, mailer=fail)
    with TestClient(app, base_url=ORIGIN, headers={"Origin": ORIGIN}) as client:
        response = client.post("/api/v1/demo/request-code", json={"email": "a@example.com"})
        assert response.status_code == 503
        assert "challenge_id" not in response.text
        assert not client.get("/api/v1/demo/session").json()["authenticated"]


def test_nonfinite_csv_metrics_are_json_safe():
    from hypercast4d.demo_jobs import csv_rows
    rows = csv_rows(b"mae,mse,parameters\nnan,inf,3\n")
    assert rows == [{"mae": None, "mse": None, "parameters": 3}]
    json.dumps(rows, allow_nan=False)


def test_csrf_content_type_and_stream_size_guards(demo):
    client, _, _, _ = demo
    assert client.post("/api/v1/demo/request-code", json={"email": "a@example.com"}, headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/v1/demo/request-code", content="email=a@example.com").status_code == 415
    assert client.post("/api/v1/demo/request-code", content=b"x" * (policy.MAX_BODY + 1), headers={"Content-Type": "application/json"}).status_code == 413


@pytest.mark.parametrize("evaluation", [
    {"epochs": 6}, {"epochs": True}, {"cells": [{"window": 61, "horizon": 1}]},
    {"cells": [{"window": 10, "horizon": 1}, {"window": 20, "horizon": 1}]},
    {"seeds": [7, 9]}, {"preset": "robust"}, {"remaining_tuning": True},
    {"remaining_replication": {}}, {"data_path": "data/raw/paper_data.xlsx"},
    {"folds": [{"train_fraction": .5, "validation_fraction": .3}]},
    {"batch_size": 1024}, {"evaluation_batch_size": 4096}, {"device": "cpu"},
    {"learning_rate": float("inf")},
])
def test_evaluation_policy_rejects_cost_or_dataset_overrides(evaluation):
    with pytest.raises(ValueError):
        policy.evaluation(evaluation)


@pytest.mark.parametrize("execution", [{"target": "local"}, {"gpu": "H100"}, {"gpu_count": 8}, {"timeout_seconds": 86400}, {"target": "gcp"}])
def test_execution_policy_cannot_be_overridden(execution):
    with pytest.raises(ValueError):
        policy.execution(execution)


def test_jobs_are_private_and_cancelled_runs_consume_daily_quota(demo):
    client, app, gateway, sent = demo
    verify(client, sent)
    first = submit(client)
    assert first.status_code == 200, first.text
    job_id = first.json()["id"]
    assert submit(client).status_code == 429
    assert client.post(f"/api/v1/jobs/{job_id}/cancel", json={}).status_code == 200
    assert gateway.cancelled == [job_id]
    for _ in range(2):
        job_id = submit(client).json()["id"]
        client.post(f"/api/v1/jobs/{job_id}/cancel", json={})
    assert submit(client).status_code == 429
    assert client.get("/api/v1/demo/session").json()["remaining_runs"] == 0
    assert client.post(f"/api/v1/jobs/{job_id}/final-test", json={}).status_code == 403
    saved = client.post("/api/v1/architectures", json=preset()).json()
    client.cookies.clear()
    verify(client, sent, "other@example.com")
    assert client.get("/api/v1/jobs").json() == []
    assert client.get("/api/v1/architectures").json() == []
    for suffix in ("", "/log", "/weights", "/predictions.csv"):
        assert client.get(f"/api/v1/jobs/{job_id}{suffix}").status_code == 404
    assert client.post(f"/api/v1/jobs/{job_id}/cancel", json={}).status_code == 404
    assert client.get(f"/api/v1/architectures/{saved['id']}").status_code == 404
    assert client.post("/api/v1/architectures", json={**preset(), "id": saved["id"]}).status_code == 404


def test_atomic_concurrent_reservations_and_monthly_cap(tmp_path):
    store = DemoStore(tmp_path, PEPPER)
    def reserve(i):
        try:
            store.reserve_run("owner", {"id": str(i), "status": {"state": "queued"}})
            return True
        except LimitError:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(reserve, range(8))) == 1
    with store.transaction() as db:
        db.executemany("INSERT INTO events VALUES ('run', 'someone', ?)", [(store.clock(),)] * 199)
    with pytest.raises(LimitError, match="shared demo allowance"):
        store.reserve_run("new-owner", {"id": "extra", "status": {"state": "queued"}})
    assert len(store.jobs(None)) == 1


def test_quota_survives_restart_and_resets_next_day(tmp_path):
    clock = [1_000_000.0]
    store = DemoStore(tmp_path, PEPPER, clock=lambda: clock[0])
    for i in range(3):
        store.reserve_run("owner", {"id": str(i), "status": {"state": "complete"}})
    assert DemoStore(tmp_path, PEPPER, clock=lambda: clock[0]).quota("owner")["remaining_runs"] == 0
    clock[0] += 86400
    assert store.quota("owner")["remaining_runs"] == 3


@pytest.mark.parametrize("name", [p["preset_id"] for p in presets() if p["preset_id"].startswith("tslib-")])
@pytest.mark.parametrize("window,horizon", [(10, 1), (60, 20)])
def test_all_15_tslib_templates_fit_demo_policy(name, window, horizon):
    spec = policy.inspect("convert", {"architecture": preset(name), "window": window, "horizon": horizon})
    result = policy.inspect("validate", {"architecture": spec, "window": window, "horizon": horizon})
    assert result["parameters"] <= policy.LIMITS["parameters"]


def test_real_demo_training_materializes_private_results(demo):
    client, app, gateway, sent = demo
    verify(client, sent)
    spec = policy.inspect("convert", {"architecture": preset()})
    # A shape-preserving HyperDense inserted directly before the residual join.
    from hypercast4d.model_editing import edit_layer
    spec = edit_layer(spec, "insert", node_id="demo-hyper", kind="hyper_dense", params={"shape_mode": "preserve", "algebra": "quaternion"},
                      before="add_1", port="args/0", cells=[{"window": 10, "horizon": 1}])
    response = submit(client, architecture=spec)
    assert response.status_code == 200, response.text
    request = gateway.requests[0]
    result = train(request, require_cuda=False)
    assert result["ok"], result["log"]
    gateway.results[request["job_id"]] = result
    jobs = client.get("/api/v1/jobs").json()
    assert jobs[0]["status"]["state"] == "complete"
    assert jobs[0]["summary"] and jobs[0]["runs"]
    assert client.get(f"/api/v1/jobs/{request['job_id']}/predictions.csv").status_code == 200
    assert client.get(f"/api/v1/jobs/{request['job_id']}/weights").status_code == 200


def test_worker_rechecks_policy_before_training():
    with pytest.raises(ValueError, match="epochs"):
        train({"architecture": preset(), "evaluation": {"epochs": 500}, "execution": policy.EXECUTION})


def test_oversized_models_and_expired_artifacts(tmp_path):
    spec = preset()
    spec["layers"][0]["params"]["d_model"] = 256
    with pytest.raises(ValueError, match="d_model"):
        policy.architecture(spec)
    clock = [1_000_000.0]
    store = DemoStore(tmp_path, PEPPER, clock=lambda: clock[0])
    store.reserve_run("owner", {"id": "job", "status": {"state": "complete"}})
    path = tmp_path / "artifacts/job"
    path.mkdir(parents=True)
    (path / "training.log").write_text("old results")
    clock[0] += 8 * 86400
    store.cleanup()
    assert not path.exists()
    assert store.jobs("owner") == []
