"""Check deployment boundaries without using a token or starting cloud workers."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load(relative):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deploy_script_refuses_wrong_workspace(monkeypatch):
    module = load("scripts/deploy_demo.py")
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.setattr(module.sys, "argv", ["deploy_demo.py", "--profile", "demo", "--workspace", "hypercast-demo"])
    calls = []
    def command(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="Workspace: research-workspace (id)")
    monkeypatch.setattr(module.subprocess, "run", command)
    with pytest.raises(SystemExit):
        module.main()
    assert len(calls) == 1
    assert "deploy" not in calls[0]


def test_deploy_script_requires_nonoverlapping_api_containers(monkeypatch):
    module = load("scripts/deploy_demo.py")
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.setattr(module.sys, "argv", ["deploy_demo.py", "--profile", "demo", "--workspace", "hypercast-demo"])
    calls = []
    def command(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="Workspace: hypercast-demo (id)")
    monkeypatch.setattr(module.subprocess, "run", command)
    module.main()
    assert calls[1][-2:] == ["--strategy", "recreate"]
    assert calls[1][calls[1].index("--profile") + 1] == "demo"


def test_gateway_distinguishes_poll_wait_from_training_timeout(monkeypatch):
    modal = pytest.importorskip("modal")
    module = load("deploy/modal_demo.py")
    error = [modal.exception.TimeoutError("not ready")]
    class Call:
        def get(self, **kwargs):
            raise error[0]
    monkeypatch.setattr(module.modal, "FunctionCall", SimpleNamespace(from_id=lambda _: Call()))
    gateway = module.ModalGateway()
    assert gateway.poll("call") is None
    error[0] = modal.exception.FunctionTimeoutError("worker expired")
    assert gateway.poll("call")["ok"] is False


def test_inspection_queue_timeout_cancels_unused_work(monkeypatch):
    modal = pytest.importorskip("modal")
    module = load("deploy/modal_demo.py")
    cancelled = []
    class Call:
        def get(self, timeout):
            assert timeout == 40
            raise modal.exception.TimeoutError("queue busy")
        def cancel(self, **kwargs):
            cancelled.append(kwargs)
    monkeypatch.setattr(module, "inspect_demo", SimpleNamespace(spawn=lambda *args: Call()))
    with pytest.raises(ValueError, match="queue is busy"):
        module.ModalGateway().inspect("validate", {})
    assert cancelled == [{"terminate_containers": True}]
