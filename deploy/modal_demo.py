"""Deploy ONLY to the dedicated demo workspace; see docs/deployment.md."""
import os
from pathlib import Path

import modal

app = modal.App("hypercast-demo")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", "numpy==2.5.3", "pandas==3.0.5", "scipy==1.18.1",
                 "einops==0.8.2", "openpyxl==3.1.5", "fastapi==0.141.1",
                 "PyYAML>=6,<7", "scikit-learn>=1.3,<2", "tensorboard>=2.16,<2.21")
    .env({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONPATH": "/opt/hypercast"})
    .add_local_dir(Path(__file__).resolve().parents[1] / "src" / "hypercast4d",
                   remote_path="/opt/hypercast/hypercast4d",
                   ignore=["**/__pycache__/**", "**/web_dist/**"])
)
volume = modal.Volume.from_name("hypercast-demo-state", create_if_missing=True)


@app.function(image=image, gpu="L4", cpu=(2, 2), memory=(4096, 4096),
              timeout=120, startup_timeout=60, retries=0,
              max_containers=1, min_containers=0, buffer_containers=0, scaledown_window=2)
def train_demo(request):
    from hypercast4d.demo_jobs import train
    return train(request)


@app.function(image=image, cpu=(1, 2), memory=(2048, 4096), timeout=20,
              startup_timeout=60, retries=0, max_containers=1,
              min_containers=0, buffer_containers=0, scaledown_window=2)
def inspect_demo(operation, payload):
    from hypercast4d.demo_policy import inspect
    try:
        return {"ok": True, "result": inspect(operation, payload)}
    except (ValueError, TypeError, KeyError, RuntimeError) as error:
        return {"ok": False, "error": str(error)[:1000]}


class ModalGateway:
    def spawn(self, request):
        return train_demo.spawn(request).object_id

    def inspect(self, operation, payload):
        call = inspect_demo.spawn(operation, payload)
        try:
            result = call.get(timeout=40)
        except modal.exception.FunctionTimeoutError as error:
            raise ValueError("This architecture exceeded the demo's 20-second checking limit.") from error
        except modal.exception.TimeoutError as error:
            call.cancel(terminate_containers=True)
            raise ValueError("The demo checking queue is busy. Please try again shortly.") from error
        if not result["ok"]:
            raise ValueError(result["error"])
        return result["result"]

    def poll(self, call_id):
        try:
            return modal.FunctionCall.from_id(call_id).get(timeout=0)
        except modal.exception.FunctionTimeoutError:
            return {"ok": False, "error": "The two-minute demo limit was reached. Try a smaller model or fewer epochs."}
        except modal.exception.TimeoutError:
            return None  # Still queued/running; this is the poll deadline, not the worker deadline.
        except (modal.exception.ExecutionError, modal.exception.NotFoundError):
            return {"ok": False, "error": "The demo worker stopped or its result expired. Try a smaller model."}

    def cancel(self, call_id):
        modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)


@app.function(image=image, cpu=(0.25, 1), memory=(2048, 4096), timeout=60,
              max_containers=1, min_containers=0, scaledown_window=30,
              volumes={"/state": volume}, secrets=[modal.Secret.from_name("hypercast-demo-config")])
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def api():
    from hypercast4d.demo_api import create_demo_app
    # Missing configuration must prevent startup, never expose the research API.
    for key in ("DEMO_PUBLIC_ORIGIN", "DEMO_SESSION_SECRET", "DEMO_EMAIL_FROM", "RESEND_API_KEY"):
        if not os.environ.get(key):
            raise RuntimeError(f"Missing demo configuration: {key}")
    return create_demo_app(Path("/state"), ModalGateway(), origin=os.environ["DEMO_PUBLIC_ORIGIN"],
                           pepper=os.environ["DEMO_SESSION_SECRET"], persist=volume.commit)
