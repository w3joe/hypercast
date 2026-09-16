"""Single-L4 worker for the sealed-tail Crossformer baseline search."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import time
import traceback


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _partition(values, start, end, window, horizon=5):
    import numpy as np
    starts = np.arange(max(window, start), end - horizon + 1)
    return (np.stack([values[t-window:t] for t in starts]).astype("float32"),
            np.stack([values[t:t+horizon, 0] for t in starts]).astype("float32"), starts)


def train_trial(plan, data, job, directory, device="cuda"):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from hypercast4d.crossformer_baseline import build
    from hypercast4d.training import seed_everything

    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    bundle = np.load(io.BytesIO(data), allow_pickle=False)
    if set(bundle.files) != {"values"}:
        raise ValueError("Unexpected data bundle")
    values = bundle["values"]
    if len(values) != 14400:
        raise ValueError("Untouched tail must not be present")
    window = job["window"]
    train_x, train_y, _ = _partition(values, 0, 8640, window)
    inner_x, inner_y, _ = _partition(values, 8640, 11520, window)
    dev_x, dev_y, origins = _partition(values, 11520, 14400, window)
    train = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    inner = TensorDataset(torch.from_numpy(inner_x), torch.from_numpy(inner_y))
    model = build(job).to(device)
    seed_everything(job["seed"])
    generator = torch.Generator().manual_seed(job["seed"])
    train_loader = DataLoader(train, batch_size=32, shuffle=True, generator=generator)
    inner_loader = DataLoader(inner, batch_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=job["learning_rate"],
                                 betas=(.9, .999), eps=1e-7)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=.98 if job["schedule"] == "exponential" else 1.0)
    best = float("inf"); best_epoch = 0; best_state = None; stale = 0; history = []
    start = time.monotonic(); deadline = start + job["timeout_seconds"] - 60
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    for epoch in range(job["epochs"]):
        if stale >= job["patience"]:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Fit deadline")
        model.train(); total = 0.; count = 0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x.to(device)) - y.to(device)).abs().mean()
            if not bool(torch.isfinite(loss)): raise RuntimeError("Nonfinite loss")
            loss.backward(); optimizer.step(); total += loss.item() * len(x); count += len(x)
        model.eval(); val = 0.; items = 0
        with torch.inference_mode():
            for x, y in inner_loader:
                loss = (model(x.to(device)) - y.to(device)).abs().mean()
                val += loss.item() * len(x); items += len(x)
        value = val / items
        history.append(dict(epoch=epoch+1, learning_rate=optimizer.param_groups[0]["lr"],
                            train_mae=total/count, inner_mae=value))
        if value < best:
            best, best_epoch, stale = value, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        scheduler.step()
    if best_state is None: raise RuntimeError("No checkpoint")
    model.load_state_dict(best_state); model.eval()
    checkpoint = directory / "best.pt"
    torch.save(dict(state_dict=best_state, job=job, best_epoch=best_epoch,
                    best_inner_mae=best), checkpoint)
    restored = build(job).to(device); restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]); restored.eval()
    with torch.inference_mode():
        probe = torch.from_numpy(dev_x[:32]).to(device)
        torch.testing.assert_close(model(probe), restored(probe), atol=0, rtol=0)
        predictions = []
        for begin in range(0, len(dev_x), 64):
            predictions.append(restored(torch.from_numpy(dev_x[begin:begin+64]).to(device)).cpu().numpy())
    predicted = np.concatenate(predictions).astype(float)
    span = plan["scaler_span"][0]; offset = plan["scaler_minimum"][0]
    predicted = predicted * span + offset; actual = dev_y.astype(float) * span + offset
    anchor = np.repeat(dev_x[:, -1, 0:1], 5, axis=1) * span + offset
    seasonal = dev_x[:, np.arange(5) + window - 24, 0] * span + offset
    error = predicted - actual
    result = dict(job=job, best_epoch=best_epoch, epochs_ran=len(history),
                  patience_exhausted=stale >= job["patience"], best_inner_mae=best,
                  development_mae=float(np.abs(error).mean()),
                  development_rmse=float(np.sqrt(np.square(error).mean())),
                  development_bias=float(error.mean()),
                  persistence_mae=float(np.abs(anchor-actual).mean()),
                  seasonal_24_mae=float(np.abs(seasonal-actual).mean()),
                  beats_persistence=bool(np.abs(error).mean() < np.abs(anchor-actual).mean()),
                  parameters=sum(p.numel() for p in model.parameters()),
                  elapsed_seconds=time.monotonic()-start,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None,
                  checkpoint_replay_passed=True, checkpoint_sha256=_sha(checkpoint), history=history,
                  development_origins=[int(origins[0]), int(origins[-1])])
    np.savez_compressed(directory/"development_predictions.npz", prediction=predicted,
                        actual=actual, persistence=anchor, seasonal_24=seasonal,
                        target_start=origins)
    (directory/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def worker(plan, data, job):
    import torch
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp); start = time.monotonic()
        try:
            if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name(0):
                raise RuntimeError("L4 required")
            result = train_trial(plan, data, job, root/"trial", "cuda")
            status = dict(ok=True, result=result, gpu=torch.cuda.get_device_name(0))
        except Exception:
            status = dict(ok=False, error=traceback.format_exc())
        status["worker_elapsed_seconds"] = time.monotonic() - start
        (root/"status.json").write_text(json.dumps(status, indent=2, allow_nan=False))
        members = list(root.iterdir()); payload = root/"payload.tar.gz"
        with tarfile.open(payload, "w:gz") as archive:
            for path in members: archive.add(path, arcname=path.name)
        return status, payload.read_bytes()

