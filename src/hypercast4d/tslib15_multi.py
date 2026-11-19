"""Multi-dataset generalization of the 15-backbone × 8-arm TSLib comparison.

Preserves the exact architecture, single-site HyperDense/low-rank replacement
protocol, relative-residual head, IEEE FP32 deterministic cuDNN setup, and
paired initialization checks from :mod:`hypercast4d.tslib15` while supporting
arbitrary channel counts (``features``) and memory-efficient sliding windows
across all registered benchmarks (``ETTh2``, ``ETTm1``, ``ETTm2``, ``Weather``,
``Electricity``, ``Traffic``, ``Exchange``, ``ILI``, ``NYC Taxi``, and ``ETTh1``).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .data import load_dataset
from .datasets import get_dataset_spec
from .hyperdense_transfer import ARMS, GroupedSpectral, ieee_precision
from .hyperdense_transfer_pinned import CONSTANTS_SHA256, constants
from .remaining_controls import (
    ChannelMap,
    ContiguousHyperDense,
    digest_except,
    dimensions,
)
from .tslib15 import BACKBONES, SITES
from .upstream_models import UpstreamSequence, upstream_defaults


PROTOCOL = "tslib15-multi-v1-deterministic-cudnn"
NEW_BENCHMARK_DATASETS: tuple[str, ...] = (
    "etth2",
    "ettm1",
    "ettm2",
    "weather",
    "electricity",
    "traffic",
    "exchange",
    "ili",
    "nyc_taxi",
)
DEFAULT_CONTEXT = 32
DEFAULT_HORIZON = 5
DEFAULT_EVALUATION_SEEDS: tuple[int, ...] = (3301, 3302, 3303)


def stream_seed(dataset_id: str, seed: int, backbone: str, role: str) -> int:
    """Derive a deterministic 63-bit RNG seed for ``(dataset, seed, backbone, role)``."""
    key = f"{PROTOCOL}/{dataset_id}/{seed}/{backbone}/{role}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**63 - 1)


class TransferMulti(nn.Module):
    """Arbitrary-channel relative-residual wrapper for any of the 15 TSLib backbones."""

    def __init__(
        self,
        backbone: str,
        features: int,
        window: int = DEFAULT_CONTEXT,
        horizon: int = DEFAULT_HORIZON,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.features = features
        self.window = window
        self.horizon = horizon
        self.layers = nn.ModuleList(
            [
                UpstreamSequence(
                    backbone, window, features, **upstream_defaults(backbone)
                ),
                nn.Flatten(start_dim=1),
            ]
        )
        self.output = nn.Linear(window * features, horizon)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict ``horizon`` target steps from ``[batch, window, features]``."""
        if x.ndim != 3 or x.shape[1:] != (self.window, self.features):
            raise ValueError(
                f"Expected [batch, {self.window}, {self.features}], got {tuple(x.shape)}"
            )
        anchor = x[:, -1, 0:1]
        y = x - x[:, -1:, :]
        for layer in self.layers:
            y = layer(y)
        return self.output(y) + anchor


def build_model(
    backbone: str,
    arm: str,
    seed: int,
    *,
    features: int,
    dataset_id: str,
    window: int = DEFAULT_CONTEXT,
    horizon: int = DEFAULT_HORIZON,
) -> tuple[TransferMulti, dict[str, Any]]:
    """Build a paired ``(backbone, arm, seed)`` model and provenance report for ``features``."""
    if backbone not in SITES or arm not in ARMS:
        raise ValueError(f"Unknown backbone/arm: {backbone!r}/{arm!r}")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(
            stream_seed(dataset_id, seed, backbone, "native")
        )
        model = TransferMulti(backbone, features=features, window=window, horizon=horizon)
        if backbone == "film":
            buffers = dict(model.named_buffers())
            with torch.no_grad():
                for key, value in constants().items():
                    buffers[key].copy_(value)
        site = "layers.0.model." + SITES[backbone]
        old = model.get_submodule(site)
        untouched = digest_except(model, site)
        rank = None
        gap = None
        if backbone == "film":
            nin = nout = 512
            bias = False
            new: nn.Module = (
                old
                if arm == "native"
                else GroupedSpectral(
                    old, arm, stream_seed(dataset_id, seed, backbone, "spectral"), 1.0
                )
            )
            if arm.startswith("lowrank"):
                rank = 512 // (2 * int(arm[-1]))
                gap = 0
        else:
            nin, nout, bias = dimensions(old)
            if nin % 8 or nout % 8:
                raise ValueError(f"Indivisible site: {backbone} (C={features}): {nin}/{nout}")
            torch.random.default_generator.manual_seed(
                stream_seed(dataset_id, seed, backbone, "selected")
            )
            if arm == "native":
                new = old
            else:
                variance = 2.0 / (nin + nout)
                if arm == "real":
                    new = nn.Linear(nin, nout, bias=bias)
                elif arm.startswith("lowrank"):
                    dimension = int(arm[-1])
                    rank = max(1, nin * nout // (dimension * (nin + nout)))
                    gap = rank * (nin + nout) - nin * nout // dimension
                    new = nn.Sequential(
                        nn.Linear(nin, rank, bias=False),
                        nn.Linear(rank, nout, bias=bias),
                    )
                else:
                    dimension = {"complex": 2, "quaternion": 4, "octonion": 8}[arm]
                    new = ContiguousHyperDense(
                        nin // dimension, nout // dimension, arm, bias=bias
                    )
                with torch.no_grad():
                    factors = list(new) if rank else [new]
                    for idx, factor in enumerate(factors):
                        torch.random.default_generator.manual_seed(
                            stream_seed(dataset_id, seed, backbone, f"factor{idx}")
                        )
                        std = (variance / rank) ** 0.25 if rank else math.sqrt(variance)
                        factor.weight.normal_(0, std)
                        if factor.bias is not None:
                            factor.bias.zero_()
                if isinstance(old, nn.Conv1d):
                    new = ChannelMap(new)
        parent, _, key = site.rpartition(".")
        model.get_submodule(parent)._modules[key] = new
        assert digest_except(model, site) == untouched

    report = dict(
        protocol=PROTOCOL,
        dataset=dataset_id,
        backbone=backbone,
        arm=arm,
        seed=seed,
        features=features,
        window=window,
        horizon=horizon,
        site=site,
        untouched_sha256=untouched,
        parameters=sum(p.numel() for p in model.parameters()),
        selected_parameters=sum(p.numel() for p in new.parameters()),
        in_features=nin,
        out_features=nout,
        rank=rank,
        lowrank_weight_budget_gap=gap,
        native_is_complex=backbone == "film",
        constants_sha256=CONSTANTS_SHA256 if backbone == "film" else None,
    )
    return model, report


class SlidingWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Zero-copy sliding-window view over a scaled 2D ``[rows, features]`` matrix.

    Avoids materializing ``(num_windows, window, features)`` 3D copies in RAM,
    keeping memory bounded even on 321-channel Electricity and 862-channel Traffic.
    """

    def __init__(
        self,
        values: np.ndarray,
        starts: np.ndarray,
        *,
        window: int = DEFAULT_CONTEXT,
        horizon: int = DEFAULT_HORIZON,
        inputs_only: bool = False,
    ) -> None:
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.window = window
        self.horizon = horizon
        self.inputs_only = inputs_only

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> Any:
        t = int(self.starts[index])
        x = self.values[t - self.window : t]
        if self.inputs_only:
            return x
        y = self.values[t : t + self.horizon, 0]
        return x, y


def split_bounds(row_count: int) -> dict[str, tuple[int, int]]:
    """Return 45% / 15% / 20% / 20% chronological half-open row bounds matching ETTh1."""
    train_end = int(row_count * 0.45)
    inner_end = int(row_count * 0.60)
    dev_end = int(row_count * 0.80)
    return {
        "train": (0, train_end),
        "inner": (train_end, inner_end),
        "development": (inner_end, dev_end),
        "test": (dev_end, row_count),
    }


def target_starts(
    start: int,
    end: int,
    *,
    window: int = DEFAULT_CONTEXT,
    horizon: int = DEFAULT_HORIZON,
) -> np.ndarray:
    """Return valid 0-based lead-1 indices whose target horizon lies wholly in ``[start, end)``."""
    starts = np.arange(max(window, start), end - horizon + 1, dtype=np.int64)
    if len(starts) == 0:
        raise ValueError(f"Split [{start}, {end}) produced no complete windows for w={window}/h={horizon}")
    return starts


def prepare_dataset_bundle(
    dataset_id: str,
    project_root: Path | None = None,
    *,
    window: int = DEFAULT_CONTEXT,
    horizon: int = DEFAULT_HORIZON,
    include_test: bool = False,
    max_rows: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Scale ``dataset_id`` strictly on training rows and serialize a compact ``.npz`` bundle.

    When ``include_test=False``, rows at or after ``development_end`` are omitted
    from the serialized array so calibration/development workers never receive
    held-out test targets. When ``max_rows`` is set (for example ``14400`` to
    match ETTh1's chronological prefix), only the leading ``max_rows`` rows are
    partitioned.
    """
    spec = get_dataset_spec(dataset_id)
    root = (project_root or Path.cwd()).resolve()
    data_file = root / spec.data_path
    frame = load_dataset(data_file, spec.target_column)
    full_row_count = int(len(frame))
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.iloc[:max_rows]
    raw = frame.to_numpy(dtype=np.float64)
    row_count, features = raw.shape
    bounds = split_bounds(row_count)
    train_end = bounds["train"][1]
    dev_end = bounds["development"][1]

    minimum = raw[:train_end].min(axis=0)
    span = raw[:train_end].max(axis=0) - minimum
    span[span == 0.0] = 1.0

    visible_rows = row_count if include_test else dev_end
    scaled = ((raw[:visible_rows] - minimum) / span).astype(np.float32)

    arrays: dict[str, np.ndarray] = {"values": scaled}
    partitions: dict[str, dict[str, Any]] = {}
    for role, (a, b) in bounds.items():
        starts = target_starts(a, b, window=window, horizon=horizon)
        partitions[role] = {
            "rows": [a, b],
            "origins": int(len(starts)),
            "first_target_date": str(frame.index[int(starts[0])]),
            "last_target_date": str(frame.index[int(starts[-1]) + horizon - 1]),
        }
        if include_test or role != "test":
            arrays[f"{role}_starts"] = starts

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    blob = buffer.getvalue()

    manifest = {
        "protocol": PROTOCOL,
        "dataset": spec.id,
        "label": spec.label,
        "data_path": spec.data_path,
        "source_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
        "bundle_sha256": hashlib.sha256(blob).hexdigest(),
        "file_rows": full_row_count,
        "rows": int(row_count),
        "max_rows": max_rows,
        "features": int(features),
        "target": spec.target_column,
        "frequency": spec.frequency,
        "context": window,
        "horizon": horizon,
        "partitions": partitions,
        "scaler_rows": [0, train_end],
        "scaler_minimum": minimum.tolist(),
        "scaler_span": span.tolist(),
        "include_test": include_test,
    }
    return blob, manifest


def run_trial(
    manifest: dict[str, Any],
    bundle_bytes: bytes,
    job: dict[str, Any],
    work_dir: Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    """Execute one calibration, development, or test trial with replay verification."""
    from .hyperdense_transfer_training import finite, run as train_run
    from .representation_pilot import score, sha
    from .training import predict

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    arrays = np.load(io.BytesIO(bundle_bytes), allow_pickle=False)
    values = arrays["values"]
    phase = str(job["phase"])
    calibration = phase == "calibration"
    window = int(manifest["context"])
    horizon = int(manifest["horizon"])
    features = int(manifest["features"])

    train_ds = SlidingWindowDataset(
        values, arrays["train_starts"], window=window, horizon=horizon
    )
    inner_ds = SlidingWindowDataset(
        values,
        arrays["inner_starts"],
        window=window,
        horizon=horizon,
        inputs_only=calibration,
    )

    model, control = build_model(
        job["backbone"],
        job["arm"],
        int(job["seed"]),
        features=features,
        dataset_id=manifest["dataset"],
        window=window,
        horizon=horizon,
    )

    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    ceiling = int(job.get("epochs", 2 if calibration else 100))
    batch_size = int(job.get("batch_size", 32))
    start = time.monotonic()

    if batch_size == 32:
        result = train_run(
            model,
            train_ds,
            inner_ds if not calibration else torch.stack([inner_ds[i] for i in range(min(len(inner_ds), 256))]),
            work_dir,
            seed=int(job["seed"]),
            epochs=ceiling,
            learning_rate=float(job.get("learning_rate", 0.001)),
            calibration=calibration,
            device=device,
            metadata={"protocol": PROTOCOL, "job": job, "control": control, "dataset": manifest["dataset"]},
            deadline=start + max(1.0, float(job.get("timeout_seconds", 3600)) - 30.0),
        )
    else:
        raise ValueError("Frozen protocol uses batch_size=32")

    elapsed = time.monotonic() - start
    peak_gpu_bytes = (
        int(torch.cuda.max_memory_allocated())
        if str(device).startswith("cuda")
        else None
    )
    result.update(
        job=job,
        control=control,
        training_elapsed_seconds=elapsed,
        peak_gpu_bytes=peak_gpu_bytes,
    )

    ckpt_path = work_dir / ("latest.pt" if calibration else "best.pt")
    checkpoint = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    finite(checkpoint)
    restored, _ = build_model(
        job["backbone"],
        job["arm"],
        int(job["seed"]),
        features=features,
        dataset_id=manifest["dataset"],
        window=window,
        horizon=horizon,
    )
    restored.load_state_dict(checkpoint["state_dict"])
    restored.to(device).eval()
    model.eval()

    sample_count = min(32, len(arrays["inner_starts"]))
    sample = torch.stack(
        [
            torch.as_tensor(values[int(t) - window : int(t)], dtype=torch.float32)
            for t in arrays["inner_starts"][:sample_count]
        ]
    ).to(device)
    with ieee_precision(), torch.inference_mode():
        torch.testing.assert_close(restored(sample), model(sample), atol=0, rtol=0)
    result["checkpoint_replay_passed"] = True
    result["latest_checkpoint_sha256"] = sha(work_dir / "latest.pt")

    if not calibration:
        eval_starts = arrays[f"{phase}_starts"]
        eval_ds = SlidingWindowDataset(
            values, eval_starts, window=window, horizon=horizon
        )
        with ieee_precision():
            inner_pred = predict(restored, inner_ds, torch.device(device), 32)
            inner_y = np.stack(
                [values[int(t) : int(t) + horizon, 0] for t in arrays["inner_starts"]]
            )
            if not np.isclose(
                float(np.abs(inner_pred - inner_y).mean()),
                result["best_inner_mae"],
                rtol=2e-5,
                atol=1e-7,
            ):
                raise RuntimeError("Inner checkpoint score mismatch")
            predicted_scaled = predict(restored, eval_ds, torch.device(device), 32)

        span = float(manifest["scaler_span"][0])
        offset = float(manifest["scaler_minimum"][0])
        predicted = predicted_scaled.astype(np.float64) * span + offset
        actual = (
            np.stack([values[int(t) : int(t) + horizon, 0] for t in eval_starts]).astype(
                np.float64
            )
            * span
            + offset
        )
        origins_raw = values[eval_starts - 1, 0].astype(np.float64) * span + offset
        persistence = np.repeat(origins_raw[:, None], horizon, axis=1)
        if not np.isfinite(predicted).all():
            raise RuntimeError("Nonfinite predictions")

        np.savez_compressed(
            work_dir / "predictions.npz",
            prediction=predicted,
            actual=actual,
            persistence=persistence,
            target_start=eval_starts,
        )
        result.update(
            **score(predicted, actual, persistence),
            rmse=float(np.sqrt(np.mean((predicted - actual) ** 2))),
            best_checkpoint_sha256=sha(work_dir / "best.pt"),
            test_scored=phase == "test",
        )
        if phase == "test":
            timings: dict[str, Any] = {}
            with ieee_precision(), torch.inference_mode():
                for batch in (1, 32):
                    if len(eval_starts) < batch:
                        continue
                    z = torch.stack(
                        [
                            torch.as_tensor(
                                values[int(t) - window : int(t)], dtype=torch.float32
                            )
                            for t in eval_starts[:batch]
                        ]
                    ).to(device)
                    for _ in range(10):
                        restored(z)
                    repeats = []
                    for _ in range(5):
                        if str(device).startswith("cuda"):
                            torch.cuda.synchronize()
                        lap = time.perf_counter()
                        for _ in range(20):
                            restored(z)
                        if str(device).startswith("cuda"):
                            torch.cuda.synchronize()
                        repeats.append((time.perf_counter() - lap) * 1000.0 / 20.0)
                    timings[str(batch)] = {
                        "median_ms": float(np.median(repeats)),
                        "repeat_ms": repeats,
                    }
            result["inference"] = timings

    result["ceiling_reached"] = (not calibration) and (result["epochs_ran"] == ceiling)
    result["epoch_ceiling"] = ceiling
    (work_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result
