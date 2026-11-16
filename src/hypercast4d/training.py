"""Deterministic training and evaluation helpers."""

from __future__ import annotations

import copy
import platform
import random
import resource
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class FitResult:
    """Summary statistics returned upon completing a model training run.

    Parameters
    ----------
    epochs_ran : int
        Number of training epochs completed before stopping or reaching ``epochs``.
    best_validation_loss : float
        Lowest mean scaled validation loss observed across completed epochs.
    train_seconds : float
        Wall-clock training duration in seconds.
    process_peak_rss_mb : float
        OS-reported process peak resident set size in MiB after fitting.
    peak_gpu_allocated_mb : float or None, default=None
        Peak PyTorch CUDA allocator usage in MiB when ``device.type == "cuda"``.
    best_epoch : int, default=0
        1-based epoch index achieving ``best_validation_loss``.
    """

    epochs_ran: int
    best_validation_loss: float
    train_seconds: float
    process_peak_rss_mb: float
    peak_gpu_allocated_mb: float | None = None
    best_epoch: int = 0


def seed_everything(seed: int | None) -> None:
    """Seed Python ``random``, NumPy, and PyTorch CPU/CUDA RNGs when ``seed`` is given."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _peak_rss_mb() -> float:
    """Return process peak RSS in MiB (handling macOS byte vs. Linux KiB units)."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if platform.system() == "Darwin" else 1024.0
    return value / divisor


def fit_model(
    model: nn.Module,
    train_data: TensorDataset,
    validation_data: TensorDataset,
    *,
    seed: int | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
    adam_amsgrad: bool,
    loss_name: str,
    shuffle: bool,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    restore_best_weights: bool,
    device: torch.device,
    epoch_callback: Callable[[int, float, float], None] | None = None,
    early_stopping_relative_delta: float = 0.0,
) -> FitResult:
    """Train ``model`` with Adam, optional early stopping, and checkpoint tracking.

    Parameters
    ----------
    model : torch.nn.Module
        Forecasting network modified in place.
    train_data : torch.utils.data.TensorDataset
        Scaled ``(features, targets)`` training dataset.
    validation_data : torch.utils.data.TensorDataset
        Scaled ``(features, targets)`` validation dataset used for loss tracking
        and optional early stopping.
    seed : int or None
        Deterministic RNG and ``DataLoader`` shuffle seed.
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        Training and validation mini-batch size.
    learning_rate : float
        Adam step size.
    adam_beta1, adam_beta2 : float
        Adam exponential decay rates (``0.9`` and ``0.999`` in Keras/PyTorch).
    adam_epsilon : float
        Adam numerical stability term (``1e-7`` matches Keras 2.12).
    adam_amsgrad : bool
        Enable the AMSGrad variant of Adam.
    loss_name : {"mse", "mae", "huber"}
        Training and validation loss objective.
    shuffle : bool
        Shuffle training windows at each epoch using a seeded generator.
    early_stopping_patience : int or None
        Stop after this many consecutive non-improving epochs, or ``None`` to
        run all ``epochs``.
    early_stopping_min_delta : float
        Minimum absolute validation-loss decrease counted as an improvement.
    restore_best_weights : bool
        Reload the lowest-validation-loss state dictionary upon completion.
    device : torch.device
        Target execution device (``cpu``, ``mps``, or ``cuda``).
    epoch_callback : callable or None, default=None
        Optional hook invoked as ``epoch_callback(epoch_1based, train_loss, val_loss)``.
    early_stopping_relative_delta : float, default=0.0
        Minimum relative improvement fraction combined with ``early_stopping_min_delta``.

    Returns
    -------
    FitResult
        Timing, memory, best-epoch, and validation-loss summary.
    """
    seed_everything(seed)
    model.to(device)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
    validation_loader = DataLoader(validation_data, batch_size=batch_size)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
        amsgrad=adam_amsgrad,
    )
    if loss_name == "mse":
        criterion: nn.Module = nn.MSELoss()
    elif loss_name == "mae":
        criterion = nn.L1Loss()
    elif loss_name == "huber":
        criterion = nn.HuberLoss()
    else:
        raise ValueError("loss must be 'mse', 'mae', or 'huber'")
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    epochs_ran = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_train_items = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite training loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(features)
            total_train_items += len(features)

        model.eval()
        total_loss = 0.0
        total_items = 0
        with torch.no_grad():
            for features, targets in validation_loader:
                features, targets = features.to(device), targets.to(device)
                batch_loss = criterion(model(features), targets)
                total_loss += batch_loss.item() * len(features)
                total_items += len(features)
        validation_loss = total_loss / total_items
        if not np.isfinite(validation_loss):
            raise RuntimeError(f"Nonfinite validation loss at epoch {epoch + 1}")
        epochs_ran = epoch + 1
        if epoch_callback is not None:
            epoch_callback(
                epochs_ran,
                total_train_loss / total_train_items,
                validation_loss,
            )
        threshold = max(early_stopping_min_delta, abs(best_loss) * early_stopping_relative_delta) if np.isfinite(best_loss) else 0.0
        if validation_loss < best_loss - threshold:
            best_loss = validation_loss
            best_epoch = epochs_ran
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if (
                early_stopping_patience is not None
                and stale_epochs >= early_stopping_patience
            ):
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    if restore_best_weights:
        model.load_state_dict(best_state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return FitResult(
        epochs_ran=epochs_ran,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        train_seconds=time.perf_counter() - start,
        process_peak_rss_mb=_peak_rss_mb(),
        peak_gpu_allocated_mb=(torch.cuda.max_memory_allocated(device) / 1024**2
                               if device.type == "cuda" else None),
    )


def benchmark_inference(model: nn.Module, features: torch.Tensor, device: torch.device,
                        batch_size: int = 32, warmup: int = 10, repeats: int = 30) -> dict[str, Any]:
    """Measure resident-input forward latency in milliseconds per fixed-size batch.

    Parameters
    ----------
    model : torch.nn.Module
        Evaluated model placed on ``device``.
    features : torch.Tensor
        Feature pool containing at least ``batch_size`` examples.
    device : torch.device
        Execution device; CUDA streams are synchronized around timing boundaries.
    batch_size : int, default=32
        Fixed batch size sliced from ``features``.
    warmup : int, default=10
        Unmeasured warmup forward passes before starting the timer.
    repeats : int, default=30
        Measured forward passes averaged into ``inference_ms_per_batch``.

    Returns
    -------
    dict[str, Any]
        Dictionary with ``inference_batch_size``, ``inference_warmup``,
        ``inference_repeats``, and ``inference_ms_per_batch``.
    """
    if len(features) < batch_size:
        raise ValueError("Not enough examples for the fixed inference benchmark batch")
    inputs = features[:batch_size].to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.perf_counter() - start
    return {"inference_batch_size": batch_size, "inference_warmup": warmup,
            "inference_repeats": repeats, "inference_ms_per_batch": 1000 * duration / repeats}


def predict(
    model: nn.Module,
    dataset: TensorDataset,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Run batched ``eval()`` inference over ``dataset`` and return a NumPy array."""
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for features, _ in DataLoader(dataset, batch_size=batch_size):
            batches.append(model(features.to(device)).cpu().numpy())
    return np.concatenate(batches)


def error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute Mean Absolute Error (``mae``) and Mean Squared Error (``mse``).

    Examples
    --------
    >>> import numpy as np
    >>> from hypercast4d.training import error_metrics
    >>> error_metrics(np.array([2.0, 4.0]), np.array([1.0, 2.0]))
    {'mae': 1.5, 'mse': 2.5}
    """
    residual = prediction - target
    return {
        "mae": float(np.mean(np.abs(residual))),
        "mse": float(np.mean(np.square(residual))),
    }
