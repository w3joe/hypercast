"""Bounded, exact-value visualization artifacts for weights and forecasts.

All snapshots sample exact parameter or prediction entries without synthetic
interpolation, bounding payload sizes so browser inspectors remain responsive
even on wide models or long evaluation series.
"""

from __future__ import annotations

import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import torch

from .layers import HyperDense, ShapePreservingHyperDense


def weight_snapshot(layer: HyperDense, name: str) -> dict[str, Any]:
    """Extract component kernels and the expanded real block matrix for display.

    Samples up to 8 evenly spaced hypercomplex input and output unit indices
    from ``layer`` and materializes the exact ``(d * len(ins), d * len(outs))``
    effective real matrix induced by ``x @ W`` in component-major order.

    Parameters
    ----------
    layer : HyperDense
        Hypercomplex dense module to inspect.
    name : str
        Dotted submodule path identifying ``layer`` inside its parent model.

    Returns
    -------
    dict[str, Any]
        Serializable dictionary containing ``algebra``, ``dimension``,
        ``in_features``, ``out_features``, sampled ``input_indices`` and
        ``output_indices``, ``components`` (shape ``[d, n_s, m_s]``),
        ``effective`` real block matrix, ``weight_parameters``, and equivalent
        unconstrained ``dense_parameters``; or ``{"name": name, "error": ...}``
        if non-finite weights are encountered.
    """
    weight = layer.weight.detach()
    ins = torch.linspace(0, layer.in_features - 1, min(8, layer.in_features)).long()
    outs = torch.linspace(0, layer.out_features - 1, min(8, layer.out_features)).long()
    components = weight[:, ins.to(weight.device)][:, :, outs.to(weight.device)].cpu()
    constants = layer.algebra.constants.to(components)
    # Materialize the exact block-structured real matrix corresponding to x @ W.
    effective = torch.einsum("bac,aio->bico", constants, components).reshape(
        layer.component_count * len(ins), layer.component_count * len(outs)
    )
    if not torch.isfinite(components).all() or not torch.isfinite(effective).all():
        return {"name": name, "error": "Weights contain non-finite values"}
    return dict(
        name=name,
        algebra=layer.algebra.name,
        dimension=layer.component_count,
        in_features=layer.in_features,
        out_features=layer.out_features,
        input_indices=ins.tolist(),
        output_indices=outs.tolist(),
        components=components.tolist(),
        effective=effective.tolist(),
        weight_parameters=weight.numel(),
        dense_parameters=layer.component_count * weight.numel(),
    )


def model_weights(model: torch.nn.Module) -> dict[str, Any]:
    """Collect weight snapshots for up to 32 ``HyperDense`` submodules.

    Parameters
    ----------
    model : torch.nn.Module
        Compiled pipeline or graph forecaster. If ``model`` exposes a
        ``bindings`` attribute (as on ``GraphForecaster``), each snapshot is
        annotated with its corresponding graph ``node_ids`` and any parent
        ``ShapePreservingHyperDense`` ``shape_fit`` record.

    Returns
    -------
    dict[str, Any]
        Dictionary with ``"layers"`` (list of snapshots) and
        ``"omitted_layers"`` (count of additional ``HyperDense`` modules beyond
        the 32-layer cap).
    """
    layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, HyperDense)
    ]
    snapshots = [weight_snapshot(module, name) for name, module in layers[:32]]
    for snapshot in snapshots:
        parent_name = (
            snapshot["name"].rsplit(".", 1)[0] if "." in snapshot["name"] else ""
        )
        parent = model.get_submodule(parent_name)
        if isinstance(parent, ShapePreservingHyperDense):
            snapshot["shape_fit"] = parent.shape_fit
        snapshot["node_ids"] = [
            node
            for node, binding in getattr(model, "bindings", {}).items()
            if snapshot["name"] == f"blocks.{binding}"
            or snapshot["name"].startswith(f"blocks.{binding}.")
        ]
    return {"layers": snapshots, "omitted_layers": max(0, len(layers) - 32)}


def save_weights(
    destination: Path, model: torch.nn.Module, **identity: Any
) -> None:
    """Atomically record trained ``HyperDense`` snapshots in ``weights.json``.

    Parameters
    ----------
    destination : Path
        Job output directory containing (or receiving) ``weights.json``.
    model : torch.nn.Module
        Trained model whose ``HyperDense`` layers should be snapshotted.
    **identity : Any
        Run coordinates—must include ``window``, ``horizon``, ``seed``, and
        ``fold`` (plus optional metadata such as ``split``).
    """
    path = destination / "weights.json"
    records = json.loads(path.read_text()) if path.exists() else {}
    key = "/".join(str(identity[k]) for k in ("window", "horizon", "seed", "fold"))
    records[key] = {**identity, "source": "trained", **model_weights(model)}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(records, allow_nan=False))
    temporary.replace(path)


def read_forecasts(
    path: Path,
    *,
    window: int,
    horizon: int,
    seed: int,
    fold: int,
    lead: int,
    trial: str = "",
) -> dict[str, Any]:
    """Read up to the latest 2,000 individual forecast rows for one run and lead.

    Parameters
    ----------
    path : Path
        Path to a ``predictions.csv`` file generated during evaluation.
    window : int
        Historical lookback window length to filter on.
    horizon : int
        Total forecast horizon length to filter on.
    seed : int
        Random seed identifier to filter on.
    fold : int
        Validation fold index (or ``0`` for final-test runs).
    lead : int
        1-based forecast step ahead (``1 <= lead <= horizon``).
    trial : str, default=""
        Optional batch-study trial identifier.

    Returns
    -------
    dict[str, Any]
        Dictionary with ``"rows"`` (chronologically sorted forecast records),
        ``"total"`` valid matching rows, ``"invalid"`` skipped rows, and
        ``"limit"`` (``2000``).
    """
    rows: deque[dict[str, Any]] = deque(maxlen=2000)
    total = invalid = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            try:
                if (raw.get("trial_id") or "") != trial:
                    continue
                if any(
                    float(raw[k]) != value
                    for k, value in dict(
                        window=window, horizon=horizon, seed=seed, fold=fold, lead=lead
                    ).items()
                ):
                    continue
                row: dict[str, Any] = {
                    k: float(raw[k])
                    for k in ("origin_row", "actual", "prediction", "persistence")
                }
                if not all(math.isfinite(v) for v in row.values()):
                    invalid += 1
                    continue
                rows.append(
                    {
                        **row,
                        "target_date": raw["target_date"],
                        "forecast_origin": raw["forecast_origin"],
                    }
                )
                total += 1
            except (KeyError, TypeError, ValueError):
                invalid += 1
    return {
        "rows": sorted(rows, key=lambda r: r["origin_row"]),
        "total": total,
        "invalid": invalid,
        "limit": 2000,
    }
