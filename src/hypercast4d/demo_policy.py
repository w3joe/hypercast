"""Public demo limits, independent of the unrestricted research application."""
from __future__ import annotations

import io
import json
import math
from functools import lru_cache
from typing import Any

LIMITS = {
    "run_seconds": 120,
    "epochs": 5,
    "runs_per_day": 3,
    "runs_per_month_total": 200,
    "queue_size": 10,
    # FiLM has ~12.6M parameters at window=60 even in its smallest exposed preset.
    "parameters": 15_000_000,
    "window": 60,
    "horizon": 20,
    "rows": 512,
    "saved_architectures": 20,
    "retention_days": 7,
}
DATA_PATH = "data/demo/synthetic.xlsx"
EXECUTION = {"target": "modal", "gpu": "L4", "timeout_seconds": LIMITS["run_seconds"]}
MAX_BODY = 512_000


def bounded_int(value: Any, label: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"Demo {label} must be an integer from {low} to {high}.")
    return value


def cell(window: Any = 10, horizon: Any = 1) -> tuple[int, int]:
    return (bounded_int(window, "window", 2, LIMITS["window"]),
            bounded_int(horizon, "horizon", 1, LIMITS["horizon"]))


def _json_bounds(value: Any, depth: int = 0) -> None:
    if depth > 24:
        raise ValueError("Demo configuration is nested too deeply.")
    if isinstance(value, dict):
        if len(value) > 2000:
            raise ValueError("Demo configuration is too large.")
        for item in value.values():
            _json_bounds(item, depth + 1)
    elif isinstance(value, list):
        if len(value) > 4000:
            raise ValueError("Demo configuration is too large.")
        for item in value:
            _json_bounds(item, depth + 1)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Demo values must be finite.")


def architecture(raw: dict) -> dict:
    """Cheap limits before tracing/allocating, followed by canonical validation.

    Actual construction is also confined to a timed, memory-limited worker.
    Shape checks alone cannot bound the cost of arbitrary branch wiring.
    """
    from .architecture import normalize_architecture_spec
    if not isinstance(raw, dict):
        raise ValueError("Architecture must be an object.")
    _json_bounds(raw)
    if len(json.dumps(raw, allow_nan=False)) > MAX_BODY:
        raise ValueError("Demo architecture is too large.")
    if raw.get("schema_version") == 2:
        sources = raw.get("sources", {})
        if not isinstance(sources, dict) or not 1 <= len(sources) <= 2:
            raise ValueError("Demo graphs support at most two starting models.")
        if len(raw.get("nodes", [])) > 1500 or len(raw.get("edges", [])) > 4000:
            raise ValueError("Demo graph exceeds the node or connection limit.")
        for source in sources.values():
            architecture(source)
        custom = [n for n in raw.get("nodes", []) if n.get("kind") != "source"]
        if len(custom) > 12:
            raise ValueError("Demo graphs support at most 12 added layers.")
        for node in custom:
            _params(node.get("params", {}))
    else:
        layers = raw.get("layers", [])
        if not isinstance(layers, list) or len(layers) > 12:
            raise ValueError("Demo architectures support at most 12 layers.")
        if sum(str(n.get("type", "")).startswith("tslib_") for n in layers) > 2:
            raise ValueError("Demo architectures support at most two TSLib models.")
        for layer in layers:
            _params(layer.get("params", {}))
            if layer.get("internal_overrides"):
                raise ValueError("For demo internal edits, convert to an editable graph first.")
    spec = normalize_architecture_spec(raw)
    # Apply to defaults as well as explicitly supplied parameters.
    for layer in spec.get("layers", []):
        _params(layer["params"])
    return spec


def _params(params: dict) -> None:
    if not isinstance(params, dict):
        raise ValueError("Layer parameters must be an object.")
    caps = {"units": 64, "channels": 64, "hidden_size": 64, "d_model": 64,
            "num_layers": 2, "n_heads": 8, "output_steps": 60,
            "dilation": 8, "kernel_size": 31, "patch_size": 32, "stride": 32}
    for key, cap in caps.items():
        if key in params:
            bounded_int(params[key], key, 1, cap)
    if "dilations" in params:
        values = params["dilations"]
        if not isinstance(values, list) or len(values) > 4:
            raise ValueError("Demo TCN supports at most four dilations.")
        for value in values:
            bounded_int(value, "dilation", 1, 8)
    if "shape" in params:
        shape = params["shape"]
        if not isinstance(shape, list) or not 2 <= len(shape) <= 4:
            raise ValueError("Demo reshape supports two to four axes.")
        for value in shape:
            bounded_int(value, "reshape dimension", -1, 4096)


def evaluation(raw: dict | None) -> dict:
    from .playground_runner import EVALUATION_DEFAULTS, normalize_evaluation
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError("Evaluation must be an object.")
    allowed = set(EVALUATION_DEFAULTS) | {"preset", "cells", "seeds", "epochs", "folds"}
    if set(raw) - allowed:
        raise ValueError("Research sweeps and custom evaluation controls are unavailable in the demo.")
    if raw.get("preset", "quick") != "quick":
        raise ValueError("The public demo supports Quick experiments only.")
    cells = raw.get("cells", [{"window": 10, "horizon": 1}])
    if not isinstance(cells, list) or len(cells) != 1:
        raise ValueError("Choose exactly one window/horizon combination for the demo.")
    window, horizon = cell(cells[0].get("window"), cells[0].get("horizon"))
    epochs = bounded_int(raw.get("epochs", 3), "epochs", 1, LIMITS["epochs"])
    seed = raw.get("seeds", [7])
    if not isinstance(seed, list) or len(seed) != 1:
        raise ValueError("Demo experiments support one seed.")
    bounded_int(seed[0], "seed", 0, 2**31 - 1)
    fixed = {"preset": "quick", "data_path": DATA_PATH, "target_column": "Signal",
             "batch_size": 32, "evaluation_batch_size": 64, "device": "cuda",
             "folds": [{"train_fraction": 0.70, "validation_fraction": 0.15}],
             "early_stopping_patience": 2, "restore_best_weights": True}
    for key, value in fixed.items():
        if key in raw and raw[key] != value:
            raise ValueError(f"Demo {key} is fixed; use the demo evaluation settings.")
    result = normalize_evaluation({**raw, **fixed, "epochs": epochs, "seeds": seed,
                                   "cells": [{"window": window, "horizon": horizon}]})
    _json_bounds(result)
    if not 0.00001 <= result["learning_rate"] <= 0.01:
        raise ValueError("Demo learning rate must be between 0.00001 and 0.01.")
    return result


def execution(raw: dict | None) -> dict:
    if raw is not None and (not isinstance(raw, dict) or any(
        key not in EXECUTION or value != EXECUTION[key] for key, value in raw.items()
    )):
        raise ValueError("The demo uses one L4 with a two-minute limit; compute overrides are unavailable.")
    return dict(EXECUTION)


@lru_cache(maxsize=1)
def dataset() -> bytes:
    """Original synthetic series; no private files or external dataset licences."""
    import numpy as np
    import pandas as pd
    t = np.arange(LIMITS["rows"])
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=len(t)),
        "Signal": np.sin(t / 9) + .3 * np.cos(t / 25) + .002 * t + rng.normal(0, .03, len(t)),
        "Season": np.sin((t + 2) / 9), "Trend": .002 * t,
        "Cycle": np.cos(t / 25)})
    output = io.BytesIO()
    frame.to_excel(output, index=False)
    return output.getvalue()


def inspect(operation: str, payload: dict) -> dict:
    """Only invoked in a CPU worker with a strict timeout and memory cap."""
    from .architecture import build_architecture, validate_architecture
    from .graph_architecture import convert_to_graph, describe_graph
    from .models import parameter_count
    spec = architecture(payload["architecture"])
    window, horizon = cell(payload.get("window", 10), payload.get("horizon", 1))
    if operation == "convert":
        result = convert_to_graph(spec, window, horizon)
        architecture(result)
        return result
    model = build_architecture(spec, window, horizon)
    if parameter_count(model) > LIMITS["parameters"]:
        raise ValueError(f"Demo models are limited to {LIMITS['parameters']:,} parameters. Reduce layer widths or layers.")
    if operation == "validate":
        return validate_architecture(spec, window, horizon)
    if operation == "describe":
        return describe_graph(spec, window, horizon)
    if operation == "weights":
        from .visualizations import model_weights
        return {"source": "initialized", **model_weights(model)}
    if operation == "internal-graph":
        from .internal_graph import architecture_internal_graph
        return architecture_internal_graph(spec, payload["layer_id"], window, horizon)
    if operation == "edit":
        from .model_editing import checked, edit_layer
        edit = payload["edit"]
        _params(edit.get("params", {}))
        cells = [{"window": window, "horizon": horizon}]
        result = edit_layer(spec, edit["action"], node_id=edit.get("id"), kind=edit.get("kind"),
                            params=edit.get("params", {}), before=edit.get("before"),
                            after=edit.get("after"), port=edit.get("port"), cells=cells)
        architecture(result)
        result, warnings = checked(result, cells, allow_invalid=bool(payload.get("allow_invalid", False)))
        return {"spec": result, "warnings": warnings}
    raise ValueError("Unsupported demo operation.")
