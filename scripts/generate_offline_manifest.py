#!/usr/bin/env python3
"""Generate the checked-in data used by the browser-only playground."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hypercast4d.architecture import layer_catalog, presets  # noqa: E402
from hypercast4d.graph_architecture import convert_to_graph, validate_graph  # noqa: E402
from hypercast4d.method_collection import method_collection  # noqa: E402
from hypercast4d.playground_runner import (  # noqa: E402
    EVALUATION_DEFAULTS,
    EVALUATION_PRESETS,
)


REFERENCE_WINDOW = 10
REFERENCE_HORIZON = 1


def _stable_metadata(value: Any) -> Any:
    """Remove process-specific addresses from callable descriptions."""
    if isinstance(value, dict):
        return {key: _stable_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable_metadata(item) for item in value]
    if isinstance(value, str):
        return re.sub(r" at 0x[0-9a-fA-F]+(?=>)", "", value)
    return value


def _auto_fit_connections(graph: dict[str, Any], metadata: dict[str, Any], dimensions: dict[str, int]) -> list[dict[str, Any]]:
    """Precalculate shape-preserving HyperDense fits at tensor connections."""
    fits: list[dict[str, Any]] = []
    for edge in graph["edges"]:
        shape = metadata.get(edge["source"], {}).get("shape")
        if not isinstance(shape, list) or len(shape) < 2 or not shape or not all(isinstance(item, int) for item in shape):
            continue
        width = shape[-1]
        algebras = {}
        for algebra, dimension in dimensions.items():
            units = (width + dimension - 1) // dimension
            padded = units * dimension
            algebras[algebra] = {
                "axis": -1,
                "input_width": width,
                "padded_width": padded,
                "units": units,
                "output_width": width,
                "padding": padded - width,
                "crop": padded - width,
            }
        fits.append({**edge, "input_shape": shape, "algebras": algebras})
    return fits


def build_manifest() -> dict[str, Any]:
    """Return deterministic catalog and reference metadata for every preset."""
    source_presets = presets()
    entries: dict[str, Any] = {}
    catalog_layers = layer_catalog()
    dimensions = catalog_layers["algebra_dimensions"]
    for preset in source_presets:
        torch.manual_seed(0)
        preset_id = preset.get("preset_id")
        if not preset_id:
            raise ValueError(f"Preset has no preset_id: {preset.get('name', 'unnamed')}")
        graph = convert_to_graph(preset, REFERENCE_WINDOW, REFERENCE_HORIZON)
        # Preserve the catalog identity because graph normalization intentionally
        # strips fields which are unrelated to model execution.
        graph["preset_id"] = preset_id
        graph["locked"] = bool(preset.get("locked", False))
        reference = validate_graph(graph, REFERENCE_WINDOW, REFERENCE_HORIZON)
        stable_nodes = _stable_metadata(reference["graph_nodes"])
        entries[preset_id] = {
            "graph": graph,
            "graph_nodes": stable_nodes,
            "parameters": reference["parameters"],
            "warnings": reference["warnings"],
            "auto_fit_connections": _auto_fit_connections(graph, stable_nodes, dimensions),
        }

    catalog = {
        **catalog_layers,
        "presets": source_presets,
        "method_collection": method_collection(),
        "evaluation_presets": EVALUATION_PRESETS,
        "evaluation_defaults": EVALUATION_DEFAULTS,
        "offline": {
            "reference_window": REFERENCE_WINDOW,
            "reference_horizon": REFERENCE_HORIZON,
            "preset_count": len(entries),
        },
    }
    return {
        "schema_version": 1,
        "reference_cell": {"window": REFERENCE_WINDOW, "horizon": REFERENCE_HORIZON},
        "catalog": catalog,
        "presets": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "web" / "public" / "offline-manifest.json",
    )
    args = parser.parse_args()
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest['presets'])} presets to {args.output}")


if __name__ == "__main__":
    main()
