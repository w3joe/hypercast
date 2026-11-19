"""Tests for the 11-dataset registry, multi-format loader, and arbitrary-channel models."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
import torch

from hypercast4d.architecture import build_architecture, presets, validate_architecture
from hypercast4d.cli import main as cli_main
from hypercast4d.data import load_dataset, prepare_windows
from hypercast4d.datasets import (
    DATASETS,
    adapt_spec_features,
    build_nyc_taxi_dataset,
    dataset_catalog,
    get_dataset_spec,
)
from hypercast4d.graph_architecture import convert_to_graph
from hypercast4d.model_editing import edit_layer
from hypercast4d.playground import create_app
from hypercast4d.playground_runner import normalize_evaluation, run_job


EXPECTED_DATASETS = {
    "copper": (4, "Copper"),
    "etth1": (7, "OT"),
    "etth2": (7, "OT"),
    "ettm1": (7, "OT"),
    "ettm2": (7, "OT"),
    "weather": (21, "OT"),
    "electricity": (321, "OT"),
    "traffic": (862, "OT"),
    "exchange": (8, "OT"),
    "ili": (7, "OT"),
    "nyc_taxi": (8, "trip_count"),
}


def test_registry_covers_all_tslib_and_nyc_taxi_datasets() -> None:
    assert set(DATASETS) == set(EXPECTED_DATASETS)
    for key, (features, target) in EXPECTED_DATASETS.items():
        spec = get_dataset_spec(key)
        assert spec.features == features
        assert spec.target_column == target
        assert len(spec.columns) == features
        assert target in spec.columns
    assert get_dataset_spec("ecl").id == "electricity"
    assert get_dataset_spec("exchange_rate").id == "exchange"
    assert get_dataset_spec("national_illness").id == "ili"
    assert get_dataset_spec("nyc-taxi").id == "nyc_taxi"


def test_normalize_evaluation_selects_dataset_default_target() -> None:
    weather_eval = normalize_evaluation({"data_path": "data/external/tslib/weather.csv"})
    assert weather_eval["target_column"] == "OT"
    taxi_eval = normalize_evaluation({"data_path": "data/external/nyc_taxi/nyc_taxi.csv"})
    assert taxi_eval["target_column"] == "trip_count"


def test_load_dataset_reads_csv_and_parquet_with_arbitrary_channels(tmp_path: Path) -> None:
    dates = pd.date_range("2024-01-01", periods=64, freq="1h")
    raw = pd.DataFrame(
        {
            "date": dates,
            **{f"c{i}": np.sin(np.linspace(0, 4, 64) + i) for i in range(20)},
            "OT": np.linspace(10.0, 25.0, 64),
        }
    )
    csv_path = tmp_path / "weather_sample.csv"
    parquet_path = tmp_path / "weather_sample.parquet"
    raw.to_csv(csv_path, index=False)
    raw.to_parquet(parquet_path, index=False)

    for path in (csv_path, parquet_path):
        loaded = load_dataset(path, "OT")
        assert loaded.shape == (64, 21)
        assert loaded.columns[0] == "OT"
        prepared = prepare_windows(loaded, window=8, horizon=2, train_fraction=0.6, validation_fraction=0.2)
        assert prepared.train.x.shape[1:] == (8, 21)


@pytest.mark.parametrize("features", [7, 8, 21, 321, 862])
def test_editable_tslib_and_msgnet_compile_and_autofit_across_dataset_widths(features: int) -> None:
    for preset_id in ("tslib-tsmixer", "tslib-msgnet"):
        base = next(p for p in presets() if p["preset_id"] == preset_id)
        adapted = adapt_spec_features(base, features)
        graph = convert_to_graph(adapted, window=10, horizon=2, features=features)
        incoming = next(e for e in graph["edges"] if e["source"] == "getitem_1")
        edited = edit_layer(
            graph,
            "insert",
            node_id="hyper_in",
            kind="hyper_dense",
            params={"algebra": "octonion", "shape_mode": "preserve"},
            before=incoming["target"],
            port=incoming["port"],
            cells=[{"window": 10, "horizon": 2}],
            features=features,
        )
        info = validate_architecture(edited, window=10, horizon=2, features=features)
        assert info["valid"] is True
        assert info["input_shape"] == f"[B, 10, {features}]"
        model = build_architecture(edited, window=10, horizon=2, features=features).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 10, features))
        assert tuple(out.shape) == (2, 2)


def test_nyc_taxi_aggregation_and_end_to_end_worker_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_dir = tmp_path / "data" / "external" / "nyc_taxi" / "raw"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(
        {"LocationID": [161, 132, 61, 1], "Borough": ["Manhattan", "Queens", "Brooklyn", "EWR"]}
    ).to_csv(raw_dir / "taxi_zone_lookup.csv", index=False)

    timestamps = pd.date_range("2024-01-01 00:15:00", periods=160, freq="1h")
    pd.DataFrame(
        {
            "tpep_pickup_datetime": timestamps,
            "tpep_dropoff_datetime": timestamps + pd.Timedelta(minutes=14),
            "PULocationID": [161, 132, 61, 1] * 40,
            "trip_distance": np.linspace(1.5, 9.0, 160),
            "fare_amount": np.linspace(12.0, 42.0, 160),
        }
    ).to_parquet(raw_dir / "yellow_tripdata_2024-01.parquet", index=False)

    manifest = build_nyc_taxi_dataset(
        tmp_path, months=("2024-01",), force=False, allow_updated_source=True
    )
    assert manifest["features"] == 8
    assert manifest["rows"] == 31 * 24

    monkeypatch.chdir(tmp_path)
    tsmixer = next(p for p in presets() if p["preset_id"] == "tslib-tsmixer")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "request.json").write_text(
        json.dumps(
            {
                "phase": "validation",
                "architecture": tsmixer,
                "evaluation": {
                    "preset": "quick",
                    "data_path": "data/external/nyc_taxi/nyc_taxi.csv",
                    "epochs": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    run_job(job_dir)
    status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "complete"
    runs = pd.read_csv(job_dir / "runs.csv")
    assert len(runs) == 1 and np.isfinite(runs.loc[0, "mae"])


def test_api_and_cli_expose_dataset_catalog_and_validation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "external" / "tslib"
    data_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=80, freq="1h"),
            **{col: np.linspace(1.0, 10.0, 80) + i for i, col in enumerate(DATASETS["etth2"].columns)},
        }
    )
    frame.to_csv(data_dir / "ETTh2.csv", index=False)

    app = create_app(results_root=tmp_path / "results", project_root=tmp_path)
    with TestClient(app) as client:
        cat = client.get("/api/v1/catalog").json()
        assert len(cat["datasets"]) == 11
        etth2 = next(d for d in cat["datasets"] if d["id"] == "etth2")
        assert etth2["available"] is True and etth2["features"] == 7

        preset = next(p for p in cat["presets"] if p["preset_id"] == "tslib-dlinear")
        graph = client.post(
            "/api/v1/architectures/convert",
            json={"architecture": preset, "window": 10, "horizon": 2, "data_path": etth2["data_path"]},
        ).json()
        validated = client.post(
            "/api/v1/architectures/validate",
            json={"architecture": graph, "window": 10, "horizon": 2, "data_path": etth2["data_path"]},
        ).json()
        assert validated["valid"] is True
        assert validated["input_shape"] == "[B, 10, 7]"

    model_file = tmp_path / "etth2_model.yaml"
    assert cli_main(["--project-root", str(tmp_path), "dataset", "list", "--json"]) == 0
    assert cli_main(["--project-root", str(tmp_path), "model", "create", "tslib-dlinear", "--dataset", "etth2", "--out", str(model_file)]) == 0
    assert cli_main(["--project-root", str(tmp_path), "experiment", "run", str(model_file), "--dataset", "etth2", "--preset", "quick", "--dry-run", "--json"]) == 0
