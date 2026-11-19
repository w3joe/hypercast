"""Unit tests for the multi-dataset 15-backbone × 8-arm runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from hypercast4d.datasets import DATASETS
from hypercast4d.tslib15 import ARMS
from hypercast4d.tslib15_multi import (
    NEW_BENCHMARK_DATASETS,
    build_model,
    prepare_dataset_bundle,
    run_trial,
)


def test_new_benchmark_datasets_match_registry() -> None:
    assert set(NEW_BENCHMARK_DATASETS) == {
        "etth2", "ettm1", "ettm2", "weather", "electricity", "traffic", "exchange", "ili", "nyc_taxi"
    }
    for ds_id in NEW_BENCHMARK_DATASETS:
        assert ds_id in DATASETS


@pytest.mark.parametrize("features", [7, 8, 21])
def test_build_model_preserves_untouched_weights_across_arms(features: int) -> None:
    digests = set()
    for arm in ARMS:
        model, report = build_model("tsmixer", arm, 3301, features=features, dataset_id="synthetic")
        digests.add(report["untouched_sha256"])
        with torch.no_grad():
            y = model(torch.randn(2, 32, features))
        assert tuple(y.shape) == (2, 5)
    assert len(digests) == 1


def test_prepare_bundle_and_run_calibration_and_test_trial_on_cpu(tmp_path: Path) -> None:
    # Build a small synthetic NYC Taxi CSV matching the registered path.
    csv_path = tmp_path / DATASETS["nyc_taxi"].data_path
    csv_path.parent.mkdir(parents=True)
    n_rows = 240
    t = np.linspace(0, 12, n_rows)
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n_rows, freq="1h"),
            **{col: np.sin(t + i) + 20.0 + i for i, col in enumerate(DATASETS["nyc_taxi"].columns)},
        }
    ).to_csv(csv_path, index=False)

    cal_blob, cal_manifest = prepare_dataset_bundle("nyc_taxi", tmp_path, include_test=False)
    assert cal_manifest["features"] == 8
    assert cal_manifest["include_test"] is False
    cal_res = run_trial(
        cal_manifest,
        cal_blob,
        {"dataset": "nyc_taxi", "phase": "calibration", "backbone": "dlinear", "arm": "quaternion", "seed": 3102, "epochs": 2, "learning_rate": 0.001},
        tmp_path / "cal",
        device="cpu",
    )
    assert cal_res["checkpoint_replay_passed"] is True
    assert cal_res["epochs_ran"] == 2
    assert cal_res["test_scored"] is False

    test_blob, test_manifest = prepare_dataset_bundle("nyc_taxi", tmp_path, include_test=True)
    test_res = run_trial(
        test_manifest,
        test_blob,
        {"dataset": "nyc_taxi", "phase": "test", "backbone": "dlinear", "arm": "quaternion", "seed": 3301, "epochs": 2, "learning_rate": 0.001},
        tmp_path / "test",
        device="cpu",
    )
    assert test_res["checkpoint_replay_passed"] is True
    assert test_res["test_scored"] is True
    assert np.isfinite(test_res["mae"])
    assert (tmp_path / "test" / "predictions.npz").is_file()
