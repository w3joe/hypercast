import csv
import json

import pytest
import torch
from fastapi.testclient import TestClient

from hypercast4d.architecture import presets
from hypercast4d.layers import HyperDense
from hypercast4d.playground import create_app
from hypercast4d.visualizations import read_forecasts, save_weights, weight_snapshot


@pytest.mark.parametrize('algebra', ['complex', 'quaternion', 'coquaternion', 'cl11', 'octonion'])
def test_effective_matrix_matches_executed_layer(algebra):
    layer = HyperDense(3, 2, algebra, bias=False).double()
    snapshot = weight_snapshot(layer, 'layer')
    matrix = torch.tensor(snapshot['effective'], dtype=torch.float64)
    inputs = torch.randn(7, layer.component_count * 3, dtype=torch.float64)
    torch.testing.assert_close(inputs @ matrix, layer(inputs))


def test_snapshot_sampling_preserves_exact_values_and_trial_identity(tmp_path):
    model = torch.nn.Sequential(HyperDense(15, 12, 'quaternion'))
    save_weights(tmp_path, model, window=32, horizon=5, seed=1, fold=0)
    original = model[0].weight.detach().clone()
    with torch.no_grad():
        model[0].weight.add_(1)
    save_weights(tmp_path, model, window=32, horizon=5, seed=2, fold=0)
    records = json.loads((tmp_path / 'weights.json').read_text())
    assert len(records) == 2
    layer = records['32/5/1/0']['layers'][0]
    assert len(layer['input_indices']) == len(layer['output_indices']) == 8
    expected = original[:, layer['input_indices']][:, :, layer['output_indices']]
    torch.testing.assert_close(torch.tensor(layer['components']), expected)


def test_forecasts_keep_cell_replicate_lead_and_missingness_separate(tmp_path):
    path = tmp_path / 'predictions.csv'
    base = dict(window=32, horizon=5, seed=1, fold=0, lead=2, origin_row=4,
                actual=10, prediction=11, persistence=9, target_date='2026-01-02', forecast_origin='2026-01-01')
    rows = [base, {**base, 'lead': 1}, {**base, 'fold': 1}, {**base, 'actual': 'nan'}, {**base, 'seed': 2}]
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=base.keys()); writer.writeheader(); writer.writerows(rows)
    result = read_forecasts(path, window=32, horizon=5, seed=1, fold=0, lead=2)
    assert result['total'] == 1 and result['invalid'] == 1
    assert result['rows'][0]['actual'] == 10


def test_initialized_preview_is_labelled_and_does_not_change_rng(tmp_path):
    app = create_app(tmp_path / 'results', tmp_path)
    client = TestClient(app)
    spec = next(p for p in presets() if p['preset_id'] == 'paper-quaternion')
    state = torch.random.get_rng_state()
    response = client.post('/api/v1/architectures/weights', json={'architecture': spec, 'window': 10, 'horizon': 1})
    assert response.status_code == 200
    assert response.json()['source'] == 'initialized'
    assert response.json()['layers']
    assert torch.equal(state, torch.random.get_rng_state())
    assert client.get('/api/v1/jobs/missing/weights').status_code == 404
    assert client.get('/api/v1/jobs/missing/forecasts?window=10&horizon=1&seed=1&fold=0').status_code == 404


def test_batched_forecasts_are_filtered_by_trial(tmp_path):
    path = tmp_path / 'predictions.csv'
    base = dict(window=10, horizon=1, seed=7, fold=0, lead=1, origin_row=1, actual=3,
                prediction=4, persistence=2, target_date='2026-01-02', forecast_origin='2026-01-01', trial_id='A')
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=base.keys()); writer.writeheader()
        writer.writerows([base, {**base, 'prediction': 99, 'trial_id': 'B'}])
    result = read_forecasts(path, window=10, horizon=1, seed=7, fold=0, lead=1, trial='A')
    assert result['total'] == 1 and result['rows'][0]['prediction'] == 4
