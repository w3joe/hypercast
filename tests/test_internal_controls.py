import json
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from hypercast4d.internal_controls import nested_windows, preflight
from hypercast4d.playground_runner import normalize_evaluation
from hypercast4d.training import fit_model


def test_actual_graph_pairing_and_surgery_equivalence():
    report = preflight('cpu')
    assert report['passed'] and report['architecture_shapes_checked'] == 24
    assert all(t['realizable_teacher_max_error'] < 1e-10 for t in report['teacher_capacity'])


@pytest.mark.parametrize('fraction', [.55, .65, .70, .75])
def test_nested_splits_isolate_stopping_and_scaler_from_outer_data(fraction):
    frame = pd.DataFrame(np.arange(4000).reshape(1000, 4).astype(float))
    p, inner, audit = nested_windows(frame, 32, 5, fraction, .10)
    boundary = audit['train_rows'][1]
    outer_start = audit['outer_rows'][0]
    assert np.max(p.train.target_start + 5) <= boundary
    assert np.min(inner.target_start) >= boundary
    assert np.max(inner.target_start + 5) <= outer_start
    assert np.min(p.validation.target_start) >= outer_start
    assert np.max(p.validation.target_start + 5) <= audit['outer_rows'][1]
    changed = frame.copy()
    changed.iloc[outer_start:] = 1e12
    q, inner_q, _ = nested_windows(changed, 32, 5, fraction, .10)
    np.testing.assert_array_equal(p.train.x, q.train.x)
    np.testing.assert_array_equal(inner.x, inner_q.x)
    np.testing.assert_array_equal(inner.y, inner_q.y)
    changed.iloc[boundary:] = -1e12
    q, _, _ = nested_windows(changed, 32, 5, fraction, .10)
    np.testing.assert_array_equal(p.scaler.minimum, q.scaler.minimum)
    np.testing.assert_array_equal(p.scaler.span, q.scaler.span)


def test_relative_early_stop_and_selected_epoch(monkeypatch):
    losses = iter([1., .9995, .9989, .9988, .9987])
    class Loss(nn.Module):
        def forward(self, prediction, target):
            return prediction.sum() * 0 + (1. if torch.is_grad_enabled() else next(losses))
    monkeypatch.setattr(nn, 'MSELoss', Loss)
    data = TensorDataset(torch.ones(2, 1), torch.ones(2, 1))
    fit = fit_model(nn.Linear(1, 1), data, data, seed=1, epochs=10, batch_size=2,
        learning_rate=.001, adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-7,
        adam_amsgrad=False, loss_name='mse', shuffle=True, early_stopping_patience=2,
        early_stopping_min_delta=0., early_stopping_relative_delta=.001,
        restore_best_weights=True, device=torch.device('cpu'))
    assert fit.best_epoch == 3 and fit.epochs_ran == 5
    assert fit.best_validation_loss == pytest.approx(.9989)


def test_nested_runner_uses_inner_for_fit_and_outer_for_scores(tmp_path, monkeypatch):
    from hypercast4d import playground_runner as runner
    from hypercast4d.architecture import presets
    from hypercast4d.graph_architecture import convert_to_graph
    frame = pd.DataFrame(np.random.default_rng(1).normal(size=(500, 4)), columns=['Copper', 'b', 'c', 'd'])
    monkeypatch.setattr(runner, 'load_paper_data', lambda *args: frame)
    real_fit = runner.fit_model
    evaluation = dict(initialization='internal-matched-v1', nested_stopping=True,
        early_stopping_relative_delta=.001, restore_best_weights=True,
        cells=[dict(window=32, horizon=5)], seeds=[907], epochs=2)
    expected, inner, audit = nested_windows(frame, 32, 5, .70, .15)
    def checked_fit(model, train, validation, **kwargs):
        torch.testing.assert_close(validation.tensors[1], inner.as_dataset().tensors[1])
        assert len(train) == len(expected.train.x)
        return real_fit(model, train, validation, **kwargs)
    monkeypatch.setattr(runner, 'fit_model', checked_fit)
    spec = convert_to_graph(next(s for s in presets() if s['preset_id'] == 'tslib-dlinear'), 32, 5)
    runner.run_validation(tmp_path, dict(architecture=spec, evaluation=evaluation))
    rows = pd.read_csv(tmp_path / 'runs.csv')
    assert rows.iloc[0]['validation_samples'] == len(expected.validation.x)
    assert rows.iloc[0]['stopping_split'] == 'inner'
    assert json.loads((tmp_path / 'split_audit.json').read_text())[0]['scaler_rows'] == audit['scaler_rows']


def test_new_options_do_not_change_historical_defaults():
    assert 'nested_stopping' not in normalize_evaluation({})
    assert normalize_evaluation({'initialization': 'internal-matched-v1'})['initialization'] == 'internal-matched-v1'
    with pytest.raises(ValueError):
        normalize_evaluation({'early_stopping_relative_delta': float('nan')})
