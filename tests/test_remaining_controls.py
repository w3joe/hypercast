from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from hypercast4d.architecture import build_architecture, presets, normalize_architecture_spec
from hypercast4d.experiment_controls import initialize_for_evaluation
from hypercast4d.playground_runner import normalize_evaluation
from hypercast4d.remaining_controls import BACKBONES, VARIANTS, initialize, digest_except


def spec(backbone):
    return normalize_architecture_spec(next(p for p in presets() if p['preset_id'] == 'tslib-' + backbone))


def evaluation(backbone, variant):
    return dict(initialization='remaining-internal-v1', replacement=dict(backbone=backbone, variant=variant),
        nested_stopping=True, restore_best_weights=True, cells=[dict(window=32, horizon=5)],
        seeds=[907], epochs=2, early_stopping_patience=20, early_stopping_relative_delta=.001)


@pytest.mark.parametrize('backbone', BACKBONES)
def test_all_arms_pair_untouched_parameters_and_buffers_and_preserve_rng(backbone):
    shared = None
    for variant in VARIANTS:
        architecture = spec(backbone)
        model = build_architecture(architecture, 32, 5)
        before = torch.random.get_rng_state().clone()
        report = initialize_for_evaluation(model, architecture, normalize_evaluation(evaluation(backbone, variant)),
            seed=907, window=32, horizon=5, fold=1)
        assert torch.equal(before, torch.random.get_rng_state())
        if shared is None:
            shared = report['untouched_sha256']
        assert shared == report['untouched_sha256'] == digest_except(model, report['selected_site'])
        assert report['whole_model_parameters'] == sum(p.numel() for p in model.parameters())
    changed = initialize(model, architecture, evaluation(backbone, 'real'), seed=908, window=32, horizon=5, fold=1)
    assert changed['untouched_sha256'] != shared


@pytest.mark.parametrize('backbone', ['dlinear', 'tsmixer', 'itransformer', 'unknown'])
def test_excluded_or_unknown_backbones_cannot_enter_expansion(backbone):
    with pytest.raises(ValueError):
        normalize_evaluation(evaluation(backbone, 'real'))


def test_original_architecture_required_and_legacy_options_unchanged():
    assert 'replacement' not in normalize_evaluation({})
    architecture = spec('patchtst'); architecture['input']['feature_order'] = [3, 2, 1, 0]
    model = build_architecture(architecture, 32, 5)
    with pytest.raises(ValueError, match='differs'):
        initialize(model, architecture, evaluation('patchtst', 'real'), seed=907, window=32, horizon=5, fold=1)
    for field in ['replacement', 'nested_stopping', 'restore_best_weights']:
        ev = evaluation('patchtst', 'real'); ev.pop(field)
        with pytest.raises(ValueError): normalize_evaluation(ev)


def test_scinet_precision_is_equal_for_both_paths_and_restored_on_error():
    from hypercast4d.remaining_controls import precision_context
    conv, matmul = torch.backends.cudnn.conv, torch.backends.cuda.matmul
    before = (conv.fp32_precision, matmul.fp32_precision)
    with pytest.raises(RuntimeError, match='intentional'):
        with precision_context('scinet'):
            assert conv.fp32_precision == matmul.fp32_precision == 'ieee'
            raise RuntimeError('intentional')
    assert (conv.fp32_precision, matmul.fp32_precision) == before
    with precision_context('patchtst'):
        assert (conv.fp32_precision, matmul.fp32_precision) == before


@pytest.mark.parametrize('backbone', ['patchtst', 'scinet', 'film'])
def test_worker_trains_the_replacement_and_returns_audits(tmp_path, monkeypatch, backbone):
    from hypercast4d import playground_runner as runner
    frame = pd.DataFrame(np.random.default_rng(1).normal(size=(200, 4)), columns=['Copper', 'b', 'c', 'd'])
    monkeypatch.setattr(runner, 'load_paper_data', lambda *args: frame)
    runner.run_validation(tmp_path, dict(architecture=spec(backbone), evaluation=evaluation(backbone, 'octonion')))
    runs = pd.read_csv(tmp_path / 'runs.csv')
    report = json.loads((tmp_path / 'initialization.json').read_text())[0]
    assert report['variant'] == 'octonion' and report['protocol'] == 'remaining-internal-v1'
    assert report['whole_model_parameters'] == runs.iloc[0]['parameters']
    assert runs.iloc[0]['stopping_split'] == 'inner'
    assert (tmp_path / 'split_audit.json').exists()
    with pytest.raises(ValueError, match='independent-data'):
        runner.run_final_test(tmp_path, dict(architecture=spec(backbone), evaluation=evaluation(backbone, 'octonion')))
