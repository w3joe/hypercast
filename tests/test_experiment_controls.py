from copy import deepcopy
import math

import pytest
import torch

from hypercast4d.architecture import build_architecture, presets
from hypercast4d.experiment_controls import initialize_for_evaluation, numerical_preflight
from hypercast4d.playground_runner import normalize_evaluation


def spec_for(backbone, variant):
    spec = deepcopy(next(s for s in presets() if s['preset_id'] == backbone))
    spec.pop('preset_id', None)
    spec['locked'] = False
    layers = [{'id': 'input_lift', 'type': 'dense', 'params': {'units': 32}},
              {'id': 'lift_activation', 'type': 'activation', 'params': {'kind': 'relu'}}]
    if variant == 'real_dense':
        layers.append({'id': 'test_layer', 'type': 'dense', 'params': {'units': 32}})
    elif variant in ('complex', 'quaternion', 'octonion'):
        d = {'complex': 2, 'quaternion': 4, 'octonion': 8}[variant]
        layers.append({'id': 'test_layer', 'type': 'hyper_dense', 'params': {'units': 32 // d, 'algebra': variant}})
    elif variant.startswith('rank'):
        layers.extend([{'id': 'test_layer', 'type': 'dense', 'params': {'units': int(variant[4:])}},
                       {'id': 'test_layer_expand', 'type': 'dense', 'params': {'units': 32}}])
    else:
        raise ValueError(variant)
    layers.append({'id': 'test_activation', 'type': 'activation', 'params': {'kind': 'relu'}})
    spec['layers'] = layers + spec['layers']
    return spec


def controlled(spec, seed=19, window=20, horizon=5):
    model = build_architecture(spec, window, horizon)
    rng = torch.get_rng_state().clone()
    report = initialize_for_evaluation(model, spec, {'initialization': 'matched-v1'},
                                      seed=seed, window=window, horizon=horizon, fold=1)
    assert torch.equal(rng, torch.get_rng_state())
    return model, report


@pytest.mark.parametrize('backbone', ['tslib-dlinear', 'tslib-tsmixer', 'tslib-itransformer'])
def test_all_shared_module_states_are_paired_despite_different_map_constructors(backbone):
    reports = [controlled(spec_for(backbone, v))[1] for v in
               ['real_dense', 'complex', 'quaternion', 'octonion', 'rank8', 'rank4', 'rank2']]
    for role in ['input_lift', 'core', 'forecast_head']:
        assert len({r['module_sha256'][role] for r in reports}) == 1, role
    changed = controlled(spec_for(backbone, 'real_dense'), seed=31)[1]
    assert changed['module_sha256']['forecast_head'] != reports[0]['module_sha256']['forecast_head']


@pytest.mark.parametrize('variant', ['real_dense', 'complex', 'quaternion', 'octonion', 'rank8', 'rank4', 'rank2'])
def test_effective_map_variance_matches_real_dense(variant):
    from hypercast4d.experiment_controls import explicit_real_matrix
    from hypercast4d.layers import HyperDense
    spec = spec_for('tslib-dlinear', variant)
    values = []
    for seed in range(48):
        model, _ = controlled(spec, seed)
        mapping = model.layers[2]
        if isinstance(mapping, HyperDense):
            weight = explicit_real_matrix(mapping)
        elif variant.startswith('rank'):
            weight = model.layers[3].weight @ mapping.weight
        else:
            weight = mapping.weight
        values.append(float(weight.detach().square().mean()))
    assert .8 < sum(values) / len(values) / (1 / 32) < 1.2


def test_legacy_is_noop_and_old_evaluation_hash_inputs_stay_unchanged():
    spec = spec_for('tslib-tsmixer', 'quaternion')
    model = build_architecture(spec, 20, 5)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    assert initialize_for_evaluation(model, spec, {}, seed=7, window=20, horizon=5, fold=1) == {}
    for k, v in model.state_dict().items():
        assert torch.equal(before[k], v)
    assert 'initialization' not in normalize_evaluation({'preset': 'standard'})
    assert normalize_evaluation({'initialization': 'matched-v1', 'benchmark': True})['initialization'] == 'matched-v1'
    with pytest.raises(ValueError, match='initialization'):
        normalize_evaluation({'initialization': 'typo'})


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA preflight runs on Modal'))])
def test_forward_gradients_and_initial_variance_preflight(device):
    assert numerical_preflight(device)['passed']


@pytest.mark.parametrize('backbone', ['tslib-dlinear', 'tslib-tsmixer', 'tslib-itransformer'])
@pytest.mark.parametrize('cell', [(20, 5), (60, 10)])
def test_controlled_planned_shapes_have_finite_forward_and_gradients(backbone, cell):
    for variant in ['real_dense', 'complex', 'quaternion', 'octonion', 'rank8', 'rank4', 'rank2']:
        model, _ = controlled(spec_for(backbone, variant), window=cell[0], horizon=cell[1])
        prediction = model(torch.randn(2, cell[0], 4))
        assert prediction.shape == (2, cell[1])
        prediction.square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
