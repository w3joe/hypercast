"""Opt-in controls for the L4 experiment; published/default presets stay unchanged."""
from __future__ import annotations

import hashlib
import math

import torch
from torch import nn

from .layers import HyperDense
from .upstream_models import UpstreamSequence

INITIALIZATION_PROTOCOL = 'matched-v1'


def _stream_seed(seed: int, window: int, horizon: int, fold: int, role: str) -> int:
    key = f'{INITIALIZATION_PROTOCOL}/{seed}/{window}/{horizon}/{fold}/{role}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % (2**63 - 1)


def state_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize_for_evaluation(model, architecture, evaluation, *, seed: int, window: int,
                              horizon: int, fold: int) -> dict:
    """Pair modules by stable semantic role, independent of constructor RNG use.

    Dense and HyperDense use normal entries with variance 2/(real_in+real_out),
    gain 1 and zero bias. Symmetric low-rank factors match that *effective*
    entry variance in expectation. Their product distribution is not Gaussian.
    Upstream blocks retain their native initialization, reconstructed in a
    dedicated role stream. Pairing applies only when module shapes match.
    """
    if evaluation.get('initialization', 'legacy') == 'legacy':
        return {}
    if evaluation['initialization'] == 'remaining-internal-v1':
        from .remaining_controls import initialize
        return initialize(model, architecture, evaluation, seed=seed, window=window, horizon=horizon, fold=fold)
    if evaluation['initialization'] == 'internal-matched-v1':
        from .internal_controls import initialize
        return initialize(model, architecture, seed=seed, window=window, horizon=horizon, fold=fold)
    if evaluation['initialization'] != INITIALIZATION_PROTOCOL:
        raise ValueError('Unsupported controlled initialization protocol')
    if architecture.get('schema_version', 1) != 1:
        raise ValueError('matched-v1 currently requires a sequential architecture')
    if any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('Apply controlled initialization before moving model to the GPU')
    layers = list(zip(architecture['layers'], model.layers))
    low_rank_std = {}
    for (spec, first), (next_spec, second) in zip(layers, layers[1:]):
        if (spec['id'] == 'test_layer' and next_spec['id'] == 'test_layer_expand'
                and isinstance(first, nn.Linear) and isinstance(second, nn.Linear)):
            if first.out_features != second.in_features:
                raise ValueError('Invalid low-rank factor shapes')
            variance = 2.0 / (first.in_features + second.out_features)
            std = (variance / first.out_features) ** .25
            low_rank_std.update({spec['id']: std, next_spec['id']: std})
    report = {'protocol': INITIALIZATION_PROTOCOL, 'seed': seed, 'window': window,
              'horizon': horizon, 'fold': fold, 'module_sha256': {}, 'weight_variance': {}}
    for spec, module in layers + [({'id': 'forecast_head'}, model.output)]:
        role = spec['id']
        with torch.random.fork_rng(devices=[]):
            # CPU-only state changes; leave the global/loader/dropout streams intact.
            torch.random.default_generator.manual_seed(_stream_seed(seed, window, horizon, fold, role))
            with torch.no_grad():
                if isinstance(module, UpstreamSequence):
                    reference = UpstreamSequence(module.method, module.steps, module.width, **spec['params'])
                    module.load_state_dict(reference.state_dict())
                elif isinstance(module, HyperDense):
                    real_in = module.component_count * module.in_features
                    real_out = module.component_count * module.out_features
                    module.weight.normal_(0, math.sqrt(2.0 / (real_in + real_out)))
                    if module.bias is not None:
                        module.bias.zero_()
                elif isinstance(module, nn.Linear):
                    std = low_rank_std.get(role, math.sqrt(2.0 / (module.in_features + module.out_features)))
                    module.weight.normal_(0, std)
                    if module.bias is not None:
                        module.bias.zero_()
                elif any(True for _ in module.parameters()):
                    raise ValueError(f'matched-v1 does not support parameterized layer {role}: {type(module).__name__}')
        if role == 'forecast_head' and architecture.get('head', {}).get('zero_initialize'):
            with torch.no_grad():
                module.weight.zero_()
                module.bias.zero_()
        if any(True for _ in module.parameters()):
            report['module_sha256'][role] = state_digest(module)
            if isinstance(module, (nn.Linear, HyperDense)):
                report['weight_variance'][role] = float(module.weight.detach().var(unbiased=False))
    return report


def explicit_real_matrix(layer: HyperDense) -> torch.Tensor:
    """Independent block-matrix reference for component-major x @ W."""
    d = layer.component_count
    constants = layer.algebra.constants.to(layer.weight)
    return torch.cat([
        torch.cat([sum(constants[b, a, out] * layer.weight[a] for a in range(d))
                   for out in range(d)], dim=1)
        for b in range(d)
    ], dim=0)


def numerical_preflight(device: str | torch.device) -> dict:
    """Deterministic forward/gradient and variance checks without real data."""
    device = torch.device(device)
    results = []
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1729)
        for algebra, d in [('complex', 2), ('quaternion', 4), ('octonion', 8)]:
            layer = HyperDense(32 // d, 24 // d, algebra).double().to(device)
            with torch.no_grad():
                layer.weight.normal_(0, math.sqrt(2 / 56))
            x = torch.linspace(-.8, .9, 96, device=device, dtype=torch.float64).reshape(3, 32).requires_grad_()
            actual = layer(x)
            expected = x @ explicit_real_matrix(layer) + layer.bias.reshape(-1)
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
            params = (x, layer.weight, layer.bias)
            actual_grad = torch.autograd.grad(actual.square().mean(), params)
            expected_grad = torch.autograd.grad(expected.square().mean(), params)
            for a, b in zip(actual_grad, expected_grad):
                torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
                if not bool(torch.isfinite(a).all()):
                    raise RuntimeError('Nonfinite algebra gradient')
            variances = []
            for _ in range(32):
                with torch.no_grad():
                    layer.weight.normal_(0, math.sqrt(2 / 56))
                variances.append(float(explicit_real_matrix(layer).detach().var(unbiased=False)))
            relative = sum(variances) / len(variances) / (2 / 56)
            if not .8 < relative < 1.2:
                raise RuntimeError(f'{algebra}: initialization variance check failed: {relative}')
            results.append({'algebra': algebra, 'real_shape': [32, 24],
                'forward_and_gradients': 'passed', 'effective_variance_ratio': relative})
    from copy import deepcopy
    from .architecture import build_architecture, presets
    checked_shapes = 0
    for base in presets():
        if base.get('preset_id') not in ('tslib-dlinear', 'tslib-tsmixer', 'tslib-itransformer'):
            continue
        for window, horizon in ((10, 1), (20, 1), (20, 5), (40, 5), (60, 10), (60, 20)):
            shared = {}
            for variant in ('original', 'lift_only', 'real_dense', 'complex', 'quaternion', 'octonion', 'rank8', 'rank4', 'rank2'):
                spec = deepcopy(base)
                spec.pop('preset_id', None)
                spec['locked'] = False
                frontend = []
                if variant != 'original':
                    frontend = [{'id': 'input_lift', 'type': 'dense', 'params': {'units': 32}},
                                {'id': 'lift_activation', 'type': 'activation', 'params': {'kind': 'relu'}}]
                if variant == 'real_dense':
                    frontend.append({'id': 'test_layer', 'type': 'dense', 'params': {'units': 32}})
                elif variant in ('complex', 'quaternion', 'octonion'):
                    dimension = {'complex': 2, 'quaternion': 4, 'octonion': 8}[variant]
                    frontend.append({'id': 'test_layer', 'type': 'hyper_dense', 'params': {'units': 32 // dimension, 'algebra': variant}})
                elif variant.startswith('rank'):
                    frontend.extend([{'id': 'test_layer', 'type': 'dense', 'params': {'units': int(variant[4:])}},
                                     {'id': 'test_layer_expand', 'type': 'dense', 'params': {'units': 32}}])
                if variant not in ('original', 'lift_only'):
                    frontend.append({'id': 'test_activation', 'type': 'activation', 'params': {'kind': 'relu'}})
                spec['layers'] = frontend + spec['layers']
                model = build_architecture(spec, window, horizon)
                report = initialize_for_evaluation(model, spec, {'initialization': INITIALIZATION_PROTOCOL},
                    seed=7, window=window, horizon=horizon, fold=1)
                if variant != 'original':
                    for role in ('input_lift', 'core', 'forecast_head'):
                        digest = report['module_sha256'][role]
                        if shared.setdefault(role, digest) != digest:
                            raise RuntimeError(f'Unpaired initialization: {base["name"]}/{role}')
                model.to(device)
                x = torch.linspace(-1, 1, 2 * window * 4, device=device).reshape(2, window, 4)
                prediction = model(x)
                if prediction.shape != (2, horizon) or not bool(torch.isfinite(prediction).all()):
                    raise RuntimeError(f'Invalid controlled architecture output: {variant}')
                prediction.square().mean().backward()
                if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise RuntimeError(f'Invalid controlled architecture gradient: {variant}')
                checked_shapes += 1
    return {'device': str(device), 'algebras': results, 'architecture_shapes_checked': checked_shapes,
            'shared_initialization': 'paired', 'passed': True}
