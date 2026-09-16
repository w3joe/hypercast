"""Frozen single-site graph experiment, independent of the historical frontend study."""
from copy import deepcopy
import hashlib
import math

import numpy as np
import torch
from torch import nn

from .architecture import build_architecture
from .data import MinMaxStats, PreparedData, WindowSplit
from .experiment_controls import explicit_real_matrix, state_digest
from .graph_architecture import convert_to_graph
from .layers import HyperDense

PROTOCOL = 'internal-matched-v1'
SITES = {
    'tslib_dlinear': 'layers_0_model_linear_seasonal',
    'tslib_tsmixer': 'layers_0_model_model_0_temporal_0',
    'tslib_itransformer': 'layers_0_model_encoder_attn_layers_0_attention_out_projection',
}
VARIANTS = ('native', 'real', 'complex', 'quaternion', 'octonion', 'rank8', 'rank4', 'rank2')


def variant_spec(native, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown internal variant')
    spec = deepcopy(native)
    site = SITES[spec['sources']['s0']['layers'][0]['type']]
    spec['name'] = spec['sources']['s0']['name'] + ' / internal ' + variant
    if variant == 'native':
        return spec
    node = next(n for n in spec['nodes'] if n['id'] == site)
    node.clear()
    node.update(id=site, kind='dense', params={'units': 32, 'bias': True})
    if variant in ('complex', 'quaternion', 'octonion'):
        d = {'complex': 2, 'quaternion': 4, 'octonion': 8}[variant]
        node.update(kind='hyper_dense', params={'units': 32 // d, 'bias': True, 'algebra': variant})
    for edge in spec['edges']:
        if edge['target'] == site:
            edge['port'] = 'x'
    if variant.startswith('rank'):
        node['params'] = {'units': int(variant[4:]), 'bias': False}
        second = site + '_expand'
        spec['nodes'].append({'id': second, 'kind': 'dense', 'params': {'units': 32, 'bias': True}})
        for edge in spec['edges']:
            if edge['source'] == site:
                edge['source'] = second
        spec['edges'].append({'source': site, 'target': second, 'port': 'x'})
    return spec


def _seed(seed, window, horizon, fold, role):
    key = f'{PROTOCOL}/{seed}/{window}/{horizon}/{fold}/{role}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % (2**63 - 1)


def initialize(model, architecture, *, seed, window, horizon, fold):
    if architecture.get('schema_version') != 2 or set(architecture['sources']) != {'s0'} or window != 32:
        raise ValueError('internal-matched-v1 requires a single-source graph and window 32')
    if any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('Initialize on CPU before moving to accelerator')
    source = architecture['sources']['s0']
    site = SITES[source['layers'][0]['type']]
    report = dict(protocol=PROTOCOL, seed=seed, window=window, horizon=horizon, fold=fold,
                  selected_site=site, untouched_sha256={}, selected_sha256={})
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.random.default_generator.manual_seed(_seed(seed, window, horizon, fold, 'native-reference'))
        native_spec = convert_to_graph(source, window, horizon)
        reference = build_architecture(native_spec, window, horizon)
        # Module registry positions change after graph edits. Pair by graph node identity.
        for node in architecture['nodes']:
            key = node['id']
            if key in (site, site + '_expand') or key not in model.bindings:
                continue
            if node.get('source_ref') != reference.nodes[key].get('source_ref'):
                raise ValueError(f'Untouched source identity changed: {key}')
            module = model.blocks[model.bindings[key]]
            module.load_state_dict(reference.blocks[reference.bindings[key]].state_dict(), strict=True)
            report['untouched_sha256'][key] = state_digest(module)
        # Graph constants include buffers and any get_attr parameters.
        for key, binding in model.bindings.items():
            if key.startswith('state:'):
                model.constants[binding].copy_(reference.constants[reference.bindings[key]])
        report['constants_sha256'] = state_digest(model.constants)
        first = model.blocks[model.bindings[site]]
        native = model.nodes[site]['kind'] == 'source'
        selected = [site] + ([site + '_expand'] if site + '_expand' in model.bindings else [])
        for role in selected:
            module = model.blocks[model.bindings[role]]
            torch.random.default_generator.manual_seed(_seed(seed, window, horizon, fold, role))
            if native:
                module.load_state_dict(reference.blocks[reference.bindings[site]].state_dict())
            else:
                std = (1 / (32 * first.out_features)) ** .25 if len(selected) == 2 else math.sqrt(1 / 32)
                module.weight.normal_(0, std)
                if module.bias is not None:
                    module.bias.zero_()
            report['selected_sha256'][role] = state_digest(module)
        report['selected_parameters'] = sum(p.numel() for role in selected
            for p in model.blocks[model.bindings[role]].parameters())
        matrix = (first.weight.T if isinstance(first, nn.Linear) else explicit_real_matrix(first))
        if len(selected) == 2:
            matrix = matrix @ model.blocks[model.bindings[selected[1]]].weight.T
        report['effective_rank'] = int(torch.linalg.matrix_rank(matrix.double()))
        report['effective_weight_variance'] = float(matrix.var(unbiased=False))
    return report


def nested_windows(frame, window, horizon, train_fraction, validation_fraction):
    """Inner stop targets are inside the nominal training prefix; outer targets never select epochs."""
    values = frame.to_numpy(dtype=np.float64)
    nominal = int(len(values) * train_fraction)
    train_end = int(nominal * .85)
    outer_end = int(len(values) * (train_fraction + validation_fraction))
    scaler = MinMaxStats.fit(values[:train_end])
    scaled = scaler.transform(values).astype(np.float32)

    def split(start, end):
        starts = np.arange(max(window, start), end - horizon + 1, dtype=np.int64)
        if len(starts) == 0:
            raise ValueError('Nested split has no complete target windows')
        return WindowSplit(np.stack([scaled[t-window:t] for t in starts]),
                           np.stack([scaled[t:t+horizon, 0] for t in starts]), starts)

    train, inner, outer = split(0, train_end), split(train_end, nominal), split(nominal, outer_end)
    audit = dict(train_rows=[0, train_end], inner_rows=[train_end, nominal],
                 outer_rows=[nominal, outer_end], scaler_rows=[0, train_end],
                 scaler_minimum=scaler.minimum.tolist(), scaler_span=scaler.span.tolist(),
                 train_samples=len(train.x), inner_samples=len(inner.x), outer_samples=len(outer.x),
                 checkpoint_selection='inner_only', bounds='half-open; all targets inside their partition')
    return PreparedData(train, outer, inner, scaler, tuple(frame.columns), train_end, outer_end), inner, audit


def preflight(device='cpu'):
    """Actual graph pairing/equivalence and independent algebra/teacher recovery gates."""
    from .architecture import presets
    device = torch.device(device)
    reports, teachers = [], []
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        torch.random.default_generator.manual_seed(1729)
        for base in presets():
            if base.get('preset_id') not in ('tslib-dlinear', 'tslib-tsmixer', 'tslib-itransformer'):
                continue
            native_spec = convert_to_graph(base, 32, 5)
            shared = None
            native_model = None
            for variant in VARIANTS:
                spec = variant_spec(native_spec, variant)
                model = build_architecture(spec, 32, 5)
                before = torch.random.get_rng_state().clone()
                report = initialize(model, spec, seed=907, window=32, horizon=5, fold=1)
                assert torch.equal(before, torch.random.get_rng_state()), 'Initialization changed global RNG'
                pairing = (report['untouched_sha256'], report['constants_sha256'])
                if shared is None:
                    shared = pairing
                assert shared == pairing, 'Untouched weights differ across variants'
                expected = dict(native=1056, real=1056, complex=544, quaternion=288,
                                octonion=160, rank8=544, rank4=288, rank2=160)[variant]
                assert report['selected_parameters'] == expected
                model.to(device).eval()
                x = torch.linspace(-1, 1, 256, device=device).reshape(2, 32, 4)
                if variant == 'native':
                    native_model = model
                if variant == 'real':
                    # Exact native-weight surgery must preserve full model outputs/input gradients.
                    selected = model.blocks[model.bindings[report['selected_site']]]
                    saved = deepcopy(selected.state_dict())
                    selected.load_state_dict(native_model.blocks[native_model.bindings[report['selected_site']]].state_dict())
                    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
                    ya, yb = model(a), native_model(b)
                    torch.testing.assert_close(ya, yb)
                    torch.testing.assert_close(torch.autograd.grad(ya.sum(), a)[0], torch.autograd.grad(yb.sum(), b)[0])
                    for dtype in (torch.float32, torch.float64):
                        for training in (False, True):
                            left, right = deepcopy(model).to(dtype), deepcopy(native_model).to(dtype)
                            left.train(training); right.train(training)
                            xa, xb = x.to(dtype).clone().requires_grad_(), x.to(dtype).clone().requires_grad_()
                            torch.manual_seed(73)
                            ya = left(xa)
                            torch.manual_seed(73)
                            yb = right(xb)
                            torch.testing.assert_close(ya, yb)
                            ya.square().mean().backward(); yb.square().mean().backward()
                            torch.testing.assert_close(xa.grad, xb.grad)
                            for key in left.nodes:
                                if key not in left.bindings:
                                    continue
                                lp = dict(left.blocks[left.bindings[key]].named_parameters())
                                rp = dict(right.blocks[right.bindings[key]].named_parameters())
                                for name in lp:
                                    assert (lp[name].grad is None) == (rp[name].grad is None)
                                    if lp[name].grad is not None:
                                        torch.testing.assert_close(lp[name].grad, rp[name].grad)
                            torch.optim.Adam(left.parameters(), lr=.001).step()
                            torch.optim.Adam(right.parameters(), lr=.001).step()
                            for key in left.nodes:
                                if key in left.bindings:
                                    lm, rm = left.blocks[left.bindings[key]], right.blocks[right.bindings[key]]
                                    for name, value in lm.state_dict().items():
                                        torch.testing.assert_close(value, rm.state_dict()[name])
                    selected.load_state_dict(saved)
                output = model(x)
                assert output.shape == (2, 5) and bool(torch.isfinite(output).all())
                output.square().mean().backward()
                assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                assert all(p.grad is not None for p in model.blocks[model.bindings[report['selected_site']]].parameters())
                reports.append(dict(backbone=base['preset_id'], variant=variant, **report))
        for algebra, d in [('complex', 2), ('quaternion', 4), ('octonion', 8)]:
            layer = HyperDense(32//d, 32//d, algebra).double().to(device)
            x = torch.randn(64, 32, dtype=torch.float64).to(device).requires_grad_()
            actual = layer(x)
            expected = x @ explicit_real_matrix(layer) + layer.bias.reshape(-1)
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
            ag = torch.autograd.grad(actual.square().mean(), (x, layer.weight, layer.bias))
            eg = torch.autograd.grad(expected.square().mean(), (x, layer.weight, layer.bias))
            for a, b in zip(ag, eg):
                torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
            # Orthogonal basis projection recovers a realizable teacher and quantifies
            # irreducible capacity error on an unstructured teacher (no optimization confound).
            basis = []
            for index in range(layer.weight.numel()):
                with torch.no_grad():
                    layer.weight.zero_(); layer.weight.reshape(-1)[index] = 1
                basis.append(explicit_real_matrix(layer).detach().reshape(-1))
            basis = torch.stack(basis)
            target = torch.randn(1024, dtype=torch.float64).to(device)
            project = lambda y: ((basis @ y) / d) @ basis
            structured = torch.randn(len(basis), dtype=torch.float64).to(device) @ basis
            torch.testing.assert_close(project(structured), structured, atol=1e-10, rtol=1e-10)
            teachers.append(dict(algebra=algebra, realizable_teacher_max_error=float((project(structured)-structured).abs().max()),
                random_teacher_relative_mse=float((project(target)-target).square().sum()/target.square().sum()),
                method='closed-form orthogonal projection; diagnostic, not training evidence'))
    return dict(passed=True, device=str(device), architecture_shapes_checked=len(reports),
                graphs=reports, teacher_capacity=teachers)
