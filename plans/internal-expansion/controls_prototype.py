"""Offline eager-model replacement prototype. Not imported by the active worker.

Promote into a versioned worker protocol only after the current stage stops and
source provenance is recorded. Keeps original data-dependent model forwards.
"""
from copy import deepcopy
import hashlib
import math

import torch
from torch import nn

from hypercast4d.architecture import build_architecture, presets
from hypercast4d.experiment_controls import explicit_real_matrix
from hypercast4d.layers import HyperDense

PROTOCOL = 'remaining-internal-v1-prototype'
SITES = {
    'patchtst': 'encoder.attn_layers.0.attention.out_projection',
    'timesnet': 'predict_linear',
    'crossformer': 'encoder.encode_blocks.0.encode_layers.0.time_attention.out_projection',
    'frets': 'fc.0',
    'lightts': 'layer_3.spatial_proj.0',
    'micn': 'regression',
    'msgnet': 'seq2pred.seq2pred',
    'scinet': 'projection_1',
    'segrnn': 'predict.1',
    'timemixer': 'pdm_blocks.0.mixing_multi_scale_trend.up_sampling_layers.0.2',
    'timexer': 'encoder.layers.0.self_attention.out_projection',
}
VARIANTS = ('native', 'real', 'complex', 'quaternion', 'octonion', 'lowrank2', 'lowrank4', 'lowrank8')


class ContiguousHyperDense(HyperDense):
    """Preserve Linear's output layout for upstream consumers that call view()."""
    def forward(self, x):
        return super().forward(x).contiguous()


class ChannelMap(nn.Module):
    """Exact kernel-one, stride-one, ungrouped Conv1d equivalent."""
    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping

    def forward(self, x):
        return self.mapping(x.transpose(1, 2)).transpose(1, 2)


def digest_except(model, prefix):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name == prefix or name.startswith(prefix + '.'):
            continue
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def dimensions(module):
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features, module.bias is not None
    if isinstance(module, nn.Conv1d) and module.kernel_size == (1,) and module.stride == (1,) and module.padding == (0,) and module.groups == 1:
        return module.in_channels, module.out_channels, module.bias is not None
    raise ValueError('Expected a Linear or an exact pointwise Conv1d')


def build_case(backbone, variant, seed=907, window=32, horizon=5, fold=1):
    if backbone not in SITES or variant not in VARIANTS or window != 32:
        raise ValueError('Unsupported backbone, variant or context; original three are excluded')
    path = 'layers.0.model.' + SITES[backbone]
    stream = f'{PROTOCOL}/{backbone}/{seed}/{window}/{horizon}/{fold}'
    seed_for = lambda role: int.from_bytes(hashlib.sha256((stream + '/' + role).encode()).digest()[:8], 'big') % (2**63-1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed_for('native'))
        spec = deepcopy(next(p for p in presets() if p['preset_id'] == 'tslib-' + backbone))
        model = build_architecture(spec, window, horizon)
        original = model.get_submodule(path)
        width_in, width_out, bias = dimensions(original)
        assert width_in % 8 == width_out % 8 == 0
        untouched = digest_except(model, path)
        report = dict(protocol=PROTOCOL, backbone=backbone, variant=variant, selected_site=path,
            in_features=width_in, out_features=width_out, bias=bias, seed=seed, window=window,
            horizon=horizon, fold=fold, untouched_sha256=untouched, rank=None)
        if variant == 'native':
            mapping = original
        else:
            torch.random.default_generator.manual_seed(seed_for(path))
            variance = 2 / (width_in + width_out)
            if variant == 'real':
                mapping = nn.Linear(width_in, width_out, bias=bias)
            elif variant.startswith('lowrank'):
                dimension = int(variant[len('lowrank'):])
                # Some narrow rectangular maps have no positive rank below the
                # hypercomplex budget. Keep a real rank-one control and disclose
                # its excess rather than silently omit it or change model width.
                rank = max(1, (width_in * width_out) // (dimension * (width_in + width_out)))
                mapping = nn.Sequential(nn.Linear(width_in, rank, bias=False), nn.Linear(rank, width_out, bias=bias))
                report.update(rank=rank, matched_dimension=dimension)
            else:
                dimension = {'complex': 2, 'quaternion': 4, 'octonion': 8}[variant]
                mapping = ContiguousHyperDense(width_in // dimension, width_out // dimension, variant, bias=bias)
            with torch.no_grad():
                factors = list(mapping) if isinstance(mapping, nn.Sequential) else [mapping]
                for index, factor in enumerate(factors):
                    torch.random.default_generator.manual_seed(seed_for(path + f'/factor{index}'))
                    std = (variance / report['rank']) ** .25 if report['rank'] else math.sqrt(variance)
                    factor.weight.normal_(0, std)
                    if factor.bias is not None:
                        factor.bias.zero_()
            replacement = ChannelMap(mapping) if isinstance(original, nn.Conv1d) else mapping
            parent, _, key = path.rpartition('.')
            model.get_submodule(parent)._modules[key] = replacement
        assert digest_except(model, path) == untouched
        report['selected_parameters'] = sum(p.numel() for p in mapping.parameters())
        report['whole_model_parameters'] = sum(p.numel() for p in model.parameters())
        if report['rank']:
            report['hypercomplex_parameter_budget'] = width_in * width_out // report['matched_dimension'] + width_out * bias
            report['budget_gap'] = report['hypercomplex_parameter_budget'] - report['selected_parameters']
            report['budget_match'] = ('exact' if report['budget_gap'] == 0 else
                'below_budget' if report['budget_gap'] > 0 else 'nearest_positive_rank_exceeds_budget')
    return model, report


def real_equivalence(backbone, device='cpu', training=False):
    """Native-to-identically-weighted real surgery, including surrounding gradients."""
    native, report = build_case(backbone, 'native')
    edited, _ = build_case(backbone, 'real')
    path = report['selected_site']
    source = native.get_submodule(path)
    destination = edited.get_submodule(path)
    if isinstance(destination, ChannelMap):
        destination = destination.mapping
    with torch.no_grad():
        destination.weight.copy_(source.weight.squeeze(-1) if isinstance(source, nn.Conv1d) else source.weight)
        if source.bias is not None:
            destination.bias.copy_(source.bias)
    native.to(device).train(training); edited.to(device).train(training)
    x = torch.linspace(-1, 1, 256, device=device).reshape(2, 32, 4)
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    device = torch.device(device)
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        torch.manual_seed(1717)
        ya = native(a)
        torch.manual_seed(1717)
        yb = edited(b)
    torch.testing.assert_close(ya, yb, atol=2e-5, rtol=2e-4)
    ya.square().mean().backward(); yb.square().mean().backward()
    torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-4)
    left, right = dict(native.named_parameters()), dict(edited.named_parameters())
    for name, parameter in left.items():
        if name.startswith(path + '.'):
            continue
        assert (parameter.grad is None) == (right[name].grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, right[name].grad, atol=2e-5, rtol=2e-4)
    expected_weight_grad = source.weight.grad.squeeze(-1) if isinstance(source, nn.Conv1d) else source.weight.grad
    torch.testing.assert_close(expected_weight_grad, destination.weight.grad, atol=2e-5, rtol=2e-4)
    if source.bias is not None:
        torch.testing.assert_close(source.bias.grad, destination.bias.grad, atol=2e-5, rtol=2e-4)
    torch.optim.Adam(native.parameters(), lr=.001, eps=1e-7).step()
    torch.optim.Adam(edited.parameters(), lr=.001, eps=1e-7).step()
    for name, parameter in left.items():
        if not name.startswith(path + '.'):
            torch.testing.assert_close(parameter, right[name], atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(source.weight.squeeze(-1) if isinstance(source, nn.Conv1d) else source.weight,
                               destination.weight, atol=2e-5, rtol=2e-4)
    if source.bias is not None:
        torch.testing.assert_close(source.bias, destination.bias, atol=2e-5, rtol=2e-4)
    return dict(backbone=backbone, output_input_and_parameter_gradients='passed',
                paired_adam_step='passed', training=training, device=str(device))


def rectangular_reference_checks(device='cpu'):
    reports = []
    device = torch.device(device)
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        torch.manual_seed(1818)
        for width_in, width_out, bias in [(32, 32, True), (32, 64, True), (32, 64, False),
                                          (4096, 256, True), (16, 16, True), (32, 8, True)]:
            for algebra, dimension in [('complex', 2), ('quaternion', 4), ('octonion', 8)]:
                layer = ContiguousHyperDense(width_in // dimension, width_out // dimension, algebra, bias=bias).double().to(device)
                with torch.no_grad():
                    layer.weight.normal_(0, math.sqrt(2 / (width_in + width_out)))
                x = torch.linspace(-.7, .9, 2 * width_in, dtype=torch.float64, device=device).reshape(2, width_in).requires_grad_()
                actual = layer(x)
                expected = x @ explicit_real_matrix(layer)
                if bias:
                    expected = expected + layer.bias.reshape(-1)
                torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
                parameters = (x, layer.weight) + ((layer.bias,) if bias else ())
                actual_grads = torch.autograd.grad(actual.square().mean(), parameters)
                reference_grads = torch.autograd.grad(expected.square().mean(), parameters)
                for a, b in zip(actual_grads, reference_grads):
                    torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
                reports.append(dict(in_features=width_in, out_features=width_out, bias=bias,
                    algebra=algebra, forward_backward='passed', device=str(device)))
    return reports


def cpu_preflight():
    reports, equivalence = [], []
    for backbone in SITES:
        for training in (False, True):
            equivalence.append(real_equivalence(backbone, training=training))
        shared = None
        for variant in VARIANTS:
            before = torch.random.get_rng_state().clone()
            model, report = build_case(backbone, variant)
            assert torch.equal(before, torch.random.get_rng_state())
            if shared is None:
                shared = report['untouched_sha256']
            assert report['untouched_sha256'] == shared
            selected = model.get_submodule(report['selected_site'])
            calls = []
            handle = selected.register_forward_hook(lambda module, args, output: calls.append(tuple(output.shape)))
            model.eval()
            prediction = model(torch.linspace(-1, 1, 256).reshape(2, 32, 4))
            assert prediction.shape == (2, 5) and bool(torch.isfinite(prediction).all())
            prediction.square().mean().backward()
            handle.remove()
            assert calls and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in selected.parameters())
            assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            report.update(executed_calls=len(calls), cpu_forward_backward='passed')
            reports.append(report)
        print(f'{backbone}: eight variants passed', flush=True)
    return dict(protocol=PROTOCOL, passed=True, tested_configurations=len(reports),
        equivalence=equivalence, configurations=reports, rectangular_reference=rectangular_reference_checks(),
        pending=['CUDA gates', 'worker/CLI integration after active stage',
                 'FiLM spectral feasibility', 'costed stage admission'],
        cloud_ready=False)


if __name__ == '__main__':
    import json
    from pathlib import Path
    torch.set_num_threads(2)
    result = cpu_preflight()
    Path(__file__).with_name('replacement-prototype-preflight.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('equivalence', 'configurations')}, indent=2))
