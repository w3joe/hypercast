"""First-scale FiLM spectral replacement; paired with remaining-internal-v1."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .remaining_controls import ContiguousHyperDense, VARIANTS, digest_except
from hypercast4d.architecture import build_architecture, presets
from hypercast4d.experiment_controls import explicit_real_matrix

SITE = 'layers.0.model.spec_conv_1.0'


class SpectralRealBlock(nn.Module):
    """Exact real-block implementation retaining the native complex parameter ties."""
    def __init__(self, original):
        super().__init__()
        self.weights_real = nn.Parameter(original.weights_real.detach().clone())
        self.weights_imag = nn.Parameter(original.weights_imag.detach().clone())
        self.modes, self.out_channels = original.modes, original.out_channels

    def apply_mode(self, a, k):
        real, imag = self.weights_real[:, :, k], self.weights_imag[:, :, k]
        weight = torch.cat([torch.cat([real.T, -imag.T], dim=1),
                            torch.cat([imag.T, real.T], dim=1)], dim=0)
        return F.linear(torch.cat([a.real, a.imag], dim=-1), weight)

    def forward(self, x):
        spectrum = torch.fft.rfft(x)
        transformed = []
        for k in range(self.modes):
            y = self.apply_mode(spectrum[..., k], k)
            transformed.append(torch.complex(y[..., :self.out_channels], y[..., self.out_channels:]))
        # Match the upstream operator's explicit complex64 output buffer and frequency mask.
        result = torch.zeros(*x.shape[:2], self.out_channels, x.shape[-1] // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        result[..., :self.modes] = torch.stack(transformed, dim=-1)
        return torch.fft.irfft(result, n=x.shape[-1])


class StructuredSpectral(SpectralRealBlock):
    """Replace each existing frequency map while preserving FFT, mask and inverse FFT."""
    def __init__(self, original, variant):
        nn.Module.__init__(self)
        self.modes, self.out_channels = original.modes, original.out_channels
        width_in, width_out = 2 * original.in_channels, 2 * original.out_channels
        variance = 2 / (width_in + width_out)
        self.maps = nn.ModuleList()
        for k in range(self.modes):
            if variant == 'real':
                layer = nn.Linear(width_in, width_out, bias=False)
                rank = None
            elif variant.startswith('lowrank'):
                dimension = int(variant[len('lowrank'):])
                rank = (width_in * width_out) // (dimension * (width_in + width_out))
                assert rank > 0
                layer = nn.Sequential(nn.Linear(width_in, rank, bias=False), nn.Linear(rank, width_out, bias=False))
            else:
                dimension = {'complex': 2, 'quaternion': 4, 'octonion': 8}[variant]
                layer = ContiguousHyperDense(width_in // dimension, width_out // dimension, variant, bias=False)
                rank = None
            with torch.no_grad():
                factors = list(layer) if rank else [layer]
                for factor in factors:
                    factor.weight.normal_(0, (variance / rank) ** .25 if rank else math.sqrt(variance))
            self.maps.append(layer)

    def apply_mode(self, a, k):
        return self.maps[k](torch.cat([a.real, a.imag], dim=-1))


def native(seed=907, window=32, horizon=5, fold=1):
    with torch.random.fork_rng(devices=[]):
        key = f'remaining-internal-v1/film/{seed}/{window}/{horizon}/{fold}/native'
        torch.random.default_generator.manual_seed(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % (2**63-1))
        spec = deepcopy(next(p for p in presets() if p['preset_id'] == 'tslib-film'))
        return build_architecture(spec, window, horizon)


def build_case(variant, seed=907, window=32, horizon=5, fold=1):
    if variant not in VARIANTS:
        raise ValueError('Unknown FiLM variant')
    model = native(seed, window, horizon, fold)
    original = model.get_submodule(SITE)
    if variant != 'native':
        with torch.random.fork_rng(devices=[]):
            key = f'remaining-internal-v1/film/{seed}/{window}/{horizon}/{fold}/spectral'
            torch.random.default_generator.manual_seed(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % (2**63-1))
            model.layers[0].model.spec_conv_1[0] = StructuredSpectral(original, variant)
    return model


def equivalence(training=False, device='cpu'):
    left, right = native(), native()
    original = left.get_submodule(SITE)
    right.layers[0].model.spec_conv_1[0] = SpectralRealBlock(original)
    left.to(device).train(training); right.to(device).train(training)
    a = torch.linspace(-1, 1, 256, device=device).reshape(2, 32, 4).requires_grad_()
    b = a.detach().clone().requires_grad_()
    ya, yb = left(a), right(b)
    torch.testing.assert_close(ya, yb, atol=1e-6, rtol=1e-4)
    ya.square().mean().backward(); yb.square().mean().backward()
    torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)
    rp = dict(right.named_parameters())
    for name, p in left.named_parameters():
        assert (p.grad is None) == (rp[name].grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, rp[name].grad, atol=1e-6, rtol=1e-4)
    torch.optim.Adam(left.parameters(), lr=.001, eps=1e-7).step()
    torch.optim.Adam(right.parameters(), lr=.001, eps=1e-7).step()
    for name, p in left.named_parameters():
        torch.testing.assert_close(p, rp[name], atol=1e-6, rtol=1e-4)
    return dict(training=training, forward_gradients_and_adam='passed')


def structured_reference(device='cpu'):
    """Independent real matrices for all sixteen selected frequency maps."""
    reports = []
    for variant in ('complex', 'quaternion', 'octonion'):
        module = build_case(variant).get_submodule(SITE).double().to(device)
        x = torch.linspace(-.7, .8, 256*32, dtype=torch.float64, device=device).reshape(1, 1, 256, 32).requires_grad_()
        actual = module(x)
        spectrum = torch.fft.rfft(x)
        transformed = []
        for k, mapping in enumerate(module.maps):
            a = spectrum[..., k]
            y = torch.cat([a.real, a.imag], dim=-1) @ explicit_real_matrix(mapping)
            transformed.append(torch.complex(y[..., :256], y[..., 256:]))
        out = torch.zeros(1, 1, 256, 17, dtype=torch.cfloat, device=device)
        out[..., :16] = torch.stack(transformed, dim=-1)
        expected = torch.fft.irfft(out, n=32)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        parameters = (x, *module.parameters())
        a = torch.autograd.grad(actual.square().mean(), parameters)
        b = torch.autograd.grad(expected.square().mean(), parameters)
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
        reports.append(dict(variant=variant, independent_spectral_forward_and_gradients='passed'))
    return reports


def preflight():
    checks = [equivalence(training) for training in (False, True)]
    shared = None
    rows = []
    for variant in VARIANTS:
        model = build_case(variant)
        digest = digest_except(model, SITE)
        if shared is None:
            shared = digest
        assert shared == digest
        model.eval()
        result = model(torch.linspace(-1, 1, 256).reshape(2, 32, 4))
        assert result.shape == (2, 5) and bool(torch.isfinite(result).all())
        result.square().mean().backward()
        selected = model.get_submodule(SITE)
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in selected.parameters())
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        rows.append(dict(variant=variant, selected_parameters=sum(p.numel() for p in selected.parameters()),
            whole_model_parameters=sum(p.numel() for p in model.parameters()), untouched_sha256=digest,
            cpu_forward_backward='passed'))
        print(f'FiLM spectral {variant}: passed', flush=True)
    return dict(passed=True, configurations=rows, equivalence=checks, site=SITE,
        real_dimensions_per_frequency=[512, 512], frequency_maps=16, bias=False,
        hypothesis='Structured replacement of first-scale native complex spectral maps; distinct from ordinary dense substitution',
        native_is_already_complex=True, component_layout='256 real coordinates followed by 256 imaginary coordinates',
        pending=['CUDA gates', 'independent structured spectral checks', 'versioned initialization integration', 'costed admission'],
        cloud_ready=False)

