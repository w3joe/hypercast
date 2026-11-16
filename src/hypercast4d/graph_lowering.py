"""Source-symbolic PyTorch FX lowering for v1 pipeline and TSLib models.

Rather than recording a brittle data-dependent execution trace, this module
symbolically traces models with ``torch.fx`` while substituting explicit
registered helper kernels for shape/FFT/folding routines that standard FX
proxies cannot step through. Trainable convolutions, projections, and attention
layers remain distinct ``call_module`` nodes in the lowered graph.
"""

from __future__ import annotations

from contextlib import ExitStack
import math
import threading
from typing import Any
from unittest.mock import patch

import einops
import torch
from torch import nn
from torch.fx import GraphModule, Proxy, Tracer

from .upstream_models import UpstreamSequence


LOWERING_LOCK = threading.RLock()


def period_select(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-``k`` dominant FFT periods and their batch amplitude weights."""
    amplitudes = torch.fft.rfft(x, dim=1).abs()
    frequency = amplitudes.mean(0).mean(-1)
    frequency[0] = -torch.inf
    indices = frequency.topk(k).indices
    return x.shape[1] // indices, amplitudes.mean(-1)[:, indices]


def period_fold(x: torch.Tensor, period: int | torch.Tensor) -> torch.Tensor:
    """Fold a ``[B, T, C]`` sequence into a 2D ``[B, C, num_periods, period]`` grid."""
    period = int(period)
    batch, length, width = x.shape
    padded = math.ceil(length / period) * period
    x = torch.nn.functional.pad(x, (0, 0, 0, padded - length))
    return x.reshape(batch, padded // period, period, width).permute(0, 3, 1, 2).contiguous()


def period_unfold(x: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Unfold a 2D period grid back to the original ``[B, T, C]`` length."""
    return x.permute(0, 2, 3, 1).reshape(original.shape[0], -1, original.shape[2])[:, :original.shape[1]]


def period_attention_fold(x: torch.Tensor, period: int | torch.Tensor) -> torch.Tensor:
    """Fold ``[B, T, C]`` into period-aligned ``[B * num_periods, period, C]`` windows."""
    folded = period_fold(x, period)
    return folded.permute(0, 2, 3, 1).reshape(-1, int(period), x.shape[2])


def period_attention_unfold(x: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Restore ``[B * num_periods, period, C]`` attention outputs to ``[B, T, C]``."""
    return x.reshape(original.shape[0], -1, original.shape[2])[:, :original.shape[1]]


def period_combine(outputs: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """Aggregate multi-period branch outputs using softmax-normalized FFT weights."""
    stacked = torch.stack(outputs, dim=-1)
    weights = torch.softmax(weights, dim=1)[:, None, None, :]
    return (stacked * weights).sum(-1)


def even_pad(x: torch.Tensor) -> torch.Tensor:
    """Replicate the final step when sequence length ``T`` is odd (for SCINet splitting)."""
    return torch.cat([x, x[:, -1:, :]], dim=1) if x.shape[1] % 2 else x


def segment_merge_layout(x: torch.Tensor, window: int) -> torch.Tensor:
    """Concatenate ``window`` strided sub-segments along the channel axis (Crossformer)."""
    remainder = x.shape[2] % window
    if remainder:
        x = torch.cat((x, x[:, :, -(window - remainder):, :]), dim=-2)
    return torch.cat([x[:, :, i::window, :] for i in range(window)], dim=-1)


def even_trim(x: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Trim SCINet odd-branch length when ``original`` had an odd sequence length."""
    return x[:, :original.shape[1] // 2, :] if original.shape[1] % 2 else x


def interleave(even: torch.Tensor, odd: torch.Tensor) -> torch.Tensor:
    """Re-interleave even and odd temporal subsequences in SCINet tree blocks."""
    length = min(even.shape[1], odd.shape[1])
    result = torch.stack([even[:, :length], odd[:, :length]], dim=2).flatten(1, 2)
    return torch.cat([result, even[:, length:]], dim=1)


def multi_decompose(x: torch.Tensor, kernels: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply TSLib multi-kernel moving-average decomposition on ``x``'s device."""
    from ._vendor.tslib.layers.Autoformer_EncDec import series_decomp_multi
    return series_decomp_multi(kernels).to(x.device)(x)


def positional_encoding(x: torch.Tensor, inverse_timescales: torch.Tensor) -> torch.Tensor:
    """Compute sinusoidal positional encodings matching SCINet's buffer layout."""
    scaled = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)[:, None] * inverse_timescales[None, :]
    signal = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)
    return signal[None, :, :x.shape[2]]


def _has_proxy(value: Any) -> bool:
    """Return ``True`` if ``value`` or any nested element is a ``torch.fx.Proxy``."""
    if isinstance(value, Proxy):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_proxy(x) for x in value)
    if isinstance(value, dict):
        return any(_has_proxy(x) for x in value.values())
    return False


class SourceTracer(Tracer):
    """Custom ``torch.fx.Tracer`` preserving submodule ownership and TSLib idioms.

    Records the enclosing ``layers.<i>`` submodule path in ``node.meta["owner"]``
    so the graph converter can group lowered operations into collapsible stages,
    and rewrites non-FX-traceable TSLib blocks (``TimesBlock``, ``ScaleGraphBlock``,
    ``SCINet``, ``SegMerging``, ``StandardNorm``) in terms of registered kernels
    while keeping their trainable leaf layers exposed.
    """

    def __init__(self) -> None:
        super().__init__(autowrap_functions=(einops.rearrange, einops.repeat))
        self.norm_stats: dict[int, tuple[Any, Any]] = {}
        self.owners: list[str] = []

    def create_proxy(self, *args: Any, **kwargs: Any) -> Proxy:
        """Create an FX proxy annotated with the active submodule ``owner`` path."""
        result = super().create_proxy(*args, **kwargs)
        if self.owners:
            result.node.meta['owner'] = self.owners[-1]
        return result

    def kernel(self, function: Any, *args: Any, **kwargs: Any) -> Proxy:
        """Emit a ``call_function`` proxy node for a registered helper kernel."""
        return self.create_proxy('call_function', function, args, kwargs)

    def is_leaf_module(self, module: nn.Module, path: str) -> bool:
        """Treat ``HyperDense`` and custom spectral/embedding modules as atomic leaves."""
        if type(module).__name__ in {'HyperDense', 'HiPPO_LegT', 'SpectralConv1d', 'PositionalEmbedding', 'FixedEmbedding'}:
            return True
        return super().is_leaf_module(module, path)

    def call_module(
        self, module: nn.Module, forward: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Track the active submodule path on ``self.owners`` while lowering ``module``."""
        path = self.submodule_paths.get(module, self.owners[-1] if self.owners else '')
        self.owners.append(path)
        try:
            return self.lower_module(module, forward, args, kwargs)
        finally:
            self.owners.pop()

    def lower_module(
        self, module: nn.Module, forward: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Lower TSLib-specific composite blocks using explicit FX-safe helper kernels."""
        kind = type(module).__name__
        package = type(module).__module__
        if isinstance(module, nn.AvgPool1d) and module not in self.submodule_paths:
            return self.kernel(torch.nn.functional.avg_pool1d, args[0], module.kernel_size,
                               module.stride, module.padding, module.ceil_mode, module.count_include_pad)
        if isinstance(module, UpstreamSequence):
            x = module.prepare_input(args[0])
            return module.model(x, None, torch.zeros_like(x), None)[:, -module.steps:, :]
        if kind == 'series_decomp_multi':
            return self.kernel(multi_decompose, args[0], module.kernel_size)
        if kind == 'SegMerging' and package.endswith('Crossformer_EncDec'):
            return module.linear_trans(module.norm(self.kernel(segment_merge_layout, args[0], module.win_size)))
        if kind == 'Normalize' and package.endswith('StandardNorm'):
            x, mode = args
            if module.non_norm:
                return x
            if mode == 'norm':
                mean = x[:, -1:, :] if module.subtract_last else x.mean(1, keepdim=True).detach()
                std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + module.eps).detach()
                self.norm_stats[id(module)] = mean, std
                x = (x - mean) / std
                return x * module.affine_weight + module.affine_bias if module.affine else x
            mean, std = self.norm_stats[id(module)]
            if module.affine:
                x = (x - module.affine_bias) / (module.affine_weight + module.eps * module.eps)
            return x * std + mean
        if kind == 'TimesBlock':
            x = args[0]
            periods, weights = self.kernel(period_select, x, module.k)
            outputs = [self.kernel(period_unfold, module.conv(self.kernel(period_fold, x, periods[i])), x)
                       for i in range(module.k)]
            return self.kernel(period_combine, outputs, weights) + x
        if kind == 'ScaleGraphBlock':
            x = args[0]
            periods, weights = self.kernel(period_select, x, module.k)
            outputs = []
            for i in range(module.k):
                x = module.gconv[i](x)
                out = self.kernel(period_attention_fold, x, periods[i])
                out = module.gelu(module.norm(module.att0(out)))
                outputs.append(self.kernel(period_attention_unfold, out, x))
            return self.kernel(period_combine, outputs, weights) + x
        if kind == 'SCINet' and package.endswith('.SCINet'):
            x = args[0]
            even, odd = module.working_block(self.kernel(even_pad, x))
            odd = self.kernel(even_trim, odd, x)
            if module.current_level:
                even, odd = module.SCINet_Tree_even(even), module.SCINet_Tree_odd(odd)
            return self.kernel(interleave, even, odd)
        if kind == 'Model' and package.endswith('.SCINet'):
            x = args[0]
            mean = x.mean(1, keepdim=True).detach()
            std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
            x = (x - mean) / std
            x = x + self.kernel(positional_encoding, x, module.inv_timescales)
            out = module.projection_1(module.sci_net_1(x) + x)
            if module.num_stacks != 1:
                combined = torch.cat((x, out), dim=1)
                out = module.projection_2(module.sci_net_2(combined) + combined)
            return torch.cat([torch.zeros_like(args[0]), out * std + mean], dim=1)
        return super().call_module(module, forward, args, kwargs)


def lower_source(model: nn.Module) -> GraphModule:
    """Symbolically lower ``model`` into a functional ``torch.fx.GraphModule``.

    Rewrites in-place tensor methods (``masked_fill_``, ``add_``, ``sub_``,
    ``mul_``, ``div_``) into out-of-place functional calls so dataflow edges are
    explicit, and binds functional dropout calls to the live ``training``
    attribute rather than freezing ``model.training`` at trace time.

    Parameters
    ----------
    model : torch.nn.Module
        Compiled Schema v1 ``ExperimentalForecaster`` to lower.

    Returns
    -------
    torch.fx.GraphModule
        Symbolically traced graph module suitable for Schema v2 graph conversion.
    """
    with LOWERING_LOCK, ExitStack() as stack:
        tracer = SourceTracer()
        for name in ('zeros', 'ones', 'arange', 'empty', 'full', 'tensor', 'eye', 'reshape'):
            original = getattr(torch, name)
            def factory(*args, _original=original, **kwargs):
                if _has_proxy((args, kwargs)):
                    return tracer.kernel(_original, *args, **kwargs)
                return _original(*args, **kwargs)
            stack.enter_context(patch.object(torch, name, factory))
        graph = tracer.trace(model)
        # Make mutation dependencies explicit: attention reads the result of
        # masked_fill_, even when the source ignores that method's return value.
        sequence = list(graph.nodes)
        for index, node in enumerate(sequence):
            if node.op == 'call_method' and node.target in {'masked_fill_', 'add_', 'sub_', 'mul_', 'div_'}:
                base = node.args[0]
                node.target = str(node.target)[:-1]
                for later in sequence[index + 1:]:
                    later.replace_input_with(base, node)
        # Functional dropout must follow GraphModule.train()/eval(), not capture
        # the source module's mode at conversion time.
        for node in list(graph.nodes):
            if node.op == 'call_function' and node.target in {
                torch.nn.functional.dropout, torch.nn.functional.dropout1d,
                torch.nn.functional.dropout2d, torch.nn.functional.dropout3d,
            }:
                with graph.inserting_before(node):
                    training = graph.get_attr('training')
                if len(node.args) > 2:
                    node.args = (*node.args[:2], training, *node.args[3:])
                else:
                    node.kwargs = {**node.kwargs, 'training': training}
    return GraphModule(model, graph)
