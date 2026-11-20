r"""Multi-layer cross-channel hypercomplex blocks, controls, and TSLib surgery.

This module extends single-site ``HyperDense`` replacement into full multi-layer
cross-channel architectures and matched experimental controls:

- [`HyperRMSNorm`][hypercast4d.multilayer_hyperdense.HyperRMSNorm] — joint
  per-unit root-mean-square normalization over $d$ basis components that
  preserves relative hypercomplex phase/direction between stacked layers.
- [`HyperGate`][hypercast4d.multilayer_hyperdense.HyperGate] — scalar-gated
  hypercomplex activation scaling all $d$ basis components of each unit
  together via a learned basis+norm gate.
- [`BlockDiagonalLinear`][hypercast4d.multilayer_hyperdense.BlockDiagonalLinear]
  and [`LowRankLinear`][hypercast4d.multilayer_hyperdense.LowRankLinear] —
  parameter-matched uncoupled and low-rank controls isolating cross-component
  bilinear products $e_a e_b$ from parameter count alone.
- [`HyperChannelBlock`][hypercast4d.multilayer_hyperdense.HyperChannelBlock] and
  [`HyperChannelMixer`][hypercast4d.multilayer_hyperdense.HyperChannelMixer] —
  learned basis-routing stem ($C \to d \cdot M$) plus stacked cross-channel
  hypercomplex mixing blocks supporting both ``"compress"`` ($1/d$ parameters)
  and ``"isoparam"`` ($\sqrt{d}$-widened equal-parameter) regimes.
- [`build_multilayer_model`][hypercast4d.multilayer_hyperdense.build_multilayer_model] —
  unified builder for ``"hyper_channel_mixer"`` and multi-layer cross-channel
  TSLib variants (``"itransformer_cc"``, ``"tsmixer_cc"``, ``"crossformer_cc"``).

Examples
--------
>>> import torch
>>> from hypercast4d.multilayer_hyperdense import HyperChannelMixer
>>> model = HyperChannelMixer(
...     window=32, horizon=5, features=7, arm="quaternion", budget_mode="isoparam"
... )
>>> x = torch.randn(4, 32, 7)
>>> tuple(model(x).shape)
(4, 5)
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .algebras import ALGEBRAS, get_algebra
from .remaining_controls import ChannelMap, ContiguousHyperDense
from .upstream_models import UpstreamSequence

PROTOCOL = "multilayer-hyperdense-v1"

ALGEBRA_ARMS: tuple[str, ...] = tuple(ALGEBRAS.keys())
LOWRANK_ARMS: tuple[str, ...] = ("lowrank2", "lowrank4", "lowrank8")
BLOCKDIAG_ARMS: tuple[str, ...] = ("blockdiag2", "blockdiag4", "blockdiag8")
MULTILAYER_ARMS: tuple[str, ...] = (
    ("native", "real") + ALGEBRA_ARMS + LOWRANK_ARMS + BLOCKDIAG_ARMS
)
MULTILAYER_MODELS: tuple[str, ...] = (
    "hyper_channel_mixer",
    "itransformer_cc",
    "tsmixer_cc",
    "crossformer_cc",
)


def arm_dimension(arm: str) -> int:
    r"""Return the structural group/compression factor $d \in \{1, 2, 4, 8\}$ for ``arm``.

    Parameters
    ----------
    arm : str
        Operator arm identifier from ``MULTILAYER_ARMS``.

    Returns
    -------
    int
        ``1`` for ``"native"`` or ``"real"``, ``algebra.component_count`` for
        hypercomplex algebras, or the trailing divisor ``2``, ``4``, ``8`` for
        ``"lowrank*"`` and ``"blockdiag*"`` controls.
    """
    if arm in ("native", "real"):
        return 1
    if arm in ALGEBRAS:
        return get_algebra(arm).component_count
    if arm in LOWRANK_ARMS or arm in BLOCKDIAG_ARMS:
        return int(arm[-1])
    raise ValueError(f"Unsupported multilayer arm: {arm!r}")


def stream_seed(*parts: object) -> int:
    """Derive a deterministic 63-bit RNG seed from ``parts`` under ``PROTOCOL``."""
    payload = "/".join([PROTOCOL, *(str(p) for p in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def digest_except_many(module: nn.Module, excluded_prefixes: set[str]) -> str:
    """Compute a SHA-256 digest of all parameters and buffers outside ``excluded_prefixes``."""
    hasher = hashlib.sha256()
    excluded = tuple(sorted(excluded_prefixes))

    def _is_excluded(name: str) -> bool:
        return any(name == p or name.startswith(f"{p}.") for p in excluded)

    for kind, items in (
        ("param", module.named_parameters()),
        ("buffer", module.named_buffers()),
    ):
        for name, tensor in items:
            if _is_excluded(name):
                continue
            arr = tensor.detach().cpu().contiguous()
            hasher.update(f"{kind}:{name}:{tuple(arr.shape)}:{arr.dtype};".encode("utf-8"))
            hasher.update(arr.numpy().tobytes())
    return hasher.hexdigest()


class HyperRMSNorm(nn.Module):
    r"""Joint root-mean-square normalization across the $d$ components of each unit.

    Given an input of shape ``(*, d * num_units)`` in component-major layout,
    reshapes to ``(*, d, num_units)`` and normalizes each hypercomplex unit across
    its $d$ basis coordinates by a single shared scalar:

    $$\mathrm{HyperRMSNorm}(z)_{b, u} = \gamma_u \, \frac{z_{b, u}}
        {\sqrt{\frac{1}{d}\sum_{a=0}^{d-1} z_{a, u}^2 + \varepsilon}}.$$

    Scaling all $d$ basis coordinates equally preserves hypercomplex phase /
    orientation between stacked [`HyperDense`][hypercast4d.layers.HyperDense] layers.

    Parameters
    ----------
    component_count : int
        Number of real components $d \ge 1$ per hypercomplex unit.
    num_units : int
        Number of hypercomplex units $M \ge 1$ (total real width is ``d * M``).
    eps : float, default=1e-6
        Numerical stability epsilon inside the square root.
    """

    def __init__(
        self, component_count: int, num_units: int, eps: float = 1e-6
    ) -> None:
        super().__init__()
        if component_count < 1 or num_units < 1:
            raise ValueError("component_count and num_units must be positive")
        self.component_count = component_count
        self.num_units = num_units
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_units))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalize ``(*, d * num_units)`` jointly over each $d$-component tuple."""
        expected = self.component_count * self.num_units
        if inputs.shape[-1] != expected:
            raise ValueError(
                f"Expected last dim {expected}, got {inputs.shape[-1]}"
            )
        grouped = inputs.reshape(*inputs.shape[:-1], self.component_count, self.num_units)
        rms_inv = torch.rsqrt(grouped.square().mean(dim=-2, keepdim=True) + self.eps)
        scaled = grouped * rms_inv * self.weight.unsqueeze(0)
        return scaled.reshape(inputs.shape)


class HyperGate(nn.Module):
    r"""Scalar-gated hypercomplex nonlinearity preserving $d$-component orientation.

    Computes a real scalar gate $g_u$ for each hypercomplex unit $u \in \{0, \dots, M-1\}$
    from both a linear basis projection and the Euclidean tuple norm, then scales all
    $d$ components of unit $u$ together via $\mathrm{SiLU}(g_u)$:

    $$g_u = \sum_{a=0}^{d-1} W_{a, u} z_{a, u} + \beta_u \|z_{:, u}\|_2 + c_u,
      \qquad y_{b, u} = z_{b, u}\, \mathrm{SiLU}(g_u).$$

    Parameters
    ----------
    component_count : int
        Number of real basis components $d \ge 1$.
    num_units : int
        Number of hypercomplex units $M \ge 1$.
    eps : float, default=1e-6
        Epsilon inside the Euclidean norm $\sqrt{\sum_a z_{a, u}^2 + \varepsilon}$.
    """

    def __init__(
        self, component_count: int, num_units: int, eps: float = 1e-6
    ) -> None:
        super().__init__()
        if component_count < 1 or num_units < 1:
            raise ValueError("component_count and num_units must be positive")
        self.component_count = component_count
        self.num_units = num_units
        self.eps = eps
        self.gate_weight = nn.Parameter(torch.empty(component_count, num_units))
        self.norm_weight = nn.Parameter(torch.zeros(num_units))
        self.bias = nn.Parameter(torch.zeros(num_units))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize basis gate weights with unit-variance scaling and zero biases."""
        nn.init.normal_(self.gate_weight, mean=0.0, std=1.0 / math.sqrt(self.component_count))
        nn.init.zeros_(self.norm_weight)
        nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply tuple-preserving gated nonlinearity to ``(*, d * num_units)``."""
        expected = self.component_count * self.num_units
        if inputs.shape[-1] != expected:
            raise ValueError(
                f"Expected last dim {expected}, got {inputs.shape[-1]}"
            )
        grouped = inputs.reshape(*inputs.shape[:-1], self.component_count, self.num_units)
        linear_term = (grouped * self.gate_weight).sum(dim=-2)
        norm_term = torch.sqrt(grouped.square().sum(dim=-2) + self.eps) * self.norm_weight
        gate = F.silu(linear_term + norm_term + self.bias).unsqueeze(-2)
        return (grouped * gate).reshape(inputs.shape)


class BlockDiagonalLinear(nn.Module):
    r"""Parameter-matched uncoupled block-diagonal linear control ($d$ independent blocks).

    Applies $d$ independent ``(in_per_block, out_per_block)`` linear maps to the
    $d$ component slices without any cross-component terms ($e_a e_b$ for $a \neq b$).
    Uses the exact same weight count ($d \cdot n \cdot m$) and bias count ($d \cdot m$)
    as [`HyperDense`][hypercast4d.layers.HyperDense], isolating the effect of
    cross-component bilinear coupling from block sparsity.

    Parameters
    ----------
    num_blocks : int
        Number of independent diagonal blocks $d \ge 1$.
    in_per_block : int
        Input features $n \ge 1$ per block (total input width is ``d * n``).
    out_per_block : int
        Output features $m \ge 1$ per block (total output width is ``d * m``).
    bias : bool, default=True
        If ``True``, includes a learnable bias of shape ``(d, out_per_block)``.
    """

    def __init__(
        self,
        num_blocks: int,
        in_per_block: int,
        out_per_block: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_blocks < 1 or in_per_block < 1 or out_per_block < 1:
            raise ValueError("num_blocks, in_per_block, and out_per_block must be positive")
        self.num_blocks = num_blocks
        self.in_per_block = in_per_block
        self.out_per_block = out_per_block
        self.weight = nn.Parameter(torch.empty(num_blocks, in_per_block, out_per_block))
        self.bias = (
            nn.Parameter(torch.zeros(num_blocks, out_per_block)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize each block with fan-in/fan-out variance matching dense ``Linear``."""
        std = math.sqrt(2.0 / (self.in_per_block + self.out_per_block))
        nn.init.normal_(self.weight, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply independent block-diagonal projections to ``(*, d * in_per_block)``."""
        expected = self.num_blocks * self.in_per_block
        if inputs.shape[-1] != expected:
            raise ValueError(
                f"Expected last dim {expected}, got {inputs.shape[-1]}"
            )
        grouped = inputs.reshape(*inputs.shape[:-1], self.num_blocks, self.in_per_block)
        out = torch.einsum("...bi,bio->...bo", grouped, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*inputs.shape[:-1], self.num_blocks * self.out_per_block).contiguous()


class LowRankLinear(nn.Module):
    r"""Parameter-matched low-rank linear bottleneck ``Linear(in, r) -> Linear(r, out)``.

    Parameters
    ----------
    in_features : int
        Input width $D_{\mathrm{in}}$.
    out_features : int
        Output width $D_{\mathrm{out}}$.
    rank : int
        Bottleneck rank $r \ge 1$.
    bias : bool, default=True
        If ``True``, attaches a learnable bias to the outer projection.
    """

    def __init__(
        self, in_features: int, out_features: int, rank: int, bias: bool = True
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1 or rank < 1:
            raise ValueError("in_features, out_features, and rank must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize factors so expanded product entries match dense Glorot variance."""
        variance = 2.0 / (self.in_features + self.out_features)
        std = (variance / self.rank) ** 0.25
        nn.init.normal_(self.down.weight, mean=0.0, std=std)
        nn.init.normal_(self.up.weight, mean=0.0, std=std)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project ``(*, in_features)`` through rank-``r`` bottleneck to ``(*, out_features)``."""
        return self.up(self.down(inputs)).contiguous()


def build_controlled_operator(
    in_features: int,
    out_features: int,
    *,
    arm: str,
    bias: bool = True,
    seed: int | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    r"""Construct a variance-matched dense, hypercomplex, low-rank, or block-diagonal map.

    Parameters
    ----------
    in_features : int
        Real input width $D_{\mathrm{in}}$.
    out_features : int
        Real output width $D_{\mathrm{out}}$.
    arm : str
        Arm name in ``MULTILAYER_ARMS`` (excluding ``"native"``).
    bias : bool, default=True
        Whether to include an additive bias.
    seed : int or None, default=None
        Optional deterministic initialization seed.

    Returns
    -------
    tuple[torch.nn.Module, dict[str, Any]]
        ``(operator, metadata)`` with parameter counts and rank/block diagnostics.
    """
    if arm == "native" or arm not in MULTILAYER_ARMS:
        raise ValueError(f"Unsupported controlled operator arm: {arm!r}")
    d = arm_dimension(arm)
    variance = 2.0 / (in_features + out_features)
    rank: int | None = None
    budget_gap = 0

    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.random.default_generator.manual_seed(seed)
        if arm == "real":
            op: nn.Module = nn.Linear(in_features, out_features, bias=bias)
            with torch.no_grad():
                op.weight.normal_(0.0, math.sqrt(variance))
                if op.bias is not None:
                    op.bias.zero_()
        elif arm in ALGEBRAS:
            if in_features % d != 0 or out_features % d != 0:
                raise ValueError(
                    f"Arm {arm!r} (d={d}) requires widths divisible by {d}, "
                    f"got {in_features} -> {out_features}"
                )
            op = ContiguousHyperDense(
                in_features // d, out_features // d, arm, bias=bias
            )
            with torch.no_grad():
                op.weight.normal_(0.0, math.sqrt(variance))
                if op.bias is not None:
                    op.bias.zero_()
        elif arm in BLOCKDIAG_ARMS:
            if in_features % d != 0 or out_features % d != 0:
                raise ValueError(
                    f"Arm {arm!r} (d={d}) requires widths divisible by {d}, "
                    f"got {in_features} -> {out_features}"
                )
            op = BlockDiagonalLinear(
                d, in_features // d, out_features // d, bias=bias
            )
            with torch.no_grad():
                # Each block sums over in_features/d inputs; scale variance by d to
                # preserve output activation variance matching real/HyperDense.
                op.weight.normal_(0.0, math.sqrt(d * variance))
                if op.bias is not None:
                    op.bias.zero_()
        elif arm in LOWRANK_ARMS:
            target_weights = (in_features * out_features) // d
            rank = max(1, round(target_weights / (in_features + out_features)))
            budget_gap = rank * (in_features + out_features) - target_weights
            op = LowRankLinear(in_features, out_features, rank=rank, bias=bias)
            with torch.no_grad():
                if seed is not None:
                    torch.random.default_generator.manual_seed(
                        stream_seed(seed, "lowrank_down")
                    )
                std = (variance / rank) ** 0.25
                op.down.weight.normal_(0.0, std)
                if seed is not None:
                    torch.random.default_generator.manual_seed(
                        stream_seed(seed, "lowrank_up")
                    )
                op.up.weight.normal_(0.0, std)
                if op.up.bias is not None:
                    op.up.bias.zero_()
        else:  # pragma: no cover
            raise ValueError(f"Unhandled arm: {arm!r}")

    meta = {
        "arm": arm,
        "dimension": d,
        "in_features": in_features,
        "out_features": out_features,
        "parameters": sum(p.numel() for p in op.parameters()),
        "rank": rank,
        "lowrank_weight_budget_gap": budget_gap,
    }
    return op, meta


def resolve_latent_width(
    base_width: int, arm: str, budget_mode: str = "compress"
) -> int:
    r"""Compute latent real width $D$ for ``(base_width, arm, budget_mode)``.

    In ``"compress"`` mode, returns ``base_width`` unchanged (giving $1/d$ block
    parameter compression). In ``"isoparam"`` mode, scales width by $\sqrt{d}$
    rounded to the nearest multiple of ``max(8, d)`` so hypercomplex/control
    cross-channel blocks match ``"real"``'s parameter budget at ``base_width``.

    Parameters
    ----------
    base_width : int
        Baseline real latent width $D_0$ (must be divisible by 8).
    arm : str
        Operator arm from ``MULTILAYER_ARMS``.
    budget_mode : {"compress", "isoparam"}, default="compress"
        Whether to compress parameters by $1/d$ or widen latent channels by $\sqrt{d}$.

    Returns
    -------
    int
        Effective real latent width $D$ divisible by 8.
    """
    if base_width < 8 or base_width % 8 != 0:
        raise ValueError(f"base_width must be a positive multiple of 8, got {base_width}")
    if budget_mode == "compress":
        return base_width
    if budget_mode != "isoparam":
        raise ValueError(f"budget_mode must be 'compress' or 'isoparam', got {budget_mode!r}")
    d = arm_dimension(arm)
    if d == 1:
        return base_width
    scaled = round(base_width * math.sqrt(d) / 8.0) * 8
    return max(8, scaled)


class HyperChannelBlock(nn.Module):
    """Residual temporal + cross-channel hypercomplex mixing block on ``[B, T, D]``.

    Each block applies:

    1. **Temporal mixing** (shared architecture across all arms): pre-norm +
       two-layer MLP along the ``steps`` axis per latent channel.
    2. **Cross-channel mixing** (governed by ``arm``): ``HyperRMSNorm`` (or
       ``LayerNorm``) + controlled expansion ``D -> expansion * D`` +
       ``HyperGate`` (or ``GELU``) + controlled contraction ``expansion * D -> D``.

    Parameters
    ----------
    steps : int
        Sequence length $T$.
    latent_width : int
        Latent channel width $D$ (divisible by 8).
    arm : str
        Operator arm in ``MULTILAYER_ARMS`` (``"native"`` aliases ``"real"``).
    expansion : int, default=2
        Hidden width multiplier inside the cross-channel FFN.
    dropout : float, default=0.1
        Dropout probability inside temporal and cross-channel branches.
    algebra_norm : bool, default=True
        Use [`HyperRMSNorm`][hypercast4d.multilayer_hyperdense.HyperRMSNorm]
        when ``d > 1`` (falls back to ``nn.LayerNorm`` when ``False`` or ``d == 1``).
    algebra_gate : bool, default=True
        Use [`HyperGate`][hypercast4d.multilayer_hyperdense.HyperGate] when
        ``d > 1`` (falls back to ``nn.GELU`` when ``False`` or ``d == 1``).
    seed : int, default=0
        Deterministic initialization seed for this block.
    """

    def __init__(
        self,
        steps: int,
        latent_width: int,
        arm: str,
        *,
        expansion: int = 2,
        dropout: float = 0.1,
        algebra_norm: bool = True,
        algebra_gate: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        effective_arm = "real" if arm == "native" else arm
        d = arm_dimension(effective_arm)
        hidden_width = latent_width * expansion

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(stream_seed(seed, "temporal"))
            self.temporal_norm = nn.LayerNorm(latent_width)
            self.temporal_mlp = nn.Sequential(
                nn.Linear(steps, steps),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(steps, steps),
                nn.Dropout(dropout),
            )

        if algebra_norm and d > 1:
            self.channel_norm: nn.Module = HyperRMSNorm(d, latent_width // d)
        else:
            self.channel_norm = nn.LayerNorm(latent_width)

        self.fc1, self.fc1_meta = build_controlled_operator(
            latent_width,
            hidden_width,
            arm=effective_arm,
            bias=True,
            seed=stream_seed(seed, "fc1"),
        )
        if algebra_gate and d > 1:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(stream_seed(seed, "gate"))
                self.activation: nn.Module = HyperGate(d, hidden_width // d)
        else:
            self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout)
        self.fc2, self.fc2_meta = build_controlled_operator(
            hidden_width,
            latent_width,
            arm=effective_arm,
            bias=True,
            seed=stream_seed(seed, "fc2"),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply residual temporal and cross-channel mixing to ``[B, T, D]``."""
        y = self.temporal_norm(inputs).transpose(1, 2)
        inputs = inputs + self.temporal_mlp(y).transpose(1, 2)

        z = self.channel_norm(inputs)
        z = self.fc1(z)
        z = self.activation(z)
        z = self.dropout(z)
        z = self.fc2(z)
        return inputs + self.dropout(z)


class HyperChannelMixer(nn.Module):
    r"""Multi-layer cross-channel hypercomplex forecaster with learned basis routing.

    Maps centered relative inputs ``[B, window, features]`` through a learned
    basis router ``Linear(features, latent_width)``, ``num_layers`` stacked
    [`HyperChannelBlock`][hypercast4d.multilayer_hyperdense.HyperChannelBlock]
    modules, and a zero-initialized residual forecast head predicting
    ``[B, horizon]`` anchored at the last observed target value ``x[:, -1, 0]``.

    Parameters
    ----------
    window : int
        Historical context window length $T$.
    horizon : int
        Forecast horizon length $H$.
    features : int
        Number of input time series $C \ge 1$.
    arm : str, default="quaternion"
        Operator arm in ``MULTILAYER_ARMS``.
    base_width : int, default=64
        Baseline latent width $D_0$ (divisible by 8).
    num_layers : int, default=3
        Number of stacked cross-channel mixing blocks $L \ge 1$.
    expansion : int, default=2
        Cross-channel FFN expansion ratio.
    dropout : float, default=0.1
        Dropout rate inside blocks.
    budget_mode : {"compress", "isoparam"}, default="compress"
        ``"compress"`` keeps width ``base_width`` ($1/d$ block parameters);
        ``"isoparam"`` widens latent channels by $\sqrt{d}$ to match ``"real"``'s
        parameter budget.
    algebra_norm : bool, default=True
        Enable tuple-preserving ``HyperRMSNorm`` on $d > 1$ arms.
    algebra_gate : bool, default=True
        Enable tuple-preserving ``HyperGate`` on $d > 1$ arms.
    seed : int, default=0
        Deterministic initialization seed.
    """

    def __init__(
        self,
        window: int,
        horizon: int,
        features: int,
        arm: str = "quaternion",
        *,
        base_width: int = 64,
        num_layers: int = 3,
        expansion: int = 2,
        dropout: float = 0.1,
        budget_mode: str = "compress",
        algebra_norm: bool = True,
        algebra_gate: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if arm not in MULTILAYER_ARMS:
            raise ValueError(f"Unsupported arm: {arm!r}")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.window = window
        self.horizon = horizon
        self.features = features
        self.arm = arm
        self.base_width = base_width
        self.latent_width = resolve_latent_width(base_width, arm, budget_mode)
        self.num_layers = num_layers
        self.budget_mode = budget_mode

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(stream_seed(seed, "stem"))
            self.stem = nn.Linear(features, self.latent_width)

        self.blocks = nn.ModuleList(
            [
                HyperChannelBlock(
                    steps=window,
                    latent_width=self.latent_width,
                    arm=arm,
                    expansion=expansion,
                    dropout=dropout,
                    algebra_norm=algebra_norm,
                    algebra_gate=algebra_gate,
                    seed=stream_seed(seed, f"block{layer_idx}"),
                )
                for layer_idx in range(num_layers)
            ]
        )
        d = arm_dimension("real" if arm == "native" else arm)
        self.final_norm: nn.Module = (
            HyperRMSNorm(d, self.latent_width // d)
            if algebra_norm and d > 1
            else nn.LayerNorm(self.latent_width)
        )
        self.head = nn.Linear(window * self.latent_width, horizon)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict ``[B, horizon]`` target trajectory from ``[B, window, features]``."""
        if inputs.ndim != 3 or inputs.shape[1:] != (self.window, self.features):
            raise ValueError(
                f"Expected [batch, {self.window}, {self.features}], got {tuple(inputs.shape)}"
            )
        anchor = inputs[:, -1, 0:1]
        centered = inputs - inputs[:, -1:, :]
        latent = self.stem(centered)
        for block in self.blocks:
            latent = block(latent)
        latent = self.final_norm(latent)
        return self.head(latent.flatten(start_dim=1)) + anchor


class RoutedTSLibTransfer(nn.Module):
    """Relative-residual multi-layer TSLib wrapper with optional channel basis routing.

    When ``routed_width`` is set (used by ``tsmixer_cc`` so arbitrary channel
    counts $C$ map into an 8-divisible hypercomplex latent space), applies a
    learned ``Linear(features, routed_width)`` stem before the TSLib sequence
    model and predicts directly from the flattened ``[B, window, routed_width]``
    sequence representation.
    """

    def __init__(
        self,
        backbone: str,
        features: int,
        window: int,
        horizon: int,
        *,
        d_model: int = 64,
        num_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        routed_width: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.features = features
        self.window = window
        self.horizon = horizon
        self.routed_width = routed_width
        inner_width = routed_width if routed_width is not None else features

        self.stem: nn.Module = (
            nn.Linear(features, inner_width)
            if routed_width is not None
            else nn.Identity()
        )
        kw: dict[str, Any] = {
            "d_model": d_model,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        if backbone in {"itransformer", "crossformer"}:
            kw["n_heads"] = n_heads
        self.sequence = UpstreamSequence(backbone, window, inner_width, **kw)
        self.flatten = nn.Flatten(start_dim=1)
        self.output = nn.Linear(window * inner_width, horizon)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict ``[B, horizon]`` from ``[B, window, features]``."""
        if inputs.ndim != 3 or inputs.shape[1:] != (self.window, self.features):
            raise ValueError(
                f"Expected [batch, {self.window}, {self.features}], got {tuple(inputs.shape)}"
            )
        anchor = inputs[:, -1, 0:1]
        centered = inputs - inputs[:, -1:, :]
        latent = self.stem(centered)
        seq = self.sequence(latent)
        return self.output(self.flatten(seq)) + anchor


def cross_channel_sites(model_name: str, inner_model: nn.Module) -> list[str]:
    """Enumerate all cross-channel ``Linear``/``Conv1d(k=1)`` sites inside ``inner_model``.

    Parameters
    ----------
    model_name : {"itransformer_cc", "tsmixer_cc", "crossformer_cc"}
        Multi-site TSLib variant identifier.
    inner_model : torch.nn.Module
        Underlying ``UpstreamSequence.model`` instance.

    Returns
    -------
    list[str]
        Sorted dotted submodule names targeted for multi-layer cross-channel replacement.
    """
    sites: list[str] = []
    for name, mod in inner_model.named_modules():
        if not isinstance(mod, (nn.Linear, nn.Conv1d)):
            continue
        if isinstance(mod, nn.Conv1d) and mod.kernel_size != (1,):
            continue
        if model_name == "tsmixer_cc":
            if ".channel." in f".{name}.":
                sites.append(name)
        elif model_name == "itransformer_cc":
            if name.startswith("encoder.attn_layers.") and (
                ".attention." in name or name.endswith(".conv1") or name.endswith(".conv2")
            ):
                sites.append(name)
        elif model_name == "crossformer_cc":
            if (
                ".dim_sender." in name
                or ".dim_receiver." in name
                or ".self_attention.MLP2." in name
                or (name.startswith("encoder.") and ".MLP2." in name)
            ):
                sites.append(name)
        else:
            raise ValueError(f"Unknown multi-site TSLib model: {model_name!r}")
    if not sites:
        raise RuntimeError(f"No cross-channel sites matched for {model_name!r}")
    return sites


def build_multilayer_model(
    model_name: str,
    arm: str,
    seed: int,
    *,
    features: int,
    dataset_id: str = "synthetic",
    window: int = 32,
    horizon: int = 5,
    base_width: int = 64,
    num_layers: int = 3,
    budget_mode: str = "compress",
    algebra_norm: bool = True,
    algebra_gate: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    r"""Build a multi-layer cross-channel model and full provenance/budget report.

    Supports both the native [`HyperChannelMixer`][hypercast4d.multilayer_hyperdense.HyperChannelMixer]
    (``"hyper_channel_mixer"``) and multi-layer cross-channel TSLib surgeries
    (``"itransformer_cc"``, ``"tsmixer_cc"``, ``"crossformer_cc"``).

    Parameters
    ----------
    model_name : str
        One of ``MULTILAYER_MODELS``.
    arm : str
        One of ``MULTILAYER_ARMS``.
    seed : int
        Deterministic experiment seed.
    features : int
        Input channel count $C \ge 1$.
    dataset_id : str, default="synthetic"
        Dataset tag mixed into deterministic RNG streams.
    window : int, default=32
        Context window length.
    horizon : int, default=5
        Forecast horizon length.
    base_width : int, default=64
        Baseline latent/model width $D_0$ (divisible by 8).
    num_layers : int, default=3
        Number of stacked layers/blocks $L \ge 1$.
    budget_mode : {"compress", "isoparam"}, default="compress"
        Parameter regime: ``"compress"`` ($1/d$ block weights) or ``"isoparam"``
        ($\sqrt{d}$-widened equal-parameter latent width).
    algebra_norm : bool, default=True
        Use ``HyperRMSNorm`` inside ``HyperChannelMixer`` when ``d > 1``.
    algebra_gate : bool, default=True
        Use ``HyperGate`` inside ``HyperChannelMixer`` when ``d > 1``.

    Returns
    -------
    tuple[torch.nn.Module, dict[str, Any]]
        ``(model, report)`` containing replaced site list, parameter share,
        compression ratio, and untouched-parameter SHA-256 verification.
    """
    if model_name not in MULTILAYER_MODELS:
        raise ValueError(f"Unknown model_name {model_name!r}; expected one of {MULTILAYER_MODELS}")
    if arm not in MULTILAYER_ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {MULTILAYER_ARMS}")

    root_seed = stream_seed(dataset_id, model_name, seed)
    if model_name == "hyper_channel_mixer":
        model = HyperChannelMixer(
            window=window,
            horizon=horizon,
            features=features,
            arm=arm,
            base_width=base_width,
            num_layers=num_layers,
            budget_mode=budget_mode,
            algebra_norm=algebra_norm,
            algebra_gate=algebra_gate,
            seed=root_seed,
        )
        site_params = sum(
            sum(p.numel() for p in blk.fc1.parameters())
            + sum(p.numel() for p in blk.fc2.parameters())
            for blk in model.blocks
        )
        total_params = sum(p.numel() for p in model.parameters())
        report: dict[str, Any] = {
            "protocol": PROTOCOL,
            "dataset": dataset_id,
            "model_name": model_name,
            "arm": arm,
            "dimension": arm_dimension("real" if arm == "native" else arm),
            "budget_mode": budget_mode,
            "seed": seed,
            "features": features,
            "window": window,
            "horizon": horizon,
            "base_width": base_width,
            "effective_width": model.latent_width,
            "num_layers": num_layers,
            "algebra_norm": algebra_norm,
            "algebra_gate": algebra_gate,
            "replaced_site_count": 2 * num_layers,
            "parameters": total_params,
            "selected_parameters": site_params,
            "selected_parameter_share": site_params / max(total_params, 1),
        }
        return model, report

    backbone = model_name.removesuffix("_cc")
    effective_width = resolve_latent_width(base_width, arm, budget_mode)
    routed_width = effective_width if model_name == "tsmixer_cc" else None

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(stream_seed(root_seed, "native_init"))
        wrapper = RoutedTSLibTransfer(
            backbone=backbone,
            features=features,
            window=window,
            horizon=horizon,
            d_model=effective_width,
            num_layers=num_layers,
            n_heads=4,
            dropout=0.1,
            routed_width=routed_width,
        )

    rel_sites = cross_channel_sites(model_name, wrapper.sequence.model)
    full_sites = [f"sequence.model.{s}" for s in rel_sites]
    untouched = digest_except_many(wrapper, set(full_sites))

    selected_params = 0
    total_gap = 0
    if arm == "native":
        for full_site in full_sites:
            sub = wrapper.get_submodule(full_site)
            selected_params += sum(p.numel() for p in sub.parameters())
    else:
        for idx, full_site in enumerate(full_sites):
            old = wrapper.get_submodule(full_site)
            if isinstance(old, nn.Linear):
                nin, nout, has_bias = old.in_features, old.out_features, old.bias is not None
                is_conv1d = False
            elif isinstance(old, nn.Conv1d):
                nin, nout, has_bias = old.in_channels, old.out_channels, old.bias is not None
                is_conv1d = True
            else:  # pragma: no cover
                raise TypeError(f"Unexpected site module type: {type(old)}")

            new_op, meta = build_controlled_operator(
                nin,
                nout,
                arm=arm,
                bias=has_bias,
                seed=stream_seed(root_seed, f"site{idx}", full_site),
            )
            replacement: nn.Module = ChannelMap(new_op) if is_conv1d else new_op
            parent, _, key = full_site.rpartition(".")
            wrapper.get_submodule(parent)._modules[key] = replacement
            selected_params += meta["parameters"]
            total_gap += abs(int(meta["lowrank_weight_budget_gap"]))

    if digest_except_many(wrapper, set(full_sites)) != untouched:
        raise RuntimeError("Untouched parameters mutated during multi-site replacement")

    total_params = sum(p.numel() for p in wrapper.parameters())
    report = {
        "protocol": PROTOCOL,
        "dataset": dataset_id,
        "model_name": model_name,
        "backbone": backbone,
        "arm": arm,
        "dimension": arm_dimension("real" if arm == "native" else arm),
        "budget_mode": budget_mode,
        "seed": seed,
        "features": features,
        "window": window,
        "horizon": horizon,
        "base_width": base_width,
        "effective_width": effective_width,
        "routed_width": routed_width,
        "num_layers": num_layers,
        "replaced_site_count": len(full_sites),
        "sites": full_sites,
        "untouched_sha256": untouched,
        "parameters": total_params,
        "selected_parameters": selected_params,
        "selected_parameter_share": selected_params / max(total_params, 1),
        "total_abs_lowrank_budget_gap": total_gap,
    }
    return wrapper, report
