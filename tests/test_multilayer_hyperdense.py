"""Unit tests for multi-layer cross-channel HyperDense blocks and controls."""

from __future__ import annotations

import pytest
import torch

from hypercast4d.multilayer_hyperdense import (
    ALGEBRA_ARMS,
    MULTILAYER_MODELS,
    BlockDiagonalLinear,
    HyperChannelMixer,
    HyperGate,
    HyperRMSNorm,
    build_controlled_operator,
    build_multilayer_model,
    resolve_latent_width,
)


def test_hyper_rms_norm_preserves_component_direction() -> None:
    norm = HyperRMSNorm(component_count=4, num_units=6)
    x = torch.randn(3, 8, 24)
    y1 = norm(x)
    # Scaling the full token vector by a positive scalar alpha leaves HyperRMSNorm invariant
    # and preserves every unit's relative direction across its 4 basis components
    alpha = torch.rand(3, 8, 1) * 4.0 + 0.5
    y2 = norm(x * alpha)
    torch.testing.assert_close(y1, y2, atol=1e-4, rtol=1e-4)
    xg = x.reshape(3, 8, 4, 6)
    yg = y1.reshape(3, 8, 4, 6)
    ratios = yg / (xg + 1e-8)
    for b in range(1, 4):
        torch.testing.assert_close(ratios[:, :, 0, :], ratios[:, :, b, :], atol=1e-4, rtol=1e-4)


def test_hyper_gate_scales_basis_components_uniformly_per_unit() -> None:
    gate = HyperGate(component_count=4, num_units=5)
    x = torch.randn(2, 6, 20) + 0.5
    y = gate(x)
    xg = x.reshape(2, 6, 4, 5)
    yg = y.reshape(2, 6, 4, 5)
    # Ratio y[b] / x[b] must be identical across all 4 basis components b=0..3
    ratios = yg / xg
    for b in range(1, 4):
        torch.testing.assert_close(ratios[:, :, 0, :], ratios[:, :, b, :], atol=1e-5, rtol=1e-5)
    # Bounded sigmoid gating guarantees ||y||_2 <= ||x||_2 even at large input scale
    large = torch.randn(2, 6, 20) * 100.0
    assert gate(large).detach().norm().item() <= large.norm().item() + 1e-4


@pytest.mark.parametrize(
    ("hd_arm", "bd_arm", "lr_arm", "d"),
    [
        ("complex", "blockdiag2", "lowrank2", 2),
        ("split_complex", "blockdiag2", "lowrank2", 2),
        ("quaternion", "blockdiag4", "lowrank4", 4),
        ("coquaternion", "blockdiag4", "lowrank4", 4),
        ("cl11", "blockdiag4", "lowrank4", 4),
        ("octonion", "blockdiag8", "lowrank8", 8),
    ],
)
def test_controlled_operators_match_parameter_budgets_and_variances(
    hd_arm: str, bd_arm: str, lr_arm: str, d: int
) -> None:
    real_op, real_meta = build_controlled_operator(64, 128, arm="real", seed=101)
    hd_op, hd_meta = build_controlled_operator(64, 128, arm=hd_arm, seed=101)
    bd_op, bd_meta = build_controlled_operator(64, 128, arm=bd_arm, seed=101)
    lr_op, lr_meta = build_controlled_operator(64, 128, arm=lr_arm, seed=101)

    assert isinstance(bd_op, BlockDiagonalLinear)
    # Exact parameter parity between HyperDense and BlockDiagonalLinear
    assert hd_meta["parameters"] == bd_meta["parameters"]
    expected_weights = (64 * 128) // d
    assert hd_meta["parameters"] == expected_weights + 128
    # LowRank integer rank must be the closest possible integer rank (gap <= (in + out) // 2)
    assert abs(lr_meta["parameters"] - hd_meta["parameters"]) <= (64 + 128) // 2

    # Check forward output activation variance parity on unit-variance Gaussian inputs
    torch.manual_seed(7)
    x = torch.randn(512, 64)
    var_real = real_op(x).detach().var().item()
    var_hd = hd_op(x).detach().var().item()
    var_bd = bd_op(x).detach().var().item()
    var_lr = lr_op(x).detach().var().item()
    for v in (var_hd, var_bd, var_lr):
        assert 0.5 * var_real < v < 1.8 * var_real


@pytest.mark.parametrize("features", [7, 21, 321])
def test_hyper_channel_mixer_compress_and_isoparam_modes(features: int) -> None:
    real_m, real_rep = build_multilayer_model(
        "hyper_channel_mixer", "real", 42, features=features, budget_mode="compress"
    )
    quat_comp, comp_rep = build_multilayer_model(
        "hyper_channel_mixer", "quaternion", 42, features=features, budget_mode="compress"
    )
    quat_iso, iso_rep = build_multilayer_model(
        "hyper_channel_mixer", "quaternion", 42, features=features, budget_mode="isoparam"
    )

    assert resolve_latent_width(64, "quaternion", "isoparam") == 128
    # In compress mode, cross-channel FFN weights are ~1/4 of real
    assert comp_rep["selected_parameters"] < 0.30 * real_rep["selected_parameters"]
    # In isoparam mode, d=4 at width 128 matches d=1 real at width 64 nearly 1:1
    ratio = iso_rep["selected_parameters"] / real_rep["selected_parameters"]
    assert 0.98 <= ratio <= 1.05

    x = torch.randn(4, 32, features)
    for m in (real_m, quat_comp, quat_iso):
        out = m(x)
        assert out.shape == (4, 5)
        # Zero-initialized residual head starts at exact persistence forecast!
        expected_anchor = x[:, -1, 0:1].expand(4, 5)
        torch.testing.assert_close(out, expected_anchor)


@pytest.mark.parametrize("model_name", MULTILAYER_MODELS)
@pytest.mark.parametrize("arm", ["native", "real", "quaternion", "octonion", "lowrank4", "blockdiag4"])
def test_all_multilayer_models_forward_backward_and_high_site_coverage(
    model_name: str, arm: str
) -> None:
    model, report = build_multilayer_model(
        model_name,
        arm,
        seed=99,
        features=7,
        base_width=32,
        num_layers=2,
        budget_mode="compress",
    )
    assert report["replaced_site_count"] >= 4
    if arm == "real":
        # Multi-layer cross-channel sites cover a major fraction of the model (unlike 1.3% in single-site!)
        assert report["selected_parameter_share"] > 0.25

    x = torch.randn(3, 32, 7, requires_grad=True)
    y = model(x)
    assert y.shape == (3, 5)
    loss = (y - torch.randn(3, 5)).abs().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_all_six_algebras_supported_in_hyper_channel_mixer() -> None:
    x = torch.randn(2, 32, 8)
    for alg in ALGEBRA_ARMS:
        model = HyperChannelMixer(32, 5, 8, arm=alg, base_width=32, num_layers=2)
        out = model(x)
        assert out.shape == (2, 5)
