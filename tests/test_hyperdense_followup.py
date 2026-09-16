"""Correctness checks for the isolated, inference-only optimization prototypes."""
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('followup_profile', Path(__file__).resolve().parents[1]/'scripts/profile_hyperdense_followup.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('algebra,dimension', [('complex',2),('quaternion',4),('octonion',8)])
@pytest.mark.parametrize('bias', [False,True])
def test_frozen_export_preserves_map_and_input_gradient(algebra, dimension, bias):
    torch.manual_seed(64)
    original = module.ContiguousHyperDense(16//dimension,24//dimension,algebra,bias=bias).double()
    for implementation in [module.CachedConstants(original),module.export_dense(original)]:
        x=torch.randn(2,3,16,dtype=torch.float64,requires_grad=True)
        y=x.detach().clone().requires_grad_()
        expected=original(x); actual=implementation(y)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        expected.square().sum().backward();actual.square().sum().backward()
        torch.testing.assert_close(x.grad,y.grad,atol=1e-11,rtol=1e-11)
        assert all(not p.requires_grad for p in implementation.parameters())
        snapshot=implementation(y.detach()).detach().clone()
        with torch.no_grad():original.weight.add_(.125)
        torch.testing.assert_close(implementation(y.detach()),snapshot,atol=0,rtol=0)
        with torch.no_grad():original.weight.sub_(.125)


def test_conversion_keeps_selected_layer_live_in_full_model():
    torch.set_num_threads(2)
    torch.manual_seed(73)
    original,_=module.build_case('micn','octonion',seed=818)
    original.eval(); dense=module.replace_layers(module.deepcopy(original),'dense').eval()
    x=torch.randn(2,32,4)
    with torch.inference_mode():
        y=original(x)
        torch.testing.assert_close(dense(x),y,atol=2e-5,rtol=2e-4)
        for layer in original.modules():
            if isinstance(layer,module.HyperDense):layer.weight.zero_()
        assert (original(x)-y).abs().max().item()>1e-4
