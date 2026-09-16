import numpy as np
import pytest
import torch
from hypercast4d.native_transfer import NativeTransfer,build
from hypercast4d.representation_pilot import build_model

torch.set_num_threads(2)

@pytest.mark.parametrize('backbone',['micn','film'])
@pytest.mark.parametrize('mode',['levels_direct','relative_residual'])
def test_four_channel_wrapper_matches_historical_graph(backbone,mode):
    old,_=build_model(backbone,mode,1101);new=NativeTransfer(backbone,4,mode=mode)
    new.load_state_dict(old.state_dict(),strict=True)
    old.eval();new.eval()
    x=torch.randn(2,32,4);a=x.clone().requires_grad_();b=x.clone().requires_grad_()
    ya,yb=old(a),new(b)
    torch.testing.assert_close(ya,yb,atol=0,rtol=0)
    ya.square().mean().backward();yb.square().mean().backward()
    torch.testing.assert_close(a.grad,b.grad,atol=0,rtol=0)

@pytest.mark.parametrize('backbone',['micn','film'])
def test_seven_channels_paired_and_gradient_recovery(backbone):
    direct=build(backbone,2101,'levels_direct');residual=build(backbone,2101,'relative_residual')
    for name,p in direct.state_dict().items():
        if not name.startswith('output.'):torch.testing.assert_close(p,residual.state_dict()[name],atol=0,rtol=0)
    direct.eval();residual.eval();x=torch.randn(2,32,7)
    torch.testing.assert_close(residual(x),x[:,-1,0:1].expand(-1,5),atol=0,rtol=0)
    x.requires_grad_();direct(x).square().mean().backward()
    assert all(x.grad[:,:,i].abs().sum()>0 for i in range(7))
    opt=torch.optim.Adam(residual.parameters(),lr=.001)
    (residual(x.detach())-x[:,-1,0:1].detach()-1).square().mean().backward()
    assert residual.output.weight.grad.abs().sum()>0
    opt.step();opt.zero_grad();residual(x.detach()).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in residual.layers.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in residual.parameters())


def test_partition_keeps_targets_in_role_and_seasonal_baseline_causal():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_etth1_transfer.py'
    spec=importlib.util.spec_from_file_location('ett_prepare',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    raw=np.arange(180*7,dtype='float32').reshape(180,7)
    part=m.partition(raw,100,160)
    starts=part['target_start']
    assert starts.min()==100 and starts.max()+4==159
    np.testing.assert_array_equal(part['y'],raw[starts[:,None]+np.arange(5),0])
    # Same-observation daily baseline for each of five forecast leads.
    seasonal=part['x'][:,np.arange(5)+32-24,0]
    np.testing.assert_array_equal(seasonal,raw[starts[:,None]+np.arange(5)-24,0])
    assert all(i<32 for i in np.arange(5)+32-24)


def test_timing_calibration_never_needs_validation_targets():
    import importlib.util,hashlib,io
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/calibrate_etth1_transfer.py'
    spec=importlib.util.spec_from_file_location('ett_calibration',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    rng=np.random.default_rng(2);buffer=io.BytesIO()
    np.savez(buffer,train_x=rng.normal(size=(2,32,7)).astype('float32'),train_y=rng.normal(size=(2,5)).astype('float32'),inner_x=rng.normal(size=(2,32,7)).astype('float32'))
    blob=buffer.getvalue();manifest=dict(source_code_sha256={},bundle_sha256=hashlib.sha256(blob).hexdigest())
    r=m.calibrate(manifest,blob,device='cpu',archive_checkpoints=True)
    assert len(r['checkpoints'])==4
    checkpoint=torch.load(io.BytesIO(next(iter(r['checkpoints'].values()))),weights_only=True,map_location='cpu')
    assert checkpoint['epoch']==3 and checkpoint['optimizer_state_dict']['state']
    assert 'sampler_rng_state' in checkpoint
    assert r['passed'] and len(r['rows'])==4 and not r['test_scored'] and not r['validation_scored']
    assert all(x['epochs']==3 and len(x['epoch_seconds'])==3 for x in r['rows'])
