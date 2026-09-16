"""Synthetic gates for the versioned transfer controls; no evaluation data."""
from copy import deepcopy
import torch
from torch import nn
from .hyperdense_transfer_pinned import build,ARMS,SITES,ieee_precision
from .remaining_spectral import SpectralRealBlock
from .experiment_controls import explicit_real_matrix
from .layers import HyperDense
from .training import seed_everything


def expanded(layer):
    if isinstance(layer,HyperDense):return explicit_real_matrix(layer).T
    if isinstance(layer,nn.Sequential):return layer[1].weight@layer[0].weight
    return layer.weight


def real_surgery(backbone,device,training):
    left,_=build(backbone,'native',2390);right=deepcopy(left);site=SITES[backbone]
    if backbone=='film':right.layers[0].model.spec_conv_1[0]=SpectralRealBlock(right.get_submodule(site))
    else:
        old=right.get_submodule(site);new=nn.Linear(32,32);new.load_state_dict(old.state_dict());right.layers[0].model.regression=new
    with torch.no_grad():left.output.weight.fill_(.002);right.output.weight.copy_(left.output.weight)
    left.to(device).train(training);right.to(device).train(training)
    x=torch.linspace(-.8,.9,448,device=device).reshape(2,32,7);a=x.clone().requires_grad_();b=x.clone().requires_grad_()
    seed_everything(2390);ya=left(a);seed_everything(2390);yb=right(b)
    torch.testing.assert_close(ya,yb,atol=2e-6,rtol=2e-4)
    ya.square().mean().backward();yb.square().mean().backward();torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-4)
    lp,rp=dict(left.named_parameters()),dict(right.named_parameters())
    for name,p in lp.items():
        assert (p.grad is None)==(rp[name].grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,rp[name].grad,atol=2e-6,rtol=2e-4)
    torch.optim.Adam(left.parameters(),lr=.001,eps=1e-7).step();torch.optim.Adam(right.parameters(),lr=.001,eps=1e-7).step()
    for name,p in lp.items():torch.testing.assert_close(p,rp[name],atol=2e-6,rtol=2e-4)
    return dict(backbone=backbone,training=training,passed=True)


def structured_reference(backbone,arm,device):
    model,_=build(backbone,arm,2391);layer=model.get_submodule(SITES[backbone]).double().to(device)
    if backbone=='micn':
        x=torch.linspace(-.5,.7,64,device=device,dtype=torch.float64).reshape(2,32).requires_grad_()
        actual=layer(x);expected=x@expanded(layer).T+layer.bias.reshape(-1)
    else:
        x=torch.linspace(-.5,.7,256*32,device=device,dtype=torch.float64).reshape(1,1,256,32).requires_grad_()
        actual=layer(x);spectrum=torch.fft.rfft(x);transformed=[]
        for k,mapping in enumerate(layer.maps):
            a=spectrum[...,k];packed=torch.cat([a.real,a.imag],dim=-1)[...,layer.input_permutation]
            y=(packed@expanded(mapping).T)[...,layer.output_inverse]
            transformed.append(torch.complex(y[...,:256],y[...,256:]))
        result=torch.zeros(1,1,256,17,device=device,dtype=torch.cfloat);result[...,:16]=torch.stack(transformed,dim=-1)
        expected=torch.fft.irfft(result,n=32)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-4)
    params=(x,*layer.parameters());a=torch.autograd.grad(actual.square().mean(),params);b=torch.autograd.grad(expected.square().mean(),params)
    for p,q in zip(a,b):torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-4)
    return dict(backbone=backbone,arm=arm,passed=True)


def continuation_gate(device):
    import tempfile
    from pathlib import Path
    from torch.utils.data import TensorDataset
    from .hyperdense_transfer_training import run
    torch.manual_seed(2392)
    model=nn.Sequential(nn.Flatten(),nn.Dropout(.2),nn.Linear(8,2))
    x=torch.randn(70,2,4);y=torch.randn(70,2);train=TensorDataset(x,y);inner=TensorDataset(x[:13],y[:13])
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);a,b=deepcopy(model),deepcopy(model)
        options=dict(seed=2392,schedule='exponential',device=device)
        first=run(a,train,inner,root/'full',epochs=5,**options)
        run(b,train,inner,root/'split',epochs=2,**options)
        second=run(b,train,inner,root/'split',epochs=5,resume=True,**options)
        assert first['history']==second['history']
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,atol=0,rtol=0)
    return dict(device=str(device),passed=True,scope='Exact dropout/Adam/exponential-scheduler continuation on synthetic linear model; not proof of every full backbone CUDA trajectory')


def gate(device='cpu'):
    torch.set_num_threads(2);rows=[];equivalence=[];references=[]
    with ieee_precision():
        for backbone in SITES:
            for training in (False,True):equivalence.append(real_surgery(backbone,device,training))
            for arm in ('complex','quaternion','octonion'):references.append(structured_reference(backbone,arm,device))
            hashes=[]
            for arm in ARMS:
                seed_everything(2300);model,report=build(backbone,arm,2300);hashes.append(report['untouched_sha256']);model.to(device).train()
                x=torch.linspace(-.8,.9,448,device=device).reshape(2,32,7);anchor=x[:,-1,0:1].expand(-1,5)
                calls=[];selected=model.get_submodule(SITES[backbone]);hook=selected.register_forward_hook(lambda m,a,o:calls.append(tuple(o.shape)))
                y=model(x);torch.testing.assert_close(y,anchor,rtol=0,atol=0)
                opt=torch.optim.Adam(model.parameters(),lr=.001,eps=1e-7)
                (y-anchor-1).square().mean().backward();assert model.output.weight.grad.abs().sum()>0;opt.step();opt.zero_grad(set_to_none=True)
                model(x).square().mean().backward();hook.remove()
                assert calls and any(p.grad is not None and p.grad.abs().sum()>0 for p in selected.parameters())
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                report.update(passed=True,selected_calls=len(calls))
                if str(device).startswith('cuda'):
                    model.eval()
                    with torch.inference_mode():
                        gpu_prediction=model(x).cpu();cpu_model=deepcopy(model).cpu();cpu_prediction=cpu_model(x.cpu())
                    torch.testing.assert_close(cpu_prediction,gpu_prediction,atol=2e-5,rtol=2e-4)
                    report['cpu_cuda_max_absolute_difference']=float((cpu_prediction-gpu_prediction).abs().max())
                    del cpu_model,cpu_prediction,gpu_prediction
                rows.append(report)
                del opt,model,selected,y
            assert len(set(hashes))==1
    return dict(passed=True,device=str(device),rows=rows,equivalence=equivalence,references=references,precision='IEEE FP32',continuation=continuation_gate(device),test_scored=False)
