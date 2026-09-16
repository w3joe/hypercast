"""Isolated seven-channel, selected-site comparison for the 2026-09-15 study."""
import hashlib
import math
from copy import deepcopy
from contextlib import contextmanager
import torch
from torch import nn
from .native_transfer import NativeTransfer
from .upstream_models import UpstreamSequence, upstream_defaults
from .remaining_controls import SITES as REMAINING, dimensions, digest_except, ChannelMap, ContiguousHyperDense
from .hyperdense_transfer import ARMS, GroupedSpectral, ieee_precision
from .hyperdense_transfer_pinned import constants, CONSTANTS_SHA256

PROTOCOL = 'tslib15-etth1-v2-deterministic-cudnn'
SITES = {**{'dlinear':'Linear_Seasonal', 'tsmixer':'model.0.temporal.0',
    'itransformer':'encoder.attn_layers.0.attention.out_projection'}, **REMAINING,
    'film':'spec_conv_1.0'}
BACKBONES = tuple(SITES)

def stream(seed, backbone, role):
    return int.from_bytes(hashlib.sha256(f'{PROTOCOL}/{seed}/{backbone}/{role}'.encode()).digest()[:8], 'big') % (2**63-1)

class Transfer15(NativeTransfer):
    def __init__(self, backbone):
        nn.Module.__init__(self)
        self.features, self.mode = 7, 'relative_residual'
        self.layers = nn.ModuleList([UpstreamSequence(backbone,32,7,**upstream_defaults(backbone)), nn.Flatten(start_dim=1)])
        self.output = nn.Linear(224,5)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

def build(backbone, arm, seed):
    if backbone not in SITES or arm not in ARMS: raise ValueError('Unknown model/arm')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(stream(seed,backbone,'native'))
        model = Transfer15(backbone)
        if backbone == 'film':
            buffers = dict(model.named_buffers())
            with torch.no_grad():
                for k,v in constants().items(): buffers[k].copy_(v)
        site = 'layers.0.model.' + SITES[backbone]
        old = model.get_submodule(site); untouched = digest_except(model,site)
        rank = None; gap = None
        if backbone == 'film':
            nin=nout=512; bias=False
            new = old if arm=='native' else GroupedSpectral(old,arm,stream(seed,backbone,'spectral'),1.)
            if arm.startswith('lowrank'): rank=512//(2*int(arm[-1]));gap=0
        else:
            nin,nout,bias=dimensions(old)
            if nin%8 or nout%8: raise ValueError(f'Indivisible site: {backbone}: {nin}/{nout}')
            torch.random.default_generator.manual_seed(stream(seed,backbone,'selected'))
            if arm=='native': new=old
            else:
                variance=2/(nin+nout)
                if arm=='real': new=nn.Linear(nin,nout,bias=bias)
                elif arm.startswith('lowrank'):
                    d=int(arm[-1]); rank=max(1,nin*nout//(d*(nin+nout)))
                    gap=rank*(nin+nout)-nin*nout//d
                    new=nn.Sequential(nn.Linear(nin,rank,bias=False),nn.Linear(rank,nout,bias=bias))
                else:
                    d={'complex':2,'quaternion':4,'octonion':8}[arm]
                    new=ContiguousHyperDense(nin//d,nout//d,arm,bias=bias)
                with torch.no_grad():
                    for i,f in enumerate(list(new) if rank else [new]):
                        torch.random.default_generator.manual_seed(stream(seed,backbone,f'factor{i}'))
                        f.weight.normal_(0,(variance/rank)**.25 if rank else math.sqrt(variance))
                        if f.bias is not None: f.bias.zero_()
                if isinstance(old,nn.Conv1d): new=ChannelMap(new)
        parent,_,key=site.rpartition('.');model.get_submodule(parent)._modules[key]=new
        assert digest_except(model,site)==untouched
    return model,dict(protocol=PROTOCOL,backbone=backbone,arm=arm,seed=seed,site=site,
        untouched_sha256=untouched,parameters=sum(p.numel() for p in model.parameters()),
        selected_parameters=sum(p.numel() for p in new.parameters()),in_features=nin,out_features=nout,
        rank=rank,lowrank_weight_budget_gap=gap,native_is_complex=backbone=='film',
        constants_sha256=CONSTANTS_SHA256 if backbone=='film' else None)

def replace(model,site,value):
    parent,_,key=site.rpartition('.');model.get_submodule(parent)._modules[key]=value

@contextmanager
def cudnn_enabled(enabled):
    previous=torch.backends.cudnn.enabled
    try:
        torch.backends.cudnn.enabled=enabled
        yield
    finally:torch.backends.cudnn.enabled=previous

def gate(backbone,device='cpu'):
    """Synthetic controls, without any development or final-test targets."""
    from .training import seed_everything
    from .remaining_spectral import SpectralRealBlock
    from .hyperdense_transfer_pinned_checks import expanded, structured_reference
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;rows=[]
    with ieee_precision():
        for training in (False,True):
            a,report=build(backbone,'native',3100);b=deepcopy(a);site=report['site'];old=b.get_submodule(site)
            if backbone=='film':new=SpectralRealBlock(old)
            elif isinstance(old,nn.Conv1d):
                nin,nout,bias=dimensions(old);mapping=nn.Linear(nin,nout,bias=bias)
                with torch.no_grad():
                    mapping.weight.copy_(old.weight[:,:,0])
                    if bias:mapping.bias.copy_(old.bias)
                new=ChannelMap(mapping)
            else: new=deepcopy(old)
            replace(b,site,new)
            with torch.no_grad(): a.output.weight.fill_(.002);b.output.weight.copy_(a.output.weight)
            a.to(device).train(training);b.to(device).train(training)
            seed_everything(3100);x=torch.randn(2,32,7,device=device);xa=x.clone().requires_grad_();xb=x.clone().requires_grad_()
            # cuDNN GRU backward requires training mode; only this eval diagnostic disables it.
            with cudnn_enabled(not (backbone=='segrnn' and not training)):
                seed_everything(3100);ya=a(xa);seed_everything(3100);yb=b(xb)
                torch.testing.assert_close(ya,yb,atol=2e-6,rtol=2e-4)
                ya.square().mean().backward();yb.square().mean().backward()
            torch.testing.assert_close(xa.grad,xb.grad,atol=2e-6,rtol=2e-4)
            pa,pb=dict(a.named_parameters()),dict(b.named_parameters())
            def corresponding(name):
                if isinstance(old,nn.Conv1d) and name.startswith(site+'.'):
                    return pb[name.replace(site+'.',site+'.mapping.')], True
                return pb[name],False
            for name,p in pa.items():
                q,conv=corresponding(name)
                assert (p.grad is None)==(q.grad is None)
                if p.grad is not None:
                    pg=p.grad[:,:,0] if conv and p.ndim==3 else p.grad
                    torch.testing.assert_close(pg,q.grad,atol=2e-6,rtol=2e-4)
            torch.optim.Adam(a.parameters(),lr=.001,eps=1e-7).step();torch.optim.Adam(b.parameters(),lr=.001,eps=1e-7).step()
            for name,p in pa.items():
                q,conv=corresponding(name);torch.testing.assert_close(p[:,:,0] if conv and p.ndim==3 else p,q,atol=2e-6,rtol=2e-4,msg=lambda msg: name+': '+msg)
        for arm in ARMS:
            seed_everything(3101);m,r=build(backbone,arm,3101);m.to(device).train()
            x=torch.randn(3,32,7,device=device);anchor=x[:,-1,0:1].expand(-1,5)
            y=m(x);torch.testing.assert_close(y,anchor,rtol=0,atol=0)
            opt=torch.optim.Adam(m.parameters(),lr=.001,eps=1e-7)
            (y-anchor-1).square().mean().backward();opt.step();opt.zero_grad(set_to_none=True)
            x.requires_grad_();m(x).square().mean().backward()
            assert all(x.grad[:,:,i].abs().sum()>0 for i in range(7))
            selected=m.get_submodule(r['site'])
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in selected.parameters())
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
            if arm in ('complex','quaternion','octonion'):
                if backbone=='film': structured_reference('film',arm,device)
                else:
                    layer=deepcopy(selected.mapping if isinstance(selected,ChannelMap) else selected).double()
                    z=torch.randn(2,r['in_features'],dtype=torch.float64,device=device,requires_grad=True)
                    actual=layer(z);expected=z@expanded(layer).T
                    if layer.bias is not None:expected=expected+layer.bias.reshape(-1)
                    torch.testing.assert_close(actual,expected,atol=1e-9,rtol=1e-7)
                    pars=(z,*layer.parameters());ga=torch.autograd.grad(actual.square().mean(),pars);gb=torch.autograd.grad(expected.square().mean(),pars)
                    for aa,bb in zip(ga,gb):torch.testing.assert_close(aa,bb,atol=1e-9,rtol=1e-7)
            r['passed']=True;rows.append(r)
        assert len({r['untouched_sha256'] for r in rows})==1
    return dict(passed=True,backbone=backbone,device=device,rows=rows,precision='IEEE FP32')
