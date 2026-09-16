"""Versioned seven-channel HyperDense surgery; historical controls stay frozen."""
from contextlib import contextmanager
import hashlib
import math
import torch
from torch import nn
from .native_transfer import build as native_build
from .remaining_controls import ContiguousHyperDense, digest_except
from .remaining_spectral import SpectralRealBlock

PROTOCOL='hyperdense-etth1-v2'
ARMS=('native','real','complex','quaternion','octonion','lowrank2','lowrank4','lowrank8')
SITES={'micn':'layers.0.model.regression','film':'layers.0.model.spec_conv_1.0'}
DIMENSIONS={'complex':2,'quaternion':4,'octonion':8}


@contextmanager
def ieee_precision():
    conv,matmul=torch.backends.cudnn.conv,torch.backends.cuda.matmul
    before=(conv.fp32_precision,matmul.fp32_precision)
    conv.fp32_precision=matmul.fp32_precision='ieee'
    try:yield
    finally:conv.fp32_precision,matmul.fp32_precision=before


def stream_seed(seed,backbone,role):
    return int.from_bytes(hashlib.sha256(f'{PROTOCOL}/{seed}/{backbone}/{role}'.encode()).digest()[:8],'big')%(2**63-1)


def mapping(nin,nout,bias,arm,seed,backbone,role,gain):
    if arm not in ARMS[1:] or gain not in (.5,1.):raise ValueError('Invalid controlled mapping')
    variance=2/(nin+nout);rank=None
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(stream_seed(seed,backbone,role))
        if arm=='real':layer=nn.Linear(nin,nout,bias=bias)
        elif arm.startswith('lowrank'):
            rank=nin*nout//(int(arm[-1])*(nin+nout))
            if rank<1 or rank*(nin+nout)!=nin*nout//int(arm[-1]):raise ValueError('Exact rank budget required')
            layer=nn.Sequential(nn.Linear(nin,rank,bias=False),nn.Linear(rank,nout,bias=bias))
        else:
            dim=DIMENSIONS[arm]
            if nin%dim or nout%dim:raise ValueError('Indivisible operator')
            layer=ContiguousHyperDense(nin//dim,nout//dim,arm,bias=bias)
        with torch.no_grad():
            for i,factor in enumerate(list(layer) if rank else [layer]):
                torch.random.default_generator.manual_seed(stream_seed(seed,backbone,f'{role}/factor{i}'))
                std=math.sqrt(gain)*(variance/rank)**.25 if rank else gain*math.sqrt(variance)
                factor.weight.normal_(0,std)
                if factor.bias is not None:factor.bias.zero_()
    return layer


def spectral_permutation(width,dimension):
    if dimension not in (2,4,8) or width%(dimension//2):raise ValueError('Invalid spectral grouping')
    blocks=torch.arange(width).reshape(dimension//2,-1)
    return torch.stack([blocks,blocks+width],dim=1).reshape(-1)


class GroupedSpectral(SpectralRealBlock):
    def __init__(self,original,arm,seed,gain):
        nn.Module.__init__(self)
        self.modes,self.out_channels=original.modes,original.out_channels
        dimension=DIMENSIONS.get(arm,int(arm[-1]) if arm.startswith('lowrank') else 2)
        perm=spectral_permutation(original.in_channels,dimension)
        self.register_buffer('input_permutation',perm)
        self.register_buffer('output_inverse',torch.argsort(spectral_permutation(original.out_channels,dimension)))
        self.maps=nn.ModuleList([mapping(2*original.in_channels,2*original.out_channels,False,arm,seed,'film',f'frequency{k}',gain) for k in range(self.modes)])

    def apply_mode(self,a,k):
        x=torch.cat([a.real,a.imag],dim=-1)[...,self.input_permutation]
        return self.maps[k](x)[...,self.output_inverse]


def build(backbone,arm,seed,gain=1.):
    if backbone not in SITES or arm not in ARMS or gain not in (.5,1.):raise ValueError('Unsupported transfer control')
    model=native_build(backbone,seed,'relative_residual',features=7)
    site=SITES[backbone];original=model.get_submodule(site);untouched=digest_except(model,site)
    if arm=='native':
        with torch.no_grad():
            for name,p in original.named_parameters():
                if 'weight' in name:p.mul_(gain)
        replacement=original
    elif backbone=='micn':replacement=mapping(32,32,True,arm,seed,backbone,'regression',gain)
    else:replacement=GroupedSpectral(original,arm,seed,gain)
    parent,_,key=site.rpartition('.');model.get_submodule(parent)._modules[key]=replacement
    assert digest_except(model,site)==untouched
    report=dict(protocol=PROTOCOL,backbone=backbone,arm=arm,seed=seed,gain=gain,features=7,context=32,horizon=5,
        site=site,untouched_sha256=untouched,native_is_complex=backbone=='film',
        parameters=sum(p.numel() for p in model.parameters()),selected_parameters=sum(p.numel() for p in replacement.parameters()),
        controlled_expanded_variance=gain**2/(32 if backbone=='micn' else 512) if arm!='native' else None,
        selected_weight_second_moments={n:float(p.detach().abs().square().mean()) for n,p in replacement.named_parameters() if 'weight' in n},
        component_layout='chronological blocks' if backbone=='micn' else 'paired real/imaginary contiguous latent blocks')
    return model,report
