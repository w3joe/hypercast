"""Isolated arbitrary-channel native wrapper for the next development study.

Preserves the historical 32-step upstream latent forecast + flattened head.
No changes to the frozen Copper implementation or generic app schema.
"""
import hashlib
import torch
from torch import nn
from .upstream_models import UpstreamSequence, upstream_defaults

MODES=('levels_direct','relative_residual')


class NativeTransfer(nn.Module):
    def __init__(self,backbone,features=7,window=32,horizon=5,mode='levels_direct'):
        super().__init__()
        if backbone not in ('micn','film') or mode not in MODES or features<1 or window!=32 or horizon!=5:
            raise ValueError('Native transfer scope is MICN/FiLM, context32/horizon5')
        self.features,self.mode=features,mode
        self.layers=nn.ModuleList([UpstreamSequence(backbone,window,features,**upstream_defaults(backbone)),nn.Flatten(start_dim=1)])
        self.output=nn.Linear(window*features,horizon)
        nn.init.xavier_uniform_(self.output.weight);nn.init.zeros_(self.output.bias)
        if mode=='relative_residual':
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(32,self.features):raise ValueError('Wrong input shape')
        anchor=x[:,-1,0:1]
        y=x-x[:,-1:,:] if self.mode=='relative_residual' else x
        for layer in self.layers:y=layer(y)
        y=self.output(y)
        return y+anchor if self.mode=='relative_residual' else y


def build(backbone,seed,mode,features=7):
    key=f'native-transfer-v1/{backbone}/{seed}/{features}/32/5'
    value=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')%(2**63-1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(value)
        return NativeTransfer(backbone,features=features,mode=mode)
