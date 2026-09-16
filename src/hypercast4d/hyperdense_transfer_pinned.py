"""Pin FiLM numerical constants across CPU hosts; preserve learned initialization."""
from pathlib import Path
import hashlib
import torch
from .hyperdense_transfer import build as unpinned_build,ARMS,SITES,mapping,spectral_permutation,ieee_precision
from .remaining_controls import digest_except

CONSTANTS_SHA256="643b6e3339570a3668ac8c0f2d0ad95fff9ba9b0b61fc330ffe7b948b76ad491"

def constants():
    path=Path(__file__).parent/'_constants/film_legendre_v1.pt'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CONSTANTS_SHA256:raise ValueError('Canonical FiLM constants changed')
    return torch.load(path,weights_only=True,map_location='cpu')


def build(backbone,arm,seed,gain=1.):
    model,report=unpinned_build(backbone,arm,seed,gain)
    if backbone=='film':
        values=constants();buffers=dict(model.named_buffers())
        assert set(values)=={k for k in buffers if '.legts.' in k}
        with torch.no_grad():
            for k,v in values.items():buffers[k].copy_(v)
        report['constants_sha256']=CONSTANTS_SHA256
    report['protocol']='hyperdense-etth1-v3-pinned-constants'
    report['untouched_sha256']=digest_except(model,SITES[backbone])
    return model,report
