"""Local, inference-only experiments; never changes the frozen training code."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn

from hypercast4d.layers import HyperDense
from hypercast4d.experiment_controls import explicit_real_matrix
from hypercast4d.remaining_controls import ContiguousHyperDense, build_case

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/hyperdense-followup-local'


class CachedConstants(nn.Module):
    """Same contraction with constants resident in the weight dtype/device."""
    def __init__(self, source):
        super().__init__()
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.register_buffer('constants', source.algebra.constants.to(source.weight).clone())
        self.register_buffer('bias', None if source.bias is None else source.bias.detach().clone())
        self.dimension = source.component_count
        self.width = source.in_features
        self.contiguous = isinstance(source, ContiguousHyperDense)

    def forward(self, x):
        y = torch.einsum('bac,aio,...bi->...co', self.constants, self.weight,
                         x.reshape(*x.shape[:-1], self.dimension, self.width))
        if self.bias is not None:
            y = y + self.bias
        y = y.reshape(*x.shape[:-1], -1)
        return y.contiguous() if self.contiguous else y


def export_dense(source):
    """Frozen inference snapshot. Re-export if weights change; expands storage."""
    d = source.component_count
    out = nn.Linear(d * source.in_features, d * source.out_features, bias=source.bias is not None,
                    device=source.weight.device, dtype=source.weight.dtype)
    with torch.no_grad():
        out.weight.copy_(explicit_real_matrix(source).T)
        if source.bias is not None:
            out.bias.copy_(source.bias.reshape(-1))
    return out.requires_grad_(False).eval()


def replace_layers(module, mode):
    for name, child in list(module.named_children()):
        if isinstance(child, HyperDense):
            setattr(module, name, CachedConstants(child) if mode == 'cached' else export_dense(child))
        else:
            replace_layers(child, mode)
    return module


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def timings(implementations, x, device, repeats=15, rounds=5):
    samples = {name: [] for name in implementations}
    with torch.inference_mode():
        for module in implementations.values():
            for _ in range(5):
                module(x)
        sync(device)
        for turn in range(rounds):
            names = list(implementations)
            random.Random(807 + turn).shuffle(names)
            for name in names:
                sync(device)
                start = time.perf_counter()
                for _ in range(repeats):
                    implementations[name](x)
                sync(device)
                samples[name].append(1000 * (time.perf_counter() - start) / repeats)
    return {name: dict(median_ms=float(np.median(v)), min_ms=min(v), max_ms=max(v),
                       rounds_ms=v) for name, v in samples.items()}


def storage(module):
    return sum(t.numel() * t.element_size() for t in list(module.parameters()) + list(module.buffers()))


def equivalent(implementations, x):
    with torch.inference_mode():
        ref = implementations['original'](x)
        result = {}
        for name, module in implementations.items():
            actual = module(x)
            torch.testing.assert_close(actual, ref, atol=2e-5, rtol=2e-4)
            result[name] = float((actual-ref).abs().max())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', nargs='+', default=['cpu', 'mps'])
    parser.add_argument('--output', type=Path, default=OUT/'inference.json')
    args = parser.parse_args()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(818)
    state = json.loads((ROOT/'results/modal-l4-internal/controller/controller_state.json').read_text())
    shapes = {}
    for c in state['candidates']:
        if c['stage_name'] != 'remaining-replication':
            continue
        init = json.loads((ROOT/'results/modal-l4-internal/jobs'/c['job_id']/'initialization.json').read_text())
        r = next(r for r in init if r['variant'] == 'real')
        shapes.setdefault((r['in_features'], r['out_features'], r['bias']), []).append(c['backbone'])
    output = dict(platform=platform.platform(), torch_version=torch.__version__, cpu_threads=2,
                  limitations='Fresh random weights; layer microbenchmarks have synthetic flat batches, not full-model activation shapes. Local CPU/MPS timings cannot establish L4 latency. Dense export expands resident storage. No retraining or Modal submissions.',
                  layers=[], full_models=[], errors=[], source_sha256={})
    for path in ['src/hypercast4d/layers.py', 'scripts/profile_hyperdense_followup.py']:
        output['source_sha256'][path] = hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    # Independent high-precision basis-vector and leading-axis checks, including bias.
    checks = 0
    for name, d in [('complex', 2), ('quaternion', 4), ('octonion', 8)]:
        for bias in (True, False):
            original = ContiguousHyperDense(16//d, 24//d, name, bias=bias).double().eval()
            dense = export_dense(original)
            for x in [torch.eye(16, dtype=torch.float64), torch.randn(2,3,16,dtype=torch.float64)]:
                with torch.inference_mode():
                    torch.testing.assert_close(original(x), dense(x), atol=1e-12, rtol=1e-12)
                checks += 1
    output['float64_equivalence_checks'] = checks
    for device in args.devices:
        for (width_in, width_out, bias), models in shapes.items():
            for algebra, d in [('complex',2), ('quaternion',4), ('octonion',8)]:
                original = ContiguousHyperDense(width_in//d, width_out//d, algebra, bias=bias).to(device).eval()
                implementations = dict(original=original, cached=CachedConstants(original), dense=export_dense(original))
                for batch in (1,32,256):
                    x = torch.randn(batch, width_in, device=device)
                    error = equivalent(implementations,x)
                    measured = timings(implementations,x,device)
                    output['layers'].append(dict(device=device, width_in=width_in, width_out=width_out,
                        bias=bias, backbones=models, algebra=algebra, batch=batch, max_abs_error=error,
                        storage_bytes={n:storage(m) for n,m in implementations.items()}, timings=measured))
                print(f'{device} {width_in}->{width_out} {algebra} complete', flush=True)
                args.output.write_text(json.dumps(output,indent=2)+'\n')
    # Actual MICN/FiLM graph on CPU: controlled fresh weights, original vs equivalent implementation.
    for backbone, algebra in [('micn','octonion'), ('film','quaternion')]:
        original, _ = build_case(backbone,algebra,seed=818)
        original.eval()
        implementations = dict(original=original,
            cached=replace_layers(deepcopy(original),'cached').eval(),
            dense=replace_layers(deepcopy(original),'dense').eval())
        for batch in (1,32):
            x = torch.randn(batch,32,4)
            output['full_models'].append(dict(device='cpu',backbone=backbone,algebra=algebra,batch=batch,
                max_abs_error=equivalent(implementations,x), timings=timings(implementations,x,'cpu',repeats=5),
                storage_bytes={n:storage(m) for n,m in implementations.items()}))
        print(f'Full {backbone} CPU checks complete',flush=True)
    # Operator attribution is CPU-only and separate from timing trials.
    original = ContiguousHyperDense(8,8,'quaternion').eval()
    impl = dict(original=original,cached=CachedConstants(original),dense=export_dense(original))
    x = torch.randn(32,32)
    output['cpu_operator_profiles'] = {}
    for name,module in impl.items():
        with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profile:
            for _ in range(20):
                module(x)
        events = profile.key_averages()
        output['cpu_operator_profiles'][name] = [dict(operator=e.key,calls=e.count,
            self_cpu_time_us=e.self_cpu_time_total, self_cpu_memory_bytes=e.self_cpu_memory_usage) for e in events]
        (args.output.parent/f'profile-{name}.txt').write_text(events.table(sort_by='self_cpu_time_total',row_limit=20))
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(f'Saved {args.output}',flush=True)


if __name__ == '__main__':
    main()
