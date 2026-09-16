"""CUDA-only inference gate, separate from historical local prototypes."""
from copy import deepcopy
import json
from pathlib import Path
import torch
from profile_hyperdense_followup import (ContiguousHyperDense, CachedConstants, export_dense,
    build_case, replace_layers, equivalent, timings, storage)


def measure(implementations,x):
    result=dict(max_abs_error=equivalent(implementations,x),timings=timings(implementations,x,'cuda'),
                storage_bytes={n:storage(m) for n,m in implementations.items()},memory={})
    for name,module in implementations.items():
        torch.cuda.synchronize()
        baseline=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            y=module(x)
            torch.cuda.synchronize()
        result['memory'][name]=dict(baseline_allocated_bytes=baseline,
            incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated()-baseline)
        del y
    return result


def run(output,shapes):
    if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Requires one L4')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(818)
    result=dict(gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,layers=[],full_models=[],
        limitations='Fresh weights; resident inputs; incremental allocator peaks include output and workspace, not process/device memory. All three implementations coexist. Dense export expands deployment storage.')
    def save():
        (output/'inference.json').write_text(json.dumps(result,indent=2))
    for wi,wo,bias in shapes:
        for algebra,d in [('complex',2),('quaternion',4),('octonion',8)]:
            original=ContiguousHyperDense(wi//d,wo//d,algebra,bias=bias).cuda().eval()
            impl=dict(original=original,cached=CachedConstants(original),dense=export_dense(original))
            for batch in [1,32,256]:
                x=torch.randn(batch,wi,device='cuda')
                result['layers'].append(dict(in_features=wi,out_features=wo,bias=bias,algebra=algebra,batch=batch,**measure(impl,x)))
                save()
    for backbone,algebra in [('micn','octonion'),('film','quaternion')]:
        original,_=build_case(backbone,algebra,seed=818);original.cuda().eval()
        impl=dict(original=original,cached=replace_layers(deepcopy(original),'cached'),dense=replace_layers(deepcopy(original),'dense'))
        for batch in [1,32]:
            x=torch.randn(batch,32,4,device='cuda')
            result['full_models'].append(dict(backbone=backbone,algebra=algebra,batch=batch,**measure(impl,x)))
            save()
    original=ContiguousHyperDense(8,8,'quaternion').cuda().eval()
    x=torch.randn(32,32,device='cuda')
    for name,m in dict(original=original,cached=CachedConstants(original),dense=export_dense(original)).items():
        with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],profile_memory=True) as p:
            for _ in range(20):m(x)
            torch.cuda.synchronize()
        p.export_chrome_trace(str(output/f'profile-{name}.json'))
    save()
