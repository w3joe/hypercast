"""Frozen single-L4 worker. Calibration never receives held-out labels."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import time
import traceback

def train_trial(plan,data,job,directory,device='cuda'):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset
    from hypercast4d.tslib15 import build,ieee_precision
    from hypercast4d.hyperdense_transfer_training import run,finite
    from hypercast4d.training import predict
    from hypercast4d.representation_pilot import score,sha
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    arrays=np.load(io.BytesIO(data),allow_pickle=False);phase=job['phase'];cal=phase=='calibration'
    required={'train_x','train_y','inner_x'} if cal else {'train_x','train_y','inner_x','inner_y',phase+'_x',phase+'_y',phase+'_target_start'}
    if set(arrays.files)!=required:raise ValueError('Unexpected labels in worker bundle')
    def dataset(name):return TensorDataset(torch.from_numpy(arrays[name+'_x']),torch.from_numpy(arrays[name+'_y']))
    model,control=build(job['backbone'],job['arm'],job['seed'])
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    ceiling=3 if cal else job.get('epochs',150)
    result=run(model,dataset('train'),torch.from_numpy(arrays['inner_x']) if cal else dataset('inner'),directory,
        seed=job['seed'],epochs=3 if cal else ceiling,learning_rate=job['learning_rate'],calibration=cal,device=device,
        metadata=dict(protocol=plan['protocol'],source_sha256=plan['source_sha256'],job=job,control=control,dataset=plan['dataset_manifest']),
        deadline=start+max(1,min(job['timeout_seconds']-30,plan['gpu_finish_epoch']-time.time()-90)))
    result.update(job=job,control=control,training_elapsed_seconds=time.monotonic()-start,
        peak_gpu_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None)
    file=directory/('latest.pt' if cal else 'best.pt');c=torch.load(file,weights_only=True,map_location='cpu');finite(c)
    restored,_=build(job['backbone'],job['arm'],job['seed']);restored.load_state_dict(c['state_dict']);restored.to(device).eval();model.eval()
    with ieee_precision(),torch.inference_mode():
        sample=torch.from_numpy(arrays['inner_x'][:32]).to(device)
        torch.testing.assert_close(restored(sample),model(sample),atol=0,rtol=0)
    result['checkpoint_replay_passed']=True
    result['latest_checkpoint_sha256']=sha(directory/'latest.pt')
    if not cal:
        with ieee_precision():
            inner=predict(restored,dataset('inner'),torch.device(device),32)
            if not np.isclose(np.abs(inner-arrays['inner_y']).mean(),result['best_inner_mae'],rtol=2e-5,atol=1e-7):raise RuntimeError('Inner checkpoint score mismatch')
            predicted=predict(restored,dataset(phase),torch.device(device),32)
        manifest=plan['dataset_manifest'];span,offset=manifest['scaler_span'][0],manifest['scaler_minimum'][0]
        predicted=predicted.astype(float)*span+offset;actual=arrays[phase+'_y'].astype(float)*span+offset;x=arrays[phase+'_x'].astype(float)
        persistence=np.repeat(x[:,-1,0:1],5,axis=1)*span+offset;seasonal=x[:,np.arange(5)+8,0]*span+offset
        if not np.isfinite(predicted).all():raise RuntimeError('Nonfinite predictions')
        np.savez_compressed(directory/'predictions.npz',prediction=predicted,actual=actual,persistence=persistence,
            seasonal_24=seasonal,target_start=arrays[phase+'_target_start'])
        result.update(**score(predicted,actual,persistence),rmse=float(np.sqrt(np.mean((predicted-actual)**2))),
            seasonal_24_mae=float(np.abs(actual-seasonal).mean()),best_checkpoint_sha256=sha(directory/'best.pt'),test_scored=phase=='test')
        if phase=='test':
            timings={}
            with ieee_precision(),torch.inference_mode():
                for batch in (1,32):
                    z=torch.from_numpy(arrays['test_x'][:batch]).to(device)
                    for _ in range(10):restored(z)
                    repeats=[]
                    for _ in range(5):
                        if str(device).startswith('cuda'):torch.cuda.synchronize()
                        lap=time.perf_counter()
                        for _ in range(20):restored(z)
                        if str(device).startswith('cuda'):torch.cuda.synchronize()
                        repeats.append((time.perf_counter()-lap)*1000/20)
                    timings[str(batch)]=dict(median_ms=float(np.median(repeats)),repeat_ms=repeats)
            result['inference']=timings
    result['ceiling_reached']=not cal and result['epochs_ran']==ceiling
    result['epoch_ceiling']=ceiling
    result['checkpoint_bytes']={p.name:p.stat().st_size for p in directory.glob('*.pt')}
    (directory/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result

def worker(plan,data,job):
    import torch
    import hypercast4d.tslib15 as module
    torch.set_num_threads(2);root=Path(module.__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp);start=time.monotonic()
        try:
            if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name(0):raise RuntimeError('L4 required')
            for name,digest in plan['source_sha256'].items():
                if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Source changed: {name}')
            if hashlib.sha256(data).hexdigest()!=plan['data_sha256'][job['phase']]:raise ValueError('Data changed')
            if time.time()>=plan['gpu_finish_epoch']-120:raise TimeoutError('Global deadline')
            if job['phase']=='calibration':
                if job.get('run_gate',True):
                    gate=module.gate(job['backbone'],'cuda');(path/'gate.json').write_text(json.dumps(gate,indent=2))
                for arm in ([job['arm']] if 'arm' in job else module.ARMS):
                    sub={**job,'arm':arm};train_trial(plan,data,sub,path/arm)
            else:train_trial(plan,data,job,path/'trial')
            status=dict(ok=True,gpu=torch.cuda.get_device_name(0),torch_version=str(torch.__version__))
        except Exception:status=dict(ok=False,error=traceback.format_exc())
        status.update(job=job,elapsed_seconds=time.monotonic()-start)
        (path/'status.json').write_text(json.dumps(status,indent=2));members=list(path.iterdir())
        payload=path/'payload.tar.gz'
        with tarfile.open(payload,mode='w:gz') as archive:
            for p in members:archive.add(p,arcname=p.name)
        return status,payload.read_bytes()
