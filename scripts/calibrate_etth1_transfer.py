"""One bounded train-only timing job; requires a separately authorized cap."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from hypercast4d.native_transfer import build
from hypercast4d.training import seed_everything
from hypercast4d.representation_pilot import sha

ROOT=Path(__file__).resolve().parents[1]


def calibrate(manifest,data,device='cuda',archive_checkpoints=False):
    import hypercast4d.native_transfer as module
    root=Path(module.__file__).resolve().parents[2]
    for name,digest in manifest['source_code_sha256'].items():
        if sha(root/name)!=digest:raise ValueError(f'Frozen source changed: {name}')
    import hashlib
    if hashlib.sha256(data).hexdigest()!=manifest['bundle_sha256']:raise ValueError('Data changed')
    if device=='cuda' and (not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name(0)):
        raise RuntimeError('Requires L4')
    torch.set_num_threads(2)
    arrays=np.load(io.BytesIO(data),allow_pickle=False)
    train=TensorDataset(torch.from_numpy(arrays['train_x']),torch.from_numpy(arrays['train_y']))
    inner_inputs=DataLoader(torch.from_numpy(arrays['inner_x']),batch_size=32)
    rows=[];checkpoints={}
    for b in ['micn','film']:
        for mode in ['levels_direct','relative_residual']:
            seed_everything(2100);model=build(b,2100,mode).to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001,eps=1e-7)
            loader=DataLoader(train,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(2100))
            if device=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            start=time.perf_counter();times=[];validation_times=[]
            for epoch in range(3):
                if device=='cuda':torch.cuda.synchronize()
                lap=time.perf_counter();model.train()
                for x,y in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss=(model(x.to(device))-y.to(device)).abs().mean()
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite calibration loss')
                    loss.backward()
                    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise RuntimeError('Nonfinite gradients')
                    optimizer.step()
                if device=='cuda':torch.cuda.synchronize()
                times.append(time.perf_counter()-lap)
                lap=time.perf_counter();model.eval()
                with torch.inference_mode():
                    for x in inner_inputs:
                        prediction=model(x.to(device))
                        if not torch.isfinite(prediction).all():raise RuntimeError('Nonfinite calibration output')
                if device=='cuda':torch.cuda.synchronize()
                validation_times.append(time.perf_counter()-lap)
            rows.append(dict(backbone=b,mode=mode,seed=2100,train_samples=len(train),epochs=3,
                epoch_seconds=times,inner_forward_seconds=validation_times,total_seconds=time.perf_counter()-start,
                parameters=sum(p.numel() for p in model.parameters()),
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None))
            if archive_checkpoints:
                import random
                buffer=io.BytesIO();numpy_state=np.random.get_state()
                torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    optimizer_state_dict=optimizer.state_dict(),epoch=3,backbone=b,mode=mode,seed=2100,
                    features=7,window=32,horizon=5,manifest=manifest,
                    torch_rng_state=torch.get_rng_state(),cuda_rng_states=torch.cuda.get_rng_state_all() if device=='cuda' else [],
                    sampler_rng_state=loader.generator.get_state(),python_rng_state=random.getstate(),
                    numpy_rng_state=dict(name=numpy_state[0],keys=torch.tensor(numpy_state[1].astype('int64')),
                        position=numpy_state[2],has_gauss=numpy_state[3],cached_gaussian=numpy_state[4])),buffer)
                checkpoints[f'{b}-{mode}.pt']=buffer.getvalue()
            del optimizer,model
    return dict(passed=True,device=device,rows=rows,validation_scored=False,test_scored=False,
                purpose='Timing only; seed excluded from development comparisons',checkpoints=checkpoints)


def remote_worker(manifest,data):
    try:return calibrate(manifest,data,archive_checkpoints=True)
    except Exception:return dict(passed=False,error=traceback.format_exc())


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--allocation',type=Path,required=True);p.add_argument('--authorized-cap-usd',type=float,required=True)
    args=p.parse_args();manifest=json.loads((args.bundle/'manifest.json').read_text())
    for name,digest in manifest['source_code_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Source changed: {name}')
    data=(args.bundle/'development.npz').read_bytes()
    assert sha(args.bundle/'development.npz')==manifest['bundle_sha256']
    cli=str(Path(sys.executable).with_name('modal'))
    def read(*parts):return json.loads(subprocess.run([cli,*parts,'--json'],check=True,capture_output=True,text=True,timeout=30).stdout)
    if any(int(a.get('tasks',0)) for a in read('app','list')):raise RuntimeError('Active Modal tasks')
    rates=read('billing','rates');hourly=float(rates['gpu_hour_cost_l4'])+2*float(rates['cpu_hour_cost'])+4*float(rates['mem_gib_hour_cost'])
    from launch_representation_pilot import admission
    cost=admission([dict(timeout_seconds=600)],hourly,args.authorized_cap_usd)[0]
    args.allocation.mkdir(parents=True,exist_ok=False)
    ledger=dict(cap_usd=args.authorized_cap_usd,reserve_usd=1,reservation_usd=cost,status='reserved',rates=rates,manifest_sha256=sha(args.bundle/'manifest.json'))
    def save():(args.allocation/'ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    save()
    import modal,fcntl
    lock=(ROOT/'results/.representation-pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    image=(modal.Image.debian_slim(python_version='3.12')
        .pip_install('einops==0.8.2','scipy==1.18.1','numpy==2.5.3','openpyxl==3.1.5','pandas==3.0.5','torch==2.14.0')
        .env({'PYTHONPATH':'/opt/pilot/src:/opt/pilot/scripts'})
        .add_local_dir(ROOT/'src',remote_path='/opt/pilot/src',ignore=['**/__pycache__/**','**/web_dist/**'])
        .add_local_dir(ROOT/'scripts',remote_path='/opt/pilot/scripts',ignore=['**/__pycache__/**'])
        .add_local_dir(ROOT/'tests',remote_path='/opt/pilot/tests',ignore=['**/__pycache__/**']))
    app=modal.App('hypercast4d-etth1-calibration')
    fn=app.function(image=image,gpu='L4',cpu=2,memory=4096,timeout=600,startup_timeout=120,retries=0,max_containers=1,scaledown_window=2,serialized=True)(remote_worker)
    try:
        with app.run():
            call=fn.spawn(manifest,data);ledger.update(status='submitted',call_id=call.object_id);save()
            result=call.get()
        for name,blob in result.pop('checkpoints',{}).items():
            (args.allocation/name).write_bytes(blob)
        (args.allocation/'calibration.json').write_text(json.dumps(result,indent=2)+'\n')
        ledger['status']='complete' if result['passed'] else 'failed'
    except BaseException:
        ledger['status']='failed';raise
    finally:save()


if __name__=='__main__':main()
