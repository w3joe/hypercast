"""Bounded twelve-fit native development stage; final-test tensors never supplied."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


def train_trial(manifest,data,job,directory,device='cuda',epochs=150,deadline=None,plan=None):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset
    from hypercast4d.native_transfer import build
    from hypercast4d.training import predict
    from hypercast4d.transfer_training import fit
    from hypercast4d.representation_pilot import score,sha
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    arrays=np.load(io.BytesIO(data),allow_pickle=False)
    def dataset(name):return TensorDataset(torch.from_numpy(arrays[name+'_x']),torch.from_numpy(arrays[name+'_y']))
    model=build(job['backbone'],job['seed'],job['mode'])
    metadata=dict(backbone=job['backbone'],seed=job['seed'],mode=job['mode'],features=7,window=32,horizon=5,dataset_manifest=manifest,stage_plan=plan)
    start=time.perf_counter()
    fitted=fit(model,dataset('train'),dataset('inner'),directory,seed=job['seed'],epochs=epochs,
        device=device,metadata=metadata,deadline=deadline)
    # Reconstruct the selected inference artifact, not the latest training state.
    checkpoint=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
    restored=build(job['backbone'],job['seed'],job['mode']).to(device)
    restored.load_state_dict(checkpoint['state_dict'],strict=True)
    inner=predict(restored,dataset('inner'),torch.device(device),32)
    if not np.isclose(np.abs(inner-arrays['inner_y']).mean(),fitted['best_inner_mae'],rtol=2e-5,atol=1e-7):
        raise RuntimeError('Selected checkpoint MAE replay mismatch')
    span,minimum=manifest['scaler_span'][0],manifest['scaler_minimum'][0]
    predicted=predict(restored,dataset('development'),torch.device(device),32).astype(np.float64)*span+minimum
    actual=arrays['development_y'].astype(np.float64)*span+minimum
    x=arrays['development_x'].astype(np.float64)
    persistence=np.repeat(x[:,-1,0:1],5,axis=1)*span+minimum
    seasonal=x[:,np.arange(5)+32-24,0]*span+minimum
    if not np.isfinite(predicted).all():raise RuntimeError('Nonfinite development predictions')
    np.savez_compressed(directory/'development-predictions.npz',prediction=predicted,actual=actual,persistence=persistence,
        seasonal_24_hour=seasonal,target_start=arrays['development_target_start'])
    result=dict(**job,**fitted,**score(predicted,actual,persistence),
        seasonal_24_mae=float(np.abs(seasonal-actual).mean()),
        parameters=sum(p.numel() for p in model.parameters()),wall_seconds=time.perf_counter()-start,
        best_checkpoint_sha256=sha(directory/'best.pt'),latest_checkpoint_sha256=sha(directory/'latest.pt'),
        checkpoint_replay_passed=True,test_scored=False)
    (directory/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def worker(manifest,data,plan,job):
    import torch
    import hypercast4d.native_transfer as native
    root=Path(native.__file__).resolve().parents[2]
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as temp:
        directory=Path(temp)/'trial';start=time.monotonic();status={}
        try:
            if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name(0):raise RuntimeError('L4 required')
            for name,digest in plan['source_sha256'].items():
                if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Source changed: {name}')
            if hashlib.sha256(data).hexdigest()!=manifest['bundle_sha256']:raise ValueError('Dataset changed')
            train_trial(manifest,data,job,directory,deadline=start+job['timeout_seconds']-90,plan=plan)
            status=dict(ok=True,gpu=torch.cuda.get_device_name(0))
        except Exception:status=dict(ok=False,error=traceback.format_exc())
        status.update(elapsed_seconds=time.monotonic()-start,job=job)
        (Path(temp)/'status.json').write_text(json.dumps(status,indent=2))
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            archive.add(Path(temp)/'status.json',arcname='status.json')
            if directory.exists():archive.add(directory,arcname='trial')
        return status,buffer.getvalue()


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--allocation',type=Path,required=True);p.add_argument('--authorized-cap-usd',type=float,required=True)
    a=p.parse_args()
    from hypercast4d.representation_pilot import sha
    from launch_representation_pilot import admission
    plan=json.loads(a.plan.read_text());manifest=json.loads((a.bundle/'manifest.json').read_text());data=(a.bundle/'development.npz').read_bytes()
    assert sha(a.bundle/'manifest.json')==plan['dataset_manifest_sha256']
    assert sha(a.bundle/'development.npz')==manifest['bundle_sha256']
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Source changed: {name}')
    cli=str(Path(sys.executable).with_name('modal'))
    def read(*args):return json.loads(subprocess.run([cli,*args,'--json'],capture_output=True,text=True,check=True,timeout=30).stdout)
    if any(int(x.get('tasks',0)) for x in read('app','list')):raise RuntimeError('Other Modal tasks active')
    rates=read('billing','rates');hourly=float(rates['gpu_hour_cost_l4'])+2*float(rates['cpu_hour_cost'])+4*float(rates['mem_gib_hour_cost'])
    jobs=plan['jobs'];costs=admission(jobs,hourly,a.authorized_cap_usd)
    import fcntl
    lock=(ROOT/'results/.representation-pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a.allocation.mkdir(parents=True,exist_ok=False)
    ledger=dict(cap_usd=a.authorized_cap_usd,reserve_usd=1,rates=rates,plan_sha256=sha(a.plan),attempts=[],status='admitted')
    def save():
        temp=a.allocation/'ledger.tmp';temp.write_text(json.dumps(ledger,indent=2));temp.replace(a.allocation/'ledger.json')
    save()
    import modal
    image=(modal.Image.debian_slim(python_version='3.12')
        .pip_install('einops==0.8.2','scipy==1.18.1','numpy==2.5.3','openpyxl==3.1.5','pandas==3.0.5','torch==2.14.0')
        .env({'PYTHONPATH':'/opt/pilot/src:/opt/pilot/scripts'})
        .add_local_dir(ROOT/'src',remote_path='/opt/pilot/src',ignore=['**/__pycache__/**','**/web_dist/**'])
        .add_local_dir(ROOT/'scripts',remote_path='/opt/pilot/scripts',ignore=['**/__pycache__/**'])
        .add_local_dir(ROOT/'tests',remote_path='/opt/pilot/tests',ignore=['**/__pycache__/**']))
    app=modal.App('hypercast4d-etth1-development')
    remote=app.function(image=image,gpu='L4',cpu=2,memory=4096,retries=0,max_containers=2,
        scaledown_window=2,startup_timeout=120,serialized=True)(worker)
    try:
        for start in range(0,len(jobs),2):
            indexes=list(range(start,min(start+2,len(jobs))));attempts={}
            for i in indexes:
                attempts[i]=dict(index=i,job=jobs[i],reservation_usd=costs[i],status='reserved');ledger['attempts'].append(attempts[i])
            save();errors=[]
            with app.run():
                calls={}
                for i in indexes:
                    calls[i]=remote.with_options(timeout=jobs[i]['timeout_seconds']).spawn(manifest,data,plan,jobs[i])
                    attempts[i].update(status='submitted',call_id=calls[i].object_id);save()
                for i,call in calls.items():
                    try:
                        status,archive=call.get();(a.allocation/f'job-{i:02d}.tar.gz').write_bytes(archive)
                        attempts[i].update(status='complete' if status['ok'] else 'failed',runtime=status)
                        if not status['ok']:errors.append(status['error'])
                    except Exception:
                        error=traceback.format_exc();attempts[i].update(status='failed',error=error);errors.append(error)
                    save()
            if errors:raise RuntimeError('\n'.join(errors))
        ledger['status']='complete'
    except BaseException:ledger['status']='stopped_no_retry';raise
    finally:save()


if __name__=='__main__':main()
