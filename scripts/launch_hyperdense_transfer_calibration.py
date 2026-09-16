"""New $14 bounded calibration allocation: one gate, then eight two-L4 waves."""
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


def calibrate(manifest,data,job,directory,device='cuda',deadline=None):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset
    from hypercast4d.hyperdense_transfer import build
    from hypercast4d.hyperdense_transfer_training import run
    arrays=np.load(io.BytesIO(data),allow_pickle=False)
    if set(arrays.files)!={'train_x','train_y','inner_x'}:raise ValueError('Calibration bundle must omit evaluation labels and development inputs')
    model,control=build(job['backbone'],job['arm'],job['seed'],job['gain'])
    train=TensorDataset(torch.from_numpy(arrays['train_x']),torch.from_numpy(arrays['train_y']))
    inner=torch.from_numpy(arrays['inner_x'])
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    result=run(model,train,inner,directory,seed=job['seed'],epochs=job['epochs'],learning_rate=job['learning_rate'],schedule=job['schedule'],calibration=True,
        device=device,metadata=dict(manifest=manifest,job=job,control=control),deadline=deadline)
    # Replay the calibration's latest state, not a validation-selected "best" artifact.
    checkpoint=torch.load(Path(directory)/'latest.pt',weights_only=True,map_location='cpu')
    restored,_=build(job['backbone'],job['arm'],job['seed'],job['gain']);restored.load_state_dict(checkpoint['state_dict']);restored.to(device).eval();model.eval()
    from hypercast4d.hyperdense_transfer import ieee_precision
    with ieee_precision(),torch.inference_mode():
        x=inner[:32].to(device);left=model(x);right=restored(x);torch.testing.assert_close(left,right,atol=0,rtol=0)
    result.update(control=control,train_samples=len(train),inner_samples=len(inner),checkpoint_replay_passed=True,
        checkpoint_sha256=hashlib.sha256((Path(directory)/'latest.pt').read_bytes()).hexdigest(),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
    (Path(directory)/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def worker(manifest,data,job):
    import torch
    import hypercast4d.hyperdense_transfer as module
    torch.set_num_threads(2);root=Path(module.__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp);start=time.monotonic()
        try:
            if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name(0):raise RuntimeError('L4 required')
            for name,digest in manifest['source_sha256'].items():
                if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Source changed: {name}')
            if hashlib.sha256(data).hexdigest()!=manifest['data_sha256']:raise ValueError('Data changed')
            if job['kind']=='gate':
                from hypercast4d.hyperdense_transfer_checks import gate
                (path/'gate.json').write_text(json.dumps(gate('cuda'),indent=2,allow_nan=False))
            else:calibrate(manifest,data,job,path/'trial',deadline=start+job['timeout_seconds']-90)
            status=dict(ok=True,gpu=torch.cuda.get_device_name(0))
        except Exception:status=dict(ok=False,error=traceback.format_exc())
        status.update(job=job,elapsed_seconds=time.monotonic()-start,torch_version=str(torch.__version__),cuda_version=torch.version.cuda)
        (path/'status.json').write_text(json.dumps(status,indent=2))
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            for p in path.iterdir():archive.add(p,arcname=p.name)
        return status,buffer.getvalue()


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--allocation',type=Path,required=True);p.add_argument('--authorized-cap-usd',type=float,required=True);a=p.parse_args()
    from hypercast4d.representation_pilot import sha
    from launch_representation_pilot import admission
    manifest=json.loads((a.bundle/'manifest.json').read_text());data=(a.bundle/'calibration.npz').read_bytes();jobs=manifest['jobs']
    if sha(a.bundle/'calibration.npz')!=manifest['data_sha256']:raise ValueError('Data changed')
    for name,digest in manifest['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Source changed: {name}')
    cli=str(Path(sys.executable).with_name('modal'))
    def read(*args):return json.loads(subprocess.run([cli,*args,'--json'],capture_output=True,text=True,check=True,timeout=30).stdout)
    if any(int(x.get('tasks',0)) for x in read('app','list')):raise RuntimeError('Other Modal tasks active')
    rates=read('billing','rates');hourly=float(rates['gpu_hour_cost_l4'])+2*float(rates['cpu_hour_cost'])+4*float(rates['mem_gib_hour_cost']);costs=admission(jobs,hourly,a.authorized_cap_usd)
    import fcntl
    lock=(ROOT/'results/.representation-pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a.allocation.mkdir(parents=True,exist_ok=False)
    ledger=dict(cap_usd=a.authorized_cap_usd,reserve_usd=1,rates=rates,manifest_sha256=sha(a.bundle/'manifest.json'),attempts=[],status='admitted')
    def save():
        temp=a.allocation/'ledger.tmp';temp.write_text(json.dumps(ledger,indent=2));temp.replace(a.allocation/'ledger.json')
    save()
    import modal
    image=(modal.Image.debian_slim(python_version='3.12').pip_install('einops==0.8.2','scipy==1.18.1','numpy==2.5.3','openpyxl==3.1.5','pandas==3.0.5','torch==2.14.0')
        .env({'PYTHONPATH':'/opt/pilot/src:/opt/pilot/scripts'})
        .add_local_dir(ROOT/'src',remote_path='/opt/pilot/src',ignore=['**/__pycache__/**','**/web_dist/**'])
        .add_local_dir(ROOT/'scripts',remote_path='/opt/pilot/scripts',ignore=['**/__pycache__/**'])
        .add_local_dir(ROOT/'tests',remote_path='/opt/pilot/tests',ignore=['**/__pycache__/**']))
    app=modal.App('hypercast4d-hyperdense-etth1-calibration')
    remote=app.function(image=image,gpu='L4',cpu=2,memory=4096,retries=0,timeout=600,startup_timeout=120,max_containers=2,scaledown_window=2,serialized=True)(worker)
    waves=[[0]]+[list(range(i,min(i+2,len(jobs)))) for i in range(1,len(jobs),2)]
    try:
        for indexes in waves:
            attempts={}
            for i in indexes:
                attempts[i]=dict(index=i,job=jobs[i],reservation_usd=costs[i],status='reserved');ledger['attempts'].append(attempts[i])
            save();errors=[]
            with app.run():
                calls={}
                try:
                    for i in indexes:
                        calls[i]=remote.spawn(manifest,data,jobs[i]);attempts[i].update(status='submitted',call_id=calls[i].object_id);save()
                except Exception:errors.append(traceback.format_exc())
                for i,call in calls.items():
                    try:
                        status,archive=call.get();(a.allocation/f'job-{i:02d}.tar.gz').write_bytes(archive)
                        attempts[i].update(status='complete' if status['ok'] else 'failed',runtime=status)
                        if not status['ok']:errors.append(status['error'])
                    except Exception:
                        error=traceback.format_exc();attempts[i].update(status='failed',error=error);errors.append(error)
                    save()
                for i in set(indexes)-set(calls):attempts[i].update(status='submission_failed');save()
            if errors:raise RuntimeError('\n'.join(errors))
        ledger['status']='complete'
    except BaseException:ledger['status']='stopped_no_retry';raise
    finally:save()

if __name__=='__main__':main()
