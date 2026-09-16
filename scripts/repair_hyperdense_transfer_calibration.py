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
    from hypercast4d.hyperdense_transfer_pinned import build
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
    import hypercast4d.hyperdense_transfer_pinned as module
    torch.set_num_threads(2);root=Path(module.__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp);start=time.monotonic()
        try:
            if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name(0):raise RuntimeError('L4 required')
            for name,digest in manifest['source_sha256'].items():
                if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Source changed: {name}')
            if hashlib.sha256(data).hexdigest()!=manifest['data_sha256']:raise ValueError('Data changed')
            from hypercast4d.hyperdense_transfer_pinned_checks import gate
            g=gate('cuda');(path/'gate.json').write_text(json.dumps(g,indent=2,allow_nan=False))
            for index,config in enumerate(job['configurations']):
                result=calibrate(manifest,data,config,path/f'trial-{index:02d}',deadline=start+job['timeout_seconds']-90)
                expected=next(r for r in g['rows'] if r['backbone']==config['backbone'] and r['arm']==config['arm'])
                if result['control']['untouched_sha256']!=expected['untouched_sha256']:raise RuntimeError('Worker initial-state pairing failed')
            status=dict(ok=True,gpu=torch.cuda.get_device_name(0))
        except Exception:status=dict(ok=False,error=traceback.format_exc())
        status.update(job=job,elapsed_seconds=time.monotonic()-start,torch_version=str(torch.__version__),cuda_version=torch.version.cuda)
        (path/'status.json').write_text(json.dumps(status,indent=2))
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            for p in path.iterdir():archive.add(p,arcname=p.name)
        return status,buffer.getvalue()


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--allocation',type=Path,required=True);a=p.parse_args()
    from hypercast4d.representation_pilot import sha
    from launch_representation_pilot import reservation
    import fcntl
    lock=(ROOT/'results/.representation-pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger_path=a.allocation/'ledger.json';ledger=json.loads(ledger_path.read_text());manifest=json.loads((a.bundle/'manifest.json').read_text());data=(a.bundle/'calibration.npz').read_bytes()
    if ledger['status']!='stopped_no_retry' or len(ledger['attempts'])!=15:raise ValueError('This repair is only for the audited stopped allocation')
    if sha(ledger_path)!=manifest['prior_ledger_sha256']:raise ValueError('Prior ledger changed')
    if sha(a.bundle/'calibration.npz')!=manifest['data_sha256']:raise ValueError('Data changed')
    for name,h in manifest['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError(f'Source changed: {name}')
    cli=str(Path(sys.executable).with_name('modal'))
    def read(*args):return json.loads(subprocess.run([cli,*args,'--json'],check=True,capture_output=True,text=True,timeout=30).stdout)
    if any(int(x['tasks']) for x in read('app','list')):raise RuntimeError('Active cloud tasks')
    rates=read('billing','rates');hourly=float(rates['gpu_hour_cost_l4'])+2*float(rates['cpu_hour_cost'])+4*float(rates['mem_gib_hour_cost']);costs=[reservation(j['timeout_seconds'],hourly) for j in manifest['jobs']]
    prior=sum(a['reservation_usd'] for a in ledger['attempts'])
    if (prior+sum(costs))*1.2+1>ledger['cap_usd']:raise ValueError('Repair plus every previous reservation exceeds allocation')
    def save():
        tmp=a.allocation/'ledger.tmp';tmp.write_text(json.dumps(ledger,indent=2));tmp.replace(ledger_path)
    for a0 in ledger['attempts']:
        if a0['index'] in (13,14):a0['status']='not_submitted_reservation_retained'
        if 1<=a0['index']<=12:a0['superseded_by_pinned_constants']=True
    ledger.update(status='manual_repair_admitted',repair_manifest_sha256=sha(a.bundle/'manifest.json'),repair_rates=rates)
    for i,(job,cost) in enumerate(zip(manifest['jobs'],costs),15):ledger['attempts'].append(dict(index=i,job=job,status='reserved',reservation_usd=cost))
    save()
    import modal
    image=(modal.Image.debian_slim(python_version='3.12').pip_install('einops==0.8.2','scipy==1.18.1','numpy==2.5.3','openpyxl==3.1.5','pandas==3.0.5','torch==2.14.0')
        .env({'PYTHONPATH':'/opt/pilot/src:/opt/pilot/scripts'})
        .add_local_dir(ROOT/'src',remote_path='/opt/pilot/src',ignore=['**/__pycache__/**','**/web_dist/**'])
        .add_local_dir(ROOT/'scripts',remote_path='/opt/pilot/scripts',ignore=['**/__pycache__/**'])
        .add_local_dir(ROOT/'tests',remote_path='/opt/pilot/tests',ignore=['**/__pycache__/**']))
    app=modal.App('hypercast4d-hyperdense-etth1-pinned-calibration')
    fn=app.function(image=image,gpu='L4',cpu=2,memory=4096,timeout=600,startup_timeout=120,max_containers=2,scaledown_window=2,retries=0,serialized=True)(worker)
    errors=[]
    try:
        with app.run():
            calls={}
            try:
                for i in (15,16):
                    attempt=ledger['attempts'][i];calls[i]=fn.spawn(manifest,data,attempt['job']);attempt.update(status='submitted',call_id=calls[i].object_id);save()
            except Exception:errors.append(traceback.format_exc())
            for i,call in calls.items():
                attempt=ledger['attempts'][i]
                try:
                    status,blob=call.get();(a.allocation/f'job-{i:02d}.tar.gz').write_bytes(blob);attempt.update(status='complete' if status['ok'] else 'failed',runtime=status)
                    if not status['ok']:errors.append(status['error'])
                except Exception:
                    error=traceback.format_exc();attempt.update(status='failed',error=error);errors.append(error)
                save()
            for i in set((15,16))-set(calls):ledger['attempts'][i]['status']='submission_failed';save()
        if errors:raise RuntimeError('\n'.join(errors))
        ledger['status']='complete_after_manual_repair'
    except BaseException:ledger['status']='stopped_after_manual_repair';raise
    finally:save()

if __name__=='__main__':main()
