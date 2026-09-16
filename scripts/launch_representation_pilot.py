"""Explicit CLI launch for a NEW allocation only. Never run without user authorization."""
import argparse
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


def worker(manifest, data, job, require_cuda=True):
    import torch
    import hypercast4d.representation_pilot as pilot
    from hypercast4d.representation_pilot import run_pair, sha
    source_root=Path(pilot.__file__).resolve().parents[2]
    torch.set_num_threads(2)
    if require_cuda and (not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name(0)):
        raise RuntimeError('L4 required')
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);bundle=root/'bundle';bundle.mkdir()
        (bundle/'manifest.json').write_text(json.dumps(manifest))
        (bundle/'development.npz').write_bytes(data)
        error=None
        start=time.monotonic()
        try:
            for name,digest in manifest['source_sha256'].items():
                if sha(source_root/name)!=digest:raise ValueError(f'Worker source changed: {name}')
            if sha(bundle/'development.npz')!=manifest['data_sha256']:
                raise ValueError('Worker data changed')
            if job['kind']=='inference':
                from profile_hyperdense_l4_pilot import run
                run(root/'artifacts',manifest['inference_shapes'])
            else:
                run_pair(bundle,root/'artifacts',job['backbone'],job['seed'],
                    'cuda' if require_cuda else 'cpu',job.get('epochs',100))
        except Exception:
            error=traceback.format_exc()
        result=dict(ok=error is None,error=error,elapsed_seconds=time.monotonic()-start,
                    gpu=torch.cuda.get_device_name(0) if require_cuda else 'cpu',job=job)
        (root/'runtime.json').write_text(json.dumps(result,indent=2))
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            archive.add(root/'runtime.json',arcname='runtime.json')
            if (root/'artifacts').exists():archive.add(root/'artifacts',arcname='artifacts')
        return result,buffer.getvalue()


def reservation(timeout, hourly):
    return (timeout+120)/3600*hourly*3+.05


def admission(jobs,hourly,cap):
    import math
    if not math.isfinite(cap) or cap<=1 or not math.isfinite(hourly) or hourly<=0:
        raise ValueError('Invalid cap or rates')
    costs=[reservation(j['timeout_seconds'],hourly) for j in jobs]
    if sum(costs)*1.2+1>cap:
        raise ValueError('Complete stage plus 20% margin and $1 reserve exceeds cap')
    return costs


def job_waves(jobs, concurrency):
    if concurrency not in (1,2) or not jobs or jobs[0]['kind']!='inference':
        raise ValueError('One inference gate then at most two concurrent jobs')
    return [[0]]+[list(range(i,min(i+concurrency,len(jobs)))) for i in range(1,len(jobs),concurrency)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--allocation',type=Path,required=True,help='New directory; existing ledgers are rejected')
    p.add_argument('--authorized-cap-usd',type=float,required=True)
    args=p.parse_args()
    from hypercast4d.representation_pilot import sha
    manifest=json.loads((args.bundle/'manifest.json').read_text())
    if sha(args.bundle/'development.npz')!=manifest['data_sha256']:raise ValueError('Data changed')
    for name,digest in manifest['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Source changed: {name}')
    cli=str(Path(sys.executable).with_name('modal'))
    def read(*command):
        return json.loads(subprocess.run([cli,*command,'--json'],capture_output=True,text=True,check=True,timeout=30).stdout)
    apps=read('app','list')
    if any(int(a.get('tasks',0))>0 for a in apps):raise RuntimeError('Active Modal tasks; stop before admission')
    rates=read('billing','rates')
    hourly=float(rates['gpu_hour_cost_l4'])+2*float(rates['cpu_hour_cost'])+4*float(rates['mem_gib_hour_cost'])
    jobs=[dict(kind='inference',timeout_seconds=600)]+[dict(kind='training',**j) for j in manifest['jobs']]
    costs=admission(jobs,hourly,args.authorized_cap_usd)
    waves=job_waves(jobs,manifest['max_concurrent_l4'])
    args.allocation.mkdir(parents=True,exist_ok=False)
    ledger=dict(cap_usd=args.authorized_cap_usd,reserve_usd=1,rates=rates,
                manifest_sha256=sha(args.bundle/'manifest.json'),attempts=[],status='admitted',
                max_concurrent_l4=manifest['max_concurrent_l4'],waves=waves)
    def save():
        temp=args.allocation/'ledger.tmp';temp.write_text(json.dumps(ledger,indent=2));temp.replace(args.allocation/'ledger.json')
    save()
    import fcntl
    lock=(ROOT/'results/.representation-pilot.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import modal
    image=(modal.Image.debian_slim(python_version='3.12')
        .pip_install('einops==0.8.2','scipy==1.18.1','numpy==2.5.3','openpyxl==3.1.5','pandas==3.0.5','torch==2.14.0')
        .env({'PYTHONPATH':'/opt/pilot/src:/opt/pilot/scripts'})
        .add_local_dir(ROOT/'src',remote_path='/opt/pilot/src',ignore=['**/__pycache__/**','**/web_dist/**'])
        .add_local_dir(ROOT/'scripts',remote_path='/opt/pilot/scripts',ignore=['**/__pycache__/**'])
        .add_local_dir(ROOT/'tests',remote_path='/opt/pilot/tests',ignore=['**/__pycache__/**']))
    app=modal.App('hypercast4d-representation-pilot')
    remote=app.function(image=image,gpu='L4',cpu=2,memory=4096,retries=0,serialized=True,
                        max_containers=manifest['max_concurrent_l4'],scaledown_window=2,startup_timeout=120)(worker)
    data=(args.bundle/'development.npz').read_bytes()
    try:
        # Charge each full reservation BEFORE starting its app/build/remote attempt.
        for wave in waves:
            attempts={}
            for index in wave:
                attempt=dict(index=index,job=jobs[index],reservation_usd=costs[index],status='reserved')
                ledger['attempts'].append(attempt);attempts[index]=attempt
            save()
            errors=[]
            with app.run():
                calls={}
                for index in wave:
                    calls[index]=remote.with_options(timeout=jobs[index]['timeout_seconds']).spawn(manifest,data,jobs[index])
                    attempts[index].update(status='submitted',call_id=calls[index].object_id);save()
                # Drain both results even if one fails; never submit a later wave after failure.
                for index,call in calls.items():
                    try:
                        result,archive=call.get()
                        (args.allocation/f'job-{index:02d}.tar.gz').write_bytes(archive)
                        attempts[index].update(status='complete' if result['ok'] else 'failed',runtime=result)
                        if not result['ok']:errors.append(result['error'])
                    except Exception:
                        error=traceback.format_exc();attempts[index].update(status='failed',error=error);errors.append(error)
                    save()
            if errors:raise RuntimeError('\n'.join(errors))
        ledger['status']='complete'
    except BaseException:
        ledger['status']='stopped_no_retry'
        raise
    finally:
        save()


if __name__=='__main__':main()
