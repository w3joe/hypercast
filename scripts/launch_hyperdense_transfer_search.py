"""Separately authorized 192-setting search plus 32 paired reranking fits."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


def top_two(rows,backbone,arm):
    candidates=[r for r in rows if r['backbone']==backbone and r['arm']==arm and r['phase']=='search']
    if len(candidates)!=12 or {r['setting_id'] for r in candidates}!=set(range(12)):raise ValueError('Incomplete setting grid')
    if any(r['seed']!=2301 or not math.isfinite(r['mae']) for r in candidates):raise ValueError('Invalid search seed or score')
    return sorted(candidates,key=lambda r:(r['mae'],r['setting_id']))[:2]


def selected_settings(rows,backbones,arms):
    chosen=[]
    for b in backbones:
        for arm in arms:
            initial=top_two(rows,b,arm);scores=[]
            for r in initial:
                follow=[x for x in rows if x['backbone']==b and x['arm']==arm and x['phase']=='reranking' and x['setting_id']==r['setting_id']]
                if len(follow)!=1:raise ValueError('Missing or duplicated second seed')
                if follow[0]['seed']!=2302 or follow[0]['settings']!=r['settings'] or not math.isfinite(follow[0]['mae']):raise ValueError('Invalid reranking seed, settings or score')
                scores.append(dict(backbone=b,arm=arm,setting_id=r['setting_id'],mean_development_mae=(r['mae']+follow[0]['mae'])/2,
                    development_mae_by_seed={str(r['seed']):r['mae'],str(follow[0]['seed']):follow[0]['mae']},settings=r['settings']))
            chosen.append(min(scores,key=lambda r:(r['mean_development_mae'],r['setting_id'])))
    return chosen


def train_trial(plan,data,job,directory,device='cuda',deadline=None,epochs=150):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset
    from hypercast4d.hyperdense_transfer_pinned import build,ieee_precision
    from hypercast4d.hyperdense_transfer_training import run
    from hypercast4d.training import predict
    from hypercast4d.representation_pilot import score,sha
    arrays=np.load(io.BytesIO(data),allow_pickle=False)
    required={'train_x','train_y','inner_x','inner_y','development_x','development_y','development_target_start'}
    if set(arrays.files)!=required:raise ValueError('Unexpected or test arrays in development package')
    def dataset(name):return TensorDataset(torch.from_numpy(arrays[name+'_x']),torch.from_numpy(arrays[name+'_y']))
    setting=job['settings'];model,control=build(job['backbone'],job['arm'],job['seed'],setting['selected_weight_amplitude'])
    metadata=dict(plan=plan,job=job,control=control)
    result=run(model,dataset('train'),dataset('inner'),directory,seed=job['seed'],epochs=epochs,learning_rate=setting['learning_rate'],schedule=setting['schedule'],device=device,metadata=metadata,deadline=deadline)
    directory=Path(directory);c=torch.load(directory/'best.pt',weights_only=True,map_location='cpu')
    restored,_=build(job['backbone'],job['arm'],job['seed'],setting['selected_weight_amplitude']);restored.load_state_dict(c['state_dict']);restored.to(device)
    with ieee_precision():
        inner=predict(restored,dataset('inner'),torch.device(device),32)
        if not np.isclose(np.abs(inner-arrays['inner_y']).mean(),result['best_inner_mae'],rtol=2e-5,atol=1e-7):raise RuntimeError('Selected checkpoint replay failed')
        predicted=predict(restored,dataset('development'),torch.device(device),32)
    manifest=plan['dataset_manifest'];span,offset=manifest['scaler_span'][0],manifest['scaler_minimum'][0]
    predicted=predicted.astype(float)*span+offset;actual=arrays['development_y'].astype(float)*span+offset;x=arrays['development_x'].astype(float)
    persistence=np.repeat(x[:,-1,0:1],5,axis=1)*span+offset;seasonal=x[:,np.arange(5)+8,0]*span+offset
    if not np.isfinite(predicted).all():raise RuntimeError('Nonfinite forecasts')
    np.savez_compressed(directory/'predictions.npz',prediction=predicted,actual=actual,persistence=persistence,seasonal_24=seasonal,target_start=arrays['development_target_start'])
    result.update(**job,control=control,**score(predicted,actual,persistence),seasonal_24_mae=float(np.abs(actual-seasonal).mean()),checkpoint_replay_passed=True,
        best_checkpoint_sha256=sha(directory/'best.pt'),latest_checkpoint_sha256=sha(directory/'latest.pt'))
    (directory/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


def worker(plan,data,job):
    import torch
    import hypercast4d.hyperdense_transfer_pinned as module
    torch.set_num_threads(2);root=Path(module.__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp);start=time.monotonic()
        try:
            if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name(0):raise RuntimeError('L4 required')
            for name,digest in plan['source_sha256'].items():
                if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Source changed: {name}')
            if hashlib.sha256(data).hexdigest()!=plan['data_sha256']:raise ValueError('Data changed')
            train_trial(plan,data,job,path/'trial',deadline=start+job['timeout_seconds']-90)
            status=dict(ok=True,gpu=torch.cuda.get_device_name(0))
        except Exception:status=dict(ok=False,error=traceback.format_exc())
        status.update(job=job,elapsed_seconds=time.monotonic()-start)
        (path/'status.json').write_text(json.dumps(status,indent=2))
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            for p in path.iterdir():archive.add(p,arcname=p.name)
        return status,buffer.getvalue()


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--allocation',type=Path,required=True);p.add_argument('--authorized-cap-usd',type=float,required=True);a=p.parse_args()
    from hypercast4d.representation_pilot import sha
    from launch_representation_pilot import admission
    plan=json.loads((a.bundle/'plan.json').read_text());data=(a.bundle/'development.npz').read_bytes()
    if hashlib.sha256(data).hexdigest()!=plan['data_sha256']:raise ValueError('Data changed')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Source changed: {name}')
    cli=str(Path(sys.executable).with_name('modal'))
    def read(*args):return json.loads(subprocess.run([cli,*args,'--json'],capture_output=True,text=True,check=True,timeout=30).stdout)
    if any(int(x.get('tasks',0)) for x in read('app','list')):raise RuntimeError('Other Modal tasks active')
    rates=read('billing','rates');hourly=float(rates['gpu_hour_cost_l4'])+2*float(rates['cpu_hour_cost'])+4*float(rates['mem_gib_hour_cost'])
    if not math.isfinite(a.authorized_cap_usd) or a.authorized_cap_usd<=0:raise ValueError('Finite positive spending cap required')
    templates=plan['jobs'];costs=admission(templates,hourly,a.authorized_cap_usd)
    import fcntl
    lock=(ROOT/'results/.representation-pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a.allocation.mkdir(parents=True,exist_ok=False);ledger=dict(cap_usd=a.authorized_cap_usd,reserve_usd=1,rates=rates,plan_sha256=sha(a.bundle/'plan.json'),attempts=[],status='admitted')
    def save():
        tmp=a.allocation/'ledger.tmp';tmp.write_text(json.dumps(ledger,indent=2));tmp.replace(a.allocation/'ledger.json')
    save()
    import modal
    image=(modal.Image.debian_slim(python_version='3.12').pip_install('einops==0.8.2','scipy==1.18.1','numpy==2.5.3','openpyxl==3.1.5','pandas==3.0.5','torch==2.14.0')
        .env({'PYTHONPATH':'/opt/pilot/src:/opt/pilot/scripts'})
        .add_local_dir(ROOT/'src',remote_path='/opt/pilot/src',ignore=['**/__pycache__/**','**/web_dist/**'])
        .add_local_dir(ROOT/'scripts',remote_path='/opt/pilot/scripts',ignore=['**/__pycache__/**'])
        .add_local_dir(ROOT/'tests',remote_path='/opt/pilot/tests',ignore=['**/__pycache__/**']))
    app=modal.App('hypercast4d-hyperdense-etth1-search')
    remote=app.function(image=image,gpu='L4',cpu=2,memory=4096,retries=0,startup_timeout=120,max_containers=2,scaledown_window=2,serialized=True)(worker)
    rows=[]
    try:
        for start in range(0,len(templates),2):
            indexes=list(range(start,min(start+2,len(templates))));attempts={}
            for i in indexes:
                job=dict(templates[i])
                if job['phase']=='reranking':
                    selected=top_two(rows,job['backbone'],job['arm'])[job.pop('rank')];job.update(setting_id=selected['setting_id'],settings=selected['settings'])
                attempts[i]=dict(index=i,job=job,reservation_usd=costs[i],status='reserved');ledger['attempts'].append(attempts[i])
            save();errors=[]
            with app.run():
                calls={}
                try:
                    for i in indexes:
                        job=attempts[i]['job'];calls[i]=remote.with_options(timeout=job['timeout_seconds']).spawn(plan,data,job);attempts[i].update(status='submitted',call_id=calls[i].object_id);save()
                except Exception:errors.append(traceback.format_exc())
                for i,call in calls.items():
                    try:
                        status,blob=call.get();(a.allocation/f'job-{i:03d}.tar.gz').write_bytes(blob);attempts[i].update(status='complete' if status['ok'] else 'failed',runtime=status)
                        if status['ok']:
                            with tarfile.open(fileobj=io.BytesIO(blob)) as archive:row=json.loads(archive.extractfile('trial/result.json').read())
                            rows.append(row)
                        else:errors.append(status['error'])
                    except Exception:
                        error=traceback.format_exc();attempts[i].update(status='failed',error=error);errors.append(error)
                    save()
                for i in set(indexes)-set(calls):attempts[i].update(status='submission_failed');save()
            if errors:raise RuntimeError('\n'.join(errors))
        chosen=selected_settings(rows,plan['backbones'],plan['arms']);(a.allocation/'selected-settings.json').write_text(json.dumps(chosen,indent=2)+'\n');ledger['status']='complete'
    except BaseException:ledger['status']='stopped_no_retry';raise
    finally:save()

if __name__=='__main__':main()
