"""Ten-worker controller with global time/spend limits and balanced admission."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from run_tslib15_worker import worker

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)
def makespan(seconds,slots=10):
    lanes=[0.]*slots
    for s in sorted(seconds,reverse=True):
        if not math.isfinite(s) or s<=0:raise ValueError('Invalid duration')
        k=min(range(slots),key=lambda i:lanes[i]);lanes[k]+=s
    return max(lanes)
def choose_rates(rows,backbones,arms):
    chosen=[]
    for b in backbones:
        for a in arms:
            candidates=[r for r in rows if r['job']['backbone']==b and r['job']['arm']==a]
            if len(candidates)!=2 or {r['job']['learning_rate'] for r in candidates}!={.0003,.001}:raise ValueError('Incomplete balanced tuning')
            r=min(candidates,key=lambda r:(r['mae'],r['job']['learning_rate']))
            if not math.isfinite(r['mae']):raise ValueError('Nonfinite score')
            chosen.append(dict(backbone=b,arm=a,learning_rate=r['job']['learning_rate']))
    return chosen
def timeout_table(rows,epochs=150):
    table={}
    for r in rows:
        j=r['job'];maximum=max(t['total_seconds'] for t in r['timings'])
        duration=math.ceil((epochs*maximum*1.25+60)/30)*30
        if duration>4700:raise ValueError('Single fit exceeds supported hard runtime')
        table[(j['backbone'],j['arm'])]=duration
    if len(table)!=120:raise ValueError('Incomplete calibration')
    return table
def freeze_test_bundle(prepared,plan,selection):
    import numpy as np
    import pandas as pd
    from prepare_etth1_transfer import partition,COLUMNS
    if len(selection)!=120 or len({(s['backbone'],s['arm']) for s in selection})!=120:raise ValueError('Incomplete selection')
    atomic_json(prepared/'selected-settings.json',selection)
    source=ROOT/'data/external/etth1-1d16c8f/ETTh1.csv';manifest=plan['dataset_manifest']
    if sha(source)!=manifest['source_sha256']:raise ValueError('Dataset changed')
    frame=pd.read_csv(source);raw=frame.iloc[:14400][COLUMNS].to_numpy(dtype=float)
    scaled=((raw-np.array(manifest['scaler_minimum']))/np.array(manifest['scaler_span'])).astype('float32')
    dev=np.load(prepared/'development.npz',allow_pickle=False)
    values={k:dev[k] for k in ('train_x','train_y','inner_x','inner_y')}
    values.update({'test_'+k:v for k,v in partition(scaled,11520,14400).items()})
    np.savez_compressed(prepared/'test.npz',**values)
    plan['data_sha256']['test']=sha(prepared/'test.npz')
    plan['selection_sha256']=sha(prepared/'selected-settings.json')
    atomic_json(prepared/'evaluation-plan.json',plan)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--prepared-version',default='v1');p.add_argument('--stage',choices=['calibration','recalibration','film-calibration','continue'],required=True);args=p.parse_args()
    import fcntl,modal
    run=args.run_dir.resolve();prepared=run/f'prepared-{args.prepared_version}';plan=json.loads((prepared/'plan.json').read_text())
    lock=(run/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    global_lock=(ROOT/'results/.representation-pilot.lock').open('a');fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if plan['cap_usd']!=80 or plan['max_l4']!=10:raise ValueError('Authorization mismatch')
    for name,digest in plan['source_sha256'].items():
        if sha(prepared/'frozen'/name)!=digest:raise ValueError(f'Frozen source changed: {name}')
    cli=str(Path(sys.executable).with_name('modal'))
    def read(*a):return json.loads(subprocess.run([cli,*a,'--json'],capture_output=True,text=True,check=True,timeout=30).stdout)
    if any(int(x.get('tasks',0)) for x in read('app','list')):raise RuntimeError('Other Modal tasks active')
    rates=read('billing','rates');hourly=float(rates['gpu_hour_cost_l4'])+2*float(rates['cpu_hour_cost'])+4*float(rates['mem_gib_hour_cost'])
    ledger_path=run/'ledger.json'
    if args.stage=='calibration':
        if ledger_path.exists():raise ValueError('Existing allocation')
        ledger=dict(cap_usd=80,max_l4=10,rates=rates,hourly_rate=hourly,attempts=[],sessions=[],status='calibration')
    else:ledger=json.loads(ledger_path.read_text())
    ledger['prepared_dir']=prepared.name
    if 'error' in ledger:ledger.setdefault('previous_errors',[]).append(ledger.pop('error'))
    ledger['status']=args.stage+'_running'
    session=dict(start_epoch=time.time(),stage=args.stage,hourly_rate=hourly,app_id=None);ledger['sessions'].append(session)
    mutex=threading.RLock();stop=threading.Event()
    def save():
        with mutex:
            now=time.time()
            ledger['conservative_upper_usd']=5+sum((s.get('end_epoch',now)-s['start_epoch'])*10*s['hourly_rate']/3600*1.1 for s in ledger['sessions'])
            atomic_json(ledger_path,ledger)
    def log(message):print(json.dumps(dict(utc_epoch=time.time(),message=message)),flush=True)
    save()
    image=(modal.Image.debian_slim(python_version='3.12')
        .pip_install('einops==0.8.2','scipy==1.18.1','numpy==2.5.3','openpyxl==3.1.5','pandas==3.0.5','torch==2.14.0')
        .env({'PYTHONPATH':'/opt/pilot/src:/opt/pilot/scripts','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        .add_local_dir(prepared/'frozen',remote_path='/opt/pilot',ignore=['**/__pycache__/**']))
    app=modal.App('hypercast4d-tslib15-seven-hour')
    remote=app.function(image=image,gpu='L4',cpu=(2,2),memory=(4096,4096),retries=0,timeout=4800,
        startup_timeout=120,max_containers=10,scaledown_window=60,serialized=True)(worker)
    def stage(jobs,phase,base_rows=()):
        ledger['status']=phase+'_running';save()
        data=(prepared/f'{phase}.npz').read_bytes()
        if hashlib.sha256(data).hexdigest()!=plan['data_sha256'][phase]:raise ValueError('Bundle changed')
        rows=list(base_rows)
        def task(job):
            with mutex:
                if stop.is_set():return None
                save()
                if time.time()+job['timeout_seconds']+60>plan['gpu_finish_epoch'] or ledger['conservative_upper_usd']>=73:
                    stop.set();raise RuntimeError('Deadline/budget admission stop')
                idx=len(ledger['attempts']);attempt=dict(index=idx,job=job,status='submitting',submitted_epoch=time.time());ledger['attempts'].append(attempt);save()
            call=None
            try:
                call=remote.spawn(plan,data,job)
                with mutex:attempt.update(status='running',call_id=call.object_id);save()
                status,blob=call.get(timeout=min(job['timeout_seconds']+180,plan['gpu_finish_epoch']-time.time()))
                path=run/f'job-{idx:03d}.tar.gz';path.write_bytes(blob)
                if not status['ok']:
                    with mutex:attempt.update(runtime=status,archive_sha256=sha(path));save()
                    raise RuntimeError(status['error'])
                extracted=[]
                with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
                    for member in archive.getmembers():
                        if member.name.endswith('/result.json'):
                            r=json.load(archive.extractfile(member));assert r['checkpoint_replay_passed'];extracted.append(r)
                        if member.name.endswith('/predictions.npz'):
                            light=run/'predictions';light.mkdir(exist_ok=True)
                            (light/f'{idx:03d}.npz').write_bytes(archive.extractfile(member).read())
                    if phase=='calibration' and job.get('run_gate',True):
                        g=json.load(archive.extractfile('gate.json'));assert g['passed']
                        atomic_json(run/f"gpu-gate-{job['backbone']}.json",g)
                with mutex:
                    attempt.update(status='complete' if status['ok'] else 'failed',runtime=status,archive_sha256=sha(path),ended_epoch=time.time());save()
                if not status['ok']:raise RuntimeError(status['error'])
                if len(extracted)!=(8 if phase=='calibration' and 'arm' not in job else 1):raise ValueError('Missing results')
                return extracted
            except BaseException:
                stop.set()
                if call is not None:
                    try:call.cancel()
                    except Exception:pass
                with mutex:attempt.update(status='failed',error=traceback.format_exc(),ended_epoch=time.time());save()
                raise
        ordered=sorted(jobs,key=lambda j:j['timeout_seconds'],reverse=True)
        with ThreadPoolExecutor(max_workers=10) as pool:
            pending={pool.submit(task,j) for j in ordered};errors=[]
            while pending:
                done,pending=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
                for f in done:
                    try:
                        result=f.result()
                        if result:rows.extend(result)
                    except BaseException:errors.append(traceback.format_exc())
                atomic_json(run/f'{phase}-results.json',rows);save()
                log(f'{phase}: {len(rows)} fits returned; {len(pending)} jobs remaining; errors={len(errors)}')
            if errors:raise RuntimeError('\n'.join(errors))
        return rows
    try:
        with app.run():
            session['app_id']=app.app_id;save()
            if args.stage in ('calibration','recalibration'):
                jobs=[dict(phase='calibration',backbone=b,seed=3102,learning_rate=.001,timeout_seconds=900) for b in plan['backbones']]
                rows=stage(jobs,'calibration');assert len(rows)==120
                ledger['status']='calibration_complete_pending_admission';save()
            elif args.stage=='film-calibration':
                existing=json.loads((run/'calibration-results.json').read_text())
                assert len(existing)==112 and all(r['job']['backbone']!='film' for r in existing)
                job=dict(phase='calibration',backbone='film',seed=3102,learning_rate=.001,timeout_seconds=900,arm='native',run_gate=True)
                rows=stage([job],'calibration',base_rows=existing)
                remaining=[{**job,'arm':arm,'run_gate':False} for arm in plan['arms'] if arm!='native']
                rows=stage(remaining,'calibration',base_rows=rows);assert len(rows)==120
                ledger['status']='calibration_complete_pending_admission';save()
            else:
                calibration=json.loads((run/'calibration-results.json').read_text());table=timeout_table(calibration)
                evaluation=[dict(phase='test',backbone=b,arm=a,seed=s,learning_rate=.001,timeout_seconds=table[(b,a)]) for b in plan['backbones'] for a in plan['arms'] for s in plan['evaluation_seeds']]
                tuning=[dict(phase='development',backbone=b,arm=a,seed=3201,learning_rate=lr,timeout_seconds=table[(b,a)]) for b in plan['backbones'] for a in plan['arms'] for lr in plan['learning_rates']]
                evtime=makespan([j['timeout_seconds']+30 for j in evaluation]);tutime=makespan([j['timeout_seconds']+30 for j in tuning]);available=plan['gpu_finish_epoch']-time.time()-300
                full=evtime+tutime<=available
                ceiling=150;alternatives=[]
                if not full and evtime>available:
                    for candidate in (125,100):
                        candidate_table=timeout_table(calibration,candidate)
                        candidate_time=makespan([candidate_table[(j['backbone'],j['arm'])]+30 for j in evaluation])
                        alternatives.append(dict(epochs=candidate,makespan_seconds=candidate_time))
                        if candidate_time<=available:
                            ceiling=candidate;table=candidate_table;evtime=candidate_time;break
                for j in evaluation:j.update(timeout_seconds=table[(j['backbone'],j['arm'])],epochs=ceiling)
                plan['epochs']=ceiling
                decision=dict(time_epoch=time.time(),evaluation_ceiling_seconds=evtime,tuning_ceiling_seconds=tutime,available_seconds=available,
                    mode='two_rate_tuning' if full else 'fixed_rate_fallback',epoch_ceiling=ceiling,alternatives=alternatives,
                    basis='Calibration max epoch × ceiling ×1.25 +60s per-fit budget +30s queue allowance; 10-slot longest-job-first simulation',
                    prospective_amendment='If fixed-rate 150 epochs does not fit, use a uniform 125 or 100 ceiling across all 15 models/eight arms/three seeds, chosen solely by calibration runtime before held-out scoring; retain patience 20 and flag ceiling hits')
                atomic_json(run/'admission.json',decision)
                if evtime>available:raise RuntimeError('Even balanced fixed-rate stage exceeds deadline estimate')
                if full:
                    development=stage(tuning,'development');selection=choose_rates(development,plan['backbones'],plan['arms'])
                else:selection=[dict(backbone=b,arm=a,learning_rate=.001) for b in plan['backbones'] for a in plan['arms']]
                # Freeze analysis block length using development only, or the declared fixed fallback.
                from analyze_tslib15 import choose_block_length
                plan['analysis']['block_length']=choose_block_length(run) if full else 24
                freeze_test_bundle(prepared,plan,selection)
                lookup={(s['backbone'],s['arm']):s['learning_rate'] for s in selection}
                for j in evaluation:j['learning_rate']=lookup[(j['backbone'],j['arm'])]
                test=stage(evaluation,'test');assert len(test)==360;ledger['status']='training_complete';save()
        session['end_epoch']=time.time();save()
        if args.stage=='continue':
            from analyze_tslib15 import analyze
            analyze(run);ledger['status']='complete'
    except BaseException:ledger['status']='stopped_for_review';ledger['error']=traceback.format_exc();raise
    finally:
        session['end_epoch']=time.time();save()
        log(f"Controller finished: {ledger['status']}; conservative upper ${ledger['conservative_upper_usd']:.2f}")
if __name__=='__main__':main()
