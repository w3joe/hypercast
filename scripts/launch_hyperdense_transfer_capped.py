"""New staged $100 allocation; frozen training implementation, balanced 16-fit blocks."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
from launch_hyperdense_transfer_search import worker,top_two,selected_settings
ROOT=Path(__file__).resolve().parents[1]

def reservation(seconds,rate):
    if not math.isfinite(seconds) or seconds<0 or not math.isfinite(rate) or rate<=0:raise ValueError('Invalid resource estimate')
    return (seconds+420)/3600*rate*1.25+.05

def charged(attempt):return attempt.get('settled_usd',attempt['reservation_usd'])

def admit(attempts,jobs,rate,cap):
    if not math.isfinite(cap) or cap<=10:raise ValueError('Invalid cap')
    return sum(charged(a) for a in attempts)+sum(reservation(j['timeout_seconds'],rate) for j in jobs)+10<=cap

def settle(attempt,elapsed,rate):
    # Only after the app stopped and archives were received; retain failed reservations.
    if attempt['status']!='complete':return
    actual=reservation(elapsed,rate)
    if actual>attempt['reservation_usd']:raise RuntimeError('Observed accounting exceeded reservation; stop')
    attempt.update(settled_usd=actual,settled_wall_seconds=elapsed,settlement_basis='spawn through stopped app; 420s overhead and 25% margin retained')

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--allocation',type=Path,required=True);p.add_argument('--authorized-cap-usd',type=float,required=True);a=p.parse_args()
    import fcntl
    from hypercast4d.representation_pilot import sha
    lock=(ROOT/'results/.representation-pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((a.bundle/'plan.json').read_text());data=(a.bundle/'development.npz').read_bytes()
    if sha(a.bundle/'development.npz')!=plan['data_sha256']:raise ValueError('Data changed')
    for name,h in plan['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError(f'Source changed: {name}')
    cli=str(Path(sys.executable).with_name('modal'))
    def read(*args):return json.loads(subprocess.run([cli,*args,'--json'],capture_output=True,text=True,check=True,timeout=30).stdout)
    if any(int(x.get('tasks',0)) for x in read('app','list')):raise RuntimeError('Other Modal tasks active')
    rates=read('billing','rates');rate=float(rates['gpu_hour_cost_l4'])+2*float(rates['cpu_hour_cost'])+4*float(rates['mem_gib_hour_cost'])
    if a.authorized_cap_usd!=100 or plan['authorized_cap_usd']!=100:raise ValueError('This allocation is authorized for $100 only')
    a.allocation.mkdir(parents=True,exist_ok=False)
    ledger=dict(cap_usd=100,protected_reserve_usd=10,rates=rates,hourly_rate=rate,plan_sha256=sha(a.bundle/'plan.json'),attempts=[],status='admitted',accounting='new staged rule; old ledgers unchanged',completed_blocks=0)
    def save():
        ledger['accounted_usd']=sum(charged(x) for x in ledger['attempts']);tmp=a.allocation/'ledger.tmp';tmp.write_text(json.dumps(ledger,indent=2));tmp.replace(a.allocation/'ledger.json')
    def log(message):print(json.dumps(dict(time=time.time(),message=message,accounted_usd=ledger.get('accounted_usd'))),flush=True)
    save()
    import modal
    image=(modal.Image.debian_slim(python_version='3.12').pip_install('einops==0.8.2','scipy==1.18.1','numpy==2.5.3','openpyxl==3.1.5','pandas==3.0.5','torch==2.14.0')
        .env({'PYTHONPATH':'/opt/pilot/src:/opt/pilot/scripts'})
        .add_local_dir(ROOT/'src',remote_path='/opt/pilot/src',ignore=['**/__pycache__/**','**/web_dist/**'])
        .add_local_dir(ROOT/'scripts',remote_path='/opt/pilot/scripts',ignore=['**/__pycache__/**'])
        .add_local_dir(ROOT/'tests',remote_path='/opt/pilot/tests',ignore=['**/__pycache__/**']))
    app=modal.App('hypercast4d-hyperdense-etth1-capped-search')
    remote=app.function(image=image,gpu='L4',cpu=2,memory=4096,retries=0,startup_timeout=120,max_containers=2,scaledown_window=2,serialized=True)(worker)
    rows=[]
    try:
        for block_id,templates in enumerate(plan['blocks']):
            if not admit(ledger['attempts'],templates,rate,100):ledger['status']='stopped_budget_before_balanced_block';save();log('Budget stop; no partial new block admitted');return
            offset=len(ledger['attempts'])
            for template in templates:
                job=dict(template)
                if job['phase']=='reranking':
                    winner=top_two(rows,job['backbone'],job['arm'])[job.pop('rank')];job.update(setting_id=winner['setting_id'],settings=winner['settings'])
                ledger['attempts'].append(dict(index=len(ledger['attempts']),block=block_id,job=job,reservation_usd=reservation(job['timeout_seconds'],rate),status='reserved'))
            save();log(f'Admitted balanced block {block_id+1}/14')
            for start in range(offset,offset+16,2):
                attempts=ledger['attempts'][start:start+2];errors=[];started={}
                with app.run():
                    calls={}
                    try:
                        for attempt in attempts:
                            i=attempt['index'];started[i]=time.monotonic();calls[i]=remote.with_options(timeout=attempt['job']['timeout_seconds']).spawn(plan,data,attempt['job']);attempt.update(status='submitted',call_id=calls[i].object_id);save()
                    except Exception:errors.append(traceback.format_exc())
                    for attempt in attempts:
                        i=attempt['index']
                        if i not in calls:attempt['status']='submission_failed';save();continue
                        try:
                            remaining=max(1,attempt['job']['timeout_seconds']+420-(time.monotonic()-started[i]))
                            status,blob=calls[i].get(timeout=remaining)
                            path=a.allocation/f'job-{i:03d}.tar.gz';path.write_bytes(blob)
                            attempt.update(status='complete' if status['ok'] else 'failed',runtime=status,archive_sha256=sha(path))
                            if status['ok']:
                                with tarfile.open(fileobj=io.BytesIO(blob)) as t:row=json.loads(t.extractfile('trial/result.json').read())
                                assert row['backbone']==attempt['job']['backbone'] and row['arm']==attempt['job']['arm'] and row['checkpoint_replay_passed']
                                rows.append(row)
                            else:errors.append(status['error'])
                        except Exception:attempt.update(status='failed',error=traceback.format_exc());errors.append(attempt['error'])
                        save()
                ended=time.monotonic()
                for attempt in attempts:
                    if attempt['index'] in started:settle(attempt,ended-started[attempt['index']],rate)
                (a.allocation/'development-results.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n');save();log(f'Completed wave ending at fit {start+2}; errors={len(errors)}')
                if errors:raise RuntimeError('\n'.join(errors))
            ledger['completed_blocks']+=1;save()
        chosen=selected_settings(rows,plan['backbones'],plan['arms']);(a.allocation/'selected-settings.json').write_text(json.dumps(chosen,indent=2)+'\n');ledger['status']='complete';log('All 224 fits completed')
    except BaseException:ledger['status']='stopped_no_retry';raise
    finally:save()
if __name__=='__main__':main()
