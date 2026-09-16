"""Read-only CPU audit of completed GPU archives, optionally as they arrive."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time
import traceback

def sha_bytes(blob):return hashlib.sha256(blob).hexdigest()
def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def audit_one(run,attempt,plan):
    import numpy as np
    import torch
    from hypercast4d.tslib15 import build
    from hypercast4d.hyperdense_transfer_training import finite
    index=attempt['index'];job=attempt['job'];path=run/f'job-{index:03d}.tar.gz'
    # Hash the actual returned archive, independently of the collector's check.
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==attempt['archive_sha256']
    with tarfile.open(path) as archive:
        status=json.load(archive.extractfile('status.json'));result=json.load(archive.extractfile('trial/result.json'))
        assert status['ok'] and status['job']==job and 'L4' in status['gpu']
        assert result['job']==job and result['checkpoint_replay_passed'] and result['test_scored']
        best_blob=archive.extractfile('trial/best.pt').read();latest_blob=archive.extractfile('trial/latest.pt').read()
        prediction_blob=archive.extractfile('trial/predictions.npz').read()
    assert sha_bytes(best_blob)==result['best_checkpoint_sha256']
    assert sha_bytes(latest_blob)==result['latest_checkpoint_sha256']
    assert prediction_blob==(run/'predictions'/f'{index:03d}.npz').read_bytes()
    best=torch.load(io.BytesIO(best_blob),weights_only=True,map_location='cpu')
    latest=torch.load(io.BytesIO(latest_blob),weights_only=True,map_location='cpu');finite(best);finite(latest)
    assert latest['metadata']==best['metadata']
    assert latest['metadata']['source_sha256']==plan['source_sha256']
    assert latest['metadata']['job']==job and latest['metadata']['control']==result['control']
    assert latest['epoch']==result['epochs_ran']<=job['epochs']
    assert latest['history']==result['history'] and latest['best_epoch']==best['best_epoch']==result['best_epoch']
    assert latest['best_loss']==best['best_inner_mae']==result['best_inner_mae']
    assert min(x['inner_mae'] for x in latest['history'])==latest['best_loss']
    assert latest['config']['seed']==job['seed'] and latest['config']['learning_rate']==job['learning_rate']
    assert not latest['config']['calibration'] and latest['config']['patience']==20
    assert latest['optimizer_state_dict']['state'] and latest['rng']['cuda']
    assert latest['rng']['sampler'].numel()>0 and latest['scheduler_state_dict']['last_epoch']==latest['epoch']
    for name,value in best['state_dict'].items():
        assert torch.equal(value,latest['best_state_dict'][name]),name
    model,control=build(job['backbone'],job['arm'],job['seed'])
    # CPU normal/positional initialization is not promised bit-identical across
    # macOS ARM and Linux x86. Check structural reconstruction here; require
    # exact untouched-state pairing across the actual GPU comparison arms below.
    assert {k:v for k,v in control.items() if k!='untouched_sha256'}=={k:v for k,v in result['control'].items() if k!='untouched_sha256'}
    model.load_state_dict(best['state_dict'],strict=True)
    predictions=np.load(io.BytesIO(prediction_blob),allow_pickle=False)
    arrays=np.load(run/plan['_prepared_dir']/'test.npz',allow_pickle=False)
    span=plan['dataset_manifest']['scaler_span'][0];offset=plan['dataset_manifest']['scaler_minimum'][0]
    expected=arrays['test_y'].astype(float)*span+offset
    np.testing.assert_array_equal(predictions['actual'],expected)
    np.testing.assert_array_equal(predictions['target_start'],arrays['test_target_start'])
    np.testing.assert_array_equal(predictions['persistence'],np.repeat(arrays['test_x'][:,-1,0:1].astype(float),5,axis=1)*span+offset)
    np.testing.assert_array_equal(predictions['seasonal_24'],arrays['test_x'][:,np.arange(5)+8,0].astype(float)*span+offset)
    loss=predictions['prediction']-expected;assert np.isfinite(loss).all()
    np.testing.assert_allclose(np.abs(loss).mean(),result['mae'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(np.sqrt(np.square(loss).mean()),result['rmse'],rtol=1e-12,atol=1e-12)
    return dict(index=index,backbone=job['backbone'],arm=job['arm'],seed=job['seed'],passed=True,
        best_checkpoint_sha256=result['best_checkpoint_sha256'],latest_checkpoint_sha256=result['latest_checkpoint_sha256'],
        gpu_initial_untouched_sha256=result['control']['untouched_sha256'],cpu_initial_untouched_sha256=control['untouched_sha256'],
        cpu_gpu_initialization_byte_equal=control['untouched_sha256']==result['control']['untouched_sha256'],
        archive_bytes=path.stat().st_size,epochs=result['epochs_ran'],selected_epoch=result['best_epoch'])

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--watch',action='store_true');args=p.parse_args();run=args.run.resolve()
    ledger=json.loads((run/'ledger.json').read_text());prepared=ledger['prepared_dir']
    plan=json.loads((run/prepared/'evaluation-plan.json').read_text());plan['_prepared_dir']=prepared
    sys.path.insert(0,str(run/prepared/'frozen/src'))
    import torch
    torch.set_num_threads(2)
    out=run/'artifact-audit.json'
    report=json.loads(out.read_text()) if out.exists() else dict(passed=False,rows=[],errors=[],audit_source_sha256=sha_bytes(Path(__file__).read_bytes()))
    known={r['index'] for r in report['rows']}|{r['index'] for r in report['errors']}
    while True:
        ledger=json.loads((run/'ledger.json').read_text())
        for a in ledger['attempts']:
            if a['job']['phase']!='test' or a['status']!='complete' or a['index'] in known:continue
            try:
                row=audit_one(run,a,plan)
                for prior in report['rows']:
                    if (prior['backbone'],prior['seed'])==(row['backbone'],row['seed']):
                        assert prior['gpu_initial_untouched_sha256']==row['gpu_initial_untouched_sha256'],'GPU comparison arms have different untouched initialization'
                report['rows'].append(row)
            except Exception:report['errors'].append(dict(index=a['index'],error=traceback.format_exc()))
            known.add(a['index']);report.update(audited=len(report['rows']),failed=len(report['errors']),updated_epoch=time.time(),passed=len(report['rows'])==360 and not report['errors'])
            atomic(out,report)
            print(json.dumps(dict(audited=report['audited'],failed=report['failed'],last_index=a['index'])),flush=True)
        if not args.watch or len(known)>=360 or report['errors'] or time.time()>plan['deadline_epoch']:break
        if ledger['status']=='stopped_for_review' and not any(a['status'] in ('running','submitting') for a in ledger['attempts'] if a['job']['phase']=='test'):break
        time.sleep(20)
    atomic(out,report)
if __name__=='__main__':main()
