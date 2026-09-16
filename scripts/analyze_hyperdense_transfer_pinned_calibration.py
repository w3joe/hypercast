"""Audit returned train-only calibration artifacts and recost prospective stages."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile
import torch
from hypercast4d.hyperdense_transfer_training import finite

ROOT=Path(__file__).resolve().parents[1]


def analyze(allocation,bundle):
    torch.set_num_threads(2)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    ledger=json.loads((allocation/'ledger.json').read_text());manifest=json.loads((bundle/'manifest.json').read_text())
    assert sha(bundle/'manifest.json')==ledger['repair_manifest_sha256']
    assert sha(bundle/'calibration.npz')==manifest['data_sha256']
    for name,digest in manifest['source_sha256'].items():assert sha(ROOT/name)==digest,name
    with tarfile.open(bundle/'source.tar.gz') as source:
        for name,digest in manifest['source_sha256'].items():assert hashlib.sha256(source.extractfile(name).read()).hexdigest()==digest
    assert ledger['status']=='complete_after_manual_repair'
    prior=json.loads((allocation/'ledger-before-manual-repair.json').read_text())
    assert sha(allocation/'ledger-before-manual-repair.json')==manifest['prior_ledger_sha256']
    assert len(ledger['attempts'])==17 and len(prior['attempts'])==15
    for old,new in zip(prior['attempts'],ledger['attempts']):
        assert old['reservation_usd']==new['reservation_usd'] and old['job']==new['job']
        if old['index']<=12:assert (allocation/f"job-{old['index']:02d}.tar.gz").is_file()
    rows=[];audits=[];gates=[];pending=[]
    from hypercast4d.hyperdense_transfer_pinned import constants
    canonical=constants()
    for a in ledger['attempts'][15:]:
        assert a['status']=='complete'
        path=allocation/f"job-{a['index']:02d}.tar.gz"
        with tarfile.open(path) as t:
            status=json.loads(t.extractfile('status.json').read());assert status['ok'] and 'L4' in status['gpu']
            assert status['job']==a['job']
            gate=json.loads(t.extractfile('gate.json').read());assert gate['passed'] and gate['continuation']['passed'] and len(gate['rows'])==16
            assert all(r['passed'] for field in ['rows','equivalence','references'] for r in gate[field])
            gates.append(gate)
            for j,config in enumerate(a['job']['configurations']):
                result=json.loads(t.extractfile(f'trial-{j:02d}/result.json').read());blob=t.extractfile(f'trial-{j:02d}/latest.pt').read()
                assert hashlib.sha256(blob).hexdigest()==result['checkpoint_sha256']
                c=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True);finite(c)
                assert c['epoch']==3 and c['config']['calibration'] and c['metadata']['manifest']==manifest and c['metadata']['job']==config
                assert c['metadata']['control']==result['control'] and c['history']==result['history']
                assert c['optimizer_state_dict']['state'] and c['scheduler_state_dict']['last_epoch']==3
                assert c['rng']['cuda'] and c['rng']['sampler'].numel()>0
                assert c['best_loss'] is None and len(c['timings'])==3
                for name,value in c['state_dict'].items():torch.testing.assert_close(value,c['best_state_dict'][name],atol=0,rtol=0)
                assert result['epochs_ran']==3 and result['checkpoint_replay_passed'] and result['calibration'] and not result['validation_scored'] and not result['test_scored']
                assert result['train_samples']==6444 and result['inner_samples']==2156
                for i,h in enumerate(result['history']):
                    assert set(h)=={'epoch','lr','train_mae'} and h['epoch']==i+1
                    assert math.isclose(h['lr'],.001*.98**i,rel_tol=1e-12)
                assert len(result['timings'])==3 and all(r['total_seconds']>0 for r in result['timings'])
                control=result['control'];gate_row=next(r for r in gate['rows'] if r['backbone']==control['backbone'] and r['arm']==control['arm'])
                assert control['parameters']==gate_row['parameters'] and control['untouched_sha256']==gate_row['untouched_sha256']
                rows.append(result)
                audits.append(dict(index=a['index'],archive_sha256=sha(path),checkpoint_sha256=result['checkpoint_sha256'],passed=True))
                if config['backbone']=='film':
                    for name,value in canonical.items():torch.testing.assert_close(c['state_dict'][name],value,atol=0,rtol=0)
    assert {(r['control']['backbone'],r['control']['arm']) for r in rows}=={(r['backbone'],r['arm']) for r in gates[0]['rows']}
    assert len(rows)==16 and len(gates)==2
    for left,right in zip(gates[0]['rows'],gates[1]['rows']):assert left['untouched_sha256']==right['untouched_sha256']
    full=True
    report=dict(complete=full,completed_calibration_fits=len(rows),superseded_completed_fits=12,unsubmitted_reservations_retained=2,gate_passed=gate is not None,gates=gates,rows=rows,audits=audits,pending=pending,
        conservative_accounted_usd=sum(a['reservation_usd'] for a in ledger['attempts']),authorized_cap_usd=ledger['cap_usd'],
        validation_scored=False,development_scored=False,test_scored=False,frozen_sources_preserved=True)
    if full:
        rate=float(ledger['repair_rates']['gpu_hour_cost_l4'])+2*float(ledger['repair_rates']['cpu_hour_cost'])+4*float(ledger['repair_rates']['mem_gib_hour_cost'])
        legacy_rows=[]
        for index in range(1,13):
            with tarfile.open(allocation/f'job-{index:02d}.tar.gz') as t:
                r=json.loads(t.extractfile('trial/result.json').read())
                legacy_rows.append(r)
        report['cost_uses_superseded_timing_envelope']=True
        estimates={}
        for b in ['micn','film']:
            candidates=[(r['control']['arm'],t) for r in rows+legacy_rows if r['control']['backbone']==b for t in r['timings']]
            arm,worst=max(candidates,key=lambda v:v[1]['total_seconds']);timeout=math.ceil((150*1.5*worst['total_seconds']+120)/30)*30
            reserve=(timeout+120)/3600*rate*3+.05
            estimates[b]=dict(slowest_arm=arm,max_combined_epoch_seconds=worst['total_seconds'],timeout_seconds=timeout,reservation_per_fit_usd=reserve)
        stages={}
        for name,per_model in [('search_and_reranking',112),('confirmation',40)]:
            reserved=sum(r['reservation_per_fit_usd']*per_model for r in estimates.values())
            stages[name]=dict(fits=per_model*2,reservations_usd=reserved,admission_with_20pct_margin_and_1usd_reserve=reserved*1.2+1,proposed_new_cap_usd=math.ceil(reserved*1.2+1))
        report['cost_proposal']=dict(rate_used=rate,model_common_timeouts=estimates,stages=stages,inference_profiling_not_included=True,
            authorized=False,limitations='Three calibration epochs, with the conservative timing envelope also retaining superseded runs to cover host variability; no development or final-test scoring; timing includes validation-input forwarding and checkpoint I/O but not validation target arithmetic or end-of-fit artifact transfer. 1.5x epoch allowance plus overhead, 3x resource reservation, stage margin and reserve retained. Reprice and finish implementation before admission.')
    (allocation/'pinned-analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','audits','gate','gates']},indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--allocation',type=Path,default=ROOT/'results/hyperdense-etth1-v2/calibration-001');p.add_argument('--bundle',type=Path,default=ROOT/'results/hyperdense-etth1-v2/prepared-calibration-v2-pinned');a=p.parse_args();analyze(a.allocation,a.bundle)
