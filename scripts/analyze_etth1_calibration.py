"""Audit the completed timing job and conservatively estimate the native stage."""
import hashlib
import json
import math
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/'results/etth1-transfer/calibration-001'


def main():
    result=json.loads((DIR/'calibration.json').read_text())
    ledger=json.loads((DIR/'ledger.json').read_text())
    manifest=json.loads((ROOT/'results/etth1-transfer/prepared-v3/manifest.json').read_text())
    assert ledger['status']=='complete' and result['passed'] and result['device']=='cuda'
    assert not result['validation_scored'] and not result['test_scored']
    rows=result['rows'];assert len(rows)==4
    assert {(r['backbone'],r['mode']) for r in rows}=={(b,m) for b in ['micn','film'] for m in ['levels_direct','relative_residual']}
    checkpoints=[]
    for r in rows:
        assert r['epochs']==3 and r['train_samples']==6444
        for key in ['epoch_seconds','inner_forward_seconds']:
            assert len(r[key])==3 and all(math.isfinite(x) and x>0 for x in r[key])
        path=DIR/f"{r['backbone']}-{r['mode']}.pt"
        c=torch.load(path,map_location='cpu',weights_only=True)
        assert c['epoch']==3 and c['manifest']==manifest and c['features']==7
        assert c['backbone']==r['backbone'] and c['mode']==r['mode']
        assert c['optimizer_state_dict']['state'] and len(c['cuda_rng_states'])==1
        assert all(torch.isfinite(t).all() for t in c['state_dict'].values())
        assert all(k in c for k in ['sampler_rng_state','torch_rng_state','python_rng_state','numpy_rng_state'])
        checkpoints.append(dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    rates=ledger['rates']
    hourly=float(rates['gpu_hour_cost_l4'])+2*float(rates['cpu_hour_cost'])+4*float(rates['mem_gib_hour_cost'])
    jobs=[];models=[]
    for b in ['micn','film']:
        epoch=max(t+v for r in rows if r['backbone']==b for t,v in zip(r['epoch_seconds'],r['inner_forward_seconds']))
        # Preserve equal ceilings and timeout allowance across the paired formulations.
        timeout=max(300,math.ceil((epoch*150*1.5+120)/30)*30)
        reservation=(timeout+120)/3600*hourly*3+.05
        models.append(dict(backbone=b,conservative_epoch_seconds=epoch,timeout_per_fit_seconds=timeout,
            reservation_per_fit_usd=reservation,whole_stage_fits=6))
        for seed in [2201,2202,2203]:
            for mode in ['levels_direct','relative_residual']:
                jobs.append(dict(backbone=b,seed=seed,mode=mode,epochs=150,patience=20,timeout_seconds=timeout,reservation_usd=reservation))
    total=sum(j['reservation_usd'] for j in jobs)
    report=dict(calibration_passed=True,checkpoint_audit_passed=True,checkpoints=checkpoints,
        validation_scored=False,test_scored=False,calibration_conservative_accounted_usd=ledger['reservation_usd'],
        calibration_cap_usd=ledger['cap_usd'],hourly_rate_usd=hourly,models=models,jobs=jobs,
        next_stage_reservations_usd=total,next_stage_with_margin_reserve_usd=total*1.2+1,
        suggested_next_stage_cap_usd=math.ceil(total*1.2+1),next_stage_authorized=False,
        cost_method='Maximum observed paired train+inner-forward epoch time per backbone, including first epoch; x150 x1.5 allowance, +120s per fit overhead, timeout rounded up to30s. Reserve timeout+120 at3x rates +$.05/job. Twelve fits, 20% margin plus $1 reserve.',
        limitations='Three-epoch native calibration only. Gradient finiteness checks add overhead; full checkpoint-selection/serialization and development scoring remain to be validated. Estimate is for this frozen native workload, not all eight HyperDense arms or a final-test study. Final-test data stay unscored.')
    (DIR/'audit-and-cost-proposal.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['jobs','checkpoints']},indent=2))


if __name__=='__main__':main()
