"""Audit returned development artifacts; no final-test access or scoring."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import numpy as np
import torch
from hypercast4d.representation_pilot import score

ROOT=Path(__file__).resolve().parents[1]


def finite_tensors(value):
    if isinstance(value,torch.Tensor):assert bool(torch.isfinite(value).all())
    elif isinstance(value,dict):
        for child in value.values():finite_tensors(child)
    elif isinstance(value,(tuple,list)):
        for child in value:finite_tensors(child)


def analyze(directory):
    directory=Path(directory)
    torch.set_num_threads(2)
    ledger=json.loads((directory/'ledger.json').read_text())
    plan_path=ROOT/'results/etth1-transfer/development-prepared-v1/plan.json'
    plan=json.loads(plan_path.read_text())
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest()==ledger['plan_sha256']
    bundle=ROOT/'results/etth1-transfer/prepared-v3'
    manifest=json.loads((bundle/'manifest.json').read_text())
    assert hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest()==plan['dataset_manifest_sha256']
    assert hashlib.sha256((bundle/'development.npz').read_bytes()).hexdigest()==manifest['bundle_sha256']
    data=np.load(bundle/'development.npz',allow_pickle=False)
    assert not any('test' in k for k in data.files)
    scale,offset=manifest['scaler_span'][0],manifest['scaler_minimum'][0]
    actual=data['development_y'].astype(np.float64)*scale+offset
    persistence=np.repeat(data['development_x'][:,-1,0:1].astype(np.float64),5,axis=1)*scale+offset
    seasonal=data['development_x'][:,np.arange(5)+32-24,0].astype(np.float64)*scale+offset
    starts=data['development_target_start'];assert actual.shape==(2876,5)
    thirds=np.array_split(np.arange(len(actual)),3)
    rows=[];audits=[];partial=[]
    for attempt in ledger['attempts']:
        path=directory/f"job-{attempt['index']:02d}.tar.gz"
        if attempt['status']!='complete':
            partial.append(dict(index=attempt['index'],status=attempt['status'],archive_present=path.exists()))
            continue
        with tarfile.open(path) as archive:
            def read(name):return archive.extractfile(name).read()
            status=json.loads(read('status.json'));assert status['ok'] and 'L4' in status['gpu']
            row=json.loads(read('trial/result.json'))
            for k in ['backbone','mode','seed','epochs','patience']:assert row[k]==attempt['job'][k]
            assert not row['test_scored'] and row['checkpoint_replay_passed']
            p=np.load(io.BytesIO(read('trial/development-predictions.npz')),allow_pickle=False)
            for k,v in [('actual',actual),('persistence',persistence),('seasonal_24_hour',seasonal),('target_start',starts)]:np.testing.assert_array_equal(p[k],v)
            assert p['prediction'].shape==actual.shape and np.isfinite(p['prediction']).all()
            metrics=score(p['prediction'],actual,persistence)
            for k,v in metrics.items():np.testing.assert_allclose(row[k],v,rtol=1e-10,atol=1e-12)
            assert np.isclose(row['seasonal_24_mae'],np.abs(seasonal-actual).mean())
            for which in ['best','latest']:
                blob=read(f'trial/{which}.pt')
                assert hashlib.sha256(blob).hexdigest()==row[f'{which}_checkpoint_sha256']
                checkpoint=torch.load(io.BytesIO(blob),weights_only=True,map_location='cpu')
                finite_tensors(checkpoint)
                assert checkpoint['metadata']['dataset_manifest']==manifest
                assert checkpoint['metadata']['stage_plan']==plan
                if which=='best':
                    assert checkpoint['best_epoch']==row['best_epoch']
                    assert checkpoint['best_inner_mae']==row['best_inner_mae']
                    best_state=checkpoint['state_dict']
                else:
                    assert checkpoint['epoch']==row['epochs_ran']
                    assert checkpoint['history']==row['history']
                    assert checkpoint['optimizer_state_dict']['state'] and checkpoint['rng']['cuda']
                    for name,value in best_state.items():torch.testing.assert_close(value,checkpoint['best_state_dict'][name],atol=0,rtol=0)
            from hypercast4d.native_transfer import build
            replay_model=build(row['backbone'],row['seed'],row['mode']).eval()
            replay_model.load_state_dict(best_state,strict=True)
            indexes=np.concatenate([np.arange(32),np.arange(1408,1440),np.arange(2848,2876)])
            with torch.inference_mode():
                replay=torch.cat([replay_model(torch.from_numpy(data['development_x'][part])) for part in [indexes[:32],indexes[32:64],indexes[64:]]]).numpy().astype(np.float64)*scale+offset
            replay_passed=bool(np.allclose(replay,p['prediction'][indexes],rtol=2e-4,atol=2e-5))
            replay_error=float(np.max(np.abs(replay-p['prediction'][indexes])))
            del replay_model
            history=row.pop('history');selected=min(history,key=lambda r:r['inner_mae'])
            assert selected['epoch']==row['best_epoch'] and selected['inner_mae']==row['best_inner_mae']
            assert len(history)==row['epochs_ran']<=150
            assert row['stale_epochs']==row['epochs_ran']-row['best_epoch']
            assert row['patience_exhausted']==(row['stale_epochs']>=20)
            row.update(mae_over_persistence=row['mae']/row['persistence_mae'],
                mae_over_seasonal24=row['mae']/row['seasonal_24_mae'],
                mse_over_persistence=row['mse']/float(np.square(persistence-actual).mean()),
                per_lead_mae_reduction_vs_persistence_percent=(100*(1-np.array(row['per_lead_mae'])/np.abs(persistence-actual).mean(axis=0))).tolist(),
                third_mae=[float(np.abs(p['prediction'][s]-actual[s]).mean()) for s in thirds])
            rows.append(row)
            audits.append(dict(index=attempt['index'],passed=True,predictions=int(actual.size),
                archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                cpu_replay_origins=len(indexes),cpu_replay_passed=replay_passed,cpu_replay_max_absolute_error=replay_error,cpu_replay_rtol=2e-4,cpu_replay_atol=2e-5))
    paired=[]
    for b in ['micn','film']:
        for seed in [2201,2202,2203]:
            pair={r['mode']:r for r in rows if r['backbone']==b and r['seed']==seed}
            if len(pair)!=2:continue
            d,r=pair['levels_direct'],pair['relative_residual']
            paired.append(dict(backbone=b,seed=seed,mae_reduction_percent=100*(1-r['mae']/d['mae']),
                mse_reduction_percent=100*(1-r['mse']/d['mse']),
                inner_mae_reduction_percent=100*(1-r['best_inner_mae']/d['best_inner_mae']),
                third_mae_reduction_percent=[100*(1-y/x) for x,y in zip(d['third_mae'],r['third_mae'])]))
    full=len(rows)==12 and ledger['status']=='complete'
    report=dict(status=ledger['status'],complete=full,completed_fits=len(rows),rows=rows,paired=paired,audits=audits,pending_or_failed=partial,
        conservative_accounted_usd=sum(r['reservation_usd'] for r in ledger['attempts']),cap_usd=ledger['cap_usd'],reserve_usd=ledger['reserve_usd'],
        baselines=dict(persistence_mae=float(np.abs(persistence-actual).mean()),seasonal_24_mae=float(np.abs(seasonal-actual).mean()),persistence_per_lead_mae=np.abs(persistence-actual).mean(axis=0).tolist()),
        convergence_flags=[{k:r[k] for k in ['backbone','mode','seed','epochs_ran','best_epoch','stale_epochs']} for r in rows if not r['patience_exhausted']],
        cpu_replay_flags=[a for a in audits if not a['cpu_replay_passed']],
        final_test_scored=False,limitations='Development evaluation on one new-to-project dataset, not independent final confirmation or evidence of HyperDense benefits. Three-seed summaries and chronological thirds are descriptive; no significance claim.')
    if full:
        report['model_summary']={b:dict(median_paired_mae_reduction_percent=float(np.median([p['mae_reduction_percent'] for p in paired if p['backbone']==b])),relative_wins=sum(p['mae_reduction_percent']>0 for p in paired if p['backbone']==b)) for b in ['micn','film']}
    (directory/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','audits']},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--allocation',type=Path,default=ROOT/'results/etth1-transfer/development-001')
    analyze(parser.parse_args().allocation)
