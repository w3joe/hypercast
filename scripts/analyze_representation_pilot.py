"""Audit returned pilot archives without selecting on outer/final data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import numpy as np
import torch
from hypercast4d.representation_pilot import advancement, score


def analyze(allocation,bundle):
    allocation,bundle=Path(allocation),Path(bundle)
    ledger=json.loads((allocation/'ledger.json').read_text())
    manifest=json.loads((bundle/'manifest.json').read_text())
    assert hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest()==ledger['manifest_sha256']
    data=np.load(bundle/'development.npz',allow_pickle=False)
    span,minimum=manifest['scaler_span'][0],manifest['scaler_minimum'][0]
    actual=data['inner_y']*span+minimum
    persistence=np.repeat(data['inner_x'][:,-1,0:1],5,axis=1)*span+minimum
    rows=[];inference=None;archives=[]
    for attempt in ledger['attempts']:
        if attempt['status']!='complete':continue
        path=allocation/f"job-{attempt['index']:02d}.tar.gz"
        archives.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        with tarfile.open(path) as archive:
            def read(name):return archive.extractfile(name).read()
            runtime=json.loads(read('runtime.json'));assert runtime['ok'] and 'L4' in runtime['gpu']
            if attempt['job']['kind']=='inference':
                inference=json.loads(read('artifacts/inference.json'))
                assert len(inference['layers'])==63 and len(inference['full_models'])==4
                continue
            scores=json.loads(read('artifacts/scores.json'));assert len(scores)==2
            shared=[]
            for row in scores:
                prefix='artifacts/'+row['mode']+'/'
                prediction=np.load(io.BytesIO(read(prefix+'predictions.npz')),allow_pickle=False)
                np.testing.assert_array_equal(prediction['actual'],actual)
                np.testing.assert_array_equal(prediction['persistence'],persistence)
                np.testing.assert_array_equal(prediction['target_start'],data['inner_target_start'])
                metrics=score(prediction['prediction'],actual,persistence)
                for k,v in metrics.items():np.testing.assert_allclose(v,row[k],rtol=1e-6,atol=1e-8)
                assert np.isfinite(prediction['prediction']).all()
                checkpoint_bytes=read(prefix+'checkpoint.pt')
                assert hashlib.sha256(checkpoint_bytes).hexdigest()==row['checkpoint_sha256']
                checkpoint=torch.load(io.BytesIO(checkpoint_bytes),map_location='cpu',weights_only=True)
                assert checkpoint['manifest']==manifest
                assert checkpoint['mode']==row['mode'] and checkpoint['seed']==row['seed']
                assert checkpoint['backbone']==row['backbone']==attempt['job']['backbone']
                assert checkpoint['seed']==attempt['job']['seed']
                shared.append(checkpoint['initialization']['paired_except_head_sha256'])
                curve=json.loads(read(prefix+'curve.json'))
                best=min(curve,key=lambda r:r['inner_mae'])
                assert best['epoch']==row['best_epoch']
                assert best['inner_mae']==row['best_validation_loss']
                row['mae_over_persistence']=row['mae']/row['persistence_mae']
                rows.append(row)
            assert shared[0]==shared[1]
    report=dict(ledger_status=ledger['status'],successful_fits=len(rows),rows=rows,archives=archives,
        conservative_accounted_usd=sum(a['reservation_usd'] for a in ledger['attempts']),cap_usd=ledger['cap_usd'],
        reserve_usd=ledger['reserve_usd'],limitations='Inner-development data only; checkpoint-selected scores are exploratory, not independent confirmation.')
    if inference:
        report['inference']=dict(layer_cases=len(inference['layers']),
            cached_median_layer_speedup=float(np.median([r['timings']['original']['median_ms']/r['timings']['cached']['median_ms'] for r in inference['layers']])),
            dense_median_layer_speedup=float(np.median([r['timings']['original']['median_ms']/r['timings']['dense']['median_ms'] for r in inference['layers']])),
            full_models=inference['full_models'])
    if len(rows)==12:
        report['development_gate']=advancement(rows)
        report['convergence_flags']=[{k:r[k] for k in ['backbone','seed','mode','best_epoch','epochs_ran']} for r in rows if r['convergence_review_required']]
        report['epoch_ceiling_fits']=[{k:r[k] for k in ['backbone','seed','mode','best_epoch','epochs_ran']} for r in rows if r['epochs_ran']==100]
        report['full_advancement_cleared']=report['development_gate']['advance'] and not report['convergence_flags'] and not report['epoch_ceiling_fits']
        report['advancement_interpretation']='Paired score criterion only. Ceiling fits require convergence review before declaring the full advancement gate cleared; no new stage is authorized.'
        report['paired_improvements']=[]
        for backbone in ['micn','film']:
            changes=[]
            for seed in [1101,1102,1103]:
                pair={r['mode']:r for r in rows if r['backbone']==backbone and r['seed']==seed}
                changes.append(100*(1-pair['relative_residual']['mae']/pair['levels_direct']['mae']))
            report['paired_improvements'].append(dict(backbone=backbone,mae_reduction_percent=changes,median_reduction_percent=float(np.median(changes))))
    (allocation/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','archives','inference']},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--allocation',required=True);p.add_argument('--bundle',required=True)
    a=p.parse_args();analyze(a.allocation,a.bundle)
