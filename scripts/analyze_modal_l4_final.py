#!/usr/bin/env python3
"""Analyze terminal final jobs; retain missing budget-blocked controls explicitly."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_modal_l4_robustness import BACKBONES, VARIANTS, PAIRS, block_means, relative_effect


def analyze(root, output):
    state=json.loads((root/'controller/controller_state.json').read_text())
    candidates=[c for c in state['candidates'] if c.get('stage_name')=='locked-final']
    assert len(candidates)==27
    if any(c['status'] not in {'complete','blocked_by_budget','exhausted'} for c in candidates):
        raise ValueError('Wait until the final queue is terminal before comparing scores')
    for path,digest in state['frozen_inputs'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    missing=[dict(backbone=c['backbone'],variant=c['variant'],status=c['status']) for c in candidates if c['status']!='complete']
    summaries=[];losses={};reference=None;baseline=None;sources={};leads=[];paired=defaultdict(set)
    for c in candidates:
        if c['status']!='complete':continue
        p=root/'jobs'/c['job_id'];rows=list(csv.DictReader((p/'runs.csv').open()))
        curves=list(csv.DictReader((p/'learning_curves.csv').open()));initializations=json.loads((p/'initialization.json').read_text())
        request=json.loads((p/'request.json').read_text());parent=json.loads((root/'jobs'/c['parent_job_id']/'request.json').read_text())
        assert request['evaluation']==parent['evaluation'] and request['phase']=='final_test'
        assert len(rows)==2 and all(r['epochs_ran']=='150' and r['split']=='test' and r['fold']=='0' for r in rows)
        assert {int(r['seed']) for r in rows}=={101,211} and len(curves)==300 and len(initializations)==2
        for seed in [101,211]:
            part=[r for r in curves if int(r['seed'])==seed]
            assert len(part)==150 and {int(r['epoch']) for r in part}==set(range(1,151))
        assert all(np.isfinite(float(r[k])) for r in curves for k in ['train_loss','validation_loss'])
        for init in initializations:
            assert init['fold']==0 and init['protocol']=='matched-v1'
            if c['variant']!='original':
                for key in ['input_lift','core','forecast_head']:paired[c['backbone'],init['seed'],key].add(init['module_sha256'][key])
        prediction_path=p/'predictions.csv';sources[c['job_id']]=hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        groups=defaultdict(list)
        for row in csv.DictReader(prediction_path.open()):
            assert row['split']=='test' and row['fold']=='0';groups[int(row['seed'])].append(row)
        assert set(groups)=={101,211}
        per_seed=[]
        for seed in [101,211]:
            predictions=sorted(groups[seed],key=lambda r:(int(r['origin_row']),int(r['lead'])))
            keys=[(int(r['origin_row']),int(r['lead']),r['target_date']) for r in predictions]
            if reference is not None:assert keys==reference
            reference=keys
            values=np.array([[float(r[k]) for k in ['actual','prediction','persistence']] for r in predictions])
            assert np.isfinite(values).all()
            if baseline is not None:np.testing.assert_allclose(baseline,values[:,[0,2]],rtol=0,atol=1e-6)
            baseline=values[:,[0,2]]
            error=np.abs(values[:,1]-values[:,0]);persistence=np.abs(values[:,2]-values[:,0])
            saved=next(r for r in rows if int(r['seed'])==seed)
            np.testing.assert_allclose(error.mean(),float(saved['mae']),rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(error.mean()/persistence.mean(),float(saved['mae_ratio']),rtol=1e-5,atol=1e-6)
            assert len(keys)%5==0 and all([k[1] for k in keys[i:i+5]]==[1,2,3,4,5] for i in range(0,len(keys),5))
            per_seed.append(error.reshape(-1,5).mean(axis=1))
            for lead in range(1,6):
                mask=np.array([k[1]==lead for k in keys])
                leads.append(dict(backbone=c['backbone'],variant=c['variant'],seed=seed,lead=lead,
                    mae=float(error[mask].mean()),mse=float(np.mean((values[mask,1]-values[mask,0])**2))))
        losses[c['backbone'],c['variant']]=np.mean(per_seed,axis=0)
        losses['persistence','persistence']=persistence.reshape(-1,5).mean(axis=1)
        metrics=['mae','mse','mae_ratio','parameters','train_seconds','peak_gpu_allocated_mb','inference_ms_per_batch']
        assert all(np.isfinite(float(r[k])) for r in rows for k in metrics)
        summaries.append(dict(backbone=c['backbone'],variant=c['variant'],job_id=c['job_id'],
            learning_rate=c['evaluation']['learning_rate'],**{k:float(np.mean([float(r[k]) for r in rows])) for k in metrics},
            seed_ratios={r['seed']:float(r['mae_ratio']) for r in rows}))
    assert all(len(v)==1 for v in paired.values())
    columns=sorted(losses);values=np.column_stack([losses[k] for k in columns]);contrasts=[]
    for length in [60,30,120]:
        samples=block_means(values,length,np.random.default_rng(20260911));means=values.mean(axis=0)
        for b in BACKBONES:
            for a,control in PAIRS:
                key=(b,a);other=('persistence','persistence') if control=='persistence' else (b,control)
                row=dict(backbone=b,candidate=a,control=control,block_length=length)
                if key not in losses or other not in losses:
                    row.update(status='unavailable_missing_control',improvement_percent=None,ci95_low=None,ci95_high=None,family95_low=None,family95_high=None)
                else:
                    ai=columns.index(key);bi=columns.index(other)
                    effects=relative_effect(samples[:,ai],samples[:,bi]);q=np.quantile(effects,[.025,.975,.05/72,1-.05/72])
                    row.update(status='available',improvement_percent=float(relative_effect(means[ai],means[bi])),
                        ci95_low=float(q[0]),ci95_high=float(q[1]),family95_low=float(q[2]),family95_high=float(q[3]))
                contrasts.append(row)
    result=dict(matrix_complete=not missing,completed_configurations=len(summaries),planned_configurations=27,
        completed_fits=2*len(summaries),planned_fits=54,missing=missing,summaries=summaries,contrasts=contrasts,
        origins=len(values),effective_blocks={str(n):len(values)/n for n in [30,60,120]},resamples=10000,
        bootstrap_seed=20260911,multiplicity_family_size=36,source_prediction_sha256=sources,
        conservative_spend_usd=max(state['budget']['estimated_spend_usd'],state['budget']['confirmed_spend_usd']))
    output.mkdir(parents=True,exist_ok=True);(output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in [('contrasts',contrasts),('per_lead',leads)]:
        with (output/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lookup={(r['backbone'],r['variant']):r for r in summaries}
    text=['# Locked retrospective final results — incomplete matrix','',
        f"{2*len(summaries)}/54 fits completed, all at 150 epochs. The budget guard prevented the remaining controls from running. Missing: "+', '.join(f"{m['backbone']} {m['variant']}" for m in missing)+'.', '',
        'This is an incomplete final comparison. No aggregate winner or complete confirmation is declared. Available paired contrasts are reported transparently; unavailable controls are marked missing, never imputed or removed from the original 36-comparison correction. Earlier complete robustness results remain separate.', '',
        '## Descriptive final scores', '', 'Mean over seeds 101/211 of MAE/persistence; lower is better, 1 matches persistence.', '',
        '| Backbone | Original | Lift only | Real dense | 2D | 4D | 8D | Rank-8 | Rank-4 | Rank-2 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for b in BACKBONES:
        text.append('| '+b+' | '+' | '.join(f"{lookup[b,v]['mae_ratio']:.3f}" if (b,v) in lookup else 'not run' for v in VARIANTS)+' |')
    text+=['','## Available paired contrasts against real dense','',
        'Positive improvement favors HyperDense. Family-wise 95% intervals retain the full 36-contrast correction. Block length 60; lengths 30/120 are in the accompanying data. These are conditional, descriptive intervals on a historically exposed dataset.', '',
        '| Backbone | Dimension | Improvement % | Family-wise interval |','|---|---|---:|---|']
    for r in contrasts:
        if r['block_length']==60 and r['control']=='real_dense' and r['status']=='available':
            text.append(f"| {r['backbone']} | {r['candidate']} | {r['improvement_percent']:.1f} | [{r['family95_low']:.1f}, {r['family95_high']:.1f}] |")
    text+=['','## Limits','',f"The test has {len(values)} forecast origins, about {len(values)/60:.1f} effective blocks at length 60. Overlapping horizons, strong time dependence, two seeds, limited extreme-quantile precision, historical test exposure and the incomplete matrix restrict inference. Sensitivity-dependent effects are inconclusive; nonsignificance does not imply equivalence. Parameter/timing/seed summaries and prediction hashes are in analysis.json; per-lead errors and all available/missing contrasts are saved separately.", '',
        'Training/validation recipes were frozen before this final evaluation. No pending jobs were selected or modified using test scores, and no new recipe is proposed from these scores.']
    (output/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:result[k] for k in ['matrix_complete','completed_fits','missing','origins','conservative_spend_usd']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results-root',type=Path,default=Path('results/modal-l4-runner'))
    parser.add_argument('--out',type=Path,default=Path('results/modal-l4-runner/controller/final-analysis'))
    args=parser.parse_args();analyze(args.results_root,args.out)
