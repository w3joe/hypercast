"""Paired model/seed/origin reporting with simultaneous block-bootstrap intervals."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

def choose_block_length(run):
    ledger=json.loads((run/'ledger.json').read_text());lengths=[]
    for a in ledger['attempts']:
        j=a['job']
        if a['status']!='complete' or j['phase']!='development' or j['arm']!='real':continue
        p=np.load(run/'predictions'/f"{a['index']:03d}.npz",allow_pickle=False)
        e=np.abs(p['prediction']-p['actual']).mean(1);e=e-e.mean();den=np.dot(e,e)
        acf=np.array([np.dot(e[:-lag],e[lag:])/den if den else 0 for lag in range(1,193)])
        found=next((lag for lag in range(1,169) if np.all(np.abs(acf[lag-1:lag+23])<.1)),168)
        lengths.append(found)
    return int(np.clip(max(lengths,default=24),24,168))

def bootstrap_intervals(base,other,block,resamples=10000,seed=3401):
    # Input [contrast, paired seed, origin]. Resample complete forecast-origin errors.
    c,s,n=base.shape;assert other.shape==base.shape
    point=100*(base.mean((1,2))-other.mean((1,2)))/base.mean((1,2))
    joined=np.concatenate([base,other],axis=0);extended=np.concatenate([joined,joined],axis=2)
    prefix=np.concatenate([np.zeros((*joined.shape[:2],1)),np.cumsum(extended,axis=2)],axis=2)
    starts=np.arange(n);full=prefix[:,:,starts+block]-prefix[:,:,starts]
    count,tail=divmod(n,block)
    remainder=prefix[:,:,starts+tail]-prefix[:,:,starts] if tail else None
    rng=np.random.default_rng(seed);samples=[]
    for begin in range(0,resamples,100):
        b=min(100,resamples-begin);seed_ids=rng.integers(s,size=(b,s));indices=rng.integers(n,size=(b,s,count))
        totals=full[:,seed_ids[:,:,None],indices].sum((2,3))
        if tail:
            last=rng.integers(n,size=(b,s));totals+=remainder[:,seed_ids,last].sum(2)
        means=totals/(s*n);samples.append((100*(means[:c]-means[c:])/means[:c]).T)
    boot=np.concatenate(samples);sd=boot.std(0,ddof=1);safe=np.maximum(sd,1e-12)
    critical=np.quantile(np.max(np.abs((boot-point)/safe),axis=1),.95)
    return point,point-critical*sd,point+critical*sd

def analyze(run):
    run=Path(run);ledger=json.loads((run/'ledger.json').read_text());plan=json.loads((run/ledger.get('prepared_dir','prepared-v1')/'evaluation-plan.json').read_text())
    results=json.loads((run/'test-results.json').read_text());assert len(results)==360
    lookup={(r['job']['backbone'],r['job']['arm'],r['job']['seed']):r for r in results};assert len(lookup)==360
    errors={};audits=[];expected_actual=None;expected_origins=None
    for a in ledger['attempts']:
        j=a['job']
        if j['phase']!='test':continue
        assert a['status']=='complete' and a['runtime']['ok']
        p=np.load(run/'predictions'/f"{a['index']:03d}.npz",allow_pickle=False)
        if expected_actual is None:expected_actual=p['actual'];expected_origins=p['target_start']
        np.testing.assert_array_equal(p['actual'],expected_actual);np.testing.assert_array_equal(p['target_start'],expected_origins)
        assert expected_origins.min()==11520 and expected_origins.max()+4==14399
        assert all(np.isfinite(p[k]).all() for k in p.files)
        r=lookup[(j['backbone'],j['arm'],j['seed'])]
        loss=np.abs(p['prediction']-p['actual']);np.testing.assert_allclose(loss.mean(),r['mae'],rtol=1e-10)
        errors[(j['backbone'],j['arm'],j['seed'])]=loss.mean(1)
        r['lead_mae']=loss.mean(0).tolist()
        audits.append(dict(index=a['index'],backbone=j['backbone'],arm=j['arm'],seed=j['seed'],passed=True))
    summaries=[]
    for b in plan['backbones']:
        hashes=[]
        for arm in plan['arms']:
            rr=[lookup[(b,arm,s)] for s in plan['evaluation_seeds']]
            for s,r in zip(plan['evaluation_seeds'],rr):
                assert r['control']['untouched_sha256']==lookup[(b,'real',s)]['control']['untouched_sha256']
            summaries.append(dict(backbone=b,arm=arm,mae=float(np.mean([r['mae'] for r in rr])),
                seed_mae=[r['mae'] for r in rr],rmse=float(np.mean([r['rmse'] for r in rr])),
                parameters=rr[0]['control']['parameters'],selected_parameters=rr[0]['control']['selected_parameters'],
                lowrank_weight_budget_gap=rr[0]['control']['lowrank_weight_budget_gap'],
                mean_training_seconds=float(np.mean([r['training_elapsed_seconds'] for r in rr])),
                ceiling_fits=sum(r['ceiling_reached'] for r in rr),patience_exhausted_fits=sum(r['patience_exhausted'] for r in rr),
                peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in rr),
                latency_batch1_ms=float(np.median([r['inference']['1']['median_ms'] for r in rr])),
                latency_batch32_ms=float(np.median([r['inference']['32']['median_ms'] for r in rr])),
                lead_mae=np.mean([r['lead_mae'] for r in rr],axis=0).tolist()))
    contrasts=[(b,a) for b in plan['backbones'] for a in ('complex','quaternion','octonion')]
    base=np.stack([np.stack([errors[(b,'real',s)] for s in plan['evaluation_seeds']]) for b,a in contrasts])
    other=np.stack([np.stack([errors[(b,a,s)] for s in plan['evaluation_seeds']]) for b,a in contrasts])
    intervals={};block=plan['analysis']['block_length']
    for length in sorted({max(5,block//2),block,block*2,168}):
        point,low,high=bootstrap_intervals(base,other,length,plan['analysis']['bootstrap_resamples'])
        intervals[str(length)]=[dict(backbone=b,arm=a,improvement_pct=float(p),lower_pct=float(l),upper_pct=float(h)) for (b,a),p,l,h in zip(contrasts,point,low,high)]
    primary=intervals[str(block)];summ={(r['backbone'],r['arm']):r for r in summaries}
    for r in summaries:
        b=r['backbone'];r['improvement_vs_real_pct']=100*(summ[(b,'real')]['mae']-r['mae'])/summ[(b,'real')]['mae']
        r['improvement_vs_native_pct']=100*(summ[(b,'native')]['mae']-r['mae'])/summ[(b,'native')]['mae']
    sample=np.load(run/'predictions'/f"{audits[0]['index']:03d}.npz",allow_pickle=False)
    persistence=float(np.abs(sample['actual']-sample['persistence']).mean());seasonal=float(np.abs(sample['actual']-sample['seasonal_24']).mean())
    report=dict(complete=True,models=15,arms=8,seeds=3,fits=360,primary_block_length=block,
        bootstrap_intervals=intervals,summaries=summaries,audits=audits,persistence_mae=persistence,seasonal_mae=seasonal,
        primary_point_wins=sum(r['improvement_pct']>0 for r in primary),primary_adjusted_wins=sum(r['lower_pct']>0 for r in primary),
        practical_adjusted_wins=sum(r['lower_pct']>2 for r in primary),conservative_upper_usd=ledger['conservative_upper_usd'],
        adjusted_wins_all_block_lengths=sum(all(v[i]['lower_pct']>0 for v in intervals.values()) for i in range(45)),
        limitations=['Single dataset and selected internal sites, custom 32/5 task','Three paired seeds give limited seed uncertainty precision',
            'Native/low-rank secondary comparisons are descriptive','Ceiling-hit fits are not claimed converged','FiLM native is already complex'])
    (run/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    with (run/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    lines=['# All-15 dense versus HyperDense results','',
        f'Completed {len(results)} held-out evaluation fits: 15 backbones × eight arms × three paired seeds.',
        f'ETTh1, seven channels, OT target, 32-hour context, five-hour horizon. Persistence MAE: {persistence:.5f}; seasonal-24 MAE: {seasonal:.5f}.','',
        f"Of 45 primary comparisons, {report['primary_point_wins']} improve dense by point estimate; {report['primary_adjusted_wins']} have simultaneous 95% intervals above zero; {report['practical_adjusted_wins']} exceed the 2% practical threshold with those intervals.",'',
        'Positive percentages favour HyperDense. Brackets give simultaneous 95% intervals; all dimensions are retained.','',
        '| Model | Dense MAE | 2D improvement % [CI] | 4D improvement % [CI] | 8D improvement % [CI] |',
        '|---|---:|---:|---:|---:|']
    for b in plan['backbones']:
        rr=[r for r in primary if r['backbone']==b]
        cells=[f"{r['improvement_pct']:.2f} [{r['lower_pct']:.2f}, {r['upper_pct']:.2f}]" for r in rr]
        lines.append(f"| {b} | {summ[(b,'real')]['mae']:.5f} | "+' | '.join(cells)+' |')
    lines+=['','## Interpretation and limits','',*['- '+x for x in report['limitations']],
        f"- {sum(r['ceiling_fits'] for r in summaries)} fits reached the {plan['epochs']}-epoch ceiling; report convergence flags alongside scores.",
        f"- Conservative compute upper accounting: ${ledger['conservative_upper_usd']:.2f} of $80; this is not the final provider invoice.",
        '- Summary CSV and full JSON include native/low-rank scores, seed scores, parameters, timings, memory, audit and block-length sensitivity.','']
    (run/'report.md').write_text('\n'.join(lines));return report
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);analyze(p.parse_args().run)
