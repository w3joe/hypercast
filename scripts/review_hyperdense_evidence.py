"""Post-hoc synthesis of existing results only; no fitting or new-period scoring."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/hyperdense-evidence-review'


def main():
    OUT.mkdir(exist_ok=True)
    allocation=ROOT/'results/hyperdense-representation-pilot/allocation-001'
    sources=[allocation/'analysis.json',ROOT/'results/modal-l4-internal/controller/remaining-replication-analysis.json',
        ROOT/'results/modal-l4-internal/controller/internal-development-review.json']
    a=json.loads(sources[0].read_text());old=json.loads(sources[1].read_text())['scores']
    report=dict(method='Existing 207-origin development forecasts partitioned into three equal consecutive groups of 69 origins, retaining all five leads. Descriptive post-hoc diagnostics only; no uncertainty claims, model selection, new targets or fitting.',
                pilot=[],remaining_models=[],source_sha256={})
    for i in range(1,7):
        path=allocation/f'job-{i:02d}.tar.gz';sources.append(path)
        with tarfile.open(path) as archive:
            scores=json.loads(archive.extractfile('artifacts/scores.json').read());arrays={}
            for r in scores:
                f=np.load(io.BytesIO(archive.extractfile(f"artifacts/{r['mode']}/predictions.npz").read()),allow_pickle=False)
                arrays[r['mode']]={k:f[k] for k in f.files}
            d,r=arrays['levels_direct'],arrays['relative_residual']
            np.testing.assert_array_equal(d['actual'],r['actual']);np.testing.assert_array_equal(d['target_start'],r['target_start'])
            de=np.abs(d['prediction']-d['actual']).mean(1);re=np.abs(r['prediction']-r['actual']).mean(1)
            baseline=np.abs(r['persistence']-r['actual']).mean(0)
            report['pilot'].append(dict(backbone=scores[0]['backbone'],seed=scores[0]['seed'],
                relative_origin_win_fraction=float((re<de).mean()),
                chronological_third_improvement_percent=[float(100*(1-re[s].mean()/de[s].mean())) for s in np.array_split(np.arange(len(de)),3)],
                relative_per_lead_mae_over_persistence=(np.abs(r['prediction']-r['actual']).mean(0)/baseline).tolist(),
                direct_per_lead_mae_over_persistence=(np.abs(d['prediction']-d['actual']).mean(0)/baseline).tolist()))
    for b in sorted({r['backbone'] for r in old}):
        rows={r['variant']:r for r in old if r['backbone']==b}
        h=min((rows[k] for k in ['complex','quaternion','octonion']),key=lambda r:r['mae'])
        low=min((rows[k] for k in ['lowrank2','lowrank4','lowrank8']),key=lambda r:r['mae'])
        report['remaining_models'].append(dict(backbone=b,descriptive_best_hyper=h['variant'],
            best_hyper_mae_ratio=h['mae_ratio'],best_hyper_vs_native_mae_change_percent=100*(h['mae']/rows['native']['mae']-1),
            best_hyper_vs_best_lowrank_mae_change_percent=100*(h['mae']/low['mae']-1),best_lowrank=low['variant'],
            hyper_parameter_saving_vs_native_percent=100*(1-h['parameters']/rows['native']['parameters'])))
    report['pilot_mean_metrics']={}
    for b in ['micn','film']:
        report['pilot_mean_metrics'][b]={m:{k:float(np.mean([r[k] for r in a['rows'] if r['backbone']==b and r['mode']==m])) for k in ['mae','mse','bias']} for m in ['levels_direct','relative_residual']}
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report['source_sha256'][str(Path(__file__).resolve().relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'existing-results-review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['pilot'],indent=2))


if __name__=='__main__':main()
