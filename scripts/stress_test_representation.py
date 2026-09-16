"""Frozen-checkpoint CPU evaluation on already-exposed outer Copper dates."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

import numpy as np
import torch
from hypercast4d.data import load_paper_data
from hypercast4d.internal_controls import nested_windows
from hypercast4d.representation_pilot import build_model, score, sha
from hypercast4d.training import predict

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/hyperdense-outer-stress-v1'
ALLOCATION=ROOT/'results/hyperdense-representation-pilot/allocation-001'


def main():
    torch.set_num_threads(2)
    manifest=json.loads((ROOT/'results/hyperdense-representation-pilot/prepared-v2-parallel/manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Frozen source changed: {name}')
    source=ROOT/'data/raw/paper_data.xlsx'
    assert sha(source)==manifest['source_data_sha256']
    frame=load_paper_data(source,'Copper')
    prepared,inner,audit=nested_windows(frame,32,5,.7,.15)
    for k in audit:assert audit[k]==manifest[k],k
    outer=prepared.validation;starts=outer.target_start
    raw=frame.to_numpy(dtype=np.float64)
    actual=raw[starts[:,None]+np.arange(5),0]
    persistence=np.repeat(raw[starts-1,0:1],5,axis=1)
    assert outer.y.shape==(297,5)
    np.testing.assert_allclose(prepared.scaler.inverse_target(outer.y),actual,atol=5e-7,rtol=0)
    cases=[]
    for i in range(1,7):
        path=ALLOCATION/f'job-{i:02d}.tar.gz'
        with tarfile.open(path) as archive:
            for row in json.loads(archive.extractfile('artifacts/scores.json').read()):
                name=f"artifacts/{row['mode']}/checkpoint.pt"
                blob=archive.extractfile(name).read()
                assert hashlib.sha256(blob).hexdigest()==row['checkpoint_sha256']
                cases.append(dict(archive=path.name,member=name,backbone=row['backbone'],seed=row['seed'],mode=row['mode'],checkpoint_sha256=row['checkpoint_sha256']))
    assert len(cases)==12
    OUT.mkdir(exist_ok=False)
    thirds=np.array_split(np.arange(297),3)
    freeze=dict(protocol='retrospective-outer-stress-v1',created_before_inference=True,
        cases=cases,source_data_sha256=sha(source),script_sha256=sha(__file__),audit=audit,
        outer_first_target=str(frame.index[starts[0]].date()),outer_last_target=str(frame.index[starts[-1]+4].date()),
        score='Raw-unit MAE primary; MSE, bias, per-lead errors, equal chronological thirds and origin above training maximum secondary',
        regime='Origin Copper exceeds the original training maximum; all leads retained together',
        device='cpu',threads=2,batch=32,training=False,selection=False,
        limitations='Already-exposed retrospective development period. All 12 unchanged checkpoints evaluated; no final-tail scoring or prospective confirmation.')
    (OUT/'manifest.json').write_text(json.dumps(freeze,indent=2)+'\n')
    maximum=manifest['scaler_minimum'][0]+manifest['scaler_span'][0]
    masks={'origin_above_train_max':raw[starts-1,0]>maximum,'origin_not_above_train_max':raw[starts-1,0]<=maximum}
    records=[]
    for case in cases:
        with tarfile.open(ALLOCATION/case['archive']) as archive:
            checkpoint=torch.load(io.BytesIO(archive.extractfile(case['member']).read()),weights_only=True,map_location='cpu')
            archived=np.load(io.BytesIO(archive.extractfile(f"artifacts/{case['mode']}/predictions.npz").read()))['prediction']
        assert checkpoint['manifest']==manifest
        model,_=build_model(case['backbone'],case['mode'],case['seed'])
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        # Cross-device replay gate on original inner predictions precedes outer scoring.
        replay=prepared.scaler.inverse_target(predict(model,inner.as_dataset(),torch.device('cpu'),32))
        np.testing.assert_allclose(replay,archived,atol=2e-5,rtol=2e-4)
        start=time.perf_counter()
        prediction=prepared.scaler.inverse_target(predict(model,outer.as_dataset(),torch.device('cpu'),32).astype(np.float64))
        assert np.isfinite(prediction).all()
        tag=f"{case['backbone']}-{case['seed']}-{case['mode']}"
        np.savez_compressed(OUT/(tag+'.npz'),prediction=prediction,actual=actual,persistence=persistence,target_start=starts)
        metrics=score(prediction,actual,persistence)
        r=dict(**case,**metrics,mae_over_persistence=metrics['mae']/metrics['persistence_mae'],
            cpu_inference_seconds=time.perf_counter()-start,inner_replay_max_abs_error=float(np.max(np.abs(replay-archived))),
            thirds=[score(prediction[s],actual[s],persistence[s]) for s in thirds],
            regimes={k:dict(origins=int(m.sum()),**score(prediction[m],actual[m],persistence[m])) for k,m in masks.items()})
        records.append(r)
        (OUT/'scores.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
        print(tag,round(r['mae_over_persistence'],4),flush=True)
    paired=[]
    for b in ['micn','film']:
        for seed in [1101,1102,1103]:
            p={r['mode']:r for r in records if r['backbone']==b and r['seed']==seed};d,r=p['levels_direct'],p['relative_residual']
            paired.append(dict(backbone=b,seed=seed,mae_reduction_percent=100*(1-r['mae']/d['mae']),
                third_reductions_percent=[100*(1-y['mae']/x['mae']) for x,y in zip(d['thirds'],r['thirds'])]))
    (OUT/'summary.json').write_text(json.dumps(dict(completed=12,paired=paired,limitations=freeze['limitations']),indent=2)+'\n')


if __name__=='__main__':main()
