"""Freeze ETTh1 roles without exposing test tensors to development code."""
import json
from pathlib import Path
import tarfile
import numpy as np
import pandas as pd
from hypercast4d.representation_pilot import sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/etth1-transfer/prepared-v3'
COLUMNS=['OT','HUFL','HULL','MUFL','MULL','LUFL','LULL']
BOUNDS={'train':(0,6480),'inner':(6480,8640),'development':(8640,11520),'test':(11520,14400)}


def partition(values,start,end,window=32,horizon=5):
    starts=np.arange(max(window,start),end-horizon+1)
    if not len(starts):raise ValueError('No complete windows')
    return dict(x=np.stack([values[t-window:t] for t in starts]),
                y=np.stack([values[t:t+horizon,0] for t in starts]),target_start=starts)


def prepare():
    source=ROOT/'data/external/etth1-1d16c8f/ETTh1.csv'
    frame=pd.read_csv(source,parse_dates=['date'])
    if len(frame)<14400 or set(frame.columns)!={'date',*COLUMNS}:raise ValueError('Unexpected ETTh1 schema')
    if not frame.date.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():raise ValueError('Nonhourly or duplicate dates')
    # Never summarize/scale the final-test target values for development.
    raw=frame.iloc[:11520][COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():raise ValueError('Invalid development inputs')
    minimum=raw[:6480].min(0);span=raw[:6480].max(0)-minimum;span[span==0]=1
    scaled=((raw-minimum)/span).astype('float32')
    arrays={}
    for label,(a,b) in BOUNDS.items():
        if label=='test':continue
        for key,value in partition(scaled,a,b).items():arrays[label+'_'+key]=value
    OUT.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(OUT/'development.npz',**arrays)
    source_paths=list((ROOT/'src/hypercast4d').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'tests/test_native_transfer.py',ROOT/'scripts/calibrate_etth1_transfer.py',ROOT/'scripts/launch_representation_pilot.py']
    manifest=dict(protocol='etth1-native-transfer-v1',status='prepared_no_paid_authorization',
        dataset='ETTh1',source_url='https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv',
        source_commit='1d16c8f4f943005d613b5bc962e9eeb06058cf07',source_sha256=sha(source),bundle_sha256=sha(OUT/'development.npz'),
        file_rows=len(frame),columns=COLUMNS,target='OT',frequency='hourly',context=32,horizon=5,
        task='Custom short-horizon target forecasting using all seven channels; not standard long-horizon benchmark reproduction',
        partitions={k:dict(rows=list(v),first_target_date=str(frame.date.iloc[max(32,v[0])]),last_target_date=str(frame.date.iloc[v[1]-1]),origins=v[1]-max(32,v[0])-4) for k,v in BOUNDS.items()},
        unused_tail_rows=[14400,len(frame)],scaler_rows=[0,6480],scaler_minimum=minimum.tolist(),scaler_span=span.tolist(),
        baselines=['last_observation','seasonal_24_hour'],baseline_information='Seasonal lead h uses target history at h-24, within the same 32-hour context for h=1..5',
        roles='Train fits weights/scaler. Inner selects checkpoints. Development compares packages/settings. Final test withheld from development tensors and must be scored only after protocol freeze.',
        calibration=dict(backbones=['micn','film'],modes=['levels_direct','relative_residual'],seed=2100,epochs=3,time_inner_forward_without_targets=True,score_inner=False,score_development=False,score_test=False,max_l4=1,timeout_seconds=600),
        proposed_development=dict(seeds=[2201,2202,2203],fits=12,epochs=150,patience=20,loss='mae',optimizer='Adam',learning_rate=.001,betas=[.9,.999],epsilon=1e-7,batch_size=32,checkpoint='inner_MAE',ranking='development_MAE',concurrency_l4=2),
        selection_rule='Keep both models; report all paired development differences. Reconsider package if evidence disagrees. No automatic HyperDense stage or final-test scoring.',
        source_code_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(source_paths)},
        prior_use_audit='Repository search found ETTh1 only in proposal documentation, no previous run manifests. Public benchmark status does not prove absence of external exposure.',
        limitations='New dataset for this project, not a prospective collection. This manifest freezes partitions before training; successful independent confirmation still requires frozen model/settings and no test-directed changes.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(OUT/'source.tar.gz','w:gz') as archive:
        for path in source_paths:archive.add(path,arcname=str(path.relative_to(ROOT)))
    print(json.dumps(dict(partitions=manifest['partitions'],file_rows=len(frame),source_sha256=manifest['source_sha256']),indent=2))


if __name__=='__main__':prepare()
