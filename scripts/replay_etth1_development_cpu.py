"""Post-hoc CPU portability diagnostic on already-exposed development data only."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import numpy as np
import torch
from hypercast4d.native_transfer import build
from hypercast4d.representation_pilot import score

ROOT=Path(__file__).resolve().parents[1]
ALLOCATION=ROOT/'results/etth1-transfer/development-001'


def run():
    torch.set_num_threads(2)
    bundle=ROOT/'results/etth1-transfer/prepared-v3'
    manifest=json.loads((bundle/'manifest.json').read_text())
    assert hashlib.sha256((bundle/'development.npz').read_bytes()).hexdigest()==manifest['bundle_sha256']
    data=np.load(bundle/'development.npz',allow_pickle=False)
    assert not any('test' in k for k in data.files)
    x=data['development_x'];span,offset=manifest['scaler_span'][0],manifest['scaler_minimum'][0]
    ledger=json.loads((ALLOCATION/'ledger.json').read_text());rows=[]
    previous=ALLOCATION/'cpu-replay.json'
    cached={r['index']:r for r in json.loads(previous.read_text())['rows']} if previous.exists() else {}
    for a in ledger['attempts']:
        if a['status']!='complete':continue
        archive_path=ALLOCATION/f"job-{a['index']:02d}.tar.gz"
        digest=hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if cached.get(a['index'],{}).get('archive_sha256')==digest:
            rows.append(cached[a['index']]);continue
        with tarfile.open(archive_path) as t:
            c=torch.load(io.BytesIO(t.extractfile('trial/best.pt').read()),map_location='cpu',weights_only=True)
            saved=np.load(io.BytesIO(t.extractfile('trial/development-predictions.npz').read()),allow_pickle=False)
            r=json.loads(t.extractfile('trial/result.json').read())
            model=build(r['backbone'],r['seed'],r['mode']).eval();model.load_state_dict(c['state_dict'])
            with torch.inference_mode():
                prediction=torch.cat([model(torch.from_numpy(x[i:i+32])) for i in range(0,len(x),32)]).numpy().astype(np.float64)*span+offset
            m=score(prediction,saved['actual'],saved['persistence'])
            rows.append(dict(index=a['index'],archive_sha256=digest,backbone=r['backbone'],seed=r['seed'],mode=r['mode'],cpu_mae=m['mae'],l4_mae=r['mae'],
                mae_difference=m['mae']-r['mae'],maximum_absolute_prediction_difference=float(np.max(np.abs(prediction-saved['prediction']))),
                strict_tolerance_passed=bool(np.allclose(prediction,saved['prediction'],rtol=2e-4,atol=2e-5))))
    pairs=[]
    for b in ['micn','film']:
        for seed in [2201,2202,2203]:
            rs={r['mode']:r for r in rows if r['backbone']==b and r['seed']==seed}
            if len(rs)<2:continue
            d,r=rs['levels_direct'],rs['relative_residual']
            pairs.append(dict(backbone=b,seed=seed,cpu_mae_reduction_percent=100*(1-r['cpu_mae']/d['cpu_mae']),l4_mae_reduction_percent=100*(1-r['l4_mae']/d['l4_mae'])))
    result=dict(completed_fits=len(rows),rows=rows,paired=pairs,rtol=2e-4,atol=2e-5,final_test_scored=False,
        interpretation='Diagnostic replay on CPU of already-scored development windows, not an independent evaluation. L4 predictions remain the frozen primary results. No tolerance was relaxed after observing mismatches.',
        cpu_torch_version=str(torch.__version__))
    (ALLOCATION/'cpu-replay.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':run()
