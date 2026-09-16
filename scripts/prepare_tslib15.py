"""Prepare versioned development/calibration inputs and freeze source."""
import hashlib
import json
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
from hypercast4d.tslib15 import BACKBONES,ARMS,PROTOCOL

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/tslib15-20260915'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(version='v1'):
    out=RUN/f'prepared-{version}';out.mkdir(exist_ok=False)
    auth=json.loads((RUN/'authorization.json').read_text())
    gates=json.loads((RUN/'cpu-gates.json').read_text())
    assert len(gates)==15 and all(g['passed'] and len(g['rows'])==8 for g in gates)
    old=ROOT/'results/etth1-transfer/prepared-v3';manifest=json.loads((old/'manifest.json').read_text())
    source=ROOT/'data/external/etth1-1d16c8f/ETTh1.csv'
    assert sha(source)==manifest['source_sha256'] and sha(old/'development.npz')==manifest['bundle_sha256']
    arrays=np.load(old/'development.npz',allow_pickle=False)
    calibration={k:arrays[k] for k in ('train_x','train_y','inner_x')}
    development={k:arrays[k] for k in ('train_x','train_y','inner_x','inner_y','development_x','development_y','development_target_start')}
    np.savez_compressed(out/'calibration.npz',**calibration);np.savez_compressed(out/'development.npz',**development)
    # Test data is created only after the controller freezes selected settings.
    files=list((ROOT/'src').rglob('*.py'))+list((ROOT/'src/hypercast4d/_constants').glob('*.pt'))
    files += [ROOT/'scripts'/n for n in ('run_tslib15_worker.py','run_tslib15_controller.py','prepare_tslib15.py','analyze_tslib15.py')]
    files += [ROOT/'docs/tslib_15_seven_hour_plan.md',ROOT/'docs/tslib_15_execution.md',ROOT/'tests/test_tslib15.py',ROOT/'scripts/plot_tslib15.py']
    hashes={}
    for p in files:
        relative=p.relative_to(ROOT);dest=out/'frozen'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);hashes[str(relative)]=sha(dest)
    plan=dict(protocol=PROTOCOL,authorization=auth,cap_usd=80,max_l4=10,dataset_manifest=manifest,
        source_sha256=hashes,data_sha256={s:sha(out/f'{s}.npz') for s in ('calibration','development')},
        backbones=list(BACKBONES),arms=list(ARMS),calibration_seed=3102,tuning_seed=3201,evaluation_seeds=[3301,3302,3303],
        learning_rates=[.0003,.001],epochs=150,patience=20,learning_rate_tie_rule='lowest MAE then lower learning rate',
        gpu_finish_epoch=datetime.fromisoformat(auth['gpu_finish_utc'].replace('Z','+00:00')).timestamp(),
        deadline_epoch=datetime.fromisoformat(auth['deadline_utc'].replace('Z','+00:00')).timestamp(),
        analysis=dict(primary_contrasts=45,bootstrap_resamples=10000,confidence=.95,practical_mae_improvement_pct=2,
            block_rule='development real-arm origin-MAE ACF first abs<0.1 for 24 consecutive lags; max across models, clamp 24..168; fallback 24',
            sensitivity='half and double block length plus 168-hour weekly check',seed_resampling='paired across arms and models',multiplicity='centered max-standardized bootstrap simultaneous intervals'))
    (out/'plan.json').write_text(json.dumps(plan,indent=2));print(out)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--version',default='v1');prepare(p.parse_args().version)
