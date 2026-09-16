"""Freeze the equally tuned development study after audited L4 calibration; no launch."""
import argparse
import json
import tarfile
from pathlib import Path
import numpy as np
from hypercast4d.representation_pilot import sha
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    calibration=ROOT/'results/hyperdense-etth1-v2/calibration-001/pinned-analysis.json';report=json.loads(calibration.read_text())
    assert report['complete'] and report['completed_calibration_fits']==16
    proposal_path=ROOT/'plans/hyperdense-etth1-v2/proposal.json';proposal=json.loads(proposal_path.read_text())
    parent=ROOT/'results/etth1-transfer/prepared-v3';meta=json.loads((parent/'manifest.json').read_text());assert sha(parent/'development.npz')==meta['bundle_sha256']
    pinned=ROOT/'results/hyperdense-etth1-v2/prepared-calibration-v2-pinned/manifest.json'
    for name,h in json.loads(pinned.read_text())['source_sha256'].items():assert sha(ROOT/name)==h,name
    a.output.mkdir(parents=True,exist_ok=False)
    keys=['train_x','train_y','inner_x','inner_y','development_x','development_y','development_target_start']
    arrays=np.load(parent/'development.npz',allow_pickle=False);np.savez_compressed(a.output/'development.npz',**{k:arrays[k] for k in keys})
    estimates=report['cost_proposal']['model_common_timeouts'];jobs=[]
    for b in proposal['backbones']:
        for arm in proposal['arms']:
            for setting in proposal['grid']:
                jobs.append(dict(phase='search',backbone=b,arm=arm,seed=2301,setting_id=setting['id'],settings=setting,timeout_seconds=estimates[b]['timeout_seconds']))
    for b in proposal['backbones']:
        for arm in proposal['arms']:
            for rank in range(2):jobs.append(dict(phase='reranking',backbone=b,arm=arm,seed=2302,rank=rank,timeout_seconds=estimates[b]['timeout_seconds']))
    assert len(jobs)==224
    files=sorted(list((ROOT/'src/hypercast4d').rglob('*.py'))+[ROOT/'src/hypercast4d/_constants/film_legendre_v1.pt']+[ROOT/'scripts'/n for n in ['launch_hyperdense_transfer_search.py','prepare_hyperdense_transfer_search.py','launch_representation_pilot.py']]+[ROOT/'tests/test_hyperdense_transfer_search.py'])
    plan=dict(protocol='hyperdense-etth1-v3-pinned-search',status='prepared_not_authorized',backbones=proposal['backbones'],arms=proposal['arms'],jobs=jobs,proposal=proposal,proposal_sha256=sha(proposal_path),dataset_manifest=meta,dataset_manifest_sha256=sha(parent/'manifest.json'),calibration_analysis_sha256=sha(calibration),data_sha256=sha(a.output/'development.npz'),array_keys=keys,source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},concurrent_l4=2,retries=0,final_test_supplied=False,cost=report['cost_proposal']['stages']['search_and_reranking'],selection='Lowest development MAE selects two settings per arm; mean across seeds 2301 and 2302 selects final setting. Tie: setting id. Inner MAE alone selects epochs.',confirmation_not_included=True,inference_not_included=True)
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with tarfile.open(a.output/'source.tar.gz','w:gz') as t:
        for f in files:t.add(f,arcname=str(f.relative_to(ROOT)))
    print(json.dumps(dict(fits=len(jobs),source_files=len(files),cost=plan['cost'],authorized=False),indent=2))
if __name__=='__main__':main()
