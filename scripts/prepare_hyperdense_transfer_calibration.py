"""Freeze a train-only calibration bundle and source snapshot; no launch."""
import argparse
import json
import tarfile
from pathlib import Path
import numpy as np
from hypercast4d.representation_pilot import sha
from hypercast4d.hyperdense_transfer import ARMS

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    original=ROOT/'results/etth1-transfer/prepared-v3';proposal=ROOT/'plans/hyperdense-etth1-v2/proposal.json'
    meta=json.loads((original/'manifest.json').read_text())
    assert sha(original/'development.npz')==meta['bundle_sha256']
    for manifest,key in [('results/modal-l4-internal/controller/controller_state.json','frozen_inputs'),('results/hyperdense-representation-pilot/prepared-v2-parallel/manifest.json','source_sha256'),('results/etth1-transfer/prepared-v3/manifest.json','source_code_sha256'),('results/etth1-transfer/development-prepared-v1/plan.json','source_sha256')]:
        for name,h in json.loads((ROOT/manifest).read_text())[key].items():assert sha(ROOT/name)==h,name
    a.output.mkdir(parents=True,exist_ok=False)
    arrays=np.load(original/'development.npz',allow_pickle=False)
    np.savez_compressed(a.output/'calibration.npz',**{k:arrays[k] for k in ['train_x','train_y','inner_x']})
    files=list((ROOT/'src/hypercast4d').rglob('*.py'))+[ROOT/'scripts'/n for n in ['launch_hyperdense_transfer_calibration.py','prepare_hyperdense_transfer_calibration.py','launch_representation_pilot.py']]+[ROOT/'tests/test_hyperdense_transfer.py']
    jobs=[dict(kind='gate',timeout_seconds=600)]+[dict(kind='calibration',backbone=b,arm=arm,seed=2300,gain=1.,learning_rate=.001,schedule='exponential',epochs=3,timeout_seconds=600) for b in ['micn','film'] for arm in ARMS]
    manifest=dict(protocol='hyperdense-etth1-v2-calibration',jobs=jobs,authorized_cap_usd=14,concurrent_l4=2,retries=0,
        dataset_manifest=meta,parent_dataset_manifest_sha256=sha(original/'manifest.json'),proposal_sha256=sha(proposal),
        data_sha256=sha(a.output/'calibration.npz'),array_keys=['train_x','train_y','inner_x'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},validation_labels_supplied=False,development_labels_supplied=False,test_labels_supplied=False,
        calibration_selection='none',settings_note='Representative high rate/full amplitude, exponential scheduler exercised. All twelve search settings have identical operator shapes.',
        pending_after_calibration=['full search launcher and admission','final test scorer and inference benchmark','fresh spending cap for full search'])
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(a.output/'source.tar.gz','w:gz') as archive:
        for p in files:archive.add(p,arcname=str(p.relative_to(ROOT)))
    print(json.dumps(dict(jobs=len(jobs),data_keys=manifest['array_keys'],source_files=len(files),frozen_prior_sources_unchanged=True),indent=2))

if __name__=='__main__':main()
