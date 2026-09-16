"""Freeze a development-only pilot bundle. Never submits jobs or edits old ledgers."""
import argparse
import json
from pathlib import Path
import tarfile
import numpy as np
from hypercast4d.data import load_paper_data
from hypercast4d.internal_controls import nested_windows
from hypercast4d.representation_pilot import sha, SEEDS

ROOT = Path(__file__).resolve().parents[1]


def prepare(output, concurrency=1):
    if concurrency not in (1,2):raise ValueError('Only one or two L4s supported')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT/'data/raw/paper_data.xlsx'
    frame = load_paper_data(source, 'Copper')
    prepared, inner, audit = nested_windows(frame,32,5,.7,.15)
    np.savez_compressed(output/'development.npz', train_x=prepared.train.x, train_y=prepared.train.y,
        train_target_start=prepared.train.target_start, inner_x=inner.x, inner_y=inner.y,
        inner_target_start=inner.target_start)
    names = list((ROOT/'src/hypercast4d').rglob('*.py'))
    names += [ROOT/'scripts'/name for name in ['prepare_representation_pilot.py',
        'run_representation_pilot.py','launch_representation_pilot.py',
        'profile_hyperdense_l4_pilot.py','profile_hyperdense_followup.py']]
    names += [ROOT/'tests/test_representation_pilot.py',ROOT/'tests/test_hyperdense_followup.py']
    previous=json.loads((ROOT/'results/hyperdense-followup-local/inference.json').read_text())
    shapes=sorted({(r['width_in'],r['width_out'],r['bias']) for r in previous['layers']})
    manifest = dict(protocol='native-representation-pilot-v1', status='prepared_not_authorized',
        source_data_sha256=sha(source), data_sha256=sha(output/'development.npz'),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(names)},
        **audit, feature_order=list(frame.columns), target='Copper', window=32, horizon=5,
        date_bounds={name:[str(frame.index[a].date()),str(frame.index[b-1].date())]
                     for name,(a,b) in [('train',audit['train_rows']),('inner',audit['inner_rows'])]},
        selection='inner MAE only; outer and final test not bundled or scored',
        loss='mae', optimizer=dict(name='Adam',lr=.001,betas=[.9,.999],eps=1e-7,weight_decay=0),
        epochs=100, patience=20, relative_delta=0, batch_size=32, shuffle=True,
        jobs=[dict(backbone=b,seed=s,modes=['levels_direct','relative_residual'],
                   timeout_seconds=300 if b=='micn' else 570) for b in ('micn','film') for s in SEEDS],
        inference_shapes=shapes,
        max_concurrent_l4=concurrency, retries=0, requested_cap_usd=6, protected_reserve_usd=1,
        limitations='Reused development dates; package changes representation, head and initialization together. FiLM native is complex.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(output/'source.tar.gz','w:gz') as archive:
        for p in sorted(names):
            archive.add(p,arcname=str(p.relative_to(ROOT)))
    print(json.dumps({k:manifest[k] for k in ['status','date_bounds','train_samples','inner_samples','jobs']},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--concurrency',type=int,choices=[1,2],default=1)
    args=parser.parse_args()
    prepare(args.output,args.concurrency)
