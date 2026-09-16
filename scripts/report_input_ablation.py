"""Audit and summarize an input ablation; optionally publish completed runs locally."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--publish-root', type=Path)
    args = parser.parse_args()
    root = args.output
    manifest = json.loads((root / 'manifest.json').read_text())
    runs = pd.read_csv(root / 'all_runs.csv')
    assert len(manifest['models']) == 18 and len(runs) == 108
    assert set(runs['split']) == {'validation'}
    assert set(runs['epochs_ran']) == {manifest['evaluation']['epochs']}
    assert not runs.duplicated(['model', 'variant', 'window', 'horizon', 'seed', 'fold']).any()
    assert runs.groupby(['model', 'variant']).size().eq(6).all()
    for directory in (root / 'jobs').iterdir():
        status = json.loads((directory / 'status.json').read_text())
        assert status['state'] == 'complete' and status['completed'] == 6
    assert runs[['mae', 'mse', 'mae_ratio', 'mse_ratio']].notna().all().all()
    variants = ['original', 'lift_only', 'dense_control', 'hyper_2d', 'hyper_3d', 'hyper_4d']
    models = ['dlinear', 'tsmixer', 'itransformer']
    names = dict(original='Original', lift_only='Dense 4→12 only', dense_control='Dense 4→12 + Dense 12→12',
                 hyper_2d='Dense 4→12 + HyperDense 2D', hyper_3d='Dense 4→12 + HyperDense 3D', hyper_4d='Dense 4→12 + HyperDense 4D')
    summary = pd.read_csv(root / 'comparison.csv')
    ratios = summary.pivot(index='variant', columns='model', values='mae_ratio')
    text = [f"# HyperDense input ablation — {manifest['evaluation']['epochs']} epochs", '',
            'Validation only; no held-out test runs. 18 configurations × 2 window/horizon settings × 3 seeds = 108 fitted runs.', '',
            '## Main comparison', '',
            'Mean MAE divided by persistence MAE, averaged equally across seeds and settings. Lower is better; **1.0 = repeating the last observed price**. These are ratios, not percentage errors.', '',
            '| Input configuration | DLinear | TSMixer | iTransformer |',
            '| --- | ---: | ---: | ---: |']
    for variant in variants:
        text.append('| ' + names[variant] + ' | ' + ' | '.join(f'{ratios.loc[variant, model]:.4f}' for model in models) + ' |')
    text += ['', '## Protocol and interpretation', '',
             '- Dataset: local paper_data.xlsx, 2,008 observations, four input features, Copper target.',
             '- Chronological train/validation/test allocation: 70% / 15% / 15%. Normalization fitted to training data only; the last 15% was not evaluated.',
             f"- Epochs: {manifest['evaluation']['epochs']}; batch size 128; seeds 7, 19, 31; window/horizon 10/1 and 20/5; Adam learning rate 0.001. Fixed final epoch, no early stopping or best-weight restoration.",
             '- CPU execution, two Torch threads. All candidates used identical evaluation settings.',
             '- 2D = complex (6 output units); 3D = cyclic tricomplex (4 units); 4D = quaternion (3 units). Every HyperDense input/output has 12 real-valued features.',
             '- A common learned 4→12 Dense lift is necessary for a matched-width 3D comparison because four is not divisible by three. No activation was inserted between the lift and HyperDense.',
             '- The projection-only and two-Dense controls distinguish input widening and added linear parameterization from the hypercomplex structure.',
             '- Widths are matched, parameter counts are not: the 12→12 stage has 156 parameters for Dense, 84 for 2D, 60 for 3D, and 48 for 4D.',
             '- Changing dimension also changes the algebra and parameter sharing. These runs do not isolate dimension alone or establish statistical significance (only three seeds and one chronological split).',
             '- These are existing playground TSLib-core adaptations with the playground forecast head, not full published-model benchmark reproductions.',
             '- The unchanged-architecture and hybrid initialization streams differ even with matched seeds; initial weights are not identical.', '',
             '## Per-setting mean MAE ratios', '',
             '| Model | Configuration | 10/1 | 20/5 |', '| --- | --- | ---: | ---: |']
    cells = pd.read_csv(root / 'by_cell.csv')
    for model in models:
        for variant in variants:
            rows = cells[(cells.model == model) & (cells.variant == variant)].sort_values('window')
            text.append(f'| {model} | {names[variant]} | ' + ' | '.join(f'{v:.4f}' for v in rows.mae_ratio) + ' |')
    text += ['', '## Artifacts', '',
             '- [Individual run metrics](all_runs.csv)', '- [Per-setting means and standard deviations](by_cell.csv)',
             '- [Aggregate comparison and parameter counts](comparison.csv)', '- [Exact requests and architecture index](manifest.json)',
             '- `jobs/` contains requests, status, logs, per-lead metrics, forecast predictions and diagnostics.', '']
    (root / 'REPORT.md').write_text('\n'.join(text))
    (root / 'audit.json').write_text(json.dumps({'jobs': 18, 'fits': 108, 'all_complete': True,
        'validation_only': True, 'unique_trials': True, 'torch_version': torch.__version__,
        'dataset_sha256': hashlib.sha256(Path(manifest['evaluation']['data_path']).read_bytes()).hexdigest()}, indent=2))
    if args.publish_root:
        # Only new IDs are copied; existing playground jobs are never changed.
        for folder in ('jobs', 'architectures'):
            for source in (root / folder).iterdir():
                assert not (args.publish_root / folder / source.name).exists(), f'Already exists: {source.name}'
        for folder in ('jobs', 'architectures'):
            destination = args.publish_root / folder
            destination.mkdir(parents=True, exist_ok=True)
            for source in (root / folder).iterdir():
                if source.is_dir():
                    shutil.copytree(source, destination / source.name)
                else:
                    shutil.copy2(source, destination / source.name)
        print('Published 18 completed runs and architecture records to', args.publish_root)
    print('\n'.join(text[:15]))


if __name__ == '__main__':
    main()
