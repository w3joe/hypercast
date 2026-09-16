"""Reproducible validation-only HyperDense input ablation; no cloud execution."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from contextlib import redirect_stdout, redirect_stderr

import pandas as pd
import torch

from hypercast4d.architecture import presets, build_architecture, normalize_architecture_spec
from hypercast4d.playground import JobManager
from hypercast4d.playground_runner import normalize_evaluation, run_job


def candidates():
    catalogue = {item['preset_id']: item for item in presets()}
    for model in ('dlinear', 'tsmixer', 'itransformer'):
        for variant in ('original', 'lift_only', 'dense_control', 'hyper_2d', 'hyper_3d', 'hyper_4d'):
            spec = copy.deepcopy(catalogue[f'tslib-{model}'])
            spec.pop('preset_id', None)
            spec['locked'] = False
            spec['name'] = f'{model} · input {variant}'
            prefix = []
            if variant != 'original':
                prefix.append({'id': 'input_lift', 'type': 'dense', 'params': {'units': 12}})
            if variant == 'dense_control':
                prefix.append({'id': 'input_dense', 'type': 'dense', 'params': {'units': 12}})
            if variant.startswith('hyper_'):
                dimension = int(variant[6])
                prefix.append({'id': 'input_hyper', 'type': 'hyper_dense', 'params': {
                    'units': 12 // dimension, 'algebra': {2: 'complex', 3: 'tricomplex', 4: 'quaternion'}[dimension],
                }})
            spec['layers'] = prefix + spec['layers']
            yield model, variant, normalize_architecture_spec(spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Use a new output directory; existing experiments are never overwritten.')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    evaluation = normalize_evaluation({'preset': 'standard', 'cells': [
        {'window': 10, 'horizon': 1}, {'window': 20, 'horizon': 5}],
        'seeds': [7, 19, 31], 'epochs': args.epochs, 'batch_size': 128, 'device': 'cpu'})
    specs = list(candidates())
    for _, _, spec in specs:
        for cell in evaluation['cells']:
            with torch.no_grad():
                output = build_architecture(spec, cell['window'], cell['horizon'])(torch.zeros(2, cell['window'], 4))
            assert output.shape == (2, cell['horizon']), (spec['name'], output.shape)
    print('Preflight passed: 18 configurations × 2 cells. Starting 108 fits.', flush=True)
    manager = JobManager(args.output, Path.cwd())
    manifest = {'evaluation': evaluation, 'models': [], 'threads': 2, 'validation_only': True}
    results = []
    started = time.monotonic()
    for index, (model, variant, spec) in enumerate(specs, 1):
        record = manager.save_architecture(spec)
        job = manager.submit_validation(spec, evaluation, {'target': 'local'})
        destination = manager.jobs_root / job['id']
        manifest['models'].append({'model': model, 'variant': variant, 'job_id': job['id'], 'architecture': record})
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        print(f'[{index}/18] {model} {variant} started', flush=True)
        with (destination / 'training.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
            run_job(destination)
        frame = pd.read_csv(destination / 'runs.csv')
        assert len(frame) == 6 and set(frame['split']) == {'validation'}
        frame.insert(0, 'variant', variant)
        frame.insert(0, 'model', model)
        results.append(frame)
        pd.concat(results, ignore_index=True).to_csv(args.output / 'all_runs.csv', index=False)
        print(f'[{index}/18] done · MAE/persistence {frame.mae_ratio.mean():.4f} · elapsed {time.monotonic()-started:.0f}s', flush=True)
    runs = pd.concat(results, ignore_index=True)
    by_cell = runs.groupby(['model', 'variant', 'window', 'horizon']).agg(
        mae=('mae', 'mean'), mae_std=('mae', 'std'), mse=('mse', 'mean'),
        mae_ratio=('mae_ratio', 'mean'), mse_ratio=('mse_ratio', 'mean'),
        parameters=('parameters', 'mean'), train_seconds=('train_seconds', 'sum')).reset_index()
    by_cell.to_csv(args.output / 'by_cell.csv', index=False)
    summary = runs.groupby(['model', 'variant']).agg(
        mae_ratio=('mae_ratio', 'mean'), mse_ratio=('mse_ratio', 'mean'),
        parameters=('parameters', 'mean'), train_seconds=('train_seconds', 'sum')).reset_index()
    summary.to_csv(args.output / 'comparison.csv', index=False)
    print(summary.to_string(index=False), flush=True)
    print(f'Complete: {args.output.resolve()} · {time.monotonic()-started:.0f}s', flush=True)


if __name__ == '__main__':
    main()
