#!/usr/bin/env python3
"""Reproducible paired analysis of the frozen, complete robustness matrix."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BACKBONES = ['tslib-dlinear', 'tslib-tsmixer', 'tslib-itransformer']
VARIANTS = ['original', 'lift_only', 'real_dense', 'complex', 'quaternion', 'octonion',
            'real_rank8', 'real_rank4', 'real_rank2']
HYPER = ['complex', 'quaternion', 'octonion']
PAIRS = [(v, 'real_dense') for v in HYPER] + list(zip(HYPER, ['real_rank8', 'real_rank4', 'real_rank2'])) + [
    ('complex', 'quaternion'), ('complex', 'octonion'), ('quaternion', 'octonion')] + [(v, 'persistence') for v in HYPER]


def block_means(values, length, rng, resamples=10000):
    """Joint circular block resampling; columns share the same sampled origins."""
    n = len(values)
    result = np.empty((resamples, values.shape[1]))
    blocks = (n + length - 1) // length
    for start in range(0, resamples, 100):
        count = min(100, resamples - start)
        starts = rng.integers(n, size=(count, blocks))
        indices = ((starts[..., None] + np.arange(length)) % n).reshape(count, -1)[:, :n]
        result[start:start + count] = values[indices].mean(axis=1)
    return result


def relative_effect(candidate, control):
    return 100 * (1 - np.asarray(candidate) / np.asarray(control))


def analyze(root, output):
    state = json.loads((root / 'controller/controller_state.json').read_text())
    if state.get('schedule_status') != 'controlled-robustness_complete':
        raise ValueError('Analyze only the complete, reconciled robustness matrix')
    for path, expected in state['frozen_inputs'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen input changed: {path}')
    candidates = [c for c in state['candidates'] if c.get('stage_name') == 'controlled-robustness']
    assert len(candidates) == 27 and all(c['status'] == 'complete' for c in candidates)
    expected = {(seed, fold) for seed in [101, 211] for fold in [1, 2, 3]}
    losses = {}; references = {}; actuals = {}; summaries = []; per_lead = []; sources = {}; paired = defaultdict(set)
    for c in candidates:
        directory = root / 'jobs' / c['job_id']
        rows = list(csv.DictReader((directory / 'runs.csv').open()))
        curves = list(csv.DictReader((directory / 'learning_curves.csv').open()))
        initialization = json.loads((directory / 'initialization.json').read_text())
        status = json.loads((directory / 'status.json').read_text())
        assert status['state'] == 'complete' and status['actual_gpu'] == 'NVIDIA L4'
        assert len(rows) == 6 and {(int(x['seed']), int(x['fold'])) for x in rows} == expected
        assert all(int(x['epochs_ran']) == 150 for x in rows) and len(curves) == 900 and len(initialization) == 6
        assert all(np.isfinite(float(x[k])) for x in curves for k in ['train_loss', 'validation_loss'])
        for seed, fold in expected:
            part = [x for x in curves if (int(x['seed']), int(x['fold'])) == (seed, fold)]
            assert len(part) == 150 and {int(x['epoch']) for x in part} == set(range(1, 151))
        for init in initialization:
            assert init['protocol'] == 'matched-v1'
            if c['variant'] != 'original':
                for key in ['input_lift', 'core', 'forecast_head']:
                    paired[c['backbone'], init['seed'], init['fold'], key].add(init['module_sha256'][key])
        prediction_path = directory / 'predictions.csv'
        sources[c['job_id']] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        groups = defaultdict(list)
        for row in csv.DictReader(prediction_path.open()):
            assert row['split'] == 'validation'
            groups[int(row['seed']), int(row['fold'])].append(row)
        assert set(groups) == expected
        for seed, fold in sorted(expected):
            predictions = sorted(groups[seed, fold], key=lambda x: (int(x['origin_row']), int(x['lead'])))
            keys = [(int(x['origin_row']), int(x['lead']), x['target_date']) for x in predictions]
            if fold in references: assert references[fold] == keys
            references[fold] = keys
            values = np.array([[float(x[k]) for k in ['actual', 'prediction', 'persistence']] for x in predictions])
            assert np.isfinite(values).all()
            if fold in actuals: np.testing.assert_allclose(actuals[fold], values[:, [0, 2]], rtol=0, atol=1e-6)
            actuals[fold] = values[:, [0, 2]]
            error = np.abs(values[:, 1] - values[:, 0]); pe = np.abs(values[:, 2] - values[:, 0])
            row = next(x for x in rows if int(x['seed']) == seed and int(x['fold']) == fold)
            np.testing.assert_allclose(error.mean(), float(row['mae']), rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(error.mean() / pe.mean(), float(row['mae_ratio']), rtol=1e-5, atol=1e-6)
            assert len(keys) % 5 == 0 and all([k[1] for k in keys[i:i+5]] == [1,2,3,4,5] for i in range(0, len(keys), 5))
            losses[c['backbone'], c['variant'], seed, fold] = error.reshape(-1, 5).mean(axis=1)
            losses['persistence', seed, fold] = pe.reshape(-1, 5).mean(axis=1)
            for lead in range(1, 6):
                mask = np.array([k[1] == lead for k in keys])
                per_lead.append(dict(backbone=c['backbone'], variant=c['variant'], seed=seed, fold=fold,
                    lead=lead, mae=float(error[mask].mean()), mse=float(np.mean((values[mask, 1]-values[mask, 0])**2))))
        metrics = ['mae', 'mse', 'mae_ratio', 'parameters', 'train_seconds', 'peak_gpu_allocated_mb', 'inference_ms_per_batch']
        assert all(np.isfinite(float(x[k])) for x in rows for k in metrics)
        summary = {'backbone': c['backbone'], 'variant': c['variant'], 'job_id': c['job_id'],
            'learning_rate': c['evaluation']['learning_rate'], **{k: float(np.mean([float(x[k]) for x in rows])) for k in metrics},
            'fold_ratios': {str(f): float(np.mean([float(x['mae_ratio']) for x in rows if int(x['fold']) == f])) for f in [1,2,3]},
            'seed_ratios': {str(seed): float(np.mean([float(x['mae_ratio']) for x in rows if int(x['seed']) == seed])) for seed in [101,211]},
            'final_over_best_validation_loss': float(np.mean([float(part[-1]['validation_loss']) / min(float(x['validation_loss']) for x in part)
                for seed, fold in sorted(expected) for part in [[x for x in curves if int(x['seed']) == seed and int(x['fold']) == fold]]]))}
        summaries.append(summary)
    assert all(len(v) == 1 for v in paired.values())
    columns = [(b, v) for b in BACKBONES for v in VARIANTS] + [('persistence', 'persistence')]
    arrays = {}
    for fold in [1,2,3]:
        arrays[fold] = np.column_stack([np.mean([losses[b, v, seed, fold] for seed in [101,211]], axis=0)
            if b != 'persistence' else np.mean([losses['persistence', seed, fold] for seed in [101,211]], axis=0) for b, v in columns])
    # Match the primary equal-fold MAE/persistence outcome, not a pooled-date MAE.
    scores = np.mean([a.mean(axis=0) / a[:, -1].mean() for a in arrays.values()], axis=0)
    intervals = []
    for length in [60,30,120]:
        rng = np.random.default_rng(20260911)
        means = [block_means(a, length, rng) for a in arrays.values()]
        samples = np.mean([m / m[:, -1:] for m in means], axis=0)
        for b in BACKBONES:
            for candidate, control in PAIRS:
                ai = columns.index((b, candidate)); bi = 27 if control == 'persistence' else columns.index((b, control))
                effects = relative_effect(samples[:, ai], samples[:, bi])
                q = np.quantile(effects, [.025, .975, .05/72, 1-.05/72])
                intervals.append(dict(backbone=b, candidate=candidate, control=control, block_length=length,
                    improvement_percent=float(relative_effect(scores[ai], scores[bi])), ci95_low=float(q[0]), ci95_high=float(q[1]),
                    family95_low=float(q[2]), family95_high=float(q[3])))
    output.mkdir(parents=True, exist_ok=True)
    result = dict(complete=True, configurations=27, fits=162, failures=0, summaries=summaries, contrasts=intervals,
        fold_origins={str(f): len(a) for f,a in arrays.items()},
        effective_blocks={str(length): {str(f): len(a)/length for f,a in arrays.items()} for length in [30,60,120]},
        estimator='equal-fold mean MAE/persistence, seeds averaged; paired relative differences of aggregate scores',
        resamples=10000, bootstrap_seed=20260911, source_prediction_sha256=sources,
        conservative_spend_usd=max(state['budget']['estimated_spend_usd'],state['budget']['confirmed_spend_usd']))
    (output / 'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    for name, rows in [('contrasts', intervals), ('per_lead', per_lead)]:
        with (output / f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lookup={(r['backbone'],r['variant']):r for r in summaries}
    text=['# Complete robustness analysis', '', '162 fits; 27 configurations; seeds 101/211; three chronological folds; window 20/horizon 5; 150 fixed epochs. No failures. All predictions reconcile with saved MAE metrics; paired initialization and frozen inputs pass checks. No test data read.', '',
          '## Primary outcome', '', 'Equal-fold and equal-seed mean MAE/persistence; lower is better, 1 matches persistence.', '',
          '| Backbone | Original | Lift only | Real dense | 2D | 4D | 8D | Rank-8 | Rank-4 | Rank-2 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for b in BACKBONES:
        text.append('| '+b+' | '+' | '.join(f"{lookup[b,v]['mae_ratio']:.3f}" for v in VARIANTS)+' |')
    text += ['', '## Paired effects against real dense', '', 'Positive values favor HyperDense. Intervals use the prespecified block length 60 and Bonferroni adjustment across 36 contrasts.', '',
             '| Backbone | Dimension | Improvement % | Family-wise 95% interval |', '|---|---|---:|---|']
    for row in intervals:
        if row['block_length']==60 and row['control']=='real_dense':
            text.append(f"| {row['backbone']} | {row['candidate']} | {row['improvement_percent']:.1f} | [{row['family95_low']:.1f}, {row['family95_high']:.1f}] |")
    text += ['', '## Interpretation limits', '', 'The validation dates overlap earlier development exposure. These results assess seed and period stability conditional on the selected training recipe, not an independent holdout or cross-dataset superiority. The block length reaches the development dependence-search cap, and each fold has only about three effective blocks at length 60. Sensitivity lengths 30/120 and small seed count limit uncertainty claims. Extreme adjusted quantiles have limited Monte Carlo precision. Do not label a nonsignificant effect equivalent; use the 2% practical margin and report sensitivity-dependent results as inconclusive.', '',
             'All fold and seed scores, runtime/memory/parameter summaries, convergence diagnostics and prediction hashes are in analysis.json. All 36 contrasts at block lengths 60/30/120 are in contrasts.csv; per-lead MAE/MSE are in per_lead.csv. A lower parameter count alone does not establish speed or forecasting benefit. Final-epoch versus best-validation-loss ratios are descriptive diagnostics, not retrospective checkpoint selection.']
    (output/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:result[k] for k in ['fits','failures','fold_origins','conservative_spend_usd']},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,default=Path('results/modal-l4-runner'))
    parser.add_argument('--out',type=Path,default=Path('results/modal-l4-runner/controller/robustness-analysis'))
    args=parser.parse_args();analyze(args.results_root,args.out)
