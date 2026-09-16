#!/usr/bin/env python3
"""Frozen exploratory paired-block analysis; requires all replication fits complete."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'results/modal-l4-internal'


def simultaneous_intervals(differences, block_length, *, draws=10000, seed=20260911):
    """Common circular blocks preserve both temporal and cross-contrast pairing."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Expected finite origin-by-contrast differences')
    n = len(values)
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, values.shape[1]))
    blocks = int(np.ceil(n / block_length))
    for start in range(0, draws, 100):
        count = min(100, draws-start)
        origins = rng.integers(0, n, size=(count, blocks))
        indices = ((origins[:, :, None] + np.arange(block_length)) % n).reshape(count, -1)[:, :n]
        # Frequency weights avoid allocating draws x origins x contrasts.
        weights = np.zeros((count, n))
        for i, index in enumerate(indices):
            weights[i] = np.bincount(index, minlength=n) / n
        samples[start:start+count] = weights @ values
    observed = values.mean(axis=0)
    standard = samples.std(axis=0, ddof=1)
    variable = standard > 1e-15
    maxima = np.zeros(draws)
    if variable.any():
        maxima = np.abs((samples[:, variable]-observed[variable]) / standard[variable]).max(axis=1)
    critical = float(np.quantile(maxima, .95, method='higher'))
    return dict(mean=observed.tolist(), lower=(observed-critical*standard).tolist(),
        upper=(observed+critical*standard).tolist(), bootstrap_standard_error=standard.tolist(),
        critical_value=critical, degenerate=(~variable).tolist(), block_length=block_length,
        effective_blocks=n/block_length)


def main():
    state = json.loads((ROOT/'controller/controller_state.json').read_text())
    candidates = [c for c in state['candidates'] if c['stage_name']=='remaining-replication']
    if len(candidates)!=12 or any(c['status']!='complete' for c in candidates):
        raise RuntimeError('All twelve replication batches must complete before this analysis')
    plan = json.loads((ROOT/'controller/replication-provenance/analysis-plan.json').read_text())
    metadata, differences, scores = [], [], []
    common_origins = None
    for candidate in candidates:
        directory = ROOT/'jobs'/candidate['job_id']
        manifest = json.loads((directory/'batch-manifest.json').read_text())
        if manifest['completed']!=8 or not all(x['status']=='complete' for x in manifest['trials']):
            raise RuntimeError('Incomplete replication artifacts')
        runs = pd.read_csv(directory/'runs.csv')
        frame = pd.read_csv(directory/'predictions.csv')
        if len(runs)!=8 or set(frame.seed)!={401} or set(frame.trial_status)!={'complete'}:
            raise RuntimeError('Unexpected seed, fit count or failure state')
        losses = {}
        for row in runs.to_dict('records'):
            if row['learning_rate'] != candidate['evaluation']['remaining_replication'][row['variant']]:
                raise RuntimeError('Frozen learning rate changed')
            part = frame[frame.trial_id==row['trial_id']].copy()
            if len(part)!=1485 or part.duplicated(['origin_row','lead']).any():
                raise RuntimeError('Missing or duplicate forecast targets')
            part['loss'] = abs(part.prediction-part.actual)
            loss = part.groupby('origin_row', sort=True).loss.mean()
            if common_origins is None:common_origins = loss.index.to_numpy()
            if not np.array_equal(common_origins,loss.index.to_numpy()):
                raise RuntimeError('Forecast origins do not pair')
            losses[row['variant']] = loss.to_numpy()
            if not np.isclose(loss.mean(),row['mae']):raise RuntimeError('Saved metric mismatch')
            scores.append({**row,'job_id':candidate['job_id']})
        part['loss'] = abs(part.persistence-part.actual)
        losses['persistence'] = part.groupby('origin_row',sort=True).loss.mean().to_numpy()
        pairs = [(v,u) for v,k in [('complex','lowrank2'),('quaternion','lowrank4'),('octonion','lowrank8')]
                 for u in ['real','native','persistence',k]]
        pairs += [('complex','quaternion'),('complex','octonion'),('quaternion','octonion')]
        for left,right in pairs:
            differences.append(losses[left]-losses[right])
            metadata.append(dict(backbone=candidate['backbone'],left=left,right=right,
                relative_mae_change_percent=100*(losses[left].mean()/losses[right].mean()-1)))
    matrix = np.stack(differences,axis=1)
    assert matrix.shape == (297,180)
    output = dict(stage='remaining-replication',scores=scores,contrasts=metadata,
        uncertainty={str(length):simultaneous_intervals(matrix,length,draws=plan['replicates'],seed=plan['random_seed'])
                     for length in [plan['block_length'],*plan['sensitivity_lengths']]},
        limitations=plan['limitations'],negative_difference_favours='left',
        selected_final_epoch=sum(r['best_epoch']==r['epochs_ran'] for r in scores))
    (ROOT/'controller/remaining-replication-analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(fits=len(scores),contrasts=len(metadata),selected_final_epoch=output['selected_final_epoch'],
                         limitations=output['limitations']),indent=2))


if __name__=='__main__':main()
