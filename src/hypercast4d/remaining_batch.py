"""Sequential, fully labelled eight-arm/two-rate tuning within one GPU job."""
from copy import deepcopy
import json
import tarfile
import time

import pandas as pd

from .remaining_controls import VARIANTS

RATES = (.0003, .001)
CSV_FILES = ('runs.csv', 'per_lead.csv', 'predictions.csv', 'learning_curves.csv', 'summary.csv')
JSON_FILES = ('initialization.json', 'split_audit.json', 'summary.json')


def trial_grid(evaluation):
    """Order is frozen, and every variant gets the same two rates."""
    rows = []
    if evaluation.get('remaining_replication'):
        for variant in VARIANTS:
            for seed in evaluation['seeds']:
                child = deepcopy(evaluation)
                rate = child.pop('remaining_replication')[variant]
                child.pop('remaining_tuning', None)
                child.update(seeds=[seed], learning_rate=rate, remaining_preflight=False)
                child['replacement']['variant'] = variant
                rows.append(dict(trial_id=f'{variant}-seed{seed}-lr{rate}',
                    backbone=child['replacement']['backbone'], variant=variant,
                    learning_rate=rate, seed=seed, evaluation=child))
        return rows
    for variant in VARIANTS:
        for rate in RATES:
            child = deepcopy(evaluation)
            child.pop('remaining_tuning')
            child['remaining_preflight'] = False
            child['replacement']['variant'] = variant
            child['learning_rate'] = rate
            rows.append(dict(trial_id=f'{variant}-lr{rate}', backbone=child['replacement']['backbone'],
                variant=variant, learning_rate=rate, evaluation=child))
    return rows


def collect_artifacts(directory, trials):
    """Preserve both root tables and each complete/partial child directory."""
    from .playground_runner import _atomic_json
    for filename in CSV_FILES:
        frames = []
        for trial in trials:
            path = directory / 'trials' / trial['trial_id'] / filename
            if path.exists():
                frame = pd.read_csv(path)
                for key in ('trial_id', 'backbone', 'variant', 'learning_rate'):
                    frame[key] = trial[key]
                if 'seed' in trial:
                    frame['seed'] = trial['seed']
                frame['trial_status'] = json.loads((path.parent / 'status.json').read_text()).get('state')
                frames.append(frame)
        if frames:
            target = directory / filename
            temporary = target.with_suffix('.tmp')
            pd.concat(frames, ignore_index=True).to_csv(temporary, index=False)
            temporary.replace(target)
    for filename in JSON_FILES:
        rows = []
        for trial in trials:
            path = directory / 'trials' / trial['trial_id'] / filename
            if path.exists():
                labels = {key: trial[key] for key in ('trial_id', 'backbone', 'variant', 'learning_rate')}
                if 'seed' in trial:
                    labels['seed'] = trial['seed']
                rows.extend({**row, **labels} for row in json.loads(path.read_text()))
        if rows:
            _atomic_json(directory / filename, rows)
    temporary = directory / 'batch-artifacts.tmp'
    weights = {}
    for trial in trials:
        path = directory / 'trials' / trial['trial_id'] / 'weights.json'
        if path.is_file():
            weights.update({f"{trial['trial_id']}/{key}": {**value, 'trial_id': trial['trial_id']}
                            for key, value in json.loads(path.read_text()).items()})
    if weights:
        _atomic_json(directory / 'weights.json', weights)
    with tarfile.open(temporary, 'w:gz') as archive:
        archive.add(directory / 'trials', arcname='trials')
    temporary.replace(directory / 'batch-artifacts.tar.gz')


def run_batch(directory, request):
    from .playground_runner import normalize_evaluation, run_validation, _atomic_json, _status
    evaluation = normalize_evaluation(request['evaluation'])
    trials = trial_grid(evaluation)
    total = len(trials)
    timeout = float(request.get('execution', {}).get('timeout_seconds', 3600))
    if timeout <= 120:
        raise ValueError('Batched tuning requires more than 120 seconds for safe artifact return')
    deadline = time.monotonic() + timeout - 90
    completed = 0
    manifest = dict(protocol='remaining-replication-batch-v1' if evaluation.get('remaining_replication') else 'remaining-tuning-batch-v1', expected=total, completed=0,
        trials=[{**t, 'status': 'pending'} for t in trials],
        hard_timeout_seconds=timeout, soft_margin_seconds=90)
    _atomic_json(directory / 'batch-manifest.json', manifest)
    _status(directory, state='running', completed=0, total=total, phase='validation')
    for index, trial in enumerate(trials):
        child_dir = directory / 'trials' / trial['trial_id']
        child_dir.mkdir(parents=True, exist_ok=False)
        child_request = {**request, 'evaluation': trial['evaluation']}
        _atomic_json(child_dir / 'request.json', child_request)
        child_request['_deadline_monotonic'] = deadline
        print(f'Batch fit {index + 1}/{total}: {trial["trial_id"]}', flush=True)
        _status(directory, current={k: trial[k] for k in ('trial_id', 'variant', 'learning_rate')})
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError('Batch soft deadline reached before next fit')
            run_validation(child_dir, child_request)
            status = json.loads((child_dir / 'status.json').read_text())
            if status['state'] != 'complete' or status['completed'] != 1:
                raise RuntimeError('Child fit is not complete')
            completed += 1
            manifest['trials'][index]['status'] = 'complete'
        except BaseException as error:
            manifest['trials'][index].update(status='failed', error=str(error))
            _status(child_dir, state='failed', error=str(error))
            _status(directory, state='failed', completed=completed, total=total, error=str(error))
            raise
        finally:
            manifest['completed'] = completed
            _atomic_json(directory / 'batch-manifest.json', manifest)
            collect_artifacts(directory, trials)
            _status(directory, completed=completed, total=total)
    _status(directory, state='complete', completed=total, total=total, current=None)
