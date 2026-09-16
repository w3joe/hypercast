import io
import json
import tarfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hypercast4d.architecture import normalize_architecture_spec, presets
from hypercast4d import playground_runner as runner
from hypercast4d import remaining_batch as batch
from hypercast4d.modal_runner import _execute_payload
from hypercast4d.remaining_controls import VARIANTS


def request():
    architecture = normalize_architecture_spec(next(p for p in presets() if p['preset_id']=='tslib-patchtst'))
    return dict(architecture=architecture, evaluation=dict(preset='standard', seeds=[701], epochs=2,
        cells=[dict(window=32,horizon=5)], initialization='remaining-internal-v1',
        replacement=dict(backbone='patchtst',variant='native'), remaining_tuning=True,
        nested_stopping=True, restore_best_weights=True, early_stopping_patience=20,
        early_stopping_relative_delta=.001, device='cpu', data_path='data/synthetic.xlsx'),
        execution=dict(target='modal',gpu='L4',timeout_seconds=300))


@pytest.fixture
def synthetic(monkeypatch):
    frame=pd.DataFrame(np.random.default_rng(23).normal(size=(200,4)),columns=['Copper','b','c','d'])
    monkeypatch.setattr(runner,'load_paper_data',lambda *args:frame)
    return frame


def test_remote_batch_returns_all_labelled_fits_audits_and_original_artifacts(synthetic,tmp_path):
    req=request()
    result=_execute_payload(json.dumps(req),b'synthetic data',None,require_cuda=False)
    assert result['ok'],result['error']
    assert result['status']['completed']==result['status']['total']==16
    rows=pd.read_csv(io.BytesIO(result['files']['runs.csv']))
    assert len(rows)==16 and set(zip(rows.variant,rows.learning_rate))=={(v,r) for v in VARIANTS for r in batch.RATES}
    assert set(rows.trial_status)=={'complete'} and set(rows.stopping_split)=={'inner'}
    initialization=json.loads(result['files']['initialization.json'])
    assert len(initialization)==16 and len({r['untouched_sha256'] for r in initialization})==1
    assert len(json.loads(result['files']['split_audit.json']))==16
    curves=pd.read_csv(io.BytesIO(result['files']['learning_curves.csv']))
    assert len(curves.groupby('trial_id'))==16
    with tarfile.open(fileobj=io.BytesIO(result['files']['batch-artifacts.tar.gz'])) as archive:
        archived=archive.extractfile('trials/octonion-lr0.001/predictions.csv').read()
    # A late batch fit must match an isolated fit despite previous RNG consumption.
    isolated=next(t for t in batch.trial_grid(runner.normalize_evaluation(req['evaluation']))
                  if t['trial_id']=='octonion-lr0.001')
    runner.run_validation(tmp_path,{**req,'evaluation':isolated['evaluation']})
    pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(archived)),pd.read_csv(tmp_path/'predictions.csv'))


def test_remote_failure_returns_prior_completed_and_partial_artifacts(synthetic,monkeypatch):
    original=runner.run_validation
    def failing(directory,req):
        if not req['evaluation'].get('remaining_tuning') and req['evaluation']['replacement']['variant']=='real':
            original(directory,req)
            raise RuntimeError('injected failure after fit artifacts')
        return original(directory,req)
    monkeypatch.setattr(runner,'run_validation',failing)
    result=_execute_payload(json.dumps(request()),b'synthetic',None,require_cuda=False)
    assert not result['ok'] and result['status']['completed']==2 and result['status']['total']==16
    manifest=json.loads(result['files']['batch-manifest.json'])
    assert manifest['trials'][2]['status']=='failed' and manifest['trials'][3]['status']=='pending'
    rows=pd.read_csv(io.BytesIO(result['files']['runs.csv']))
    assert list(rows.trial_status)==['complete','complete','failed']
    with tarfile.open(fileobj=io.BytesIO(result['files']['batch-artifacts.tar.gz'])) as archive:
        assert 'trials/real-lr0.0003/learning_curves.csv' in archive.getnames()


def test_soft_deadline_preserves_manifest_without_starting_a_fit(synthetic,monkeypatch):
    ticks=iter([0.,1000.])
    monkeypatch.setattr(batch,'time',SimpleNamespace(monotonic=lambda:next(ticks)))
    result=_execute_payload(json.dumps(request()),b'synthetic',None,require_cuda=False)
    assert not result['ok'] and 'soft deadline' in result['error']
    assert result['status']['completed']==0
    assert 'batch-artifacts.tar.gz' in result['files']


def test_batch_requires_single_paired_cell_seed_fold_and_native_template():
    assert 'remaining_tuning' not in runner.normalize_evaluation({})
    for change in [dict(seeds=[1,2]),dict(preset='robust'),dict(remaining_preflight=True),
                   dict(replacement=dict(backbone='patchtst',variant='complex'))]:
        with pytest.raises(ValueError,match='remaining_tuning'):
            runner.normalize_evaluation({**request()['evaluation'],**change})


@pytest.mark.parametrize('seeds', [[401], [401, 503]])
def test_replication_uses_frozen_per_arm_rates_and_pairs_each_seed(synthetic, seeds):
    req = request()
    req['evaluation'].pop('remaining_tuning')
    rates = {variant: batch.RATES[index % 2] for index, variant in enumerate(VARIANTS)}
    req['evaluation'].update(remaining_replication=rates, seeds=seeds)
    result = _execute_payload(json.dumps(req), b'synthetic', None, require_cuda=False)
    assert result['ok'], result['error']
    assert result['status']['completed'] == result['status']['total'] == 8 * len(seeds)
    rows = pd.read_csv(io.BytesIO(result['files']['runs.csv']))
    assert set(zip(rows.variant, rows.seed)) == {(v, s) for v in VARIANTS for s in seeds}
    assert all(r.learning_rate == rates[r.variant] for r in rows.itertuples())
    reports = json.loads(result['files']['initialization.json'])
    by_seed = {seed: {r['untouched_sha256'] for r in reports if r['seed'] == seed} for seed in seeds}
    assert all(len(values) == 1 for values in by_seed.values())
    assert len(set.union(*by_seed.values())) == len(seeds)
    manifest = json.loads(result['files']['batch-manifest.json'])
    assert manifest['protocol'] == 'remaining-replication-batch-v1'
    assert all(not t['evaluation'].get('remaining_replication') and len(t['evaluation']['seeds']) == 1 for t in manifest['trials'])


def test_replication_rejects_missing_rates_or_simultaneous_tuning():
    ev = request()['evaluation']
    with pytest.raises(ValueError, match='remaining_replication'):
        runner.normalize_evaluation({**ev, 'remaining_replication': {v: .001 for v in VARIANTS}})
    ev.pop('remaining_tuning')
    for rates in [{}, {v: .1 for v in VARIANTS}, {v: .001 for v in VARIANTS[:-1]}]:
        with pytest.raises(ValueError, match='remaining_replication'):
            runner.normalize_evaluation({**ev, 'remaining_replication': rates})
