import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from hypercast4d.comparison_archives import list_archives, read_archive_forecasts
from hypercast4d.playground import create_app


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def archive(tmp_path):
    source, root = tmp_path / 'study', tmp_path / 'workspace'
    dataset = tmp_path / 'data.csv'
    dataset.write_text('date,OT\n' + ''.join(f'2026-01-01 {i:02d}:00:00,{i}\n' for i in range(20)))
    partitions = {name: {'rows': rows, 'origins': count} for name, rows, count in (
        ('train', [0, 4], 2), ('inner', [4, 7], 2), ('development', [7, 10], 2), ('test', [10, 15], 4))}
    plan = dict(protocol='tslib15-test', backbones=['dlinear', 'tsmixer'], arms=['native'], evaluation_seeds=[1, 2], patience=20,
                dataset_manifest=dict(context=2, horizon=2, target='OT', task='Synthetic fixture',
                                      source_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(), partitions=partitions))
    write_json(source / 'prepared/evaluation-plan.json', plan)
    results, attempts = [], []
    (source / 'predictions').mkdir()
    for backbone in plan['backbones']:
        for seed in plan['evaluation_seeds']:
            index = len(results)
            job = dict(phase='test', backbone=backbone, arm='native', seed=seed, epochs=10, learning_rate=.001)
            actual = np.arange(8).reshape(4, 2).astype(float) + 10
            prediction, persistence = actual + seed * .1, actual + .5
            error = prediction - actual
            np.savez(source / f'predictions/{index:03d}.npz', actual=actual, prediction=prediction, persistence=persistence, target_start=np.arange(10, 14))
            results.append(dict(job=job, test_scored=True, checkpoint_replay_passed=True,
                                mae=float(np.abs(error).mean()), mse=float(np.square(error).mean()), rmse=float(np.sqrt(np.square(error).mean())),
                                persistence_mae=.5, bias=float(error.mean()), control={'parameters': 100},
                                training_elapsed_seconds=12, epochs_ran=5, ceiling_reached=False, inference={'32': {'median_ms': .5}}))
            attempts.append(dict(index=index, job=job, status='complete', ended_epoch=1_000_000))
    write_json(source / 'ledger.json', dict(status='complete', prepared_dir='prepared', attempts=attempts))
    write_json(source / 'test-results.json', results)
    write_json(source / 'artifact-audit.json', dict(passed=True, rows=[{}] * len(results), errors=[]))
    write_json(root / 'comparison-archives/study.json', dict(format='tslib15-v1', source='study', dataset='data.csv'))
    return tmp_path, root, source, dataset


def test_archives_keep_raw_paired_scores_provenance_and_read_only_identity(archive):
    project, root, source, _ = archive
    before = (source / 'test-results.json').read_bytes()
    records = list_archives(root, project)
    assert len(records) == 2
    assert {r['archive']['backbone'] for r in records} == {'dlinear', 'tsmixer'}
    for record in records:
        assert record['status']['phase'] == 'final_test'
        assert 'architecture' not in record['request']
        assert record['request']['evaluation']['split_metadata']['test'] == [10, 15]
        assert [r['seed'] for r in record['runs']] == [1, 2]
        assert record['summary'][0]['mae_mean'] == pytest.approx(.15)
        assert record['summary'][0]['mae_ratio'] == pytest.approx(.3)
        assert len(record['per_lead']) == 4
        assert record['runs'][0]['persistence_mse'] == .25
        assert 'directional_accuracy' not in record['runs'][0]  # Do not invent unrecorded diagnostics.
    assert before == (source / 'test-results.json').read_bytes()
    assert not (root / 'jobs').exists()


def test_forecast_overlay_uses_actual_dates_leads_and_saved_predictions(archive):
    project, root, _, _ = archive
    record = list_archives(root, project)[0]
    data = read_archive_forecasts(root, project, record['id'], window=2, horizon=2, seed=2, fold=1, lead=2)
    assert data['total'] == 4 and data['invalid'] == 0
    assert data['rows'][0] == dict(origin_row=9, forecast_origin='2026-01-01 09:00:00', target_date='2026-01-01 11:00:00', actual=11, prediction=11.2, persistence=11.5)
    with pytest.raises(ValueError, match='Unknown archived replicate'):
        read_archive_forecasts(root, project, record['id'], window=2, horizon=2, seed=99, fold=1, lead=1)
    with pytest.raises(ValueError, match='Invalid forecast'):
        read_archive_forecasts(root, project, record['id'], window=2, horizon=2, seed=1, fold=1, lead=3)


@pytest.mark.parametrize('problem', ['incomplete', 'failed_audit', 'metric_mismatch', 'dataset_changed'])
def test_rejects_incomplete_or_changed_evidence(archive, problem):
    project, root, source, dataset = archive
    if problem == 'failed_audit':
        write_json(source / 'artifact-audit.json', dict(passed=False, rows=[], errors=['bad archive']))
    elif problem == 'dataset_changed':
        dataset.write_text(dataset.read_text() + '\n')
    else:
        path = source / 'test-results.json'; results = json.loads(path.read_text())
        if problem == 'incomplete': results.pop()
        else: results[0]['mae'] = 99
        write_json(path, results)
    with pytest.raises(ValueError):
        list_archives(root, project)


def test_archive_api_does_not_add_jobs_or_allow_training(archive):
    project, root, _, _ = archive
    client = TestClient(create_app(root, project))
    records = client.get('/api/v1/comparison-archives').json()
    assert len(records) == 2
    assert client.get('/api/v1/jobs').json() == []
    record_id = records[0]['id']
    assert client.post(f'/api/v1/jobs/{record_id}/final-test').status_code == 404
    response = client.get(f'/api/v1/comparison-archives/{record_id}/forecasts', params=dict(window=2, horizon=2, seed=1, fold=1, lead=1))
    assert response.status_code == 200 and response.json()['total'] == 4
    assert client.get('/api/v1/comparison-archives/missing/forecasts', params=dict(window=2, horizon=2, seed=1, fold=1)).status_code == 404
