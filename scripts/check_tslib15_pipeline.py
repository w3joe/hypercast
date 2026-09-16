"""Read-only check of scaling, temporal alignment and archived TSLib15 scores.

Reconstructs tensors directly from the original CSV, without calling the training
pipeline's partition/scaling helpers. Does not train or alter experiment artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(project, run):
    ledger = json.loads((run / 'ledger.json').read_text())
    prepared = run / ledger['prepared_dir']
    plan = json.loads((prepared / 'evaluation-plan.json').read_text())
    manifest = plan['dataset_manifest']
    source = project / 'data/external/etth1-1d16c8f/ETTh1.csv'
    assert sha(source) == manifest['source_sha256'], 'Source CSV changed'
    worker = 'scripts/run_tslib15_worker.py'
    assert sha(prepared / 'frozen' / worker) == plan['source_sha256'][worker]
    for phase, digest in plan['data_sha256'].items():
        assert sha(prepared / f'{phase}.npz') == digest, f'{phase} bundle changed'
    frame = pd.read_csv(source)
    dates = pd.to_datetime(frame['date'])
    assert dates.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
    columns = manifest['columns']
    assert columns[0] == manifest['target'] == 'OT'
    raw = frame[columns].to_numpy(dtype=np.float64)
    assert np.isfinite(raw).all()
    start, end = manifest['scaler_rows']
    assert [start, end] == manifest['partitions']['train']['rows']
    minimum = raw[start:end].min(axis=0)
    span = raw[start:end].max(axis=0) - minimum
    span[span == 0] = 1
    np.testing.assert_array_equal(minimum, manifest['scaler_minimum'])
    np.testing.assert_array_equal(span, manifest['scaler_span'])
    scaled = ((raw - minimum) / span).astype(np.float32)
    window, horizon = manifest['context'], manifest['horizon']
    checked = {}
    with np.load(prepared / 'development.npz', allow_pickle=False) as development, np.load(prepared / 'test.npz', allow_pickle=False) as test:
        for phase, split in manifest['partitions'].items():
            begin, finish = split['rows']
            origins = np.arange(max(window, begin), finish - horizon + 1)
            inputs = origins[:, None] + np.arange(-window, 0)
            targets = origins[:, None] + np.arange(horizon)
            assert np.all(inputs[:, -1] < targets[:, 0])
            assert targets.min() >= begin and targets.max() < finish
            bundle = test if phase == 'test' else development
            np.testing.assert_array_equal(bundle[phase + '_x'], scaled[inputs])
            np.testing.assert_array_equal(bundle[phase + '_y'], scaled[targets, 0])
            if phase + '_target_start' in bundle:
                np.testing.assert_array_equal(bundle[phase + '_target_start'], origins)
            if phase in ('train', 'inner'):
                np.testing.assert_array_equal(test[phase + '_x'], development[phase + '_x'])
                np.testing.assert_array_equal(test[phase + '_y'], development[phase + '_y'])
            checked[phase] = dict(origins=len(origins), first_target_row=int(targets.min()), last_target_row=int(targets.max()))
        test_origins = test['test_target_start']
        actual = test['test_y'].astype(float) * span[0] + minimum[0]
        persistence = np.repeat(test['test_x'][:, -1, 0:1].astype(float), horizon, axis=1) * span[0] + minimum[0]
        raw_targets = raw[test_origins[:, None] + np.arange(horizon), 0]
        raw_persistence = np.repeat(raw[test_origins - 1, 0:1], horizon, axis=1)
        np.testing.assert_allclose(actual, raw_targets, rtol=0, atol=3e-6)
        np.testing.assert_allclose(persistence, raw_persistence, rtol=0, atol=3e-6)
    results = json.loads((run / 'test-results.json').read_text())
    key = lambda job: (job['backbone'], job['arm'], job['seed'])
    scores = {key(result['job']): result for result in results}
    expected = {(b, a, s) for b in plan['backbones'] for a in plan['arms'] for s in plan['evaluation_seeds']}
    assert set(scores) == expected and len(scores) == len(results)
    attempts = [a for a in ledger['attempts'] if a['job']['phase'] == 'test' and a['status'] == 'complete']
    assert len(attempts) == len(expected) and {key(a['job']) for a in attempts} == expected
    for attempt in attempts:
        result = scores[key(attempt['job'])]
        with np.load(run / 'predictions' / f"{attempt['index']:03d}.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['actual'], actual)
            np.testing.assert_array_equal(saved['persistence'], persistence)
            np.testing.assert_array_equal(saved['target_start'], test_origins)
            prediction = saved['prediction']
            assert prediction.shape == actual.shape and np.isfinite(prediction).all()
            mae = np.abs(prediction - actual).mean()
            mse = np.square(prediction - actual).mean()
            baseline_mae = np.abs(persistence - actual).mean()
            np.testing.assert_allclose([mae, mse, baseline_mae], [result['mae'], result['mse'], result['persistence_mae']], rtol=1e-12, atol=1e-12)
    dlinear = [r for r in results if r['job']['backbone'] == 'dlinear' and r['job']['arm'] == 'native']
    baseline = float(np.abs(persistence - actual).mean())
    return dict(passed=True, source_sha256=manifest['source_sha256'], archived_fits_checked=len(attempts),
        partitions=checked, scaler='Training-only min–max', scaler_rows=[start, end], target_column=columns[0],
        target_minimum=float(minimum[0]), target_span=float(span[0]),
        max_raw_target_roundtrip_error=float(np.abs(actual - raw_targets).max()),
        persistence_mae_original_units=baseline,
        dlinear_native_mean_mae_original_units=float(np.mean([r['mae'] for r in dlinear])),
        dlinear_native_mean_mae_ratio=float(np.mean([r['mae'] / r['persistence_mae'] for r in dlinear])),
        scope='Verifies saved tensors, scale roundtrip, origins and saved prediction metrics; does not rerun training or infer absence of every possible pipeline bug.')


if __name__ == '__main__':
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=project / 'results/tslib15-20260915')
    args = parser.parse_args()
    print(json.dumps(check(project, args.run), indent=2))
