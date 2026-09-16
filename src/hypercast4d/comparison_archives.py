"""Read-only adapters for completed research archives, never executable jobs."""
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _json(path):
    return json.loads(Path(path).read_text())


def _inside(project, path):
    resolved = (project / path).resolve()
    if not resolved.is_relative_to(project.resolve()):
        raise ValueError('Comparison archive must be inside this project')
    return resolved


def _registered(root, project):
    for file in sorted((root / 'comparison-archives').glob('*.json')):
        item = _json(file)
        if item.get('format') != 'tslib15-v1':
            raise ValueError(f'Unsupported comparison archive format: {file.name}')
        source, data = _inside(project, item['source']), _inside(project, item['dataset'])
        ledger = _json(source / 'ledger.json')
        plan = _inside(source, str(Path(ledger['prepared_dir']) / 'evaluation-plan.json'))
        paths = [source / name for name in ('ledger.json', 'test-results.json', 'artifact-audit.json')] + [plan, data]
        version = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
        yield source, data, plan, version


@lru_cache(maxsize=8)
def _load(source, dataset, plan_path, version):
    plan, ledger = _json(plan_path), _json(source / 'ledger.json')
    results, audit = _json(source / 'test-results.json'), _json(source / 'artifact-audit.json')
    expected = {(b, a, s) for b in plan['backbones'] for a in plan['arms'] for s in plan['evaluation_seeds']}
    keys = [(r['job']['backbone'], r['job']['arm'], r['job']['seed']) for r in results]
    if (ledger['status'] != 'complete' or not audit.get('passed') or audit.get('errors')
            or len(audit['rows']) != len(expected) or len(keys) != len(expected) or set(keys) != expected):
        raise ValueError('Comparison requires the complete, audited model/variant/seed matrix')
    manifest = plan['dataset_manifest']
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != manifest['source_sha256']:
        raise ValueError('Comparison dataset does not match its recorded hash')
    attempts = {(a['job']['backbone'], a['job']['arm'], a['job']['seed']): a
                for a in ledger['attempts'] if a['job']['phase'] == 'test' and a['status'] == 'complete'}
    partitions = manifest['partitions']
    split_metadata = {k: v['rows'] for k, v in partitions.items()}
    split_hash = hashlib.sha256(json.dumps(split_metadata, sort_keys=True).encode()).hexdigest()
    window, horizon = manifest['context'], manifest['horizon']
    grouped, forecasts = defaultdict(list), {}
    expected_actual = expected_origins = None
    for result in results:
        job = result['job']; key = (job['backbone'], job['arm'], job['seed'])
        if job['phase'] != 'test' or not result['test_scored'] or not result['checkpoint_replay_passed']:
            raise ValueError('Only completed final-test fits may be compared here')
        attempt = attempts[key]
        path = source / 'predictions' / f"{attempt['index']:03d}.npz"
        with np.load(path, allow_pickle=False) as data:
            prediction, actual, persistence, origins = [data[k] for k in ('prediction', 'actual', 'persistence', 'target_start')]
            if actual.shape != (partitions['test']['origins'], horizon) or prediction.shape != actual.shape or persistence.shape != actual.shape:
                raise ValueError('Archived forecast shape does not match the evaluation plan')
            if not all(np.isfinite(x).all() for x in (prediction, actual, persistence, origins)):
                raise ValueError('Nonfinite archived forecasts')
            if len(np.unique(origins)) != len(origins) or origins.min() != partitions['test']['rows'][0] or origins.max() + horizon != partitions['test']['rows'][1]:
                raise ValueError('Archived forecast origins do not match the test partition')
            if expected_actual is None:
                expected_actual, expected_origins = actual.copy(), origins.copy()
            elif not np.array_equal(actual, expected_actual) or not np.array_equal(origins, expected_origins):
                raise ValueError('Archived models have different evaluation targets or origins')
            error, baseline = prediction - actual, persistence - actual
            mae, mse, baseline_mae = float(np.abs(error).mean()), float(np.square(error).mean()), float(np.abs(baseline).mean())
            if not all(np.isclose(value, result[field], rtol=1e-9, atol=1e-10) for value, field in ((mae, 'mae'), (mse, 'mse'), (baseline_mae, 'persistence_mae'))):
                raise ValueError('Archived metrics do not match the saved predictions')
            common = dict(window=window, horizon=horizon, seed=job['seed'], fold=1, split='test',
                          train_samples=partitions['train']['origins'], validation_samples=partitions['inner']['origins'], test_samples=len(origins))
            row = dict(common, mae=result['mae'], mse=result['mse'], rmse=result['rmse'], bias=result['bias'],
                       persistence_mae=result['persistence_mae'], persistence_mse=float(np.square(baseline).mean()),
                       parameters=result['control']['parameters'], train_seconds=result['training_elapsed_seconds'],
                       epochs_ran=result['epochs_ran'], p95_abs_error=float(np.quantile(np.abs(error), .95)),
                       sample_count=int(error.size), forecast_origins=len(origins), ceiling_reached=int(result['ceiling_reached']),
                       latency_batch32_ms=result['inference']['32']['median_ms'])
            leads = [dict(common, lead=i + 1, mae=float(np.abs(error[:, i]).mean()), mse=float(np.square(error[:, i]).mean()),
                          bias=float(error[:, i].mean()), persistence_mae=float(np.abs(baseline[:, i]).mean()),
                          persistence_mse=float(np.square(baseline[:, i]).mean())) for i in range(horizon)]
        grouped[(job['backbone'], job['arm'])].append((row, leads, result, attempt))
        forecasts[key] = path
    records = []
    for arm in plan['arms']:
        for backbone in plan['backbones']:
            fits = sorted(grouped[(backbone, arm)], key=lambda item: item[0]['seed'])
            rows = [item[0] for item in fits]; first = fits[0][2]
            epochs = {r[2]['job']['epochs'] for r in fits}; rates = {r[2]['job']['learning_rate'] for r in fits}
            if len(epochs) != 1 or len(rates) != 1:
                raise ValueError('Mixed recipes within an archived model/variant')
            display = {'dlinear': 'DLinear', 'tsmixer': 'TSMixer', 'itransformer': 'iTransformer', 'patchtst': 'PatchTST',
                       'timesnet': 'TimesNet', 'crossformer': 'Crossformer', 'frets': 'FreTS', 'lightts': 'LightTS', 'micn': 'MICN',
                       'msgnet': 'MSGNet', 'scinet': 'SCINet', 'segrnn': 'SegRNN', 'timemixer': 'TimeMixer', 'timexer': 'TimeXer', 'film': 'FiLM'}.get(backbone, backbone)
            evaluation = dict(protocol=plan['protocol'], preset='archived-study', data_path=dataset.name, target_column=manifest['target'],
                              data_sha256=manifest['source_sha256'], split_sha256=split_hash, split_metadata=split_metadata,
                              target_scale='Original OT units', cells=[dict(window=window, horizon=horizon)], seeds=plan['evaluation_seeds'],
                              epochs=next(iter(epochs)), learning_rate=next(iter(rates)), batch_size=32, evaluation_batch_size=32,
                              loss='mae', adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-7, adam_amsgrad=False, shuffle=True,
                              early_stopping_patience=plan['patience'], early_stopping_min_delta=0, restore_best_weights=True, device='cuda')
            stamp = datetime.fromtimestamp(max(f[3]['ended_epoch'] for f in fits), timezone.utc).isoformat()
            record_id = f'archive-{source.name}-{backbone}-{arm}'
            records.append(dict(id=record_id, request=dict(evaluation=evaluation, execution=dict(target='modal', gpu='L4')),
                status=dict(id=record_id, protocol=plan['protocol'], execution_target='modal', state='complete', phase='final_test', preset='archived-study', architecture_name=f'{display} · {arm}',
                            created_at=stamp, updated_at=stamp, completed=len(rows), total=len(rows)),
                summary=[dict(window=window, horizon=horizon, mae_mean=float(np.mean([r['mae'] for r in rows])),
                              mse_mean=float(np.mean([r['mse'] for r in rows])),
                              mae_std=float(np.std([r['mae'] for r in rows], ddof=1)) if len(rows) > 1 else None,
                              mse_std=float(np.std([r['mse'] for r in rows], ddof=1)) if len(rows) > 1 else None,
                              persistence_mae=float(np.mean([r['persistence_mae'] for r in rows])),
                              persistence_mse=float(np.mean([r['persistence_mse'] for r in rows])),
                              parameters=rows[0]['parameters'], train_seconds=float(np.mean([r['train_seconds'] for r in rows])),
                              epochs_median=float(np.median([r['epochs_ran'] for r in rows])),
                              mae_ratio=float(np.mean([r['mae'] / r['persistence_mae'] for r in rows])),
                              mse_ratio=float(np.mean([r['mse'] / r['persistence_mse'] for r in rows])))], runs=rows,
                per_lead=[lead for fit in fits for lead in fit[1]],
                archive=dict(collection='TSLib 15 · ETTh1', backbone=backbone, variant=arm, source=source.name,
                             target_scale=evaluation['target_scale'], task=manifest['task'], forecasts=True,
                             split_description=' · '.join(f'{name}: rows {span[0]}–{span[1] - 1}' for name, span in split_metadata.items()),
                             note=f'Completed archived study; read-only. {len(plan["evaluation_seeds"])} paired seeds, one fixed test partition. Whiskers on this page are seed ranges, not the study’s simultaneous bootstrap intervals.')))
    return records, forecasts


def list_archives(root: Path, project: Path):
    return [record for source, dataset, plan, version in _registered(root, project)
            for record in _load(source, dataset, plan, version)[0]]


def read_archive_forecasts(root: Path, project: Path, record_id: str, *, window: int, horizon: int, seed: int, fold: int, lead: int):
    if not 1 <= lead <= horizon or window < 1:
        raise ValueError('Invalid forecast cell or lead')
    for source, dataset, plan, version in _registered(root, project):
        records, forecasts = _load(source, dataset, plan, version)
        record = next((r for r in records if r['id'] == record_id), None)
        if record is None:
            continue
        if fold != 1 or not any(r['window'] == window and r['horizon'] == horizon and r['seed'] == seed for r in record['runs']):
            raise ValueError('Unknown archived replicate or evaluation cell')
        with dataset.open(newline='') as file:
            dates = [r['date'] for r in csv.DictReader(file)]
        path = forecasts[(record['archive']['backbone'], record['archive']['variant'], seed)]
        with np.load(path, allow_pickle=False) as data:
            indices = np.argsort(data['target_start'])[-2000:]
            rows = [dict(origin_row=int(data['target_start'][i]) - 1, forecast_origin=dates[int(data['target_start'][i]) - 1],
                         target_date=dates[int(data['target_start'][i]) + lead - 1], actual=float(data['actual'][i, lead - 1]),
                         prediction=float(data['prediction'][i, lead - 1]), persistence=float(data['persistence'][i, lead - 1])) for i in indices]
            return dict(rows=rows, total=len(data['target_start']), invalid=0, limit=2000)
    raise KeyError(record_id)
