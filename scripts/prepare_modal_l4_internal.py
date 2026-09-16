#!/usr/bin/env python3
"""Prepare or advance the separately authorized $20 internal-replacement study."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tarfile

import pandas as pd
import torch

import run_modal_l4_controller as controller
from hypercast4d.internal_controls import preflight, variant_spec, VARIANTS
from hypercast4d.model_editing import checked, load_model, write_model

ROOT = controller.PROJECT_ROOT / 'results/modal-l4-internal'
BACKBONES = ('dlinear', 'tsmixer', 'itransformer')


def candidate(backbone, variant, path, evaluation, stage, *, timeout=600, depends=()):
    return dict(id=f'{backbone}-{variant}-{stage}-lr{evaluation["learning_rate"]}',
        backbone=backbone, variant=variant, variant_name=variant,
        architecture_file=str(path), evaluation=deepcopy(evaluation), stage_name=stage,
        timeout_seconds=timeout, max_retries=0, status='pending', job_id=None, attempts=0,
        reserved_usd=0., estimated_cost_usd=0., started_at=None, finished_at=None,
        status_message='', depends_on=list(depends))


def freeze(state):
    project = controller.PROJECT_ROOT
    paths = list((project / 'src/hypercast4d').rglob('*.py'))
    paths += list((ROOT / 'specs').glob('*.yaml'))
    paths += [project / p for p in ('pyproject.toml', 'data/raw/paper_data.xlsx',
        'scripts/run_modal_l4_controller.py', 'scripts/prepare_modal_l4_internal.py',
        'tests/test_internal_controls.py', 'docs/modal_l4_internal_replacement_plan.md',
        'docs/modal_l4_internal_operations.md')]
    state['frozen_inputs'] = {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    directory = ROOT / 'controller/provenance'
    directory.mkdir(exist_ok=False)
    with tarfile.open(directory / 'source-data-specs.tar.gz', 'w:gz') as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(project)))
    (directory / 'sha256.json').write_text(json.dumps(state['frozen_inputs'], indent=2) + '\n')


def prepare():
    directory = ROOT / 'controller'
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / controller.STATE_FILENAME
    if state_path.exists():
        raise RuntimeError('New study already exists; resume it without resetting its allocation')
    if controller.remote_apps_active():
        raise RuntimeError('Another Modal job is active')
    report = preflight('cpu')
    (directory / 'cpu-preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    evaluation = dict(preset='standard', cells=[dict(window=32, horizon=5)], seeds=[907],
        epochs=5, batch_size=32, learning_rate=.001, initialization='internal-matched-v1',
        nested_stopping=True, early_stopping_patience=20, early_stopping_min_delta=0.,
        early_stopping_relative_delta=.001, restore_best_weights=True, benchmark=True)
    rows = []
    for backbone in BACKBONES:
        native = load_model(controller.PROJECT_ROOT / f'plans/internal-replacement/{backbone}-graph.yaml')
        for variant in VARIANTS:
            spec, _ = checked(variant_spec(native, variant), evaluation['cells'])
            path = Path('specs') / f'{backbone}-{variant}.yaml'
            write_model(ROOT / path, spec)
            row = candidate(backbone, variant, path, evaluation, 'internal-pilot',
                timeout=900 if not rows else 300, depends=[rows[0]['id']] if rows else [])
            if not rows:
                row['evaluation']['internal_preflight'] = True
            rows.append(row)
    baseline = controller.parse_monthly_summary()
    previous_path = controller.DEFAULT_RESULTS_ROOT / 'controller' / controller.STATE_FILENAME
    previous = controller.load_state(previous_path)
    state = dict(experiment_id='internal-replacement-2026-09-11', created_at=controller._utcnow(),
        updated_at=controller._utcnow(), project_root=str(controller.PROJECT_ROOT), results_root=str(ROOT),
        stage_name='internal-pilot', schedule_status='prepared',
        stage=dict(evaluation=evaluation, execution={'target': 'modal', 'gpu': 'L4'}),
        rates=controller.load_modal_rates(),
        budget=dict(total_usd=20., safety_reserve_usd=1., max_job_seconds=3600, avg_fit_seconds=90.,
            jobs_submitted=0, confirmed_spend_usd=0., estimated_spend_usd=0., max_retries=0,
            uncertainty_factor=3., startup_cost_usd=.05),
        billing=dict(baseline_metered_usd=baseline['metered_cost'], baseline_billed_usd=baseline['billed_cost']),
        previous_allocation=dict(state_file=str(previous_path), sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest(),
            accounted_spend_usd=controller.accounted_spend(previous), total_usd=previous['budget']['total_usd'],
            relation='Separate additional $20 explicitly authorized by user; previous allocation remains closed'),
        candidates=rows, independent_final_allowed=False)
    controller.migrate_state(state)
    freeze(state)
    controller.save_state(state_path, state)
    print(json.dumps(dict(stage=state['stage_name'], candidates=len(rows), budget=state['budget'],
        previous_allocation=state['previous_allocation']), indent=2))


def read_runs(state, row):
    directory = Path(state['results_root']) / 'jobs' / row['job_id']
    # Fail closed on missing scientific audit artifacts, even if the backend reports success.
    init = json.loads((directory / 'initialization.json').read_text())
    audit = json.loads((directory / 'split_audit.json').read_text())
    runs = pd.read_csv(directory / 'runs.csv')
    if not init or not audit or not len(runs) or not (runs['stopping_split'] == 'inner').all():
        raise RuntimeError('Missing internal-initialization/nested-stopping evidence')
    if not all(x['protocol'] == 'internal-matched-v1' for x in init):
        raise RuntimeError('Wrong initialization protocol')
    if row['evaluation'].get('internal_preflight'):
        if not json.loads((directory / 'preflight.json').read_text())['passed']:
            raise RuntimeError('GPU preflight failed')
    return runs


def extend(state):
    """Only complete, balanced stages advance; the old ledger is never touched."""
    if not all(c['status'] == 'complete' for c in state['candidates']):
        raise RuntimeError('Complete every scheduled job before advancing')
    prior = state['stage_name']
    sources = [c for c in state['candidates'] if c['stage_name'] == prior]
    expected = {(b, v) for b in BACKBONES for v in VARIANTS}
    if prior in ('internal-pilot', 'internal-development'):
        repetitions = 1 if prior == 'internal-pilot' else 2
        if len(sources) != 24 * repetitions or {(c['backbone'], c['variant']) for c in sources} != expected:
            raise RuntimeError('Incomplete balanced internal comparison')
    data = {c['id']: read_runs(state, c) for c in sources}
    if prior == 'internal-pilot':
        stage = 'internal-development'
        additions = []
        for row in sources:
            for lr in (.0003, .001):
                evaluation = {**row['evaluation'], 'learning_rate': lr, 'epochs': 150, 'seeds': [701]}
                evaluation.pop('internal_preflight', None)
                additions.append(candidate(row['backbone'], row['variant'], row['architecture_file'],
                    evaluation, stage, timeout=900, depends=[row['id']]))
        seconds = sum(float(df['train_seconds'].sum()) for df in data.values()) * 60
        decision = dict(reason='Equal two-rate development grid; pilot scores do not select arms', seeds=[701])
    elif prior == 'internal-development':
        stage = 'internal-robustness'
        selected = []
        for backbone in BACKBONES:
            for variant in VARIANTS:
                options = [c for c in sources if (c['backbone'], c['variant']) == (backbone, variant)]
                if len(options) != 2:
                    raise RuntimeError('Incomplete two-rate comparison')
                scores = {c['id']: float(data[c['id']]['mae_ratio'].mean()) for c in options}
                best = min(scores.values())
                if not all(math.isfinite(v) and v > 0 for v in scores.values()):
                    raise RuntimeError('Invalid development score')
                selected.append(min((c for c in options if scores[c['id']] <= best * 1.001),
                    key=lambda c: c['evaluation']['learning_rate']))
        # Reduce only replication, symmetrically, if measured cost rules out all three seeds.
        seconds_per_seed = sum(float(data[c['id']]['train_seconds'].sum()) for c in selected) * 3
        headroom = state['budget']['total_usd'] - state['budget']['safety_reserve_usd'] - controller.accounted_spend(state)
        seeds = [401, 503, 601]
        while seeds and cost(state, seconds_per_seed * len(seeds), 24) + 1.5 > headroom:
            seeds.pop()
        if not seeds:
            raise RuntimeError('No complete balanced robustness matrix fits remaining allocation')
        additions = []
        for row in selected:
            evaluation = {**row['evaluation'], 'preset': 'robust', 'seeds': seeds}
            evaluation.pop('folds', None)
            additions.append(candidate(row['backbone'], row['variant'], row['architecture_file'],
                evaluation, stage, timeout=1800, depends=[row['id']]))
        seconds = seconds_per_seed * len(seeds)
        decision = dict(reason='Frozen development MAE ratio; 0.1% ties choose lower rate', seeds=seeds,
            selected=[dict(backbone=c['backbone'], variant=c['variant'], learning_rate=c['evaluation']['learning_rate'],
                           job_id=c['job_id']) for c in selected])
    else:
        raise RuntimeError('Primary matrix complete: AI must analyze it and preregister any affordable follow-up')
    estimate = cost(state, seconds, len(additions))
    headroom = state['budget']['total_usd'] - state['budget']['safety_reserve_usd'] - controller.accounted_spend(state)
    if estimate + 1.5 > headroom:
        raise RuntimeError(f'Balanced {stage} estimated ${estimate:.2f} plus end-of-queue margin exceeds ${headroom:.2f}')
    result = deepcopy(state)
    result['candidates'].extend(additions)
    result.update(stage_name=stage, schedule_status='prepared')
    result.setdefault('stage_history', []).append(dict(stage=prior, completed_at=controller._utcnow(),
        accounted_spend_usd=controller.accounted_spend(state)))
    result.setdefault('stage_decisions', []).append(dict(stage=stage, estimated_cost_usd=estimate,
        headroom_usd=headroom, jobs=len(additions), decision=decision,
        caveat='Exploratory historical Copper; overlapping development/robustness periods; no independent final'))
    controller.migrate_state(result)
    return result


def cost(state, seconds, jobs):
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError('Invalid measured training duration')
    return controller.estimate_job_cost_usd(1, state['rates'], seconds + jobs * 40,
        state['budget']['uncertainty_factor'], jobs * state['budget']['startup_cost_usd'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--advance', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    directory = ROOT / 'controller'
    directory.mkdir(parents=True, exist_ok=True)
    with controller.ensure_single_controller(directory / '.controller.lock'):
        if not args.advance:
            prepare()
            return
        path = directory / controller.STATE_FILENAME
        state = controller.load_state(path)
        controller.verify_frozen_inputs(state)
        if controller.remote_apps_active():
            raise RuntimeError('Wait for existing Modal activity to stop')
        backend = controller.ensure_backend(controller.PROJECT_ROOT, ROOT)
        jobs = controller.request(backend['url'], '/api/v1/jobs')
        _, active = controller.reconcile_candidate_state(state, jobs, controller._utcnow(), state['rates'], state['stage']['evaluation'])
        if active:
            raise RuntimeError('Unresolved active job')
        controller.update_cost_ledger(state, jobs)
        rates = controller.load_modal_rates()
        state['rates'] = {k: max(state['rates'][k], v) for k, v in rates.items()}
        controller.refresh_billing(state, controller.parse_monthly_summary())
        prepared = extend(state)
        decision_path = directory / f'{prepared["stage_name"]}-decision.json'
        if decision_path.exists():
            raise RuntimeError('Stage decision already exists; inspect before retrying')
        decision_path.write_text(json.dumps(prepared['stage_decisions'][-1], indent=2) + '\n')
        prepared['frozen_inputs'][str(decision_path.relative_to(controller.PROJECT_ROOT))] = hashlib.sha256(decision_path.read_bytes()).hexdigest()
        controller.save_state(path, prepared)
        print(json.dumps(prepared['stage_decisions'][-1], indent=2))


if __name__ == '__main__':
    main()
