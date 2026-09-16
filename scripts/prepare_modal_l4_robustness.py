#!/usr/bin/env python3
"""Append the preregistered robustness stage to the existing single-L4 queue."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil

import run_modal_l4_controller as controller
from report_modal_l4_development import report


def extend(state, selection):
    """Preserve history and spending, and require the complete balanced selection."""
    if any(c.get('stage_name') == 'controlled-robustness' for c in state['candidates']):
        raise RuntimeError('Robustness already prepared; resume the existing controller')
    if state.get('schedule_status') != 'controlled-development_complete' or not all(
            c['status'] == 'complete' for c in state['candidates']):
        raise RuntimeError('Finish and reconcile every development job first')
    originals = [c for c in state['candidates'] if c.get('stage_name') == 'pilot']
    expected = {(b, v.key) for b in controller.BACKBONES for v in controller.VARIANTS}
    selected = {(c['backbone'], c['variant']): c for c in selection.get('selected', [])}
    if (not selection.get('complete') or len(selection.get('selected', [])) != 27
            or set(selected) != expected or {(c['backbone'], c['variant']) for c in originals} != expected):
        raise RuntimeError('Require all 27 configurations and their frozen rates')
    seconds = sum(float(c['train_seconds']) for c in selected.values()) * 6 * (.65 / .70)
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError('Invalid measured development timings')
    estimate = controller.estimate_job_cost_usd(1, state['rates'], seconds + 27 * 30,
        state['budget']['uncertainty_factor'], 27 * state['budget']['startup_cost_usd'])
    headroom = state['budget']['total_usd'] - state['budget']['safety_reserve_usd'] - controller.accounted_spend(state)
    if estimate > headroom:
        raise RuntimeError(f'Robustness estimate ${estimate:.2f} exceeds headroom ${headroom:.2f}')
    additions = []
    for original in originals:
        chosen = selected[original['backbone'], original['variant']]
        if chosen['learning_rate'] not in (.0003, .001):
            raise RuntimeError('Selected learning rate outside frozen development grid')
        parent = next((c for c in state['candidates'] if c.get('stage_name') == 'controlled-development'
                       and c.get('job_id') == chosen['job_id']), None)
        if (parent is None or (parent['backbone'], parent['variant']) != (original['backbone'], original['variant'])
                or parent['evaluation']['learning_rate'] != chosen['learning_rate']):
            raise RuntimeError('Frozen selection does not identify its development parent')
        evaluation = {**deepcopy(parent['evaluation']), 'preset': 'robust', 'seeds': [101, 211]}
        evaluation.pop('folds', None)
        if controller.estimate_trials(controller.normalize_evaluation(evaluation)) != 6:
            raise RuntimeError('Robustness candidate must contain six fits')
        row = {key: original[key] for key in ('backbone', 'variant', 'variant_name', 'architecture_file')}
        row.update(id=f"{original['id']}-robust-seeds101-211", stage_name='controlled-robustness',
            evaluation=evaluation, timeout_seconds=1200, status='pending', job_id=None, attempts=0,
            reserved_usd=0., estimated_cost_usd=0., started_at=None, finished_at=None, status_message='',
            depends_on=[parent['id']], selected_development_job_id=chosen['job_id'])
        additions.append(row)
    # Work on a copy: validation failures must never partly append a stage.
    result = deepcopy(state)
    result['candidates'].extend(additions)
    result.setdefault('stage_history', []).append({'stage': 'controlled-development', 'completed': 54,
        'finished_at': state.get('updated_at'), 'accounted_spend_usd': controller.accounted_spend(state)})
    result.update(stage_name='controlled-robustness', schedule_status='prepared',
        robustness_plan={'fits': 162, 'jobs': 27, 'estimated_cost_usd': estimate,
            'headroom_at_preparation_usd': headroom, 'timeout_seconds': 1200})
    controller.migrate_state(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=controller.DEFAULT_RESULTS_ROOT)
    args = parser.parse_args()
    root = args.results_root.resolve(); directory = root / 'controller'
    with controller.ensure_single_controller(directory / '.controller.lock'):
        state = controller.load_state(directory / controller.STATE_FILENAME)
        controller.verify_frozen_inputs(state)
        if controller.remote_apps_active():
            raise RuntimeError('Wait for the existing Modal job to stop')
        backend = controller.ensure_backend(controller.PROJECT_ROOT, root)
        jobs = controller.request(backend['url'], '/api/v1/jobs')
        _, active = controller.reconcile_candidate_state(state, jobs, controller._utcnow(), state['rates'], state['stage']['evaluation'])
        if active:
            raise RuntimeError('Unresolved active or missing job prevents stage preparation')
        controller.update_cost_ledger(state, jobs)
        rates = controller.load_modal_rates()
        state['rates'] = {k: max(state['rates'][k], v) for k, v in rates.items()}
        controller.refresh_billing(state, controller.parse_monthly_summary())
        selection_path = directory / 'development_selection.json'
        selection = json.loads(selection_path.read_text())
        current = report(root)
        if not current['complete'] or current['selected'] != selection['selected']:
            raise RuntimeError('Frozen rates differ from verified development artifacts')
        prepared = extend(state, selection)
        provenance = directory / 'robustness-provenance'
        provenance.mkdir(exist_ok=False)
        shutil.copy2(directory / controller.STATE_FILENAME, provenance / 'preparation-state.json')
        tracked = ['scripts/prepare_modal_l4_robustness.py', 'scripts/report_modal_l4_development.py',
                   'docs/modal_l4_robustness_stage.md', str(selection_path.relative_to(controller.PROJECT_ROOT))]
        tracked.extend(str((directory / name).relative_to(controller.PROJECT_ROOT)) for name in
                       ('robustness-dry-runs.json', 'development_analysis.md',
                        'development_convergence.json', 'development_dependence.json'))
        for relative in tracked:
            path = controller.PROJECT_ROOT / relative
            prepared['frozen_inputs'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            shutil.copy2(path, provenance / path.name)
        (provenance / 'sha256.json').write_text(json.dumps(prepared['frozen_inputs'], indent=2) + '\n')
        (provenance / 'prepared-manifest.json').write_text(json.dumps(prepared, indent=2) + '\n')
        prepared['robustness_provenance'] = str(provenance)
        prepared['updated_at'] = controller._utcnow()
        controller.save_state(directory / controller.STATE_FILENAME, prepared)
        print(json.dumps(prepared['robustness_plan'], indent=2))


if __name__ == '__main__':
    main()
