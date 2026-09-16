#!/usr/bin/env python3
"""Verify and append the remaining-backbone calibration to the existing $20 ledger."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

import torch

import run_modal_l4_controller as controller
from prepare_modal_l4_internal import candidate
from hypercast4d.architecture import presets
from hypercast4d.model_editing import checked, write_model
from hypercast4d.remaining_controls import BACKBONES, VARIANTS, PROTOCOL, preflight

ROOT = controller.PROJECT_ROOT / 'results/modal-l4-internal'
PLAN = controller.PROJECT_ROOT / 'plans/internal-expansion'
STAGE = 'remaining-calibration'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluation(backbone, variant):
    return dict(preset='standard', cells=[dict(window=32, horizon=5)], seeds=[907],
        epochs=5, batch_size=32, learning_rate=.001, initialization=PROTOCOL,
        replacement=dict(backbone=backbone, variant=variant), nested_stopping=True,
        early_stopping_patience=20, early_stopping_min_delta=0.,
        early_stopping_relative_delta=.001, restore_best_weights=True, benchmark=True,
        remaining_preflight=variant == 'native')


def validation_inputs():
    paths = list((controller.PROJECT_ROOT / 'src/hypercast4d').rglob('*.py'))
    paths += [Path(__file__).resolve(), controller.PROJECT_ROOT / 'scripts/run_modal_l4_controller.py']
    return {str(p.relative_to(controller.PROJECT_ROOT)): sha(p) for p in paths}


def verify():
    """Exercise real CLI normalization for all 96 configurations and CPU gates."""
    report = dict(protocol=PROTOCOL, passed=False, source_hashes=validation_inputs(), cpu=[], cli=[])
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        hashes = set()
        for backbone in BACKBONES:
            report['cpu'].append(preflight(backbone, 'cpu'))
            spec, _ = checked(next(p for p in presets() if p['preset_id'] == 'tslib-' + backbone),
                              [dict(window=32, horizon=5)])
            spec_path = directory / f'{backbone}.yaml'
            write_model(spec_path, spec)
            for variant in VARIANTS:
                ev_path = directory / 'evaluation.json'
                ev_path.write_text(json.dumps(evaluation(backbone, variant)))
                command = [str(Path(sys.executable).with_name('hypercast')), 'experiment', 'run',
                    str(spec_path), '--evaluation', str(ev_path), '--target', 'modal', '--gpu', 'L4',
                    '--device', 'cuda', '--dry-run', '--json']
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                payload = json.loads(result.stdout)
                assert payload['dry_run'] and payload['trials'] == 1
                assert payload['evaluation']['replacement'] == dict(backbone=backbone, variant=variant)
                assert payload['candidate_hash'] not in hashes
                hashes.add(payload['candidate_hash'])
                report['cli'].append(dict(backbone=backbone, variant=variant, **payload))
            print(f'CPU and CLI verified: {backbone}, eight variants', flush=True)
    assert len(hashes) == 96 and report['source_hashes'] == validation_inputs()
    report.update(passed=True, completed_at=controller._utcnow(), cuda_validation='deferred to bounded calibration jobs')
    (PLAN / 'integrated-verification.json').write_text(json.dumps(report, indent=2) + '\n')


def prepare():
    directory = ROOT / 'controller'
    path = directory / controller.STATE_FILENAME
    with controller.ensure_single_controller(directory / '.controller.lock'):
        state = controller.load_state(path)
        if state['stage_name'] != 'internal-development' or not all(c['status'] == 'complete' for c in state['candidates']):
            raise RuntimeError('Calibration may be appended exactly once, after completed internal development')
        if controller.remote_apps_active():
            raise RuntimeError('Wait for all remote experiment activity to stop')
        review = json.loads((directory / 'internal-development-review.json').read_text())
        assert review['passed'] and review['completed_fits'] == 48
        previous = state['previous_allocation']
        assert sha(Path(previous['state_file'])) == previous['sha256']
        provenance = directory / 'expansion-provenance'
        before = json.loads((provenance / 'preparation-state.json').read_text())
        assert before['frozen_inputs'] == state['frozen_inputs']
        with tarfile.open(provenance / 'pre-expansion-source.tar.gz') as archive:
            for name, expected in before['frozen_inputs'].items():
                assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected, name
        allowed = {'src/hypercast4d/experiment_controls.py', 'src/hypercast4d/playground_runner.py',
                   'src/hypercast4d/cli.py', 'docs/modal_l4_internal_operations.md'}
        changed = {name for name, expected in before['frozen_inputs'].items()
                   if sha(controller.PROJECT_ROOT / name) != expected}
        if changed - allowed:
            raise RuntimeError(f'Unexpected modification to frozen input: {sorted(changed - allowed)}')
        verification = json.loads((PLAN / 'integrated-verification.json').read_text())
        assert verification['passed'] and verification['source_hashes'] == validation_inputs()
        assert len(verification['cpu']) == 12 and len(verification['cli']) == 96
        backend = controller.ensure_backend(controller.PROJECT_ROOT, ROOT)
        jobs = controller.request(backend['url'], '/api/v1/jobs')
        _, active = controller.reconcile_candidate_state(state, jobs, controller._utcnow(),
            state['rates'], state['stage']['evaluation'])
        if active:
            raise RuntimeError('Unresolved backend job')
        controller.update_cost_ledger(state, jobs)
        state['rates'] = {k: max(state['rates'][k], v) for k, v in controller.load_modal_rates().items()}
        controller.refresh_billing(state, controller.parse_monthly_summary())
        # Persist the reconciled ledger even if preparation cannot be admitted.
        controller.save_state(path, state)
        reserve = controller.estimate_job_cost_usd(1, state['rates'], 720,
            state['budget']['uncertainty_factor'], state['budget']['startup_cost_usd'])
        headroom = state['budget']['total_usd'] - state['budget']['safety_reserve_usd'] - controller.accounted_spend(state)
        if reserve * 12 > headroom:
            raise RuntimeError('Full timeout reservations for all twelve calibrations exceed remaining funds')
        result = deepcopy(state)
        additions = []
        manifest = []
        for backbone in BACKBONES:
            spec, _ = checked(next(p for p in presets() if p['preset_id'] == 'tslib-' + backbone),
                              [dict(window=32, horizon=5)])
            spec_path = Path('specs') / f'remaining-{backbone}.yaml'
            write_model(ROOT / spec_path, spec)
            row = candidate(backbone, 'native', spec_path, evaluation(backbone, 'native'), STAGE,
                timeout=600, depends=[additions[-1]['id']] if additions else [])
            additions.append(row)
            for variant in VARIANTS:
                manifest.append(dict(backbone=backbone, variant=variant, architecture_file=str(spec_path),
                    evaluation=evaluation(backbone, variant), phase=STAGE if variant == 'native' else 'remaining-pilot'))
        result['candidates'].extend(additions)
        result.update(stage_name=STAGE, schedule_status='prepared')
        result.setdefault('stage_history', []).append(dict(stage='internal-development',
            completed_at=controller._utcnow(), accounted_spend_usd=controller.accounted_spend(state),
            next_stage_override='User requests remaining twelve; original-three robustness superseded'))
        decision = dict(stage=STAGE, prepared_at=controller._utcnow(), jobs=12,
            full_timeout_reservations_usd=round(reserve * 12, 6), headroom_usd=headroom,
            accounted_spend_usd=controller.accounted_spend(state), excluded=['dlinear', 'tsmixer', 'itransformer'],
            target_configurations=96, pending_after_calibration=84,
            next_stage='Cost complete balanced remaining-pilot from these measured timings before append; do not select on accuracy',
            changed_prior_inputs=sorted(changed), prior_archive_sha256=sha(provenance / 'pre-expansion-source.tar.gz'),
            scope='Eleven dense/pointwise interventions plus separate FiLM spectral intervention',
            independent_confirmation=False)
        result.setdefault('stage_decisions', []).append(decision)
        controller.migrate_state(result)
        assert [c['candidate_hash'] for c in result['candidates'][:len(state['candidates'])]] == [c['candidate_hash'] for c in state['candidates']]
        assert len({c['candidate_hash'] for c in result['candidates']}) == len(result['candidates'])
        for filename, payload in [('remaining-manifest.json', manifest), ('remaining-calibration-decision.json', decision)]:
            destination = directory / filename
            if destination.exists():
                raise RuntimeError('Preparation artifact already exists; inspect before retrying')
            destination.write_text(json.dumps(payload, indent=2) + '\n')
        frozen = set(before['frozen_inputs']) | set(validation_inputs())
        frozen.update(str(p.relative_to(controller.PROJECT_ROOT)) for p in (ROOT / 'specs').glob('*.yaml'))
        frozen.update(['docs/modal_l4_remaining_models_plan.md', 'tests/test_remaining_controls.py',
            'plans/internal-expansion/integrated-verification.json',
            'results/modal-l4-internal/controller/remaining-manifest.json',
            'results/modal-l4-internal/controller/remaining-calibration-decision.json',
            'results/modal-l4-internal/controller/internal-development-review.json'])
        result['frozen_inputs'] = {name: sha(controller.PROJECT_ROOT / name) for name in sorted(frozen)}
        archive_path = provenance / 'remaining-source-data-specs.tar.gz'
        if archive_path.exists():
            raise RuntimeError('Expansion archive already exists')
        with tarfile.open(archive_path, 'w:gz') as archive:
            for name in result['frozen_inputs']:
                archive.add(controller.PROJECT_ROOT / name, arcname=name)
        (provenance / 'remaining-sha256.json').write_text(json.dumps(result['frozen_inputs'], indent=2) + '\n')
        controller.verify_frozen_inputs(result)
        controller.save_state(path, result)
        (provenance / 'remaining-prepared-state.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='CPU gates and all 96 CLI dry runs; no cloud submission')
    parser.add_argument('--prepare', action='store_true', help='Append twelve native calibrations to the existing ledger')
    args = parser.parse_args()
    if args.verify == args.prepare:
        parser.error('Choose exactly one of --verify or --prepare')
    torch.set_num_threads(2)
    verify() if args.verify else prepare()
