"""Regression coverage for resume, concurrency and cloud budget boundaries."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest

path = Path(__file__).parents[1] / 'scripts/run_modal_l4_controller.py'
spec = importlib.util.spec_from_file_location('l4_controller', path)
c = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = c
spec.loader.exec_module(c)


@pytest.fixture
def state(tmp_path):
    evaluation = {'preset': 'standard', 'cells': [{'window': 10, 'horizon': 1}], 'seeds': [7], 'epochs': 10}
    execution = {'target': 'modal', 'gpu': 'L4'}
    specs = tmp_path / 'specs'
    specs.mkdir()
    candidates = c.create_candidate_list(tmp_path, evaluation, execution, specs)
    result = {'stage_name': 'pilot', 'results_root': str(tmp_path), 'stage': {'evaluation': evaluation, 'execution': execution},
        'candidates': candidates, 'rates': {'gpu_l4_per_hour': .8, 'cpu_core_per_hour': .0473, 'mem_gib_hour_cost': .008},
        'budget': {'total_usd': 20, 'safety_reserve_usd': 1, 'max_job_seconds': 3600, 'uncertainty_factor': 3,
                   'startup_cost_usd': .05, 'max_retries': 1, 'confirmed_spend_usd': 0},
        'billing': {'baseline_metered_usd': 2, 'baseline_billed_usd': 0}}
    c.migrate_state(result)
    return result


def job(state, name='job-1', status='complete', bounded=False):
    candidate = state['candidates'][0]
    architecture = c.load_model(Path(state['results_root']) / candidate['architecture_file'])
    execution = dict(state['stage']['execution'])
    if bounded:
        execution['timeout_seconds'] = 3600
    evaluation = c.normalize_evaluation({**state['stage']['evaluation'], 'device': 'cuda'})
    return {'id': name, 'request': {'phase': 'validation', 'architecture': architecture,
        'evaluation': evaluation, 'execution': execution,
        'candidate_hash': c.architecture_hash(architecture, {**evaluation, 'execution': execution})},
        'status': {'state': status, 'completed': 1, 'total': 1, 'created_at': '2026-09-10T20:00:00Z', 'updated_at': '2026-09-10T20:01:00Z'},
        'runs': [{'train_seconds': '1'}]}


def reconcile(state, jobs):
    return c.reconcile_candidate_state(state, jobs, c._utcnow(), state['rates'], state['stage']['evaluation'])


def test_recovers_old_hash_and_counts_duplicate_without_rerunning(state):
    candidate = state['candidates'][0]
    candidate.update(status='missing', attempts=1)
    jobs = [job(state), job(state, 'job-2')]
    reconcile(state, jobs)
    assert candidate['status'] == 'complete'
    assert candidate['job_id'] == 'job-1'
    assert candidate['attempts'] == 2
    assert c.choose_next_candidate(state)['id'] != candidate['id']
    c.update_cost_ledger(state, jobs)
    assert len(state['cost_ledger']) == 2
    cost = state['budget']['estimated_spend_usd']
    c.update_cost_ledger(state, jobs)
    assert state['budget']['estimated_spend_usd'] == cost
    assert cost > .1


def test_new_timeout_matches_same_scientific_candidate(state):
    assert job(state)['request']['candidate_hash'] != job(state, bounded=True)['request']['candidate_hash']
    reconcile(state, [job(state, bounded=True)])
    assert state['candidates'][0]['status'] == 'complete'


def test_lost_submission_blocks_instead_of_resubmitting(state):
    state['candidates'][0].update(status='submitting', attempts=1, reserved_usd=3)
    _, active = reconcile(state, [])
    assert active
    assert state['candidates'][0]['status'] == 'missing'
    assert state['candidates'][0]['reserved_usd'] == 3


def test_any_active_duplicate_occupies_slot_even_with_success(state):
    _, active = reconcile(state, [job(state), job(state, 'job-2', 'running')])
    assert active
    assert state['candidates'][0]['status'] == 'running'


def test_unknown_active_job_occupies_slot(state):
    other = job(state, status='running')
    other['request']['evaluation']['seeds'] = [99]
    assert reconcile(state, [other])[1]


def test_complete_without_artifacts_requires_review(state):
    broken = job(state)
    broken['runs'] = []
    reconcile(state, [broken])
    assert state['candidates'][0]['status'] == 'invalid_artifacts'


def test_budget_reserves_full_timeout_and_counts_delayed_billing(state):
    candidate = state['candidates'][0]
    reservation = c.expected_cost_for_candidate(state, candidate, 1)
    assert reservation > 2.8
    state['budget']['estimated_spend_usd'] = 18
    assert not c.can_submit_candidate(state, candidate, 1)
    state['budget']['estimated_spend_usd'] = 0
    assert c.can_submit_candidate(state, candidate, 1)


def test_retry_count_allows_one_retry_then_exhausts(state):
    candidate = state['candidates'][0]
    candidate.update(status='failed', attempts=1)
    assert c.choose_next_candidate(state) is candidate
    candidate['attempts'] = 2
    assert c.choose_next_candidate(state) is not candidate
    assert candidate['status'] == 'exhausted'


def test_stale_lock_pid_is_truncated_and_second_controller_rejected(tmp_path):
    lock = tmp_path / 'lock'
    lock.write_text('999999999999999999')
    with c.ensure_single_controller(lock):
        assert lock.read_text().strip() == str(c.os.getpid())
        with pytest.raises(RuntimeError, match='already active'):
            with c.ensure_single_controller(lock):
                pass
    assert lock.read_text() == ''


def test_stage_change_cannot_reset_budget(state, tmp_path):
    state_path = tmp_path / 'state.json'
    state['stage_name'] = 'confirmation'
    c.save_state(state_path, state)
    with pytest.raises(RuntimeError, match='preserve'):
        c.initialize_state(state_path, tmp_path, 'pilot')


def test_billing_failure_never_submits(state, tmp_path, monkeypatch):
    state_path = tmp_path / 'controller.json'
    c.save_state(state_path, state)
    monkeypatch.setattr(c, 'ensure_backend', lambda *args: {'url': 'http://test'})
    def request(url, path, payload=None):
        assert payload is None, 'unexpected cloud submission'
        return []
    monkeypatch.setattr(c, 'request', request)
    monkeypatch.setattr(c, 'load_modal_rates', lambda: state['rates'])
    def failed():
        raise RuntimeError('billing unavailable')
    monkeypatch.setattr(c, 'parse_monthly_summary', failed)
    with pytest.raises(RuntimeError, match='billing unavailable'):
        c.run_cycle(state_path, tmp_path, tmp_path/'lock', False, 1, None)
    saved = c.load_state(state_path)
    assert all(row['attempts'] == 0 for row in saved['candidates'])


def test_submission_response_lost_recovers_next_cycle(state, tmp_path, monkeypatch):
    state_path = tmp_path / 'controller.json'
    c.save_state(state_path, state)
    monkeypatch.setattr(c, 'ensure_backend', lambda *args: {'url': 'http://test'})
    monkeypatch.setattr(c, 'load_modal_rates', lambda: state['rates'])
    monkeypatch.setattr(c, 'parse_monthly_summary', lambda: {'metered_cost': 2, 'billed_cost': 0})
    monkeypatch.setattr(c, 'remote_apps_active', lambda: False)
    created = []
    def request(url, path, payload=None):
        if payload is None:
            return created
        assert payload['execution']['timeout_seconds'] == 3600
        created.append(job(state))
        raise TimeoutError('response lost')
    monkeypatch.setattr(c, 'request', request)
    with pytest.raises(TimeoutError):
        c.run_cycle(state_path, tmp_path, tmp_path/'lock', False, 1, None)
    saved = c.load_state(state_path)
    assert saved['candidates'][0]['status'] == 'submitting'
    assert saved['candidates'][0]['reserved_usd'] > 2.8
    c.run_cycle(state_path, tmp_path, tmp_path/'lock', False, 1, None, reconcile_only=True)
    saved = c.load_state(state_path)
    assert saved['candidates'][0]['status'] == 'complete'
    assert len(created) == 1


def test_stage_extension_preserves_costs_and_runs_every_control_at_both_rates(state):
    for row in state['candidates']:
        row['status'] = 'complete'
    state['budget']['confirmed_spend_usd'] = 2.5
    state['cost_ledger'] = {'prior': {'estimated_usd': 3}}
    c.prepare_controlled_development(state)
    assert state['budget']['confirmed_spend_usd'] == 2.5
    assert state['cost_ledger'] == {'prior': {'estimated_usd': 3}}
    assert len(state['candidates']) == 82
    assert c.choose_next_candidate(state)['id'] == 'controlled-preflight'
    c.prepare_controlled_development(state)
    assert len(state['candidates']) == 82
    development = [r for r in state['candidates'] if r['stage_name'] == 'controlled-development']
    assert len(development) == 54
    for variant in c.VARIANTS:
        assert sum(r['variant'] == variant.key for r in development) == 6
    assert all(c.candidate_evaluation(state, row)['epochs'] == 150 for row in development)
    assert all(row['depends_on'] == ['controlled-preflight'] for row in development)


def test_failed_preflight_blocks_development(state):
    for row in state['candidates']: row['status'] = 'complete'
    c.prepare_controlled_development(state)
    preflight = next(r for r in state['candidates'] if r['id'] == 'controlled-preflight')
    preflight.update(status='failed', attempts=2)
    assert c.choose_next_candidate(state) is None
    assert preflight['status'] == 'exhausted'


def test_changed_frozen_source_blocks_submissions(state, tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(c, 'PROJECT_ROOT', tmp_path)
    source = tmp_path / 'source.py'
    source.write_text('old')
    state['frozen_inputs'] = {'source.py': hashlib.sha256(source.read_bytes()).hexdigest()}
    c.verify_frozen_inputs(state)
    source.write_text('new')
    with pytest.raises(RuntimeError, match='Frozen experiment input changed'):
        c.verify_frozen_inputs(state)


def test_final_phase_does_not_reuse_validation_and_expects_one_fold(state):
    from copy import deepcopy
    validation=job(state)
    final_candidate=deepcopy(state['candidates'][0])
    final_candidate.update(id='final',phase='final_test',parent_job_id='job-1',max_retries=0)
    state['candidates'].append(final_candidate);c.migrate_state(state)
    final_job=deepcopy(validation);final_job['id']='final-job';final_job['request']['phase']='final_test'
    reconcile(state,[validation,final_job])
    assert state['candidates'][0]['job_id']=='job-1'
    assert final_candidate['job_id']=='final-job' and final_candidate['status']=='complete'
    assert final_candidate['candidate_hash'] != state['candidates'][0]['candidate_hash']


def test_final_submission_inherits_parent_and_limits_timeout(state, monkeypatch):
    candidate=state['candidates'][0]
    candidate.update(phase='final_test',parent_job_id='robust-parent',timeout_seconds=300,max_retries=0)
    calls=[]
    monkeypatch.setattr(c,'request',lambda url,path,payload: calls.append((path,payload)) or {'id':'final-job'})
    c.submit_candidate('http://test',state,candidate,2)
    assert calls==[('/api/v1/jobs/robust-parent/final-test',{'timeout_seconds':300})]
    candidate.update(status='failed',attempts=1)
    assert c.choose_next_candidate(state) is not candidate
    assert candidate['status']=='exhausted'
