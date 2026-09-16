"""Stage admission must preserve costs, controls and the selected parent jobs."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest
from test_modal_l4_controller import state

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import prepare_modal_l4_robustness as robust


def completed(state):
    for row in state['candidates']:
        row['status'] = 'complete'
    robust.controller.prepare_controlled_development(state)
    for i, row in enumerate(state['candidates']):
        row.update(status='complete', job_id=f'job-{i}')
    state['schedule_status'] = 'controlled-development_complete'
    state['cost_ledger'] = {'prior-duplicate': {'estimated_usd': 8.}}
    state['budget']['estimated_spend_usd'] = 8.
    rows = [r for r in state['candidates'] if r.get('stage_name') == 'controlled-development'
            and r['evaluation']['learning_rate'] == .0003]
    selection = {'complete': True, 'selected': [dict(backbone=r['backbone'], variant=r['variant'],
        job_id=r['job_id'], learning_rate=.0003, train_seconds=40.) for r in rows]}
    return selection


def test_balanced_append_keeps_history_and_six_fits_per_job(state):
    selection = completed(state); before = copy.deepcopy(state)
    result = robust.extend(state, selection)
    assert state == before
    assert result['candidates'][:82] == before['candidates']
    assert result['cost_ledger'] == before['cost_ledger']
    assert result['budget'] == before['budget']
    rows = result['candidates'][82:]
    assert len(rows) == 27
    assert len({r['candidate_hash'] for r in rows}) == 27
    assert not {r['candidate_hash'] for r in rows} & {r['candidate_hash'] for r in before['candidates']}
    for row in rows:
        ev = robust.controller.normalize_evaluation(row['evaluation'])
        assert robust.controller.estimate_trials(ev) == 6
        assert ev['seeds'] == [101, 211] and ev['epochs'] == 150
        assert row['timeout_seconds'] == 1200
        assert robust.controller.can_submit_candidate(result, row, 6)
    with pytest.raises(RuntimeError, match='already prepared'):
        robust.extend(result, selection)


@pytest.mark.parametrize('corruption', ['incomplete', 'missing_control', 'wrong_parent', 'budget'])
def test_unsafe_extension_rejected_without_mutating_history(state, corruption):
    selection = completed(state)
    if corruption == 'incomplete': state['candidates'][-1]['status'] = 'running'
    if corruption == 'missing_control': selection['selected'].pop()
    if corruption == 'wrong_parent': selection['selected'][0]['job_id'] = 'unrelated'
    if corruption == 'budget': state['budget']['estimated_spend_usd'] = 18.
    before = copy.deepcopy(state)
    with pytest.raises(RuntimeError): robust.extend(state, selection)
    assert state == before
