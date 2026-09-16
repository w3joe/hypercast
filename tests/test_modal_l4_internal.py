from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import prepare_modal_l4_internal as study


@pytest.fixture
def state(monkeypatch):
    evaluation = dict(learning_rate=.001, epochs=5, seeds=[907],
        initialization='internal-matched-v1', nested_stopping=True)
    rows = [study.candidate(b, v, 'unused.yaml', evaluation, 'internal-pilot')
            for b in study.BACKBONES for v in study.VARIANTS]
    for i, row in enumerate(rows):
        row.update(status='complete', job_id=f'job{i}')
    monkeypatch.setattr(study, 'read_runs', lambda *a: pd.DataFrame([dict(train_seconds=.2, mae_ratio=1.)]))
    monkeypatch.setattr(study.controller, 'migrate_state', lambda s: None)
    return dict(stage_name='internal-pilot', candidates=rows,
        budget=dict(total_usd=20., safety_reserve_usd=1., estimated_spend_usd=2.,
            confirmed_spend_usd=1., uncertainty_factor=3., startup_cost_usd=.05),
        rates=dict(gpu_l4_per_hour=.8, cpu_core_per_hour=.05, mem_gib_hour_cost=.006),
        cost_ledger={'all-attempts': {'estimated_usd': 2.}}, previous_allocation={'total_usd':20.})


def test_stage_extension_preserves_every_cost_and_all_controls(state):
    before = deepcopy(state)
    dev = study.extend(state)
    assert state == before
    assert len(dev['candidates']) == 72
    assert dev['cost_ledger'] == state['cost_ledger'] and dev['budget'] == state['budget']
    assert dev['previous_allocation'] == state['previous_allocation']
    for row in dev['candidates'][24:]:
        assert row['evaluation']['epochs'] == 150 and row['evaluation']['seeds'] == [701]
        row['status'] = 'complete'
    robust = study.extend(dev)
    assert len(robust['candidates']) == 96
    for row in robust['candidates'][72:]:
        ev = study.controller.normalize_evaluation(row['evaluation'])
        assert len(ev['folds']) == 3 and ev['seeds'] == [401, 503, 601]
        assert ev['learning_rate'] == .0003  # Lower rate wins exact tie.
    assert robust['cost_ledger'] == state['cost_ledger'] and robust['budget'] == state['budget']


@pytest.mark.parametrize('corruption', ['missing_arm', 'failed', 'budget'])
def test_invalid_or_unaffordable_stage_never_partly_mutates_state(state, corruption):
    if corruption == 'missing_arm':
        state['candidates'].pop()
    elif corruption == 'failed':
        state['candidates'][0]['status'] = 'failed'
    else:
        state['budget']['estimated_spend_usd'] = 18.
    before = deepcopy(state)
    with pytest.raises(RuntimeError):
        study.extend(state)
    assert state == before
