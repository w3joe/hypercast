import copy
import sys
from pathlib import Path

import pytest
from test_modal_l4_controller import state
from test_modal_l4_robustness import completed

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import prepare_modal_l4_robustness as robust
import prepare_modal_l4_final as final


def parents(state):
    selection=completed(state);s=robust.extend(state,selection);jobs=[]
    for i,row in enumerate(s['candidates']):
        if row.get('stage_name')!='controlled-robustness':continue
        row.update(status='complete',job_id=f'robust-{i}')
        jobs.append({'id':row['job_id'],'request':{'phase':'validation'},'status':{'state':'complete'},
            'runs':[{'epochs_ran':150,'train_seconds':35} for _ in range(6)]})
    s['schedule_status']='controlled-robustness_complete';s['updated_at']='2026-09-11T01:23:42Z'
    s['budget']['estimated_spend_usd']=14.72
    return s,jobs


def test_all_final_jobs_preserve_parents_costs_seeds_and_limits(state):
    s,jobs=parents(state);old=copy.deepcopy(s);result=final.extend(s,jobs)
    assert s==old and result['candidates'][:109]==old['candidates']
    assert result['budget']==old['budget'] and result['cost_ledger']==old['cost_ledger']
    rows=result['candidates'][109:];assert len(rows)==27
    for row in rows:
        parent=next(p for p in old['candidates'] if p['job_id']==row['parent_job_id'])
        assert row['evaluation']==parent['evaluation']
        assert row['timeout_seconds']==300 and row['max_retries']==0
        assert row['candidate_hash'].startswith('final-test:')
        assert row['candidate_hash'] != parent['candidate_hash']


def test_final_budget_reservation_and_incomplete_parents_block_stage(state):
    s,jobs=parents(state);s['budget']['estimated_spend_usd']=18
    with pytest.raises(RuntimeError,match='budget'):final.extend(s,jobs)
    s['budget']['estimated_spend_usd']=14.72;jobs[-1]['runs'].pop()
    with pytest.raises(RuntimeError,match='parent'):final.extend(s,jobs)
