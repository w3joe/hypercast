#!/usr/bin/env python3
"""Prepare all locked final comparisons without resetting programme accounting."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import run_modal_l4_controller as c

VERSIONED = {'src/hypercast4d/playground.py', 'src/hypercast4d/cli.py',
             'src/hypercast4d/playground_runner.py', 'scripts/run_modal_l4_controller.py'}


def extend(state, jobs):
    if state.get('schedule_status') != 'controlled-robustness_complete' or any(
            x['status'] != 'complete' for x in state['candidates']):
        raise RuntimeError('Complete and reconcile robustness before preparing the final stage')
    parents = [x for x in state['candidates'] if x.get('stage_name') == 'controlled-robustness']
    if len(parents) != 27 or {(x['backbone'], x['variant']) for x in parents} != {
            (b,v.key) for b in c.BACKBONES for v in c.VARIANTS}:
        raise RuntimeError('Require every family and real control')
    by_id = {j['id']:j for j in jobs}; additions=[]; forecast=0.
    available = state['budget']['total_usd']-state['budget']['safety_reserve_usd']-c.accounted_spend(state)
    for parent in parents:
        job=by_id[parent['job_id']]
        if job['request']['phase'] != 'validation' or job['status']['state'] != 'complete' or len(job['runs']) != 6:
            raise RuntimeError('Invalid robustness parent')
        if not all(int(r['epochs_ran']) == 150 for r in job['runs']):
            raise RuntimeError('Every final fit must inherit 150 epochs')
        evaluation=deepcopy(parent['evaluation'])
        if evaluation['seeds'] != [101,211] or evaluation['initialization'] != 'matched-v1':
            raise RuntimeError('Frozen seed/initialization mismatch')
        seconds=sum(float(r['train_seconds']) for r in job['runs'])/3*(.85/.65)
        expected=c.estimate_job_cost_usd(1,state['rates'],seconds+30,state['budget']['uncertainty_factor'],state['budget']['startup_cost_usd'])
        row={k:parent[k] for k in ['backbone','variant','variant_name','architecture_file']}
        row.update(id=parent['id']+'-final',stage_name='locked-final',phase='final_test',parent_job_id=parent['job_id'],
            depends_on=[parent['id']],evaluation=evaluation,timeout_seconds=300,max_retries=0,status='pending',
            job_id=None,attempts=0,reserved_usd=0.,estimated_cost_usd=0.,started_at=None,finished_at=None,
            status_message='',forecast_cost_usd=expected)
        reserve=c.expected_cost_for_candidate(state,row,2)
        if forecast+reserve > available or forecast+expected > available:
            raise RuntimeError('Forecast cannot safely admit every final job within the existing budget')
        forecast+=expected; additions.append(row)
    result=deepcopy(state);result['candidates'].extend(additions)
    result.setdefault('stage_history',[]).append({'stage':'controlled-robustness','completed':27,'fits':162,
        'finished_at':state['updated_at'],'accounted_spend_usd':c.accounted_spend(state)})
    result.update(stage_name='locked-final',schedule_status='prepared',final_plan={'jobs':27,'fits':54,
        'estimated_cost_usd':forecast,'headroom_at_preparation_usd':available,'timeout_seconds':300})
    c.migrate_state(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results-root',type=Path,default=c.DEFAULT_RESULTS_ROOT)
    args=parser.parse_args();root=args.results_root.resolve();directory=root/'controller'
    with c.ensure_single_controller(directory/'.controller.lock'):
        state=c.load_state(directory/c.STATE_FILENAME)
        changed={p for p,h in state['frozen_inputs'].items() if hashlib.sha256((c.PROJECT_ROOT/p).read_bytes()).hexdigest()!=h}
        if changed != VERSIONED: raise RuntimeError(f'Unexpected source changes: {changed}')
        if c.remote_apps_active():raise RuntimeError('Cloud work must finish before preparation')
        backend=c.ensure_backend(c.PROJECT_ROOT,root);jobs=c.request(backend['url'],'/api/v1/jobs')
        c.migrate_state(state)
        _,active=c.reconcile_candidate_state(state,jobs,c._utcnow(),state['rates'],state['stage']['evaluation'])
        if active:raise RuntimeError('Active or missing jobs prevent preparation')
        if any(j['request']['phase']=='final_test' for j in jobs):raise RuntimeError('This programme already has final-test exposure')
        c.update_cost_ledger(state,jobs)
        rates=c.load_modal_rates();state['rates']={k:max(state['rates'][k],v) for k,v in rates.items()}
        c.refresh_billing(state,c.parse_monthly_summary())
        analysis=json.loads((directory/'robustness-analysis/analysis.json').read_text())
        if not analysis['complete'] or analysis['fits']!=162:raise RuntimeError('Complete robustness analysis required')
        prepared=extend(state,jobs)
        provenance=directory/'final-provenance';provenance.mkdir(exist_ok=False)
        shutil.copy2(directory/c.STATE_FILENAME,provenance/'previous-state.json')
        (provenance/'previous-sha256.json').write_text(json.dumps(state['frozen_inputs'],indent=2)+'\n')
        tracked=set(state['frozen_inputs']) | {'scripts/prepare_modal_l4_final.py','scripts/analyze_modal_l4_robustness.py',
            'docs/modal_l4_final_stage.md',str((directory/'controlled-provenance/test-exposure-audit.json').relative_to(c.PROJECT_ROOT))}
        tracked.update(str(p.relative_to(c.PROJECT_ROOT)) for p in (directory/'robustness-analysis').iterdir() if p.is_file())
        prepared['frozen_inputs']={p:hashlib.sha256((c.PROJECT_ROOT/p).read_bytes()).hexdigest() for p in sorted(tracked)}
        with tarfile.open(provenance/'source-and-analysis.tar.gz','w:gz') as archive:
            for relative in sorted(tracked):archive.add(c.PROJECT_ROOT/relative,arcname=relative)
        (provenance/'sha256.json').write_text(json.dumps(prepared['frozen_inputs'],indent=2)+'\n')
        prepared['final_provenance']=str(provenance);prepared['updated_at']=c._utcnow()
        (provenance/'prepared-manifest.json').write_text(json.dumps(prepared,indent=2)+'\n')
        c.save_state(directory/c.STATE_FILENAME,prepared)
        print(json.dumps(prepared['final_plan'],indent=2))


if __name__=='__main__':main()
