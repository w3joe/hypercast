#!/usr/bin/env python3
"""Freeze fixed rates and append a costed 96-fit fresh-seed follow-up."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

import run_modal_l4_controller as controller
from prepare_modal_l4_internal import candidate
from prepare_modal_l4_remaining import ROOT
from hypercast4d.playground_runner import normalize_evaluation
from hypercast4d.remaining_controls import BACKBONES, VARIANTS
from hypercast4d.remaining_batch import trial_grid

STAGE = 'remaining-replication'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    directory=ROOT/'controller';path=directory/controller.STATE_FILENAME;provenance=directory/'replication-provenance'
    with controller.ensure_single_controller(directory/'.controller.lock'):
        state=controller.load_state(path)
        if state['stage_name']!='remaining-tuning' or not all(c['status']=='complete' for c in state['candidates']):
            raise RuntimeError('Complete tuning must precede this one-time follow-up')
        if controller.remote_apps_active():raise RuntimeError('Cloud activity must stop before versioning')
        before=json.loads((provenance/'before-state.json').read_text())
        assert state['frozen_inputs']==before['frozen_inputs']
        with tarfile.open(provenance/'before-source.tar.gz') as archive:
            for name,expected in before['frozen_inputs'].items():
                assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==expected,name
        changed={name for name,h in state['frozen_inputs'].items() if sha(controller.PROJECT_ROOT/name)!=h}
        allowed={'src/hypercast4d/playground_runner.py','src/hypercast4d/remaining_batch.py',
            'src/hypercast4d/modal_runner.py','src/hypercast4d/cli.py','scripts/run_modal_l4_controller.py',
            'tests/test_remaining_batch.py'}
        if changed-allowed:raise RuntimeError(f'Unexpected changed input: {changed-allowed}')
        previous=state['previous_allocation'];assert sha(Path(previous['state_file']))==previous['sha256']
        review=json.loads((directory/'remaining-tuning-review.json').read_text())
        assert review['passed'] and review['completed_fits']==192 and len(review['selected_development_rows'])==96
        selections=review['selected_development_rows']
        rows=[];cli=[];choices=[]
        with tempfile.TemporaryDirectory() as temporary:
            ev_path=Path(temporary)/'evaluation.json'
            for backbone in BACKBONES:
                selected=[r for r in selections if r['backbone']==backbone]
                assert {r['variant'] for r in selected}==set(VARIANTS) and len(selected)==8
                old=next(c for c in state['candidates'] if c['stage_name']=='remaining-tuning' and c['backbone']==backbone)
                rates={r['variant']:r['learning_rate'] for r in selected}
                # Recompute the frozen selection rule instead of trusting a summary alone.
                for row in selected:
                    options=[r for r in review['records'] if r['backbone']==backbone and r['variant']==row['variant']]
                    best=min(x['mae_ratio'] for x in options)
                    chosen=min((x for x in options if x['mae_ratio']<=best*1.001),key=lambda x:x['learning_rate'])
                    assert chosen['trial_id']==row['trial_id']
                evaluation={**old['evaluation'],'seeds':[401],'epochs':50,'remaining_replication':rates}
                evaluation.pop('remaining_tuning');evaluation=normalize_evaluation(evaluation)
                seconds=sum(r['train_seconds']/r['epochs_ran']*50 for r in selected)*1.2
                if not math.isfinite(seconds) or seconds<=0:raise RuntimeError('Invalid measured runtime')
                timeout=max(300,math.ceil((seconds+120)/30)*30)
                if timeout>3600:raise RuntimeError('Timeout exceeds authorized bound')
                row=candidate(backbone,'all-eight-seed401',old['architecture_file'],evaluation,STAGE,
                    timeout=timeout,depends=[rows[-1]['id']] if rows else [c['id'] for c in state['candidates'] if c['stage_name']=='remaining-tuning'])
                rows.append(row);choices.extend(selected)
                ev_path.write_text(json.dumps(evaluation))
                result=subprocess.run([str(Path(sys.executable).with_name('hypercast')),'experiment','run',
                    str(ROOT/old['architecture_file']),'--evaluation',str(ev_path),'--target','modal','--gpu','L4',
                    '--device','cuda','--dry-run','--json'],text=True,capture_output=True,check=True)
                payload=json.loads(result.stdout);assert payload['trials']==8
                children=trial_grid(payload['evaluation']);assert len(children)==8 and all(t['seed']==401 for t in children)
                cli.append(dict(backbone=backbone,dry_run=payload,children=children))
                print('CLI verified:',backbone,'eight fixed-rate fits',flush=True)
        assert len({r['dry_run']['candidate_hash'] for r in cli})==12
        backend=controller.ensure_backend(controller.PROJECT_ROOT,ROOT);jobs=controller.request(backend['url'],'/api/v1/jobs')
        _,active=controller.reconcile_candidate_state(state,jobs,controller._utcnow(),state['rates'],state['stage']['evaluation'])
        if active or controller.remote_apps_active():raise RuntimeError('Unexpected concurrent activity')
        controller.update_cost_ledger(state,jobs)
        state['rates']={k:max(state['rates'][k],v) for k,v in controller.load_modal_rates().items()}
        controller.refresh_billing(state,controller.parse_monthly_summary());controller.save_state(path,state)
        reservations=[controller.expected_cost_for_candidate(state,row,8) for row in rows]
        full=sum(reservations);headroom=state['budget']['total_usd']-state['budget']['safety_reserve_usd']-controller.accounted_spend(state)
        if full+.5>headroom:raise RuntimeError(f'Full reservations ${full:.2f} plus $.50 margin exceed ${headroom:.2f}')
        decision=dict(stage=STAGE,prepared_at=controller._utcnow(),jobs=12,fits=96,seeds=[401],epochs_max=50,
            accounted_spend_usd=controller.accounted_spend(state),headroom_usd=headroom,
            full_timeout_reservations_usd=round(full,6),additional_margin_usd=.5,
            rates_rule='Per-arm development MAE/persistence; .1% ties choose lower rate',
            reduction='Two fresh seeds x50 epochs required about $10.42 full reservations and cannot fit; reduce uniformly to one fresh seed while preserving all models and controls',
            cost_method='Measured selected-arm seconds per executed epoch x50 x1.2 seed/runtime allowance +120 batch seconds; round timeout upward; reserve timeout+120 at existing 3x rates and startup allowance',
            per_backbone=[dict(backbone=c['backbone'],timeout_seconds=c['timeout_seconds'],reservation_usd=v) for c,v in zip(rows,reservations)],
            prior_tuning_final_epoch_fits=91,changed_inputs=sorted(changed),
            analysis_plan='controller/replication-provenance/analysis-plan.json',
            limitations='One fresh seed, reused development dates and higher training ceiling; limited replication, no independent confirmation',
            next_action='Audit all 96 fits and run scripts/analyze_remaining_replication.py only after complete; inspect remaining funds and convergence before further work')
        for name,payload in [('fixed-rate-selections.json',choices),('cli-verification.json',cli),('decision.json',decision)]:
            destination=provenance/name
            if destination.exists():raise RuntimeError('Preparation artifact exists; inspect before retrying')
            destination.write_text(json.dumps(payload,indent=2)+'\n')
        old_hashes=[c['candidate_hash'] for c in state['candidates']]
        state['candidates'].extend(rows);state.setdefault('stage_history',[]).append(dict(stage='remaining-tuning',completed_fits=192,completed_at=controller._utcnow(),accounted_spend_usd=controller.accounted_spend(state)))
        state.setdefault('stage_decisions',[]).append(decision);state.update(stage_name=STAGE,schedule_status='prepared')
        controller.migrate_state(state);assert old_hashes==[c['candidate_hash'] for c in state['candidates'][:len(old_hashes)]]
        assert len({c['candidate_hash'] for c in state['candidates']})==len(state['candidates'])
        names=set(state['frozen_inputs'])
        names.update(str(p.relative_to(controller.PROJECT_ROOT)) for p in (controller.PROJECT_ROOT/'src/hypercast4d').rglob('*.py'))
        names.update(['scripts/prepare_modal_l4_replication.py','scripts/analyze_remaining_replication.py',
            'tests/test_remaining_replication_analysis.py','docs/modal_l4_remaining_replication_stage.md',
            'results/modal-l4-internal/controller/remaining-tuning-review.json'])
        names.update(str((provenance/n).relative_to(controller.PROJECT_ROOT)) for n in ['analysis-plan.json','fixed-rate-selections.json','cli-verification.json','decision.json'])
        state['frozen_inputs']={n:sha(controller.PROJECT_ROOT/n) for n in sorted(names)}
        target=provenance/'replication-source.tar.gz'
        if target.exists():raise RuntimeError('Replication source archive exists')
        with tarfile.open(target,'w:gz') as archive:
            for name in state['frozen_inputs']:archive.add(controller.PROJECT_ROOT/name,arcname=name)
        controller.verify_frozen_inputs(state);controller.save_state(path,state)
        (provenance/'prepared-state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(decision,indent=2))


if __name__=='__main__':main()
