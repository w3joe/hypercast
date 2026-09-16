#!/usr/bin/env python3
"""Prepare one balanced, cost-admitted two-rate stage for all remaining models."""
import argparse
from copy import deepcopy
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
from prepare_modal_l4_remaining import ROOT, validation_inputs
from hypercast4d.remaining_controls import BACKBONES, VARIANTS
from hypercast4d.remaining_batch import trial_grid
from hypercast4d.playground_runner import normalize_evaluation

STAGE = 'remaining-tuning'
EPOCHS = 25


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {**validation_inputs(), str(Path(__file__).resolve().relative_to(controller.PROJECT_ROOT)):sha(Path(__file__))}


def additions(state):
    review = json.loads((ROOT/'controller/remaining-calibration-review.json').read_text())
    assert review['passed'] and len(review['rows'])==12
    durations={r['backbone']:r['train_seconds'] for r in review['rows']}
    calibration={r['backbone']:r for r in state['candidates'] if r['stage_name']=='remaining-calibration'}
    assert set(calibration)==set(BACKBONES) and all(c['status']=='complete' for c in calibration.values())
    rows=[]
    for backbone in BACKBONES:
        previous=calibration[backbone]
        evaluation={**previous['evaluation'],'remaining_preflight':False,'remaining_tuning':True,
                    'seeds':[701],'epochs':EPOCHS,'learning_rate':.0003}
        evaluation=normalize_evaluation(evaluation)
        # 16 fits, epoch scaling and a 2x allowance for slower variants, then
        # 120 seconds for per-fit initialization, evaluation and artifact work.
        seconds=16*durations[backbone]*EPOCHS/5*2
        if not math.isfinite(seconds) or seconds<=0:raise RuntimeError('Invalid calibration duration')
        timeout=max(300,math.ceil((seconds+120)/30)*30)
        if timeout>3600:raise RuntimeError('Batch exceeds authorized maximum job duration')
        row=candidate(backbone,'all-eight-two-rates',previous['architecture_file'],evaluation,STAGE,
            timeout=timeout,depends=[rows[-1]['id']] if rows else [c['id'] for c in calibration.values()])
        rows.append(row)
    return rows


def verify():
    state=controller.load_state(ROOT/'controller/controller_state.json')
    rows=additions(state);hashes=source_hashes();reports=[]
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'evaluation.json'
        for row in rows:
            path.write_text(json.dumps(row['evaluation']))
            result=subprocess.run([str(Path(sys.executable).with_name('hypercast')),'experiment','run',
                str(ROOT/row['architecture_file']),'--evaluation',str(path),'--target','modal','--gpu','L4',
                '--device','cuda','--dry-run','--json'],text=True,capture_output=True,check=True)
            result=json.loads(result.stdout)
            assert result['trials']==16
            children=trial_grid(result['evaluation'])
            assert len(children)==16 and {x['variant'] for x in children}==set(VARIANTS)
            reports.append(dict(backbone=row['backbone'],dry_run=result,children=children))
            print('CLI verified:',row['backbone'],'16 labelled fits',flush=True)
    assert len({r['dry_run']['candidate_hash'] for r in reports})==12
    assert hashes==source_hashes()
    (ROOT/'controller/tuning-provenance/cli-verification.json').write_text(json.dumps(
        dict(passed=True,source_hashes=hashes,configurations=reports),indent=2)+'\n')


def prepare():
    directory=ROOT/'controller';path=directory/controller.STATE_FILENAME;provenance=directory/'tuning-provenance'
    with controller.ensure_single_controller(directory/'.controller.lock'):
        state=controller.load_state(path)
        if state['stage_name']!='remaining-calibration' or not all(c['status']=='complete' for c in state['candidates']):
            raise RuntimeError('All calibrations must complete; tuning may be appended only once')
        if controller.remote_apps_active():raise RuntimeError('Wait for current cloud activity')
        before=json.loads((provenance/'before-state.json').read_text())
        assert before['frozen_inputs']==state['frozen_inputs']
        with tarfile.open(provenance/'before-source.tar.gz') as archive:
            for name,expected in before['frozen_inputs'].items():
                assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==expected,name
        allowed={'src/hypercast4d/playground_runner.py','src/hypercast4d/modal_runner.py',
                 'src/hypercast4d/cli.py','scripts/run_modal_l4_controller.py'}
        changed={n for n,h in state['frozen_inputs'].items() if sha(controller.PROJECT_ROOT/n)!=h}
        if changed-allowed:raise RuntimeError(f'Unexpected changed input: {changed-allowed}')
        previous=state['previous_allocation'];assert sha(Path(previous['state_file']))==previous['sha256']
        verification=json.loads((provenance/'cli-verification.json').read_text())
        assert verification['passed'] and verification['source_hashes']==source_hashes()
        backend=controller.ensure_backend(controller.PROJECT_ROOT,ROOT)
        jobs=controller.request(backend['url'],'/api/v1/jobs')
        _,active=controller.reconcile_candidate_state(state,jobs,controller._utcnow(),state['rates'],state['stage']['evaluation'])
        if active:raise RuntimeError('Unresolved active job')
        controller.update_cost_ledger(state,jobs)
        state['rates']={k:max(state['rates'][k],v) for k,v in controller.load_modal_rates().items()}
        controller.refresh_billing(state,controller.parse_monthly_summary());controller.save_state(path,state)
        rows=additions(state)
        reservations=[controller.expected_cost_for_candidate(state,c,16) for c in rows]
        full=sum(reservations)
        headroom=state['budget']['total_usd']-state['budget']['safety_reserve_usd']-controller.accounted_spend(state)
        if full+1.5>headroom:raise RuntimeError(f'Complete tuning reservations ${full:.2f} plus margin exceed ${headroom:.2f}')
        decision=dict(stage=STAGE,prepared_at=controller._utcnow(),jobs=12,fits=192,epochs_max=EPOCHS,
            learning_rates=[.0003,.001],seeds=[701],backbones=list(BACKBONES),variants=list(VARIANTS),
            full_timeout_reservations_usd=round(full,6),additional_margin_usd=1.5,headroom_usd=headroom,
            accounted_spend_usd=controller.accounted_spend(state),changed_inputs=sorted(changed),
            per_backbone=[dict(backbone=c['backbone'],timeout_seconds=c['timeout_seconds'],reservation_usd=v) for c,v in zip(rows,reservations)],
            estimation='Five-epoch native timings x16 x5 x2 variant allowance, plus 120 seconds batch overhead; full timeout plus another 120 seconds admission overhead priced at 3x plus startup',
            supersedes='84 additional five-epoch fits; user prioritizes balanced hyperparameter tuning',
            limitations='25 epochs and one seed may undertrain models; exploratory development, no independent or robustness claim',
            next_action='Audit all 192 fits; report per-arm convergence and paired comparisons, cost any further stage before submitting')
        state.setdefault('stage_history',[]).append(dict(stage='remaining-calibration',completed_at=controller._utcnow(),
            completed=12,failed_attempts=1,accounted_spend_usd=controller.accounted_spend(state)))
        state.setdefault('stage_decisions',[]).append(decision)
        old_hashes=[c['candidate_hash'] for c in state['candidates']]
        state['candidates'].extend(rows);state.update(stage_name=STAGE,schedule_status='prepared')
        controller.migrate_state(state)
        assert old_hashes==[c['candidate_hash'] for c in state['candidates'][:len(old_hashes)]]
        assert len({c['candidate_hash'] for c in state['candidates']})==len(state['candidates'])
        decision_path=directory/'remaining-tuning-decision.json'
        if decision_path.exists():raise RuntimeError('Decision already exists; inspect before retry')
        decision_path.write_text(json.dumps(decision,indent=2)+'\n')
        names=set(state['frozen_inputs'])|set(source_hashes())
        names.update(['tests/test_remaining_batch.py','docs/modal_l4_remaining_tuning_stage.md',
            'results/modal-l4-internal/controller/remaining-calibration-review.json',
            'results/modal-l4-internal/controller/remaining-tuning-decision.json',
            'results/modal-l4-internal/controller/tuning-provenance/cli-verification.json'])
        state['frozen_inputs']={n:sha(controller.PROJECT_ROOT/n) for n in sorted(names)}
        target=provenance/'tuning-source.tar.gz'
        if target.exists():raise RuntimeError('Tuning archive already exists')
        with tarfile.open(target,'w:gz') as archive:
            for name in state['frozen_inputs']:archive.add(controller.PROJECT_ROOT/name,arcname=name)
        controller.verify_frozen_inputs(state);controller.save_state(path,state)
        (provenance/'prepared-state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(decision,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--verify',action='store_true');group.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    verify() if args.verify else prepare()
