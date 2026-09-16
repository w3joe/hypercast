"""Freeze calibrated native stage; never submits or modifies old ledgers."""
import json
from pathlib import Path
import tarfile
from hypercast4d.representation_pilot import sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    bundle=ROOT/'results/etth1-transfer/prepared-v3'
    manifest=json.loads((bundle/'manifest.json').read_text())
    audit_path=ROOT/'results/etth1-transfer/calibration-001/audit-and-cost-proposal.json'
    audit=json.loads(audit_path.read_text())
    assert audit['calibration_passed'] and audit['checkpoint_audit_passed']
    for name,digest in manifest['source_code_sha256'].items():assert sha(ROOT/name)==digest,name
    expected={(b,m,s) for b in ['micn','film'] for m in ['levels_direct','relative_residual'] for s in [2201,2202,2203]}
    jobs=audit['jobs'];assert len(jobs)==12 and {(r['backbone'],r['mode'],r['seed']) for r in jobs}==expected
    files=list((ROOT/'src/hypercast4d').rglob('*.py'))+[ROOT/'scripts'/n for n in ['run_etth1_development.py','prepare_etth1_development.py','launch_representation_pilot.py']]+[ROOT/'tests/test_transfer_training.py',ROOT/'tests/test_native_transfer.py']
    out=ROOT/'results/etth1-transfer/development-prepared-v1';out.mkdir(exist_ok=False)
    plan=dict(protocol='etth1-native-development-v1',status='prepared_not_authorized',jobs=jobs,
        dataset_manifest_sha256=sha(bundle/'manifest.json'),dataset_bundle_sha256=sha(bundle/'development.npz'),
        calibration_audit_sha256=sha(audit_path),concurrent_l4=2,retries=0,
        primary_score='Development MAE, after inner-MAE checkpoint selection',
        learning_rate=.001,optimizer='Adam',betas=[.9,.999],epsilon=1e-7,weight_decay=0,
        batch_size=32,epochs=150,patience=20,minimum_improvement=0,
        target='OT',channels=7,context=32,horizon=5,
        baselines=['persistence','seasonal_24_hour'],test_scoring=False,
        checkpoint_policy='Atomic latest model/Adam/RNG/sampler/best-state checkpoint each completed epoch; selected best inference artifact after stopping. No automatic resume or retries.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        proposed_new_cap_usd=audit['suggested_next_stage_cap_usd'],full_reservations_usd=audit['next_stage_reservations_usd'],protected_reserve_usd=1,
        review_rule='Retain both models and every paired seed. Review convergence and paired development effects against direct, persistence and seasonal baselines before any representation choice or further stage. Report disagreements; no automatic final-test or HyperDense tuning launch.',
        limitation='Costs estimated from three calibration epochs with conservative margin. CUDA exact-resume equivalence not yet tested; latest state supports future audited recovery, not an automatic continuation.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with tarfile.open(out/'source.tar.gz','w:gz') as archive:
        for path in files:archive.add(path,arcname=str(path.relative_to(ROOT)))
    print(json.dumps(dict(status=plan['status'],fits=len(jobs),cap_usd=plan['proposed_new_cap_usd'],reservations_usd=plan['full_reservations_usd']),indent=2))


if __name__=='__main__':main()
