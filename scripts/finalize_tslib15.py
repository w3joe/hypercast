"""Assemble the prespecified descriptive metrics and verified final report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

def finalize(run):
    report=json.loads((run/'analysis.json').read_text());audit=json.loads((run/'artifact-audit.json').read_text())
    ledger=json.loads((run/'ledger.json').read_text());results=json.loads((run/'test-results.json').read_text())
    assert report['complete'] and audit['passed'] and len(audit['rows'])==360 and not audit['errors']
    assert ledger['status']=='complete' and len(results)==360
    summaries=report['summaries'];lookup={(r['backbone'],r['arm']):r for r in summaries}
    for row in summaries:
        b,a=row['backbone'],row['arm'];real=lookup[(b,'real')];native=lookup[(b,'native')]
        fits=[r for r in results if (r['job']['backbone'],r['job']['arm'])==(b,a)]
        row.update(parameters_saved_vs_real_pct=100*(real['parameters']-row['parameters'])/real['parameters'],
            parameters_saved_vs_native_pct=100*(native['parameters']-row['parameters'])/native['parameters'],
            latency_ratio_vs_real=row['latency_batch32_ms']/real['latency_batch32_ms'],
            mae_over_persistence=row['mae']/report['persistence_mae'],mean_bias=float(np.mean([r['bias'] for r in fits])),
            mean_mse=float(np.mean([r['mse'] for r in fits])),mean_epochs=float(np.mean([r['epochs_ran'] for r in fits])),
            mean_best_epoch=float(np.mean([r['best_epoch'] for r in fits])))
    with (run/'full-summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    hyper=[r for r in summaries if r['arm'] in ('complex','quaternion','octonion')]
    operational=dict(artifact_audits_passed=360,exact_gpu_initialization_pairing=True,
        cpu_gpu_initialization_byte_mismatches=sum(not r['cpu_gpu_initialization_byte_equal'] for r in audit['rows']),
        successful_evaluation_fits=360,successful_calibration_fits=120,
        failed_attempts=[a['index'] for a in ledger['attempts'] if a['status']=='failed'],
        epoch_ceiling=100,early_stopping_completed=sum(r['patience_exhausted'] for r in results),
        ceiling_hits=sum(r['ceiling_reached'] for r in results),
        maximum_concurrency=10,cap_usd=80,conservative_upper_usd=ledger['conservative_upper_usd'],
        evaluation_worker_hours=sum(a['runtime']['elapsed_seconds'] for a in ledger['attempts'] if a['job']['phase']=='test')/3600,
        hyper_point_wins_vs_native=sum(r['improvement_vs_native_pct']>0 for r in hyper),
        hyper_point_wins_vs_persistence=sum(r['mae_over_persistence']<1 for r in hyper),
        report_generated_epoch=time.time(),report_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (run/'completion-summary.json').write_text(json.dumps(operational,indent=2))
    full=(run/'report.md').read_text()
    full+='\n## Final artifact audit\n\n'
    full+='All 360 returned archives passed independent local checks of archive/checkpoint hashes, finite model and optimizer state, exact best-state selection, strict checkpoint reconstruction, forecast alignment, persistence/seasonal baselines and recomputed MAE/RMSE. The actual GPU comparison arms have identical untouched initialization within each backbone/seed.\n\n'
    full+=f"Fresh CPU versus GPU-worker initialization hashes differed in {operational['cpu_gpu_initialization_byte_mismatches']} fits. This is recorded as a cross-platform initialization diagnostic; CPU byte-identical initialization is not claimed. Saved checkpoints reconstruct strictly, and same-device GPU prediction replay passed for every fit. The first stricter CPU audit and its revised scope are preserved.\n"
    full+='\n## Compression and latency\n\nThese are descriptive summaries across 15 selected sites. Positive savings mean fewer whole-model parameters. Latency above 1 means slower than controlled real.\n\n'
    full+='| Arm | Median parameters saved vs real | Median parameters saved vs native | Median batch-32 latency / real |\n|---|---:|---:|---:|\n'
    for a in ('complex','quaternion','octonion'):
        rr=[r for r in hyper if r['arm']==a]
        full+=f"| {a} | {np.median([r['parameters_saved_vs_real_pct'] for r in rr]):.2f}% | {np.median([r['parameters_saved_vs_native_pct'] for r in rr]):.2f}% | {np.median([r['latency_ratio_vs_real'] for r in rr]):.2f}× |\n"
    full+='\n## Scope and operations\n\n'
    full+='The calibrated fixed-rate fallback used Adam at 0.001, maximum 100 epochs, patience 20 and three fresh paired seeds. The two-rate tuning stage was omitted uniformly. All 15 backbones and all eight arms were retained. The seven-hour deadline and $80 allocation were unchanged.\n\n'
    full+=f"{operational['early_stopping_completed']} fits exhausted patience; {operational['ceiling_hits']} reached the epoch ceiling. Stopping rules do not prove global convergence. {report['adjusted_wins_all_block_lengths']} favourable primary contrasts retained simultaneous intervals above zero across every prespecified block-length sensitivity check.\n\n"
    full+=f"Conservative upper accounting: **${operational['conservative_upper_usd']:.2f} of $80**. This prices ten workers for entire app sessions, including idle time, plus margin and ancillary reserve; it is not the final provider invoice. Successful evaluation workers totalled {operational['evaluation_worker_hours']:.2f} worker-hours. Two failed calibration attempts remain in the ledger.\n\n"
    full+='The full CSV includes all 120 model/arm summaries, seed errors, MAE/MSE/RMSE, bias, convergence flags, parameter savings and latency ratios. Seed-level curves, predictions and best/latest checkpoints remain in the numbered archives.\n'
    (run/'full-report.md').write_text(full)
    from plot_tslib15 import plot
    plot(run)
    return operational

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--watch',action='store_true');a=p.parse_args()
    while a.watch:
        audit=a.run/'artifact-audit.json';analysis=a.run/'analysis.json';ledger=json.loads((a.run/'ledger.json').read_text())
        if ledger['status']=='complete' and analysis.exists() and audit.exists() and json.loads(audit.read_text()).get('passed'):break
        if ledger['status']=='stopped_for_review':raise RuntimeError('Training stopped; finalization requires review')
        time.sleep(20)
    print(json.dumps(finalize(a.run),indent=2))
