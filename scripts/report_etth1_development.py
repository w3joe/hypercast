"""Render the audited native development study without accessing final-test data."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def report(allocation):
    s=json.loads((allocation/'analysis.json').read_text())
    if not s['complete'] or len(s['audits'])!=12 or not all(a['passed'] for a in s['audits']):
        raise ValueError('All twelve audited fits are required')
    cpu=json.loads((allocation/'cpu-replay.json').read_text());assert cpu['completed_fits']==12
    rows=s['rows']; p=s['baselines']['persistence_mae']; seasonal=s['baselines']['seasonal_24_mae']
    lines=['# ETTh1 native development results','',
        f"Relative/residual improved all six paired comparisons. Median paired development-MAE reductions were {s['model_summary']['micn']['median_paired_mae_reduction_percent']:.2f}% for MICN and {s['model_summary']['film']['median_paired_mae_reduction_percent']:.2f}% for FiLM.", '',
        'All twelve authorized native fits completed on Modal L4s in waves of two. This stage compares levels/direct prediction with the relative-input, persistence-residual package; it does not replace internal layers with HyperDense.', '',
        'The frozen task uses all seven ETTh1 channels, OT as target, a 32-hour context and five-hour horizon. Training and inner checkpoint selection precede the 2,876 development origins from 26 June through 23 October 2017. These dates are now exposed development data. Final-test tensors were absent from the worker bundle and no final-test scores were computed.', '',
        '## Forecasting results','',
        f'Raw-unit development MAE: persistence **{p:.6f}**; 24-hour seasonal persistence **{seasonal:.6f}**. Positive reductions mean improvement. Medians summarize three paired seeds, not independent forecast observations.','',
        '| Backbone | Formulation | Median MAE | MAE range | Median reduction vs persistence | Parameters |',
        '|---|---|---:|---:|---:|---:|']
    for b in ['micn','film']:
        for mode in ['levels_direct','relative_residual']:
            rs=[r for r in rows if r['backbone']==b and r['mode']==mode];mae=[r['mae'] for r in rs]
            lines.append(f"| {b.upper() if b=='micn' else 'FiLM'} | {mode} | {np.median(mae):.6f} | {min(mae):.6f}–{max(mae):.6f} | {100*(1-np.median(mae)/p):.2f}% | {rs[0]['parameters']:,} |")
    lines+=['','| Backbone | Seed | Relative MAE reduction vs direct | Relative MSE reduction vs direct | Inner MAE reduction vs direct | MAE reductions across chronological thirds |','|---|---:|---:|---:|---:|---|']
    for r in s['paired']:
        thirds=', '.join(f'{x:.2f}%' for x in r['third_mae_reduction_percent'])
        lines.append(f"| {r['backbone']} | {r['seed']} | {r['mae_reduction_percent']:.2f}% | {r['mse_reduction_percent']:.2f}% | {r['inner_mae_reduction_percent']:.2f}% | {thirds} |")
    lines+=['','The chronological thirds are descriptive post-hoc checks. Forecast windows and leads overlap; they are not independent samples, and no significance claim is made. This comparison changes relative inputs, residual prediction and head initialization together, so it cannot attribute an effect to any one component.','',
        '## Selection and convergence','',
        'All fits use Adam at 0.001, batch 32, MAE training, strict inner-MAE checkpoint selection, a 150-epoch ceiling and patience 20. Development scores were computed after selection and did not affect the training run.', '',
        '| Backbone | Seed | Formulation | Best epoch | Epochs run | Stale epochs | Worker seconds |',
        '|---|---:|---|---:|---:|---:|---:|']
    ledger=json.loads((allocation/'ledger.json').read_text())
    for r,a in zip(rows,ledger['attempts']):
        lines.append(f"| {r['backbone']} | {r['seed']} | {r['mode']} | {r['best_epoch']} | {r['epochs_ran']} | {r['stale_epochs']} | {a['runtime']['elapsed_seconds']:.1f} |")
    lines+=['',f"Fits not exhausting patience: **{len(s['convergence_flags'])}**. Early stopping is evidence of the specified stopping rule being met, not proof of global optimization convergence. Worker times include training, per-epoch checkpoint writes and evaluation; they are not inference benchmarks.",'',
        '## Controls, artifacts and budget','',
        f"The return audit passed all twelve archives and {sum(a['predictions'] for a in s['audits']):,} saved forecast values. It checks frozen targets, origins and both baselines; recomputes MAE/MSE/bias and lead scores; verifies checkpoint hashes and metadata; and checks selected weights against the best state stored in the latest resumable checkpoint. Each worker also reloaded its selected checkpoint and reproduced inner MAE within tolerance.",'',
        'Every fit archives best inference weights plus latest model, Adam, Python/NumPy/Torch/CUDA RNG, sampler, best-state and training-history state. Exact CPU resume was tested; exact CUDA continuation has not been established. Historical frozen inputs and the current source archives passed preservation checks. The eleven focused wrapper/training tests passed before launch.','',
        f"The new **$19 allocation conservatively accounts for ${s['conservative_accounted_usd']:.6f}**, retaining timeout-based reservations even when fits finish early. This is not the final invoice. The $1 protected reserve is not spent; prior allocations remain closed. There were no failed attempts or retries. No larger paid stage or final-test evaluation was launched.",'',
        '- [Audited results, per-lead metrics and archive hashes](../results/etth1-transfer/development-001/analysis.json)',
        '- [Full CPU checkpoint replay](../results/etth1-transfer/development-001/cpu-replay.json)',
        '- [Allocation ledger](../results/etth1-transfer/development-001/ledger.json)',
        '- [Frozen stage plan](../results/etth1-transfer/development-prepared-v1/plan.json)',
        '- [Source preservation audit](../results/etth1-transfer/development-001/source-preservation-audit.json)',
        '- [Calibration and launch protocol](etth1_calibration_results_and_development_plan.md)','']
    leads=['## Horizon detail','',
        'Relative/residual MAE reduction versus persistence by forecast lead. Negative values mean the model is worse. Aggregate improvements do not imply superiority at every lead.','',
        '| Backbone | Seed | +1 hour | +2 hours | +3 hours | +4 hours | +5 hours |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['mode']=='relative_residual':
            values=' | '.join(f'{x:.2f}%' for x in r['per_lead_mae_reduction_vs_persistence_percent'])
            leads.append(f"| {r['backbone']} | {r['seed']} | {values} |")
    leads+=['']
    lines[lines.index('## Selection and convergence'):lines.index('## Selection and convergence')]=leads
    failures=[r for r in cpu['rows'] if not r['strict_tolerance_passed']]
    portability=(f"A separate post-hoc CPU replay covered all development forecasts from all twelve checkpoints. "
        f"{len(failures)} fits failed the fixed pointwise tolerance (rtol 2e-4, atol 2e-5); "
        f"maximum forecast difference was {max(r['maximum_absolute_prediction_difference'] for r in cpu['rows']):.6f} raw units, "
        f"and maximum absolute aggregate MAE change was {max(abs(r['mae_difference']) for r in cpu['rows']):.8f}. "
        "The fixed tolerance was not widened. CPU versus CUDA numerical execution may explain the differences, but the exact source was not isolated. "
        "The CPU paired rankings are recorded alongside the L4 rankings; L4 remains the frozen primary evaluation. "
        "This limits claims of cross-device numerical equivalence and does not replace the successful same-device selected-checkpoint replay.")
    lines[lines.index('## Controls, artifacts and budget'):lines.index('## Controls, artifacts and budget')]=[
        '## CPU portability diagnostic','',portability,'']
    meets=all(s['model_summary'][b]['relative_wins']>=2 and s['model_summary'][b]['median_paired_mae_reduction_percent']>0 for b in ['micn','film']) and not s['convergence_flags']
    decision=('The package meets the proposed representation advancement rule on this development dataset: both models improve in at least two of three paired seeds and all fits complete the stopping checks. Freeze relative/residual for subsequent development comparisons; retain both models and all eight arms.' if meets else
        'The representation advancement rule is not cleanly satisfied for both models. Retain the disagreements and resolve them before claiming a common formulation or launching the full grid.')
    lines+=['## Interpretation and next scope','',decision,'',
        'This is evidence about native-model representation, not hypercomplex algebra. MICN inner selection scores are mixed despite consistent development improvement. ETTh1 is one public dataset on a custom short-horizon task, and additional seeds on these dates would remain development evidence.','',
        'The next implementation needs native, controlled real, 2D/4D/8D and exactly budget-matched rank controls on the seven-channel wrapper; equal AdamW learning-rate/weight-decay/selected-operator-initialization settings; untouched-weight pairing; forward/gradient and reconstruction checks; and explicit compute-precision recording. The current Adam-only development runner does not implement that grid. Keep inference-only optimizations separate.','',
        'An analytic site inventory found MICN 8D would save only 1.0843% of whole-model parameters versus native. FiLM 4D would save 16.6636%, and FiLM 8D 24.99546% versus its already-complex native model. Thus none of these single-site replacements can meet a literal 25% native-relative compression threshold. Do not round 24.99546% into a threshold pass. Savings against expanded controlled-real FiLM are larger and must be identified as a different baseline. Retain the original threshold for that claim or prospectively define a different question before any new scored comparison. These are analytic parameter counts, not tested seven-channel HyperDense models.','',
        'Recalibrate all arms before proposing a new full-stage budget: native timing cannot price spectral HyperDense or low-rank execution reliably. The former $48 estimate is not valid for this workload. The final-test period remains untouched until model/settings, contrasts and analysis are frozen; no paid tuning or final-test launch follows automatically.','',
        '- [Next control sites and analytic parameter budgets](../results/etth1-transfer/development-001/next-control-site-inventory.json)','']
    import os
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/hypercast4d-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharey=True)
    for ax,b,label in zip(axes,['micn','film'],['MICN','FiLM']):
        for seed,color in zip([2201,2202,2203],['#0072B2','#D55E00','#009E73']):
            pair={r['mode']:r for r in rows if r['backbone']==b and r['seed']==seed}
            ax.plot([0,1],[pair[m]['mae_over_persistence'] for m in ['levels_direct','relative_residual']],
                    'o-',label=str(seed),color=color,linewidth=1.6)
        ax.axhline(1,color='black',linestyle='--',linewidth=1,label='Persistence')
        ax.set_xticks([0,1],['Levels/direct','Relative/residual']);ax.set_title(label)
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Development MAE / persistence MAE')
    axes[1].legend(title='Paired seed',fontsize=8)
    fig.suptitle('ETTh1 native representation pilot — lower is better')
    fig.tight_layout()
    figure=ROOT/'docs/figures/etth1-native-development';figure.parent.mkdir(exist_ok=True)
    fig.savefig(figure.with_suffix('.png'),dpi=180);fig.savefig(figure.with_suffix('.pdf'));plt.close(fig)
    lines[lines.index('## Selection and convergence'):lines.index('## Selection and convergence')]=[
        '![Paired development MAE relative to persistence](figures/etth1-native-development.png)', '']
    path=ROOT/'docs/etth1_native_development_results.md';path.write_text('\n'.join(lines));print(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--allocation',type=Path,default=ROOT/'results/etth1-transfer/development-001');report(p.parse_args().allocation)
