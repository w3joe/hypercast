"""Report local follow-up evidence and cost a conditional future L4 study."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hypercast4d.data import load_paper_data

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/hyperdense-followup-local'


def main():
    perf=json.loads((OUT/'inference.json').read_text())
    diag=json.loads((OUT/'persistence-diagnostics.json').read_text())
    prior=json.loads((ROOT/'results/modal-l4-internal/controller/remaining-replication-analysis.json').read_text())
    rates=json.loads((OUT/'current-rates.json').read_text())
    hourly=rates['rates']['gpu_l4_per_hour']+2*rates['rates']['cpu_core_per_hour']+4*rates['rates']['mem_gib_hour_cost']
    selected={(r['backbone'],r['variant']):r for r in prior['scores'] if r['backbone'] in ['micn','film']}
    variants=['native','real','complex','quaternion','octonion','lowrank2','lowrank4','lowrank8']

    def reservation(timeout):return (timeout+120)*hourly/3600*3+.05

    def pack(backbone, arms, multiplier):
        batches=[]; current=[]; elapsed=0.
        for arm in arms:
            r=selected[backbone,arm]
            seconds=r['train_seconds']/r['epochs_ran']*100*multiplier*1.5
            if seconds+120>3000:raise ValueError('One estimated fit exceeds batch runtime target; revise design before launch')
            if elapsed+seconds+120>3000 and current:
                batches.append((current,elapsed));current=[];elapsed=0.
            current.append(arm);elapsed+=seconds
        if current:batches.append((current,elapsed))
        return [dict(backbone=backbone,arms=group,fits=len(group),timeout_seconds=max(300,math.ceil((seconds+120)/30)*30),
                     reservation_usd=reservation(max(300,math.ceil((seconds+120)/30)*30))) for group,seconds in batches]

    scenarios=[]
    for workload in [1,4]:
        pilot=[];tuning=[];confirmation=[]
        for b in ['micn','film']:
            for seed in range(3):pilot.extend(pack(b,['native']*2,workload))
            for config in range(8):tuning.extend(pack(b,variants,workload))
            for seed in range(5):confirmation.extend(pack(b,variants,workload))
        stagecost=lambda jobs:round(sum(j['reservation_usd'] for j in jobs),2)
        total=reservation(600)+stagecost(pilot)+stagecost(tuning)+stagecost(confirmation)
        scenarios.append(dict(workload_multiplier=workload,inference_check_reservation_usd=round(reservation(600),2),
            representation_pilot=dict(fits=12,jobs=pilot,reservation_usd=stagecost(pilot)),
            equal_tuning=dict(fits=128,jobs=tuning,reservation_usd=stagecost(tuning)),
            confirmation=dict(fits=80,jobs=confirmation,reservation_usd=stagecost(confirmation)),
            reservations_usd=round(total,2),suggested_cap_with_20_percent_margin_and_1_dollar_reserve=math.ceil(total*1.2+1)))
    cost=dict(status='proposal_only_not_submitted',rates=rates,resources=dict(gpu='L4',concurrent_gpus=1,cpu_cores=2,memory_gib=4),
        assumptions='100 epochs per fit; historical selected-arm training seconds/epoch; 1.5x runtime allowance; 120 seconds batch overhead; timeouts rounded to 30 seconds, min 300, packing target max 3000; reserve timeout+120 at 3x resource rates plus $0.05/job. Two models, one dataset per scenario. New data, representation and optimizer settings may change runtime. Recalibrate before admission.',scenarios=scenarios)
    (OUT/'confirmation-cost-proposal.json').write_text(json.dumps(cost,indent=2)+'\n')

    names={'patchtst':'PatchTST','timesnet':'TimesNet','crossformer':'Crossformer','frets':'FreTS','lightts':'LightTS',
           'micn':'MICN','msgnet':'MSGNet','scinet':'SCINet','segrnn':'SegRNN','timemixer':'TimeMixer','timexer':'TimeXer','film':'FiLM'}
    frame=load_paper_data(ROOT/'data/raw/paper_data.xlsx','Copper').iloc[:1706]
    fig,axes=plt.subplots(2,1,figsize=(10,8.5),layout='constrained')
    ax=axes[0];ax.plot(frame.index,frame.Copper,color='#244a70',linewidth=1)
    for label,start,end,color in [('Training',0,1194,'#dce8ef'),('Inner validation',1194,1405,'#f6edc5'),('Outer evaluation',1405,1706,'#f8d9cc')]:
        ax.axvspan(frame.index[start],frame.index[end-1],color=color,alpha=.6,label=label)
    ax.axhline(diag['partitions']['train']['target_max'],linestyle='--',color='#9b2d30',label='Training maximum')
    ax.set_title('Copper moved well beyond the training price range',loc='left');ax.set_ylabel('Copper value (source units)');ax.legend(fontsize=8,ncol=4)
    native=[r for r in diag['arms'] if r['variant']=='native']
    ax=axes[1];positions=np.arange(len(native));width=.36
    for offset,regime,label,color in [(-width/2,'origin_inside_train_range','Origin inside training range','#376b8d'),(width/2,'origin_above_train_max','Origin above training maximum','#d47a50')]:
        ax.bar(positions+offset,[r['by_origin_regime'][regime]['mae_ratio'] for r in native],width,label=label,color=color)
    ax.axhline(1,color='#555555',linestyle='--');ax.set_xticks(positions,[names[r['backbone']] for r in native],rotation=30,ha='right')
    ax.set_ylabel('MAE / persistence MAE');ax.set_title('Native-model errors increase in the higher-price period',loc='left');ax.legend(fontsize=9)
    fig.supxlabel('Post-hoc descriptive groups, not randomized conditions. Price range and time period are confounded.',fontsize=9)
    dest=ROOT/'docs/figures/hyperdense-followup-diagnostics.png';fig.savefig(dest,dpi=180);plt.close(fig)

    lines=['# HyperDense follow-up: implementation and persistence diagnostics','',
        'Completed locally on 11 September 2026. No Modal inference or training jobs were launched. The completed studies, their frozen source files, budget ledgers and paused monitor remain unchanged. New implementation variants are isolated inference-only prototypes in `scripts/profile_hyperdense_followup.py`.',
        '', '## Findings that change the next experiment','',
        '**Inference overhead is measurable locally.** Across 63 matched layer cases on the Apple GPU (seven study-derived shape/bias combinations, three algebras and three batch sizes), caching constants produced a median 2.93× speedup. Pre-expanding the learned hypercomplex map into an ordinary dense matrix produced a median 13.55× layer speedup. These are ratios across microbenchmarks, not end-to-end application speedups or estimates for L4.',
        '', '**The evaluation period differs sharply from training.** Copper training values ranged from 1.9395 to 3.2930; evaluation values reached 4.7785. About 75.75% of evaluation rows were above the training maximum. Three other observed series also frequently lay outside their training ranges. This is a distribution-shift hypothesis supported by descriptive evidence, not a demonstrated sole cause of forecast errors.',
        '', '**The models learn, but their forecasts transfer poorly.** Eighteen of 96 arms beat persistence on inner-validation MSE, yet all 96 lost on outer MAE. All twelve native models and 95 of 96 total arms had negative average forecast bias. Native Crossformer illustrates the problem: MAE/persistence was 1.07 when the forecast origin was within the training price range and 7.55 when it was above that range. Those subgroups are also different time periods, so this is not a causal experiment.',
        '', '![Price range and native forecast errors](figures/hyperdense-followup-diagnostics.png)',
        '', '## Implementation experiment','',
        'The original algebra constants are CPU float64 tensors outside the module buffer registry. Every float32 forward requests a dtype conversion; accelerator forwards also request device transfer. The CPU operator trace counted 20 `aten::_to_copy` calls over 20 original forwards and none for the cached implementation. Both contraction implementations still used two `bmm` calls per forward, while dense export used one `addmm`. The profiler is for attribution only; separate unprofiled runs supply latency measurements.',
        '', 'Each case used identical weights and inputs across original, cached-constant and dense-export implementations, five warm-up calls, five rounds with shuffled implementation order and fifteen calls per round. Inputs were resident before timing; CPU used two threads, MPS synchronized at timing boundaries. All 126 device/shape/algebra/batch cases passed float32 equivalence checks (atol 2e-5, rtol 2e-4). Twelve float64 basis/leading-axis checks also passed at 1e-12. Seven focused tests additionally check gradients with respect to inputs, immutable export snapshots and an active selected layer in the full MICN graph.',
        '', 'Full-graph checks used freshly initialized MICN 8D and FiLM 4D on CPU, batch sizes 1 and 32. Archived trained weights were not available, so these are not reruns of trained checkpoints. Existing saved forecasts remain the evidence for forecasting accuracy. Selected layers were verified to affect model outputs.',
        '', '| CPU full model | Batch | Original ms | Cached ms | Dense export ms |', '|---|---:|---:|---:|---:|']
    for row in perf['full_models']:
        lines.append(f'| {names[row["backbone"]]} {row["algebra"]} | {row["batch"]} | '+ ' | '.join(f'{row["timings"][v]["median_ms"]:.3f}' for v in ['original','cached','dense'])+' |')
    lines += ['', 'FiLM dense export reduced whole-model CPU latency by about 14.1% at batch 1 and 7.6% at batch 32. MICN showed no clear gain at batch 32; timing ranges overlapped. Most model computation is outside the selected layer. Expanding FiLM increased registered parameter/buffer storage from 21,993,028 to 34,575,940 bytes. These counts exclude activations, allocator overhead and unregistered shared algebra tensors; they are not peak process memory. Caching preserves compact weights; dense export trades resident storage for simpler execution. Exported weights must be regenerated after any training update.',
        '', '## Data, target and learning checks','',
        'Recomputed 142,560 saved forecast rows across all 96 arms against source data: target dates, lead alignment, forecast origins and persistence were correct. Maximum raw-target reconstruction discrepancy was 1.61e-7, consistent with float32 scaling roundoff; persistence exactly matched the raw origin price. Training-only scaler values and chronological target partitions matched all saved audits. No target clipping was found in the inspected preparation/inverse-scaling path. These checks do not prove every component of each model is bug-free.',
        '', '| Partition | Dates | Forecast origins | Persistence MAE | Copper rows above training maximum |', '|---|---|---:|---:|---:|']
    for label,row in diag['partitions'].items():lines.append(f'| {label} | {row["start_date"]}–{row["end_date"]} | {row["origins"]} | {row["persistence_mae"]:.5f} | {100*row["target_above_train_max_fraction"]:.1f}% |')
    lines += ['', 'Eighty-eight of 96 arms lost to persistence at every individual forecast lead. Ten selected their final executed epoch, so incomplete training remains possible, but simply adding epochs does not address the observed change in data distribution. Training and checkpoint selection optimized MSE whereas the headline metric is MAE; both should be reported and the selection objective specified in the next protocol.',
        '', '## Small CPU baseline probes','',
        'Fitted ridge regressions only on the 1,158 actual training windows, predicting a correction to persistence. One used flattened level inputs; the other subtracted each feature’s latest observation from its history before flattening. Both used the same train-only standardization and seven fixed ridge penalties (0.01 to 10,000), selected by inner MSE. No outer score selected a penalty. A drift baseline used only the average training price change. These are post-hoc diagnostic probes, not independent evidence of predictive superiority.',
        '', '| Probe | Selected penalty | Inner MAE / persistence | Outer MAE / persistence |', '|---|---:|---:|---:|']
    for method in ['ridge_residual_levels','ridge_residual_relative_to_latest','train_mean_drift']:
        rows={r['split']:r for r in diag['baseline_probes'] if r['method']==method}
        lines.append(f'| {method} | {rows["inner"].get("alpha","—")} | {rows["inner"]["mae_ratio"]:.4f} | {rows["outer"]["mae_ratio"]:.4f} |')
    lines += ['', 'The relative-input ridge reduced inner MSE by 9.65% versus persistence, but outer MAE was still 1.47% worse. The level-input ridge was 13.56% worse on outer MAE and chose the largest available penalty, signalling a boundary optimum in that small search. This supports testing level-relative features and a persistence residual head; it does not show that preprocessing fixes the neural models.',
        '', '## Next action','',
        'Validate cached constants and exported dense maps on L4 before making GPU performance claims. For accuracy, first run a small native-model representation pilot with identical observations and a shared residual-output formulation. Only advance to hypercomplex tuning if the pilot gives useful evidence. Preserve all algebra/control arms within the selected models, record the selection of MICN/FiLM as post-hoc, and use fresh data for any independent confirmation.',
        '', '[Costed conditional study plan](hyperdense_followup_plan.md). [Machine-readable diagnostics](../results/hyperdense-followup-local/persistence-diagnostics.json). [Timing measurements and profiles](../results/hyperdense-followup-local/inference.json).', '']
    (ROOT/'docs/hyperdense_followup_results.md').write_text('\n'.join(lines))

    base=scenarios[0]
    plan=['# Conditional HyperDense confirmation plan','',
        'Prepared 11 September 2026 after local diagnostics. This is a costed proposal, not a launched job queue. The prior $20 allocation remains closed. No new spending cap has been authorized. Dataset selection and its frozen split manifest must precede final admission; the following estimates assume the earlier workload, not an arbitrarily larger benchmark.',
        '', '## Questions and order','',
        '1. **Implementation:** on one L4, do caching and equivalent dense export remove the observed layer overhead? Run the same algebra/shape/batch matrix, with synchronized repeated timing, numerical equivalence, device allocation measurements and operator profiling. Separately retain compact checkpoint size and expanded deployment storage. Budget one 600-second job; no training needed.',
        '2. **Representation pilot:** use native MICN and FiLM only, 3 paired development seeds (1101–1103), two formulations: the existing levels/direct output and per-feature histories relative to their last value plus a persistence-residual output head. Keep the observed inputs and learned head dimensions matched. The existing direct-output arm keeps its native initialization; the residual arm starts at persistence with a zero-initialized learned correction. Because representation, output formulation and head initialization change together, this pilot tests a package, not their separate causal contributions. A later factorial ablation would be needed to distinguish them. Use up to 100 epochs, inner MAE checkpoint selection, patience 20 and the same optimizer settings. Twelve fits. This pilot must use development data; never use final test outcomes to decide whether to advance.',
        '3. **Balanced tuning, conditional on the pilot:** retain MICN and FiLM and all eight arms (native, controlled real, 2D/4D/8D and corresponding low-rank controls). Freeze the selected input/output formulation using development evidence. Give each arm eight settings: learning rate {0.0003,0.001} × AdamW weight decay {0,0.0001} × selected-operator initialization multiplier {0.5,1}. Biases and all untouched weights remain paired. Apply the multiplier to the native selected operator too; document complex operators and low-rank factors so effective matrix variance is controlled. Use one common tuning seed (1201), up to 100 epochs and identical inner MAE stopping. This is 128 fits. Dropout remains at the shared architecture default because there is no verified common independently configurable site; do not invent an unequal dropout search. Implement and test these evaluation fields before any launch—the current two-rate runner does not implement this proposed grid.',
        '4. **Confirmation:** freeze each arm’s setting, then run all sixteen model/arm combinations on five fresh paired seeds (1301–1305), each with the same 100-epoch ceiling and stopping protocol: 80 fits per dataset. Use an independently frozen chronological test period or a new dataset with development/test separation. Reused Copper dates remain exploratory. Persist the restored trained weights, architecture, preprocessing and split metadata so future profiling can use actual trained checkpoints.',
        '', '## Advancement and success rules','',
        'Advance from the representation pilot only if both models complete integrity/convergence checks and the proposed formulation improves median development MAE over the existing formulation in at least two of three paired seeds per model; retain per-seed scores and do not call this a significance test. If the two models disagree, revise the hypothesis rather than silently dropping the less favourable model from a supposedly balanced comparison. Stop after the pilot if it adds no useful evidence. Budget admission must cover the complete next stage with the reserve intact.',
        '', 'Use MAE as the primary tuning, checkpoint-selection and final accuracy objective; also report MSE, bias, per-lead error and calibration. An accuracy success must improve native and controlled real forecasts, with persistence serving as the practical benchmark. Preregister a minimum useful improvement (proposed: 2% relative MAE). A compression success requires a predefined noninferiority margin (proposed: at most 2% MAE increase against the stronger native/real baseline) and at least 25% whole-model parameter reduction. These thresholds are proposals, not conclusions from current results. Simultaneous confidence intervals must satisfy the relevant margins; point estimates alone do not qualify.',
        '', 'Analyze seeds and time separately: retain paired seed-level effects, use forecast origins with all leads kept together, preserve time dependence with blocked resampling and adjust the preregistered family of model/algebra contrasts. Choose block lengths using development data only and report sensitivity; insufficient effective test blocks prevent a robust claim. For a claim beyond one dataset, repeat on at least two independent datasets and cost each separately. Do not select a winner on the final test and then treat it as an independently confirmed result.',
        '', '## Cost and admission','',
        f'Pricing was read through the existing CLI at {rates["captured_at"]}: L4 ${rates["rates"]["gpu_l4_per_hour"]:.4f}/hour, two CPU cores and 4 GiB memory, combined ${hourly:.4f}/hour before uncertainty allowances. One L4 at a time. Estimates use measured training seconds per epoch, 100 epochs, 1.5× runtime allowance, 120 seconds per-batch overhead, upward timeout rounding, timeout plus 120 seconds reserved at 3× rates, and $0.05 startup allowance per job. No automatic retries; all failures consume the ledger.',
        '', '| Scenario (one dataset) | L4 inference check | 12-fit pilot | 128-fit tuning | 80-fit confirmation | Total reservations | Proposed cap incl. margin/reserve |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in scenarios:
        plan.append(f'| {s["workload_multiplier"]}× earlier training workload | ${s["inference_check_reservation_usd"]:.2f} | ${s["representation_pilot"]["reservation_usd"]:.2f} | ${s["equal_tuning"]["reservation_usd"]:.2f} | ${s["confirmation"]["reservation_usd"]:.2f} | ${s["reservations_usd"]:.2f} | ${s["suggested_cap_with_20_percent_margin_and_1_dollar_reserve"]} |')
    pilot_cap=math.ceil((base['inference_check_reservation_usd']+base['representation_pilot']['reservation_usd'])*1.2+1)
    plan += ['', f'For only the L4 inference check plus the representation pilot at the historical workload, the proposed separate cap is **${pilot_cap}**, including 20% margin and a $1 reserve. The entire programme is conditional; these are conservative admission estimates, not quoted bills. A fourfold workload is an illustrative scaling scenario, not a guarantee that a particular dataset costs four times as much. Fresh data, optimizer changes, compilation and representation can change runtime. Recalibrate actual target data and reprice before launching; stop if the whole stage does not fit.',
        '', 'Implementation/data work still required before paid confirmation: freeze the new dataset and dates, implement the proposed representation/optimizer/initialization controls, verify equivalence and gradient gates, save trained checkpoints, generate CLI manifests and check the complete stage reservation against a newly authorized cap. The local investigation does not authorize or implicitly start that later study.',
        '', '[Detailed batch reservations](../results/hyperdense-followup-local/confirmation-cost-proposal.json). [Local evidence](hyperdense_followup_results.md).', '']
    (ROOT/'docs/hyperdense_followup_plan.md').write_text('\n'.join(plan))
    print(json.dumps(dict(costs=[{k:v for k,v in s.items() if k in ['workload_multiplier','reservations_usd','suggested_cap_with_20_percent_margin_and_1_dollar_reserve']} for s in scenarios],pilot_cap=pilot_cap),indent=2))


if __name__=='__main__':main()
