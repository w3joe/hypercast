"""Render the complete audited pilot, including unfavourable seeds and limitations."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ALLOCATION=ROOT/'results/hyperdense-representation-pilot/allocation-001'


def main():
    a=json.loads((ALLOCATION/'analysis.json').read_text())
    if a['ledger_status']!='complete' or a['successful_fits']!=12:
        raise RuntimeError('Complete audited pilot required')
    rows=a['rows']; inf=a['inference']
    lines=['# Authorized HyperDense pilot results','',
        'Completed on Modal L4 after authorization of a new $6 cap and parallel execution. The inference gate ran first; the six paired training jobs ran in waves of two. All seven jobs completed without retries. All twelve restored checkpoints, curves and development forecasts were archived and passed the returned-artifact audit. No outer or final-test predictions were generated.','',
        '## Representation package','',
        'Native MICN and native FiLM each received seeds 1101–1103, a levels/direct arm and a relative/residual arm. The latter changes input centering, output formulation and external-head initialization together. Both use MAE training and inner-MAE checkpoint selection, the same Adam settings, up to 100 epochs and patience 20. FiLM native is already complex.','',
        '| Model | Seed | Direct MAE / persistence | Relative MAE / persistence | Relative MAE reduction | Direct best / ran epochs | Relative best / ran epochs |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for b in ['micn','film']:
        for seed in [1101,1102,1103]:
            pair={r['mode']:r for r in rows if r['backbone']==b and r['seed']==seed}
            d,r=pair['levels_direct'],pair['relative_residual']
            change=100*(1-r['mae']/d['mae'])
            lines.append(f"| {b.upper()} | {seed} | {d['mae_over_persistence']:.4f} | {r['mae_over_persistence']:.4f} | {change:.2f}% | {d['best_epoch']} / {d['epochs_ran']} | {r['best_epoch']} / {r['epochs_ran']} |")
    lines+=['',f"The prespecified paired development **score criterion** {'passed' if a['development_gate']['advance'] else 'did not pass'} for the two-model package. This is a development screening rule, not a significance test or an accuracy claim on independent data. The full advancement gate is not yet cleared because a ceiling fit requires convergence review.",'']
    for row in a['paired_improvements']:
        lines.append(f"- {row['backbone'].upper()}: median paired MAE reduction {row['median_reduction_percent']:.2f}% across all three seeds.")
    ceilings=[f"{r['backbone']}/{r['seed']}/{r['mode']}" for r in rows if r['epochs_ran']==100]
    lines+=['',f"Training reached the 100-epoch ceiling in {len(ceilings)} fit(s): {', '.join(ceilings) or 'none'}. Final-executed-epoch checkpoint flags: {len(a['convergence_flags'])}. Patience stopping is evidence of a plateau under this protocol, not proof of global convergence.",'',
        'The 207 development origins span targets from 2019-10-09 through 2020-08-10 and were already inspected in earlier work. Their persistence MAE is about 0.04339. These scores also selected checkpoints. They cannot confirm an outer-period forecasting gain or establish that HyperDense improves accuracy; only native backbones were trained in this pilot.','',
        '## L4 inference','',
        f"All 63 layer cases and four full-model cases passed float32 numerical equivalence. Cached constants achieved a median {inf['cached_median_layer_speedup']:.2f}× layer speedup; equivalent dense export achieved {inf['dense_median_layer_speedup']:.2f}×. These are medians of casewise timing ratios, not throughput estimates for an application.",'',
        '| Full model | Batch | Original ms | Cached ms | Dense export ms | Dense latency reduction |',
        '|---|---:|---:|---:|---:|---:|']
    for r in inf['full_models']:
        t={k:v['median_ms'] for k,v in r['timings'].items()}
        lines.append(f"| {r['backbone'].upper()} | {r['batch']} | {t['original']:.3f} | {t['cached']:.3f} | {t['dense']:.3f} | {100*(1-t['dense']/t['original']):.1f}% |")
    lines+=['',
        'The inference job uses matched fresh weights for MICN 8D and FiLM 4D, synchronized repeated timings and separate CUDA operator traces. It precedes pilot training and does not profile the trained native checkpoints. Dense export expands resident weight storage; cached constants preserve compact weights. Peak allocator increments are reported with all implementations resident and exclude external process memory. These trials do not establish an end-to-end deployment speedup or performance uncertainty across independent machines.','',
        '## Budget and next decision','',
        f"Conservative accounted reservations total **${a['conservative_accounted_usd']:.2f} of $6**, including startup/runtime allowances. The protected $1 reserve remains intact, and ${a['cap_usd']-a['conservative_accounted_usd']:.2f} is unspent against this conservative accounting. This is not a final Modal invoice. Prior allocation ledgers and frozen source archives remain unchanged.",'',
        'The pilot allocation ends here. The representation evidence can inform a separately frozen balanced eight-arm study, but this run authorizes neither that larger study nor new datasets. Independent data, optimizer/initialization grid controls, recosting and an explicit new cap are still needed before confirmation.','',
        '## Artifacts','',
        '- [Full scores, paired gate, per-lead errors and audit](../results/hyperdense-representation-pilot/allocation-001/analysis.json)',
        '- [New allocation ledger and call IDs](../results/hyperdense-representation-pilot/allocation-001/ledger.json)',
        '- [Frozen parallel manifest](../results/hyperdense-representation-pilot/prepared-v2-parallel/manifest.json)',
        '- [User authorization record](../results/hyperdense-representation-pilot/authorization.json)',
        '- `allocation-001/job-00.tar.gz` contains inference results and CUDA traces; `job-01.tar.gz` through `job-06.tar.gz` contain the twelve trained checkpoints and diagnostics.',
        '- `prepared-v1` and `prepared-v2-parallel` retain their separate frozen source archives.','']
    (ROOT/'docs/hyperdense_representation_pilot_results.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
