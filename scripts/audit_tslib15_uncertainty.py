"""Supplement the frozen analysis by resampling test hours jointly across seeds.

Declared while training was running, before reviewing test-score comparisons.
The original analysis is retained; this audit preserves common temporal shocks
across seeds instead of treating each seed's test hours independently.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def intervals(base, other, block, resamples=10000, seed=3401):
    c, s, n = base.shape
    assert other.shape == base.shape
    point = 100 * (base.mean((1, 2)) - other.mean((1, 2))) / base.mean((1, 2))
    joined = np.concatenate([base, other], axis=0)
    extended = np.concatenate([joined, joined], axis=2)
    prefix = np.concatenate([np.zeros((*joined.shape[:2], 1)), np.cumsum(extended, axis=2)], axis=2)
    starts = np.arange(n)
    full = prefix[:, :, starts + block] - prefix[:, :, starts]
    count, tail = divmod(n, block)
    remainder = prefix[:, :, starts + tail] - prefix[:, :, starts] if tail else None
    rng = np.random.default_rng(seed)
    samples = []
    for begin in range(0, resamples, 100):
        b = min(100, resamples - begin)
        seed_ids = rng.integers(s, size=(b, s))
        # One origin-block sample per replicate, shared across seeds and contrasts.
        indices = rng.integers(n, size=(b, count))
        totals = full[:, seed_ids[:, :, None], indices[:, None, :]].sum((2, 3))
        if tail:
            last = rng.integers(n, size=b)
            totals += remainder[:, seed_ids, last[:, None]].sum(2)
        means = totals / (s * n)
        samples.append((100 * (means[:c] - means[c:]) / means[:c]).T)
    boot = np.concatenate(samples)
    sd = boot.std(0, ddof=1)
    critical = np.quantile(np.max(np.abs((boot - point) / np.maximum(sd, 1e-12)), axis=1), .95)
    return point, point - critical * sd, point + critical * sd


def analyze(run):
    ledger = json.loads((run / 'ledger.json').read_text())
    assert ledger['status'] == 'complete'
    original = json.loads((run / 'analysis.json').read_text())
    plan = json.loads((run / ledger['prepared_dir'] / 'evaluation-plan.json').read_text())
    errors = {}
    for a in ledger['attempts']:
        j = a['job']
        if j['phase'] != 'test':
            continue
        assert a['status'] == 'complete'
        with np.load(run / 'predictions' / f"{a['index']:03d}.npz", allow_pickle=False) as p:
            errors[(j['backbone'], j['arm'], j['seed'])] = np.abs(p['prediction'] - p['actual']).mean(1)
    contrasts = [(b, a) for b in plan['backbones'] for a in ('complex', 'quaternion', 'octonion')]
    base = np.stack([np.stack([errors[(b, 'real', s)] for s in plan['evaluation_seeds']]) for b, a in contrasts])
    other = np.stack([np.stack([errors[(b, a, s)] for s in plan['evaluation_seeds']]) for b, a in contrasts])
    rows = {}
    for block in (12, 24, 48, 168):
        point, low, high = intervals(base, other, block)
        rows[str(block)] = [dict(backbone=b, arm=a, improvement_pct=float(p), lower_pct=float(l), upper_pct=float(h))
                           for (b, a), p, l, h in zip(contrasts, point, low, high)]
    primary = rows['24']
    report = dict(method='Shared circular origin blocks across all paired seeds and contrasts; seed resampling retained',
                  resamples=10000, intervals=rows,
                  adjusted_wins=sum(r['lower_pct'] > 0 for r in primary),
                  practical_adjusted_wins=sum(r['lower_pct'] > 2 for r in primary),
                  adjusted_wins_all_block_lengths=sum(all(v[i]['lower_pct'] > 0 for v in rows.values()) for i in range(45)),
                  original_adjusted_wins=original['primary_adjusted_wins'])
    (run / 'joint-origin-bootstrap.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    lines = ['# Shared-origin uncertainty audit', '',
             'All seeds see the same held-out hours. This supplementary bootstrap resamples each time block jointly across seeds, preserving common temporal shocks. It retains paired seed resampling, 10,000 replicates and simultaneous 95% intervals across all 45 contrasts. The original frozen analysis remains available.', '',
             f"At block length 24, {report['adjusted_wins']} contrasts have intervals above zero; {report['practical_adjusted_wins']} have lower bounds above 2%. {report['adjusted_wins_all_block_lengths']} remain above zero at every block length (12, 24, 48, 168).", '',
             '| Model | Arm | MAE reduction % | Shared-origin simultaneous 95% interval |',
             '|---|---|---:|---:|']
    for r in primary:
        lines.append(f"| {r['backbone']} | {r['arm']} | {r['improvement_pct']:.2f} | [{r['lower_pct']:.2f}, {r['upper_pct']:.2f}] |")
    lines += ['', 'Three seeds and one dataset still limit generalization. This is an uncertainty audit, not new model selection or tuning.', '']
    (run / 'joint-origin-bootstrap.md').write_text('\n'.join(lines))
    return {k: v for k, v in report.items() if k != 'intervals'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()
    while args.watch:
        ledger = json.loads((args.run / 'ledger.json').read_text())
        if ledger['status'] == 'complete':
            break
        if ledger['status'] == 'stopped_for_review':
            raise RuntimeError('Training stopped; uncertainty audit requires completed study')
        time.sleep(20)
    print(json.dumps(analyze(args.run), indent=2))
