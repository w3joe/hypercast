#!/usr/bin/env python3
"""Summarize controlled development; freeze rates only after the balanced sweep."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def choose_rate(rows):
    """Prespecified relative 0.1% tie region; prefer the lower learning rate."""
    best = min(row['mae_ratio'] for row in rows)
    if not math.isfinite(best) or best <= 0:
        raise ValueError('Invalid development score')
    tied = [row for row in rows if row['mae_ratio'] <= best * 1.001]
    return min(tied, key=lambda row: row['learning_rate'])


def report(root):
    state = json.loads((root / 'controller/controller_state.json').read_text())
    candidates = [c for c in state['candidates'] if c.get('stage_name') == 'controlled-development']
    grouped = defaultdict(list)
    counts = Counter()
    for candidate in candidates:
        if not candidate.get('job_id'):
            counts[candidate['status']] += 1
            continue
        directory = root / 'jobs' / candidate['job_id']
        status = json.loads((directory / 'status.json').read_text())
        counts[status['state']] += 1
        if status['state'] != 'complete':
            continue
        with (directory / 'runs.csv').open() as handle:
            fits = list(csv.DictReader(handle))
        if len(fits) != 1 or fits[0]['epochs_ran'] != '150':
            raise ValueError('Development artifact does not match the frozen protocol')
        fit = fits[0]
        grouped[(candidate['backbone'], candidate['variant'])].append({
            'learning_rate': candidate['evaluation']['learning_rate'], 'mae_ratio': float(fit['mae_ratio']),
            'train_seconds': float(fit['train_seconds']), 'job_id': candidate['job_id']})
    complete = (state.get('schedule_status') == 'controlled-development_complete' and len(candidates) == 54 and counts['complete'] == 54 and len(grouped) == 27
                and all({r['learning_rate'] for r in rows} == {.0003, .001} for rows in grouped.values()))
    budget = state['budget']
    spend = max(budget['confirmed_spend_usd'], budget['estimated_spend_usd'])
    result = {'complete': complete, 'counts': dict(counts), 'accounted_spend_usd': spend,
              'stage': 'controlled-development', 'selection_rule': 'lowest MAE/persistence; relative 0.1% tie chooses lower rate'}
    if complete:
        selected = [{'backbone': b, 'variant': v, **choose_rate(rows)} for (b, v), rows in sorted(grouped.items())]
        rates = state['rates'];per_second = (rates['gpu_l4_per_hour'] + 2*rates['cpu_core_per_hour'] + 4*rates['mem_gib_hour_cost']) / 3600
        # Provisional robust follow-up: 2 seeds x 3 folds, one 20/5 cell, 27 jobs.
        train_seconds = sum(row['train_seconds'] for row in selected) * 6 * (.65 / .70)
        cost = (train_seconds + 27*30) * per_second * budget['uncertainty_factor'] + 27*budget['startup_cost_usd']
        headroom = budget['total_usd'] - budget['safety_reserve_usd'] - spend
        result.update(selected=selected, provisional_robust_fits=162,
                      provisional_robust_cost_usd=cost, budget_headroom_usd=headroom,
                      provisional_robust_fits_budget=cost <= headroom)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=Path('results/modal-l4-runner'))
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args(); result = report(args.results_root)
    if args.freeze:
        if not result['complete']:
            raise SystemExit('Development is incomplete; no learning-rate selections were frozen.')
        path = args.results_root / 'controller/development_selection.json'
        if path.exists():
            raise SystemExit('A selection already exists; inspect it rather than overwriting.')
        path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
