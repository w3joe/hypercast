"""Expose a completed TSLib15 archive in Compare without creating training jobs."""
import argparse
import json
from pathlib import Path
from hypercast4d.comparison_archives import list_archives

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, default=Path('results/playground'))
    parser.add_argument('--project', type=Path, default=Path.cwd())
    args = parser.parse_args()
    project = args.project.resolve()
    item = dict(format='tslib15-v1', source=str(args.source.resolve().relative_to(project)), dataset=str(args.dataset.resolve().relative_to(project)))
    folder = args.workspace / 'comparison-archives'; folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f'{args.source.name}.json'
    previous = destination.read_bytes() if destination.exists() else None
    temporary = destination.with_suffix('.tmp'); temporary.write_text(json.dumps(item, indent=2) + '\n'); temporary.replace(destination)
    try:
        records = list_archives(args.workspace.resolve(), project)
    except Exception:
        if previous is None: destination.unlink()
        else: destination.write_bytes(previous)
        raise
    added = [r for r in records if r['archive']['source'] == args.source.name]
    print(f'Registered {len(added)} model/variant results, {sum(len(r["runs"]) for r in added)} fits. No jobs created.')
