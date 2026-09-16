"""Unified model editing and single-experiment command line interface."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.error

import yaml

from .architecture import presets, validate_architecture, architecture_hash
from .compute import normalize_execution
from .graph_architecture import convert_to_graph
from .model_editing import load_model, write_model, checked, layer_types, describe_layers, edit_layer, connect_graph
from .playground_runner import normalize_evaluation
from .workspace_runtime import request, discover, ensure_backend, stop_backend


def emit(value, machine=False):
    if machine:
        print(json.dumps(value, allow_nan=False))
    elif isinstance(value, str):
        print(value)
    else:
        print(yaml.safe_dump(value, sort_keys=False, allow_unicode=True).rstrip())


def assignments(values):
    result = {}
    for pair in values or []:
        key, separator, value = pair.partition('=')
        if not separator or not key or key in result:
            raise ValueError('Use unique key=value assignments')
        result[key] = yaml.safe_load(value)
    json.dumps(result, allow_nan=False)
    return result


def parse_cells(value):
    try:
        cells = [{'window': int(pair.split('/')[0]), 'horizon': int(pair.split('/')[1])} for pair in value.split(',')]
        if any(len(pair.split('/')) != 2 for pair in value.split(',')):
            raise ValueError()
        return normalize_evaluation({'cells': cells})['cells']
    except (ValueError, IndexError) as error:
        raise ValueError('Cells must be window/horizon pairs, e.g. 10/1,20/5') from error


def globals_for(parser, defaults=False):
    default = None if defaults else argparse.SUPPRESS
    parser.add_argument('--project-root', default=default)
    parser.add_argument('--results-root', default=default)
    parser.add_argument('--json', dest='machine', action='store_true', default=False if defaults else argparse.SUPPRESS)


def parser():
    root = argparse.ArgumentParser(prog='hypercast', description=__doc__)
    globals_for(root, True)
    groups = root.add_subparsers(dest='group', required=True)
    def group(name):
        return groups.add_parser(name).add_subparsers(dest='action', required=True)
    def command(parent, name, file=False, cells=False):
        leaf = parent.add_parser(name)
        globals_for(leaf)
        if file:
            leaf.add_argument('file', type=Path)
        if cells:
            leaf.add_argument('--cells', default='10/1')
        return leaf
    model = group('model')
    command(model, 'list')
    create = command(model, 'create')
    create.add_argument('preset'); create.add_argument('--out', type=Path, required=True); create.add_argument('--name')
    for action in ('inspect', 'validate', 'expand'):
        leaf = command(model, action, file=True, cells=True)
        if action == 'expand':
            leaf.add_argument('--out', type=Path, required=True)
    command(model, 'save', file=True, cells=True)
    export = command(model, 'export'); export.add_argument('id'); export.add_argument('--out', type=Path, required=True)
    layer = group('layer')
    command(layer, 'types'); command(layer, 'list', file=True, cells=True)
    for action in ('add', 'insert', 'set', 'replace', 'remove'):
        leaf = command(layer, action, file=True, cells=True)
        if action in ('set', 'replace', 'remove'):
            leaf.add_argument('id')
        else:
            leaf.add_argument('--id', required=True)
        if action in ('add', 'insert', 'replace'):
            leaf.add_argument('--type', dest='kind', required=True)
        leaf.add_argument('--set', action='append', default=[])
        leaf.add_argument('--out', type=Path)
        leaf.add_argument('--allow-invalid', action='store_true')
        if action == 'insert':
            anchors = leaf.add_mutually_exclusive_group(required=True)
            anchors.add_argument('--before'); anchors.add_argument('--after')
            leaf.add_argument('--port')
    graph = group('graph')
    for action in ('connect', 'disconnect', 'output'):
        leaf = command(graph, action, file=True, cells=True)
        leaf.add_argument('--out', type=Path); leaf.add_argument('--allow-invalid', action='store_true')
        if action == 'output':
            leaf.add_argument('id')
        else:
            leaf.add_argument('--target', required=True); leaf.add_argument('--port', required=True)
        if action == 'connect':
            leaf.add_argument('--source', required=True); leaf.add_argument('--replace', action='store_true')
    experiment = group('experiment')
    run = command(experiment, 'run', file=True)
    run.add_argument('--evaluation', type=Path)
    run.add_argument('--eval', action='append', default=[])
    for key in ('preset', 'cells', 'seeds', 'device'):
        run.add_argument('--' + key)
    for key in ('epochs', 'batch-size'):
        run.add_argument('--' + key, type=int)
    run.add_argument('--learning-rate', type=float)
    run.add_argument('--target', choices=['local', 'modal', 'gcp'], default='local')
    run.add_argument('--gpu'); run.add_argument('--gpu-count', type=int)
    run.add_argument('--dry-run', action='store_true'); run.add_argument('--detach', action='store_true'); run.add_argument('--yes', action='store_true')
    final = command(experiment, 'final-test'); final.add_argument('id')
    final.add_argument('--yes', action='store_true'); final.add_argument('--detach', action='store_true')
    final.add_argument('--timeout-seconds', type=int)
    job = group('job')
    command(job, 'list')
    for action in ('status', 'logs', 'cancel', 'results'):
        leaf = command(job, action); leaf.add_argument('id')
        if action in ('status', 'logs'):
            leaf.add_argument('--follow', action='store_true')
        if action == 'results':
            leaf.add_argument('--out', type=Path)
    server = group('server')
    start = command(server, 'start'); start.add_argument('--port', type=int, default=0)
    command(server, 'status'); command(server, 'stop')
    return root


def evaluation_for(args):
    raw = yaml.safe_load(args.evaluation.read_text()) if args.evaluation else {}
    if not isinstance(raw, dict):
        raise ValueError('Evaluation file must contain an object')
    raw.update(assignments(args.eval))
    supported = set(normalize_evaluation({})) | {'initialization', 'benchmark', 'numerical_preflight',
        'internal_preflight', 'remaining_preflight', 'remaining_tuning', 'remaining_replication', 'replacement', 'nested_stopping', 'early_stopping_relative_delta'}
    if set(raw) - supported:
        raise ValueError(f'Unknown evaluation settings: {sorted(set(raw) - supported)}')
    if 'folds' in raw:
        candidate = normalize_evaluation(raw)
        if raw['folds'] != candidate['folds']:
            raise ValueError('Custom folds are not supported; choose an evaluation preset')
    for key in ('preset', 'epochs', 'batch_size', 'learning_rate', 'device'):
        if getattr(args, key, None) is not None:
            raw[key] = getattr(args, key)
    if args.cells:
        raw['cells'] = parse_cells(args.cells)
    if args.seeds:
        raw['seeds'] = [int(seed) for seed in args.seeds.split(',')]
    if args.target != 'local':
        raw['device'] = 'cuda'
    return normalize_evaluation(raw)


def confirmation(args, execution, *, final=False):
    if not final and execution['target'] == 'local':
        return
    message = ('This evaluates the held-out test set once. ' if final else '')
    if execution['target'] != 'local':
        message += f"Billed {execution['target']} execution: {execution}. Dataset and code will be uploaded. "
    print(message, file=sys.stderr)
    if not args.yes and (not sys.stdin.isatty() or input('Continue? [y/N] ').strip().lower() != 'y'):
        raise ValueError('Not submitted. Supply --yes to confirm non-interactively.')


def wait_job(url, job_id):
    previous = None
    try:
        while True:
            job = request(url, f'/api/v1/jobs/{job_id}')
            status = job['status']
            update = (status['state'], status.get('completed'), status.get('total'), status.get('current'))
            if update != previous:
                print(f"{job_id}: {status['state']} {status.get('completed', 0)}/{status.get('total', 0)} {status.get('current') or ''}", file=sys.stderr)
                previous = update
            if status['state'] in {'complete', 'failed', 'cancelled', 'interrupted'}:
                return job, 0 if status['state'] == 'complete' else 130 if status['state'] == 'cancelled' else 4
            time.sleep(.5)
    except KeyboardInterrupt:
        print(f'Stopped waiting. Job {job_id} remains submitted. Cancel with: hypercast job cancel {job_id}', file=sys.stderr)
        return {'job_id': job_id, 'detached': True}, 130


def dispatch(args):
    project = Path(args.project_root or Path.cwd()).resolve()
    results = Path(args.results_root or project / 'results/playground')
    if not results.is_absolute():
        results = project / results
    results = results.resolve()
    # Explicit file arguments retain shell-relative meaning; only dataset paths
    # and backend execution are rooted at --project-root.
    cells = parse_cells(args.cells) if getattr(args, 'cells', None) and args.group != 'experiment' else [{'window': 10, 'horizon': 1}]
    backend = lambda: ensure_backend(project, results)
    if args.group == 'server':
        found = ensure_backend(project, results, args.port) if args.action == 'start' else discover(project, results)
        if args.action == 'stop' and found:
            stop_backend(found)
        return found or {'running': False}, 0
    if args.group == 'model':
        if args.action == 'list':
            return [{'id': spec['preset_id'], 'name': spec['name']} for spec in presets()], 0
        if args.action == 'create':
            spec = next((s for s in presets() if s['preset_id'] == args.preset), None)
            if spec is None:
                raise ValueError('Unknown preset; use hypercast model list')
            spec.pop('preset_id', None); spec['locked'] = False
            if args.name:
                spec['name'] = args.name
            spec, _ = checked(spec, cells)
            write_model(args.out, spec)
            return {'file': str(args.out), 'name': spec['name']}, 0
        if args.action == 'export':
            record = request(backend()['url'], f'/api/v1/architectures/{args.id}')
            write_model(args.out, record)
            return {'file': str(args.out)}, 0
        spec = load_model(args.file)
        if args.action == 'inspect':
            return {'name': spec['name'], 'schema_version': spec.get('schema_version', 1), 'layers': describe_layers(spec, cells)}, 0
        if args.action == 'expand':
            expanded = convert_to_graph(spec, cells[0]['window'], cells[0]['horizon'])
            for key in ('locked', 'view'):
                if key in spec:
                    expanded[key] = spec[key]
            checked(expanded, cells)
            write_model(args.out, expanded)
            return {'file': str(args.out), 'nodes': len(expanded['nodes'])}, 0
        spec, _ = checked(spec, cells)
        if args.action == 'save':
            return request(backend()['url'], '/api/v1/architectures', spec), 0
        return {'valid': True, 'cells': [validate_architecture(spec, c['window'], c['horizon']) for c in cells]}, 0
    if args.group in ('layer', 'graph'):
        if args.action == 'types':
            from .algebras import ALGEBRAS
            return {'types': list(layer_types().values()), 'algebras': {key: value.component_count for key, value in ALGEBRAS.items()}, 'hyperdense_units': 'Manual output width = units × algebra components; graph shape_mode=preserve infers units, pads and crops to preserve the input width'}, 0
        spec = load_model(args.file)
        if args.action == 'list':
            return describe_layers(spec, cells), 0
        if spec.get('locked'):
            raise ValueError('Locked model: create an editable preset copy first')
        sharing_warning = None
        if args.group == 'layer' and args.action == 'replace' and spec.get('schema_version') == 2:
            old = next((n for n in spec['nodes'] if n['id'] == args.id), {})
            if old.get('module_ref') and sum(n.get('module_ref') == old['module_ref'] for n in spec['nodes']) > 1:
                sharing_warning = 'The replacement makes this graph call independent of its previously shared weights.'
        if args.group == 'layer':
            spec = edit_layer(spec, args.action, node_id=args.id, kind=getattr(args, 'kind', None),
                              params=assignments(args.set), before=getattr(args, 'before', None),
                              after=getattr(args, 'after', None), port=getattr(args, 'port', None), cells=cells)
        else:
            if spec.get('schema_version') != 2:
                raise ValueError('Use model expand before graph editing')
            if args.action == 'connect':
                spec = connect_graph(spec, args.source, args.target, args.port, replace=args.replace)
            elif args.action == 'disconnect':
                found = [e for e in spec['edges'] if e['target'] == args.target and e['port'] == args.port]
                if not found:
                    raise ValueError('No connection at that target port')
                spec['edges'] = [e for e in spec['edges'] if e not in found]
            else:
                if args.id not in {n['id'] for n in spec['nodes']}:
                    raise ValueError('Unknown graph output node')
                spec['output'] = args.id
        spec, notices = checked(spec, cells, allow_invalid=args.allow_invalid)
        if sharing_warning:
            notices.append(sharing_warning)
        destination = args.out or args.file
        write_model(destination, spec, overwrite=destination.resolve() == args.file.resolve())
        return {'file': str(destination), 'warnings': notices,
                'layers': describe_layers(spec, cells) if not notices else []}, 0
    if args.group == 'experiment':
        if args.action == 'run':
            evaluation = evaluation_for(args)
            if args.target == 'local' and args.gpu or args.target != 'gcp' and args.gpu_count is not None:
                raise ValueError('GPU selection requires a cloud target; GPU count is supported on GCP only')
            raw = {'target': args.target}
            if args.gpu:
                raw['gpu'] = args.gpu
            if args.gpu_count is not None:
                raw['gpu_count'] = args.gpu_count
            execution = normalize_execution(raw)
            spec, _ = checked(load_model(args.file), evaluation['cells'])
            dataset = (project / evaluation['data_path']).resolve()
            if not dataset.is_relative_to(project / 'data') or not dataset.is_file():
                raise ValueError('Dataset must exist inside the selected project data directory')
            if args.dry_run:
                return {'dry_run': True, 'execution': execution, 'evaluation': evaluation,
                        'candidate_hash': architecture_hash(spec, {**evaluation, 'execution': execution}),
                        'trials': len(evaluation['cells']) * len(evaluation['seeds']) * len(evaluation['folds']) * (8 if evaluation.get('remaining_replication') else 16 if evaluation.get('remaining_tuning') else 1),
                        'cells': [validate_architecture(spec, c['window'], c['horizon']) for c in evaluation['cells']]}, 0
            confirmation(args, execution)
            url = backend()['url']
            record = request(url, '/api/v1/architectures', spec)
            job = request(url, '/api/v1/jobs', {'architecture': spec, 'evaluation': evaluation, 'execution': execution})
            print(f"Architecture {record['id']} · job {job['id']} · UI {url}", file=sys.stderr)
        else:
            url = backend()['url']
            parent = request(url, f'/api/v1/jobs/{args.id}')
            confirmation(args, parent['request'].get('execution', {'target': 'local'}), final=True)
            payload = {} if args.timeout_seconds is None else {'timeout_seconds': args.timeout_seconds}
            job = request(url, f'/api/v1/jobs/{args.id}/final-test', payload)
        result, code = (job, 0) if args.detach else wait_job(url, job['id'])
        if args.machine or 'status' not in result:
            return result, code
        return {'job_id': result['id'], 'state': result['status']['state'], 'ui': url,
                'artifacts': str(results / 'jobs' / result['id']),
                'summary': [{key: row.get(key) for key in ('window', 'horizon', 'mae_mean', 'mse_mean', 'mae_ratio', 'mse_ratio')}
                            for row in result.get('summary', [])]}, code
    if args.group == 'job':
        found = discover(project, results)
        if not found:
            raise RuntimeError('No backend running; use hypercast server start')
        url = found['url']
        if args.action == 'list':
            return [{'id': j['id'], **j['status']} for j in request(url, '/api/v1/jobs')], 0
        if not args.id or '/' in args.id or '..' in args.id:
            raise ValueError('Invalid job ID')
        path = f'/api/v1/jobs/{args.id}'
        if args.action == 'cancel':
            return request(url, path + '/cancel', {}), 0
        if args.action == 'status' and args.follow:
            return wait_job(url, args.id)
        if args.action == 'logs':
            previous = ''
            while True:
                content = request(url, path + '/log')
                if not args.follow:
                    return content, 0
                if content != previous:
                    chunk = content[len(previous):] if content.startswith(previous) else content
                    if args.machine:
                        emit({'job_id': args.id, 'log': chunk}, True)
                    else:
                        print(chunk, end='', flush=True)
                    previous = content
                if request(url, path)['status']['state'] in {'complete', 'failed', 'cancelled', 'interrupted'}:
                    return {'finished': True}, 0
                time.sleep(.5)
        job = request(url, path)
        if args.action == 'results':
            if args.out:
                if args.out.exists():
                    raise ValueError('Results export path already exists')
                source = results / 'jobs' / args.id
                args.out.mkdir(parents=True)
                for filename in ('request.json', 'status.json', 'runs.csv', 'per_lead.csv', 'summary.json', 'summary.csv', 'predictions.csv', 'diagnostics.json', 'weights.json', 'training.log'):
                    if (source / filename).is_file():
                        shutil.copy2(source / filename, args.out / filename)
            return {'id': args.id, 'state': job['status']['state'], 'summary': job['summary'],
                    'metric_note': 'MAE/persistence and MSE/persistence: lower is better; 1 equals persistence',
                    'artifacts': str(results / 'jobs' / args.id)}, 0
        return job['status'], 0
    raise ValueError('Unsupported command')


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        value, code = dispatch(args)
        emit(value, args.machine)
        return code
    except KeyboardInterrupt:
        print('Stopped observing; submitted jobs were not cancelled.', file=sys.stderr)
        return 130
    except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
        emit({'error': str(error), 'exit_code': 2}, True) if args.machine else print(str(error), file=sys.stderr)
        return 2
    except (RuntimeError, OSError, urllib.error.URLError) as error:
        emit({'error': str(error), 'exit_code': 3}, True) if args.machine else print(str(error), file=sys.stderr)
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
