import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hypercast4d.cli import main
from hypercast4d.architecture import presets, validate_architecture
from hypercast4d.graph_architecture import convert_to_graph
from hypercast4d.model_editing import checked, connect_graph, dense_swap, edit_layer, load_model, write_model
from hypercast4d.workspace_runtime import workspace_owner


@pytest.fixture
def spec():
    return copy.deepcopy(next(s for s in presets() if s['preset_id'] == 'tslib_dlinear'.replace('_', '-')))


def test_cli_create_insert_set_validate(tmp_path, capsys):
    path = tmp_path / 'model.yaml'
    assert main(['model', 'create', 'tslib-dlinear', '--out', str(path), '--json']) == 0
    assert json.loads(capsys.readouterr().out)['file'] == str(path)
    assert main(['layer', 'insert', str(path), '--before', 'core', '--id', 'lift', '--type', 'dense', '--set', 'units=12']) == 0
    assert main(['layer', 'insert', str(path), '--before', 'core', '--id', 'hyper', '--type', 'hyper_dense', '--set', 'algebra=tricomplex', '--set', 'units=4']) == 0
    assert main(['model', 'validate', str(path), '--cells', '10/1,20/5']) == 0
    assert [n['id'] for n in load_model(path)['layers']][:3] == ['lift', 'hyper', 'core']
    before = path.read_bytes()
    assert main(['layer', 'set', str(path), 'hyper', '--set', 'algebra=imaginary']) == 2
    assert path.read_bytes() == before
    assert main(['layer', 'set', str(path), 'lift', '--set', 'unitz=16']) == 2
    assert path.read_bytes() == before


def test_dry_run_does_not_start_backend_or_write_results(tmp_path, monkeypatch, capsys, spec):
    path = tmp_path / 'model.json'
    write_model(path, spec)
    monkeypatch.setattr('hypercast4d.cli.ensure_backend', lambda *args: pytest.fail('Backend started during dry run'))
    root = tmp_path / 'results'
    assert main(['experiment', 'run', str(path), '--results-root', str(root), '--target', 'gcp', '--gpu', 'L4', '--gpu-count', '16', '--dry-run', '--json']) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['execution']['vm_count'] == 2
    assert output['trials'] == 1
    assert not root.exists()


def test_cloud_run_requires_confirmation_before_backend(tmp_path, monkeypatch, spec):
    path = tmp_path / 'model.yaml'; write_model(path, spec)
    monkeypatch.setattr('hypercast4d.cli.ensure_backend', lambda *args: pytest.fail('Unexpected backend startup'))
    monkeypatch.setattr('sys.stdin.isatty', lambda: False)
    assert main(['experiment', 'run', str(path), '--target', 'gcp']) == 2


def test_graph_edit_preserves_layout_and_rejects_cycles(tmp_path, spec):
    graph = convert_to_graph(spec, 10, 1)
    graph['view'] = {'positions': {'inputs': {'x': 10, 'y': 20}}, 'collapsed': []}
    edge = graph['edges'][0]
    with pytest.raises(ValueError, match='already connected'):
        connect_graph(graph, edge['source'], edge['target'], edge['port'])
    with pytest.raises(ValueError, match='cycle'):
        connect_graph(graph, edge['target'], edge['source'], 'x')
    next_spec = connect_graph(graph, edge['source'], edge['target'], edge['port'], replace=True)
    assert next_spec['view'] == graph['view']
    path = tmp_path / 'graph.yaml'; write_model(path, next_spec)
    assert load_model(path)['view'] == graph['view']


def test_dense_swap_validates_dimensions_and_keeps_original(spec):
    cells = [{'window': 10, 'horizon': 1}, {'window': 20, 'horizon': 5}]
    spec = edit_layer(spec, 'insert', node_id='lift', kind='dense', params={'units': 12}, before='core', cells=cells)
    spec = edit_layer(spec, 'insert', node_id='dense', kind='dense', params={'units': 12}, before='core', cells=cells)
    result = dense_swap(spec, 'dense', 'hyper_dense', 'tricomplex', cells)
    assert result['layers'][1]['params']['units'] == 4
    assert spec['layers'][1]['type'] == 'dense'
    with pytest.raises(ValueError, match='divisible'):
        dense_swap(spec, 'lift', 'hyper_dense', 'tricomplex', cells)


def test_workspace_lock_excludes_second_owner(tmp_path):
    with workspace_owner(tmp_path, tmp_path / 'results'):
        with pytest.raises(RuntimeError, match='owns'):
            with workspace_owner(tmp_path, tmp_path / 'results'):
                pytest.fail('Two owners')


def test_locked_edit_rejected(spec):
    spec['locked'] = True
    with pytest.raises(ValueError, match='Locked'):
        edit_layer(spec, 'remove', node_id='core')


def test_invalid_draft_requires_explicit_flag(spec):
    graph = convert_to_graph(spec, 10, 1)
    graph['edges'] = []
    with pytest.raises(ValueError):
        checked(graph, [{'window': 10, 'horizon': 1}])
    _, notices = checked(graph, [{'window': 10, 'horizon': 1}], allow_invalid=True)
    assert notices


def test_cli_parameter_json_types():
    from hypercast4d.cli import assignments
    assert assignments(['bias=false', 'shape=[0, -1]', 'algebra=complex']) == {'bias': False, 'shape': [0, -1], 'algebra': 'complex'}
    with pytest.raises(ValueError):
        assignments(['p=.nan'])


def test_api_startup_lock_precedes_recovery(tmp_path):
    from hypercast4d.playground import create_app
    from fastapi.testclient import TestClient
    job = tmp_path / 'jobs/active'; job.mkdir(parents=True)
    (job / 'status.json').write_text(json.dumps({'state': 'running'}))
    (job / 'request.json').write_text('{}')
    with workspace_owner(tmp_path, tmp_path):
        app = create_app(tmp_path, tmp_path)
        assert json.loads((job / 'status.json').read_text())['state'] == 'running'
        with pytest.raises(RuntimeError, match='owns'):
            with TestClient(app):
                pass
        assert json.loads((job / 'status.json').read_text())['state'] == 'running'


def test_shared_edit_endpoint_matches_offline_edit(tmp_path, spec):
    from hypercast4d.playground import create_app
    from fastapi.testclient import TestClient
    cells = [{'window': 10, 'horizon': 1}]
    payload = {'action': 'insert', 'id': 'lift', 'kind': 'dense', 'params': {'units': 12}, 'before': 'core'}
    expected, _ = checked(edit_layer(spec, 'insert', node_id='lift', kind='dense', params={'units': 12}, before='core', cells=cells), cells)
    with TestClient(create_app(tmp_path, tmp_path)) as client:
        result = client.post('/api/v1/architectures/edit', json={'architecture': spec, 'edit': payload, 'cells': cells})
        assert result.status_code == 200, result.text
        assert result.json()['spec'] == expected


def test_graph_dense_swap_all_cells_and_bias(spec):
    cells = [{'window': 10, 'horizon': 1}, {'window': 20, 'horizon': 5}]
    spec = edit_layer(spec, 'insert', node_id='lift', kind='dense', params={'units': 12}, before='core', cells=cells)
    graph = convert_to_graph(spec, 10, 1)
    info = validate_architecture(graph, 10, 1)['graph_nodes']
    node_id = next(n for n, meta in info.items() if meta.get('label') == 'Linear' and meta['shape'][-1] == 12)
    result = dense_swap(graph, node_id, 'hyper_dense', 'complex', cells)
    node = next(n for n in result['nodes'] if n['id'] == node_id)
    assert node['params'] == {'units': 6, 'bias': True, 'algebra': 'complex'}
    assert 'module_ref' not in node


def test_draft_cannot_hide_bad_parameters(spec):
    graph = convert_to_graph(spec, 10, 1)
    with pytest.raises(ValueError):
        edit_layer(graph, 'add', node_id='bad', kind='dense', params={'units': -1}, cells=[{'window': 10, 'horizon': 1}])


def test_wait_interrupt_detaches_without_cancelling(monkeypatch):
    from hypercast4d.cli import wait_job
    def interrupted(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr('hypercast4d.cli.request', interrupted)
    value, code = wait_job('http://127.0.0.1:1', 'test-job')
    assert value == {'job_id': 'test-job', 'detached': True} and code == 130


def test_simultaneous_backend_launches_share_owner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from hypercast4d.workspace_runtime import ensure_backend, stop_backend, discover
    project = Path(__file__).resolve().parents[1]
    root = tmp_path / 'workspace'
    found = None
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(ensure_backend, project, root) for _ in range(2)]
            first, second = [future.result() for future in futures]
        found = first
        assert first['pid'] == second['pid']
        assert first['url'] == second['url']
        assert discover(project, root)['pid'] == first['pid']
        with pytest.raises(RuntimeError, match='mismatch'):
            discover(tmp_path, root)
    finally:
        found = found or discover(project, root)
        if found:
            stop_backend(found)


def test_stale_discovery_is_not_adopted(tmp_path):
    from hypercast4d.workspace_runtime import discover
    runtime = tmp_path / '.runtime'; runtime.mkdir()
    (runtime / 'server.json').write_text(json.dumps({'url': 'http://127.0.0.1:1', 'pid': 999999}))
    assert discover(tmp_path, tmp_path) is None


def test_cli_and_api_job_hashes_match(tmp_path, spec, capsys):
    from hypercast4d.playground import JobManager
    path = tmp_path / 'model.yaml'; write_model(path, spec)
    assert main(['experiment', 'run', str(path), '--preset', 'standard', '--cells', '10/1,20/5', '--epochs', '2', '--dry-run', '--json']) == 0
    dry = json.loads(capsys.readouterr().out)
    manager = JobManager(tmp_path / 'results', Path.cwd())
    job = manager.submit_validation(spec, dry['evaluation'], dry['execution'])
    assert job['request']['candidate_hash'] == dry['candidate_hash']


def test_no_new_worker_if_unregistered_default_server(monkeypatch, tmp_path):
    from hypercast4d.workspace_runtime import ensure_backend
    from io import BytesIO
    monkeypatch.setattr('hypercast4d.workspace_runtime.urllib.request.urlopen', lambda *a, **kw: BytesIO(b'{"ok":true}'))
    monkeypatch.setattr('hypercast4d.workspace_runtime.subprocess.Popen', lambda *a, **kw: pytest.fail('Started competing worker'))
    with pytest.raises(RuntimeError, match='unregistered'):
        ensure_backend(tmp_path, tmp_path / 'results/playground')


def test_controlled_initialization_and_benchmark_are_available_in_cli(tmp_path, capsys, spec):
    path = tmp_path / 'model.yaml'
    write_model(path, spec)
    assert main(['experiment', 'run', str(path), '--target', 'modal', '--gpu', 'L4',
                 '--epochs', '150', '--eval', 'initialization=matched-v1',
                 '--eval', 'benchmark=true', '--dry-run', '--json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['evaluation']['initialization'] == 'matched-v1'
    assert result['evaluation']['benchmark'] is True
    assert result['evaluation']['epochs'] == 150
