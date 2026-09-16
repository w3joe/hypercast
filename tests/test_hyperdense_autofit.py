"""Shape-safe insertion must survive real graph joins, training and persistence."""
import copy
import io
import json

import pytest
import torch
from fastapi.testclient import TestClient

from hypercast4d.algebras import ALGEBRAS
from hypercast4d.architecture import architecture_hash, build_architecture, presets, validate_architecture
from hypercast4d.graph_architecture import convert_to_graph
from hypercast4d.layers import ShapePreservingHyperDense
from hypercast4d.model_editing import checked, describe_layers, edit_layer, load_model, write_model
from hypercast4d.playground import create_app
from hypercast4d.upstream_models import UPSTREAM_MODELS
from hypercast4d.visualizations import model_weights


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def source(method='tsmixer', width=4):
    spec = copy.deepcopy(next(p for p in presets() if p['preset_id'] == f'tslib-{method}'))
    spec['input']['feature_order'] = list(range(width))
    return spec


def insert(graph, edge, params=None):
    return edit_layer(graph, 'insert', node_id='autofit', kind='hyper_dense',
                      before=edge['target'], port=edge['port'], params=params,
                      cells=[{'window': 10, 'horizon': 1}])


@pytest.mark.parametrize('algebra', ALGEBRAS)
@pytest.mark.parametrize('width', [1, 3, 4, 7, 8])
@pytest.mark.parametrize('prefix', [(2,), (2, 5), (2, 3, 5)])
def test_padding_cropping_exact_reference_and_gradients(algebra, width, prefix):
    layer = ShapePreservingHyperDense(width, algebra).double()
    x = torch.randn(*prefix, width, dtype=torch.float64, requires_grad=True)
    actual = layer(x)
    expected = layer.hyper(torch.nn.functional.pad(x, (0, layer.padded_width - width)))[..., :width]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == x.shape
    assert actual.is_contiguous()
    assert actual.view(prefix[0], -1).numel() == x.numel()
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    before = layer.hyper.weight.detach().clone()
    torch.optim.Adam(layer.parameters(), lr=.001).step()
    assert not torch.equal(before, layer.hyper.weight)


@pytest.mark.parametrize('bad', [(torch.ones(2, 4),), torch.tensor(1.), torch.ones(4),
                                 torch.ones(2, 4, dtype=torch.long), torch.ones(2, 4, dtype=torch.complex64)])
def test_unsupported_connections_and_dynamic_width(bad):
    layer = ShapePreservingHyperDense(4, 'quaternion')
    with pytest.raises(ValueError, match='real floating-point tensor'):
        layer(bad)
    with pytest.raises(ValueError, match='constructed for last-axis width 4, got 7'):
        layer(torch.ones(2, 7))


@pytest.mark.parametrize('method', UPSTREAM_MODELS)
@pytest.mark.parametrize('side', ['before', 'after'])
def test_all_tslib_boundaries_across_cells_train_and_restore(method, side):
    graph = convert_to_graph(source(method, width=3))
    info = validate_architecture(graph, 10, 1)['graph_nodes']
    members = {n['id'] for n in graph['nodes'] if n.get('group') == 'layer-core'}
    edges = [e for e in graph['edges'] if (e['source'] in members) != (e['target'] in members)
             and (e['target'] in members) == (side == 'before')]
    edge = next(e for e in edges if isinstance(info[e['source']]['shape'], list)
                and len(info[e['source']]['shape']) >= 2
                and all(isinstance(v, int) for v in info[e['source']]['shape']))
    fitted = insert(graph, edge)
    for window, horizon in [(2, 1), (10, 3), (33, 7)]:
        model = build_architecture(fitted, window, horizon)
        assert model.metadata['autofit']['shape'] == model.metadata[edge['source']]['shape']
        x = torch.randn(2, window, 4)
        prediction = model(x)
        assert prediction.shape == (2, horizon) and torch.isfinite(prediction).all()
        prediction.square().mean().backward()
        hyper = model.blocks[model.bindings['autofit']].hyper
        assert hyper.weight.grad is not None and torch.isfinite(hyper.weight.grad).all()
        torch.optim.Adam(model.parameters(), lr=.001).step()
        saved = io.BytesIO()
        torch.save(model.state_dict(), saved)
        saved.seek(0)
        restored = build_architecture(json.loads(json.dumps(fitted)), window, horizon)
        restored.load_state_dict(torch.load(saved, weights_only=True))
        torch.testing.assert_close(restored.eval()(x), model.eval()(x), rtol=0, atol=0)


def test_residual_32_versus_4_repaired_without_mutating_legacy(tmp_path):
    graph = convert_to_graph(source())
    edge = next(e for e in graph['edges'] if e['target'] == 'add_1' and e['port'] == 'args/0')
    broken = insert(graph, edge, {'shape_mode': 'manual', 'units': 8})
    broken['nodes'][-1]['params'].pop('shape_mode')  # Existing saved graphs keep their old behavior.
    original = copy.deepcopy(broken)
    with pytest.raises(ValueError, match=r'Node add_1:.*32.*4.*input shapes'):
        validate_architecture(broken, 10, 1)
    cells = [{'window': 10, 'horizon': 1}, {'window': 33, 'horizon': 7}]
    repaired = edit_layer(broken, 'set', node_id='autofit', params={'shape_mode': 'preserve'}, cells=cells)
    normalized, _ = checked(repaired, cells)
    assert broken == original
    assert architecture_hash(broken) != architecture_hash(repaired)
    model = build_architecture(normalized, 10, 1)
    fit = model.metadata['autofit']['shape_fit']
    assert fit == dict(axis=-1, input_width=4, padded_width=4, units=1, output_width=4, padding=0, crop=0)
    snapshot = next(s for s in model_weights(model)['layers'] if 'autofit' in s['node_ids'])
    assert snapshot['shape_fit'] == fit
    listing = next(row for row in describe_layers(normalized, cells) if row['id'] == 'autofit')
    assert listing['hypercomplex']['units'] == 1
    assert listing['hypercomplex']['real_output_width'] == 4
    for extension in ['json', 'yaml']:
        path = tmp_path / f'model.{extension}'
        write_model(path, normalized)
        loaded, _ = checked(load_model(path), cells)
        assert architecture_hash(loaded) == architecture_hash(normalized)


@pytest.mark.parametrize('method,rank', [('tsmixer', 3), ('crossformer', 4)])
def test_internal_transposed_and_higher_rank_connections(method, rank):
    graph = convert_to_graph(source(method))
    info = validate_architecture(graph, 10, 1)['graph_nodes']
    edge = next(e for e in graph['edges'] if info.get(e['source'], {}).get('label') in {'transpose', 'permute', 'rearrange'}
                and len(info[e['source']]['shape']) == rank
                and info.get(e['target'], {}).get('label') != 'getattr')
    model = build_architecture(insert(graph, edge), 10, 1)
    assert model.metadata['autofit']['shape'] == info[edge['source']]['shape']
    output = model(torch.randn(2, 10, 4))
    output.square().mean().backward()
    assert torch.isfinite(output).all()


def test_api_insert_repair_and_reject_invalid_modes(tmp_path):
    client = TestClient(create_app(tmp_path / 'results', tmp_path))
    graph = convert_to_graph(source())
    edge = next(e for e in graph['edges'] if e['target'] == 'add_1' and e['port'] == 'args/0')
    payload = dict(architecture=graph, cells=[{'window': 10, 'horizon': 1}, {'window': 33, 'horizon': 7}],
                   edit=dict(action='insert', id='autofit', kind='hyper_dense', before=edge['target'], port=edge['port']))
    response = client.post('/api/v1/architectures/edit', json=payload)
    assert response.status_code == 200, response.text
    fitted = response.json()['spec']
    assert fitted['nodes'][-1]['params']['shape_mode'] == 'preserve'
    payload.update(architecture=fitted, edit=dict(action='set', id='autofit', params={'shape_mode': 'bogus'}))
    assert client.post('/api/v1/architectures/edit', json=payload).status_code == 422
    fitted['nodes'][-1]['params']['shape_mode'] = 'manual'
    payload['edit']['params']['shape_mode'] = 'preserve'
    response = client.post('/api/v1/architectures/edit', json=payload)
    assert response.status_code == 200, response.text
