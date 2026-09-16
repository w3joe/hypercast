"""Offline architecture editing shared by CLI and HTTP clients."""
import copy
import json
from pathlib import Path
import tempfile

import yaml

from .algebras import get_algebra
from .architecture import normalize_architecture_spec, validate_architecture, layer_catalog, _normalize_layer_params
from .graph_architecture import NEW_OPS, describe_graph, normalize_hyperdense_params


def document(raw):
    if not isinstance(raw, dict):
        raise ValueError('Architecture must be an object')
    spec = copy.deepcopy(raw.get('spec', raw))
    if 'view' in raw:
        spec['view'] = copy.deepcopy(raw['view'])
    return spec


def load_model(path):
    return document(yaml.safe_load(Path(path).read_text()))


def write_model(path, spec, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise ValueError(f'{path} exists; choose another --out path')
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(spec, indent=2, allow_nan=False) + '\n' if path.suffix == '.json'
               else yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.hypercast-', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def checked(spec, cells, *, allow_invalid=False):
    normalized = normalize_architecture_spec(spec)
    for key in ('view', 'locked', 'preset_id'):
        if key in spec:
            normalized[key] = copy.deepcopy(spec[key])
    notices = []
    for cell in cells:
        try:
            validate_architecture(normalized, cell['window'], cell['horizon'])
        except (ValueError, RuntimeError, TypeError) as error:
            if not allow_invalid:
                raise ValueError(f"{cell['window']}/{cell['horizon']}: {error}") from error
            notices.append(str(error))
    return normalized, notices


def connect_graph(spec, source, target, port, *, replace=False):
    if spec.get('schema_version') != 2:
        raise ValueError('Use model expand before editing graph connections')
    ids = {node['id'] for node in spec['nodes']}
    if source not in ids or target not in ids:
        raise ValueError('Connection endpoint is missing')
    existing = [e for e in spec['edges'] if e['target'] == target and e['port'] == port]
    if existing and not replace:
        raise ValueError('Input already connected; use --replace explicitly')
    edges = [e for e in spec['edges'] if e not in existing]
    pending, visited = [target], set()
    while pending:
        node = pending.pop()
        if node == source:
            raise ValueError('Connection would create a cycle')
        if node not in visited:
            visited.add(node)
            pending.extend(e['target'] for e in edges if e['source'] == node)
    return {**spec, 'edges': edges + [{'source': source, 'target': target, 'port': port}]}


def layer_types():
    return {item['type']: item for group in layer_catalog()['categories'] for item in group['layers']} | {
        name: {'type': name, 'label': name, 'defaults': defaults} for name, defaults in {
            'add': {}, 'multiply': {}, 'concat': {'dim': -1}, 'reshape': {'shape': [0, -1]},
            'permute': {'dims': [0, 2, 1]}, 'softmax': {'dim': -1}}.items()}


def validate_params(kind, params, *, graph=False):
    """Drafts may have shape/connectivity errors, never malformed parameters."""
    json.dumps(params, allow_nan=False)
    if graph and kind == 'hyper_dense':
        normalize_hyperdense_params(params)
    elif kind not in {'add', 'multiply', 'concat', 'reshape', 'permute', 'softmax'}:
        _normalize_layer_params(kind, params)
        if 'bias' in params and type(params['bias']) is not bool:
            raise ValueError('bias must be true or false')
    elif kind in {'concat', 'softmax'}:
        if type(params.get('dim')) is not int:
            raise ValueError('dim must be an integer')
    elif kind in {'reshape', 'permute'}:
        values = params.get('shape' if kind == 'reshape' else 'dims')
        if not isinstance(values, list) or not values or len(values) > 8 or any(type(v) is not int for v in values):
            raise ValueError('Shape / axes must be a nonempty integer list with at most 8 entries')
        if kind == 'reshape' and (min(values) < -1 or values.count(-1) > 1):
            raise ValueError('Reshape permits one inferred (-1) dimension')
        if kind == 'permute' and sorted(values) != list(range(len(values))):
            raise ValueError('Permutation must contain each axis exactly once')


def describe_layers(spec, cells):
    cell = cells[0]
    if spec.get('schema_version') == 2:
        info = describe_graph(spec, cell['window'], cell['horizon'])['graph_nodes']
        try:
            info.update(validate_architecture(spec, cell['window'], cell['horizon'])['graph_nodes'])
        except (ValueError, RuntimeError):
            pass
        rows = [{**node, **info.get(node['id'], {}), 'params': node.get('params', {})} for node in spec['nodes']]
    else:
        result = validate_architecture(spec, cell['window'], cell['horizon'])
        rows = [{**node, **(result['trace'][index] if index < len(result['trace']) else {})}
                for index, node in enumerate(spec['layers'])]
    for row in rows:
        params = {**row.get('settings', {}), **row.get('params', {})}
        if params.get('algebra'):
            dimension = get_algebra(params['algebra']).component_count
            if params.get('shape_mode') == 'preserve':
                fit = row.get('shape_fit', {})
                row['hypercomplex'] = {'components': dimension, 'units': fit.get('units'),
                                       'real_output_width': fit.get('output_width'),
                                       'shape_mode': 'preserve', 'padded_width': fit.get('padded_width')}
            else:
                units = params.get('units', params.get('out_features', 8))
                row['hypercomplex'] = {'components': dimension, 'units': units, 'real_output_width': units * dimension}
        if row.get('module_ref'):
            row['shared_calls'] = [n['id'] for n in spec['nodes'] if n.get('module_ref') == row['module_ref']]
    return rows


def dense_swap(spec, node_id, target, algebra, cells):
    """Match canvas semantics: replacing one graph call detaches its shared weights."""
    next_spec = copy.deepcopy(spec)
    graph = spec.get('schema_version') == 2
    nodes = next_spec['nodes' if graph else 'layers']
    node = next((node for node in nodes if node['id'] == node_id), None)
    if node is None:
        raise ValueError('Layer ID not found')
    sizes = []
    for cell in cells:
        if graph:
            info = validate_architecture(spec, cell['window'], cell['horizon'])['graph_nodes']
            meta = info.get(node_id, {})
            if node['kind'] not in {'dense', 'hyper_dense'} and meta.get('label') not in {'Linear', 'HyperDense'}:
                raise ValueError('Select a Dense or HyperDense layer')
            inputs = [e for e in spec['edges'] if e['target'] == node_id]
            if len(inputs) != 1:
                raise ValueError('Dense replacement requires exactly one connected input')
            input_shape, output_shape = info[inputs[0]['source']]['shape'], meta['shape']
            if not isinstance(input_shape, list) or not isinstance(output_shape, list):
                raise ValueError('Layer shapes must be validated tensors')
            width_in, width_out = input_shape[-1], output_shape[-1]
            bias = {**meta.get('settings', {}), **node.get('params', {})}.get('bias', True)
        else:
            from .architecture import build_architecture
            from .layers import HyperDense
            from torch import nn
            model = build_architecture(spec, cell['window'], cell['horizon'])
            layer = model.layers[next(i for i, n in enumerate(nodes) if n['id'] == node_id)]
            if not isinstance(layer, (nn.Linear, HyperDense)):
                raise ValueError('Select a Dense or HyperDense layer')
            dimension = layer.component_count if isinstance(layer, HyperDense) else 1
            width_in, width_out = layer.in_features * dimension, layer.out_features * dimension
            bias = layer.bias is not None
        dimension = get_algebra(algebra).component_count if target == 'hyper_dense' else 1
        if width_in % dimension or width_out % dimension:
            raise ValueError(f'Widths {width_in}→{width_out} must be divisible by {dimension}; no adapter was added')
        sizes.append((width_out // dimension, bias))
    if len(set(sizes)) != 1:
        raise ValueError('Replacement width differs across evaluation cells')
    params = {'units': sizes[0][0]}
    if graph:
        params['bias'] = sizes[0][1]
    if target == 'hyper_dense':
        params['algebra'] = algebra
    replacement = {'id': node_id, 'kind' if graph else 'type': target, 'params': params}
    for key in ('group', 'label'):
        if key in node:
            replacement[key] = node[key]
    nodes[nodes.index(node)] = replacement
    if graph:
        next_spec['edges'] = [{**e, 'port': 'x'} if e['target'] == node_id else e for e in spec['edges']]
    checked(next_spec, cells)
    return next_spec


def edit_layer(spec, action, *, node_id=None, kind=None, params=None, before=None, after=None, port=None, cells=None):
    if spec.get('locked'):
        raise ValueError('Locked model: create an editable preset copy first')
    result = copy.deepcopy(spec)
    graph = result.get('schema_version') == 2
    nodes = result['nodes' if graph else 'layers']
    params = params or {}
    existing = next((n for n in nodes if n['id'] == node_id), None)
    if action in {'set', 'replace', 'remove'} and existing is None:
        raise ValueError(f'Unknown layer ID {node_id}')
    if action in {'add', 'insert'} and (not node_id or existing):
        raise ValueError('Supply a new unique --id')
    if action == 'remove':
        if graph and result['output'] == node_id:
            raise ValueError('Choose another graph output before removing this node')
        nodes.remove(existing)
        if graph:
            result['edges'] = [e for e in result['edges'] if e['source'] != node_id and e['target'] != node_id]
            result.get('view', {}).get('positions', {}).pop(node_id, None)
        return result
    if action == 'set':
        if graph and existing['kind'] == 'source':
            meta = describe_graph(spec, cells[0]['window'], cells[0]['horizon'])['graph_nodes'][node_id]
            if set(params) - set(meta.get('settings', {})):
                raise ValueError('Unsupported internal parameter; inspect layer settings first')
            for key, value in params.items():
                expected = meta['settings'][key]
                if type(expected) in (bool, int) and type(value) is not type(expected):
                    raise ValueError(f'{key} must have type {type(expected).__name__}')
                if key in {'out_features', 'out_channels', 'hidden_size', 'num_layers', 'groups'} and not 1 <= value <= 4096:
                    raise ValueError(f'{key} must be between 1 and 4096')
                if key in {'p', 'dropout'} and (type(value) not in (int, float) or not 0 <= value < 1):
                    raise ValueError(f'{key} must be in [0, 1)')
            if 'algebra' in params:
                get_algebra(params['algebra'])
        else:
            type_name = existing['kind' if graph else 'type']
            allowed = ({'units', 'algebra', 'bias', 'shape_mode'} if graph and type_name == 'hyper_dense' else
                       set(layer_types()[type_name]['defaults']) | set(_normalize_layer_params(type_name, existing['params']) if type_name not in {'add', 'multiply', 'concat', 'reshape', 'permute', 'softmax'} else existing['params']))
            if graph and type_name in {'dense', 'hyper_dense'}:
                allowed.add('bias')
            if set(params) - allowed:
                raise ValueError(f'Unknown parameters: {sorted(set(params) - allowed)}')
            validate_params(type_name, {**existing['params'], **params}, graph=graph)
        existing['params'].update(params)
        if graph and existing.get('module_ref'):
            for node in nodes:
                if node.get('module_ref') == existing['module_ref']:
                    node['params'].update(params)
        return result
    if not kind or kind not in layer_types() or (graph and kind not in NEW_OPS):
        raise ValueError('Unsupported layer type for this architecture format')
    if action == 'replace' and kind in {'dense', 'hyper_dense'}:
        if set(params) - {'algebra'}:
            raise ValueError('Dense swaps preserve width; use layer set afterwards to change parameters')
        return dense_swap(spec, node_id, kind, params.get('algebra', 'quaternion'), cells)
    defaults = dict(layer_types()[kind]['defaults'])
    if graph and kind == 'hyper_dense':
        defaults['shape_mode'] = 'preserve'
    allowed = set(defaults)
    if kind not in {'add', 'multiply', 'concat', 'reshape', 'permute', 'softmax'}:
        allowed |= set(_normalize_layer_params(kind, defaults))
    if graph and kind in {'dense', 'hyper_dense'}:
        allowed.add('bias')
    if set(params) - allowed:
        raise ValueError(f'Unknown parameters: {sorted(set(params) - allowed)}')
    node = {'id': node_id, 'kind' if graph else 'type': kind, 'params': {**defaults, **params}}
    validate_params(kind, node['params'], graph=graph)
    if action == 'replace':
        if graph:
            ports = ['a', 'b'] if kind in {'add', 'multiply', 'concat'} else ['x']
            incoming = [e for e in result['edges'] if e['target'] == node_id]
            if len(ports) != len(incoming):
                raise ValueError('Replacement has different input ports; add and connect it explicitly')
            for edge, target_port in zip(incoming, ports):
                edge['port'] = target_port
            if existing.get('group'):
                node['group'] = existing['group']
        nodes[nodes.index(existing)] = node
    elif action == 'add':
        nodes.append(node)
    elif action == 'insert':
        anchor_id = before or after
        anchor = next((n for n in nodes if n['id'] == anchor_id), None)
        if anchor is None or bool(before) == bool(after):
            raise ValueError('Choose exactly one existing --before or --after layer')
        if not graph:
            nodes.insert(nodes.index(anchor) + bool(after), node)
        else:
            if kind in {'add', 'multiply', 'concat'}:
                raise ValueError('Add multi-input operations separately and connect their ports explicitly')
            affected = [e for e in result['edges'] if (e['target'] == before if before else e['source'] == after)]
            if before and port:
                affected = [e for e in affected if e['port'] == port]
            if len(affected) != 1:
                raise ValueError('Insertion requires one unambiguous connected edge; use --port or explicit graph connect')
            edge = affected[0]
            source, target, target_port = edge['source'], edge['target'], edge['port']
            if anchor.get('group'):
                node['group'] = anchor['group']
            nodes.append(node)
            result['edges'].remove(edge)
            result['edges'].extend([{'source': source, 'target': node_id, 'port': 'x'},
                                    {'source': node_id, 'target': target, 'port': target_port}])
    return result
