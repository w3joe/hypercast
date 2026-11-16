"""Version-two executable graph specifications and their PyTorch DAG executor.

Schema v2 represents any lowered HyperCAST or TSLib forecasting model as an
explicit directed acyclic graph (DAG) of ``source`` operators (bound to a
symbolically lowered v1 recipe) and custom palette operators (``NEW_OPS``).

Execution Invariants
--------------------
- Source bindings identify operators in a trusted, symbolically lowered template
  produced by [`lower_source`][hypercast4d.graph_lowering.lower_source].
- Only the persisted ``nodes`` and ``edges`` execute during ``forward``; the v1
  ``sources`` dictionary acts solely as a construction/constant recipe and is
  never invoked as a fallback whole-model forward pass.
- Disconnected subgraphs not reachable backward from ``spec["output"]`` are
  pruned from the topological execution order and surfaced as validation
  warnings rather than silently executed.
"""

from __future__ import annotations

import copy
import hashlib
import heapq
import json
from typing import Any

import torch
from torch import nn
from torch.fx import Node

from .algebras import get_algebra
from .graph_lowering import lower_source
from .layers import HyperDense, ShapePreservingHyperDense


REVISION = 'source-graph-1-tslib-4e938a1767106324dd753b2a44832bf870a0252e'
NEW_OPS = {'dense', 'activation', 'dropout', 'add', 'multiply', 'concat', 'flatten',
           'mean_pool', 'last_state', 'layer_norm', 'causal_conv', 'tcn', 'gru', 'lstm', 'hyper_dense',
           'reshape', 'permute', 'softmax'}


def normalize_hyperdense_params(params: dict[str, Any]) -> dict[str, Any]:
    """Validate Schema v2 ``hyper_dense`` parameters including ``shape_mode``.

    Parameters
    ----------
    params : dict[str, Any]
        Raw node parameter mapping. Supports ``algebra``, ``units``, ``bias``
        (``bool``), and ``shape_mode`` (``"manual"`` or ``"preserve"``). Absent
        ``shape_mode`` defaults to ``"manual"`` for backward compatibility.

    Returns
    -------
    dict[str, Any]
        Normalized parameter dictionary.

    Raises
    ------
    ValueError or ArchitectureError
        If ``shape_mode``, ``bias``, ``algebra``, or ``units`` are invalid.
    """
    from .architecture import _normalize_layer_params
    mode = params.get('shape_mode', 'manual')
    if mode not in ('manual', 'preserve'):
        raise ValueError('HyperDense shape_mode must be preserve or manual')
    if type(params.get('bias', True)) is not bool:
        raise ValueError('bias must be true or false')
    # Units are derived from the connection in preserve mode, not a saved preview.
    normalized = _normalize_layer_params('hyper_dense', {**params, **({'units': 1} if mode == 'preserve' else {})})
    return {**normalized, 'shape_mode': mode, 'bias': params.get('bias', True)}


def normalize_graph(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a Schema v2 executable graph specification.

    Parameters
    ----------
    raw : dict[str, Any]
        Candidate Schema v2 graph dictionary containing ``schema_version=2``,
        ``revision``, ``name``, ``sources``, ``nodes``, ``edges``, ``groups``,
        and ``output``.

    Returns
    -------
    dict[str, Any]
        Deep-copied, canonicalized Schema v2 graph specification.

    Raises
    ------
    ValueError
        If revision, bounds, node IDs, operator kinds, source bindings, port
        uniqueness, or group definitions violate Schema v2 invariants.
    """
    from .architecture import normalize_architecture_spec
    if raw.get('schema_version') != 2 or raw.get('revision') != REVISION:
        raise ValueError('Unsupported graph schema or template revision')
    name = raw.get('name', 'Untitled graph')
    if not isinstance(name, str) or not name.strip() or len(name) > 80:
        raise ValueError('Graph name must contain 1–80 characters')
    sources = raw.get('sources', {})
    if not isinstance(sources, dict) or not 1 <= len(sources) <= 16:
        raise ValueError('Graph needs 1–16 initialization sources')
    normalized_sources = {}
    for key, value in sources.items():
        if not isinstance(key, str) or not key or len(key) > 64 or not isinstance(value, dict) or value.get('schema_version', 1) != 1:
            raise ValueError('Invalid initialization source')
        normalized_sources[key] = normalize_architecture_spec(value)
    nodes, edges = raw.get('nodes'), raw.get('edges')
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 5000 or not isinstance(edges, list) or len(edges) > 30000:
        raise ValueError('Graph supports 1–5000 nodes and at most 30000 edges')
    result_nodes, ids = [], set()
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError('Invalid graph node')
        key = node.get('id')
        if not isinstance(key, str) or not key or len(key) > 160 or key in ids:
            raise ValueError('Invalid or duplicate graph node id')
        ids.add(key)
        kind = node.get('kind')
        if kind not in NEW_OPS | {'source'}:
            raise ValueError(f'{key}: unknown graph operation {kind}')
        params = node.get('params', {})
        if not isinstance(params, dict) or len(json.dumps(params, allow_nan=False)) > 8192:
            raise ValueError(f'{key}: invalid parameters')
        if kind == 'hyper_dense':
            normalize_hyperdense_params(params)
        item = {'id': key, 'kind': kind, 'params': copy.deepcopy(params)}
        if kind == 'source':
            ref = node.get('source_ref', {})
            if not isinstance(ref, dict) or ref.get('source') not in sources or not isinstance(ref.get('node'), str):
                raise ValueError(f'{key}: invalid source binding')
            item['source_ref'] = {'source': ref['source'], 'node': ref['node']}
        if 'module_ref' in node:
            if not isinstance(node['module_ref'], str) or len(node['module_ref']) > 256:
                raise ValueError(f'{key}: invalid shared-module reference')
            item['module_ref'] = node['module_ref']
        if node.get('group'):
            item['group'] = str(node['group'])
        if node.get('label'):
            item['label'] = str(node['label'])[:160]
        result_nodes.append(item)
    result_edges, ports = [], set()
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError('Invalid graph edge')
        source, target, port = edge.get('source'), edge.get('target'), edge.get('port')
        if source not in ids or target not in ids or not isinstance(port, str) or len(port) > 256:
            raise ValueError('Edge references an invalid node or port')
        if (target, port) in ports:
            raise ValueError(f'{target}: multiple connections to port {port}')
        ports.add((target, port))
        result_edges.append({'source': source, 'target': target, 'port': port})
    output = raw.get('output')
    if output not in ids:
        raise ValueError('Select an existing node as graph output')
    groups = raw.get('groups', [])
    if not isinstance(groups, list) or len(groups) > 5000:
        raise ValueError('Invalid graph groups')
    if any(not isinstance(g, dict) or not isinstance(g.get('id'), str) or not g['id'] for g in groups):
        raise ValueError('Invalid graph group')
    if len({g['id'] for g in groups}) != len(groups):
        raise ValueError('Duplicate graph group')
    return {'schema_version': 2, 'revision': REVISION, 'name': name.strip(),
            'sources': normalized_sources, 'nodes': result_nodes, 'edges': result_edges,
            'groups': [{'id': str(g['id']), 'label': str(g.get('label', g['id']))[:160]} for g in groups], 'output': output}


def _walk(value: Any, path: str):
    """Yield ``(port_path, fx_node)`` pairs nested inside FX args/kwargs/slices."""
    if isinstance(value, Node):
        yield path, value
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _walk(item, f'{path}/{i}')
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, f'{path}/{key}')
    elif isinstance(value, slice):
        for key in ('start', 'stop', 'step'):
            yield from _walk(getattr(value, key), f'{path}/{key}')


def _bind(value: Any, path: str, incoming: dict[str, str], values: dict[str, Any]) -> Any:
    """Substitute live runtime tensor values into an FX argument template."""
    if isinstance(value, Node):
        if path not in incoming:
            raise ValueError(f'Missing connection to {path}')
        return values[incoming[path]]
    if isinstance(value, tuple):
        return tuple(_bind(v, f'{path}/{i}', incoming, values) for i, v in enumerate(value))
    if isinstance(value, list):
        return [_bind(v, f'{path}/{i}', incoming, values) for i, v in enumerate(value)]
    if isinstance(value, dict):
        return {k: _bind(v, f'{path}/{k}', incoming, values) for k, v in value.items()}
    if isinstance(value, slice):
        return slice(*(_bind(getattr(value, k), f'{path}/{k}', incoming, values) for k in ('start', 'stop', 'step')))
    return value


def _attribute(root: nn.Module, path: str) -> Any:
    """Resolve a dotted attribute path on ``root``."""
    current: Any = root
    for part in path.split('.'):
        current = getattr(current, part)
    return current


def _shapes(value: Any) -> Any:
    """Summarize tensor or nested-sequence shapes for node inspector metadata."""
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return [_shapes(x) for x in value]
    return type(value).__name__


def convert_to_graph(spec: dict[str, Any], window: int = 10, horizon: int = 1) -> dict[str, Any]:
    """Lower a Schema v1 sequential architecture into an editable Schema v2 graph.

    Parameters
    ----------
    spec : dict[str, Any]
        Schema v1 pipeline specification (or an existing Schema v2 graph, which
        is returned normalized as-is).
    window : int, default=10
        Reference historical window length used during symbolic lowering.
    horizon : int, default=1
        Reference forecast horizon used during symbolic lowering.

    Returns
    -------
    dict[str, Any]
        Normalized Schema v2 graph specification with ``sources={"s0": spec}``,
        explicit ``nodes``, port-addressed ``edges``, stage ``groups``, and
        ``output`` node ID.

    Examples
    --------
    >>> from hypercast4d.architecture import presets
    >>> from hypercast4d.graph_architecture import convert_to_graph
    >>> v1 = next(p for p in presets() if p["preset_id"] == "paper-quaternion")
    >>> v2 = convert_to_graph(v1, window=10, horizon=1)
    >>> v2["schema_version"]
    2
    """
    from .architecture import build_architecture, normalize_architecture_spec
    spec = normalize_architecture_spec(spec)
    if spec['schema_version'] == 2:
        return spec
    with torch.random.fork_rng(devices=[]):
        source = lower_source(build_architecture(spec, window, horizon))
    nodes, edges, groups = [], [], {}
    for node in source.graph.nodes:
        key = node.name
        path = node.meta.get('owner', str(node.target) if node.op in {'call_module', 'get_attr'} else '')
        if not path.startswith('layers.') and str(node.target).startswith('layers.'):
            path = str(node.target)
        group = None
        if path.startswith('layers.'):
            index = int(path.split('.')[1])
            layer = spec['layers'][index]
            group = f'layer-{layer["id"]}'
            groups[group] = {'id': group, 'label': layer['type'].replace('tslib_', '')}
        item = {'id': key, 'kind': 'source', 'source_ref': {'source': 's0', 'node': key}, 'params': {}}
        if group:
            item['group'] = group
        if node.op == 'call_module' or (node.op == 'get_attr' and isinstance(_attribute(source, str(node.target)), torch.Tensor)):
            item['module_ref'] = f's0:{node.target}'
        nodes.append(item)
        for port, parent in [*_walk(node.args, 'args'), *_walk(node.kwargs, 'kwargs')]:
            edges.append({'source': parent.name, 'target': key, 'port': port})
    return normalize_graph({'schema_version': 2, 'revision': REVISION, 'name': spec['name'],
                            'sources': {'s0': spec}, 'nodes': nodes, 'edges': edges,
                            'groups': list(groups.values()), 'output': nodes[-1]['id']})


def _bounded(value: Any, name: str, maximum: int = 4096) -> int:
    """Validate an integer module setting in ``[1, maximum]``."""
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f'{name} must be an integer from 1 to {maximum}')
    return value


def module_settings(module: nn.Module) -> dict[str, Any]:
    """Extract editable hyperparameter settings from a leaf ``nn.Module``.

    Parameters
    ----------
    module : torch.nn.Module
        Leaf PyTorch module inspected from a lowered source template.

    Returns
    -------
    dict[str, Any]
        Editable settings dictionary (empty when ``module`` has no user-tunable
        parameters).
    """
    if isinstance(module, HyperDense):
        return {'out_features': module.out_features, 'algebra': module.algebra.name, 'bias': module.bias is not None}
    if isinstance(module, nn.Linear):
        return {'out_features': module.out_features, 'bias': module.bias is not None}
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        return {'out_channels': module.out_channels, 'kernel_size': list(module.kernel_size),
                'stride': list(module.stride), 'padding': module.padding if isinstance(module.padding, str) else list(module.padding),
                'dilation': list(module.dilation), 'groups': module.groups, 'bias': module.bias is not None}
    if isinstance(module, nn.Dropout):
        return {'p': module.p}
    if isinstance(module, (nn.GRU, nn.LSTM)):
        return {'hidden_size': module.hidden_size, 'num_layers': module.num_layers, 'dropout': module.dropout}
    if isinstance(module, nn.LeakyReLU):
        return {'negative_slope': module.negative_slope}
    return {}


def adapt_module(
    original: nn.Module, params: dict[str, Any], inputs: tuple[Any, ...]
) -> nn.Module:
    """Clone or re-instantiate a lowered source leaf module for new widths/settings.

    When upstream graph edits alter the incoming feature/channel width or user
    parameters override ``module_settings(original)``, a fresh compatible module
    is constructed; otherwise ``original`` is deep-copied with its initialized
    weights intact.

    Parameters
    ----------
    original : torch.nn.Module
        Template submodule from the lowered source recipe.
    params : dict[str, Any]
        Requested parameter overrides stored on the graph node.
    inputs : tuple[Any, ...]
        Actual input arguments observed during the dry-run construction pass.

    Returns
    -------
    torch.nn.Module
        Adapted or deep-copied module ready for registration in
        ``GraphForecaster.blocks``.
    """
    settings = module_settings(original)
    if set(params) - set(settings):
        raise ValueError(f'Unsupported settings for {type(original).__name__}: {set(params) - set(settings)}')
    requested = {**settings, **params}
    x = inputs[0] if inputs else None
    if isinstance(original, HyperDense):
        algebra = get_algebra(requested['algebra'])
        dimension = algebra.component_count
        if x.shape[-1] % dimension:
            raise ValueError(
                f'HyperDense input width must be divisible by {dimension} '
                f'for {algebra.name}'
            )
        if x.shape[-1] != original.in_features * dimension or requested != settings:
            return HyperDense(x.shape[-1] // dimension, _bounded(requested['out_features'], 'out_features'), algebra, bool(requested['bias']))
    elif isinstance(original, nn.Linear):
        width = _bounded(requested['out_features'], 'out_features')
        if x.shape[-1] != original.in_features or requested != settings:
            return nn.Linear(x.shape[-1], width, bias=bool(requested['bias']))
    elif isinstance(original, (nn.Conv1d, nn.Conv2d)):
        width = _bounded(requested['out_channels'], 'out_channels')
        if x.shape[1] != original.in_channels or requested != settings:
            return type(original)(x.shape[1], width, kernel_size=requested['kernel_size'], stride=requested['stride'],
                                  padding=requested['padding'], dilation=requested['dilation'],
                                  groups=_bounded(requested['groups'], 'groups'), bias=bool(requested['bias']),
                                  padding_mode=original.padding_mode)
    elif isinstance(original, nn.LayerNorm) and tuple(x.shape[-len(original.normalized_shape):]) != original.normalized_shape:
        return nn.LayerNorm(tuple(x.shape[-len(original.normalized_shape):]), eps=original.eps,
                            elementwise_affine=original.elementwise_affine)
    elif isinstance(original, nn.Dropout):
        p = float(requested['p'])
        if not 0 <= p <= .95:
            raise ValueError('Dropout must be between 0 and 0.95')
        if p != original.p:
            return nn.Dropout(p)
    elif isinstance(original, (nn.GRU, nn.LSTM)):
        if x.shape[-1] != original.input_size or requested != settings:
            return type(original)(x.shape[-1], _bounded(requested['hidden_size'], 'hidden_size'),
                                  num_layers=_bounded(requested['num_layers'], 'num_layers', 8),
                                  dropout=float(requested['dropout']), batch_first=original.batch_first,
                                  bidirectional=original.bidirectional, bias=original.bias)
    elif isinstance(original, nn.LeakyReLU) and requested != settings:
        return nn.LeakyReLU(float(requested['negative_slope']), inplace=False)
    return copy.deepcopy(original)


class TensorState(nn.Module):
    """Container registering learned tensors as parameters and constants as buffers."""

    def __len__(self) -> int:
        return len(self._parameters) + len(self._buffers)

    def __getitem__(self, key: str) -> torch.Tensor:
        return getattr(self, key)

    def add(self, key: str, value: torch.Tensor) -> None:
        """Clone and register ``value`` as a parameter or non-learned buffer."""
        if isinstance(value, nn.Parameter):
            self.register_parameter(key, nn.Parameter(value.detach().clone(), requires_grad=value.requires_grad))
        else:
            self.register_buffer(key, value.detach().clone())


class GraphForecaster(nn.Module):
    """Executable PyTorch DAG compiled from a Schema v2 graph specification.

    During construction, ``GraphForecaster`` lowers each initialization recipe in
    ``spec["sources"]``, computes a deterministic topological schedule pruned to
    ancestors of ``spec["output"]``, and executes a synthetic ``(2, window, features)``
    dry run under a forked RNG to materialize active submodules, verify port
    shapes, and populate inspector metadata.

    Parameters
    ----------
    spec : dict[str, Any]
        Schema v2 graph specification.
    window : int
        Historical lookback window length ``W``.
    horizon : int
        Forecast horizon ``H``.
    features : int, default=4
        Number of input series per time step.

    Attributes
    ----------
    blocks : torch.nn.ModuleDict
        Active trainable/callable leaf modules keyed by ``"m0"``, ``"m1"``, ...
    constants : TensorState
        Registered tensor parameters and buffers referenced via ``get_attr``.
    order : list[str]
        Deterministic topological execution order of active node IDs.
    metadata : dict[str, dict[str, Any]]
        Per-node port, label, setting, and validated tensor-shape records.
    """

    def __init__(
        self, spec: dict[str, Any], window: int, horizon: int, features: int = 4
    ) -> None:
        super().__init__()
        from .architecture import build_architecture
        self.spec = normalize_graph(spec)
        self.window, self.horizon, self.features = window, horizon, features
        self.blocks = nn.ModuleDict()
        self.constants = TensorState()
        self.bindings, self.references, self.metadata = {}, {}, {}
        # Not registered: these are construction recipes, never trained/executed
        # as whole-model forwards. Active copied leaves register under blocks.
        sources = {key: lower_source(build_architecture(value, window, horizon, features))
                   for key, value in self.spec['sources'].items()}
        object.__setattr__(self, '_sources', sources)
        self.templates = {key: {n.name: n for n in value.graph.nodes} for key, value in sources.items()}
        self.nodes = {n['id']: n for n in self.spec['nodes']}
        for node in self.nodes.values():
            if node['kind'] == 'source' and node['source_ref']['node'] not in self.templates[node['source_ref']['source']]:
                raise ValueError(f"Node {node['id']}: source binding is unavailable")
        self.incoming = {key: {} for key in self.nodes}
        for edge in self.spec['edges']:
            self.incoming[edge['target']][edge['port']] = edge['source']
        self.order = self._order()
        self._constructing = True
        self._param_origins = {}
        self.custom_references = {}
        self.shared_types = {}
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            self.eval()
            self.forward(torch.zeros(2, window, features))
        self._constructing = False
        self.train()
        self.receptive_field = window  # Conservative bound for arbitrary DAGs.

    def _order(self) -> list[str]:
        """Compute deterministic Kahn topological order pruned to ``output`` ancestors."""
        parents = {key: set(inputs.values()) for key, inputs in self.incoming.items()}
        children = {key: [] for key in parents}
        for key, values in parents.items():
            for parent in values:
                children[parent].append(key)
        positions = {key: i for i, key in enumerate(self.nodes)}
        queue = [(positions[key], key) for key in parents if not parents[key]]
        heapq.heapify(queue)
        ordered = []
        while queue:
            _, key = heapq.heappop(queue)
            ordered.append(key)
            for child in children[key]:
                parents[child].discard(key)
                if not parents[child]:
                    heapq.heappush(queue, (positions[child], child))
        if len(ordered) != len(self.nodes):
            raise ValueError('Graph contains a cycle; use a recurrent operation instead')
        needed = {self.spec['output']}
        for key in reversed(ordered):
            if key in needed:
                needed.update(self.incoming[key].values())
        return [key for key in ordered if key in needed]

    def _source(self, node, values, inputs):
        ref = node['source_ref']
        template = self.templates[ref['source']].get(ref['node'])
        if template is None:
            raise ValueError('Source binding is unavailable for this configuration')
        if template.op != 'call_module' and node['params']:
            raise ValueError('Replace this structural operation with a palette node to change its behavior')
        incoming = self.incoming[node['id']]
        ports = [p for p, _ in [*_walk(template.args, 'args'), *_walk(template.kwargs, 'kwargs')]]
        if set(incoming) != set(ports):
            raise ValueError(f'Expected ports {ports}; connected {list(incoming)}')
        args = _bind(template.args, 'args', incoming, values)
        kwargs = _bind(template.kwargs, 'kwargs', incoming, values)
        root = self._sources[ref['source']]
        if template.op == 'placeholder':
            value, label = inputs, 'Input'
        elif template.op == 'output':
            value, label = args[0], 'Forecast output'
        elif template.op == 'get_attr':
            value = self.training if template.target == 'training' else self._constant(ref['source'], str(template.target), root, node)
            label = f'State · {template.target}'
        elif template.op == 'call_module':
            original = root.get_submodule(str(template.target))
            module = self._module(node, original, args)
            value, label = module(*args, **kwargs), type(module).__name__
        elif template.op == 'call_function':
            if node['params']:
                raise ValueError('Replace this operation with a palette node to change its behavior')
            value = template.target(*args, **kwargs)
            label = getattr(template.target, '__name__', str(template.target))
        elif template.op == 'call_method':
            if node['params']:
                raise ValueError('Replace this operation with a palette node to change its behavior')
            value = getattr(args[0], str(template.target))(*args[1:], **kwargs)
            label = str(template.target)
        else:
            raise ValueError('Unsupported source operation')
        if self._constructing:
            settings = module_settings(self.blocks[self.bindings[node['id']]]) if template.op == 'call_module' else {}
            self.metadata[node['id']] = {'label': label, 'ports': ports, 'shape': _shapes(value),
                                         'settings': settings, 'source_path': str(template.target),
                                         'category': template.op}
        return value

    def _module(self, node, original, args):
        key = node.get('module_ref', node['id'])
        if key in self.custom_references:
            raise ValueError('Cannot share a source module with a custom operation; replace both first')
        signature = (type(original), repr(original))
        if key in self.shared_types and self.shared_types[key] != signature:
            raise ValueError('Shared modules must have matching operator types and initial dimensions')
        self.shared_types[key] = signature
        if node['id'] not in self.bindings:
            if key not in self.references:
                registry = f'm{len(self.blocks)}'
                module = adapt_module(original, node['params'], args)
                module.train(self.training)
                self.blocks[registry] = module
                self.references[key] = (registry, node['params'])
                ref = node.get('source_ref')
                if ref:
                    target = str(self.templates[ref['source']][ref['node']].target)
                    for name, parameter in module.named_parameters():
                        self._param_origins[(ref['source'], f'{target}.{name}')] = parameter
            registry, params = self.references[key]
            if params != node['params']:
                raise ValueError('Shared module calls must have identical settings')
            self.bindings[node['id']] = registry
        return self.blocks[self.bindings[node['id']]]

    def _constant(self, source, path, root, node):
        key = 'state:' + node.get('module_ref', f'{source}:{path}')
        binding = self.bindings.get(key)
        if binding is None:
            value = _attribute(root, path)
            if not isinstance(value, torch.Tensor):
                return value
            binding = f'p{len(self.constants)}'
            self.constants.add(binding, value)
            self.bindings[key] = binding
            if isinstance(value, nn.Parameter):
                self._param_origins[(source, path)] = self.constants[binding]
        return self.constants[binding]

    def _new(self, node, values):
        from .architecture import _build_layer, _normalize_layer_params, TensorShape
        kind, params = node['kind'], node['params']
        ports = ['a', 'b'] if kind in {'add', 'multiply', 'concat'} else ['x']
        incoming = self.incoming[node['id']]
        if set(incoming) != set(ports):
            raise ValueError(f'Expected ports {ports}')
        args = [values[incoming[p]] for p in ports]
        if kind in {'add', 'multiply'}:
            if args[0].shape != args[1].shape:
                raise ValueError('Join inputs must have equal shapes; add an explicit projection')
            value = args[0] + args[1] if kind == 'add' else args[0] * args[1]
        elif kind == 'concat':
            value = torch.cat(args, dim=int(params.get('dim', -1)))
        elif kind == 'reshape':
            shape = params.get('shape', [0, -1])
            if not isinstance(shape, list) or not 1 <= len(shape) <= 8 or any(type(d) is not int or d < -1 or d > 1_000_000 for d in shape):
                raise ValueError('Reshape shape must be 1–8 dimensions; 0 copies the corresponding input dimension, -1 infers it')
            value = args[0].reshape([args[0].shape[i] if d == 0 else d for i, d in enumerate(shape)])
        elif kind == 'permute':
            dims = params.get('dims', [0, 2, 1])
            if not isinstance(dims, list) or any(type(d) is not int for d in dims) or sorted(dims) != list(range(args[0].ndim)):
                raise ValueError('Permute dims must list every input axis exactly once')
            value = args[0].permute(*dims)
        elif kind == 'softmax':
            value = torch.softmax(args[0], dim=int(params.get('dim', -1)))
        else:
            if node['id'] not in self.bindings:
                key = node.get('module_ref', node['id'])
                if key in self.references:
                    raise ValueError('Cannot share a custom operation with a source module; replace both first')
                shared = self.custom_references.get(key)
                if shared:
                    if shared[:2] != (kind, params):
                        raise ValueError('Shared custom operations must have identical types and settings')
                    self.bindings[node['id']] = shared[2]
            if node['id'] not in self.bindings:
                p = dict(params)
                if kind == 'dense' and p.get('units') == 'horizon':
                    p['units'] = self.horizon
                normalized = normalize_hyperdense_params(p) if kind == 'hyper_dense' else _normalize_layer_params(kind, p)
                x = args[0]
                preserve = kind == 'hyper_dense' and normalized['shape_mode'] == 'preserve'
                if preserve:
                    ShapePreservingHyperDense.validate_input(x)
                shape = TensorShape('sequence', x.shape[-1], x.shape[1]) if x.ndim == 3 else TensorShape('vector', x.shape[-1])
                if kind in {'dense', 'hyper_dense'}:
                    if x.ndim < 2:
                        raise ValueError('Dense layers require a batch axis and a feature axis')
                    if type(p.get('bias', True)) is not bool:
                        raise ValueError('bias must be true or false')
                    if kind == 'hyper_dense':
                        algebra = get_algebra(normalized['algebra'])
                        dimension = algebra.component_count
                        if not preserve and x.shape[-1] % dimension:
                            raise ValueError(
                                f'HyperDense input width must be divisible by '
                                f'{dimension} for {algebra.name}'
                            )
                        module = (ShapePreservingHyperDense(x.shape[-1], algebra, bias=p.get('bias', True)) if preserve
                                  else HyperDense(x.shape[-1] // dimension, normalized['units'], algebra, bias=p.get('bias', True)))
                    else:
                        module = nn.Linear(x.shape[-1], normalized['units'], bias=p.get('bias', True))
                else:
                    module, _, _ = _build_layer({'type': kind, 'params': normalized}, shape)
                registry = f'm{len(self.blocks)}'
                module.train(self.training)
                self.blocks[registry] = module
                self.bindings[node['id']] = registry
                self.custom_references[key] = (kind, params, registry)
            value = self.blocks[self.bindings[node['id']]](*args)
        if self._constructing:
            self.metadata[node['id']] = {'label': kind, 'ports': ports, 'shape': _shapes(value),
                                         'settings': params, 'category': 'custom', 'source_path': None}
            module = self.blocks[self.bindings[node['id']]] if node['id'] in self.bindings else None
            if isinstance(module, ShapePreservingHyperDense):
                self.metadata[node['id']]['shape_fit'] = module.shape_fit
        return value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Execute active graph nodes in topological order on ``[B, window, features]``.

        Parameters
        ----------
        inputs : torch.Tensor
            Historical input batch of shape ``(batch, window, features)``.

        Returns
        -------
        torch.Tensor
            Forecast tensor of shape ``(batch, horizon)``.

        Raises
        ------
        ValueError
            If any node raises a shape/port error or the designated output node
            does not produce shape ``(batch, horizon)``.
        """
        values: dict[str, Any] = {}
        for key in self.order:
            node = self.nodes[key]
            try:
                values[key] = self._source(node, values, inputs) if node['kind'] == 'source' else self._new(node, values)
            except Exception as error:
                shapes = {port: _shapes(values[parent]) for port, parent in self.incoming[key].items() if parent in values}
                raise ValueError(f'Node {key}: {error}; input shapes: {shapes}') from error
        result = values[self.spec['output']]
        if not isinstance(result, torch.Tensor) or result.shape != (inputs.shape[0], self.horizon):
            raise ValueError(f'Forecast output must be [batch, {self.horizon}], got {_shapes(result)}')
        return result


def validate_graph(
    spec: dict[str, Any], window: int, horizon: int, features: int = 4
) -> dict[str, Any]:
    """Compile and dry-run ``spec`` to verify port wiring and tensor shapes.

    Parameters
    ----------
    spec : dict[str, Any]
        Schema v2 graph specification.
    window : int
        Historical window length ``W``.
    horizon : int
        Forecast horizon ``H``.
    features : int, default=4
        Number of input channels.

    Returns
    -------
    dict[str, Any]
        Validation report with ``valid=True``, normalized ``spec``,
        ``parameters``, ``graph_nodes`` metadata, and disconnected-node
        ``warnings``.
    """
    from .models import parameter_count
    with torch.random.fork_rng(devices=[]):
        model = GraphForecaster(spec, window, horizon, features)
    return {'valid': True, 'spec': model.spec, 'input_shape': f'[B, {window}, {features}]',
            'output_shape': f'[B, {horizon}]', 'parameters': parameter_count(model),
            'receptive_field': window, 'trace': [], 'graph_nodes': model.metadata,
            'warnings': [f'{n}: disconnected from forecast output' for n in model.nodes if n not in model.order]}


def describe_graph(
    spec: dict[str, Any], window: int = 10, horizon: int = 1
) -> dict[str, Any]:
    """Return static port and parameter contracts without requiring valid shapes.

    Used by the Builder UI and CLI to inspect and repair incomplete drafts whose
    edges or intermediate tensor dimensions do not yet validate.

    Parameters
    ----------
    spec : dict[str, Any]
        Schema v2 graph specification.
    window : int, default=10
        Reference window length for lowering ``sources``.
    horizon : int, default=1
        Reference forecast horizon for lowering ``sources``.

    Returns
    -------
    dict[str, Any]
        Mapping ``{"graph_nodes": {node_id: {...}}}`` describing labels, input
        ports, editable settings, and FX operator categories.
    """
    from .architecture import build_architecture
    spec = normalize_graph(spec)
    with torch.random.fork_rng(devices=[]):
        sources = {key: lower_source(build_architecture(value, window, horizon)) for key, value in spec['sources'].items()}
    templates = {key: {n.name: n for n in source.graph.nodes} for key, source in sources.items()}
    metadata = {}
    for node in spec['nodes']:
        kind = node['kind']
        if kind != 'source':
            metadata[node['id']] = {'label': kind, 'ports': ['a', 'b'] if kind in {'add', 'multiply', 'concat'} else ['x'],
                                    'settings': node['params'], 'category': 'custom', 'shape': None}
            continue
        ref = node['source_ref']
        template = templates[ref['source']].get(ref['node'])
        if template is None:
            raise ValueError(f"Node {node['id']}: source binding is unavailable")
        module = sources[ref['source']].get_submodule(str(template.target)) if template.op == 'call_module' else None
        metadata[node['id']] = {'label': type(module).__name__ if module else getattr(template.target, '__name__', str(template.target)),
            'ports': [p for p, _ in [*_walk(template.args, 'args'), *_walk(template.kwargs, 'kwargs')]],
            'settings': module_settings(module) if module else {}, 'category': template.op, 'source_path': str(template.target), 'shape': None}
    return {'graph_nodes': metadata}


def graph_hash(
    spec: dict[str, Any], evaluation: dict[str, Any] | None = None
) -> str:
    """Compute a layout-invariant SHA-256 hash of a Schema v2 graph.

    Strips presentation-only attributes (``name``, ``groups``, node ``group`` and
    ``label``) and canonicalizes edge ordering while preserving ``nodes`` array
    order (which tie-breaks topological scheduling of independent branches).

    Parameters
    ----------
    spec : dict[str, Any]
        Schema v2 graph specification.
    evaluation : dict[str, Any] or None, default=None
        Optional evaluation configuration to bind into the hash.

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 digest.
    """
    canonical = normalize_graph(spec)
    canonical.pop('name')
    canonical.pop('groups')
    for source in canonical['sources'].values():
        source.pop('name', None)
    for node in canonical['nodes']:
        node.pop('group', None)
        node.pop('label', None)
    # Array order is the deterministic tie-break for independent stochastic
    # branches, so it is execution state rather than canvas layout.
    canonical['edges'].sort(key=lambda x: (x['target'], x['port'], x['source']))
    return hashlib.sha256(json.dumps({'architecture': canonical, 'evaluation': evaluation}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
