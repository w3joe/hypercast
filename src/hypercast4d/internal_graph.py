"""Observed tensor-dependency graphs for a single CPU evaluation forward pass.

Leaf modules are atomic nodes. Tensor operations between modules propagate
dependencies; multi-source operations become explicit junctions. This is a
sample execution view for the legacy v1 internal inspector, not a static proof
of every possible control-flow path.
"""

from __future__ import annotations

import threading
from typing import Any

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from .architecture import build_architecture
from .upstream_models import UpstreamSequence


_GRAPH_LOCK = threading.Lock()


def _tensors(value: Any):
    """Yield every ``torch.Tensor`` nested within ``value``."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


class DependencyRecorder(TorchDispatchMode):
    """``TorchDispatchMode`` tracing tensor provenance between leaf ``nn.Module`` calls."""

    def __init__(self) -> None:
        super().__init__()
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, str]] = []
        self.origins: dict[int, list[str]] = {}
        self.retained: list[torch.Tensor] = []  # Prevent Python object-id reuse during this short trace.
        self.active: list[tuple[str, str | None, list[str]]] = []

    def sources(self, value: Any) -> list[str]:
        """Return sorted upstream node IDs contributing to tensors in ``value``."""
        return sorted({source for tensor in _tensors(value)
                       for source in self.origins.get(id(tensor), ())})

    def remember(self, value: Any, sources: list[str]) -> None:
        """Associate every tensor in ``value`` with upstream ``sources``."""
        for tensor in _tensors(value):
            self.origins[id(tensor)] = sources
            self.retained.append(tensor)

    def node(
        self,
        label: str,
        kind: str,
        sources: list[str],
        output: Any = None,
        path: str | None = None,
        target: str | None = None,
    ) -> str:
        """Append a recorded graph node (capped at 1,500 preview nodes) and return its ID."""
        if len(self.nodes) >= 1500:
            raise ValueError('Internal graph exceeds the 1500-node preview limit')
        key = f'n{len(self.nodes)}'
        shapes = [list(t.shape) for t in _tensors(output)]
        self.nodes.append({'id': key, 'label': label, 'kind': kind, 'path': path,
                           'target': target, 'shapes': shapes})
        self.edges.extend({'source': source, 'target': key} for source in sources)
        return key

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self.active:
            return func(*args, **kwargs)
        sources = self.sources((args, kwargs))
        output = func(*args, **kwargs)
        if len(sources) > 1:
            operation = str(func).split('.')[1]
            label = {'add': 'Add', 'add_': 'Add', 'mul': 'Multiply', 'cat': 'Concatenate',
                     'sub': 'Subtract', 'div': 'Divide', 'bmm': 'Batch matmul',
                     'mm': 'Matmul'}.get(operation, operation)
            sources = [self.node(label, 'operation', sources, output)]
        self.remember(output, sources)
        return output


def trace_core(core: UpstreamSequence) -> dict[str, Any]:
    """Record a single synthetic CPU evaluation pass through ``core``.

    Parameters
    ----------
    core : UpstreamSequence
        Wrapped TSLib sequence block to trace.

    Returns
    -------
    dict[str, Any]
        Observed dependency graph with ``nodes``, ``edges``, editable
        ``targets``, ``mode="observed"``, and explanatory ``notice``.
    """
    recorder = DependencyRecorder()
    handles = []
    editable = {t['path']: t for t in core.internal_targets if not t['blocked_by']}
    for path, module in core.model.named_modules():
        if not path or list(module.children()):
            continue
        target = path if path in editable else next(
            (parent for parent in editable if path.startswith(parent + '.') and editable[parent]['override']), None)

        def before(mod, inputs, path=path, target=target):
            recorder.active.append((path, target, recorder.sources(inputs)))

        def after(mod, inputs, output):
            path, target, sources = recorder.active.pop()
            key = recorder.node(type(mod).__name__, 'module', sources, output, path, target)
            recorder.remember(output, [key])

        handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
    try:
        core.eval()
        # Deterministic synthetic history, without consulting the user's dataset.
        history = torch.randn(1, core.steps, core.width)
        key = recorder.node('History', 'input', [], history)
        recorder.remember(history, [key])
        with torch.no_grad(), recorder:
            output = core(history)
        output_id = recorder.node('Core output', 'output', recorder.sources(output), output)
        # Only nodes that contribute to the returned tensor belong in the view.
        required = {output_id}
        for edge in reversed(recorder.edges):
            if edge['target'] in required:
                required.add(edge['source'])
        return {
            'nodes': [node for node in recorder.nodes if node['id'] in required],
            'edges': [edge for edge in recorder.edges if edge['source'] in required and edge['target'] in required],
            'targets': core.internal_targets,
            'mode': 'observed',
            'notice': 'One synthetic CPU eval forward pass. Leaf modules are atomic; single-source tensor operations '
                      '(such as reshapes) are collapsed into edges. Junctions show observed multi-source operations. '
                      'This is not every possible runtime branch. Shapes use preview batch size 1.',
        }
    finally:
        for handle in handles:
            handle.remove()


def architecture_internal_graph(
    spec: dict[str, Any], layer_id: str, window: int = 10, horizon: int = 1
) -> dict[str, Any]:
    """Build ``spec`` under an isolated RNG seed and trace TSLib block ``layer_id``.

    Parameters
    ----------
    spec : dict[str, Any]
        Schema v1 architecture specification containing ``layer_id``.
    layer_id : str
        Identifier of the ``tslib_*`` layer to inspect.
    window : int, default=10
        Preview lookback window length.
    horizon : int, default=1
        Preview forecast horizon.

    Returns
    -------
    dict[str, Any]
        Observed dependency graph returned by ``trace_core``.
    """
    # Construction and tracing must not advance the application's CPU RNG.
    with _GRAPH_LOCK, torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        model = build_architecture(spec, window, horizon)
        for layer, module in zip(model.spec['layers'], model.layers):
            if layer['id'] == layer_id:
                if not isinstance(module, UpstreamSequence):
                    raise ValueError('Only TSLib model blocks have an internal graph')
                return trace_core(module)
        raise ValueError('Unknown model block id')
