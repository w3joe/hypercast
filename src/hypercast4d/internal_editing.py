"""Shape-preserving edits to actual dense modules inside pinned model cores.

The original module hierarchy and forward connections are retained. This is not
an unrestricted graph compiler: only ``nn.Linear`` and pure dense
``nn.Sequential`` subtrees can be replaced. No user-supplied Python or arbitrary
attribute paths execute.
"""

from __future__ import annotations

from collections.abc import Iterable
import math
import re
from typing import Any

from torch import nn


ACTIVATIONS = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU,
               'tanh': nn.Tanh, 'linear': nn.Identity}
POINTWISE = (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Identity, nn.Dropout)


def normalize_internal_overrides(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate Schema v1 ``internal_overrides`` mappings for TSLib model blocks.

    Parameters
    ----------
    raw : dict[str, Any]
        Mapping from dotted submodule paths (for example ``"model.0.temporal"``)
        to override specifications with optional ``hidden_units`` (up to 4 ints
        in ``[1, 512]``), ``activation``, ``dropout``, and ``bias``.

    Returns
    -------
    dict[str, dict[str, Any]]
        Canonicalized override dictionary rejected if parent and child paths
        overlap.
    """
    if not isinstance(raw, dict) or len(raw) > 64:
        raise ValueError('internal_overrides must be an object with at most 64 edits')
    if any(not isinstance(path, str) for path in raw):
        raise ValueError('Invalid internal module path')
    result = {}
    for path, edit in sorted(raw.items()):
        if not isinstance(path, str) or len(path) > 256 or not re.fullmatch(r'[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*', path):
            raise ValueError('Invalid internal module path')
        if not isinstance(edit, dict) or set(edit) - {'hidden_units', 'activation', 'dropout', 'bias'}:
            raise ValueError(f'{path}: unsupported internal edit fields')
        units = edit.get('hidden_units', [])
        if not isinstance(units, list) or len(units) > 4 or any(type(n) is not int or not 1 <= n <= 512 for n in units):
            raise ValueError(f'{path}: use up to four hidden widths between 1 and 512')
        activation = edit.get('activation', 'relu')
        if not isinstance(activation, str) or activation not in ACTIVATIONS:
            raise ValueError(f'{path}: unsupported activation')
        dropout = edit.get('dropout', 0.)
        if type(dropout) not in (float, int) or not math.isfinite(dropout) or not 0 <= dropout <= .95:
            raise ValueError(f'{path}: dropout must be between 0 and 0.95')
        bias = edit.get('bias', True)
        if type(bias) is not bool:
            raise ValueError(f'{path}: bias must be boolean')
        result[path] = {'hidden_units': units[:], 'activation': activation, 'dropout': float(dropout), 'bias': bias}
    for path in result:
        if any(path.startswith(parent + '.') for parent in result):
            raise ValueError(f'{path}: overlapping parent and child internal edits; reset one first')
    return result


def dense_targets(
    model: nn.Module, excluded: Iterable[str] = ()
) -> list[dict[str, Any]]:
    """Enumerate editable ``Linear`` and pure-dense ``Sequential`` subtrees in ``model``.

    Parameters
    ----------
    model : torch.nn.Module
        Root upstream TSLib model instance.
    excluded : Iterable[str], default=()
        Dotted submodule path prefixes to skip (such as unused calendar
        embeddings).

    Returns
    -------
    list[dict[str, Any]]
        Metadata records with ``path``, ``kind`` (``"dense"`` or
        ``"dense_stack"``), ``in_features``, ``out_features``, ``hidden_units``,
        and human-readable ``structure``.
    """
    targets = []
    for path, module in model.named_modules():
        if not path or any(path == prefix or path.startswith(prefix + '.') for prefix in excluded):
            continue
        children = list(module.children())
        if isinstance(module, nn.Linear):
            linears, sequence = [module], [module]
        elif (isinstance(module, nn.Sequential) and children
              and isinstance(children[0], nn.Linear)
              and all(isinstance(child, (nn.Linear, *POINTWISE)) for child in children)):
            linears = [child for child in children if isinstance(child, nn.Linear)]
            sequence = children
        else:
            continue
        targets.append({
            'path': path, 'kind': 'dense' if isinstance(module, nn.Linear) else 'dense_stack',
            'in_features': linears[0].in_features, 'out_features': linears[-1].out_features,
            'hidden_units': [layer.out_features for layer in linears[:-1]],
            'structure': [f'Linear({child.in_features} → {child.out_features})' if isinstance(child, nn.Linear)
                          else repr(child) for child in sequence],
        })
    return targets


def apply_internal_overrides(
    model: nn.Module,
    overrides: dict[str, Any],
    excluded: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Replace allowlisted dense subtrees in ``model`` while preserving boundary widths.

    Parameters
    ----------
    model : torch.nn.Module
        Target PyTorch model modified in place.
    overrides : dict[str, Any]
        Raw or normalized ``internal_overrides`` mapping.
    excluded : Iterable[str], default=()
        Submodule path prefixes excluded from editing.

    Returns
    -------
    list[dict[str, Any]]
        Updated ``dense_targets`` list annotated with active ``override`` and
        ``blocked_by`` conflict markers.
    """
    targets = dense_targets(model, excluded)
    indexed = {target['path']: target for target in targets}
    edits = normalize_internal_overrides(overrides)
    for path, edit in edits.items():
        if path not in indexed:
            raise ValueError(f'{path}: not an editable dense module in this model configuration')
        target = indexed[path]
        widths = [target['in_features'], *edit['hidden_units'], target['out_features']]
        layers = []
        for index, (width_in, width_out) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(width_in, width_out, bias=edit['bias']))
            if index < len(widths) - 2:
                layers.extend([ACTIVATIONS[edit['activation']](), nn.Dropout(edit['dropout'])])
        parent_path, _, key = path.rpartition('.')
        parent = model.get_submodule(parent_path) if parent_path else model
        # Replace an existing registered module only. Paths were allowlisted from
        # named_modules above; never allow creation of arbitrary attributes.
        parent._modules[key] = nn.Sequential(*layers)
    for target in targets:
        path = target['path']
        target['override'] = edits.get(path)
        target['blocked_by'] = next((edited for edited in edits if edited != path and
                                    (path.startswith(edited + '.') or edited.startswith(path + '.'))), None)
    return targets
