"""Bounded, exact-value visualization artifacts; no synthetic evaluation data."""
from __future__ import annotations

import csv
import json
import math
from collections import deque
from pathlib import Path

import torch

from .layers import HyperDense, ShapePreservingHyperDense


def weight_snapshot(layer: HyperDense, name: str) -> dict:
    weight = layer.weight.detach()
    ins = torch.linspace(0, layer.in_features - 1, min(8, layer.in_features)).long()
    outs = torch.linspace(0, layer.out_features - 1, min(8, layer.out_features)).long()
    components = weight[:, ins.to(weight.device)][:, :, outs.to(weight.device)].cpu()
    constants = layer.algebra.constants.to(components)
    effective = torch.einsum('bac,aio->bico', constants, components).reshape(
        layer.component_count * len(ins), layer.component_count * len(outs))
    if not torch.isfinite(components).all() or not torch.isfinite(effective).all():
        return {'name': name, 'error': 'Weights contain non-finite values'}
    return dict(name=name, algebra=layer.algebra.name, dimension=layer.component_count,
                in_features=layer.in_features, out_features=layer.out_features,
                input_indices=ins.tolist(), output_indices=outs.tolist(),
                components=components.tolist(), effective=effective.tolist(),
                weight_parameters=weight.numel(), dense_parameters=layer.component_count * weight.numel())


def model_weights(model: torch.nn.Module) -> dict:
    layers = [(name, module) for name, module in model.named_modules() if isinstance(module, HyperDense)]
    snapshots = [weight_snapshot(module, name) for name, module in layers[:32]]
    for snapshot in snapshots:
        parent_name = snapshot['name'].rsplit('.', 1)[0] if '.' in snapshot['name'] else ''
        parent = model.get_submodule(parent_name)
        if isinstance(parent, ShapePreservingHyperDense):
            snapshot['shape_fit'] = parent.shape_fit
        snapshot['node_ids'] = [node for node, binding in getattr(model, 'bindings', {}).items()
                                if snapshot['name'] == f'blocks.{binding}' or snapshot['name'].startswith(f'blocks.{binding}.')]
    return {'layers': snapshots,
            'omitted_layers': max(0, len(layers) - 32)}


def save_weights(destination: Path, model: torch.nn.Module, **identity) -> None:
    path = destination / 'weights.json'
    records = json.loads(path.read_text()) if path.exists() else {}
    key = '/'.join(str(identity[k]) for k in ('window', 'horizon', 'seed', 'fold'))
    records[key] = {**identity, 'source': 'trained', **model_weights(model)}
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(records, allow_nan=False))
    temporary.replace(path)


def read_forecasts(path: Path, *, window: int, horizon: int, seed: int, fold: int, lead: int, trial: str = '') -> dict:
    rows = deque(maxlen=2000)
    total = invalid = 0
    with path.open(newline='', encoding='utf-8') as handle:
        for raw in csv.DictReader(handle):
            try:
                if (raw.get('trial_id') or '') != trial:
                    continue
                if any(float(raw[k]) != value for k, value in dict(window=window, horizon=horizon, seed=seed, fold=fold, lead=lead).items()):
                    continue
                row = {k: float(raw[k]) for k in ('origin_row', 'actual', 'prediction', 'persistence')}
                if not all(math.isfinite(v) for v in row.values()):
                    invalid += 1
                    continue
                rows.append({**row, 'target_date': raw['target_date'], 'forecast_origin': raw['forecast_origin']})
                total += 1
            except (KeyError, TypeError, ValueError):
                invalid += 1
    return {'rows': sorted(rows, key=lambda r: r['origin_row']), 'total': total, 'invalid': invalid, 'limit': 2000}
