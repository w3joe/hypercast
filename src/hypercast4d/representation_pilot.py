"""Isolated native representation pilot; no scheduler or spending authority."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from .remaining_controls import build_case, digest_except
from .training import fit_model, predict

MODES = ('levels_direct', 'relative_residual')
SEEDS = (1101, 1102, 1103)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_model(backbone, mode, seed):
    if backbone not in ('micn', 'film') or mode not in MODES:
        raise ValueError('Pilot is restricted to native MICN/FiLM and two formulations')
    model, report = build_case(backbone, 'native', seed=seed)
    if not isinstance(model.output, nn.Linear) or model.head_type != 'direct':
        raise ValueError('Expected the original external linear direct head')
    shared = digest_except(model, 'output')
    if mode == 'relative_residual':
        model.representation = 'centered'
        model.head_type = 'persistence_residual'
        nn.init.zeros_(model.output.weight)
        nn.init.zeros_(model.output.bias)
    if digest_except(model, 'output') != shared:
        raise RuntimeError('Untouched initialization changed')
    report.update(mode=mode, paired_except_head_sha256=shared,
                  architecture=model.spec, representation=model.representation,
                  head_type=model.head_type)
    return model, report


def score(pred, actual, persistence):
    error = pred - actual
    return dict(mae=float(np.abs(error).mean()), mse=float(np.square(error).mean()),
                bias=float(error.mean()), per_lead_mae=np.abs(error).mean(0).tolist(),
                per_lead_mse=np.square(error).mean(0).tolist(),
                per_lead_bias=error.mean(0).tolist(),
                persistence_mae=float(np.abs(persistence-actual).mean()))


def advancement(rows):
    """Incomplete, duplicate, nonfinite or invalid runs cannot pass the gate."""
    expected = {(b, s, m) for b in ('micn', 'film') for s in SEEDS for m in MODES}
    if len(rows) != 12 or {(r['backbone'], r['seed'], r['mode']) for r in rows} != expected:
        raise ValueError('Advancement requires all twelve unique fits')
    outcomes = {}
    for backbone in ('micn', 'film'):
        differences = []
        for seed in SEEDS:
            pair = {r['mode']: r for r in rows if r['backbone'] == backbone and r['seed'] == seed}
            if not all(r['integrity_passed'] and np.isfinite(r['mae']) for r in pair.values()):
                raise ValueError('Invalid fit cannot enter advancement')
            differences.append(pair[MODES[1]]['mae'] - pair[MODES[0]]['mae'])
        outcomes[backbone] = dict(paired_mae_differences=differences,
            passes=sum(d < 0 for d in differences) >= 2 and float(np.median(differences)) < 0)
    return dict(advance=all(r['passes'] for r in outcomes.values()), models=outcomes,
                caveat='Exploratory development gate; convergence flags require review, not a significance test')


def run_pair(bundle, output, backbone, seed, device='cpu', epochs=100):
    """Execute one paired job. Bundle contains train/inner only, never final test."""
    if seed not in SEEDS or backbone not in ('micn', 'film') or not 1 <= epochs <= 100:
        raise ValueError('Invalid pilot fit scope')
    bundle, output = Path(bundle), Path(output)
    manifest = json.loads((bundle/'manifest.json').read_text())
    if sha(bundle/'development.npz') != manifest['data_sha256']:
        raise ValueError('Frozen data digest mismatch')
    root = Path(__file__).resolve().parents[2]
    for name, digest in manifest['source_sha256'].items():
        if sha(root/name) != digest:
            raise ValueError(f'Frozen source digest mismatch: {name}')
    output.mkdir(parents=True, exist_ok=False)
    data = np.load(bundle/'development.npz', allow_pickle=False)
    train = TensorDataset(torch.from_numpy(data['train_x']), torch.from_numpy(data['train_y']))
    inner = TensorDataset(torch.from_numpy(data['inner_x']), torch.from_numpy(data['inner_y']))
    rows = []
    for mode in MODES:
        directory = output/mode
        directory.mkdir()
        model, init = build_model(backbone, mode, seed)
        (directory/'initialization.json').write_text(json.dumps(init, indent=2))
        curve = []
        def callback(epoch, train_loss, validation_loss):
            curve.append(dict(epoch=epoch, train_mae=train_loss, inner_mae=validation_loss))
            (directory/'curve.json').write_text(json.dumps(curve))
        fit = fit_model(model, train, inner, seed=seed, epochs=epochs, batch_size=32,
            learning_rate=.001, adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-7,
            adam_amsgrad=False, loss_name='mae', shuffle=True, early_stopping_patience=20,
            early_stopping_min_delta=0., early_stopping_relative_delta=0.,
            restore_best_weights=True, device=torch.device(device), epoch_callback=callback)
        predicted = predict(model, inner, torch.device(device), 32)
        if not np.isclose(np.abs(predicted-data['inner_y']).mean(), fit.best_validation_loss, rtol=2e-5, atol=1e-7):
            raise RuntimeError('Restored checkpoint MAE does not match selection score')
        checkpoint = dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            backbone=backbone, mode=mode, seed=seed, initialization=init,
            fit=asdict(fit), manifest=manifest)
        torch.save(checkpoint, directory/'checkpoint.pt')
        restored, _ = build_model(backbone, mode, seed)
        restored.load_state_dict(torch.load(directory/'checkpoint.pt', weights_only=True)['state_dict'])
        restored.to(device)
        np.testing.assert_allclose(predict(restored, inner, torch.device(device), 32), predicted, rtol=2e-5, atol=1e-7)
        span, minimum = manifest['scaler_span'][0], manifest['scaler_minimum'][0]
        actual = data['inner_y']*span+minimum
        persistence = np.repeat(data['inner_x'][:,-1,0:1],5,axis=1)*span+minimum
        predicted = predicted*span+minimum
        np.savez_compressed(directory/'predictions.npz', prediction=predicted, actual=actual,
                            persistence=persistence, target_start=data['inner_target_start'])
        row = dict(backbone=backbone, seed=seed, mode=mode, **asdict(fit),
            **score(predicted, actual, persistence), integrity_passed=True,
            checkpoint_sha256=sha(directory/'checkpoint.pt'),
            convergence_review_required=fit.best_epoch == fit.epochs_ran,
            whole_model_parameters=sum(p.numel() for p in model.parameters()))
        rows.append(row)
        (output/'scores.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    return rows
