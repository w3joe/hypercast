"""Post-hoc diagnostics of saved forecasts; train-only CPU baseline probes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hypercast4d.data import load_paper_data
from hypercast4d.internal_controls import nested_windows

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'results/modal-l4-internal'
OUT = ROOT/'results/hyperdense-followup-local'


def metrics(pred, actual, persistence):
    residual = pred-actual
    mae = np.abs(residual).mean()
    mse = np.square(residual).mean()
    return dict(mae=float(mae),mse=float(mse),mae_ratio=float(mae/np.abs(persistence-actual).mean()),
        mse_ratio=float(mse/np.square(persistence-actual).mean()),bias=float(residual.mean()),
        predicted_change_mean=float((pred-persistence).mean()),actual_change_mean=float((actual-persistence).mean()),
        mse_bias_fraction=float(residual.mean()**2/mse),underprediction_fraction=float((pred<actual).mean()))


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    state=json.loads((RUNS/'controller/controller_state.json').read_text())
    candidates=[c for c in state['candidates'] if c['stage_name']=='remaining-replication']
    assert len(candidates)==12 and all(c['status']=='complete' for c in candidates)
    request=json.loads((RUNS/'jobs'/candidates[0]['job_id']/'request.json').read_text())
    evaluation=request['evaluation']
    frame=load_paper_data(ROOT/evaluation['data_path'],evaluation['target_column'])
    prepared,inner,audit=nested_windows(frame,32,5,.7,.15)
    # Only summarize rows already used in the study. The later data is not scored.
    frame=frame.iloc[:audit['outer_rows'][1]]
    raw=frame.to_numpy()
    scale=prepared.scaler
    output=dict(limitations='Post-hoc development diagnostics on already-inspected Copper dates. CPU ridge probes are not neural model confirmation. No later test rows scored; no L4 jobs submitted.',
                audit=audit, partitions={},arms=[],baseline_probes=[],baseline_grid=[],input_sha256={})
    for path in [ROOT/evaluation['data_path'],RUNS/'controller/controller_state.json']:
        output['input_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for label,bounds,split in [('train',audit['train_rows'],prepared.train),('inner',audit['inner_rows'],inner),('outer',audit['outer_rows'],prepared.validation)]:
        values=raw[slice(*bounds)]
        actual=scale.inverse_target(split.y.astype(np.float64))
        persistence=scale.inverse_target(np.repeat(split.x[:,-1,0:1].astype(np.float64),5,axis=1))
        output['partitions'][label]=dict(start_date=str(frame.index[bounds[0]].date()),end_date=str(frame.index[bounds[1]-1].date()),
            rows=len(values),origins=len(split.y),target_min=float(values[:,0].min()),target_max=float(values[:,0].max()),
            target_above_train_max_fraction=float((values[:,0]>scale.minimum[0]+scale.span[0]).mean()),
            feature_outside_training_range_fraction={col:float(((values[:,i]<scale.minimum[i])|(values[:,i]>scale.minimum[i]+scale.span[i])).mean()) for i,col in enumerate(frame.columns)},
            persistence_mae=float(np.abs(persistence-actual).mean()),persistence_mse=float(np.square(persistence-actual).mean()),
            persistence_mse_scaled=float(np.square(split.y-split.x[:,-1,0:1]).mean()),
            persistence_bias=float((persistence-actual).mean()))
    source_max_actual_error=0.; source_max_persistence_error=0.
    archived_weight_files=[]
    import tarfile
    for c in candidates:
        directory=RUNS/'jobs'/c['job_id']
        preds=pd.read_csv(directory/'predictions.csv')
        curves=pd.read_csv(directory/'learning_curves.csv')
        runs=pd.read_csv(directory/'runs.csv')
        splits=json.loads((directory/'split_audit.json').read_text())
        for s in splits:
            for key in audit:
                assert s[key]==audit[key],(c['backbone'],key)
        with tarfile.open(directory/'batch-artifacts.tar.gz') as archive:
            archived_weight_files.extend(n for n in archive.getnames() if n.endswith(('.pt','.pth','.safetensors')))
        for row in runs.to_dict('records'):
            p=preds[preds.trial_id==row['trial_id']].sort_values(['origin_row','lead'])
            assert len(p)==1485 and not p.duplicated(['origin_row','lead']).any()
            expected_origins=np.repeat(prepared.validation.target_start-1,5)
            assert np.array_equal(p.origin_row,expected_origins)
            assert np.array_equal(p.lead,np.tile(np.arange(1,6),297))
            expected_actual=raw[p.origin_row.to_numpy()+p.lead.to_numpy(),0]
            expected_origin=raw[p.origin_row,0]
            assert np.allclose(p.actual,expected_actual,atol=5e-7,rtol=0)
            assert np.allclose(p.persistence,expected_origin,atol=5e-7,rtol=0)
            assert np.array_equal(pd.to_datetime(p.target_date).to_numpy(),frame.index[p.origin_row+p.lead].to_numpy())
            source_max_actual_error=max(source_max_actual_error,float(np.abs(p.actual-expected_actual).max()))
            source_max_persistence_error=max(source_max_persistence_error,float(np.abs(p.persistence-expected_origin).max()))
            m=metrics(p.prediction.to_numpy(),p.actual.to_numpy(),p.persistence.to_numpy())
            assert np.isclose(m['mae'],row['mae']) and np.isclose(m['mse'],row['mse'])
            curve=curves[curves.trial_id==row['trial_id']].sort_values('epoch')
            chosen=curve[curve.epoch==row['best_epoch']].iloc[0]
            assert np.isclose(chosen.validation_loss,row['best_validation_loss_scaled'])
            groups={}
            for label,mask in [('origin_inside_train_range',p.origin_price.between(scale.minimum[0],scale.minimum[0]+scale.span[0])),
                               ('origin_above_train_max',p.origin_price>scale.minimum[0]+scale.span[0])]:
                part=p[mask]
                if len(part):groups[label]=dict(targets=len(part),**metrics(part.prediction.to_numpy(),part.actual.to_numpy(),part.persistence.to_numpy()))
            lead_metrics={str(lead):metrics(part.prediction.to_numpy(),part.actual.to_numpy(),part.persistence.to_numpy()) for lead,part in p.groupby('lead')}
            output['arms'].append(dict(backbone=c['backbone'],variant=row['variant'],**m,by_lead=lead_metrics,by_origin_regime=groups,
                epochs_ran=row['epochs_ran'],best_epoch=row['best_epoch'],
                best_inner_mse_over_persistence=float(chosen.validation_loss/output['partitions']['inner']['persistence_mse_scaled']),
                last_inner_mse_over_first=float(curve.iloc[-1].validation_loss/curve.iloc[0].validation_loss),
                best_inner_over_training_mse=float(chosen.validation_loss/chosen.train_loss)))
    output['integrity']=dict(arms=96,forecasts=96*1485,source_max_actual_error=source_max_actual_error,
        source_max_persistence_error=source_max_persistence_error,source_dates_and_origins_verified=True,
        partition_scaling_verified=True,archived_trained_weight_files=archived_weight_files)
    # Small diagnostic baselines: all fitted on actual train rows/windows only.
    # Grid is fixed here before computation; inner MSE chooses alpha, never outer MAE.
    alphas=[.01,.1,1.,10.,100.,1000.,10000.]
    splits={'train':prepared.train,'inner':inner,'outer':prepared.validation}
    for feature_mode in ['levels','relative_to_latest']:
        def features(split):
            x=split.x.astype(np.float64)
            if feature_mode=='relative_to_latest':x=x-x[:,-1:,:]
            return x.reshape(len(x),-1)
        tr=features(prepared.train); mean=tr.mean(0); std=tr.std(0); std[std<1e-10]=1.
        xs={k:(features(v)-mean)/std for k,v in splits.items()}
        # Predict corrections around persistence; same absolute observations as the neural models.
        ys={k:v.y.astype(np.float64)-v.x[:,-1,0:1].astype(np.float64) for k,v in splits.items()}
        intercept=ys['train'].mean(0)
        gram=xs['train'].T@xs['train']; rhs=xs['train'].T@(ys['train']-intercept)
        solutions=[]
        for alpha in alphas:
            beta=np.linalg.solve(gram+alpha*np.eye(gram.shape[0]),rhs)
            loss=float(np.square(xs['inner']@beta+intercept-ys['inner']).mean())
            solutions.append((loss,alpha,beta))
            output['baseline_grid'].append(dict(feature_mode=feature_mode,alpha=alpha,inner_mse_scaled=loss))
        loss,alpha,beta=min(solutions,key=lambda item:(item[0],-item[1]))
        for name in ['inner','outer']:
            part=splits[name]; persistence=scale.inverse_target(np.repeat(part.x[:,-1,0:1].astype(np.float64),5,axis=1))
            actual=scale.inverse_target(part.y.astype(np.float64))
            pred=persistence+(xs[name]@beta+intercept)*scale.span[0]
            output['baseline_probes'].append(dict(method='ridge_residual_'+feature_mode,split=name,alpha=alpha,
                selected_on='inner_mse',train_samples=len(tr),**metrics(pred,actual,persistence)))
    drift=np.diff(raw[:audit['train_rows'][1],0]).mean()
    for name in ['inner','outer']:
        part=splits[name]; persistence=scale.inverse_target(np.repeat(part.x[:,-1,0:1].astype(np.float64),5,axis=1))
        actual=scale.inverse_target(part.y.astype(np.float64))
        pred=persistence+drift*np.arange(1,6)
        output['baseline_probes'].append(dict(method='train_mean_drift',split=name,**metrics(pred,actual,persistence)))
    (OUT/'persistence-diagnostics.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(partitions=output['partitions'],integrity=output['integrity'],baseline_probes=output['baseline_probes'],
        native=[{k:v for k,v in a.items() if k not in ['by_lead','by_origin_regime']} for a in output['arms'] if a['variant']=='native']),indent=2))


if __name__=='__main__':main()
