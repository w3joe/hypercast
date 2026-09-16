from copy import deepcopy
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset
from hypercast4d.transfer_training import fit

def test_resume_reproduces_uninterrupted_dropout_training(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(41)
    model=nn.Sequential(nn.Flatten(),nn.Dropout(.3),nn.Linear(8,2))
    x=torch.randn(70,2,4);y=torch.randn(70,2);train=TensorDataset(x,y);inner=TensorDataset(x[:13],y[:13])
    full=deepcopy(model);split=deepcopy(model)
    a=fit(full,train,inner,tmp_path/'full',seed=11,epochs=5)
    fit(split,train,inner,tmp_path/'split',seed=11,epochs=2)
    b=fit(split,train,inner,tmp_path/'split',seed=11,epochs=5,resume=True)
    assert a==b
    for p,q in zip(full.parameters(),split.parameters()):torch.testing.assert_close(p,q,atol=0,rtol=0)
    last=torch.load(tmp_path/'split/latest.pt',weights_only=True)
    assert last['optimizer_state_dict']['state'] and last['epoch']==5
    assert a['best_inner_mae']==min(r['inner_mae'] for r in a['history'])
    with torch.inference_mode():assert (full(x[:13])-y[:13]).abs().mean().item()==pytest.approx(a['best_inner_mae'])
    with pytest.raises(ValueError,match='metadata'):
        fit(split,train,inner,tmp_path/'split',seed=11,epochs=5,resume=True,metadata={'changed':True})


def test_nonfinite_or_deadline_does_not_create_success_checkpoint(tmp_path):
    model=nn.Linear(2,1);data=TensorDataset(torch.zeros(3,2),torch.full((3,1),float('nan')))
    with pytest.raises(RuntimeError,match='Nonfinite'):fit(model,data,data,tmp_path/'bad',seed=1,epochs=1)
    assert not (tmp_path/'bad/best.pt').exists()
    with pytest.raises(TimeoutError):fit(model,data,data,tmp_path/'late',seed=1,epochs=1,deadline=0)
    assert not (tmp_path/'late/best.pt').exists()


def test_development_labels_cannot_change_training_or_selected_weights(tmp_path):
    import importlib.util,io,numpy as np
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/run_etth1_development.py'
    spec=importlib.util.spec_from_file_location('etth1_development',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(10)
    arrays={}
    for label in ['train','inner','development']:
        arrays[label+'_x']=rng.normal(size=(4,32,7)).astype('float32')
        arrays[label+'_y']=rng.normal(size=(4,5)).astype('float32')
    arrays['development_target_start']=np.arange(4)+100
    manifest=dict(scaler_span=[1]*7,scaler_minimum=[0]*7)
    job=dict(backbone='micn',mode='relative_residual',seed=2201)
    def run(name):
        buffer=io.BytesIO();np.savez(buffer,**arrays)
        return module.train_trial(manifest,buffer.getvalue(),job,tmp_path/name,device='cpu',epochs=2)
    first=run('first');arrays['development_y']+=100;second=run('second')
    assert first['history']==second['history'] and first['best_epoch']==second['best_epoch']
    a=torch.load(tmp_path/'first/best.pt',weights_only=True);b=torch.load(tmp_path/'second/best.pt',weights_only=True)
    for key,value in a['state_dict'].items():torch.testing.assert_close(value,b['state_dict'][key],atol=0,rtol=0)
    assert first['mae']!=second['mae'] and not first['test_scored'] and not second['test_scored']
    assert first['checkpoint_replay_passed'] and second['checkpoint_replay_passed']
