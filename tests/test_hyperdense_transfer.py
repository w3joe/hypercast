from copy import deepcopy
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset
from hypercast4d.hyperdense_transfer import build,mapping,spectral_permutation,ARMS,SITES
from hypercast4d.hyperdense_transfer_checks import gate,expanded
from hypercast4d.hyperdense_transfer_training import run
from hypercast4d.native_transfer import build as native_build

torch.set_num_threads(2)


def test_all_control_gates():
    result=gate('cpu');assert result['passed'] and len(result['rows'])==16
    expected={'micn':[82636,82636,82124,81868,81740,82124,81868,81740], 'film':[6292599,8389751,6292599,5244023,4719735,6292599,5244023,4719735]}
    for backbone,counts in expected.items():assert [r['parameters'] for r in result['rows'] if r['backbone']==backbone]==counts


@pytest.mark.parametrize('backbone',['micn','film'])
def test_native_reconstruction_gain_and_constructor_rng(backbone):
    before=torch.get_rng_state().clone();a,_=build(backbone,'native',2300);assert torch.equal(before,torch.get_rng_state())
    b=native_build(backbone,2300,'relative_residual')
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
    for arm in ARMS:
        a,_=build(backbone,arm,2300,1.);b,_=build(backbone,arm,2300,.5)
        site=SITES[backbone]
        for k,v in a.state_dict().items():
            if not k.startswith(site+'.'):torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
        sa,sb=a.get_submodule(site),b.get_submodule(site)
        if arm=='native':
            for n,p in sa.named_parameters():torch.testing.assert_close(dict(sb.named_parameters())[n],p*(.5 if 'weight' in n else 1),atol=0,rtol=0)
        else:
            for x,y in zip(sa.maps if backbone=='film' else [sa],sb.maps if backbone=='film' else [sb]):torch.testing.assert_close(expanded(y),.5*expanded(x),atol=1e-7,rtol=1e-5)


def test_component_pairing():
    for q in [2,4,8]:
        p=spectral_permutation(256,q);assert torch.equal(torch.sort(p).values,torch.arange(512))
        b=p.reshape(q//2,2,-1);assert torch.equal(b[:,1]-b[:,0],torch.full_like(b[:,0],256))
        assert torch.equal(p[torch.argsort(p)],torch.arange(512))


def test_effective_variance_across_initialization_seeds():
    for arm in ARMS[1:]:
        moments=[]
        for seed in range(200):
            w=expanded(mapping(32,32,True,arm,seed,'micn','variance',1.)).detach()
            moments.append(float(w.square().mean()))
        assert np.mean(moments)==pytest.approx(1/32,rel=.08)


@pytest.mark.parametrize('calibration',[False,True])
@pytest.mark.parametrize('schedule',['constant','exponential'])
def test_exact_checkpoint_continuation(tmp_path,calibration,schedule):
    torch.manual_seed(1);model=nn.Sequential(nn.Flatten(),nn.Dropout(.2),nn.Linear(8,2));x=torch.randn(70,2,4);y=torch.randn(70,2)
    train=TensorDataset(x,y);inner=x[:13] if calibration else TensorDataset(x[:13],y[:13])
    a,b=deepcopy(model),deepcopy(model)
    kw=dict(seed=7,schedule=schedule,calibration=calibration,metadata={'test':1})
    full=run(a,train,inner,tmp_path/'full',epochs=5,**kw)
    run(b,train,inner,tmp_path/'split',epochs=2,**kw);split=run(b,train,inner,tmp_path/'split',epochs=5,resume=True,**kw)
    assert full['history']==split['history']
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,atol=0,rtol=0)
    state=torch.load(tmp_path/'split/latest.pt',weights_only=True);assert state['epoch']==5 and state['scheduler_state_dict']['last_epoch']==5
    assert [r['lr'] for r in full['history']]==pytest.approx([.001*(.98 if schedule=='exponential' else 1)**i for i in range(5)])
    with pytest.raises(ValueError,match='metadata'):run(b,train,inner,tmp_path/'split',epochs=5,resume=True,**dict(kw,metadata={'changed':1}))


def test_nonfinite_deadline_and_calibration_labels_rejected(tmp_path):
    m=nn.Linear(2,1);x=torch.ones(2,2);bad=TensorDataset(x,torch.full((2,1),float('nan')))
    with pytest.raises(RuntimeError,match='Nonfinite'):run(m,bad,x,tmp_path/'bad',seed=1,epochs=1,calibration=True)
    with pytest.raises(TimeoutError):run(m,bad,x,tmp_path/'late',seed=1,epochs=1,calibration=True,deadline=0)
    with pytest.raises(ValueError,match='inputs only'):run(m,bad,bad,tmp_path/'labels',seed=1,epochs=1,calibration=True)
    assert not (tmp_path/'bad/latest.pt').exists()


def test_training_and_inner_target_roles(tmp_path):
    x=torch.ones(4,2);y=torch.ones(4,1);model=nn.Linear(2,1)
    with torch.no_grad():model.weight.zero_();model.bias.zero_()
    results=[];states=[]
    for name,train_y,inner_y in [('near',y,torch.zeros_like(y)),('far',y,y),('changed_train',-y,y)]:
        result=run(deepcopy(model),TensorDataset(x,train_y),TensorDataset(x,inner_y),tmp_path/name,seed=1,epochs=5)
        results.append(result);states.append(torch.load(tmp_path/name/'latest.pt',weights_only=True))
    assert results[0]['best_epoch']==1 and results[1]['best_epoch']==5
    assert [r['train_mae'] for r in results[0]['history']]==[r['train_mae'] for r in results[1]['history']]
    for k,v in states[0]['state_dict'].items():torch.testing.assert_close(v,states[1]['state_dict'][k],atol=0,rtol=0)
    assert not torch.equal(states[0]['state_dict']['weight'],states[2]['state_dict']['weight'])


def test_calibration_runner_checkpoint_and_label_isolation(tmp_path):
    import importlib.util,io
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/launch_hyperdense_transfer_calibration.py'
    spec=importlib.util.spec_from_file_location('hc_calibration',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    rng=np.random.default_rng(7)
    arrays=dict(train_x=rng.normal(size=(4,32,7)).astype('float32'),train_y=rng.normal(size=(4,5)).astype('float32'),inner_x=rng.normal(size=(4,32,7)).astype('float32'))
    buffer=io.BytesIO();np.savez(buffer,**arrays)
    job=dict(backbone='micn',arm='octonion',seed=2300,gain=1.,epochs=3,learning_rate=.001,schedule='exponential')
    result=m.calibrate({},buffer.getvalue(),job,tmp_path/'cal',device='cpu')
    assert result['checkpoint_replay_passed'] and result['epochs_ran']==3 and not result['validation_scored'] and not result['test_scored']
    checkpoint=torch.load(tmp_path/'cal/latest.pt',weights_only=True);assert checkpoint['epoch']==3 and checkpoint['scheduler_state_dict']['last_epoch']==3
    assert len(checkpoint['timings'])==3 and len(result['timings'])==3
    assert all(t['total_seconds']>0 for t in result['timings'])
    for name in ['inner_y','development_y','test_y']:
        buffer=io.BytesIO();np.savez(buffer,**arrays,**{name:np.ones((4,5),dtype='float32')})
        with pytest.raises(ValueError,match='omit evaluation labels'):m.calibrate({},buffer.getvalue(),job,tmp_path/name,device='cpu')
