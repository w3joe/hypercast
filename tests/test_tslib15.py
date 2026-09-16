import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from hypercast4d.tslib15 import build,BACKBONES,ARMS

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_tslib15_controller import choose_rates,makespan
from analyze_tslib15 import bootstrap_intervals

def test_all_models_pair_untouched_state_and_include_all_inputs():
    torch.set_num_threads(2)
    for b in BACKBONES:
        hashes=[]
        for a in ARMS:
            m,r=build(b,a,3200);hashes.append(r['untouched_sha256'])
            assert m.features==7 and r['parameters']>r['selected_parameters']
        assert len(set(hashes))==1

def test_tuning_rejects_partial_grid_and_breaks_ties_by_rate():
    rows=[dict(job=dict(backbone='a',arm='real',learning_rate=lr),mae=1.) for lr in [.001,.0003]]
    assert choose_rates(rows,['a'],['real'])[0]['learning_rate']==.0003
    with pytest.raises(ValueError):choose_rates(rows[:1],['a'],['real'])
    assert makespan([5,5,5],2)==10

def test_bootstrap_preserves_pairing_and_known_proportional_effect():
    rng=np.random.default_rng(5);base=rng.uniform(.3,2,size=(4,3,90));other=base*.9
    point,low,high=bootstrap_intervals(base,other,12,resamples=200)
    np.testing.assert_allclose(point,10,atol=1e-10)
    np.testing.assert_allclose(low,10,atol=1e-10);np.testing.assert_allclose(high,10,atol=1e-10)

def test_test_labels_cannot_change_training_and_epoch_ceiling_is_explicit(tmp_path):
    import io,time
    from run_tslib15_worker import train_trial
    rng=np.random.default_rng(50)
    values=dict(train_x=rng.normal(size=(35,32,7)).astype('float32'),train_y=rng.normal(size=(35,5)).astype('float32'),
        inner_x=rng.normal(size=(8,32,7)).astype('float32'),inner_y=rng.normal(size=(8,5)).astype('float32'),
        test_x=rng.normal(size=(8,32,7)).astype('float32'),test_y=rng.normal(size=(8,5)).astype('float32'),test_target_start=np.arange(8))
    plan=dict(protocol='synthetic-test',source_sha256={},gpu_finish_epoch=time.time()+1000,
        dataset_manifest=dict(scaler_span=[1],scaler_minimum=[0]))
    job=dict(phase='test',backbone='dlinear',arm='quaternion',seed=3702,learning_rate=.001,timeout_seconds=600,epochs=3)
    first=io.BytesIO();np.savez(first,**values)
    a=train_trial(plan,first.getvalue(),job,tmp_path/'first','cpu')
    values['test_y']=values['test_y']+100
    second=io.BytesIO();np.savez(second,**values)
    b=train_trial(plan,second.getvalue(),job,tmp_path/'second','cpu')
    aa=torch.load(tmp_path/'first/best.pt',weights_only=True)['state_dict'];bb=torch.load(tmp_path/'second/best.pt',weights_only=True)['state_dict']
    for name in aa:torch.testing.assert_close(aa[name],bb[name],atol=0,rtol=0)
    assert a['epochs_ran']==b['epochs_ran']==3 and a['ceiling_reached'] and b['ceiling_reached']
    assert b['mae']>a['mae']+90
