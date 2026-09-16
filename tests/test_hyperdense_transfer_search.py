import importlib.util
import io
from pathlib import Path
import numpy as np
import pytest
import torch

P=Path(__file__).resolve().parents[1]/'scripts/launch_hyperdense_transfer_search.py'
spec=importlib.util.spec_from_file_location('hd_search',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_equal_grid_ranking_and_second_seed_selection():
    rows=[dict(backbone='micn',arm='real',phase='search',seed=2301,setting_id=i,settings={'id':i},mae=1 if i<2 else 2) for i in range(12)]
    assert [r['setting_id'] for r in m.top_two(rows,'micn','real')]==[0,1]
    with pytest.raises(ValueError):m.top_two(rows[:-1],'micn','real')
    rows += [dict(backbone='micn',arm='real',phase='reranking',seed=2302,setting_id=i,settings={'id':i},mae=v) for i,v in [(0,3),(1,1)]]
    assert m.selected_settings(rows,['micn'],['real'])[0]['setting_id']==1
    with pytest.raises(ValueError):m.selected_settings(rows[:-1],['micn'],['real'])

def test_development_targets_do_not_affect_fitted_weights_or_inner_selection(tmp_path):
    torch.set_num_threads(2);rng=np.random.default_rng(32)
    arrays={k:rng.normal(size=shape).astype('float32') for k,shape in [('train_x',(3,32,7)),('train_y',(3,5)),('inner_x',(2,32,7)),('inner_y',(2,5)),('development_x',(2,32,7)),('development_y',(2,5))]}
    arrays['development_target_start']=np.array([100,101]);plan={'dataset_manifest':{'scaler_span':[2.],'scaler_minimum':[3.]}}
    job=dict(backbone='micn',arm='quaternion',seed=2301,phase='search',setting_id=0,settings=dict(selected_weight_amplitude=.5,learning_rate=.0001,schedule='exponential'))
    def execute(name):
        blob=io.BytesIO();np.savez(blob,**arrays)
        return m.train_trial(plan,blob.getvalue(),job,tmp_path/name,device='cpu',epochs=2)
    left=execute('a');arrays['development_y']+=100;right=execute('b')
    assert left['history']==right['history'] and left['best_inner_mae']==right['best_inner_mae']
    assert right['mae']>left['mae']+100
    a=torch.load(tmp_path/'a/best.pt',weights_only=True);b=torch.load(tmp_path/'b/best.pt',weights_only=True)
    for name,value in a['state_dict'].items():torch.testing.assert_close(value,b['state_dict'][name],atol=0,rtol=0)
    arrays['test_y']=np.zeros((2,5),dtype='float32')
    with pytest.raises(ValueError,match='test arrays'):execute('c')
