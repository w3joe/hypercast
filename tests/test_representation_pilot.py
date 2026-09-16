import json
from pathlib import Path
import importlib.util
import numpy as np
import pytest
import torch
from hypercast4d.representation_pilot import build_model, advancement, run_pair, sha
from hypercast4d.remaining_controls import build_case

torch.set_num_threads(2)

@pytest.mark.parametrize('backbone',['micn','film'])
def test_pairing_persistence_relative_inputs_and_gradient_recovery(backbone):
    direct,a=build_model(backbone,'levels_direct',1101)
    residual,b=build_model(backbone,'relative_residual',1101)
    native,_=build_case(backbone,'native',1101)
    assert a['paired_except_head_sha256']==b['paired_except_head_sha256']
    assert sum(p.numel() for p in direct.parameters())==sum(p.numel() for p in residual.parameters())
    direct.eval();native.eval();residual.eval()
    x=torch.randn(2,32,4)
    torch.testing.assert_close(direct(x),native(x),atol=0,rtol=0)
    torch.testing.assert_close(residual(x),x[:,-1,0:1].expand(-1,5),atol=0,rtol=0)
    seen=[]
    handle=residual.layers[0].register_forward_pre_hook(lambda m,args:seen.append(args[0].detach().clone()))
    opt=torch.optim.Adam(residual.parameters(),lr=.001)
    (residual(x)-(x[:,-1,0:1]+1)).square().mean().backward()
    assert residual.output.weight.grad.abs().sum()>0
    opt.step();opt.zero_grad()
    residual(x).square().mean().backward()
    handle.remove()
    torch.testing.assert_close(seen[0],x-x[:,-1:,:])
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in residual.layers.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in residual.parameters())
    # Each feature's absolute shift disappears; target shift is added back exactly.
    offset=torch.tensor([3.,-2.,7.,1.])[None,None,:]
    torch.testing.assert_close(residual(x+offset),residual(x)+3,atol=2e-5,rtol=2e-4)


def test_advancement_requires_both_models_and_all_pairs():
    rows=[dict(backbone=b,seed=s,mode=m,mae=1. if m=='levels_direct' else .9,
               integrity_passed=True) for b in ['micn','film'] for s in [1101,1102,1103]
          for m in ['levels_direct','relative_residual']]
    assert advancement(rows)['advance']
    with pytest.raises(ValueError):advancement(rows[:-1])
    rows[-1]['mae']=float('nan')
    with pytest.raises(ValueError):advancement(rows)
    for r in rows:
        if r['backbone']=='film' and r['mode']=='relative_residual':r['mae']=1.1
    assert not advancement(rows)['advance']


@pytest.mark.parametrize('backbone',['micn','film'])
def test_fit_restore_archive_and_data_tamper(tmp_path,backbone):
    bundle=tmp_path/'bundle';bundle.mkdir()
    rng=np.random.default_rng(93)
    x=rng.normal(size=(6,32,4)).astype('float32')
    y=x[:,-1,0:1]+np.ones((6,5),dtype='float32')*.1
    np.savez(bundle/'development.npz',train_x=x,train_y=y,inner_x=x,inner_y=y,inner_target_start=np.arange(6)+32)
    (bundle/'manifest.json').write_text(json.dumps(dict(data_sha256=sha(bundle/'development.npz'),
        source_sha256={},scaler_span=[1]*4,scaler_minimum=[0]*4)))
    rows=run_pair(bundle,tmp_path/'run',backbone,1101,epochs=2)
    assert len(rows)==2 and all(r['integrity_passed'] for r in rows)
    for mode in ['levels_direct','relative_residual']:
        c=torch.load(tmp_path/'run'/mode/'checkpoint.pt',weights_only=True)
        assert c['fit']['best_epoch'] in [1,2]
        assert c['mode']==mode
    with (bundle/'development.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='digest'):run_pair(bundle,tmp_path/'bad',backbone,1101,epochs=1)


def test_frozen_bundle_excludes_outer_and_preserves_target_alignment(tmp_path):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('prepare_pilot',root/'scripts/prepare_representation_pilot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    bundle=tmp_path/'frozen';module.prepare(bundle)
    manifest=json.loads((bundle/'manifest.json').read_text())
    data=np.load(bundle/'development.npz')
    assert len(manifest['jobs'])==6
    assert not any('outer' in k or 'test' in k for k in data.files)
    from hypercast4d.data import load_paper_data
    frame=load_paper_data(root/'data/raw/paper_data.xlsx','Copper')
    train_end=manifest['train_rows'][1]
    raw=frame.to_numpy()
    np.testing.assert_array_equal(manifest['scaler_minimum'],raw[:train_end].min(0))
    for label in ['train','inner']:
        starts=data[label+'_target_start'];a,b=manifest[label+'_rows']
        assert starts.min()>=max(32,a) and (starts+4).max()<b
        reconstructed=data[label+'_y']*manifest['scaler_span'][0]+manifest['scaler_minimum'][0]
        np.testing.assert_allclose(reconstructed,raw[starts[:,None]+np.arange(5),0],rtol=1e-6)
    assert data['train_x'].shape==(1158,32,4)
    assert data['inner_x'].shape==(207,32,4)
    assert manifest['status']=='prepared_not_authorized'


def launcher_module():
    path=Path(__file__).resolve().parents[1]/'scripts/launch_representation_pilot.py'
    spec=importlib.util.spec_from_file_location('pilot_launch',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_budget_complete_stage_reserve_and_invalid_caps():
    module=launcher_module()
    jobs=[dict(timeout_seconds=t) for t in [600,300,300,300,570,570,570]]
    costs=module.admission(jobs,.9266,6)
    assert sum(costs)==pytest.approx(3.477275)
    for cap in [5.,0.,-1.,float('nan'),float('inf')]:
        with pytest.raises(ValueError):module.admission(jobs,.9266,cap)


def test_worker_returns_partial_artifacts_on_failure(tmp_path,monkeypatch):
    import io,tarfile,hashlib
    import hypercast4d.representation_pilot as pilot
    def fail(bundle,output,*args):
        output.mkdir();(output/'partial.txt').write_text('retain me')
        raise RuntimeError('injected failure')
    monkeypatch.setattr(pilot,'run_pair',fail)
    module=launcher_module()
    result,archive=module.worker(dict(source_sha256={},data_sha256=hashlib.sha256(b'data').hexdigest()),
        b'data',dict(kind='training',backbone='micn',seed=1101),require_cuda=False)
    assert not result['ok'] and 'injected failure' in result['error']
    with tarfile.open(fileobj=io.BytesIO(archive)) as f:
        assert f.extractfile('artifacts/partial.txt').read()==b'retain me'
        assert 'runtime.json' in f.getnames()


def test_parallel_waves_preserve_inference_gate_and_scope():
    module=launcher_module()
    jobs=[dict(kind='inference')]+[dict(kind='training') for _ in range(6)]
    waves=module.job_waves(jobs,2)
    assert waves==[[0],[1,2],[3,4],[5,6]]
    assert [i for wave in waves for i in wave]==list(range(7))
    with pytest.raises(ValueError):module.job_waves(jobs,3)
