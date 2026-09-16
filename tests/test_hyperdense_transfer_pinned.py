from pathlib import Path
import importlib.util
import io
import numpy as np
import torch
from hypercast4d.hyperdense_transfer_pinned import build,constants,ARMS
from hypercast4d.hyperdense_transfer_pinned_checks import gate


def test_pinned_gates_and_all_arms_have_identical_fixed_buffers():
    g=gate('cpu');assert g['passed'] and len(g['rows'])==16
    expected=constants()
    for arm in ARMS:
        model,_=build('film',arm,2300)
        for name,value in expected.items():torch.testing.assert_close(dict(model.named_buffers())[name],value,atol=0,rtol=0)


def test_pinning_overrides_host_regeneration(monkeypatch):
    import hypercast4d.hyperdense_transfer_pinned as m
    original=m.unpinned_build
    def changed(*a,**kw):
        model,report=original(*a,**kw)
        with torch.no_grad():
            for name,b in model.named_buffers():
                if '.legts.' in name:b.add_(.00001)
        return model,report
    before,expected=m.build('film','real',2300);monkeypatch.setattr(m,'unpinned_build',changed);after,actual=m.build('film','real',2300)
    assert actual['untouched_sha256']==expected['untouched_sha256']
    for name,value in before.state_dict().items():torch.testing.assert_close(value,after.state_dict()[name],atol=0,rtol=0)


def test_repaired_calibration_archives_canonical_state(tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts/repair_hyperdense_transfer_calibration.py'
    spec=importlib.util.spec_from_file_location('pinned_cal',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    rng=np.random.default_rng(9);buffer=io.BytesIO();np.savez(buffer,train_x=rng.normal(size=(2,32,7)).astype('float32'),train_y=rng.normal(size=(2,5)).astype('float32'),inner_x=rng.normal(size=(2,32,7)).astype('float32'))
    job=dict(backbone='film',arm='octonion',seed=2300,gain=1.,epochs=3,learning_rate=.001,schedule='exponential')
    r=m.calibrate({},buffer.getvalue(),job,tmp_path,device='cpu');assert r['checkpoint_replay_passed']
    c=torch.load(tmp_path/'latest.pt',weights_only=True)
    for name,value in constants().items():torch.testing.assert_close(c['state_dict'][name],value,atol=0,rtol=0)
