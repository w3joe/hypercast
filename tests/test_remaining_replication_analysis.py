import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('replication_analysis', Path(__file__).parents[1]/'scripts/analyze_remaining_replication.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_common_blocks_preserve_opposite_contrasts_and_determinism():
    rng=np.random.default_rng(21)
    x=np.cumsum(rng.normal(size=100))
    result=module.simultaneous_intervals(np.column_stack([x,-x]),10,draws=500,seed=3)
    assert result==module.simultaneous_intervals(np.column_stack([x,-x]),10,draws=500,seed=3)
    assert result['mean'][0]==pytest.approx(-result['mean'][1])
    assert result['lower'][0]==pytest.approx(-result['upper'][1])
    iid=module.simultaneous_intervals(np.column_stack([x,-x]),1,draws=500,seed=3)
    assert result['bootstrap_standard_error'][0]>iid['bootstrap_standard_error'][0]


def test_degenerate_and_nonfinite_input_are_explicit():
    result=module.simultaneous_intervals(np.zeros((50,3)),5,draws=100)
    assert result['degenerate']==[True,True,True]
    assert result['lower']==result['upper']==[0,0,0]
    with pytest.raises(ValueError):module.simultaneous_intervals([[float('nan')]],1)
