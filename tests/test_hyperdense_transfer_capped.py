import importlib.util
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from launch_hyperdense_transfer_capped import reservation,admit,settle,charged

def test_whole_block_admission_retains_inflight_failed_costs():
    jobs=[{'timeout_seconds':3570}]*16
    cost=sum(reservation(3570,.9266) for _ in jobs)
    assert admit([],jobs,.9266,cost+10)
    assert not admit([],jobs,.9266,cost+9.99)
    a={'status':'failed','reservation_usd':20}
    settle(a,1,.9266);assert charged(a)==20
    assert not admit([a],jobs,.9266,cost+10)

def test_settlement_keeps_original_reservation_and_overheads():
    a={'status':'complete','reservation_usd':reservation(3570,.9266)}
    old=a['reservation_usd'];settle(a,100,.9266)
    assert a['reservation_usd']==old and charged(a)==reservation(100,.9266)<old
    with pytest.raises(RuntimeError):settle(a,4000,.9266)
    for cap in [float('nan'),float('inf'),0]:
        with pytest.raises(ValueError):admit([],[],.9266,cap)
