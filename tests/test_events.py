import numpy as np
import pytest
from isac_wm.events import EventQueue, Observation, SnapshotBuffer


def obs(uid=1, capture=0., available=.2, modality="radio"):
    payload=np.ones(4) if modality=="radio" else np.ones((3,24,24))
    return Observation(uid,modality,capture,available,payload,1 if modality=="radio" else 0)


def test_no_early_release():
    q=EventQueue(); q.push(obs()); assert q.release(.199)==[]
    assert [o.uid for o in q.release(.2)]==[1]


def test_deterministic_ties():
    q=EventQueue()
    for uid in [3,1,2]: q.push(obs(uid))
    assert [o.uid for o in q.release(1)]==[1,2,3]


def test_oosm_arrives_in_availability_order():
    q=EventQueue(); q.push(obs(1,0,.4)); q.push(obs(2,.1,.2))
    assert q.release(.2)[0].capture_s==.1
    assert q.release(.4)[0].capture_s==0


def test_duplicate_rejected_after_delivery():
    q=EventQueue(); q.push(obs()); q.release(1)
    with pytest.raises(ValueError,match="Duplicate"): q.push(obs())


def test_overflow_explicit():
    q=EventQueue(1);q.push(obs())
    with pytest.raises(OverflowError): q.push(obs(2))


@pytest.mark.parametrize("capture,available",[(1,0),(0,float("nan")),(-1,0),(0,float("inf"))])
def test_invalid_time(capture,available):
    with pytest.raises(ValueError):obs(capture=capture,available=available)


def test_clock_cannot_reverse():
    q=EventQueue();q.release(1)
    with pytest.raises(ValueError):q.release(.9)
    with pytest.raises(ValueError):q.push(obs())


def test_payload_immutable():
    x=obs()
    with pytest.raises(ValueError):x.payload[0]=3


def test_no_double_counted_evidence():
    b=SnapshotBuffer(24)
    s=b.update([obs()],.2,.1);assert s["new"][1]==1
    s=b.update([],.3,.1);assert s["new"].sum()==0


def test_old_packet_does_not_replace_latest_snapshot():
    b=SnapshotBuffer(24);b.update([obs(2,.1,.2)],.2,.1)
    s=b.update([obs(1,0,.4)],.4,.1)
    assert s["age"][1]==pytest.approx(.3)
    assert b.ignored_older==1 and s["new"].sum()==0


def test_future_packet_in_snapshot_rejected():
    with pytest.raises(ValueError,match="Future"):
        SnapshotBuffer(24).update([obs()],.1,.1)
