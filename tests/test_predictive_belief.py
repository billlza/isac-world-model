import math

import pytest

from isac_wm.evidence_reasoning import RFPoint, VisionBearing
from isac_wm.predictive_belief import PredictiveBelief


def rf(uid, t, r, speed=1.0, sigma=0.2):
    return RFPoint(uid, uid, t, t + 0.05, r, 0.1, speed, range_sigma_m=sigma, speed_sigma_mps=0.15)


def test_no_evidence_and_future_embargo():
    m = PredictiveBelief()
    assert m.predict_observation(0.5)["status"] == "unknown_no_evidence"
    with pytest.raises(ValueError):
        m.ingest(rf(1, 1.0, 10), now_s=0.9)
    assert not m.history


def test_prediction_update_and_counterevidence():
    m = PredictiveBelief()
    m.ingest(rf(1, 0, 10), now_s=0.05)
    before = m.predict_observation(0.8)
    assert len(before["hypotheses"]) == 2
    m.ingest(rf(2, 1.0, 11.0), now_s=1.05)
    after = m.predict_observation(0.8)
    assert after["hypotheses"][1]["range_m"] > before["hypotheses"][1]["range_m"]
    assert after["material"] == "unknown"
    m.ingest(rf(3, 1.5, 80.0), now_s=1.55)
    assert m.contradictions >= 1
    assert m.explain()["object_identity"] == "unknown"


def test_vision_conflict_and_duplicate_protection():
    m = PredictiveBelief()
    m.ingest(rf(1, 0, 12), now_s=0.05)
    v = VisionBearing(99, 1, 0, 0.06, 0.9)
    result = m.ingest(v, now_s=0.07)
    assert result["last_event"] == "visual_rf_bearing_conflict"
    with pytest.raises(ValueError):
        m.ingest(v, now_s=0.08)


def test_delayed_older_observation_not_rewind():
    m = PredictiveBelief()
    m.ingest(rf(1, 1, 10), now_s=1.1)
    result = m.ingest(rf(2, 0.5, 9), now_s=1.2)
    assert result["last_event"] == "late_rf_ignored"


def test_memory_and_capacity():
    m = PredictiveBelief(max_history=2, max_ids=3)
    for i in range(3):
        m.ingest(rf(i, float(i), 10 + i), now_s=i + 0.1)
    assert len(m.history) == 2
    with pytest.raises(OverflowError):
        m.ingest(rf(4, 4, 14), now_s=4.1)


def test_invalid_horizons():
    m = PredictiveBelief()
    for t in (-1, math.nan, 61):
        with pytest.raises(ValueError):
            m.predict_observation(t)


def test_action_conditioned_prediction_and_cost():
    m = PredictiveBelief()
    m.ingest(rf(1, 0, 18), now_s=0.05)
    f = m.forecast_actions(0.2)
    assert f["future_truth_used"] is False
    assert f["hypotheses_by_action"]["skip"]["packet"] is None
    quick = f["hypotheses_by_action"]["quick"]
    precise = f["hypotheses_by_action"]["precise"]
    assert quick["earliest_available_s"] < precise["earliest_available_s"]
    assert quick["resource_fraction"] < precise["resource_fraction"]
    assert (
        quick["prediction"]["hypotheses"][0]["range_sigma_m_assumed"]
        > precise["prediction"]["hypotheses"][0]["range_sigma_m_assumed"]
    )
    assert precise["bearing_prediction"] == "unknown_without_angular_dynamics"
    with pytest.raises(ValueError):
        m.forecast_actions(0.1, actions=(1, 1))


def test_capture_time_then_propagate_and_stale_rejection():
    m = PredictiveBelief()
    m.ingest(rf(1, 0.0, 10), now_s=0.05)
    report = m.ingest(rf(2, 1.0, 11), now_s=1.5)
    assert report["clock_s"] == pytest.approx(1.5)
    # Posterior update happened at 1.0 and the motion hypothesis advances to 1.5.
    assert report["prediction"]["hypotheses"][1]["range_m"] >= 11.0
    stale = m.ingest(rf(3, 1.1, 11.1), now_s=1.6)
    assert stale["last_event"] == "late_rf_ignored"


def test_invalid_now_has_no_partial_state_mutation():
    m = PredictiveBelief()
    o = rf(7, 0, 10)
    for t in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            m.ingest(o, now_s=t)
        assert not m.seen
        assert m.hypotheses is None


def test_real_signal_level_phy_adapter_to_predictive_memory():
    from isac_wm.evidence_reasoning import from_phy_result
    from isac_wm.phy_ofdm import OFDMConfig, evaluate_frame

    cfg = OFDMConfig()
    tracker = PredictiveBelief()
    for i in range(2):
        result = evaluate_frame(cfg, probe_fraction=0.2, seed=2026 + i)
        packet = from_phy_result(
            result,
            uid=900 + i,
            frame=i,
            capture_s=2.0 * i,
            available_s=2.0 * i + 0.1,
        )
        tracker.ingest(packet, now_s=2.0 * i + 0.1)
    predictions = tracker.forecast_actions(0.2)
    assert len(predictions["hypotheses_by_action"]["precise"]["prediction"]["hypotheses"]) == 2
    assert predictions["future_truth_used"] is False
    assert tracker.explain()["object_material"] == "unknown"
