"""Contract and adversarial tests: no false certainty, no future truth or evidence reuse."""

import math

import numpy as np
import pytest

from isac_wm.evidence_reasoning import (
    CausalEvidenceMemory,
    RFPoint,
    VisionBearing,
    detect_visual_bearing,
    from_phy_result,
    render_synthetic_camera,
)


def rf(frame, bearing=18, *, available=None, captured=None, uid=None):
    capture = frame * 0.1 if captured is None else captured
    arrival = capture + 0.04 if available is None else available
    return RFPoint(
        uid=3 * frame if uid is None else uid,
        frame=frame,
        capture_s=capture,
        available_s=arrival,
        range_m=18 + 0.1 * frame,
        bearing_rad=math.radians(bearing),
        radial_speed_mps=1.0,
    )


def visual(frame, bearing=18, *, available=None, captured=None, uid=None):
    capture = frame * 0.1 if captured is None else captured
    arrival = capture + 0.08 if available is None else available
    return VisionBearing(
        uid=3 * frame + 1 if uid is None else uid,
        frame=frame,
        capture_s=capture,
        available_s=arrival,
        bearing_rad=math.radians(bearing),
    )


def feed(memory, frame, visual_bearing=18, rf_bearing=18, reverse=False):
    obs = [rf(frame, rf_bearing), visual(frame, visual_bearing)]
    if reverse:
        obs.reverse()
    for item in obs:
        memory.ingest(item, now_s=frame * 0.1 + 0.08)


def test_agreeing_modalities_require_two_independent_frames():
    memory = CausalEvidenceMemory()
    feed(memory, 0)
    one = memory.explanation()
    assert one["association_status"] == "insufficient_evidence"
    feed(memory, 1)
    result = memory.explanation()
    assert result["association_status"] == "angularly_consistent_identity_unresolved"
    assert result["identity_confirmed"] is False
    assert "distinct_objects_aligned_in_bearing" in result["alternative_explanations"]
    assert result["pair_count"] == 2
    assert result["object_category"] == result["material"] == result["shape"] == "unknown"
    assert len(result["comparisons"]) == 2


def test_contradiction_not_fused_or_called_object_identity():
    mem = CausalEvidenceMemory()
    feed(mem, 0, visual_bearing=-34, rf_bearing=18)
    feed(mem, 1, visual_bearing=-34, rf_bearing=18)
    out = mem.explanation()
    assert out["association_status"] == "cross_modal_conflict"
    assert out["contradicting_pairs"] == 2
    assert out["relative_same_object_score_uncalibrated"] < 0.05
    assert out["observation_request"] == "reobserve_with_independent_view_or_frequency"
    assert out["object_category"] == "unknown"


def test_future_observations_are_embargoed_and_clock_monotonic():
    m = CausalEvidenceMemory()
    with pytest.raises(ValueError, match="future"):
        m.ingest(rf(2), now_s=0.1)
    assert m.explanation()["pair_count"] == 0
    m.ingest(rf(2), now_s=0.24)
    with pytest.raises(ValueError, match="rewind"):
        m.ingest(visual(2), now_s=0.1)
    m.ingest(visual(2), now_s=0.28)
    assert m.explanation()["pair_count"] == 1
    with pytest.raises(ValueError, match="backwards"):
        m.advance(0.26)


def test_repeated_packet_cannot_multiply_support():
    mem = CausalEvidenceMemory()
    feed(mem, 0)
    before = mem.explanation()
    with pytest.raises(ValueError, match="Duplicate"):
        mem.ingest(rf(0), now_s=0.08)
    assert mem.explanation()["pair_count"] == before["pair_count"]


def test_memory_evicts_stale_and_old_pairs_no_longer_vote():
    mem = CausalEvidenceMemory(max_frames=3)
    for frame in range(5):
        feed(mem, frame)
    result = mem.explanation()
    assert result["pair_count"] == 3
    assert [x["frame"] for x in result["comparisons"]] == [2, 3, 4]
    mem.ingest(rf(0, uid=991, available=0.55), now_s=0.55)
    assert mem.ignored_stale == 1
    assert mem.explanation()["pair_count"] == 3


def test_delayed_older_frame_does_not_evict_newer_one():
    mem = CausalEvidenceMemory(max_frames=3)
    for frame in (1, 3, 4):
        feed(mem, frame)
    mem.ingest(rf(2, uid=998, available=0.48), now_s=0.48)
    assert sorted(mem.frames) == [2, 3, 4]
    assert 4 in mem.frames


def test_wrong_modality_packet_fails_without_mutating_dedup_state():
    m = CausalEvidenceMemory()
    m.ingest(rf(0), now_s=0.08)
    with pytest.raises(ValueError, match="same-modality"):
        m.ingest(rf(0, uid=1000), now_s=0.08)
    assert 1000 not in m.seen
    assert m.explanation()["pair_count"] == 0


def test_different_capture_times_do_not_produce_false_pair_evidence():
    mem = CausalEvidenceMemory(max_lag_s=0.05)
    mem.ingest(rf(0), now_s=0.2)
    mem.ingest(visual(0, captured=0.18, available=0.19), now_s=0.2)
    assert mem.explanation()["pair_count"] == 0
    assert mem.incomparable == 1


def test_reasoner_simulates_hypothesis_discriminating_viewpoint_not_identity():
    mem = CausalEvidenceMemory()
    assert mem.counterfactual_parallax_test() is None
    mem.ingest(rf(0), now_s=0.05)
    baseline = mem.counterfactual_parallax_test(max_lateral_baseline_m=0)
    shifted = mem.counterfactual_parallax_test(max_lateral_baseline_m=4)
    assert baseline["separation_deg"] == pytest.approx(0)
    assert shifted["separation_deg"] > 0
    assert shifted["identity_confirmed"] is False
    assert shifted["hypothesis_other"].startswith("distinct")
    assert "not measured" in shifted["note"]
    with pytest.raises(ValueError):
        mem.counterfactual_parallax_test(alternative_depth_offset_m=-1)


def test_old_agreement_expires_instead_of_remaining_live_identity_evidence():
    m = CausalEvidenceMemory(max_evidence_age_s=0.25)
    feed(m, 0)
    feed(m, 1)
    assert m.explanation()["association_status"] == "angularly_consistent_identity_unresolved"
    m.advance(0.6)
    out = m.explanation()
    assert out["association_status"] == "stale_evidence"
    assert out["evidence_expired"] is True
    assert out["identity_confirmed"] is False


def test_absent_camera_stays_unknown_and_does_not_invent_bearing_motion():
    m = CausalEvidenceMemory()
    m.ingest(rf(0), now_s=0.05)
    m.advance(0.5)
    result = m.explanation()
    assert result["pair_count"] == 0
    assert result["association_status"] == "insufficient_evidence"
    assert result["radial_prediction"]["range_m"] > 18
    assert result["radial_prediction"]["bearing_at_current_time"].startswith("unknown")
    assert result["object_category"] == "unknown"


def test_opposite_arrival_order_same_result():
    a, b = CausalEvidenceMemory(), CausalEvidenceMemory()
    for frame in (0, 1, 2):
        feed(a, frame)
        feed(b, frame, reverse=True)
    assert a.explanation()["relative_same_object_score_uncalibrated"] == pytest.approx(
        b.explanation()["relative_same_object_score_uncalibrated"]
    )


def test_camera_extracts_only_bearing_not_identity_or_range():
    image = render_synthetic_camera(22.0, seed=8)
    observation = detect_visual_bearing(image, uid=8, frame=0, capture_s=0, available_s=0.04)
    assert observation is not None
    assert abs(math.degrees(observation.bearing_rad) - 22.0) < 3.0
    assert not hasattr(observation, "material")
    assert not hasattr(observation, "range_m")


def test_occlusion_and_distractor_trigger_abstention():
    occ = render_synthetic_camera(22, occluded=True)
    ambiguous = render_synthetic_camera(22, distractor_bearing_deg=-35)
    for image in (occ, ambiguous):
        assert detect_visual_bearing(image, uid=8, frame=0, capture_s=0, available_s=0.04) is None


def test_future_truth_changes_cannot_change_rf_evidence():
    base = {
        "schema_version": 1,
        "truth_for_evaluation_only": {"range_m": 3, "bearing_deg": 50},
        "sensing": {"range_m": 19, "bearing_deg": 16, "radial_speed_mps": 1.2},
    }
    evidence = from_phy_result(base, uid=1, frame=0, capture_s=0.0, available_s=0.04)
    base["truth_for_evaluation_only"]["range_m"] = 99999
    base["truth_for_evaluation_only"]["bearing_deg"] = -55
    assert from_phy_result(base, uid=1, frame=0, capture_s=0.0, available_s=0.04) == evidence


@pytest.mark.parametrize(
    "bad",
    [
        {"range_m": -1},
        {"range_m": float("nan")},
        {"bearing_rad": math.pi},
        {"bearing_sigma_rad": -1},
        {"speed_sigma_mps": float("inf")},
        {"capture_s": -0.1},
        {"available_s": 0},
    ],
)
def test_rf_input_contract_fail_closed(bad):
    params = rf(1).__dict__.copy()
    params.update(bad)
    with pytest.raises(ValueError):
        RFPoint(**params)


def test_max_id_budget_fails_closed():
    mem = CausalEvidenceMemory(max_frames=2, max_ids=4)
    for frame in (0, 1):
        feed(mem, frame)
    with pytest.raises(OverflowError, match="budget"):
        mem.ingest(rf(2), now_s=0.4)


def test_bad_image_rejected_no_uncalibrated_posterior():
    with pytest.raises(ValueError):
        detect_visual_bearing(np.full((3, 64, 64), float("nan")), uid=0, frame=0, capture_s=0, available_s=0)
    with pytest.raises(ValueError):
        detect_visual_bearing(np.zeros((3, 8, 8)), uid=0, frame=0, capture_s=0, available_s=0)
