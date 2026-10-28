"""Integration contract: physical estimator -> separate camera -> timed memory."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("phy_reasoning_demo", ROOT / "scripts/phy_reasoning_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def fake_evaluator(cfg, *, probe_fraction, seed):
    # Return a realistic PHY report shape with simulator truth deliberately wrong.
    return {
        "schema_version": 1,
        "truth_for_evaluation_only": {
            "range_m": 0.001,
            "bearing_deg": -59,
            "radial_speed_mps": -200,
        },
        "probe_fraction_realized": probe_fraction,
        "sensing": {
            "range_m": 18.75,
            "bearing_deg": 18,
            "radial_speed_mps": 1.25,
        },
        "communication": {
            "ber": 0.02,
            "nominal_raw_payload_rate_bps": 750_000,
        },
    }


def check_demo(tmp_path, scenario):
    pytest.importorskip("isac_wm.phy_ofdm")
    return demo.run_demo(tmp_path / scenario, evaluator=fake_evaluator, scenario=scenario)


def test_matched_sensors_support_consistent_hypothesis(tmp_path):
    out = check_demo(tmp_path, "compatible")
    assert out["association_status"] == "angularly_consistent_identity_unresolved"
    assert out["identity_confirmed"] is False
    assert out["material"] == out["object_category"] == "unknown"
    data = json.loads((tmp_path / "compatible" / "timeline.json").read_text())
    assert data[0]["belief_before_release"]["pair_count"] == 0
    assert data[0]["belief_after_release"]["pair_count"] == 0
    assert any(x["belief_after_release"]["pair_count"] >= 2 for x in data)
    assert all(x["available_s"] >= x["capture_s"] for x in data)


def test_counterfactual_camera_disagreement_triggers_reobservation(tmp_path):
    out = check_demo(tmp_path, "conflicting_view")
    assert out["association_status"] == "cross_modal_conflict"
    assert out["hypothesis_evidence"]["contradicting_pairs"] >= 2


def test_camera_occlusion_never_turns_rf_into_semantic_class(tmp_path):
    out = check_demo(tmp_path, "occluded")
    assert out["association_status"] == "insufficient_evidence"
    assert out["event_count"] == out["frames"]


def test_collinear_counterexample_preserves_unresolved_identity(tmp_path):
    out = check_demo(tmp_path, "collinear_view")
    assert out["association_status"] == "angularly_consistent_identity_unresolved"
    assert out["identity_confirmed"] is False
    assert "distinct_objects_aligned_in_bearing" in out["hypothesis_evidence"]["alternative_explanations"]


def test_distractor_abstains(tmp_path):
    out = check_demo(tmp_path, "ambiguous_view")
    assert out["pair_count"] == 0
    assert out["material"] == "unknown"


def test_reject_existing_output_and_invalid_args(tmp_path):
    pytest.importorskip("isac_wm.phy_ofdm")
    out = tmp_path / "first"
    demo.run_demo(out, frames=2, evaluator=fake_evaluator)
    with pytest.raises(FileExistsError):
        demo.run_demo(out, evaluator=fake_evaluator)
    with pytest.raises(ValueError):
        demo.run_demo(tmp_path / "bad", frames=0, evaluator=fake_evaluator)
    with pytest.raises(ValueError):
        demo.run_demo(tmp_path / "bad", probe_fraction=1.0, evaluator=fake_evaluator)
    with pytest.raises(ValueError):
        demo.run_demo(tmp_path / "bad", scenario="imaginary", evaluator=fake_evaluator)


def test_real_phy_first_stage_contract(tmp_path):
    """This test is run in CI with the actual physical module and pinned stack."""
    from isac_wm.phy_ofdm import OFDMConfig

    config = OFDMConfig(radar_snr_db=18.0)
    out = demo.run_demo(tmp_path / "real", frames=2, seed=2026, phy_config=config)
    assert out["event_count"] >= 2
    report = json.loads((tmp_path / "real" / "sensor_observations.json").read_text())
    assert len(report) == 2
    assert all("rf_estimate_m" in x for x in report)
    assert "truth_for_evaluation_only" not in (tmp_path / "real" / "timeline.json").read_text()
