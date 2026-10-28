"""PHY -> delayed RF/vision evidence -> bounded, falsifiable hypotheses.

No simulator ground-truth labels enter the reasoning step. This is a controlled
physical-signal + synthetic-perspective-camera integration diagnostic, not a
trained object classifier, calibrated uncertainty, or a safe driving controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

from isac_wm.evidence_reasoning import (
    CausalEvidenceMemory,
    detect_visual_bearing,
    from_phy_result,
    render_synthetic_camera,
)

SCENARIOS = ("compatible", "conflicting_view", "occluded", "ambiguous_view", "collinear_view")


def _json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run_demo(
    out: Path,
    *,
    frames: int = 4,
    seed: int = 2600,
    probe_fraction: float = 0.4,
    scenario: str = "compatible",
    evaluator=None,
    phy_config=None,
) -> dict:
    """Simulate distinct RF and image evidence of a common moving point.

    evaluator is injected for tests; normal use calls actual noisy PHY waveform
    simulation + matched detection. A capture timestamp is NOT its availability.
    """
    if type(frames) is not int or not 2 <= frames <= 16:
        raise ValueError("Demo requires 2 to 16 independent evidence frames")
    if type(seed) is not int or not 0 <= seed < 2**32 - frames:
        raise ValueError("Invalid seed range")
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    if not isinstance(probe_fraction, (float, int)) or not math.isfinite(probe_fraction):
        raise ValueError("Probe fraction must be finite")
    if not 0 <= probe_fraction < 0.8:
        raise ValueError("Probe fraction outside safe configured range")
    out = Path(out)
    if out.exists():
        raise FileExistsError("Output path must be new; no experiment evidence overwrite")
    from isac_wm.phy_ofdm import OFDMConfig, evaluate_frame

    base = phy_config or OFDMConfig()
    if not isinstance(base, OFDMConfig):
        raise TypeError("PHY configuration must be OFDMConfig")
    eval_frame = evaluator or evaluate_frame
    memory = CausalEvidenceMemory(max_frames=max(4, frames))
    timeline: list[dict] = []
    physiology: list[dict] = []
    communications: list[dict] = []
    events: list = []
    frame_period_s = 0.2
    burst_s = base.symbols * base.pulse_period_s
    if burst_s >= frame_period_s:
        raise ValueError("PHY burst exceeds the proposed observation time frame")
    for frame in range(frames):
        stamp = frame * frame_period_s
        geometry = replace(
            base,
            target_range_m=base.target_range_m + frame * frame_period_s * base.target_radial_speed_mps,
        )
        report = eval_frame(geometry, probe_fraction=probe_fraction, seed=seed + frame)
        # Absolute truth is read here only to drive a separate simulator's image
        # renderer, NEVER as an input to the evidence memory or RF adapter.
        true_bearing = geometry.target_bearing_deg
        if scenario == "conflicting_view":
            true_bearing = max(-58, min(58, true_bearing - 46))
        # In collinear_view the *camera object is distinct* and 7.5 m farther
        # than the radar scatterer, but the current synthetic camera only
        # observes bearing. No input to the reasoner can reveal that depth.
        image = render_synthetic_camera(
            true_bearing,
            seed=seed + 100 + frame,
            occluded=scenario == "occluded",
            distractor_bearing_deg=-35 if scenario == "ambiguous_view" else None,
        )
        # Measurement quality parameters are ASSUMED floors, not calibrated
        # confidence read out of a matched-filter's peak energy.
        radar = from_phy_result(
            report,
            uid=3 * frame,
            frame=frame,
            capture_s=stamp,
            available_s=stamp + burst_s + 0.035,
        )
        camera = detect_visual_bearing(
            image,
            uid=3 * frame + 1,
            frame=frame,
            capture_s=stamp,
            available_s=stamp + 0.09,
        )
        events.append(radar)
        if camera is not None:
            events.append(camera)
        physiology.append(
            {
                "frame": frame,
                "rf_estimate_m": radar.range_m,
                "rf_bearing_deg": math.degrees(radar.bearing_rad),
                "rf_radial_speed_mps": radar.radial_speed_mps,
                "vision_bearing_deg": None if camera is None else math.degrees(camera.bearing_rad),
                "camera_had_detection": camera is not None,
            }
        )
        communications.append(
            {
                "frame": frame,
                "uncoded_ber": report["communication"]["ber"],
                "nominal_payload_rate_bps": report["communication"]["nominal_raw_payload_rate_bps"],
                "probe_fraction_realized": report["probe_fraction_realized"],
            }
        )

    events.sort(key=lambda e: (e.available_s, e.uid))
    for item in events:
        # A visible pre-release snapshot explicitly demonstrates embargo:
        # the incoming packet cannot influence the inference before availability.
        before = memory.explanation()
        memory.ingest(item, now_s=item.available_s)
        after = memory.explanation()
        timeline.append(
            {
                "frame": item.frame,
                "uid": item.uid,
                "modality": "rf" if item.uid % 3 == 0 else "vision",
                "capture_s": item.capture_s,
                "available_s": item.available_s,
                "belief_before_release": {
                    "status": before["association_status"],
                    "pair_count": before["pair_count"],
                },
                "belief_after_release": {
                    "status": after["association_status"],
                    "pair_count": after["pair_count"],
                },
            }
        )
    memory.advance(events[-1].available_s + 0.05)
    result = memory.explanation()
    summary = {
        "schema_version": 1,
        "scenario": scenario,
        "frames": frames,
        "rf_events": len(physiology),
        "event_count": len(events),
        "pair_count": result["pair_count"],
        "association_status": result["association_status"],
        "identity_confirmed": result["identity_confirmed"],
        "object_category": result["object_category"],
        "material": result["material"],
        "hypothesis_evidence": result,
        "limitations": [
            "No trained category detector or material/shape inversion exists",
            "Visual evidence is an independent synthetic perspective bearing cue, not a natural photograph",
            "A single radar peak and radial speed do not identify an object",
            "Angular agreement across modalities cannot rule out collinear distinct objects",
            "Sensor covariance floors and likelihoods are uncalibrated modeling assumptions",
            "Pilot CSI / BER are simulated; computation time and RF hardware are not validated",
        ],
    }
    root_sources = [Path(__file__), Path(__file__).parents[1] / "src/isac_wm/evidence_reasoning.py"]
    provenance = {
        "version": "research-phy-reasoning-v1",
        "seed": seed,
        "probe_fraction": probe_fraction,
        "scenario": scenario,
        "model": "two-hypothesis Gaussian-angle versus unrelated-uniform evidence score",
        "collinear_counterexample": scenario == "collinear_view",
        "collinear_visual_target_range_m_evaluation_only": (
            base.target_range_m + 7.5 if scenario == "collinear_view" else None
        ),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root_sources},
        "PHY_config": base.to_dict(),
        "synthetic_truth_for_evaluation_only": {
            "initial_range_m": base.target_range_m,
            "bearing_deg": base.target_bearing_deg,
            "radial_speed_mps": base.target_radial_speed_mps,
        },
    }
    out.mkdir(parents=True, exist_ok=False)
    _json(out / "manifest.json", provenance)
    _json(out / "timeline.json", timeline)
    _json(out / "sensor_observations.json", physiology)
    _json(out / "communication.json", communications)
    _json(out / "hypotheses.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2600)
    parser.add_argument("--probe-fraction", type=float, default=0.4)
    parser.add_argument("--scenario", choices=SCENARIOS, default="compatible")
    a = parser.parse_args()
    result = run_demo(
        a.out, frames=a.frames, seed=a.seed, probe_fraction=a.probe_fraction, scenario=a.scenario
    )
    print(json.dumps({"status": result["association_status"], "files": str(a.out)}, indent=2))


if __name__ == "__main__":
    main()
