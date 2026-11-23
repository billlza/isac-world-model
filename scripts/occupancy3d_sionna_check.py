"""Run a bounded Sionna RT CPU check of the 3D point-scatterer measurement law.

This is a separate forward-model check. Its four noiseless single-target scenes
are not training or test data for the learned occupancy benchmark.

Optional dependencies: sionna-rt==2.2.0 (Mitsuba 3.9.1, Dr.Jit 1.5.0).
Reference API: https://nvlabs.github.io/sionna/rt/tutorials/RCS.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0


def run_check(*, threads: int = 2, frequency_hz: float = 60e9) -> dict:
    """Check radar-equation power, propagation delay, and a height ambiguity."""
    if threads < 1 or not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("threads and frequency_hz must be positive")

    import drjit as dr
    import mitsuba as mi

    # Select the CPU explicitly; never substitute a different backend silently.
    mi.set_variant("llvm_ad_mono_polarized")
    dr.set_thread_count(threads)
    from sionna.rt import PlanarArray, Receiver, Transmitter, load_scene
    from sionna.rt.rcs import ConstantRCSSensingTarget, RCSSolver

    if version("sionna-rt") != "2.2.0":
        raise RuntimeError("This reproducibility check requires sionna-rt==2.2.0")
    if mi.variant() != "llvm_ad_mono_polarized":
        raise RuntimeError("Sionna import changed the requested LLVM CPU backend")

    settings = {
        "max_depth": 1,
        "buffer_size_per_sp": 1024,
        "samples_per_sp": 256,
        "synthetic_array": False,
        "los": True,
        "specular_reflection": False,
        "refraction": False,
        "seed": 20261123,
    }
    sigma_m2 = 1.0
    case_positions = [
        ("same_height_A", [0.0, 0.0, 1.5], [2.0, 0.0, 0.75]),
        ("same_height_B", [0.0, 0.0, 1.5], [2.0, 0.0, 2.25]),
        ("lower_sensor_A", [0.0, 0.0, 0.75], [2.0, 0.0, 0.75]),
        ("lower_sensor_B", [0.0, 0.0, 0.75], [2.0, 0.0, 2.25]),
    ]
    records = []
    began = time.perf_counter()
    solver = RCSSolver(deterministic=True)
    for name, sensor_position, target_position in case_positions:
        case_began = time.perf_counter()
        scene = load_scene()
        scene.frequency = frequency_hz
        scene.tx_array = PlanarArray(
            num_rows=1, num_cols=1, pattern="iso", polarization="V"
        )
        scene.rx_array = scene.tx_array
        scene.add(Transmitter("tx", position=sensor_position))
        scene.add(Receiver("rx", position=sensor_position))
        scene.add(
            ConstantRCSSensingTarget(
                "target",
                sigma=sigma_m2,
                position=target_position,
                length=0.1,
                width=0.1,
                height=0.1,
            )
        )
        paths = solver(scene, **settings)
        amplitude, delay = paths.cir(out_type="numpy", normalize_delays=False)
        amplitude = np.asarray(amplitude).reshape(-1)
        delay = np.asarray(delay).reshape(-1)
        if amplitude.size != 1 or delay.size != 1:
            raise RuntimeError(f"{name}: expected one direct sensing path, got {amplitude.shape}/{delay.shape}")
        a = complex(amplitude[0])
        tau_s = float(delay[0])
        if tau_s <= 0 or not np.isfinite(a.real + a.imag + tau_s):
            raise RuntimeError(f"{name}: missing or non-finite sensing path")

        distance_m = float(np.linalg.norm(np.asarray(target_position) - sensor_position))
        wavelength_m = float(scene.wavelength.numpy()[0])
        expected_power = sigma_m2 * wavelength_m**2 / ((4 * np.pi) ** 3 * distance_m**4)
        measured_power = float(abs(a) ** 2)
        measured_range_m = tau_s * SPEED_OF_LIGHT / 2
        relative_power_error = abs(measured_power / expected_power - 1)
        range_error_m = abs(measured_range_m - distance_m)
        # This tolerance accommodates float32 path coefficients and delays.
        if relative_power_error > 1e-4 or range_error_m > 1e-5:
            raise AssertionError(f"{name}: radar-equation or delay agreement failed")
        records.append(
            {
                "case": name,
                "sensor_position_m": sensor_position,
                "target_position_m": target_position,
                "analytic_range_m": distance_m,
                "sionna_range_m": measured_range_m,
                "absolute_range_error_m": range_error_m,
                "analytic_channel_power": expected_power,
                "sionna_channel_power": measured_power,
                "relative_power_error": relative_power_error,
                "path_delay_s": tau_s,
                "path_amplitude_real_imag": [a.real, a.imag],
                "elapsed_s": time.perf_counter() - case_began,
            }
        )

    same_range_difference_m = abs(records[0]["sionna_range_m"] - records[1]["sionna_range_m"])
    different_range_difference_m = records[3]["sionna_range_m"] - records[2]["sionna_range_m"]
    if same_range_difference_m > 1e-5 or abs(different_range_difference_m - 0.5) > 1e-5:
        raise AssertionError("The Sionna delay estimates failed the stated mirror-height check")
    return {
        "schema": "isac.occupancy3d.sionna_forward_check.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "passed",
        "role": "independent_forward_model_check_not_used_for_occupancy_training",
        "scope": (
            "Four noiseless single isotropic RCS targets in empty scenes, known monostatic poses. "
            "Validates radial attenuation and propagation delay, not room reconstruction, "
            "multipath, occlusion, material recovery, or sim-to-real accuracy. "
            "Complex amplitudes are recorded but no full coherent phase-agreement claim is made."
        ),
        "versions": {p: version(p) for p in ["sionna-rt", "mitsuba", "drjit", "numpy", "torch"]},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend": mi.variant(),
        "threads": threads,
        "frequency_hz": frequency_hz,
        "sigma_m2": sigma_m2,
        "solver": {"deterministic": True, **settings},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "documentation": [
            "https://nvlabs.github.io/sionna/rt/tutorials/RCS.html",
            "https://nvlabs.github.io/sionna/rt/api/rcs.html",
        ],
        "cases": records,
        "summary": {
            "same_height_range_difference_m": same_range_difference_m,
            "lower_sensor_range_difference_m": different_range_difference_m,
            "max_absolute_range_error_m": max(r["absolute_range_error_m"] for r in records),
            "max_relative_power_error": max(r["relative_power_error"] for r in records),
            "elapsed_s": time.perf_counter() - began,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    result = run_check(threads=args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": result["status"], **result["summary"]}))


if __name__ == "__main__":
    main()
