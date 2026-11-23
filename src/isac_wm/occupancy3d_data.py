"""Controlled 3D RF inverse problem under a first-Born scattering approximation.

The target is binary 3D scattering support. Each occupied cell contains one
weak point scatterer at its center, with a known real strength in the main
experiment. A cuboid labels an arrangement of these discrete scatterers, not an
accurate electromagnetic integral over a uniform solid dielectric cuboid.
The constant cell-volume factor is a chosen scatterer-strength weight absorbed
into calibration. Shadowing, multiple scattering, material interfaces and
antenna directivity are deliberately absent. These assumptions make the
observation operator explicit and permit independent support-geometry and
held-out RF evaluation.

All grids use (x, y, z) axis order and C-order flattening. Labels and geometry
specifications never enter ``ObservationBatch.model_inputs``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

C_M_S = 299_792_458.0
DATA_VERSION = "occupancy3d-born-v1"
Layout = Literal["same_height", "height_diverse"]


@dataclass(frozen=True)
class GridSpec:
    shape: tuple[int, int, int] = (12, 12, 8)
    lower_m: tuple[float, float, float] = (-3.0, -3.0, 0.0)
    upper_m: tuple[float, float, float] = (3.0, 3.0, 3.0)

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(int(n) != n or n < 2 for n in self.shape):
            raise ValueError("Grid shape must contain three integers >= 2")
        if len(self.lower_m) != 3 or len(self.upper_m) != 3:
            raise ValueError("Grid bounds must contain three coordinates")
        if not np.all(np.isfinite([self.lower_m, self.upper_m])):
            raise ValueError("Grid bounds must be finite")
        if any(lo >= hi for lo, hi in zip(self.lower_m, self.upper_m, strict=True)):
            raise ValueError("Grid lower bounds must precede upper bounds")

    @property
    def spacing_m(self) -> tuple[float, float, float]:
        return tuple((hi - lo) / n for lo, hi, n in
                     zip(self.lower_m, self.upper_m, self.shape, strict=True))

    @property
    def coordinates_m(self) -> np.ndarray:
        axes = [lo + (np.arange(n, dtype=np.float64) + 0.5) * step
                for lo, n, step in zip(self.lower_m, self.shape, self.spacing_m, strict=True)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape))

    @property
    def voxel_volume_m3(self) -> float:
        return float(np.prod(self.spacing_m))


@dataclass(frozen=True)
class RFConfig:
    n_poses: int = 12
    n_frequencies: int = 32
    carrier_hz: float = 24e9
    bandwidth_hz: float = 500e6
    radius_m: float = 5.0
    noise_std: float = 0.02
    heldout_n_poses: int = 8

    def __post_init__(self) -> None:
        for value in (self.n_poses, self.n_frequencies, self.heldout_n_poses):
            if int(value) != value or value < 2:
                raise ValueError("Pose and frequency counts must be integers >= 2")
        if not np.all(np.isfinite([self.carrier_hz, self.bandwidth_hz,
                                   self.radius_m, self.noise_std])):
            raise ValueError("RF configuration must be finite")
        if self.bandwidth_hz <= 0 or self.carrier_hz <= self.bandwidth_hz / 2:
            raise ValueError("RF frequencies must be strictly positive")
        if self.radius_m <= 0 or self.noise_std < 0:
            raise ValueError("Radius must be positive and noise nonnegative")


def point_scattering_matrix(
    coordinates_m: np.ndarray,
    sensors_m: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    cell_volume_m3: float = 1.0,
    normalization: float = 1.0,
) -> np.ndarray:
    """Return A[(pose,frequency),voxel] for coherent monostatic scattering.

    A = cell_volume / (normalization * range**2) * exp(-j*4*pi*f*range/c).
    The propagation-amplitude convention absorbs known system constants and
    scatterer strength into a common calibration. The cell-volume factor is a
    fixed point-strength convention, not a claim of continuum integration
    accuracy at coarse voxel spacing. Frequencies vary fastest.
    """
    xyz = np.asarray(coordinates_m, dtype=np.float64)
    sensors = np.asarray(sensors_m, dtype=np.float64)
    freq = np.asarray(frequencies_hz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or sensors.ndim != 2 or sensors.shape[1] != 3:
        raise ValueError("Coordinates and sensors must have shape [count,3]")
    if not len(xyz) or not len(sensors) or freq.ndim != 1 or not len(freq):
        raise ValueError("Scattering matrix inputs must be nonempty")
    if not all(np.all(np.isfinite(x)) for x in (xyz, sensors, freq)) or np.any(freq <= 0):
        raise ValueError("Coordinates must be finite and frequencies positive")
    if not np.isfinite(cell_volume_m3) or cell_volume_m3 <= 0:
        raise ValueError("Cell volume must be finite and positive")
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("Normalization must be finite and positive")
    ranges = np.linalg.norm(sensors[:, None, :] - xyz[None, :, :], axis=-1)
    if np.any(ranges <= 1e-8):
        raise ValueError("A scatterer coincides with a sensor")
    phase = (-4.0 * np.pi / C_M_S) * freq[None, :, None] * ranges[:, None, :]
    amplitude = (cell_volume_m3 / normalization) / ranges[:, None, :] ** 2
    return (amplitude * np.exp(1j * phase)).reshape(-1, len(xyz)).astype(np.complex64)


@dataclass(frozen=True)
class RFOperator:
    matrix: np.ndarray
    sensors_m: np.ndarray
    frequencies_hz: np.ndarray
    grid: GridSpec
    config: RFConfig
    layout: Layout
    heldout: bool
    normalization: float

    def forward(self, occupancy: np.ndarray) -> np.ndarray:
        values = np.asarray(occupancy, dtype=np.float32)
        if values.shape[-3:] == self.grid.shape:
            flat = values.reshape(*values.shape[:-3], self.grid.n_voxels)
        elif values.shape[-1:] == (self.grid.n_voxels,):
            flat = values
        else:
            raise ValueError("Input must end in the grid shape or flattened voxel count")
        if not np.all(np.isfinite(flat)):
            raise ValueError("Scattering density must be finite")
        return np.asarray(flat @ self.matrix.T, dtype=np.complex64)

    def metadata(self) -> dict:
        max_range = float(np.linalg.norm(
            self.sensors_m[:, None, :] - self.grid.coordinates_m[None, :, :], axis=-1).max())
        return {
            "data_version": DATA_VERSION,
            "model": "coherent_monostatic_discrete_first_Born_point_scattering",
            "grid": asdict(self.grid),
            "rf_config": asdict(self.config),
            "layout": self.layout,
            "heldout": self.heldout,
            "sensors_m": self.sensors_m.tolist(),
            "frequencies_hz": self.frequencies_hz.tolist(),
            "normalization": self.normalization,
            "normalization_rule": "voxel_volume/radius^2*sqrt(training_poses*frequencies)",
            "conventional_range_resolution_m": C_M_S / (2 * self.config.bandwidth_hz),
            "stepped_frequency_unambiguous_range_m": C_M_S / (2 * np.diff(self.frequencies_hz)[0]),
            "maximum_sensor_voxel_range_m": max_range,
            "operator_shape": list(self.matrix.shape),
            "operator_sha256": hashlib.sha256(self.matrix.tobytes()).hexdigest(),
            "assumptions": ["known sensor poses", "coherent calibrated complex RF",
                            "isotropic antenna and scatterer response", "single scattering",
                            "no occlusion or multipath", "unit relative scatterer strength in nominal model",
                            "one weak point scatterer per occupied voxel center",
                            "cuboid-shaped discrete support, not continuum dielectric cuboids"],
        }


def make_operator(
    grid: GridSpec = GridSpec(),
    rf: RFConfig = RFConfig(),
    layout: Layout = "same_height",
    *,
    heldout: bool = False,
) -> RFOperator:
    if layout not in ("same_height", "height_diverse"):
        raise ValueError("Unknown measurement layout")
    count = rf.heldout_n_poses if heldout else rf.n_poses
    # An irrational angular fraction avoids coincidences between the two grids.
    offset = np.pi / np.sqrt(7.0) if heldout else 0.0
    angles = np.arange(count, dtype=np.float64) * (2.0 * np.pi / count) + offset
    center = (np.asarray(grid.lower_m) + np.asarray(grid.upper_m)) / 2
    sensors = np.column_stack((center[0] + rf.radius_m * np.cos(angles),
                               center[1] + rf.radius_m * np.sin(angles),
                               np.full(count, center[2])))
    if layout == "height_diverse":
        height = grid.upper_m[2] - grid.lower_m[2]
        fractions = np.array([1 / 6, 5 / 6, 1 / 3, 2 / 3], dtype=np.float64)
        sensors[:, 2] = grid.lower_m[2] + height * np.resize(fractions, count)
    frequencies = np.linspace(rf.carrier_hz - rf.bandwidth_hz / 2,
                              rf.carrier_hz + rf.bandwidth_hz / 2, rf.n_frequencies)
    unambiguous_range = C_M_S / (2 * (frequencies[1] - frequencies[0]))
    max_range = np.linalg.norm(sensors[:, None, :] - grid.coordinates_m[None, :, :], axis=-1).max()
    if max_range >= unambiguous_range:
        raise ValueError("Scene exceeds stepped-frequency unambiguous range; adjust bandwidth or frequency count")
    normalization = (grid.voxel_volume_m3 / rf.radius_m**2
                     * np.sqrt(rf.n_poses * rf.n_frequencies))
    matrix = point_scattering_matrix(grid.coordinates_m, sensors, frequencies,
                                     cell_volume_m3=grid.voxel_volume_m3,
                                     normalization=normalization)
    return RFOperator(matrix, sensors, frequencies, grid, rf, layout, heldout, normalization)


def scene_hash(occupancy: np.ndarray, grid: GridSpec) -> str:
    values = np.asarray(occupancy)
    if values.shape != grid.shape or not np.all((values == 0) | (values == 1)):
        raise ValueError("Scene hash requires binary occupancy with the declared shape")
    header = json.dumps(asdict(grid), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + np.packbits(values.astype(np.uint8)).tobytes()).hexdigest()


@dataclass(frozen=True)
class SceneBatch:
    occupancy: np.ndarray
    scene_hashes: tuple[str, ...]
    pair_ids: tuple[str, ...]
    specs: tuple[dict, ...]
    grid: GridSpec
    seed: int

    @property
    def hashes(self) -> tuple[str, ...]:
        return self.scene_hashes

    @property
    def dataset_hash(self) -> str:
        return hashlib.sha256("\n".join(self.scene_hashes).encode()).hexdigest()

    def metadata(self) -> dict:
        return {"data_version": DATA_VERSION, "seed": self.seed, "grid": asdict(self.grid),
                "scene_count": len(self.occupancy), "dataset_sha256": self.dataset_hash,
                "scene_hashes": list(self.scene_hashes), "mirror_pair_ids": list(self.pair_ids),
                "scene_specs_for_supervision_and_audit_only": list(self.specs)}


def generate_scenes(
    n: int,
    seed: int,
    grid: GridSpec = GridSpec(),
    *,
    excluded_hashes: tuple[str, ...] | set[str] = (),
    include_wall_panels: bool = False,
) -> SceneBatch:
    """Generate independent cuboid families; each family and its z mirror stay together.

    Suspended as well as floor-touching volumes are intentional: a floor-only
    prior would break the mirror ambiguity without additional RF information.
    Optional panels are one-cell-thick finite volumes, not full room walls.
    """
    if int(n) != n or n < 2 or n % 2:
        raise ValueError("Scene count must be a positive even number >= 2")
    if min(grid.shape) < 4:
        raise ValueError("Cuboid generator requires at least four cells per axis")
    rng = np.random.default_rng(seed)
    occupied, hashes, pair_ids, specs = [], [], [], []
    seen = set(excluded_hashes)
    for _ in range(max(1000, n * 100)):
        if len(occupied) == n:
            break
        scene = np.zeros(grid.shape, dtype=np.float32)
        boxes = []
        for _ in range(int(rng.integers(1, 4))):
            widths = [int(rng.integers(2, min(size, 5) + 1)) for size in grid.shape[:2]]
            widths.append(int(rng.integers(1, min(grid.shape[2], 4) + 1)))
            low = [int(rng.integers(0, size - width + 1))
                   for size, width in zip(grid.shape, widths, strict=True)]
            high = [a + width for a, width in zip(low, widths, strict=True)]
            scene[tuple(slice(a, b) for a, b in zip(low, high, strict=True))] = 1
            boxes.append({"kind": "cuboid", "lower_index": low, "upper_index_exclusive": high})
        if include_wall_panels and rng.random() < 0.5:
            axis = int(rng.integers(0, 2))
            low = [int(rng.integers(0, max(1, size - 3))) for size in grid.shape]
            high = [min(size, a + 3) for size, a in zip(grid.shape, low, strict=True)]
            low[axis] = 0 if rng.random() < 0.5 else grid.shape[axis] - 1
            high[axis] = low[axis] + 1
            scene[tuple(slice(a, b) for a, b in zip(low, high, strict=True))] = 1
            boxes.append({"kind": "finite_wall_panel", "lower_index": low,
                          "upper_index_exclusive": high})
        mirror = scene[:, :, ::-1].copy()
        h1, h2 = scene_hash(scene, grid), scene_hash(mirror, grid)
        if h1 == h2 or h1 in seen or h2 in seen:
            continue
        family = hashlib.sha256("|".join(sorted((h1, h2))).encode()).hexdigest()
        mirrored_boxes = []
        for box in boxes:
            lo, hi = list(box["lower_index"]), list(box["upper_index_exclusive"])
            lo[2], hi[2] = grid.shape[2] - hi[2], grid.shape[2] - lo[2]
            mirrored_boxes.append({"kind": box["kind"], "lower_index": lo,
                                   "upper_index_exclusive": hi})
        occupied.extend((scene, mirror))
        hashes.extend((h1, h2))
        pair_ids.extend((family, family))
        specs.extend(({"boxes": boxes, "mirror": False}, {"boxes": mirrored_boxes, "mirror": True}))
        seen.update((h1, h2))
    if len(occupied) != n:
        raise RuntimeError("Unable to generate enough distinct scene families; no partial dataset returned")
    return SceneBatch(np.stack(occupied), tuple(hashes), tuple(pair_ids), tuple(specs), grid, int(seed))


@dataclass(frozen=True)
class ObservationBatch:
    y: np.ndarray
    clean_y: np.ndarray
    noise_std: float
    noise_seed: int
    operator: RFOperator
    unit_susceptibility: bool

    def model_inputs(self) -> dict[str, np.ndarray]:
        """Only calibrated measurements and instrument geometry, never labels."""
        return {"rf": self.y, "sensors_m": self.operator.sensors_m,
                "frequencies_hz": self.operator.frequencies_hz}

    def metadata(self) -> dict:
        signal_power = np.mean(np.abs(self.clean_y.astype(np.complex128)) ** 2, axis=-1)
        if self.noise_std > 0:
            snr = 10 * np.log10(np.maximum(signal_power, 1e-30) / self.noise_std**2)
            snr_summary = {"min": float(snr.min()), "mean": float(snr.mean()), "max": float(snr.max())}
        else:
            snr_summary = None
        return {"noise_seed": self.noise_seed, "complex_noise_rms": self.noise_std,
                "noise_distribution": "independent_circular_complex_Gaussian",
                "normalization_uses_scene_truth": False,
                "unit_susceptibility": self.unit_susceptibility,
                "snr_db_for_evaluation_only": snr_summary,
                "observations_sha256": hashlib.sha256(self.y.tobytes()).hexdigest()}


def observe_scenes(
    scenes: SceneBatch,
    operator: RFOperator,
    noise_seed: int,
    *,
    noise_std: float | None = None,
    susceptibility: np.ndarray | None = None,
) -> ObservationBatch:
    if scenes.grid != operator.grid:
        raise ValueError("Scene and operator grids disagree")
    sigma = operator.config.noise_std if noise_std is None else float(noise_std)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Noise must be finite and nonnegative")
    density = scenes.occupancy
    if susceptibility is not None:
        material = np.asarray(susceptibility, dtype=np.float32)
        allowed = (scenes.grid.shape, scenes.occupancy.shape)
        if material.shape not in allowed or not np.all(np.isfinite(material)) or np.any(material < 0):
            raise ValueError("Susceptibility must be finite, nonnegative and match one or all scene grids")
        density = density * material
    clean = operator.forward(density)
    rng = np.random.default_rng(noise_seed)
    noise = sigma / np.sqrt(2) * (rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape))
    y = np.asarray(clean + noise, dtype=np.complex64)
    return ObservationBatch(y, clean, sigma, int(noise_seed), operator, susceptibility is None)
