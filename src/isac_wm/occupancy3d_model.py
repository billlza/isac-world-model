"""Random-initialized geometry readout from causal, calibrated RF sufficient statistics.

The RF backprojection is a fixed physical operation, not a pretrained encoder.
This module models static discrete weak-point-scatterer occupancy. It does not implement
opaque-surface rendering, object motion, or a learned uncertainty certificate.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def weight_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _features(
    numerator: np.ndarray,
    energy: np.ndarray,
    coordinates_m: np.ndarray,
    shape: tuple[int, int, int],
    reference_energy: float,
    measurement_fraction: float,
) -> np.ndarray:
    """Only measured RF and fixed acquisition geometry enter these features."""
    if reference_energy <= 0 or not 0 < measurement_fraction <= 1:
        raise ValueError("A positive information scale and measurement fraction are required")
    bp = numerator / np.maximum(energy[None], 1e-12)
    center = (coordinates_m.max(axis=0) + coordinates_m.min(axis=0)) / 2
    half_width = (coordinates_m.max(axis=0) - coordinates_m.min(axis=0)) / 2
    xyz = (coordinates_m - center) / np.maximum(half_width, 1e-12)
    batch = numerator.shape[0]
    channels = [
        np.arcsinh(bp.real),
        np.arcsinh(bp.imag),
        np.broadcast_to(np.log1p(energy / reference_energy)[None], bp.shape),
        np.full(bp.shape, measurement_fraction),
        *[np.broadcast_to(xyz[None, :, k], bp.shape) for k in range(3)],
    ]
    result = np.stack(channels, axis=1).reshape(batch, 7, *shape).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite RF features")
    return result


def prefix_features(
    y: np.ndarray,
    matrix: np.ndarray,
    coordinates_m: np.ndarray,
    shape: tuple[int, int, int],
    n_frequencies: int,
    prefix_poses: int,
) -> np.ndarray:
    """Causal prefix; changing RF after prefix_poses cannot change the output."""
    y, matrix = np.asarray(y), np.asarray(matrix)
    if y.ndim != 2 or matrix.ndim != 2 or y.shape[1] != matrix.shape[0]:
        raise ValueError("Expected observations [batch,measurement] and operator [measurement,voxel]")
    if matrix.shape[1] != int(np.prod(shape)) or coordinates_m.shape != (matrix.shape[1], 3):
        raise ValueError("Grid and operator dimensions disagree")
    if n_frequencies < 1 or matrix.shape[0] % n_frequencies:
        raise ValueError("Operator rows must be complete poses")
    rows = int(prefix_poses) * n_frequencies
    if prefix_poses != int(prefix_poses) or not 0 < rows <= matrix.shape[0]:
        raise ValueError("Invalid prefix")
    selected = matrix[:rows]
    if not np.isfinite(y[:, :rows]).all() or not np.isfinite(matrix).all():
        raise ValueError("Non-finite accepted RF or operator")
    numerator = y[:, :rows] @ selected.conj()
    energy = np.square(np.abs(selected)).sum(axis=0)
    reference = float(np.square(np.abs(matrix)).sum(axis=0).mean())
    return _features(numerator, energy, coordinates_m, shape, reference, rows / matrix.shape[0])


class StaticRFMemory:
    """Additive static-scene RF memory with explicit receipt/capture contracts.

    Older captures that arrive late remain useful to a static map; this memory
    accepts them once, provided they are available by the current update time.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        coordinates_m: np.ndarray,
        shape: tuple[int, int, int],
        n_frequencies: int,
        batch_size: int = 1,
    ):
        self.matrix = np.asarray(matrix)
        self.coordinates_m = np.asarray(coordinates_m)
        self.shape = tuple(shape)
        self.n_frequencies = int(n_frequencies)
        if batch_size < 1 or n_frequencies < 1 or self.matrix.ndim != 2:
            raise ValueError("Invalid memory dimensions")
        if self.matrix.shape[0] % n_frequencies or self.matrix.shape[1] != int(np.prod(shape)):
            raise ValueError("Grid and measurement dimensions disagree")
        if self.coordinates_m.shape != (self.matrix.shape[1], 3):
            raise ValueError("Coordinate shape mismatch")
        if not np.isfinite(self.matrix).all() or not np.isfinite(self.coordinates_m).all():
            raise ValueError("Non-finite observation geometry")
        self.numerator = np.zeros((batch_size, self.matrix.shape[1]), np.complex128)
        self.energy = np.zeros(self.matrix.shape[1], np.float64)
        self.reference_energy = float(np.square(np.abs(self.matrix)).sum(axis=0).mean())
        self.accepted: set[int] = set()
        self.now_s = -float("inf")

    def update(
        self, y_pose: np.ndarray, pose_index: int, *, capture_s: float, available_s: float, now_s: float
    ) -> None:
        if not np.isfinite([capture_s, available_s, now_s]).all():
            raise ValueError("Timestamps must be finite")
        if not capture_s <= available_s <= now_s or now_s < self.now_s:
            raise ValueError("Capture/availability/current-time contract violated")
        if pose_index != int(pose_index) or not 0 <= pose_index < self.matrix.shape[0] // self.n_frequencies:
            raise ValueError("Invalid pose index")
        if pose_index in self.accepted:
            raise ValueError("Duplicate measurement cannot be fused twice")
        y_pose = np.asarray(y_pose)
        if y_pose.shape != (self.numerator.shape[0], self.n_frequencies) or not np.isfinite(y_pose).all():
            raise ValueError("Invalid measured RF batch")
        start = pose_index * self.n_frequencies
        block = self.matrix[start : start + self.n_frequencies]
        contribution = y_pose @ block.conj()
        information = np.square(np.abs(block)).sum(axis=0)
        self.numerator += contribution
        self.energy += information
        self.accepted.add(pose_index)
        self.now_s = now_s

    def features(self) -> np.ndarray:
        return _features(
            self.numerator,
            self.energy,
            self.coordinates_m,
            self.shape,
            self.reference_energy,
            len(self.accepted) * self.n_frequencies / self.matrix.shape[0],
        )


class OccupancyReadout(nn.Module):
    """Small 3D encoder/decoder; all trainable layers start from random weights."""

    def __init__(self, channels: int = 12):
        super().__init__()
        if channels < 4:
            raise ValueError("At least four feature channels are required")
        self.channels = channels
        self.local = nn.Sequential(
            nn.Conv3d(7, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.SiLU(),
        )
        self.context = nn.Sequential(
            nn.AvgPool3d(2),
            nn.Conv3d(channels, 2 * channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(2 * channels, 2 * channels, 3, padding=1),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Conv3d(3 * channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(channels, 1, 1),
        )
        nn.init.constant_(self.output[-1].bias, -2.0)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 5 or features.shape[1] != 7:
            raise ValueError("Features must have shape [batch,7,x,y,z]")
        local = self.local(features)
        context = F.interpolate(
            self.context(local), size=local.shape[-3:], mode="trilinear", align_corners=False
        )
        return self.output(torch.cat([local, context], dim=1)).squeeze(1)


def geometry_loss(logits: Tensor, target: Tensor) -> Tensor:
    if logits.shape != target.shape or logits.ndim != 4:
        raise ValueError("Geometry prediction and supervision must be [batch,x,y,z]")
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = logits.sigmoid().flatten(1)
    truth = target.flatten(1)
    dice = (2 * (prob * truth).sum(1) + 1) / (prob.sum(1) + truth.sum(1) + 1)
    return bce + 0.5 * (1 - dice.mean())


def rf_consistency_loss(probability: Tensor, observed_y: Tensor, matrix: Tensor) -> Tensor:
    """No geometry labels in this term; target is accepted measured complex RF."""
    if not observed_y.is_complex() or not matrix.is_complex():
        raise ValueError("RF consistency requires complex calibrated observations and operator")
    flat = probability.flatten(1).to(matrix.dtype)
    predicted = flat @ matrix.T
    if predicted.shape != observed_y.shape:
        raise ValueError("Predicted and accepted RF dimensions disagree")
    residual = (predicted - observed_y).abs().square().mean(1)
    energy = observed_y.abs().square().mean(1).clamp_min(1e-8)
    return (residual / energy).mean()
