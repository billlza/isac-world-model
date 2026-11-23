"""Independent metrics and constrained linear baselines for the 3D RF pilot.

The forward contract is ``y = A @ occupancy.ravel()`` for calibrated, real,
voxel-centred point-scattering support in [0, 1]. Arrays use (x, y, z) grid order.
These baselines do not receive scene geometry, object identities, or held-out
measurements. Any threshold or regularization selection belongs on validation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _linear_inputs(operator: np.ndarray, measurements: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    a = np.asarray(operator)
    y = np.asarray(measurements)
    if a.ndim != 2 or min(a.shape) < 1 or y.ndim not in (1, 2):
        raise ValueError("Expected operator [measurements, voxels] and y [batch, measurements]")
    unbatched = y.ndim == 1
    if unbatched:
        y = y[None]
    if y.shape[1] != a.shape[0] or y.shape[0] == 0:
        raise ValueError("Measurement dimensions do not match the operator")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(y)):
        raise ValueError("Linear inverse inputs must be finite")
    dtype = np.complex128 if a.dtype in (np.dtype("complex128"), np.dtype("float64")) else np.complex64
    return np.asarray(a, dtype=dtype), np.asarray(y, dtype=dtype), unbatched


def matched_filter(operator: np.ndarray, measurements: np.ndarray) -> np.ndarray:
    """Column-energy-corrected real backprojection, clipped to physical [0, 1]."""
    a, y, unbatched = _linear_inputs(operator, measurements)
    energy = np.sum(np.abs(a) ** 2, axis=0)
    if np.any(energy <= 0):
        raise ValueError("Matched filtering requires nonzero operator columns")
    estimate = np.clip(np.real(y @ a.conj()) / energy, 0.0, 1.0)
    return estimate[0] if unbatched else estimate


def _spectral_estimate(gram: np.ndarray, iterations: int = 48) -> float:
    # A random deterministic start avoids the all-ones vector being orthogonal
    # to the leading eigenspace. FISTA also checks its quadratic upper bound.
    vector = np.random.default_rng(0).normal(size=gram.shape[0]).astype(gram.dtype)
    vector /= np.linalg.norm(vector)
    for _ in range(iterations):
        product = gram @ vector
        norm = float(np.linalg.norm(product))
        if norm <= np.finfo(gram.dtype).tiny:
            return 0.0
        vector = product / norm
    return max(0.0, float(vector @ (gram @ vector)))


def fista_box_inverse(
    operator: np.ndarray,
    measurements: np.ndarray,
    *,
    iterations: int = 100,
    ridge: float = 1e-3,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Minimize 0.5||Ax-y||² + 0.5*ridge*||x||² subject to 0 <= x <= 1.

    Real normal equations preserve coherent complex measurements. A deterministic
    spectral estimate initializes the step; quadratic backtracking verifies the
    step for each batch. Stopping uses a fixed iteration budget, never labels.
    """
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    a, y, unbatched = _linear_inputs(operator, measurements)
    gram = a.real.T @ a.real + a.imag.T @ a.imag
    gram.flat[:: gram.shape[0] + 1] += ridge
    rhs = y.real @ a.real + y.imag @ a.imag
    if initial is None:
        x = np.zeros_like(rhs)
    else:
        x = np.asarray(initial, dtype=gram.dtype)
        if unbatched and x.ndim == 1:
            x = x[None]
        if x.shape != rhs.shape or not np.all(np.isfinite(x)) or np.any((x < 0) | (x > 1)):
            raise ValueError("initial must match the estimate shape and lie in [0, 1]")
        x = x.copy()
    z = x.copy()
    momentum = 1.0
    lipschitz = _spectral_estimate(gram) * 1.05
    if lipschitz <= 0:
        if np.any(rhs != 0):
            raise ValueError("Degenerate normal operator with nonzero right hand side")
        return x[0] if unbatched else x
    for _ in range(iterations):
        z_gram = z @ gram
        gradient = z_gram - rhs
        f_z = float(np.sum(z * (0.5 * z_gram - rhs), dtype=np.float64))
        for _attempt in range(32):
            candidate = np.clip(z - gradient / lipschitz, 0.0, 1.0)
            delta = candidate - z
            f_candidate = float(np.sum(candidate * (0.5 * (candidate @ gram) - rhs), dtype=np.float64))
            bound = f_z + float(np.sum(delta * gradient, dtype=np.float64))
            bound += 0.5 * lipschitz * float(np.sum(delta * delta, dtype=np.float64))
            tolerance = 2e-6 * (1.0 + abs(f_candidate) + abs(bound))
            if f_candidate <= bound + tolerance:
                break
            lipschitz *= 2.0
        else:
            raise FloatingPointError("FISTA quadratic backtracking did not converge")
        next_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
        z = candidate + ((momentum - 1.0) / next_momentum) * (candidate - x)
        x = candidate
        momentum = next_momentum
    if not np.all(np.isfinite(x)):
        raise FloatingPointError("Non-finite inverse estimate")
    return x[0] if unbatched else x


def fixed_height_25d(
    operator: np.ndarray,
    measurements: np.ndarray,
    grid_shape: Sequence[int],
    vertical_template: np.ndarray,
    *,
    iterations: int = 100,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Solve XY coefficients with a shared, frozen training-derived Z profile.

    The returned field is [batch, x, y, z] (or [x, y, z] for unbatched input).
    A fixed extrusion is a binary template; a training-mean height prior can be
    a soft template. The profile cannot be estimated from each test scene.
    """
    if len(grid_shape) != 3 or any(int(v) != v or v < 1 for v in grid_shape):
        raise ValueError("grid_shape must have three positive dimensions")
    shape = tuple(int(v) for v in grid_shape)
    a, y, unbatched = _linear_inputs(operator, measurements)
    profile = np.asarray(vertical_template)
    if a.shape[1] != math.prod(shape) or profile.shape != (shape[2],):
        raise ValueError("Grid or vertical template does not match the operator")
    if not np.all(np.isfinite(profile)) or np.any((profile < 0) | (profile > 1)) or not np.any(profile > 0):
        raise ValueError("vertical_template must be nonzero, finite, and lie in [0, 1]")
    reduced = np.einsum("mxyz,z->mxy", a.reshape(a.shape[0], *shape), profile).reshape(a.shape[0], -1)
    xy = fista_box_inverse(reduced, y, iterations=iterations, ridge=ridge)
    field = xy.reshape(y.shape[0], shape[0], shape[1], 1) * profile
    return field[0] if unbatched else field


def complex_nmse(prediction: np.ndarray, reference: np.ndarray, *, axis: int = -1) -> np.ndarray | float:
    """Linear-scale squared complex error / reference energy, per scene.

    An all-zero reference is rejected because normalized error is undefined.
    Evaluate against clean held-out signals and noisy measurements separately.
    """
    p, r = np.asarray(prediction), np.asarray(reference)
    if p.shape != r.shape or p.size == 0 or not np.all(np.isfinite(p)) or not np.all(np.isfinite(r)):
        raise ValueError("Prediction and reference must be finite, nonempty, and shape matched")
    energy = np.sum(np.abs(r) ** 2, axis=axis)
    if np.any(energy <= 0):
        raise ValueError("NMSE is undefined for an all-zero reference")
    result = np.sum(np.abs(p - r) ** 2, axis=axis) / energy
    return float(result) if np.ndim(result) == 0 else result


def _surface(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = mask.copy()
    for axis in range(3):
        for offset in (0, 2):
            slices = [slice(1, -1)] * 3
            slices[axis] = slice(offset, offset + mask.shape[axis])
            interior &= padded[tuple(slices)]
    return mask & ~interior


def _fully_known_neighborhood(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=True)
    known = mask.copy()
    for axis in range(3):
        for offset in (0, 2):
            slices = [slice(1, -1)] * 3
            slices[axis] = slice(offset, offset + mask.shape[axis])
            known &= padded[tuple(slices)]
    return known


def _nearest_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if len(query) == 0 or len(reference) == 0:
        raise ValueError("Nearest-distance sets must be nonempty")
    reference_squared = np.sum(reference * reference, axis=1)
    output = []
    for start in range(0, len(query), 256):
        part = query[start : start + 256]
        squared = np.sum(part * part, axis=1)[:, None] + reference_squared[None]
        squared -= 2.0 * (part @ reference.T)
        output.append(np.sqrt(np.maximum(np.min(squared, axis=1), 0.0)))
    return np.concatenate(output)


def occupancy_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    threshold: float = 0.5,
    valid_mask: np.ndarray | None = None,
    calibration_bins: int = 10,
    surface_tolerance_m: float | None = None,
) -> dict[str, Any]:
    """Metrics for one (x,y,z) occupancy volume, in world metre units.

    Height is the top occupied voxel edge in each XY column. The primary height
    RMSE uses the union of predicted/true occupied, fully-known columns, so false
    positives and missed columns are included. It is not object matching accuracy.
    Surface Chamfer is half the sum of mean bidirectional Euclidean distances.
    Empty predictions are counted explicitly and do not acquire a finite Chamfer.
    """
    p = np.asarray(probability)
    t = np.asarray(target)
    spacing = np.asarray(spacing_xyz, dtype=np.float64)
    if p.ndim != 3 or t.shape != p.shape or p.size == 0 or np.iscomplexobj(p):
        raise ValueError("probability and target must share a nonempty 3D shape")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probability must be finite and lie in [0, 1]")
    if not np.all(np.isfinite(t)) or not np.all((t == 0) | (t == 1)):
        raise ValueError("target must contain only binary finite labels")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing_xyz must contain three positive finite values")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1 or int(calibration_bins) != calibration_bins or calibration_bins < 1:
        raise ValueError("Invalid probability threshold or calibration bin count")
    valid = np.ones(p.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != p.shape or not np.any(valid):
        raise ValueError("valid_mask must match the grid and include at least one voxel")
    truth = t.astype(bool)
    predicted = p >= threshold
    tp = int(np.count_nonzero(predicted & truth & valid))
    fp = int(np.count_nonzero(predicted & ~truth & valid))
    fn = int(np.count_nonzero(~predicted & truth & valid))
    union = tp + fp + fn
    values, labels = p[valid].astype(np.float64), t[valid].astype(np.float64)
    safe = np.clip(values, 1e-7, 1 - 1e-7)
    indices = np.minimum((values * calibration_bins).astype(np.int64), calibration_bins - 1)
    ece = 0.0
    for index in range(calibration_bins):
        selected = indices == index
        if np.any(selected):
            ece += np.mean(selected) * abs(float(np.mean(values[selected]) - np.mean(labels[selected])))
    z_edges = (np.arange(p.shape[2], dtype=np.float64) + 1.0) * spacing[2]
    pred_height = np.max(predicted * z_edges, axis=2)
    true_height = np.max(truth * z_edges, axis=2)
    full_columns = np.all(valid, axis=2)
    height_columns = ((pred_height > 0) | (true_height > 0)) & full_columns
    true_columns = (true_height > 0) & full_columns
    height_error = pred_height[height_columns] - true_height[height_columns]
    true_height_error = pred_height[true_columns] - true_height[true_columns]
    result: dict[str, Any] = {
        "iou_3d": tp / union if union else 1.0,
        "precision": tp / (tp + fp) if tp + fp else float(tp + fn == 0),
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "brier": float(np.mean((values - labels) ** 2)),
        "binary_nll": float(np.mean(-labels * np.log(safe) - (1 - labels) * np.log1p(-safe))),
        "ece": float(ece),
        "height_column_rmse_m": float(np.sqrt(np.mean(height_error ** 2))) if len(height_error) else 0.0,
        "height_column_mae_m": float(np.mean(np.abs(height_error))) if len(height_error) else 0.0,
        "height_true_column_rmse_m": float(np.sqrt(np.mean(true_height_error ** 2))) if len(true_height_error) else None,
        "height_evaluated_columns": int(np.count_nonzero(height_columns)),
        "missed_columns": int(np.count_nonzero((true_height > 0) & (pred_height == 0) & full_columns)),
        "false_positive_columns": int(np.count_nonzero((true_height == 0) & (pred_height > 0) & full_columns)),
        "valid_voxels": int(np.count_nonzero(valid)),
        "predicted_occupied_voxels": tp + fp,
        "true_occupied_voxels": tp + fn,
    }
    known = _fully_known_neighborhood(valid)
    pred_points = (np.argwhere(_surface(predicted) & known) + 0.5) * spacing
    true_points = (np.argwhere(_surface(truth) & known) + 0.5) * spacing
    tolerance = float(np.min(spacing)) if surface_tolerance_m is None else float(surface_tolerance_m)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("surface_tolerance_m must be positive and finite")
    result["surface_tolerance_m"] = tolerance
    result["empty_predicted_surface"] = int(len(pred_points) == 0)
    if len(pred_points) and len(true_points):
        pred_to_true = _nearest_distances(pred_points, true_points)
        true_to_pred = _nearest_distances(true_points, pred_points)
        precision = float(np.mean(pred_to_true <= tolerance))
        recall = float(np.mean(true_to_pred <= tolerance))
        result["surface_chamfer_m"] = float(0.5 * (np.mean(pred_to_true) + np.mean(true_to_pred)))
        result["surface_f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    elif not len(pred_points) and not len(true_points):
        result["surface_chamfer_m"] = 0.0
        result["surface_f1"] = 1.0
    else:
        result["surface_chamfer_m"] = None
        result["surface_f1"] = 0.0
    return result


def select_occupancy_threshold(
    validation_probability: np.ndarray,
    validation_target: np.ndarray,
    *,
    candidates: Sequence[float] = tuple(np.linspace(0.1, 0.9, 17)),
) -> dict[str, float]:
    """Select one threshold by macro scene IoU; break ties closest to 0.5."""
    p, t = np.asarray(validation_probability), np.asarray(validation_target)
    if p.shape != t.shape or p.ndim < 2 or len(p) == 0:
        raise ValueError("Validation volumes require a batch dimension and matched shapes")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)) or not np.all((t == 0) | (t == 1)):
        raise ValueError("Invalid validation probabilities or labels")
    thresholds = np.asarray(candidates, dtype=np.float64)
    if thresholds.ndim != 1 or len(thresholds) == 0 or not np.all(np.isfinite(thresholds)) or np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("Threshold candidates must lie in [0, 1]")
    truth = t.reshape(len(t), -1).astype(bool)
    best: tuple[float, float, float] | None = None
    for threshold in thresholds:
        prediction = p.reshape(len(p), -1) >= threshold
        intersection = np.count_nonzero(prediction & truth, axis=1)
        union = np.count_nonzero(prediction | truth, axis=1)
        score = float(np.mean(np.divide(intersection, union, out=np.ones_like(union, dtype=float), where=union > 0)))
        ranked = (score, -abs(float(threshold) - 0.5), -float(threshold))
        if best is None or ranked > best:
            best = ranked
    assert best is not None
    return {"threshold": -best[2], "validation_macro_iou_3d": best[0]}


def select_fista_configuration(
    operator: np.ndarray,
    validation_measurements: np.ndarray,
    validation_target: np.ndarray,
    query_operator: np.ndarray,
    validation_query_reference: np.ndarray,
    *,
    iteration_candidates: Sequence[int] = (60, 180, 400),
    ridge_candidates: Sequence[float] = (0.0, 1e-3, 1e-2),
    near_best_iou_tolerance: float = 0.005,
) -> dict[str, Any]:
    """Select inverse settings using geometry and a common validation RF query.

    Predeclared rule: retain settings within 0.005 absolute macro IoU of the
    best validation geometry, then minimize validation query RF NMSE. Exact
    RF ties prefer fewer iterations then smaller ridge. Every tried result is
    retained. The query reference is validation supervision, never model input.
    """
    a, y, unbatched = _linear_inputs(operator, validation_measurements)
    if unbatched:
        raise ValueError("Configuration selection requires a validation batch")
    target = np.asarray(validation_target)
    query = np.asarray(query_operator)
    reference = np.asarray(validation_query_reference)
    if target.shape[0] != y.shape[0] or target.size != y.shape[0] * a.shape[1]:
        raise ValueError("Validation target does not match scenes or voxels")
    if query.ndim != 2 or query.shape[1] != a.shape[1] or reference.shape != (len(y), query.shape[0]):
        raise ValueError("Held-out query operator and reference do not match")
    if not math.isfinite(near_best_iou_tolerance) or near_best_iou_tolerance < 0:
        raise ValueError("IoU tolerance must be finite and nonnegative")
    if not iteration_candidates or not ridge_candidates:
        raise ValueError("At least one iteration and ridge candidate are required")
    rows = []
    for iterations in iteration_candidates:
        if int(iterations) != iterations or iterations < 1:
            raise ValueError("Iteration candidates must be positive integers")
        for ridge in ridge_candidates:
            estimate = fista_box_inverse(a, y, iterations=int(iterations), ridge=float(ridge))
            selected_threshold = select_occupancy_threshold(estimate, target.reshape(len(y), -1))
            query_prediction = estimate @ query.T
            rows.append({
                "iterations": int(iterations),
                "ridge": float(ridge),
                **selected_threshold,
                "validation_query_nmse": float(np.mean(complex_nmse(query_prediction, reference))),
            })
    best_iou = max(row["validation_macro_iou_3d"] for row in rows)
    eligible = [row for row in rows if row["validation_macro_iou_3d"] >= best_iou - near_best_iou_tolerance]
    selected = min(eligible, key=lambda row: (row["validation_query_nmse"], row["iterations"], row["ridge"]))
    return {
        "selected": dict(selected),
        "candidates": rows,
        "selection_rule": "Within tolerance of best validation macro IoU, then lowest common-query RF NMSE",
        "near_best_iou_tolerance": float(near_best_iou_tolerance),
        "selection_split": "validation",
    }


def select_fixed_height_configuration(
    operator: np.ndarray,
    validation_measurements: np.ndarray,
    validation_target: np.ndarray,
    query_operator: np.ndarray,
    validation_query_reference: np.ndarray,
    grid_shape: Sequence[int],
    profiles: Mapping[str, np.ndarray],
    *,
    iteration_candidates: Sequence[int] = (60, 180, 400),
    ridge_candidates: Sequence[float] = (0.0, 1e-3, 1e-2),
    near_best_iou_tolerance: float = 0.005,
) -> dict[str, Any]:
    """Validate fixed Z profiles and inverse settings after full 3D expansion.

    Profiles must be predefined or computed from training scenes only, e.g.
    ``{"all_ones": ones_z, "training_mean_z": mean_z / mean_z.max()}``.
    The objective ordering matches ``select_fista_configuration`` exactly.
    """
    a, y, unbatched = _linear_inputs(operator, validation_measurements)
    if unbatched:
        raise ValueError("Configuration selection requires a validation batch")
    if len(grid_shape) != 3 or any(int(v) != v or v < 1 for v in grid_shape):
        raise ValueError("grid_shape must have three positive integer dimensions")
    shape = tuple(int(v) for v in grid_shape)
    target = np.asarray(validation_target)
    query = np.asarray(query_operator)
    reference = np.asarray(validation_query_reference)
    if a.shape[1] != math.prod(shape) or target.size != len(y) * a.shape[1] or target.shape[0] != len(y):
        raise ValueError("Validation target or grid does not match the operator")
    if query.ndim != 2 or query.shape[1] != a.shape[1] or reference.shape != (len(y), query.shape[0]):
        raise ValueError("Held-out query operator and reference do not match")
    if not profiles or not iteration_candidates or not ridge_candidates:
        raise ValueError("Profiles and inverse configuration candidates must be nonempty")
    if not math.isfinite(near_best_iou_tolerance) or near_best_iou_tolerance < 0:
        raise ValueError("IoU tolerance must be finite and nonnegative")
    rows = []
    for profile_name, profile in profiles.items():
        if not isinstance(profile_name, str) or not profile_name:
            raise ValueError("Each profile must have a nonempty string name")
        for iterations in iteration_candidates:
            if int(iterations) != iterations or iterations < 1:
                raise ValueError("Iteration candidates must be positive integers")
            for ridge in ridge_candidates:
                estimate = fixed_height_25d(
                    a, y, shape, profile, iterations=int(iterations), ridge=float(ridge)
                ).reshape(len(y), -1)
                threshold = select_occupancy_threshold(estimate, target.reshape(len(y), -1))
                query_prediction = estimate @ query.T
                rows.append({
                    "profile_name": profile_name,
                    "iterations": int(iterations),
                    "ridge": float(ridge),
                    **threshold,
                    "validation_query_nmse": float(np.mean(complex_nmse(query_prediction, reference))),
                })
    best_iou = max(row["validation_macro_iou_3d"] for row in rows)
    eligible = [row for row in rows if row["validation_macro_iou_3d"] >= best_iou - near_best_iou_tolerance]
    selected = min(eligible, key=lambda row: (
        row["validation_query_nmse"], row["iterations"], row["ridge"], row["profile_name"]
    ))
    return {
        "selected": dict(selected),
        "candidates": rows,
        "selection_rule": "Within tolerance of best validation macro IoU, then lowest common-query RF NMSE",
        "near_best_iou_tolerance": float(near_best_iou_tolerance),
        "selection_split": "validation",
        "profile_source_requirement": "Predefined or derived from training scenes only",
    }


def paired_family_bootstrap(
    values_a: Sequence[float],
    values_b: Sequence[float],
    family_ids: Sequence[Any],
    *,
    confidence: float = 0.95,
    repetitions: int = 2000,
    seed: int = 20261123,
) -> dict[str, Any]:
    """Paired percentile interval, resampling independent scene families.

    Mirror variants/views of a scene stay together. Each family has equal weight;
    voxels and frequencies are never counted as independent experimental units.
    This interval describes scene sampling, not variation across training seeds.
    """
    a, b = np.asarray(values_a, dtype=np.float64), np.asarray(values_b, dtype=np.float64)
    families = np.asarray(family_ids)
    if a.ndim != 1 or a.shape != b.shape or a.shape != families.shape or len(a) < 1:
        raise ValueError("Paired metric arrays and family IDs must be nonempty and aligned")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Bootstrap inputs cannot silently drop failed or non-finite scenes")
    if not 0 < confidence < 1 or repetitions < 1:
        raise ValueError("Invalid confidence or bootstrap repetition count")
    distinct = np.unique(families)
    delta = a - b
    grouped = np.asarray([np.mean(delta[families == family]) for family in distinct])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(grouped), size=(repetitions, len(grouped)))
    means = np.mean(grouped[draws], axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return {
        "mean_difference_a_minus_b": float(np.mean(grouped)),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "repetitions": repetitions,
        "seed": seed,
        "independent_families": len(grouped),
        "scenes": len(a),
        "resampling_unit": "scene_family",
    }
