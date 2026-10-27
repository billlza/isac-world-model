"""Independent-episode split conformal diagnostic for position uncertainty.

One score per independent episode: max normalized position error after startup.
Coverage is marginal over exchangeable *episodes* with an unchanged estimator,
policy and simulator distribution. It is NOT a guarantee of conditional accepted
risk, OOD robustness, or physical safety. Frames are not assumed independent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .artifacts import model_fingerprint, write_json
from .model import EventBeliefModel
from .runtime import RuntimeConfig
from .simulator import SceneConfig
from .validation import integer
from .voi import VoIConfig


def fit_episode_quantile(runs: list[dict], alpha: float = 0.1, scored_from: int = 5) -> dict:
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    integer(scored_from, "scored_from", 0)
    ids = [r["seed"] for r in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("Independent calibration episode IDs must be unique")
    n = len(runs)
    rank = math.ceil((n + 1) * (1 - alpha))
    if not n or rank > n:
        raise ValueError("Too few independent episodes for a finite conformal quantile")
    scores = []
    for run in runs:
        trace = run["trace"][scored_from:]
        if not trace:
            raise ValueError("No scored calibration frames")
        values = []
        for row in trace:
            variance = np.asarray(row["position_variance"], dtype=float)
            difference = np.asarray(row["estimate"][:2]) - np.asarray(row["truth"][:2])
            if variance.shape != (2,) or (variance <= 0).any() or not np.isfinite(variance).all():
                raise ValueError("Calibration needs strictly positive finite variances")
            value = float(np.sqrt(np.sum(difference**2 / variance)))
            if not math.isfinite(value):
                raise ValueError("Non-finite calibration score")
            values.append(value)
        scores.append(max(values))
    q = float(np.partition(scores, rank - 1)[rank - 1])
    return {
        "quantile": max(q, 1e-12),
        "alpha": alpha,
        "n_episodes": n,
        "rank": rank,
        "episode_seeds": ids,
        "episode_scores": scores,
        "score": "episode maximum diagonal-Mahalanobis position error",
        "scored_from_step": scored_from,
    }


def calibrate(
    out: Path,
    model: EventBeliefModel,
    scene: SceneConfig,
    device: torch.device,
    episodes: int = 32,
    seed: int = 20_000_000,
    alpha: float = 0.1,
    policies: tuple[str, ...] = ("fixed", "guarded"),
    *,
    runtime: RuntimeConfig | None = None,
    voi_config: VoIConfig | None = None,
) -> dict:
    from .evaluation import rollout

    runtime = runtime or RuntimeConfig()
    if runtime.mode == "measured":
        raise ValueError("Calibration needs a repeatable fixed timing profile, not measured timing")
    if (
        not policies
        or len(policies) != len(set(policies))
        or set(policies) - {"fixed", "skip", "predictive", "risk", "guarded", "voi"}
    ):
        raise ValueError("Invalid calibration policies")
    integer(episodes, "episodes", 1, 10000)
    integer(seed, "seed", 0, 2**32 - 1 - episodes)
    if out.exists():
        raise FileExistsError("Calibration output exists; use a new file")
    seeds = set(range(seed, seed + episodes))
    if seeds.intersection(getattr(model, "provenance_seeds", set())):
        raise ValueError("Calibration overlaps training/validation episodes")
    fingerprint = model_fingerprint(model)
    variants = {}
    for estimator in ("kalman", "model"):
        for policy in policies:
            runs = [
                rollout(i, scene, estimator, policy, model, device, runtime=runtime, voi_config=voi_config) for i in sorted(seeds)
            ]
            variants[f"{estimator}+{policy}"] = fit_episode_quantile(runs, alpha)
    artifact = {
        "schema_version": 2,
        "runtime": runtime.to_dict(),
        "voi_configuration": (voi_config or VoIConfig()).__dict__,
        "model_fingerprint": fingerprint,
        "scene": scene.to_dict(),
        "variants": variants,
        "scope": "episode-marginal, same-policy, same-distribution; not conditional accepted-risk assurance",
    }
    write_json(out, artifact)
    return artifact


def load_calibration(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Calibration must be a regular non-symlink file")
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Calibration artifact too large")
    data = json.loads(path.read_text())
    if (
        not isinstance(data, dict)
        or data.get("schema_version") not in {1, 2}
        or not isinstance(data.get("variants"), dict)
    ):
        raise ValueError("Invalid calibration artifact")
    if data.get("schema_version") == 2:
        RuntimeConfig.from_dict(data.get("runtime"))
    fingerprint = data.get("model_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(c not in "0123456789abcdef" for c in fingerprint)
    ):
        raise ValueError("Invalid model fingerprint")
    if (
        not isinstance(data.get("scene"), dict)
        or not isinstance(data.get("scope"), str)
        or not data["variants"]
    ):
        raise ValueError("Missing calibration metadata")
    SceneConfig(**data["scene"])
    for key, value in data["variants"].items():
        if key not in {
            f"{e}+{p}" for e in ("kalman", "model") for p in ("fixed", "skip", "predictive", "risk", "guarded", "voi")
        } or not isinstance(value, dict):
            raise ValueError("Invalid calibration variant")
        if value.get("scored_from_step") != 5:
            raise ValueError("Unsupported scoring window")
        q = value.get("quantile")
        if isinstance(q, bool) or not isinstance(q, (float, int)) or not math.isfinite(q) or q <= 0:
            raise ValueError("Invalid calibration quantile")
        seeds = value.get("episode_seeds")
        if (
            not isinstance(seeds, list)
            or not all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x < 2**32 for x in seeds)
            or len(seeds) != len(set(seeds))
            or len(seeds) != value.get("n_episodes")
        ):
            raise ValueError("Invalid calibration provenance")
    return data


def validate_calibration(
    data: dict,
    model: EventBeliefModel,
    scene: SceneConfig,
    test_seeds: list[int],
    *,
    runtime: RuntimeConfig | None = None,
    voi_config: VoIConfig | None = None,
) -> None:
    runtime = runtime or RuntimeConfig()
    expected_runtime = data.get("runtime", RuntimeConfig().to_dict())
    if expected_runtime != runtime.to_dict():
        raise ValueError("Calibration runtime mismatch; recalibrate for the new execution timing")
    if runtime.mode == "measured":
        raise ValueError("Measured timing cannot use fixed-profile calibration")
    if data.get("voi_configuration", VoIConfig().__dict__) != (voi_config or VoIConfig()).__dict__:
        raise ValueError("Calibration VoI configuration mismatch")
    if data["scene"] != scene.to_dict() or data["model_fingerprint"] != model_fingerprint(model):
        raise ValueError("Calibration model/scene mismatch; recalibrate instead of reusing a certificate")
    for value in data["variants"].values():
        if set(value["episode_seeds"]).intersection(test_seeds):
            raise ValueError("Test episodes overlap calibration episodes")
