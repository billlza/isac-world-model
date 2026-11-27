"""Strict configuration: a misspelled budget or objective must not be ignored."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

from .model import ModelConfig


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 1000
    batch_size: int = 8
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    gradient_accumulation: int = 1
    clip_grad_norm: float = 1.0
    seed: int = 7
    validate_every: int = 100
    checkpoint_every: int = 100
    cfg_dropout: float = 0.15
    temporal_weight: float = 0.1
    precision: str = "float32"
    ema_decay: float = 0.995
    threads: int = 2

    def __post_init__(self):
        for name in ("steps", "batch_size", "gradient_accumulation", "validate_every", "checkpoint_every", "threads"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"training.{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("training.seed must be a nonnegative integer")
        for name in ("learning_rate", "weight_decay", "clip_grad_norm", "cfg_dropout", "temporal_weight", "ema_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"training.{name} must be finite")
        if self.learning_rate <= 0 or self.clip_grad_norm <= 0:
            raise ValueError("learning_rate and clip_grad_norm must be positive")
        if self.weight_decay < 0 or self.temporal_weight < 0:
            raise ValueError("weight_decay and temporal_weight must be nonnegative")
        if not 0 <= self.cfg_dropout < 1 or not 0 <= self.ema_decay < 1:
            raise ValueError("cfg_dropout and ema_decay must be in [0, 1)")
        if self.precision not in ("float32", "bfloat16"):
            raise ValueError("precision must be float32 or bfloat16")


@dataclass(frozen=True)
class SamplingConfig:
    steps: int = 30
    guidance_scale: float = 3.0
    fps: int = 8

    def __post_init__(self):
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("sampling.steps must be positive")
        if type(self.fps) is not int or self.fps < 1:
            raise ValueError("sampling.fps must be positive")
        if isinstance(self.guidance_scale, bool) or not isinstance(self.guidance_scale, (int, float)) or not math.isfinite(self.guidance_scale) or self.guidance_scale < 0:
            raise ValueError("sampling.guidance_scale must be finite and nonnegative")


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    name: str
    model: ModelConfig
    training: TrainingConfig
    sampling: SamplingConfig

    def to_dict(self):
        return asdict(self)


def _construct(cls, value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {label} keys: {sorted(unknown)}")
    return cls(**value)


def config_from_dict(value):
    expected = {"schema_version", "name", "model", "training", "sampling"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"Config requires exactly {sorted(expected)}")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported config schema_version")
    if not isinstance(value["name"], str) or not value["name"].strip():
        raise ValueError("Config name must be nonempty")
    return ExperimentConfig(
        1, value["name"], _construct(ModelConfig, value["model"], "model"),
        _construct(TrainingConfig, value["training"], "training"),
        _construct(SamplingConfig, value["sampling"], "sampling"),
    )


def load_config(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate configuration key: {key}")
            result[key] = value
        return result

    return config_from_dict(json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique))
