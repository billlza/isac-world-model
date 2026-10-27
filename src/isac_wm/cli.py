from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .artifacts import write_json
from .calibration import calibrate, load_calibration
from .evaluation import evaluate
from .imm import IMMConfig
from .model import ModelConfig
from .runtime import RuntimeConfig
from .safety import MonitorConfig
from .scenarios import SCENARIOS, scenario_config
from .simulator import SceneConfig
from .structured import StructuredConfig
from .toolchain import inspect_toolchain
from .training import load_model, resolve_device, runtime_info, train
from .viewer import write_viewer
from .voi import VoIConfig


def _policies(text: str) -> tuple[str, ...]:
    values = tuple(x.strip() for x in text.split(","))
    if (
        not values
        or len(values) != len(set(values))
        or set(values) - {"fixed", "skip", "predictive", "risk", "guarded", "voi"}
    ):
        raise argparse.ArgumentTypeError("Use distinct policies from fixed,skip,predictive,risk,guarded,voi")
    return values


def _estimators(text: str) -> tuple[str, ...]:
    values = tuple(x.strip() for x in text.split(","))
    allowed = {"kalman", "kalman-gated", "imm", "imm-gated", "model"}
    if not values or len(values) != len(set(values)) or set(values) - allowed:
        raise argparse.ArgumentTypeError(
            "Use distinct estimators from kalman,kalman-gated,imm,imm-gated,model"
        )
    return values


def _estimators(text: str) -> tuple[str, ...]:
    va¶»§q«^