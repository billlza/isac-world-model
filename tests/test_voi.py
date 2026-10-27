"""Causal delay-aware covariance VoI: numerical, resource and execution tests."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from isac_wm.evaluation import evaluate, rollout
from isac_wm.runtime import RuntimeConfig
from isac_wm.simulator import ACTIONS, SceneConfig, ToyScene
from isac_wm.voi import VoIConfig, choose_sensing_by_voi

X = np.array([3.0, 4.0, 0.2, -0.1])


def decision(p=0.5, tokens=0.6, cfg=None, *, scene=None, wait=0.0):
    cov = np.diag([p, p, 0.2, 0.2])
    return choose_sensing_by_voi(
        X,
        cov,
        scene or SceneConfig(steps=12),
        tokens,
        planned_actuation_wait_s=wait,
        configuration=cfg,
    )


def test_low_uncertainty_skips_high_uncertainty_senses():
    assert decision(0.00001).action == 0
    result = decision(2.0)
    assert result.action in (1, 2)
    assert result.gains_m2[2] > 0
    assert result.lost_mbit[2] > result.lost_mbit[1] > 0


def test_delayed_measurement_has_less_value_at_common_horizon():
    base = decision(0.5)
    late = decision(0.5, scene=SceneConfig(steps=12, radio_delay_extra_s=0.3))
    assert late.gains_m2[1] < base.gains_m2[1]
    assert late.gains_m2[2] < base.gains_m2[2]
    assert late.delivery_ticks[1] > base.delivery_ticks[1]
    assert decision(1, wait=2.0).action == 0
    assert decision(1, scene=SceneConfig(steps=12, radio_delay_extra_s=2.0)).action == 0


def test_price_and_budget_restrict_actions_without_nonfinite_json():
    none = decis¶»§q«^