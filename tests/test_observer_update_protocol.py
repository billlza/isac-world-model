"""The training/validation recurrence must match released-evidence inference."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from isac_wm.active_observation import (
    Anchor,
    ExperimentConfig,
    HypothesisMemory,
    ObservationNetwork,
    observation_geometry,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "observer_update_study", ROOT / "scripts/active_observation_study.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class RecordingObserver(ObservationNetwork):
    """Deterministic predictions, the real GRU update, and observable input state."""

    def __init__(self):
        super().__init__(use_memory=True)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(0.04)
        self.input_states = []
        self.predictions = []

    def predict(self, features, base_mean, base_sigma, memory):
        self.input_states.append(memory.detach().clone())
        prediction = base_mean + 0.1 * memory[..., 0]
        self.predictions.append(prediction.detach().clone())
        return prediction, torch.ones_like(base_sigma)


def evidence_sequence(cfg):
    anchor = Anchor(18, 12, 0, 0.1)
    captures = (1.0, 2.0, 3.0)
    geometry = [observation_geometry(anchor, 4.0, capture, cfg) for capture in captures]
    means, sigmas, features = map(np.stack, zip(*geometry, strict=True))
    # Row 0 is compatible/incompatible/compatible at the default threshold.
    # Row 1 stays compatible, checking that a rejected row cannot freeze a batch.
    observations = [[means[0, 0], means[1, 1] + 4.0, means[2, 0]], means[:, 1]]
    data = {
        name: torch.tensor(np.stack([values, values]), dtype=torch.float32)
        for name, values in (("features", features), ("means", means), ("sigmas", sigmas))
    }
    data["observations"] = torch.tensor(np.array(observations), dtype=torch.float32)
    data["labels"] = torch.tensor([0, 1], dtype=torch.long)
    return anchor, captures, data


@pytest.mark.parametrize("rejection_sigma", [3.0, 5.0])
def test_sequence_loss_uses_the_same_hidden_states_as_released_evidence(rejection_sigma):
    cfg = ExperimentConfig(rejection_sigma=rejection_sigma)
    anchor, captures, data = evidence_sequence(cfg)
    trained = RecordingObserver()
    loss = study.sequence_loss(trained, data, rejection_sigma=cfg.rejection_sigma)
    for row in range(2):
        online = HypothesisMemory(anchor, cfg, RecordingObserver())
        released_states = []
        for step, capture in enumerate(captures):
            torch.testing.assert_close(trained.input_states[step][row], online.hidden)
            mean, sigma = online.predict(4.0, capture)
            np.testing.assert_allclose(trained.predictions[step][row].numpy(), mean, rtol=1e-5, atol=1e-6)
            event = {
                "uid": step,
                "lateral_m": 4.0,
                "capture_s": capture,
                "available_s": capture + cfg.delivery_s,
                "prediction_mean_deg": mean.tolist(),
                "prediction_sigma_deg": sigma.tolist(),
                "observation_deg": float(data["observations"][row, step]),
            }
            online.release(event, event["available_s"])
            released_states.append(online.hidden.clone())
        assert not torch.equal(released_states[0], torch.zeros_like(online.hidden))
        if row == 0 and rejection_sigma == 3.0:
            torch.testing.assert_close(released_states[1], released_states[0])
        else:
            assert not torch.equal(released_states[1], released_states[0])
        assert not torch.equal(released_states[2], released_states[1])

    # The incompatible observation still contributes its full current-step NLL.
    predictions = torch.stack(trained.predictions, dim=1)
    supervised = predictions.gather(2, data["labels"][:, None, None].expand(-1, 3, 1)).squeeze(-1)
    nll = 0.5 * (data["observations"] - supervised).square() + math.log(math.sqrt(2 * math.pi))
    assert nll[0, 1] > 20
    torch.testing.assert_close(loss.detach(), nll.mean())


def test_supervision_labels_do_not_select_the_recurrent_update_mask():
    _, _, data = evidence_sequence(ExperimentConfig())
    original, relabelled = RecordingObserver(), RecordingObserver()
    loss = study.sequence_loss(original, data)
    changed_loss = study.sequence_loss(relabelled, {**data, "labels": 1 - data["labels"]})
    for first, second in zip(original.input_states, relabelled.input_states, strict=True):
        torch.testing.assert_close(first, second)
    assert not torch.isclose(loss, changed_loss)


def test_training_and_checkpoint_selection_receive_the_same_threshold(monkeypatch):
    _, _, data = evidence_sequence(ExperimentConfig())
    calls = []

    def recorded_loss(model, batch, *, rejection_sigma):
        calls.append((model.training, rejection_sigma))
        return next(model.parameters()).sum() * 0 + 1

    monkeypatch.setattr(study, "sequence_loss", recorded_loss)
    study.train_observer(data, data, use_memory=True, epochs=1, seed=7, rejection_sigma=5.0)
    assert calls == [(True, 5.0), (False, 5.0)]
