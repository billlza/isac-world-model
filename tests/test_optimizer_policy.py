"""Controlled width learning rates retain the control and valid continuation."""

import copy
import json
from unittest.mock import patch

import pytest
import torch
from torch import nn

from isac_wm.artifacts import (
    model_fingerprint,
    read_checkpoint,
    read_resume_checkpoint,
    save_resume_checkpoint,
)
from isac_wm.model import ModelConfig, training_loss
from isac_wm.optimizer_policy import (
    FIXED_WIDTH_PREFIXES,
    describe_optimizer_policy,
    make_optimizer,
)
from isac_wm.simulator import SceneConfig, recorded_episode
from isac_wm.structured import StructuredConfig, build_model
from isac_wm.training import batchify, train


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_config():
    return StructuredConfig(feature_dim=8, hidden_dim=16, memory_size=4, latent_state_norm="layernorm")


@pytest.fixture
def common():
    return dict(
        episodes=4,
        batch_size=2,
        device_name="cpu",
        seed=7,
        scene=SceneConfig(steps=8, image_size=16),
        model_config=small_config(),
        learning_rate=3e-4,
        train_seed_base=50_000,
        validation_seed_base=60_000,
        batch_order_seed=37,
        validation_episodes=3,
        validation_every_updates=1,
        vision_loss_weight=1.0,
    )


def assert_exact(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for actual, expected in zip(left, right):
            assert_exact(actual, expected)
    elif isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    else:
        assert left == right


def update(model, optimizer, batch, scene):
    optimizer.zero_grad(set_to_none=True)
    output = model(batch, forecast_s=3 * scene.dt, batched_encoding=True, return_visual=True)
    loss, _ = training_loss(output, batch, vision_loss_weight=1.0)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach()


def test_default_matches_literal_historical_adamw_after_real_updates(common):
    torch.manual_seed(7)
    model = build_model(small_config())
    reference = copy.deepcopy(model)
    rng = torch.get_rng_state().clone()
    optimizer, policy = make_optimizer(model, 3e-4)
    literal = torch.optim.AdamW(reference.parameters(), lr=3e-4, weight_decay=1e-4)
    assert_exact(torch.get_rng_state(), rng)
    assert policy["mode"] == "legacy_single_group"
    assert len(optimizer.param_groups) == 1
    assert "name" not in optimizer.param_groups[0]
    assert "param_names" not in optimizer.param_groups[0]
    assert_exact(optimizer.state_dict(), literal.state_dict())
    batch = batchify([recorded_episode(s, common["scene"]) for s in (50_000, 50_001)], torch.device("cpu"))
    for _ in range(3):
        assert_exact(
            update(model, optimizer, batch, common["scene"]),
            update(reference, literal, batch, common["scene"]),
        )
        assert_exact(model.state_dict(), reference.state_dict())
        assert_exact(optimizer.state_dict(), literal.state_dict())


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_policy_creation_preserves_every_initial_parameter_and_rng(scale):
    torch.manual_seed(11)
    model = build_model(small_config())
    fingerprint = model_fingerprint(model)
    visual = model_fingerprint(model.visual)
    rng = torch.get_rng_state().clone()
    optimizer, policy = make_optimizer(model, 3e-4, scale)
    assert model_fingerprint(model) == fingerprint
    assert model_fingerprint(model.visual) == visual
    assert_exact(torch.get_rng_state(), rng)
    assert len(policy["sha256"]) == 64
    live = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(live) == len({id(parameter) for parameter in live}) == len(list(model.parameters()))
    assert {id(parameter) for parameter in live} == {id(parameter) for parameter in model.parameters()}


@pytest.mark.parametrize("width, count", [(512, 3_883_798), (2048, 59_563_798)])
def test_formal_width_partition_counts_and_fixed_encoder_learning_rate(width, count):
    with torch.device("meta"):
        model = build_model(StructuredConfig(hidden_dim=width, latent_state_norm="layernorm"))
    policy = describe_optimizer_policy(model, 3e-4, 0.5)
    fixed, scaled = policy["groups"]
    assert policy["parameters"] == count
    assert fixed["name"] == "fixed_width"
    assert fixed["parameters"] == 3846
    assert set(fixed["prefixes"]) == {"visual", "radio_encoder", "action_embedding"}
    assert fixed["learning_rate"] == 3e-4
    assert scaled["name"] == "width_dependent"
    assert scaled["parameters"] == count - 3846
    assert scaled["learning_rate"] == 1.5e-4
    names = fixed["parameter_names"] + scaled["parameter_names"]
    assert len(names) == len(set(names)) == policy["parameter_tensors"]
    assert set(names) == set(dict(model.named_parameters()))
    for name in names:
        assert (name in fixed["parameter_names"]) == (name.split(".", 1)[0] in FIXED_WIDTH_PREFIXES)


def test_first_update_keeps_fixed_modules_and_halves_only_registered_width_updates(common):
    torch.manual_seed(7)
    model = build_model(small_config())
    reference = copy.deepcopy(model)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer, _ = make_optimizer(model, 3e-4, 0.5)
    literal = torch.optim.AdamW(reference.parameters(), lr=3e-4, weight_decay=1e-4)
    batch = batchify([recorded_episode(s, common["scene"]) for s in (50_000, 50_001)], torch.device("cpu"))
    assert_exact(
        update(model, optimizer, batch, common["scene"]), update(reference, literal, batch, common["scene"])
    )
    changed = {"fixed": 0, "width": 0}
    for (name, actual), (reference_name, expected) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert name == reference_name
        if name.split(".", 1)[0] in FIXED_WIDTH_PREFIXES:
            assert_exact(actual, expected)
            changed["fixed"] += int(not torch.equal(actual, before[name]))
        else:
            # float32 subtraction has its own rounding. Test the actual first
            # AdamW update, including decoupled decay, against half the control.
            torch.testing.assert_close(
                actual - before[name], 0.5 * (expected - before[name]), rtol=3e-4, atol=4e-8
            )
            changed["width"] += int(not torch.equal(actual, before[name]))
    assert all(count > 0 for count in changed.values())


def test_unregistered_or_shared_parameters_fail_closed_and_legacy_model_is_unscaled_only():
    model = build_model(small_config())
    model.new_head = nn.Linear(16, 2)
    with pytest.raises(ValueError, match="prefixes"):
        make_optimizer(model, 3e-4, 0.5)
    model = build_model(small_config())
    model.alias = model.noise.weight
    with pytest.raises(ValueError, match="aliased/shared"):
        make_optimizer(model, 3e-4, 0.5)
    legacy = build_model(ModelConfig(feature_dim=8, hidden_dim=8))
    optimizer, policy = make_optimizer(legacy, 3e-4)
    assert len(optimizer.param_groups) == 1
    assert policy["parameters"] == sum(parameter.numel() for parameter in legacy.parameters())
    with pytest.raises(ValueError, match="StructuredEventModel"):
        make_optimizer(legacy, 3e-4, 0.5)


@pytest.mark.parametrize("scale", [True, 0, -0.5, 1.1, float("nan"), float("inf")])
def test_invalid_scale_rejected_before_data_generation(tmp_path, common, scale):
    with patch("isac_wm.training.recorded_episode") as generator, pytest.raises(ValueError):
        train(tmp_path / "invalid", epochs=1, width_lr_scale=scale, **common)
    generator.assert_not_called()


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_real_train_resume_is_exact_and_all_artifacts_record_policy(tmp_path, common, scale):
    full = train(tmp_path / "full", epochs=2, width_lr_scale=scale, **common)
    train(tmp_path / "part", epochs=1, width_lr_scale=scale, **common)
    resumed = train(
        tmp_path / "resumed", epochs=2, resume=tmp_path / "part/resume.pt", width_lr_scale=scale, **common
    )
    expected, actual = [read_resume_checkpoint(tmp_path / name / "resume.pt") for name in ("full", "resumed")]
    for field in ("state_dict", "optimizer", "torch_rng"):
        assert_exact(actual[field], expected[field])
    for field in ("history", "epoch_history", "best_rmse", "numpy_rng_json", "optimizer_updates"):
        assert actual[field] == expected[field]
    assert full["initial_parameters_sha256"] == resumed["initial_parameters_sha256"]
    for name in ("full", "part", "resumed"):
        manifest = json.loads((tmp_path / name / "manifest.json").read_text())
        best = read_checkpoint(tmp_path / name / "model.pt")
        saved = read_resume_checkpoint(tmp_path / name / "resume.pt")
        for artifact in (manifest, best, saved, saved["best_checkpoint"]):
            assert artifact["optimizer_policy"] == full["optimizer_policy"]
            assert artifact["width_lr_scale"] == scale


def test_same_seed_training_initial_hash_does_not_depend_on_policy(tmp_path, common):
    control = train(tmp_path / "control", epochs=1, width_lr_scale=1.0, **common)
    candidate = train(tmp_path / "candidate", epochs=1, width_lr_scale=0.5, **common)
    assert control["initial_parameters_sha256"] == candidate["initial_parameters_sha256"]
    assert control["initial_visual_parameters_sha256"] == candidate["initial_visual_parameters_sha256"]
    assert control["model"] == candidate["model"]
    assert control["seed_plan"] == candidate["seed_plan"]
    assert control["train_episode_seeds"] == candidate["train_episode_seeds"]
    assert control["event_counts_seen"] == candidate["event_counts_seen"]


@pytest.fixture
def saved_candidate(tmp_path, common):
    train(tmp_path / "source", epochs=1, width_lr_scale=0.5, **common)
    return read_resume_checkpoint(tmp_path / "source/resume.pt")


@pytest.mark.parametrize(
    "fault",
    [
        "policy",
        "scale",
        "group_count",
        "lr",
        "group_name",
        "param_names",
        "indices",
        "moments",
        "best_policy",
    ],
)
def test_resume_rejects_policy_or_actual_optimizer_mismatch_before_step(
    tmp_path, common, saved_candidate, fault
):
    saved = copy.deepcopy(saved_candidate)
    groups = saved["optimizer"]["param_groups"]
    if fault == "policy":
        saved["optimizer_policy"]["groups"][1]["learning_rate"] *= 2
    elif fault == "scale":
        saved["width_lr_scale"] = 1.0
    elif fault == "group_count":
        groups.pop()
    elif fault == "lr":
        groups[1]["lr"] *= 2
    elif fault == "group_name":
        groups[1]["name"] = "unregistered"
    elif fault == "param_names":
        groups[1]["param_names"] = list(reversed(groups[1]["param_names"]))
    elif fault == "indices":
        groups[1]["params"][0] = groups[0]["params"][0]
    elif fault == "moments":
        next(iter(saved["optimizer"]["state"].values()))["exp_avg"] = torch.zeros(1)
    else:
        saved["best_checkpoint"]["width_lr_scale"] = 1.0
    source = tmp_path / "invalid.pt"
    save_resume_checkpoint(source, saved)
    with patch.object(torch.optim.AdamW, "step") as step, pytest.raises(ValueError, match="optimizer"):
        train(tmp_path / "rejected", epochs=2, resume=source, width_lr_scale=0.5, **common)
    step.assert_not_called()


def test_resume_cannot_change_policy_even_when_checkpoint_is_valid(tmp_path, common, saved_candidate):
    with patch.object(torch.optim.AdamW, "step") as step, pytest.raises(ValueError, match="optimizer policy"):
        train(
            tmp_path / "rejected",
            epochs=2,
            resume=tmp_path / "source/resume.pt",
            width_lr_scale=1.0,
            **common,
        )
    step.assert_not_called()


def test_loaded_live_optimizer_is_rechecked_before_any_step(tmp_path, common, saved_candidate):
    original = torch.optim.AdamW.load_state_dict

    def altered_load(optimizer, state):
        original(optimizer, state)
        optimizer.param_groups[1]["lr"] *= 2

    with (
        patch.object(torch.optim.AdamW, "load_state_dict", altered_load),
        patch.object(torch.optim.AdamW, "step") as step,
        pytest.raises(ValueError, match="optimizer learning rate"),
    ):
        train(
            tmp_path / "rejected",
            epochs=2,
            resume=tmp_path / "source/resume.pt",
            width_lr_scale=0.5,
            **common,
        )
    step.assert_not_called()


def test_genuine_legacy_resume_is_accepted_only_unscaled_and_preserves_best(tmp_path, common):
    train(tmp_path / "full", epochs=2, **common)
    train(tmp_path / "part", epochs=1, **common)
    legacy = read_resume_checkpoint(tmp_path / "part/resume.pt")
    for artifact in (legacy, legacy["best_checkpoint"]):
        del artifact["optimizer_policy"]
        del artifact["width_lr_scale"]
    source = tmp_path / "legacy.pt"
    save_resume_checkpoint(source, legacy)
    resumed = train(tmp_path / "resumed", epochs=2, resume=source, **common)
    expected = read_resume_checkpoint(tmp_path / "full/resume.pt")
    actual = read_resume_checkpoint(tmp_path / "resumed/resume.pt")
    assert resumed["optimizer_restored_from_legacy_single_group"]
    assert_exact(actual["state_dict"], expected["state_dict"])
    assert_exact(actual["optimizer"], expected["optimizer"])
    assert actual["history"] == expected["history"]
    with patch("isac_wm.training._validation_position_rmse", return_value=1000.0):
        train(tmp_path / "preserved", epochs=2, resume=source, **common)
    best = read_checkpoint(tmp_path / "preserved/model.pt")
    assert_exact(best["state_dict"], legacy["best_checkpoint"]["state_dict"])
    assert best["optimizer_policy"] == resumed["optimizer_policy"]
    assert best["optimizer_policy_origin"] == "validated_legacy_single_group_resume"
    with pytest.raises(ValueError, match="optimizer policy"):
        train(tmp_path / "scaled", epochs=2, resume=source, width_lr_scale=0.5, **common)
    legacy["optimizer"]["param_groups"][0]["lr"] *= 2
    save_resume_checkpoint(tmp_path / "legacy-bad-lr.pt", legacy)
    with pytest.raises(ValueError, match="optimizer learning rate"):
        train(tmp_path / "invalid", epochs=2, resume=tmp_path / "legacy-bad-lr.pt", **common)
