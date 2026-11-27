"""Real optimizer, provenance, held-out selection, and interruption recovery tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from text_video import engine
from text_video.config import config_from_dict
from text_video.data import make_procedural_dataset
from text_video.io import ARCHITECTURE, load_checkpoint, save_checkpoint, sha256_file
from text_video.objective import flow_objective


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_config():
    return config_from_dict({
        "schema_version": 1,
        "name": "training-contract-test",
        "model": {
            "height": 16, "width": 16, "frames": 4, "patch_size": 4,
            "hidden": 16, "depth": 1, "heads": 4, "text_layers": 1,
            "max_text_bytes": 128, "dropout": 0.1,
        },
        "training": {
            "steps": 12, "batch_size": 2, "learning_rate": 0.005,
            "weight_decay": 0.01, "gradient_accumulation": 1,
            "clip_grad_norm": 1.0, "seed": 17, "validate_every": 2,
            "checkpoint_every": 2, "cfg_dropout": 0.25,
            "temporal_weight": 0.1, "precision": "float32",
            "ema_decay": 0.6, "threads": 1,
        },
        "sampling": {"steps": 2, "guidance_scale": 2.0, "fps": 8},
    })


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return make_procedural_dataset(
        tmp_path / "data", train=7, validation=3, test=2,
        frames=4, height=16, width=16, seed=29,
    )


def _scalar_clip(values: list[float]) -> torch.Tensor:
    return torch.tensor(values).view(1, 1, len(values), 1, 1)


@pytest.mark.parametrize("time", [0.0, 0.3, 1.0])
def test_flow_target_has_correct_direction_and_preserves_true_motion(time: float) -> None:
    clean = _scalar_clip([0.1, 0.9, -0.4])
    noise = _scalar_clip([-1.2, 0.2, 0.8])
    target = clean - noise
    weights = torch.ones(1, 3)
    loss, metrics = flow_objective(target, clean, noise, torch.tensor([time]), weights)
    assert loss.item() == 0
    assert metrics["flow_loss"] == 0 and metrics["temporal_loss"] == 0
    reversed_loss, _ = flow_objective(-target, clean, noise, torch.tensor([time]), weights)
    assert reversed_loss.item() > 0


def test_weighted_objective_matches_analytic_reference_and_temporal_support() -> None:
    # Independent scalar case: errors [1,3,2], trust [1,1/2,1/4], t=1/4.
    # Temporal reconstruction errors are 3/4 times the velocity errors.
    prediction = _scalar_clip([1.0, 3.0, 2.0])
    clean = torch.zeros_like(prediction)
    expected_flow = (1 + 9 / 2 + 4 / 4) / (1 + 1 / 2 + 1 / 4)
    expected_temporal = ((1.5**2) / 2 + (0.75**2) / 4) / (1 / 2 + 1 / 4)
    loss, metrics = flow_objective(
        prediction, clean, clean, torch.tensor([0.25]), torch.tensor([[1.0, 0.5, 0.25]]),
        temporal_weight=0.2,
    )
    assert metrics["flow_loss"] == pytest.approx(expected_flow)
    assert metrics["temporal_loss"] == pytest.approx(expected_temporal)
    assert metrics["reference_support"] == pytest.approx(1.75)
    assert metrics["temporal_support"] == pytest.approx(0.75)
    assert loss.item() == pytest.approx(expected_flow + 0.2 * expected_temporal)


def test_untrusted_frame_and_external_trust_receive_no_gradient() -> None:
    prediction = _scalar_clip([1.0, 100.0, -2.0]).requires_grad_()
    reference = torch.zeros_like(prediction)
    weights = torch.tensor([[1.0, 0.0, 1.0]], requires_grad=True)
    loss, metrics = flow_objective(
        prediction, reference, reference, torch.tensor([0.2]), weights, temporal_weight=10.0,
    )
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad[0, 0, 1, 0, 0].item() == 0
    assert prediction.grad[0, 0, 0, 0, 0].abs().item() > 0
    assert prediction.grad[0, 0, 2, 0, 0].abs().item() > 0
    assert weights.grad is None
    assert metrics["temporal_support"] == 0
    assert metrics["temporal_loss"] == 0


def test_objective_rejects_no_reference_support_and_nonfinite_predictions() -> None:
    video = torch.zeros(1, 3, 2, 2, 2)
    with pytest.raises(ValueError, match="no trusted reference support"):
        flow_objective(video, video, video, torch.tensor([0.5]), torch.zeros(1, 2))
    prediction = video.clone()
    prediction[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="Non-finite"):
        flow_objective(prediction, video, video, torch.tensor([0.5]), torch.ones(1, 2))


@pytest.mark.parametrize("section", [None, "model", "training", "sampling"])
def test_config_rejects_unknown_keys_in_every_section(tiny_config, section: str | None) -> None:
    config = tiny_config.to_dict()
    target = config if section is None else config[section]
    target["misspelled_training_budget"] = 999
    with pytest.raises(ValueError, match="requires exactly|Unknown"):
        config_from_dict(config)


def test_cuda_request_without_gpu_fails_before_creating_run(
    tmp_path: Path, tiny_config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "must-not-start"
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        engine.train(tiny_config, tmp_path / "unused-manifest.jsonl", output, device_name="cuda")
    assert not output.exists()
    with pytest.raises(RuntimeError, match="bfloat16"):
        engine.resolve_device("cpu", "bfloat16")


def test_checkpoint_hash_failure_precedes_deserialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "weights.pt"
    save_checkpoint(checkpoint, {
        "schema_version": 1, "architecture": ARCHITECTURE,
        "model_state": {"weight": torch.ones(2)},
    })
    checkpoint.write_bytes(checkpoint.read_bytes() + b"unexpected appended bytes")

    def must_not_deserialize(*args, **kwargs):
        pytest.fail("Unverified checkpoint bytes reached torch.load")

    monkeypatch.setattr(torch, "load", must_not_deserialize)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_checkpoint(checkpoint)
    checkpoint.with_name(checkpoint.name + ".sha256").unlink()
    with pytest.raises(ValueError, match="sidecar missing"):
        load_checkpoint(checkpoint)


def test_real_cpu_training_improves_validation_without_sampling_test(
    tmp_path: Path, tiny_config, corpus: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled_splits: list[str] = []
    evaluated_splits: list[str] = []
    original_batch = engine._batch
    original_evaluate = engine.evaluate_model
    allow_test = False

    def observed_batch(dataset, indices, device):
        sampled_splits.append(dataset.split)
        assert dataset.split != "test" or allow_test
        return original_batch(dataset, indices, device)

    def observed_evaluate(model, dataset, config, device, seed=17001):
        evaluated_splits.append(dataset.split)
        return original_evaluate(model, dataset, config, device, seed=seed)

    monkeypatch.setattr(engine, "_batch", observed_batch)
    monkeypatch.setattr(engine, "evaluate_model", observed_evaluate)
    output = tmp_path / "trained"
    summary = engine.train(tiny_config, corpus, output, device_name="cpu")
    assert summary["status"] == "completed" and summary["step"] == tiny_config.training.steps
    assert summary["final_validation"]["total_loss"] < summary["initial_validation"]["total_loss"]
    assert summary["selected_step"] > 0
    assert set(sampled_splits) == {"train", "validation"}
    assert set(evaluated_splits) == {"validation"}
    assert not summary["test_used_during_training"]
    selected = load_checkpoint(output / "selected.pt")
    assert selected["provenance"]["selection_split"] == "validation"
    assert selected["provenance"]["initialization"] == "random"
    assert selected["provenance"]["data"]["split_counts"] == {"train": 7, "validation": 3, "test": 2}
    assert sha256_file(output / "selected.pt") == summary["selected_checkpoint_sha256"]
    assert json.loads((output / "status.json").read_text())["status"] == "completed"

    # Only this explicit post-training call is allowed to score the held-out test set.
    allow_test = True
    result = engine.evaluate_checkpoint(
        output / "selected.pt", corpus, tmp_path / "test-evaluation", device_name="cpu", split="test",
    )
    assert result["split"] == "test" and result["samples"] == 2
    assert evaluated_splits[-1] == "test"
    assert result["noise_seed"] != summary["final_validation"]["noise_seed"]


def _assert_exact_tree(actual, expected, location: str = "root") -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), location
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, location
        assert torch.equal(actual, expected), f"Tensor changed at {location}"
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), location
        for key in expected:
            _assert_exact_tree(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), location
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _assert_exact_tree(actual_item, expected_item, f"{location}[{index}]")
    else:
        assert actual == expected, location


def test_failure_on_third_optimizer_update_resumes_exactly_at_original_budget(
    tmp_path: Path, tiny_config, corpus: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(tiny_config, training=replace(
        tiny_config.training, steps=6, checkpoint_every=2, validate_every=2,
        gradient_accumulation=2,
    ))
    uninterrupted = tmp_path / "uninterrupted"
    interrupted = tmp_path / "interrupted"
    expected_summary = engine.train(config, corpus, uninterrupted, device_name="cpu")
    real_step = torch.optim.AdamW.step
    calls = 0

    def fail_third_step(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected interruption at optimizer update three")
        return real_step(optimizer, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", fail_third_step)
        with pytest.raises(RuntimeError, match="injected interruption"):
            engine.train(config, corpus, interrupted, device_name="cpu")
    failed_status = json.loads((interrupted / "status.json").read_text())
    assert failed_status["status"] == "failed" and failed_status["step"] == 2
    snapshot = load_checkpoint(interrupted / "resume.pt")
    assert snapshot["step"] == 2 and snapshot["config"]["training"]["steps"] == 6
    assert not (interrupted / "final.pt").exists()

    recovered = engine.train(config, corpus, interrupted, device_name="cpu", resume=True)
    assert recovered["step"] == expected_summary["step"] == 6
    expected = load_checkpoint(uninterrupted / "resume.pt")
    actual = load_checkpoint(interrupted / "resume.pt")
    for key in (
        "model_state", "ema_state", "optimizer_state", "data_rng", "noise_rng",
        "torch_rng", "cuda_rng", "best_state", "best_step", "best_score",
        "history", "validation_history",
    ):
        _assert_exact_tree(actual[key], expected[key], key)
    for filename in ("selected.pt", "final.pt"):
        _assert_exact_tree(
            load_checkpoint(interrupted / filename)["model_state"],
            load_checkpoint(uninterrupted / filename)["model_state"], filename,
        )
    with pytest.raises(ValueError, match="already completed"):
        engine.train(config, corpus, interrupted, device_name="cpu", resume=True)


def test_sft_continues_own_checkpoint_and_preserves_initialization_provenance(
    tmp_path: Path, tiny_config, corpus: Path,
) -> None:
    pretrain_config = replace(tiny_config, training=replace(tiny_config.training, steps=4))
    pretrain_out = tmp_path / "pretrain"
    initial = engine.train(pretrain_config, corpus, pretrain_out, device_name="cpu")
    initializer = pretrain_out / "final.pt"
    sft_config = replace(tiny_config, training=replace(tiny_config.training, steps=2))
    sft_out = tmp_path / "sft"
    result = engine.train(sft_config, corpus, sft_out, device_name="cpu", stage="sft", init_checkpoint=initializer)
    assert result["status"] == "completed" and result["stage"] == "sft"
    assert result["initial_validation"]["total_loss"] == initial["final_validation"]["total_loss"]
    weights = load_checkpoint(sft_out / "final.pt")
    assert weights["provenance"]["initialization"] == "own_checkpoint"
    assert weights["provenance"]["initial_checkpoint_sha256"] == sha256_file(initializer)
    assert weights["provenance"]["initial_checkpoint_stage"] == "pretrain"
    assert weights["provenance"]["initial_checkpoint_step"] == 4
    with pytest.raises(ValueError, match="pretrain starts random"):
        engine.train(tiny_config, corpus, tmp_path / "wrong-pretrain", device_name="cpu", init_checkpoint=initializer)


def test_sft_refuses_hashed_external_architecture_before_creating_run(
    tmp_path: Path, tiny_config, corpus: Path,
) -> None:
    external = tmp_path / "external.pt"
    save_checkpoint(external, {
        "schema_version": 1, "architecture": "unrelated_pretrained_video_architecture",
        "kind": "weights", "model_state": {},
    })
    output = tmp_path / "forbidden-sft"
    with pytest.raises(ValueError, match="different architecture"):
        engine.train(tiny_config, corpus, output, device_name="cpu", stage="sft", init_checkpoint=external)
    assert not output.exists()


@pytest.mark.parametrize("marker,replacement", [
    ('"name": "training-contract-test"', '"name": "training-contract-test", "name": "overwritten"'),
    ('"hidden": 16', '"hidden": 16, "hidden": 32'),
    ('"steps": 12', '"steps": 12, "steps": 13'),
    ('"steps": 2', '"steps": 2, "steps": 3'),
])
def test_config_file_rejects_duplicate_keys_before_values_are_overwritten(
    tmp_path: Path, tiny_config, marker: str, replacement: str,
) -> None:
    from text_video.config import load_config

    encoded = json.dumps(tiny_config.to_dict())
    assert encoded.count(marker) == 1
    path = tmp_path / "ambiguous-config.json"
    path.write_text(encoded.replace(marker, replacement, 1), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)duplicate"):
        load_config(path)


@pytest.mark.parametrize("zero_temporal_support", [False, True])
def test_accumulation_support_scaling_matches_concatenated_batch_gradient(
    zero_temporal_support: bool,
) -> None:
    if zero_temporal_support:
        weights = [
            torch.tensor([[1.0, 0.0, 0.0, 0.9]]),
            torch.tensor([[0.2, 0.0, 0.4, 0.0], [1.0, 0.0, 0.0, 1.0]]),
            torch.tensor([[0.4, 0.0, 0.0, 0.6]]),
        ]
    else:
        weights = [
            torch.tensor([[1.0, 0.1, 0.0, 0.9]]),
            torch.tensor([[0.2, 0.3, 0.4, 0.7], [1.0, 1.0, 1.0, 1.0]]),
            torch.tensor([[0.4, 0.0, 0.0, 0.6]]),
        ]
    scales = engine.accumulation_scales(weights)
    assert len(scales) == len(weights)
    assert sum(pair[0] for pair in scales) == pytest.approx(1.0)
    assert sum(pair[1] for pair in scales) == pytest.approx(0.0 if zero_temporal_support else 1.0)
    assert scales[-1][1] == 0.0

    generator = torch.Generator().manual_seed(41)
    shape = (sum(weight.shape[0] for weight in weights), 3, 4, 2, 2)
    clean = torch.randn(shape, generator=generator).tanh()
    noise = torch.randn(shape, generator=generator)
    values = torch.randn(shape, generator=generator)
    times = torch.tensor([0.1, 0.3, 0.6, 0.8])
    joined_prediction = values.clone().requires_grad_()
    joined_loss, _ = flow_objective(
        joined_prediction, clean, noise, times, torch.cat(weights), temporal_weight=0.7,
    )
    joined_gradient, = torch.autograd.grad(joined_loss, joined_prediction)

    accumulated_prediction = values.clone().requires_grad_()
    scaled_losses, raw_losses = [], []
    offset = 0
    for weight, (flow_scale, temporal_scale) in zip(weights, scales, strict=True):
        end = offset + weight.shape[0]
        arguments = (
            accumulated_prediction[offset:end], clean[offset:end], noise[offset:end],
            times[offset:end], weight,
        )
        raw_loss, raw_metrics = flow_objective(*arguments, temporal_weight=0.7)
        scaled_loss, scaled_metrics = flow_objective(
            *arguments, temporal_weight=0.7, flow_scale=flow_scale, temporal_scale=temporal_scale,
        )
        assert scaled_metrics == raw_metrics
        raw_losses.append(raw_loss)
        scaled_losses.append(scaled_loss)
        offset = end
    accumulated_loss = sum(scaled_losses)
    accumulated_gradient, = torch.autograd.grad(accumulated_loss, accumulated_prediction, retain_graph=True)
    torch.testing.assert_close(accumulated_loss, joined_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(accumulated_gradient, joined_gradient, atol=1e-7, rtol=1e-6)

    # This deliberately heterogeneous batch exposes the old equal-microbatch mean error.
    naive_gradient, = torch.autograd.grad(sum(raw_losses) / len(raw_losses), accumulated_prediction)
    assert (naive_gradient - joined_gradient).abs().max().item() > 1e-4


def _lineage_pretrain(tmp_path: Path, config, corpus: Path) -> Path:
    config = replace(config, training=replace(config.training, steps=2))
    output = tmp_path / "lineage-pretrain"
    engine.train(config, corpus, output, device_name="cpu")
    return output / "final.pt"


def _rewrite_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _lineage_identity(row: dict, *, validated_record: bool = False) -> tuple:
    decoded_key = "_decoded_sha256" if validated_record else "decoded_sha256"
    return row["id"], row["group_id"], row["split"], row["sha256"], row[decoded_key]


def test_decoded_digest_is_computed_and_overrides_untrusted_manifest_field(corpus: Path) -> None:
    from text_video.data import validate_manifest

    expected = validate_manifest(corpus)[0]["_decoded_sha256"]
    rows = [json.loads(line) for line in corpus.read_text().splitlines()]
    rows[0]["_decoded_sha256"] = "0" * 64
    _rewrite_manifest(corpus, rows)
    observed = validate_manifest(corpus)[0]["_decoded_sha256"]
    assert observed == expected and observed != "0" * 64


@pytest.mark.parametrize("conflict", ["group", "file_bytes", "repacked_pixels"])
def test_sft_rejects_previous_stage_split_leakage_even_when_old_manifest_is_absent(
    tmp_path: Path, tiny_config, corpus: Path, conflict: str,
) -> None:
    import numpy as np

    from text_video.data import validate_manifest

    initializer = _lineage_pretrain(tmp_path, tiny_config, corpus)
    old_records = validate_manifest(corpus)
    lineage = load_checkpoint(initializer)["provenance"]["split_lineage"]
    assert len(lineage) == len(old_records)
    assert {_lineage_identity(row) for row in lineage} == {
        _lineage_identity(row, validated_record=True) for row in old_records
    }
    original_train = next(row for row in old_records if row["split"] == "train")
    fresh = make_procedural_dataset(
        tmp_path / "new-data", train=7, validation=3, test=2,
        frames=4, height=16, width=16, seed=53,
    )
    rows = [json.loads(line) for line in fresh.read_text().splitlines()]
    moved = next(row for row in rows if row["split"] == "validation")
    if conflict == "group":
        moved["group_id"] = original_train["group_id"]
        assert moved["sha256"] != original_train["sha256"]
    else:
        original_path = corpus.parent / original_train["path"]
        if conflict == "file_bytes":
            moved["path"] = str(original_path)
            moved["sha256"] = original_train["sha256"]
        else:
            with np.load(original_path, allow_pickle=False) as archive:
                pixels = archive["frames"].copy()
            repacked = fresh.parent / "repacked-original.npz"
            np.savez(repacked, frames=pixels)
            moved["path"] = repacked.name
            moved["sha256"] = sha256_file(repacked)
            assert moved["sha256"] != original_train["sha256"]
        moved["caption"] = original_train["caption"]
        moved["reference_weight"] = original_train["reference_weight"]
        assert moved["group_id"] != original_train["group_id"]
    _rewrite_manifest(fresh, rows)
    assert len(validate_manifest(fresh)) == 12  # No within-manifest leakage explains this rejection.
    output = tmp_path / "rejected-sft"
    with pytest.raises(ValueError, match="(?i)(split|lineage)"):
        engine.train(tiny_config, fresh, output, device_name="cpu", stage="sft", init_checkpoint=initializer)
    assert not output.exists()


@pytest.mark.parametrize("new_data", [False, True])
def test_sft_allows_same_split_or_new_episodes_and_carries_all_ancestor_lineage(
    tmp_path: Path, tiny_config, corpus: Path, new_data: bool,
) -> None:
    from text_video.data import validate_manifest

    initializer = _lineage_pretrain(tmp_path, tiny_config, corpus)
    original = load_checkpoint(initializer)["provenance"]["split_lineage"]
    manifest = corpus
    if new_data:
        manifest = make_procedural_dataset(
            tmp_path / "fresh-episodes", train=7, validation=3, test=2,
            frames=4, height=16, width=16, seed=67,
        )
    config = replace(tiny_config, training=replace(tiny_config.training, steps=2))
    output = tmp_path / "allowed-sft"
    result = engine.train(config, manifest, output, device_name="cpu", stage="sft", init_checkpoint=initializer)
    assert result["status"] == "completed"
    inherited = load_checkpoint(output / "final.pt")["provenance"]["split_lineage"]
    expected = {_lineage_identity(row) for row in original}
    expected.update(_lineage_identity(row, validated_record=True) for row in validate_manifest(manifest))
    assert {_lineage_identity(row) for row in inherited} == expected
    assert len(inherited) == len(expected)

    if new_data:
        # Stage three must still protect the first pretraining split, even though
        # stage two used a completely disjoint corpus.
        third = make_procedural_dataset(
            tmp_path / "third-stage-data", train=7, validation=3, test=2,
            frames=4, height=16, width=16, seed=71,
        )
        rows = [json.loads(line) for line in third.read_text().splitlines()]
        original_train = next(row for row in original if row["split"] == "train")
        next(row for row in rows if row["split"] == "test")["group_id"] = original_train["group_id"]
        _rewrite_manifest(third, rows)
        with pytest.raises(ValueError, match="(?i)(split|lineage)"):
            engine.train(
                config, third, tmp_path / "rejected-grandchild", device_name="cpu", stage="sft",
                init_checkpoint=output / "final.pt",
            )
