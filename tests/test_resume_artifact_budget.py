"""Continuation state has a larger explicit bound; inference limits stay fixed."""

import io
import zipfile
from unittest.mock import patch

import pytest
import torch

from isac_wm import artifacts
from isac_wm.model import ModelConfig
from isac_wm.simulator import SceneConfig
from isac_wm.training import load_model, train


def resume_payload():
    return {
        "schema_version": 2,
        "state_dict": {"weights": torch.arange(1024, dtype=torch.float32)},
        "optimizer": {"state": {}, "param_groups": []},
        "numpy_rng_json": "{}",
        "torch_rng": torch.get_rng_state(),
        "completed_epochs": 1,
        "best_checkpoint": None,
        "history": [],
        "model_config": {},
        "scene_config": {},
    }


def test_default_model_budget_stays_256_mib_and_resume_budget_is_finite():
    assert artifacts.MAX_CHECKPOINT_BYTES == 256 * 1024**2
    assert artifacts.MAX_RESUME_CHECKPOINT_BYTES == 2 * 1024**3


def test_larger_resume_requires_explicit_api_and_cannot_enter_model_loader(tmp_path, monkeypatch):
    # Scale only the byte limits, not the serialization/verification code, to
    # exercise the actual boundary without a gigabyte fixture.
    monkeypatch.setattr(artifacts, "MAX_CHECKPOINT_BYTES", 4 * 1024)
    monkeypatch.setattr(artifacts, "MAX_RESUME_CHECKPOINT_BYTES", 64 * 1024)
    payload = resume_payload()
    path = tmp_path / "resume.pt"
    with pytest.raises(ValueError, match="artifact budget"):
        artifacts.save_checkpoint(path, payload)
    assert not path.exists()
    assert not path.with_suffix(".pt.sha256").exists()

    artifacts.save_resume_checkpoint(path, payload)
    assert artifacts.MAX_CHECKPOINT_BYTES < path.stat().st_size < artifacts.MAX_RESUME_CHECKPOINT_BYTES
    restored = artifacts.read_resume_checkpoint(path)
    torch.testing.assert_close(restored["state_dict"], payload["state_dict"], rtol=0, atol=0)
    torch.testing.assert_close(restored["torch_rng"], payload["torch_rng"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="file-size limit"):
        artifacts.read_checkpoint(path)
    with pytest.raises(ValueError, match="file-size limit"):
        load_model(path, "cpu")

    # Raising the bound does not bypass checksum verification or weights-only loading.
    with patch.object(artifacts.torch, "load", wraps=torch.load) as loader:
        artifacts.read_resume_checkpoint(path)
    assert loader.call_args.kwargs["weights_only"] is True
    path.with_suffix(".pt.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        artifacts.read_resume_checkpoint(path)


def test_resume_size_cap_applies_to_save_and_read_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "resume.pt"
    payload = resume_payload()
    artifacts.save_resume_checkpoint(path, payload)
    original = path.read_bytes()
    original_hash = path.with_suffix(".pt.sha256").read_bytes()
    monkeypatch.setattr(artifacts, "MAX_RESUME_CHECKPOINT_BYTES", len(original) - 1)
    with pytest.raises(ValueError, match="artifact budget"):
        artifacts.save_resume_checkpoint(path, payload)
    assert path.read_bytes() == original
    assert path.with_suffix(".pt.sha256").read_bytes() == original_hash
    with patch.object(artifacts.torch, "load") as loader:
        with pytest.raises(ValueError, match="file-size limit"):
            artifacts.read_resume_checkpoint(path)
    loader.assert_not_called()


@pytest.mark.parametrize("resume", [False, True])
def test_each_budget_bounds_expanded_zip_before_torch_load(tmp_path, monkeypatch, resume):
    name = "MAX_RESUME_CHECKPOINT_BYTES" if resume else "MAX_CHECKPOINT_BYTES"
    monkeypatch.setattr(artifacts, name, 1024)
    path = tmp_path / "compressed.pt"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("payload", b"x" * 1025)
    assert path.stat().st_size < 1024
    reader = artifacts.read_resume_checkpoint if resume else artifacts.read_checkpoint
    with patch.object(artifacts.torch, "load") as loader:
        with pytest.raises(ValueError, match="Expanded checkpoint"):
            reader(path)
    loader.assert_not_called()


def test_resume_rechecks_the_bounded_bytes_if_file_grows_after_stat(tmp_path, monkeypatch):
    path = tmp_path / "growing.pt"
    path.write_bytes(b"small")
    monkeypatch.setattr(artifacts, "MAX_RESUME_CHECKPOINT_BYTES", 1024)
    with patch.object(type(path), "open", return_value=io.BytesIO(b"x" * 1025)):
        with pytest.raises(ValueError, match="file-size limit"):
            artifacts.read_resume_checkpoint(path)


@pytest.mark.parametrize(
    "change",
    [
        {"optimizer": None},
        {"optimizer": {"state": [], "param_groups": []}},
        {"completed_epochs": True},
        {"completed_epochs": 0},
        {"torch_rng": "not a tensor"},
        {"best_checkpoint": []},
    ],
)
def test_resume_apis_reject_malformed_continuation_metadata(tmp_path, change):
    path = tmp_path / "invalid.pt"
    value = resume_payload() | change
    with pytest.raises(ValueError, match="training continuation fields"):
        artifacts.save_resume_checkpoint(path, value)
    assert not path.exists()
    # Existing general artifact API remains backwards compatible; the explicit
    # continuation loader rejects the same malformed payload after integrity checks.
    artifacts.save_checkpoint(path, value)
    with pytest.raises(ValueError, match="training continuation fields"):
        artifacts.read_resume_checkpoint(path)


def test_resume_api_requires_continuation_not_just_model_weights(tmp_path):
    path = tmp_path / "model.pt"
    value = {"schema_version": 2, "state_dict": {"x": torch.ones(2)}}
    artifacts.save_checkpoint(path, value)
    with pytest.raises(ValueError, match="missing training continuation fields"):
        artifacts.read_resume_checkpoint(path)
    with pytest.raises(ValueError, match="missing training continuation fields"):
        artifacts.save_resume_checkpoint(tmp_path / "not-resume.pt", value)


def test_training_uses_resume_budget_and_matches_uninterrupted_run(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    common = dict(
        episodes=2,
        batch_size=2,
        device_name="cpu",
        seed=7,
        scene=SceneConfig(steps=8, image_size=16),
        model_config=ModelConfig(feature_dim=8, hidden_dim=8),
        train_seed_base=50_000,
        validation_seed_base=60_000,
        batch_order_seed=37,
        validation_episodes=2,
        validation_every_updates=1,
    )
    try:
        train(tmp_path / "first", epochs=1, **common)
        model_size = (tmp_path / "first/model.pt").stat().st_size
        resume_size = (tmp_path / "first/resume.pt").stat().st_size
        # The optimizer/selected model make continuation larger even for this
        # tiny network. A model-sized default cap must not break save or resume.
        model_bound = model_size + 1024
        assert model_bound < resume_size
        monkeypatch.setattr(artifacts, "MAX_CHECKPOINT_BYTES", model_bound)
        monkeypatch.setattr(artifacts, "MAX_RESUME_CHECKPOINT_BYTES", resume_size * 2)
        train(tmp_path / "bounded", epochs=1, **common)
        train(tmp_path / "resumed", epochs=2, resume=tmp_path / "bounded/resume.pt", **common)
        train(tmp_path / "whole", epochs=2, **common)
        resumed = artifacts.read_resume_checkpoint(tmp_path / "resumed/resume.pt")
        whole = artifacts.read_resume_checkpoint(tmp_path / "whole/resume.pt")
        for field in ("state_dict", "optimizer", "torch_rng"):
            torch.testing.assert_close(resumed[field], whole[field], rtol=0, atol=0)
        assert resumed["history"] == whole["history"]
        assert resumed["completed_epochs"] == 2
        load_model(tmp_path / "resumed/model.pt", "cpu")
        with pytest.raises(ValueError, match="file-size limit"):
            artifacts.read_checkpoint(tmp_path / "resumed/resume.pt")
    finally:
        torch.set_num_threads(previous_threads)
