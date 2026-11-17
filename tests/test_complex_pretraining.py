"""Scientific split, continuation, completion and inference-artifact contracts."""

import json
from dataclasses import replace

import pytest
import torch

import isac_wm.pretraining as study
from isac_wm.artifacts import model_fingerprint, read_checkpoint, sha256_file
from isac_wm.simulator import SceneConfig
from isac_wm.training import batchify, load_model


@pytest.fixture(autouse=True)
def small_runtime(monkeypatch):
    torch.set_num_threads(2)
    catalogue = study.scene_profiles
    monkeypatch.setattr(
        study,
        "scene_profiles",
        lambda scene: {
            name: profile for name, profile in catalogue(scene).items() if name in {"nominal", "combined"}
        },
    )


def test_seed_namespaces_preserve_old_holdouts_and_pair_data():
    seen = set()
    for profile in ("smoke", "development", "formal"):
        config = study.profile_config(profile)
        splits = []
        for split in ("train", "validation", "test"):
            plan = study.episode_plan(config, "mixed-control", split)
            seeds = {row["seed"] for row in plan}
            assert len(seeds) == len(plan)
            assert min(seeds) >= 610_000_000
            assert not seeds & seen
            seen.update(seeds)
            splits.append(seeds)
        assert not (splits[0] & splits[1] | splits[1] & splits[2] | splits[0] & splits[2])
        assert study.episode_plan(config, "mixed-control", "train") == study.episode_plan(
            config, "mixed-enhanced", "train"
        )
        nominal = study.episode_plan(config, "nominal-control", "train")
        assert {row["seed"] for row in nominal} == splits[0]
        assert all(row["profile"] == "nominal" and not row["appearance"] for row in nominal)


def test_multihorizon_evaluation_matches_model_and_ignores_labels():
    config = study.profile_config("smoke")
    model = study.build_model(study.model_config(config, "mixed-enhanced")).eval()
    batch = batchify([study.make_episode(610_000_100, SceneConfig(steps=12))], torch.device("cpu"))
    with torch.inference_mode():
        actual = study._prediction(model, batch, config.horizons, 0.1)
        for horizon in config.horizons:
            expected = model(batch, forecast_s=horizon * 0.1, batched_encoding=True)
            torch.testing.assert_close(actual["mean"], expected["mean"], rtol=0, atol=0)
            torch.testing.assert_close(actual["future"][horizon], expected["future_mean"], rtol=0, atol=0)
        for key in ("state_target", "state_reference_variance", "state_reference_validity", "action_target"):
            batch[key].fill_(99)
        changed = study._prediction(model, batch, config.horizons, 0.1)
        torch.testing.assert_close(actual["mean"], changed["mean"], rtol=0, atol=0)


def test_completed_study_has_loadable_weights_and_detects_corruption(tmp_path):
    out = tmp_path / "study"
    result = study.run_pretraining(out, study.profile_config("smoke"))
    assert result["status"] == "completed"
    assert set(result["planned_cells"]) == set(result["cells"])
    assert len(result["cells"]) == 3
    initial = []
    for cell in result["cells"].values():
        study.verify_cell_artifacts(out, cell)
        manifest = json.loads((out / cell["manifest"]).read_text())
        model, scene, _ = load_model(out / cell["path"] / "model.pt", "cpu")
        assert scene.steps == 12
        assert model_fingerprint(model) != manifest["initial_parameters_sha256"]
        if cell["recipe"] != "mixed-enhanced":
            initial.append(manifest["initial_parameters_sha256"])
    assert initial[0] == initial[1]
    resumed = study.run_pretraining(out, study.profile_config("smoke"), resume=True)
    assert resumed["artifacts_sha256"] == result["artifacts_sha256"]
    with pytest.raises(FileExistsError, match="overwrite"):
        study.run_pretraining(out, study.profile_config("smoke"))
    cell = next(iter(result["cells"].values()))
    metric_path = out / cell["metrics"]
    metric_path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        study.run_pretraining(out, study.profile_config("smoke"), resume=True)
    assert json.loads((out / "study.json").read_text())["status"] == "failed"


def test_completed_epoch_resume_is_exact(tmp_path, monkeypatch):
    config = replace(study.profile_config("smoke"), epochs=2)
    source = study.source_provenance(study.Path(__file__).resolve().parents[1])
    original_save = study.save_resume_checkpoint

    def interrupt_after_commit(path, value):
        original_save(path, value)
        raise RuntimeError("interrupted after committed epoch")

    monkeypatch.setattr(study, "save_resume_checkpoint", interrupt_after_commit)
    with pytest.raises(RuntimeError, match="committed epoch"):
        study.train_cell(tmp_path / "resumed", config, "mixed-enhanced", 7, torch.device("cpu"), source)
    monkeypatch.setattr(study, "save_resume_checkpoint", original_save)
    resumed = study.train_cell(
        tmp_path / "resumed", config, "mixed-enhanced", 7, torch.device("cpu"), source, resume=True
    )
    whole = study.train_cell(tmp_path / "whole", config, "mixed-enhanced", 7, torch.device("cpu"), source)
    for filename in ("model.pt", "final.pt"):
        left = read_checkpoint(tmp_path / "resumed" / resumed["path"] / filename)
        right = read_checkpoint(tmp_path / "whole" / whole["path"] / filename)
        for key in left["state_dict"]:
            torch.testing.assert_close(left["state_dict"][key], right["state_dict"][key], rtol=0, atol=0)
    assert resumed["test_macro_position_rmse_m"] == whole["test_macro_position_rmse_m"]


def test_failure_before_first_epoch_preserves_attempt_and_restarts(tmp_path, monkeypatch):
    out = tmp_path / "early"
    original_loss = study.pretraining_loss

    def fail_before_update(*args, **kwargs):
        raise RuntimeError("injected loss failure")

    monkeypatch.setattr(study, "pretraining_loss", fail_before_update)
    with pytest.raises(RuntimeError, match="injected loss"):
        study.run_pretraining(out, study.profile_config("smoke"))
    failed = json.loads((out / "study.json").read_text())
    assert failed["status"] == "failed"
    assert not list(out.glob("*/resume.pt"))
    monkeypatch.setattr(study, "pretraining_loss", original_loss)
    completed = study.run_pretraining(out, study.profile_config("smoke"), resume=True)
    assert completed["status"] == "completed"
    recovery = completed["recovery_events"][0]
    assert recovery["action"] == "restart_from_random_initialization"
    preserved = study.Path(recovery["preserved_attempt"])
    assert json.loads((preserved / "manifest.json").read_text())["status"] == "failed"


def test_arbitrary_hash_list_cannot_assert_completion(tmp_path):
    path = tmp_path / "unrelated.txt"
    path.write_text("unrelated")
    with pytest.raises(ValueError):
        study.verify_cell_artifacts(
            tmp_path,
            {
                "status": "completed",
                "artifacts_sha256": {path.name: sha256_file(path)},
            },
        )
