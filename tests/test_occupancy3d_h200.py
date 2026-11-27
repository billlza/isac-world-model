"""CPU checks for launch preparation and explicit CUDA refusal, not GPU validation."""

import ast
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_scientific_functions_match_frozen_pilot():
    def functions(path):
        tree = ast.parse(path.read_text())
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

    original = functions(ROOT / "scripts/occupancy3d_study.py")
    gpu = functions(ROOT / "scripts/occupancy3d_h200_study.py")
    for name in original.keys() - {"run", "main", "source_hashes"}:
        assert gpu[name] == original[name], name


def test_prepare_is_not_submission_and_is_repeatable(tmp_path):
    output = tmp_path / "prepared"
    command = [
        "bash",
        str(ROOT / "scripts/submit_occupancy3d_pretrain.sh"),
        "--output-root",
        str(output),
    ]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(first.stdout)["status"] == "prepared_not_submitted"
    assert json.loads(second.stdout)["status"] == "prepared_not_submitted"
    assert not (output / ".submission").exists()
    manifest = json.loads((output / "submission_manifest.json").read_text())
    assert manifest["gpu_validation"] == "not_executed"
    assert manifest["configuration"]["stages"]["RL"] == "not_implemented_not_executed"
    assert manifest["configuration"]["stages"]["RLHF"] == "not_implemented_not_executed"


def test_actual_cuda_preflight_fails_without_cuda(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "occupancy3d_h200_study", ROOT / "scripts/occupancy3d_h200_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no device fallback"):
        module.cuda_training_preflight(4, 0.2)


def test_prior_scene_hashes_reject_duplicates_and_pin_contents(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "occupancy3d_h200_study", ROOT / "scripts/occupancy3d_h200_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "prior.json"
    scene_hash = "a" * 64
    path.write_text(json.dumps({"scene_count": 1, "scene_hashes": [scene_hash]}))
    values, record = module.load_prior_scene_hashes(path)
    assert values == {scene_hash}
    assert record["scene_count"] == 1
    assert len(record["sha256"]) == 64
    path.write_text(json.dumps({"scene_count": 2, "scene_hashes": [scene_hash] * 2}))
    with pytest.raises(ValueError, match="uniqueness"):
        module.load_prior_scene_hashes(path)
