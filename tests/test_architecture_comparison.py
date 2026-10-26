"""Architecture selection must preserve failures and held-out comparisons."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "architecture_comparison", ROOT / "scripts/compare_architectures.py"
)
comparison = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = comparison
spec.loader.exec_module(comparison)


def test_compare_protocol_and_overwrite(tmp_path):
    result = comparison.compare(tmp_path / "study", seeds=(7,), quick=True)
    assert result["status"] == "passed"
    assert result["scope"] == "pipeline smoke only"
    assert len(result["training"]) == 3
    assert len(result["tests"]) == 15
    assert result["selected_by_validation"] == result["validation_ranking"][0]["architecture"]
    assert all(set(x["summary"]) == {"kalman+fixed", "model+fixed"} for x in result["tests"])
    for item in result["tests"]:
        assert {r["seed"] for r in item["per_episode"]} == {60_000_000}
    original = (tmp_path / "study/comparison.json").read_bytes()
    with pytest.raises(FileExistsError):
        comparison.compare(tmp_path / "study", quick=True)
    assert (tmp_path / "study/comparison.json").read_bytes() == original


def test_compare_failure_record(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate training failure")

    monkeypatch.setattr(comparison, "train", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        comparison.compare(tmp_path / "failed", seeds=(7,), quick=True)
    result = json.loads((tmp_path / "failed/comparison.json").read_text())
    assert result["status"] == "failed"
    assert "deliberate" in result["error"]
    assert result["training"] == []


@pytest.mark.parametrize(
    "kwargs",
    [{"seeds": ()}, {"seeds": (7, 7)}, {"seeds": (100,)}, {"episodes": 0}, {"test_episodes": 10_001}],
)
def test_compare_invalid_protocol(tmp_path, kwargs):
    with pytest.raises(ValueError):
        comparison.compare(tmp_path / "invalid", **kwargs)
    assert not (tmp_path / "invalid").exists()
