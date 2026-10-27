"""Fail-closed study summaries from complete source-bound evidence."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("study_statistics", ROOT / "scripts/study_statistics.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _study(path: Path, *, status: str = "passed", fingerprint: dict | None = None):
    item = {
        "status": status,
        "source_sha256": fingerprint or {"toy": "expected"},
        "protocol": {"train_seeds": [7], "scenarios": [["nominal", 0]]},
        "scores": [
            {
                "scenario": "nominal",
                "compute_ms": 0,
                "per_episode": {
                    "structured": [{"seed": 7, "rmse_m": 0.5}],
                    "structured-attention": [{"seed": 7, "rmse_m": 0.3}],
                    "kalman": [{"seed": 7, "rmse_m": 0.2}],
                },
            }
        ],
        "training": [],
        "baseline_validation": {},
    }
    path.write_text(json.dumps(item), encoding="utf-8")
    return path


def test_completed_source_bound_study_can_be_summarized(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "sources", lambda: {"toy": "expected"})
    study = _study(tmp_path / "complete.json")
    out = tmp_path / "results.json"
    result = module.summarize(study, None, out)
    assert result["original_study_status"] == "passed"
    assert result["verification_status"] == "passed_complete_run_and_source_hash_match"
    assert result["scenarios"][0]["attention_minus_structured_mean_m"] == pytest.approx(-0.2)
    prior = out.read_bytes()
    with pytest.raises(FileExistsError):
        module.summarize(study, None, out)
    assert prior == out.read_bytes()


def test_source_mismatch_rejected_without_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "sources", lambda: {"toy": "new-code"})
    study = _study(tmp_path / "wrong.json")
    with pytest.raises(ValueError, match="source hash differs"):
        module.summarize(study, None, tmp_path / "should-not-exist.json")
    assert not (tmp_path / "should-not-exist.json").exists()


def test_failed_study_requires_verified_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "sources", lambda: {"toy": "expected"})
    study = _study(tmp_path / "failed.json", status="failed")
    with pytest.raises(ValueError, match="Only completed passed"):
        module.summarize(study, None, tmp_path / "nope.json")
