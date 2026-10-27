"""Paired validation/test split and explicit failure contracts for stronger baselines."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("estimator_study", ROOT / "scripts/compare_estimators.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_paired_study_seeds_and_non_overwrite(tmp_path):
    result = module.compare(tmp_path / "paired", seeds=(7, 11), quick=True)
    assert result["status"] == "passed"
    assert result["protocol"]["test_seed_starts"] == [70_000_000, 70_010_000]
    assert len(result["training"]) == 4
    assert len(result["scores"]) == 16
    assert all(len(row["per_episode"]) == 6 for row in result["scores"])
    assert all(d["pairs"] == 2 for d in result["attention_ablation"].values())
    assert result["baseline_validation"]["kalman"]["chosen"] in (0.1, 0.8, 3.0)
    prior = (tmp_path / "paired/comparison.json").read_bytes()
    saved = json.loads(prior)
    assert saved["scores"] == result["scores"]
    assert not (tmp_path / "paired/traces").exists()
    diagnostics = {
        "ignored_older_snapshot_packets",
        "overflow_dropped_count",
        "undelivered_at_end",
        "deadline_miss_fraction",
        "service_p95_ms",
        "service_deadline_miss_fraction",
        "started_jobs",
        "completed_jobs",
        "unfinished_jobs",
    }
    for row in saved["scores"]:
        for episodes in row["per_episode"].values():
            for episode in episodes:
                metrics = episode["metrics"]
                assert diagnostics <= metrics.keys()
                assert metrics["position_rmse_m"] == episode["rmse_m"]
                assert metrics["service_p95_ms"] == row["compute_ms"]
                assert metrics["service_deadline_miss_fraction"] == (1.0 if row["compute_ms"] > 100 else 0.0)
                assert sum(episode["status_counts"].values()) == metrics["scored_count"]
                assert episode["runtime"]["compute_ms"] == row["compute_ms"]
                if row["compute_ms"]:
                    assert metrics["unfinished_jobs"] == 1
    with pytest.raises(FileExistsError):
        module.compare(tmp_path / "paired", quick=True)
    assert prior == (tmp_path / "paired/comparison.json").read_bytes()


def test_study_fails_closed_and_preserves_evidence(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("intentional baseline failure")

    monkeypatch.setattr(module, "tune_baselines", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        module.compare(tmp_path / "failed", seeds=(7,), quick=True)
    result = json.loads((tmp_path / "failed/comparison.json").read_text())
    assert result["status"] == "failed"
    assert "intentional" in result["error"]


@pytest.mark.parametrize(
    "kwargs", [{"seeds": ()}, {"seeds": (7, 7)}, {"seeds": (100,)}, {"test_episodes": 10000}, {"episodes": 0}]
)
def test_bad_protocol_never_creates_output(tmp_path, kwargs):
    with pytest.raises(ValueError):
        module.compare(tmp_path / "invalid", **kwargs)
    assert not (tmp_path / "invalid").exists()
