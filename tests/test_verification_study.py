import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verification_study", ROOT / "scripts/verification_study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_paired_study_preserves_all_outcomes_and_same_initial_evidence(tmp_path):
    torch.set_num_threads(2)
    report = study.run_study(tmp_path / "paired", episodes=1, seed=11600)
    assert len(report["episodes"]) == 12
    assert len(report["summary"]) == 12
    assert len(report["paired_episodes"]) == 6
    for scenario in study.SCENARIOS:
        rows = [r for r in report["episodes"] if r["scenario"] == scenario]
        assert rows[0]["anchor"] == rows[1]["anchor"]
        assert rows[0]["initial_phy"] == rows[1]["initial_phy"]
        assert {r["planner"] for r in rows} == set(study.PLANNERS)
        for row in rows:
            assert row["metrics"]["travel_m"] <= row["config"]["travel_budget_m"]
    stored = json.loads((tmp_path / "paired/summary.json").read_text())
    assert stored["provenance"]["status"] == "completed"
    with pytest.raises(FileExistsError):
        study.run_study(tmp_path / "paired", episodes=1)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        study.summarize(report["episodes"][:-1], [11600])


@pytest.mark.parametrize("kwargs", [{"episodes": 0}, {"seed": -1}, {"episodes": True}])
def test_invalid_study_is_rejected_before_writing(tmp_path, kwargs):
    with pytest.raises(ValueError):
        study.run_study(tmp_path / "invalid", **kwargs)
    assert not (tmp_path / "invalid").exists()


def test_failed_study_retains_failure(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate paired generator failure")

    monkeypatch.setattr(study.SyntheticWorld, "initial_rf", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        study.run_study(tmp_path / "failed", episodes=1)
    manifest = json.loads((tmp_path / "failed/manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["failure_type"] == "RuntimeError"
    assert not (tmp_path / "failed/study.json").exists()
