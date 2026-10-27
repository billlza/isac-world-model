"""Study produces inspectable, source-bound paired evidence."""

import json

import pytest

from scripts.voi_study import run_study


def test_small_paired_study_has_all_conditions(tmp_path):
    result = run_study(tmp_path / "study", episodes=2, steps=8, seed=81_000_000)
    loaded = json.loads((tmp_path / "study/report.json").read_text())
    assert result == loaded
    assert len(loaded["experiments"]) == 10
    assert len(loaded["source_sha256"]) == 5
    for entry in loaded["experiments"].values():
        assert set(entry["methods"]) == {"fixed", "guarded", "skip", "voi"}
        for method in entry["methods"].values():
            assert len(method["trajectories"]) == 2
        assert "voi_minus_fixed" in entry["paired"]
    with pytest.raises(FileExistsError):
        run_study(tmp_path / "study", episodes=1, steps=8)
