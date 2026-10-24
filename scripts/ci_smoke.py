"""End-to-end smoke; real train/save/resume/calibrate/evaluate/stress operations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def run(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    common = ["--device", "cpu", "--threads", "2"]
    def cli(*args):
        subprocess.run([sys.executable, "-m", "isac_wm", *map(str, args), *common],
                       check=True, timeout=180)
    train = out / "train"
    resumed = out / "resumed"
    cli("train", "--out", train, "--epochs", 1, "--episodes", 4, "--batch-size", 2)
    cli("train", "--out", resumed, "--epochs", 2, "--episodes", 4, "--batch-size", 2,
        "--resume", train / "resume.pt")
    calibration = out / "calibration.json"
    cli("calibrate", "--checkpoint", resumed / "model.pt", "--out", calibration,
        "--episodes", 10, "--latency-mode", "fixed", "--compute-ms", 150)
    cli("evaluate", "--checkpoint", resumed / "model.pt", "--calibration", calibration,
        "--out", out / "evaluation", "--episodes", 2,
        "--latency-mode", "fixed", "--compute-ms", 150)
    cli("stress", "--checkpoint", resumed / "model.pt", "--out", out / "stress",
        "--episodes", 1, "--latency-mode", "fixed", "--compute-ms", 150)
    report = json.loads((out / "evaluation/evaluation.json").read_text())
    if report["runtime"]["mode"] != "fixed":
        raise AssertionError("Missing runtime profile in evaluation")
    stress = json.loads((out / "stress/stress.json").read_text())
    if len(stress.get("scenarios", {})) != 6 or any(
            value.get("status") != "completed" for value in stress["scenarios"].values()):
        raise AssertionError("Stress smoke reported failures")
    print("Train -> resume -> calibrate -> delayed evaluation -> six-scenario stress: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
