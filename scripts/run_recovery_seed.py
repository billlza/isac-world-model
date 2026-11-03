"""Run one formally registered Hamilton seed after a successful whole-card preflight."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def main():
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(7, 11, 13), required=True)
    args = parser.parse_args()
    out = args.dir.resolve()
    evidence = json.loads((out / "preflight.json").read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (
        evidence["status"] != "passed"
        or evidence["slurm"]["job_id"] != os.environ["SLURM_JOB_ID"]
        or evidence["git_commit"] != head
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("Study requires current-job, current-revision H200 preflight evidence")
    versions = [platform.python_version(), np.__version__, str(torch.__version__), torch.version.cuda]
    if versions != ["3.13.9", "2.3.4", "2.9.0+cu128", "12.8"]:
        raise RuntimeError(f"Unverified school stack: {versions}")
    manifest = out / "study.json"
    record = {
        "status": "running",
        "git_commit": head,
        "training_seed": args.seed,
        "runtime": versions,
        "slurm_array_id": os.environ["SLURM_ARRAY_JOB_ID"],
        "stages": [],
    }
    with manifest.open("x") as handle:
        json.dump(record, handle)

    def save():
        tmp = out / ".study.json.tmp"
        tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        tmp.replace(manifest)

    def run(label, argv):
        stage = {"name": label, "status": "running", "command": argv}
        record["stages"].append(stage)
        save()
        start = time.monotonic()
        try:
            subprocess.run([sys.executable, *argv], check=True)
        except BaseException:
            stage.update(status="failed", wall_seconds=time.monotonic() - start)
            raise
        stage.update(status="completed", wall_seconds=time.monotonic() - start)
        save()

    expected = [args.seed * 100000 + i for i in range(96)]
    val = [10000000 + args.seed * 100000 + i for i in range(12)]
    try:
        for h in (96, 192):
            for arch in ("structured", "structured-attention"):
                label = f"{arch}-{h}"
                dest = out / label
                run(
                    label,
                    [
                        "-m",
                        "isac_wm.cli",
                        "train",
                        "--device",
                        "cuda",
                        "--threads",
                        "2",
                        "--features",
                        "48",
                        "--hidden",
                        str(h),
                        "--memory-size",
                        "16",
                        "--heads",
                        "4",
                        "--architecture",
                        arch,
                        "--steps",
                        "64",
                        "--episodes",
                        "96",
                        "--batch-size",
                        "8",
                        "--epochs",
                        "16",
                        "--patience",
                        "0",
                        "--seed",
                        str(args.seed),
                        "--out",
                        str(dest),
                    ],
                )
                m = json.loads((dest / "manifest.json").read_text())
                if (
                    m["runtime"]["device"] != "cuda"
                    or m["completed_epochs"] != 16
                    or m["train_episode_seeds"] != expected
                    or m["validation_episode_seeds"] != val
                    or m["model"]["hidden_dim"] != h
                    or m["model"]["architecture"] != arch
                    or m["resume_source"] is not None
                ):
                    raise RuntimeError(f"Training output does not match registered design: {label}")
                record["stages"][-1].update(
                    parameters=m["parameters"], best_validation_rmse_m=m["best_validation_rmse_m"]
                )
            run(
                f"evaluate-{h}",
                [
                    "scripts/score_scale_recovery.py",
                    "--structured",
                    str(out / f"structured-{h}" / "model.pt"),
                    "--attention",
                    str(out / f"structured-attention-{h}" / "model.pt"),
                    "--device",
                    "cuda",
                    "--episodes",
                    "24",
                    "--seed-base",
                    "93621000",
                    "--steps",
                    "64",
                    "--out",
                    str(out / f"recovery-{h}.json"),
                ],
            )
        record["status"] = "completed"
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
