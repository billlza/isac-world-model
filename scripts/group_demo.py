"""Prepare an offline meeting demo from a clean clone; never download weights.

The quick preset tests the pipeline, not model quality. All comparisons use the
same checkpoint and held-out episode seeds. Existing outputs are never replaced.
"""

from __future__ import annotations

import argparse
import html
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from evidence import sha256, source_snapshot, tree_digest

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("nominal-0ms", "nominal", 0),
    ("nominal-150ms", "nominal", 150),
    ("camera-loss-150ms", "camera_loss", 150),
)
NOTE = (
    "Synthetic 2-D tracking, not real camera/radar, SLAM, waveform ISAC or quantum hardware. "
    "Service times are simulated and tick-quantized, not Mac/GPU benchmarks. "
    "No calibration is applied: uncalibrated outputs are not safety-certified. "
    "Same airtime cap does not imply equal actual sensing usage."
)


def save_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def write_index(out: Path, summaries: dict) -> None:
    sections = []
    for name, metrics in summaries.items():
        rows = []
        for method, values in metrics.items():
            cells = [html.escape(method)]
            for key in ("position_rmse_m", "position_p95_m", "sensing_airtime_fraction"):
                value = values.get(key)
                cells.append("n/a" if value is None else f"{value:.4f}")
            rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in cells) + "</tr>")
        sections.append(
            f'<h2><a href="{name}/replay.html">{html.escape(name)} — open replay</a></h2>'
            "<table><tr><th>Estimator / policy</th><th>RMSE (m)</th>"
            "<th>p95 error (m)</th><th>Sensing airtime</th></tr>" + "".join(rows) + "</table>"
        )
    text = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>ISAC research demo</title><style>"
        "body{font:17px system-ui;max-width:1000px;margin:36px auto;padding:0 20px;line-height:1.6}"
        "table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid;text-align:left}"
        "</style><h1>Asynchronous visual–radio tracking</h1><p>"
        + html.escape(NOTE)
        + "</p><p>Read each case in order: nominal, processing delay, then camera failure. "
        "The plots include Kalman and learned estimators; no method is hidden.</p>"
        + "".join(sections)
        + '<p><a href="demo-manifest.json">Provenance and commands</a> · '
        '<a href="summary.json">Raw summary</a></p></html>'
    )
    (out / "index.html").write_text(text, encoding="utf-8")


def prepare(out: Path, *, checkpoint: Path | None = None, quick: bool = False) -> dict:
    if checkpoint is not None:
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise ValueError("Checkpoint must be an existing regular file; omit it to train from scratch")
        checkpoint = checkpoint.resolve()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = source_snapshot(ROOT)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "preset": "pipeline-smoke" if quick else "meeting",
        "scope": NOTE,
        "source_tree_sha256": tree_digest(source),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy")},
        "test_seed_start": 30_000_000,
        "commands": [],
    }
    report_path = out / "demo-manifest.json"
    save_json(report_path, manifest)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

    def cli(name: str, *args: object) -> None:
        command = [sys.executable, "-m", "isac_wm", *map(str, args), "--device", "cpu", "--threads", "2"]
        item = {"name": name, "command": command, "status": "running", "log": f"{name}.log"}
        manifest["commands"].append(item)
        save_json(report_path, manifest)
        print(f"{name}: running", flush=True)
        start = time.perf_counter()
        try:
            with (out / item["log"]).open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800
                )
        except (OSError, subprocess.TimeoutExpired):
            item.update(status="failed", exit_code=None)
            item["duration_s"] = time.perf_counter() - start
            save_json(report_path, manifest)
            raise
        item.update(status="passed" if result.returncode == 0 else "failed", exit_code=result.returncode)
        item["duration_s"] = time.perf_counter() - start
        save_json(report_path, manifest)
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {out / item['log']}")

    try:
        if checkpoint is None:
            cli(
                "training",
                "train",
                "--out",
                out / "train",
                "--episodes",
                4 if quick else 64,
                "--epochs",
                1 if quick else 12,
                "--batch-size",
                2 if quick else 8,
                "--features",
                16 if quick else 48,
                "--hidden",
                32 if quick else 96,
                "--steps",
                12 if quick else 48,
            )
            checkpoint = out / "train/model.pt"
        manifest["checkpoint"] = str(checkpoint)
        manifest["checkpoint_sha256"] = sha256(checkpoint)
        summaries = {}
        for name, scenario, milliseconds in CASES:
            cli(
                name,
                "evaluate",
                "--checkpoint",
                checkpoint,
                "--out",
                out / name,
                "--episodes",
                1 if quick else 4,
                "--seed",
                manifest["test_seed_start"],
                "--scenario",
                scenario,
                "--latency-mode",
                "fixed",
                "--compute-ms",
                milliseconds,
                "--policies",
                "fixed,guarded",
            )
            report = json.loads((out / name / "evaluation.json").read_text())
            if (
                report["runtime"]["mode"] != "fixed"
                or report["test_seed_start"] != manifest["test_seed_start"]
            ):
                raise ValueError("Demo result does not match the declared protocol")
            if not (out / name / "replay.html").is_file():
                raise ValueError("Missing offline replay")
            summaries[name] = report["summary"]
        if source_snapshot(ROOT) != source or sha256(checkpoint) != manifest["checkpoint_sha256"]:
            raise ValueError("Source or checkpoint changed during demonstration")
        save_json(out / "summary.json", summaries)
        write_index(out, summaries)
        manifest["artifact_sha256"] = {
            str(path.relative_to(out)): sha256(path)
            for path in sorted(out.rglob("*"))
            if path.is_file() and path != report_path
        }
        manifest["status"] = "passed"
        return manifest
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        save_json(report_path, manifest)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--quick", action="store_true", help="One-epoch pipeline smoke; not an accuracy result"
    )
    args = parser.parse_args(argv)
    try:
        prepare(args.out, checkpoint=args.checkpoint, quick=args.quick)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"Demo failed: {exc}", file=sys.stderr)
        return 1
    print(f"Open {(args.out.resolve() / 'index.html').as_uri()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
