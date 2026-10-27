from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .artifacts import write_json
from .calibration import calibrate, load_calibration
from .evaluation import evaluate
from .imm import IMMConfig
from .model import ModelConfig
from .runtime import RuntimeConfig
from .safety import MonitorConfig
from .scenarios import SCENARIOS, scenario_config
from .simulator import SceneConfig
from .structured import StructuredConfig
from .toolchain import inspect_toolchain
from .training import load_model, resolve_device, runtime_info, train
from .viewer import write_viewer


def _policies(text: str) -> tuple[str, ...]:
    values = tuple(x.strip() for x in text.split(","))
    if (
        not values
        or len(values) != len(set(values))
        or set(values) - {"fixed", "predictive", "risk", "guarded"}
    ):
        raise argparse.ArgumentTypeError("Use distinct policies from fixed,predictive,risk,guarded")
    return values


def _estimators(text: str) -> tuple[str, ...]:
    values = tuple(x.strip() for x in text.split(","))
    allowed = {"kalman", "kalman-gated", "imm", "imm-gated", "model"}
    if not values or len(values) != len(set(values)) or set(values) - allowed:
        raise argparse.ArgumentTypeError(
            "Use distinct estimators from kalman,kalman-gated,imm,imm-gated,model"
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent visual-radio world model; research use only")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", help="Train all weights from random initialization, or resume your own run")
    p.add_argument("--out", type=Path, default=Path("runs/train"))
    p.add_argument(
        "--epochs", type=int, default=10, help="Total epoch target, including epochs before resume"
    )
    p.add_argument("--episodes", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--architecture", choices=["gru", "structured", "structured-attention"], default="gru")
    p.add_argument("--memory-size", type=int, default=16)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--features", type=int, default=48)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--resume", type=Path)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=0, help="0 disables early stopping")
    p.add_argument("--max-dataset-gb", type=float, default=4.0)
    p = sub.add_parser("evaluate", help="Paired independent evaluation with optional episode calibration")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("runs/eval"))
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=30_000_000)
    p.add_argument("--policies", type=_policies, default=("fixed", "guarded"))
    p.add_argument("--calibration", type=Path)
    p.add_argument("--estimators", type=_estimators, default=("kalman", "model"))
    p.add_argument("--kf-process-noise", type=float, default=0.8)
    p.add_argument("--imm-switch-rate", type=float, default=0.6)
    p.add_argument("--scenario", choices=SCENARIOS, default="nominal")
    p.add_argument("--budget", type=float, default=0.1)
    p.add_argument("--deadline-ms", type=float, default=100.0)
    p.add_argument("--max-age", type=float, default=0.6)
    p = sub.add_parser(
        "calibrate", help="Calibrate on separate whole episodes, not dependent individual frames"
    )
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("runs/calibration.json"))
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--seed", type=int, default=20_000_000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--policies", type=_policies, default=("fixed", "guarded"))
    p = sub.add_parser("stress", help="Paired fault scenarios; never reuse nominal calibration under shift")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("runs/stress"))
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--seed", type=int, default=40_000_000)
    p.add_argument("--policies", type=_policies, default=("fixed", "guarded"))
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--estimators", type=_estimators, default=("kalman", "model"))
    p.add_argument("--deadline-ms", type=float, default=100.0)
    p = sub.add_parser("doctor", help="Report actual available runtime")
    for p in sub.choices.values():
        p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
        p.add_argument("--threads", type=int, default=4)
    for command in ("evaluate", "calibrate", "stress"):
        p = sub.choices[command]
        p.add_argument("--latency-mode", choices=["reference", "fixed", "measured"], default="reference")
        p.add_argument(
            "--compute-ms",
            type=float,
            default=0.0,
            help="Service time in fixed mode, not a measured benchmark",
        )
        p.add_argument("--queue-capacity", type=int, default=4096)
    args = parser.parse_args()
    if not 1 <= args.threads <= 1024:
        parser.error("--threads must be between 1 and 1024")
    torch.set_num_threads(args.threads)
    try:
        if args.command == "train":
            if not 0 < args.max_dataset_gb <= 1024:
                parser.error("--max-dataset-gb must be in (0,1024]")
            result = train(
                args.out,
                args.epochs,
                args.episodes,
                args.batch_size,
                args.device,
                args.seed,
                scene=SceneConfig(steps=args.steps, dt=args.dt),
                model_config=(
                    ModelConfig(args.features, args.hidden)
                    if args.architecture == "gru"
                    else StructuredConfig(
                        args.features, args.hidden, args.architecture, args.memory_size, args.heads
                    )
                ),
                resume=args.resume,
                learning_rate=args.learning_rate,
                patience=args.patience,
                max_dataset_bytes=int(args.max_dataset_gb * 1024**3),
            )
        elif args.command in {"evaluate", "calibrate", "stress"}:
            runtime = RuntimeConfig(args.latency_mode, args.compute_ms, args.queue_capacity)
            model, scene, device = load_model(args.checkpoint, args.device)
            if args.command == "calibrate":
                artifact = calibrate(
                    args.out,
                    model,
                    scene,
                    device,
                    args.episodes,
                    args.seed,
                    args.alpha,
                    args.policies,
                    runtime=runtime,
                )
                result = {
                    "path": str(args.out),
                    "scope": artifact["scope"],
                    "quantiles": {k: v["quantile"] for k, v in artifact["variants"].items()},
                }
            elif args.command == "evaluate":
                calibration = load_calibration(args.calibration) if args.calibration else None
                scene = scenario_config(args.scenario, scene)
                report = evaluate(
                    args.out,
                    model,
                    scene,
                    device,
                    args.episodes,
                    args.seed,
                    policies=args.policies,
                    calibration=calibration,
                    sensor_budget=args.budget,
                    runtime=runtime,
                    estimators=args.estimators,
                    baseline_process_noise=args.kf_process_noise,
                    imm_config=IMMConfig(switch_rate_hz=args.imm_switch_rate),
                    monitor=MonitorConfig(
                        max_information_age_s=args.max_age, compute_deadline_ms=args.deadline_ms
                    ),
                )
                write_viewer(report, args.out / "replay.html")
                result = {"summary": report["summary"], "viewer": str(args.out / "replay.html")}
            else:
                if args.out.exists() and any(args.out.iterdir()):
                    raise FileExistsError("Stress output must be a new directory")
                if len(args.scenarios) != len(set(args.scenarios)):
                    raise ValueError("Duplicate stress scenarios")
                result = {
                    "schema_version": 1,
                    "scenarios": {},
                    "note": "Fault injection, not safety certification; all failures retained.",
                }
                for name in args.scenarios:
                    try:
                        report = evaluate(
                            args.out / name,
                            model,
                            scenario_config(name, scene),
                            device,
                            args.episodes,
                            args.seed,
                            policies=args.policies,
                            estimators=args.estimators,
                            runtime=runtime,
                            monitor=MonitorConfig(compute_deadline_ms=args.deadline_ms),
                        )
                        write_viewer(report, args.out / name / "replay.html")
                        result["scenarios"][name] = {"status": "completed", "summary": report["summary"]}
                    except (ValueError, RuntimeError, FloatingPointError) as exc:
                        result["scenarios"][name] = {"status": "failed", "error": str(exc)}
                write_json(args.out / "stress.json", result)
        else:
            result = runtime_info(resolve_device(args.device))
            result["toolchain"] = inspect_toolchain()
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.command == "stress" and any(v["status"] != "completed" for v in result["scenarios"].values()):
            raise SystemExit(1)
    except (ValueError, OSError, MemoryError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
