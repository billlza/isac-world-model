from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .training import train, load_model, runtime_info, resolve_device
from .evaluation import evaluate
from .viewer import write_viewer
from .model import ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent visual-radio world-model prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", help="Train every model component from random initialization")
    p.add_argument("--out", type=Path, default=Path("runs/train"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--episodes", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--features", type=int, default=48)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("evaluate", help="Paired independent test episodes; preserve all variant results")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("runs/eval"))
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=30_000_000)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("doctor", help="Report actual runtime; never assume GPU availability")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    if hasattr(args, "threads"):
        if args.threads < 1:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)
    if args.command == "train":
        result = train(args.out, args.epochs, args.episodes, args.batch_size, args.device, args.seed,
                       model_config=ModelConfig(args.features, args.hidden))
    elif args.command == "evaluate":
        model, scene, device = load_model(args.checkpoint, args.device)
        report = evaluate(args.out, model, scene, device, args.episodes, args.seed)
        write_viewer(report, args.out/"replay.html")
        result = {"summary": report["summary"], "viewer": str(args.out/"replay.html")}
    else:
        result = runtime_info(resolve_device(args.device))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
