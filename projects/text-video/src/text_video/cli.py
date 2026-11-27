"""One command line for data validation, training, and actual video generation."""

import argparse
import json
from pathlib import Path
import sys

from .config import load_config
from .data import make_procedural_dataset
from .engine import evaluate_checkpoint, train, validate_data
from .generate import generate


def build_parser():
    parser = argparse.ArgumentParser(description="Independent text-conditioned video flow model (random initialization)")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("make-data", help="Original synthetic development corpus, not a general video benchmark")
    data.add_argument("--out", required=True)
    for name, default in (("train", 256), ("validation", 32), ("test", 32), ("frames", 8), ("height", 32), ("width", 32), ("seed", 7)):
        data.add_argument("--" + name, type=int, default=default)
    validate = sub.add_parser("validate-data")
    validate.add_argument("--config", required=True)
    validate.add_argument("--manifest", required=True)
    training = sub.add_parser("train")
    for name in ("config", "manifest", "out"):
        training.add_argument("--" + name, required=True)
    training.add_argument("--device", required=True, choices=("cpu", "cuda"))
    training.add_argument("--stage", default="pretrain", choices=("pretrain", "sft"))
    training.add_argument("--init-checkpoint")
    training.add_argument("--resume", action="store_true")
    sampling = sub.add_parser("sample")
    sampling.add_argument("--checkpoint", required=True)
    sampling.add_argument("--prompt", action="append", required=True)
    sampling.add_argument("--out", required=True)
    sampling.add_argument("--device", required=True, choices=("cpu", "cuda"))
    sampling.add_argument("--seed", type=int, default=101)
    sampling.add_argument("--same-seed", action="store_true", help="Use the same noise for a controlled prompt comparison")
    sampling.add_argument("--steps", type=int)
    sampling.add_argument("--guidance-scale", type=float)
    sampling.add_argument("--preview-scale", type=int, default=8)
    evaluation = sub.add_parser("evaluate")
    for name in ("checkpoint", "manifest", "out"):
        evaluation.add_argument("--" + name, required=True)
    evaluation.add_argument("--device", required=True, choices=("cpu", "cuda"))
    evaluation.add_argument("--split", choices=("validation", "test"), default="test")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "make-data":
            manifest = make_procedural_dataset(args.out, train=args.train, validation=args.validation, test=args.test, frames=args.frames, height=args.height, width=args.width, seed=args.seed)
            result = {"manifest": str(Path(manifest).resolve()), "role": "synthetic-development"}
        elif args.command == "validate-data":
            result = validate_data(args.manifest, load_config(args.config))
        elif args.command == "train":
            result = train(load_config(args.config), args.manifest, args.out, device_name=args.device, stage=args.stage, init_checkpoint=args.init_checkpoint, resume=args.resume)
        elif args.command == "sample":
            result = generate(args.checkpoint, args.prompt, args.out, device_name=args.device, seed=args.seed, steps=args.steps, guidance_scale=args.guidance_scale, preview_scale=args.preview_scale, same_seed=args.same_seed)
        else:
            result = evaluate_checkpoint(args.checkpoint, args.manifest, args.out, device_name=args.device, split=args.split)
    except (ValueError, OSError, RuntimeError, FloatingPointError) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
