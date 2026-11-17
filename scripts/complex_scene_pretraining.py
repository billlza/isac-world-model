#!/usr/bin/env python3
"""Run the random-init, mixed-scene three-recipe pretraining study."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from isac_wm.pretraining import profile_config, run_pretraining  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "development", "formal"), default="development")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--resume", action="store_true", help="Continue verified complete-epoch checkpoints in --out"
    )
    parser.add_argument("--epochs", type=int, help="Explicit budget override, recorded in the protocol")
    parser.add_argument(
        "--episodes", type=int, help="Explicit corpus-size override, recorded in the protocol"
    )
    parser.add_argument(
        "--initialization-seeds", type=int, nargs="+", help="Paired initialization seeds for all recipes"
    )
    args = parser.parse_args()
    if not 1 <= args.threads <= 64:
        parser.error("--threads must be in [1,64]")
    torch.set_num_threads(args.threads)
    config = profile_config(args.profile)
    overrides = {
        name: getattr(args, name) for name in ("epochs", "episodes") if getattr(args, name) is not None
    }
    if args.initialization_seeds is not None:
        overrides["initialization_seeds"] = tuple(args.initialization_seeds)
    config = replace(config, **overrides)
    result = run_pretraining(args.out, config, device_name=args.device, source_root=ROOT, resume=args.resume)
    print(
        json.dumps({"status": result["status"], "out": str(args.out), "comparison": result["comparison"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
