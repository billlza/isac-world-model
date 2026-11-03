"""Check the actual Hamilton inputs against the previously collected experiment evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REFERENCE = Path(__file__).resolve().parents[1] / "examples/memory_evidence/round2_provenance.json"


def verify(directory, reference):
    records = reference["records"]
    weights = reference["checkpoints"]
    expected_records = {
        f"seed-{s}/{f}" for s in (7, 11, 13) for f in ("study.json", "recovery-96.json", "recovery-192.json")
    }
    expected_weights = {
        f"seed-{s}/{a}-{h}/model.pt"
        for s in (7, 11, 13)
        for a in ("structured", "structured-attention")
        for h in (96, 192)
    }
    if set(records) != expected_records or set(weights) != expected_weights or reference["job"] != "19341792":
        raise ValueError("Reference does not cover the registered job and all 12 trained checkpoints")
    for group in (records, weights):
        for name, expected in group.items():
            with (directory / name).open("rb") as f:
                found = hashlib.file_digest(f, "sha256").hexdigest()
            if found != expected:
                raise ValueError(f"Different bytes than the round-two evidence: {name}")
    return {"status": "verified", **reference}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--array-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = verify(args.array_dir, json.loads(REFERENCE.read_text()))
    with args.out.open("x") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
