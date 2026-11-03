"""Optional independent reproduction on the archived CPU development checkpoint (not Hamilton)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    import numpy as np
    import torch
    from frozen_recovery import evaluate_case

    from isac_wm.artifacts import sha256_file
    from isac_wm.training import load_model

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--artifact-dir", type=Path, required=True, help="Extracted recovery-scale-development artifact"
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    evidence = Path(__file__).resolve().parents[1] / "examples/memory_evidence/development_four_cells.json"
    previous = json.loads(evidence.read_text())
    weight = args.artifact_dir / "structured-attention-96/model.pt"
    if sha256_file(weight) != previous["checkpoint_sha256"]:
        raise ValueError("Wrong development checkpoint")
    model, cfg, device = load_model(weight, "cpu")
    original = json.loads((args.artifact_dir / "recovery-96.json").read_text())
    reference = {(r["seed"], r["condition"]): r for r in previous["case_auc_m_s"]}
    seen = {}
    for row in original["episodes"]:
        s, cond = row["seed"], row["condition"]
        if (s, cond) in seen or (s, cond) not in reference:
            raise ValueError("Unexpected or duplicate development case")
        run = evaluate_case(model, cfg, device, s, cond, row)
        computed = {cell: v["position_error_auc_m_s"] for cell, v in run["cells"].items()}
        expected = reference[s, cond]["auc_m_s"]
        if computed.keys() != expected.keys() or any(
            not np.isclose(v, expected[k], rtol=0, atol=0.0002) for k, v in computed.items()
        ):
            raise ValueError("Counterfactual does not reproduce independent development replay")
        seen[s, cond] = computed
    if seen.keys() != reference.keys():
        raise ValueError("Missing development case")
    with args.out.open("x") as f:
        json.dump({f"{s}/{c}": scores for (s, c), scores in seen.items()}, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
