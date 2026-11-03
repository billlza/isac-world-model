"""Paired width and event-memory contrasts for a complete two-by-two study."""

from __future__ import annotations

import numpy as np


def contrasts(cells):
    """Each cell maps (training seed, challenge trajectory seed) to a scalar error."""
    expected = {(a, h) for a in ("structured", "structured-attention") for h in (96, 192)}
    if set(cells) != expected:
        raise ValueError("Need both architectures at both widths")
    indices = set(cells[("structured", 96)])
    if not indices or any(set(cell) != indices for cell in cells.values()):
        raise ValueError("Missing or mismatched paired endpoints")
    seeds = sorted({s for s, _ in indices})
    episodes = sorted({e for _, e in indices})
    if len(seeds) < 2 or len(episodes) < 2 or len(indices) != len(seeds) * len(episodes):
        raise ValueError("Need a complete crossed model-seed by episode-seed panel")
    measures = {
        k: np.array([[cell[(s, e)] for e in episodes] for s in seeds], dtype=np.float64)
        for k, cell in cells.items()
    }
    if not all(np.isfinite(x).all() for x in measures.values()):
        raise ValueError("Non-finite paired endpoints")
    s0, s1 = measures["structured", 96], measures["structured", 192]
    a0, a1 = measures["structured-attention", 96], measures["structured-attention", 192]
    differences = {
        "attention_minus_structured_at_96": a0 - s0,
        "attention_minus_structured_at_192": a1 - s1,
        "wide_minus_narrow_structured": s1 - s0,
        "wide_minus_narrow_attention": a1 - a0,
        "interaction_width_effect_attention_minus_structured": (a1 - a0) - (s1 - s0),
    }
    # Shared columns represent the same target trajectory for every initialization.
    # Draw the training and trajectory indices separately, and pair all four cells.
    rng = np.random.default_rng(20261103)
    rows = rng.integers(len(seeds), size=(4000, len(seeds), 1))
    cols = rng.integers(len(episodes), size=(4000, 1, len(episodes)))
    result = {}
    for label, d in differences.items():
        b = d[rows, cols].mean(axis=(1, 2))
        result[label] = {
            "mean": float(d.mean()),
            "per_training_seed": dict(zip(map(str, seeds), d.mean(axis=1).tolist(), strict=True)),
            "negative_pairs": int((d < 0).sum()),
            "positive_pairs": int((d > 0).sum()),
            "ties": int((d == 0).sum()),
            "exploratory_crossed_bootstrap_95_percentile": np.quantile(b, [0.025, 0.975]).tolist(),
        }
    return {
        "training_seeds": seeds,
        "episode_seeds": episodes,
        "contrasts": result,
        "interpretation": "negative favors first named variant; exploratory interval with few training seeds",
    }


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    panels = {cond: {} for cond in ("control", "maneuver")}
    recovery_failures = {}
    initials = set()
    checkpoints = {}
    for s in (7, 11, 13):
        directory = args.array_dir / f"seed-{s}"
        run = json.loads((directory / "study.json").read_text())
        if run["status"] != "completed" or run["training_seed"] != s:
            raise ValueError("Missing or incomplete training seed")
        initials.add((run["git_commit"], run["slurm_array_id"]))
        for h in (96, 192):
            pairs = json.loads((directory / f"recovery-{h}.json").read_text())
            if pairs["status"] != "completed" or pairs["width"] != h:
                raise ValueError("Invalid completed pair")
            checkpoints[f"{s}-{h}"] = pairs["checkpoints_sha256"]
            seen = set()
            for row in pairs["episodes"]:
                e, cond = row["seed"], row["condition"]
                if (e, cond) in seen or cond not in panels or not 93621000 <= e <= 93621023:
                    raise ValueError("Duplicate or unregistered held-out case")
                seen.add((e, cond))
                for arch, metrics in row["scores"].items():
                    if arch not in ("structured", "structured-attention"):
                        continue
                    panels[cond].setdefault((arch, h), {})[s, e] = metrics["position_error_auc_m_s"]
                    k = f"{s}/{arch}/{h}/{cond}"
                    recovery_failures[k] = recovery_failures.get(k, 0) + int(metrics["not_recovered_by_1s"])
            if len(seen) != 48:
                raise ValueError("Incomplete grid of registered held-out cases")
    if len(initials) != 1:
        raise ValueError("Mixed revisions or array IDs")
    result = {"revision_and_array": list(initials.pop()), "primary_unit": "m*s",
              "per_condition": {c: contrasts(p) for c, p in panels.items()},
              "unrecovered_counts_out_of_24": recovery_failures, "checkpoint_digests": checkpoints}
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    for cond, report in result["per_condition"].items():
        print(cond, {k: round(x["mean"], 5) for k, x in report["contrasts"].items()})


if __name__ == "__main__":
    main()
