"""Frozen, paired camera/old-key interventions on the archived Hamilton round-two checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from recovery_challenge_core import END, WINDOW, observed_trace, score

from isac_wm.events import SnapshotBuffer

SOURCE = "b2547848b09b8c8cb3e7a03ce0154595465006d6"
TEST = range(93621000, 93621024)
TOLERANCE_M = 0.0002
# Original immutable recovery files in the independently downloaded array-19341792 archive.
REFERENCE_DIGEST = {
    (7, 96): "5b9268a3d040f6565159d3585f95a63e12e24a63a30b64648d27ab9739bb62c6",
    (7, 192): "3a7d75dfdf3eb1588b6a66c820629dc879a06eeda114d96940bb86f5ee6ee774",
    (11, 96): "ca427c1361e56ed0ede9cd4fc5f58f7ccbd2cc305513282672d6727d43693baa",
    (11, 192): "5eb7137aee95921fb642541b87b1e0c76d92a49b17f88e1e593a2ce381ccf8d9",
    (13, 96): "58f0d3d44c5f320dfb0bffe68dd9c0acc514e6c51030fa9a227d225ee3d8b0e1",
    (13, 192): "28c065626c97a8333009ff93cb4de73e2772a93e99478445c80c82947947b97c",
}


def reference(directory, seed, width):
    raw = (directory / f"recovery-{width}.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_DIGEST[seed, width]:
        raise ValueError("Recovery reference differs from the archived 19341792 bytes")
    ref = json.loads(raw)
    rows = {(x["seed"], x["condition"]): x for x in ref["episodes"]}
    grid = {(s, c) for s in TEST for c in ("control", "maneuver")}
    if ref["status"] != "completed" or set(rows) != grid or len(ref["episodes"]) != len(grid):
        raise ValueError("Incomplete reference grid")
    return ref["checkpoints_sha256"], rows


def counterfactual_snapshots(cfg, deliveries, factual, anchor):
    """Withhold only delivered vision captured after the camera reopened; preserve RF packets and age."""
    buffer = SnapshotBuffer(cfg.image_size)
    muted, removed = [], []
    for k, ready in enumerate(deliveries):
        blocked = [o for o in ready if o.modality == "vision" and o.capture_s >= END * cfg.dt]
        keep = [o for o in ready if not (o.modality == "vision" and o.capture_s >= END * cfg.dt)]
        if blocked:
            removed.append((k, len(blocked)))
        snap = buffer.update(keep, k * cfg.dt, cfg.dt)
        muted.append(snap)
        if k < anchor and any(not np.array_equal(v, snap[key]) for key, v in factual[k].items()):
            raise ValueError("Camera intervention affected pre-delivery information")
        if any(not np.array_equal(snap[key][idx], factual[k][key][idx]) for key, idx in (
            ("radio", slice(None)), ("new", 1), ("valid", 1), ("age", 1)
        )):
            raise ValueError("Camera intervention changed the RF information set")
    if not removed or removed[0][0] != anchor or anchor + WINDOW >= len(muted):
        raise ValueError("Incorrect paired intervention anchor/window")
    return muted, sum(n for _, n in removed)


def anchor_memory(memory, *, mask_old):
    """The used mask is the boolean validity of the explicit event ring, not the recurrent state.

    At the tick BEFORE anchor processing every occupied slot predates delivery by construction.
    Clear those validity bits once; the normal model appends every new token, preserves its null
    token and thereafter evicts normally. Do not clear GRU, vision/RF embeddings or kinematics.
    """
    result = tuple(x.clone() for x in memory)
    if mask_old:
        used = result[7]
        if len(result) != 9 or used.ndim != 2 or str(used.dtype) != "torch.bool":
            raise ValueError("Unexpected structured memory schema")
        used.zero_()
    return result


def forward_part(model, cfg, device, memory, snaps):
    import torch

    from isac_wm.training import INPUT_KEYS

    estimates, predictions = [], []
    for snap in snaps:
        batch = {k: torch.from_numpy(snap[k])[None].to(device) for k in INPUT_KEYS}
        current, memory = model.step(batch, memory)
        future = model.forecast(current["hidden"], 3 * cfg.dt)
        estimates.append(current["mean"][0, :2])
        predictions.append(future["mean"][0, :2])
    return memory, torch.stack(estimates), torch.stack(predictions)


def factual_replay_check(scored, ref):
    """Fail closed: test individual factual time points, not only an aggregate error."""
    a = np.asarray(scored["post_errors_m"])
    b = np.asarray(ref["post_errors_m"])
    if a.shape != (11,) or b.shape != (11,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid factual error series")
    deltas = [float(np.max(np.abs(a-b)))]
    for k in ("position_error_auc_m_s", "blackout_rmse_m", "three_step_forecast_rmse_m"):
        deltas.append(abs(scored[k] - ref[k]))
    if not np.isfinite(deltas).all() or max(deltas) > TOLERANCE_M:
        raise ValueError(f"Factual scores do not reproduce original reference: {max(deltas)}")
    if any(scored[k] != ref[k] for k in ("not_recovered_by_1s", "recovery_confirmed_s")):
        raise ValueError("Factual threshold outcome differs from original reference")
    return deltas[0]


def evaluate_case(model, cfg, device, seed, cond, ref):
    import torch

    snaps, ready, truth, meta = observed_trace(seed, cfg, cond == "maneuver")
    if meta != ref["observations"]:
        raise ValueError("Observations differ from paired reference")
    anchor = meta["first_post_blackout_camera_delivery_tick"]
    muted, removed_count = counterfactual_snapshots(cfg, ready, snaps, anchor)
    arch = model.config.architecture
    intervene_keys = arch == "structured-attention" and model.config.hidden_dim == 96
    cells = {}
    with torch.inference_mode():
        pre, pre_xy, pre_future = forward_part(model, cfg, device, model.initial(1, device), snaps[:anchor])
        prior_slots = int(pre[7].sum().item())
        if intervene_keys and not prior_slots:
            raise ValueError("Empty preanchor A96 bank: the intended old-key treatment is unavailable")
        for camera, stream in (("on", snaps), ("off", muted)):
            for keys in (("on", "off") if intervene_keys else ("on",)):
                memory = anchor_memory(pre, mask_old=keys == "off")
                _, xy, forecast = forward_part(model, cfg, device, memory, stream[anchor:])
                xy = torch.cat((pre_xy, xy)).cpu().numpy() * cfg.room_m
                forecast = torch.cat((pre_future, forecast)).cpu().numpy() * cfg.room_m
                result = score(xy, truth[:, :2], anchor)
                result["prearrival_error_m"] = float(np.linalg.norm(xy[anchor-1] - truth[anchor-1, :2]))
                e = forecast[5:-3] - truth[8:, :2]
                result["three_step_forecast_rmse_m"] = float(np.sqrt(np.mean(np.sum(e**2, axis=1))))
                cells[f"camera_{camera}_oldkeys_{keys}"] = result
    dev = factual_replay_check(cells["camera_on_oldkeys_on"], ref["scores"][arch])
    return {
        "seed": seed, "condition": cond, "observations": meta, "camera_packets_withheld": removed_count,
        "preanchor_occupied_event_slots": prior_slots, "factual_curve_max_abs_deviation_m": dev, "cells": cells,
    }


def summarize(rows):
    """Paired per-seed means; positive gain means that providing information reduced error."""
    report = {}
    for cond in ("control", "maneuver"):
        selected = [r["cells"] for r in rows if r["condition"] == cond]
        if not selected:
            raise ValueError(f"No {cond} cases")
        on = "camera_on_oldkeys_on"
        off = "camera_off_oldkeys_on"
        cuts = [(off, on, "camera_benefit_with_normal_memory")]
        if "camera_on_oldkeys_off" in selected[0]:
            cuts += [
                ("camera_off_oldkeys_off", "camera_on_oldkeys_off", "camera_benefit_without_preanchor_keys"),
                ("camera_on_oldkeys_off", on, "preanchor_key_benefit_camera_on"),
                ("camera_off_oldkeys_off", off, "preanchor_key_benefit_camera_off"),
            ]
        gains = {
            name: np.array([r[a]["position_error_auc_m_s"]-r[b]["position_error_auc_m_s"] for r in selected])
            for a, b, name in cuts
        }
        if len(cuts) > 1:
            gains["camera_benefit_normal_minus_masked_preanchor_keys"] = (
                gains["camera_benefit_with_normal_memory"] - gains["camera_benefit_without_preanchor_keys"]
            )
        report[cond] = {
            "paired_auc_effects_m_s": {
                name: {
                    "mean": float(d.mean()), "positive": int((d>0).sum()),
                    "negative": int((d<0).sum()), "ties": int((d==0).sum()),
                }
                for name, d in gains.items()
            },
            "by_cell": {
                cell: {
                    "n": len(selected),
                    "mean_auc_m_s": float(np.mean([r[cell]["position_error_auc_m_s"] for r in selected])),
                    "confirmed_by_1s": sum(not r[cell]["not_recovered_by_1s"] for r in selected),
                    "confirmed_then_outside_at_1s": sum(
                        not r[cell]["not_recovered_by_1s"] and r[cell]["post_errors_m"][-1]>0.25
                        for r in selected
                    ),
                    "mean_prearrival_error_m": float(np.mean([r[cell]["prearrival_error_m"] for r in selected])),
                    "mean_post_errors_m": np.mean([r[cell]["post_errors_m"] for r in selected],axis=0).tolist(),
                }
                for cell in selected[0]
            },
        }
    return report


def run(array_dir, out, device_name, seeds):
    import torch

    from isac_wm.artifacts import model_fingerprint, sha256_file, write_json
    from isac_wm.training import load_model, runtime_info

    if not seeds or len(set(seeds)) != len(seeds) or set(seeds)-{7,11,13} or device_name not in ("cpu","cuda"):
        raise ValueError("Use distinct archived seeds and an explicit device")
    torch.set_num_threads(2)
    commit = subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=Path(__file__).resolve().parents[1],text=True
    ).strip()
    report = {
        "schema_version": 1, "status": "running", "source_commit": SOURCE, "analysis_commit": commit,
        "original_array_id": "19341792", "requested_device": device_name, "selected_training_seeds": seeds,
        "analysis_file_sha256": sha256_file(Path(__file__)), "python": platform.python_version(),
        "numpy": np.__version__, "torch": str(torch.__version__), "tolerance_m": TOLERANCE_M,
        "scope": "Paired diagnostic on previously analyzed holdouts; no new training, no independent generalization claim.",
        "oldkeys_off": "Only A96: invalidate preanchor event-bank validity once immediately before first recovered frame delivery. Recurrent state and all postanchor writes remain intact.",
        "camera_off": "Withhold at delivery every vision packet captured on/after reopening. RF, actions, truth, and reference scoring anchor unchanged.",
        "models": [],
    }
    with out.open("x", encoding="utf-8") as handle:
        json.dump(report,handle,allow_nan=False)
    try:
        for s in seeds:
            directory = array_dir / f"seed-{s}"
            study = json.loads((directory / "study.json").read_bytes())
            if (study["status"],study["git_commit"],study["training_seed"],study["slurm_array_id"]) != (
                "completed", SOURCE,s,"19341792",
            ):
                raise ValueError("Incorrect training provenance")
            expected_train = {s*100_000+i for i in range(96)}
            expected_val = {10_000_000+s*100_000+i for i in range(12)}
            for h in (96,192):
                hashes, reference_rows = reference(directory,s,h)
                for a in ("structured","structured-attention"):
                    path = directory / f"{a}-{h}" / "model.pt"
                    if sha256_file(path) != hashes[a]:
                        raise ValueError(f"Original checkpoint hash mismatch: {path}")
                    model, cfg, device = load_model(path,device_name)
                    if (model.config.architecture != a or model.config.hidden_dim != h or cfg.steps != 64
                            or model.provenance_seeds != expected_train|expected_val
                            or model.provenance_seeds.intersection(TEST)):
                        raise ValueError("Unexpected checkpoint architecture or training/evaluation split")
                    weights = model_fingerprint(model)
                    entry = {
                        "training_seed": s, "architecture": a, "width": h, "status":"running",
                        "checkpoint_sha256":hashes[a], "reference_sha256":REFERENCE_DIGEST[s,h],
                        "loaded_parameter_sha256":weights, "runtime":runtime_info(device), "episodes":[],
                    }
                    report["models"].append(entry)
                    write_json(out,report)
                    start=time.monotonic()
                    for (episode_seed,cond),row in reference_rows.items():
                        entry["episodes"].append(evaluate_case(model,cfg,device,episode_seed,cond,row))
                    if model_fingerprint(model)!=weights or sha256_file(path)!=hashes[a]:
                        raise RuntimeError("Frozen model or original checkpoint changed")
                    entry.update(status="completed",wall_seconds=time.monotonic()-start,summary=summarize(entry["episodes"]))
                    write_json(out,report)
                    print(f"completed frozen replay seed={s}, architecture={a}, width={h}",flush=True)
        report["status"]="completed"
    except BaseException as exc:
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        if report["models"] and report["models"][-1]["status"]=="running":
            report["models"][-1]["status"]="failed"
        raise
    finally:
        write_json(out,report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array-dir",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--device",required=True,choices=("cpu","cuda"))
    parser.add_argument("--seeds",nargs="+",type=int,choices=(7,11,13),default=[7,11,13])
    args=parser.parse_args()
    run(args.array_dir, args.out, args.device, args.seeds)


if __name__ == "__main__":
    main()
