"""Matched CPU microbenchmarks. No accelerator or end-to-end timing claim."""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from isac_wm.artifacts import write_json
from isac_wm.baselines import ReferenceReplayKalman, ReplayKalman
from isac_wm.events import EventQueue
from isac_wm.model import EventBeliefModel, ModelConfig
from isac_wm.simulator import SceneConfig, ToyScene, recorded_episode
from isac_wm.training import batchify, runtime_info, seed_all


def observations(steps):
    c = SceneConfig(steps=steps, delay_jitter_s=0.18)
    s = ToyScene(940, c)
    q = EventQueue()
    rows = []
    for k in range(steps):
        now = k * c.dt
        rows.append((now, q.release(now)))
        for o in (s.camera(k), s.radio(k, 1)):
            if o is not None:
                q.push(o)
    return rows


def measure_filter(cls, rows):
    f = cls(10.0)
    out = []
    start = time.perf_counter()
    for now, ready in rows:
        f.ingest(ready, now)
        out.append(f.estimate(now))
    return time.perf_counter() - start, out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--out", type=Path, default=Path("runs/benchmark.json"))
    a.add_argument("--repeats", type=int, default=5)
    args = a.parse_args()
    if not 3 <= args.repeats <= 100:
        raise ValueError("repeats in [3,100]")
    torch.set_num_threads(2)
    seed_all(7)
    result = {
        "runtime": runtime_info(torch.device("cpu")),
        "scope": "matched CPU microbenchmarks; not full deployment speed",
        "filter": {},
    }
    for n in (48, 240):
        rows = observations(n)
        _, reference = measure_filter(ReferenceReplayKalman, rows)
        _, cached = measure_filter(ReplayKalman, rows)
        maxerr = max(
            float(np.max(np.abs(a - b)))
            for pair, pair2 in zip(reference, cached)
            for a, b in zip(pair, pair2)
        )
        timings = {"reference": [], "cached": []}
        for i in range(args.repeats):
            order = [("reference", ReferenceReplayKalman), ("cached", ReplayKalman)]
            for name, cls in order[:: (-1 if i % 2 else 1)]:
                timings[name].append(measure_filter(cls, rows)[0] * 1000)
        med = {k: statistics.median(v) for k, v in timings.items()}
        result["filter"][str(n)] = {
            "median_ms": med,
            "raw_ms": timings,
            "speedup": med["reference"] / med["cached"],
            "max_absolute_output_difference": maxerr,
        }
    c = SceneConfig(steps=48)
    batch = batchify([recorded_episode(i, c) for i in range(8)], torch.device("cpu"))
    m = EventBeliefModel(ModelConfig()).eval()
    with torch.inference_mode():
        reference = m(batch, batched_encoding=False)
        new = m(batch, batched_encoding=True)
        maxerr = max(float((reference[k] - new[k]).abs().max()) for k in reference)
        for _ in range(2):
            m(batch, batched_encoding=False)
            m(batch, batched_encoding=True)
        ts = {"sequential": [], "batched": []}
        for i in range(args.repeats):
            for name, flag in [("sequential", False), ("batched", True)][:: (-1 if i % 2 else 1)]:
                t = time.perf_counter()
                m(batch, batched_encoding=flag)
                ts[name].append((time.perf_counter() - t) * 1000)
    med = {k: statistics.median(v) for k, v in ts.items()}
    result["model"] = {
        "batch": 8,
        "steps": 48,
        "median_ms": med,
        "raw_ms": ts,
        "speedup": med["sequential"] / med["batched"],
        "max_absolute_output_difference": maxerr,
    }
    write_json(args.out, result)
    print(result)


if __name__ == "__main__":
    main()
