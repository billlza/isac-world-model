# Portable active-observation record — 2026-10-28

Open `START_MATLAB.m` from the repository root. This directory contains the first
predeclared held-out seed for **every** scenario and method, together with the
aggregate outcomes of all test episodes. It is recorded Python experiment data;
actual MATLAB execution is still pending on the new Mac.

Generation command:

```bash
bash scripts/env.sh python scripts/active_observation_study.py \
  --out runs/active-observation-20261028 \
  --train-episodes 64 --validation-episodes 16 --test-episodes 4 \
  --epochs 12 --seed 271828
```

There are 64 training episodes, 16 validation episodes, and 4 held-out episode
seeds per scenario. Six scenarios × six methods × four seeds yield 144 evaluated
rollouts. Each model starts randomly; weights are selected on validation NLL.
The JSON includes exact source/data/checkpoint hashes, seed splits, runtime and
initial actual OFDM reports. Full local records and small checkpoints remain in
the generated run directory; the MATLAB replay needs neither checkpoint.

## Observed outcomes, including failures

This tiny test is a mechanism/smoke study, **not a superiority result**. The
recurrent model's selected validation NLL was 1.3539; the memoryless model's was
1.3056 on the same validation episodes (lower is better). This run does **not**
establish a benefit from recurrent memory.

| Scenario | Learned-memory result | Fixed-measurement result |
| --- | --- | --- |
| same | 75% final abstention | 0% final abstention |
| different | 50% final abstention | 0% final abstention |
| unresolvable | 100% unknown; no captures | 100% unknown; four captures |
| outside_candidates | 100% unknown; no explicit candidate-set rejection | 100% candidate-set rejection |
| camera_loss | 100% unknown; missing detections retained | 100% unknown; missing detections retained |
| late | 100% final abstention | 25% final abstention |

The two-candidate VoI proxy can stop after one strongly discriminative or
contradictory sample, before the confirmation/rejection gate has enough evidence.
That mismatch is a recorded open research problem, not a reason to lower the
gate or remove failed runs. All methods had zero *observed* wrong confirmations
in these four test seeds; high abstention and tiny sample size prevent a safety
interpretation.

Fixed/random methods typically spend more resources. Their final error and the
adaptive methods' NLL are not comparisons at identical realized measurements.
Inspect achieved transport/travel costs as well as the shared budget caps.
No claim is made about real radio goodput, material identification, multi-object
tracking, or a fully trained RGB/RF world model.

See [the full MATLAB guide](../../docs/MATLAB_ACTIVE_OBSERVATION.zh-CN.md) for
native checks, UI validation and reproduction on the new Mac.
