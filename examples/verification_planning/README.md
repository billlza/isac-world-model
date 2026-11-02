# Paired verification-planning evidence — 2026-11-02

This is a completed geometry-only CPU experiment comparing two **action rules**
with the same corrected evidence state machine. `legacy-voi` is not the entire
original October implementation. No neural model was trained for this comparison.

## Provenance and complete records

- Source commit: `8d7dd73af01659e16d3a571e4b381bc7d94bbe80`.
- [CI run](https://github.com/billlza/isac-world-model/actions/runs/41545039429).
- [Original Linux/Python 3.14.9 artifact](https://github.com/billlza/isac-world-model/actions/runs/41545039429/artifacts/13187702083).
- [Manifest](2026-11-02/manifest.json): configuration, runtime, source hashes and all eight seeds.
- [Summary](2026-11-02/summary.json): every scenario/planner aggregate and every paired episode delta.
- [Full study, gzip-compressed](2026-11-02/study.json.gz): all 96 rollouts, initial PHY outputs,
  observations, decisions, timestamps, weights, failures and resource counts.

The full uncompressed JSON SHA-256 is
`e27364eaec58154f3c8516691a22746409ef10e9df383e4cf651cec22734588f`.
The gzip SHA-256 is
`5b121367a610da3204b3206391224f92f0479fbe725b5f44819fbde92ae8d52e`.
The archive contains the original JSON bytes, with a deterministic gzip header.

```bash
python scripts/verification_study.py --out runs/verification-paired --episodes 8 --seed 20261103
```

Seeds were predeclared as 20261103–20261110; six scenarios and two planners yield
96 rollouts. Both planners receive the same initial RF evidence for each paired
world. Runtime was Python 3.14.9, NumPy 2.5.4 and PyTorch 2.14.1+cpu, with two CPU
threads. The measured 2.35 s study duration is specific to that CI run, not an
H200 training-time estimate.

## Observed trade-offs

All values below are over eight episodes per scenario and planner. Arrows mean
`legacy-voi` → `verification-aware`. Abstention includes candidate-set rejection;
explicit rejection is reported separately.

| Scenario | Final outcome | Mean captures | Mean travel | Mean reserved camera bits |
| --- | --- | ---: | ---: | ---: |
| same | Abstention 8/8 → 0/8 | 1.75 → 2.00 | 4 → 12 m | 1792 → 2048 |
| different | Abstention 3/8 → 0/8 | 2.50 → 2.125 | 4 → 12 m | 2560 → 2176 |
| unresolvable | Unknown 8/8 → 8/8 | 0 → 0 | 0 → 0 m | 0 → 0 |
| outside_candidates | Rejected 8/8 → 8/8 | 2 → 2 | 4 → 12 m | 2048 → 2048 |
| camera_loss | Unknown 8/8 → 8/8 | 4 → 4 | 4 → 4 m | 4096 → 4096 |
| late | Abstention 5/8 → 3/8 | 1.75 → 2.00 | 4 → 4 m | 1792 → 2048 |

Neither planner produced an observed wrong confirmation in this small sample.
The normal identifying scenarios show a useful completion improvement, while
the outside-candidate scenario exposes unnecessary extra movement without a
rejection-rate benefit. Persistent camera loss still consumes four attempted
measurements under both rules. These outcomes motivate further work on the
value and cost of verification; they do not establish an optimal scheduler.

The two planners share budget caps, not identical realized resource use.
Spatial separation does not make observations independent, and same-view
verification is explicitly recorded when a different feasible view is absent.
The test contains only eight underlying episode seeds reused across scenarios,
so 96 rollouts must not be described as 96 independent environments. Zero
observed errors do not establish safety. No recurrent-memory, real-data,
online-relearning or H200 result follows from this study.

## Engineering validation at this source commit

The Linux/Python 3.14.9 Python profile passed 397 tests with 91.37% coverage,
integration, distribution and installed-wheel checks. The first workflow run
reported a source-format failure (one dictionary-comprehension layout);
that layout is corrected in the subsequent commit. Use [PR #6 checks](https://github.com/billlza/isac-world-model/pull/6)
for the final whole-workflow status rather than calling the first run fully green.
