# IMM and learned world-model architecture comparison — 2026-10-27

## Model responsibilities

- **Sensing** obtains optical/radio observations. The synthetic radio measurement proxy is not waveform-level ISAC.
- **World model** encodes and assimilates delivered observations, maintains temporal state, forecasts motion and can predict an action-conditioned hypothetical observation. It is *not* the physical sensor.
- **Decision/scheduler** selects whether/how to sense under the time-sharing proxy; this study uses the same **fixed** policy for all estimators, so it isolates *tracking*, not learned sensing policy benefits.
- **Classic estimators**: tuned constant-velocity KF, its innovation-gated variant, and IMM switching among three acceleration *noise intensities* (not a CT-turn IMM).

## Reproducible study protocol

- Original study source: `c37ae5a430a6b3017233b1998cbc0f264edaf4a7`; complete run **passed**, with all 21 recorded source fingerprints matching that commit. Runtime: Linux CPU, Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5.
- 3 independently initialized training seeds [7, 11, 13]; 96 synthetic training episodes and 16 epochs each. Test groups use disjoint episode seeds.
- 8 scenario/latency conditions × 3 groups × 24 trajectories × 6 estimators = **3,456 estimator–episode evaluations**.
- Same scenario, seed, sensing budget and *imposed simulation service time* within a group; model parameter count, actual compute and training costs differ and must not be suppressed.
- Validation-only baseline tuning; no hyperparameter changes based on test outcomes. Scene geometry is still the same synthetic 10 m room family.
- Sources of model inputs are observations available by the decision time. Ground truth is accessible only to simulation/training and post-hoc evaluator.

The integration branch also includes the remote `a3394f7` history, including the browser replay fix and isolated toolchain setup, plus formatting required by the pinned Ruff version. These changes alter source fingerprints. The numerical results below remain evidence for the original study revision, not a rerun under the repository's locked runtime. Preserve the original reports and checkpoints; use the original commit when checking their source hashes, or generate a new study for the integrated revision.

## Mean per-trajectory RMSE (m; lower is better)

| Scenario | KF | IMM | Gated KF | Gated IMM | Structured | Structured + attention |
|---|---:|---:|---:|---:|---:|---:|
| nominal / 0 ms | 0.233 | 0.232 | 0.239 | 0.237 | 0.297 | 0.266 |
| nominal / 150 ms | 0.423 | 0.416 | 0.436 | 0.419 | 0.659 | 0.547 |
| maneuver / 150 ms | 0.734 | 0.723 | 0.901 | 0.731 | 0.891 | 0.809 |
| radio_outliers / 150 ms | 1.432 | 2.106 | 0.473 | 0.438 | 1.672 | 1.219 |
| radio_bias / 150 ms | 0.724 | 0.750 | 0.924 | 0.806 | 0.990 | 0.762 |
| sparse_camera / 150 ms | 0.740 | 0.752 | 0.764 | 0.772 | 0.771 | 0.754 |
| camera_loss / 150 ms | 0.782 | 0.800 | 0.830 | 0.812 | 0.780 | 0.774 |
| blackout / 150 ms | 3.238 | 3.238 | 3.238 | 3.238 | 3.405 | 3.220 |

## Attention-specific ablation

A negative **attention − structured** difference means the attention variant has lower RMSE. Only **three independent training-seed groups** were run, so the hierarchical resampling ranges remain exploratory; they are **not** proof of a universal performance improvement.

| Scenario | Mean ΔRMSE (m) | Hierarchical resampling 95% range (m) |
|---|---:|---:|
| nominal / 0 ms | -0.031 | [-0.091, +0.008] |
| nominal / 150 ms | -0.112 | [-0.311, +0.012] |
| maneuver / 150 ms | -0.081 | [-0.197, +0.002] |
| radio_outliers / 150 ms | -0.453 | [-1.203, +0.026] |
| radio_bias / 150 ms | -0.229 | [-0.584, -0.034] |
| sparse_camera / 150 ms | -0.018 | [-0.050, +0.012] |
| camera_loss / 150 ms | -0.006 | [-0.028, +0.020] |
| blackout / 150 ms | -0.184 | [-0.380, -0.000] |

## Decision on candidate technology (research recommendation)

1. **Keep IMM and gated IMM as mandatory classical comparators.** Gating performs extremely well under isolated radio outliers; in nominal and maneuvering settings, appropriately tuned classical estimators are often better.
2. **Keep structured physics + learned residual as the primary *neural research family*.** This provides physical state semantics, a learnable residual, and a bounded event memory for async sensing.
3. **Keep bounded multihead attention optional, not universally enabled.** It often improves the matched structured model but adds model parameters and compute. With three training seeds and non-matched capacities, the increment is not yet robustly established across all cases.
4. Next distinct contribution candidate: **decision-sufficient predictive state and value of a future delivered observation**, rather than merely adding a Transformer to the tracker. The current `predictive` scheduler remains a covariance heuristic, not a learned counterfactual VoI policy.

## Critical scope boundaries

- Stronger baselines are not exhaustive: no independently tuned CA/CT-mode IMM, robust M-estimators, or complete Bayesian smoother.
- Synthetic orthographic camera and processed polar radio measurements, with no RF waveform simulator, real sensor noise characterization, geometric SLAM, or quantum measurement.
- Delayed evidence is replayed by classic estimators; the neural snapshot implementation handles delayed and OOS evidence differently. This difference can influence the results.
- The test scenes share a room generator and are not independent environment families; more training seeds, distinct layouts and real recordings are needed for transportability.
- The attention architecture differs in parameter count; parameter/compute matching is a future **incremental-benefit** test, not a newly imposed requirement for having a working research prototype.
- No calibrated risk guarantee or certification. Uncertainty scores require separate calibration and stress checks.

## Run again

```bash
bash scripts/env.sh python scripts/compare_estimators.py --quick --seeds 7,11 --out runs/imm-smoke
bash scripts/env.sh python scripts/compare_estimators.py --seeds 7,11,13 --episodes 96 --epochs 16 --test-episodes 24 --validate-episodes 8 --out runs/imm-study
bash scripts/env.sh python scripts/study_statistics.py --study runs/imm-study/comparison.json --out runs/imm-study-statistics.json
```

Do not copy the three-seed numerical conclusions to new datasets or hardware without a new evaluation.
