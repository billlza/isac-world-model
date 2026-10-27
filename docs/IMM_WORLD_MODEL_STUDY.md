# World model, IMM, and sensor-model mismatch: research protocol

## Distinct responsibilities

- **Sensing:** the camera and toy radio observation generators produce observations with capture and availability times. ISAC waveform-level estimation, hardware and full 3D SLAM are **not implemented**.
- **Classical state estimation:** `ReplayKalman` and the new `ReplayIMM` ingest **delivered** observations and maintain causal state distributions; IMM dynamically weights a low/medium/high process-noise model bank.
- **Our world model:** `StructuredEventModel` learns an environment state representation, a continuous-interval dynamics residual, state uncertainty and an action-conditioned **predicted observation**. An optional, strictly bounded event-memory multihead-attention module retrieves previously encoded observations. The learned model is trained **from random initialization**.
- **Scheduling:** the existing fixed / guarded / predictive heuristics choose whether to request sensing. The present `predictive` heuristic is still only a covariance proxy, **not** a learned, calibrated counterfactual value-of-information estimator.
- **Communication:** a stated AWGN time-sharing proxy, not waveform-level communication throughput or an actual ISAC capacity region.

## Why IMM is a meaningful stronger baseline

With a hidden Markovian motion mode, IMM has three sub-filters using acceleration-process-noise intensities `q=(0.05,0.8,6.0)`. Its mode transition over an elapsed time `dt` uses:

```
P_ij(dt) = (1 - exp(-lambda dt))/M + exp(-lambda dt) * [i == j]
```

The model-mixing step is followed by Gaussian predictive likelihood updates and exact moment-matched state/covariance combinations. NIS-gated IMM rejects an outlying position observation only if its innovation is incompatible with the **mixture prediction**. The gate threshold is the predeclared 99th percentile of the nominal 2D chi-square distribution; this is not a calibrated guarantee under model mismatch.

The bank varies **process noise**, not the exact target turn acceleration: this is a well-specified `IMM-CV` comparator, not a claim to implement every IMM/CT/CA variant. In this toy scene the simulator has piecewise random acceleration and reflection boundaries. A future matched turn-model IMM/UKF is another possible comparator, depending on the selected physical task.

Both IMM and KF fully reprocess **delivered** out-of-sequence observations by capture time, with prefix caching; the neural snapshot path may ignore older-after-newer packets. This is a real algorithmic asymmetry that must be reported, not hidden.

## Reproduce the study

All commands require a locked environment and a new output directory.

```bash
bash scripts/env.sh python scripts/compare_estimators.py --quick --seeds 7,11 --out runs/imm-smoke

bash scripts/env.sh python scripts/compare_estimators.py \
  --seeds 7,11,13 --episodes 96 --epochs 16 \
  --test-episodes 24 --validate-episodes 8 \
  --out runs/imm-study
```

The script selects KF acceleration spectral density and IMM switching rate using **validation-only** trajectories. Held-out test trajectory IDs are disjoint between training-seed groups; the scenario variants share each trajectory **within** a group for paired comparisons. Attention and no-attention structured models use identical training seeds and backbone dimensions but **not** identical parameter counts or execution costs. The comparison script reports exploratory episode-bootstrap intervals; use `scripts/study_statistics.py` on an independently verified finished run for a group-aware bootstrap over training seeds and trajectories. Neither interval is a risk certificate. The script preserves per-episode RMSE, p95 error, resource usage, CPU latency, model checkpoints, training time, source fingerprints and failures.

Schema version 2 also retains the complete evaluator `metrics`, `status_counts`, and `runtime` for every estimator and episode, even without `--keep-trace`. This includes late snapshot discards, queue overflow, observations still undelivered at termination, unfinished inference jobs, and measured-compute versus imposed-service deadline outcomes. The original short metric names remain for compatibility; a missing diagnostic in a version 1 record is unknown, not zero.

Existing calibration artifacts were fitted with KF process noise 0.8. Calibrated KF evaluation rejects a different `--kf-process-noise`; tuned KF comparisons remain available without calibration. IMM and gated variants require their own calibration support before calibrated claims can be made.

To evaluate a trusted, locally trained checkpoint with all classical alternatives:

```bash
bash scripts/env.sh isac-wm evaluate \
  --checkpoint runs/structured-7/model.pt \
  --estimators kalman,kalman-gated,imm,imm-gated,model \
  --policies fixed \
  --latency-mode fixed --compute-ms 150 \
  --scenario radio_outliers --episodes 24 \
  --out runs/strong-baselines
```

> `radio_bias` and `sparse_camera` add held-out sensor mismatch without changing the underlying target trajectory. A common tested scenario label does **not** make indoor rooms, vehicle routes and RF propagation geometrically independent. These remain synthetic orthographic scenes; hardware and new environment-family validation are separate milestones.

## Required interpretation

- Report paired trajectory differences rather than selecting an architecture using test data.
- A confidence interval overlapping zero **does not establish a robust attention benefit**; one favorable training seed is not enough.
- A smaller RMSE must be considered alongside computation, sensing airtime and latency. A gate can reject helpful observations; higher robustness is not automatically free.
- A well-tuned model-bank estimator can beat a train-from-scratch network; this is a valid, useful result.
- Do not claim that a trained world model is the sensor, that the system performs quantum ISAC, or that it gives calibrated guarantees.

Sources for the classic approach: Blom & Bar-Shalom (IEEE TAC 1988, doi:10.1109/9.1299); Bar-Shalom et al. (IEEE TAES 1989, doi:10.1109/7.18693). Modern robust IMM with model-mismatch/outlier evaluation is active research; our implementation does not assert novelty from the IMM itself.
