# Validation record — 23 October 2026

## Completed in this authoring session

- Python bytecode compilation, editable package installation, CLI execution and Bash syntax checks.
- The generated HTML viewer was rendered in Chromium; changing the run and frame slider worked with zero JavaScript errors.
- **40 pytest cases passed** on Linux x86-64, Python 3.13.5,
  PyTorch 2.10.0+cpu, NumPy 2.3.5.
- Tests cover causal input release, duplicate evidence, queue overflow, nonfinite
  timestamps, deterministic paired simulation, budget conservation, OOS replay,
  no leakage from future inputs or supervision labels, differentiable training,
  small-batch learning, safe checkpoint round trip and the full four-variant run.
- A 142,312-parameter model was trained from random initialization on 192 generated
  episodes for 24 epochs. Entire validation episodes were held out. Best validation
  position RMSE: approximately **0.3488 m**. CPU training took approximately 102 s
  in this environment; it is not an Apple/NVIDIA comparison.
- Eight additional, independent test episodes were evaluated, with identical seeds
  across four estimator/policy combinations. The evaluator preserves startup,
  failure, uncertainty, resource, queue and timing records.

## Preserved first experiment: no superiority claim

| Variant | Mean episode position RMSE (m) | Mean episode p95 (m) | Mean sensing fraction |
|---|---:|---:|---:|
| Replay Kalman + fixed | 0.22873 | 0.43499 | 0.1000 |
| Replay Kalman + predictive heuristic | 0.22873 | 0.43499 | 0.1000 |
| Random-init trained model + fixed | 0.36339 | 0.58406 | 0.1000 |
| Random-init trained model + predictive heuristic | 0.36339 | 0.58406 | 0.1000 |

The learned model has **not beaten the classical baseline** in this test. Both
policies selected the same actions in these runs; no scheduling benefit was
observed. The learned uncertainty head rejected all results under the deliberately
uncalibrated acceptance rule; its conditional accepted-error metric is `null`,
not zero. These outcomes are stored rather than omitted.

The two initial training attempts are both preserved in `examples/`: an 8-epoch
smoke run and the 24-epoch development run. More epochs/data were added before the
independent test, not selected to obtain a winning test result.

## Not completed

- **GitHub push**: the connected GitHub API exposed read operations only. No remote
  repository, branch, issue or pull request was created or changed.
- **Rust compilation/tests**: no Rust compiler was installed in the execution
  environment. An attempt to download the official installer failed due to DNS/
  network access. Seven Rust tests and the CI job are supplied but NOT reported
  as passed. The Python reference event queue is tested; native integration is not.
- **MPS/CUDA runs**: code paths exist, but no Apple or NVIDIA device was available.
- **Scientific evaluation**: no multi-training-seed study, calibration guarantee,
  real sensor experiment, waveform-level ISAC, quantum model, novelty proof or
  performance-leading claim.
- Complete streaming latency: measured compute time is not injected into the
  simulator clock yet; do not interpret compute-only timing as a deadline guarantee.

## Meaning of this milestone

The from-scratch training/evaluation chain, evidence timing and test harness are
working. The preserved weaknesses directly define the next research tasks:
uncertainty calibration, richer observation/motion models, learned value-at-arrival,
stronger tuned baselines and measured streaming resource effects.
