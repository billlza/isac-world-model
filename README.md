# ISAC World Model · v0.1.0

**An independently authored, randomly initialized visual–radio research prototype.**

Status: executable first experiment, not a validated new architecture or deployed
ISAC/quantum system. No pretrained weights, teacher models, private datasets,
World-Mirror code, or restricted model outputs are used.

[中文说明](docs/README.zh-CN.md) · [Model and experiment contract](docs/DESIGN.md)
· [Validation record](docs/VALIDATION.md) · [Provenance](docs/PROVENANCE.md)

## What is implemented

- Independently implemented synthetic RGB and range/bearing/radial-velocity sensor.
- Acquisition time is distinct from availability time. Sensor outputs are released
  causally; sensing is requested only after a decision, not before it.
- Randomly initialized image and radio encoders, separate modality memories,
  time-conditioned latent dynamics, probabilistic state/forecast heads, and an
  action-conditioned observation prediction head.
- Environment dynamics are **not** conditioned on sensing actions. The sensor model
  and resource cost are: scanning does not make the target move.
- Training, safe checkpoint loading, held-out episode validation, four-way paired
  evaluation and a self-contained HTML trace viewer.
- A classical full-history causal replay Kalman baseline and an explicit shared
  airtime budget. The policy is currently a covariance-reduction **heuristic**, not
  a learned or theoretically certified counterfactual value-of-information policy.
- A dependency-free Rust event broker with Rust unit tests and a simple line
  protocol. The current Python experiment uses its Python reference queue. Native
  FFI/runtime integration is a separate next step, not claimed complete.

## Quick start

Python 3.11+ and a platform-appropriate PyTorch installation are required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
isac-wm doctor
isac-wm train --out runs/train --epochs 24 --episodes 192 --device auto
isac-wm evaluate --checkpoint runs/train/model.pt --out runs/eval --episodes 8
```

The delivered ZIP also includes `checkpoints/development.pt`, a small checkpoint
trained from random initialization in this session (ignored by Git). You may
pass it to `isac-wm evaluate --checkpoint checkpoints/development.pt --out runs/replay`.

Open `runs/eval/replay.html` locally. No web service or external model is contacted.
On Windows, activate the environment using `.venv\Scripts\activate`.

`--device auto` selects CUDA, then MPS, then CPU according to runtime availability.
Only CPU execution was tested in the authoring session. MPS and CUDA are supported
code paths, not measured performance claims. Failures do not silently fall back to
CPU when a specific accelerator was requested.

For the native event core, with a Rust toolchain installed:

```bash
cargo test --workspace
cargo build --release --workspace
printf 'P 1 0 100\nR 99\nR 100\n' | target/release/isac-event-replay
```

## What the experiment means

A 2-D target moves independently of the sensing schedule. A synthetic, calibrated
orthographic camera renders 24x24 RGB images; the radio sensor emits processed
range/bearing/radial-velocity features with action-dependent noise and delay.
There is an intentionally simple shared-frame communications opportunity-cost
model. It is **not** waveform-level sensing, a calibrated multipath channel, a
camera perspective reconstruction method, or a quantum measurement model.

Each evaluation uses the same independent scene seeds for:

1. replay Kalman + fixed quick sensing;
2. replay Kalman + predictive budget heuristic;
3. learned model + fixed quick sensing;
4. learned model + predictive budget heuristic.

All have the same upper airtime cap. Actual sensing expenditure may differ and is
reported. A Pareto comparison or matched realized-budget sweep is still required.
Model variances and the viewer's acceptance ellipse are **uncalibrated**. They do
not certify safety or establish error-probability guarantees.

Training labels include simulation truth and later acquired radio outputs. Those
labels are not estimator inputs. Tests verify that changing future inputs,
labels, or future actions cannot change earlier state predictions.

Reported compute latency is measured on the executing machine. It is reported
separately and **does not yet advance the simulator clock**. Sensor delays are
simulated. Thus these are not complete streaming-perception latency results.

## Repository layout

```text
src/isac_wm/       model, training, simulator, baselines, events, evaluation, viewer
crates/event-core/ independently implemented native event queue
scripts/          smoke check and private GitHub publisher
examples/         preserved authoring-session measurements, without cherry-picking
tests/            causality, timing, budget, gradients and regression tests
docs/             design contract, scope, provenance and validation
```

## Publishing to GitHub

The publisher is opt-in and targets a **new private repository**, not an existing
project. It requires GitHub CLI (`gh`) authenticated as `billlza`. It refuses to
modify an existing remote repository or reuse an existing `origin`.

```bash
bash scripts/publish_private.sh billlza/isac-world-model
```

It does not ask for, print or store a token. The script uses your local GitHub CLI
login. No remote push is claimed until this command completes and the commit is
visible on GitHub. Use a different name only by passing it explicitly.

## License / research ownership

No public reuse license is selected for this initial private research artifact.
Original source was generated specifically for this project; dependencies retain
their own licenses. Agree ownership/publication arrangements before changing the
repository to public or choosing a redistribution license. See `RIGHTS.md`.
