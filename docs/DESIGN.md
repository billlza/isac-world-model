# Design contract — v0.1.0

## Scientific scope

This release tests whether the complete engineering path can execute: physical
scene -> delayed visual/radio evidence -> learned state/forecast -> sensing action
-> new evidence, with a communications opportunity cost. It is the first runnable
model, not an established novelty or performance claim.

The long-term project may add Quantum ISAC after an appropriate adviser defines
a quantum state, measurement and resource model. This release contains **none**.

## Independently designed model organization

`EventBeliefModel` has 142,312 parameters under the default configuration.
All trainable tensors are initialized by PyTorch from the supplied random seed.

1. A small convolutional image encoder preserves a 4x4 spatial grid before a learned
   projection. A separate MLP encodes normalized radio measurements.
2. Separate GRU cells keep modality-specific memory. A stored payload is encoded
   but cannot trigger an update twice; `new` and `valid` flags control the update.
3. Latent state evolves with real delta-time: `h_prior = h + dt*f(h)`.
   This is a first-order learned transition, not an exact continuous-time solver.
4. A shared GRU performs evidence-driven correction using the modality memories,
   acquisition ages, valid/new masks and delta-time.
5. State and forecast heads output diagonal Gaussian parameters for normalized
   x,y,vx,vy. The forecast has no sensing-action input.
6. An action-conditioned head predicts radio measurements around a differentiable
   geometric mapping of the predicted state. Its residual is bounded. It is
   trained, but its uncertainty is NOT yet the scheduler's observation-noise model.

This is a proposed composition of familiar CNN/MLP/GRU mechanisms. Changing the
name or random initialization does not establish architectural novelty. No
pretrained backbone is silently used.

## Time and evidence contract

- `capture_s`: physical sampling instant (point-sample abstraction in this version).
- `available_s`: earliest instant at which the algorithm can use the observation.
- State estimates and labels refer to the current simulated decision time.
- A measurement requested at a decision cannot affect that same decision.
- Duplicate event IDs are rejected. Queue overflow raises an explicit exception.
- The neural snapshot buffer keeps the latest acquisition for each modality;
  late older packets are counted and ignored, not relabelled as fresh evidence.
- The Kalman baseline reprocesses all already-arrived observations in acquisition
  order and propagates the result to the current time. This is deliberately more
  capable in OOSM handling, but has growing replay cost. Fixed-lag smoothing and
  an efficient learned OOSM update are future work.
- The Rust queue uses integer nanoseconds; Python uses seconds. A native adapter
  with explicit rounding/parity tests is not connected in this version.

## State, sensor and resource model

The target state is `[x_m,y_m,vx_mps,vy_mps]`. The neural state targets divide by
`[room_m,room_m,2,2]`; metrics convert back to metres. The camera is synthetic,
calibrated, orthographic and sometimes occluded. It is not a real RGB perspective
camera. No camera geometry, depth or wireless channel is claimed inferred from
raw real observations in this release.

The processed RF observation is:

`[range/(sqrt(2)*room), cos(bearing), sin(bearing), radial_velocity/2]`.

Three actions are defined: skip, quick and precise. Their configured airtime,
noise and delay are explicit assumptions, **not empirical sensor performance**.
Changing an action does not change the target or the paired camera noise.

Communications opportunity-cost proxy:

`bits_k = (1-rho(action_k))*frame_duration*bandwidth*log2(1+SNR)`.

This assumes time sharing, fixed link SNR, no packet/PHY overhead and ideal AWGN
spectral efficiency. It is not measured throughput or a new ISAC capacity bound.
Shared resource models are motivated by joint sensing-communications design,
including dynamic time allocation; see the supplied survey in PROVENANCE.md.

## Learning objective

Supervised current-state Gaussian NLL + 0.5*three-step-future NLL +
0.15*action-conditioned observation NLL. All labels are separately named `*_target`.
The label branches cannot affect a forward state prediction. Gaussian NLL is
implemented without the constant term, so negative loss values are normal.

Simulation truth and future observations provide training supervision only. The
current implementation is not self-supervised internet-scale pretraining, and no
claim is made that a latent vector is an exact posterior belief distribution.

## Evaluation and claim boundaries

Train/validation/test split by entire generated episode seeds. The first five
startup frames are excluded from aggregate error metrics but retained in traces.
A seed list and runtime versions are stored. The preserved run is a small,
one-training-seed engineering experiment, not statistically strong research evidence.

The acceptance rule uses an uncalibrated Gaussian-radius heuristic. Report both
acceptance and error among accepted results; `null` means no accepted samples,
not zero risk. The policy uses a documented covariance-reduction proxy with
model forecasts, not a trained/optimal counterfactual value estimator.

The CPU runtime is recorded. It is NOT automatically fed back into simulation
clock time; a streaming execution-cost model remains a required next extension
before claims about end-to-end latency. Real datasets, multipath/ghost targets,
matched realized-budget Pareto sweeps, scene-distribution shifts and a tuned IMM
baseline are subsequent experiments, not silently considered complete.

## Next implementation increments

- Calibrate predicted uncertainty on a held-out calibration split.
- Generate paired counterfactual observations and train value-at-arrival estimates.
- Use a deadline/queue model so measured inference time affects availability.
- Add raw-signal / ray-tracing observation adapters and a measured camera stream.
- Integrate the native event queue after parity/FFI tests on the deployment machine.
- Scale model/data only after inspecting learning and compute profiles.
