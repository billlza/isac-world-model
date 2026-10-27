# Arrival-time Value of Information (VoI) — research baseline

Status: **implemented and locally exercised, not a validated superiority claim.**

## What this adds

`src/isac_wm/voi.py` introduces a **transparent, non-learned** lookahead
scheduler (`voi`) in addition to `fixed`, `guarded`, and the new **always-skip
control** (`skip`). It combines:

- Present-state covariance (classical or learned, if valid and PSD), with a
  continuous-time white-acceleration *assumption* and explicit action-specific
  range/bearing measurement uncertainty;
- Acquisition only after the currently executing job completes, then a
  radio-result delay rounded up to simulation ticks;
- A common future horizon across all actions, assigning no benefit to a radio
  result before it is actually available;
- An explicit cost in communication bits from the *toy* AWGN time-sharing
  proxy, with the conversion factor **m² of average position MSE per lost Mbit**;
- A token-bucket cap. No sensing request can spend unavailable airtime.

For a candidate action `a`, after propagating the current covariance `P` to
its future acquisition time, let `P_a` denote the hypothetical Gaussian
posterior covariance conditional on taking that action's measurement. A
noise-model covariance reduction is scored at each common future tick `k`:

```
benefit(a) = (1 / H) * sum_{k=1}^H
             1[measurement_a has become available by k] *
             tr( (F_k (P - P_a) F_k^T)_{position,position} )

score(a) = benefit(a) - lambda * lost_Mbit(a)
```

Here `F_k` propagates from acquisition to prediction time, and the posterior
covariance update uses the action's geometric range/bearing noise matrix and
Joseph form. `skip` has score zero. The simulator still determines whether the
measurement is dropped or an outlier; **the selector does not read future
truth or hypothetical realized packets**. All candidate gains, net scores,
delivery ticks and lost communication bits can be inspected in the event trace.

**Limits:** This calculation neglects future camera observations and
sensor-evidence correlations, treats uncertain receiver/measurement models as
Gaussian, and assigns no extra processing queue after radio availability.
Time-varying biases, false observations and miscalibrated neural covariances
can invalidate the proxy. It is not a learned model, an information-theoretic
bound, an ISAC waveform implementation, or a certified safety system.

## Runnable examples

After project dependencies are installed in the isolated environment:

```bash
# Paired, no-checkpoint KF / IMM experiment, fixed 150 ms *assumed* compute time
bash scripts/env.sh python scripts/voi_study.py \
  --out runs/voi-study-1 --episodes 12 --steps 48 --voi-tradeoff 4

# CLI accepts every one of these explicitly priced candidate policies
bash scripts/env.sh isac-wm evaluate \
  --checkpoint /path/to/your/OWN/model.pt \
  --policies fixed,skip,guarded,voi \
  --estimators kalman,imm-gated,model \
  --voi-horizon 0.8 --voi-tradeoff 4 \
  --latency-mode fixed --compute-ms 150 \
  --out runs/voi-eval-1
```

Vary `--voi-tradeoff` on a separate **validation** set before evaluating on
new test seeds. Fixed, skip, guarded and VoI must all be compared. Do **not**
choose a winning weight retrospectively on final test traces.

Calibration stores the VoI configuration and rejects reuse with a different
tradeoff or horizon. The simulator includes no real OFDM, beamforming, MIMO
radio hardware, QKD, or measurement-validated latency; the results are only for
its synthetic orthographic/processed-feature environment.

## Current design decision

The original structured/attention model remains opt-in. `voi` is a **decision
baseline**, not a claimed learned contribution. Next research direction: learn
and validate an action-conditioned observation model and the expected *task*
risk reduction at the result's real availability time, using held-out scenes
and validation-only hyperparameter selection. Require a stronger maneuver IMM
and a calibrated RF signal/channel model before claiming ISAC superiority.
