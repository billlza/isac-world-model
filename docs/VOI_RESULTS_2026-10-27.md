# Arrival-time VoI — controlled paired comparison (2026-10-27)

**Evidence status:** 24 independent test trajectories per configuration, 48 time steps, fixed 150 ms hypothetical service time. The same seeds are paired across policies. These are synthetic processed RF features and an AWGN airtime proxy, not raw waveform ISAC or measured hardware.

Parameter `lambda=4 m² per lost Mbit` is an illustrative *predeclared comparison point*, not optimized on these final seeds. Fixed and VoI may consume unequal airtime.

| Scenario/estimator | Fixed RMSE / airtime | Guarded | Skip | VoI |
|---|---|---|---|---|
| nominal/kalman | 0.378 m / 0.048 | 0.399 m / 0.025 | 0.413 m / 0.000 | 0.378 m / 0.048 |
| nominal/imm-gated | 0.372 m / 0.048 | 0.393 m / 0.026 | 0.425 m / 0.000 | 0.371 m / 0.043 |
| maneuver/kalman | 0.727 m / 0.048 | 0.863 m / 0.026 | 1.630 m / 0.000 | 0.727 m / 0.048 |
| maneuver/imm-gated | 0.772 m / 0.048 | 0.845 m / 0.031 | 1.604 m / 0.000 | 0.772 m / 0.050 |
| radio_outliers/kalman | 1.396 m / 0.048 | 1.124 m / 0.025 | 0.413 m / 0.000 | 1.396 m / 0.048 |
| radio_outliers/imm-gated | 0.373 m / 0.048 | 0.388 m / 0.027 | 0.425 m / 0.000 | 0.374 m / 0.043 |
| radio_bias/kalman | 0.706 m / 0.048 | 0.657 m / 0.025 | 0.413 m / 0.000 | 0.706 m / 0.048 |
| radio_bias/imm-gated | 0.770 m / 0.048 | 0.764 m / 0.029 | 0.425 m / 0.000 | 0.825 m / 0.049 |
| camera_loss/kalman | 0.722 m / 0.048 | 0.853 m / 0.026 | 2.836 m / 0.000 | 0.722 m / 0.048 |
| camera_loss/imm-gated | 0.769 m / 0.048 | 0.818 m / 0.031 | 2.836 m / 0.000 | 0.769 m / 0.048 |

### Interpretation

- For nominal/IMM-gated, VoI is ~0.371 m vs fixed ~0.372 m, at ~0.043 vs 0.048 sensing fraction. This difference alone cannot establish robust improvement.
- At persistent RF bias/IMM-gated, the VoI outcome is worse than fixed or guarded, even with comparable airtime. The Gaussian/no-bias assumption fails.
- Under camera loss, always-skip deteriorates substantially; avoiding all sensing is not generally a viable solution.
- Higher or lower VoI cost weights can radically change airtime and tracking quality. Performance claims require validation-only weight selection, held-out scenarios, and a Pareto analysis.
- The work is a stronger closed-loop *baseline* and traceable decision contract, not a demonstrated novel learned scheduling law. No unobserved future packets were consumed.

Full machine-readable per-trajectory report is delivered separately (not embedded into git history). Source SHA-256 hashes in that report bind the tested simulation and scheduler files.
