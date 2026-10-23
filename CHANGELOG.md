# Changelog

## 0.2.0 — 2026-10-23

- Cache exact OOSM Kalman posterior prefixes; retain reference implementation for parity and profiling.
- Batch encode newly delivered modality inputs; avoid redundant encoder work; preserve state_dict compatibility.
- Validate bounded payloads, finite values, IDs, times and whole batches before state mutation.
- Atomic checkpoints, integrity sidecars, restricted bounded loaders, early stopping and deterministic CPU resume.
- Whole-episode uncertainty calibration with model/scene/policy/split binding; default uncalibrated abstention.
- Add experimental arrival-risk and freshness-guarded policies; retain observed failures.
- Six paired fault scenarios, richer runtime/quality/status metrics, standalone CPU benchmark script.
- Rebuild local replay with requestAnimationFrame, speed/pause/scrub/keyboard and responsive layout.
- Add native broker ID-history cap and next-arrival API (source only; Rust toolchain unavailable locally).
- 87 Python tests and real browser interaction checks; no GPU, hardware or safety-certification claim.
