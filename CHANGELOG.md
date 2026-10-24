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

## 0.3.0 — execution-time integrity and mandatory checks

- Single-worker runtime with explicit fixed/measured service delays, output embargo,
  delayed sensing actuation, capture while busy and logged bounded-queue overflow.
- Calibration schema 2 binds execution protocol; rejects mismatched/unstable timing.
- Exact-prefix default encoding and separate fast training path; future-gradient tests.
- Same-byte checkpoint verification/load, checksum validation and bounded ZIP metadata.
- Conservative training-dataset peak estimate; actual end-to-end CLI smoke and wheel install.
- Rust 1.85.1 lock: atomic batch queue admission, bounded text protocol, 15 native tests,
  rustfmt/Clippy/release build verified in an isolated Linux environment.
- 25,005 randomized protocol comparisons and equivalent Python oracle tests.
- CPU pytest/branch coverage, pinned Actions, explicit blocked/failed exit statuses,
  timeout process-group cleanup, artifact logs, source-fingerprint stability checks.
- Fixed-timing sensitivity results retained, including model regression under delay.
- No new large-training or real-device superiority claim; v0.2 checkpoints preserved.
