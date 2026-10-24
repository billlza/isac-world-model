# Execution-time contract (v0.3)

The runtime is a **single non-preemptive worker** with a bounded observation queue.
It does not queue model jobs: when free, the worker uses the latest available
snapshot. Sensors continue to acquire while it computes. The neural snapshot
still discards older-acquired late packets; the Kalman reference can retrodict.

- `reference`: old instantaneous logical execution; host compute is measured but
  does not delay outputs. Kept explicitly for comparison and old calibrations.
- `fixed --compute-ms N`: hypothetical declared service time. A result and its
  sensing command are unavailable until computation finishes. Radio acquisition
  occurs when the command is actually executed, not at the old input timestamp.
- `measured`: use synchronized host compute duration. Diagnostic and machine-load
  dependent; rejected for statistical calibration. Not a real-time certification.

Publication and action execution round **up** to the simulator tick. With dt=0.1s,
50ms and 100ms service delays can have the same tick-level effect. These sweeps
must not be used to rank hardware at millisecond precision. Renderer overhead,
OS scheduling, networking, and actuator dynamics are not a fully emulated device.

Between publications, the last published state is propagated with constant
velocity and an explicit white-acceleration covariance model. This is a modeling
assumption, not a proven physical error bound. Before first publication, a neutral
room-center prior is displayed with `no_output`, never accepted as evidence.

Each job logs input cutoff, IDs, maximum available timestamp, finish time,
publication time, requested action and executed action. Each frame distinguishes
state time, result availability and evidence age. Queue overflow drops the **new**
packet and logs its ID; the pending packet budget stays bounded. Completed work
is never retroactively used on earlier frames.

Calibration schema 2 binds the full execution profile. Model/scene/policy/seeds
retain the previous checks. Switching delay, queue budget or covariance inflation
requires recalibration. Schema 1 remains valid only for the reference profile.
Episode-marginal calibration is not conditional accepted-risk assurance and
is not an OOD or safety certificate.

The streaming path defaults to exact-prefix event encoding. Training explicitly
uses batched encoding for throughput. The two are tolerance-compared, including
gradients; a separate gradient test rules out dependence on future inputs. CUDA
and MPS numerical behavior has not been validated in this release.
