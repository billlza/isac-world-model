# Verification-aware active observation

## Problem and scope

The October active-observation policy ranks camera actions by a two-candidate
Bayes-error reduction proxy minus declared costs. A sufficiently strong first
observation can make this proxy smaller than the action cost, even though the
confirmation protocol still needs a second informative observation. An
all-candidates-incompatible observation can likewise call for a second check
without offering much two-candidate classification gain.

This change adds an explicit verification rule. It does not claim an optimal
value-of-information solution, calibrated confidence or independent evidence.
The simulator, motion budget, message cost and arrival-time constraints remain
part of every proposed action.

## Evidence protocol

| State | Rule |
| --- | --- |
| No compatible informative observation | Use the ordinary action objective; allow unknown |
| One compatible informative observation | Request a feasible, geometrically informative confirmation view |
| One all-candidates contradiction | Suspend confirmation and request a feasible check |
| Two consecutive valid contradictions | Reject the existing candidate set |
| Missing detection | Retain the existing evidence state; missing is neither support nor contradiction |

`accepted`, `informative` and `contradictions` retain cumulative diagnostic
counts. `supporting_informative` counts compatible informative detections since
the most recent contradiction. A contradiction resets only that support count,
not the audit history or previous predictions. Two fresh supports are required
before reconfirmation. Compatibility here means compatibility with at least
one candidate in the set, not two votes for the eventual winner. Accumulated
candidate weights are retained and the relative weight threshold still applies
afterward; this protocol alone does not eliminate an overconcentrated historical
belief or establish successful adaptation to a changed target identity.

An incompatible packet contributes to predictive NLL and the recorded trace,
but does not reinforce the least-wrong candidate or update its ordinary neural
hidden state. A compatible detection breaks the contradiction streak. Missing
detections do not break a streak of valid contradictory detections. Rejection
is latched for the current fixed candidate set; generating and evaluating a new
candidate set is a separate, unimplemented capability.

The neural observer is trained on nominal sequences. Training and inference
both gate recurrent updates using the predicted candidate-set compatibility;
the current-step supervised NLL still includes incompatible observations. Old
checkpoints were selected under the earlier update protocol, so their memory
performance must be reevaluated. The geometry-only paired study below isolates
the action rule and cannot establish a neural-memory advantage.

## Action rule and audit fields

All candidate actions are first filtered by movement budget and packet-arrival
deadline. Confirmation verification additionally requires physical geometric
separation under the declared uncertainty floors; neural confidence cannot
manufacture identifiability. Contradiction verification can use a view where
both candidates predict the same bearing, because such a view can still
contradict both predictions.

Among eligible verification views, choose greatest separation from the last
arrived valid detection, then earlier arrival and lower travel. If only the
same view is feasible, its new packet can be used, with
`verification_same_view=true`. View separation and distinct packet IDs do not
remove shared RF-anchor or camera bias. The two-observation rule is an explicit
experimental protocol, not a statistical certificate.

The original score is preserved, including negative values, rather than being
relabelled as positive information gain. Each chosen action records
`selection_reason`, `verification_view_separation_m` and
`verification_same_view`. Trace entries expose contradiction streak, support
count and pending verification. No hidden simulator labels enter this rule.

## Controlled comparison

`--planner verification-aware` is the new default.
`--planner legacy-voi` retains the old action criterion while sharing the
corrected evidence protocol. It is **not** the entire original October
algorithm. Existing October artifacts remain historical evidence.

Run the small geometry-only paired study using an existing project runtime:

```bash
python scripts/verification_study.py --out runs/verification-paired --episodes 8 --seed 20261103
```

The predeclared seed list, six scenarios, same initial RF evidence and both
planners are retained in the output. Eight seeds per scenario produce 96
rollouts. `summary.json` includes actual resource costs and per-episode paired
deltas. `study.json` includes every trace and failure. Predictive NLL is stored
per episode, but policy-selected observations are different evaluation sets;
do not interpret a policy-to-policy NLL difference as a matched prediction
comparison. Source hashes, runtime versions and the actual commit identify the
run. Existing output directories are refused; failures leave a failed manifest
and a nonzero exit.

The normal CI executes this study once on Linux/Python 3.14.9 and uploads it
beside the required Python evidence. This is CPU validation, not an H200 run.
Meaningful follow-up is an error/completion/resource trade-off across more
seeds, followed by common-sequence evaluation of learned memory and no-memory
models. No improvement is assumed before the recorded outcomes are inspected.
