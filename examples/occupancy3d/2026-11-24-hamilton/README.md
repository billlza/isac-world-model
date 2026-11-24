# Completed Hamilton 3D pretraining evidence

Job 19558762 executed commit `3111d2c31d2770b87f63545095ad7a59bc48b644`. The application completed all 12 training stages and 84 evaluation cells.

The CSV tables retain all 60 seed-averaged results and all 300 paired-family contrasts. The audit verifies source, checkpoints, split identities and every reported summary. The matched-test comparison freezes both old and new checkpoints and original validation thresholds.

`numerical_replay_audit.json` records aggregate agreement and the stricter per-scene bound that was not met; it is not a bitwise-replay claim. The full result package also contains the original uploaded run, per-scene outputs, recovered RF arrays, predictions and analysis utilities. Training source remains at the pinned Git commit.

Report: [H200 results](../../../docs/RF_3D_H200_RESULTS_2026-11-24.zh-CN.md).
