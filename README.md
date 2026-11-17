# ISAC World Model — v0.4.1 release engineering

## Complex-scene pretraining from random initialization

The new pretraining study brings maneuvering, occlusion, asynchronous delays,
radio outliers/bias, sparse camera sampling and their combinations into training.
It compares a nominal-data control, a mixed-data control, and a mixed-data model
with a larger visual encoder, a wider acceleration range and a differentiable
displacement objective. All three use the same held-out evaluation episodes.

```bash
bash scripts/env.sh python scripts/complex_scene_pretraining.py \
  --out runs/complex-development --profile development --device cpu --threads 2
```

Selected, final and continuation weights are saved with checksums. The new
Hamilton launcher pins the exact source revision and archives the weights too.
See [the training protocol, recovery guide and scope](docs/COMPLEX_PRETRAINING.zh-CN.md).
This stage trains the existing synthetic RGB/radio state model for complex-scene
state estimation and multi-step physical prediction.

The completed two-initialization development study measured a 12.10% reduction
in two-second forecast macro RMSE for mixed-data training, with a 1.07% increase
in current-position macro RMSE. Radio-outlier and combined-condition tracking
improved by 21.34% and 16.79%. The combined enhanced configuration did not beat the
mixed-data control under this budget. Full results and scope are in the protocol;
the first formal H200 submission was rejected by the partition resource policy.
The corrected launcher leaves CPU and memory allocation to the GPU partition;
formal execution still requires a new scheduler receipt.

## Large-model optimization and trajectory error replay

Phase C completed six Hamilton cells (job 19404810). At equal data and update
budgets, mean best validation position RMSE was 0.331078 m for A512, 0.437621 m
for A1024 and 0.443610 m for A2048. All four paired larger-width best-checkpoint
contrasts were worse under that recipe. The 59.56M-parameter A2048 remains in
the next registered experiment alongside an A512 reference.

Phase D compares the original A2048 optimizer with half the learning rate for
width-dependent modules while preserving the visual/RF encoder learning rate,
initial weights, observation cadence and selection budget. It first replays
retained Phase C checkpoints, then records both selected and final model errors
per trajectory and tick, including P95/P99, maxima, fixed exceedance bins and
matched streaming/batched execution. No fresh test is consumed.

See [the Phase D protocol and Hamilton submission guide](docs/WIDTH_OPTIMIZATION_STUDY.zh-CN.md)
and [the completed Phase C design](docs/NORMALIZED_CAPACITY_STUDY.zh-CN.md).
Phase D is an implemented experiment awaiting formal results; its descriptive
error thresholds are not validated physical operating limits.

## Active observation loop and MATLAB replay

`scripts/active_observation_study.py` executes candidate-view predictions, moves a
synthetic perspective camera, and releases its newly extracted evidence only at
the declared arrival time. It compares a geometric VoI baseline, small randomly
initialized recurrent / memoryless observation learners, fixed/random sensing,
and an always-unknown control across six scenarios, retaining failed outcomes.

On a new Mac with base MATLAB, open **`START_MATLAB.m`** and click Run: the native
OFDM checks run first, then the portable interactive replay opens. Viewing the
included record needs no Python installation or additional MATLAB toolboxes.
MATLAB execution is **pending verification on the new computer**.
See [the macOS MATLAB guide and experimental limits](docs/MATLAB_ACTIVE_OBSERVATION.zh-CN.md).

This is a small learned observation/residual-memory experiment, not joint training
of the complete RGB/RF neural World Model or evidence of autonomous safety.

## Falsifiable RF–vision evidence memory (separate experimental reasoning stage)

This is **not** material recognition or a deployed autonomous controller. `src/isac_wm/evidence_reasoning.py` records bounded, timestamped radar/vision evidence and compares *same-object* versus *unrelated-object* explanations without using ground-truth labels. It retains explicit counterevidence and returns **unknown identity**, including when angles agree but independent depth is missing. Vision here is an intentionally restricted synthetic perspective bearing cue; existing orthographic-image neural checkpoints are NOT drop-in compatible.

```bash
bash scripts/env.sh python scripts/phy_reasoning_demo.py --out runs/reason-agree --scenario compatible --frames 4
bash scripts/env.sh python scripts/phy_reasoning_demo.py --out runs/reason-conflict --scenario conflicting_view --frames 4
bash scripts/env.sh python scripts/phy_reasoning_demo.py --out runs/reason-occluded --scenario occluded --frames 4
bash scripts/env.sh python scripts/phy_reasoning_demo.py --out runs/reason-collinear --scenario collinear_view --frames 4
```

See [the Chinese design, assumptions and failure-model document](docs/EVIDENCE_REASONING_WORLD_MODEL.zh-CN.md). This is a **model-based, interpretable baseline** preceding the planned learned, action-conditioned predictive world model; RF error floors and association score are *not calibrated probabilities*.


## Reproducible MIMO-OFDM PHY baseline (separate first-stage experiment)

A **common-waveform**, CPU-only PHY baseline now lives in `src/isac_wm/phy_ofdm.py`:
2-Tx / 4-Rx QPSK MIMO downlink with noisy pilot-estimated CSI, plus an 8-Rx monostatic radar array using the *same* OFDM transmissions to estimate target range, bearing and Doppler. It contains a common moving scatterer in the communication and sensing models. Dedicated sector-directed probe symbols consume payload REs; ordinary transmitted data also serve as radar illumination.

```bash
# After the normal project-local bootstrap, run reproducible paired simulations.
bash scripts/env.sh python scripts/phy_sweep.py \\
  --out runs/phy-baseline --trials 16 --fractions 0,0.2,0.4 --radar-snr-db -35

# On a Mac with MATLAB, base-MATLAB FFT/CP/radar sanity checks plus CSV plot:
# addpath('scripts'); matlab_phy_smoke('runs/phy-baseline/summary.csv');
```

See [the PHY derivation, assumptions and MATLAB instructions](docs/MIMO_OFDM_PHY.zh-CN.md) and [reproducible example results](examples/phy_first_stage/manifest.json). **This is a physics-facing benchmark, not yet the radio backend of the learned World Model, not real 5G NR or material classification.** MathWorks also provides a distinct bistatic [communication-centric MIMO-OFDM ISAC example](https://www.mathworks.com/help/phased/ug/integrated-sensing-and-communication-2-communication-centric-approach-using-mimo-ofdm.html).


## GitHub and clean-clone demonstration

The private repository is `billlza/isac-world-model`. Check GitHub Actions for the
exact commit being demonstrated; historical archive CI reports are not current-run proof.
**Checkpoints are not committed to Git.** A fresh clone must train one or use a
trusted local checkpoint. No external pretrained weights are downloaded.

```bash
# After setup, create results before the meeting:
bash scripts/env.sh python scripts/group_demo.py --out runs/meeting-20261027
# Pipeline-only smoke (one epoch, not an accuracy benchmark):
bash scripts/env.sh python scripts/group_demo.py --quick --out runs/meeting-smoke
```

Open `runs/meeting-20261027/index.html` in a browser. The three offline replays compare
nominal/zero service time, nominal/150ms service time, and camera loss/150ms service time.
All use one checkpoint and the same held-out seeds. Outputs are never overwritten.
See [the meeting runbook](docs/GROUP_MEETING.zh-CN.md) for setup, compilation, demonstration
order, interpretation and failure recovery.

## First setup and verification

```bash
INSTALL_RUSTUP=1 bash scripts/bootstrap.sh
# Use project-local toolchains again in any new shell:
bash scripts/env.sh python scripts/ci.py --profile all --out validation/local-ci/my-run
bash scripts/env.sh python scripts/evidence.py --report validation/local-ci/my-run/ci-report.json
```

The default is the pinned Linux CPU / macOS Apple Silicon Python stack. CUDA is not installed
silently; an accelerator backend needs its own pinned dependency profile and hardware validation.
Do not interpret successful CPU CI as a CUDA or MPS benchmark.

A release archive may include `history.bundle`; the publisher can restore Git metadata without modifying
source or generating a replacement commit. It will not push the bundle, checkpoints or local data.

```bash
# Local preflight only, no remote mutation:
python3 scripts/publish_private.py
# After installing GitHub CLI and logging in as billlza locally:
gh auth login --hostname github.com --git-protocol https --web
python3 scripts/publish_private.py --execute
```

For first publication only: the publisher creates a **private** repository, refuses existing different history, never
force-pushes, checks the remote lock/runtime/workflow bytes and waits for exact-commit CI. It
records upload status separately from CI status. Existing empty private repositories are allowed.
Credentials stay in the user's GitHub CLI configuration, not this project or chat.

See `docs/RELEASE_v041.md` for scope and failure recovery.

## 延迟进入模型与决策

```bash
bash scripts/env.sh isac-wm evaluate --checkpoint runs/train-a/model.pt \
  --out runs/fixed-150ms --episodes 4 --device cpu --threads 2 \
  --latency-mode fixed --compute-ms 150
```

先按下节训练，或使用上面的 group_demo 入口。`v020.pt` 只存在于旧交付包，不在 GitHub 仓库中。运行输出 `evaluation.json` 和独立 HTML 回放；旧准确率不当成新结果。

- `reference`：保留旧的逻辑瞬时推理模式，仅用于比较和旧校准兼容。
- `fixed`：指定假设服务时延；计算完成前，输出和感知指令都不可用。
- `measured`：以实际主机推理耗时驱动虚拟时间，依赖机器负载，只作诊断。

传感器采集在推理忙碌时继续；队列有界，丢弃新包必须记入日志。无输出时显示 `no_output`。结果发布取上一个已完成任务，并按声明的运动/协方差模型传播到当前时刻。

**时间分辨率仍由仿真 dt 决定。**默认 dt=100ms 时，结果发布向上取整到下一 tick；不能用这一离散实验宣称硬件有几毫秒的优势。`fixed` 的150ms不是这台电脑的实测耗时。完整语义见 `docs/RUNTIME.md`。

## 从零训练、恢复与校准

```bash
bash scripts/env.sh isac-wm train --out runs/train-a --episodes 192 --epochs 24 \
  --batch-size 8 --features 48 --hidden 96 --device cpu --threads 2
bash scripts/env.sh isac-wm train --resume runs/train-a/resume.pt --out runs/train-b \
  --episodes 192 --epochs 40 --batch-size 8 --features 48 --hidden 96 \
  --device cpu --threads 2
bash scripts/env.sh isac-wm calibrate --checkpoint runs/train-a/model.pt \
  --out runs/calibration-150ms.json --episodes 32 \
  --latency-mode fixed --compute-ms 150 --device cpu --threads 2
bash scripts/env.sh isac-wm evaluate --checkpoint runs/train-a/model.pt \
  --calibration runs/calibration-150ms.json --out runs/test-150ms --episodes 8 \
  --latency-mode fixed --compute-ms 150 --device cpu --threads 2
```

校准绑定模型、场景、策略、分割和**运行时协议**。改时延/队列上限/协方差模型，需重新校准；实测时延模式不接受固定配置校准。独立轨迹覆盖不是条件接受风险保证，也不是对分布外或物理安全的保证。

训练显式使用批量编码，逐步评测默认严格前缀模式。既测试两条路径的数值/梯度一致性，也独立测试未来输入梯度为零。数据集内存预算是预估分配检查，不是整个进程的硬内存隔离。

## 故障测试与硬件

```bash
bash scripts/env.sh isac-wm stress --checkpoint runs/train-a/model.pt --out runs/stress-150ms \
  --episodes 4 --latency-mode fixed --compute-ms 150 --device cpu --threads 2
bash scripts/env.sh python scripts/latency_sweep.py --checkpoint runs/train-a/model.pt --out runs/timing-sweep
```

故障扫描中的任一场景失败都会写记录并返回非零状态。明确选择不可用的 CUDA/MPS 会报错，不会冒充 GPU。本版实际测试 CPU；GitHub macOS job 也是 CPU，不等于 MPS 已经验证。
