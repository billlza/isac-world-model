# ISAC World Model — v0.4.1 release engineering

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
