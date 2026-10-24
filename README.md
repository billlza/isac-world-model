# ISAC World Model — v0.3.0

自主实现、随机初始化的视觉—无线世界模型研究原型。无外部预训练权重、教师模型或私有实验数据。当前为合成二维观测与共享时隙代理；Quantum、原始无线波形和真实设备尚未实现。

本版补齐**显式计算服务时钟、结果延迟发布、感知指令延迟执行、Rust 原生队列验证与可失败的 CI 门禁**。详细证据见 `docs/VALIDATION_v030.md`。测试通过不等于模型已经超过经典基线，也不等于安全认证。

## 安装与检查

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c ci/constraints.txt -e '.[dev]'
isac-wm doctor --device cpu
python scripts/ci.py --profile python --out validation/my-python-ci
```

使用 Python 3.11–3.13。完整工具链可使用 `bash scripts/bootstrap.sh`：Rust 固定为 1.85.1，带 rustfmt/Clippy。脚本不会静默安装系统链接器；rustup 不存在时需先安装或显式授权下载安装器。开发依赖是直接版本约束，不是完整传递依赖的哈希锁。见 `docs/CI.md`。

```bash
# 全部门禁：缺工具或任何失败都不允许返回成功
python scripts/ci.py --profile all --out validation/my-full-ci
```

输出目录必须是新的，避免覆盖旧证据。退出码 0=通过、1=失败、2=缺前提条件。原生队列是独立组件；当前主实验使用 Python 参考队列，并用相同随机协议参考验证二者行为。

## 延迟进入模型与决策

```bash
isac-wm evaluate --checkpoint checkpoints/v020.pt \
  --out runs/fixed-150ms --episodes 4 --device cpu --threads 2 \
  --latency-mode fixed --compute-ms 150
```

附带 `v020.pt` 是上一版自主训练检查点，本版用它做配对回归和延迟敏感性评测，没有冒充新的精度突破。运行输出 `evaluation.json` 和独立 HTML 回放。

- `reference`：保留旧的逻辑瞬时推理模式，仅用于比较和旧校准兼容。
- `fixed`：指定假设服务时延；计算完成前，输出和感知指令都不可用。
- `measured`：以实际主机推理耗时驱动虚拟时间，依赖机器负载，只作诊断。

传感器采集在推理忙碌时继续；队列有界，丢弃新包必须记入日志。无输出时显示 `no_output`。结果发布取上一个已完成任务，并按声明的运动/协方差模型传播到当前时刻。

**时间分辨率仍由仿真 dt 决定。**默认 dt=100ms 时，结果发布向上取整到下一 tick；不能用这一离散实验宣称硬件有几毫秒的优势。`fixed` 的150ms不是这台电脑的实测耗时。完整语义见 `docs/RUNTIME.md`。

## 从零训练、恢复与校准

```bash
isac-wm train --out runs/train-a --episodes 192 --epochs 24 \
  --batch-size 8 --features 48 --hidden 96 --device cpu --threads 2
isac-wm train --resume runs/train-a/resume.pt --out runs/train-b \
  --episodes 192 --epochs 40 --batch-size 8 --features 48 --hidden 96 \
  --device cpu --threads 2
isac-wm calibrate --checkpoint runs/train-a/model.pt \
  --out runs/calibration-150ms.json --episodes 32 \
  --latency-mode fixed --compute-ms 150 --device cpu --threads 2
isac-wm evaluate --checkpoint runs/train-a/model.pt \
  --calibration runs/calibration-150ms.json --out runs/test-150ms --episodes 8 \
  --latency-mode fixed --compute-ms 150 --device cpu --threads 2
```

校准绑定模型、场景、策略、分割和**运行时协议**。改时延/队列上限/协方差模型，需重新校准；实测时延模式不接受固定配置校准。独立轨迹覆盖不是条件接受风险保证，也不是对分布外或物理安全的保证。

训练显式使用批量编码，逐步评测默认严格前缀模式。既测试两条路径的数值/梯度一致性，也独立测试未来输入梯度为零。数据集内存预算是预估分配检查，不是整个进程的硬内存隔离。

## 故障测试与硬件

```bash
isac-wm stress --checkpoint checkpoints/v020.pt --out runs/stress-150ms \
  --episodes 4 --latency-mode fixed --compute-ms 150 --device cpu --threads 2
python scripts/latency_sweep.py --checkpoint checkpoints/v020.pt --out runs/timing-sweep
```

故障扫描中的任一场景失败都会写记录并返回非零状态。明确选择不可用的 CUDA/MPS 会报错，不会冒充 GPU。本版实际测试 CPU；GitHub macOS job 也是 CPU，不等于 MPS 已经验证。

## 发布和 GitHub CI

本版没有完成 GitHub 推送，不能把本地/隔离环境执行当成 GitHub Actions 绿灯。工作流定义 Linux/Python 3.11、3.13，macOS/Python 3.12 CPU、Rust、静态检查与聚合门禁。Actions固定SHA，默认只读权限，不使用密钥和特权PR触发器。

```bash
# 新私有仓库；拒绝覆盖已有仓库与 origin
bash scripts/publish_private.sh billlza/isac-world-model
# 已推送后核对具体提交、触发并等待远端 CI
bash scripts/verify_remote_ci.sh billlza/isac-world-model main
```

需要由用户已认证的 GitHub CLI 执行；不向聊天提供访问令牌。机构/合作方的公开与知识产权安排仍按 `RIGHTS.md` 确认。
