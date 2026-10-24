# ISAC World Model — v0.4.0

自主实现、随机初始化的视觉—无线世界模型研究原型。无外部预训练权重、教师模型或私有实验数据。当前为合成二维观测与共享时隙代理；Quantum、原始无线波形和真实设备尚未实现。

**本版重点：最新稳定依赖、可复现工具链、完整 CI 实跑。**参考环境 Python 3.14.9 / PyTorch 2.14.1+cpu / NumPy 2.5.4 / Rust 1.99.1。17 个必需步骤全部通过，154 项 Python 测试、15 项 Rust 测试、Ruff、已知漏洞审计和打包安装检查通过。详情见 `docs/VALIDATION_v040.md`。

## 安装与全工具链检查

```bash
# 在新解压目录内；需要一个现有 Python >=3.10 和系统链接器。
# INSTALL_RUSTUP=1 明确允许下载安装器，只安装到本项目 .toolchains/。
INSTALL_RUSTUP=1 bash scripts/bootstrap.sh
```

脚本创建 `.bootstrap/`、`.venv/`、`.toolchains/`；不需要 sudo，不改变系统默认 Rust/Python。已有 `.venv` 时先保存需要的内容，再显式设置 `RECREATE_VENV=1`。macOS 的系统链接器来自 Xcode Command Line Tools，Linux 需要系统 C 链接工具；脚本不会静默安装系统包。

已有 uv 0.13.5，只安装锁定 Python 开发环境：

```bash
uv python install 3.14.9
uv sync --locked --extra dev --python 3.14.9
source .venv/bin/activate
isac-wm doctor --device cpu
python scripts/dependencies.py check
python scripts/dependencies.py environment
python scripts/ci.py --profile python --out validation/my-python-ci
```

全部门禁（Rust 须在 PATH 中；bootstrap 已自动处理自己的子进程）：

```bash
export CARGO_HOME="$PWD/.toolchains/cargo"
export RUSTUP_HOME="$PWD/.toolchains/rustup"
export PATH="$PWD/.venv/bin:$CARGO_HOME/bin:$PATH"
python scripts/ci.py --profile all --out validation/my-full-ci
```

输出目录必须是新的。退出码 0=全部通过、1=检查失败、2=缺工具等前提条件。`all` 包括 Python、Ruff、已知依赖漏洞审计和 Rust；已知漏洞查询需要联网。它不自动取得 GitHub 登录状态，也不声称验证了不可用的加速器。

## “最新”的精确含义

版本核对日为 **2026-10-24**。直接依赖均为当时核实到的稳定版本，传递依赖以 `uv.lock` 锁定。49 个实际安装第三方包中48个为最新稳定版；唯一传递例外为 mpmath 1.3.0，因为最新 SymPy 1.14.0 要求 `<1.4`，不能强装 mpmath 1.4.1。

Python 最新为3.15.1，但 PyTorch 2.14.1尚无 CPython 3.15预编译包，因此使用3.14.9；项目支持范围改为 `>=3.12,<3.15`，不再声称支持不满足 NumPy 要求的3.11。详见 `docs/DEPENDENCIES.md`。

`uv.lock` 包含精确版本、来源与哈希；当前审核平台是 Linux x86_64 CPU 和 macOS Apple Silicon 的 Python 3.12–3.14包集合。这里只实测了 Linux CPU。Linux 默认选择官方 CPU PyTorch 索引；CUDA训练环境应另行验证并锁定，不能用本轮结果冒充。可选 Playwright 1.64.1已锁定，本轮未执行浏览器验证。

```bash
# 只读查询，不会自动安装/升级或修改锁文件
python scripts/dependencies.py latest --out runs/latest-versions.json
python scripts/audit_dependencies.py --out runs/dependency-audit.json
```

## 延迟进入模型与决策

```bash
isac-wm evaluate --checkpoint checkpoints/v020.pt \
  --out runs/fixed-150ms --episodes 4 --device cpu --threads 2 \
  --latency-mode fixed --compute-ms 150
```

附带 `v020.pt` 是上一版自主训练检查点，v0.3 用它做过配对延迟评测；本版只做工具链与集成训练验证，不把旧准确率当成新结果。运行输出 `evaluation.json` 和独立 HTML 回放。

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

## GitHub CI 与发布

工作流定义 Linux/Python 3.12.15、3.13.17、3.14.9，以及 macOS/Python 3.14.9的 CPU job；另有双平台 Rust、质量和已知漏洞审计、最终汇总门禁。Actions 更新并固定到完整提交SHA，只读权限，不保留checkout凭据。

**本轮全门禁实跑于隔离 Linux 环境，尚未触发 GitHub Actions。**当前连接查询 `billlza/isac-world-model` 返回404，不伪造远端成功。macOS job尚未执行，不能据此声称 MPS已测。

```bash
# 在你已登录 gh 的电脑上，为新仓库使用原有保守发布脚本
bash scripts/publish_private.sh billlza/isac-world-model
bash scripts/verify_remote_ci.sh billlza/isac-world-model main
```

脚本拒绝覆盖已存在的仓库或origin。默认私有；公开与知识产权安排见 `RIGHTS.md`。不要向聊天粘贴令牌。
