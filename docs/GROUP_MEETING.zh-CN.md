# 组会演示与运行指南（2026-10-27）

## 今天先准备，明天不现场装依赖

在已上传的仓库工作目录中更新；不要覆盖未提交的修改：

```bash
git status --short
git pull --ff-only
bash scripts/env.sh isac-wm doctor --device cpu
```

已有 `.venv` 和项目工具链就直接用 `env.sh`。全新机器才执行：

```bash
INSTALL_RUSTUP=1 bash scripts/bootstrap.sh
```

首次引导需要 Git、Python >=3.10、系统 C 链接器；macOS 可由 Xcode Command Line Tools
提供。安装脚本不使用 sudo、不改系统默认工具链，已有 `.venv` 不会被默默覆盖。
这些 Bash 命令面向 macOS/Linux；Windows 使用 WSL2，不当成原生 PowerShell 命令。

## 干净克隆不包含检查点

`checkpoints/` 不在 Git 历史中。旧交付 ZIP 附带的 `v020.pt` 不能被假定存在于 GitHub。
下面的入口没有外部模型下载，会先从随机初始化训练一个小型模型，然后生成三组回放：

```bash
bash scripts/env.sh python scripts/group_demo.py --out runs/meeting-20261027
```

默认使用 CPU、两个线程、64 条训练轨迹和 12 个 epoch；模型 features=48、hidden=96，
每条轨迹 48 步。它是可重复演示配置，不是训练充分性或性能领先的声明。
训练种子是 7，测试从 30000000 开始；三组使用同一检查点与测试种子。
命令不覆盖已有输出。再次运行请换目录名。

只检查管线能否工作：

```bash
bash scripts/env.sh python scripts/group_demo.py --quick --out runs/meeting-smoke
```

`--quick` 只有一个 epoch、四条训练轨迹、12 个时间步及更小网络，不能拿来评价精度。
已有自己训练的可信检查点时可以复用：

```bash
bash scripts/env.sh python scripts/group_demo.py \
  --checkpoint runs/train-a/model.pt --out runs/meeting-existing
```

模型原有的训练/验证种子与评测种子重叠检查仍然生效。缺文件会明确失败，不会偷偷下载权重。

## 打开离线演示

macOS：

```bash
open runs/meeting-20261027/index.html
```

Linux 桌面：

```bash
xdg-open runs/meeting-20261027/index.html
```

也可以用浏览器打开该文件。页面及回放不需要外网、账号或远程推理。
如果单位浏览器禁止 file URL，只在本机回环接口服务这个输出目录：

```bash
bash scripts/env.sh python -m http.server 8765 \
  --bind 127.0.0.1 --directory runs/meeting-20261027
```

然后打开 `http://127.0.0.1:8765`。不要把整个仓库、私人日志或权重目录公开到网络。

`index.html` 汇总全部方法，链接三组回放：

1. `nominal-0ms`：名义场景、零假设计算服务时间。
2. `nominal-150ms`：相同检查点及种子，计算服务时间改为 150ms。
3. `camera-loss-150ms`：相机完全丢失，同时保留 150ms 假设服务时间。

每组包含 Kalman/model × fixed/guarded，两种调度共享预算上限，实际用量可能不同。
回放虚线是真值，只供评价；实线是估计。下拉框切换方法和轨迹；播放/暂停、拖动、
方向键可看逐帧信息。重点展示 capture/availability/output 时间、证据年龄和感知动作。
150ms 是模拟服务时间，不是你的电脑实测速度；默认 dt=100ms，发布向上取整到 tick。
未加载校准，监测状态 `uncalibrated` 是预期结果，不是程序报错，也不是安全保证。
若超过配置的截止时限，超时状态也必须保留。

## 建议的 8 分钟汇报

**第 1 分钟：问题。** “我希望研究异步视觉与无线观测下，预测模型怎样决定额外感知何时有价值；
通信资源、处理延迟和跟踪误差必须一起评价。”

**第 2 分钟：当前实现。** “已经有从随机初始化训练的小型视觉—无线时序模型、
状态/未来预测、卡尔曼基线和有界事件回放。当前输入是二维合成图像与处理后的无线特征，
不是实拍视频、原始雷达波形、完整 SLAM 或量子实验。”

**第 3—5 分钟：运行三组回放。** 先看正常轨迹，再改变服务时间，最后关闭视觉。
比较同一 checkpoint 与 seed；不切换到更有利的模型或删掉基线。强调“结果到达时是否仍有用”。

**第 6 分钟：如实解释结果。** 展示本轮 `summary.json` 的实际数字，包括模型落后于卡尔曼的部分。
“我已经有一个能暴露模型问题的试验台；是否有新机制优势，要通过后续同预算实验建立。”

**第 7 分钟：下一步。** 明确动态模型、动作条件观测模型和观测价值估计分别承担什么。
固定预测器比较调度，固定调度比较预测器，避免把收益来源混在一起。

**第 8 分钟：请导师收敛物理任务。** “先固定节点跟踪外部目标，还是移动终端定位与建图？
从哪个组内信号/观测模型起步？现有感知时隙和 AWGN 通信代理应怎样替换成更可靠的模型？”

CI只用一页或一句话说明可重复运行和错误检测能力；不要占据主要研究讨论时间。
现有三篇论文分别对应联合设计、性能界与具体几何融合实例，demo 目前不是第三篇的复现。

## 编译、测试和重新训练

Python 模型不需要手工编译，锁定安装即可。Rust 是独立事件队列组件，尚未替代主模型的 Python 队列。

```bash
bash scripts/env.sh cargo build --workspace --release --locked --offline
bash scripts/env.sh cargo test --workspace --locked --offline
printf 'P 1 0 100\nR 99\nR 100\n' | ./target/release/isac-event-replay
```

输出应依次为 `OK`、`READY `、`READY 1`，说明可用时刻以前不释放事件（单位纳秒）。

```bash
bash scripts/env.sh python -m pytest
bash scripts/env.sh python scripts/ci.py --profile all --out validation/local-ci/meeting-check
```

完整 CI 包含在线依赖审计和冷安装检查，应在会前运行，不在组会上临时执行。
浏览器测试单列，远端 CI 将它列为必需项：

```bash
bash scripts/env.sh uv sync --locked --extra dev --extra browser
bash scripts/env.sh python -m playwright install chromium
bash scripts/env.sh python scripts/browser_smoke.py --out validation/browser-local.json
```

CPU 与原生编译测试不是 MPS/CUDA 性能证明。明天优先使用 CPU 已准备好的离线回放。

## 结果与备份

`demo-manifest.json` 保存命令、状态、实际 Python/包版本、源码摘要、检查点哈希及产物哈希。
`summary.json` 是全部方法的原始汇总；每个子目录有 `evaluation.json` 和 `replay.html`；
`.log` 保留失败。失败不会写成成功的 index 页面。
会前复制整个输出目录作备份，重新打开一次，并保存一次自己的 GitHub Actions 运行链接。
不要拿旧源码的 CI 证据给修改后的代码背书。
