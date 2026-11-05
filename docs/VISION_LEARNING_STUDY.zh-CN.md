# Phase A：学习率 × 直接视觉监督诊断

这轮保持 A512 的架构、参数量和观测时序不变，分别改变学习率和直接视觉定位损失，检查大模型早期学习过程。正式实验只使用已经登记的训练、验证轨迹，不读取新的测试集。先完成这轮诊断，再决定后续模型或训练方案。

## 1. 已固定的研究问题

前一轮 capacity/data 实验的 A512 与 A96 使用相同的观测时间安排；宽度本身没有改变摄像机采样、观测交付或状态更新的 tick。下一轮记录实际事件数以及网络内部尺度，区分以下量：

- **观测频率**：每段轨迹的相机、RF 新事件及其到达时间。
- **证据更新频率**：某个 tick 是否接收到了有效新观测。
- **计算速度**：GPU 上实际计算所需时间。
- **学习有效性**：视觉定位误差、预测/融合状态误差、梯度和参数更新、测量噪声和加速度输出是否饱和。

这些量可以相关，定义上并不相同。记录事件计数能够直接检验“扩大参数是否使观测或更新变稀疏”；学习率与视觉监督对照用于观察学习过程的变化。

直接视觉监督的权重 `1.0` 是本轮预先选定的诊断权重。它不代表已找出最佳权重；若有明确收益，后续才安排独立的权重选择与正式测试。

## 2. 完整八组实验

| 隐藏宽度 | 学习率 | 视觉定位损失权重 | 初始化 |
|---|---:|---:|---|
| 512 | 0.001 | 0.0 | 7、11 |
| 512 | 0.001 | 1.0 | 7、11 |
| 512 | 0.0003 | 0.0 | 7、11 |
| 512 | 0.0003 | 1.0 | 7、11 |

所有组均从随机权重开始，不导入旧模型，不续训。总参数固定为 **3,883,798**；视觉编码器仍为 **1,230** 参数。`structured-attention`、feature 48、memory 16、heads 4、24×24 图像、64 tick、`dt=0.1 s` 均保持不变。

每个初始化在四个学习率/损失组合中必须具有一致的全模型指纹和视觉编码器指纹；初始化 7 与 11 必须不同。数据生成、批次顺序、训练更新数和模型选择机会保持一致。

| 协议 | 正式 H200 实验 | CI 开发烟测 |
|---|---:|---:|
| cell 数量 | 8 | 8 |
| 实际模型宽度 | 512 | 512 |
| 训练 episode 数 | 1536 | 8 |
| 训练种子起点 | 700000 | 82800000 |
| 验证 episode 数 | 192 | 8 |
| 验证种子起点 | 74611000 | 82810000 |
| batch size | 8 | 4 |
| batch order seed | 37 | 37 |
| optimizer updates | 384 | 8 |
| 完整 epochs | 2 | 4 |
| 验证间隔 | 每 96 updates | 每 2 updates |
| checkpoint 选择机会 | 4 | 4 |
| 设备 | 学校完整 H200 | CPU |
| 新测试集读取 | 无 | 无 |

正式训练种子范围为 `700000..701535`，验证范围为 `74611000..74611191`。开发烟测种子与正式数据和先前测试集分离。烟测保留完整八组和真实 A512 结构，主要验证流程及证据，不用于科研结论。

## 3. 视觉损失及时间含义

图像具有采集时间与交付时间。直接视觉监督使用**被实际接受的图像事件在采集时刻的目标坐标**，并以房间尺寸归一化。不能把延迟图像直接监督为交付时刻的真实位置。

参与该损失的条件同时满足：图像是新事件、输入有效、目标在采集时可见。陈旧重复图像和完全遮挡画面不被强制回归到不存在的可见目标。没有符合条件的图像时，该辅助损失及有效计数均为零。

记有效图像掩码为 \(m_i\)，视觉位置预测及采集时刻目标位置为归一化二维坐标 \(\hat p_i,p_i\)，则登记的损失为：

\[
L_{\mathrm{vision}}=
\frac{\sum_i m_i\,\frac{1}{2}\|\hat p_i-p_i\|^2}
{\max(1,\sum_i m_i)}.
\]

训练目标为原有损失加 `vision_loss_weight × L_vision`。权重零的组保持原有优化目标；额外诊断仍计算视觉误差，使对照组与实验组可以比较。

标签 `vision_capture_target_xy`、`vision_target_mask`、`vision_capture_s`、`vision_event_uid` 仅供训练和诊断。在线输入仍然只有 `image/radio/age/valid/new/dt`，没有把真值坐标传入模型的推理接口。

## 4. 早期诊断与结果解释

正式诊断 updates：`0, 1, 2, 4, 8, 16, 32, 64, 96, 192, 384`。

烟测诊断 updates：`0, 1, 2, 4, 8`。

诊断固定重放验证集前四段轨迹，最多四段、64 tick。它们不增加 checkpoint 选择机会；最佳 checkpoint 仍然只按四次完整验证集 RMSE 选择。诊断观察器不做反向传播，不修改输入，恢复随机数状态和各层训练状态。

每个 cell 保留：

- 初始、最佳与最后一次验证指标，包括完整验证集位置 RMSE 和固定诊断子集视觉采集位置 RMSE。
- 全部指定时刻的视觉定位误差、视觉测量方差、attention 残差与 hidden 尺度、噪声及加速度输出饱和程度。
- 被观察训练更新的各分支梯度、参数范数、实际更新范数及相对更新量。
- 实际新视觉事件、新 RF 事件、证据更新 tick、仅预测 tick、可见视觉采集数。
- 训练耗时及包括验证、诊断开销在内的平均 optimizer updates/s。这个数值不等于在线控制决策时延。

汇总分别列出相同学习率下“视觉损失 1 − 0”、相同视觉权重下“学习率 0.0003 − 0.001”及二者交互项，保留两个初始化的所有结果。误差指标负差值表示改善。两个初始化的均值是描述性结果，不用于宣称任意训练随机种子的总体置信区间。

若低学习率使早期饱和明显减轻，同时状态误差改善，则支持优化尺度是相关因素；若直接视觉监督稳定改善采集位置估计，则支持视觉学习信号是可改善环节。仍需联合查看两个初始化、内部诊断和完整验证结果；单个最优 cell 不能单独判定过去失败的全部原因。

## 5. Hamilton 提交与防重复

本轮使用学校已有的 `python/3.13.9cuda`，固定核验 Python 3.13.9、NumPy 2.3.4、PyTorch 2.9.0+cu128、CUDA build 12.8。申请 `cuda` 分区的一张 `h200_nvl`，wall-time 上限两小时。没有新增虚拟环境、依赖安装或静默设备回退。

提交器使用上一轮已存在的 `$HOME/isac-capacity-3225ce7435c4` 作为已知 Git bundle 基础，其 HEAD 必须为 `3225ce7435c41e1695b7d8f3f1740586b6edbd00`。旧实验目录与结果保持原样。本轮在 `$HOME/isac-vision-<新提交前12位>` 建立普通 Git checkout。

发布完整提交 SHA 后，在 **Mac 原项目目录** 使用以下流程；将 `PUBLISHED_FULL_SHA` 替换为这次公布的 40 位 SHA。

```bash
(
  set -e
  set -o pipefail
  cd "$HOME/Desktop/isac-world-model"
  vision_pin=PUBLISHED_FULL_SHA
  vision_ref=refs/remotes/origin/research/vision-supervision-h200
  git fetch origin "refs/heads/research/vision-supervision-h200:${vision_ref}"
  if [ "$(git rev-parse "$vision_ref")" != "$vision_pin" ]; then
    echo '分支与已验证版本不一致，停止提交。'
    exit 1
  fi
  git show "${vision_pin}:scripts/submit_vision_learning_from_mac.sh" |
    bash -s -- "$vision_pin" bhmq03
)
```

`${vision_pin}:path` 的大括号要保留，避免 Mac zsh 将冒号后的内容解释为变量修饰。

提交器先确认本地版本、远端主机、bundle 前置版本、远端源码整洁性及学校 Python 版本，再创建 `.vision-submission` 提交占位。Slurm 成功回执保存到该目录的 `receipt`。重复执行会打印已保存的作业号，不重复申请 GPU。

若网络中断或调度器返回不明确，提交占位保留。此时先检查 `squeue --me` 与 `receipt.pending/submission.stderr`，确定没有已提交任务后再处理；不要直接删除占位重复发作业。

## 6. 查看进度与取回结果

提交成功会打印实际作业号、日志路径、结果路径及可直接粘贴的下载命令。以下 `<job>`、`<prefix>` 代表回执中的真实值。

```bash
# 在 Hamilton 上
sacct -X -j <job> --format=JobID,State,ExitCode,Elapsed
squeue -j <job> -o "%.18i %.12T %.10M %R"
tail -n 40 "$HOME/isac-vision-<prefix>/slurm-<job>.out"
cat "$HOME/isac-vision-<prefix>/runs/vision-<job>/status.json"
```

`PENDING (Resources)` 表示等待资源，不能提前读取尚未生成的训练结果。训练中 `study.json` 会记录各 cell 的未开始、运行中、已完成状态。阶段状态原子写入；失败保留已有证据，旧结果目录不能覆盖。

满足 Slurm `COMPLETED 0:0`，且最终 `status.json` 为 `completed/done` 后，下载：

- `isac-vision-<job>.tar.gz`
- 最终 `status.json`
- 如需详细排错，再下载 `slurm-<job>.out`。

证据包仅含 JSON 和 checkpoint SHA-256 sidecar。打包器会实际读取并核验 Hamilton 上八个选中 checkpoint 的字节哈希，但不把权重打入压缩包。`bundle_manifest.json` 明确列出权重路径、哈希和 `weights_included:false`。

包中 `job_status_at_packaging.json` 是打包当时的阶段快照，可能仍为 `running/package`；它不冒充最终调度器状态。最终状态来自另取的 `status.json` 与 `sacct`。

打包器拒绝缺失 cell、变更的训练/验证协议、缺失诊断、指纹不一致、事件数变化、非有限数值或改变的 checkpoint/history。归档通过独占 staging 文件和不可覆盖发布生成，避免覆盖旧证据。

## 7. 本地与 CI 验证

不需要 Torch/GPU 的协议、提交、失败保留和打包测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_vision_learning*.py' -v
bash -n scripts/hamilton_vision_learning.sh scripts/submit_vision_learning_from_mac.sh
```

真实 Torch 开发烟测使用现有测试环境：

```bash
PYTHONPATH=src python scripts/vision_learning_study.py \
  --out runs/vision-learning-smoke \
  --device cpu --profile smoke
```

输出目录必须尚不存在。代码审查、数值单测和真实八组烟测分别核验不同层面的行为；正式科研结论以 H200 返回的完整证据包为准。
