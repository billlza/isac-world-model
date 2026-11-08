# Phase B：递归隐藏状态尺度的配对实验

本轮检验一个具体机制：约束完整递归 hidden 的输出尺度，能否减轻训练后期的融合退化，并保留已经观察到的视觉定位收益。固定 A512 的参数量、视觉监督、学习率、数据和更新预算，仅改变递归 hidden 的归一化条件。

## 1. Phase A 提供的实验依据

Hamilton 作业 `19373715` 已完成八组正式训练与验证。源码固定为
`0656dd4051001edbc08a8468375271e378cc9002`，结果归档
`isac-vision-19373715.tar.gz` 的 SHA-256 为
`bb3a973d1faeb69c280d0b93f28ed7d53f5c87d44781965b9ce443f937fcd9f5`。

相同视觉监督条件和初始化下，学习率由 `1e-3` 降到 `3e-4` 后，四个配对的最佳验证误差均改善。低学习率、视觉监督权重为 1 的两个初始化出现了不同学习轨迹：

| Phase A 条件 | 最佳融合位置 RMSE | 末次融合位置 RMSE | 末次视觉诊断 RMSE |
|---|---:|---:|---:|
| `3e-4 / vision=1 / init=7` | 0.325374 m | 0.325374 m | 0.044578 m |
| `3e-4 / vision=1 / init=11` | 0.403300 m | 1.156187 m | 3.447915 m |

融合指标使用完整的 **192 条验证轨迹**。视觉诊断使用固定 **4 条验证轨迹中的 101 次可见图像捕获**，目标对应图像采集时刻。`init=7` 的视觉定位已达到约 **4.46 cm**，完整验证融合误差约为 **32.54 cm**。

`init=11` 从更新 192 到 384，hidden RMS 由 0.709777 增至 11.457467，约放大 **16.14 倍**；加速度和视觉、无线噪声读出也出现明显饱和。同期视觉误差由 3.632473 m 略降至 3.447915 m。这些伴随变化支持进一步干预递归状态与物理读出的耦合，但尚未确定单一因果起点。Phase A 的内部诊断没有更新 288 的快照，本轮补上该观察点。

结果与原设计口径见 [Phase A 协议](VISION_LEARNING_STUDY.zh-CN.md)。本轮重新训练四组配对单元，保留 Phase A 的原始结果与最佳权重，不把 NCC 的独立开发诊断拼入正式学习曲线。

## 2. 唯一结构因素及其放置点

配置 `latent_state_norm` 有两个条件：

| 条件 | 行为 |
|---|---|
| `none` | 默认旧路径；维持原有计算与旧配置序列化格式 |
| `layernorm` | 完整 hidden 更新后执行无可训练参数的 LayerNorm，再用于物理读出并写回递归 memory |

归一化覆盖逐 tick 更新中新观测触发的融合路径和没有新观测的预测路径；物理位置、速度及协方差不直接归一化。query、memory key 和 feedforward 输入原有的 LayerNorm 保留。新增位置作用于所有残差合成后的 hidden。独立 `forecast` 接口只外推物理状态，沿用上一次更新写回的 hidden，不额外重复归一化。

设单个 hidden 向量为 \(h\)，参数无关的变换为

\[
\operatorname{LN}_{\epsilon}(h)
=\frac{h-\operatorname{mean}(h)}
{\sqrt{\operatorname{Var}(h)+\epsilon}}.
\]

使用 `elementwise_affine=False`、`epsilon=1e-5`，不引入可训练缩放或偏置。精确算术下，输出 RMS 为
\(\sqrt{\operatorname{Var}(h)/(\operatorname{Var}(h)+\epsilon)}\)，不超过 1，通常接近 1。原 hidden 的 RMS 小于 1 时，归一化可能使它增大；因此不能把“RMS 下降”设为必然结果。

这个变换去除了 hidden 的均值和幅度信息，可能改变有效表征及噪声校准。物理读出层的权重仍可增长，输出 logits 仍可能饱和。下一轮的判断依据是配对任务表现，归一化后的数值界本身不构成性能改善证据。

源实现见 [structured.py](../src/isac_wm/structured.py)。训练已有全局梯度范数裁剪 `1.0`；本轮保持该优化设置。[training.py](../src/isac_wm/training.py) 中的梯度裁剪约束参数梯度，与此处约束前向递归 hidden 的位置不同。

## 3. 四组正式实验

| cell | 递归归一化 | 初始化 |
|---|---|---:|
| `a512-norm-none-init-7` | none | 7 |
| `a512-norm-none-init-11` | none | 11 |
| `a512-norm-layernorm-init-7` | layernorm | 7 |
| `a512-norm-layernorm-init-11` | layernorm | 11 |

所有组使用 `structured-attention`、hidden 512、feature 48、memory 16、heads 4；图像为 24×24，轨迹为 64 tick，`dt=0.1 s`。总参数保持 **3,883,798**，视觉编码器保持 **1,230** 参数。学习率固定为 `0.0003`，直接视觉监督权重固定为 `1.0`。

每个初始化在两个归一化条件下使用相同的初始参数和视觉编码器指纹；初始化 7 与 11 的指纹必须不同。所有模型从随机权重开始，不续训已有模型。保持相同的训练数据、批次顺序、实际观测事件数、优化器、数值精度与验证选择机会。

| 协议 | 正式 Hamilton 实验 | CI 开发验证 |
|---|---:|---:|
| cell 数量 | 4 | 4 |
| 模型宽度 | 512 | 512 |
| 训练轨迹数 | 1536 | 16 |
| 训练种子起点 | 700000 | 89110000 |
| 验证轨迹数 | 192 | 8 |
| 验证种子起点 | 74611000 | 89120000 |
| batch size | 8 | 8 |
| batch order seed | 37 | 37 |
| 优化器更新数 | 384 | 8 |
| 完整 epochs | 2 | 4 |
| 完整验证 updates | 96、192、288、384 | 2、4、6、8 |
| checkpoint 选择机会 | 4 | 4 |
| 执行设备 | 完整 H200 | CPU |
| fresh test 使用 | 无 | 无 |

正式训练种子为 `700000..701535`，验证种子为 `74611000..74611191`。开发种子为 `89110000..89110015` 与 `89120000..89120007`，与正式数据分开。开发验证运行真实 A512 的前向、反向及优化器更新，用于检查四组流程、兼容性和证据；科研结论来自正式训练结果。

视觉标签继续对应实际接受图像的采集时刻，只有 fresh、valid、visible 的图像参与直接定位监督。在线输入保持 `image/radio/age/valid/new/dt`；训练目标和诊断所用真值不进入在线推理接口。

## 4. 诊断和模型选择

正式内部诊断 updates：

```text
0, 1, 2, 4, 8, 16, 32, 64, 96, 192, 288, 384
```

开发内部诊断 updates：

```text
0, 1, 2, 4, 6, 8
```

诊断仍使用验证集前四条固定轨迹，最多 64 tick。增加内部观察点不会增加完整验证或最佳 checkpoint 的选择次数。观察器恢复模型各层模式和随机数状态，保留原有输入、参数、梯度及优化流程。

每组保存完整验证历史，以及视觉定位误差、归一化前后 hidden 尺度、attention 残差、加速度和噪声输出饱和、测量方差、增益、创新、分支梯度与实际参数更新。`hidden_pre_norm_rms` 和 `hidden_post_norm_rms` 分别记录归一化边界前后尺度；沿用的 `hidden_rms` 与后者相同，表示物理读出和递归写回实际使用的 hidden。`hidden_scope` 记录这一口径，避免将两种位置的数值混为一条曲线。

必须分别报告四次完整验证中的 **最佳位置 RMSE** 和第 384 次更新的 **固定终点 RMSE**。末次视觉诊断对应更新 384，不自动代表最佳 checkpoint 的视觉能力。视觉位置误差仅涵盖可见捕获；测量方差、熵和修正诊断涵盖 fresh-valid 图像，包括遮挡事件，分析时保持各自样本口径。

主要比较是每个初始化下 `layernorm − none` 的误差差值。负值表示误差降低。两个初始化的均值是条件描述性结果。重点检查：

1. init 11 的终点误差及后半段退化是否改善；
2. init 7 已有的融合与视觉定位收益是否保留；
3. 最佳值与终点值是否呈现一致或不同的变化。

隐藏状态、梯度、噪声和增益帮助解释上述任务指标。成功的 Phase A 单元也存在部分输出饱和，全部饱和比例归零不作为成功条件。

## 5. Hamilton 提交与结果保留

脚本复用学校的 `python/3.13.9cuda`，核验 Python 3.13.9、NumPy 2.3.4、PyTorch 2.9.0+cu128 和 CUDA build 12.8。正式作业申请 `cuda` 分区的一张完整 `h200_nvl`；预检核验实际分配和 CUDA 运算，不改用 CPU 或 MIG，不缩小 batch、模型或序列。

Mac 提交器为 [submit_recurrent_state_from_mac.sh](../scripts/submit_recurrent_state_from_mac.sh)，作业脚本为 [hamilton_recurrent_state.sh](../scripts/hamilton_recurrent_state.sh)。提交器传输自包含 Git bundle，在
`$HOME/isac-state-<源码前12位>` 建立独立的普通 Git checkout，不依赖旧 Hamilton checkout 的前置提交。旧实验目录与结果保持原样。

公布通过验证的完整提交后，在 **Mac 原项目目录** 执行以下流程；`PUBLISHED_FULL_SHA` 必须替换为对应的 40 位源码 SHA。

```bash
(
  set -e
  set -o pipefail
  cd "$HOME/Desktop/isac-world-model"
  state_pin=PUBLISHED_FULL_SHA
  state_ref=refs/remotes/origin/research/recurrent-state-stability
  git fetch origin "refs/heads/research/recurrent-state-stability:${state_ref}"
  if [ "$(git rev-parse "$state_ref")" != "$state_pin" ]; then
    echo '分支与已验证版本不一致，停止提交。'
    exit 1
  fi
  git show "${state_pin}:scripts/submit_recurrent_state_from_mac.sh" |
    bash -s -- "$state_pin" bhmq03
)
```

`${state_pin}:path` 的大括号保留，避免 Mac zsh 的冒号变量修饰。学校密码由用户在 SSH 提示中输入。

提交器在 `.state-submission` 中保存源码提交、请求、阶段、回执和实际作业号；重复调用优先返回已有作业号。若网络中断或 Slurm 回应不明确，保留提交占位并先核对队列与回执，避免未确认结果时重复发起训练。新结果目录为
`$HOME/isac-state-<源码前12位>/runs/state-<作业号>`。

## 6. 查看状态与取回结果

提交器打印实际作业号、日志、结果路径和下载命令。拿到实际回执后，在 Hamilton 查看：

```bash
sacct -X -j <job> --format=JobID,State,ExitCode,Elapsed
squeue -j <job> -o "%.18i %.12T %.10M %R"
tail -n 40 "$HOME/isac-state-<prefix>/slurm-<job>.out"
cat "$HOME/isac-state-<prefix>/runs/state-<job>/status.json"
```

`PENDING` 时尚未生成的训练文件是正常状态。训练开始后，阶段状态和逐单元状态持续保留；失败保留已完成证据，现有结果目录不能覆盖。

待 Slurm 为 `COMPLETED 0:0`，且最终 `status.json` 为 `completed/done`，取回
`isac-state-<job>.tar.gz` 和最终 `status.json`。打包器
[package_recurrent_state_study.py](../scripts/package_recurrent_state_study.py) 核对四组协议、完整训练和诊断记录、参数指纹、实际事件数及服务器上选中 checkpoint 的字节哈希。权重保留在 Hamilton，证据包明确记录 `weights_included:false`。

包中的 `job_status_at_packaging.json` 记录打包当时的阶段；最终作业状态来自另取的 `status.json` 与 `sacct`。缺失或变化的证据会使打包失败，不发布不完整的成功归档。

## 7. 开发验证与 CI

协议、提交与打包测试，以及 shell 语法检查：

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_recurrent_state*.py' -v
bash -n scripts/hamilton_recurrent_state.sh scripts/submit_recurrent_state_from_mac.sh
```

有 PyTorch 的测试环境中，运行数值、兼容性与真实四组开发训练：

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_latent_state_norm.py' -v
PYTHONPATH=src python scripts/recurrent_state_study.py \
  --out runs/recurrent-state-development \
  --device cpu --profile development
```

输出目录须尚不存在。CI 在已有 Hamilton 版本兼容任务中使用 Python 3.13.9、NumPy 2.3.4 和 PyTorch 2.9.0+cpu；开发步骤显式采用 CPU。它保留视觉监督及诊断回归测试，执行四个真实 A512 单元，核对配对初始权重、事件计数、预算和 checkpoint 字节哈希，并重新加载选中模型确认归一化配置保留。

研究分支 `research/recurrent-state-stability` 的 CI 总门要求该兼容任务成功；任务被跳过不能替代通过。开发证据以 `recurrent-state-pytorch-2.9-cpu` artifact 保留 JSON 与哈希记录。具体通过数、CI 链接、最终源码 SHA 和正式作业号以执行后的真实回执为准。
