# 复杂场景随机初始化预训练

## 本轮训练目标

接续 Phase D 的源码 `30fe45f5837b6ed09fe739aa2371166a89b43fb9`，本轮改变此前神经训练集中主要使用单一简化场景的安排。复杂运动、遮挡、延迟、无线异常和外观变化现在进入训练 corpus，并保留等数据量、等更新次数和等选模次数的对照。

当前模型学习的是**合成单目标 RGB/RF 序列中的状态与观测预测**：位置、速度、不确定性和多时距未来状态。它没有语言解码器。复杂自然语言问答、多目标语义理解、真实图片重建不属于这轮成绩的评价对象。

历史模型权重没有复制进本轮；所有主模型从随机初始化开始。新训练的 nominal-control 是同一基础架构下的当轮对照，不等于直接重测 Hamilton 上保存的历史 checkpoint。

## 三组对照与数据

| recipe | 训练数据 | 视觉编码器 | 每轴加速度范围 | 位移辅助项权重 |
|---|---|---|---|---|
| nominal-control | nominal | 16 通道、1 层卷积 | ±0.5 m/s² | 0 |
| mixed-control | 11 类混合条件 | 16 通道、1 层卷积 | ±0.5 m/s² | 0 |
| mixed-enhanced | 与 mixed-control 完全相同 | 64 通道、3 层卷积 | ±6.0 m/s² | 0.1 |

全部采用 `structured-attention`、feature dimension 48、latent layer normalization、同一 AdamW 配置和相同 batch-order RNG。nominal-control 与 mixed-control 在同一初始化种子下具有完全相同的初始参数。增强模型新增参数会改变后续随机数抽样，因此它与对照是同种子、同预算比较，初始参数摘要分别记录。

11 类条件按固定顺序为：nominal、maneuver、delay、occlusion、radio_outliers、radio_bias、sparse_camera、maneuver_delay、occlusion_outliers、maneuver_bias、combined。训练 episode 循环分配条件，每个 epoch 再打乱采样顺序。除 nominal 外，混合数据与评估数据包含颜色变化、背景纹理和静态低对比度非目标 clutter。增强按实际 capture UID 固定；缓存帧不在每个时刻重新随机化。图像增强不读取真值、不补画被遮挡目标。

nominal-control 到 mixed-control 的变化包含场景与外观两部分；mixed-control 到 mixed-enhanced 的变化包含视觉容量、加速度表达范围和位移目标三部分。三组比较可以判断整套干预的效果，不能独立分解每个部件的增益。

| 配置 | train episodes | epochs | batch | hidden | 验证/测试每类 episode | 初始化种子 | 预测时距 |
|---|---:|---:|---:|---:|---|---|---|
| smoke | 4 | 1 | 2 | 32 | 1 / 1 | 7 | 0.3、0.6 s |
| development 默认 | 96 | 8 | 8 | 96 | 4 / 8 | 7 | 0.3、1.0、2.0 s |
| development 首轮扩展 | 176 | 12 | 8 | 96 | 4 / 8 | 7 | 0.3、1.0、2.0 s |
| formal | 3072 | 8 | 16 | 512 | 16 / 32 | 7、11 | 0.3、1.0、2.0 s |

smoke 的 4 条训练序列不能覆盖全部条件，只用于执行链路验证。development 每条序列 40 步，formal 每条 64 步；dt 均为 0.1 s，图像 24×24。formal 共 6 个 cell，每个 cell 1536 次参数更新。实际运行时间由硬件与日志确定，4 小时是调度上限。

数据 namespace 为 smoke 610M、development 620M、formal 630M；各自 train 从基址开始，validation 从基址加 1M 开始，test 从基址加 2M 开始。不同条件的验证/测试 seed 间隔 10,000。它们避开历史 20M 校准、30M 测试及此前研究的 episode 区间。所有 recipe 共享完全相同的 validation/test 计划；两个 mixed recipe 共享完全相同的 train 计划。初始化种子不改变数据。

## 标签可信度和目标函数

每条序列新增 `state_reference_variance[T,4]` 与 `state_reference_validity[T,4]`，分别表示外部参考误差方差和坐标有效性。这些字段只进入监督路径。传感器噪声、观测方差和模型自报置信度不被拿来替代参考质量。

本轮模拟器提供精确状态真值，因此实际训练 variance=0、validity=1。非零参考方差、失效坐标和 NaN 隔离已经有数值与梯度测试；本轮模型成绩只反映精确模拟参考下的训练，不是错误标注鲁棒性实证。系统性错误参考需要外部修正或失效标记，仅增加方差不能消除偏差。

损失分项为：

```text
state NLL
+ 0.5 × future NLL
+ 0.15 × observation NLL
+ 1.0 × capture-time visual MSE
+ process_weight × reference-weighted displacement MSE
```

状态与未来 NLL 把外部参考方差加到预测方差上，在有效坐标中计算。这个解释要求参考误差近似独立、零均值；已知有偏的标签应修正或排除。未来状态使用实际抽中的 horizon 对齐标签；每个 batch 在预定时距中等概率选一个，所有时距的实际更新次数写入 history。

位移项比较 `(future_mean_t - mean_t)` 与 `(reference_{t+h} - reference_t)`，仅使用两端均有效的坐标，按 `1/(1 + var_t + var_{t+h})` 调整。共同的状态偏差在位移中抵消；真实转弯保留在目标中，不施加匀速要求。它提供可核对的运动监督梯度，是相关辅助目标，没有实现自然语言思考步骤评分、错误根因奖励分配或 RLHF。各项按自身有效支持平均，原始项、加权项及支持数量都记录。

## 评估、选模和完成条件

每个完整 epoch 用验证集各条件的位置 RMSE 的算术平均选模。只在选模结束后评估 test。测试序列不出现在训练或选模中，但来自相同的预定义条件族，因此衡量这些复杂条件下的新轨迹泛化。未见条件族、真实传感器域和工业任务需要另外的数据。

结果报告当前状态的宏平均/逐条件 RMSE、逐条件 P95、0.3/1.0/2.0 s 预测 RMSE和坐标 90% 区间经验覆盖率。前 5 个 warmup tick 不进入这些位置指标。预测指标的起点也在 warmup 之后，且只比较序列内存在的未来标签。覆盖率描述实际覆盖情况，不是安全认证。

测试使用离线因果序列推理，保留观测采集/可用时间和 stale/fresh 语义。这里没有测量调度延迟、硬实时任务成功率或新增闭环感知策略；这些仍由原有 runtime/evaluation 实验单独衡量。全传感器 blackout 不被当成可以由训练恢复观测信息的成功样本。

每个 cell 保存：

- `model.pt`：仅由验证集选择的模型。
- `final.pt`：最后一次参数更新后的模型。
- `resume.pt`：完整 epoch 边界的参数、Adam 状态、RNG、历史和已选模型。
- 三个 checkpoint 的 `.sha256`、`manifest.json`、`history.json`、`metrics.json`、`episode_plan.json`。

`study.json` 列出全部 `planned_cells`、实际状态和完整文件摘要。完成校验要求每个预定 cell 都完成全部 epoch/更新预算，产物齐全、摘要正确、selected/final/resume 与历史和选模指标相符，执行源码没有在运行期间改变。分数好坏不决定运行状态；误差下降必须从实际指标另外判断。

## 本地运行与恢复

使用项目已经验证的运行环境：

```bash
bash scripts/env.sh python scripts/complex_scene_pretraining.py \
  --out runs/complex-development --profile development --device cpu --threads 2 \
  --episodes 176 --epochs 12
```

同一个输出目录默认拒绝覆盖。中断后，保持源码、runtime、配置和数据计划一致，使用同一参数加 `--resume`。这表示继续原定预算，不会无声增加 epoch。已有完成 cell 会校验产物后跳过。

```bash
bash scripts/env.sh python scripts/complex_scene_pretraining.py \
  --out runs/complex-development --profile development --device cpu --threads 2 \
  --episodes 176 --epochs 12 --resume
```

若首次完整 epoch 之前失败，还没有可恢复的 checkpoint，旧 cell 目录会移入相邻的 `<study>-interrupted/`，明确记录恢复事件，然后从该 cell 原来的随机种子重新训练。已有完整 checkpoint 损坏时会停止，不会跳过校验或自动降级重训。

推送 `research/complex-scene-pretraining` 会触发独立的 CPU development workflow：使用 Hamilton 对应的 Python 3.13.9、PyTorch 2.9.0 CPU 版和 NumPy 2.3.4，先测试再运行三组训练，保留权重及结果为 Actions artifact。它验证 CPU 兼容运行，不构成 H200/CUDA 执行记录。

## Hamilton 正式运行

在 Mac 的现有 Git 仓库中 fetch 本分支，使用发布时给出的完整 commit SHA 创建独立 worktree，再执行新 launcher。下面的 `EXACT_FULL_SHA` 应替换为经过验证的完整 40 位 SHA：

```bash
git fetch origin refs/heads/research/complex-scene-pretraining:refs/remotes/origin/research/complex-scene-pretraining
git worktree add --detach ../isac-complex-pretraining EXACT_FULL_SHA
cd ../isac-complex-pretraining
bash scripts/submit_complex_pretraining_from_mac.sh EXACT_FULL_SHA DURHAM_SHORT_USERNAME
```

launcher 传输完整 Git bundle，远端创建 `isac-pretrain-<SHA前12位>`，不依赖历史 checkpoint。它检查固定源码、登录主机与账号，使用原子提交声明防重复；只接受真实 `sbatch --parsable` 回执。不确定的提交状态保留现场供查询，不自动再次提交。

资源请求为 cuda 分区、1 张完整 H200 NVL、16 CPU、64 GiB RAM、最多 4 小时。计算节点先运行既有硬件预检，固定学校 Python 3.13.9 / NumPy 2.3.4 / Torch 2.9.0+cu128 / CUDA 12.8，然后启动 formal。

输出位于 `runs/pretrain-<JOBID>/`。必须同时检查 Slurm 的 `COMPLETED` 与 `ExitCode=0:0`，以及该目录的 `status.json`。提交回执只说明调度器接受任务。完整训练、源文件核验、包含全部权重的归档及归档重读校验成功后，顶层状态才为 completed。

归档包含 selected/final/resume 权重、逐文件 SHA256、运行记录、源码摘要和打包当时的状态快照。最终 `status.json`、`package.json` 与 `.tar.gz.sha256` 单独下载；launcher 输出具体命令。目录或回执已存在时保留原结果。

## 实现与验证入口

- [模型配置与视觉编码器](../src/isac_wm/structured.py)
- [场景与捕获一致的外观增强](../src/isac_wm/pretraining_data.py)
- [参考质量与分项目标](../src/isac_wm/pretraining_objective.py)
- [训练、选模、恢复与完成核验](../src/isac_wm/pretraining.py)
- [命令行入口](../scripts/complex_scene_pretraining.py)
- [Hamilton 提交入口](../scripts/submit_complex_pretraining_from_mac.sh)
- [含权重的归档核验](../scripts/package_complex_pretraining.py)

局部测试覆盖旧配置的逐位兼容、真实梯度、未来信息隔离、时间对齐、参考方差与失效坐标、种子隔离、完整 epoch 精确续训、首 epoch 前失败恢复、错误完成声明、提交防重复和真实归档完整性。真实 CPU 训练日志与模型成绩以运行产物为准；mock 调度测试不计作 Hamilton 提交。
