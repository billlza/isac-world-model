# Phase D：大模型优化与逐轨迹误差核查

## 1. 本轮依据与目标

Phase C 已完成容量筛选，Hamilton 作业为 `19404810`，源码为
`249407dbbe4039f68d7cd54c6cc336a3f9c96f06`。六组正式训练、验证、配对种子、
实际观测次数和检查点记录已核对。归档 `isac-scale-19404810.tar.gz` 的 SHA-256 为
`308f25886a082f73c7fc672407047fc62dbedaf441202fa3303f5ea207a9febb`。

| 隐藏宽度 | 参数量 | 初始化 7 最佳融合 RMSE | 初始化 11 最佳融合 RMSE |
|---|---:|---:|---:|
| A512 | 3,883,798 | 0.266762 m | 0.395395 m |
| A1024 | 15,103,766 | 0.430206 m | 0.445036 m |
| A2048 | 59,563,798 | 0.443239 m | 0.443981 m |

这些是完整 192 条验证轨迹上的当前二维位置 RMSE。最佳检查点分别从四次相同预算的
验证中选出。**Phase C 已显示原配方下加宽没有改善最佳验证误差。** 本轮继续训练
59.56M 参数的 A2048，并针对已有的优化诊断做单因素干预。

已观测到的首步参数相对更新 `||Δθ|| / ||θ||`，两个初始化的均值在 A512 为
0.8647%，A2048 为 1.7172%。大宽度还出现较大的注意力残差及加速度、噪声输出头饱和。
这些现象支持检验宽度相关模块的步长；它们不单独证明学习率是唯一原因，也不证明减半最优。
因此保留原 A2048 配方作为同轮对照，不同时改变架构、损失、梯度裁剪、数据量和视觉输入。

另一个需要定位的结果来自 A512/init-11：其末次视觉诊断 RMSE 为 0.052554 m，
末次完整融合验证 RMSE 为 0.480299 m。前者仅覆盖四条轨迹的 101 次可见图像捕获，
后者覆盖 192 条轨迹；诊断使用逐步编码，完整验证使用批量编码。
本轮用**同一检查点、同一批轨迹和相同批尺寸**分开比较编码路径和批尺寸，
不把不同样本支持集的两个数直接相减归因。

## 2. 已登记的六组训练

| 组别 | 参数量 | 宽度相关模块学习率 | 视觉、RF 编码器和动作嵌入学习率 | 初始化 |
|---|---:|---:|---:|---|
| A512 原配方 | 3,883,798 | 3e-4 | 3e-4 | 7、11 |
| A2048 原配方 | 59,563,798 | 3e-4 | 3e-4 | 7、11 |
| A2048 宽度相关步长减半 | 59,563,798 | 1.5e-4 | 3e-4 | 7、11 |

“宽度相关模块”由逐个命名参数的完整分组定义，包含 recurrent fusion、dynamics、
attention、事件/时间投影、相关归一化、加速度/噪声和预测观测输出头。
固定宽度组为 `visual`、`radio_encoder`、`action_embedding`，共 3,846 个参数，
其中视觉编码器仍为 1,230 个参数。遗漏或重复分组会报错。分组、实际学习率、参数数量、
名字及结构指纹写入 manifest 和两类检查点。

当 `width_lr_scale=1.0` 时，保留原单参数组 AdamW 的构造与顺序。
候选组使用两个显式参数组；AdamW 的 `weight_decay=1e-4` 和全局裁剪阈值 1.0 不变。
相同初始化下，两种 A2048 策略的**全部初始权重必须一致**；跨宽度的视觉初始权重也必须一致。
六组不会有六个不同的全模型初始指纹：相同 A2048 初始化的两个策略故意配对，
共有四个不同的宽度/初始化组合。

| 固定条件 | 正式 Hamilton 训练 | CPU 开发执行 |
|---|---|---|
| 真实模型 | 上表六组 | 同样六组，保留完整 A2048 |
| 轨迹长度、图像 | 64 ticks、24×24 | 相同 |
| 训练轨迹 | 1,536，种子 700000 起 | 8，种子 89150000 起 |
| 验证轨迹 | 192，种子 74611000 起 | 8，种子 89160000 起 |
| Batch / minibatch 排序种子 | 8 / 37 | 相同 |
| 更新次数 | 384，2 epochs | 4，4 epochs |
| 完整验证位置 | 96、192、288、384 | 1、2、3、4 |
| 视觉监督权重 | 1.0 | 1.0 |
| Recurrent LayerNorm | 完整 hidden 更新后，affine=false，eps=1e-5 | 相同 |
| 初始化 | 全部随机初始化，各组重新训练 | 相同 |
| fresh test | 不读取 | 不读取；开发种子与正式种子分离 |

正式训练仍申请一张完整 H200 NVL，时限 4 小时，使用 Hamilton 已验证的
Python 3.13.9 / NumPy 2.3.4 / PyTorch 2.9.0+cu128。不静默退回 CPU、MIG、小模型或小 batch。
计算预算匹配更新数与数据暴露，不声称各组 FLOPs 相同。

## 3. 对旧检查点的只读重放

原运行保留在：

`$HOME/isac-scale-249407dbbe40/runs/scale-19404810`

本轮先读取该目录的正式状态、study、manifest、训练和诊断记录，再重放
`a512-layernorm-init-11/model.pt` 与 `resume.pt`：

- `model.pt` 是 update 192 的最佳权重。
- `resume.pt` 中顶层 `state_dict` 是 update 384 的末次权重。
- `resume.pt` 内嵌的 `best_checkpoint` 不作为末次权重。

旧 `study.json` 固定 SHA-256：
`b9ecba054a7e2ddb1f26bca8312f751160c38236e7988d871ac305c7679ba2ee`。
旧 manifest 固定 SHA-256：
`bacd15d358823c7655dd7052b5b4cbdb7fff520968fd81be782acedba3211c08`。
旧最佳权重固定 SHA-256：
`35a24d691085748867be24763ad3e95f62696e97714bf3cd27dfea0ca9fa766f`。

末次 resume 文件的摘要未包含在 Phase C 的下载归档中，本轮将首次记录其服务器 sidecar
与实际字节摘要，并核验该文件的协议、更新数和真实 tensor。该证据边界会保留在结果中。
模型文件经受限、带 SHA 校验的 weights-only 读取；最佳模型上限 256 MiB，续训文件显式上限
2 GiB。原文件不改动，重放不进行优化器更新。旧目录或权重缺失时会明确失败，不替换成重新训练的模型。

先比较完整 192 条轨迹上批量编码得到的 RMSE 与原记录，然后比较：

1. 完整 192 条轨迹，batch=8，逐步编码与批量编码。
2. 前四条相同轨迹、batch=4，仅改变编码路径。
3. 相同的前四条轨迹和批量编码，比较 batch=4 与包含它们的 batch=8。

比较记录每个 tick 的输出差异、首个超出登记数值容差的位置和误差分布。
数值容差只用于定位计算差异，不作为物理安全阈值。发现路径差异本身作为研究结果保留；
文件完整性、非有限值和协议错误才中止执行。

## 4. 误差目标如何进入本轮评价

六组新模型的**最佳与末次检查点**都执行上述两条完整验证路径，并输出：

- 每条轨迹、每个评价 tick 的二维位置欧氏误差；整体和每条轨迹的 RMSE。
- P95、P99、最大误差与超过 0.1、0.25、0.5、1.0 m 的次数及比例。
- 同一检查点的可见图像捕获误差，避免把“最佳融合成绩”与“末次视觉成绩”拼成一组结果。
- 可见性、图像年龄、RF 新鲜度等分层，以及修正幅度与测量方差诊断。
- 初始化 7、11 各自的候选减对照差值；保留改善和退化，不只报均值。

所有真值只在预测返回后用于评估。在线模型输入保持不变。
误差分位数是相关序列上的描述性统计，不当作独立样本置信界。
这组固定超差档位用于暴露大误差，不等于已经定义或验证了设备操作限值。
最佳检查点选择仍按原融合 RMSE，当前不据尾部结果重新选 checkpoint。

平均定位精度和偶发大偏差都会影响后续工程决策。本轮的实施交付是优化策略与可追踪的
大误差证据；降低误差是否成功，必须由正式配对实验返回的数据回答。
真实设备的速度、制动距离、遮挡范围及允许误差决定实际动作边界，当前实验不添加自主物理控制，
结果明确记录 `deployment_validated:false`。

## 5. 提交与取回

Mac 入口为 [submit_width_optimization_from_mac.sh](../scripts/submit_width_optimization_from_mac.sh)，
Slurm 入口为 [hamilton_width_optimization.sh](../scripts/hamilton_width_optimization.sh)。
自包含 Git bundle 部署到独立目录 `$HOME/isac-optim-<源码前12位>`，回执保存在 `.optim-submission`。
旧 Phase C 目录只读。提交脚本检查已保存回执，避免网络中断后的重复提交。

以下示意命令中的 `PUBLISHED_FULL_SHA` 须替换为实际通过核对的完整发布提交；
聊天中的执行命令会给出该具体值：

```bash
(
  set -e
  set -o pipefail
  cd "$HOME/Desktop/isac-world-model"
  optim_pin=PUBLISHED_FULL_SHA
  optim_ref=refs/remotes/origin/research/width-optimization-h200
  git fetch origin "refs/heads/research/width-optimization-h200:${optim_ref}"
  if [ "$(git rev-parse "$optim_ref")" != "$optim_pin" ]; then
    printf '%s\n' '分支版本与已验证提交不同，停止提交。' >&2
    exit 1
  fi
  git show "${optim_pin}:scripts/submit_width_optimization_from_mac.sh" |
    bash -s -- "$optim_pin" bhmq03
)
```

保留 `${optim_pin}:path` 的大括号，避免 zsh 冒号变量修饰。学校密码由用户在 SSH 提示中输入。
作业名为 `isac-optim`，结果为 `runs/optim-<实际作业号>`，只有成功 Slurm 回执才代表已经提交。
等待 Slurm 为 `COMPLETED 0:0` 且最终 `status.json` 为 `completed/done` 后，
按提交器打印的路径取回压缩包和最终状态。

证据归档包含两类权重的摘要、旧来源证据、六组训练/诊断记录、全部重放结果与重算对比。
大体积 `.pt` 权重留在 Hamilton，归档明确 `weights_included:false`。

## 6. 开发验证

```bash
PYTHONPATH=src python -m pytest tests/test_optimizer_policy.py tests/test_checkpoint_replay.py tests/test_width_optimization_study.py tests/test_width_optimization_job_contract.py -q
bash -n scripts/hamilton_width_optimization.sh scripts/submit_width_optimization_from_mac.sh
PYTHONPATH=src python scripts/width_optimization_study.py \
  --out runs/width-optimization-development --device cpu --profile development
```

CPU 开发执行覆盖全部六组实际宽度、四次更新、最佳和末次权重持久化、逐轨迹重放与配对校验。
它不读取学校旧目录，也不产生正式精度结论。Hamilton 兼容 CI 使用 PyTorch 2.9.0+cpu，
研究分支的 CI 总门要求该任务成功，不能以跳过替代通过。
