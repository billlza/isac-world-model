# Phase C：固定稳定配方的模型容量实验

本轮比较 A512、A1024、A2048，检验已经取得收益的递归状态 LayerNorm 配方在更大模型上的表现。六个单元全部重新随机初始化；在本轮内计算配对差值。实验固定数据暴露量、优化器更新次数、观测事件和验证机会，扩大隐藏状态及其关联模块的容量。

## 1. Phase B 已经完成的结果

Hamilton 作业 `19404348` 完成四组正式训练，Slurm 为 `COMPLETED 0:0`，最终阶段为 `completed/done`。源码为 `841142a8bb59fd0bb8c67156fd74ec125955e2f3`；归档 `isac-state-19404348.tar.gz` 的 SHA-256 为 `44f10ba6c4fa8204fd4c02a5fa9aaa5eedbbbe8af01abf4be7707c8a127ffacc`。归档成员与最终状态已经核对。

| 初始化 | 无状态归一化：末次融合 RMSE | LayerNorm：末次融合 RMSE | 相对下降 | LayerNorm：末次视觉诊断 RMSE |
|---|---:|---:|---:|---:|
| 7 | 0.341885 m | 0.259641 m | 24.06% | 0.041114 m |
| 11 | 0.411931 m | 0.272337 m | 33.89% | 0.052601 m |

四组的最佳完整验证结果均出现在第 384 次更新。两个初始化的融合 RMSE 算术均值从 0.376908 m 降至 0.265989 m，下降 29.43%；这不是将所有轨迹误差合并后重新计算的 pooled RMSE。融合指标覆盖完整的 192 条验证轨迹；视觉诊断覆盖固定四条轨迹中的 101 次可见图像捕获。两种指标保留不同口径。

这组配对结果支持在后续容量实验中固定使用 LayerNorm。Phase A 与 Phase B 的未归一化重复运行存在数值轨迹差异，因此本轮重新训练 A512，将历史结果作为参照而不代替同轮对照。Phase B 的完整协议见 [RECURRENT_STATE_STUDY.zh-CN.md](RECURRENT_STATE_STUDY.zh-CN.md)。

## 2. 六个正式单元与精确参数量

| 单元 | hidden dim | 初始化 | 总参数量 |
|---|---:|---:|---:|
| `a512-layernorm-init-7` | 512 | 7 | 3,883,798 |
| `a512-layernorm-init-11` | 512 | 11 | 3,883,798 |
| `a1024-layernorm-init-7` | 1024 | 7 | 15,103,766 |
| `a1024-layernorm-init-11` | 1024 | 11 | 15,103,766 |
| `a2048-layernorm-init-7` | 2048 | 7 | 59,563,798 |
| `a2048-layernorm-init-11` | 2048 | 11 | 59,563,798 |

固定 `feature_dim=48`、`memory_size=16`、`heads=4`，总参数量为

\[
P(h)=14h^2+410h+3862.
\]

最大一档为 **0.059564B**，约为 A512 的 **15.34 倍**。循环状态、融合、注意力、动力学和读出等宽度相关部分随之扩大。空间视觉编码器保持 **1,230 参数**，图像尺寸保持 24×24，视觉特征维度和记忆长度固定。因此本轮回答的是状态与融合容量的收益，视觉编码器本身的容量另有定义。

三档均使用参数免费的完整 hidden 更新 LayerNorm：`eps=1e-5`、`elementwise_affine=False`，应用于有新观测和预测两条 hidden 更新路径；物理位置、速度和协方差不直接归一化。保持全局梯度裁剪阈值 1.0、AdamW 与已有数值精度；学习率固定 `3e-4`，直接视觉监督权重固定 `1.0`。

相同初始化种子的三个宽度具有相同的初始视觉编码器指纹。完整模型的形状不同，完整指纹必须不同；种子 7 和 11 的初始化指纹也必须不同。视觉编码器先于宽度相关模块构造，因此无需额外复制权重或改变初始化规则。

## 3. 数据、更新预算和开发数据隔离

| 项目 | 正式 Hamilton 训练 | CPU 开发执行 |
|---|---:|---:|
| 宽度 | 512 / 1024 / 2048 | 512 / 1024 / 2048 |
| 初始化 | 7 / 11 | 7 / 11 |
| 训练轨迹数 | 1536 | 8 |
| 验证轨迹数 | 192 | 8 |
| batch size | 8 | 8 |
| 优化器更新数 | 384 | 4 |
| 完整 epoch 数 | 2 | 4 |
| 完整验证 update | 96 / 192 / 288 / 384 | 1 / 2 / 3 / 4 |
| checkpoint 选择机会 | 4 | 4 |
| 训练种子起点 | 700000 | 89130000 |
| 验证种子起点 | 74611000 | 89140000 |
| batch order seed | 37 | 37 |
| fresh test | 保留 | 保留 |

所有正式单元使用相同的 64 tick 轨迹、`dt=0.1 s`、训练及验证种子列表与批次顺序。相同更新数意味着相同训练样本暴露量；更大模型每次更新的计算量和运行时间可以不同，本轮不宣称等 FLOPs。

内部诊断位置为 `0,1,2,4,8,16,32,64,96,192,288,384`；开发执行使用 `0,1,2,3,4`。诊断与完整验证分开计数，内部快照不增加模型选择机会。正式和开发都执行真实的三档模型；开发缩短的是数据和更新预算。

视觉监督对应实际接受图像的采集时刻，仅 fresh、valid、visible 图像提供直接定位标签。在线输入不增加目标真值或未来观测。六组实际看到的视觉、无线、证据更新与纯预测事件数必须一致。

## 4. 评价和数值证据

主要报告每个初始化下的 **A1024 − A512** 与 **A2048 − A512** 差值，以及两个初始化的描述性均值。误差差值为负表示改善。报告项目包括：

1. 四次完整验证中的最佳融合位置 RMSE，以及固定终点 RMSE；
2. 末次视觉诊断 RMSE，保持可见捕获口径；
3. 末次与最佳值的差、后半段变化及最佳 checkpoint 所在 update；
4. 梯度、实际参数更新、归一化前后 hidden 尺度、输出饱和和测量校准诊断；
5. 实际训练时间、GPU peak allocated / reserved 内存、模型参数量。

六组全部保留，不因某档未改善而删除结果。较大容量的结果用于选择当前配方下表现与成本合适的模型；本轮不同时搜索学习率、视觉编码器或训练预算。

保持已有 CUDA 数值策略，记录实际 deterministic algorithms、cuDNN deterministic / benchmark、TF32、float32 matmul precision 与 `CUBLAS_WORKSPACE_CONFIG` 等设置；六组设置必须一致。固定种子和初始化哈希确认初始条件，运行时记录帮助解释数值差异。这里不新增强制确定性模式，也不声称 GPU 训练逐位一致。

## 5. 检查点、设备与作业范围

正式作业申请 `cuda` 分区的 **一张完整 H200**，六组顺序执行；时间申请上限为 **4 小时**，实际耗时在结果中记录。现有 preflight 核验完整 H200、分配、CUDA 运算和学校运行时：Python 3.13.9、NumPy 2.3.4、PyTorch 2.9.0+cu128、CUDA build 12.8。

模型与批次不在运行中自动缩小，设备不自动替换。资源或运行时不满足时，失败状态与已完成证据保留。

约 59.56M 参数的 float32 最佳权重文件仍在原有 256 MiB 模型检查点限制内。续训文件同时包含当前权重、最佳权重和优化器状态，因此使用单独的 **2 GiB** 有界读写接口；普通模型检查点继续使用 256 MiB 上限。读取保留字节哈希、文件大小、ZIP 展开大小及 `weights_only=True` 检查。每组保存最佳 `model.pt` 与完整 `resume.pt`，本轮正式训练始终从随机初始化开始。

## 6. 从 Mac 提交

提交器为 [submit_normalized_capacity_from_mac.sh](../scripts/submit_normalized_capacity_from_mac.sh)，Slurm 入口为 [hamilton_normalized_capacity.sh](../scripts/hamilton_normalized_capacity.sh)。提交器创建自包含 Git bundle，在 `$HOME/isac-scale-<源码前12位>` 建立独立 checkout，并将回执保存在 `.scale-submission`。

取得已通过 CI 的完整提交后，在 **Mac 原项目目录** 执行；`PUBLISHED_FULL_SHA` 替换为公布的 40 位提交：

```bash
(
  set -e
  set -o pipefail
  cd "$HOME/Desktop/isac-world-model"
  scale_pin=PUBLISHED_FULL_SHA
  scale_ref=refs/remotes/origin/research/normalized-capacity-h200
  git fetch origin "refs/heads/research/normalized-capacity-h200:${scale_ref}"
  if [ "$(git rev-parse "$scale_ref")" != "$scale_pin" ]; then
    echo '分支与已验证版本不一致，停止提交。'
    exit 1
  fi
  git show "${scale_pin}:scripts/submit_normalized_capacity_from_mac.sh" |
    bash -s -- "$scale_pin" bhmq03
)
```

保留 `${scale_pin}:path` 的大括号，避免 zsh 冒号变量修饰。学校密码只在 SSH 提示中输入。重复执行优先使用已有回执；提交结果不明确时先检查保存的回执与队列，避免重复作业。

## 7. 状态、结果取回与完成判据

提交成功后，脚本打印实际作业号、日志、结果目录和下载命令。作业名为 `isac-scale`，结果目录为 `runs/scale-<job>`。以实际回执为准，代码准备完成本身不代表作业已经提交。

```bash
sacct -X -j <job> --format=JobID,State,ExitCode,Elapsed
squeue -j <job> -o "%.18i %.12T %.10M %R"
tail -n 40 "$HOME/isac-scale-<prefix>/slurm-<job>.out"
cat "$HOME/isac-scale-<prefix>/runs/scale-<job>/status.json"
```

等待 Slurm 为 `COMPLETED 0:0`，且最终状态为 `completed/done` 后，取回 `isac-scale-<job>.tar.gz` 与最终 `status.json`。打包器验证六组协议、事件数、数值策略、检查点元数据、实际文件哈希与重算汇总。证据包包含结果、历史、诊断、清单和哈希，明确 `weights_included:false`；大体积权重保留在服务器。

包内 `job_status_at_packaging.json` 是打包时的状态快照；最终状态来自另行下载的 `status.json`。研究结论在正式六组结果返回后分析；开发测试负责验证训练、保存、加载和证据流程。

## 8. 开发检查

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_normalized_capacity*.py' -v
bash -n scripts/hamilton_normalized_capacity.sh scripts/submit_normalized_capacity_from_mac.sh
PYTHONPATH=src python scripts/normalized_capacity_study.py \
  --out runs/normalized-capacity-development --device cpu --profile development
```

CI 在与 Hamilton 相同版本的 PyTorch 2.9.0 CPU 构建中运行真实六组开发训练，并重新加载选中模型核对参数量、LayerNorm 配置、训练预算与数据种子。Hamilton 作业负责实际 CUDA 分配和正式训练证据。
