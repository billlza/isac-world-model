# 在新 Mac 上检查预测、求证与纠错

## 当前交付与验证边界

本阶段提供一个小规模的闭环研究实验：真实调用仓库已有 MIMO-OFDM 仿真和匹配估计器取得初始 RF 观测；随后执行相机横向位移，在独立的合成透视图像中提取新方位，再按观测实际到达时刻更新竞争假设。

新增网络从随机初始化训练，在物理几何预测上学习方位残差、观测尺度和一个小型 GRU 记忆。它使用提取后的方位量，**还没有联合训练原来的 RGB CNN、完整 RF 信号和多对象世界状态**。旧模型和旧检查点保持各自的数据契约，不能直接用来处理这个新实验。

MATLAB 有两个明确的执行部分：

- `matlab_phy_smoke.m` 在 MATLAB 内执行 OFDM CP/FFT、匹配搜索和距离/角度/Doppler 断言。
- `matlab_active_observation.m` 在 MATLAB 内交互展示 Python 已生成的真实实验记录。移动滑块或播放会更新候选几何、预先保存的预测、新证据、相对支持分数和消耗；不会重新训练网络。黑色观测叉只在 `available_s` 之后出现，评测真值不会画成模型已经知道的信息。

**目标环境是基础 MATLAB R2023b 或更新版本；不需要额外工具箱、Python、PyTorch 或 GPU 来打开已附带的回放。** 本机没有 MATLAB，MATLAB 执行和图形效果尚待新 Mac 验收，不能把 Python 测试通过当作 MATLAB 已通过。界面使用 MathWorks 文档中的 [JSON 读取](https://www.mathworks.com/help/matlab/ref/jsondecode.html)、[滑块](https://www.mathworks.com/help/matlab/ref/uislider.html) 和 [界面导出](https://www.mathworks.com/help/matlab/ref/exportapp.html) 接口。

## 新电脑上先做这三步

1. 安装并完成 MATLAB 自身的许可激活。仅为本展示，不必安装 Simulink、Deep Learning、Phased Array 或其他工具箱。
2. 解压提供的轻量展示包，或取得包含本次修改的 Git 分支。保留 `START_MATLAB.m`、`scripts/`、`examples/` 的相对目录结构。
3. 在 MATLAB 打开根目录 `START_MATLAB.m` 并点击 Run。原生 PHY 断言通过后会出现回放窗口。

也可在 MATLAB 的 Current Folder 进入仓库/展示包根目录后执行：

```matlab
run('START_MATLAB.m')
```

选择不同场景和方法，点击 Play / pause 或拖动滑块。Compare all methods 展示该场景全部测试 episode 的错误确认、弃答和资源占用，包括失败的方法。初始展示来自预先指定的第一个测试 seed，不按结果挑选。

第一次建议依次观察：`same`、`different`、`unresolvable`、`outside_candidates`、`camera_loss`、`late`。对比 `geometry_voi`、`learned_memory`、`learned_no_memory`、`fixed`、`random`、`always_unknown`。

## 在 MATLAB 中留下实际运行证据

在根目录执行以下命令，输出目录必须是新的：

```matlab
addpath('scripts')
matlab_phy_smoke()
report = matlab_active_observation_smoke( ...
    fullfile('examples','active_observation','matlab_replay.json'), ...
    fullfile('runs','matlab-validation-01'));
```

检查每个 episode、每个时间点后，生成 `matlab-report.json` 与实际界面截图 `matlab-replay.png`，记录 MATLAB 版本、平台、帧数和结果。失败会保存错误并重新抛出；失败记录不要覆盖。只有这一步在新 Mac 上成功，才能写“MATLAB 实际运行通过”。

脚本读取失败或界面报错时，保留完整错误标识和这两个目录的位置。不要通过忽略断言或更改证据文件来使演示继续。

## 实验契约

- 候选是假设的同深度对象与沿同一视线更远 7.5 m 的对象，不能据此断言材质、类别或现实身份。`outside_candidates` 的视觉对象更近，两个候选都错误。
- 感知动作只改变相机位置、观测时间和资源开销；目标按相同外生运动规律演化。预测输入只含估计 RF 锚点、过去已经到达的观测、候选动作和公开预算。
- 每次请求先保存预测，再执行图像渲染与方位提取；到达前不能改变推理状态。此小实验一次只允许一个待到达的相机请求，不等同于完整异步多传感器调度器。
- 物理可区分性门槛来自几何预测及声明的误差尺度，不能由神经网络缩小方差、编造预测差异来绕过。重复 UID 会报错；新 UID 的重复视角仍可能共享 RF 系统误差，当前经典似然积不是严格独立证据。
- `candidate_set_rejected` 是连续两次绝对预测残差越界的实验性门槛，**不是已校准的 OOD 检测器**。相对权重也不是已验证的身份概率。VoI 的两候选风险模型没有完整描述“候选都错”的价值，可能在一次观测后停止且保留未知，这属于需要报告的失败模式。
- 通信成本是**初始 OFDM 原始载荷容量重复使用、每次相机结果预留 1024 bit 的传输代理**，不含后续信道变化、重传、编码或真实相机压缩。报告实际占用与误差；相同预算上限不表示各方法消耗完全相同，不能据此宣称 ISAC 波形或真实吞吐量的优势。
- 整条 episode 按不相交 seed 划分训练、验证和测试。验证集选择网络权重；测试场景含不可辨识、候选失配、遮挡和晚到。小样本输出用于运行和机制检查，不构成统计显著性或安全证明。
- `always_unknown` 零测量且始终弃答，保留它是为了显示低错误率可能来自不做判断。报告也包含过程中曾出现的错误确认，不能用最终纠正掩盖早期错误。

## 在新 Mac 重新生成结果（可选，需 Python 环境）

展示不需要复制旧 Mac 的 `.venv`、`.toolchains`、缓存或大检查点。需要重新训练时，在新电脑的 Git checkout 中按项目的 `scripts/bootstrap.sh` 安装本地锁定环境，再运行：

```bash
bash scripts/env.sh python scripts/active_observation_study.py \
  --out runs/active-observation-new-mac \
  --train-episodes 64 --validation-episodes 16 --test-episodes 4 --epochs 12
```

输出包含完整 `study.json`、紧凑 `matlab_replay.json`、两个小型 `.pt`、校验和与包含源码/数据/运行时哈希的 `manifest.json`。输出路径已存在会拒绝覆盖，生成失败会留下失败 manifest 并返回错误。

然后在 MATLAB 打开新结果：

```matlab
addpath('scripts')
matlab_active_observation(fullfile('runs','active-observation-new-mac','matlab_replay.json'))
```

后续要检验的研究问题是：扩大独立测试场景和训练规模后，记忆是否改善条件预测、是否更快纠错，以及在匹配实际资源消耗时是否降低任务损失。当前展示不会预设这些结论。
