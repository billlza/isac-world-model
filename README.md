# ISAC World Model — v0.2.0

独立设计、随机初始化的视觉—无线世界模型研究原型。主模型不依赖 DINO、JEPA、World-Mirror 等预训练权重或教师模型。研究目标是异步观测下的状态预测、观测价值与通信资源取舍。Quantum尚未建模。

**本版已实际完成：87项Python测试、从零训练、独立测试、故障扫描、CPU微基准与浏览器回放验证。Rust/CUDA/MPS尚未实测。**

详细结果与失败见 [`docs/VALIDATION_v020.md`](docs/VALIDATION_v020.md)。未隐藏经典基线更准、实验策略失效或接受比例为零的结果。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
isac-wm doctor
pytest
```

最低Python3.11。此版本在Python3.13.5、PyTorch2.10.0+cpu、NumPy2.3.5上实测。`--device auto`选择实际可用的CUDA/MPS/CPU，明确指定不可用设备会报错，不会冒充GPU执行。

## 运行附带检查点

```bash
isac-wm evaluate --checkpoint checkpoints/v020.pt   --calibration checkpoints/calibration-v020.json   --out runs/my-evaluation --episodes 8 --threads 2
```

打开输出目录的 `replay.html`。它是自包含本地文件，支持暂停、速度、拖动、选择实验、误差曲线和状态诊断。`checkpoints/development.pt`为v0.1旧检查点，保留兼容性测试，勿与新检查点的校准文件混用。

## 自己从零训练与恢复

```bash
isac-wm train --out runs/train-a --episodes 192 --epochs 24   --batch-size 8 --features 48 --hidden 96 --threads 2

# 恢复到总计40轮；必须使用新输出目录和匹配的原训练参数。
isac-wm train --resume runs/train-a/resume.pt --out runs/train-b   --episodes 192 --epochs 40 --batch-size 8 --features 48 --hidden 96 --threads 2
```

`--patience`启用早停；`--max-dataset-gb`限制估算的数据集分配。续训保存优化器、随机状态、当前与最佳模型；本轮CPU小任务恢复与连续训练逐参数一致，不承诺跨设备位级一致。

## 校准与故障扫描

```bash
isac-wm calibrate --checkpoint runs/train-a/model.pt   --out runs/my-calibration.json --episodes 32 --alpha .1 --threads 2

isac-wm evaluate --checkpoint runs/train-a/model.pt   --calibration runs/my-calibration.json --out runs/test --episodes 8 --threads 2

isac-wm stress --checkpoint checkpoints/v020.pt   --out runs/fault-tests --episodes 4 --threads 2

# 显式实验选项：未校准方差风险策略已观察到失效，不作为默认策略。
isac-wm evaluate --checkpoint checkpoints/v020.pt --out runs/experimental   --policies fixed,predictive,risk,guarded --episodes 8 --threads 2

python scripts/benchmark_local.py --out runs/benchmark-new.json
```

校准按独立轨迹而不是逐帧划分；指纹、场景、策略和数据分割不匹配会拒绝使用。故障扫描不会把正常场景校准证书套到分布外场景。监测是有效性检查，不是车辆或真实设备安全认证。

## 模型与运行语义

- 视觉CNN和无线MLP从零初始化；模态记忆、共同状态、显式时间跨度动态、状态分布和动作条件观测预测。
- 测量动作不改变目标真实运动。
- `capture_s`描述观测时刻，`available_s`描述数据可用时刻。未到达和重复证据不可进入估计器。
- 神经快照保留最新采集时刻的模态观测；老包被记录为忽略。卡尔曼参考则按全部已交付观测重放，因此两者保留证据的机制不同。
- 缓存卡尔曼只重放受乱序插入影响的后缀；重复查询不重复注入过程噪声。
- 整序列训练时批量编码新的观测，不反复编码相同旧图像；保持时间因果性。
- `fixed`每周期请求quick；`predictive`是旧代理；`risk`是阈值启发式；`guarded`给risk加入无线证据新鲜度约束。均非已训练的VoI网络。
- 感知使用共享时隙预算。通信量是AWGN时间分配代理，不是实测链路吞吐率。

## 工程边界

输入检查、数据上限、原子产物、校验和、超时与数据陈旧监测均已加入。仅加载可信检查点；本地校验和不提供来源认证。监测不访问真值。推理耗时只被记录，尚未进入模拟时钟排队。

训练种子、校准种子、测试种子分别记录，整轨迹划分。首次5步保留在回放中但不计入汇总；所有方法一致。

## 目录

`src/isac_wm` 模型/仿真/估计/训练/校准/监测/CLI；`tests` Python测试；`crates/event-core` 原生队列；`docs` 设计与验证；`checkpoints` 本地分发检查点；`runs` 实际实验产物（不提交Git）；`validation` 执行记录。

## GitHub

本次没有远端推送。源码包配套Git bundle保留提交历史；`scripts/publish_private.sh`只在用户本机已认证、目标不存在时创建private仓库。检查点与原始运行产物被.gitignore排除，必要时另行作为受控产物存储，不要把完整原始数据塞入Git历史。

```bash
bash scripts/publish_private.sh billlza/isac-world-model
```

不得把第三方受限代码、未授权权重或内部实验数据加入仓库。研究成果所有权与公开发布按学校/合作方约定确认，见 `RIGHTS.md`。
