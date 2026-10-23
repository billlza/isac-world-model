# 第一版说明

这是可训练、可回放、可做因果输入检查的初始研究代码，不是现成模型换皮。
视觉编码器、无线编码器、模态记忆、时序动态与输出头均从随机初始化训练。
不使用 World-Mirror、DINO、JEPA 的权重或教师输出。

第一版选择一个二维移动目标和固定感知节点，以控制模型、时间与资源问题。
图像是合成俯视 RGB 图；无线侧是带噪距离、方位、径向速度的处理后特征。
它不是实际相机三维重建，也没有伪称已实现原始回波、完整物理 ISAC 或量子机制。

## 已实现

- 不同采集／到达时间、延迟与丢测；严格按可用时间释放证据。
- 自主网络与完整训练、保存、加载、独立测试流程。
- 跨整条轨迹的数据划分；验证未来样本和监督标签不会泄漏进当前估计。
- 模态独有记忆＋共同状态；显式时间步长；环境动态与感知动作分开。
- 当前位置／速度与未来状态的均值、方差输出；动作条件观测预测。
- 经典重放 Kalman 基线；固定测量与预测预算策略的 2×2 对照。
- 明确共享时间资源和通信机会成本；保留负结果与运行时间。
- Rust 原生事件队列及单元测试源码；当前 Python 主程序尚未接入该原生队列。

## 不能提前声称

不确定性尚未校准，不是安全保证。调度器目前是透明的近似启发式，
不是已经训练成功的最优观测价值网络。计算时间被测量、记录，但还未推进
仿真时钟，因此不能把本轮速度数字当作端到端实时保证。

当前运行环境只有 CPU；本次没有在 Mac MPS 或 NVIDIA CUDA 上实测。
Rust 编译器是否可用以及原生测试实际运行情况，以 VALIDATION.md 为准。

## 使用

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
isac-wm train --out runs/train --epochs 24 --episodes 192 --device auto
isac-wm evaluate --checkpoint runs/train/model.pt --out runs/eval --episodes 8
```

打开 `runs/eval/replay.html` 可以逐帧查看估计、真值、观测年龄、动作与队列。
所有实验基线结果都会保留，不会只输出神经网络获胜的例子。

## GitHub

建议先创建私有仓库，不把未公开研究默认开源。
发布脚本使用你电脑上已经登录的 GitHub CLI，不索取或嵌入令牌：

```bash
bash scripts/publish_private.sh billlza/isac-world-model
```

本地完成与远端推送是两回事；是否实际上传，以交付说明和 GitHub 远端为准。
