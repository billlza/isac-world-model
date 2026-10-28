# ISAC World Model：因果证据、视觉记忆与可反驳的环境假设（第一阶段）

**状态：已实现第一版可解释推理/跨模态证据记忆基线；不是通用视觉识别模型、材料反演器、受控车辆系统或已经学到语义的世界模型。**

## 为什么不是“看到反射就猜是什么”

无线回波受形状、尺寸、介电参数、极化、入射角、遮挡、多径和接收系统共同影响。单次距离—角度—多普勒估计一般不具有恢复材质或物体类别的可辨识性；看起来相似的反射峰也可能来自多个不同对象。二维合成相机的一处彩色亮斑也无法自动证明目标身份。

我们当前选择一个**可检查的中间里程碑**：多时刻证据维护；保留「同一目标」与「不同目标」两种相互竞争的解释；观测可用时间与采集时间分别记录；输出支持与反证来源；无法区分时输出未知，绝不把反射幅度硬译成物体类别。

## 实际数据链

1. `phy_ofdm.py`：同一组 MIMO-OFDM 发射符号用于双流通信及雷达感知；雷达输出从匹配搜索得出的距离、方向、径向速度（**不是**直接用仿真真值）。
2. `evidence_reasoning.from_phy_result()`：只读取 `.sensing` 字段，构造 `RFPoint`；忽略 `.truth_for_evaluation_only`，并明确**假定**误差尺度，不能声称校准完成。
3. 合成透视相机：单独从模拟目标状态生成 RGB 视角，用一个**刻意有限**的亮斑检测器提取方位线索，不输出类别、尺寸、真实深度。相机和无线观测来自同一个模拟场景，却通过独立的前向机制产生。
4. `CausalEvidenceMemory`：按数据**实际到达时刻**更新；同帧比对时检查采集时间差；记录事件 UID 防止重复证据计数；保留有界窗口，拒绝未来数据；过期一致性证据不再被当作当前有效认识。
5. 两个竞争假设：`H_same`（视觉和 RF 线索属于同一对象）、`H_other`（属于不同对象）。**单纯方位一致无法区分同一物体与同向不同深度的物体**，即使模型得分很高也只输出 `angularly_consistent_identity_unresolved`，`identity_confirmed=false`。角度残差的高斯模型与无关联均匀模型产生**未校准的相对证据分数**；单一观测不能触发“已确认”。
6. 导出结构化解释：相符/冲突帧的 UID、角度残差、证据比；估算当前径向距离及显式增长的误差尺度；**当前方位角运动无法从单次径向多普勒推出，所以保留未知**。

## 不确定性的边界

- `relative_same_object_score_uncalibrated` 是在规定分布假设下得到的**相对似然支持分数**，不是现实概率或安全认证的置信度。不同帧近似独立、观测误差高斯、无关对象方位均匀等假设需要验证。
- 单个 RF 匹配峰无法代表多个反射对象。当前对照只有两种关联假设，不能把 `conflicting_objects` 理解成已经确定有两辆车；它只是*现有测量下不应强制融合*。
- 不能通过系统缺乏观测、成像模糊或反射强来推断“金属、塑料、墙、人、动物、车辆”等身份。`object_category`、`shape`、`material` 固定为 `unknown`。
- 观测数据时间戳与可用时间分开；运行数据中的场景真值仅用于生成仿真和独立评估，不能进入推理器。
- 当前相机是**与旧 ToyScene 的正交俯视图不同**的合成透视相机，现有神经网络检查点不能直接接受它而不重新训练与校准。这个模块现为一个并列的可解释推理头，下一步才与可学习时序状态共享有凭据的特征。
- 当前 PHY 和新相机都没有经过真实传感器标定，未建模动态多目标、多径关联、真实复杂表面 RCS、材料频散、相机标定误差或恶意欺骗。

## 快速实验（项目已安装后）

```bash
# 两类观测一致：只有至少两次独立帧才可能支持方位一致；即便一致也无法确认是同一物体
bash scripts/env.sh python scripts/phy_reasoning_demo.py \
  --out runs/reasoning-compatible --scenario compatible --frames 4

# 视觉与雷达角度明显冲突：不把两个对象强行并成一个
bash scripts/env.sh python scripts/phy_reasoning_demo.py \
  --out runs/reasoning-conflict --scenario conflicting_view --frames 4

# 同一方位可能有两个不同距离的物体：证明不能凭角度确认身份
bash scripts/env.sh python scripts/phy_reasoning_demo.py \
  --out runs/reasoning-collinear --scenario collinear_view --frames 4

# 相机遮挡、视觉多个候选：不强行指定物体身份
bash scripts/env.sh python scripts/phy_reasoning_demo.py \
  --out runs/reasoning-occluded --scenario occluded --frames 4
bash scripts/env.sh python scripts/phy_reasoning_demo.py \
  --out runs/reasoning-ambiguous --scenario ambiguous_view --frames 4
```

每个输出目录含：`manifest.json`（参数、随机种子、源码 SHA-256 和模拟真值*仅用于评测*），`timeline.json`（入队前/实际到达后的决策变化），`sensor_observations.json`（仅估计所得物理/视觉量），`communication.json`（BER 和**未编码**载荷速率），`hypotheses.json`（假设与证据、明确的未知属性）。输出目录存在时拒绝覆盖。

要直接核实其核心性质：

```bash
bash scripts/env.sh python -m pytest tests/test_evidence_reasoning.py tests/test_phy_reasoning_demo.py
```

## 向具有学习与反事实验证能力的 World Model 推进

这项基线没有宣称“自己会思考”，但明确界定了**训练什么、证据怎样反驳模型、何时不允许给出答案**。下一阶段按下列逻辑推进：

- 建立真实或标定仿真的多频/多角/多极化和多视角图像数据，独立标注材质/形状/实例身份。**先验证可辨识性**，不把任意相关性误当因果解释。
- 将现有 `StructuredEventModel` 的视觉空间编码器和跨模态时序记忆与同一 PHY 数据分布重训练；训练独立的观测预测头，观察新证据能否否定先前假设。`H_same`/`H_other` 和更多散射来源假设可作为可解释监督/诊断头，但并非先验真值。
- 训练对**候选测量动作**的未来观测分布及到达时后验误差的预测。比较同样数据/预算下的 IMM、经典 VoI、无记忆学习模型、带记忆学习模型；报告危险虚假确认、弃答率、延迟和通信牺牲。
- 将“预测结果”与“执行动作”隔开，实测硬件延迟、标定误差和 OOD 时自动回退；自动驾驶、无人机等安全关键部署需要另外建立功能安全、验证与保护机制。

**原创方向不是把一个分类器和雷达相加，而是研究：在有限、异步且可能冲突的观测下，怎样形成可验证的环境假设、预测真正能区分假设的新观测，以及在认知失败时如何安全保留未知。**
