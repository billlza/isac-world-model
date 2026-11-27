# Independent text-to-video model

独立的文字条件视频流匹配模型，位于 `projects/text-video/`。目标是建立“文字输入 → 连续视频输出”的独立训练链路，以实际训练和采样结果判断后续规模。它使用新网络、新参数和新 checkpoint 格式；已有 ISAC / RF world model 的权重不进入此模型。

首轮实际 CPU 训练已完成 6,400 步，权重和视频均已生成；数值、可见效果和执行边界见 [2026-11-27 实验记录](RESULTS_2026-11-27.md)。

## 保留的训练设计与新模型

| 部分 | 本项目的实现 |
| --- | --- |
| 初始化 | 视频网络、文字编码器、文字字节表全部本地初始化；无外部 pretrained weights、VAE 或 tokenizer |
| 复杂场景 | 开发数据覆盖静态、两个方向移动、纵深变化、双物体、遮挡、升降；真实视频使用同一个 manifest 接口 |
| 可信参考 | 外部逐帧 `reference_weight` 进入损失；模型不能通过降低自身置信度逃避监督 |
| 过程监督 | 单独记录相邻帧变化的监督误差；不称作推理奖励或 RL |
| 训练阶段 | `pretrain` 从零开始；`sft` 继续本架构自身的预训练权重并建立新优化器 |
| 阶段状态 | RL、RLHF 尚未实现或执行，不把监督续训重新命名为 RLHF |
| 评估 | validation 选择 checkpoint；test 仅由显式训练后评估命令评分；跨阶段保持场景和内容 split 隔离 |
| 运行证据 | 源码指纹、数据指纹、配置、runtime、训练曲线、权重 hash、完整优化器和 RNG 恢复状态 |

已有三维 RF 实验的结论仍属于原任务。本项目继承训练和证据组织方式，不把原任务的 IoU 当作视频生成能力证据。

### 网络

`TextVideoModel` 是像素空间的条件 flow Transformer：

1. RGB 视频按空间 patch 编码；帧维度独立保留。
2. UTF-8 字节文字编码器提供条件，文字层也参与训练。中文可以编码，但中文语义能力需要对应的训练数据。
3. 每层分别做帧内空间注意力、同位置的跨帧注意力、文字交叉注意力和前馈变换。
4. 时间嵌入调制这些层，网络输出与视频同形状的速度场。
5. 推理从高斯噪声出发，用 Heun 积分生成整段视频，支持训练过空提示词条件的 CFG。

输出头初始为零以稳定优化，其余主干使用本地初始化。第一步先学习输出头，后续步骤梯度进入主干。完整片段内采用双向注意力，因此这是视频生成模型；它不声称是在线因果状态预测器。

当前实现没有音频生成、自动扩展长视频、视频压缩器、高清增强器或自回归续帧。后续增加这些能力应分别训练和验证。选择像素空间使第一版不依赖任何已有视觉权重，代价是分辨率扩展更昂贵。

### 目标函数

令 `x` 为参考视频、`z` 为独立高斯噪声，训练时：

```text
x_t = (1 - t) z + t x
target_velocity = x - z
x_hat = x_t + (1 - t) predicted_velocity
```

主损失是预测速度与 `x-z` 的逐帧加权均方误差。过程损失比较 `x_hat` 与 `x` 的相邻帧差分；它匹配真实变化，不把“所有帧静止”作为正确目标。相邻参考权重取两帧权重的最小值。梯度累积按整个有效批次的参考支持量归一化，帧项和帧对项分别计算。

训练包含带噪目标视频是 flow matching 的监督构造。采样阶段只接收文字和随机噪声，不读取目标视频。该目标可参考原始论文 [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)；本项目代码独立实现。

## 本地复现

从仓库根目录运行，以下路径均相对于仓库根目录。CPU 参考环境的直接依赖版本列在 `requirements-cpu.txt`；每次实际执行还会记录观察到的 runtime。

```bash
python -m pip install -r projects/text-video/requirements-cpu.txt
export PYTHONPATH="$PWD/projects/text-video/src"

python -m pytest projects/text-video/tests -q

python -m text_video make-data \
  --out projects/text-video/data/cpu-pilot-v1 \
  --train 256 --validation 32 --test 32 \
  --frames 4 --height 16 --width 16 --seed 7

python -m text_video train \
  --config projects/text-video/configs/cpu_pilot.json \
  --manifest projects/text-video/data/cpu-pilot-v1/manifest.jsonl \
  --out projects/text-video/runs/cpu-pilot-v1 \
  --device cpu --stage pretrain

python -m text_video sample \
  --checkpoint projects/text-video/runs/cpu-pilot-v1/selected.pt \
  --prompt 'A red sphere moves from left to right on a tiled floor.' \
  --prompt 'A blue cube moves toward the camera on a tiled floor.' \
  --prompt 'A green cone moves from left to right behind a gray pillar.' \
  --out projects/text-video/runs/cpu-pilot-v1/samples \
  --device cpu --seed 101

python -m text_video evaluate \
  --checkpoint projects/text-video/runs/cpu-pilot-v1/selected.pt \
  --manifest projects/text-video/data/cpu-pilot-v1/manifest.jsonl \
  --out projects/text-video/runs/cpu-pilot-v1/test \
  --device cpu
```

CPU pilot 为 16×16、4 帧、8 fps，即 0.5 秒。GIF 和 contact sheet 为最近邻放大预览，不代表高分辨率生成。MP4 使用原始尺寸，NPZ 保留编码前的准确 RGB 像素。每个样本记录提示词、seed、checkpoint hash 和采样参数。

输出目录禁止覆盖；重复实验请使用新目录。训练结果包括：

- `selected.pt`：按验证集总损失选择的权重。
- `final.pt`：最后一步权重。
- `resume.pt`：原模型、EMA、AdamW、数据抽样 RNG、噪声 RNG、全局 RNG、训练历史和原预算。
- 每个 checkpoint 对应 SHA-256 sidecar，加载前核验；原始 checkpoint 对外仍为 PyTorch weights-only 可读取格式。
- `config.json`、`provenance.json`、`history.json`、`summary.json`、`status.json`。

中断后可在原命令末尾追加 `--resume`，使用相同配置、源码、数据、runtime、阶段和初始化 checkpoint 恢复。完成任务不允许通过修改 `steps` 隐式延长；增加预算应创建新的显式阶段。CPU 同 runtime 的逐位恢复由真实故障注入测试覆盖；GPU 跨运行逐位一致没有由固定 seed 自动得到保证。

验证损失衡量固定噪声和时间点下的监督任务，不等于画面质量、提示词理解、物理一致性或通用视频生成排名。

## 数据接口

支持本地 `.npz`（`frames`，uint8 `[T,H,W,3]`）、MP4 和 GIF。每行 JSONL 必须包含：

```json
{"schema_version":1,"id":"clip-001","group_id":"episode-001","split":"train","caption":"A red sphere moves left.","path":"videos/clip-001.npz","sha256":"<actual 64-character sha256>","fps":8,"reference_weight":[1,1,1,1],"provenance":{"source":"original capture or dataset identity","license":"explicit license or permission basis"}}
```

`reference_weight` 长度必须等于源视频总帧数；每个实际使用的片段必须有正参考支持。读取前 T 帧，不插帧；空间尺寸只按配置显式缩放。模型当前没有 fps 条件输入，因此 manifest fps 必须匹配配置，时间重采样须在建 manifest 前显式完成。

数据检查覆盖：文件 SHA-256、RGB 格式、长度、文字字节上限、参考权重、来源声明，以及跨 train/validation/test 的 group、文件字节、解码像素重复。自身 checkpoint 还携带跨阶段 split 记录，监督续训不能把旧训练场景或内容变成测试样本。声明存在并不等于独立核实了其授权。项目不自动下载任何训练素材。

程序生成数据是本项目原创的 `synthetic-development` 数据，不是现实视频 benchmark；参考权重为 1，不能用这次开发实验声称已证明非均匀可信度机制的实际收益。该机制有独立数值和梯度测试。

## Hamilton 单 H200

配置提供三个明确预算：

| 配置 | 视频形状 | 模型 | 优化更新 |
| --- | --- | --- | ---: |
| `cpu_pilot.json` | 16×16，4 帧 | hidden 64，2 层，1 文字层 | 800 |
| `cpu_sft.json` | 同上 | 同上，从自身 checkpoint 监督续训 | 800 |
| `cpu_extended.json` | 同上 | 同上，继续自身监督权重；独立新 run | 4,800 |
| `h200_smoke.json` | 64×64，8 帧 | hidden 256，8 层，2 文字层 | 100 |
| `h200_development.json` | 同上 | 同上 | 20,000 |
| `h200_sft.json` | 同上 | 同上，从自身 checkpoint 继续 | 5,000 |

H200 配置为 BF16 autocast、batch 8、两次梯度累积；没有声称这些预算已在学校设备完成。先运行 100 步配置测实际吞吐、峰值资源和视频输出，再决定完整预算。不要从 CPU 时间推断 H200 完成时间。

Launcher 沿用已记录的 Hamilton `python/3.13.9cuda`、PyTorch `2.9.0+cu128`、NumPy `2.3.4` 和 CUDA `12.8`，请求 `cuda` 分区的一整张 `h200_nvl`，时间 4 小时。没有添加 CPU、内存或 account 请求。学校 module 是否包含 Pillow、imageio 和 imageio-ffmpeg 尚未在此次任务中观察，分配内会检查并记录，缺失将明确失败；脚本不会自动安装或切换环境。

在 Hamilton 的已审核、已提交且干净的 checkout 中：

```bash
module load python/3.13.9cuda
export PYTHONPATH="$PWD/projects/text-video/src"

python -m text_video make-data \
  --out projects/text-video/data/h200-development-v1 \
  --train 4096 --validation 256 --test 256 \
  --frames 8 --height 64 --width 64 --seed 7

# 用审核过的完整 40 位 commit SHA 替换下方占位符。
bash projects/text-video/scripts/submit_hamilton.sh --check \
  FULL_REVIEWED_COMMIT_SHA \
  projects/text-video/configs/h200_smoke.json \
  projects/text-video/data/h200-development-v1/manifest.jsonl \
  projects/text-video/runs/h200-smoke-v1 pretrain
```

`--check` 只完成检查和准备。同一命令改为 `--submit` 才实际调用 `sbatch`，并要求 Hamilton login host。提交器绑定源码、配置、数据内容和初始化 checkpoint，保留 `OUT.submission/` 的真实 scheduler receipt，并在分配内重新检查。重复提交有原子 claim；receipt 不明确时保留记录并停止，不自行重复提交。

监督续训将配置替换为 `h200_sft.json`，阶段改为 `sft`，最后增加自己的 `selected.pt` 路径。它允许新的、有来源的训练数据，但必须保持所有历史数据的 split 分配。

**当前 HPC wrapper 仅提交新任务，尚未实现恢复任务的独立 scheduler claim。** Trainer 的 `--resume` 已实现；在学校续跑仍需确认旧作业终止，并在明确的 GPU allocation 内使用原训练命令追加该参数。不要复用旧 receipt 或删除原 claim 来绕过重复提交保护。

## 结果解释与下一轮依赖

第一轮应先看实际神经网络样本：背景结构是否形成、物体是否可辨、文字改变是否产生对应变化、跨帧是否稳定。训练样本图与网络输出必须分别标记，不能把程序渲染的视频当作模型生成效果。

从这个开发域扩展到现实视频，需要足够的有来源图文视频数据与实际 GPU 训练。提高分辨率和时长时，可单独训练自有视频压缩器、扩展文字语料、引入长时间训练课程；这些是后续实现选项，不是当前已经完成的能力。当前项目不宣称复制 Sora 的内部架构或达到其画质。
