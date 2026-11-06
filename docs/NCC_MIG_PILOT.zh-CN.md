# NCC：A512 在 10 GB MIG 上的开发短测

## 目的与已知资源

2026-11-06 的用户调度快照显示：`gpu13` 的 4 张 Hopper、`gpu10/12` 的
10 张 Ampere 整卡均已分配；`gpu10` 的 28 个 `1g.10gb` 中用了 8 个，
`gpu12` 的 42 个中用了 9 个。53 个尚未分配的 GRES 是可尝试的调度目标，
不是立即获配的保证。`MIXED` 也不表示还有整张 GPU。

本入口明确申请一个 `gpu:1g.10gb:1`。不修改 MIG 配置、不直接登录计算节点。
是否允许本账号使用、何时启动，仍由 NCC 的 Slurm 策略决定。

本轮验证真实 A512 工作集：3,883,798 参数、batch 8、64 步、24×24 图像、
FP32、现有 AdamW 和诊断。两条支路的直接视觉监督权重分别为 0 和 1，
各运行 4 次 optimizer update。初始权重、训练语料和批次顺序相同。
开发种子为 4,080,000,000 起的 16 条训练轨迹、4,080,100,000 起的 8 条
验证轨迹，初始化 97；不读取正式 Phase A 训练/验证轨迹或 fresh test。

这项短测确认模型能否运行、实际显存和耗时，不是正式八组实验的性能结论。
不减小模型、batch 或序列来静默适配切片，也不从短测检查点续训正式实验。

## 一次提交包含两个 Slurm 作业

| 作业 | 分区 / QOS | 资源 | 工作 |
|---|---|---|---|
| `isac-ncc-runtime` | `cpu` / `debug` | 2 CPU、8 GB 主机内存、30 分钟 | 建立固定版本运行环境 |
| `isac-ncc-mig` | `res-gpu-small` / `debug` | 1 个 10 GB MIG、2 CPU、8 GB 主机内存、15 分钟 | 设备预检及真实 A512 短测 |

GPU 作业依赖 CPU 作业的 `afterok`；安装失败则不会占卡训练。
`--kill-on-invalid-dep=yes` 避免安装失败后依赖作业永久等待。环境安装与
GPU 实际执行有独立回执、状态和日志。

CPU 作业固定检查 `/usr/bin/python3` 为 Python 3.12.3，创建专用 venv，
分别从 PyPI 和 PyTorch 官方 cu128 索引安装 NumPy 2.3.4、
PyTorch 2.9.0+cu128。保留完整依赖 freeze、安装报告及其哈希。
freeze 是本次解析结果记录，不是完整的 `--require-hashes` 锁。
不依赖 `module load python/...`，不加载额外 CUDA module，不运行项目的
默认 `uv sync`（其 Linux 源选择的是另一版本的 CPU PyTorch）。

CPU 节点的 Python、venv 支持和下载网络需由实际 setup 作业确认。
缺失时明确失败并保留对应日志，不假定登录节点的安装就等同于计算节点。

## 从 Mac 提交

使用发布时提供的完整提交 SHA；下面 `PIN_FROM_RELEASE` 必须替换。
在 Mac 本机终端、原来的项目仓库执行：

```bash
(
  set -e
  set -o pipefail
  cd "$HOME/Desktop/isac-world-model"
  ncc_pin=PIN_FROM_RELEASE
  ncc_ref=refs/remotes/origin/research/ncc-mig-pilot
  git fetch origin "refs/heads/research/ncc-mig-pilot:${ncc_ref}"
  test "$(git rev-parse "$ncc_ref")" = "$ncc_pin"
  git show "${ncc_pin}:scripts/submit_ncc_mig_pilot_from_mac.sh" |
    bash -s -- "$ncc_pin" bhmq03
)
```

`${ncc_pin}:...` 的大括号不可省略：Mac 的 zsh 会将未加大括号的冒号
解释成变量修饰语。学校密码只输入 SSH 提示，不写入脚本。

上传的是完整 Git bundle，不依赖 Hamilton 的历史目录或增量 bundle。
远程源码为 `$HOME/isac-ncc-mig-<SHA前12位>`，运行环境单独位于
`$HOME/isac-ncc-runtime-<SHA前12位>`。既有目录和结果不覆盖。
已取得的 job ID 保存在 `.ncc-mig-submission/setup.job-id` 和
`.ncc-mig-submission/pilot.job-id`。重复运行只报告已有回执，不重复提交。
若提交响应不完整，保留该次 claim/receipt，先查账而不盲目再提交。

## 怎样判断完成

先看两个作业各自的 `sacct`；GPU 依赖等待与 GPU 资源等待是不同阶段。
只有 CPU 的 `runtime.json.status=ready` 表示依赖安装完成，
不表示 GPU 已经执行过。

GPU 完成后检查：

```text
runs/ncc-mig-<GPU作业号>/status.json
runs/ncc-mig-<GPU作业号>/preflight.json
runs/ncc-mig-<GPU作业号>/model-check/pilot.json
```

预检要求 Slurm 为本人在 `res-gpu-small` 分配一个运行中的 `1g.10gb`，
实际 CUDA 可见一个实例；通过 CUDA Driver API `cuDeviceGetUuid_v2` 与
`nvidia-smi -L` 的 MIG 子设备 UUID 对齐。实例总显存来自 PyTorch 的
CUDA 设备属性，不把父卡的 80 GB 误报为切片显存。

`pilot.json.status=completed` 包含两条成功的 A512 支路；各自目录保留
训练/验证历史、视觉梯度和参数更新、检查点以及 SHA-256 校验。
CUDA 峰值统计范围是实际训练/验证/检查点循环；初始 update 0 诊断不在
这一峰值范围内，其成功执行由完整短测成功体现。耗时包含指定的验证、
诊断和保存开销，不能直接当作纯优化器更新吞吐或正式八组的精确预计时间。

Slurm 对 MIG 的 `gpumem/gpuutil` accounting 可能没有支持；这些账单字段
缺失不否定实际 CUDA 更新。这里用 CUDA 执行、实际参数变化和 CUDA
allocator 峰值作为证据，不强制 CUPTI profiler 权限。

原 Hamilton 19373715 和 NCC Hopper 1161475 不由这个入口取消。
短测确认后，再根据真实显存/耗时安排正式八组所需 walltime，并明确记录
Python 3.13.9→3.12.3、H200→MIG 的运行环境迁移。原正式训练协议保持不变。

## 参考

- [NCC 文档](https://nccadmin.webspace.durham.ac.uk/)
- [NCC PyTorch 环境指南](https://nccadmin.webspace.durham.ac.uk/pytorch/)
- [PyTorch 历史版本与官方 CUDA wheel](https://pytorch.org/get-started/previous-versions/)
- [Slurm GRES 与 MIG 调度](https://slurm.schedmd.com/gres.html)
- [CUDA 12.8 Driver API 设备 UUID](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__DEVICE.html)
