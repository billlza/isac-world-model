# 可选的训练运行时间诊断

这是独立于 Phase A 科学实验的短性能测量，不保存模型、不选择 checkpoint、不报告精度提升，也不改变模型、损失或 CUDA 算子。只在已经分配的计算节点运行 GPU 测量；登录节点用于提交作业。

## 用法

使用当前工程已验证的 Python / PyTorch 环境，仓库根目录执行：

```bash
PYTHONPATH=src python scripts/profile_training_runtime.py \
  --device cuda --out runs/runtime-profile-h200-001
```

输出目录必须不存在；成功结果是 `profile.json` 且 `status` 为 `completed`。失败会尽可能保留 `status=failed` 和错误。请求 CUDA 而不可用、profiler 不支持 CUDA activity，或实际 trace 没有任何正的设备耗时时直接报错，不把 CPU-only 统计视作 CUDA 测量。CPU 诊断需显式指定 `--device cpu`。

CLI 固定比较 A96、A512（structured-attention，feature48、memory16、heads4），每段64个时间步、dt=0.1s、24×24图像。默认每个模型2次预热更新、8次无 profiler 的计时更新、3次独立 profiler 更新，batch8；这也是允许的最大更新数和 batch。可用 `--measured-updates 2 --profile-updates 1 --warmup-updates 1 --batch-size 1` 缩短诊断，须保留这些实际参数再比较。

该脚本只生成4,080,000,000开始的专用 development episode seeds，模型初始化 seed97，禁止通过 CLI 替换为正式训练／验证／测试种子。相同 CPU minibatch 只生成一次，按相同顺序供两个宽度使用；JSON 记录逐批 SHA256、完整种子、源码 SHA256、可获得的 git commit、PyTorch/CUDA/设备/精度设置和实际更新计数。

## 怎么读结果

- `warmup` 单独记录预热时间，不混入 `measured`。后者覆盖缓存 CPU 批次传输、zero_grad、forward、现有 loss 和检查、backward、梯度裁剪及 AdamW。生成数据、初始化、验证与文件存储不在该计时中。每次更新前后同步 CUDA，阶段内部不额外同步；模型已有同步保持原状。
- `counts` 区分预热、普通计时与 profiler 更新，并列出 forward 次数、批处理时间步和包含 batch 的 episode 时间步。默认每个模型13次更新、832个批处理时间步、6656个 episode 时间步，模拟时间频率不随模型宽度改变。
- `operators` 来自另一次有限 profiler pass，列出 CPU／设备 self time 前50组算子，以及 item、标量提取、solve、同步等候选事件。不把带 profiler 的时间当作正常训练速度；嵌套算子次数、total time 不可直接相加。
- `unprofiled_cuda_peak_allocated_bytes` 与 reserved 只表示 PyTorch CUDA allocator 的峰值，不是整个进程或设备全部显存。profiled峰值单列；CPU不提供等价显存数字。
- 8次测量不足以建立稳定的在线 p95 保证；JSON 中 p95 只是小样本 nearest-rank 描述。固定顺序的一次诊断也不构成宽度因果结论，确认候选优化后再作独立重复测量。
- loss 只用于确认这段真实反向传播／优化执行有限，不是科学精度指标。当前诊断固定 visual loss weight0，不混入 Phase A 的新增视觉监督因素。

现有64 tick离线循环不会因为墙钟耗时增加自动减少更新数。实时系统若超出传感器周期，才需要另外检查积压、丢帧与降频，记录采集、到达和消费时间。

## 优化选择与官方依据

首先检查 profiler 中已有同步和小算子启动开销。PyTorch2.9明确说明 [`torch.linalg.solve`](https://docs.pytorch.org/docs/2.9/generated/torch.linalg.solve.html) 在 CUDA 输入上会同步 CPU；[`solve_ex`](https://docs.pytorch.org/docs/2.9/generated/torch.linalg.solve_ex.html) 可避免默认错误检查同步，但返回的 `info` 必须正确处理，不能删掉失败检查就声称优化完成。这里仅记录候选，不替换算子。

训练与实验编排继续使用 Python/PyTorch。先尝试批处理和纯张量区域编译；[张量相关 Python 控制流](https://docs.pytorch.org/docs/2.9/compile/programming_model.common_graph_breaks.html) 会导致断图，[`torch.compile(fullgraph=True)`](https://docs.pytorch.org/docs/2.9/generated/torch.compile.html) 可让所选区域断图时明确报错。先核对前向、梯度、时间戳、new/valid mask与错误处理，再比较性能。GPU 是异步执行的，必须采用[正确的同步或 CUDA Event 计时](https://docs.pytorch.org/docs/2.9/notes/cuda.html)；[Profiler](https://docs.pytorch.org/docs/2.9/profiler.html) 用来定位算子而非直接宣称稳态加速。

已确认的热点可考虑 C++/CUDA 或 [Triton](https://triton-lang.org/main/index.html)；采集、事件队列、协议与记录层可沿用项目已有 Rust event-core，先依据其独立 oracle／parity 证据选择集成范围。仅把调用同样 CUDA kernels 的 Python 外层改成 C++ 或 Rust 不会自动提升视觉表征或减少 GPU 工作。

截至2026-11-05，[Rust1.99.1](https://blog.rust-lang.org/2026/10/15/Rust-1.99.1/) 已发布（修复了含 x86_64 FFI bool返回值在内的问题），[C++26](https://www.iso.org/standard/91179.html) 仍是 ISO DIS阶段。当前 [CUDA12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html) 支持 C++11/14/17/20，不能直接把 CUDA 编译单元改为 C++26；[PyTorch2.9扩展](https://docs.pytorch.org/docs/2.9/cpp_extension.html) 需要独立可用的编译器／NVCC工具链并核验 ABI。PyTorch报告cu128不等于已安装NVCC。版本号本身不是加速证据。
