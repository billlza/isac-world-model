# M7 Pro 实机数据接入：已确认能力与三维模型接口

日期：2026-11-24。

## 当前结论

NETGEAR Nighthawk M7 Pro **MR7450-100EUS** 可以继续作为真实蜂窝通信链路的测量来源。
已有 Mooring 采集器能够读取和保存 LTE/NR 信号质量、频段、服务状态和设备状态。
本次增加的是研究仓库中的**电脑端历史数据导入器**，用于把这类真实遥测保留为可审查、可重复分析的数据。

现有 Mooring 接口输出标量蜂窝遥测；本轮已读代码没有输出复数 CSI、原始 I/Q、
逐天线相位或多径时延。现有三维占据模型的校准复数 RF 输入与它不同。
因此，遥测导入已具备实现依据，直接用这份遥测运行现有三维房间反演仍缺测量接口。
这限定的是当前接口可支持的任务，不是对芯片内部所有可能接口的断言。

本轮没有访问用户的 M7、登录其管理页面、采集新实机数据或运行新的模型训练。
会议内容仍需以实际返回的记录为准，本文件不归因于组会中任何尚未核对的发言。

## 1. 代码与资料依据

### 设备与固件

- 官方产品页确认 MR7450 为 5G / WiFi 7 移动路由器，采用 Snapdragon X75：
  <https://www.netgear.com/uk/mobile-wifi/hotspots/mr7450/>。
- 官方手册第 56 页说明管理页面的 Network、Device、WiFi Details 和 Diagnostics。
  Diagnostics 包括蜂窝频段信息、温度、电池电压：
  <https://www.downloads.netgear.com/files/GDC/MR7400/MR7400_7450_UM_EN.pdf>。
- 官方 GPL 索引列有 MR7450 的 **10.01.14.00** 条目：
  <https://www.downloads.netgear.com/files/GDC/2649_GPLv1.html>。
  此次未取得并审阅该源码包；条目的存在不能证明当前固件具有可调用的 CSI/IQ 导出接口，
  也不能证明可构建、签名和安装自定义固件。
- 既往记录中设备 PRI / UI 版本为 `MR7450-100EUS_01.15`，
  研究基线曾记录为 `NTGX75_10.04.43.01`。这不是本轮从设备重新读取的版本。
  导入器的实际固件字段默认未知，由操作者依据对应采集记录填写；
  不把历史版本或 GPL 条目自动填成当前设备版本。

### 已有实机采集代码

本次核对 Mooring 的固定提交：

`0d6361d3bca8ea76a08cd8ef1cac6606dd730958`。

- [MR7450 字段白名单](https://github.com/billlza/mooring/blob/0d6361d3bca8ea76a08cd8ef1cac6606dd730958/crates/daemon-core/src/router/netgear/allowlist.rs)
- [历史 CSV 契约](https://github.com/billlza/mooring/blob/0d6361d3bca8ea76a08cd8ef1cac6606dd730958/crates/daemon-core/src/radio_table.rs)
- [历史格式](https://github.com/billlza/mooring/blob/0d6361d3bca8ea76a08cd8ef1cac6606dd730958/crates/daemon-core/src/radio_history.rs)
- [运行中的轮询顺序](https://github.com/billlza/mooring/blob/0d6361d3bca8ea76a08cd8ef1cac6606dd730958/crates/mooringctl/src/daemon/deployed.rs)
- [导出与分析说明、既有实机验证记录](https://github.com/billlza/mooring/blob/0d6361d3bca8ea76a08cd8ef1cac6606dd730958/docs/runbooks/radio-analysis.md)
- [早期设备测量记录](https://github.com/billlza/mooring/blob/0d6361d3bca8ea76a08cd8ef1cac6606dd730958/docs/measurements/M2.md)

这些历史文档记录过实机状态读取和历史 CSV 导出成功。
它们是此前工作的证据；本次适配器测试使用合成契约样例，不能计作新增实机结果。

## 2. 可用测量与研究用途

| 输入或能力 | 当前已读接口的情况 | 本次接入用途 |
| --- | --- | --- |
| LTE RSRP、RSRQ、SINR、RSSI | 可由已有采集器读取，CSV 可保留缺失项 | 真实链路强度与质量分析 |
| NR RSRP、RSRQ、SINR | 可读取；未报告时为空 | 5G 链路状态分析 |
| 频段、网络类型、服务状态、载波聚合数量 | CSV 中有对应字段 | 区分网络／频段变化与同一条件下的变化 |
| 温度、运行时长、会话时长 | CSV 中有对应字段 | 排查重启、温度与会话变化等混杂因素 |
| 读取失败及其原因 | 有独立记录 | 采集可用性统计，不等同于蜂窝链路断开 |
| 传输吞吐、RTT、业务包丢失率 | 本次样本 CSV 不含 | 需要单独测量后才能评价 |
| 复数 CSI / 原始 I/Q / 天线相位 / 多径时延 | 当前接口没有提供 | 不生成或补造这些字段 |
| 三维占据模型的校准复数观测 | 当前遥测不满足此输入类型 | 保持与 `occupancy3d` 输入独立 |
| 在线可用时刻与 RF 采样时刻 | 旧 CSV 未记录 | 本次数据不用于声称在线 deadline 验证 |

蜂窝标量遥测也不同于本地 WiFi 链路的逐子载波复数 CSI。
芯片和协议支持 WiFi 7，并不自动使这些测量向当前应用接口开放。

## 3. 导入器保留的语义

新模块：`src/isac_wm/m7_telemetry.py`。命令入口：`scripts/import_m7_telemetry.py`。

输入是既有 `mooringctl router history --csv samples` 的 21 列样本表。
表头是版本契约；未知表头或格式损坏会明确报错。
原始 CSV 保持不变，输出写入一个新的目录。

- 七个 RF 标量保留原单位和缺失标记；缺失不会填成物理零。
- `outcome=failed` 表示电脑未成功读取路由器，不能转换为“网络掉线”标签。
- `outcome=read, service=no_service` 才是成功读取后，设备报告无服务。
- 网络代码 PLMN 保留字符串语义，不把前导零丢掉。
- `at_unix` 与 `at_utc` 必须对应；坏行、截断、重复或乱序不会被悄悄删除或重排。
- 本版导入要求轮询开始时间严格递增。宿主机时钟回拨可能产生合法的 Mooring 历史，
  但会被本导入器明确拒绝；这是导入器的质量准入限制，不代表原历史不可能存在。
- 源文件 SHA-256、字节数、导入时间与所依据的格式版本写入来源记录。
- “格式参考提交”与“实际采集器提交”分别记录。实际采集版本不明时保持未知。
- `--data-origin` 是操作者对数据来源的声明，不是软件证明数据确由设备产生。
- 质量报告记录实际行数、失败、缺失与时间间隔。间隔不能自动换算成丢包数。

### 时间戳的关键限制

Mooring 的现有轮询先记录宿主机时间，再请求设备，随后取得完成时间；
历史表只保存了前者。因此 `at_unix` 是 **host poll start**。

导入结果保留这一语义。RF 采样时刻与逐条数据可用时刻保持未知；
不设成轮询开始，也不设成导入时刻。这里的限制影响在线因果与时延验证，
不影响已有数据作为历史遥测的真实性或离线分布分析。

CSV 本身不编码原历史文件的 `incomplete_days`。
“CSV 完整且可以读取”与“上游历史完全无缺失”分别记录。
如需核查正在写入的日志，保留 Mooring 的非 CSV 历史检查结果。

## 4. 在现有 Mac 上运行

这部分只需要 Python 3.12 或更高版本的标准库；无需安装研究模型的 Torch 环境。
保留 Mooring 原有部署方式。以下命令只读取电脑上已有的历史文件。

在 Mac 终端导出，使用新目录避免覆盖已有记录：

```bash
(
  set -e
  m7_export_dir="$(mktemp -d "$HOME/Desktop/m7-isac-XXXXXX")"
  "$HOME/Applications/Mooring.app/Contents/MacOS/mooringctl" \
    router history --csv samples > "$m7_export_dir/radio-samples.csv"
  printf '导出位置：%s\n' "$m7_export_dir/radio-samples.csv"
)
```

默认导出昨天与今天的 UTC 历史。需要较长区间时，
在 `router history` 后按 Mooring 原有接口指定 `--since YYYY-MM-DD --until YYYY-MM-DD`。
`--csv` 需要包含 2026-11-14 后此功能的 Mooring 构建。
源项目的历史存放与传输约束继续适用；本导入器不会上传文件。

在本研究仓库根目录运行，将输入路径换为刚才打印的位置：

```bash
python3 scripts/import_m7_telemetry.py \
  --input /path/to/radio-samples.csv \
  --output runs/m7-real-20261124 \
  --data-origin real_device
```

有对应采集版本记录时，可以追加：

```text
--collector-commit <实际采集器的40位提交SHA> --firmware <对应采集时的版本>
```

只有确认来自设备的导出才声明 `real_device`。
测试样例用 `synthetic`；来源不明使用默认的 `unknown`。

输出包含：

- `telemetry.jsonl`：逐条、类型独立的链路遥测；
- `quality.json`：行数、缺失、失败和时间间隔等可观察的数据质量；
- `manifest.json`：来源、时间语义、数据来源声明与任务适用性。

输出目录已存在时命令拒绝覆盖。只有表头的合法空导出会报告没有样本，
不生成“训练成功”或性能结论。真实 CSV 和导入结果不加入 Git。

## 5. 本周如何使用这台设备

先从已经存在的实机历史确认字段覆盖、采集失败和间隔，建立真实通信支路。
随后可以按网络、频段和日期划分数据，用过去观测预测后续链路状态，
比较持续值等基线，并把网络切换和设备移动等条件明确记录。
训练／验证／测试按独立日期或实验段划分，不能把相邻窗口随机打散后声称泛化。

若开展位置或遮挡实验，应独立记录位置、姿态、条件和时间。
这些记录可以检验环境上下文是否帮助通信状态预测；
当前 CSV 本身没有房间几何真值，也没有给出从 RF 恢复三维房间所需的观测。

三维预训练的已完成结果继续保留。把实测复数 RF 接入它的具体依赖是：
可重复读取的真实信号样本、测量频率和收发几何、校准约定及匹配的前向算子。
本次导入不会重新预训练、启动 RL 或产生人类偏好标注。

## 6. 验证边界

验收使用明确标为合成的契约输入，检查格式、缺失、失败状态、时间语义、
来源摘要、输出和错误处理；不把样例数值当作 M7 实测性能。
验证命令与结果记录在对应提交／PR 中。

本轮新增能力完成后，需要真实导出文件才能验证它对用户当前历史数据的兼容性。
实际预测效果、环境上下文收益和三维实机重建均属于后续实验的具体问题。
