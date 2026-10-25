# ISAC World Model v0.4.1 — 验证与发布交接

日期：2026-10-25。基于 v0.4.0 本地提交 `2b2a0fde45caf88bb1667a85645e97138d8e46a2`。

## 结论

本轮交付完成工具链、安装复现、证据绑定及首次私有发布流程的加固。
**尚未上传 GitHub，也没有 GitHub Actions 远端成功记录。** 当前连接账号已确认是
`billlza`；`billlza/isac-world-model` 查询返回 HTTP 404。该响应只说明当前连接不能读取目标，
不能单独证明目标不存在。连接器支持已有仓库内的写入，但没有仓库创建操作；本执行环境没有
已认证的 GitHub CLI。本轮未更改用户的任何已有 GitHub 仓库。

主验证及独立安装验证均实际完成全部 20 个必需检查。两者的源码范围、文件内容和源码树哈希一致。
源码/构建/工作流输入共 65 个文件；每份报告绑定 28 个顶层日志及结构化记录。

```text
adc5979f0bfa4a3f92671a1a84b1939ae530d2f557f6eed95e1069ee077b7dd0
```

完整日志分别在 `validation/v041/` 和 `validation/bootstrap-v041/`。
报告生成时间分别为 `2026-10-25T14:55:31.158244+00:00`、`2026-10-25T14:56:08.812728+00:00`。
交付包的最终本地提交见 `LOCAL_COMMIT.txt`；发布状态见 `RELEASE.json`。

## 实际验证结果

| 检查 | 结果 |
|---|---|
| 主环境完整 CI | 20/20 必需步骤通过 |
| 空目录一键 bootstrap 完整 CI | 20/20 必需步骤通过 |
| Python 测试 | 185 通过，0 失败，0 错误，0 跳过 |
| Rust 单元测试 | 15 通过 |
| 原生差分测试 | 5 个种子，共 25,005 条命令一致；另有 2 项非法输入测试 |
| Ruff / Ruff format | 通过 |
| Rustfmt / Clippy / release build | 通过，Clippy 使用 -D warnings |
| 依赖一致性及 pip check | 通过 |
| 已知漏洞公告审计 | 49 个第三方包，0 跳过，0 报告的已知漏洞 |
| wheel / sdist / 源目录外安装检查 | 通过 |
| 训练、恢复、校准、延迟评测和六类故障扫描 | 集成流程通过 |
| 全新在线安装与缓存离线安装 | 两个独立虚拟环境的 50 个包（含项目）版本清单一致 |
| 安装后功能检查 | 两个环境均在源目录外导入已安装模块，验证有限梯度并完成小型训练 |
| 源码及证据文件核验 | 两份报告均通过，65 个源码输入及每份 28 个证据文件一致 |

覆盖率：语句 1,343/1,458 = 92.11%；分支 385/484 = 79.55%；综合 88.98%。
原 88% 综合门槛未降低。覆盖率统计对象是 `isac_wm`，不是全部发布脚本；发布脚本另有回归测试。

完整 CI 必检项：contracts、dependency-policy、dependency-environment、pip-check、lock-check、
environment-snapshot、python-tests、integration、distribution、wheel-install、ruff、ruff-format、
dependency-audit、rust-toolchain、rustfmt、clippy、rust-tests、native-build、native-parity、reproducible-install。

## 版本与工具链

参考环境：Linux x86_64 CPU，2 vCPU，4 GB 内存。

| 组件 | 实测版本 |
|---|---|
| CPython | 3.14.9 |
| PyTorch | 2.14.1+cpu |
| NumPy | 2.5.4 |
| Rust / Cargo | 1.99.1 |
| Ruff | 0.17.2 |
| uv | 0.13.5 |
| pytest / pytest-cov | 9.1.1 / 7.1.0 |
| coverage | 7.17.0 |
| pip-audit | 2.10.1 |
| 系统链接器 | GCC 15.2.0（环境提供） |

本轮重新查询了 `ci/versions.json` 的全部 15 个直接包，均与注册表当前版本一致；原始查询结果见
`validation/v041-current-versions.json`。未重新宣称所有平台/传递依赖都在本轮逐一更新。
v0.4.0 已记录的兼容性决定继续保留：Python 3.14 系列适配当前 PyTorch 发行包；
SymPy 对 mpmath 的 <1.4 约束继续满足。完整锁文件、哈希、来源及范围位于 `uv.lock` 和 `ci/`。

bootstrap 在一个仅复制源码的新目录中运行，没有继承项目 .venv 或项目 .toolchains；
重新准备项目专用 Python、Rust、uv 缓存及依赖，然后运行完整 CI 和证据核验。
系统 Git、链接器及引导 Python 是明确前提，不是由脚本偷偷修改的全局工具。

## 本轮修复和机制

旧的首次发布脚本没有提交 uv.lock 和 .python-version。本轮改为恢复已交付的 Git 历史并验证
所有必需发布文件已经进入同一个干净提交，不再临时拼凑一个不完整的提交。

新增 evidence.py：全量检查集合、每项退出码、源码树、日志和结构化记录必须一致；
空报告、部分 CI、缺失/重复步骤、跳过/失败、源码变化、日志改变、路径穿越或符号链接均拒绝。
哈希用于检测变更，不是发布者签名或系统安全证明。

新增 clean_install.py：新缓存开始在线安装；再用同一批缓存在第二个新环境中以 uv --offline 安装。
使用非 editable 安装、隔离环境变量、无用户 site-packages，并在源码目录外测试真实已安装的模块。
这证明同操作系统及所选解释器下的依赖/应用安装可以复现，不是整个操作系统或编译产物逐字节一致，
也不代表没有预备缓存时整套 Python/Rust 工具链可离线安装。

新增 check_rust.py：核验实际 rustc/cargo 版本，记录 rustfmt/clippy/linker 路径与版本；缺工具不作通过。
新增 env.sh：在后续终端恢复项目局部 PATH、CARGO_HOME、RUSTUP_HOME 和 uv 路径。

首次发布脚本分开记录 upload_status 与 ci_status；创建/上传成功不等于 CI 成功。
它核验实际远端 main 的 SHA 及 uv.lock、.python-version、工作流文件字节；随后只接受该提交的工作流，
并要求包括 ci-gate 在内全部任务完成且成功。超时、取消、跳过、其他提交的绿灯均不算验收成功。
远端行为已做失败分支和契约测试；本轮没有真实 GitHub 发布试跑。

## 用户本机最短发布路径

使用完整 ZIP 解压后的新目录。不要在 GitHub 另行初始化带 README 的同名仓库，避免不同历史冲突。
需要 Git、Python >=3.10 和 GitHub CLI。仅发布不需要先安装 PyTorch 或 Rust。

```bash
cd isac-world-model

gh auth login --hostname github.com --git-protocol https --web --scopes workflow

# 仅本地检查、恢复交付 Git 历史；不联系 GitHub
python3 scripts/publish_private.py

# 创建/使用空的私有仓库，上传，核验提交并等候对应的远端 CI
python3 scripts/publish_private.py --execute
```

默认目标是 `billlza/isac-world-model`，不会公开仓库，不会 force-push，不会覆盖不同历史。
GitHub 登录在用户本机完成，不要将令牌发给对话。
发布回执保存在 `validation/publication/<时间戳>-<进程号>/receipt.json`，记录仓库地址、
本地/远端 SHA、上传状态、CI 状态、workflow URL 及各 job 结果。
等候超时后可重新只读核验，不用重新上传：

```bash
python3 scripts/publish_private.py --execute --verify-only
```

后续有新提交时，采用正常受审查的 git push；本脚本针对首次发布，故意拒绝覆盖已有不同历史。

## 在本机重新安装完整研发环境

```bash
INSTALL_RUSTUP=1 bash scripts/bootstrap.sh

# 后续终端可这样运行项目工具；bootstrap 已运行完整 CI
bash scripts/env.sh isac-wm doctor --device cpu
```

安装写入 .bootstrap、.venv 和 .toolchains，不改变系统默认 Python/Rust，不使用 sudo。
已有 .venv 时脚本拒绝覆盖；需要重建时显式设置 RECREATE_VENV=1。
在 macOS 上需要现有 Command Line Tools/Git/链接器；本轮未实际执行 macOS、MPS 或 CUDA。

## 已完成与未完成

已完成：固定版本下的完整工具链检查、空目录 bootstrap、在线/缓存离线安装复现、20 项完整 CI、
185 项 Python 与 15 项 Rust 测试、证据与交付源码绑定、归档 Git 历史恢复及发布脚本加固。

未完成：GitHub 上传、远端 Actions、macOS/MPS/CUDA 实机验证、Rust 队列接入主模型。
没有修改主模型架构或宣称精度提升。ZIP 中可选的 v020 检查点是历史资料，
本轮小型训练用于安装与集成验证，不替代模型效果研究。

## 外部操作

复用了用户既有 Vercel 测试项目和已存在的快照，新建一次临时 2 vCPU/4GB 会话执行安装验证。
无公开端口、无网站部署、无新快照、无 API 密钥写入源码。结果完整导出后会话已停止。
详见 docs/EXTERNAL_ACTIONS_v041.md。
