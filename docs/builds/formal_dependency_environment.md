# 正式依赖环境与哈希锁证据链

## 1. 文档职责

本文档定义当前项目正式 Python 依赖环境的唯一协议。该协议回答以下问题:

1. 每条 GPU 科学运行路径使用哪个精确 CPython patch、Linux 架构、CUDA 和 PyTorch 组合。
2. 人工审计的直接依赖输入与解析后的完整 wheel 闭包如何区分。
3. 父编排解释器与相互冲突的科学子解释器如何隔离。
4. Notebook、独立 CLI、runner 和论文结果包如何共享同一套 fail-closed 环境证据。

环境可执行性不能由 Notebook 内的临时安装单元证明。正式路径由已提交 registry、精确直接依赖输入、目标 Linux x86_64 环境完整哈希锁、环境 inspection、正式执行锁和 preparation report 组成连续证据链。

## 2. 分层职责

| 层级 | 当前文件 | 职责 |
| --- | --- | --- |
| 配置事实层 | `configs/dependency_profile_registry.json` | 登记 profile 名称、CPython patch、Linux 架构、CPU 或 CUDA accelerator identity 和依赖输入路径。 |
| 精确直接输入层 | `configs/dependency_profiles/*_direct.txt` | 保存人工审计后的直接依赖, 每行只能是 `name==version`。 |
| 完整闭包层 | `configs/dependency_profiles/*_lock.txt` | 保存匹配 profile 的 Linux x86_64 目标环境解析出的直接依赖与传递依赖, 每个候选 wheel 必须带真实 SHA-256。 |
| profile 运行时层 | `experiments/runtime/dependency_profiles.py` | 解析 registry、校验精确规格、计算稳定摘要、检查完整锁并核验当前解释器中的完整锁包集合。 |
| 当前解释器准备层 | `experiments/runtime/dependency_preparation.py` | 要求正式执行锁和目标完整锁, 执行 hash-locked 安装、兼容性检查与共享 inspection。 |
| 隔离环境准备层 | `experiments/runtime/isolated_dependency_environment.py` | 使用固定 `uv` 创建精确 CPython 子环境, 并在子解释器中调用当前解释器准备层。 |
| 薄 CLI 层 | `scripts/prepare_dependency_profile.py`, `scripts/prepare_isolated_dependency_environment.py` | 仅转发参数和退出码, 不保存科学机制、包清单或第二套环境规则。 |
| Notebook 入口层 | `paper_workflow/notebooks/*.ipynb` | 只传入登记的 profile id 和路径参数; 不维护包清单或安装实现。 |
| 治理审计层 | `tools/harness/audits/audit_dependency_profile_governance.py` | 检查六个 profile、直接输入、完整锁门禁、层级依赖、字段登记和动态安装禁令。 |

这一分层属于通用工程写法: 配置事实、解析器、执行器和入口相互分离, 内层不引用外层, 因而同一正式执行器可以脱离 Notebook 在普通 Linux 服务器运行。

本项目特定设计是一个父编排 profile 与五个科学执行 profile 的隔离矩阵。SLM-WM SD3.5、T2SMark SD3.5 和三套官方参考实现需要不同的 CUDA 或 Python 组合, 不得把它们压入一个可漂移的共享环境。一次正式 session 准备父 `workflow_orchestrator` 环境和当前 workflow 唯一选择的一个科学子环境, 而不是只准备一个 profile 或同时混装全部科学环境。

## 3. 一个 CPU 父 profile 与五个 CUDA 科学 profile

| profile | CPython | 平台 | accelerator | torch | torchvision | PyTorch index |
| --- | --- | --- | --- | --- | --- | --- |
| `workflow_orchestrator` | `3.12.13` | `linux/x86_64` | `cpu` | 不适用 | 不适用 | 不适用 |
| `sd35_method_runtime_gpu` | `3.12.13` | `linux/x86_64` | `12.8` | `2.11.0+cu128` | `0.26.0+cu128` | `https://download.pytorch.org/whl/cu128` |
| `t2smark_sd35_gpu` | `3.12.13` | `linux/x86_64` | `12.4` | `2.5.0+cu124` | `0.20.0+cu124` | `https://download.pytorch.org/whl/cu124` |
| `tree_ring_official_py39_cu117` | `3.9.19` | `linux/x86_64` | `11.7` | `1.13.0+cu117` | `0.14.0+cu117` | `https://download.pytorch.org/whl/cu117` |
| `gaussian_shading_official_py38_cu117` | `3.8.20` | `linux/x86_64` | `11.7` | `1.13.0+cu117` | `0.14.0+cu117` | `https://download.pytorch.org/whl/cu117` |
| `shallow_diffuse_official_py39_cu117` | `3.9.19` | `linux/x86_64` | `11.7` | `1.13.0+cu117` | `0.14.0+cu117` | `https://download.pytorch.org/whl/cu117` |

五个科学 profile 的 `torch` 与 `torchvision` 必须同时出现在 profile 和直接输入文件中。二者的 `+cu*` local version、registry 中的 CUDA 版本和 PyTorch index 后缀必须完全一致。CPU 父 profile 的 CUDA、torch、torchvision 与 PyTorch index 字段必须为 `null`, 直接输入不得出现 `torch` 或 `torchvision`。

六个直接输入文件当前共登记111个精确条目。父编排 profile 固定 `uv==0.11.28` 与 `huggingface_hub==1.20.1`; 后者供父 runner 下载固定 revision 的模型和 OpenCLIP 快照。五个科学执行 profile 均固定 `pip==24.3.1`、`setuptools==75.3.0` 和 `wheel==0.45.1`, 使目标环境中的安装器也进入完整锁证据。T2SMark 固定 `diffusers==0.32.0`; Gaussian Shading 固定 `scipy==1.10.1` 和 `huggingface_hub==0.25.2`。所有精确值在目标完整锁提交并通过对应 CPU 或 GPU smoke 前都只是受治理输入, 不能表述为已资格化环境。

## 4. 直接输入与完整锁

### 4.1 精确直接依赖输入

直接输入文件表达项目明确请求安装的包。有效行只能满足以下形式:

```text
package_name==exact_version
```

直接输入禁止版本范围、未版本化包、可漂移标签、远程 VCS 引用和安装选项。`direct_requirements_digest` 对规范化包名和精确版本计算稳定摘要, 不受换行符和条目顺序影响。

### 4.2 目标 runtime 完整 wheel 哈希锁

直接输入不能证明传递闭包。父编排完整锁必须在登记的 CPython 3.12.13 Linux x86_64 CPU runtime 中解析; 五个科学完整锁必须在满足各自 CPython、Linux x86_64、CUDA 和 PyTorch index 的 GPU runtime 中解析。目标 runtime 可以位于 Colab 或满足同一 profile 的普通 Linux 服务器。完整锁必须包含:

- 全部直接依赖。
- resolver 选择的全部传递依赖。
- 每个解析后依赖的精确版本。
- 实际候选 wheel 的 SHA-256。

完整锁必须进入 Git `HEAD`, 且工作树内容与 `HEAD` 一致。项目不接受人工推测的 wheel 摘要, 也不允许用直接输入文件代替完整锁。

候选锁审查包不能只信任物化器写出的 provenance。`write_dependency_lock_review_bundle.py` 会重新读取实际 `pip_resolver_report.json`, 使用 `load_resolved_wheels` 复核目标解释器、全部直接依赖、wheel URL 和 SHA-256, 再重建唯一规范候选文本、依赖数量与逻辑摘要。重建值必须与实际候选文件及 provenance 逐项一致; 任一文件在生成后被改写都会阻断审查包成功结论。

## 5. Readiness 与完整环境 inspection

`get_dependency_profile` 在完整锁缺失时仍返回受治理 profile, 以便入口和审计报告说明阻断原因。此时必须满足:

```text
complete_hash_lock_present = false
formal_ready = false
readiness_blockers = ["complete_hash_lock_missing"]
```

`require_dependency_profile_ready` 对该状态直接抛出阻断异常。仓库结构审计把这一状态识别为正确的 fail-closed 行为, 不能把缺锁误写成已完成环境。

完整锁存在并通过 schema 校验后, `inspect_dependency_profile_environment` 检查:

1. Python 实现与完整 patch 版本。
2. 操作系统和机器架构。
3. 对五个科学 profile 检查 torch 模块版本、torch build CUDA 版本和 CUDA 可用状态; CPU 父 profile 不导入 torch, 这些观测值为不适用。
4. 完整锁中的每个直接、传递依赖的 installed distribution 版本。
5. profile 自身的完整锁 readiness。

报告同时保留 `direct_dependencies` 和 `locked_dependencies` 映射, 并通过一次 installed distribution 查询缓存产生两组实测映射。完整锁存在时只由完整锁包集合的不一致项门控, 避免直接依赖与完整锁对同一错误重复计数; 完整锁缺失时才使用直接依赖生成诊断。inspection 返回 `expected_environment`、`observed_environment`、`mismatches`、`readiness_blockers`、`decision` 和 `inspection_digest`; 仅当全部条件精确匹配时 `decision` 才能为 `pass`。

## 6. 当前解释器 preparation report

`experiments.runtime.dependency_preparation` 只消费 registry API, 不接受调用者传入自由包规格。准备顺序为:

1. 要求 `require_published_formal_execution_lock` 返回 clean、detached、已发布提交锁。
2. 查询 profile 和稳定 summary, 并要求目标完整哈希锁 ready。
3. 检查 registry、直接输入与完整锁均已提交且无工作树漂移。
4. 检查当前解释器和平台允许执行该 profile。
5. 使用完整锁执行 `--require-hashes` 和 binary-only 安装。
6. 五个科学执行 profile 使用当前 `sys.executable -m pip check`; 父编排 profile 将 `compatibility_check_required=false` 且记录 `not_applicable_to_orchestrator`。
7. 调用共享 inspection API 复核完整锁中的全部包。
8. 写出 `outputs/dependency_profiles/<profile_id>/dependency_profile_report.json`。

preparation report 绑定 profile 身份、直接输入摘要、完整锁摘要、正式执行锁、Git 提交状态、完整 argv、返回码、标准输出、标准错误、`pip_check`、共享 inspection 和最终门禁。该报告的 `supports_paper_claim` 固定为 `false`; 它只能证明运行输入与环境资格, 不能替代图像、指标、统计检验或 baseline 结果记录。

## 7. 隔离 CPython provision 与正式 prepare

五个科学执行 profile 必须通过隔离环境 API 运行。两个 API 的职责严格分离:

- `provision_isolated_dependency_python(...)` 要求父编排 profile 的完整锁和 inspection 通过, 同时要求正式执行锁; 它不要求目标科学 profile 的完整锁。该 API 使用固定 `uv==0.11.28` 创建精确 CPython patch 与空 venv, 执行 `ensurepip`, 适用于在候选子解释器中物化目标完整锁。成功结论只能是 `provisioned=true`、`formal_ready=false`, 不能形成正式环境结论。
- `prepare_isolated_dependency_environment(...)` 额外要求目标科学 profile 的完整锁 ready, 随后在目标 Python 中执行 `-m experiments.runtime.dependency_preparation --profile <profile_id>`, 并严格复核子报告的 profile、锁摘要、正式执行锁、安装、`pip check`、inspection 与解释器摘要。全部通过后才允许 `formal_ready=true`。

`uv` 每个发行版内置一份冻结的可下载 Python distribution 列表。当前协议固定 `uv==0.11.28` 并向 `uv python install` 与 `uv venv --managed-python` 传入完整 CPython patch, 不允许只指定 major/minor。provision 还要求实际 executable 路径属于当前父解释器安装的 `uv` distribution 文件清单, 且文件 SHA-256 与 `RECORD` 登记值一致; 仅在 PATH 中提供可报告相同版本的复制文件不能通过。报告保存 distribution `RECORD` 路径与摘要、executable 的 RECORD 路径与摘要、`uv --version` 结果、目标 Python 路径与摘要以及全部命令进程证据。正式 prepare 完成后再次计算目标 Python SHA-256, 防止安装期间解释器身份漂移。

默认环境根和 managed Python 根位于 `tempfile.gettempdir()` 下的项目专用目录, 不硬编码 Colab 路径。CLI 允许显式覆盖两者, 因而同一实现可在 Colab 与普通 Linux GPU 服务器运行。`uv venv --clear` 保证同名子环境不会继承上一次 session 的残留包。

## 8. Notebook 与业务路径约束

Notebook 保持调用薄入口, 但不承担环境创建实现。正式 session 的调用关系为:

1. 父解释器使用 `scripts/prepare_dependency_profile.py --profile workflow_orchestrator` 资格化编排环境。
2. 当前 workflow 的 runner 通过内层 API 准备且只准备一个科学子 profile。
3. 子解释器中的科学 runner 产生方法或 baseline 结果。

Notebook 或业务路径不得包含 `%pip`、`--upgrade`、可漂移版本、自由 requirements 列表或自行拼装的依赖安装实现。`main/`、`experiments/` 和 `paper_experiments/` 不得导入 `scripts/`; 薄脚本可以依赖内层模块, 内层不能反向引用脚本或 Notebook。

## 9. 当前执行接线状态

隔离环境原语和五个科学 profile 的候选锁路由已经实现。Tree-Ring、Gaussian Shading 与 Shallow Diffuse 三个 official-reference runner 已直接调用 `prepare_isolated_dependency_environment` 并验证子解释器 preparation 报告。

主方法 `sd35_method_runtime_gpu` 与 T2SMark `t2smark_sd35_gpu` 的正式科学命令当前尚未完成子解释器调用接线, 仍会经过父解释器环境报告路径。因此当前项目必须保留 `science_child_execution_wiring_incomplete` 阻断状态: 这两个 profile 即使未来补齐完整锁, 在子解释器命令身份、解释器 SHA-256 和内层 preparation 报告进入各自正式运行 manifest 前, 也不得形成 formal-ready 论文结果。该阻断项描述当前代码事实, 不能由 Notebook 说明、父解释器版本匹配或候选锁生成结果代替。

## 10. 当前资格状态

当前仓库已经提交六个精确直接依赖输入, 尚未提交一个 CPU 父 runtime 与五个 GPU 科学 runtime 的完整 wheel 哈希锁。因此六个 profile 当前均为 `formal_ready=false`, 正式环境 qualification 被 `complete_hash_lock_missing` 阻断。

这一状态允许执行 CPU 结构审计、registry 查询、Python 3.8 parser 兼容性测试和 Notebook 静态检查, 不允许执行正式依赖安装、GPU 结果生成或将任一 profile 描述为已经资格化。后续必须先在各自目标 runtime 生成真实完整锁并提交, 再依次通过 CPU 父编排 preparation、科学子环境 preparation、完整 `pip check`、对应硬件 smoke 与论文运行门禁。
