# Scripts

`scripts/` 提供可脱离 Notebook 运行的命令行入口, 不承载方法数学实现。

## 代码包抽离入口

- `extract_release_package.py`: 按 `minimal_method_package`、`paper_artifact_rebuild_package` 或 `paper_experiment_execution_package` 复制唯一文件集合, 记录源路径、包内路径、逐文件 SHA-256 和源仓库提交。三个 profile 都要求源工作树 clean 并在输出目录创建新的 clean detached Git 根提交。两个论文实验 profile 继续要求六个依赖锁; 最小核心包只复验自己的 PyTorch 2.11 可安装范围与构建依赖身份。命令只允许把持久化代码包写入 `outputs/`。
- `validate_core_method_package.py`: 映射到最小核心包根目录的标准库只读验证入口。它要求 `python -I`, 复验专用 README、抽离 manifest、clean detached Git、核心依赖身份、只打包 `main` 的 `pyproject.toml`, 并导入全部核心模块。该入口不消费论文实验依赖锁。
- `validate_extracted_package.py`: 在脱离开发仓库的代码包中复算全部文件摘要, 核验 Git 跟踪集合、源提交映射、六个完整依赖锁和必需 CLI 入口。验证子进程不继承开发仓库 `PYTHONPATH`, 不包含 `paper_workflow/`, 也不导入或执行 CUDA。

## 正式依赖入口

- `prepare_dependency_profile.py`: 薄转发入口, 在当前解释器中调用 `experiments.runtime.dependency_preparation`; 仅消费已提交且 ready 的完整哈希锁。
- `prepare_isolated_dependency_environment.py`: 薄转发入口, 使用固定 `uv==0.11.28` 为五个科学 profile 创建并正式准备独立 CPython 子环境。
- `materialize_dependency_lock_candidate.py`: 在与目标 profile 精确匹配的解释器中解析完整 wheel 闭包候选, 只写 `outputs/`, 不直接修改 `configs/`。
- `write_dependency_lock_review_bundle.py`: fresh Linux x86_64 host 资格化入口, CLI 必须使用 `python -I`。脚本使用宿主 Python 标准库下载工具锁固定的 PyPI Linux x86_64 `uv` wheel, 复验 URL、平台文件名和 SHA-256 后直接提取唯一 executable, 不依赖宿主 `venv`、`pip` 或 `ensurepip`。固定 `uv` 创建精确 `workflow_orchestrator` CPython 3.12.13 子解释器, 再由该 child 运行唯一审查包实现。`workflow_orchestrator` 候选不要求自身完整锁; 五个科学 profile 在 orchestrator 锁尚未提交时会在下载前失败, 通过顺序门禁后才准备父环境、创建目标 CPython 子解释器并使用登记的 PyTorch index 解析 wheel 闭包。父 launcher 会重新读取本地与 Drive 的 manifest、精确文件集合、路径、大小和摘要, 不能用退出码0代替受治理审查包。
- `write_reviewed_dependency_hash_lock.py`: 在人工批准后离线复验 Drive 回传审查包、候选生成代码锁、当前 clean Git HEAD、三个文件摘要和 pip resolver 闭包, 并在实际写锁前再次核验 HEAD 与 clean 状态, 然后把规范候选写入 registry 登记且仍缺失的完整锁路径。CLI 只允许由目标 checkout 内同一份脚本修改该 checkout, 不允许从另一代码版本通过 `--root` 写入。该入口不覆盖已有锁、不提交 Git, 写入结论固定不支持论文 claim。
- `write_reviewed_scientific_dependency_hash_locks.py`: 在父编排锁已经提交后, 原子复验同一 clean detached commit 生成的五个科学 profile 审查包。CLI 要求按 registry 顺序逐项批准五个 profile; 全部 manifest、resolver report、wheel SHA-256 与规范锁文本通过后才写入任何目标。任一候选或写入失败都会删除本次已写文件, 因而不会留下部分科学锁集合。该入口不执行 CUDA, 不提交 Git, 也不支持论文 claim。
- `run_formal_workflow_host.py`: 正式结果的唯一 fresh-host 父入口, 必须以 `python -I` 在 Linux x86_64 调用。该入口先要求请求提交对应的 clean detached checkout, 再复用固定 URL、平台文件名和 SHA-256 的 `uv` wheel 引导, 创建 registry 指定的 CPython 3.12.13, 并以当前 HEAD 中的 `workflow_orchestrator` 完整哈希锁执行 `pip install --require-hashes --only-binary=:all:`、`pip check` 与隔离解释器复验。受治理 workflow 结果同时记录父 profile、完整锁摘要、解释器路径和解释器文件摘要; 任一命令失败均不会进入 GPU 或 CPU 闭合业务入口。
- `formal_workflow_entry.py`: 精确 `workflow_orchestrator` 子解释器中的统一 GPU / CPU 子入口。该文件位于 `scripts/`, 不依赖 `paper_workflow/`, 因而 fresh host、普通 GPU 服务器和 Colab 均执行同一内层入口。
- `formal_workflow_environment.py`: 统一发布论文运行层级、固定 FPR、模型 revision、baseline 身份和 workflow 持久化参数。`run_gpu_server_workflow.py` 直接调用该实现；Colab helper 只在外层补充 Notebook 会话起点记录。

两个 prepare 脚本不保存依赖列表、安装逻辑或环境判断。可复用实现位于 `experiments/runtime/`, 因而普通 Linux GPU 服务器与 Colab 使用同一内层 API。

## GPU 入口

- `run_gpu_server_workflow.py`: 精确 `workflow_orchestrator` 子解释器使用的 CPU 父入口, 由 `run_formal_workflow_host.py` 调用。它公开9个正式路由: `image_only_dataset`、`mechanism_ablation`、`external_baseline_tree_ring`、`external_baseline_gaussian_shading`、`external_baseline_shallow_diffuse`、`official_reference_t2smark`、`official_reference_tree_ring`、`official_reference_gaussian_shading` 和 `official_reference_shallow_diffuse`。父解释器必须先通过已提交的 `workflow_orchestrator` 完整锁与当前环境检查, 再发布正式执行锁、调用内层环境配置并进行路由编排; 主方法与消融进入 `sd35_method_runtime_gpu`, method-faithful 与 T2SMark 使用共享隔离 workflow, 三个官方参考路由进入各自独立科学 profile, 宿主环境不安装或执行科学依赖。9个路由均返回同一 `gpu_server_workflow_result` schema, 并绑定父编排依赖证据、正式执行锁、完整 workflow 环境、内层工作流摘要和可选归档记录。7条外部 baseline 路由统一复用 `paper_experiments.runners.persistent_workflow_session`; 6条活动随机化路由的持久根只能由当前 `randomization_repeat_id` 的受治理配置生成, 主方法、质量、消融与 checkpoint 共享同一个只含一次 repeat ID 的父目录。三个跨 repeat 不变路由仍可用 `--persistent-output-dir` 指向服务器持久磁盘或已挂载 Drive, 但 checkpoint 不具备论文证据资格。
- `run_image_only_dataset_runtime.py`: `experiments.runners.image_only_dataset_workload` 的薄 CLI。内层工作负载执行当前论文级别的完整主方法、仅图像检测、共同攻击和正式数据集质量评估。
- `run_runtime_rerun_ablations.py`: `experiments.ablations.mechanism_ablation_workload` 的薄 CLI。内层工作负载对同一完整 Prompt 集执行真实机制消融, 每个变体使用自己的 calibration split 冻结阈值。
- `semantic_watermark_scientific_workflow.py`: 主方法的可脱离 Notebook 父编排实现。它创建一次隔离科学执行, 写入产物级执行绑定, 复用同一受验证子解释器完成绑定打包, 并按调用者显式提供的目标目录镜像本次归档。
- `run_semantic_watermark_scientific_session.py`: 仅转发到 `experiments.runtime.semantic_watermark_scientific_session` 的薄命令入口。科学 session 实现在同一 `sd35_method_runtime_gpu` 子解释器中顺序运行主方法、质量评估和按需消融, 或在互斥的绑定打包模式中验证产物后重新归档。
- `run_gpu_method_qualification.py`: 在 `sd35_method_runtime_gpu` 精确子解释器中执行单 Prompt 真实主方法。加载 SD3.5 前先用当前设备上的小张量真实执行 `torch._assert_async`、`torch.func.linearize` 与 `torch.func.vjp`, 以提前拒绝目标 PyTorch 或 CUDA 后端不兼容；该轻量记录绑定到最终资格化报告, 但不替代716维 VAE/CLIP 真实计算图证据, 且 `supports_paper_claim=false`。
- `write_gpu_method_qualification_report.py`: 从单 Prompt 真实运行结果、更新 JSONL 和检测 JSONL 写出方法算子资格化报告, 并可选消费资源观测与登记预算。`gpu_operator_preflight_ready` 与 `gpu_resource_budget_ready` 独立判定, 资源超限不改变方法算子真实性结论。
- `build_external_baseline_command_plan.py`、`run_external_baseline_command_plan.py` 与 `validate_external_baseline_evidence.py`: 完整论文实验层相应模块的薄 CLI。可复用实现位于 `paper_experiments/baselines/command_plan_builder.py`、`command_plan_execution.py` 与 `evidence_validation_cli.py`。

独立 CUDA 服务器运行外部路由时使用以下入口.该命令与 Notebook 调用相同的恢复、科学隔离和归档实现:

```bash
python scripts/run_gpu_server_workflow.py \
  --workflow external_baseline_tree_ring \
  --paper-run-name probe_paper \
  --repository-commit <40位小写Git提交> \
  --randomization-repeat-id seed_00_key_00
```

活动随机化 GPU 路由禁止用 `--persistent-output-dir` 覆盖目录, 以免多个 seed-key repeat 共用 checkpoint 或结果包。入口必须使用当前论文运行配置派生的 `randomization_repeats/<repeat>` 持久根; Colab 由该规则得到已挂载 Drive 路径。三个跨 repeat 不变的官方参考路由可以显式提供持久目录。父解释器仍只执行 `workflow_orchestrator` 编排逻辑, 不直接导入科学依赖。

## CPU 闭合入口

- `write_randomization_repeat_evidence_package.py`: 显式选择一个已登记 seed-key repeat 的8类随机化 leaf ZIP, 以原始 ZIP 字节写出自包含证据组件。该组件固定 `randomization_aggregate_ready=false` 与 `supports_paper_claim=false`。
- `write_randomization_aggregate_provenance_package.py`: 按权威顺序接收9个 repeat 证据组件与3个跨 repeat 不变 official-reference ZIP, 重新调用生产 validator 后保存12个输入 ZIP 的原始字节。聚合包固定 `randomization_aggregate_ready=true` 与 `supports_paper_claim=false`, 只表示最终统计输入已经闭合。
- `paper_result_closure.py`: 从显式文件或目录中选择唯一有效聚合 ZIP, 核验论文运行层级、冻结 FPR、共同 clean Git 提交和聚合摘要, 随后执行5个可脱离 Notebook 的规范统计 Writer。Writer 分别重建检测与逐攻击统计、全样本及质量匹配配对优势、FID/KID 联合质量统计、真实重运行消融必要性统计、单模型内部风险参数敏感性统计。
- `run_gpu_server_result_closure.py`: CPU 汇总主机入口。`--dry-run` 仍会验证聚合来源与 Git 提交并返回真实命令计划；正式模式调用同一个 `paper_result_closure.py` 实现, 不维护第二套闭合逻辑。

闭合入口会再次读取5个统计产物的 summary、manifest、文件集合和 SHA-256, 并要求它们绑定同一聚合来源、代码提交、运行层级和 FPR。顶层结论分为两个独立字段：

1. `paper_result_evidence_ready`: 5类统计均能从聚合包内样本级记录或原始 Inception 特征重建。
2. `supports_paper_claim`: 统一阴性总体 fixed-FPR、全样本及质量匹配总体优势、正式消融必要性三项门禁同时通过。FID/KID 与单模型内部风险参数敏感性是必需证据, 但不参与中心结论投票。

完整归档只保存不可变聚合来源、5类重建统计、闭合报告及归档 manifest。任一 repeat 缺失、跨运行混选、FPR 漂移、Git 提交不一致、原始记录重算失败或输出摘要漂移都会阻断归档。

CPU 汇总命令示例：

```bash
python scripts/run_gpu_server_result_closure.py \
  --paper-run-name probe_paper \
  --randomization-aggregate-package-path outputs/randomization_aggregate/probe_paper/randomization_aggregate.zip \
  --complete-output-dir outputs/complete_result_packages \
  --repository-commit <40位小写Git提交>
```

当前没有可用的远程 Linux 服务器或本地 CUDA 环境。本地只执行 CPU 测试、静态审计和协议验证；GPU 主方法、消融和 baseline 先在 Colab 运行, 后续 `full_paper` 再迁移到独立 CUDA 服务器。五个 CUDA profile 的真实锁安装、`pip check`、torch/CUDA identity、CUDA 可用性和科学运行必须在 Colab 或 GPU 服务器完成。Notebook 不得内嵌安装逻辑或绕过已提交锁门禁。

所有持久化输出必须位于 `outputs/`; harness 报告必须位于 `outputs/audit_reports/`。
