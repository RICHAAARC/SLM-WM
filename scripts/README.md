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

- `write_randomization_repeat_evidence_package.py`: 显式选择一个已登记 seed-key repeat 的7类随机化 leaf ZIP, 以原始 ZIP 字节嵌套写出自包含证据包。该包固定 `randomization_aggregate_ready=false` 与 `supports_paper_claim=false`。
- `write_randomization_aggregate_provenance_package.py`: 按权威顺序显式接收9个单 repeat 证据组件和3个跨 repeat 不变 official-reference ZIP, 重新调用生产 validator / inspector 后保存12个输入 ZIP 的原始字节。聚合包固定 `randomization_aggregate_ready=true` 与 `supports_paper_claim=false`; 它只证明正式统计输入已闭合, 不直接支持论文结论。Manifest 的重建命令只接受当前 aggregate ZIP 路径参数, 由层内入口在临时目录安全提取12个原始 ZIP 并重新执行全部生产门禁, 不引用只能在压缩包内部看到的成员路径。
- `paper_experiments/runners/paper_claim_provenance.py`: 全部正式结论 Writer 共用的来源边界。精确9+3聚合来源构造器与独立 validator 已实现；在各 Writer 完成不可变来源对象绑定、原始记录重算和 aggregate digest 传播前, 相应公开入口仍提前拒绝且不创建输出。
- `paper_result_closure.py`: 仅承担精确9重复聚合后的论文结果闭合。没有9重复聚合证据时直接拒绝构造或执行论文闭合 DAG。
- `run_gpu_server_result_closure.py`: 精确9重复聚合后的 CPU 汇总服务器入口。单 repeat 保存与传输必须使用随机化证据打包入口。
- `write_pilot_paper_result_records.py`: 正式 records Writer 必须先取得版本化精确9重复聚合验证器返回的来源对象; 在该 Writer 完成不可变来源对象绑定前, 公开入口在读取任何输入和创建任何输出前无条件拒绝执行。单 repeat ZIP 不得通过独立物化模式写入正式结果目录。
- `write_pilot_paper_fixed_fpr_common_protocol_outputs.py`: 只负责已验证精确9重复聚合后的共同协议产物。该入口不接受单 repeat 包、调用方声明的 ready 字段或其他输入身份替代物; 在共同协议 Writer 完成不可变 aggregate 来源绑定前保持失败即关闭。
- `write_fixed_fpr_threshold_audit_outputs.py`: 正式职责是从9个重复中主方法与四个外部 baseline 的原始 observation 独立重算45个 calibration clean-negative 冻结阈值、阈值摘要和逐条判定；当前公开入口在精确聚合记录提取器接入前失败即关闭。
- `write_primary_baseline_result_candidates.py`: 从三个 method-faithful transfer manifest 的 exact-set collection 与 T2SMark 正式候选生成共同协议记录。
- `write_primary_baseline_evidence_outputs.py`: 联合核验三个 common-backbone exact-set source 与独立 T2SMark formal source, 生成四方法完整证据门禁。
- `write_official_reference_fidelity_evidence_outputs.py`: 在精确结果包物化后独立核验三个 official-reference family 的运行、validation、package input 摘要、归档治理和共同 clean 代码版本, 写入 `outputs/official_reference_fidelity_evidence/<paper_run_name>/`; 正式闭合调用使用 `--require-pass`。该入口只生成补充方法忠实度证据, 不声明主表优势。
- `write_paired_superiority_outputs.py`: 正式职责是把9个重复中的 SLM-WM 与4个主表 baseline 的完整 test Prompt × attack 记录精确配对, 并执行100000次 Prompt-clustered bootstrap、单侧 bounded Hoeffding Prompt 均值检验和跨4比较 Holm 校正；当前公开入口在跨重复原始记录重算接入前失败即关闭。
- `write_pilot_paper_result_analysis_outputs.py`: 正式职责是从跨重复 records 生成完整逐攻击 CI、优势比较表、失败记录与真实失败案例图；当前公开入口在跨重复 records Writer 接入前失败即关闭。
- `write_pilot_paper_complete_result_package.py`: 只接受通过版本化精确9重复聚合验证器返回的来源对象, 从该对象绑定的9个随机化组件与3个跨重复不变包物化输入, 再核验最终结果门禁并按当前 `paper_run_name` 独占目录执行 fail-closed 打包; 跨重复 records、共同协议和结果门禁尚未接入时公开入口在创建输出前拒绝执行, 最终 zip 摘要只写入包外 archive receipt。
- `write_paper_artifact_evidence_audit_outputs.py`: 正式职责是读取跨重复仅图像盲检原始 JSONL、冻结协议与正式表图数据并重建 ROC / DET、FID / KID 和关键统计；当前公开入口在上游正式 Writer 接入前失败即关闭。
- `write_submission_readiness_outputs.py`: 正式职责是在实际数据验证与 claim 审计之后执行投稿就绪审计；当前公开入口随上游正式 Writer 一起失败即关闭。
- `write_evidence_closure_entry_review_outputs.py`: 正式职责是由已物化的受治理审计报告生成入口清单与确定性决策；当前公开入口随上游正式 Writer 一起失败即关闭。
- `write_result_closure_gate_outputs.py`: 正式职责是联合复核精确9重复聚合来源、三方法 official-reference 忠实度、45个 method-repeat 阈值、4个主表 baseline 的配对总体优势、正式 records、表图、消融、质量和入口审计；当前公开入口在全部上游跨重复 Writer 接入前失败即关闭。

## 聚合闭合执行边界

下列17项是冻结的目标执行顺序。只有权威9个 repeat 全部完成、原始证据重算通过且版本化聚合来源验证成立后才能执行；当前尚未接入跨重复重算的公开 Writer 会在创建输出前拒绝。任何单 repeat 输入都不能调用这些论文 claim 产物。

1. `write_official_reference_fidelity_evidence_outputs.py --require-pass`
2. `write_attack_matrix_outputs.py`
3. `write_fixed_fpr_threshold_audit_outputs.py --require-pass`
4. `write_paired_superiority_outputs.py --require-pass`
5. `write_primary_baseline_method_faithful_adapter_protocol.py`
6. `write_primary_baseline_result_candidates.py`
7. `write_primary_baseline_formal_import_protocol.py`
8. `write_primary_baseline_evidence_outputs.py --require-pass`
9. `write_external_baseline_comparison_outputs.py`
10. `write_pilot_paper_result_records.py --require-existing-evidence`
11. `write_pilot_paper_fixed_fpr_common_protocol_outputs.py --require-existing-evidence`
12. `write_pilot_paper_result_analysis_outputs.py`
13. `write_paper_artifact_evidence_audit_outputs.py`
14. `write_submission_readiness_outputs.py`
15. `write_evidence_closure_entry_review_outputs.py`
16. `write_result_closure_gate_outputs.py --require-pass`
17. `write_pilot_paper_complete_result_package.py`

逐攻击结果表属于完整披露证据。主表总体 superiority claim 只由第4步的 Prompt 聚类配对统计及第16步的跨产物语义门禁决定; official-reference 忠实度证据不进入主表。

当前没有可用的远程 Linux 服务器或本地 CUDA 环境。本地只执行 CPU 测试、静态审计和协议验证；GPU 主方法、消融和 baseline 先在 Colab 运行, 后续 full_paper 再迁移到独立 CUDA 服务器。

五个 CUDA profile 的 repository 隔离子执行路径均已定义。完整锁候选只解析目标 CPython/Linux x86_64 wheel 闭包和登记 PyTorch index, 不导入 torch 或执行 CUDA, 因而可在无 GPU Linux 服务器完成; 真实锁安装、`pip check`、torch/CUDA identity、CUDA 可用性和科学运行仍必须在 Colab GPU 环境通过。Notebook 不得内嵌安装逻辑或绕过已提交锁门禁。

完整结果包的共享代码白名单只归档 `scripts/` 及更内层的可执行实现, 不归档 `paper_workflow/`、Notebook 或 Colab / Drive 包装。该边界使 CPU 服务器能够仅凭结果包中的内层实现复核精确9重复聚合来源、科学执行绑定和17步证据闭合。

所有持久化输出必须位于 `outputs/`; harness 报告必须位于 `outputs/audit_reports/`。
