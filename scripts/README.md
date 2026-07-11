# Scripts

`scripts/` 提供可脱离 Notebook 运行的命令行入口, 不承载方法数学实现。

## 正式依赖入口

- `prepare_dependency_profile.py`: 薄转发入口, 在当前解释器中调用 `experiments.runtime.dependency_preparation`; 仅消费已提交且 ready 的完整哈希锁。
- `prepare_isolated_dependency_environment.py`: 薄转发入口, 使用固定 `uv==0.11.28` 为五个科学 profile 创建并正式准备独立 CPython 子环境。
- `materialize_dependency_lock_candidate.py`: 在与目标 profile 精确匹配的解释器中解析完整 wheel 闭包候选, 只写 `outputs/`, 不直接修改 `configs/`。
- `write_dependency_lock_review_bundle.py`: fresh Linux x86_64 host 资格化入口。脚本先通过 `dependency_qualification_uv_linux_x86_64_lock.txt` 的单 wheel SHA-256 安装固定 `uv`, 创建精确 `workflow_orchestrator` CPython 3.12.13 子解释器, 再由该 child 运行唯一审查包实现。`workflow_orchestrator` 候选不要求自身完整锁; 五个科学 profile 必须先由已提交 orchestrator 完整锁准备父环境, 再创建目标 CPython 子解释器并使用登记的 PyTorch index 解析 wheel 闭包。父 launcher 会重新读取 child manifest 和三个候选文件, 不能用退出码0代替受治理审查包。
- `write_reviewed_dependency_hash_lock.py`: 在人工批准后离线复验 Drive 回传审查包、候选生成代码锁、当前 clean Git HEAD、三个文件摘要和 pip resolver 闭包, 然后把规范候选写入 registry 登记且仍缺失的完整锁路径。该入口不覆盖已有锁、不提交 Git, 写入结论固定不支持论文 claim。

两个 prepare 脚本不保存依赖列表、安装逻辑或环境判断。可复用实现位于 `experiments/runtime/`, 因而普通 Linux GPU 服务器与 Colab 使用同一内层 API。

## GPU 入口

- `run_gpu_server_workflow.py`: 可脱离 Notebook 使用的 CPU 父入口. 公开9个正式路由: `image_only_dataset`、`mechanism_ablation`、`external_baseline_tree_ring`、`external_baseline_gaussian_shading`、`external_baseline_shallow_diffuse`、`official_reference_t2smark`、`official_reference_tree_ring`、`official_reference_gaussian_shading` 和 `official_reference_shallow_diffuse`。父解释器必须先通过已提交的 `workflow_orchestrator` 完整锁与当前环境检查, 再发布正式执行锁和运行身份并进行路由编排; 主方法与消融进入 `sd35_method_runtime_gpu`, method-faithful 与 T2SMark 使用共享隔离 workflow, 三个官方参考路由进入各自独立科学 profile, 宿主环境不安装或执行科学依赖。9个路由均返回同一 `gpu_server_workflow_result` schema, 并绑定父编排依赖证据、正式执行锁、内层工作流摘要和可选归档记录。7条外部 baseline 路由统一复用 `paper_experiments.runners.persistent_workflow_session`; `--persistent-output-dir` 可以指向服务器持久磁盘或已挂载 Drive, 同时保存断点状态和正式归档, 但 checkpoint 不具备论文证据资格。
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
  --persistent-output-dir /mnt/persistent/slm_wm/probe_paper/tree_ring
```

未提供 `--persistent-output-dir` 时, 服务器入口把可交付归档与 checkpoint 写到当前仓库 `outputs/gpu_server_delivery/<paper_run_name>/<workflow_name>/`.父解释器仍只执行 `workflow_orchestrator` 编排逻辑, 不直接导入科学依赖.

## CPU 闭合入口

- `paper_result_closure.py`: 对精确10包输入执行 current-run 清理、显式物化、18步 run-scoped 证据 DAG、语义闭合门禁和最终打包, 返回本次唯一归档路径。
- `run_gpu_server_result_closure.py`: `paper_result_closure.py` 的 CPU 服务器 CLI, 先对已回传结果包执行包内身份 dry-run, 正式运行时冻结输入锁并执行完整闭合。
- `write_pilot_paper_result_records.py`: 物化受治理 records。
- `write_fixed_fpr_threshold_audit_outputs.py`: 从主方法与四个外部 baseline 的原始 observation 独立重算 calibration clean negative 冻结阈值、阈值摘要和逐条判定, 不接受仅由上游声明的 ready 状态。
- `write_primary_baseline_result_candidates.py`: 从三个 method-faithful transfer manifest 的 exact-set collection 与 T2SMark 正式候选生成共同协议记录。
- `write_primary_baseline_evidence_outputs.py`: 联合核验三个 common-backbone exact-set source 与独立 T2SMark formal source, 生成四方法完整证据门禁。
- `write_official_reference_fidelity_evidence_outputs.py`: 在精确结果包物化后独立核验三个 official-reference family 的运行、validation、package input 摘要、归档治理和共同 clean 代码版本, 写入 `outputs/official_reference_fidelity_evidence/<paper_run_name>/`; 正式闭合调用使用 `--require-pass`。该入口只生成补充方法忠实度证据, 不声明主表优势。
- `write_paired_superiority_outputs.py`: 将 SLM-WM 与4个主表 baseline 的完整 test Prompt x attack 记录精确配对, 绑定正式攻击配置、两方法冻结阈值和原始 observation 字节摘要, 写出 `paired_outcomes.jsonl`、4行总体统计表、summary 与 manifest; 正式统计固定使用100000次 Prompt-clustered bootstrap、单侧 bounded Hoeffding Prompt 均值检验和跨4比较的 Holm 校正, exact DP sign-flip 仅作 sharp-null 诊断。
- `write_pilot_paper_result_analysis_outputs.py`: 生成完整逐攻击 CI 与优势比较表; 完整披露允许存在未显著胜出的攻击, 只有 `universal_per_attack_superiority_claim_ready` 限定“每个攻击均显著胜出”的更强主张。
- `write_pilot_paper_complete_result_package.py`: 仅从显式 `--package-path` 物化输入, 重新核验 run-scoped closure input lock 与最终 result closure gate, 再按当前 `paper_run_name` 独占目录执行 fail-closed 打包; 最终 zip 摘要只写入包外 archive receipt。
- `write_paper_artifact_evidence_audit_outputs.py`: 实际读取仅图像盲检原始 JSONL、冻结协议与其余正式表图数据, 共绑定11类源文件; 从原始记录重建分数分布、ROC / DET 后逐列、逐行、逐单元格核验, 同时验证 FID / KID 两行实测状态及关键 ready 一致性, 并记录全部输入路径和字节 SHA-256; 缺表或自造曲线时 fail-closed 为 blocked。
- `write_submission_readiness_outputs.py`: 在实际数据验证与 claim 审计之后执行投稿就绪审计。
- `write_evidence_closure_entry_review_outputs.py`: 由已物化的受治理审计报告自动生成入口清单与确定性决策; 仅当全部检查项为 ready 时输出 `ready_for_evidence_closure`, 不等待人工批准。
- `write_result_closure_gate_outputs.py`: 联合复核输入锁、三方法 official-reference 忠实度、五方法阈值、4个主表 baseline 的配对总体优势、正式 records、表图数据、消融、质量和入口审计; 门禁从五份原始 observation 独立重建 paired outcomes、正式统计和完整 result metrics, 同时复验攻击记录身份、来源文件 SHA-256、schema validation、模板覆盖及 manifest 配置摘要, 只在全部受治理事实一致时允许完整打包。

## 18步执行顺序

1. `write_pilot_paper_result_records.py --materialize-only`
2. `write_official_reference_fidelity_evidence_outputs.py --require-pass`
3. `write_attack_matrix_outputs.py`
4. `write_fixed_fpr_threshold_audit_outputs.py --require-pass`
5. `write_paired_superiority_outputs.py --require-pass`
6. `write_primary_baseline_method_faithful_adapter_protocol.py`
7. `write_primary_baseline_result_candidates.py`
8. `write_primary_baseline_formal_import_protocol.py`
9. `write_primary_baseline_evidence_outputs.py --require-pass`
10. `write_external_baseline_comparison_outputs.py`
11. `write_pilot_paper_result_records.py --require-existing-evidence`
12. `write_pilot_paper_fixed_fpr_common_protocol_outputs.py --require-existing-evidence`
13. `write_pilot_paper_result_analysis_outputs.py`
14. `write_paper_artifact_evidence_audit_outputs.py`
15. `write_submission_readiness_outputs.py`
16. `write_evidence_closure_entry_review_outputs.py`
17. `write_result_closure_gate_outputs.py --require-pass`
18. `write_pilot_paper_complete_result_package.py`

逐攻击结果表属于完整披露证据。主表总体 superiority claim 只由第5步的 Prompt 聚类配对统计及第17步的跨产物语义门禁决定; official-reference 忠实度证据不进入主表。

当前 `8.216.54.104` 没有 GPU, 只能执行 CPU 闭合、审计和打包。GPU 主方法、消融和 baseline 必须在 Colab 或其他 CUDA 服务器运行。

五个 CUDA profile 的 repository 隔离子执行路径均已定义。完整锁候选只解析目标 CPython/Linux x86_64 wheel 闭包和登记 PyTorch index, 不导入 torch 或执行 CUDA, 因而可在无 GPU Linux 服务器完成; 真实锁安装、`pip check`、torch/CUDA identity、CUDA 可用性和科学运行仍必须在 Colab GPU 环境通过。Notebook 不得内嵌安装逻辑或绕过已提交锁门禁。

完整结果包的共享代码白名单只归档 `scripts/` 及更内层的可执行实现, 不归档 `paper_workflow/`、Notebook 或 Colab / Drive 包装。该边界使 CPU 服务器能够仅凭结果包中的内层实现复核输入锁、科学执行绑定和18步证据闭合。

所有持久化输出必须位于 `outputs/`; harness 报告必须位于 `outputs/audit_reports/`。
