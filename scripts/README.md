# Scripts

`scripts/` 提供可脱离 Notebook 运行的命令行入口, 不承载方法数学实现。

## 正式依赖入口

- `prepare_dependency_profile.py`: 薄转发入口, 在当前解释器中调用 `experiments.runtime.dependency_preparation`; 仅消费已提交且 ready 的完整哈希锁。
- `prepare_isolated_dependency_environment.py`: 薄转发入口, 使用固定 `uv==0.11.28` 为五个科学 profile 创建并正式准备独立 CPython 子环境。
- `materialize_dependency_lock_candidate.py`: 在与目标 profile 精确匹配的解释器中解析完整 wheel 闭包候选, 只写 `outputs/`, 不直接修改 `configs/`。
- `write_dependency_lock_review_bundle.py`: 仅允许 CPU `workflow_orchestrator` 在当前解释器物化候选且不传 PyTorch index; 其余五个 CUDA 科学 profile 必须先准备父编排环境, 再 provision 独立子解释器并在子解释器中运行同一物化器。脚本会从实际 pip report 重建规范候选锁并与候选文件及 provenance 逐项核对。

两个 prepare 脚本不保存依赖列表、安装逻辑或环境判断。可复用实现位于 `experiments/runtime/`, 因而普通 Linux GPU 服务器与 Colab 使用同一内层 API。

## GPU 入口

- `run_image_only_dataset_runtime.py`: 执行当前论文级别的完整主方法、仅图像检测和共同攻击。
- `run_runtime_rerun_ablations.py`: 对同一完整 Prompt 集执行真实机制消融, 每个变体使用自己的 calibration split 冻结阈值。
- `build_external_baseline_command_plan.py`: 为单个 Tree-Ring、Gaussian Shading 或 Shallow Diffuse common-backbone 运行构建真实 SD3.5 命令。必须显式传入当前论文级别的 `--target-fpr`。
- `run_external_baseline_command_plan.py`: 执行 baseline 命令计划并汇总 observation。

## CPU 闭合入口

- `run_gpu_server_result_closure.py`: 在 CPU 汇总服务器对已回传结果包执行包内身份 dry-run, 正式运行时冻结精确10包输入锁并执行18步论文证据闭合。
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

所有持久化输出必须位于 `outputs/`; harness 报告必须位于 `outputs/audit_reports/`。
