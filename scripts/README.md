# Scripts

`scripts/` 提供可脱离 Notebook 运行的命令行入口, 不承载方法数学实现。

## GPU 入口

- `run_image_only_dataset_runtime.py`: 执行当前论文级别的完整主方法、仅图像检测和共同攻击。
- `run_runtime_rerun_ablations.py`: 对同一完整 Prompt 集执行真实机制消融, 每个变体使用自己的 calibration split 冻结阈值。
- `build_external_baseline_command_plan.py`: 为单个 Tree-Ring、Gaussian Shading 或 Shallow Diffuse common-backbone 运行构建真实 SD3.5 命令。必须显式传入当前论文级别的 `--target-fpr`。
- `run_external_baseline_command_plan.py`: 执行 baseline 命令计划并汇总 observation。

## CPU 闭合入口

- `run_gpu_server_result_closure.py`: 从已回传结果包执行论文证据闭合。
- `write_pilot_paper_result_records.py`: 物化受治理 records。
- `write_primary_baseline_result_candidates.py`: 从三个 method-faithful transfer manifest 的 exact-set collection 与 T2SMark 正式候选生成共同协议记录。
- `write_primary_baseline_evidence_outputs.py`: 联合核验三个 common-backbone exact-set source 与独立 T2SMark formal source, 生成四方法完整证据门禁。
- `write_pilot_paper_result_analysis_outputs.py`: 生成论文统计表。
- `write_pilot_paper_complete_result_package.py`: 生成完整结果包。
- `write_paper_artifact_evidence_audit_outputs.py` 与 `write_submission_readiness_outputs.py`: 执行证据链和投稿就绪审计。

当前 `8.216.54.104` 没有 GPU, 只能执行 CPU 闭合、审计和打包。GPU 主方法、消融和 baseline 必须在 Colab 或其他 CUDA 服务器运行。

所有持久化输出必须位于 `outputs/`; harness 报告必须位于 `outputs/audit_reports/`。
