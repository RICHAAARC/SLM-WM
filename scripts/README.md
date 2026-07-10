# Scripts

此目录保存命令行辅助脚本、服务器入口、结果重建命令和发布检查命令。脚本可以包装 `main/`、`experiments/` 与 `paper_experiments/` 的正式实现, 但不应成为唯一算法实现位置。

## 服务器运行入口

- `run_gpu_server_workflow.py`: 不依赖 Notebook 或 Google Drive 的单流程服务器入口。可运行主方法、攻击、质量、method-faithful baseline 和 official reference。
- `run_gpu_server_result_closure.py`: 汇总服务器结果闭合入口。它从本地交换目录读取各计算服务器上传的结果包, 并生成完整结果包。

示例:

```bash
python scripts/run_gpu_server_workflow.py \
  --workflow aligned_rescoring \
  --paper-run-name pilot_paper \
  --result-root outputs/gpu_server_results/pilot_paper

python scripts/run_gpu_server_result_closure.py \
  --paper-run-name pilot_paper \
  --package-search-root outputs/gpu_server_results/pilot_paper \
  --complete-output-dir outputs/gpu_server_results/pilot_paper/complete_result_package
```

## 结果重建与论文证据命令

- `write_pilot_paper_result_records.py`: 从结果包物化 records, 并生成当前运行层级的正式结果记录。
- `write_pilot_paper_fixed_fpr_common_protocol_outputs.py`: 重建 fixed-FPR common protocol、bootstrap CI 和 claim readiness。
- `write_pilot_paper_complete_result_package.py`: 生成完整结果包。
- `write_attack_matrix_outputs.py`: 重建攻击矩阵表和 manifest。
- `write_external_baseline_comparison_outputs.py`: 重建外部 baseline 对比表。
- `write_internal_ablation_outputs.py`: 重建内部消融证据。
- `write_dataset_level_quality_outputs.py`: 重建 dataset-level 质量证据摘要; 正式表只输出 Inception FID / KID, proxy 指标单独输出为诊断表。

上述命令必须读取受治理 records、manifests 或结果包, 不得手工拼接正式论文结论。

## baseline 与辅助命令

- `build_external_baseline_command_plan.py`: 生成外部 baseline 显式命令计划。
- `run_external_baseline_command_plan.py`: 执行外部 baseline 命令计划并汇总 observation。
- `validate_external_baseline_evidence.py`: 校验 baseline 证据边界。
- `verify_drive_artifacts.py`: 检查 Drive 或本地同步目录中的结果包覆盖情况。
- `sync_local_outputs_to_drive.py`: 将本地结果同步到指定远程目录。
- `extract_minimal_paper_package.py`: 抽取最小方法发布包或完整论文实验发布包。

## 输出边界

所有持久化输出必须写入 `outputs/` 下的语义子目录。harness 审计报告必须写入 `outputs/audit_reports/`。脚本不得在仓库根目录或源码目录写入运行产物。
