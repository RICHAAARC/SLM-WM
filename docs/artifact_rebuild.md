# 论文产物重建规则

1. records 是数值事实来源。
2. tables、figures 与 reports 只能从 records、schema 和 manifests 重建。
3. manifest 必须记录输入路径、输出路径、配置摘要、代码版本和重建命令。
4. 结果记录缺失时重建命令必须失败或输出明确的未闭合报告, 不得写入替代数值。
5. 失败案例图必须引用真实 attacked image 文件; 图像缺失时停止构图。
6. baseline comparison 只消费 formal import 的 accepted records。
7. 正式消融表只消费重新生成、重新攻击和重新检测的变体 records。

主要重建入口:

- `scripts/write_pilot_paper_result_records.py`
- `scripts/write_pilot_paper_fixed_fpr_common_protocol_outputs.py`
- `scripts/write_pilot_paper_result_analysis_outputs.py`
- `scripts/write_external_baseline_comparison_outputs.py`
- `scripts/write_paper_artifact_evidence_audit_outputs.py`
- `scripts/write_submission_readiness_outputs.py`

全部持久化产物写入 `outputs/`。
