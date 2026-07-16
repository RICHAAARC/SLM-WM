# 论文产物重建规则

1. records 是数值事实来源。
2. tables、figures 与 reports 只能从 records、schema 和 manifests 重建。
3. manifest 必须记录输入路径、输出路径、配置摘要、代码版本和重建命令。
4. 结果记录缺失时重建命令必须失败或输出明确的未闭合报告, 不得写入替代数值。
5. 失败案例图必须引用真实 attacked image 文件; 图像缺失时停止构图。
6. baseline comparison 只消费 formal import 的 accepted records。
7. 正式消融表只消费重新生成、重新攻击和重新检测的变体 records。
8. 单模型内部参数敏感性表只消费算法原语登记的小规模单因素设置，各设置必须独立重新生成、重新攻击、重新检测和校准，不得复用名义设置阈值，也不得把敏感性结果反馈修改正式 test 参数。
9. 参数敏感性固定使用一个登记 repeat 和小规模 Prompt 子集，只形成描述性诊断与 schema 复验；不得扩张为5重复推断，不得进入论文主张或 release gate。
10. `pilot_paper` 是主投稿重建目标；`full_paper` 是可选扩展。full 产物缺失必须报告为扩展证据不完整，但不能阻断完整 pilot 结果包或把 pilot 结论外推到 FPR=0.001。
11. 图像、普通攻击、VAE/Q/K 原子和质量特征可从通过摘要复验的共享缓存重建来源关系；阈值、角色/密钥决策、逐 profile 统计和 claim audit 必须从原子 records 重新计算，不得从缓存写入最终结论。

主要重建入口:

- `scripts/write_paper_result_records.py`
- `scripts/write_paper_fixed_fpr_common_protocol_outputs.py`
- `scripts/write_paper_result_analysis_outputs.py`
- `scripts/write_external_baseline_comparison_outputs.py`
- `scripts/write_paper_artifact_evidence_audit_outputs.py`
- `scripts/write_submission_readiness_outputs.py`
- `paper_experiments/runners/single_model_parameter_sensitivity_diagnostic.py`（目标诊断 writer；是否已经实现只由项目构建状态规范登记）

全部持久化产物写入 `outputs/`。
