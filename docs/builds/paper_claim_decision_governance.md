# 论文证据到结论治理

## 一、文档职责

本文档记录正式实验事实、派生统计、论文主张决策和运行流程就绪状态之间的唯一语义关系。它不保存实验数值, 也不允许人工覆盖 records、统计表或闭合报告中的结论。

## 二、当前证据链基线

| 证据角色 | 主要生产位置 | 主要验证或消费位置 | 当前结论字段 |
| --- | --- | --- | --- |
| fixed-FPR 检测记录 | `experiments/runners/image_only_dataset_runtime.py` | `paper_experiments/analysis/fixed_fpr_threshold_audit.py`、`paper_experiments/analysis/result_closure_gate.py` | `supports_paper_claim`、fixed-FPR ready 字段 |
| baseline 公平对比 | `paper_experiments/baselines/`、`paper_experiments/analysis/paired_superiority.py` | `paper_experiments/analysis/paper_evidence_audit.py`、`paper_experiments/analysis/result_closure_gate.py` | `supports_paper_claim`、superiority ready 字段 |
| FID/KID 与配对质量 | `experiments/artifacts/dataset_level_quality_outputs.py`、`paper_experiments/analysis/randomization_dataset_quality.py` | `scripts/paper_result_closure.py`、`paper_experiments/analysis/result_closure_gate.py` | `conclusion_decision=measured_evidence_component`、`supports_paper_claim=false` |
| 机制必要性消融 | `experiments/ablations/necessity_statistics.py`、`paper_experiments/runners/randomization_ablation_necessity.py` | `scripts/paper_result_closure.py`、`paper_experiments/analysis/result_closure_gate.py` | `necessity_component_decision`、`supports_paper_claim` |
| 参数敏感性 | `experiments/ablations/branch_risk_sensitivity.py`、`paper_experiments/runners/randomization_parameter_sensitivity.py` | `scripts/paper_result_closure.py` | 完整测量 ready 字段、`supports_paper_claim` |
| 结果闭合与完整包 | `scripts/paper_result_closure.py`、`scripts/write_paper_complete_result_package.py` | `paper_experiments/analysis/result_closure_gate.py`、完整包 validator | `conclusion_decision`、`supports_paper_claim` |

当前主要问题是同名布尔字段同时出现在运行组件、统计组件、论文产物和完整包中。部分位置表示“不得直接支持论文主张”, 部分位置表示“证据完整且结论成立”, 因而不能继续把该字段当作跨层原始事实。

## 三、目标主张集合

正式论文结论至少分解为以下稳定主张标识：

```text
fixed_fpr_detection
baseline_superiority
quality_preservation
mechanism_necessity
parameter_robustness
```

每项主张只允许使用以下三态决策：

```text
supported
measured_not_supported
evidence_incomplete
```

三态决策必须同时区分：

1. `evidence_complete`: 所有预登记输入、原子记录、统计和 provenance 是否完整；
2. `scientific_support`: 完整证据是否满足预登记判据；
3. `decision`: 由前两项唯一派生的三态结论。

`registered_claim_set_supported` 只能由预登记 `required_claims` 的三态决策派生。兼容字段 `supports_paper_claim` 暂时保留, 但只能等于对应登记主张集合的派生结果, 不得由 writer 或 manifest 自行写成原始事实。

## 四、运行流程就绪边界

运行流程闭合与科学结论必须独立：

```text
workflow_transfer_ready = (
    probe_workflow_closed
    and protocol_isomorphism_ready
    and artifact_contract_isomorphic
)
```

即使 `probe_paper` 的某项主张为 `measured_not_supported`, 只要真实正式步骤执行完成、产物完整且三个 profile 协议同构, 仍可以证明相同代码和协议能够迁移到 `pilot_paper` 与 `full_paper`。该状态不得被解释为更严格 FPR 下的科学结论已经成立。

## 五、分层职责

- `experiments/` 生产真实运行记录、攻击记录、消融记录和质量原子；
- `paper_experiments/` 重建统计、逐攻击结论和分主张决策；
- `scripts/` 编排结果闭合、验证和归档；
- `paper_workflow/` 只包装 Colab session 与 Drive；
- `main/` 不参与本次治理重构。

该结构属于通用论文证据治理写法。SLM-WM 项目特定部分是固定 FPR、9个 seed-key repeat、14项机制消融、18项分支风险参数敏感性、716维局部特征水平集切空间和图像侧 Q/K 几何证据的绑定方式。
