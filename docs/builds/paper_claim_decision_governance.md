# 论文证据到结论治理

## 一、文档职责

本文档定义正式实验事实、派生统计、论文主张决策和运行流程就绪状态之间的唯一语义关系。它不保存项目进度或实验数值，也不允许人工覆盖 records、统计表或闭合报告中的结论。

三档规模与流程迁移的唯一规范是 `paper_profile_protocol_isomorphism.md`；质量估计对象、区间和非劣效边界的唯一规范是 `paper_quality_claim_governance.md`。本文件只能消费两者的受治理输出，不得复制或修改其规模字段、统计公式和阈值。项目实际接线状态、迁移差距和验证进度只由 `project_construction_state.md` 登记。

## 二、证据链职责

| 证据角色 | 生产职责 | 验证或消费职责 | 结论边界 |
| --- | --- | --- | --- |
| fixed-FPR 检测记录 | `experiments/` 生成三类真实样本观测和冻结决策协议 | `paper_experiments/` 重建逐攻击和跨重复统计 | 原子记录不得直接声明论文主张成立 |
| baseline 公平对比 | `paper_experiments/baselines/` 导入受治理真实结果 | 配对优势和证据审计模块复验共同协议 | 每种方法使用自身连续分数独立校准，不共享数值阈值 |
| FID/KID 与配对质量 | `experiments/` 生产真实图像和质量特征 | 随机化质量聚合与质量决策模块 | 诊断指标不得替代独立质量证据 |
| 机制必要性消融 | `experiments/ablations/` 从同一核心实现生成登记角色 | 随机化消融聚合器验证必要性 | 缺失角色、额外正式角色或复用主方法阈值均失败关闭 |
| 单模型内部参数敏感性 | `experiments/` 小规模单因素运行 | 诊断聚合器记录局部稳定性 | 只解释固定参数依据，不进入正式论文主张集合 |
| 结果闭合与完整包 | `scripts/` 编排闭合和归档 | 完整包 validator 重建全部摘要与决策 | 归档完整不等于科学结论成立 |

跨层出现的同名兼容布尔字段只能是受治理决策的派生结果，不能作为生产者写入的原始科学事实。

## 三、正式主张集合

正式论文主张集合精确为：

```text
fixed_fpr_detection
baseline_superiority
quality_preservation
mechanism_necessity
```

单模型内部参数敏感性是诊断证据，不是第五项主张，也不得作为 `optional_claims`、gate 角色或 `registered_claim_set_supported` 的输入。

每项正式主张只允许使用以下三态决策：

```text
supported
measured_not_supported
evidence_incomplete
```

三态决策必须同时区分：

1. `evidence_complete`：全部预登记输入、原子记录、统计和 provenance 是否完整。
2. `scientific_support`：完整证据是否满足预登记判据。
3. `decision`：由前两项唯一派生的三态结论。

`registered_claim_set_supported` 只能由 `configs/paper_claim_registry.json` 登记的全部 `required_claims` 派生。兼容字段 `supports_paper_claim` 只能等于对应登记主张集合的派生结果，不得由 writer、manifest 或 Notebook 自行写成原始事实。集中构造与校验职责位于 `paper_experiments/analysis/paper_claim_decisions.py`。

质量结论必须消费 Prompt 条件 KID、配对感知质量、独立视觉内容质量、逐攻击质量和跨攻击质量的受治理统计。任一必需原子或区间缺失时，`quality_preservation` 必须为 `evidence_incomplete`，不得由诊断代理、同源机制一致性分数或旧兼容状态替代。

## 四、运行流程就绪边界

运行流程闭合与科学结论必须独立：

```text
workflow_transfer_ready = (
    probe_workflow_closed
    and protocol_isomorphism_ready
    and artifact_contract_isomorphic
)
```

`workflow_transfer_ready=true` 只说明同一实现、决策程序和产物契约可以从 `probe_paper` 迁移到另外两个规模，不表示更严格 FPR 下的科学结论成立。即使 `probe_paper` 的某项主张为 `measured_not_supported`，只要真实正式步骤执行完成、产物完整且三档协议同构，流程迁移结论仍可成立。

`scientific_scope_by_profile` 必须分别保存三个工作点的科学作用域。只有具有自身正式结果的 profile 才能产生该 profile 的科学支持结论；不存在自身结果时必须记录 `evidence_incomplete`、`scientific_support=null` 和 `scientific_support_transferred_from_probe=false`。

## 五、分层职责

- `experiments/` 生产真实运行记录、攻击记录、最小必要消融记录和质量原子。
- `paper_experiments/` 重建统计、逐攻击结论和分主张决策。
- `scripts/` 编排结果闭合、验证和归档。
- `paper_workflow/` 只包装 Colab session 与 Drive。
- `main/` 只提供核心方法，不依赖本结论治理层。

该结构属于通用论文证据治理写法。SLM-WM 项目特定方法语义只引用两份无状态核心规范，不在本文重复公式、参数或完整角色集合。

## 六、协议变更原子性

方法或证据 schema 发生受治理变更时，必须在同一变更单元中完成：

1. 更新主张登记和必要主张集合。
2. 更新真实生产者的原子记录角色和 schema。
3. 更新主张决策输入映射，同时保持三态决策语义。
4. 更新结果闭合和完整包消费者，拒绝旧方法字段或未登记额外文件。
5. 更新 field registry 和产物契约。
6. 使用真实格式的原始记录执行生产者到消费者正向集成测试，不得手工注入最终 `supported` 决策。

验收必须分别报告 `evidence_complete`、`scientific_support` 和 `decision`。任何 writer、manifest 或 Notebook 直接写入最终支持状态都属于阻断违规。
