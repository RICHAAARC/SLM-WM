# 完整论文实验层

`paper_experiments/` 保存外部 baseline 适配、官方参考复现编排、受治理结果导入、共同协议比较、跨重复统计和论文证据审计代码。该层可以依赖 `main/` 与 `experiments/`，不得依赖 `scripts/` 或 `paper_workflow/`。

## 规范来源

- 目标方法及其正式角色、样本 schema 和决策接口只引用 `../docs/builds/method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md`。
- 三档同构只引用 `../docs/builds/paper_profile_protocol_isomorphism.md`。
- 质量结论只引用 `../docs/builds/paper_quality_claim_governance.md`。
- 主张决策只引用 `../docs/builds/paper_claim_decision_governance.md`。
- 实际迁移差距只引用 `../docs/builds/project_construction_state.md`。

本 README 不复制方法公式、参数、正式角色全集、writer 文件清单或项目完成状态。

## 子目录职责

```text
baselines/   外部 baseline 适配、官方参考复现和受治理导入协议
runners/     可脱离 Notebook 调用的完整论文实验 runner
analysis/    原始证据重建、统计推断、主张决策和闭合审计
```

## 本层职责

1. 从不可变单 repeat 证据组件重建精确5重复聚合来源，拒绝缺失、重复、额外或身份漂移成员。
2. 对主方法和主表 baseline 使用相同 Prompt 分区、7项核心攻击职责、目标 FPR 和有限样本预算算法；每种方法使用自身真实连续分数独立冻结阈值。
3. 从样本级检测记录、质量特征和消融重运行记录重建统计，不信任 producer 的派生 ready 字段。
4. 参数敏感性只作为单模型内部诊断，不进入正式论文主张、gate 角色或结果闭合必要主张集合。
5. `probe_paper`、`pilot_paper` 和 `full_paper` 固定使用相同方法、7项核心攻击、4个主表 baseline、统计程序、产物 schema 和精确5重复，只允许登记规模、目标 FPR、统计强度及其派生记录数量变化；pilot 是主投稿证据，full 是可选扩展。10项补充攻击保持跨 profile 身份与参数同构，但其运行是可选描述性扩展。
6. 可以消费通过身份摘要和等价性门禁的共享图像、攻击与公开特征原子，但必须按角色、密钥关系和 profile 重建阈值、决策、统计与主张审计。
7. official-reference 证据只承担登记的方法忠实度职责，不替代 common-backbone 主表比较。
8. 质量、配对优势和主张决策只消费 `core_claim_required` exact set；补充攻击结果单列报告，不参与 calibration、baseline 优越性、机制必要性、质量 required gate 或集合级 claim vote。

所有生产者、聚合器和 validator 都必须失败关闭。单 repeat、环境资格、依赖锁、入口可启动和完整包归档分别是不同证据层级，不能相互替代论文科学结论。
