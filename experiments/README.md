# 主方法实验层

`experiments/` 负责把 `main/` 的同一核心实现接入真实 SD3.5，并生产 Prompt split、fixed-FPR calibration/test 原始观测、真实攻击、正式消融、质量特征和实验产物。该层可以依赖 `main/`，不得依赖 `paper_experiments/`、`scripts/` 或 `paper_workflow/`。

## 规范来源

- 算法事实、参数和禁止替代项只引用 `../docs/builds/algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md`。
- 公开接口、运行数据流、正式角色和 schema 只引用 `../docs/builds/method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md`。
- 实际实现差距和验证进度只引用 `../docs/builds/project_construction_state.md`。

本 README 不复制注入索引、载体公式、强度、权重、几何搜索参数或正式角色全集。

## 本层职责

1. 调用 `main/` 的真实生成、嵌入、检测和救回接口，不重写科学方法。
2. 物化三类受治理样本和成功/失败联合观测，失败样本保留在正式分母中。
3. 使用共同的 Prompt 分区和有限样本预算算法校准完整决策器；主方法与 baseline 各自使用自身连续分数，不共享数值阈值。
4. 从同一核心实现生成登记消融，不建立简化方法副本。
5. 从真实持久化图像提取质量证据，不使用 proxy、synthetic、placeholder、随机向量或预制分数。
6. 单 Prompt 资格化只能形成 `supports_paper_claim=false` 的算子事实；论文结论必须进入外层精确9重复聚合。

任何仍验证迁移前方法的 runner 或测试只能证明历史实现回归稳定，不能产生目标方法证据。具体迁移状态只由项目构建状态文档登记。
