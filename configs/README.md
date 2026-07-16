# 配置目录

`configs/` 保存模型、方法、攻击、论文 profile、依赖和证据协议的唯一机器可读配置。配置不得定义第二套算法公式。

## 状态来源

配置迁移状态只以 `docs/builds/project_construction_state.md` 为准。现有配置在目标方法身份原子切换完成前，不代表“语义显著性自适应内容-几何双链潜空间水印”已经配置完成，也不得用于生成目标方法正式结果。

目标方法字段的精确集合见 `docs/builds/method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md#10-配置契约`。完成代码迁移时必须：

1. 删除旧风险场、Jacobian、CG、Null Space 和三注入字段；不得删除目标 Q/K 四分量、alignment 与救回所需身份。
2. 登记 S/T/R/Q 路由、单一注入索引、LF/HF/geometry 固定强度、总预算、Q/K 关系公式、几何捕获域与搜索参数。
3. 重建方法定义摘要和语义追踪登记表。
4. 让配置加载器拒绝未知旧字段，不能静默忽略。

## 保持不变的外层配置

- 模型 ID 与精确 revision。
- Prompt 来源与三档嵌套集合。
- 攻击集合和 baseline 来源登记。
- 三档 profile 的产物 schema 与命令依赖图。
- 目标字段、迁移前字段和共享字段的唯一机器生命周期登记。
- 独立 DINOv2 质量评估器。
- 依赖 profile 和全哈希锁。
- PRG 跨平台固定向量。

`probe_paper`、`pilot_paper` 和 `full_paper` 只能在登记的样本规模、目标 FPR `0.1/0.01/0.001` 和统计强度字段上变化。
