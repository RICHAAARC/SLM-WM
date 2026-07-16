# 核心方法层

`main/` 只允许保存论文核心方法和最小数学工具，不得依赖 `experiments/`、`paper_experiments/`、`scripts/`、`paper_workflow/` 或 `tools/`。

## 目标方法职责

目标“语义显著性自适应内容-几何双链潜空间水印”由内容链和几何链共同构成，二者均属于最小方法发布边界。唯一算法定义见 `docs/builds/algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md`，目标目录、公开接口和数据流见 `docs/builds/method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md`。本 README 不复制公式、参数或 schema。

## 发布边界

代码中存在某个算子不表示目标方法整体已经实现。最小方法包只发布冻结目标接口及其必要实现；开发仓库中的迁移状态、影响顺序和历史兼容信息不属于该包的运行依赖。

复用不等于原文件整体保留。目标迁移完成后不得保留可被正式入口选择的旧方法别名、兼容开关或隐藏回退。
