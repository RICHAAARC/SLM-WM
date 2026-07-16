# 独立执行层

`scripts/` 提供可脱离 Notebook 运行的 GPU、CPU 汇总、结果检查、结果打包和 release CLI。该层可以依赖 `main/`、`experiments/` 和 `paper_experiments/`，不得依赖 `paper_workflow/`。

## 规范来源

脚本不得复制目标方法、统计或门禁实现。方法资格化事实与失败条件只引用两份无状态核心规范；证据结论引用 `docs/builds/paper_claim_decision_governance.md`；实际接线状态只引用 `docs/builds/project_construction_state.md`。

## 本层职责

1. 绑定精确 Git commit、依赖 profile、模型 revision、输入摘要和输出目录。
2. 调用内层真实 runner，并把内层非零状态逐层传播到宿主进程。
3. 保持方法真实性门禁与资源预算门禁相互独立。
4. 编排精确9重复的证据聚合、结果闭合和完整包验证，不手工生成科学结论。
5. 提供三种 release profile 的抽离和包内验证入口；抽离成功只证明文件与入口身份闭合。

Notebook 只能调用本层入口，不能承载资格化、统计、结果闭合或发布逻辑。单 Prompt 资格化报告固定 `supports_paper_claim=false`。
