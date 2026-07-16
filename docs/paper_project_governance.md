# Paper Project Governance

## 目标

本框架面向论文项目。它要求论文中的关键结论、表格、图和补充材料都能追溯到 repository 中的 governed records、tables、figures、reports 和 manifests。

## 推荐流程

1. 在 `.codex/project_contract.md` 中声明论文目标和阶段。
2. 在 `main/` 中实现核心方法，不引入实验编排、论文统计或 Notebook 依赖。
3. 在 `experiments/` 中实现主方法运行、攻击、校准、质量记录和消融协议。
4. 在 `paper_experiments/` 中实现跨重复聚合、baseline 对比和论文结论决策。
5. 在 `scripts/` 中实现可脱离 Notebook 的服务器运行、结果检查和打包命令。
6. 在 `paper_workflow/` 中提供只调用 `scripts/` 的薄 Notebook 入口，不定义方法、统计或门禁逻辑。
7. 在 `docs/field_registry.md` 中登记字段。
8. 用 `tools/harness/run_all_audits.py` 检查治理规则。
9. 用 `tests/` 分层验证轻量约束、功能行为和正式流程。

## 构建约束

核心算法路径应突出方法机制、数据流和统计边界。项目不鼓励在多个业务函数中重复书写相同字段、相同配置项或相同数据结构的防御式校验与错误信息。

重复校验应收敛到以下位置:

1. 配置加载阶段校验。
2. dataclass 构造阶段校验。
3. 专门的 schema validator。
4. 轻量约束测试或功能测试。

业务函数内部只保留关键边界校验, 例如外部输入边界、跨模块协议边界、不可恢复状态边界或会导致静默错误的核心算法前置条件。该约束属于通用工程写法; 本项目的特殊要求是, SLM-WM 的核心算法实现必须优先保持方法逻辑清晰, 避免被重复防御式样板代码淹没。

## 论文 claim 规则

- supported claim 必须有证据路径。
- 证据路径必须指向 governed artifact。
- placeholder 字段不得支撑 claim。
- Notebook 中临时打印的结果不得直接作为论文 claim 依据。
