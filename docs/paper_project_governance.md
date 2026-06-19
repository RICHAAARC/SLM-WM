# Paper Project Governance

## 目标

本框架面向论文项目。它要求论文中的关键结论、表格、图和补充材料都能追溯到 repository 中的 governed records、tables、figures、reports 和 manifests。

## 推荐流程

1. 在 `.codex/project_contract.md` 中声明论文目标和阶段。
2. 在 `main/` 中实现核心方法和协议。
3. 在 `experiments/` 中实现阶段性实验 runner。
4. 在 `paper_workflow/` 中编排 Notebook 入口。
5. 在 `scripts/` 中实现数据准备、结果检查和打包命令。
6. 在 `docs/field_registry.md` 中登记字段。
7. 用 `tools/harness/run_all_audits.py` 检查治理规则。
8. 用 `tests/` 分层验证轻量约束、功能行为和正式流程。

## 论文 claim 规则

- supported claim 必须有证据路径。
- 证据路径必须指向 governed artifact。
- placeholder 字段不得支撑 claim。
- Notebook 中临时打印的结果不得直接作为论文 claim 依据。
