# Skill Name

naming_governance

## Purpose

冻结正式文件、目录、模块、配置和字段的语义命名规则。

## Scope

适用于路径、配置键、JSON key、Python dict key、测试名和报告名。

## Required Inputs

- 项目契约。
- 相关文件路径。
- 任务目标和约束条件。

## Required Outputs

- 明确的检查结论。
- 必要的阻断原因。
- 可执行的后续建议。

## Blocking Rules

- 正式名称必须有语义。
- 除 `docs/` 外, 不得使用 `docs/naming_governance.md` 中登记的过程标记词。
- 禁止弱编号、弱版本后缀和无语义名称。

## Allowed Changes

- 更新与本 skill 直接相关的文档、测试或 harness 规则。
- 保持核心运行语义不被无关治理逻辑污染。

## Forbidden Changes

- 绕过项目契约。
- 绕过 harness 审计。
- 将临时实现伪装为正式发布能力。

## Required Audit Hooks

- `run_all_audits.py`
