# Skill Name

progression_guard

## Purpose

防止项目推进跳跃和弱编号命名。

## Scope

适用于 project unit、推进配置和递进式构建决策。

## Required Inputs

- 项目契约。
- 相关文件路径。
- 任务目标和约束条件。

## Required Outputs

- 明确的检查结论。
- 必要的阻断原因。
- 可执行的后续建议。

## Blocking Rules

- 项目推进必须基于明确契约和通过的 harness gates。
- 推进单元名必须使用语义名称。

## Allowed Changes

- 更新与本 skill 直接相关的文档、测试或 harness 规则。
- 保持核心运行语义不被无关治理逻辑污染。

## Forbidden Changes

- 绕过项目契约。
- 绕过 harness 审计。
- 将临时实现伪装为正式发布能力。

## Required Audit Hooks

- `run_all_audits.py`
