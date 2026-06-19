# Skill Name

placeholder_random_field_governance

## Purpose

约束 placeholder 与 random trace 字段后缀和登记规则。

## Scope

适用于配置、records、manifests、测试 fixture 和 Markdown 代码块。

## Required Inputs

- 项目契约。
- 相关文件路径。
- 任务目标和约束条件。

## Required Outputs

- 明确的检查结论。
- 必要的阻断原因。
- 可执行的后续建议。

## Blocking Rules

- Placeholder 字段必须以 `_placeholder` 结尾。
- Random trace 字段必须以 `_random` 或 `_digest_random` 结尾。

## Allowed Changes

- 更新与本 skill 直接相关的文档、测试或 harness 规则。
- 保持核心运行语义不被无关治理逻辑污染。

## Forbidden Changes

- 绕过项目契约。
- 绕过 harness 审计。
- 将临时实现伪装为正式发布能力。

## Required Audit Hooks

- `run_all_audits.py`
