# Skill Name

repository_intake

## Purpose

在修改前检查仓库状态、目录边界和当前推进单元。

## Scope

适用于任何新增、迁移、重构或发布准备任务。

## Required Inputs

- 项目契约。
- 相关文件路径。
- 任务目标和约束条件。

## Required Outputs

- 明确的检查结论。
- 必要的阻断原因。
- 可执行的后续建议。

## Blocking Rules

- 修改前必须读取 `.codex/project_contract.md`。
- 修改前必须确认目标路径属于允许边界。

## Allowed Changes

- 更新与本 skill 直接相关的文档、测试或 harness 规则。
- 保持核心运行语义不被无关治理逻辑污染。

## Forbidden Changes

- 绕过项目契约。
- 绕过 harness 审计。
- 将临时实现伪装为正式发布能力。

## Required Audit Hooks

- `run_all_audits.py`
