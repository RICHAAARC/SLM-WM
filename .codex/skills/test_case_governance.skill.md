# Skill Name

test_case_governance

## Purpose

约束测试目录、marker、fixture 和运行成本。

## Scope

适用于 `tests/`、`pyproject.toml` 和测试相关 harness。

## Required Inputs

- 项目契约。
- 相关文件路径。
- 任务目标和约束条件。

## Required Outputs

- 明确的检查结论。
- 必要的阻断原因。
- 可执行的后续建议。

## Blocking Rules

- 根目录平铺 `tests/test_*.py` 被禁止。
- 重型测试必须进入默认排除路径。

## Allowed Changes

- 更新与本 skill 直接相关的文档、测试或 harness 规则。
- 保持核心运行语义不被无关治理逻辑污染。

## Forbidden Changes

- 绕过项目契约。
- 绕过 harness 审计。
- 将临时实现伪装为正式发布能力。

## Required Audit Hooks

- `run_all_audits.py`
