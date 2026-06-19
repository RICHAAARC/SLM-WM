# Skill Name

minimal_release

## Purpose

定义最小发布边界, 防止过早发布临时资产。

## Scope

适用于发布目录、发布清单和 release extraction。

## Required Inputs

- 项目契约。
- 相关文件路径。
- 任务目标和约束条件。

## Required Outputs

- 明确的检查结论。
- 必要的阻断原因。
- 可执行的后续建议。

## Blocking Rules

- release 前必须通过测试和 harness。
- 临时治理输出和本地审计报告默认不进入发布包。

## Allowed Changes

- 更新与本 skill 直接相关的文档、测试或 harness 规则。
- 保持核心运行语义不被无关治理逻辑污染。

## Forbidden Changes

- 绕过项目契约。
- 绕过 harness 审计。
- 将临时实现伪装为正式发布能力。

## Required Audit Hooks

- `run_all_audits.py`
