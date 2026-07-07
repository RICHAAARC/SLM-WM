# Skill Name

notebook_entrypoint

## Purpose

确保 Notebook 只作为入口, 不成为唯一实现路径。

## Scope

适用于 `.ipynb`、Notebook helper、scripts 和 artifact 输出路径。

## Required Inputs

- 项目契约。
- 相关文件路径。
- 任务目标和约束条件。

## Required Outputs

- 明确的检查结论。
- 必要的阻断原因。
- 可执行的后续建议。

## Blocking Rules

- Notebook 文件必须集中放在 `paper_workflow/notebooks/`。
- Notebook 不得直接写正式 records、tables、figures 或 reports。
- 核心协议逻辑不得只存在于 Notebook cell 中。

## Allowed Changes

- 更新与本 skill 直接相关的文档、测试或 harness 规则。
- 保持核心运行语义不被无关治理逻辑污染。

## Forbidden Changes

- 绕过项目契约。
- 绕过 harness 审计。
- 将临时实现伪装为正式发布能力。

## Required Audit Hooks

- `run_all_audits.py`
