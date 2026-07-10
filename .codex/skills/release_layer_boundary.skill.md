# Skill Name

release_layer_boundary

## Purpose

约束 SLM-WM 仓库的五层结构与发布抽离边界, 保证核心方法、主方法实验、完整论文实验、独立 CLI 和 Colab 入口职责清晰。

## Scope

适用于新增目录、移动模块、调整 runner、改造 Notebook 入口、配置发布包和引入外部 baseline 适配工程。

## Required Inputs

- `.codex/project_contract.md` 中的三层结构契约。
- `docs/file_organization.md` 与 `docs/release_boundary.md`。
- 目标改动涉及的导入路径和输出路径。

## Required Outputs

- 明确说明改动属于核心方法复现层、完整论文实验层或 Colab 运行层。
- 明确说明是否影响最小方法发布包或完整论文实验发布包。
- 若存在兼容旧入口的过渡代码, 必须说明其迁移目的和后续收敛方向。

## Blocking Rules

1. `main/` 不得导入 `experiments/`、`paper_experiments/`、`scripts/`、`paper_workflow/` 或 `external_baseline/`。
2. `experiments/` 不得导入 `paper_experiments/`、`scripts/`、`paper_workflow/` 或 `external_baseline/`。
3. `paper_experiments/` 不得导入 `scripts/` 或 `paper_workflow/`。
4. `scripts/` 不得导入 `paper_workflow/`。
5. Colab 运行层不得成为正式 records、tables、figures、reports 或 manifests 的唯一实现位置。
6. Notebook 文件必须集中放在 `paper_workflow/notebooks/`, 不得与 helper 模块平铺混放。
7. 外部 baseline 源码缓存不得进入最小方法发布包。
8. baseline 适配、官方参考复现和受治理导入不应放入 `experiments/`。

## Allowed Changes

- 将外部 baseline 相关实现迁移到 `paper_experiments/`。
- 将 Notebook 入口中的正式逻辑下沉到核心方法复现层或完整论文实验层。
- 更新 harness、测试、文档和发布抽离 profile, 以表达三层依赖方向。

## Forbidden Changes

- 为了临时跑通而让 `main/` 或 `experiments/` 反向依赖 Colab helper。
- 将第三方 baseline 源码复制进最小方法发布包。
- 将正式论文结果手工写入 Notebook cell 或未受治理脚本。

## Required Audit Hooks

- `pytest -q`
- `python tools/harness/run_all_audits.py`
