# Release Boundary

## 与 extraction profile 的关系

本文件说明发布边界原则, `docs/extraction_profiles.md` 定义可执行的抽离 profile。发布包不应默认等同于开发仓库。

## 发布包类型

### `minimal_method_package`

该包是最小论文方法代码附件, 只保留核心方法、核心协议和最小配置。

默认包含:

```text
main/core/
main/methods/
main/protocol/
configs/
README.md
pyproject.toml
```

默认排除:

```text
main/analysis/
main/cli/
experiments/
paper_experiments/
external_baseline/
scripts/
paper_workflow/
.codex/
tools/harness/
tests/constraints/
outputs/
```

### `paper_artifact_rebuild_package`

该包用于重建论文所需 tables、figures、reports 和 manifests。它可以包含 artifact builders、完整论文实验编排和轻量功能测试, 但不包含外层治理实现、Colab 入口或第三方源码缓存。

默认包含:

```text
main/
configs/
experiments/
paper_experiments/
scripts/
docs/中必要的复现和 schema 文档
tests/functional/
README.md
pyproject.toml
```

默认排除:

```text
.codex/
tools/harness/
external_baseline/
outputs/
tests/constraints/
tests/integration/
paper_workflow/
```

### `full_experiment_execution_package`

该包用于在服务器上产出论文所需完整实验结果。它包含核心方法复现层和完整论文实验层, 但仍不包含 Colab 运行层。若需要第三方官方源码, 应由使用者按 `external_baseline/` 来源登记自行拉取或挂载, 不应把第三方源码缓存直接打入发布包。

默认包含:

```text
main/
configs/
experiments/
paper_experiments/
scripts/
docs/中必要的复现和 schema 文档
tests/functional/
README.md
pyproject.toml
```

默认排除:

```text
.codex/
tools/harness/
paper_workflow/
external_baseline/中的第三方源码缓存
outputs/
tests/constraints/
tests/integration/
```

## 默认进入论文发布包

- `main/`
- `configs/`
- `scripts/` 中必要的复现脚本
- `paper_experiments/` 中完整论文实验所需的受治理适配和导入协议
- `docs/` 中的方法、复现、数据准备和模型准备文档
- `tests/` 中可公开的复现测试
- 必要的 `experiments/` paper protocol

## 默认不进入论文发布包

- `.codex/`
- `tools/harness/`
- `outputs/`
- `external_baseline/` 中的第三方源码缓存
- `paper_workflow/` 中的 Colab 入口
- 本地 Notebook 缓存
- 私有数据或本地绝对路径配置
- 未经治理的临时实验结果

## 说明

该边界适用于论文代码开源前的最小发布抽取。内部治理材料可以保留在开发仓库, 但发布包应优先服务审稿复现和读者理解。
