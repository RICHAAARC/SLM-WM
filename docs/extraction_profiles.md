# Extraction Profiles

## 文档定位

本文档定义论文相关研究项目的可抽离边界。其目标是保证开发仓库可以保留完整治理能力, 但论文发表或复现附件可以只携带必要代码、配置和说明。

## Profile 总览

| profile_name | purpose | includes_governance_layer | includes_artifact_builders | intended_audience |
| --- | --- | --- | --- | --- |
| development_repository | 完整开发仓库, 包含治理、测试、Notebook workflow 和 harness。 | true | true | 项目作者和协作者。 |
| paper_artifact_rebuild_package | 论文图表和报告重建附件, 用于从 governed records 重建 tables、figures、reports 和 manifests。 | false | true | 审稿人、读者和复现实验者。 |
| full_experiment_execution_package | 完整论文实验执行附件, 用于在服务器上产出论文证据结果。 | false | true | 需要复现实验主流程的审稿人和作者。 |
| minimal_method_package | 最小论文方法代码附件, 只保留核心方法、核心协议和最小配置。 | false | false | 只需要理解或复用方法实现的读者。 |

## `development_repository`

该 profile 保留完整仓库内容, 用于持续开发、治理审计和 Agent 协作。

### 默认包含

```text
.codex/
configs/
docs/
experiments/
external_baseline/
main/
paper_experiments/
paper_workflow/
scripts/
tests/
tools/harness/
README.md
pyproject.toml
AGENTS.md
```

### 默认排除

```text
outputs/
__pycache__/
.pytest_cache/
.venv/
dist/
build/
```

## `paper_artifact_rebuild_package`

该 profile 面向论文图表、表格、报告和 manifest 的重建。它可以包含产物生成脚本, 但不应包含 Agent 契约、harness 审计实现或本地运行输出。

### 默认包含

```text
main/
configs/
experiments/
paper_experiments/
scripts/
docs/artifact_rebuild.md
docs/field_registry.md
docs/file_organization.md
docs/release_boundary.md
docs/extraction_profiles.md
docs/intermediate_state_governance.md
tests/functional/
README.md
pyproject.toml
```

### 默认排除

```text
.codex/
tools/harness/
external_baseline/
outputs/
paper_workflow/
tests/constraints/
tests/integration/
tests/helpers/
本地 Notebook 缓存
私有数据
```

## `full_experiment_execution_package`

该 profile 面向服务器复现实验。它包含核心方法复现层和完整论文实验层, 不包含 Colab 运行入口和第三方源码缓存。外部 baseline 的官方源码应由使用者按来源登记自行下载或挂载。

### 默认包含

```text
main/
configs/
experiments/
paper_experiments/
scripts/
docs/artifact_rebuild.md
docs/field_registry.md
docs/file_organization.md
docs/release_boundary.md
docs/release_layer_boundary.md
docs/extraction_profiles.md
docs/intermediate_state_governance.md
tests/functional/
README.md
pyproject.toml
```

### 默认排除

```text
.codex/
tools/harness/
external_baseline/
outputs/
paper_workflow/
tests/constraints/
tests/integration/
tests/helpers/
本地 Notebook 缓存
私有数据
```

## `minimal_method_package`

该 profile 面向论文方法最小代码附件。它只保留能够解释和复现核心方法的最小项目包, 不携带外层治理、Notebook workflow 或论文产物重建层。

### 默认包含

```text
main/core/
main/methods/
main/protocol/
configs/
README.md
pyproject.toml
```

### 默认排除

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
tests/integration/
tests/helpers/
outputs/
```

## 依赖方向要求

核心方法层必须能够独立于外层治理层被抽离。允许的主要依赖方向如下:

```text
main/protocol/ -> main/core/
main/methods/ -> main/core/
main/analysis/ -> main/core/, main/methods/, main/protocol/
main/cli/ -> main/core/, main/methods/, main/protocol/, main/analysis/
experiments/ -> main/
paper_experiments/ -> main/, experiments/
scripts/ -> main/
paper_workflow/ -> paper_experiments/, main/, experiments/
tools/harness/ -> 任意受治理路径
tests/ -> main/, tools/harness/
```

禁止的主要依赖方向如下:

```text
main/core/ -> main/analysis/, main/cli/, experiments/, paper_experiments/, external_baseline/, scripts/, tests/, tools/, paper_workflow/
main/methods/ -> main/analysis/, main/cli/, experiments/, paper_experiments/, external_baseline/, scripts/, tests/, tools/, paper_workflow/
main/protocol/ -> main/analysis/, main/cli/, experiments/, paper_experiments/, external_baseline/, scripts/, tests/, tools/, paper_workflow/
main/analysis/ -> experiments/, paper_experiments/, scripts/, tests/, tools/, paper_workflow/
main/cli/ -> experiments/, paper_experiments/, scripts/, tests/, tools/, paper_workflow/
experiments/ -> paper_experiments/, external_baseline/, paper_workflow/
paper_experiments/ -> paper_workflow/
```

这一规则属于方法可抽离性的核心约束。项目作者的特殊设计在于: 将治理层保留在开发仓库中, 但要求论文附件抽取时能够去除治理实现, 只保留读者复现所需的代码。
