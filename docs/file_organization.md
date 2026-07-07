# 文件组织契约

## 文档定位

本文档约束论文相关研究项目的目录边界。目标是将论文方法核心、完整论文实验、Notebook workflow、辅助脚本、治理审计和发布材料分离, 避免临时实验代码污染 `main/`。

## 推荐目录

```text
main/                   核心方法、协议、评估、分析、CLI 复现能力
configs/                实验配置、协议配置、数据 manifest 模板
experiments/            SLM 项目自身实验协议、运行配置、内部消融和主方法 runner
paper_experiments/      完整论文实验产出、外部 baseline 适配、受治理导入和公平对比
paper_workflow/         Notebook / Colab workflow 入口和 session helper
paper_workflow/notebooks/ Colab Notebook 入口文件
external_baseline/      外部 baseline 官方源码缓存与来源登记, 默认不提交外部源码
scripts/                数据准备、结果检查、结果打包、release 辅助命令
docs/                   方法说明、复现说明、治理契约、投稿材料说明
tools/harness/          可执行治理审计
tests/                  分层测试目录
.codex/                 Agent 协作契约与 skill 文件
outputs/                统一本地持久化输出根目录, 默认不提交
outputs/audit_reports/  harness 审计输出, 默认不提交
```

## `main/` 边界

`main/` 只保存论文方法和最终复现需要的核心能力, 包括:

```text
main/core/              records、manifest、schema、digest、registry
main/protocol/          split、threshold、runner 接口、输出布局
main/methods/           论文方法和机制变体
main/analysis/          表格、图数据、报告和 claim audit 构建
main/cli/               命令行复现实验入口
```

核心包冻结后的 `main/` 边界进一步要求:

1. `main/core/` 可以定义 SLM-WM 方法 typed object, 但不得写入运行输出或绑定外部执行环境。
2. `main/methods/` 与 `main/protocol/` 可以引用 `main/core/`, 但不得引用实验 runner、脚本、测试、harness 或 Notebook workflow。
3. `main/analysis/` 可以消费核心对象构造产物重建所需结构, 但不得依赖 `experiments/`、`paper_experiments/`、`external_baseline/`、`scripts/`、`tests/`、`tools/harness/` 或 `paper_workflow/`。
4. 核心包边界本地报告必须写入 `outputs/core_package_boundary_freeze/`, 不得写入仓库根目录或源码目录。

## 三层结构边界

本项目采用三层结构, 依赖方向必须为:

```text
paper_workflow/ -> paper_experiments/ -> main/ 与 experiments/
```

### 核心方法复现层

核心方法复现层包含 `main/`、`experiments/` 和必要的 `configs/`。该层负责 SLM-WM 主方法、主实验攻击、内部消融、阈值协议和服务器运行入口。该层不得依赖 `paper_workflow/`、`paper_experiments/` 或 `external_baseline/`。

### 完整论文实验层

完整论文实验层放在 `paper_experiments/`。该层负责外部 baseline 的 method-faithful 适配、官方参考复现编排、受治理结果导入、共同协议对比表和论文证据闭合。该层可以依赖核心方法复现层, 但不得依赖 Colab 运行层。

### Colab 运行层

Colab 运行层放在 `paper_workflow/`。Notebook 文件统一放在 `paper_workflow/notebooks/`, 共享 helper 放在 `paper_workflow/colab_utils/` 或 `paper_workflow/notebook_utils/`。该层只负责 Notebook 入口、Google Drive 落盘、Colab session helper 和远程运行包装。正式 records、tables、figures、reports 和 manifests 的生成逻辑不得只存在于该层。

## 禁止依赖方向

```text
main/ -> tools/harness/
main/ -> tests/
main/ -> experiments/
main/ -> paper_experiments/
main/ -> paper_workflow/
main/ -> outputs/
main/ -> scripts/
main/ -> external_baseline/
main/ -> google.colab
main/ -> pydrive
experiments/ -> paper_experiments/
experiments/ -> paper_workflow/
experiments/ -> external_baseline/
paper_experiments/ -> paper_workflow/
```

## 允许依赖方向

```text
tools/harness/ -> main/
tests/ -> main/
experiments/ -> main/
paper_experiments/ -> main/
paper_experiments/ -> experiments/
scripts/ -> main/
paper_workflow/ -> paper_experiments/
paper_workflow/ -> main/
```

## `external_baseline/` 边界

`external_baseline/` 仅用于保存外部 baseline 官方源码或复现镜像的本地副本。该目录不属于 `main/` 方法核心, 不直接写出论文 records、tables、figures、reports 或 manifests。

目录内第三方源码默认由本目录 `.gitignore` 排除, 仓库只保留来源登记文件和每个 baseline 的本地说明文件。若后续需要生成外部 baseline 指标, 必须通过 `paper_experiments/baselines/` adapter 或受治理导入记录进入 `outputs/`, 不得手工修改论文表格。

## Notebook 边界

Notebook 可以作为论文实验入口, 但不得成为唯一正式实现路径。正式 records、tables、figures、reports 和 manifests 的写入逻辑应位于 `main/`、`experiments/`、`paper_experiments/` 或 `scripts/`。

## 输出文件边界

所有由仓库命令、脚本、实验 runner、artifact builder 或 harness 持久化写入的输出文件都必须位于 `outputs/` 目录下。  
harness 审计报告统一写入 `outputs/audit_reports/`。  
源码目录、文档目录和仓库根目录不得写入未受治理的运行输出或论文产物输出。


