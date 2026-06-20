# governed_project_framework

`governed_project_framework` 是一个面向论文相关研究项目的工程治理模板。它的目标不是提供某个具体算法, 而是提供一套可复制的项目构建方法, 用于约束论文方法实现、实验流程、Notebook workflow、表格图表重建、claim audit 和最小发布边界。

## 核心定位

本框架适用于论文项目, 尤其适合需要长期迭代、递进式实验、Notebook 远程运行、正式表格重建和审稿复现材料整理的研究代码库。

框架固定使用 `main/` 作为核心 Python 包目录。`main/` 保存论文方法、实验协议、核心评估、表格重建和 CLI 复现入口。

## 核心包边界

`core_package_boundary_freeze` 冻结 `main/` 的最小方法包边界。该边界的含义是:

1. `main/core/` 只保存 digest、records、manifest 和 SLM-WM 方法 typed object。
2. `main/methods/` 只保存论文方法机制与机制变体。
3. `main/protocol/` 只保存 split、threshold、runner 接口和协议对象。
4. `main/analysis/` 可以构造表格、图数据、报告和 claim audit 所需的中间结构, 但不得反向依赖实验 runner、Notebook workflow 或 harness。
5. `main/cli/` 只作为命令行复现入口, 不保存外部环境专用逻辑。

本地边界报告由 `scripts/write_core_package_boundary_outputs.py` 写入 `outputs/core_package_boundary_freeze/`。这些文件是治理输出, 不是正式论文实验结果, 默认不提交。

## 五层结构

1. 契约层: `.codex/project_contract.md` 与 `docs/` 定义推进单元、目录边界、字段、命名、测试和发布规则。
2. Skill 层: `.codex/skills/*.skill.md` 约束 Agent 或协作者在不同任务中的允许行为与禁止行为。
3. Harness 层: `tools/harness/` 将文档约束转化为可执行审计。
4. 测试层: `tests/constraints/`、`tests/functional/`、`tests/integration/` 控制默认测试成本。
5. 论文产物治理层: records、tables、figures、reports、manifests 和 claims 之间保持可追溯关系。

## 推荐目录

```text
main/                   论文方法、协议、分析、CLI 和核心复现能力
configs/                实验配置模板
experiments/            递进式实验 runner 和 paper protocol
paper_workflow/         Notebook / Colab workflow 入口和 session helper
scripts/                数据准备、结果检查、打包和发布辅助命令
docs/                   工程治理、实验协议和复现说明
tools/harness/          可执行治理审计
tests/                  分层测试目录
.codex/                 Agent 协作契约与 skill 文件
outputs/                统一本地持久化输出根目录, 默认不提交
outputs/audit_reports/  harness 审计输出, 默认不提交
```

## 必需检查

```bash
python tools/harness/inspect_repository.py
pytest -q
python tools/harness/run_all_audits.py
```

## 复制到新论文项目

1. 将本目录内容复制到新仓库根目录。
2. 修改 `.codex/project_contract.md` 中的论文目标、推进单元名称、方法对象和通过条件。
3. 在 `main/` 中实现论文方法和核心协议, 不要把正式逻辑只写在 Notebook 中。
4. 在 `experiments/` 中放置递进式实验 runner。
5. 在 `paper_workflow/` 中放置 Notebook workflow, 但 Notebook 只负责调度。
6. 在 `docs/field_registry.md` 中登记 governed fields。
7. 保持 `pytest -q` 默认只运行 `unit`、`constraint` 或 `quick` 测试。

## 发布建议

该框架可作为 GitHub 模板仓库发布。使用者应先替换论文主题、方法名称、推进单元名称和 field registry, 再接入 CI。

## 方法抽离与论文附件

模板支持将完整开发仓库抽离为不同附件 profile:

1. `development_repository`: 保留治理、harness、Notebook workflow 和全部测试。
2. `paper_artifact_rebuild_package`: 保留 records 到 tables、figures、reports、manifests 的重建能力。
3. `minimal_method_package`: 只保留 `main/core/`、`main/methods/`、`main/protocol/` 和最小配置。

抽离规则见 `docs/extraction_profiles.md`。核心原则是: 核心方法层不得依赖 `.codex/`、`tools/harness/`、`tests/`、`experiments/`、`scripts/`、`paper_workflow/` 或本地输出目录。

中间状态字段、临时字段和缓存字段的规则见 `docs/intermediate_state_governance.md`。跨边界保存的中间状态必须显式命名并登记到 `docs/field_registry.md`。
