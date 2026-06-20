# Project Contract Template

## Long-Term Goal

本项目采用 governed research project 方法构建论文相关代码库: 先定义契约、目录边界、字段注册、测试分层、Notebook 边界和论文产物重建规则, 再实现论文方法与实验流程。

## Current Construction Unit

- `project_unit`: `project_bootstrap`
- `target_construction_unit`: `core_method_runtime_construction`
- 当前推进单元只允许建立目录、文档、skill、harness、测试分层和最小 `main/` 核心包骨架。
- 当前推进单元不应引入真实大规模数据、正式实验输出、论文最终图表或发布包。

## Ordered Semantic Construction Units

1. `project_bootstrap`
2. `core_method_runtime_construction`
3. `experiment_protocol_validation`
4. `paper_artifact_rebuild_gate`
5. `submission_readiness_gate`
6. `minimal_release_extraction`

## Core Directory Rules

1. `main/` 保存论文方法、核心协议、核心评估、表格重建和 CLI 复现能力。
2. `experiments/` 保存递进式实验 runner、ablation、baseline 或 paper protocol。
3. `paper_workflow/` 保存 Notebook / Colab workflow 入口和 session helper。
4. `scripts/` 保存数据准备、结果检查、结果打包和 release 辅助命令。
5. `tools/harness/` 保存外层治理审计, 不得被 `main/` 反向依赖。
6. `.codex/` 和 `docs/` 保存协作契约与人类可读治理规则。
7. `tests/` 按运行成本和验证目标分层。
8. `outputs/` 是统一的本地持久化输出根目录, 默认不提交。
9. harness 审计报告必须写入 `outputs/audit_reports/`, 不得写入仓库根目录下的 `audit_reports/`。

## Output File Governance

1. 由 repository command、script、experiment runner、artifact builder 或 harness 持久化写入的输出文件必须位于 `outputs/` 目录下。
2. harness 审计报告统一写入 `outputs/audit_reports/`。
3. 正式论文 records、tables、figures、reports、manifests 和 release candidate 若在本地生成, 必须使用 `outputs/` 下的语义子目录。
4. 测试中的一次性临时文件仍应使用 `tmp_path` 或 `tmp_path_factory`; 若测试需要保留可检查的持久化产物, 该产物也必须位于 `outputs/` 下并默认不提交。
5. 禁止在仓库根目录或源码目录中写入未受治理的运行输出、审计输出、实验输出或论文产物输出。

## Notebook Boundary Rules

1. Notebook 是论文实验的入口和远程执行包装, 不是正式协议逻辑的唯一实现。
2. Notebook 不得直接手写正式 records、thresholds、tables、figures 或 reports。
3. Notebook 应调用 `main/`、`experiments/` 或 `scripts/` 中的 repository modules。
4. Notebook 专用 helper 放在 `paper_workflow/notebook_utils/`。
5. 跨 Notebook 共享的 Colab 或 session helper 放在 `paper_workflow/colab_utils/`。

## Paper Artifact Governance

1. records 是论文结果事实来源。
2. tables、figures 和 reports 必须可由 records 与 manifests 重建。
3. supported claims 必须绑定到 governed artifacts。
4. 手工拼接正式论文结果表、正式图数据或正式 claim audit 属于阻断违规。
5. manifests 必须记录输入、输出、配置摘要、代码版本和重建命令。

## Naming Governance

1. 正式文件名、目录名、模块名、配置键和字段名应使用 `snake_case`。
2. 正式名称必须表达职责、机制或业务含义。
3. 除 `docs/` 下的人类可读规划文件外, 正式路径、配置键、字段名、测试名、报告名和正文不得使用 `docs/naming_governance.md` 中登记的过程标记词。
4. 禁止用弱编号、弱版本后缀、`new`、`old`、`best`、`final` 等词作为正式语义。
5. 方法、实验、报告和配置应使用能表达机制、实验协议或论文职责的名称。

## Git Commit Governance

1. Git commit message 的自然语言说明必须使用中文。
2. commit subject 与 commit body 均应使用中文描述改动内容、原因和影响范围。
3. 代码标识符、路径、命令、模型名、配置键和错误码可以保留原始英文或符号形式。
4. 不得使用仅包含英文泛化词的提交信息, 例如 `fix`, `update`, `wip` 或 `misc`。

## Placeholder And Random Governance

1. Placeholder 字段必须以 `_placeholder` 结尾。
2. Random trace 字段必须以 `_random` 或 `_digest_random` 结尾。
3. Placeholder 字段不得支持 supported claims。
4. Governed fields 应先登记到 `docs/field_registry.md`。

## Validation Boundary Governance

1. 项目构建应避免在业务路径中大量重复防御式校验和错误信息构造。
2. 重复校验逻辑应优先收敛到配置加载时校验、dataclass 构造时校验、专门的 schema validator 或测试用例。
3. 业务函数内部只保留关键边界校验, 例如外部输入边界、不可恢复状态边界、跨模块协议边界或会导致静默错误的核心算法前置条件。
4. 相同字段、相同配置项、相同数据结构的错误信息不应在多个业务函数中分散维护; 应由统一解析器、schema 或测试断言提供稳定诊断。
5. 核心算法实现应优先呈现方法逻辑、数据流和统计边界, 不应被重复的防御式样板代码淹没。

## Test Governance

1. 默认 `pytest -q` 只运行 `unit`、`constraint` 或 `quick` 测试。
2. `tests/constraints/` 保存静态或轻量治理测试。
3. `tests/functional/` 保存轻量功能测试。
4. `tests/integration/` 保存集成、smoke、slow 或 formal 测试, 默认排除。
5. 测试输出必须使用 `tmp_path` 或 `tmp_path_factory`。

## Required Completion Commands

```bash
pytest -q
python tools/harness/run_all_audits.py
```
