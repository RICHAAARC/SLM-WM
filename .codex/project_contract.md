# Project Contract Template

## Long-Term Goal

本项目采用 governed research project 方法构建论文相关代码库: 先定义契约、目录边界、字段注册、测试分层、Notebook 边界和论文产物重建规则, 再实现论文方法与实验流程。

## Current Construction Unit

- `project_unit`: `core_method_runtime_construction`
- `target_construction_unit`: `experiment_protocol_validation`
- 当前推进单元允许实现真实方法算子、真实模型运行时接口、仅图像检测接口和对应轻量测试。
- 当前推进单元不得把 synthetic、proxy、counterfactual 或未完成的大规模运行结果登记为正式论文证据。
- 进入 `experiment_protocol_validation` 前, 真实分支风险场、Jacobian Null Space、尾部截断鲁棒载体、注意力几何和仅图像检测必须形成可执行闭环并通过 harness gates。
- `docs/builds/method_semantic_invariants.md` 是核心方法数学语义的权威来源；`main/methods/method_definition.py` 只能作为可执行镜像, 不得由当前实现反向改写方法定义。
- `configs/method_semantic_registry.json` 只登记公式、实现符号、CPU 性质和 GPU 原子证据之间的追踪关系, 不得自行保存 `ready`、`verified`、`pass` 或其他科学验证结论。
- 方法规范冻结只定义可证伪公式、唯一配置、失败条件、禁止替代项和验证职责, 不得据此声明实现已经通过。某个核心方法不变量提升到 `cpu_verified` 时, 必须在同一提交中完成真实实现、独立正反例性质测试、运行证据 schema 和当前方法文档对齐。全部核心不变量通过独立 CPU 语义审计前, 不得进入论文实验协议推进。
- 当前没有本地 CUDA 环境, 也没有可用的远程 Linux 服务器。真实 SD3.5 生成、精确 JVP、Q/K 梯度、再扩散攻击和正式 Inception 特征提取只能由 Colab GPU 入口执行; 本地环境只承担 CPU 测试、静态审计、记录物化和论文结果闭合。
- 在全部核心方法不变量通过独立 CPU 语义审计前, 项目保持在当前构建单元。进入后续单元仍不得把代码可执行性写成已经完成的论文实验结果；真实方法稳定性与论文结论分别需要后续 Colab GPU 科学预检和正式结果包证明。

## Ordered Semantic Construction Units

1. `project_bootstrap`
2. `core_method_runtime_construction`
3. `experiment_protocol_validation`
4. `paper_artifact_rebuild_gate`
5. `submission_readiness_gate`
6. `minimal_release_extraction`

## Core Directory Rules

1. `main/` 只保存 SLM-WM 论文核心方法及其最小数学工具, 不保存实验协议、结果分析、表格重建、CLI、Notebook、baseline 或产物治理能力。
2. `experiments/` 保存 SLM 项目自身的实验协议、运行时配置、真实攻击、正式消融、主方法 runner 和实验产物 schema, 不保存外部 baseline 适配工程。
3. `paper_experiments/` 保存完整论文实验产出能力, 包括外部 baseline 的受治理导入、方法忠实度验证、官方参考复现编排、论文级对比、证据审计和投稿就绪分析。
4. `scripts/` 保存可脱离 Notebook 运行的服务器 CLI、数据准备、结果检查、结果打包和 release 辅助命令。
5. `paper_workflow/` 保存最外层 Notebook / Colab workflow 入口和 session helper, Notebook 文件应位于 `paper_workflow/notebooks/`。
6. `tools/harness/` 保存外层治理审计, 不得被 `main/` 反向依赖。
7. `.codex/` 和 `docs/` 保存协作契约与人类可读治理规则。
8. `tests/` 按运行成本和验证目标分层。
9. `outputs/` 是统一的本地持久化输出根目录, 默认不提交。
10. harness 审计报告必须写入 `outputs/audit_reports/`, 不得写入仓库根目录下的 `audit_reports/`。

## Repository Layer Boundary Governance

本仓库采用由内向外的五层结构, 依赖方向必须保持为
`paper_workflow/ -> scripts/ -> paper_experiments/ -> experiments/ -> main/`。

1. 核心方法层: `main/` 只包含风险场、语义条件 Jacobian 低响应子空间、内容载体、注意力几何和仅图像检测等论文方法实现。该层只能依赖通用第三方库和自身模块。
2. 主方法实验层: `experiments/` 负责数据划分、fixed-FPR、模型运行、攻击、正式消融和实验产物。该层可以依赖 `main/`, 不得依赖 `paper_experiments/`、`scripts/`、`paper_workflow/` 或外部 baseline 工程。
3. 完整论文实验层: `paper_experiments/` 负责外部 baseline、公平对比、官方参考复现、受治理导入、论文证据审计和投稿就绪分析。该层可以依赖 `main/` 与 `experiments/`, 不得依赖 `scripts/` 或 `paper_workflow/`。
4. 独立执行层: `scripts/` 提供可在 GPU 服务器或 CPU 汇总服务器直接执行的 CLI。该层可以依赖前三层, 不得依赖 `paper_workflow/`。
5. Colab 运行层: `paper_workflow/` 只负责 Notebook 入口、Drive 同步、Colab session helper 和远程运行包装。正式实现必须能够脱离该层运行。
6. `external_baseline/` 只作为外部源码缓存、来源登记和经审计的适配实现目录, 不属于核心方法层, 也不进入最小方法发布包。
7. 最小方法发布包只包含 `main/` 和方法所需的最小配置; 完整论文实验发布包可以包含 `experiments/`、`paper_experiments/` 与 `scripts/`, 但必须排除 `paper_workflow/` 和未受治理的第三方源码缓存。

## Output File Governance

1. 由 repository command、script、experiment runner、artifact builder 或 harness 持久化写入的输出文件必须位于 `outputs/` 目录下。
2. harness 审计报告统一写入 `outputs/audit_reports/`。
3. 正式论文 records、tables、figures、reports、manifests 和 release candidate 若在本地生成, 必须使用 `outputs/` 下的语义子目录。
4. 测试中的一次性临时文件仍应使用 `tmp_path` 或 `tmp_path_factory`; 若测试需要保留可检查的持久化产物, 该产物也必须位于 `outputs/` 下并默认不提交。
5. 禁止在仓库根目录或源码目录中写入未受治理的运行输出、审计输出、实验输出或论文产物输出。

## Notebook Boundary Rules

1. Notebook 是论文实验的入口和远程执行包装, 不是正式协议逻辑的唯一实现。
2. Notebook 不得直接手写正式 records、thresholds、tables、figures 或 reports。
3. Notebook 应调用 `paper_workflow/` 薄包装, 薄包装再调用 `scripts/` 或更内层 repository modules; Notebook 不得直接维护第二套正式逻辑。
4. Notebook 文件统一放在 `paper_workflow/notebooks/`, 避免入口文件与 helper 模块混放。
5. Notebook 专用 helper 放在 `paper_workflow/notebook_utils/`。
6. 跨 Notebook 共享的 Colab 或 session helper 放在 `paper_workflow/colab_utils/`。

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
