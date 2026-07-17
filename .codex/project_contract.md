# Project Contract Template

## Long-Term Goal

本项目采用 governed research project 方法构建论文相关代码库: 先定义契约、目录边界、字段注册、测试分层、Notebook 边界和论文产物重建规则, 再实现论文方法与实验流程。

## Current Construction Unit

- `project_unit`: `document_ecosystem_synchronization`
- `target_construction_unit`: `core_method_runtime_construction`
- `project_unit` 只表示当前治理动作；`target_construction_unit` 表示文档定稿后的下一代码构建目标，二者都不表示完成状态。完成状态只由项目构建状态规范登记。
- 当前唯一算法原语权威来源是 `docs/builds/algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md`。
- 当前唯一方法机制与接口设计来源是 `docs/builds/method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md`。
- 当前唯一项目状态、保留/修改/移除清单和实施顺序来源是 `docs/builds/project_construction_state.md`。
- 三份核心文档是当前权威设计来源；当前5重复、主投稿 profile 与等价执行复用修订尚待机器协议同步和复核。外围契约、README、构建清单和发布文档只能单向消费三份核心文档，不得复制第二套公式、接口或迁移状态。
- 正式方法固定为“语义显著性自适应内容-几何双链潜空间水印”。具体数学语义、禁止主张和参数边界只以算法原语文档为准，本契约不重复定义。
- 当前实现与目标方法的差距只以 `docs/builds/project_construction_state.md` 为准。迁移前代码和兼容记录不得覆盖三份核心文档，也不得支持目标方法主张。
- 在核心实现、配置、登记表、CPU 性质测试和 GPU 资格化全部迁移前，正式论文结果生产处于阻断状态。历史 `c6139ced` 结果及其他旧提交结果只能作为历史工程证据。
- 当前治理单元只同步文档、机器可读目标契约及其约束测试，不修改核心方法实现；下一构建单元才允许按有状态清单修改核心方法及其直接配置、测试和 runtime 接线。任何单元都不得扩充攻击集合或 baseline，也不得用 synthetic、proxy、placeholder 或兼容性回退替代真实方法。
- 方法规范冻结只定义可证伪公式、唯一配置、失败条件、禁止替代项和验证职责，不表示实现已经通过。实现状态必须由独立测试和真实 GPU 记录决定。
- 单个 seed-key 重复只允许形成 `supports_paper_claim=false` 的证据组件。任何正式论文结论仍必须通过版本化精确5重复聚合验证。
- `pilot_paper` 是主投稿证据 profile；`full_paper` 是更严格 FPR 与更大样本规模的可选扩展。未运行或未闭合 `full_paper` 不阻断基于完整 `pilot_paper` 证据形成的投稿就绪状态，也不得被解释为 `full_paper` 科学结论已经成立。
- 三档 required claims 只消费预登记的7项核心证据攻击；其余10项攻击属于补充描述性证据，不进入 fixed-FPR、baseline superiority、quality preservation 或 mechanism necessity 的 required conjunction。核心攻击在三档、6个方法角色和4个主表 baseline 间保持完全同构；补充攻击缺失、失败或未运行不得被解释为已验证，也不得阻断完整核心证据闭合。
- clean 图像、角色无关模型原子、公开质量特征、普通攻击结果、profile 不变原子和生成共享前缀只能在身份摘要完全匹配且等价性验证通过时复用。密钥依赖模板、角色阈值、最终决策、逐 profile 校准和统计结论不得跨边界复用。

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

1. 核心方法层: `main/` 目标上只包含真实内容观测、S/T/R/Q 内容路由、二维 LF 与 HF-tail 载体、真实 Q/K 几何同步与有界恢复、三分支单次写回、仅图像检测和同阈值救回。迁移前的 Jacobian、JVP/VJP、PSD-CG 与旧多注入耦合不得进入目标最小发布包；可复用的真实 Q/K 关系和恢复算子必须保留。该层只能依赖通用第三方库和自身模块。
2. 主方法实验层: `experiments/` 负责数据划分、fixed-FPR、模型运行、攻击、正式消融和实验产物。该层可以依赖 `main/`, 不得依赖 `paper_experiments/`、`scripts/`、`paper_workflow/` 或外部 baseline 工程。
3. 完整论文实验层: `paper_experiments/` 负责外部 baseline、公平对比、官方参考复现、受治理导入、论文证据审计和投稿就绪分析。该层可以依赖 `main/` 与 `experiments/`, 不得依赖 `scripts/` 或 `paper_workflow/`。
4. 独立执行层: `scripts/` 提供可在 GPU 服务器或 CPU 汇总服务器直接执行的 CLI。该层可以依赖前三层, 不得依赖 `paper_workflow/`。
5. Colab 运行层: `paper_workflow/` 只负责 Notebook 入口、Drive 同步、Colab session helper 和远程运行包装。正式实现必须能够脱离该层运行。
6. `external_baseline/` 只作为外部源码缓存、来源登记和经审计的适配实现目录, 不属于核心方法层, 也不进入最小方法发布包。
7. 最小方法发布包只包含 `main/` 和方法所需的最小配置，其中内容链与几何链均为必需组成；完整论文实验发布包可以包含 `experiments/`、`paper_experiments/` 与 `scripts/`, 但必须排除 `paper_workflow/` 和未受治理的第三方源码缓存。

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
