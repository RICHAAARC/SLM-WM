# 项目构建规范索引

## 1. 目录职责

`docs/builds/` 只保存能够直接约束项目实现、实验协议和正式证据闭合的现行构建规范。目录中的文档必须满足以下条件：

1. 具有唯一且不可由其他文档替代的职责。
2. 明确指出机器可读事实来源、目标实现位置或正式消费位置。
3. 无状态规范只描述目标契约；当前实现状态只能写入本目录的项目构建状态规范，不得在其他构建规范中维护第二份状态。
4. 给出失败条件、实施顺序或验收入口，使维护者可以据此修改项目。
5. 不保存历史方法说明、运行操作手册、临时调查记录或论文正文草稿。

本目录的机器可读文件集合由 `scripts/build_specification_inventory.py` 中的 `BUILD_SPECIFICATION_PATHS` 唯一登记。约束测试直接比较登记集合与真实目录；新增、遗漏或额外 Markdown 文件都必须失败，不能只更新本索引而绕过结果包登记。

## 2. 必要文档清单

| 文档 | 唯一职责 | 机器事实或实现来源 | 何时使用 |
| --- | --- | --- | --- |
| `algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md` | 冻结目标方法的公式、输入输出、失败条件和主张边界 | 后续目标 `main/` 实现及方法身份登记 | 修改任何核心算法原语前首先阅读 |
| `method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md` | 将算法原语映射到目录、公开接口、运行数据流和验收边界 | 目标 `main/`、`experiments/` 接口与 schema | 设计或实现方法模块时使用 |
| `project_construction_state.md` | 记录当前仓库快照、GitNexus 影响面、保留/修改/移除清单和实施顺序 | 当前 Git 提交、GitNexus 索引及真实文件状态 | 判断当前实现差距和下一构建动作时使用 |
| `formal_dependency_environment.md` | 冻结正式 Python、CUDA、PyTorch 和完整哈希锁证据链 | `configs/dependency_profile_registry.json`、`experiments/runtime/dependency_profiles.py` | 修改依赖、宿主入口或 GPU 环境时使用 |
| `prompt_dataset_provenance.md` | 冻结 Prompt 来源、选择算法、嵌套集合和来源摘要 | `configs/prompt_source_registry.json`、`configs/prompt_selection_manifest.jsonl` | 重建或审计 Prompt bank 时使用 |
| `paper_profile_protocol_isomorphism.md` | 冻结三档允许变化的规模字段、必须同构的协议字段和 pilot 主投稿/full 扩展边界 | `configs/paper_profile_protocol_registry.json`、`paper_experiments/analysis/paper_profile_protocol_isomorphism.py` | 修改 `probe_paper`、`pilot_paper`、`full_paper` 任一协议时使用 |
| `paper_quality_claim_governance.md` | 冻结质量估计对象、统计单位、非劣效边界和逐攻击结论 | `configs/paper_quality_claim_protocol.json`、`paper_experiments/analysis/paper_quality_decisions.py` | 修改质量生产、聚合或决策时使用 |
| `paper_claim_decision_governance.md` | 冻结论文主张、证据完整性和科学支持之间的派生关系 | `configs/paper_claim_registry.json`、`paper_experiments/analysis/paper_claim_decisions.py` | 修改结论闭合和结果包门禁时最后使用 |

除本文件外，上述8份规范均不可删除或互相替代：两份无状态方法规范与一份有状态构建记录职责排他；三档规模、质量统计和最终主张决策分别约束不同机器对象，也不能合并为一份宽泛说明。

## 3. 明确排除的文档类型

以下文档不得位于本目录：

- 迁移前方法不变量记录；它只可为旧登记表提供可追踪锚点。
- GPU 会话持久化和恢复操作协议；它属于运行操作层。
- 与本目录项目构建状态规范并行的第二份项目状态文档；目录边界、字段总表和 Notebook 使用说明分别由对应外围层维护。
- 已删除的旧方法章节、技术路线、算子映射、一致性报告和核心构建指南。

被排除不表示对应信息无用，而是表示它不具有“现行项目构建规范”的权威角色。

## 4. 文档依赖关系

构建规范按以下方向消费，不允许反向覆盖：

```text
算法原语
  -> 方法机制设计
  -> 正式依赖环境
  -> Prompt 来源与三档规模协议
  -> 质量结论协议
  -> 论文主张决策
```

项目构建状态规范不进入上述无状态规范派生链。它只读取真实仓库状态并对照算法原语与方法机制设计登记差距，不能反向修改两份无状态规范。

具体约束如下：

1. 方法机制设计只能映射算法原语，不能增加第三载体、显著性模型、注意力分支或新主张。
2. 依赖环境只证明运行输入和环境身份，不证明方法或论文结论成立。
3. Prompt 来源文档定义样本总体；三档同构文档只能引用该总体，不能维护第二套 Prompt 数量。
4. 质量文档只定义质量子主张；最终是否支持 `quality_preservation` 由论文主张决策层派生。
5. 论文主张决策文档只能消费前层受治理事实，不能改写方法公式、实验结果或统计边界。

## 5. 可执行项目变更顺序

### 5.1 核心方法迁移

1. 以算法原语文档第4至9节作为唯一数学输入。
2. 按方法机制设计第2节重新执行 GitNexus upstream impact。
3. 按方法机制设计第3至5节建立目标 `main/` 模块和单 Prompt 数据流。
4. 按第8至10节原子迁移配置、方法身份、实验 runner、消融和 GPU 资格化。
5. 只有 CPU 性质测试和真实 GPU 单 Prompt 资格化都通过，才能把目标实现状态提升为 `gpu_operator_qualified`。

该步骤的实际完成状态只由项目构建状态规范登记；旧 `main/`、旧方法登记表和历史结果不能证明目标方法。

### 5.2 环境迁移

1. 从依赖 registry 和直接依赖输入重建完整哈希锁。
2. 在 Linux x86_64 目标 Python patch 中执行隔离安装、`pip check` 和环境 inspection。
3. 删除只服务旧 Jacobian、JVP/VJP、PSD-CG、三时刻写回和 Null Space 耦合的资格化前置条件。
4. 改为验证目标方法实际需要的 VAE 解码、S/T/R/Q 内容事实、二维 LF/HF-tail 载体、真实 Q/K 几何同步、有界恢复、单次写回和图像编码。

静态锁通过不能替代真实 CUDA 资格化。

### 5.3 实验协议迁移

1. 由 Prompt 来源清单逐字节重建70、700和7000条嵌套集合。
2. 由三档同构登记生成 `scale_contract`、`protocol_contract` 和 `artifact_contract`。
3. 拒绝方法摘要、检测器、攻击、baseline、消融、统计规则或产物 schema 的跨档差异。
4. `probe_paper` 流程闭合只证明相同流程可迁移，不外推 `pilot_paper` 或 `full_paper` 的科学效果。
5. `pilot_paper` 是主投稿证据；`full_paper` 是可选扩展，不进入 pilot 投稿就绪的 required-claim conjunction。

### 5.4 质量与论文结论闭合

1. 从真实持久化图像生产配对感知、独立视觉内容和分布质量原子。
2. 按 Prompt 聚类和5重复结构重建统计区间。
3. 逐攻击形成质量子主张，再派生跨攻击质量结论。
4. 将 fixed-FPR、baseline、质量和机制必要性四项结论传入统一主张决策器；参数敏感性仅由独立诊断消费者复验，不进入主张决策器或 release gate。
5. 只有 `evidence_complete=true` 且预登记判据满足时，主张才能为 `supported`。

## 6. 统一完成判据

根据本目录实施项目变更后，至少必须完成：

```powershell
pytest -q
python tools/harness/run_all_audits.py
git diff --check
```

此外还必须执行以下人工复验：

1. GitNexus `detect_changes` 未发现未登记的核心流程影响。
2. `docs/builds/` 的实际文件集合与 `BUILD_SPECIFICATION_PATHS` 完全相同。
3. 所有目标路径、配置和符号在对应阶段完成后真实存在，不以文档描述代替实现。
4. 所有历史兼容文档和历史结果仍明确标记为不能支持目标方法主张。
5. GPU 事实只由 clean detached 提交上的真实 CUDA 运行产生。
