# SLM-WM 总阶段项目构建指引：Core-first 修订版 v3

## 一、文档定位

本文档是 SLM-WM 项目的总阶段构建指引。本文档已经按当前项目框架契约修正：正式核心包以 `main/` 为准，不再要求额外顶层 `slm_wm_core/` 或 `slm_wm_runtime/` 作为正式项目边界。若未来确需新增顶层核心包，必须先更新 `.codex/project_contract.md`、目录治理文档和 harness 规则。

本文档的执行原则为：

```text
core-first -> runtime-second -> workflow-third -> paper-artifact-final
```

其含义是：

1. 先在 `main/` 内冻结论文方法核心、协议对象和最小复现能力；
2. 再在 `experiments/` 与 `scripts/` 中接入 SD3 / SD3.5 运行适配、攻击、baseline 和 ablation runner；
3. 再在 `paper_workflow/` 中接入 Notebook / Colab workflow、session helper 和外部持久化 manifest；
4. 最后通过 `main/analysis/` 与 `scripts/` 从 governed records 自动重建论文表格、图、报告和 evidence audit。

本文档不覆盖项目契约。若本文档与 `.codex/project_contract.md` 冲突，以项目契约为准。

---

## 二、项目总目标

项目最终目标是实现“语义条件化的潜空间流形水印（SLM-WM）”的方法机制，并产出可用于论文投稿的全部表格、图、报告、证据审计和最小发布包。

方法主线为：

$$
\text{semantic risk field}
\rightarrow
\text{semantic-conditioned safe null-space}
\rightarrow
\text{latent LF/HF/attention carrier decomposition}
\rightarrow
\text{fixed-FPR robust detection}
\rightarrow
\text{same-threshold geometric rescue}
\rightarrow
\text{attested final attribution}.
$$

项目完成后应具备以下能力：

1. 在 diffusion latent trajectory 内部嵌入水印，而不是在生成后图像域叠加水印；
2. 根据语义显著性、纹理复杂度和轨迹稳定性自适应选择水印方向；
3. 从 semantic-conditioned safe null-space 导出 LF、HF 和 Self-Attention geometry 三类证据方向；
4. 使用固定 calibration 协议控制 raw content decision 与 rescue 后 evidence-level decision 的整体误报；
5. 在几何恢复后复用同一内容阈值重判，禁止几何链直接判 positive；
6. 通过 governed records、tables、figures、reports 和 manifests 支撑论文 claims；
7. 导出与当前 release profile 对齐的 `minimal_method_package` 和 `paper_artifact_rebuild_package`。

---

## 三、项目框架映射

当前项目契约规定的目录职责如下。

| 目录 | 构建职责 |
|---|---|
| `main/core/` | records、manifest、digest、schema、typed objects 等通用核心结构 |
| `main/methods/` | SLM-WM 方法机制，包括 semantic route、subspace、carrier、geometry、attestation 等 |
| `main/protocol/` | split、threshold、fixed-FPR decision、runner 协议和输出布局 |
| `main/analysis/` | 表格、图数据、报告和 claim audit 构建逻辑 |
| `main/cli/` | 可选 CLI 复现入口 |
| `experiments/` | 阶段性 runner、SD runtime adapter、attack、baseline、ablation 和 paper protocol |
| `paper_workflow/` | Notebook / Colab workflow 入口、Drive/session helper 和工作流包装 |
| `scripts/` | 数据准备、结果检查、阶段运行、结果打包和 release 辅助命令 |
| `tests/constraints/` | 静态或轻量治理测试，默认执行 |
| `tests/functional/` | 轻量功能测试，默认执行 |
| `tests/integration/` | SD、Drive、slow、formal 或端到端测试，默认排除 |
| `tools/harness/` | 外层治理审计，不得被 `main/` 反向依赖 |

禁止把正式方法核心放入未登记的顶层核心包，也禁止让 `main/` 反向依赖 `experiments/`、`scripts/`、`paper_workflow/`、`tests/` 或 `tools/harness/`。

---

## 四、执行路线分级

### （一）最小可发表闭环

最小可发表闭环用于资源有限或高风险 attention update 不稳定的场景，至少包括：

1. `main/` 中的 SLM-WM 方法核心；
2. SD3 / SD3.5 或可审计 runtime fallback 的 latent injection；
3. semantic mask 到 latent mask 的真实接入；
4. semantic-conditioned safe null-space 或明确标注的可审计近似；
5. LF / HF 内容载体与统一内容分数；
6. fixed-FPR calibration；
7. 常规攻击矩阵；
8. 主表 external baseline；
9. 内部机制消融；
10. paper artifacts 与 evidence audit。

若 attention-relative latent update 不稳定，最小闭环允许将其降级为 attention graph geometry evidence 与 no-direct-positive rescue analysis，但不得声称 Self-Attention watermark 是 Full 方法主载体。

### （二）高上限完整闭环

高上限完整闭环在最小闭环基础上增加：

1. 稳定 attention-relative latent update；
2. attention graph 作为主几何同步约束；
3. 再扩散攻击闭环；
4. full-main 常规攻击全量运行；
5. full-extra 再扩散代表子集或高成本补充实验；
6. 更完整的 case study、score retention、aligned gain 和 failure analysis。

---

## 五、阶段与项目契约的关系

本文档中的 stage00 至 stage17 是 SLM-WM 构建工作包，不等同于 `.codex/project_contract.md` 中的 `project_stage`。当前项目契约的语义阶段仍为：

```text
project_bootstrap
core_method_runtime_construction
experiment_protocol_validation
paper_artifact_rebuild_gate
submission_readiness_gate
minimal_release_extraction
```

stage00 至 stage17 的推进不得绕过这些语义阶段，也不得在 `project_bootstrap` 阶段引入真实大规模数据、正式实验输出、论文最终图表或发布包。

---

## 六、阶段阻断条件总表

| 阶段 | 不得进入下一阶段的条件 |
|---|---|
| stage00 | `main/` 仍反向依赖 Drive、Colab、paper workflow、experiments、scripts、tests 或 harness |
| stage01 | 方法 typed objects 不稳定，或 `main/` 直接写正式 records / manifests |
| stage02 | synthetic smoke 无法证明 key 区分、rescue 边界和 attestation 分层 |
| stage03 | SD runtime adapter 污染 `main/`，或无法记录 unsupported reason |
| stage04 | latent injection 导致图像崩坏，或强度变化无可解释趋势 |
| stage05 | Notebook / Colab 重启后无法只凭 manifest 重载前序产物 |
| stage06 | calibration 与 test split 不独立，或 event / prompt 标识不稳定 |
| stage07 | semantic mask 未真实影响 latent feature operator 或 basis |
| stage08 | LF / HF 使用独立阈值投票作为主判 |
| stage09 | attention graph 无法稳定提取，且 unsupported reason 未记录 |
| stage10 | attention-relative update 不稳定但仍被写入 Full 主方法 |
| stage11 | 几何链直接判 positive，或恢复后未使用同一内容阈值 |
| stage12 | test split 被用于调阈值，或 rescue 后整体 FPR 未审计 |
| stage13 | 攻击结果未写 attack digest，或再扩散不稳定却作为核心主张 |
| stage14 | baseline 使用不同 prompt set / attack matrix，或手工填写结果 |
| stage15 | 消融开关未真实改变对应机制 |
| stage16 | artifact builder 不能由 records 和 manifests 重建表格、图和报告 |
| stage17 | full-main 调参、final evidence audit 未通过，或 release profiles 无法验证 |

---

## 七、阶段总览

### stage00：核心包边界冻结

目标是在 `main/` 内冻结核心包边界，确认 `main/core/`、`main/methods/`、`main/protocol/`、`main/analysis/` 和 `main/cli/` 的职责划分。边界测试应进入 `tests/constraints/`，不得新增重型默认测试。

输出以本地阶段 manifest 和边界报告为主，不产生正式论文结果。

### stage01：纯算法原语实现

目标是在 `main/methods/` 和 `main/protocol/` 中实现 synthetic / tensor 级别的 SLM-WM 原语，包括 semantic risk、safe basis、LF / HF carrier、attention stub、content score、geometry reliability、evidence / final decision。

attention 在本阶段只允许 synthetic stub，不接入真实 SD attention。

### stage02：核心方法最小闭环 smoke test

目标是用 synthetic latent tensor 验证核心链路，覆盖 clean、watermarked、wrong-key、geometric shifted、aligned recovered、unattested positive 和 final positive 等场景。

smoke records 必须由 `scripts/` 或测试 harness 根据 `main/` 返回对象生成，不能由 `main/` 直接写正式 records。

### stage03：SD3 / SD3.5 运行适配层

目标是在 `experiments/` 或 `scripts/` 中建立 SD3 / SD3.5 适配、采样回调、latent trace、VAE encode / decode 和 attention capture。该层可以依赖 `diffusers`、`transformers`、`accelerate`、`safetensors` 等运行依赖，但不得让 `main/` 反向依赖这些适配对象。

若模型权重、显存或 API 不可用，必须记录 unsupported reason，并使用 toy / synthetic adapter 完成工程测试。fallback 结果不得支持论文主张。

### stage04：最小 diffusion latent injection

目标是在真实或 fallback diffusion sampling trajectory 中验证 latent update 可注入、强度变化可解释、图像不崩坏。输出仍为本地阶段目录和 manifest，不进入正式论文结果。

### stage05：Colab / Drive 运行层

目标是在 `paper_workflow/` 中建立 Notebook / Colab workflow 入口、外部持久化路径、manifest reload 和冷启动验证。Notebook 只调用 `main/`、`experiments/` 或 `scripts/`，不得直接实现算法逻辑、阈值计算、正式 records、tables、figures 或 reports。

### stage06：Prompt、split、records 与实验协议

目标是在 `experiments/protocol/` 与 `main/protocol/` 中建立 prompt、split、event、sample role、records schema 和 calibration protocol。calibration 与 test 必须独立。

### stage07：semantic mask、risk field 与安全子空间正式实现

目标是让 semantic mask 真实进入 latent trajectory feature operator，并让 semantic route 改变 safe basis。若使用全 mask，只能作为 fallback 或消融。

### stage08：LF / HF 内容载体与内容检测统计

目标是实现 LF 主证据、HF 鲁棒补充和统一内容分数。LF / HF 不得分别设置独立主判阈值后投票。

### stage09：Self-Attention graph extraction 与几何证据

目标是捕获或复用 SD3 / SD3.5 attention 信息，构造 attention-relative graph 和 geometry evidence。该阶段只记录几何可靠性统计，不直接判 positive。

### stage10：Attention-relative latent update

目标是在 stage09 稳定后实现 attention-relative latent update。若 update 不稳定，必须降级为 evidence-only 或诊断机制，不能污染 Full 方法主张。

### stage11：同阈值几何救回集成

目标是将几何恢复接入 same-threshold rescue。几何链只恢复参考系；恢复后内容链必须复用 stage12 冻结的同一内容阈值。

### stage12：阈值校准与常规图像水印指标体系

目标是冻结 fixed-FPR calibration，并建立常规水印指标、质量指标、ROC / DET、score distribution 和 threshold degeneracy report。该阶段必须同时审计 raw content decision 与 rescue 后 evidence-level decision 的整体 FPR。

### stage13：攻击矩阵与再扩散攻击闭环

目标是建立常规攻击与再扩散攻击矩阵。再扩散攻击若不稳定，只能进入补充实验或局限性讨论。

### stage14：外部 baseline 对比

目标是在相同 prompt、split、attack matrix、clean negative 和 fixed-FPR 或可比 operating point 下比较 Tree-Ring、Gaussian Shading、Shallow Diffuse、T2SMark 等 baseline。无法运行的 baseline 必须记录 unsupported reason。

### stage15：内部消融与反工程组合证明

目标是通过机制消融证明 SLM-WM 不是工程组件叠加。每个消融必须真实改变对应机制，而不是只改 method name。

### stage16：论文图表、报告与 evidence audit 构建层

stage16 的职责是实现并验证 artifact rebuild 层。该阶段应完成表格、图、报告和 evidence audit 的 builder、schema、provenance 与 dry-run 验证。若已有 governed records，可生成预览或阶段性产物；但 stage16 不负责冻结 full-main 最终论文结果。

stage16 的产物应说明：给定 governed records 与 manifests，是否可以自动重建论文 tables、figures、reports 和 claim evidence audit。

### stage17：Pilot、Full 与提交前冻结

stage17 的职责是完成 probe、pilot、full-main 与必要 full-extra，冻结配置和最终 records，然后调用 stage16 已验证的 artifact builders 重新生成最终论文表格、图、报告和 evidence audit。最终发布包在本阶段导出和验证。

---

## 八、fixed-FPR 与 rescue 统计边界

正式 fixed-FPR 口径必须作用于完整 evidence-level 判定协议，而不是只固定内容阈值。

1. 内容阈值由 calibration split 的 clean negative 分布确定。
2. geometry reliability thresholds、rescue window 和 fail-reason gate 也必须在 calibration 或预注册协议中冻结。
3. test split 只用于报告，不得调阈值、调 rescue window 或调 fail-reason 规则。
4. rescue 仅允许作用于边界失败样本，不能对远离阈值的 negative 样本开放。
5. recovery 后必须复用同一内容阈值，不得为 aligned score 单独设置新阈值。
6. 报告 fixed-FPR 时必须同时给出 raw content FPR、rescue 后 clean negative FPR 和 rescue 后 attacked negative FPR。
7. 若 rescue 后整体 FPR 超过目标 operating point，不得声称完整系统仍满足该 fixed-FPR 目标，除非重新在 calibration split 中冻结包含 rescue 的完整决策协议。

---

## 九、论文产物与发布包要求

正式论文产物必须满足：

1. records 是事实来源；
2. tables、figures 和 reports 可由 records 与 manifests 重建；
3. supported claims 绑定到 governed artifacts；
4. placeholder 字段不得支撑 supported claims；
5. manifests 记录输入、输出、配置摘要、代码版本和重建命令。

发布包名称与当前项目 release profile 保持一致：

| profile | 作用 | 默认边界 |
|---|---|---|
| `minimal_method_package` | 最小论文方法代码附件 | 包含 `main/core/`、`main/methods/`、`main/protocol/` 和最小配置 |
| `paper_artifact_rebuild_package` | 论文图表和报告重建附件 | 包含 artifact builders、必要 scripts、configs、experiments protocol 和轻量功能测试 |

导出应优先复用当前脚本能力，例如 `scripts/extract_minimal_paper_package.py`。如需新增包装脚本，仍必须遵守 release boundary 文档和 harness 审计。

---

## 十、当前最优执行顺序

当前仓库仍处于 `project_bootstrap` 语义阶段。最优执行顺序是：

1. 先保持 `main/` 边界、字段登记、测试分层和 harness 规则稳定；
2. 再在 `main/methods/` 与 `main/protocol/` 中实现 synthetic / tensor 级方法核心；
3. 再通过 `tests/functional/` 与 `scripts/` 建立最小 smoke；
4. 再接入 `experiments/` 中的 SD runtime、attack、baseline 和 ablation runner；
5. 再接入 `paper_workflow/` 中的 Notebook / Colab workflow；
6. 最后进入 artifact rebuild、full run、evidence audit 和 release extraction。

每个阶段完成后均需运行：

```bash
pytest -q
python tools/harness/run_all_audits.py
```

若某阶段包含 integration、smoke、slow 或 formal 测试，应额外显式运行对应命令，但不得把重型测试加入默认 `pytest -q` 路径。
