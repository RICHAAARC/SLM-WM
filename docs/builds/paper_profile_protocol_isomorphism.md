# 论文运行规模协议同构治理

## 一、职责与结论边界

本文档说明 `probe_paper`、`pilot_paper` 与 `full_paper` 如何形成可机器复验的协议同构结论。该门禁比较代码与实验协议, 不比较效果数值, 也不把 probe 的科学结论外推到更严格 fixed-FPR 工作点。

三个运行规模分别使用70、700和7000个 Prompt, 注册 FPR 分别为0.1、0.01和0.001。`probe_paper` 是流程与初步可行性 profile，`pilot_paper` 是主投稿证据 profile，`full_paper` 是可选扩展 profile。它们只允许改变登记的 Prompt / 样本规模、目标 FPR、统计强度和由规模派生的记录数量；核心方法、7项核心证据攻击、baseline、数据划分原则、阈值校准原则、指标语义、随机化、命令依赖图、产物 schema、gate 角色和主张决策结构必须完全一致。10项补充攻击的配置身份和描述性边界也必须一致，但其是否实际执行不属于 profile 核心闭合条件。

## 二、唯一登记与实际来源

`configs/paper_profile_protocol_registry.json` 只冻结不能从现有 Python 注册源可靠重建的外层契约：

1. 三个 profile 的规范顺序；
2. 允许变化字段路径；
3. CPU 结果重建命令依赖边；
4. 四类正式随机化统计 writer 与一类非主张参数诊断 writer 的产物文件契约；
5. 四项正式论文主张与 gate 角色的一一对应。参数敏感性仅为诊断产物，不进入该映射。

其余身份从实际项目实现读取：

- 方法配置来自 `configs/model_sd35.yaml` 及其完整配置摘要；
- 攻击来自 `experiments.protocol.attacks.default_attack_configs`，并由机器 registry 显式登记与 `resource_profile` 正交的 `attack_evidence_role`；
- baseline 来自4个主表 baseline 的实际共同协议定义；
- split 来自固定70条块内的 3:33:34 风险分层规则；
- 阈值来自 calibration 中 clean negative、wrong-key negative 与登记 attacked negative 共同构成的完整决策器冻结协议；
- 检测、配对优势和质量指标语义来自实际统计模块与冻结质量协议；
- 随机化来自权威注册表预先冻结的5个互异 seed-key 有序配对；该集合不是 seed 与 key 的完整笛卡尔积；
- 主张结构来自 `configs/paper_claim_registry.json`。

这一实现属于通用工程写法：不能只比较 profile 名称或少量摘要, 而应把实际配置正文、执行依赖和结论规则共同纳入规范记录。SLM-WM 项目特定部分是上述 fixed-FPR、固定5重复、7项核心证据攻击、10项补充描述性攻击、4个主表 baseline 和4项正式主张的具体绑定。

### 2.1 冻结攻击证据集合

7项核心证据攻击按以下规范顺序冻结，并在三档、6个方法角色、4个主表 baseline、正样本和受攻击负样本之间形成完整共同矩阵：

1. `jpeg_compression_main`
2. `gaussian_noise_main`
3. `resize_main`
4. `crop_resize_main`
5. `rotation_main`
6. `img2img_regeneration_extra`
7. `diffusion_purification_extra`

10项补充描述性攻击按以下规范顺序冻结：

1. `gaussian_blur_main`
2. `crop_main`
3. `composite_geometric_main`
4. `photometric_distortion_main`
5. `flow_matching_inversion_regeneration_extra`
6. `sdedit_regeneration_extra`
7. `global_editing_extra`
8. `local_editing_extra`
9. `visual_paraphrase_extra`
10. `adversarial_removal_extra`

`jpeg_compression_probe` 只承担非主张的单攻击工程 smoke，不属于上述两个论文证据集合。核心集合逐项登记 `attack_evidence_role=core_claim_required`；补充集合逐项登记 `attack_evidence_role=supplementary_descriptive`。已有 ID 中的 `_main/_extra` 和 `resource_profile=full_main/full_extra` 只描述历史命名与执行资源，不得据此推断证据职责。

核心集合是四项 required claims 的唯一攻击总体。补充集合不进入 calibration、核心跨攻击聚合、质量合取、baseline superiority、mechanism necessity 或投稿就绪；若运行，只允许完整方法在冻结核心决策器下产生逐攻击描述性检测、误报和质量结果。补充结果不得选择最强 baseline 后追加自适应比较，也不得由部分成功攻击外推完整集合鲁棒性。

## 三、规范化记录

每个 profile 记录由三部分组成：

```text
profile_id
scale_contract
protocol_contract
artifact_contract
```

`scale_contract` 只保存允许变化的科学规模字段及其派生内容，包括 FPR、Prompt 文件与数量、各固定 split 的派生样本计数、三类样本角色的派生数量、质量图像数量和记录数量派生关系。Drive 结果根可以按 profile 派生，但它只是操作存储位置，不是科学协议变化字段；其派生规则必须同构。固定5重复不能在 profile 间变化。`protocol_contract` 保存所有必须一致的实验语义，包括三类样本角色、Q/K 关系公式、几何捕获域、7项核心攻击、10项补充攻击、攻击证据职责、生成式攻击对称职责和质量指标。`artifact_contract` 在三档比较视图中连接正式 claim 产物与非主张诊断产物的 writer、ready 字段和文件集合。机器登记以 `artifact_contract` 保存仅与四项 claim 一一对应的正式产物，以 `diagnostic_artifact_contract` 保存参数敏感性诊断产物；只有前者允许出现在 `gate_roles`。补充攻击报告使用独立非主张产物角色，不得进入 `gate_roles`。

比较时不对任意 JSON 路径做宽松删除。实现只比较三个 profile 的完整 `protocol_contract` 与完整 `artifact_contract`, 因而未登记字段不能借“规模变化”名义被忽略。所有差异以结构化路径写入报告。

## 四、流程迁移与科学结论

流程迁移只按以下关系派生：

```text
workflow_transfer_ready = (
    probe_workflow_closed
    and protocol_isomorphism_ready
    and artifact_contract_isomorphic
)
```

`probe_workflow_closed` 要求输入报告属于 `probe_paper`、FPR 精确为0.1、闭合检查非空、阻断数为0, 并且 `evidence_closure_allowed` 与 `result_closure_ready` 均为 true。分主张科学决策仍由统一 validator 重算, 但不参与上述流程公式。

因此有两种必须区分的情况：

1. probe 结果为 `supported`：可以说明 FPR=0.1 下登记主张成立, 并在同构门禁通过时说明相同流程可迁移；不能说明 FPR=0.01或0.001下主张成立。
2. probe 结果为 `measured_not_supported`：说明 FPR=0.1 下完整测量没有支持主张；若流程与产物仍完整且协议同构, `workflow_transfer_ready` 仍可为 true。

没有 pilot 或 full 自身正式结果时, 报告将对应科学状态写成 `evidence_incomplete`、`scientific_support=null` 和 `scientific_support_transferred_from_probe=false`。其中 pilot 缺失会阻断主投稿证据闭合；full 缺失只表示可选扩展尚未建立，不阻断完整 pilot 作用域内的投稿就绪和主张。这避免把工程可运行性误报为论文效果证据，也避免把扩展实验误设为主投稿的硬前置。

## 五、跨 profile 等价产物复用

嵌套 Prompt 身份相同且完整 provenance 匹配时，三档可以复用 profile-invariant 的生成图像、核心或补充普通攻击结果、公开 VAE/Q/K 原子和质量特征。复用记录必须绑定 source artifact digest、生产代码、模型 revision、预处理、dtype、依赖和缓存 schema，并逐成员验证。

以下内容始终是 profile-specific，禁止从 probe 或 pilot 直接复制到更严格 profile：calibration population、经验预算、置信边界、`content_threshold`、`rescue_margin_low`、最终布尔决策、统计区间和主张决策。复用只减少重复测量，不改变每档独立证据责任。

## 六、独立写出入口

在与 probe 结果相同的 clean detached commit 中运行：

```bash
python -m scripts.write_paper_profile_protocol_isomorphism_report \
  --probe-result-closure-report-path outputs/paper_result_closure/probe_paper/result_closure_gate_report.json
```

writer 只在 `outputs/paper_profile_protocol_isomorphism/` 写出报告和 `manifest.local.json`。当前 Git 身份必须与 probe 闭合报告中的 `common_code_version` 精确相同；目录已存在时拒绝混入旧产物。该命令不运行 GPU 实验, 不修改结果, 也不进入 `main/`。

## 七、方法迁移时的更新顺序与协议变更原子性

方法、角色或证据 schema 发生受治理变更时，三档协议必须在同一变更单元中完成：

1. 更新 `configs/model_sd35.yaml` 的方法配置和摘要来源。
2. 更新 `configs/paper_profile_protocol_registry.json` 的统一方法、消融和产物契约；不得把方法差异登记为允许规模差异。
3. 更新 `paper_experiments/analysis/paper_profile_protocol_isomorphism.py` 从实际注册源读取目标方法身份和正式角色登记；角色集合必须从唯一登记派生，不得在多个消费者中重复硬编码。
4. 对三个 profile 分别执行 dry-run，要求缺失路径为空且规范化 `protocol_contract`、`artifact_contract` 完全一致。
5. 使用属于同一新提交的真实 `probe_paper` 闭合报告重建同构报告。

攻击协议迁移还必须原子更新攻击 registry、核心/补充集合摘要、预期记录计数、baseline 共同模板、质量 gate、结果闭合和补充报告消费者。只更新人类可读列表而保留17项攻击全部进入 required gate，必须失败关闭。

验收必须同时满足 `profile_scale_registration_ready=true`、`protocol_isomorphism_ready=true` 和 `artifact_contract_isomorphic=true`。这些状态只证明协议和流程可迁移，不改变 pilot 或 full 的 `evidence_incomplete` 科学状态。
