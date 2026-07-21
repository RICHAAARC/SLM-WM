# 项目构建状态与变更计划

## 1. 文档职责

本文档是项目构建状态的唯一有状态来源，记录真实仓库快照与两份无状态规范之间的差距：

1. `algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md` 只定义算法公式、输入输出、冻结参数、失败条件和主张边界。
2. `method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md` 只定义模块职责、接口、数据流、依赖方向、配置契约和验收要求。
3. 本文档记录哪些机制已经存在、哪些可复用、哪些需要修改、哪些必须退出正式路径，以及实际实施顺序和完成状态。

本文件不得修改前两份规范中的算法含义。若状态描述与算法原语或方法机制设计冲突，以前两份无状态规范为准，并把冲突登记为待修复项目状态。

---

## 2. 审计基线

| 项目 | 基线 |
|---|---|
| 审计日期 | `2026-07-21` |
| 重复协议文档修订日期 | `2026-07-17` |
| profile 与等价执行文档修订日期 | `2026-07-17` |
| 攻击证据职责文档修订日期 | `2026-07-17` |
| 源代码与已审 GPU 证据基线提交 | `db324b7c86a1bef305114fe83db44dfed04fd706` |
| GitNexus 索引提交 | `69a23070e3dcec8b4a60bc997446dda1efd4b528` |
| GitNexus 索引规模 | 14,698 symbols、32,708 relationships、300 execution flows |
| 当前活动构建单元 | `content_survival_observation_colab_drive_input`（基于已发布 `0bcc0ce7fbd3e34576074deb87c25ed22ebf7343` 的 CPU-only Drive 输入适配；未启动 Colab GPU） |
| 下一目标构建单元 | 本适配独立审计；通过后仍须由独立阶段授权真实 Colab A100 运行 |
| 受治理解释器 | 仓库 `.venv` 的 CPython 3.12.13 |
| 默认测试事实 | `.venv/bin/pytest -q -s` 为 `2354 passed, 82 deselected, 380 warnings`；精确 `.venv/bin/pytest -q` 仍在收集前触发宿主 capture 临时文件 `FileNotFoundError`，未运行 runtime-heavy GPU integration 或真实 Colab/A100 作业 |
| 定向协议/cache/约束检查 | observation protocol、专用 host 与 Colab 适配合计 `73 passed`；最小验收面覆盖公开 GitHub 提交、固定 Drive 输入、薄 Notebook、A100 显存门、secret 隔离、长时 pipe 排空、进程组清理、失败落盘和最终固定 Drive 交付，不建立模型供应链或通用持久化门禁 |
| harness 与格式检查 | 10项 harness 全部通过；`git diff --check` 通过 |
| 外部源码 qualification | 显式 integration 运行 `6 failed, 0 skipped`；失败均来自4套登记真实源码目录缺失，符合缺源失败关闭边界，不属于默认测试失败 |
| 工作树说明 | S3 从已独立审计的 `db324b7` 开始；`.codex/config.toml` 是既有范围外 untracked 文件，本原子不得修改或提交；旧 S1 存储归档保持只读 |
| 核心实现判断范围 | `main/`、`experiments/`、`configs/model_sd35.yaml`、`configs/method_semantic_registry.json` 及其测试和运行入口 |

已审 GPU 证据继续精确绑定 `db324b7`；`0b6abdc` 只登记为尚未科学验证的候选 M0。当前 CPU instrumentation 原子不修改 H1-H5、`semantic_conditioned_latent_method_definition`、`FormalMethodRuntimeConfig`、routing registry、key plan、检测公式、阈值或候选顺序。未来如获批运行，必须绑定本原子后续形成的精确 clean detached 提交；本表只用于安排构建，不支持论文主张。

### 2.1 `db324b7` 后的有效状态（覆盖下文历史差距快照）

- `db324b7` 是已审 S1 GPU 证据的代码基线，不是“方法科学通过”。S1 对 `0.75/1.00/1.25` 三个固定倍率各完成33条 calibration，但三档均只有 `11/33` 满足严格 `positive_source.content_score > wrong_key_negative.content_score`，参数裁决为 `no_compatible_candidate`，三个候选均 `science_blocked`。
- 同一 S1 记录中，`final_image_preservation`、`carrier_only_final_image_preservation`、`final_image_attention_observability` 三类正式 producer 证据均为 `0/33`。S2 因而只关闭了失败事实与根因边界，没有产生新 GPU 科学结果，也没有授权晋升名义 `1.00`。
- 当前 S3 原子新增独立 `content_survival_direction_v1` 协议：registered-only 对称 probe 仅冻结共同 direction/sign，wrong-key 在 nominal replay 后独立评价；clean/carrier/full 三图由真实 nominal replay 生成，CLIP 官方全局图像特征只承担质量保持，真实最终图像 Q/K 只承担 attention observability。两类质量证据都不得替代 key ownership。
- S3 writer/loader 使用同一全量 validator 和 manifest rename-last 原子发布；旧 S1 artifact 因缺少新协议、复合方法身份、七链父子记录和三图证据而明确不兼容，不迁移、不补造。
- 对存储端 S1 正式副本的只读 paired 复算确认已有原子分量足以做诊断：三档 `content_score` 均为11/33胜，均值 registered-wrong delta 约为 `-0.0080`，未配对 AUC 约为 `0.314–0.315`；LF 同为11/33，HF-tail 为5/33，attention-sync 为0/33，attention-geometry 为6–7/33。相同图像的 `embedding_pair_ssim` 与 `source_to_evaluated_ssim` 在 registered/wrong 两侧差值精确为0。八个最困难 Prompt 在三档稳定重合，描述上包含 object-centric、balanced、structured scene；这是旧记录的组件诊断，不是新方法通过证据，也不把 CLIP/语义量解释为 key ownership。
- 本原子目前只构成 CPU 可运行性与契约实现。survival-direction 是否改善同一水印图的 registered-vs-wrong 归因只能由下一次获批的全新 GPU run 证伪；在该结果形成前不得写成方法有效、GPU qualified 或论文证据。

### 2.2 候选 M0 的定位边界与独立 observation 原子

- `0b6abdc` 冻结为候选 M0，不是科学通过状态，也不得直接进入33条 calibration 或 qualification。其已知实现偏差包括：full/carrier direction 与 common gamma 未机械共享；counterfactual 身份为手工构造；partial manifest 写入后未从实际字节重读验证；bfloat16 realized-ratio tolerance 过宽；scheduler 身份只覆盖 z10 与 timestep。本 observation 原子只登记并暴露这些混杂，不修改候选 M0。
- 新增的独立 `content_survival_observation_v1` 固定4个 Prompt：`prompt_e6db4109e01246bc`、`prompt_d026a34e14f1806b`、`prompt_4ba678eb7a3abc94`、`prompt_184d2cc052881c3e`。前两者是 S1 稳定 winner，后两者是 S1 稳定 loser；顺序、选择职责和摘要均不可在运行中调整。
- observation matrix 固定为 semantic/uniform × LF-only/HF-only/dual 的6个 cell。每个 cell 固定执行 full±、carrier± 四条 registered-only probe，以及 fixed-positive before、selected-sign after 两条 nominal chain；每 Prompt 另有一条共享 clean chain，总计37条/Prompt、148条 diffusion chain。registered 与32个预登记 domain-separated wrong key 只复用既有图像/latent评分，不增加 diffusion chain。
- 每条 chain 真实保留 post-write z10、`output_type=latent` 的 terminal pre-VAE latent，以及手工 decode 最终图像后由冻结 `_encode_image_latent` 产生的 re-encode latent。三个 latent 位置直接调用正式 `build_low_frequency_template`、`build_high_frequency_tail_template` 与 `compute_blind_content_score`；图像检测器不承担 latent 评分。独立 routed-template oracle 只用于定位损失，不进入 sign、wrong-key gate、阈值、正式盲检或主张。
- 固定计算上界为148 chain × 3 latent位置 × 33 key × blind/oracle = 29,304次评分。24个 cell 各自采用 manifest rename-last；partial cell 不算完成，只能重做相同 roster identity，冲突或篡改失败关闭。公共 checkpoint/restore primitive 不变。
- canonical protocol 固定绑定 S1 正式 source run `content_strength_sensitivity_20260721T012732CST_db324b7`、`db324b7` 完整 commit、archive checksum/manifest/run-manifest 三项 SHA、routing registry 双摘要，以及按三档 minimum/maximum registered-minus-wrong margin 排序得到的四条 Prompt 选择证据；Prompt roster 另有独立 semantic digest 与 canonical artifact SHA。
- 每个 cell 的 result、binding 与 complete manifest 传播同一完整身份：已验证 formal execution lock、实际 orchestrator/scientific profile 与 hash-lock 身份、SD3.5/vision model revision、semantic runtime config digest、Prompt text/config digest、routing 双摘要、Prompt/key roster digest 与实际 commit。CLI 只从 observation protocol 读取固定 routing 双摘要，不接受环境变量替换。
- 共享 validator 从实际 `.pt`、PNG 与 score bytes 重算三类 latent、canonical RGB 尺寸/摘要、6×3 observation、registered+32 wrong-key 顺序、blind/oracle score identity、rank/percentile/null/margin、四 probe sign 选择、严格 callback role/sign、完整 scheduler/base/run/Prompt/protocol/roster 父身份与 clean sharing；从持久化 chain/observation 重建的父子关系摘要必须同时匹配 result、binding 和 manifest。即使协调改写 record/result/binding/manifest 摘要，只要 tensor 叶子或六链共同身份未同步变化仍会拒绝。summary 只统计经过该 validator 的24个 complete manifest 和4条共享 clean chain。
- 因候选 M0 的 full/carrier probe direction 与 common gamma 未共享，修订后的记录分别保存 `nominal_pair_unique_difference_ready`、`selector_confounded`、`confounded_reasons` 与 `causal_conclusion_ready`；当前 `selector_confounded=true` 且 `causal_conclusion_ready=false` 是机械边界，不因 nominal pair 单独一致而改变。
- 所有 bundle、result、manifest 与 record 固定 `diagnostic_only=true`、`supports_paper_claim=false`、`candidate_promotion_allowed=false`、`qualification_evidence=false`。该原子即使 CPU 测试闭合，也只表示 instrumentation 可运行；GPU仍处于 HOLD，旧 S1 archive 继续只读。
- `d4bc5b3` 的第一次独立 GPU diagnostic run 在模型加载和首个 cell 之前失败关闭：正式依赖环境本身通过，但 CLI 错把嵌套在 `provision_report` 的已验证 orchestrator identity 当作正式环境报告顶层字段读取，触发 `orchestrator inspection identity is absent`。该记录精确为0个 complete cell、0个 partial cell、0条 diffusion chain、0次评分，不是 M0、方法或科学失败，也不是 observation 结果。当前 CPU 修复只把 CLI 对齐到真实 `isolated_dependency_environment_preparation_report -> provision_report -> orchestrator_inspection` schema，并复算 provision/inspection digest、commit 与 formal lock；不修改正式 producer、模型、方法、routing、检测器或 GPU 工作量。
- `69a2307` 绑定的第二次全新 diagnostic run 已证明上述 dependency/orchestrator identity 接口修复通过，但在模型加载前因 host 链没有显式提供 `SLM_WM_KEY_MATERIAL` 再次失败关闭；该记录仍为0个 model load、0个 complete/partial cell、0条 diffusion chain、0次评分，分类为 `required_key_material_input_not_provided_by_host_chain`，不是 M0、方法或科学失败。根因是临时 host driver 未接收该输入，而共享 scientific executor 的默认 runner 只能继承父环境；把 raw key 放进父环境会错误暴露给父编排、依赖安装与 inspection。
- CPU-only 修复新增 observation 专用 host 入口：从受控宿主环境显式取得 raw key 后立即从宿主环境移除，在准备精确父编排和科学依赖时不传播；精确父编排只在内存接收输入，并由 single-use command runner 仅向唯一 `run_content_survival_observation.py` scientific child 的进程环境注入。runner 在第一次合法 launch attempt 前消费能力，后续相同 target 也必须在第二个 key 环境构造前拒绝；持久化状态从真实 invocation/attempt/completed/non-target rejection 计数生成，首发 `OSError` 不得宣称 child 已启动完成。argv、父环境、命令记录、stdout/stderr、JSON/manifest、状态与异常均不得出现 raw key；持久化 host identity 只包含域隔离 SHA-256 与 `registered_watermark_key_material` 角色。缺失输入继续在 checkout、依赖准备和模型加载前失败关闭，非目标 child 或身份漂移均拒绝。该修复只闭合程序输入接口，不构成 GPU diagnostic 或科学通过。

下文第3节及后续以 `b31ffeb` 为起点的“大迁移差距”保留为历史计划，不再覆盖本节登记的 `db324b7`、S1/S2 与 S3 事实。

### 2.3 Colab 执行适配边界

- 当前新增的是 CPU-only 平台适配，不是 Colab/A100 科学运行结果。永久薄 Notebook 只通过匿名 HTTPS 获取已独立审计并发布到 `origin/main` 的提交，控制器再次核对远端 `main` 与 run request 中的40位提交完全一致，随后建立 clean detached checkout；运行期间不执行 `pull`。当前已发布基线为 `0bcc0ce7fbd3e34576074deb87c25ed22ebf7343`；Drive 输入适配补丁尚未提交或 push，控制任务必须在新提交完成独审并精确发布后再生成与该提交绑定的 `run_request.json`，不能复用现有 Drive request。
- Notebook 不实现安装、模型、方法、secret、validator 或打包逻辑，只依次调用稳定的 `prepare-drive-input`、`bootstrap-public`、`preflight`、`run` 和 `package-drive` 子命令。最后一个 cell 重新从固定 request 与本地磁盘定位控制器，不依赖前面 cell 的运行时变量，可在 success、preflight failure、OOM 或 scientific failure 后单独执行。
- 所有仓库、隔离运行目录、模型下载 cache、checkpoint、日志和中间结果只写 `/content`。SD3.5 与 CLIP 继续由既有方法 host 按 `configs/model_sd35.yaml` 的普通运行参数通过标准加载入口下载和加载；本适配不新增 revision、snapshot、文件摘要、断链、offline 重载或模型 identity 门禁。下载/加载失败按普通可打包运行失败处理，不宣称供应链增强。
- 适配控制器只要求 A100 且启动时 total 与 available 显存均不少于40,000 MiB。A100-80G 也可通过但不会自动选择；L4 或不足门禁的 A100 失败。A100-40G 如果真实加载或首 cell OOM，则诚实落盘并打包，后续切换80G需另行授权；不得启用 CPU offload、量化、attention slicing 或降低科学工作量。
- 固定输入目录为 `/content/drive/MyDrive/SLM/content-survival/inputs/`，其中非秘密 `run_request.json` 由控制任务在目标提交发布后生成，raw key 固定来自私有 `watermark_raw_key.txt`。controller 在启动边界短暂挂载 Drive，把 request 复制到固定 `/content` 路径；raw key 仅在 `run` 时从固定文件读入内存并立即卸载 Drive，经既有 single-use 内存环境传给唯一 host 链。可选 HF token 继续来自 Colab Secret。raw key 不进入 Notebook、argv、日志、JSON、archive、controller state 或本地持久文件。
- child 启动前只落盘 `watermark_used` 与 `hf_token_used` 非秘密布尔。处于 running、success 或 scientific failure 的运行在打包时必须重新从固定 Drive key 文件取得 raw key，并在 HF token 已使用时重新取得该 Secret，才能执行内容扫描；key 缺失、不可读、HF Secret 权限撤销或 usage 状态缺失都失败关闭且不复制结果。尚未向 child 传递 secret 的 bootstrap/preflight failure 仍可无 secret 独立打包。
- 科学运行期间 Drive 已卸载，不执行 Drive hot I/O，也不写 checkpoint、日志或中间结果。最后 cell 先在 `/content` 从磁盘状态分类 success/failure，完成 secret scan、文件清单、archive、detached checksum 和独立解包复验；仅全部通过后才再次 mount Drive，并按 archive 后 checksum 的顺序复制到固定 `/content/drive/MyDrive/SLM/content-survival/results/`。VM 回收前未执行该 cell 会丢失本地结果，这是当前被接受的运行边界。
- 固定科学工作量仍由既有 single-use host → scientific child 执行：4个 Prompt、24个 cell、148条 diffusion chain、29,304次评分，且保持 `diagnostic_only=true`、`supports_paper_claim=false`、`candidate_promotion_allowed=false`、`qualification_evidence=false`。CPU 适配验收不证明真实 Colab/A100 运行成功。

---

## 3. 总体状态

| 构建面 | 状态 | 判断 |
|---|---|---|
| 三份中心文档职责分离 | `core_documents_frozen` | 算法公式和6角色集合保持不变；正式随机化冻结为5个预登记 seed-key 配对，pilot 为主投稿、full 为可选扩展，攻击职责为7项核心 required 与10项补充描述性。两项原 HIGH schema 阻断已闭合，定向检查和独立只读复审均确认三份中心文档不存在定稿阻断；该状态不表示机器协议或 runtime 已实现 |
| 其他文档生态同步 | `protocol_documentation_updated` | 人类可读文档已同步6角色、5重复、pilot/full 证据职责、7/10攻击分层及缓存/恢复/并行目标；11个目标治理字段已登记到字段生命周期 target schema，但不表示 writer 已实现。机器随机化、攻击 registry、profile gate、质量/主张过滤器、缓存身份、聚合器、配置摘要和相关测试仍需原子切换，因此 `document_ecosystem_synchronized` 保持未完成 |
| 目标核心方法实现 | `not_implemented` | 正式运行仍执行716维特征、Jacobian Null Space、JVP/VJP、PSD-CG 和三时刻注入路径 |
| 内容自适应路由 | `not_implemented` | 尚无由 Prompt 条件空间显著性、Sobel 纹理、相邻 latent 响应和公开探针敏感性共同构造的 `S/T/R/Q` 路由 |
| 路由 reference registry | `not_materialized` | `configs/content_routing_reference_registry.json` 尚不存在，真实 `g_ref/r_ref/q_ref` 及其隔离参数划分仍未物化 |
| LF 主证据 | `partially_reusable` | 已存在真实二维低通密钥载体和0.70检测权重，但尚未接入目标空间路由和单时刻写回 |
| HF-tail 困难攻击补充 | `nonconformant` | 已存在确定性幅值 tail，但缺少高通前置步骤，不能作为目标 HF-tail |
| Q/K 几何同步嵌入 | `partially_reusable` | 真实 Q/K、四分量关系、稳定 token 和带密钥模板存在，但几何更新仍耦合 `JacobianNullSpaceResult` |
| 几何恢复与同阈值救回 | `partially_reusable` | 有界搜索、回正、重新编码、aligned 内容分数和同阈值救回基础存在，但仍绑定旧内容载体和旧方法身份 |
| 近阈值后按需几何搜索 | `nonconformant` | 现有检测基础可复用，但目标接口尚未显式分离 raw measurement、geometry search 和 frozen decision 编排 |
| 等价执行复用与样本级并行 | `documented_not_implemented` | clean 跨角色复用、公开原子/质量特征/攻击缓存、共享生成前缀、Prompt-repeat checkpoint 和多 GPU ownership 已冻结文档边界；runtime、cache manifest、等价性测试和调度 validator 尚未实现 |
| 核心/补充攻击证据职责 | `documented_not_implemented` | 目标文档冻结7项核心攻击进入6角色、4 baseline 和 required claims，10项补充攻击只作描述性结果；当前 `default_attack_configs()` 仍以 `resource_profile` 区分18项配置，闭合链仍把17项正式攻击全部视为共同 required 矩阵 |
| 完整 fixed-FPR 校准 | `nonconformant` | 现有 `image_only_evidence.py` 只使用 clean negative，并拟合 geometry score/confidence/sync 门；目标协议要求三组负观测分别受预算约束，几何结构门预先冻结 |
| 最终图像 Q/K 双归因 | `partially_reusable` | runtime 已有 full/carrier-only 对照基础，但尚未按新方法冻结 registered/wrong-key 与 matched content-only 原子 schema |
| 204维手工结构分量 | `remove_from_core_gate` | 当前资格化和配置仍把它作为机制保持门禁；目标方法只允许在 `experiments/` 中保留诊断，不允许筛除正式样本 |
| 单时刻三分支一次写回 | `not_implemented` | `configs/model_sd35.yaml` 仍登记 `injection_step_indices: [6, 10, 14]` |
| 目标方法 CPU 性质测试 | `not_implemented` | 现有测试主要冻结迁移前实现，不能证明目标原语 |
| 目标方法 GPU 资格化 | `blocked` | 资格化仍要求716维 JVP/VJP、PSD-CG 和三时刻写回事实 |
| 三档正式实验 | `blocked` | runner、方法摘要、攻击证据职责、消融、记录和结果包仍绑定迁移前方法身份；目标上 pilot 只由7项核心攻击闭合主投稿证据，full 只作为可选规模扩展，10项补充攻击不进入 required gate |

结论：仓库拥有可复用的 LF、真实 Q/K、几何恢复和同阈值救回基础，但还不是两份无状态规范定义的“内容域自适应嵌入 + 几何链救回重判”实现。

---

## 4. GitNexus 影响面

下表来自上述精确索引提交的 upstream impact。修改任何函数、类或方法前仍必须重新运行 GitNexus；本表不能作为后续修改的豁免。

| 符号 | 风险 | 直接依赖 | 累计影响 | 受影响流程或说明 |
|---|---:|---:|---:|---|
| `semantic_conditioned_latent_method_definition` | CRITICAL | 9 | 82 | 影响8组流程，连接主 runtime、消融、GPU 资格化、聚合与测试；必须协调切换 |
| `optimize_attention_geometry_update` | MEDIUM | 7 | 7 | 直接影响7个功能测试；需要解除 Null Space 耦合但保留真实 Q/K 关系 |
| `recover_attention_affine_alignment` | MEDIUM | 11 | 11 | 直接进入 `measure_image_only_watermark` 及10个功能测试；必须保留捕获域内恢复能力 |
| `measure_image_only_watermark` | LOW | 0 | 0 | 图谱未识别动态调用上游；真实 runner 仍消费该检测路径，实施时按至少 MEDIUM 管理 |
| `build_low_frequency_template` | LOW | 3 | 3 | 真实二维 LF 构造可复用 |
| `build_tail_robust_template` | MEDIUM | 5 | 5 | 检测和载体调用者均受影响；必须改为先高通再 tail |

CRITICAL 风险处理原则：先建立目标 schema 和消费者迁移清单，再切换方法定义；不得先删除旧字段后逐个修复外层报错。

### 4.1 CRITICAL 方法定义的9个直接依赖

以下清单来自 GitNexus `context`，是方法身份切换时必须同步处理的完整直接依赖：

| 直接依赖 | 文件 | 迁移动作 | 完成证据 |
|---|---|---|---|
| `semantic_conditioned_latent_method_definition_digest` | `main/methods/method_definition.py` | 与新方法定义同时替换 schema 和摘要输入 | 新定义与摘要逐字段重建测试 |
| `test_machine_readable_method_definition_freezes_constructive_semantics` | `tests/constraints/test_method_definition_contract.py` | 从716维/Null Space 断言切换为双链、单注入、盲检测和同阈值救回断言 | constraint 通过且不再导入旧 schema |
| `test_runtime_config_identity_binds_method_definition` | `tests/constraints/test_method_definition_contract.py` | 绑定新配置、reference registry 和新方法摘要 | 配置任一字段变更均改变摘要 |
| `_ablation_atomic_fixture` | `tests/functional/test_formal_atomic_record_rebuild.py` | 将 fixture 迁移为最小6个方法/消融角色及新字段 | 原子记录重建测试通过 |
| `ablation_atomic_records` | `tests/integration/test_result_closure_gate.py` | 迁移闭合 fixture，不得手工补造新方法结论 | 从原子记录正向重建闭合包 |
| `semantic_watermark_runtime_config_payload` | `experiments/runners/semantic_watermark_runtime.py` | 绑定单注入、`S/T/R/Q`、HF-tail、Q/K 和 reference registry | 配置 payload 与新方法定义同摘要 |
| `validate_semantic_watermark_runtime_result_provenance` | `experiments/runners/semantic_watermark_runtime.py` | 校验新方法原子、盲载体能量和最终 Q/K 归因 | 旧字段或缺失新字段时失败关闭 |
| `_scientific_content_binding_validation_parameters` | `experiments/runners/semantic_watermark_runtime.py` | 移除716/204核心绑定，改为显著性、路由和最终图像诊断 | 返回值不再依赖旧联合特征 |
| `run_semantic_watermark_runtime` | `experiments/runners/semantic_watermark_runtime.py` | 切换为索引10一次三分支写回及近阈值后按需几何搜索 | CPU 性质测试和真实 GPU 资格化 |

切换顺序固定为：方法 schema 与摘要构造器 -> 配置 payload 和只读 validator -> 测试 fixture -> runtime。每一步修改具体函数前都必须重新执行 GitNexus upstream impact。

---

## 5. 保留清单

以下机制具有真实实现基础，应保留其科学语义并迁移到正式接口，不得因删除 Null Space 而整体删除：

| 保留机制 | 主要位置 | 保留范围 |
|---|---|---|
| 稳定摘要与密钥 PRG | `main/core/digest.py`、`main/core/keyed_prg.py` | 保留确定性、跨平台向量和域隔离；为 LF、HF-tail、geometry 和公开敏感性探针使用独立 domain |
| 二维 LF 构造 | `main/methods/carrier/keyed_tensor.py::build_low_frequency_template` | 保留真实 H/W 低通、密钥载体、中心化和归一化；补接 `M_LF` 路由 |
| 幅值 tail 的确定性排序组件 | `main/methods/carrier/keyed_tensor.py::build_tail_robust_template` | 只复用 PRG、稳定排序、逐样本截断和摘要；在其前增加二维高通并重命名为 HF-tail 语义 |
| 真实 Q/K 记录和四分量关系 | `main/methods/geometry/differentiable_attention.py` | 保留 `to_q`/`to_k`、逐 head 关系、稳定 token、pair 权重、四分量、极性和摘要 |
| 有界几何搜索 | `main/methods/geometry/attention_alignment.py::recover_attention_affine_alignment` | 保留二面体 + 有界相似变换搜索、coverage、inlier、residual、正 registration objective margin、直接 Q/K identity 和失败关闭；不得宣称任意 affine |
| 图像回正与重新编码 | `main/methods/detection/image_only.py` | 保留真实图像恢复、公开 VAE 重新编码和 aligned 内容测量框架；冻结 `affine_grid/grid_sample` output-to-input 矩阵直接采样约定，不得对搜索矩阵再次求逆 |
| 公开仅图像 Q/K 条件 | `main/methods/detection/image_only.py`、`experiments/runners/semantic_watermark_runtime.py` | 保留 VAE posterior mode、规范公开噪声、scheduler 索引7、empty-text triplet、无 CFG 和直接 Q/K；迁移时不得改成随机噪声或读取原始 Prompt |
| 同阈值救回 | `experiments/protocol/image_only_evidence.py` | 保留 Prompt 级嵌套 calibration、最宽可行近阈值窗口、同一内容阈值和 geometry 不独立正判语义；移除旧 geometry score/confidence/sync 阈值拟合 |
| 单次 dtype 转换组合工具 | `main/methods/update_composition.py` | 保留 float32 有序组合、共同缩放、真实 dtype 写回摘要；运行时改为只在索引10调用一次 |
| 冻结模型与依赖身份 | `configs/model_sd35.yaml` 及 dependency registry | 保留 SD3.5、CLIP、VAE、层名、revision、依赖 profile 和 fail-closed 身份校验 |
| 独立视觉内容质量评估器 | `configs/independent_semantic_quality_evaluator.json`、`experiments/protocol/independent_semantic_quality.py` | 保留冻结 DINOv2 模型、预处理、特征层、归一化、依赖身份和只读质量证据职责；路径中的 `independent_semantic` 是兼容标识符，不表示 Prompt 语义或图文对齐；不得进入方法优化或检测 |
| 分层执行与证据治理 | `experiments/`、`paper_experiments/`、`scripts/`、`paper_workflow/` | 保留由内向外的依赖结构、真实记录、聚合、服务器入口和薄 Notebook 边界 |

---

## 6. 修改清单

### 6.1 文件级处理表

| 路径 | 当前职责 | 目标处理 | 删除条件或验收 |
|---|---|---|---|
| `main/methods/method_definition.py` | 716维局部切空间方法身份 | 原子替换为双链方法身份 | 9个直接依赖全部切换后删除旧 schema |
| `main/methods/semantic/feature_protocol.py` | 512维 CLIP + 204维结构联合特征协议 | 退出核心方法身份；可复用的 CLIP 模型身份下沉到 `content/saliency.py` | `main/` 不再构造716维联合特征 |
| `main/methods/semantic/runtime.py` | 旧语义特征运行与模型加载 | 只复用冻结 CLIP/VAE 加载、revision 和预处理身份；显著性改由新接口构造 | 无 Prompt 的全局/局部一致性不再冒充显著性 |
| `main/methods/semantic/vector_values.py` | 204维手工结构向量和旧联合特征值 | 退出 `main/` 核心门禁；如仍需诊断则迁移到 `experiments/` 并使用诊断字段身份 | `main/`、正式配置和资格化均不再导入 |
| `main/methods/semantic/branch_risk.py` | 手工 branch risk 权重和预算 | 由 `content/routing.py` 的 `S/T/R/Q` 公式替代 | 所有正式调用者为零后删除或移出最小包 |
| `main/methods/content/{saliency,texture,latent_response,local_sensitivity,routing}.py` | 不存在 | 新建五个单一职责内容观测与路由模块；统一消费 reference registry | CPU 性质测试覆盖公式、形状、反事实、公开探针和禁止输入 |
| `main/methods/carrier/keyed_tensor.py` | LF、纯幅值 tail、内容相关统计 | 保留 LF；将 tail 改为先高通后逐样本 tail；明确未掩码盲相关 | CPU 载体性质与盲统计测试通过 |
| `main/methods/geometry/differentiable_attention.py` | 真实 Q/K 与 Null Space 耦合几何更新 | 保留 Q/K 四分量和模板，移除 `JacobianNullSpaceResult` 输入 | 直接 Q/K 梯度和最终图像归因通过 |
| `main/methods/geometry/attention_alignment.py` | 有界几何搜索 | 保留算子；调用时机改为近阈值之后 | raw 分数在窗口外时搜索调用计数为零 |
| `main/methods/detection/image_only.py` | 旧内容载体测量、几何恢复和 aligned 评分 | 拆分 raw measurement、geometry search、frozen decision 编排 | `geometry_reliable` 仅作为搜索输出 |
| `main/methods/detection/evidence_decision.py` | 不存在 | 新建只应用 `FrozenDualChainDecision` 的纯决策模块；不得包含 calibration、攻击或统计 | 给定同一测量与决策身份可确定性重建全部布尔字段 |
| `main/methods/subspace/jacobian_nullspace.py` | JVP/VJP、QR、PSD-CG | 退出正式路径和最小包 | `rg`、GitNexus 和测试均证明正式调用者为零 |
| `main/methods/subspace/semantic_projection.py` | 旧联合语义特征投影和 Null Space 前置量 | 退出正式路径和最小包，不得改名后继续承担内容路由 | `rg`、GitNexus 和测试均证明正式调用者为零 |
| `main/methods/update_composition.py` | float32 组合和 dtype 写回 | 保留工具，运行时只调用一次 | 索引10单次三分支写回测试通过 |
| `main/methods/__init__.py` 及各子包 `__init__.py` | 导出迁移前方法与 subspace 接口 | 原子切换为双链公开接口，删除正式旧导出 | 最小包导入测试只暴露目标方法 |
| `configs/model_sd35.yaml` | 旧风险、Null Space 和三注入配置 | 替换为新方法唯一配置入口 | 不再包含旧字段或兼容性开关 |
| `configs/method_semantic_registry.json` | 迁移前方法不变量 | 原子替换为双链不变量 | 旧 definition pointer 全部退出正式登记 |
| `configs/core_method_dependency_identity.json` | 迁移前核心方法依赖身份 | 重新绑定目标模型、PRG、Q/K、reference registry 和单注入身份 | 摘要可从新配置逐字段重建 |
| `configs/content_routing_reference_registry.json` | 不存在 | 新建并绑定真实参数划分、观测摘要和 `g_ref/r_ref/q_ref` | 隔离 GPU 参数物化作业可逐字段重建 |
| `configs/paper_profile_protocol_registry.json`、`paper_experiments/analysis/paper_profile_protocol_isomorphism.py` | 目标登记已包含最小6角色、三类样本角色和 success/failure schema 身份，但随机化实现仍登记9重复，尚未机器登记 pilot 主投稿/full 扩展证据角色及7/10攻击职责，正式 runtime 尚未生产目标 schema | 保持三档共享角色、字段摘要、决策程序、诊断 schema、7项核心/10项补充攻击和精确5个预登记 seed-key 配对；登记 pilot 为主投稿、full 为可选扩展，只允许规模、目标 FPR 与派生统计强度变化 | 对角色、样本角色、schema 字段、5重复、核心攻击 exact-set、补充攻击身份、profile 证据职责或操作路径角色的任一漂移均失败关闭 |
| `experiments/protocol/attacks.py`、攻击矩阵 writer 与 validator | 当前1项 probe、9项 `full_main` 和8项 `full_extra` 只按资源档位分类；17项均进入正式共同矩阵 | 保留18项攻击 ID 和参数，新增与资源档位正交的 `attack_evidence_role`：7项 `core_claim_required`、10项 `supplementary_descriptive`、1项非主张 probe；核心矩阵进入全部角色/baseline/required gate，补充只允许完整方法描述性执行 | 资源档位推断证据职责、核心集合不精确、补充进入 calibration/claim、结果后改角色或部分补充冒充完整集合时失败关闭 |
| 目标 cache/checkpoint manifest 与多 GPU 调度 validator | 尚无覆盖共享生成前缀、公开 measurement package、攻击缓存和样本 ownership 的统一机器身份 | 新增完整输入摘要、producer 版本、依赖、schema、内容摘要、命中来源和等价性状态；缓存与 checkpoint 固定不支持论文主张 | 身份不全、跨密钥复用依赖原子、跨图像攻击命中、重复 ownership 或共享/独立执行不等价时失败关闭 |
| `configs/field_lifecycle_registry.json`、`docs/field_registry.md` | 已建立目标、迁移前和共享字段的确定性优先级及显式 legacy 集合，但 runtime writer 尚未整体迁移 | 核心与证据 writer 迁移时必须按同一分类拒绝 legacy 字段进入目标记录，并保持 field_name 全局唯一 | 代表字段分类、全表唯一性和目标结果禁用 legacy 的约束测试全部通过 |
| `experiments/runners/semantic_watermark_runtime.py` | 旧主 runtime | 切换为新核心接口和单次写回 | 不再导入 subspace 或旧方法定义 |
| `experiments/protocol/method_runtime_config.py` | 解析迁移前多注入、Null Space 和风险配置 | 解析唯一 `B=1`、post-step callback、reference、双载体和几何配置 | 旧字段和兼容分支失败关闭 |
| `experiments/protocol/gpu_method_qualification.py` | 716维、CG、三写回资格化 | 改为新算子、盲统计、最终 Q/K 双归因和救回顺序 | 新报告通过且 `supports_paper_claim=false` |
| `experiments/protocol/image_only_evidence.py` | clean-negative-only 嵌套校准、geometry score/confidence/sync 阈值和完整决策应用 | 保留 Prompt 分区与同阈值语义；改为三组负观测分别受预算约束，接入 lazy geometry 和冻结结构门禁 | 三组原始记录能够正向重建完整 fixed-FPR 决策，且不再拟合 geometry 分数门 |
| `experiments/protocol/paper_fixed_fpr.py` | 仍生成迁移前 `*_claim` 三档别名并把正式摘要绑定 clean-negative-only calibration 角色 | 迁移为 `probe_paper/pilot_paper/full_paper`，并只用7项核心攻击登记 clean/attacked/wrong-key 三组负观测各自计数、预算和置信边界；补充攻击使用冻结核心决策器而不回调 | 旧命名、单一 clean calibration、补充攻击进入拟合或补充结果回写核心阈值时不能进入目标摘要 |
| `experiments/ablations/` | 迁移前 branch risk 敏感性和旧机制消融 | 收敛为最小6角色开关和独立 calibration；参数敏感性只保留第7.2节单模型单因素方案 | 不存在逐项 `S/T/R/Q` 正式三档矩阵或第二套方法实现 |
| `paper_experiments/runners/randomization_ablation_necessity.py` | 聚合迁移前消融身份 | 只聚合最小6角色及其预登记比较 | 缺角色、额外角色或阈值复用均失败关闭 |
| `paper_experiments/runners/randomization_parameter_sensitivity.py` | 聚合旧 branch risk 参数且按9重复形成迁移前统计 | 退出目标入口；由目标单模型诊断 writer 聚合 reference 分位数、探针步长、内容共同倍率和 geometry 倍率的一个固定 repeat | 目标诊断产物使用新语义身份，缺失不阻断四项主张，存在时必须严格复验且不得回写正式参数 |
| `scripts/write_paper_complete_result_package.py` | 已通过唯一构建文档清单登记三份中心文档，但结果 schema 仍绑定迁移前方法 | 核心迁移时同步切换最小6角色和新字段，不再新增平行文档清单 | 打包清单与真实 writer/profile registry 完全一致 |
| `scripts/extract_release_package.py`、`scripts/validate_core_method_package.py` | 抽离和 validator 仍允许迁移前 `main/methods/subspace/` 进入最小包，只验证文件身份、依赖与可导入性 | 与核心 runtime 原子迁移：目标最小包必须包含 content/carrier/geometry/detection，存在旧 subspace 或缺少目标公开接口即失败 | 迁移前抽离成功不得称为目标核心方法发布；迁移后 validator 对旧目录和旧导出失败关闭 |
| `tools/harness/lib/method_semantic_registry.py`、`tools/harness/audits/audit_method_semantic_registry.py` | 校验迁移前方法登记和摘要 | 切换为两份无状态规范、新 registry、最小6角色及禁止旧方法身份 | harness 对额外、缺失和旧字段均失败关闭 |
| `tests/constraints/test_extraction_contract.py` 及相关 extraction tests | 仍断言抽离/安装后可导入旧 Jacobian Null Space 模块 | 与抽离器和 validator 原子切换，明确断言旧 subspace 缺失且目标双链公开接口完整 | 旧模块存在、目标目录缺失或包内引用断裂均失败 |
| 其余 `tests/constraints/`、相关 `tests/functional/` 与 `tests/integration/` | 大量 fixture 和断言仍冻结迁移前语义 | 按“定义/schema -> 配置 -> 原子记录 -> runtime -> 聚合闭合”顺序迁移，不允许手工补造最终支持结论 | 默认测试、定向闭合测试和 harness 全部通过 |

### 6.2 核心方法层

1. 新增 `main/methods/content/`，实现 Prompt 条件 CLIP patch-text 显著图、冻结 Sobel 纹理图、索引9/10相邻 latent 响应图、公开单方向探针敏感性图和 `S/T/R/Q` 路由；CLIP 文本特征固定为 EOS pooled output 经 `text_projection`。
2. 将 `build_tail_robust_template` 改为“二维高通残差 -> 每个 `B=1` 样本稳定选择 `max(1,ceil(0.20*C*H*W))` 个 TopAbs 坐标 -> L2 归一化”的 HF-tail；不能只改函数名或 metadata。
3. 将几何更新从 `JacobianNullSpaceResult.project()` 解耦，改为索引10 content-only 基底处的一次真实 Q/K 目标梯度、独立0.0010预算和最多9个比例候选。
4. 将 LF、HF-tail、geometry 以 `0.0025/0.0015/0.0010` 在 float32 中组合，在总预算 `0.0050||z||_2` 内只执行一次实际 dtype 写回。
5. 将仅图像内容分数迁移为未掩码 LF/HF-tail 的0.70/0.30盲相关；记录掩码均值、非零比例和分支有效能量，禁止筛除低容量样本。
6. 将检测执行拆为 raw content measurement、近阈值判断、按需 geometry search、可靠回正和同阈值重判；`geometry_reliable` 只能由搜索产生。搜索必须冻结 D4 矩阵与候选顺序、局部组合公式、`J(T)` 最大化、并列取最早和 output-to-input 重采样约定。
7. 在最终图像记录 registered/wrong-key Q/K 分数；单 Prompt 资格化另外运行 matched content-only 反事实并验证双归因。
8. 将204维手工结构描述符移出核心门禁，只允许在 `experiments/` 作为诊断记录。
9. 将方法定义替换为两份无状态规范的完整机器身份，并删除与716维局部切空间绑定的正式 schema。

### 6.3 配置和登记

1. 将 `injection_step_indices: [6, 10, 14]` 替换为 `method_batch_size: 1`、`callback_api: callback_on_step_end`、从0开始的 post-step callback 语义和精确整数 `injection_step_index: 10`。
2. 删除 `jacobian_candidate_count`、`null_space_rank`、CG、QR、投影能量和 JVP 响应门禁等目标方法不使用的正式配置。
3. 用 `g_ref`、`r_ref`、`q_ref`、公开探针、内容路由、HF 高通、几何预算和捕获域字段替换手工风险权重入口。
4. 新建 `configs/content_routing_reference_registry.json`，绑定参数划分、Prompt/seed 摘要、模型预处理身份、真实观测摘要、95分位算法和三个精确标量；观测总体分别固定为插值前有限正 `G`、latent 分辨率有限正 `d_R`、插值前有限正 `d_Q` 的跨成员拼接。
5. 原子替换 `configs/method_semantic_registry.json`；不得让新旧方法不变量同时被标记为正式。
6. 将机器随机化协议从迁移前3 seed × 3 key 的9重复原子切换为权威登记的5个有序 seed-key 配对；三档固定复用同一集合与顺序，只在 Prompt / 样本规模、目标 FPR、统计强度及其派生记录数量上变化；FPR 固定为 `0.1/0.01/0.001`。
7. 在 profile registry 和主张 gate 中登记 `pilot_paper` 为主投稿证据、`full_paper` 为可选扩展；full 缺失不得进入 pilot required gate，full 自身主张仍要求独立闭合。
8. 为18项攻击配置登记与 `resource_profile` 正交的证据职责：7项核心攻击完整进入6角色、4 baseline、三组 calibration 和四项 required claims；10项补充攻击只允许完整方法使用冻结核心决策器形成描述性结果；probe JPEG 保持非主张工程职责。

### 6.4 运行、检测与实验

1. 以目标方法 runtime 替换 `experiments/runners/semantic_watermark_runtime.py` 的716维和三注入执行路径。
2. 将 GPU 资格化从“716维 JVP/VJP + PSD-CG + 三写回”改为验证 `S/T/R/Q`、HF-tail、真实 Q/K、单时刻三分支、一次写回和几何救回。
3. 从同一核心实现生成最小6角色：`full_dual_chain`、`uniform_content_routing`、`lf_only_content`、`hf_tail_only_content`、`content_chain_only`、`geometry_recovery_without_embedded_sync`；删除逐项 `S/T/R/Q` 消融和 `geometry_sync_without_rescue` 等非必要完整三档矩阵。
4. 新增单模型小规模参数敏感性 runner，固定一个 Prompt 子集和一个 repeat，只运行无反馈的单因素候选。
5. 迁移 `method_role`、`watermarked_positive`、`clean_negative`、`attacked_negative`、registered/wrong-key、`scoring_key_identity_digest`、图像成员路径、`content_threshold`、可空 `rescue_margin_low`、`attack_evidence_role`、核心/补充攻击后检出率、质量记录和生成式攻击对称评测。
6. 使单 repeat、5-repeat 聚合和完整结果包消费新的方法摘要、字段和产物 schema。
7. 实现受摘要约束的 clean/measurement/attack/profile-invariant cache、Prompt-repeat 幂等 checkpoint、惰性几何搜索和样本级多 GPU ownership；共享前缀必须具有对六次独立运行的等价性测试和 fail-closed 回退。

---

## 7. 移除清单

以下内容必须在调用者完成切换后退出正式执行、配置、资格化、结果身份和最小发布包：

1. 716维联合特征水平集、一阶局部切空间和“潜流形”方法身份。
2. `main/methods/subspace/jacobian_nullspace.py` 及其 JVP/VJP、QR Null Space、PSD-CG 正式依赖。
3. 三个注入索引、三次实际 latent 写回以及与多时刻耦合的中间状态。
4. 无 Prompt 的 CLIP 全局/局部一致性被表述为语义显著性的路径。
5. 只做原始高斯幅值 TopAbs、但被表述为 HF-tail 的路径。
6. 旧 `local_geometry`、`joint_feature_width`、手工 branch risk、CG 和投影响应配置。
7. 204维手工结构描述符作为核心方法门禁或正式样本筛除条件的路径。
8. geometry score 对正式阳性判定的任何影响。
9. 仍绑定迁移前方法定义的 GPU 资格化事实、消融身份、repeat 证据、结果包和论文主张资格。

移除不等于立即物理删除文件。只有当 `rg`、GitNexus 和测试证明正式调用者为零后，才删除源文件或将纯历史记录移出发布清单。不得为兼容旧结果保留可被正式入口选择的第二套方法。

---

## 8. 不得误删或弱化

1. 不得删除 `main/methods/geometry/` 整个目录。
2. 不得删除真实 Q/K 四分量、稳定 token、pair 权重和带密钥关系模板。
3. 不得把几何恢复降级为攻击标签触发、像素启发式或预制 transform。
4. 不得删除真实图像回正、重新编码和同阈值内容重判。
5. 不得把双链误写为两个张量分支；一次写回仍包含 LF、HF-tail、geometry 三个更新分支。
6. 不得用单 Prompt、CPU fixture、mock hook、历史结果或旧提交结果声明目标方法完成。
7. 不得为了降低 GPU 成本回退为 proxy、synthetic、placeholder、随机特征或兼容性占位结果。

---

## 9. 有序实施计划

### 构建单元 A：三份中心文档冻结

1. 确认算法原语不包含仓库进度、历史实现、迁移任务和提交状态。
2. 确认方法机制设计不包含当前文件、保留/删除清单、GitNexus 快照和实施进度。
3. 确认所有有状态事实只出现在本文档。
4. 通过只针对三份中心文档的职责排他、公式唯一性、schema 闭合和交叉一致性审计。
5. 本单元只冻结三份人类可读核心文档的规范职责；机器随机化、攻击 registry、profile gate、质量/主张过滤器、缓存身份、聚合器和运行配置的迁移不属于本状态前置条件，也不阻断核心方法 runtime 构建启动。

### 构建单元 B：其他文档生态与机器协议同步

1. 在 `core_documents_frozen` 后，以三份核心文档为唯一依据继续修订其余项目文档和机器协议；该单元可与核心方法 runtime 构建按明确文件边界推进，不是核心方法实现的前置条件。
2. 同步其他人类可读文档、根契约、README、完整结果包、release profile 和构建文档 inventory，以及直接消费核心文档的机器目标契约、registry、配置摘要和约束测试；不得以同步名义改写核心算法或物化未冻结科学身份。
3. 删除或移出过期、冗余且不能承担唯一职责的旧文档，不复制第二套公式、方法接口或迁移状态。
4. 运行文档生态定向测试、默认 pytest、harness 和 `git diff --check`，通过后才能提升为 `document_ecosystem_synchronized`。该状态不是核心方法 runtime 构建的前置条件，但必须在进入 `experiment_protocol_validation`、正式证据生产或论文结果生产前完成。

### 构建单元 C：低风险内容算子

1. 在 `core_documents_frozen` 后开始，实现 Prompt 条件 patch 显著图、Sobel 纹理图、相邻 post-step latent 响应图和公开探针敏感性图，统一固定 `B=1`；本单元及后续核心 runtime 单元不得生产正式实验或论文证据。
2. 实现 `S/T/R/Q` 路由和先高通、再按 `ceil(0.20*C*H*W)` 稳定选取的 HF-tail。
3. 实现未掩码盲相关统计量，并记录掩码与有效能量。
4. 增加 CPU 性质测试；默认 pytest 不加载真实大模型。

### 构建单元 D：几何链解耦

1. 保留真实 Q/K 和恢复算子。
2. 将 `optimize_attention_geometry_update` 改为直接 Q/K 梯度与独立预算。
3. 删除 `JacobianNullSpaceResult` 输入和 JVP 复验。
4. 将搜索实现冻结为规范 token 抽样、142个粗候选、3轮局部细化、`J(T)` 最大化、并列取最早和 output-to-input 图像恢复。
5. 将检测编排改为“raw -> 近阈值 -> geometry search -> reliable -> 回正 -> 同阈值”。
6. 验证最终图像 registered/wrong-key 归因、捕获域内恢复、域外失败关闭和真实回正重编码。

### 构建单元 E：方法身份和 runtime 原子切换

1. 按第4.1节逐项迁移 CRITICAL 方法定义的9个直接依赖。
2. 先迁移解析器、schema、只读消费者和测试，再切换主 runtime、消融、资格化和打包器。
3. 将配置切换为单注入、三分支和新方法身份。
4. 清除所有正式旧字段和可选择的兼容方法入口。

### 构建单元 F：实验与证据迁移

1. 只有 `document_ecosystem_synchronized` 与目标核心 runtime 的 CPU 一致性门禁均闭合后，才能进入 `experiment_protocol_validation` 并迁移正式实验协议；在此之前不得生产正式证据或论文结果。
2. 迁移 `method_role`、三类样本、评分密钥身份、攻击证据职责、完整单样本来源身份和三组 calibration 负观测；单样本记录显式绑定 Prompt/生成输入、生成与攻击随机 seed、PRG、模型 revision、运行组件、方法定义、运行配置和路由 reference registry。以 Prompt 为单位执行1/3窗口拟合、2/3阈值冻结，并只对7项核心攻击分别约束 clean、attacked、wrong-key 的 fixed-FPR 预算；补充攻击不得进入拟合。
3. 实现 `FormalEvaluationSuccess | FormalEvaluationFailure` 判别联合 schema：冻结稳定 `failure_boundary/failure_code`，失败前可得事实照实保存、不可得字段为 `null`、`evidence_positive=false`；单 repeat 与5-repeat 聚合器必须保留失败记录在对应 detection/FPR 正式分母，禁止按缺失值删除或用 NaN、0、随机值、placeholder 插补。
4. 迁移最小6个正式方法/消融角色和单模型小规模参数敏感性。
5. 迁移单 repeat、5-repeat 聚合、核心结果包、补充描述性报告和 release profile。
6. 核验 `probe_paper`、`pilot_paper`、`full_paper` 的 schema、7项核心攻击和决策规则同构，并核验 full 或10项补充攻击缺失均不阻断 pilot 主投稿 gate。
7. 核验缓存只复用角色/密钥/profile 无关原子，Prompt-repeat 恢复不选择样本，样本级多 GPU 聚合无遗漏、重复或额外身份。
8. 执行分层 import audit 和三种 release profile dry-run，要求禁止依赖为零且 `missing_paths` 为空。

### 构建单元 G：真实 GPU 闭环

1. 在隔离参数划分上运行小规模真实 GPU reference 物化，生成 `g_ref/r_ref/q_ref` registry；报告保持 `supports_paper_claim=false`。
2. 从包含该 registry 的新 clean detached 提交运行单 Prompt GPU 方法资格化。
3. 运行单模型内部敏感性，确认名义参数附近没有明显不稳定，但不得自动回写正式参数。
4. 单攻击几何恢复和救回闭环。
5. 单 repeat 7项核心攻击共同矩阵闭环。
6. 5-repeat `probe_paper`；通过后只能证明7项核心攻击、FPR=0.1 的同构全流程及其统计结论。
7. 5-repeat `pilot_paper`；作为主投稿证据独立闭合7项核心攻击、FPR=0.01 下的四项主张和结果包。
8. 核心 pilot 闭合后，资源允许时由完整方法在冻结决策器下运行10项补充攻击并形成独立描述性报告；缺失不阻断 pilot，部分结果不得冒充完整补充集合。
9. 资源允许时再运行可选 `full_paper`；其7项核心攻击和 FPR=0.001 结论独立闭合，缺失不回退或否定 pilot 证据。

---

## 10. 状态提升规则

| 状态 | 必须满足的证据 |
|---|---|
| `core_documents_frozen` | 三份人类可读中心文档内部职责排他、公式唯一、接口/schema 闭合，并由独立智能体按固定标准确认定向审计无阻断项；后续机器协议迁移不属于该状态的前置条件 |
| `document_ecosystem_synchronized` | `core_documents_frozen` 后，其余文档、契约、机器协议、构建清单、结果包与 release profile 完成单向同步，冗余文档处理完成，默认 pytest、harness 和 diff 检查通过；它不是核心 runtime 构建前置条件，但必须在 `experiment_protocol_validation` 和正式证据生产前完成 |
| `cpu_conformant` | 目标公式、接口、配置和禁止路径均有 CPU 性质测试；旧正式入口不可选 |
| `routing_references_materialized` | 隔离参数划分在真实 GPU 上生成受摘要绑定的 `g_ref/r_ref/q_ref` registry，且不支持论文主张 |
| `gpu_operator_qualified` | clean detached 提交产生真实单 Prompt GPU 报告，`gpu_operator_preflight_ready=true` 且 `supports_paper_claim=false` |
| `single_repeat_closed` | 一个完整 repeat 覆盖全部登记方法组件、7项核心攻击、消融、质量和4个主表 baseline，并形成真实证据包；不要求补充攻击 |
| `probe_paper_closed` | 5重复、7项核心攻击、FPR=0.1、全部结论集合、质量、baseline、机制必要性和结果包闭合 |
| `pilot_paper_closed` | 主投稿证据：同一方法、7项核心攻击和 schema 在 FPR=0.01 下取得独立统计证据并闭合 required claims |
| `supplementary_attacks_reported` | 可选描述性证据：完整方法使用冻结核心决策器报告10项补充攻击；不进入任何 profile 的 required-claim conjunction |
| `full_paper_closed` | 可选规模扩展：同一方法、7项核心攻击和 schema 在 FPR=0.001 下取得独立统计证据；不是 pilot 闭合前置条件 |

资源预算状态必须与方法真实性状态分开。显存、运行时间或费用超限只阻断调度，不得篡改方法是否真实执行的判断。

---

## 11. 核心文档定稿条件

- [x] 两份无状态规范不含仓库历史、实现进度、迁移计划或保留/删除清单。
- [x] 本文档完整列出真实实现差距、保留项、修改项、移除项和不得误删项。
- [x] GitNexus 已重新索引到当前源代码提交，并重新确认 CRITICAL 影响；旧方法定义现场结果为9个直接依赖、82个累计影响符号和8组受影响流程。
- [x] CRITICAL 方法定义的9个直接依赖和主要文件均有明确处理方式及验收条件。
- [x] 三份文档之间没有公式、参数、职责或状态归属冲突；7项核心/10项补充攻击、6角色、固定5重复与 pilot/full 边界已经独立复审确认。
- [x] 盲检测、按需几何搜索、最终图像 Q/K 归因、reference registry 和单模型敏感性在三份文档中语义一致。
- [x] 204维手工结构描述符不再属于目标核心门禁。
- [x] 独立复审者已按固定标准确认三份核心文档不存在定稿阻断项。
- [x] 本轮只针对三份核心文档的轻量交叉校验和格式检查已重新通过。

两项原 HIGH 阻断已经由提交 `4edb1cbd70ee889b8eabc68bb34e34aaa75a9f07` 闭合：`attack_evidence_role` 进入 success/failure 共享身份与摘要，registered/wrong-key 几何差值进入 raw/success/failure 和 measurement digest。提交 `b31ffeb0ecd7a9d47aa36d2e35d6428494593429` 又登记11个目标生命周期字段、把文档约束测试固定为5重复并反向禁止固定9重复，同时将真实外部源码验证收敛到显式 integration qualification。独立复审和定向检查均无核心文档阻断，因此 `core_documents_frozen` 已完成。该状态只允许启动核心方法 runtime 构建；机器随机化 runtime 仍为3 seed × 3 key 的9重复，攻击职责 writer、目标 runtime、聚合器和正式证据均未迁移，不能支持任何论文结论。

---

## 12. 文档生态同步条件

- [x] 已以三份核心文档为唯一来源修订外围人类可读文档和根契约，并完成独立只读复审。
- [x] 完整结果包、release profile 和构建文档 inventory 已登记三份核心文档且无第二套方法定义。
- [x] 其他构建规范不再保存当前项目状态，过期或冗余文档已删除或收敛为无状态协议。
- [x] 本轮文档生态定向测试、默认 pytest、harness 和 `git diff --check` 已重新通过。
- [ ] 机器随机化、攻击职责 writer、attack/profile registry、配置摘要、聚合器和正式记录 producer 已按冻结目标协议完成原子同步。

三份核心文档和外围人类可读规范已经统一5重复、7项核心/10项补充攻击职责、pilot 主投稿/full 可选扩展以及等价执行优化边界；11个目标治理字段也已进入字段生命周期 target schema，但该登记不表示 writer 已实现。机器随机化 runtime 和对应聚合仍使用3 seed × 3 key 的9重复，攻击 registry、profile gate、质量/主张过滤器、cache/checkpoint manifest、攻击职责 writer、聚合器、配置摘要和正式记录 producer 尚未迁移，因此该状态继续记为 `protocol_documentation_updated`，不得提升为 `document_ecosystem_synchronized`。真实外部 baseline 源码不在默认 pytest 边界内；显式 qualification 在源码缺失时保持失败关闭。只有完成机器协议原子切换并重新通过相应验证后，才允许进入 `experiment_protocol_validation`、正式证据生产和论文结果生产；这不阻断已经激活但尚未实现的核心方法 runtime 构建。
