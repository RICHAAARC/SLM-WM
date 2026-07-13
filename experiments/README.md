# Experiments

`experiments/` 是不依赖 Notebook 的实验实现层, 依赖方向为 `experiments -> main`。

## 当前职责

- `protocol/`: 论文级 Prompt 分层、fixed-FPR、共同攻击、正式证据和运行规模配置。
- `protocol/prompt_sources.py`: 冻结 COCO 与 PartiPrompts 来源身份、7000条 6:1 交织选择清单和70/700/7000三级严格前缀；`audit_packaged_prompt_set_bytes` 可只凭结果包内嵌的来源注册表、选择清单和 Prompt 原始字节重建当前运行集合, 不依赖仓库路径。
- `runtime/`: SD3.5 模型加载、真实图像攻击、语义特征、续跑、仓库环境与正式依赖环境实现. `dependency_profiles.py` 解析一个 CPU 父编排 profile 与五个 CUDA 科学 profile, 并核验完整锁中的全部直接和传递包; `dependency_preparation.py` 执行当前解释器的 hash-locked 安装与适用的兼容性检查; `isolated_dependency_environment.py` 使用固定且经 distribution `RECORD` 验证的 `uv` 为五个科学 profile 创建独立 CPython 子环境; `isolated_scientific_execution.py` 在执行前后复验子解释器、依赖报告与正式执行锁; `semantic_watermark_scientific_session.py` 在单个主方法科学子解释器中调度运行与绑定打包两种互斥模式; `scientific_execution_binding.py` 保存可脱离临时 venv 审计的产物内证据快照; `resume_checkpoint.py` 提供不依赖 Colab 或 Drive API 的通用原子 checkpoint 发布与恢复协议; `archive_naming.py` 只提供 UTC 时间和短提交身份原语, 不保存 Notebook、baseline 或 official-reference workflow 词表.
- `runners/semantic_watermark_runtime.py`: 完整执行分支风险、716维完整语义与视觉条件 Jacobian、20候选方向的4维 Null Space、LF、Gaussian 幅值尾部截断、真实多层 Q/K 几何更新、同种子 carrier-only 总机制效应反事实、逐注入无 attention 原子持久化、三边最终完整特征保持门禁、真实 Q/K 双归因增益门禁和仅图像盲检；每次注入同时持久化完整风险/预算/mask、候选/响应/基底、三分支更新及 Q/K 输入输出的版本化内容摘要。持久化运行配置删除密钥原文字段, 只写入后缀合规的 `key_material_digest_random`。
- `protocol/formal_randomization.py`: 冻结3个生成种子与3个水印密钥的9个交叉重复, 在任何运行前预注册3个 key seed 与 key material 摘要计划, 从公开生成 seed 构造设备无关规范基础 latent, 并为主方法与全部主表 baseline 生成可逐字段核验的随机化身份。正式入口拒绝不能派生预注册 key plan 的环境根密钥；运行记录同时保存协议字段和随机摘要, 科学来源的随机字段容器只接收后缀合规的随机部分。
- `runners/image_only_dataset_workload.py`: 从唯一方法 YAML 和论文运行配置构造完整主方法工作负载, 执行主运行与正式 Inception FID / KID, 并根据科学 session 的打包边界返回续跑或完成状态。`scripts/run_image_only_dataset_runtime.py` 只转发到该模块。
- `runners/image_only_dataset_runtime.py`: 运行完整 Prompt 集, 在 calibration split 独立冻结检测协议, 只在 test split 形成论文统计; 同时从每条仅图像检测记录的真实连续分数生成分数分布、ROC 与 DET 数据。
- `ablations/mechanism_ablation_workload.py`: 构造当前论文规模的全部消融配置, 调用正式重运行消融并执行受治理打包。`scripts/run_runtime_rerun_ablations.py` 只转发到该模块。
- `ablations/runtime_rerun.py`: 对完整方法与14个消融配置分别重新生成、攻击、检测并独立校准, 包括中心化 Q/K logit、可微行内 rank、关系概率和距离调制中心化概率四个逐项留一变体；不读取或变换完整方法分数。
- `artifacts/dataset_level_quality_outputs.py`: 从真实 clean/watermarked 图像对提取正式 Inception 特征, 并构建 `fid`、`kid_mean`、`kid_std` 三行质量证据。KID 在 canonical feature population 上执行100轮均匀无放回子集估计, std 表示子集估计值的总体标准差而不是标准误。
- `artifacts/detection_score_curves.py`: 将内容主判与冻结几何救回转换为判定等价连续分数, 使用 `positive_source` 与 clean negative / wrong-key negative 的记录级真实标签, 对 test overall 与每个同时含正负样本的攻击条件枚举正负无穷端点和全部唯一观测分数, 输出可复用的完整 threshold sweep。
- `artifacts/`: 保存通用 manifest schema、连续检测统计与正式质量产物构建器。
- 正式攻击记录必须由运行端直接写入 `attack_id`、`attack_family`、`attack_name`、`resource_profile`、`attack_config_digest` 与 `attack_parameters`; 攻击矩阵只验证并传播该身份, 不根据名称后贴配置摘要。

主方法、method-faithful baseline、T2SMark 与三套 official-reference runner 均通过登记的隔离子解释器执行. 当前代码路径不保留父解释器直接执行科学实现的分支. 当前唯一外部阻断类别是六个目标完整哈希锁尚未在匹配环境完成资格审查, 其中五个 CUDA profile 必须在对应 Colab 或 Linux CUDA 环境完成审查.

三个主方法 GPU 上游 family 均使用 `outputs/<artifact>/<paper_run_name>/...`:

- `outputs/image_only_dataset_runtime/<paper_run_name>/`
- `outputs/formal_mechanism_ablation/<paper_run_name>/`
- `outputs/dataset_level_quality/<paper_run_name>/`

Colab 主方法在 `/content` 本地磁盘执行 clean detached 工作树、科学子解释器和全部 `outputs/` 写入。外层若把 `SLM_WM_RESUME_CHECKPOINT_DIR` 指向 Drive 或服务器持久磁盘, 本层只同步已完成科学单元、正式消融运行、Inception 特征 batch 和进度记录; 未配置该变量时所有 checkpoint 调用保持无操作。每个快照绑定正式执行锁、仓库相对路径、文件大小与 SHA-256, 先验证临时副本, 后原子发布 manifest; 恢复时先完整验证全部 manifest 和 payload, 再写回本地 `outputs/`。

checkpoint 是续跑中间状态, 固定使用 `evidence_eligibility=intermediate_state_only` 与 `supports_paper_claim=false`。progress 文件不能替代 `manifest.local.json`, 也不能进入完成单元或论文统计。正式 Inception feature shard 还绑定精确图像角色、实际图像 SHA-256、提取器身份、2048维有限数值与正式执行锁; 任一身份漂移、冲突 shard 或摘要损坏都会阻断复用。论文证据资格仍只由完整运行 manifest、正式归档与闭合门禁决定。

summary 必须同时记录带时区的 `generated_at`、`paper_run_name`、`target_fpr` 与唯一的 `randomization_repeat_identity` 对象。主方法运行、正式消融和数据集质量 package input 使用同一 schema v2 冻结 repeat ID、seed/key 索引和协议摘要; selector 还会与运行 summary、manifest 中的独立来源交叉核验。单 repeat summary 使用 `full_method_component_ready`、`ablation_component_ready` 或 `formal_fid_kid_component_ready` 表示可聚合科学事实, 正式消融摘要文件为 `ablation_component_summary.json`, 并固定 `randomization_aggregate_ready=false` 与 `supports_paper_claim=false`。ZIP 只包含所属 run-scoped `outputs/` family, 不收集仓库源码或其他运行层级文件。

主方法运行包必须包含 `score_distribution_table.csv`、`roc_curve_points.csv` 和 `det_curve_points.csv`。每个分数分布行绑定 observation、真实标签、连续分数、冻结阈值与混淆矩阵; ROC / DET 共享同一完整 threshold sweep, 不得由 `test_detection_metrics.csv` 中的单一 operating point 代替。

## 正式方法边界

内容载体为 `lf_content` 与 `tail_robust`。`tail_robust` 仅按 Gaussian 元素绝对幅值执行分位点尾部截断, 不具有空间频带含义。注意力稳定度来自至少两个真实 Q/K 层对应关系行的余弦一致性。嵌入与盲检共享稳定 token pair 构造规则, 但数据依赖身份分别在一次注入内部和一次盲检的 raw、registration、aligned 路径内部冻结, 不执行跨端身份比较。注册只使用与攻击配置无关的有界搜索和递减分辨率局部优化。最终 clean、carrier-only 与完整方法成图必须在 CUDA 上重新编码真实 Q/K。carrier-only 与完整方法共享 seed、scheduler、LF/tail 配置和算子, 仅 attention geometry 开关不同；首个注入前 latent 必须字节级相同, 之后允许 attention 介入造成 LF、tail 与轨迹的下游交互, 因而该比较是 attention 开关总机制效应而非纯直接效应。carrier-only 的每个更新原子必须明确没有 attention 分数、更新、关系、pair 身份和 attention Null Space, 并把 JSONL 路径、文件 SHA-256、内容摘要、反事实身份与图像 SHA-256 绑定到结果、manifest 和缓存复验。三张成图的三条完整 CLIP/视觉特征边全部通过后, 才比较自身盲选择归因增益与冻结 carrier-only pair 权重归因增益。clean 只保留为总体水印对照。

科学内容门禁同时使用 `slm_wm_tensor_content_v1` 和 `slm_wm_image_rgb_uint8_content_v1`。单次运行写入端先持久化更新 JSONL、检测 JSONL 和图像, 再从磁盘重读这些文件构造总证据；完成结果加载器同样从磁盘重算并要求记录完全相等。联合身份覆盖风险来源与预算、Null Space 八类 Tensor、实际量化写回的完整716维参考特征和 JVP Tensor、三分支更新、注入五角色 Q/K、最终三图的规范 RGB uint8 像素及 Q/K、检测 raw/aligned Q/K、公开检测噪声和 carrier-only 反事实。最终三图 Q/K 逐角色持久化公开噪声 Tensor、PRG 身份和索引0、1、2, 像素-Q/K binding 也包含噪声身份；检测必须共享该身份并从3开始连续。alignment 摘要必须规范。attention 开启时, 写后和打包前都使用同一 carrier-only validator；该 validator 同时消费完整配置或持久化脱敏配置, 并拒绝任何残留 attention 字段。数据集汇总先重算每个内嵌绑定记录；每个单元 manifest 必须自包含且无重复路径, 其配置和配置摘要必须匹配结果内脱敏配置。数据集 manifest 必须无重复并覆盖全部互斥单元叶子, 打包器拒绝仅摘要结果、遗漏叶子及叶子篡改。SHA-256 只证明包内内容一致性, 不能替代模型来源锁、真实 CUDA 运行或论文统计证据。

四分量留一变体在生成端和检测端共用同一归一化权重。被移除分量的权重严格为0, 其余三个分量各为 $1/3$；该权重同时进入梯度、回溯、最终归因、注册和同步评分。每个变体继续执行完整 Prompt、攻击和独立 calibration 流程, 因而属于真实方法重运行而不是汇总层算术消融。

主方法检测协议在 calibration clean negative 上联合冻结内容阈值、几何可靠性阈值与同阈值救回规则, 随后只在独立 test split 应用该冻结协议。正式 FPR 证据是 test clean negative 的经验 operating point 及95%单侧 Wilson 上界; 该协议不声明 calibration 导出的 split-conformal 有限样本总体 FPR 保证。

所有持久化输出必须写入 `outputs/`。本层不得导入 `paper_experiments/`、`scripts/` 或 `paper_workflow/`。
