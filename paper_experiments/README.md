# paper_experiments

`paper_experiments/` 是完整论文实验层。该目录保存外部 baseline 适配、官方参考复现编排、受治理结果导入、共同协议对比和论文证据闭合代码。

该目录可以依赖 `main/` 与 `experiments/`, 但不得依赖 `paper_workflow/`。Notebook 入口只能调用这里的正式实现, 不能把正式论文结果逻辑写在 cell 中。

## 子目录职责

```text
baselines/   外部 baseline 适配、官方参考复现和受治理导入协议
runners/     完整论文实验的服务器可复用 runner
analysis/    论文表图实际数据验证、证据审计、闭合门禁与投稿就绪分析
```

## runner 职责

- `runners/closure_package_selection.py`: 按包内身份、当前论文运行层级、fixed-FPR、baseline、ready 状态和 `outputs/` 白名单选择精确输入。单 repeat 路径只选择7类活动随机化 leaf package; 3类跨 repeat 不变官方参考包由聚合路径单独选择。论文闭合编排位于外层 `scripts/paper_result_closure.py`, 本层提供输入选择、正式实验和证据分析能力。
- `runners/randomization_aggregate_provenance.py`: 精确接收权威9个单 repeat 组件与3个跨 repeat 不变 official-reference ZIP, 保存12个输入的原始字节, 写后再次调用生产 validator / inspector, 并返回深度不可变的聚合来源对象。该对象固定 `supports_paper_claim=false`, 只能作为正式统计的输入来源。
- `runners/randomization_aggregate_record_workspace.py`: 只接收上述不可变聚合来源对象, 在临时工作区内再次复验9个 repeat component、63个活动 leaf 与3个不变 leaf。该入口按固定成员登记分别暴露9个 Prompt runtime、27个 Prompt 来源字节、45个运行 manifest、45个方法 observation、45个阈值声明、18个消融、27个质量和3个官方参考记录源；每个记录源均绑定成员字节、leaf ZIP、repeat component 与 aggregate ZIP 的完整摘要链, 不返回临时文件系统路径, 也不写持久产物。
- `runners/randomization_prompt_source_contract.py`: 从每个主方法 leaf 内嵌的来源注册表、7000条冻结选择清单和当前运行层级 Prompt 文件逐字节重建唯一 Prompt 契约, 并要求9个 repeat 的三份来源身份完全一致。该模块不读取当前仓库的 Prompt 文件, 因而无法用9份彼此一致但未登记的 runtime 文本替换受治理语料。
- `runners/randomization_method_repeat_thresholds.py`: 只接收同一个不可变聚合来源对象, 将9个 runtime records 逐条绑定到包内 Prompt 契约, 重新核对精确 runtime schema、完整正式方法配置、split、模型、生成 seed 和运行前预注册的3-key 计划, 在 CPU 上按版本化 PRG 重新生成3组基础 latent；同时核验4个 baseline 顶层运行 manifest 保存的同一完整9重复计划, 再把45组 observation 与 producer 阈值声明交给纯分析层逐一重算。该桥接入口不接受路径、Prompt 或成员覆盖, 返回的重建报告固定 `supports_paper_claim=false`。
- `runners/paper_claim_provenance.py`: 正式论文结论 Writer 的统一来源边界。各 Writer 完成不可变 aggregate 来源对象绑定与派生字段传播前, 公开入口均在读取历史输入或创建输出前失败即关闭; 单 repeat 组件或手工 ready 字段不能恢复正式结论。
- 主方法正式输入包来自 `image_only_dataset_runtime`、`runtime_rerun_ablation` 和 `dataset_level_quality`; attention capture、latent injection 或 aligned rescoring 诊断包不能替代这些输入。
- `runners/external_baseline_method_faithful.py`: 单个 Tree-Ring、Gaussian Shading 或 Shallow Diffuse 的 SD3.5 common-backbone 适配调度、关键算子数值忠实度复验、逐 Prompt 原子完成单元复算、fixed-FPR 审计、transfer manifest 和独占结果打包逻辑。断点恢复发生在真实 adapter 内部; runner 保留完成单元并清除可重建派生文件, 然后再次验证 test Prompt 攻击 exact set、数值忠实度报告与跨会话科学来源。
- `runners/isolated_scientific_workflow.py`: method-faithful 与 T2SMark 共用的父子进程调度. CPU 父进程分别选择 `sd35_method_runtime_gpu` 或 `t2smark_sd35_gpu`, 科学子解释器调用完整 runner 并写出唯一结果 envelope; 父进程验证 envelope、execution report 和依赖报告后才返回科学 runner 的 summary.
- `runners/persistent_workflow_session.py`: 三条 method-faithful、T2SMark 和三条 official-reference 共用的持久化会话.该模块把论文运行配置、科学 profile 摘要与完整哈希锁、正式执行 commit 和影响结果的环境配置绑定为唯一恢复身份, 周期性把已稳定落盘的普通文件镜像到挂载盘, 并通过 `experiments.runtime.resume_checkpoint` 发布完成单元.恢复先验证全部 manifest、路径和 SHA-256, 通过后再清理当前路由旧文件并逐文件原子发布.
- `runners/t2smark_source_runtime.py`: 按来源登记 commit 准备 T2SMark 源码并应用摘要固定的正式协议 Git diff。
- `runners/t2smark_formal_reproduction.py`: T2SMark 官方 SD3.5 路径复现、逐 Prompt 原子单元恢复、Prompt split 导出、固定 FPR 候选记录生成、受治理导入校验和结果打包逻辑。归档前从单元重新生成官方结果、adapter observation、候选记录与 validation report, 并拒绝工作区路径和额外目录进入证据包。
- `runners/tree_ring_official_reference.py`: Tree-Ring 官方参考复现、10-Prompt 原子批次恢复、官方依赖环境准备、模型仓库布局核验、受治理导入记录和结果打包逻辑。
- `runners/gaussian_shading_official_reference.py`: Gaussian Shading 官方参考复现、10-Prompt 原子批次恢复、官方依赖环境准备、严格官方依赖优先策略、受治理导入记录和结果打包逻辑。
- `runners/shallow_diffuse_official_reference.py`: Shallow Diffuse 官方参考复现、10-Prompt 原子批次恢复、官方依赖环境准备、源码运行边界核验、受治理导入记录和结果打包逻辑。
- `analysis/paired_superiority.py`: 对 SLM-WM 与4个主表 baseline 的完整 test Prompt x attack 观测执行精确配对, 以 Prompt 为聚类单位分别计算全样本优势统计和检测标签无关的质量匹配优势统计。
- `analysis/method_repeat_fixed_fpr.py`: 以单个方法和单个登记重复为唯一阈值计算单元, 从原始 clean-negative observation 独立重算精确45个 fixed-FPR 阈值。该模块同时拒绝 Prompt exact-set、模型 revision、seed、key、基础 latent、规范攻击 seed、producer 阈值声明或任一层摘要不一致的输入；返回的阈值记录、公平身份记录和汇总报告固定 `supports_paper_claim=false`, 只能作为后续跨重复统计的受治理事实。
- `analysis/paper_artifact_data_validation.py`: 实际读取11类正式表图源文件, 以仅图像盲检原始 JSONL 和冻结协议独立重建分数分布、ROC 与 DET, 再精确核验表格列序、行序、单元格、曲线端点和跨摘要 readiness 一致性。
- `analysis/formal_record_statistics.py`: 最终闭合不信任机制必要性 summary、必要性 CSV 或 FID/KID 指标表。该模块从逐 Prompt 正式消融重运行记录重新执行配对 bootstrap、Holm 校正和质量非劣门禁, 并独立复算消融攻击 seed；同时从逐图像正式 Inception feature records 以 float64 独立重算 `fid`、`kid_mean`、`kid_std` 三行, 随后逐字段比较全部派生行。全样本子集的 KID std 必须精确为0, 大样本 std 按100个冻结子集估计值的总体标准差重建。未获支持的单机制结论仍可作为真实负结果进入协议闭合, 但任何原始记录、派生统计或指标字段漂移都会 fail-closed。

外部 baseline 命令计划构建、执行和证据校验的可复用实现分别位于 `baselines/command_plan_builder.py`、`baselines/command_plan_execution.py` 与 `baselines/evidence_validation_cli.py`。`scripts/` 中的同名职责命令只提供可执行入口, 本层不依赖外层脚本。

## official-reference 忠实度证据

- `baselines/official_reference_fidelity_evidence.py`: 只读取当前论文运行下已物化的 Tree-Ring、Gaussian Shading 和 Shallow Diffuse 三个 official-reference family, 独立复算 package input 的精确文件集合与逐文件摘要, 并核验运行决策、受治理导入、validation、归档治理语义和共同 clean `code_version`。
- 输出记录固定为精确三个方法身份, 只支持补充方法忠实度证据。该证据明确设置 `main_table_eligible=false` 和 `supports_main_table_superiority_claim=false`, 不替代 common-backbone 主表比较, 也不扩展到其他 supplemental 方法。
- 聚合 CPU 结果闭合 DAG 在精确9个 repeat 组件和3个跨 repeat 不变 official-reference 包全部通过独立复验后执行该审计。语义门禁核验 `official_reference_exact_set_ready`、`common_code_version_ready`、`official_reference_fidelity_ready_count` 和 `official_reference_fidelity_evidence_ready`。

## 主表 baseline 数值忠实度

- `baselines/method_faithful_numerical_fidelity.py` 从来源登记的完整 Git commit 读取不可变源码 blob, 只编译已登记的关键定义, 并在确定性 CPU Tensor 上比较官方算子与 SD3.5 common-backbone adapter 的输出。报告绑定官方 commit、源码 blob、抽取定义 AST、adapter 文件、逐算子输入输出摘要、误差和容差, 已声明的 ready 字段不能替代数值语义重算。
- Tree-Ring 直接比较 mask、ring key、Fourier 注入和检测距离; Shallow Diffuse 直接比较 ring mask、complex random patch、Fourier 注入、检测距离和编辑时刻取整。Gaussian Shading 以官方源码中的 ChaCha20 调用契约绑定 RFC8439 known-answer test, 并比较 block voting 与条件 Gaussian 符号映射; 该路径只声明其实际覆盖的源码契约与算子等价性, 不把未执行的完整官方类伪装为逐指令等价。
- Tree-Ring、Gaussian Shading 和 Shallow Diffuse 的数值报告是各自 method-faithful 归档的必需成员。T2SMark 使用 native official result 的精确重建作为对应数值忠实度模式。四个 baseline 必须全部通过, `primary_baseline_numerical_fidelity_ready` 才能成立; 数值忠实度报告本身固定 `supports_paper_claim=false`, 只能作为公平实现资格证据。

六个 common-backbone / official-reference family 与 T2SMark 均把正式运行写入 `outputs/<artifact>/<paper_run_name>/...`。主方法、质量、消融、三个 common-backbone baseline 与 T2SMark 共7类随机化证据包, 必须由 package input 和 archive 或运行 manifest 两个独立来源共同绑定活动 `randomization_repeat_id`、seed/key 索引和随机化协议摘要。Tree-Ring、Gaussian Shading 与 Shallow Diffuse 的3个 official-reference 原环境包只承担跨 repeat 不变的数值忠实度证据, 不得伪装成随机化主表结果。归档名称包含方法身份和唯一时间后缀, 包内身份还必须绑定同一 `paper_run_name`、`target_fpr`、`baseline_id` 和 `code_version`。归档白名单只允许当前 family 的 `outputs/` 成员, 不把 runner 源码、README、source registry 或 Notebook runtime 报告复制进正式输入包。

单 repeat 输入选择与自包含证据包固定 `repeat_component_ready=true`、`randomization_aggregate_ready=false` 和 `supports_paper_claim=false`。该组件只证明当前 seed-key 单元的7类随机化包没有跨 repeat 混选; 3类不变忠实度包不复制进 component, 由聚合层独立选择一次。最终论文闭合必须精确覆盖权威9 ID 并从原始观测与特征重算统计。

method-faithful 与 T2SMark 的归档白名单还必须包含 `scientific_execution_binding.json`、workflow 内的隔离执行报告、依赖环境报告快照、命令调度报告和子进程结果 envelope. 独立 binding 以 SHA-256 绑定科学 runner 输出的 summary 与 manifest, 同时记录 profile、完整依赖锁和正式执行锁身份; 父层不事后改写科学 runner 已完成的 summary 或 manifest. Drive 下载后的 CPU 闭合只复核包内快照和执行时双向重验证标志, 不依赖 Colab 临时 venv 继续存在.

## 外部 GPU workflow 持久化边界

- 7条外部 GPU 路径使用同一个 route registry、同一个定时 generation 格式和同一个完成单元恢复原语, 不在各 official runner 内复制 Drive 逻辑.
- 定时 generation 使用不可变目录和原子 `current_checkpoint.json` 指针.每个文件记录仓库相对路径、大小与 SHA-256; 复制期间仍在变化的文件不会进入该 generation.
- 完成单元只有在 runner 返回摘要与持久化摘要一致、路由真实 ready 字段通过、论文运行层级和 baseline 身份匹配、运行 manifest 绑定当前正式执行锁后才会发布. method-faithful 与 T2SMark 还会重新验证 `scientific_execution_binding.json`.
- 所有 checkpoint 均固定 `supports_paper_claim=false`.checkpoint 只减少断线后的重复计算; 正式 records、ZIP、package input manifest、archive manifest 和 CPU 证据闭合仍是唯一论文证据路径.
- 上游科学工具已经发布的完整文件或完整科学单元可以恢复.某个上游命令尚未发布结构化完成记录时, 恢复后会重放该未完成单元, 不声称任意 Python 指令、GPU kernel 或半写图像可以原地续跑.

主方法、method-faithful、T2SMark 与三套 official-reference 的完整科学 runner 均由五个登记 CUDA profile 的隔离子解释器执行。六个 profile 均具有已登记的完整哈希锁; 锁文件通过静态 schema、直接依赖覆盖和摘要门禁只表示安装输入闭合。当前本地 Windows / CPU 环境不具备 Linux 或 CUDA 执行条件, 因而真实安装、`pip check`、torch/CUDA identity、硬件 smoke 和科学结果仍必须在后续 Colab GPU 会话中完成后才能产生正式输入包。

## 主表总体优势统计

- 主表比较只使用 SLM-WM 与 Tree-Ring、Gaussian Shading、Shallow Diffuse、T2SMark 四个正式 baseline。每个 baseline 必须与 SLM-WM 在完全相同的 test Prompt、攻击条件、生成 seed、密钥重复、基础 latent Tensor 和规范攻击 seed 上形成一一配对；配对记录必须同时绑定重复 ID、seed/key 索引、实际生成 seed、基础 latent 内容摘要和联合身份摘要。重复、缺失、额外配对键或任一随机身份不一致都会阻断统计。
- 正式随机化使用3个生成 seed 与3个水印密钥的9个交叉重复。单个 GPU 运行只物化一个活动重复并写入独占 Drive 子目录；最终论文比较必须在全部9个重复完成后按 Prompt、生成 seed 和密钥重复组织统计。任何单重复结果只能说明该登记重复下的运行事实, 不能支持跨种子或跨密钥稳定性主张。
- 单个 Prompt 的效应值是该 Prompt 跨完整攻击集合的平均二元检测差值。正式统计使用100000次 Prompt-clustered bootstrap 的95% percentile CI, 对取值位于 [-1,1] 的 Prompt 聚类差执行单侧 bounded Hoeffding 均值检验, 再对4个主表 claim p 值执行 Holm 校正。整数动态规划 exact sign-flip 只披露 sharp-null 诊断, 不参与均值优势 claim 门禁。
- 质量匹配使用未攻击 clean-watermarked pair 的实测 `embedding_pair_ssim`, 并绑定相同 Prompt、生成 seed、水印密钥与基础 latent 身份。主方法与当前 baseline 的绝对 SSIM 差不超过0.02时, 该 Prompt 进入当前 baseline 的质量匹配子集; 每个 baseline 至少需要覆盖80%的 test Prompt。不同 baseline 可以形成不同的匹配 Prompt 子集, 但每个入选 Prompt 必须覆盖完整攻击集合。
- 质量子集选择不读取主方法或 baseline 的检测标签。闭合路径会在各 baseline 的匹配子集上独立重算 Prompt-clustered bootstrap、bounded Hoeffding 检验和按预注册4次比较执行的 Holm 校正。单个 baseline 必须同时通过全样本与质量匹配优势门禁才允许其表格行支持 claim; 四个 baseline 的两类门禁全部通过后, 总体主表优势结论才可成立。
- 配对 outcome 逐条绑定规范 test Prompt 身份、正式攻击 ID、资源档位、攻击配置摘要、两方法冻结阈值摘要和原始 observation 文件字节摘要。结果闭合门禁会读取五方法原始 observation 独立重建全部 outcome、样本计数、TPR、clean/attacked FPR、质量均值、分数保持诊断和 Hoeffding 区间, 再精确核验正式 result records、来源文件 SHA-256、模板覆盖、schema validation 与 manifest 配置摘要。
- 单个 baseline 只有在总体平均差值大于0、bootstrap CI 下界大于0且 Holm 校正后 `p < 0.05` 时才满足 `paired_superiority_ready`。4个 baseline 必须全部满足该条件, `overall_paired_superiority_ready` 才能支持主表总体 superiority claim。
- 上述统计结论严格限定于当前受治理 Prompt benchmark。将结论外推到未采样自然 Prompt 总体还需要 Prompt 聚类独立或可交换假设及相应数据来源证明, 当前门禁不据此声明无条件总体优势。
- 逐攻击比较表负责完整披露每个攻击条件的效应与保守 CI, 不要求每个攻击均显著胜出。`universal_per_attack_superiority_claim_ready` 只控制“所有攻击均显著胜出”的更强主张, 不替代 Prompt 聚类配对的总体优势门禁。
- 三个 official-reference family 只承担补充方法忠实度证据, 明确设置 `main_table_eligible=false` 与 `supports_main_table_superiority_claim=false`, 不进入上述主表配对统计。

## CPU 结果闭合 DAG

每个 repeat 由 `randomization_repeat_evidence.py` 精确选择7类活动随机化 leaf ZIP, 保持 leaf 原始 ZIP 字节并写入独立的自包含 evidence package。3类跨 repeat 不变的 official-reference 忠实度包不复制进每个 repeat, 只在聚合层选择一次。单 repeat package 的 manifest 绑定 seed/key 身份、随机化协议、代码版本、执行锁摘要与7个 leaf SHA-256, 并固定 `repeat_component_ready=true`、`randomization_aggregate_ready=false`、`supports_paper_claim=false`。

聚合 package 的 manifest 以 `{aggregate_package_path}` 明确标记调用方必须提供的当前来源 ZIP, 并登记 `repository_root` 工作目录与 `self_contained_aggregate_zip` 输入模式。层内重建入口先完整验证14个成员, 再只把登记的9个 repeat component 与3个 invariant ZIP 提取到临时目录, 最后重新调用生产 Writer；重建命令不假定内部成员路径已经存在于宿主文件系统。

最终 aggregate DAG 必须在精确9个 component 全部存在后执行 official-reference 忠实度审计、攻击矩阵、45个 method-repeat fixed-FPR 阈值审计、Prompt 聚类配对优势、主表证据、正式结果 records、共同协议、结果分析、论文表图实际数据审计、投稿就绪审计、证据闭合入口审计、结果闭合语义门禁和完整结果包打包。任一 repeat 缺失、重复、身份漂移或原始记录重算失败都必须阻断后续闭合。

结果分析 payload 采用固定角色集合: 主置信区间表、逐攻击优势表、失败案例记录和失败案例图。结果分析 summary 保存角色路径、逐文件 SHA-256 与组合摘要, manifest metadata 必须传播同一绑定并由 config digest 锁定。结果闭合门禁重新读取实际文件, 并与主比较表、攻击表和质量表一起纳入 `closure_source_file_sha256`; 完整包生成前会再次读取并复算全部字节摘要, 因而删除、替换或篡改任一表图都会 fail-closed。

## 结果口径

- SLM-WM 主方法表使用“空间低通 LF”“高斯幅值尾部截断”和“Q/K attention geometry”三个机制名称。正式论文消融使用 Tail-only、No-Tail 和 No-Tail-Truncation。
- method-faithful 结果用于主表 common-backbone 对比: 同一 prompt split、同一 SD3.5 主线、同一攻击簇、同一 fixed-FPR 协议。
- Tree-Ring、Gaussian Shading 和 Shallow Diffuse 的 official reference 结果用于补充表和方法忠实度审计。T2SMark 原生 SD3.5 formal 路径直接生成主表候选, 但同样必须通过完整攻击模板、fixed-FPR 与 受治理导入 门禁。
- `probe_paper`、`pilot_paper` 和 `full_paper` 都是正式结果包, 分别支持 `probe_claim`、`pilot_claim` 和 `full_claim`。三者只允许样本数量和 fixed-FPR 标准不同。
- proxy、placeholder、fallback、synthetic 和 formal-null 证据不得进入共同协议 claim-ready 统计。
- 论文表图审计实际读取仅图像盲检原始记录、冻结协议、test 指标、记录级连续分数分布、完整 ROC / DET sweep、攻击指标、主表 baseline、正式消融和 FID / KID 表。分数分布与曲线必须由原始记录精确重建并逐单元格一致, readiness flag、路径字符串或仅满足单调性的自造曲线不能替代该核验。
- `analysis/evidence_closure_entry_review.py`: 联合投稿就绪、待补证据、claim 阻断、主表 baseline 和数据集质量等受治理输入逐项执行确定性入口判定。只有全部检查项通过时才输出 `entry_review_decision=ready_for_evidence_closure` 与 `evidence_closure_allowed=true`; 任一缺口都会自动阻断, 不使用人工批准状态。

Colab 入口通过 `paper_workflow/colab_utils/` 增加进度显示、Drive 路径和 Notebook runtime 报告; 正式协议逻辑仍以本目录与核心方法复现层为准。
