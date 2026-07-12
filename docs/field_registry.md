# Field Registry

## 文档定位

本文档是项目中 governed fields 的登记表。它只记录“当前项目实际使用或模板预留的字段实例”, 不重复解释字段治理规则。

字段 category、后缀要求和清理规则见:

```text
docs/placeholder_random_governance.md
docs/intermediate_state_governance.md
docs/artifact_rebuild.md
```

## 何时需要登记

新增字段只要进入下列任一位置, 就应先登记到本表:

```text
配置文件
records
manifests
tables
reports
Python dict key
测试 fixture
Markdown 示例
Notebook 与 repository module 的跨边界数据
```

函数内部一次性局部变量不需要登记。跨函数、跨文件、跨进程或跨 Notebook 边界保存的中间状态字段需要登记。

## 字段登记表

| field_name | category | required_suffix | allowed_in_records | allowed_in_claims | replacement_required | description |
| --- | --- | --- | --- | --- | --- | --- |
| project_unit | governance | none | true | false | false | 当前项目语义阶段。 |
| target_construction_unit | governance | none | true | false | false | 当前构建目标。 |
| run_id | protocol | none | true | false | false | 一次运行的稳定标识。 |
| record_id | protocol | none | true | false | false | 单条记录的稳定标识。 |
| split | protocol | none | true | false | false | 数据或事件划分。 |
| method_name | protocol | none | true | false | false | 实验记录中的方法名称。 |
| metric_name | protocol | none | true | false | false | 实验记录中的指标名称。 |
| metric_value | protocol | none | true | false | false | 实验记录中的指标数值。 |
| metadata | governance | none | true | false | false | records、manifest 或 typed object 的补充结构化信息。 |
| artifact_id | artifact | none | false | false | false | 受治理论文产物的稳定标识。 |
| artifact_type | artifact | none | false | false | false | 受治理论文产物类型, 例如 table、figure、report 或 manifest。 |
| input_paths | artifact | none | false | false | false | 产物重建所需输入路径。 |
| output_paths | artifact | none | false | false | false | 产物重建生成输出路径。 |
| config_digest | artifact | none | false | false | false | 产物重建配置摘要。 |
| code_version | artifact | none | false | false | false | 产物重建所用的精确40位小写 Git 提交 SHA; dirty 状态不得进入正式证据。 |
| formal_execution_lock_schema | protocol | none | false | false | false | clean detached Git 正式执行锁使用的稳定 schema 名称。 |
| formal_execution_commit | provenance | none | false | false | false | 正式执行锁绑定且与当前 HEAD 精确一致的40位小写 Git 提交 SHA。 |
| formal_execution_head_detached | governance | none | false | false | false | 正式执行时 HEAD 是否已脱离所有分支引用。 |
| formal_execution_worktree_clean | governance | none | false | false | false | 正式执行时 Git porcelain 状态是否为空, 包括未跟踪文件。 |
| formal_execution_lock_ready | governance | none | false | false | false | 提交身份、detached HEAD 与 clean 工作树是否同时满足正式执行条件。 |
| formal_execution_lock_digest | provenance | none | false | false | false | 正式执行锁核心记录的稳定 SHA-256 摘要。 |
| formal_execution_lock | provenance | none | false | false | false | 运行环境报告中经过严格 schema 校验的完整正式执行锁记录。 |
| formal_execution_run_lock | provenance | none | false | false | false | 正式运行入口与运行 manifest 写出前两次实时复验形成的完整执行锁记录。 |
| formal_execution_package_lock | provenance | none | false | false | false | 正式归档开始与归档写出后实时复验形成的完整执行锁记录。 |
| formal_execution_run_lock_digest | provenance | none | false | false | false | 单个正式输入包中运行锁的稳定 SHA-256 摘要。 |
| formal_execution_package_lock_digest | provenance | none | false | false | false | 单个正式输入包中打包锁的稳定 SHA-256 摘要。 |
| formal_execution_run_lock_digests | provenance | none | false | false | false | 闭合输入锁中由10类 package family 映射到运行锁摘要的精确字典。 |
| formal_execution_package_lock_digests | provenance | none | false | false | false | 闭合输入锁中由10类 package family 映射到打包锁摘要的精确字典。 |
| rebuild_command | artifact | none | false | false | false | 产物重建命令。 |
| generated_at | governance | none | false | false | false | 本地报告或审计摘要的生成时间。 |
| construction_unit_name | governance | none | false | false | false | 项目分阶段构建流程中的语义阶段名称。 |
| phase_status | governance | none | false | false | false | 分阶段构建状态文档中的阶段推进状态。 |
| executor | governance | none | false | false | false | 执行当前阶段推进的主体。 |
| execution_date | governance | none | false | false | false | 当前阶段推进的执行日期。 |
| input_manifest | governance | none | false | false | false | 当前阶段使用的输入 manifest。 |
| expected_output_manifest | governance | none | false | false | false | 当前阶段预期生成的输出 manifest。 |
| expected_outputs | governance | none | false | false | false | 当前阶段预期生成的输出文件集合。 |
| blocking_items | governance | none | false | false | false | 当前阶段尚未解除的阻断项。 |
| invariants | governance | none | false | false | false | 当前阶段推进时必须保持不变的约束。 |
| next_stage_entry | governance | none | false | false | false | 当前阶段通过后允许进入的下一阶段入口。 |
| command | governance | none | false | false | false | 阶段验证表中记录的验证命令。 |
| result | governance | none | false | false | false | 阶段验证表中记录的验证结果。 |
| decision | governance | none | false | false | false | 审计、阶段报告或导入检查的通过状态。 |
| checked_paths | governance | none | false | false | false | 审计或阶段报告实际检查的路径集合。 |
| violations | governance | none | false | false | false | 审计或阶段报告发现的违规项集合。 |
| reason | governance | none | false | false | false | 单个违规项的原因代码。 |
| value | governance | none | false | false | false | 单个违规项触发检查的值。 |
| summary | governance | none | false | false | false | 报告中的聚合统计信息。 |
| checked_path_count | governance | none | false | false | false | 报告检查路径数量。 |
| violation_count | governance | none | false | false | false | 报告违规项数量。 |
| package_root | governance | none | false | false | false | 阶段报告检查的核心包根目录。 |
| repository_root | governance | none | false | false | false | 阶段脚本执行时使用的仓库根目录。 |
| required_package_dirs | governance | none | false | false | false | 阶段要求存在的核心包子目录集合。 |
| forbidden_import_prefixes | governance | none | false | false | false | 核心包边界禁止导入的模块前缀集合。 |
| forbidden_text_patterns | governance | none | false | false | false | 核心包边界禁止出现的运行时路径或外层依赖文本模式。 |
| import_target | governance | none | false | false | false | 导入检查使用的目标模块名。 |
| import_succeeded | governance | none | false | false | false | 导入检查是否成功。 |
| report_digest | governance | none | false | false | false | 本地阶段报告内容的稳定摘要。 |
| summary_digest | governance | none | false | false | false | 本地阶段 summary 内容的稳定摘要。 |
| primitive_status | governance | none | false | false | false | 阶段 summary 中记录的算法原语实现状态集合。 |
| primitive_name | governance | none | false | false | false | 算法原语对象或阶段记录中的原语名称。 |
| metrics | governance | none | false | false | false | 阶段 summary 中记录的指标集合。 |
| record_count | governance | none | false | false | false | 阶段脚本生成的记录数量。 |
| carrier_digests | method | none | false | false | false | smoke metrics 中记录的 carrier 摘要集合。 |
| minimal_method_dependency | governance | none | false | false | false | minimal smoke 依赖的最小方法包入口。 |
| writes_persistent_output_by_default | governance | none | false | false | false | 脚本默认是否写出持久化输出。 |
| attention_runtime | method | none | false | false | false | attention 原语是否接入真实运行时的状态说明。 |
| branch | method | none | false | false | false | 载体派生分支名称, 例如 LF、`tail_robust` 或 attention。 |
| eta_saliency | method | none | false | false | false | 语义风险场中 saliency 权重。 |
| eta_semantic | method | none | false | false | false | 语义风险场中 semantic 权重。 |
| eta_texture | method | none | false | false | false | 语义风险场中 texture 权重。 |
| eta_stability | method | none | false | false | false | 语义风险场中 stability 权重。 |
| budget_min | method | none | false | false | false | 语义承载预算下界。 |
| budget_max | method | none | false | false | false | 语义承载预算上界。 |
| budget_gain | method | none | false | false | false | 由低风险区域提升承载预算的增益。 |
| texture_threshold | method | none | false | false | false | LF/高斯幅值尾部截断路由的纹理阈值。 |
| risk_threshold | method | none | false | false | false | LF 路由的风险阈值。 |
| stability_threshold | method | none | false | false | false | 高斯幅值尾部截断与 attention 路由的稳定性阈值。 |
| risk_values | method | none | false | false | false | 语义风险场逐位置风险值。 |
| budget_values | method | none | false | false | false | 语义承载预算逐位置数值。 |
| lf_mask | method | none | false | false | false | LF 主证据候选区域 mask。 |
| tail_mask | method | none | false | false | false | 高斯幅值尾部截断补充证据候选区域 mask。 |
| attention_mask | method | none | false | false | false | attention 几何候选区域 mask。 |
| mask_values | method | none | false | false | false | 投影到 latent 长度后的 mask 数值。 |
| masked_latent_values | method | none | false | false | false | 应用 mask 后的 latent 数值。 |
| source_length | method | none | false | false | false | 投影前 mask 长度。 |
| target_length | method | none | false | false | false | 投影后目标 latent 长度。 |
| projection_digest | method | none | false | false | false | latent mask 投影结果摘要。 |
| lf_basis | method | none | false | false | false | LF 路由投影后的子基底。 |
| tail_basis | method | none | false | false | false | 高斯幅值尾部截断路由投影后的子基底。 |
| attention_basis | method | none | false | false | false | attention 路由投影后的子基底。 |
| selected_indices | method | none | false | false | false | Jacobian 低响应子空间基底选中的 latent 位置。 |
| basis_rank | method | none | false | false | false | 语义条件 Jacobian 低响应子空间的实际秩。 |
| selected_index_count_min | metric | none | false | false | false | 当前语义子空间输出中每条安全基底选中索引数量的最小值。 |
| selected_index_count_max | metric | none | false | false | false | 当前语义子空间输出中每条安全基底选中索引数量的最大值。 |
| condition_id | method | none | false | false | false | 语义条件化水印路由对象的稳定标识。 |
| semantic_digest | method | none | false | false | false | 语义输入或语义表示的稳定摘要。 |
| semantic_tags | method | none | false | false | false | 语义条件对象携带的标签集合。 |
| risk_policy | method | none | false | false | false | 语义条件对象采用的风险策略名称。 |
| subspace_id | method | none | false | false | false | 潜空间安全子空间对象的稳定标识。 |
| basis_digest | method | none | false | false | false | 潜空间子空间基的稳定摘要。 |
| manifold_dimension | method | none | false | false | false | 潜空间流形子空间维度。 |
| safe_axes | method | none | false | false | false | 可用于水印承载的安全轴集合。 |
| carrier_id | method | none | false | false | false | 水印载体对象的稳定标识。 |
| carrier_family | method | none | false | false | false | 水印载体所属机制族。 |
| frequency_band | method | none | false | false | false | 水印载体使用的频带名称或明确的频带不适用标识。 |
| key_digest | method | none | false | false | false | 载体派生时使用的密钥材料摘要。 |
| watermark_key_digest | method | none | true | false | false | 最小 latent injection 使用的水印密钥材料摘要, 不记录原始密钥。 |
| update_values | method | none | false | false | false | 单个 carrier 生成的 latent update 数值。 |
| carrier_digest | method | none | true | false | false | 单个 carrier 的稳定摘要。 |
| carrier_source | method | none | true | false | false | 最小 latent injection 中 carrier 的来源机制。 |
| carrier_width | method | none | true | false | false | 从核心算法原语导出的 carrier 基础宽度。 |
| lf_carrier_digest | method | none | true | false | false | LF carrier 的稳定摘要。 |
| tail_carrier_digest | method | none | true | false | false | 高斯幅值尾部截断 carrier 的稳定摘要。 |
| core_update_digest | method | none | true | false | false | LF/高斯幅值尾部截断/attention carrier 合成后的核心 update 摘要。 |
| tail_threshold | method | none | false | false | false | 高斯幅值尾部截断使用的幅值分位点阈值, 不表示频率截止值。 |
| retained_fraction | method | none | false | false | false | 高斯幅值尾部截断后实际保留的模板元素比例。 |
| tail_fraction | method | none | false | false | false | 高斯幅值尾部截断的目标元素保留比例, 不定义空间频带。 |
| embedding_strength | method | none | false | false | false | 水印嵌入强度。 |
| anchor_id | method | none | false | false | false | 注意力几何锚点对象的稳定标识。 |
| attention_layer | method | none | false | false | false | 注意力锚点对应的层或模块名称。 |
| anchor_digest | method | none | false | false | false | 注意力锚点的稳定摘要。 |
| lf_update_values | method | none | false | false | false | LF 分量 latent update 数值。 |
| tail_update_values | method | none | false | false | false | 高斯幅值尾部截断分量的 latent update 数值。 |
| combined_update_values | method | none | false | false | false | 三个分量相加后的 latent update 数值。 |
| update_digest | method | none | false | false | false | 组合 latent update 摘要。 |
| evidence_id | method | none | true | false | false | 检测证据对象的稳定标识。 |
| evidence_type | method | none | true | false | false | 检测证据类型, 例如 content、geometry 或 attention。 |
| score_name | method | none | true | false | false | 检测证据分数名称。 |
| score_value | method | none | true | false | false | 检测证据分数值。 |
| lf_score | method | none | true | false | false | LF 分支归一化相关分数。 |
| tail_score | method | none | true | false | false | 高斯幅值尾部截断分支的相关分数。 |
| combined_score | method | none | true | false | false | 观测向量与实际 runtime 写入的 combined_update_values 方向之间的归一化相关分数。 |
| lf_tail_fusion_score | method | none | true | false | false | LF/高斯幅值尾部截断按固定权重融合后的诊断分数, 不作为正式 fixed-FPR 分数空间。 |
| content_score | method | none | true | false | false | fixed-FPR 正式内容分数, 当前对齐 combined_update_values 写入方向。 |
| lambda_lf | method | none | true | false | false | 内容分数中 LF 分支权重。 |
| lambda_tail | method | none | true | false | false | 内容分数中的高斯幅值尾部截断分支权重。 |
| used_independent_branch_vote | method | none | true | false | false | 是否使用 LF/高斯幅值尾部截断独立阈值投票。正式方法应为 false。 |
| registration_confidence | method | none | true | false | false | 几何恢复的注册置信度。 |
| anchor_inlier_ratio | method | none | true | false | false | attention anchor 内点比例。 |
| recovered_sync_consistency | method | none | true | false | false | 恢复后相对关系同步一致性。 |
| alignment_residual | method | none | true | false | false | 几何恢复后的残差。 |
| geometry_reliable | method | none | true | false | false | 几何恢复是否可信。 |
| direct_positive_decision | method | none | true | false | false | 几何链是否直接给出 positive。正式方法应为 false。 |
| registration_threshold | method | none | true | false | false | 几何注册置信度阈值。 |
| inlier_threshold | method | none | true | false | false | 几何内点比例阈值。 |
| sync_threshold | method | none | true | false | false | 几何同步一致性阈值。 |
| residual_threshold | method | none | true | false | false | 几何残差阈值。 |
| decision_id | method | none | true | false | false | 多证据融合检测决策的稳定标识。 |
| decision_label | method | none | true | false | false | 多证据融合检测决策标签。 |
| threshold_name | protocol | none | true | false | false | 检测或校准协议使用的阈值名称。 |
| threshold_value | protocol | none | true | false | false | 检测或校准协议使用的阈值数值。 |
| threshold_score_field | protocol | none | true | false | false | fixed-FPR 阈值实际校准和正式判定使用的分数字段。 |
| threshold_score_source_field | protocol | none | true | false | false | fixed-FPR 正式判定分数在上游记录中的来源字段。 |
| evidence_ids | method | none | true | false | false | 融合决策引用的检测证据标识集合。 |
| raw_content_score | method | none | true | false | false | 几何 rescue 前的内容分数。 |
| aligned_content_score | method | none | true | false | false | 几何对齐后的内容分数。 |
| formal_detection_score | method | none | true | false | false | fixed-FPR 正式判定使用的分数, 其来源由 threshold_score_field 说明。 |
| threshold_score_after | method | none | true | false | false | 攻击或质量保持作用后用于 fixed-FPR 判定的正式分数。 |
| content_threshold | protocol | none | true | false | false | fixed-FPR 内容阈值。 |
| score_margin | method | none | true | false | false | 内容分数相对内容阈值的边界余量。 |
| raw_content_margin | method | none | true | false | false | 原始内容分数相对内容阈值的边界余量。 |
| aligned_content_margin | method | none | true | false | false | 对齐后内容分数相对内容阈值的边界余量。 |
| formal_detection_margin | method | none | true | false | false | 正式判定分数相对 fixed-FPR 阈值的边界余量。 |
| fail_reason | method | none | true | false | false | 内容判定失败原因。 |
| rescue_margin_low | protocol | none | true | false | false | rescue 边界失败窗口下界。 |
| positive_by_content | method | none | true | false | false | 原始内容分支是否正判。 |
| formal_detection_decision | method | none | true | false | false | 按 threshold_score_field 与 fixed-FPR 阈值得到的正式检测判定。 |
| rescue_eligible | method | none | true | false | false | 样本是否满足 rescue 条件。 |
| rescue_applied | method | none | true | false | false | 是否实际应用 rescue 并通过同阈值重判。 |
| evidence_level | method | none | true | false | false | evidence-level 判定结果。 |
| evidence_decision | method | none | true | false | false | smoke 场景中的 evidence-level 判定结果。 |
| attestation_pass | method | none | true | false | false | attestation 是否验证通过。 |
| final_level | method | none | true | false | false | final-level 判定结果。 |
| final_decision | method | none | true | false | false | smoke 场景中的 final-level 判定结果。 |
| final_label | method | none | true | false | false | final-level 判定标签。 |
| key_separation_margin | method | none | true | false | false | 正确 key 与错误 key 内容分数差。 |
| score_margin_min | method | none | true | false | false | smoke 场景中的最小 score margin。 |
| rescue_trigger_rate | method | none | true | false | false | smoke 场景中 rescue_applied 的触发比例。 |
| wrong_key_over_threshold | method | none | true | false | false | 错误 key 是否超过内容阈值。 |
| geometry_unreliable_rescue_blocked | method | none | true | false | false | 几何可靠性不足时 rescue 是否被阻断。 |
| final_positive_count | method | none | true | false | false | smoke 场景 final positive 数量。 |
| evidence_positive_count | method | none | true | false | false | smoke 场景 evidence positive 数量。 |
| tail_truncation_delta | method | none | true | false | false | 高斯幅值尾部截断前后的分数差异。 |
| attestation_layering_pass | method | none | true | false | false | attestation 是否只影响 final-level 的检查结果。 |
| model_family | runtime | none | true | false | false | SD runtime adapter 使用的模型族。 |
| model_id | runtime | none | true | false | false | SD runtime adapter 使用的模型标识。 |
| model_revision | runtime | none | true | false | false | 模型 loader 实际使用的40位不可变 Hugging Face 仓库提交。 |
| vision_model_revision | runtime | none | true | false | false | 语义条件 CLIP 图像编码器实际使用的40位不可变提交。 |
| generation_model_revision | runtime | none | true | false | false | 外部 baseline observation 实际生成图像时使用的模型提交。 |
| official_model_revision | runtime | none | false | false | false | official-reference 公开镜像快照的40位不可变提交。 |
| clip_model_revision | runtime | none | false | false | false | 成对感知质量评估使用的 CLIP 模型提交。 |
| pair_clip_model_revision | runtime | none | false | false | false | T2SMark 严格图像对质量配置使用的 CLIP 模型提交。 |
| diffusion_model_source | provenance | none | false | false | false | 运行环境报告中绑定仓库、revision 和用途的扩散模型来源记录。 |
| backend_name | runtime | none | true | false | false | 实际执行 generation probe 的后端名称。 |
| backend_mode | runtime | none | true | false | false | 配置中要求的 runtime 后端模式。 |
| runtime_dependency_mode | runtime | none | true | false | false | runtime adapter 实际依赖模式。 |
| prompt | runtime | none | false | false | false | runtime 配置中的正向提示词。 |
| negative_prompt | runtime | none | false | false | false | runtime 配置中的反向提示词。 |
| prompt_digest | runtime | none | true | false | false | prompt 文本或 runtime prompt 配置的稳定摘要。 |
| seed | runtime | none | true | false | false | runtime adapter 使用的确定性种子。 |
| width | runtime | none | true | false | false | generation probe 的目标图像宽度。 |
| height | runtime | none | true | false | false | generation probe 的目标图像高度。 |
| inference_steps | runtime | none | true | false | false | runtime adapter 采样步数。 |
| guidance_scale | runtime | none | true | false | false | runtime adapter guidance scale 配置。 |
| latent_width | runtime | none | false | false | false | SD3.5 Medium runtime 使用的 latent 向量宽度。 |
| generation_id | runtime | none | true | false | false | generation record 的稳定标识。 |
| latent_digest | runtime | none | true | false | false | latent 向量或最终 latent 的稳定摘要。 |
| image_digest | runtime | none | true | false | false | 真实生成或攻击图像的稳定摘要。 |
| image_shape | runtime | none | true | false | false | generation record 记录的图像形状。 |
| quality_score | metric | none | true | false | false | source 与 evaluated 图像之间按共享高斯窗口实现计算的实测 SSIM。 |
| unsupported_reason | runtime | none | true | false | false | 真实后端不可用或能力降级的原因。 |
| unsupported_reasons | runtime | none | false | false | false | runtime summary 中聚合的 unsupported_reason 集合。 |
| trajectory_index | runtime | none | true | false | false | latent trace 在采样序列中的索引。 |
| timestep | runtime | none | true | false | false | latent trace 对应的采样时间位置。 |
| latent_shape | runtime | none | true | false | false | latent trace 中记录的 latent 形状。 |
| latent_mean | runtime | none | true | false | false | latent trace 中 latent 数值均值。 |
| latent_std | runtime | none | true | false | false | latent trace 中 latent 数值标准差。 |
| latent_min | runtime | none | true | false | false | latent trace 中 latent 数值最小值。 |
| latent_max | runtime | none | true | false | false | latent trace 中 latent 数值最大值。 |
| trace_source | runtime | none | false | false | false | latent trace 的来源后端。 |
| capture_id | runtime | none | true | false | false | attention capture record 的稳定标识。 |
| capture_backend | runtime | none | true | false | false | attention capture record 的捕获后端。 |
| supports_paper_claim | governance | none | true | false | false | 记录或摘要是否允许支持正式论文 claim。 |
| config_digests | artifact | none | false | false | false | runtime manifest 中记录的配置摘要集合。 |
| generation_record_count | runtime | none | false | false | false | runtime summary 中 generation record 数量。 |
| latent_trace_record_count | runtime | none | false | false | false | runtime summary 中 latent trace record 数量。 |
| attention_capture_record_count | runtime | none | false | false | false | runtime summary 中 attention capture record 数量。 |
| unsupported_reason_count | runtime | none | false | false | false | runtime summary 中 unsupported_reason 数量。 |
| reproducibility_digest | runtime | none | false | false | false | runtime summary 的复现摘要。 |
| mean_quality_score | runtime | none | false | false | false | generation records 的平均质量分数。 |
| probe_id | runtime | none | true | false | false | Colab 真实 runtime probe 的稳定标识。 |
| probe_decision | runtime | none | true | false | false | Colab 真实 runtime probe 的通过状态。 |
| trajectory_entry_count | runtime | none | true | false | false | 真实 runtime 捕获到的 latent trajectory 记录数量。 |
| pipeline_class | runtime | none | true | false | false | 实际加载的 diffusers pipeline 类名。 |
| device_name | runtime | none | true | false | false | 真实推理使用的设备名称。 |
| torch_dtype | runtime | none | true | false | false | 真实推理使用的 torch dtype。 |
| hf_token_env | runtime | none | false | false | false | 真实 runtime 读取 Hugging Face token 的环境变量名。 |
| torch_version | runtime | none | false | false | false | Colab runtime 中的 torch 版本。 |
| diffusers_version | runtime | none | false | false | false | Colab runtime 中的 diffusers 版本。 |
| transformers_version | runtime | none | false | false | false | Colab runtime 中的 transformers 版本。 |
| accelerate_version | runtime | none | false | false | false | Colab runtime 中的 accelerate 版本。 |
| huggingface_hub_version | runtime | none | false | false | false | Colab runtime 中的 huggingface_hub 版本。 |
| tokenizers_version | runtime | none | false | false | false | Colab runtime 中的 tokenizers 版本。 |
| safetensors_version | runtime | none | false | false | false | Colab runtime 中的 safetensors 版本。 |
| sentencepiece_version | runtime | none | false | false | false | Colab runtime 中的 sentencepiece 版本。 |
| protobuf_version | runtime | none | false | false | false | Colab runtime 中的 protobuf 版本。 |
| numpy_version | runtime | none | false | false | false | Colab runtime 中的 numpy 版本。 |
| pillow_version | runtime | none | false | false | false | Colab runtime 中的 pillow 版本。 |
| dependency_mode | runtime | none | false | false | false | 当前运行是否使用已提交完整哈希锁的固定依赖模式。 |
| profile_id | governance | none | true | false | false | 依赖准备 CLI 选择的受治理 profile 标识。 |
| execution_role | governance | none | true | false | false | 依赖 profile 对应的运行职责。 |
| profile_digest | artifact | none | true | false | false | Python、平台、CUDA、PyTorch pair 与直接输入身份的稳定摘要。 |
| profile_summary_digest | artifact | none | true | false | false | 依赖准备报告引用的 profile summary 稳定摘要。 |
| direct_dependency_input_contract | governance | none | false | false | false | registry 中区分精确直接依赖输入的契约。 |
| complete_hash_lock_contract | governance | none | false | false | false | registry 中区分目标 Linux x86_64 runtime 完整 wheel 哈希锁的契约。 |
| artifact_role | governance | none | false | false | false | 依赖输入或完整锁在正式环境中的证据职责。 |
| requirement_format | governance | none | false | false | false | 直接输入或完整锁条目的受治理文本格式。 |
| exact_operator | governance | none | false | false | false | 直接依赖输入唯一允许的精确版本运算符。 |
| materialization_environment | runtime | none | false | false | false | 完整 wheel 哈希锁必须被解析和物化的目标环境。 |
| repository_commit_required | governance | none | false | false | false | 完整锁是否必须提交后才能进入正式准备路径。 |
| formal_readiness_requires_valid_lock | governance | none | false | false | false | 正式 readiness 是否强制依赖有效完整哈希锁。 |
| profiles | governance | none | false | false | false | dependency registry 中按稳定名称索引的隔离 profile 集合。 |
| python | runtime | none | false | false | false | profile 中精确 CPython 身份的结构化记录。 |
| implementation | runtime | none | false | false | false | profile 登记的 Python 实现名称。 |
| accelerator | runtime | none | false | false | false | profile 登记的加速器运行时与 CUDA 版本。 |
| pytorch | runtime | none | false | false | false | profile 登记的 torch、torchvision 和 wheel index 组合。 |
| operating_system | runtime | none | true | false | false | profile 期望或 inspection 实测的操作系统名称。 |
| machine | runtime | none | true | false | false | profile 期望或 inspection 实测的机器架构。 |
| accelerator_runtime | runtime | none | true | false | false | profile 要求的运行时类型; 父编排为 `cpu`, 五个科学 profile 为 `cuda`。 |
| torchvision_version | runtime | none | true | false | false | profile 要求的精确 torchvision local version。 |
| index_url | runtime | none | false | false | false | registry 中 PyTorch CUDA wheel index。 |
| pytorch_index_url | runtime | none | true | false | false | CUDA profile summary 中展开的 PyTorch wheel index; CPU 父编排 profile 为 null。 |
| direct_requirements_path | artifact | none | true | false | false | 已提交精确直接依赖输入文件的仓库相对路径。 |
| direct_requirements | runtime | none | true | false | false | profile 中全部 `name==version` 直接依赖集合。 |
| direct_requirements_digest | artifact | none | true | false | false | 精确直接依赖集合的稳定语义摘要。 |
| locked_requirements | runtime | none | true | false | false | 完整哈希锁中全部直接与传递依赖的规范化 `name==version` 集合。 |
| complete_hash_lock_path | artifact | none | true | false | false | 目标环境完整 wheel 哈希锁的仓库相对路径。 |
| complete_hash_lock_present | runtime | none | true | false | false | profile 查询时完整哈希锁是否存在。 |
| complete_hash_lock_digest | artifact | none | true | false | false | 完整哈希锁条目与真实 wheel SHA-256 集合的稳定摘要。 |
| complete_hash_lock_dependency_count | metric | none | true | false | false | 完整哈希锁覆盖的直接和传递依赖数量。 |
| formal_ready | governance | none | true | false | false | profile 是否已具备可进入正式环境准备的完整锁。 |
| readiness_blockers | governance | none | true | false | false | 阻断 profile 正式准备的稳定原因列表。 |
| profile_formal_ready | governance | none | true | false | false | inspection 报告引用的 profile 完整锁 readiness。 |
| expected_environment | runtime | none | true | false | false | inspection 从 profile 展开的精确期望环境。 |
| observed_environment | runtime | none | true | false | false | inspection 从当前解释器与 torch 读取的实测环境。 |
| environment_match | governance | none | true | false | false | 当前解释器、平台、完整锁包与适用的 CUDA identity 是否精确匹配。 |
| mismatches | governance | none | true | false | false | inspection 检出的稳定环境不一致原因列表。 |
| inspection_digest | artifact | none | true | false | false | 不含时间戳和绝对路径的环境 inspection 稳定摘要。 |
| torch_module_available | runtime | none | true | false | false | CUDA 科学解释器能否加载 torch 模块; CPU 父编排 profile 为 null。 |
| torch_module_version | runtime | none | true | false | false | 当前解释器实际加载的 torch 模块版本。 |
| torch_cuda_version | runtime | none | true | false | false | 当前 torch build 报告的 CUDA 版本。 |
| direct_dependencies | runtime | none | true | false | false | inspection 中规范化包名到期望或实测版本的映射。 |
| locked_dependencies | runtime | none | true | false | false | inspection 中完整锁全部直接与传递包名到期望或实测版本的映射。 |
| report_schema | provenance | none | true | false | false | 依赖准备报告的 schema 名称。 |
| repository_commit_state | provenance | none | true | false | false | registry、直接输入和完整锁的 Git 提交状态集合。 |
| registry | provenance | none | false | false | false | 依赖准备报告中 registry 文件的提交状态记录。 |
| complete_hash_lock | provenance | none | false | false | false | 依赖准备报告中完整锁文件的提交状态记录。 |
| path | artifact | none | false | false | false | 单个受治理依赖输入的仓库相对路径。 |
| working_tree_file_present | provenance | none | false | false | false | 当前工作树中是否存在指定受治理依赖输入。 |
| head_contains_file | provenance | none | false | false | false | 当前 Git `HEAD` 是否包含受治理依赖输入。 |
| worktree_matches_head | provenance | none | false | false | false | 受治理依赖输入的工作树内容是否与 `HEAD` 一致。 |
| is_committed | provenance | none | false | false | false | 单个依赖输入是否同时存在于 `HEAD` 且无工作树漂移。 |
| all_committed | provenance | none | true | false | false | registry、直接输入和完整锁是否全部已提交且无漂移。 |
| installation | runtime | none | true | false | false | 依赖准备报告中的 hash-locked pip 执行记录。 |
| attempted | runtime | none | false | false | false | 依赖准备 CLI 是否已尝试执行完整锁安装。 |
| pip_check | runtime | none | true | false | false | 当前解释器执行依赖兼容性核验的完整 argv、返回码、标准输出、标准错误与结论。 |
| compatibility_check_required | governance | none | true | false | false | 当前 profile 是否必须执行 `sys.executable -m pip check`; 五个科学 profile 为 true, 父编排 profile 为 false。 |
| runtime_comparison | runtime | none | true | false | false | 安装后共享 inspection API 生成的精确环境比较记录。 |
| failure_reasons | governance | none | true | false | false | 依赖准备失败或被阻断的稳定原因列表。 |
| report_path | artifact | none | false | false | false | 依赖准备 CLI 写入 `outputs/` 的报告路径。 |
| error | runtime | none | false | false | false | registry 解析失败时 CLI 输出的诊断文本。 |
| operation_kind | protocol | none | true | false | false | 隔离依赖报告区分解释器 provision 与正式环境 preparation 的稳定操作类型。 |
| target_complete_hash_lock_ready | governance | none | true | false | false | provision 时目标科学 profile 是否已具备完整锁; 该值不改变 provision 的非正式性质。 |
| environment_root | runtime | none | true | false | false | 五个科学子环境所在的可覆盖跨服务器根目录。 |
| managed_python_root | runtime | none | true | false | false | 固定 `uv` 保存受管 CPython distribution 的可覆盖根目录。 |
| isolated_environment_path | runtime | none | true | false | false | 当前科学 profile 独立 venv 的绝对路径。 |
| uv_distribution_version | provenance | none | true | false | false | 父编排解释器中安装的受治理 `uv` distribution 精确版本。 |
| uv_distribution_record_path | provenance | none | true | false | false | 当前解释器所安装 `uv` distribution 的实际 `RECORD` 文件路径。 |
| uv_distribution_record_sha256 | provenance | none | true | false | false | 当前解释器所安装 `uv` distribution 的 `RECORD` 文件 SHA-256。 |
| uv_distribution_executable_record_path | provenance | none | true | false | false | `uv` executable 在当前解释器 distribution `RECORD` 中登记的路径。 |
| uv_distribution_executable_record_sha256 | provenance | none | true | false | false | 按 distribution `RECORD` 验证通过的 `uv` executable SHA-256。 |
| uv_executable_path | provenance | none | true | false | false | 实际执行的 `uv` executable 绝对路径。 |
| uv_executable_sha256 | provenance | none | true | false | false | 实际执行的 `uv` executable 文件 SHA-256。 |
| uv_reported_version | provenance | none | true | false | false | 实际 `uv --version` 进程报告的精确版本。 |
| python_executable_path | provenance | none | true | false | false | `uv` 创建的目标科学子解释器绝对路径。 |
| python_executable_sha256 | provenance | none | true | false | false | 正式依赖安装前目标科学子解释器文件 SHA-256。 |
| python_executable_sha256_after_preparation | provenance | none | true | false | false | 正式依赖安装后再次计算的目标科学子解释器文件 SHA-256。 |
| operation | protocol | none | false | false | false | 单条隔离环境命令的稳定职责标识。 |
| argv | runtime | none | true | false | false | 单条隔离环境命令未经 shell 解释的完整参数向量。 |
| environment_overrides | runtime | none | true | false | false | 单条隔离环境命令显式传入的环境变量覆盖映射。 |
| command_results | runtime | none | true | false | false | 隔离环境流程按执行顺序保存的全部命令进程证据。 |
| uv_commands | provenance | none | true | false | false | `command_results` 中实际由固定 `uv` executable 执行的命令子集。 |
| orchestrator_profile_digest | provenance | none | true | false | false | provision 绑定的父编排 profile 稳定摘要。 |
| orchestrator_complete_hash_lock_digest | provenance | none | true | false | false | provision 绑定的父编排完整哈希锁摘要。 |
| orchestrator_inspection | runtime | none | true | false | false | provision 前父编排解释器的完整锁环境实测报告。 |
| provisioned | governance | none | true | false | false | 精确 CPython 与隔离 venv 是否创建并通过 patch 核验; 该字段不表示正式依赖环境 ready。 |
| provision_report_path | artifact | none | true | false | false | 正式隔离环境报告绑定的 Python provision 报告路径。 |
| provision_report_digest | provenance | none | true | false | false | 正式隔离环境报告绑定的 Python provision 报告文件 SHA-256。 |
| provision_report | provenance | none | true | false | false | 正式隔离环境报告内嵌的完整 Python provision 记录。 |
| dependency_preparation_command | runtime | none | true | false | false | 目标子解释器调用内层 dependency preparation 的完整进程证据。 |
| dependency_preparation_report_path | artifact | none | true | false | false | 目标子解释器写出的 dependency preparation 报告路径。 |
| dependency_preparation_report_digest | provenance | none | true | false | false | 目标子解释器 dependency preparation 报告文件 SHA-256。 |
| dependency_preparation_report | provenance | none | true | false | false | 正式隔离环境报告内嵌并严格复核的子解释器 preparation 记录。 |
| formal_preparation_completed | governance | none | true | false | false | 目标完整锁安装、兼容性核验、完整 inspection 与解释器摘要复核是否全部完成。 |
| dependency_environment_report_path | artifact | none | true | false | false | 科学子解释器继承且经过路径约束的正式隔离依赖环境报告绝对路径。 |
| dependency_environment_report_digest | provenance | none | true | false | false | 父执行原语注入的正式隔离依赖环境报告 SHA-256。 |
| dependency_environment_report_actual_digest | provenance | none | true | false | false | 科学子解释器实时复算的正式隔离依赖环境报告 SHA-256。 |
| dependency_environment_report_valid | governance | none | true | false | false | 隔离科学执行原语是否已严格验证依赖环境报告及解释器身份。 |
| dependency_environment_validation_errors | governance | none | true | false | false | 隔离依赖环境报告未通过执行前验证时的稳定错误列表。 |
| python_executable_revalidated_before_child | governance | none | true | false | false | 启动科学子命令前是否再次确认解释器文件 SHA-256 未漂移。 |
| python_executable_revalidated_after_child | governance | none | true | false | false | 科学子命令结束后是否再次确认解释器文件 SHA-256 未漂移。 |
| dependency_environment_report_revalidated_before_child | governance | none | true | false | false | 启动科学子命令前是否再次确认隔离环境报告摘要未漂移。 |
| dependency_environment_report_revalidated_after_child | governance | none | true | false | false | 科学子命令结束后是否再次确认隔离环境报告摘要未漂移。 |
| formal_execution_lock_revalidated_before_child | governance | none | true | false | false | 启动科学子命令前是否实时复验同一正式执行锁。 |
| formal_execution_lock_revalidated_after_child | governance | none | true | false | false | 科学子命令结束后是否实时复验同一正式执行锁。 |
| child_argv_tail | runtime | none | true | false | false | 由调用方提供且不包含 Python executable 的科学子命令参数尾部。 |
| execution_report_path | artifact | none | true | false | false | 隔离科学执行原语在 `outputs/` 下写出的 JSON 报告路径。 |
| execution_completed | governance | none | true | false | false | 科学子命令是否以返回码0完成且全部执行后身份复核通过。 |
| scientific_execution_report | provenance | none | true | false | false | 外层 session 返回的完整隔离科学执行报告。 |
| scientific_execution_report_path | artifact | none | true | false | false | 产物内本地化隔离科学执行报告或其会话源报告路径。 |
| scientific_execution_report_digest | provenance | none | true | false | false | 本地化隔离科学执行报告的文件 SHA-256。 |
| source_dependency_environment_report_path | provenance | none | true | false | false | 科学子进程启动时使用的隔离依赖环境报告绝对源路径, 用于离线复验仓库根、工作目录与环境覆盖的一致性。 |
| scientific_profile_id | protocol | none | true | false | false | 外层 session 实际选择的唯一科学依赖 profile。 |
| scientific_profile_digest | provenance | none | true | false | false | 外层 session 绑定的科学 profile 稳定摘要。 |
| scientific_direct_requirements_digest | provenance | none | true | false | false | CPU 闭合输入锁传播的科学 profile 直接依赖输入 SHA-256。 |
| scientific_complete_hash_lock_digest | provenance | none | true | false | false | 外层 session 绑定的科学 profile 完整哈希锁摘要。 |
| scientific_complete_hash_lock_dependency_count | metric | none | true | false | false | CPU 闭合输入锁传播的完整哈希锁直接与传递依赖总数。 |
| scientific_python_executable_digest | provenance | none | true | false | false | CPU 闭合输入锁传播的实际科学解释器文件 SHA-256。 |
| scientific_execution_binding_digest | provenance | none | true | false | false | CPU 闭合输入锁记录的产物级科学执行绑定文件 SHA-256; official-reference 包不使用该字段。 |
| scientific_dependency_evidence_digest | provenance | none | true | false | false | CPU 闭合输入锁记录的包内隔离依赖环境证据文件 SHA-256。 |
| scientific_command_sequence_digest | provenance | none | true | false | false | 主方法三个结果包共同传播的实际科学子命令序列 SHA-256, 不包含 stdout 与 stderr。 |
| dependency_profile_count | metric | none | true | false | false | 完整论文结果包要求并复验的依赖 profile 总数。 |
| dependency_profile_records | provenance | none | true | false | false | 完整结果包逐 profile 保存的直接依赖、完整锁、摘要与归档成员复验记录。 |
| dependency_hash_lock_count | metric | none | true | false | false | 已通过完整哈希锁复验的依赖 profile 数量。 |
| dependency_hash_lock_archive_entries_ready | governance | none | true | false | false | 六份完整哈希锁是否全部作为精确成员进入完整论文结果包。 |
| dependency_profile_inputs_archive_entries_ready | governance | none | true | false | false | 依赖 registry 与六份直接依赖输入是否全部进入完整论文结果包。 |
| dependency_hash_locks_ready | governance | none | true | false | false | 六个依赖 profile 的锁内容、摘要、计数、正式门禁与归档成员是否全部闭合。 |
| dependency_hash_lock_failure_reason | governance | none | true | false | false | 完整依赖锁门禁未通过时的稳定阻断原因。 |
| scientific_execution_bindings | provenance | none | true | false | false | 已闭合产物角色到独立科学执行绑定记录的映射。 |
| binding_path | artifact | none | true | false | false | 单个科学执行绑定文件的路径。 |
| binding_digest | provenance | none | true | false | false | 单个科学执行绑定文件的 SHA-256。 |
| binding | provenance | none | true | false | false | 外层 session 返回的完整科学执行绑定对象。 |
| artifact_role | protocol | none | true | false | false | 科学执行绑定或重新打包记录对应的正式产物职责。 |
| scientific_command_dispatch_report_path | artifact | none | true | false | false | 产物内逐科学命令调度报告的路径。 |
| scientific_command_dispatch_report_digest | provenance | none | true | false | false | 产物内逐科学命令调度报告的 SHA-256。 |
| bound_summary_path | artifact | none | true | false | false | 科学执行绑定所约束的正式摘要路径。 |
| bound_summary_digest | provenance | none | true | false | false | 科学执行绑定所约束正式摘要的 SHA-256。 |
| bound_manifest_path | artifact | none | true | false | false | 科学执行绑定所约束的科学 runner manifest 路径. |
| bound_manifest_scientific_digest | provenance | none | true | false | false | 排除唯一打包边界锁字段后, 科学 runner manifest 的规范 JSON SHA-256。 |
| bound_manifest_digest_scope | protocol | none | true | false | false | 科学 manifest 摘要明确排除 `formal_execution_package_lock` 的固定范围标识。 |
| commands | runtime | none | true | false | false | 科学命令调度报告按实际顺序保存的独立命令证据列表。 |
| command_role | protocol | none | true | false | false | 单条科学子命令在当前 session 中的稳定职责。 |
| artifact_records | provenance | none | true | false | false | 当前科学会话逐角色保存的 summary、manifest、执行锁与摘要记录。 |
| artifact_validation_mode | protocol | none | true | false | false | 当前科学会话对已完成或续跑产物执行同会话复验的固定模式。 |
| summary_sha256 | provenance | none | true | false | false | 科学会话确认的角色化 summary 文件 SHA-256。 |
| manifest_sha256_at_session | provenance | none | true | false | false | 补充打包边界锁之前, 科学会话确认的 manifest 文件 SHA-256。 |
| manifest_scientific_digest | provenance | none | true | false | false | 排除唯一打包边界锁后的角色化 manifest 规范 JSON SHA-256。 |
| summary_protocol_decision | governance | none | true | false | false | 科学会话对角色化 summary 协议闭合状态的复验结论。 |
| artifact_state | governance | none | true | false | false | 科学命令结束后主方法、质量和消融的完成或续跑状态。 |
| runtime_progress_present | governance | none | true | false | false | 主方法续跑状态文件是否仍存在。 |
| ablation_progress_present | governance | none | true | false | false | 正式消融续跑状态文件是否仍存在。 |
| include_formal_ablation | protocol | none | true | false | false | 当前绑定重新打包是否包含已经闭合的正式消融。 |
| packaging_deferred | protocol | none | true | false | false | 科学命令阶段是否把归档推迟到外层执行绑定写入并复验之后。 |
| archives | artifact | none | true | false | false | 本次绑定重新打包产生的角色化结果包记录列表。 |
| archive_sha256 | provenance | none | true | false | false | 本次重新生成结果包的文件 SHA-256。 |
| bound_packaging_execution | provenance | none | true | false | false | 外层 session 复用科学解释器执行绑定打包的完整进程记录。 |
| packaging_result | artifact | none | true | false | false | 绑定打包命令返回的角色化结果包对象。 |
| isolated_scientific_context_required | governance | none | true | false | false | 当前依赖 profile 是否必须由隔离科学执行原语启动并注入环境上下文。 |
| isolated_scientific_context_ready | governance | none | true | false | false | 当前科学进程的报告、profile、锁、执行锁与解释器身份是否全部一致。 |
| isolated_scientific_context | runtime | none | true | false | false | `build_runtime_environment_report` 生成的隔离科学上下文严格核验记录。 |
| required | governance | none | true | false | false | 单条隔离科学上下文是否必须通过后才能形成依赖环境 readiness。 |
| ready | governance | none | true | false | false | 单条隔离科学上下文是否没有任何稳定 blocker。 |
| blockers | governance | none | true | false | false | 隔离科学上下文检测出的稳定阻断原因列表。 |
| reported_profile_digest | provenance | none | true | false | false | 注入隔离环境报告声明的依赖 profile 稳定摘要。 |
| reported_complete_hash_lock_digest | provenance | none | true | false | false | 注入隔离环境报告声明的完整 wheel 哈希锁摘要。 |
| reported_formal_execution_lock_digest | provenance | none | true | false | false | 注入隔离环境报告声明的正式执行锁摘要。 |
| reported_python_executable | provenance | none | true | false | false | 注入隔离环境报告声明的科学 Python executable 路径。 |
| reported_python_executable_sha256 | provenance | none | true | false | false | 注入隔离环境报告声明的科学 Python executable SHA-256。 |
| current_python_executable | runtime | none | true | false | false | 科学子进程实时读取的当前 `sys.executable` 绝对路径。 |
| current_python_executable_sha256 | provenance | none | true | false | false | 科学子进程实时计算的当前 `sys.executable` SHA-256。 |
| pip_version | provenance | none | true | false | false | 候选完整锁解析实际使用且由 pip report 自报的 pip distribution 版本。 |
| resolver_command | runtime | none | true | false | false | 候选完整锁物化使用的完整 pip resolver argv。 |
| resolver_return_code | runtime | none | true | false | false | 候选完整锁 pip resolver 进程返回码。 |
| resolver_stdout | runtime | none | false | false | false | 候选完整锁 pip resolver 的标准输出。 |
| resolver_stderr | runtime | none | false | false | false | 候选完整锁 pip resolver 的标准错误。 |
| pip_resolver_report_path | artifact | none | true | false | false | 候选物化器实际解析并由审查包重新验证的 pip report 路径。 |
| candidate_lock_path | artifact | none | true | false | false | 仅供人工审查的规范完整哈希锁候选路径。 |
| candidate_lock_logical_digest | provenance | none | true | false | false | 从实际 pip report 重建的规范 wheel 记录集合稳定摘要。 |
| candidate_lock_dependency_count | metric | none | true | false | false | 从实际 pip report 重建并写入候选锁的直接与传递依赖数量。 |
| candidate_hash_source | provenance | none | true | false | false | 候选 wheel SHA-256 在 pip report 中的唯一来源字段路径。 |
| review_execution_mode | protocol | none | true | false | false | 候选锁审查包区分父编排当前解释器与科学隔离子解释器的执行模式。 |
| local_bundle_dir | artifact | none | true | false | false | 单个依赖 profile 候选锁审查包在 `outputs/` 下的目录。 |
| drive_bundle_dir | artifact | none | true | false | false | 显式请求镜像时单个依赖 profile 的外部 Drive 目录。 |
| drive_copy_performed | governance | none | true | false | false | 审查包是否已经按显式请求复制并逐文件复核到 Drive。 |
| orchestrator_preparation | provenance | none | true | false | false | 科学 profile 候选锁物化前父编排环境 preparation 的报告路径、摘要与结论。 |
| isolated_python_provision | provenance | none | true | false | false | 科学 profile 候选锁物化使用的隔离 CPython provision 身份与结论。 |
| candidate_materialization | provenance | none | true | false | false | 候选锁物化器在父解释器或科学子解释器中的完整执行记录。 |
| diagnostic_message | runtime | none | false | false | false | 候选锁资格化或审查包写入失败时保存的诊断信息。 |
| python_version | runtime | none | false | false | false | Colab runtime 中的 Python 版本。 |
| python_executable | runtime | none | false | false | false | 当前进程或科学会话实际使用的 Python 解释器路径。 |
| platform | runtime | none | false | false | false | Colab runtime 的平台摘要。 |
| package_versions | runtime | none | false | false | false | Colab runtime 中关键 Python 包版本快照。 |
| cuda_available | runtime | none | false | false | false | Colab runtime 是否可用 CUDA。 |
| cuda_version | runtime | none | false | false | false | Colab runtime 中 torch 报告的 CUDA 版本。 |
| device_count | runtime | none | false | false | false | Colab runtime 中可见 GPU 设备数量。 |
| gpu_name | runtime | none | false | false | false | Colab runtime 中首个 GPU 设备名称。 |
| runtime_environment | runtime | none | false | false | false | 真实 runtime 结果 metadata 中嵌入的环境快照。 |
| scientific_unit_provenance | provenance | none | true | false | false | 单个 Prompt、正式消融运行或 Inception feature batch 在实际完成进程中写出的代码锁、依赖锁、设备和随机性来源记录。 |
| scientific_unit_id | provenance | none | true | false | false | 由完成单元精确配置或 batch 图像身份派生的稳定标识。 |
| scientific_unit_config | protocol | none | true | false | false | 单个语义水印完成单元隐藏密钥原文后的完整科学运行配置。 |
| scientific_unit_config_digest | provenance | none | true | false | false | 单个完成单元精确科学配置的稳定 SHA-256 摘要。 |
| scientific_execution_environment | runtime | none | true | false | false | 单个完成单元实际使用的依赖 profile、完整哈希锁、代码锁、Python、PyTorch、CUDA 与 GPU 身份。 |
| scientific_execution_environment_digest | provenance | none | true | false | false | 单个完成单元实际执行环境记录的稳定 SHA-256 摘要。 |
| scientific_random_identity_random | random | _random | true | false | false | 单个完成单元实际使用的生成种子、检测种子、攻击种子或明确无随机生成器模式。 |
| scientific_random_identity_digest_random | random | _digest_random | true | false | false | 单个完成单元随机性身份的稳定 SHA-256 摘要。 |
| generation_seed_random | random | _random | true | false | false | 单个 Prompt 生成 clean 与 watermarked 配对图像使用的确定性种子。 |
| public_detection_seed_random | random | _random | true | false | false | 仅图像检测的公开反演噪声使用的确定性种子。 |
| key_material_digest_random | random | _digest_random | true | false | false | 驱动 keyed carrier、候选方向与注意力目标随机轨迹的密钥材料摘要。 |
| standard_attack_seeds_random | random | _random | true | false | false | 单个完成单元按标准图像攻击标识保存的实际攻击种子映射。 |
| diffusion_attack_seeds_random | random | _random | true | false | false | 单个完成单元按再扩散攻击标识保存的实际攻击种子映射。 |
| feature_extraction_seed_random | random | _random | true | false | false | Inception eval 特征完成单元声明未使用随机生成器的固定随机性模式。 |
| scientific_unit_provenance_digest | provenance | none | true | false | false | 排除自指字段后单个科学完成单元来源记录的稳定 SHA-256 摘要。 |
| execution_device_name | runtime | none | true | false | false | 科学 tensor 实际执行的 CUDA 设备字符串, 包括可用时的设备索引。 |
| visible_cuda_device_count | runtime | none | true | false | false | 完成单元进程实际可见的 CUDA 设备数量。 |
| cuda_device_index | runtime | none | true | false | false | 科学 tensor 实际绑定且经过范围校验的 CUDA 设备索引。 |
| cuda_device_name | runtime | none | true | false | false | 按实际 CUDA 设备索引读取的 GPU 名称。 |
| cuda_device_capability | runtime | none | true | false | false | 实际 CUDA 设备的 major/minor compute capability。 |
| scientific_unit_provenance_reference_count | metric | none | true | false | false | 最终 records 或 feature rows 对完成单元来源记录的引用数量。 |
| scientific_unit_provenance_record_count | metric | none | true | false | false | 按 scientific_unit_id 去重后的真实完成单元来源记录数量。 |
| scientific_unit_provenance_records_digest | provenance | none | true | false | false | 按 scientific_unit_id 排序的全部去重来源记录稳定摘要。 |
| scientific_unit_ids | provenance | none | true | false | false | 最终汇总实际覆盖的全部科学完成单元标识集合。 |
| scientific_unit_config_digests | provenance | none | true | false | false | 最终汇总实际覆盖的全部完成单元配置摘要集合。 |
| scientific_execution_environment_digests | provenance | none | true | false | false | 跨 Colab 会话实际出现的执行环境摘要集合。 |
| scientific_dependency_profile_ids | provenance | none | true | false | false | 跨完成单元实际使用的依赖 profile 标识集合。 |
| scientific_dependency_profile_digests | provenance | none | true | false | false | 跨完成单元实际使用的依赖 profile 摘要集合。 |
| scientific_complete_hash_lock_digests | provenance | none | true | false | false | 跨完成单元实际使用的完整依赖哈希锁摘要集合。 |
| scientific_formal_execution_commits | provenance | none | true | false | false | 跨完成单元实际使用的40位正式 Git commit 集合。 |
| scientific_formal_execution_lock_digests | provenance | none | true | false | false | 跨完成单元实际使用的正式代码执行锁摘要集合。 |
| scientific_torch_versions | runtime | none | true | false | false | 跨完成单元实际加载的 PyTorch 版本集合。 |
| scientific_torch_cuda_versions | runtime | none | true | false | false | 跨完成单元实际加载的 PyTorch CUDA build 版本集合。 |
| scientific_execution_device_names | runtime | none | true | false | false | 跨完成单元实际使用的 CUDA device 字符串集合。 |
| scientific_cuda_device_names | runtime | none | true | false | false | 跨完成单元实际使用的 GPU 型号名称集合。 |
| scientific_random_identity_digests_random | random | _random | true | false | false | 跨完成单元实际随机性身份摘要集合。 |
| scientific_unit_provenance_ready | governance | none | true | false | false | 所有引用是否存在有效来源记录且同一 scientific_unit_id 不含冲突内容。 |
| scientific_unit_provenance_identity_ready | governance | none | true | false | false | Inception 特征来源是否同时绑定当前代码锁和 `sd35_method_runtime_gpu` 完整依赖锁。 |
| environment_report_path | artifact | none | true | false | false | 指向完整 runtime environment report JSON 的受治理路径。 |
| geometry_manifest_digest | artifact | none | true | false | false | 真实 attention 捕获运行引用的几何 manifest 稳定摘要。 |
| elapsed_seconds | runtime | none | true | false | false | 真实推理耗时秒数。 |
| error_message | runtime | none | false | false | false | 真实后端不可用时的错误消息。 |
| image_path | runtime | none | true | false | false | 真实推理输出图像的受治理路径。 |
| archive_name | artifact | none | false | false | false | 真实 runtime 产物 zip 文件名。 |
| archive_path | artifact | none | true | false | false | 真实 runtime 产物 zip 在 outputs 下的受治理路径。 |
| archive_digest | artifact | none | true | false | false | 真实 runtime 产物 zip 的 SHA-256 摘要。 |
| archive_payload_digest | artifact | none | false | false | false | 打包输入条目内容与路径的稳定摘要, 用于避免 zip 自引用摘要问题。 |
| archive_digest_scope | governance | none | false | false | false | archive digest 字段对应的摘要边界, 例如最终 zip 文件或外部 sidecar。 |
| archive_entry_count | artifact | none | true | false | false | 真实 runtime 产物 zip 中包含的文件数量。 |
| entry_payload_digest | artifact | none | false | false | false | package input manifest 记录的打包输入条目稳定摘要。 |
| final_archive_digest_available_in_sidecar | governance | none | false | false | false | 最终 zip 文件 SHA-256 是否写入同目录 sidecar summary 与 manifest。 |
| drive_output_dir | artifact | none | false | false | false | Colab 镜像保存到 Google Drive 的目标目录。 |
| drive_archive_path | artifact | none | true | false | false | Colab 镜像保存到 Google Drive 后的 zip 路径。 |
| drive_archive_digest | artifact | none | true | false | false | Google Drive 镜像 zip 的 SHA-256 摘要。 |
| injection_id | method | none | true | false | false | 最小 latent injection 运行的稳定标识。 |
| run_decision | runtime | none | true | false | false | 最小 latent injection 运行是否形成可审计结果。 |
| clean_image_path | artifact | none | true | false | false | clean paired image 在 outputs 下的受治理路径。 |
| watermarked_image_path | artifact | none | true | false | false | watermarked paired image 在 outputs 下的受治理路径。 |
| clean_image_digest | artifact | none | true | false | false | clean paired image 的 SHA-256 摘要。 |
| watermarked_image_digest | artifact | none | true | false | false | watermarked paired image 的 SHA-256 摘要。 |
| latent_update_count | method | none | true | false | false | 最小 latent injection 运行中实际执行的 latent update 数量。 |
| update_index | method | none | true | false | false | latent update 在注入集合中的顺序索引。 |
| injection_strength | method | none | true | false | false | 注入到 latent 的 carrier 扰动强度。 |
| injection_step_indices | method | none | true | false | false | 执行 latent injection 的采样位置集合。 |
| latent_digest_before | method | none | true | false | false | latent injection 前的 latent tensor 稳定摘要。 |
| latent_digest_after | method | none | true | false | false | latent injection 后的 latent tensor 稳定摘要。 |
| update_norm | method | none | true | false | false | 注入扰动 tensor 的二范数。 |
| latent_norm_before | method | none | true | false | false | latent injection 前 latent tensor 的二范数。 |
| latent_norm_after | method | none | true | false | false | latent injection 后 latent tensor 的二范数。 |
| relative_update_norm | method | none | true | false | false | update_norm 与 latent_norm_before 的比值。 |
| psnr | metric | none | true | false | false | paired image 的 PSNR 指标。 |
| ssim | metric | none | true | false | false | paired image 的全图轻量 SSIM 指标。 |
| mse | metric | none | true | false | false | paired image 的均方误差。 |
| mean_abs_error | metric | none | true | false | false | paired image 的平均绝对误差。 |
| workflow_name | governance | none | true | false | false | Colab Drive workflow 的语义名称。 |
| workflow_decision | governance | none | true | false | false | Colab Drive workflow 的可审计通过状态。 |
| drive_root | artifact | none | true | false | false | Google Drive 或本地镜像根目录。 |
| drive_workflow_dir | artifact | none | true | false | false | Drive 中保存 workflow manifest 与镜像摘要的目录。 |
| drive_local_output_dir | artifact | none | true | false | false | Drive 中保存本地 outputs 镜像文件的目录。 |
| local_output_dir | artifact | none | true | false | false | Colab Drive workflow 在 outputs 下写入报告的目录。 |
| local_manifests | artifact | none | true | false | false | 被 workflow 发现并登记的本地 manifest 引用集合。 |
| local_manifest_count | artifact | none | true | false | false | 被 workflow 发现的本地 manifest 数量。 |
| manifest_path | artifact | none | true | false | false | manifest 文件路径。 |
| manifest_digest | artifact | none | true | false | false | manifest 文件或 manifest payload 的稳定摘要。 |
| output_count | artifact | none | true | false | false | 单个 manifest 登记的输出路径数量。 |
| dependency_decision | runtime | none | true | false | false | Colab workflow 依赖检查是否通过。 |
| dependency_count | runtime | none | true | false | false | Colab workflow 检查的依赖数量。 |
| missing_dependency_count | runtime | none | true | false | false | Colab workflow 缺失依赖数量。 |
| dependency_name | runtime | none | true | false | false | Colab workflow 检查的依赖名称。 |
| dependencies | runtime | none | true | false | false | Colab workflow 依赖检查记录集合。 |
| module_available | runtime | none | true | false | false | Colab workflow 中单个依赖是否可导入。 |
| installed_version | runtime | none | true | false | false | Colab workflow 中单个依赖的已安装版本。 |
| dependency_report | runtime | none | true | false | false | Colab workflow 的依赖快照报告。 |
| dependency_report_digest | artifact | none | false | false | false | Colab workflow 依赖报告 payload 摘要。 |
| mount_decision | runtime | none | true | false | false | Google Drive 挂载动作或跳过动作的状态。 |
| mount_point | runtime | none | true | false | false | Google Drive 挂载点。 |
| mounted | runtime | none | true | false | false | Google Drive 是否被当前 workflow 视为已挂载。 |
| mount_report_digest | artifact | none | false | false | false | Google Drive 挂载报告 payload 摘要。 |
| sync_report_digest | artifact | none | false | false | false | 本地 outputs 镜像报告 payload 摘要。 |
| source_path | artifact | none | true | false | false | 被镜像文件在仓库中的来源路径。 |
| destination_path | artifact | none | true | false | false | 被镜像文件在 Drive workflow 目录中的目标路径。 |
| file_digest | artifact | none | true | false | false | 被镜像文件的 SHA-256 摘要。 |
| byte_count | artifact | none | true | false | false | 被镜像文件的字节数。 |
| copy_decision | artifact | none | true | false | false | 单个文件镜像操作是否通过。 |
| mirrored_files | artifact | none | true | false | false | 已镜像文件记录集合。 |
| mirrored_file_count | artifact | none | true | false | false | 已镜像文件数量。 |
| input_file_count | artifact | none | true | false | false | workflow 本次登记的输入文件数量。 |
| input_manifest_digest | artifact | none | false | false | false | Drive input manifest payload 摘要。 |
| output_manifest_digest | artifact | none | false | false | false | Drive output manifest payload 摘要。 |
| reload_decision | artifact | none | true | false | false | 根据 Drive manifest 重载校验的通过状态。 |
| verified_file_count | artifact | none | true | false | false | 重载校验中摘要匹配的文件数量。 |
| missing_input_count | artifact | none | true | false | false | 重载校验中缺失的登记文件数量。 |
| missing_input_paths | artifact | none | true | false | false | 重载校验中缺失的登记文件路径集合。 |
| digest_mismatch_count | artifact | none | true | false | false | 重载校验中摘要不一致的文件数量。 |
| prompt_id | protocol | none | true | false | false | prompt 协议记录的稳定标识。 |
| prompt_set | protocol | none | true | false | false | prompt 所属集合, 例如 probe_paper、pilot_paper 或 full_paper。 |
| prompt_index | protocol | none | true | false | false | prompt 在所属集合中的稳定序号。 |
| prompt_text | protocol | none | true | false | false | 规范化后的 prompt 文本。 |
| risk_profile | method | none | true | false | false | 由 prompt 语义标签派生的轻量风险配置名称。 |
| split_names | protocol | none | false | false | false | split manifest 中登记的划分名称集合。 |
| split_counts | protocol | none | false | false | false | split manifest 或统计摘要中的各 split 样本数量。 |
| split_prompt_ids | protocol | none | false | false | false | split manifest 中按 split 聚合的 prompt_id 集合。 |
| sample_role | protocol | none | true | false | false | 事件协议中样本角色, 例如 positive_source、clean_negative 或 attacked_negative。 |
| event_id | protocol | none | true | false | false | 事件协议记录的稳定标识。 |
| event_family | protocol | none | true | false | false | 事件所属机制族, 例如 watermark_embedding、clean_generation 或 attack_generation。 |
| attack_family | protocol | none | true | false | false | 事件协议登记的攻击族。 |
| protocol_decision | governance | none | true | false | false | 协议记录、manifest 或统计摘要的通过状态。 |
| prompt_count | protocol | none | false | false | false | prompt 协议记录数量。 |
| event_count | protocol | none | false | false | false | 事件协议记录数量。 |
| sample_role_counts | protocol | none | false | false | false | 按 sample_role 聚合的事件数量。 |
| prompt_set_counts | protocol | none | false | false | false | 按 prompt_set 聚合的 prompt 数量。 |
| calibration_test_disjoint | protocol | none | false | false | false | calibration 与 test 的 prompt_id 是否无交叉。 |
| prompt_records | artifact | none | false | false | false | prompt manifest 中嵌入的 prompt 协议记录集合。 |
| event_records | artifact | none | false | false | false | event protocol manifest 中嵌入的事件协议记录集合。 |
| prompt_manifest_digest | artifact | none | false | false | false | prompt manifest payload 的稳定摘要。 |
| split_manifest_digest | artifact | none | false | false | false | split manifest payload 的稳定摘要。 |
| event_protocol_digest | artifact | none | false | false | false | event protocol manifest payload 的稳定摘要。 |
| prompt_statistics_digest | artifact | none | false | false | false | prompt statistics payload 的稳定摘要。 |
| source_archive | artifact | none | false | false | false | prompt bank 导入摘要中的来源 archive 路径。 |
| source_archive_digest | artifact | none | false | false | false | prompt bank 来源 archive 的 SHA-256 摘要。 |
| prompt_counts | protocol | none | false | false | false | prompt bank 导入摘要中按 prompt_set 聚合的 prompt 数量。 |
| sanitized_prompt_counts | protocol | none | false | false | false | prompt bank 导入摘要中因治理规则被替换的 prompt 数量。 |
| route_record_id | protocol | none | true | false | false | 语义路由 record 的稳定标识。 |
| subspace_plan_id | protocol | none | true | false | false | 安全子空间 plan record 的稳定标识。 |
| route_id | method | none | true | false | false | 语义条件路由对象的稳定标识。 |
| route_label | method | none | true | false | false | 语义条件路由的机制标签。 |
| route_digest | method | none | true | false | false | 语义条件路由 payload 的稳定摘要。 |
| risk_field_digest | method | none | true | false | false | 语义风险场 payload 的稳定摘要。 |
| mask_source | method | none | true | false | false | 语义掩码来源类型。 |
| mask_source_digest | method | none | true | false | false | 语义掩码来源和掩码值的稳定摘要。 |
| latent_mask_values | method | none | true | false | false | 投影到 latent 长度后的语义掩码值。 |
| masked_feature_values | method | none | true | false | false | 语义掩码作用后归一化的 latent 特征值。 |
| latent_mask_digest | method | none | true | false | false | latent mask 投影结果的稳定摘要。 |
| feature_operator_digest | method | none | true | false | false | latent feature operator 的稳定摘要。 |
| trajectory_feature_digest | method | none | true | false | false | 轨迹特征 payload 的稳定摘要。 |
| approximate_jvp_values | method | none | true | false | false | 近似 JVP 的逐轴数值。 |
| approximate_jvp_digest | method | none | true | false | false | 近似 JVP payload 的稳定摘要。 |
| jvp_estimator_name | method | none | true | false | false | 近似 JVP 估计器名称。 |
| basis_strategy | method | none | true | false | false | 安全基底求解策略。 |
| semantic_mask_enabled | method | none | true | false | false | 安全基底求解是否启用语义掩码。 |
| basis_digests | artifact | none | false | false | false | 各基底策略的摘要集合。 |
| route_projection_digest | method | none | true | false | false | 安全基底按语义路由投影后的稳定摘要。 |
| semantic_route_record_count | method | none | false | false | false | 语义路由 records 数量。 |
| subspace_plan_record_count | method | none | false | false | false | 安全子空间 plan records 数量。 |
| mask_projection_report_count | method | none | false | false | false | mask projection report 数量。 |
| unique_route_digest_count | method | none | false | false | false | 唯一语义路由摘要数量。 |
| semantic_mask_changed_basis_count | method | none | false | false | false | 启用语义掩码后基底相对无语义掩码路径发生变化的记录数量。 |
| basis_strategies | method | none | false | false | false | 当前输出中可运行的基底策略集合。 |
| risk_field_length | method | none | false | false | false | 语义风险场向量长度。 |
| projection_operator | method | none | false | false | false | mask 投影算子名称。 |
| feature_length | method | none | false | false | false | latent feature operator 输出长度。 |
| jvp_value_count | method | none | false | false | false | 近似 JVP 数值数量。 |
| plan_digest | method | none | false | false | false | 安全子空间 plan payload 的稳定摘要。 |
| lf_basis_count | method | none | false | false | false | LF 路由投影后的基底行数。 |
| tail_basis_count | method | none | false | false | false | 高斯幅值尾部截断路由投影后的基底行数。 |
| attention_basis_count | method | none | false | false | false | attention 路由投影后的基底行数。 |
| content_detection_record_id | method | none | true | false | false | 内容检测 record 的稳定标识。 |
| content_mode | method | none | true | false | false | LF/高斯幅值尾部截断内容载体机制开关名称。 |
| mechanism_scores | method | none | true | false | false | 同一观测样本在各内容机制开关下的统一内容分数集合。 |
| lf_enabled | method | none | true | false | false | 内容 update 组合时是否启用 LF 主证据分量。 |
| tail_enabled | method | none | true | false | false | 内容 update 组合时是否启用高斯幅值尾部截断补充分量。 |
| tail_truncation_enabled | method | none | true | false | false | 高斯幅值尾部截断内容载体是否启用截断。 |
| content_update_digest | method | none | true | false | false | LF/高斯幅值尾部截断内容 update 组合 payload 的稳定摘要。 |
| content_chain_digest | method | none | true | false | false | 内容载体链路组合后的稳定摘要。 |
| lf_content_carrier_digest | method | none | true | false | false | LF 内容载体 payload 的稳定摘要。 |
| tail_content_carrier_digest | method | none | true | false | false | 高斯幅值尾部截断内容载体 payload 的稳定摘要。 |
| score_digest | method | none | true | false | false | 统一内容分数 payload 的稳定摘要。 |
| fixed_fpr_ready | method | none | true | false | false | 内容分数是否已保持可进入 fixed-FPR 校准的统计边界。 |
| content_detection_record_count | method | none | false | false | false | 内容检测 records 数量。 |
| content_modes | method | none | false | false | false | 当前内容载体产物支持的机制开关集合。 |
| attention_graph_id | method | none | true | false | false | 注意力锚点图 record 的稳定标识。 |
| stable_token_indices | method | none | true | false | false | 从 attention graph 中选出的稳定 token 索引集合。 |
| relative_relation_values | method | none | true | false | false | 稳定 token 集内部的相对注意力关系权重。 |
| anchor_graph_digest | method | none | true | false | false | 注意力锚点图 payload 的稳定摘要。 |
| attention_relation_consistency | method | none | true | false | false | 稳定 token 双向注意力关系一致性。 |
| geometry_evidence_record_id | method | none | true | false | false | 注意力几何证据 record 的稳定标识。 |
| graph_source | method | none | true | false | false | 注意力图 record 的矩阵来源说明。 |
| geometry_source | method | none | true | false | false | 几何证据 record 的来源说明。 |
| attention_capture_record_count | method | none | false | false | false | attention capture records 数量。 |
| attention_graph_record_count | method | none | false | false | false | 注意力锚点图 records 数量。 |
| geometry_evidence_record_count | method | none | false | false | false | 几何证据 records 数量。 |
| real_attention_capture_count | method | none | false | false | false | 无 unsupported reason 且含有有界 attention_matrix_preview 的真实 attention capture records 数量。 |
| unsupported_capture_count | method | none | false | false | false | 带 unsupported reason 的 attention capture records 数量。 |
| attention_relation_consistency_mean | metric | none | false | false | false | 注意力关系一致性的均值。 |
| anchor_inlier_ratio_mean | metric | none | false | false | false | anchor inlier ratio 的均值。 |
| registration_confidence_mean | metric | none | false | false | false | registration confidence 的均值。 |
| recovered_sync_consistency_mean | metric | none | false | false | false | recovered sync consistency 的均值。 |
| alignment_residual_mean | metric | none | false | false | false | alignment residual 的均值。 |
| geometry_reliable_count | method | none | false | false | false | 几何证据中满足可靠性条件的记录数量。 |
| direct_positive_decision_used | method | none | false | false | false | 几何证据链是否使用了直接 positive 判定。 |
| attention_geometry_ready | method | none | false | false | false | 注意力几何证据是否已具备进入后续真实 attention 相对更新的条件。 |
| attention_records_path | artifact | none | true | false | false | 用于重建注意力几何证据的 attention capture records 输入路径。 |
| attention_matrix_preview | method | none | true | false | false | 真实 attention 捕获中保存的有界 attention matrix 预览。 |
| attention_matrix_source | method | none | true | false | false | 注意力几何重建时使用的矩阵来源, 例如 preview 或 digest replay。 |
| attention_token_indices | method | none | true | false | false | 真实 attention map 预览中保留的 token 索引集合。 |
| capture_tensor_shape | method | none | true | false | false | 真实 attention hook 捕获到的 tensor 形状。 |
| geometry_manifest_path | artifact | none | true | false | false | 真实 attention 捕获运行引用的几何 manifest 路径。 |
| geometry_summary_path | artifact | none | true | false | false | 真实 attention 捕获运行引用的几何 summary 路径。 |
| entry_paths | artifact | none | false | false | false | 压缩包输入 manifest 中登记的入包文件路径集合。 |
| entry_count | artifact | none | false | false | false | 压缩包输入 manifest 中登记的入包文件数量。 |
| attention_geometry_source_path | artifact | none | false | false | false | attention-relative latent update 使用的 ready attention geometry 输入路径。 |
| attention_carrier_record_count | method | none | false | false | false | attention-relative carrier records 数量。 |
| active_update_count | method | none | false | false | false | 通过稳定性边界并执行 active update 的 attention carrier 数量。 |
| evidence_only_count | method | none | false | false | false | 因几何证据不可靠或 update 不稳定而降级为 evidence-only 的 carrier 数量。 |
| attention_update_stable_count | method | none | false | false | false | carrier 级别满足 attention update 稳定性边界的数量。 |
| attention_update_stability_row_count | method | none | false | false | false | attention update 强度稳定性表中的行数。 |
| target_relation_values | method | none | true | false | false | attention-relative update 使用的目标相对关系权重。 |
| baseline_relation_values | method | none | true | false | false | 未执行 attention update 时的相对关系近似。 |
| relation_gradient_values | method | none | true | false | false | 相对关系误差投影到语义安全轴后的可审计梯度近似。 |
| relation_loss_before | method | none | true | false | false | 执行 attention update 前的几何关系损失。 |
| relation_loss_after | method | none | true | false | false | 执行 attention update 后或强度模拟后的几何关系损失。 |
| relation_loss_delta | method | none | true | false | false | attention update 前后几何关系损失下降量。 |
| relation_consistency_before | method | none | true | false | false | 执行 attention update 前的相对关系一致性。 |
| relation_consistency_after | method | none | true | false | false | 执行 attention update 后或强度模拟后的相对关系一致性。 |
| projected_update_norm | method | none | true | false | false | 投影到语义安全轴后的 attention update 二范数。 |
| attention_update_strength | method | none | true | false | false | attention update 强度曲线中的实际强度值。 |
| attention_update_stable | method | none | true | false | false | 单条 carrier 或强度行是否满足 update 稳定性边界。 |
| attention_relative_carrier_digest | method | none | true | false | false | attention-relative carrier 的稳定摘要。 |
| quality_metric_name | metric | none | true | true | false | 数据集级正式质量指标名称, 当前精确取 FID 或 KID。 |
| quality_metric_value | metric | none | true | true | false | 从真实生成图像与真实参考图像特征计算的 FID 或 KID 数值。 |
| image_quality_metrics_ready | method | none | false | false | false | 是否已经完成真实图像质量指标测量。 |
| full_method_claim_ready | governance | none | true | true | false | 主方法真实生成、攻击、仅图像检测和科学算子门禁是否共同支持当前论文层级结论。 |
| selected_attention_carrier_id | method | none | true | false | false | 真实 attention latent injection 中选用的 active carrier 标识。 |
| attention_geometry_package_path | artifact | none | true | false | false | 真实 attention latent injection 使用的 geometry 输入包路径。 |
| method_manifest_path | artifact | none | true | false | false | 真实运行引用的 attention latent update 方法 manifest 路径。 |
| attention_latent_injection_package_path | artifact | none | false | false | false | 真实 attention latent injection 打包产物路径。 |
| attention_latent_injection_package_digest | artifact | none | false | false | false | 真实 attention latent injection 打包产物 SHA256 摘要。 |
| aligned_detection_record_id | method | none | false | false | false | 几何恢复后内容重判记录的稳定标识。 |
| aligned_detection_record_count | metric | none | false | false | false | 几何恢复后内容重判记录数量。 |
| rescue_ablation_mode | method | none | false | false | false | 几何 rescue 消融模式, 用于区分完整 rescue、无 rescue、无 attention anchor 和反例审计。 |
| rescue_score_gain | method | none | false | false | false | 几何对齐后内容分数相对原始内容分数的增益。 |
| rescue_score_gain_mean | metric | none | false | false | false | 几何对齐后内容分数增益均值。 |
| raw_content_positive_count | metric | none | false | false | false | 原始内容分支正判记录数量。 |
| raw_content_failed_count | metric | none | false | false | false | 原始内容分支未正判记录数量。 |
| rescue_eligible_count | metric | none | false | false | false | 满足 rescue 触发前置条件的记录数量。 |
| rescue_applied_count | metric | none | false | false | false | 实际完成 rescue 并通过同阈值重判的记录数量。 |
| full_rescue_record_count | metric | none | false | false | false | full_rescue 消融模式下的记录数量。 |
| full_rescue_applied_count | metric | none | false | false | false | full_rescue 消融模式下实际完成 rescue 的记录数量。 |
| raw_content_clean_fpr | metric | none | false | false | false | clean negative 上原始内容分支误报率。 |
| formal_detection_score_clean_fpr | metric | none | false | false | false | clean negative 上正式判定分数的误报率。 |
| formal_detection_score_attacked_fpr | metric | none | false | false | false | attacked negative 上正式判定分数的诊断误报率。 |
| evidence_clean_fpr | metric | none | false | false | false | clean negative 上 rescue 后 evidence-level 误报率。 |
| evidence_attacked_fpr | metric | none | false | false | false | attacked negative 上 rescue 后 evidence-level 误报率。 |
| geo_direct_positive_audit_decision | method | none | false | false | false | 仅用于反例审计的几何直接判正风险指示, 不进入正式 evidence decision。 |
| geo_direct_positive_audit_rate | metric | none | false | false | false | clean negative 上几何直接判正反例审计触发率。 |
| geo_direct_positive_audit_formal_method | method | none | false | false | false | 几何直接判正反例审计是否进入正式方法, 正式方法中必须为 false。 |
| direct_positive_decision_used | method | none | false | false | false | 是否在正式 evidence decision 中使用几何直接判正。 |
| operating_point_id | metric | none | false | false | false | fixed-FPR operating point 的稳定标识。 |
| target_fpr | protocol | none | false | false | false | 阈值校准协议的目标误报率。 |
| calibrated_content_threshold | protocol | none | false | false | false | 由 calibration clean negative 冻结的内容阈值。 |
| calibrated_detection_threshold | protocol | none | false | false | false | 由 calibration clean negative 在 threshold_score_field 指定分数空间中冻结的正式检测阈值。 |
| formal_detection_claim_ready | governance | none | false | false | false | 正式判定分数是否同时满足校准 split fixed-FPR 边界和测试 clean negative 经验 FPR 诊断边界。 |
| calibration_negative_count | metric | none | false | false | false | 用于阈值冻结的 calibration clean negative 样本数。 |
| allowed_false_positive_count | metric | none | false | false | false | 目标 FPR 下允许的 false positive 数量。 |
| nominal_allowed_false_positive_count | metric | none | false | false | false | 按 floor(target_fpr * negative_count) 得到的名义 false positive 预算。 |
| confidence_controlled_false_positive_count | metric | none | false | false | false | 在 calibration FPR 置信上界不超过目标 FPR 时允许的保守 false positive 预算。 |
| false_positive_budget_mode | protocol | none | false | false | false | fixed-FPR 阈值冻结使用的 false positive 预算模式, 例如 empirical 或 confidence_controlled。 |
| calibration_confidence_level | protocol | none | false | false | false | calibration FPR 置信上界使用的单侧置信水平。 |
| calibration_fpr_confidence_upper_bound | metric | none | false | false | false | calibration split false positive rate 的单侧置信上界。 |
| calibration_confidence_boundary_ready | governance | none | false | false | false | calibration FPR 置信上界是否不超过目标 FPR, 用于 full_paper 论文级 fixed-FPR 声明门禁。 |
| observed_false_positive_count | metric | none | false | false | false | 阈值冻结数据上的实际 false positive 数量。 |
| observed_fpr | metric | none | false | false | false | 阈值冻结数据上的实际 FPR。 |
| threshold_tie_count | metric | none | false | false | false | 与冻结阈值数值相同的样本数量。 |
| threshold_degenerate | metric | none | false | false | false | 阈值是否存在退化或并列导致的 FPR 风险。 |
| threshold_source | protocol | none | false | false | false | 阈值来源。主方法使用 calibration clean negative 联合冻结完整 evidence 协议, 正式 FPR 由独立 test clean negative 经验值及单侧 Wilson 上界支持; 外部 baseline 可使用其受审计的固定 score-map conformal 阈值。 |
| evaluation_split | protocol | none | false | false | false | 正式结果指标所属的数据划分, 论文比较记录必须为 test, 不得混入 calibration 或 dev。 |
| rescue_window_frozen | protocol | none | false | false | false | rescue window 是否已冻结。 |
| fail_reason_gate_frozen | protocol | none | false | false | false | fail reason gate 是否已冻结。 |
| evidence_fpr_exceeds_target | metric | none | false | false | false | 测试 clean negative evidence-level FPR 是否超过目标 operating point, 不包含 attacked negative 诊断分母; 该字段用于经验诊断, 不直接阻断阈值冻结 workflow。 |
| fixed_fpr_control_scope | protocol | none | false | false | false | fixed-FPR 阈值冻结时实际使用的控制样本范围, 正式协议应为 calibration clean negative。 |
| fixed_fpr_denominator_role | protocol | none | false | false | false | fixed-FPR 统计分母的样本角色说明, 当前只允许 clean negative 作为控制分母。 |
| rescue_control_scope | protocol | none | false | false | false | rescue 后 evidence-level FPR 的控制样本范围, 当前为 evidence clean negative。 |
| rescue_changes_fpr_denominator | protocol | none | false | false | false | rescue 是否改变 fixed-FPR 分母, 当前必须为 false。 |
| attacked_negative_boundary_role | governance | none | false | false | false | attacked negative 在 fixed-FPR 协议中的边界角色, 当前仅作为 robustness diagnostic。 |
| attacked_negative_governs_fixed_fpr | governance | none | false | false | false | attacked negative FPR 是否参与 fixed-FPR 阈值控制, 当前必须为 false。 |
| calibration_fpr_exceeds_target | metric | none | false | false | false | calibration clean negative 上冻结阈值的实际 FPR 是否超过目标 FPR, 该字段控制 fixed-FPR 校准门禁。 |
| test_clean_fpr_exceeds_target | metric | none | false | false | false | 测试 clean negative 上 evidence-level FPR 是否超过目标 FPR, 该字段用于经验泛化诊断。 |
| formal_detection_test_clean_fpr_exceeds_target | metric | none | false | false | false | 测试 clean negative 上 formal detection score FPR 是否超过目标 FPR, 该字段用于正式判定分数的经验泛化诊断。 |
| attacked_fpr_diagnostic_exceeds_target | metric | none | false | false | false | attacked negative evidence-level FPR 是否超过目标 FPR, 该字段只作为攻击鲁棒性诊断。 |
| fixed_fpr_boundary_ready | metric | none | false | false | false | fixed-FPR 阈值冻结边界是否未退化、calibration FPR 未超标并可被下游检测协议复用。 |
| rescue_boundary_ready | metric | none | false | false | false | rescue 统计边界是否已冻结且不改变 fixed-FPR 分母; 测试 clean negative FPR 超标仅作为经验诊断。 |
| fixed_fpr_and_rescue_boundary_ready | metric | none | false | false | false | fixed-FPR 阈值冻结边界和 rescue 协议边界是否同时满足当前工程审计要求。 |
| workflow_calibration_ready | governance | none | false | false | false | 阈值校准 workflow 是否满足可继续下游重建的工程门禁。 |
| paper_claim_empirical_fpr_ready | governance | none | false | false | false | 测试 clean negative 经验 FPR 是否足以支持论文声明层面的 fixed-FPR 结论。 |
| statistical_boundary | protocol | none | false | false | false | FPR 审计表中单行记录所属的统计边界名称。 |
| governs_fixed_fpr | governance | none | false | false | false | FPR 审计表中单行记录是否参与 fixed-FPR 控制边界。 |
| raw_content_claim_ready | claim | none | false | false | false | raw content 分支是否满足当前 fixed-FPR 口径。 |
| true_positive_rate | metric | none | false | false | false | positive source 上的 true positive rate。 |
| false_positive_rate | metric | none | false | false | false | 冻结 operating point 下 clean negative 的 false positive rate。 |
| false_negative_rate | metric | none | false | false | false | DET 曲线点中的 false negative rate。 |
| raw_score_auc | metric | none | false | false | false | raw content score 对 positive / clean negative 的 AUC。 |
| aligned_score_auc | metric | none | false | false | false | aligned content score 对 positive / clean negative 的 AUC。 |
| rescue_applied_rate | metric | none | false | false | false | 阈值校准口径下 rescue_applied 的比例。 |
| metric_name | metric | none | false | false | false | 常规指标名称。 |
| metric_value | metric | none | false | false | false | 常规指标数值。 |
| metric_source | metric | none | false | false | false | 常规指标来源。 |
| metric_status | metric | none | false | false | false | 指标状态, 例如 measured 或 unsupported。 |
| fpr_exceeds_target | metric | none | false | false | false | 某一 FPR 审计口径是否超过目标 FPR。 |
| decision_scope | metric | none | false | false | false | FPR 审计中的判定范围。 |
| claim_id | claim | none | false | true | false | claim 审计表中的声明标识。 |
| evidence_path | claim | none | false | true | false | claim 绑定的证据路径。 |
| attack_record_id | protocol | none | true | false | false | 攻击检测 record 的稳定标识。 |
| attack_id | protocol | none | true | false | false | 单个攻击配置的稳定标识。 |
| attack_name | protocol | none | true | false | false | 攻击配置的语义名称。 |
| attack_transform_name | protocol | none | false | false | false | 图像级攻击实际执行的变换参数名称, 用于区分共同协议攻击名称与具体实现参数。 |
| primary_baseline_attacked_image_count | metric | none | false | false | false | 主表 external baseline method-faithful adapter 生成的攻击后图像总数。 |
| attacked_image_count_by_baseline | metric | none | false | false | false | 按 baseline id 聚合的攻击后图像数量。 |
| formal_image_attack_families | protocol | none | false | false | false | method-faithful adapter 默认覆盖的正式图像级攻击名称列表。 |
| attack_strength | protocol | none | true | false | false | 攻击配置使用的归一化强度。 |
| resource_profile | protocol | none | true | false | false | 攻击矩阵运行使用的资源档位, 例如 probe、full_main 或 full_extra。 |
| requires_gpu | protocol | none | true | false | false | 攻击配置是否需要真实 GPU 推理或重生成能力。 |
| attack_parameters | protocol | none | true | false | false | 攻击配置的具体参数字典。 |
| attack_config_digest | artifact | none | true | false | false | 攻击配置 payload 的稳定摘要。 |
| attack_record_digest | artifact | none | true | false | false | 攻击检测 record payload 的稳定摘要。 |
| source_record_id | protocol | none | true | false | false | 攻击记录引用的源检测 record 标识。 |
| source_image_digest | artifact | none | true | false | false | 真实源图像文件的稳定摘要。 |
| source_image_digest_source | artifact | none | true | false | false | source image digest 的来源说明。 |
| attacked_image_digest | artifact | none | true | false | false | 真实攻击后图像文件的稳定摘要。 |
| attacked_image_digest_source | artifact | none | true | false | false | attacked image digest 的来源说明。 |
| attacked_image_available | artifact | none | true | false | false | 是否存在真实可读取的攻击后图像文件。 |
| attack_performed | protocol | none | true | false | false | 当前记录是否实际执行了本地可用攻击路径。 |
| real_attack_record_id | protocol | none | true | false | false | 真实图像级攻击闭环中单条记录的稳定标识。 |
| source_image_id | artifact | none | true | false | false | 真实 source image 文件的稳定标识。 |
| source_image_path | artifact | none | true | false | false | 真实 source image 文件的受治理路径。 |
| attacked_image_path | artifact | none | true | false | false | 真实 attacked image 文件的受治理路径。 |
| attack_implementation | protocol | none | true | false | false | 真实图像级攻击运行使用的具体 pipeline 或算子机制。 |
| attack_execution | protocol | none | true | false | false | 单次攻击的冻结参数、随机种子、mask 与检测器查询轨迹。 |
| attack_seed_random | random | none | true | false | false | 生成单次随机攻击变换的可复现种子。 |
| effective_parameters | protocol | none | true | false | false | 单次攻击实际消费的冻结参数集合。 |
| local_edit_mask_digest | artifact | none | true | false | false | 局部 inpainting 白色编辑 mask 的 SHA-256 摘要。 |
| local_edit_mask_area_ratio | metric | none | true | false | false | 局部 inpainting 白色编辑区域占整幅图像的实际面积比例。 |
| detector_query_trace | protocol | none | true | false | false | 检测器引导去水印中源图像与全部候选的强度、种子和实测分数序列。 |
| candidate_seed_random | random | none | true | false | false | 检测器引导去水印单个候选使用的可复现种子。 |
| image_only_detection | metric | none | true | false | false | T2SMark 对真实 clean/watermarked 图像使用同一正式密钥得到的仅图像检测分数对象。 |
| clean_score | metric | none | true | false | false | 真实 clean negative 图像在正式检测密钥下的连续分数。 |
| watermarked_score | metric | none | true | false | false | 真实 watermarked positive 图像在正式检测密钥下的连续分数。 |
| detection_score | metric | none | true | false | false | 单幅候选或攻击后图像由对应方法真实检测器计算的连续分数。 |
| detection_method | method | none | true | false | false | 攻击后重跑检测时使用的受治理检测方法名称。 |
| detection_threshold | protocol | none | true | false | false | 攻击后重跑检测使用的检测阈值。 |
| attacked_image_registry_path | artifact | none | false | false | false | 真实 attacked image 注册表 JSONL 文件路径。 |
| attack_family_metrics_path | artifact | none | false | false | false | 真实图像级攻击分组指标表路径。 |
| real_attack_record_count | metric | none | false | false | false | 真实图像级攻击检测记录数量。 |
| real_attacked_image_count | metric | none | false | false | false | 已生成并登记 digest 的真实 attacked image 文件数量。 |
| regeneration_attack_record_count | metric | none | false | false | false | 再扩散类攻击检测记录数量。 |
| required_regeneration_attack_count | metric | none | false | false | false | 证据门禁要求的真实 GPU 攻击类型数量。 |
| measured_regeneration_attack_count | metric | none | false | false | false | 已在真实 GPU workflow 中完成测量的攻击类型数量。 |
| real_attacked_image_closed_loop_ready | metric | none | false | false | false | 真实 source / attacked image 文件、路径和 digest 是否完成闭环。 |
| regeneration_attack_gpu_validation_ready | metric | none | false | false | false | img2img、flow-matching inversion、SDEdit 和 diffusion purification 是否已由真实 GPU workflow 生成并测量。 |
| attack_detection_rerun_ready | metric | none | false | false | false | 真实 attacked image 生成后是否已重跑攻击后检测记录。 |
| formal_attack_detection_ready | metric | none | false | false | false | 真实 attacked image 是否已经转换为 attack matrix 兼容正式检测记录。 |
| formal_records_path | artifact | none | false | false | false | 真实攻击闭环写出的 attack matrix 兼容检测记录 JSONL 路径。 |
| aligned_rescoring_drive_dir | artifact | none | false | false | false | Colab workflow 查找前序 aligned rescoring 结果包的 Google Drive 目录。 |
| threshold_calibration_drive_dir | artifact | none | false | false | false | Colab workflow 查找 fixed-FPR 阈值校准结果包的 Google Drive 目录。 |
| threshold_calibration_package_path | artifact | none | false | false | false | 被解包为正式检测边界输入的 threshold calibration 结果包路径。 |
| threshold_calibration_package_digest | artifact | none | false | false | false | threshold calibration 结果包 SHA256 摘要。 |
| formal_boundary_ready | metric | none | true | false | false | 单条真实攻击正式检测记录是否已读取 fixed-FPR 与 rescue 边界。 |
| measured_record_count | metric | none | false | false | false | 单个攻击分组中已实测的记录数量。 |
| detection_positive_rate | metric | none | false | false | false | 单个攻击分组中重跑检测判正的比例。 |
| raw_content_score_before | metric | none | true | false | false | 攻击前 raw content score。 |
| raw_content_score_after | metric | none | true | false | false | 攻击后 raw content score。 |
| aligned_content_score_before | metric | none | true | false | false | 攻击前 aligned content score。 |
| aligned_content_score_after | metric | none | true | false | false | 攻击后 aligned content score。 |
| lf_score_retention | metric | none | true | false | false | LF 内容分数在攻击后的保持率。 |
| tail_score_retention | metric | none | true | false | false | 高斯幅值尾部截断内容分数在攻击后的保持率。 |
| score_retention | metric | none | true | false | false | 统一内容分数在攻击后的保持率。 |
| attack_record_count | metric | none | false | false | false | 攻击矩阵检测 records 数量。 |
| attack_config_count | metric | none | false | false | false | 攻击矩阵配置数量。 |
| attack_family_count | metric | none | false | false | false | 攻击矩阵中唯一攻击族数量。 |
| supported_record_count | metric | none | false | false | false | 具备真实正式证据并可进入当前论文表格统计的记录数量。 |
| unsupported_record_count | metric | none | false | false | false | 因真实资源或输入缺失而不能进入本地统计的记录数量。 |
| gpu_attack_unsupported_count | metric | none | false | false | false | 需要真实 GPU 但当前尚未由真实 formal records 覆盖的攻击类型数量。 |
| attack_manifest_path | artifact | none | true | false | false | 攻击矩阵专用 manifest 路径。 |
| attack_metrics_ready | artifact | none | false | false | false | 完整正式攻击矩阵的真实检测指标是否可从记录重建。 |
| clean_false_positive_rate | metric | none | false | false | false | clean negative 在攻击矩阵统计中的 false positive rate。 |
| attacked_false_positive_rate | metric | none | false | false | false | attacked negative 在攻击矩阵统计中的 false positive rate。 |
| score_retention_mean | metric | none | false | false | false | 单一检测器内部的攻击前后分数稳定性诊断均值, 合法范围为 [0,1], 不用于跨方法正式比较. |
| geometry_reliable_rate | metric | none | false | false | false | 攻击分组内几何可靠记录比例。 |
| rescue_rate | metric | none | false | false | false | 攻击分组内 rescue_applied 比例。 |
| input_manifests | artifact | none | false | false | false | 攻击矩阵或重建 manifest 引用的输入 manifest 集合。 |
| input_records_path | artifact | none | true | false | false | 攻击矩阵重建读取的源 records 路径。 |
| input_thresholds_path | artifact | none | true | false | false | 攻击矩阵重建读取的 fixed-FPR 阈值文件路径。 |
| input_threshold_report_path | artifact | none | true | false | false | 攻击矩阵重建读取的阈值边界报告路径。 |
| real_attack_records_path | artifact | none | false | false | false | 攻击矩阵可选读取的真实 attacked image formal records JSONL 路径。 |
| formal_real_attack_record_count | metric | none | false | false | false | 已并入攻击矩阵的真实 attacked image formal detection record 数量。 |
| performed_attack_record_count | metric | none | false | false | false | 已由真实攻击图像与真实仅图像检测闭环产生的记录数量。 |
| gpu_attack_real_measurement_missing_count | metric | none | false | false | false | 默认再扩散攻击清单中尚未由真实 GPU formal records 覆盖的攻击类型数量。 |
| resource_profiles | protocol | none | false | false | false | 攻击矩阵中出现的资源档位集合。 |
| conventional_attack_names | protocol | none | false | false | false | 当前攻击矩阵登记的常规攻击名称集合。 |
| regeneration_attack_names | protocol | none | false | false | false | 当前攻击矩阵登记的再扩散攻击名称集合。 |
| real_regeneration_attack_names | protocol | none | false | false | false | 已由真实 attacked image formal records 覆盖的再扩散攻击名称集合。 |
| required_real_gpu_attack_count | metric | none | true | false | false | 共同攻击矩阵中需要真实 GPU 图像攻击闭环覆盖的攻击名称数量。|
| measured_real_gpu_attack_count | metric | none | true | false | false | 已由真实 GPU 图像攻击 formal records 覆盖的攻击名称数量。|
| real_gpu_attack_validation_ready | governance | none | true | false | false | 再扩散、高级编辑和自适应去水印等真实 GPU 攻击是否已全部覆盖。|
| real_gpu_attack_names | protocol | none | true | false | false | 已由真实 GPU attacked image formal records 覆盖的攻击名称集合。|
| evaluation_boundary | protocol | none | false | false | false | 攻击后检测复用的 fixed-FPR 与 rescue 统计边界。 |
| regeneration_attack_status | governance | none | false | false | false | 再扩散攻击是否已有真实产物支持的状态说明。 |
| source_supports_paper_claim | claim | none | true | false | false | 源记录是否支持论文主张的继承状态。 |
| score_retention_min | metric | none | false | false | false | 攻击分组内 score retention 最小值。 |
| score_retention_max | metric | none | false | false | false | 攻击分组内 score retention 最大值。 |
| lf_score_retention_mean | metric | none | false | false | false | 攻击分组内 LF score retention 均值。 |
| tail_score_retention_mean | metric | none | false | false | false | 攻击分组内高斯幅值尾部截断 score retention 均值。 |
| positive_count | metric | none | false | false | false | 当前攻击设置下完整 test split positive 记录数量。 |
| negative_count | metric | none | false | false | false | 用于固定 FPR 评估的完整 test split clean negative 记录数量。 |
| baseline_id | protocol | none | true | false | false | 外部 baseline 的稳定语义标识。 |
| baseline_family | protocol | none | true | false | false | 外部 baseline 所属方法族。 |
| baseline_name | protocol | none | true | false | false | 外部 baseline 的论文或方法显示名称。 |
| comparison_group | protocol | none | true | false | false | 外部 baseline 在主表或补充表中的对比分组。 |
| expected_input_mode | protocol | none | true | false | false | 外部 baseline 预期输入模式, 例如 diffusion latent 或 image space。 |
| baseline_observation_id | protocol | none | true | false | false | 外部 baseline 在共同协议边界下的观测记录标识。 |
| baseline_observation_digest | artifact | none | true | false | false | 外部 baseline 观测记录 payload 的稳定摘要。 |
| comparable_operating_point | protocol | none | true | false | false | 外部 baseline 与当前方法对齐使用的可比较 operating point。 |
| common_prompt_protocol_ready | protocol | none | false | false | false | 是否已将 baseline 对齐到相同 prompt 协议。 |
| common_attack_protocol_ready | protocol | none | false | false | false | 是否已将 baseline 对齐到相同攻击矩阵协议。 |
| common_threshold_protocol_ready | protocol | none | false | false | false | 是否已将 baseline 对齐到相同 fixed-FPR 或可比较阈值协议。 |
| baseline_protocol_compatible | protocol | none | false | false | false | baseline adapter 是否可登记到当前共同协议边界。 |
| baseline_requires_gpu | runtime | none | false | false | false | 外部 baseline 是否需要 GPU 才能复现实验结果。 |
| baseline_requires_training | runtime | none | false | false | false | 外部 baseline 是否需要训练或微调才可复现实验结果。 |
| baseline_adapter_ready | runtime | none | false | false | false | 仓库是否已有该外部 baseline 的协议 adapter。 |
| baseline_official_code_ready | runtime | none | false | false | false | 外部 baseline 官方代码是否已接入并通过本项目协议检查。 |
| baseline_reproduced_result_ready | runtime | none | false | false | false | 外部 baseline 是否已有本项目内复现实验结果。 |
| baseline_imported_result_ready | runtime | none | false | false | false | 外部 baseline 是否已有受治理导入结果。 |
| baseline_result_source | artifact | none | false | false | false | 外部 baseline 指标结果来源。 |
| baseline_count | metric | none | false | false | false | 外部 baseline 对比清单中的 baseline 数量。 |
| baseline_observation_count | metric | none | false | false | false | 外部 baseline 观测记录数量。 |
| comparable_baseline_count | metric | none | false | false | false | 可登记到共同协议边界的外部 baseline 数量。 |
| baseline_result_ready_count | metric | none | false | false | false | 已具备复现或导入结果的外部 baseline 数量。 |
| baseline_result_ready | metric | none | false | false | false | 外部 baseline 对比 manifest 中的单数结果就绪状态。 |
| baseline_results_ready | metric | none | false | false | false | 外部 baseline 对比运行中所有 baseline 结果是否就绪。 |
| comparison_protocol_ready | protocol | none | false | false | false | 外部 baseline 对比所需共同协议边界是否已经可审计。 |
| attack_manifest_supports_paper_claim | claim | none | false | false | false | 外部 baseline 对比输入的攻击矩阵 manifest 是否支持论文主张。 |
| method_id | protocol | none | true | false | false | 对比表中的方法标识。 |
| method_role | protocol | none | true | false | false | 对比表中的方法角色, 精确区分 proposed_method 与 primary_baseline。 |
| comparison_scope | protocol | none | true | false | false | 对比表行对应的统计或协议范围。 |
| ablation_record_id | protocol | none | true | false | false | 内部消融 record 的稳定标识。 |
| ablation_record_digest | artifact | none | true | false | false | 内部消融 record payload 的稳定摘要。 |
| ablation_id | protocol | none | true | false | false | 内部机制消融配置的稳定标识。 |
| ablation_name | protocol | none | true | false | false | 内部机制消融配置的显示名称。 |
| mechanism_group | protocol | none | true | false | false | 消融配置所属机制组。 |
| ablated_mechanism | protocol | none | true | false | false | 被关闭或替换的具体机制。 |
| mechanism_change | protocol | none | true | false | false | 消融配置实际施加的机制改变。 |
| mechanism_change_digest | artifact | none | true | false | false | 消融机制改变配置的稳定摘要。 |
| ablated_evidence_decision | metric | none | true | false | false | 消融后在相同 fixed-FPR 边界下的 evidence 判定。 |
| ablated_detection_decision | metric | none | true | false | false | 消融后同时考虑 attestation gate 的最终检测判定。 |
| ablated_score_retention | metric | none | true | false | false | 消融后重新计算的 score retention。 |
| ablated_lf_score_retention | metric | none | true | false | false | 消融后重新计算的 LF score retention。 |
| ablated_tail_score_retention | metric | none | true | false | false | 消融后重新计算的高斯幅值尾部截断 score retention。 |
| ablated_geometry_reliable | metric | none | true | false | false | 消融后重新计算的几何可靠性。 |
| ablated_rescue_applied | metric | none | true | false | false | 消融后重新计算的 rescue 触发状态。 |
| ablated_raw_content_score_after | metric | none | true | false | false | 消融后攻击后的 raw content score。 |
| ablated_aligned_content_score_after | metric | none | true | false | false | 消融后攻击后的 aligned content score。 |
| attestation_required | claim | none | true | false | false | 消融记录是否要求 attestation gate。 |
| attestation_available | claim | none | true | false | false | 消融记录是否具备 attestation。 |
| attestation_available_rate | metric | none | false | false | false | 消融分组内 attestation 可用比例。 |
| formal_method_allowed | claim | none | true | false | false | 消融配置是否允许被视为正式方法候选。 |
| claim_status | claim | none | true | false | false | 消融记录的论文主张支持边界。 |
| mechanism_explanation | claim | none | true | false | false | 消融配置对应的机制解释。 |
| expected_failure_mode | claim | none | true | false | false | 消融配置预期暴露的失效模式。 |
| ablation_count | metric | none | false | false | false | 内部消融配置数量。 |
| ablation_record_count | metric | none | false | false | false | 内部消融 records 数量。 |
| ablation_claim_input_record_count | metric | none | false | false | false | 进入正式 standalone ablation claim 的 formal attacked image watermark rescore 记录数量。 |
| ablation_claim_total_source_record_count | metric | none | false | false | false | 内部消融输入筛选前的源攻击 records 数量。 |
| ablation_claim_excluded_record_count | metric | none | false | false | false | 被正式消融 claim gate 排除的非声明记录数量。 |
| ablation_claim_excluded_record_examples | governance | none | false | false | false | 被正式消融 claim gate 排除的代表性记录示例。 |
| ablation_claim_metric_status | governance | none | false | false | false | 正式消融 claim 使用的统一 metric_status 边界。 |
| ablation_claim_formal_input_ready | governance | none | false | false | false | 内部消融 claim 输入是否全部来自正式 attacked image watermark rescore 记录。 |
| ablation_claim_gate_ready | governance | none | false | false | false | 内部消融 standalone claim 是否通过正式门禁。 |
| ablation_standalone_claim_ready | claim | none | false | false | false | 内部机制消融是否可作为独立论文主张证据。 |
| strong_ablation_standalone_claim_ready | claim | none | false | false | false | 核心强消融集合是否具备独立论文主张证据。 |
| core_ablation_claim_ready | governance | none | false | false | false | 核心机制消融集合是否全部由正式输入覆盖并支持 claim。 |
| core_ablation_claim_ids | protocol | none | false | false | false | 进入强消融 standalone claim 的核心消融标识集合。 |
| core_ablation_ready_count | metric | none | false | false | false | 已通过正式 claim gate 的核心消融数量。 |
| core_ablation_required_count | metric | none | false | false | false | 强消融 standalone claim 要求覆盖的核心消融数量。 |
| attack_formal_evidence_ready | governance | none | false | false | false | 消融 claim 依赖的攻击矩阵是否已完成真实图像级 formal evidence。 |
| ablation_real_linkage_boundary | governance | none | false | false | false | 内部消融与真实攻击闭环之间的证据链接边界说明。 |
| ablation_claim_input_filter | governance | none | false | false | false | 内部消融 claim 输入筛选报告的嵌套摘要。 |
| mechanism_group_count | metric | none | false | false | false | 内部消融覆盖的机制组数量。 |
| mechanism_groups | protocol | none | false | false | false | 内部消融覆盖的机制组集合。 |
| degradation_chain | claim | none | false | false | false | 按退化强度排序的消融链条。 |
| degradation_chain_rank | metric | none | false | false | false | 消融表中相对完整方法的退化排序。 |
| ablation_protocol_ready | protocol | none | false | false | false | 内部消融协议是否已生成完整消融记录和表格。 |
| mechanism_coverage_ready | protocol | none | false | false | false | 内部消融是否覆盖预定义机制组。 |
| external_baseline_result_ready | metric | none | false | false | false | 外部 baseline 真实结果是否已可供下游对照。 |
| compared_to_ablation_id | protocol | none | false | false | false | pairwise delta 对比使用的参考消融标识。 |
| full_metric_value | metric | none | false | false | false | 完整方法参考行中的指标值。 |
| ablated_metric_value | metric | none | false | false | false | 消融方法行中的指标值。 |
| delta_value | metric | none | false | false | false | 消融方法相对完整方法的指标差值。 |
| degradation_direction | metric | none | false | false | false | 指标差值对应的退化方向解释。 |
| mechanism_interpretation | claim | none | false | false | false | pairwise delta 行中的机制解释。 |
| registry_name | governance | none | false | false | false | 来源登记文件的稳定名称。 |
| registry_status | governance | none | false | false | false | 来源登记文件当前治理状态。 |
| source_root | artifact | none | false | false | false | 外部 baseline 源码缓存根目录。 |
| managed_by | governance | none | false | false | false | 来源登记文件遵循的治理契约。 |
| baseline_sources | artifact | none | false | false | false | 外部 baseline 来源登记条目集合。 |
| source_dir | artifact | none | true | false | false | 单个外部 baseline 的本地源码缓存目录。 |
| source_status | runtime | none | false | false | false | 单个外部 baseline 官方源码的本地获取状态。 |
| official_repository_url | runtime | none | false | false | false | 外部 baseline 官方源码仓库 URL。 |
| official_repository_commit | runtime | none | false | false | false | 外部 baseline 官方源码仓库提交标识。 |
| source_license | governance | none | false | false | false | 外部 baseline 官方源码许可证记录。 |
| local_code_tracked | governance | none | false | false | false | 外部 baseline 第三方源码是否由本仓库跟踪。 |
| result_status | metric | none | false | false | false | 外部 baseline 复现或导入结果状态。 |
| paper_claim_support | claim | none | false | false | false | 外部 baseline 来源或结果是否支持论文主张。 |
| aligned_rescoring_record_id | protocol | none | true | false | false | 真实 aligned rescoring record 的稳定标识。 |
| aligned_rescoring_record_count | metric | none | false | false | false | 真实 aligned rescoring records 数量。 |
| aligned_rescoring_ready | method | none | true | false | false | 单条真实 aligned rescoring 记录是否具备真实 latent 投影重打分。 |
| aligned_rescoring_package_path | artifact | none | false | false | false | 向下游传播的真实 aligned rescoring 结果包路径。 |
| aligned_rescoring_package_digest | artifact | none | false | false | false | 向下游传播的真实 aligned rescoring 结果包 SHA256 摘要。 |
| aligned_rescoring_quality_metrics_ready | metric | none | false | false | false | 真实 aligned rescoring 结果包中的 pair-level 质量指标是否已完成并可向下游传播。 |
| real_aligned_rescore_count | metric | none | false | false | false | 具备真实 latent 投影重打分的记录数量。 |
| selected_attention_carrier_count | metric | none | false | false | false | 真实 aligned rescoring 运行中选用的 active attention carrier 数量。 |
| real_raw_content_score | metric | none | true | false | false | 对齐前真实 latent 投影重新计算得到的内容分数。 |
| real_aligned_content_score | metric | none | true | false | false | 对齐后真实 latent 投影重新计算得到的内容分数。 |
| real_rescoring_score_gain | metric | none | true | false | false | 真实 aligned content score 相对 raw content score 的增益。 |
| score_space_name | protocol | none | true | false | false | fixed-FPR 校准、攻击检测或结果摘要实际使用的分数空间名称。 |
| score_space_alignment_ready | governance | none | false | false | false | fixed-FPR 校准分数空间是否已与真实检测分数空间对齐。 |
| real_score_calibration_ready | governance | none | false | false | false | fixed-FPR 阈值是否基于真实 aligned rescoring 分数记录校准。 |
| calibration_records_source | protocol | none | false | false | false | fixed-FPR 校准记录来源, 例如 aligned_rescoring_real_scores。 |
| calibration_records_source_member | artifact | none | false | false | false | fixed-FPR 校准记录在输入压缩包中的成员路径。 |
| calibration_record_count | metric | none | false | false | false | fixed-FPR 校准实际消费的记录数量。 |
| raw_content_score_source_field | protocol | none | true | false | false | 归一化 raw_content_score 时使用的原始字段名。 |
| aligned_content_score_source_field | protocol | none | true | false | false | 归一化 aligned_content_score 时使用的原始字段名。 |
| real_lf_score_before | metric | none | true | false | false | 对齐前真实 latent 投影上的 LF 内容分数。 |
| real_lf_score_after | metric | none | true | false | false | 对齐后真实 latent 投影上的 LF 内容分数。 |
| real_tail_score_before | metric | none | true | false | false | 对齐前真实 latent 投影上的高斯幅值尾部截断内容分数。 |
| real_tail_score_after | metric | none | true | false | false | 对齐后真实 latent 投影上的高斯幅值尾部截断内容分数。 |
| real_combined_score_before | metric | none | true | false | false | 对齐前真实 latent 投影上的 combined 内容分数。 |
| real_combined_score_after | metric | none | true | false | false | 对齐后真实 latent 投影上的 combined 内容分数。 |
| real_lf_tail_fusion_score_before | metric | none | true | false | false | 对齐前真实 latent 投影上的 LF/高斯幅值尾部截断加权诊断分数。 |
| real_lf_tail_fusion_score_after | metric | none | true | false | false | 对齐后真实 latent 投影上的 LF/高斯幅值尾部截断加权诊断分数。 |
| latent_projection_digest_before | artifact | none | true | false | false | 对齐前真实 latent 投影向量的稳定摘要。 |
| latent_projection_digest_after | artifact | none | true | false | false | 对齐后真实 latent 投影向量的稳定摘要。 |
| latent_projection_values_before | method | none | true | false | false | 对齐前真实 latent 投影到内容检测维度后的有界向量。 |
| latent_projection_values_after | method | none | true | false | false | 对齐后真实 latent 投影到内容检测维度后的有界向量。 |
| latent_projection_mode | method | none | true | false | false | 真实 latent 到内容检测向量的投影模式。 |
| latent_projection_boundary_before | method | none | true | false | false | real_raw_content_score 使用的真实 latent 边界, 应固定为首次注入前未被 watermark update 污染的 latent。 |
| latent_projection_boundary_after | method | none | true | false | false | real_aligned_content_score 使用的真实 latent 边界, 应为完成所有 runtime 注入后的 aligned latent。 |
| first_injection_trajectory_index | protocol | none | true | false | false | 真实 aligned rescoring 中首次 runtime 注入所在的采样轨迹索引。 |
| first_injection_timestep | protocol | none | true | false | false | 真实 aligned rescoring 中首次 runtime 注入对应的 diffusion timestep。 |
| final_injection_trajectory_index | protocol | none | true | false | false | 真实 aligned rescoring 中最终 runtime 注入所在的采样轨迹索引。 |
| final_injection_timestep | protocol | none | true | false | false | 真实 aligned rescoring 中最终 runtime 注入对应的 diffusion timestep。 |
| content_carrier_source | method | none | true | false | false | 真实 runtime carrier 中内容载体分量的来源。 |
| runtime_content_update_digest | method | none | true | false | false | 真实 runtime latent 写入所使用的 content update 摘要。 |
| runtime_content_detection_record_id | method | none | true | false | false | 真实 runtime latent 写入所绑定的内容检测 record 标识。 |
| runtime_content_sample_role | protocol | none | true | false | false | 真实 runtime latent 写入所绑定的内容样本角色。 |
| runtime_content_weight | method | none | true | false | false | 真实 runtime carrier 中 content 分量的组合权重。 |
| runtime_attention_weight | method | none | true | false | false | 真实 runtime carrier 中 attention 分量的组合权重。 |
| runtime_attention_alignment | method | none | true | false | false | 真实 runtime carrier 中 attention 分量与 content 分量的方向一致性。 |
| runtime_attention_sign | method | none | true | false | false | 真实 runtime carrier 中 attention 分量为避免抵消内容信号所采用的符号。 |
| output_records_path | artifact | none | false | false | false | 真实 aligned rescoring result 中登记的 records 输出路径。 |
| quality_metrics_path | artifact | none | false | false | false | 真实 aligned rescoring result 中登记的质量指标表路径。 |
| aligned_image_path | artifact | none | true | false | false | 真实 aligned rescoring 运行保存的 aligned image 路径。 |
| aligned_image_digest | artifact | none | true | false | false | 真实 aligned rescoring 运行保存的 aligned image 摘要。 |
| perceptual_metrics_ready | metric | none | false | false | false | LPIPS / CLIP 等感知指标是否已经完成计算。 |
| lpips | metric | none | false | false | false | Learned Perceptual Image Patch Similarity 指标值或 unsupported 状态。 |
| lpips_status | metric | none | false | false | false | LPIPS 指标计算状态。 |
| lpips_error_type | metric | none | false | false | false | LPIPS 指标不可用时记录的异常类型, 用于定位 Colab 运行失败边界。 |
| lpips_error_message | metric | none | false | false | false | LPIPS 指标不可用时记录的压缩异常信息, 用于定位 Colab 运行失败边界。 |
| lpips_network | runtime | none | false | false | false | aligned rescoring 中 LPIPS 使用的 backbone 名称, 例如 alex。 |
| clip_score | metric | none | false | false | false | CLIP 图文一致性或图像一致性指标值或 unsupported 状态。 |
| clip_score_clean | metric | none | false | false | false | clean image 与 prompt 的 CLIP 图文一致性分数。 |
| clip_score_aligned | metric | none | false | false | false | aligned image 与 prompt 的 CLIP 图文一致性分数。 |
| clip_score_delta | metric | none | false | false | false | aligned CLIP score 相对 clean CLIP score 的差值。 |
| clip_score_status | metric | none | false | false | false | CLIP score 指标计算状态。 |
| clip_score_error_type | metric | none | false | false | false | CLIP score 指标不可用时记录的异常类型, 用于定位 Colab 运行失败边界。 |
| clip_score_error_message | metric | none | false | false | false | CLIP score 指标不可用时记录的压缩异常信息, 用于定位 Colab 运行失败边界。 |
| clip_model_id | runtime | none | false | false | false | aligned rescoring 中用于计算 CLIP score 的模型标识。 |
| enable_pair_perceptual_metrics | runtime | none | false | false | false | 是否在真实 aligned rescoring workflow 中尝试计算 LPIPS 与 CLIP pair-level 指标。 |
| require_pair_perceptual_metrics | runtime | none | false | false | false | 是否要求 LPIPS 与 CLIP pair-level 指标完成后才允许 aligned rescoring run_decision 为 pass。 |
| perceptual_metric_device_name | runtime | none | false | false | false | LPIPS 与 CLIP pair-level 指标计算使用的设备名称。 |
| fid | metric | none | false | false | false | Fréchet Inception Distance 指标值或 unsupported 状态。 |
| fid_status | metric | none | false | false | false | FID 指标计算状态。 |
| kid | metric | none | false | false | false | Kernel Inception Distance 指标值或 unsupported 状态。 |
| kid_status | metric | none | false | false | false | KID 指标计算状态。 |
| dataset_quality_record_id | artifact | none | true | false | false | 数据集级质量图像对记录的稳定标识。 |
| dataset_quality_record_digest | artifact | none | true | false | false | 数据集级质量图像对记录的稳定摘要。 |
| image_pair_index | protocol | none | true | false | false | 数据集级质量图像对在 registry 中的序号。 |
| image_pair_role | protocol | none | true | false | false | 数据集级质量图像对的比较角色或攻击名称。 |
| comparison_image_path | artifact | none | true | false | false | 数据集级质量图像对中的 comparison 图像路径。 |
| comparison_image_digest | artifact | none | true | false | false | 数据集级质量图像对中的 comparison 图像摘要。 |
| feature_backend | protocol | none | true | false | false | 数据集级质量指标使用的图像特征后端。 |
| paper_metric_name | metric | none | false | false | false | 正式质量指标在论文中的规范名称。 |
| source_image_count | metric | none | false | false | false | 数据集级质量输入中的 source 图像数量。 |
| comparison_image_count | metric | none | false | false | false | 数据集级质量输入中的 comparison 图像数量。 |
| sample_pair_count | metric | none | false | false | false | 数据集级质量输入中的图像配对数量。 |
| formal_quality_metric_count | metric | none | false | false | false | 数据集级正式质量指标表中的指标行数量, 正式表只包含 FID / KID。 |
| formal_fid_kid_ready | governance | none | false | false | false | 正式 FID / KID 是否已由论文约定特征后端完成。 |
| formal_fid_kid_metric_names_ready | governance | none | false | false | false | 正式 FID 与 KID 两个指标是否均已测量, 防止单个指标被误判为完整质量结论。 |
| formal_fid_kid_claim_gate_ready | governance | none | false | false | false | 数据集级正式 FID / KID 是否通过论文质量主张门禁。 |
| formal_fid_kid_claim_blocker | governance | none | false | false | false | 阻断正式 FID / KID 质量主张的原因。 |
| dataset_quality_claim_boundary | governance | none | false | false | false | 数据集级质量证据能够支撑的论文主张边界。 |
| dataset_quality_summary_path | artifact | none | false | false | false | 数据集级质量摘要 JSON 路径。 |
| dataset_quality_formal_metrics_path | artifact | none | false | false | false | 只包含正式 FID / KID 行的数据集级质量指标表路径。 |
| dataset_quality_formal_feature_import_report_path | artifact | none | false | false | false | 数据集级质量正式特征导入报告路径。 |
| dataset_quality_formal_feature_records_path | artifact | none | false | false | false | 当前论文运行目录内规范化保存的正式 Inception 特征 JSONL 路径。 |
| dataset_quality_image_role | protocol | none | true | false | false | 数据集级质量正式特征记录对应 source 或 comparison 图像角色。 |
| feature_vector | artifact | none | true | false | false | 由 Inception 或论文约定视觉特征后端导出的单张图像特征向量。 |
| formal_feature_record_count | metric | none | false | false | false | 规范正式特征 JSONL 的记录数量, 必须等于当前 Prompt 数量的2倍。 |
| expected_feature_pair_count | metric | none | false | false | false | 当前论文运行层级要求完整覆盖的 source / comparison 特征配对数量。 |
| accepted_feature_pair_count | metric | none | false | false | false | 可用于正式 FID / KID 协议的 source / comparison 特征配对数量。 |
| missing_feature_pair_count | metric | none | false | false | false | 正式特征导入中缺失 source 或 comparison 特征的图像对数量。 |
| feature_issue_count | metric | none | false | false | false | 正式特征导入 schema 检查发现的问题数量。 |
| formal_feature_records_sha256 | provenance | none | false | false | false | 当前 run 规范正式特征 JSONL 文件的字节级 SHA-256。 |
| feature_dimension | metric | none | false | false | false | 正式特征记录中的视觉特征维度。 |
| formal_feature_backend_ready | governance | none | false | false | false | 数据集级质量正式视觉特征后端是否已导入并通过 schema 检查。 |
| formal_sample_scale_ready | governance | none | false | false | false | 数据集级质量正式 FID / KID 是否具备足够样本规模。 |
| formal_min_sample_count | governance | none | false | false | false | 数据集级质量正式 FID / KID 协议要求的最小图像对数量。 |
| formal_feature_records_path | artifact | none | false | false | false | 数据集级质量正式视觉特征 JSONL 记录路径。 |
| feature_model_name | runtime | none | false | false | false | 数据集级质量正式视觉特征提取所使用的模型名称。 |
| feature_device_name | runtime | none | false | false | false | 数据集级质量正式视觉特征提取所使用的运行设备。 |
| dataset_quality_metrics_path | artifact | none | false | false | false | 数据集级正式质量指标表路径, 与 dataset_quality_formal_metrics_path 等价。 |
| real_attack_registry_path | artifact | none | false | false | false | 数据集级质量脚本读取的真实攻击图像 registry 路径。 |
| dataset_quality_image_resolution_records_path | artifact | none | false | false | false | 数据集级质量图像解析记录 JSONL 路径。 |
| image_resolution_record_id | artifact | none | true | false | false | 单个数据集级质量图像解析记录的稳定标识。 |
| image_resolution_record_digest | artifact | none | true | false | false | 单个数据集级质量图像解析记录的稳定摘要。 |
| requested_image_path | artifact | none | true | false | false | 数据集级质量指标请求读取的原始图像路径。 |
| resolved_image_path | artifact | none | true | false | false | 图像解析流程最终找到或物化的可读取图像路径。 |
| resolved_from_package_path | artifact | none | true | false | false | 图像从前序结果 ZIP 物化时对应的来源包路径。 |
| resolved_image_digest | artifact | none | true | false | false | 图像解析流程最终读取到的图像文件摘要。 |
| resolved_from_package_digest | artifact | none | true | false | false | 图像来源 ZIP 包的 SHA-256 摘要。 |
| resolution_status | governance | none | true | false | false | 单个图像路径的解析状态, 例如已存在、从输入包物化或缺失。 |
| materialized_image_input | artifact | none | true | false | false | 图像是否由前序结果 ZIP 物化到 outputs 下作为本次质量指标输入。 |
| image_resolution_record_count | metric | none | false | false | false | 数据集级质量图像解析记录总数。 |
| resolved_image_file_count | metric | none | false | false | false | 数据集级质量图像解析流程成功找到的图像文件数量。 |
| missing_image_file_count | metric | none | false | false | false | 数据集级质量图像解析流程仍然缺失的图像文件数量。 |
| materialized_image_input_count | metric | none | false | false | false | 从前序结果 ZIP 物化到 outputs 下的图像输入数量。 |
| input_package_count | metric | none | false | false | false | 数据集级质量指标脚本读取的前序结果 ZIP 数量。 |
| real_attack_evaluation_drive_package_path | artifact | none | false | false | false | Google Drive 中真实攻击闭环前序包路径。 |
| real_attack_evaluation_drive_package_digest | artifact | none | false | false | false | Google Drive 中真实攻击闭环前序包 SHA-256 摘要。 |
| real_attack_evaluation_input_package_path | artifact | none | false | false | false | 复制到本次数据集级质量输入目录的真实攻击闭环包路径。 |
| real_attack_evaluation_input_package_digest | artifact | none | false | false | false | 本次数据集级质量输入目录中真实攻击闭环包的 SHA-256 摘要。 |
| real_attack_extracted_entry_count | metric | none | false | false | false | 从真实攻击闭环前序包中解出的 registry 与 attacked image 文件数量。 |
| real_attack_extracted_entries | artifact | none | false | false | false | 从真实攻击闭环前序包中解出的 registry 与 attacked image 文件路径集合。 |
| dataset_quality_summary | artifact | none | false | false | false | 论文证据审计中数据集级质量摘要的逻辑路径键。 |
| dataset_quality_metrics | artifact | none | false | false | false | 论文证据审计中数据集级正式质量指标表的逻辑路径键。 |
| dataset_quality_diagnostic_metrics | artifact | none | false | false | false | 论文证据审计中数据集级诊断质量指标表的逻辑路径键, 不支撑 claim。 |
| audit_item_id | governance | none | false | false | false | 论文证据审计表中单个可重建产物检查项的稳定标识。 |
| artifact_kind | artifact | none | false | false | false | 论文证据审计中产物类型, 例如 table 或 figure_data。 |
| artifact_name | artifact | none | false | false | false | 论文证据审计中产物的人类可读名称。 |
| source_paths | artifact | none | false | false | false | 论文图表或审计产物可重建时依赖的上游路径集合。 |
| source_path_map | artifact | none | false | false | false | 论文证据审计输入中由脚本层注入的逻辑路径到受治理路径映射。 |
| builder_status | artifact | none | false | false | false | 论文产物构建器的当前状态, 例如可重建预览或阻断。 |
| paper_ready | artifact | none | false | false | false | 单个论文产物是否已经具备投稿级使用条件。 |
| claim_text | claim | none | false | false | false | 被审计论文主张的人类可读表述。 |
| claim_decision | claim | none | false | false | false | 被审计论文主张的当前判定, 例如 unsupported 或 preview_only。 |
| claim_scope | claim | none | false | false | false | 被审计论文主张所属的证据范围。 |
| paper_claim_supported | claim | none | false | false | false | 单条 claim audit 行是否支持论文主张。 |
| blocker_count | governance | none | false | false | false | 单个审计对象关联的阻断原因数量。 |
| primary_blocker | governance | none | false | false | false | 单个审计对象最主要的阻断原因代码。 |
| gap_id | governance | none | false | false | false | 投稿前证据缺口的稳定标识。 |
| gap_area | governance | none | false | false | false | 投稿前证据缺口所属的实验或产物区域。 |
| blocker_severity | governance | none | false | false | false | 证据缺口或阻断项的严重程度。 |
| required_action | governance | none | false | false | false | 关闭证据缺口所需执行的具体补证动作。 |
| related_artifacts | artifact | none | false | false | false | 证据缺口关联的上游或下游产物路径集合。 |
| closes_claim_ids | claim | none | false | false | false | 证据缺口关闭后可解除阻断的 claim 标识集合。 |
| recommended_order | governance | none | false | false | false | 补证任务建议执行顺序。 |
| submission_ready | governance | none | false | false | false | 当前证据包是否已经具备投稿冻结条件。 |
| critical_gap_count | metric | none | false | false | false | 当前审计中严重证据缺口数量。 |
| blocking_claim_count | metric | none | false | false | false | 当前审计中不可支持或仅预览的 claim 数量。 |
| artifact_builder_ready | artifact | none | false | false | false | 所有注册的论文表格与图数据是否均具备可重建的构建路径。 |
| paper_artifact_claim_ready | governance | none | false | false | false | 所有注册的论文表格与图数据是否均已达到正式论文证据门禁。 |
| paper_artifact_audit_ready | artifact | none | false | false | false | 论文图表证据审计链路是否已经完成本地可审计输出。 |
| claim_audit_row_count | metric | none | false | false | false | claim audit 表中的审计行数量。 |
| table_readiness_row_count | metric | none | false | false | false | 论文表格 readiness 表中的审计行数量。 |
| figure_readiness_row_count | metric | none | false | false | false | 论文图数据 readiness 表中的审计行数量。 |
| rebuildable_artifact_count | metric | none | false | false | false | 当前可由受治理输入重建的论文产物数量。 |
| blocked_artifact_count | metric | none | false | false | false | 当前被阻断的论文产物数量。 |
| paper_ready_artifact_count | metric | none | false | false | false | 当前已经达到投稿级条件的论文产物数量。 |
| primary_blockers | governance | none | false | false | false | 投稿冻结阻断报告中的主要阻断项列表。 |
| recommended_next_action | governance | none | false | false | false | 审计报告建议优先执行的下一步补证动作。 |
| gap_count | metric | none | false | false | false | 当前审计列出的证据缺口总数。 |
| dry_run_decision | governance | none | false | false | false | 论文证据审计 dry-run 的通过或失败判定。 |
| artifact_builder_readiness_report | artifact | none | false | false | false | dry-run 报告中嵌入的论文产物构建器 readiness 摘要。 |
| submission_blocker_report | governance | none | false | false | false | dry-run 报告中嵌入的投稿冻结阻断摘要。 |
| evidence_manifest | governance | none | false | false | false | 投稿就绪门禁输入中引用的证据审计 manifest 摘要。 |
| builder_report | artifact | none | false | false | false | 投稿就绪门禁输入中引用的产物构建器 readiness 报告。 |
| blocker_report | governance | none | false | false | false | 投稿就绪门禁输入中引用的投稿阻断报告。 |
| evidence_gaps | governance | none | false | false | false | 投稿就绪门禁输入中引用的证据缺口集合。 |
| release_profiles | governance | none | false | false | false | 投稿就绪门禁输入中引用的 release profile dry-run 摘要集合。 |
| required_input_id | governance | none | false | false | false | 投稿就绪门禁中待补齐输入的稳定标识。 |
| required_input_area | governance | none | false | false | false | 投稿就绪门禁中待补齐输入所属的证据区域。 |
| required_input_severity | governance | none | false | false | false | 投稿就绪门禁中待补齐输入的阻断严重程度。 |
| input_ready | governance | none | false | false | false | 单个待补齐输入是否已经满足投稿冻结条件。 |
| profile_name | governance | none | false | false | false | release 抽取配置或正式依赖 profile 的稳定名称。 |
| root_path | governance | none | false | false | false | release 抽取 dry-run 使用的仓库根路径。 |
| output_path | artifact | none | false | false | false | release 抽取 dry-run 指定的输出路径。 |
| copied_files | artifact | none | false | false | false | release 抽取 dry-run 将会复制的文件清单。 |
| missing_paths | artifact | none | false | false | false | release 抽取 dry-run 中缺失的输入路径清单。 |
| excluded_parts | governance | none | false | false | false | release 抽取配置中排除的路径片段集合。 |
| dry_run | governance | none | false | false | false | release 抽取或审计命令是否只执行 dry-run。 |
| release_profile_name | governance | none | false | false | false | 投稿就绪门禁中被审计的 release profile 名称。 |
| release_profile_file_count | metric | none | false | false | false | 单个 release profile dry-run 中可复制文件数量。 |
| release_profile_missing_count | metric | none | false | false | false | 单个 release profile dry-run 中缺失路径数量。 |
| release_dry_run_ready | governance | none | false | false | false | release profile dry-run 是否可执行并具备文件清单。 |
| release_package_allowed | governance | none | false | false | false | 当前证据边界下是否允许导出 release package。 |
| package_freeze_allowed | governance | none | false | false | false | 当前证据边界下是否允许冻结投稿候选包。 |
| release_scope | governance | none | false | false | false | 当前 release 产物允许使用的范围。 |
| readiness_decision | governance | none | false | false | false | 投稿就绪门禁的总体判定。 |
| critical_required_input_count | metric | none | false | false | false | 投稿就绪门禁中 critical 待补齐输入数量。 |
| required_input_count | metric | none | false | false | false | 投稿就绪门禁中待补齐输入总数。 |
| release_profile_count | metric | none | false | false | false | 投稿就绪门禁中审计的 release profile 数量。 |
| limitations | governance | none | false | false | false | 投稿就绪门禁报告中显式列出的适用边界和限制。 |
| entry_review_ready | governance | none | false | false | false | 论文投稿级证据闭合入口审计是否已生成确定性判定。 |
| evidence_closure_allowed | governance | none | false | false | false | 当前受治理证据是否允许进入论文投稿级证据闭合。 |
| entry_review_decision | governance | none | false | false | false | 证据闭合入口审计的总体判定。 |
| review_item_count | metric | none | false | false | false | 证据闭合入口审计清单中的检查项数量。 |
| blocked_review_item_count | metric | none | false | false | false | 证据闭合入口审计清单中仍被阻断的检查项数量。 |
| blocked_review_item_ids | governance | none | false | false | false | 证据闭合入口审计清单中仍被阻断的检查项 id 集合。 |
| review_item_id | governance | none | false | false | false | 证据闭合入口审计清单中的检查项 id。 |
| review_area | governance | none | false | false | false | 证据闭合入口审计清单中的检查领域。 |
| review_status | governance | none | false | false | false | 证据闭合入口审计清单中单项检查状态。 |
| source_artifact | artifact | none | false | false | false | 单项入口审计检查依据的受治理产物路径。 |
| blocker_reason | governance | none | false | false | false | 单项入口审计检查未通过时的阻断原因。 |
| audit_note | governance | none | false | false | false | 解释单项入口审计的检查内容与受治理证据边界。 |
| threshold_calibration_ready | artifact | none | false | false | false | threshold calibration 结果是否已经生成可供下游读取的阈值与审计产物。 |
| geometric_rescue_ready | artifact | none | false | false | false | 几何恢复记录是否已经由前序结果包重建并满足下游阈值校准输入要求。 |
| geometric_rescue_record_count | metric | none | false | false | false | 本次 threshold calibration workflow 重建得到的几何恢复记录数量。 |
| threshold_manifest_path | artifact | none | false | false | false | threshold calibration manifest 的输出路径。 |
| geometric_rescue_manifest_path | artifact | none | false | false | false | 几何恢复 manifest 的输出路径。 |
| threshold_report_path | artifact | none | false | false | false | threshold degeneracy report 的输出路径。 |
| rescue_audit_path | artifact | none | false | false | false | 几何恢复审计摘要的输出路径。 |
| attention_injection_drive_package_path | artifact | none | false | false | false | Google Drive 中 attention latent injection 前序包路径。 |
| attention_injection_drive_package_digest | artifact | none | false | false | false | Google Drive 中 attention latent injection 前序包 SHA256 摘要。 |
| attention_injection_input_package_path | artifact | none | false | false | false | 复制到本次 threshold calibration 输入目录的 attention latent injection 包路径。 |
| attention_injection_input_package_digest | artifact | none | false | false | false | 复制到本次 threshold calibration 输入目录的 attention latent injection 包 SHA256 摘要。 |
| aligned_rescoring_drive_package_path | artifact | none | false | false | false | Google Drive 中 aligned rescoring 前序包路径。 |
| aligned_rescoring_drive_package_digest | artifact | none | false | false | false | Google Drive 中 aligned rescoring 前序包 SHA256 摘要。 |
| aligned_rescoring_input_package_path | artifact | none | false | false | false | 复制到本次 threshold calibration 输入目录的 aligned rescoring 包路径。 |
| aligned_rescoring_input_package_digest | artifact | none | false | false | false | 复制到本次 threshold calibration 输入目录的 aligned rescoring 包 SHA256 摘要。 |
| content_records_path | artifact | none | false | false | false | 从 aligned rescoring 前序包解出的内容检测记录路径。 |
| content_extracted_entry_count | metric | none | false | false | false | 从 aligned rescoring 前序包解出的内容检测输入文件数量。 |
| content_extracted_entries | artifact | none | false | false | false | 从 aligned rescoring 前序包解出的内容检测输入文件列表。 |
| embedded_digest_scope | governance | none | false | false | false | 说明 zip 内嵌 archive 摘要文件与外部最终摘要之间的 digest 记录边界。 |
| baseline_result_record_id | protocol | none | true | false | false | 外部 baseline 受治理导入结果记录的稳定标识。 |
| baseline_result_digest | artifact | none | true | false | false | 外部 baseline 受治理导入结果 payload 的稳定摘要。 |
| result_protocol_name | protocol | none | true | false | false | 外部 baseline 结果所遵循的共同实验协议名称。 |
| result_source_type | artifact | none | true | false | false | 外部 baseline 指标来自官方复现还是受治理导入。 |
| baseline_result_source_digest | artifact | none | true | false | false | 外部 baseline 指标来源文件或来源包的稳定摘要。 |
| official_repository_url | artifact | none | false | false | false | 外部 baseline 官方源码仓库地址。 |
| official_repository_commit | artifact | none | false | false | false | 外部 baseline 官方源码本地缓存对应的提交标识。 |
| official_repository_branch | artifact | none | false | false | false | 外部 baseline 官方源码本地缓存对应的分支名称。 |
| official_source_ready_count | metric | none | false | false | false | 已在本地源码缓存中可检查的外部 baseline 数量。 |
| imported_baseline_result_count | metric | none | false | false | false | 已导入共同协议结果记录的外部 baseline 观测数量。 |
| baseline_source_registry_ready | governance | none | false | false | false | 外部 baseline 源码登记文件是否已被本次对比流程读取。 |
| baseline_source_registry_path | artifact | none | false | false | false | 外部 baseline 源码登记文件路径。 |
| baseline_result_records_path | artifact | none | false | false | false | 受治理外部 baseline 结果 JSONL 输入路径。 |
| source_registry_digest | artifact | none | false | false | false | 外部 baseline 源码登记内容的稳定摘要。 |
| imported_result_digest | artifact | none | false | false | false | 外部 baseline 受治理导入结果集合的稳定摘要。 |
| baseline_execution_id | protocol | none | true | false | false | 主表外部 baseline 官方复现计划记录的稳定标识。 |
| baseline_execution_digest | artifact | none | true | false | false | 主表外部 baseline 官方复现计划 payload 的稳定摘要。 |
| command_profile_name | protocol | none | false | false | false | 外部 baseline 官方运行入口的命令画像名称。 |
| dependency_profile | runtime | none | true | false | false | 外部 baseline runner 引用的完整受治理依赖 profile 记录。 |
| official_entrypoint | artifact | none | true | false | false | 外部 baseline 官方源码中的运行入口脚本。 |
| reproduction_command | runtime | none | true | false | false | 外部 baseline 官方复现建议命令模板。 |
| expected_result_adapter | protocol | none | true | false | false | 官方输出转换为共同协议结果记录时需要使用的适配器名称。 |
| model_alignment_status | protocol | none | true | false | false | 外部 baseline 与 SD3.5 主线模型边界的对齐状态。 |
| result_import_required | governance | none | true | false | false | 外部 baseline 是否仍需导入共同协议结果记录。 |
| result_record_template_id | protocol | none | true | false | false | 外部 baseline 共同协议结果导入模板的稳定标识。 |
| result_record_template_digest | artifact | none | true | false | false | 外部 baseline 共同协议结果导入模板 payload 的稳定摘要。 |
| required_metric_fields | protocol | none | true | false | false | 外部 baseline 结果导入时必须提供的指标字段集合。 |
| required_source_fields | protocol | none | true | false | false | 外部 baseline 结果导入时必须提供的来源字段集合。 |
| primary_baseline_count | metric | none | false | false | false | 主表外部 baseline 数量。 |
| primary_source_ready_count | metric | none | false | false | false | 主表外部 baseline 中官方源码入口可检查的数量。 |
| result_record_template_count | metric | none | false | false | false | 主表外部 baseline 需要补齐的共同协议结果模板数量。 |
| primary_baseline_plan_ready | governance | none | false | false | false | 主表外部 baseline 官方复现计划是否已可审计。 |
| result_import_template_ready | governance | none | false | false | false | 主表外部 baseline 共同协议结果导入模板是否已可审计。 |
| execution_plan_digest | artifact | none | false | false | false | 主表外部 baseline 官方复现计划集合的稳定摘要。 |
| result_template_digest | artifact | none | false | false | false | 主表外部 baseline 结果导入模板集合的稳定摘要。 |
| adapter_path | artifact | none | true | false | false | 外部 baseline 的项目维护 adapter 路径。 |
| adapter_status | runtime | none | true | false | false | 外部 baseline adapter 当前可执行或需补齐的状态。 |
| official_source_tracked | governance | none | false | false | false | 第三方官方源码是否由本项目 git 跟踪。 |
| baseline_command_plan_manifest_path | artifact | none | false | false | false | 外部 baseline 命令计划 manifest 路径。 |
| baseline_command_results_path | artifact | none | true | false | false | 单 baseline transfer manifest 绑定的 command result 相对路径。 |
| baseline_observations_path | artifact | none | false | false | false | 外部 baseline observation 输出路径。 |
| baseline_command_id | protocol | none | true | false | false | 外部 baseline 命令计划中单条命令的稳定标识。 |
| baseline_command_digest | artifact | none | true | false | false | 外部 baseline 命令计划 payload 的稳定摘要。 |
| command_count | metric | none | false | false | false | 外部 baseline 命令计划中的命令数量。 |
| failed_command_count | metric | none | false | false | false | 外部 baseline 命令执行失败数量。 |
| baseline_ids | protocol | none | true | false | false | 外部 baseline 命令计划或执行 manifest 中涉及的 baseline id 集合。 |
| observation_count | metric | none | false | false | false | 外部 baseline adapter 或执行 manifest 中的 observation 数量。 |
| return_code | runtime | none | false | false | false | 外部 baseline 命令进程返回码。 |
| stdout | runtime | none | false | false | false | 外部 baseline 命令标准输出文本。 |
| stderr | runtime | none | false | false | false | 外部 baseline 命令标准错误文本。 |
| working_directory | runtime | none | false | false | false | 外部 baseline 命令执行工作目录。 |
| timeout_seconds | runtime | none | false | false | false | 外部 baseline 命令允许运行的最长秒数。 |
| command_results_path | artifact | none | false | false | false | 外部 baseline 命令结果文件路径。 |
| observations_path | artifact | none | false | false | false | 主表 baseline 证据边界读取的 observation 输入路径。 |
| source_registry_path | artifact | none | false | false | false | 主表 baseline 证据边界读取的源码登记输入路径。 |
| execution_digest | artifact | none | true | false | false | 外部 baseline 执行 manifest 的稳定摘要。 |
| evidence_paths | artifact | none | true | false | false | 外部 baseline 正式结果所绑定的证据文件路径集合。 |
| formal_result_claim | claim | none | false | false | false | 外部 baseline 结果是否声明可作为正式论文对比证据。 |
| execution_boundary | governance | none | true | false | false | 外部 baseline 执行 manifest 对工程链路与论文证据边界的说明。 |
| producer_id | governance | none | true | false | false | 产物生成器或 adapter 的稳定标识。 |
| producer_role | governance | none | true | false | false | 产物生成器在流程中的职责。 |
| detection_decision | metric | none | false | false | false | 外部 baseline 单条 observation 根据 score 和 threshold 得到的检测判定。 |
| score_name | metric | none | true | false | false | 外部 baseline observation 中分数字段的语义名称。 |
| higher_is_positive | protocol | none | false | false | false | 分数越高是否代表越倾向水印阳性。 |
| threshold_source | protocol | none | true | false | false | 外部 baseline observation 中 threshold 的来源说明。 |
| bit_accuracy | metric | none | false | false | false | T2SMark 或同类方法记录的 bit 级准确率。 |
| key_accuracy | metric | none | false | false | false | T2SMark 或同类方法记录的 key 级准确率。 |
| t2smark_result_index | protocol | none | false | false | false | T2SMark 官方 results.json 中的样本索引。 |
| image_pair_count | metric | none | false | false | false | T2SMark adapter 输入 image pair 数量。 |
| t2smark_result_count | metric | none | false | false | false | T2SMark 官方 results.json 中可读取的样本数量。 |
| t2smark_run_name | protocol | none | true | false | false | T2SMark 正式运行、复用和打包路径绑定的论文运行层级名称。 |
| missing_result_indices | metric | none | true | false | false | T2SMark 输入 image pair 中缺少官方结果的索引集合。 |
| adapter_digest | artifact | none | true | false | false | 外部 baseline adapter manifest 的稳定摘要。 |
| generation_model_id | runtime | none | true | false | false | 生成外部 baseline 图像所使用的模型标识。|
| watermark_parameters | protocol | none | true | false | false | 外部 baseline method-faithful adapter 的水印参数摘要。|
| message_mapping | protocol | none | true | false | false | Gaussian Shading adapter 中 message 到 latent noise 的映射说明。|
| tree_ring_adapter_mode | protocol | none | false | false | false | Tree-Ring adapter 在命令计划中的运行模式。|
| gaussian_shading_adapter_mode | protocol | none | false | false | false | Gaussian Shading adapter 在命令计划中的运行模式。|
| shallow_diffuse_adapter_mode | protocol | none | false | false | false | Shallow Diffuse adapter 在命令计划中的运行模式。|
| tree_ring_attack_families | protocol | none | false | false | false | Tree-Ring adapter 内部可选轻量攻击族列表。|
| gaussian_shading_attack_families | protocol | none | false | false | false | Gaussian Shading adapter 内部可选轻量攻击族列表。|
| shallow_diffuse_attack_families | protocol | none | false | false | false | Shallow Diffuse adapter 内部可选轻量攻击族列表。|
| channel_copy | protocol | none | true | false | false | Gaussian Shading bit 在 SD3.5 latent 通道维度上的重复因子。|
| hw_copy | protocol | none | true | false | false | Gaussian Shading bit 在 SD3.5 latent 空间维度上的重复因子。|
| shallow_injection_mode | protocol | none | true | false | false | Shallow Diffuse adapter 单条 observation 使用的浅层注入执行模式。|
| shallow_injection_modes | protocol | none | false | false | false | Shallow Diffuse adapter manifest 中出现过的浅层注入执行模式集合。|
| edit_fraction | protocol | none | true | false | false | Shallow Diffuse 在 denoising 过程中的注入位置比例。|
| w_inner_radius | protocol | none | true | false | false | Shallow Diffuse ring mask 的内半径。|
| w_mask_shape | protocol | none | true | false | false | Shallow Diffuse watermark mask 形状。|
| w_injection | protocol | none | true | false | false | Shallow Diffuse watermark 注入模式。|
| w_measurement | protocol | none | true | false | false | Shallow Diffuse watermark 检测度量。|
| source_role | protocol | none | true | false | false | attacked image manifest 中源图像的角色。|
| primary_baseline_formal_import_protocol_name | protocol | none | true | false | false | 主表 external baseline 正式结果导入协议名称。 |
| formal_import_input_record_count | metric | none | false | false | false | 正式导入 validator 接收的候选记录数量。 |
| accepted_formal_import_count | metric | none | false | false | false | 通过正式导入 validator 的记录数量。 |
| rejected_formal_import_count | metric | none | false | false | false | 未通过正式导入 validator 的记录数量。 |
| formal_import_issue_count | metric | none | false | false | false | 正式导入 validator 发现的问题数量。 |
| formal_import_validation_ready | governance | none | false | false | false | 主表 baseline 正式导入候选记录是否全部通过 schema 校验。 |
| formal_import_validation_report_path | artifact | none | false | false | false | 正式导入 validator 报告路径。 |
| formal_evidence_path_resolution_report_path | artifact | none | false | false | false | 正式导入候选记录 evidence paths 在当前工作区或挂载目录下的可解析状态报告路径。 |
| formal_evidence_path_reference_count | metric | none | false | false | false | 正式导入候选记录中声明的 evidence path 引用数量。 |
| existing_formal_evidence_path_count | metric | none | false | false | false | 当前工作区或挂载目录下可解析的正式 evidence path 数量。 |
| direct_formal_evidence_path_count | metric | none | false | false | false | 直接按记录原始路径即可解析的正式 evidence path 数量。 |
| search_resolved_formal_evidence_path_count | metric | none | false | false | false | 通过显式外部镜像根目录按文件名解析到的正式 evidence path 数量。 |
| missing_formal_evidence_path_count | metric | none | false | false | false | 当前工作区或挂载目录下不可解析的正式 evidence path 数量。 |
| formal_evidence_path_resolution_ready | governance | none | false | false | false | 正式导入候选记录的 evidence paths 是否在当前审计边界内全部可解析。 |
| evidence_search_roots | protocol | none | false | false | false | 正式 evidence path 解析时使用的显式外部镜像根目录集合。 |
| resolved_formal_evidence_paths | artifact | none | false | false | false | 在当前审计边界内解析成功的正式 evidence path 集合。 |
| formal_evidence_path_missing_baseline_ids | protocol | none | false | false | false | 存在不可解析 evidence paths 的主表 baseline id 集合。 |
| missing_formal_evidence_paths | artifact | none | false | false | false | 当前工作区或挂载目录下不可解析的正式 evidence path 集合。 |
| formal_import_candidate_records_path | artifact | none | false | false | false | T2SMark 或其他 baseline 写出的正式导入候选 JSONL 路径。 |
| formal_import_candidate_record_count | metric | none | false | false | false | T2SMark 或其他 baseline 写出的正式导入候选记录数量。 |
| candidate_record_count | metric | none | false | false | false | 单个主表 baseline 在正式导入 readiness 行中的候选记录数量。 |
| candidate_record_digest | artifact | none | false | false | false | 主表 external baseline 候选记录集合的稳定摘要。|
| validation_report_digest | artifact | none | false | false | false | 主表 external baseline 候选校验报告的稳定摘要。|
| formal_import_readiness_digest | artifact | none | false | false | false | 主表 external baseline 正式导入 readiness 表的稳定摘要。|
| formal_import_readiness_summary_digest | artifact | none | false | false | false | 主表 external baseline 正式导入 readiness 摘要的稳定摘要。|
| paper_claim_boundary | governance | none | false | false | false | 当前证据与正式论文 claim 之间的边界说明。|
| covered_primary_baseline_count | metric | none | false | false | false | 小样本证据已覆盖的主表 external baseline 数量。|
| covered_primary_baseline_ids | protocol | none | false | false | false | 小样本证据已覆盖的主表 external baseline id 集合。|
| missing_primary_baseline_ids | protocol | none | false | false | false | 小样本证据尚未覆盖的主表 external baseline id 集合。|
| formal_import_ready | governance | none | true | false | false | 单条小样本证据是否已经通过正式导入 validator。|
| formal_import_ready_count | metric | none | false | false | false | 已通过正式导入 validator 的小样本候选记录数量。|
| formal_import_ready_ids | protocol | none | false | false | false | 已通过正式导入 validator 的 baseline id 集合。|
| formal_import_blocking_reasons | governance | none | true | false | false | 单条候选记录未通过正式导入 validator 的原因集合。|
| formal_full_paper_run_requested | governance | none | false | false | false | 当前流程是否请求正式 full paper 规模运行。|
| formal_full_paper_run_permitted | governance | none | false | false | false | 当前项目边界是否允许正式 full paper 规模运行。|
| excluded_operating_points | protocol | none | false | false | false | 当前小样本边界显式排除的论文级 operating point 集合。|
| paper_claim_ready | governance | none | false | false | false | 当前证据是否允许支持论文级 claim。|
| candidate_records_path | artifact | none | false | false | false | 小样本证据写出脚本读取的候选记录路径。|
| validation_report_path | artifact | none | false | false | false | 小样本证据写出脚本读取的 validator 报告路径。|
| prompt_protocol_name | protocol | none | false | false | false | 正式导入结果绑定的 prompt 协议名称。 |
| prompt_protocol_digest | artifact | none | true | false | false | 正式导入结果绑定的 prompt 协议摘要。 |
| prompt_set_name | protocol | none | false | false | false | 正式导入或运行计划绑定的 prompt set 名称。 |
| selected_prompt_count | metric | none | false | false | false | 当前运行实际选择的 prompt 数量。 |
| prompt_limit | protocol | none | false | false | false | 当前论文运行层级的 Prompt 处理上限。 |
| paper_run_prompt_count | metric | none | false | false | false | 当前论文运行层级的 Prompt 源文件总数。 |
| paper_run_prompt_source_path | artifact | none | false | false | false | 当前论文运行层级的 Prompt 源文件路径。 |
| t2smark_formal_reproduction_ready | artifact | none | false | false | false | T2SMark SD3.5 Medium 正式攻击矩阵运行与 adapter 转换链路是否跑通。 |
| t2smark_formal_import_validation_report_path | artifact | none | false | false | false | T2SMark formal 正式导入候选记录的 validator 报告路径。 |
| formal_attack_families | protocol | none | false | false | false | T2SMark formal 运行要求官方入口实际执行的正式攻击名称集合。 |
| t2smark_formal_attack_names | protocol | none | false | false | false | T2SMark formal 结果必须覆盖的受治理攻击名称集合。 |
| t2smark_formal_attack_result_count | metric | none | false | false | false | 同时包含全部正式攻击结果的 T2SMark 样本数。 |
| t2smark_formal_attack_ready | governance | none | false | false | false | T2SMark 每个已选 Prompt 样本是否均完成全部正式攻击。 |
| example_digest_random | random | _digest_random | true | false | false | 可复现随机轨迹的 digest 字段。 |
| example_state_intermediate | intermediate | _intermediate | true | false | true | 跨步骤保存的示例中间状态字段, 正式产物生成前需要清理或迁移。 |
| example_artifact_temporary | temporary | _temporary | false | false | true | 可清理的示例临时产物标记。 |
| example_result_cache | cache | _cache | false | false | false | 可由输入、配置和代码重建的示例缓存标记。 |
| external_baseline_method_faithful_ready | artifact | none | false | false | false | 外部 baseline 真实 method-faithful 链路是否已跑通并生成可审计结果包。|
| t2smark_real_method_faithful_ready | artifact | none | false | false | false | T2SMark 官方 SD3.5 Medium 真实 GPU 最小复现是否已生成或复用成功。|
| t2smark_official_result_generated | artifact | none | false | false | false | T2SMark 官方 results.json 是否由本次运行生成。|
| t2smark_official_result_reused | artifact | none | false | false | false | T2SMark 官方 results.json 是否来自通过当前配置核对的本地或 Drive 结果。|
| t2smark_source_available | artifact | none | false | false | false | T2SMark 官方源码入口在当前工作区中是否可用。|
| t2smark_source_downloaded | artifact | none | false | false | false | T2SMark 官方源码缓存是否由本次冷启动流程下载。|
| source_available | artifact | none | false | false | false | 外部源码缓存入口文件是否存在且可用于后续命令。|
| source_downloaded | artifact | none | false | false | false | 外部源码缓存是否由本次命令补齐。|
| source_entry_path | artifact | none | false | false | false | 外部源码缓存入口脚本路径。|
| adapter_execution_ready | artifact | none | false | false | false | 外部 baseline adapter 命令计划是否执行并通过证据边界校验。|
| adapter_observation_count | metric | none | false | false | false | 外部 baseline adapter 输出的 observation 数量。|
| primary_baseline_adapter_ready | artifact | none | false | false | false | 当前单个 common-backbone external baseline adapter 是否完成真实运行并通过证据边界校验。|
| primary_baseline_observation_count | metric | none | false | false | false | 当前单个 common-backbone external baseline adapter 输出的 observation 数量。|
| primary_baseline_prompt_plan_path | artifact | none | false | false | false | 当前单个 common-backbone method-faithful adapter 读取的完整受治理 Prompt 计划路径。|
| primary_baseline_evidence_id | artifact | none | true | false | false | 主表 external baseline 证据边界记录的稳定标识。|
| primary_baseline_evidence_digest | artifact | none | true | false | false | 主表 external baseline 证据边界记录的稳定摘要。|
| adapter_run_ready | artifact | none | false | false | false | 单个主表 external baseline 的真实 adapter 命令是否成功并产生正式 observation。|
| adapter_run_ready_count | metric | none | false | false | false | 真实 adapter 运行已成功的主表 external baseline 数量。|
| adapter_run_ready_ids | protocol | none | false | false | false | 真实 adapter 运行已成功的主表 external baseline id 集合。|
| adapter_run_observation_count | metric | none | false | false | false | 单个主表 external baseline 在 method-faithful 链路中产生的 observation 数量。|
| adapter_run_execution_devices | runtime | none | false | false | false | 单个主表 external baseline method-faithful observation 中记录的执行设备集合。|
| adapter_run_sample_roles | protocol | none | false | false | false | 单个主表 external baseline method-faithful observation 中记录的样本角色集合。|
| adapter_run_latent_shapes | runtime | none | false | false | false | 单个主表 external baseline method-faithful observation 中记录的 latent shape 集合。|
| method_faithful_adapter_ready | governance | none | false | false | false | 主表 external baseline 是否已达到方法忠实 SD3.5 适配边界。|
| method_faithful_adapter_status_id | artifact | none | true | false | false | 主表 external baseline 方法忠实适配协议状态记录的稳定标识。|
| method_faithful_adapter_status_digest | artifact | none | true | false | false | 主表 external baseline 方法忠实适配协议状态记录的稳定摘要。|
| protocol_role | protocol | none | true | false | false | baseline 在当前协议中的角色, 例如 method-faithful adapter 必需项或 native official reproduction。|
| expected_adapter_boundary | governance | none | true | false | false | 方法忠实适配协议要求的 adapter 边界名称。|
| observed_adapter_boundaries | governance | none | true | false | false | observation 中实际出现的 adapter 边界集合。|
| clean_negative_count | metric | none | true | false | false | 方法忠实适配协议中 clean negative observation 数量。|
| positive_source_count | metric | none | true | false | false | 方法忠实适配协议中 positive source observation 数量。|
| attacked_observation_count | metric | none | true | false | false | 方法忠实适配协议中攻击后 observation 数量。|
| score_protocol_ready | governance | none | true | false | false | observation 是否具备 score、threshold、score_name 与分数方向字段。|
| image_provenance_ready | governance | none | true | false | false | observation 是否具备 image_path 与 image_digest 图像 provenance 字段。|
| formal_import_candidate_allowed | governance | none | true | false | false | 单个 baseline 是否允许作为正式导入候选继续进入后续共同协议。|
| method_faithful_adapter_required_count | metric | none | false | false | false | 方法忠实 SD3.5 适配协议要求覆盖的主表 baseline 数量。|
| method_faithful_adapter_ready_count | metric | none | false | false | false | 已达到方法忠实 SD3.5 适配边界的主表 baseline 数量。|
| method_faithful_adapter_ready_ids | protocol | none | false | false | false | 已达到方法忠实 SD3.5 适配边界的主表 baseline id 集合。|
| missing_method_faithful_adapter_ids | protocol | none | false | false | false | 尚未达到方法忠实 SD3.5 适配边界的主表 baseline id 集合。|
| native_official_reproduction_ids | protocol | none | false | false | false | 不需要 method-faithful adapter 的 native official reproduction baseline id 集合。|
| method_faithful_adapter_protocol_ready | governance | none | false | false | false | 主表 diffusion watermark baseline 是否均达到方法忠实 SD3.5 适配协议边界。|
| formal_import_candidate_allowed_ids | protocol | none | false | false | false | 已允许进入正式导入候选的 method-faithful adapter baseline id 集合。|
| input_observation_count | metric | none | false | false | false | 协议写出脚本读取到的输入 observation 数量。|
| paper_run_prompt_protocol_ready | governance | none | false | false | false | 主表 external baseline 是否已覆盖当前论文运行层级的完整 Prompt 协议。|
| fixed_fpr_baseline_calibration_ready | governance | none | false | false | false | 主表 external baseline 是否已完成 fixed-FPR 校准。|
| attack_matrix_baseline_detection_ready | governance | none | false | false | false | 主表 external baseline 是否已完成共同攻击矩阵下的检测。|
| formal_evidence_paths_ready | governance | none | false | false | false | 主表 external baseline 是否已绑定正式证据路径。|
| formal_result_ready | governance | none | false | false | false | 单个主表 external baseline 是否已具备正式共同协议结果。|
| formal_result_ready_count | metric | none | false | false | false | 已具备正式共同协议结果的主表 external baseline 数量。|
| formal_result_ready_ids | protocol | none | false | false | false | 已具备正式共同协议结果的主表 external baseline id 集合。|
| primary_baseline_formal_ready | governance | none | false | false | false | 四个主表 external baseline 是否全部具备正式共同协议结果。|
| blocked_primary_baseline_ids | protocol | none | false | false | false | 尚未具备正式共同协议结果的主表 external baseline id 集合。|
| blocking_reason_count | metric | none | false | false | false | 单个主表 baseline readiness 行中的阻断原因数量。|
| blocking_reasons | governance | none | false | false | false | 证据边界记录中阻断正式结果声明的原因集合。|
| dominant_blocking_reasons | governance | none | false | false | false | 主表 baseline 正式导入 readiness 摘要中的主要阻断原因集合。|
| dominant_formal_import_blocking_reasons | governance | none | false | false | false | 外部 baseline 对比运行报告中透传的正式导入主要阻断原因集合。|
| missing_formal_attack_resource_profile | governance | none | false | false | false | 主表 baseline 候选是否缺少当前正式攻击矩阵要求的资源档位边界。|
| missing_paper_run_prompt_protocol | governance | none | false | false | false | 主表 baseline 候选是否缺少当前论文运行层级的完整 Prompt 协议边界。|
| missing_fixed_fpr_baseline_calibration | governance | none | false | false | false | 主表 baseline 候选是否缺少 fixed-FPR baseline 校准边界。|
| missing_attack_matrix_baseline_detection | governance | none | false | false | false | 主表 baseline 候选是否缺少共同攻击矩阵检测边界。|
| expected_formal_template_count | metric | none | false | false | false | 单个主表 baseline 需要覆盖的正式共同协议模板数量。|
| formal_template_record_count | metric | none | false | false | false | 主表 baseline 正式共同协议模板总数量。|
| candidate_template_match_count | metric | none | false | false | false | 单个主表 baseline 候选记录中匹配正式模板键的数量。|
| accepted_template_match_count | metric | none | false | false | false | 单个主表 baseline 已通过 validator 且匹配正式模板键的数量。|
| missing_candidate_template_count | metric | none | false | false | false | 主表 baseline 正式共同协议模板中尚未出现候选记录匹配的数量。|
| missing_formal_template_count | metric | none | false | false | false | 主表 baseline 正式共同协议模板仍缺失的数量。|
| unexpected_candidate_record_count | metric | none | false | false | false | 候选记录中不属于当前正式攻击模板键的记录数量。|
| unexpected_accepted_record_count | metric | none | false | false | false | 通过行级校验但不属于当前正式攻击模板键的记录数量。|
| duplicate_candidate_template_count | metric | none | false | false | false | 候选记录中重复的 baseline 与攻击模板键数量。|
| duplicate_accepted_template_count | metric | none | false | false | false | 已通过行级校验记录中重复的 baseline 与攻击模板键数量。|
| formal_template_coverage_ready | governance | none | false | false | false | 单个主表 baseline 是否已覆盖所有正式共同协议模板。|
| formal_template_coverage_ready_count | metric | none | false | false | false | 已覆盖所有正式共同协议模板的主表 baseline 数量。|
| formal_template_coverage_ready_ids | protocol | none | false | false | false | 已覆盖所有正式共同协议模板的主表 baseline id 集合。|
| primary_baseline_formal_template_coverage_ready | governance | none | false | false | false | 四个主表 external baseline 是否全部覆盖正式共同协议模板。|
| formal_template_coverage_digest | artifact | none | false | false | false | 主表 external baseline 正式模板覆盖表的稳定摘要。|
| formal_template_coverage_summary_digest | artifact | none | false | false | false | 主表 external baseline 正式模板覆盖摘要的稳定摘要。|
| formal_evidence_collection_id | artifact | none | false | false | false | 主表 external baseline 正式证据收集计划行的稳定标识。 |
| formal_evidence_collection_digest | artifact | none | false | false | false | 主表 external baseline 正式证据收集计划行的稳定摘要。 |
| formal_evidence_collection_ready | governance | none | false | false | false | 单条正式模板是否已经具备可导入的正式证据记录。 |
| formal_evidence_collection_task_count | metric | none | false | false | false | 主表 external baseline 正式证据收集任务总数。 |
| ready_formal_evidence_collection_task_count | metric | none | false | false | false | 已具备正式证据记录的主表 external baseline 任务数量。 |
| missing_formal_evidence_collection_task_count | metric | none | false | false | false | 仍需补齐正式证据记录的主表 external baseline 任务数量。 |
| primary_baseline_formal_evidence_collection_ready | governance | none | false | false | false | 四个主表 external baseline 是否全部完成正式证据收集。 |
| required_collection_actions | protocol | none | false | false | false | 正式证据收集计划中对缺失模板给出的后续补证动作集合。 |
| required_result_record_path | artifact | none | false | false | false | 正式模板或证据收集计划要求写入或导入的当前 `paper_run_name` 独占结果记录路径。 |
| formal_evidence_collection_plan_digest | artifact | none | false | false | false | 主表 external baseline 正式证据收集计划的稳定摘要。 |
| formal_evidence_collection_summary_digest | artifact | none | false | false | false | 主表 external baseline 正式证据收集摘要的稳定摘要。 |
| adapter_boundary | governance | none | false | false | false | adapter observation 或 manifest 对工程 smoke 与正式论文证据边界的说明。|
| execution_device | runtime | none | false | false | false | adapter 张量或诊断分数实际执行设备。|
| torch_available | runtime | none | false | false | false | adapter 运行环境中是否可导入 torch。|
| adapter_seed | runtime | none | false | false | false | adapter 为单条 prompt 派生的可复现整数种子。|
| score_metadata | runtime | none | false | false | false | adapter manifest 中按 prompt 记录的轻量分数计算元数据。|
| adapter_unsupported_reason | governance | none | false | false | false | 外部 baseline adapter 未能通过 method-faithful 链路时记录的边界原因。|
| official_result_generated | artifact | none | false | false | false | 外部 baseline 官方结果是否由本次命令生成。|
| official_result_reused | artifact | none | false | false | false | 外部 baseline 官方结果是否由通过当前配置核对的本地结果复用。|
| official_generation_reason | governance | none | false | false | false | 外部 baseline 官方结果生成或复用决策的原因。|
| official_results_path | artifact | none | false | false | false | 外部 baseline 官方结果文件路径。|
| official_return_code | runtime | none | false | false | false | 外部 baseline 官方入口命令返回码。|
| official_command | runtime | none | false | false | false | 外部 baseline 官方入口命令 argv 列表。|
| image_pairs_path | artifact | none | false | false | false | 外部 baseline adapter 使用的 image pair 输入路径。|
| baseline_execution_manifest_path | artifact | none | false | false | false | 外部 baseline adapter 执行 manifest 路径。|
| command_plan_path | artifact | none | false | false | false | 外部 baseline adapter 命令计划路径。|
| clone_return_code | runtime | none | false | false | false | 外部源码缓存 git clone 命令返回码。|
| checkout_return_code | runtime | none | false | false | false | 外部源码缓存 git checkout 命令返回码。|
| t2smark_source_patch_applied | artifact | none | false | false | false | T2SMark 官方源码兼容补丁是否由本次 helper 应用。|
| source_patch_applied | artifact | none | false | false | false | 外部源码缓存兼容补丁是否在本次调用中写入源码缓存。|
| source_patch_needed | artifact | none | false | false | false | 外部源码缓存是否在本次调用中检测到需要补丁写入。|
| source_patch_path | artifact | none | false | false | false | 外部源码缓存兼容补丁作用的文件路径。|
| source_patch_reason | governance | none | false | false | false | 外部源码缓存兼容补丁被应用的原因说明。|
| source_patch_report | artifact | none | false | false | false | 外部源码缓存兼容补丁报告对象。|
| source_prepare_skipped | governance | none | false | false | false | 已有外部官方结果可复用时是否跳过源码缓存准备。|

| prompt_file | protocol | none | false | false | false | prompt 协议使用的配置文件路径。|
| risk_profile_counts | protocol | none | false | false | false | 按 risk_profile 聚合的 prompt 数量。|
| calibration_clean_negative_count | metric | none | false | false | false | fixed-FPR 校准 split 中 clean negative 样本数量。|
| test_clean_negative_count | metric | none | false | false | false | fixed-FPR 测试 split 中 clean negative 样本数量。|
| protocol_target_ready | governance | none | true | false | false | 单方法阈值审计中的目标 FPR 是否与当前论文运行协议完全一致。 |
| protocol_value_ready | governance | none | true | false | false | 单方法阈值及其摘要是否与从 calibration clean negative 独立重算的结果完全一致。 |
| detection_decision_ready | governance | none | true | false | false | observation 的逐条检测判定是否与独立重算的冻结阈值判定完全一致。 |
| split_count_ready | governance | none | true | false | false | 单方法 calibration 与 test clean negative 数量是否与当前论文层级的固定划分完全一致。 |
| fixed_fpr_threshold_ready | governance | none | true | false | false | 单方法是否通过目标值、阈值重算、逐条判定与样本数量联合门禁。 |
| observation_source_sha256 | provenance | none | true | true | false | 单方法 fixed-FPR 阈值审计实际读取的 observation 文件字节 SHA-256。 |
| expected_method_ids | protocol | none | false | false | false | 统一 fixed-FPR 阈值审计必须覆盖的精确方法标识集合。 |
| audited_method_ids | protocol | none | false | false | false | 统一 fixed-FPR 阈值审计实际覆盖的方法标识序列。 |
| audited_method_count | metric | none | false | false | false | 统一 fixed-FPR 阈值审计实际核验的方法数量。 |
| method_identity_ready | governance | none | false | false | false | 实际方法标识是否无重复且与受治理方法集合完全相等。 |
| all_method_thresholds_ready | governance | none | false | false | false | 主方法与四个外部 baseline 是否全部通过各自的独立阈值重算门禁。 |
| method_observation_source_sha256_map | provenance | none | true | true | false | 五方法身份到实际 observation 文件字节 SHA-256 的精确映射。 |
| threshold_audit_rows_digest | provenance | none | true | true | false | 对按方法身份规范排序的完整 fixed-FPR 阈值审计行计算的稳定 SHA-256 摘要。 |
| threshold_observation_binding_ready | governance | none | false | false | false | 五方法身份、observation 文件字节摘要与冻结阈值摘要是否形成无缺失且格式有效的精确绑定。 |
| fixed_fpr_threshold_audit_ready | governance | none | false | false | false | 五个方法的身份、目标 FPR、冻结阈值和逐条判定是否共同完成正式证据审计。 |
| pilot_paper_common_protocol_ready | governance | none | false | false | false | pilot_paper 级 fixed-FPR 共同协议是否完成运行前治理冻结。|
| pilot_paper_prompt_count | metric | none | false | false | false | pilot_paper prompt split 中的 prompt 数量。|
| pilot_paper_prompt_split_ready | governance | none | false | false | false | pilot_paper prompt split 是否可供共同协议使用。|
| pilot_paper_target_fpr | protocol | none | false | false | false | pilot_paper 共同协议使用的 fixed-FPR 目标值。|
| paper_target_fpr | protocol | none | false | false | false | 当前论文运行层级使用的 fixed-FPR 目标值, pilot_paper 默认为 0.01, full_paper 默认为 0.001。|
| expected_target_fpr | protocol | none | false | false | false | 当前论文运行层级按协议应匹配的 fixed-FPR 目标值。|
| pilot_paper_negative_count_minimum_required | metric | none | false | false | false | pilot_paper fixed-FPR 校准所要求的最小 clean negative 数量。|
| minimum_clean_negative_count | metric | none | false | false | false | fixed-FPR 协议要求的完整 test split clean negative 样本数, 三类运行层级分别为34、340、3400。|
| minimum_result_positive_count | metric | none | false | false | false | 结果导入 schema 要求的完整 test split positive 样本数, 与当前运行层级的 test 数量一致。|
| minimum_result_negative_count | metric | none | false | false | false | 结果导入 schema 要求的完整 test split negative 样本数, 不允许不完整统计进入论文 claim 边界。|
| minimum_result_attacked_negative_count | metric | none | false | false | false | 结果导入 schema 要求的每个攻击设置完整 test split attacked negative 样本数。|
| pilot_paper_negative_count_ready | governance | none | false | false | false | pilot_paper prompt split 中 clean negative 数量是否满足最小要求。|
| pilot_paper_attack_count | metric | none | false | false | false | pilot_paper 共同协议攻击矩阵中的攻击配置数量。|
| pilot_paper_method_count | metric | none | false | false | false | pilot_paper 共同协议覆盖的方法数量。|
| pilot_paper_import_template_count | metric | none | false | false | false | pilot_paper 共同协议生成的 method × attack 导入模板数量。|
| pilot_paper_result_import_ready | governance | none | false | false | false | pilot_paper 结果候选记录是否已通过受治理导入 schema。|
| accepted_pilot_paper_import_count | metric | none | false | false | false | 已通过 pilot_paper 受治理导入 schema 的结果记录数量。|
| rejected_pilot_paper_import_count | metric | none | false | false | false | 未通过 pilot_paper 受治理导入 schema 的结果记录数量。|
| pilot_paper_import_issue_count | metric | none | false | false | false | pilot_paper 受治理导入 schema 发现的问题数量。|
| accepted_pilot_paper_claim_record_count | metric | none | false | false | false | 已通过 pilot_paper 受治理导入且显式允许支撑 pilot_paper 论文主张的结果记录数量。|
| pilot_paper_claim_record_ready | governance | none | false | false | false | pilot_paper 结果导入记录是否已覆盖模板并具备 pilot_paper 论文主张记录边界。|
| pilot_paper_evidence_coverage_ready | governance | none | false | false | false | pilot_paper 结果是否已覆盖共同协议模板并完成受治理证据导入。|
| pilot_paper_effectiveness_gate_ready | governance | none | false | false | false | pilot_paper 结果是否通过方法效果门禁并允许支持优势性主张。|
| pilot_paper_effectiveness_gate_reason | governance | none | false | false | false | pilot_paper 方法效果门禁未通过或通过的原因代码。|
| slm_wm_mean_true_positive_rate | metric | none | false | false | false | SLM-WM 在共同攻击模板上的平均 true positive rate。|
| slm_wm_mean_false_positive_rate | metric | none | false | false | false | SLM-WM 在共同攻击模板上的平均 false positive rate。|
| slm_wm_fixed_fpr_boundary_ready | governance | none | false | false | false | SLM-WM 平均 false positive rate 是否满足 fixed-FPR 目标边界。|
| best_baseline_mean_true_positive_rate | metric | none | false | false | false | 主表 baseline 中平均 true positive rate 最高的方法结果。|
| best_baseline_method_id | protocol | none | false | false | false | 主表 baseline 中平均 true positive rate 最高的方法标识。|
| missing_baseline_methods | governance | none | false | false | false | 方法效果门禁中缺失结果记录的 baseline 方法集合。|
| method_attack_templates_aligned | governance | none | false | false | false | SLM-WM 与 baseline 是否在相同攻击模板上进行比较。|
| method_attack_template_keys_unique | governance | none | false | false | false | 各方法的攻击模板结果键是否均无重复记录。|
| paper_run_result_import_coverage_ready | governance | none | false | false | false | 当前论文运行的已接受结果是否严格覆盖 method × attack 模板。|
| paper_run_result_missing_template_count | metric | none | false | false | false | 当前论文运行已接受结果中缺失的 method × attack 模板数。|
| paper_run_result_unexpected_template_count | metric | none | false | false | false | 当前论文运行已接受结果中额外的 method × attack 模板数。|
| paper_run_result_duplicate_template_count | metric | none | false | false | false | 当前论文运行已接受结果中重复的 method × attack 模板行数。|
| paper_run_template_registry_unique | governance | none | false | false | false | 当前论文运行的正式导入模板注册表是否无重复键。|
| pilot_paper_claim_ready | governance | none | false | false | false | pilot_paper 规模论文主张是否已由同一 prompt split、同一攻击矩阵、fixed-FPR 协议和受治理导入记录闭合。|
| pilot_paper_supports_superiority_claim | governance | none | false | false | false | pilot_paper 结果是否允许在 pilot_paper 样本规模边界内支撑方法优越性主张。|
| full_paper_claim_ready | governance | none | false | false | false | full_paper 规模论文主张是否已具备证据闭合。pilot_paper 共同协议中必须为 false。|
| paper_claim_scale | governance | none | false | false | false | 当前结果或协议允许支撑的论文主张规模, 例如 pilot_paper 或 full_paper。|
| paper_run_claim_type | governance | none | false | false | false | 当前论文运行层级对应的正式主张类型, probe_paper、pilot_paper 与 full_paper 分别对应 probe_claim、pilot_claim 与 full_claim。|
| probe_claim_ready | governance | none | false | false | false | probe_paper 小规模正式流程主张是否已经通过同一协议门禁。|
| pilot_claim_ready | governance | none | false | false | false | pilot_paper 中规模正式流程主张是否已经通过同一协议门禁。|
| full_claim_ready | governance | none | false | false | false | full_paper 全规模正式流程主张是否已经通过同一协议门禁。|
| strict_formal_evidence_required | governance | none | false | false | false | 当前共同协议是否要求结果记录只能来自正式真实测量证据。|
| strict_formal_result_ready | governance | none | true | false | false | 单条共同协议结果记录是否已经通过正式真实测量证据门禁。|
| nonformal_evidence_rejection_policy | governance | none | false | false | false | 非正式证据进入共同协议结果导入时的拒绝策略。|
| paper_run_complete_result_package_ready | governance | none | false | false | false | 当前论文运行层级完整结果包是否已经覆盖所有必需输出目录并完成归档。|
| probe_paper_complete_result_package_ready | governance | none | false | false | false | probe_paper 完整结果包是否已经覆盖所有必需输出目录并完成归档。|
| pilot_paper_complete_result_package_ready | governance | none | false | false | false | pilot_paper 完整结果包是否已经覆盖所有必需输出目录并完成归档。|
| full_paper_complete_result_package_ready | governance | none | false | false | false | full_paper 完整结果包是否已经覆盖所有必需输出目录并完成归档。|
| paper_run_claim_ready | governance | none | false | false | false | 当前论文运行层级的正式主张是否已经通过共同协议、证据覆盖和优势性门禁。|
| full_paper_claim_boundary | governance | none | false | false | false | pilot_paper 结果与 full_paper 规模论文主张之间的样本规模边界说明。|
| prompt_split_digest | artifact | none | false | false | false | 共同协议使用的 prompt split 稳定摘要。|
| attack_matrix_digest | artifact | none | false | false | false | 共同协议使用的攻击矩阵稳定摘要。|
| fixed_fpr_protocol_digest | artifact | none | false | false | false | fixed-FPR 校准协议的稳定摘要。|
| confidence_interval_method | protocol | none | false | false | false | 显式有界样本均值使用的分布无关置信区间方法, 当前固定为 `bounded_hoeffding`, 区间宽度由对应指标的上下界决定.|
| confidence_level | protocol | none | false | false | false | 置信区间使用的置信水平。|
| result_scope | protocol | none | false | false | false | 结果记录所属的共同协议范围。|
| result_claim_scope | governance | none | false | false | false | 结果记录允许支撑的声明范围。|
| governed_import_required | governance | none | false | false | false | 方法结果是否必须通过受治理导入协议进入对比表。|
| method_ids | protocol | none | false | false | false | 当前共同协议要求覆盖的方法 id 集合。|
| required_rate_fields | protocol | none | false | false | false | 导入 schema 要求按 [0,1] 率值校验的字段集合。|
| metric_bounds | protocol | none | false | false | false | 导入 schema 为每个正式指标声明数学上下界, 检测率与分数保持率使用 [0,1], SSIM 质量均值使用 [-1,1].|
| ci_field_groups | protocol | none | false | false | false | 导入 schema 中 metric 与置信区间字段组合, 每组区间必须服从 `metric_bounds` 声明的范围.|
| ci_count_fields | protocol | none | false | false | false | 导入 schema 为每个置信区间指标声明原始样本计数字段, 用于精确重建 bounded Hoeffding 区间.|
| pilot_paper_result_template_id | artifact | none | true | false | false | pilot_paper 共同协议结果导入模板行的稳定标识。|
| pilot_paper_result_template_digest | artifact | none | true | false | false | pilot_paper 共同协议结果导入模板行的稳定摘要。|
| quality_score_mean | metric | none | true | false | false | 当前论文运行层级共同协议中 attacked positive 的实测 SSIM 均值, 合法范围为 [-1,1], 不执行 [0,1] 裁剪.|
| quality_score_ci_low | metric | none | true | false | false | SSIM 均值 Hoeffding 置信区间下界, 使用 [-1,1] 观测范围和 range_width=2.|
| quality_score_ci_high | metric | none | true | false | false | SSIM 均值 Hoeffding 置信区间上界, 使用 [-1,1] 观测范围和 range_width=2.|
| score_retention_ci_low | metric | none | true | false | false | 攻击后分数保持率 Hoeffding 置信区间下界, 使用 [0,1] 观测范围和 range_width=1.|
| score_retention_ci_high | metric | none | true | false | false | 攻击后分数保持率 Hoeffding 置信区间上界, 使用 [0,1] 观测范围和 range_width=1.|
| baseline_formal_import_record_accepted | governance | none | true | false | false | 外部 baseline 候选记录是否已经被主表受治理导入报告接受。|
| template_covered | governance | none | true | false | false | 单个 pilot_paper method × attack 模板是否已被结果记录覆盖。|
| pilot_paper_result_record_digest | artifact | none | true | false | false | pilot_paper 共同协议结果记录的稳定摘要。|
| pilot_paper_result_record_id | artifact | none | true | false | false | pilot_paper 共同协议结果记录的稳定标识。|
| result_source_kind | governance | none | true | false | false | pilot_paper 结果记录来源类型, 区分 SLM-WM 攻击矩阵与外部 baseline 结果。|
| dataset_quality_formal_metric_ready | governance | none | true | false | false | 数据集级正式 FID / KID 指标是否可以支撑共同协议结果记录。|
| attacked_image_latent_rescore | method | none | true | false | false | attacked image 经过 VAE latent 投影后的直接水印重检测摘要对象。|
| attacked_image_rescore_count | metric | none | true | false | false | 已完成 attacked image 直接水印重检测的记录数量。|
| attacked_image_rescore_ready | governance | none | true | false | false | attacked image 直接水印重检测是否覆盖当前闭环内全部真实 attacked image。|
| attacked_image_rescore_performed | governance | none | true | false | false | 是否已经对 attacked image 执行直接水印重检测。|
| attacked_image_rescore_required_for_claim | governance | none | true | false | false | 当前记录若要支撑论文主张是否仍需要 attacked image 直接水印重检测。|
| same_threshold_rescue_source | governance | none | true | false | false | same-threshold 几何恢复重判所使用的分数来源边界。|
| attacked_latent_projection_digest | artifact | none | true | false | false | attacked image 经 VAE 编码和槽位投影后的稳定摘要。|
| watermark_coordinate | metric | none | true | false | false | attacked image latent 投影在 source clean-to-aligned 水印方向上的无界坐标。|
| bounded_watermark_coordinate | metric | none | true | false | false | 用于正式分数映射的 bounded watermark coordinate。|
| decision_mode | protocol | none | true | false | false | fixed-FPR 诊断表中的判定模式, 用于区分 raw、aligned 和 rescue 后 evidence。|
| score_field | metric | none | true | false | false | fixed-FPR 诊断表使用的分数字段或布尔判定字段。|
| score_mode_operating_points | artifact | none | true | false | false | raw、aligned 和 rescue 后 evidence 三种判定模式的 fixed-FPR operating point 诊断行集合。|
| detector_input_access_mode | governance | none | true | false | false | 检测器需要访问的输入类型, 用于区分 generation latent trace、最终图像或方法原生输入。|
| blind_image_detector | governance | none | true | false | false | 检测器是否仅依赖最终图像而不访问生成轨迹或内部 latent。|
| latent_trace_detection_claim_boundary | governance | none | true | false | false | 依赖生成轨迹的检测结果不能被误表述为 blind image watermark detection 的声明边界。|
| baseline_fairness_boundary | governance | none | true | false | false | 外部 baseline 对比所需的检测权限公平性边界说明。|
| runtime_weighted_content_component_norm | metric | none | true | false | false | 运行时写入 carrier 中 content 分量乘权重后的范数。|
| runtime_weighted_attention_component_norm | metric | none | true | false | false | 运行时写入 carrier 中 attention geometry 分量乘权重后的范数。|
| runtime_content_attention_norm_ratio | metric | none | true | false | false | 运行时 content 分量与 attention 分量的加权范数比。|
| runtime_final_carrier_content_cosine | metric | none | true | false | false | 最终写入 carrier 与 content carrier 方向的余弦相似度。|
| runtime_final_carrier_attention_cosine | metric | none | true | false | false | 最终写入 carrier 与带符号 attention carrier 方向的余弦相似度。|
| content_update_width | metric | none | true | false | false | aligned rescoring 批次中用于 latent projection 的内容 update 向量宽度。|
| content_update_width_consistent | governance | none | true | false | false | aligned rescoring 批次内所有 content update 是否具有一致向量宽度。|
| skipped_output_entries | governance | none | true | false | false | 被结果物化层跳过的 zip 条目名称集合。|
| materialized_output_entries_digest | artifact | none | true | false | false | 已物化 outputs/ 条目名称集合的稳定摘要。|
| skipped_output_entry_count | metric | none | true | false | false | 由于非 outputs/ 路径或路径越界而被跳过的结果包条目数量。|
| materialized_output_entry_count | metric | none | true | false | false | 从结果包中成功物化到 outputs/ 的条目数量。|
| input_package_paths | artifact | none | true | false | false | 参与结果物化的 zip 结果包路径集合。|
| materialization_report | artifact | none | true | false | false | 从 Google Drive 或其他结果包物化 outputs 条目的摘要报告对象。|
| missing_template_examples | governance | none | true | false | false | pilot_paper 结果记录物化摘要中用于诊断的缺失模板示例集合。|
| pilot_paper_template_coverage_ready | governance | none | true | false | false | pilot_paper 结果记录是否覆盖全部 method × attack 导入模板。|
| pilot_paper_template_missing_count | metric | none | true | false | false | 尚未被结果记录覆盖的 pilot_paper 导入模板数量。|
| pilot_paper_template_covered_count | metric | none | true | false | false | 已被结果记录覆盖的 pilot_paper 导入模板数量。|
| pilot_paper_template_record_count | metric | none | true | false | false | pilot_paper 共同协议结果导入模板总数量。|
| pilot_paper_result_method_ids | protocol | none | true | false | false | pilot_paper 结果记录实际覆盖的方法 id 集合。|
| pilot_paper_result_record_count | metric | none | true | false | false | pilot_paper 共同协议结果记录物化后的记录数量。|
| expected_superiority_row_count | metric | none | true | false | false | 正式攻击矩阵应产生的逐攻击优越性比较行数。|
| expected_result_record_count | metric | none | true | false | false | 五种方法与正式攻击矩阵笛卡尔积对应的预期结果记录数。|
| actual_result_record_count | metric | none | true | false | false | 结果分析实际读取的记录行数, 包含可能的重复键。|
| unique_result_record_key_count | metric | none | true | false | false | 结果记录中唯一 method × attack 模板键的数量。|
| duplicate_result_record_count | metric | none | true | false | false | 结果记录中重复 method × attack 模板键的多余行数。|
| missing_result_record_count | metric | none | true | false | false | 五种方法的正式 method × attack 模板中缺失的记录数。|
| unexpected_result_record_count | metric | none | true | false | false | 不属于当前正式 method × attack 模板的记录键数。|
| result_template_coverage_ready | governance | none | true | false | false | 结果记录是否无缺失、无额外且无重复地覆盖正式 method × attack 模板。|
| confidence_interval_row_count | metric | none | true | false | false | 结果分析表中完成披露的 method × attack 置信区间行数。|
| per_attack_superiority_row_count | metric | none | true | false | false | 逐攻击表中 SLM-WM 与当前攻击下最强主表 baseline 的比较行数。|
| superiority_claim_ready_count | metric | none | true | false | false | 逐攻击比较中保守置信区间下达到显著优势的披露行数, 不作为完整分析闭合的必要条件。|
| per_attack_ci_coverage_ready | governance | none | true | false | false | 所有正式结果记录是否均形成完整、自洽且可审计的逐攻击置信区间行。|
| per_attack_superiority_evaluation_ready | governance | none | true | false | false | 每个正式攻击是否均形成可审计的 SLM-WM 与最强主表 baseline 比较行。|
| universal_per_attack_superiority_claim_ready | governance | none | true | false | false | 是否每个正式攻击的保守置信区间均支持 SLM-WM 显著胜出; 该字段只限定全攻击普遍优势主张。|
| failure_case_record_count | metric | none | true | false | false | 失败案例图绑定的真实攻击检测失败记录数量。|
| failure_case_figure_ready | governance | none | true | false | false | 失败案例图是否已由受治理攻击记录和实际攻击图像重建。|
| missing_result_record_examples | governance | none | true | false | false | 结果模板覆盖报告中的缺失键示例。|
| unexpected_result_record_examples | governance | none | true | false | false | 结果模板覆盖报告中的额外键示例。|
| prompt_split_ready | governance | none | false | false | false | prompt split 是否满足共同协议使用条件。|
| branch_name | method | none | true | false | false | 分支风险场、Null Space 或载体所属的机制分支。 |
| bundle_digest | method | none | true | false | false | 三个分支风险场组合后的稳定摘要。 |
| eligible_indices | method | none | true | false | false | 分支风险阈值允许承载的位置索引。 |
| jvp_mode | method | none | true | false | false | Jacobian 方向导数的实现方式, 正式值为 torch_func_linearize_exact_jvp、torch_autograd_exact_jvp 或 torch_autograd_exact_jvp_compatibility。 |
| solver_digest | method | none | true | false | false | 真实 JVP/SVD 子空间求解记录摘要。 |
| response_residual | metric | none | true | true | false | 小奇异值系数在联合响应矩阵上的残差范数。 |
| orthogonality_error | metric | none | true | true | false | latent 安全基底相对单位矩阵的正交误差。 |
| projection_energy_retention | metric | none | true | true | false | 固定检测模板投影到安全子空间后保留的能量比例。 |
| tail_robust_score | metric | none | true | true | false | 高斯幅值尾部截断载体的图像盲检相关分数。 |
| attention_geometry_score | metric | none | true | true | false | 真实 Q/K attention 与密钥关系签名的一致性分数。 |
| formal_evidence_positive | metric | none | true | true | false | 冻结完整 evidence 协议后的正式布尔判定。 |
| frozen_content_threshold | protocol | none | true | true | false | calibration clean negative 冻结的内容阈值。 |
| threshold_digest | protocol | none | true | true | false | calibration 分数集合、冻结阈值、阈值来源和相关 evidence 配置的稳定摘要。 |
| jacobian_candidate_count | method | none | true | false | false | 每个分支参与真实 JVP 的候选方向数量。 |
| null_space_rank | method | none | true | false | false | 每个分支保留的小响应 latent 基底秩。 |
| lf_relative_strength | method | none | true | false | false | LF 更新相对当前 latent 范数的强度。 |
| tail_relative_strength | method | none | true | false | false | 尾部截断载体相对当前 latent 范数的强度。 |
| attention_relative_strength | method | none | true | false | false | 注意力几何更新相对当前 latent 范数的强度。 |
| tail_fraction | method | none | true | false | false | 标准高斯模板保留的幅值尾部比例。 |
| source_id | provenance | none | true | false | false | 外部 Prompt 数据来源稳定标识。 |
| source_url | provenance | none | true | false | false | 外部 Prompt 项目主页。 |
| revision_url | provenance | none | false | false | false | 直接指向资源登记40位提交树的 Hugging Face URL。 |
| upstream_repository_id | provenance | none | false | false | false | 公开镜像对应的原始上游仓库标识。 |
| upstream_access_status | provenance | none | false | false | false | 审计时原始上游仓库的可访问状态。 |
| registry_schema | provenance | none | false | false | false | 不可变资源或正式依赖 profile 登记表的 schema 名称。 |
| schema_version | provenance | none | false | false | false | 受治理登记表或报告的结构版本。 |
| source_name | provenance | none | false | false | false | 模型或数据资源在登记表中的稳定语义名称。 |
| provider | provenance | none | false | false | false | 资源实际获取服务提供方。 |
| repository_id | provenance | none | false | false | false | Hugging Face 资源的 owner/name 仓库标识。 |
| repository_type | provenance | none | false | false | false | 资源仓库是 model 还是 dataset。 |
| revision | provenance | none | false | false | false | 资源登记表中的40位小写十六进制不可变提交。 |
| access_policy | provenance | none | false | false | false | 资源的公开或受限访问策略。 |
| usage_roles | provenance | none | false | false | false | 资源在正式实验中允许的明确职责集合。 |
| sources | provenance | none | false | false | false | 不可变模型与数据来源登记表中的资源身份映射。 |
| required_files | provenance | none | false | false | false | 单个模型来源必须逐文件满足的相对路径、SHA-256 和字节大小约束。 |
| vision_model_source | provenance | none | false | false | false | 运行环境报告中绑定仓库、revision 和用途的 CLIP 语义编码器来源记录。 |
| model_snapshot_content | provenance | none | false | false | false | official-reference 本地模型目录的逐文件路径、大小和 SHA-256 记录。 |
| reused_model_snapshot_content | provenance | none | false | false | false | 复用前从已有报告与本地文件重算一致的模型快照记录。 |
| snapshot_content_digest | provenance | none | false | false | false | 本地模型快照仓库、revision 与逐文件记录的稳定摘要。 |
| allow_patterns | provenance | none | false | false | false | 模型快照唯一允许物化并进入逐文件摘要的仓库相对匹配模式集合。 |
| model_source_ready | governance | none | true | false | false | official-reference 是否同时绑定登记仓库、精确 revision 和文件级快照摘要。 |
| model_source_repository_id | provenance | none | true | false | false | official-reference summary 绑定的公开镜像仓库标识。 |
| model_source_revision | provenance | none | true | false | false | official-reference summary 绑定的40位不可变模型提交。 |
| model_snapshot_content_digest | provenance | none | true | false | false | official-reference summary 绑定的本地模型逐文件快照摘要。 |
| model_snapshot_allow_patterns | provenance | none | true | false | false | official-reference summary 绑定的受治理模型组件选择器集合。 |
| model_snapshot_scope_ready | governance | none | true | false | false | 本地模型快照是否只包含正式 loader 所需组件且选择器与登记契约一致。 |
| openclip_checkpoint_requested | protocol | none | false | false | false | 当前 official-reference 是否请求物化精确 OpenCLIP checkpoint。 |
| openclip_checkpoint_ready | governance | none | false | false | false | OpenCLIP checkpoint 是否通过登记仓库、revision、文件大小与 SHA-256 核验。 |
| openclip_source_name | provenance | none | false | false | false | OpenCLIP checkpoint 在不可变模型来源登记表中的稳定名称。 |
| openclip_usage_role | protocol | none | false | false | false | OpenCLIP checkpoint 在 official-reference 中承担的受治理编码器职责。 |
| openclip_model_name | protocol | none | true | false | false | official-reference 实际构造的 OpenCLIP 模型架构名称。 |
| openclip_repository_id | provenance | none | true | false | false | official-reference 实际使用的 OpenCLIP checkpoint 仓库标识。 |
| openclip_revision | provenance | none | true | false | false | official-reference 实际使用的 OpenCLIP checkpoint 40位不可变提交。 |
| openclip_snapshot_dir | provenance | none | false | false | false | 由仓库标识与精确 revision 共同确定的共享 OpenCLIP 快照目录。 |
| openclip_checkpoint_filename | provenance | none | true | false | false | official-reference 传给 OpenCLIP loader 的固定 checkpoint 文件名。 |
| openclip_checkpoint_path | provenance | none | false | false | false | official-reference 传给 OpenCLIP loader 的本地普通文件路径。 |
| openclip_checkpoint_sha256 | provenance | none | true | false | false | OpenCLIP checkpoint 文件内容的登记 SHA-256。 |
| openclip_checkpoint_size_bytes | provenance | none | false | false | false | OpenCLIP checkpoint 文件的登记字节大小。 |
| openclip_snapshot_content_digest | provenance | none | true | false | false | OpenCLIP 共享快照逐文件证据的稳定摘要。 |
| openclip_source_ready | governance | none | true | false | false | official-reference 是否绑定精确 OpenCLIP 架构、仓库、revision、checkpoint 哈希与快照摘要。 |
| snapshot_validation_error | governance | none | false | false | false | 既有本地模型快照无法通过文件级复用校验时的阻断原因。 |
| file_count | provenance | none | false | false | false | 模型快照文件记录数量。 |
| files | provenance | none | false | false | false | 模型快照中受摘要绑定的逐文件记录。 |
| size_bytes | provenance | none | false | false | false | 模型快照单个文件的字节大小。 |
| source_data_url | provenance | none | true | false | false | 固定 revision 的原始 Prompt 文件地址。 |
| source_revision | provenance | none | true | false | false | 外部 Prompt 数据固定 Git revision。 |
| source_file_sha256 | provenance | none | true | false | false | 外部 Prompt 原始文件 SHA-256。 |
| source_record_count | provenance | none | true | false | false | 外部 Prompt 原始文件记录数量。 |
| license | provenance | none | true | false | false | 外部 Prompt 数据随来源仓库登记的许可证。 |
| selection_policy | provenance | none | true | false | false | Prompt 补充数据的确定性分层选择规则。 |
| normalization_policy | provenance | none | true | false | false | Prompt 去重使用的文本规范化规则。 |
| prompt_sets | provenance | none | true | false | false | 各 Prompt 集新增数量、类别分布和摘要。 |
| original_count | provenance | none | true | false | false | Prompt 集补充前数量。 |
| added_count | provenance | none | true | false | false | Prompt 集从登记来源新增的数量。 |
| result_count | provenance | none | true | false | false | Prompt 集补充后的数量。 |
| category_counts | provenance | none | true | false | false | 新增 Prompt 的来源类别分布。 |
| challenge_counts | provenance | none | true | false | false | 新增 Prompt 的来源挑战类型分布。 |
| selected_prompt_digest | provenance | none | true | false | false | 确定性选择后 Prompt 文本集合摘要。 |
| attacked_negative_count | metric | none | false | true | false | 当前攻击设置下完整 test split attacked negative 记录数量, 与 clean negative 数量分开统计。 |
| generation_rerun | protocol | none | true | true | false | 消融配置是否重新执行真实图像生成。 |
| counterfactual_score_transform_used | protocol | none | true | true | false | 消融是否使用禁止替代正式实验的分数变换。 |
| jvp_modes | method | none | true | false | false | 同一注入时刻各载体分支实际使用的精确 JVP 实现模式集合。 |
| content_failure_reason | method | none | true | true | false | 单次图像盲检在当前检测配置下的内容主判失败分类。 |
| formal_content_failure_reason | protocol | none | true | true | false | 应用 calibration 冻结阈值后重新计算的内容主判失败分类。 |
| attention_applied_update_strength | metric | none | true | true | false | 注意力几何单调回溯搜索最终接受的 latent 更新二范数。 |
| attention_backtracking_step_count | metric | none | true | true | false | 注意力几何单调回溯搜索为获得真实 Q/K 分数提升所执行的缩减次数。 |
| completed_prompt_count | metric | none | true | false | false | Colab 数据集续跑中已经通过 manifest 校验的 Prompt 数量。 |
| remaining_prompt_count | metric | none | true | false | false | Colab 数据集续跑中尚未完成的 Prompt 数量。 |
| resumed_prompt_count | metric | none | true | false | false | 当前会话从已完成缓存复用的 Prompt 数量。 |
| new_prompt_count | metric | none | true | false | false | 当前会话新增完成的 Prompt 数量。 |
| completed_run_count | metric | none | true | false | false | 正式消融续跑中已经通过 manifest 校验的配置运行数量。 |
| remaining_run_count | metric | none | true | false | false | 正式消融续跑中尚未完成的配置运行数量。 |
| resumed_run_count | metric | none | true | false | false | 当前会话从缓存复用的正式消融配置运行数量。 |
| new_run_count | metric | none | true | false | false | 当前会话新增完成的正式消融配置运行数量。 |
| active_workflow | governance | none | true | false | false | Colab 续跑当前正在推进的数据集主流程或正式消融流程。 |
| formal_ablation_requested | governance | none | true | false | false | 当前 Colab 会话是否被显式要求推进真实机制消融。 |
| mirrored_archives | artifact | none | true | false | false | 已从持久化工作区镜像到论文运行目录的正式结果包路径映射。 |
| formal_feature_origin | provenance | none | true | false | false | 正式 FID/KID 特征来自直接兼容 Inception 提取还是受治理特征记录导入。 |
| feature_extractor_id | provenance | none | true | false | false | 单张图像正式特征所使用的精确提取器、版本与特征层标识。 |
| max_new_prompts_per_session | protocol | none | true | false | false | Colab 单次会话允许新增完成的最大 Prompt 数量, 0 表示不限制。 |
| max_new_runs_per_session | protocol | none | true | false | false | Colab 单次会话允许新增完成的最大正式消融配置运行数量, 0 表示不限制。 |
| completed_prompt_digest | artifact | none | true | false | false | 已完成 Prompt 标识集合的稳定摘要, 用于续跑状态核对。 |
| preferred_direction_count | method | none | true | false | false | Jacobian 候选矩阵中显式纳入的固定载体或注意力梯度方向数量。 |
| preferred_direction_role | method | none | true | false | false | 优先候选方向在当前分支中承担的固定载体或注意力梯度角色。 |
| minimum_projection_energy_retention | protocol | none | true | false | false | 固定盲检模板投影到语义安全子空间后必须保留的最小能量比例。 |
| relative_response_residual | metric | none | true | true | false | 选中子空间平均响应相对全部候选平均响应的归一化残差。 |
| maximum_relative_response_residual | protocol | none | true | false | false | 语义条件低响应子空间允许的最大归一化响应残差。 |
| scientific_update_record_count | metric | none | true | false | false | 数据集正式运行实际生成的关键科学算子注入记录数量。 |
| expected_scientific_update_record_count | metric | none | true | false | false | Prompt 数量乘以冻结注入时刻数量得到的预期科学算子记录数。 |
| scientific_operator_failure_count | metric | none | true | false | false | 未通过精确 JVP、残差、正交性、载体能量或 Q/K 单调提升检查的注入记录数。 |
| scientific_operator_gate_ready | governance | none | true | false | false | 全部注入记录是否通过真实关键科学算子门禁。 |
| formal_feature_extractor_ids | provenance | none | true | false | false | 正式质量记录实际出现的特征提取器标识集合。 |
| canonical_formal_feature_extractor_ready | governance | none | true | false | false | 正式质量记录是否全部来自冻结的 torch-fidelity 0.4.0 兼容 Inception 提取器。 |
| branch_risk_records | method | none | true | true | false | 单次注入中三个分支风险场的摘要映射。 |
| risk_value_mean | metric | none | true | true | false | 单个分支空间风险值的均值。 |
| budget_value_mean | metric | none | true | true | false | 单个分支进入硬资格边界前的承载预算均值。 |
| eligible_position_count | metric | none | true | true | false | 单个分支通过风险资格阈值的 latent 空间位置数量。 |
| risk_field_position_count | metric | none | true | true | false | 单个分支风险场覆盖的 latent 空间位置总数。 |
| scientific_autograd_compatibility | environment | none | true | false | false | 为精确 forward AD 与 latent 输入梯度固定的模型注意力实现摘要。 |
| clip_attention_implementation | environment | none | true | false | false | CLIP 视觉编码器使用的注意力实现, 当前正式值为 eager。 |
| vae_attention_processor | environment | none | true | false | false | VAE 可微解码使用的 Diffusers attention processor。 |
| baseline_observations_sha256 | provenance | none | true | false | false | 单 baseline transfer manifest 绑定的 observation 文件 SHA-256。 |
| baseline_command_results_sha256 | provenance | none | true | false | false | 单 baseline command result 文件 SHA-256。 |
| prompt_plan_path | artifact | none | true | false | false | 单 baseline transfer manifest 绑定的完整受治理 Prompt 计划路径。 |
| prompt_plan_sha256 | provenance | none | true | false | false | 当前 baseline 完整 Prompt 计划文件 SHA-256。 |
| adapter_manifest_path | artifact | none | true | false | false | 单 baseline transfer manifest 绑定的 adapter manifest 路径。 |
| adapter_manifest_sha256 | provenance | none | true | false | false | 当前 baseline adapter manifest 文件 SHA-256。 |
| execution_manifest_path | artifact | none | true | false | false | 单 baseline transfer manifest 绑定的 execution manifest 路径。 |
| execution_manifest_sha256 | provenance | none | true | false | false | 当前 baseline execution manifest 文件 SHA-256。 |
| paper_run_name | protocol | none | true | false | false | 产物所属的唯一论文运行层级, 取值为 probe_paper、pilot_paper 或 full_paper。 |
| formal_attack_names | protocol | none | true | false | false | 当前正式运行实际要求完整覆盖的受治理攻击名称集合。 |
| threshold | protocol | none | true | false | false | observation 及其 transfer 记录共享的 calibration clean negative 冻结阈值。 |
| generation_protocol | protocol | none | true | false | false | common-backbone 模型、采样步数、guidance 和图像尺寸的结构化配置。 |
| detection_protocol | protocol | none | true | false | false | 仅图像访问边界、反演步数与目标 FPR 的结构化配置。 |
| transfer_ready | governance | none | true | false | false | 单 baseline transfer 交换面是否通过身份、计数、阈值与攻击集合校验。 |
| fixed_fpr_observation_evidence_path | artifact | none | true | false | false | 正式导入记录绑定的 observation JSON 或 JSONL 路径, 必须同时出现在 evidence_paths 中。 |
| fixed_fpr_observation_evidence_digest | provenance | none | true | false | false | 对正式导入 observation 列表执行规范序列化后得到的 SHA-256 摘要。 |
| required_threshold_fields | protocol | none | true | false | false | 主表 baseline 正式导入 schema 要求完整提供的 fixed-FPR 阈值 provenance 字段集合。 |
| protocol_patch_path | artifact | none | true | false | false | T2SMark 正式协议 Git diff 的受治理路径。 |
| protocol_patch_sha256 | provenance | none | true | false | false | T2SMark 正式协议 Git diff 文件的 SHA-256。 |
| protocol_patch_applied | governance | none | true | false | false | T2SMark 正式协议 Git diff 是否由本次源码准备流程应用。 |
| protocol_patch_ready | governance | none | true | false | false | T2SMark 正式协议 Git diff 是否已应用且通过反向校验。 |
| patched_source_sha256 | provenance | none | true | false | false | T2SMark 正式协议涉及源码文件的相对路径到 SHA-256 映射。 |
| source_worktree_exact | governance | none | true | false | false | 外部源码工作树是否恰好等于登记 commit 加受治理协议 Git diff, 不含额外修改。 |
| source_worktree_digest | provenance | none | true | false | false | 外部源码登记 commit、固定 diff 与受影响文件摘要组成的稳定摘要。 |
| source_revision_ready | governance | none | true | false | false | 外部源码 commit、工作树和正式协议补丁是否共同通过版本门禁。 |
| protocol_binding_name | protocol | none | true | false | false | T2SMark 官方结果复用必须匹配的正式协议绑定名称。 |
| canonical_prompt_digest | provenance | none | true | false | false | 当前论文层级完整受治理 Prompt 集合的规范摘要。 |
| protocol_binding_digest | provenance | none | true | false | false | T2SMark Prompt、预算、攻击、源码和随机种子协议绑定对象的稳定摘要。 |
| protocol_evidence_digest | provenance | none | true | false | false | T2SMark 协议绑定对象与官方结果、设置文件摘要共同形成的证据摘要。 |
| official_results_sha256 | provenance | none | true | false | false | T2SMark 官方 `results.json` 文件的 SHA-256。 |
| official_settings_sha256 | provenance | none | true | false | false | T2SMark 官方运行设置文件的 SHA-256。 |
| official_protocol_binding_ready | governance | none | true | false | false | T2SMark 官方结果是否完整匹配当前正式协议绑定和证据摘要。 |
| official_protocol_binding_digest | provenance | none | true | false | false | 已核验 T2SMark 官方结果所绑定正式协议对象的稳定摘要。 |
| official_protocol_evidence_digest | provenance | none | true | false | false | 已核验 T2SMark 官方结果协议与事实文件共同形成的证据摘要。 |
| official_protocol_binding_path | artifact | none | true | false | false | T2SMark 官方结果旁路保存的正式协议绑定记录路径。 |
| entry_sha256 | provenance | none | true | false | false | 精确结果包白名单中每个归档成员路径到 SHA-256 的映射。 |
| source_to_evaluated_ssim | metric | none | true | true | false | 单条检测记录从 source 图像到实际 evaluated 图像的高斯窗口 SSIM。 |
| source_to_evaluated_psnr | metric | none | true | true | false | 单条检测记录从 source 图像到实际 evaluated 图像的 PSNR。 |
| source_to_evaluated_mse | metric | none | true | true | false | 单条检测记录从 source 图像到实际 evaluated 图像的均方误差。 |
| source_to_evaluated_ssim_mean | metric | none | true | true | false | 同一攻击与角色下 source-to-evaluated SSIM 均值。 |
| source_to_evaluated_psnr_mean | metric | none | true | true | false | 同一攻击与角色下全量 PSNR 均值; 全组图像均完全一致且每条 MSE 为0时允许为正无穷, 不允许选择性忽略样本或混合有限值与正无穷。 |
| attacked_positive_source_to_attacked_ssim_mean | metric | none | true | true | false | 攻击后 watermarked positive 相对其未攻击 source 的 SSIM 均值。 |
| package_family | provenance | none | true | false | false | CPU 论文闭合输入锁中记录的唯一上游结果包职责族。 |
| package_path | artifact | none | true | false | false | CPU 论文闭合选择后冻结的上游 ZIP 显式绝对路径。 |
| package_sha256 | provenance | none | true | false | false | CPU 论文闭合选择后冻结的上游 ZIP 文件 SHA-256。 |
| closure_input_package_count | metric | none | true | false | false | CPU 论文闭合输入锁覆盖的互异结果包 family 数量, 正式值为10。 |
| closure_input_packages | artifact | none | true | false | false | CPU 论文闭合输入锁冻结的 family、路径、摘要、论文层级、FPR、代码版本和生成时间记录集合。 |
| closure_input_lock_digest | provenance | none | true | false | false | 对论文闭合输入锁核心内容执行规范序列化后得到的 SHA-256。 |
| selected_package_paths | artifact | none | false | false | false | CPU 论文闭合 dry-run 报告返回的10个显式上游 ZIP 路径。 |
| closure_input_lock_path | artifact | none | false | false | false | 当前论文运行层级的 CPU 闭合输入锁输出路径。 |
| closure_input_lock_manifest_path | artifact | none | false | false | false | 当前论文运行层级输入锁独立 manifest 的输出路径。 |
| closure_input_lock_written | governance | none | false | false | false | 当前选择调用是否请求并成功写出输入锁及其独立 manifest。 |
| closure_input_selection_ready | governance | none | false | false | false | 10个结果包 family 是否均通过包内身份、白名单、CRC 和 ready 门禁。 |
| closure_input_lock_ready | governance | none | false | false | false | 输入锁 manifest 是否绑定10个精确结果包及当前论文运行协议。 |
| closure_input_lock_present | governance | none | false | false | false | 完整结果包生成前是否在当前论文运行层级目录发现 closure input lock。 |
| closure_input_lock_manifest_ready | governance | none | false | false | false | 完整结果包复核是否确认 input lock manifest 的身份、输出路径、层级、FPR、数量和锁摘要完全一致。 |
| closure_input_lock_digest_ready | governance | none | false | false | false | 完整结果包复算的 closure input lock 规范摘要是否等于锁中声明值。 |
| closure_input_family_ready | governance | none | false | false | false | Closure input lock 是否恰好覆盖受治理的10个互异结果包 family。 |
| closure_input_scope_ready | governance | none | false | false | false | Closure input lock 顶层与逐包记录的论文运行层级和目标 FPR 是否完全一致。 |
| closure_input_explicit_paths_ready | governance | none | false | false | false | 完整结果包收到的显式 package paths 是否与 closure input lock 中的路径集合完全相等。 |
| closure_input_package_digests_ready | governance | none | false | false | false | 完整结果包生成前重新计算的10个 ZIP 文件摘要是否与 closure input lock 完全一致。 |
| closure_input_package_metadata_ready | governance | none | false | false | false | Closure input lock 的每个结果包是否都提供非空代码版本和生成时间。 |
| result_closure_gate_report_path | artifact | none | false | false | false | 完整结果包绑定的当前论文运行层级 result closure gate 报告路径。 |
| result_closure_gate_report_present | governance | none | false | false | false | 完整结果包生成前是否发现当前论文运行层级的 result closure gate 报告。 |
| result_closure_gate_manifest_path | artifact | none | false | false | false | 完整结果包绑定的当前论文运行层级 result closure gate manifest 路径。 |
| result_closure_gate_manifest_ready | governance | none | false | false | false | Result closure gate manifest 身份与报告、自身输出路径是否完全绑定。 |
| payload_entry_paths | artifact | none | false | false | false | 完整结果包中除内部 package metadata 外的当前论文运行层级受治理文件路径集合。 |
| payload_entry_count | metric | none | false | false | false | 完整结果包中除内部 package metadata 外的受治理文件数量。 |
| entry_paths_digest | provenance | none | false | false | false | 完整结果包全部归档成员路径列表的稳定摘要。 |
| archive_entry_digest | provenance | none | false | false | false | 完整结果包 readiness 中全部归档成员路径列表的稳定摘要。 |
| explicit_package_paths | artifact | none | false | false | false | 完整结果包实际复核并物化的显式上游 ZIP 路径集合。 |
| materialization_skipped | governance | none | false | false | false | 调用者是否显式要求复用已物化 outputs 并跳过 ZIP 条目提取。 |
| closure_check_count | metric | none | false | false | false | 当前运行层级结果闭合语义门禁执行的检查总数。 |
| blocked_check_count | metric | none | false | false | false | 结果闭合语义门禁中未通过的检查数量。 |
| blocked_check_ids | governance | none | false | false | false | 结果闭合语义门禁中未通过检查的稳定标识集合。 |
| checks | governance | none | false | false | false | 结果闭合报告保存的逐项语义门禁记录集合。 |
| source_artifact_digests | provenance | none | false | false | false | 结果闭合报告对各类正式输入证据计算的稳定摘要映射。 |
| result_closure_ready | governance | none | false | false | false | 当前运行层级的攻击、阈值、baseline、结果、消融、质量与投稿证据是否全部闭合。 |
| closure_decision | governance | none | false | false | false | 结果闭合语义门禁的最终判定, 仅允许 pass 或 blocked。 |
| expected_prompt_count | metric | none | false | false | false | 当前论文运行层级按冻结协议要求的完整 Prompt 数量。 |
| expected_test_count | metric | none | false | false | false | 当前论文运行层级按冻结划分要求的完整 test Prompt 数量。 |
| check_id | governance | none | true | false | false | 单项结果闭合语义检查的稳定标识。 |
| check_area | governance | none | true | false | false | 单项结果闭合语义检查所属的证据领域。 |
| check_status | governance | none | true | false | false | 单项结果闭合语义检查的 pass 或 blocked 状态。 |
| source_artifacts | artifact | none | true | false | false | 单项结果闭合检查实际消费的受治理证据对象集合。 |
| threshold_audit_digest | provenance | none | false | false | false | 五方法统一 fixed-FPR 阈值审计输入的稳定摘要。 |
| baseline_comparison_digest | provenance | none | false | false | false | 外部 baseline 正式比较报告与 manifest 的稳定摘要。 |
| result_records_digest | provenance | none | false | false | false | 当前运行层级正式结果 records 的稳定摘要。 |
| common_protocol_digest | provenance | none | false | false | false | 当前运行层级共同协议 summary 与 schema 的稳定摘要。 |
| result_analysis_digest | provenance | none | false | false | false | 当前运行层级论文结果分析 summary 的稳定摘要。 |
| formal_ablation_digest | provenance | none | false | false | false | 当前运行层级正式重运行消融 summary 的稳定摘要。 |
| formal_fid_kid_digest | provenance | none | false | false | false | 当前运行层级正式 FID/KID summary 的稳定摘要。 |
| paper_evidence_audit_digest | provenance | none | false | false | false | 论文 claim 证据审计 builder 与 blocker 报告的稳定摘要。 |
| submission_readiness_digest | provenance | none | false | false | false | 投稿就绪报告的稳定摘要。 |
| entry_review_digest | provenance | none | false | false | false | 论文证据闭合入口审查报告的稳定摘要。 |
| generation_latent_trace_required | protocol | none | true | false | false | 正式检测是否依赖生成 latent 轨迹; 仅图像盲检正式路径必须为 false。 |
| expected_ablation_ids | protocol | none | false | false | false | 正式机制消融唯一允许的8项受治理标识序列。 |
| actual_ablation_ids | protocol | none | false | false | false | 当前正式消融实际运行并汇总的机制标识序列。 |
| ablation_spec_digest | provenance | none | false | false | false | 对实际消融配置完整字段序列计算的稳定摘要。 |
| ablation_exact_set_ready | governance | none | false | false | false | 实际消融标识、顺序和配置摘要是否与唯一8项正式规范完全一致。 |
| registry_prompt_count | metric | none | false | false | false | 数据集质量图像 registry 中实际出现的 Prompt 记录数量。 |
| duplicate_registry_prompt_id_count | metric | none | false | false | false | 数据集质量 registry 中重复 Prompt 标识的数量。 |
| missing_registry_prompt_id_count | metric | none | false | false | false | 当前受治理 Prompt 集中未被质量 registry 覆盖的标识数量。 |
| unexpected_registry_prompt_id_count | metric | none | false | false | false | 质量 registry 中不属于当前受治理 Prompt 集的标识数量。 |
| canonical_prompt_id_digest | provenance | none | false | false | false | 当前论文运行精确 Prompt 标识集合的稳定摘要。 |
| registry_prompt_id_digest | provenance | none | false | false | false | 数据集质量 registry 实际 Prompt 标识集合的稳定摘要。 |
| prompt_registry_exact_set_ready | governance | none | false | false | false | 质量 registry 是否无缺失、无额外、无重复地一对一覆盖当前 Prompt 集。 |
| expected_prompt_id_digest | provenance | none | false | false | false | 结果闭合门禁从当前受治理 Prompt 文件独立重算的标识集合摘要。 |
| expected_prompt_split_digest | provenance | none | false | false | false | 结果闭合门禁从当前 Prompt 文件和规范划分规则独立重算的完整 split 摘要。 |
| expected_calibration_prompt_id_digest | provenance | none | false | false | false | 结果闭合门禁从当前 Prompt 文件独立重算的 calibration Prompt 标识集合摘要。 |
| expected_test_prompt_id_digest | provenance | none | false | false | false | 结果闭合门禁从当前受治理 Prompt 文件独立重算的规范 test split 标识集合摘要。 |
| calibration_prompt_id_digest | provenance | none | true | false | false | 共同协议从当前受治理 Prompt 文件重算的 calibration Prompt 标识集合摘要。 |
| attack_detection_records | artifact | none | false | false | false | 结果闭合门禁实际读取并逐条复验身份与摘要的正式攻击检测记录集合。 |
| attacked_image_registry | artifact | none | false | false | false | 结果闭合门禁实际读取并与攻击检测记录精确投影比对的 attacked image registry。 |
| result_record_validation_report | artifact | none | false | false | false | 正式 result records 的 schema validation 报告, 最终门禁会独立重算并精确比对。 |
| result_record_template_coverage | artifact | none | false | false | false | 正式 method x attack 六元身份模板覆盖表, 最终门禁会按规范注册表重建。 |
| require_existing_evidence | protocol | none | false | false | false | result record 构造时是否要求每个 evidence path 已存在, 并进入 manifest 配置摘要。 |
| test_prompt_id_digest | provenance | none | true | false | false | 共同协议从受治理 Prompt 文件当前 test split 独立重算的标识集合摘要。 |
| dataset_quality_feature_records_sha256 | provenance | none | false | false | false | 结果闭合门禁对规范正式特征 JSONL 文件独立计算的字节摘要。 |
| common_code_version | provenance | none | true | false | false | CPU 结果闭合输入锁中10个上游结果包共享的精确40位小写 clean Git 提交 SHA。 |
| closure_input_common_code_version_ready | governance | none | false | false | false | 完整结果打包前逐包代码版本是否均为同一规范化 clean Git 提交。 |
| closure_source_file_sha256 | provenance | none | false | false | false | 结果闭合门禁全部实际读取文件的规范路径到文件字节 SHA-256 映射。 |
| closure_source_file_digest | provenance | none | false | false | false | 对结果闭合门禁输入文件字节摘要映射执行规范序列化后得到的 SHA-256。 |
| result_closure_gate_report_digest | provenance | none | false | false | false | 完整结果打包前从 gate manifest 读取并对 gate 报告重算的稳定摘要。 |
| result_closure_gate_report_digest_ready | governance | none | false | false | false | Gate 报告当前内容的稳定摘要是否仍等于 gate manifest 绑定值。 |
| result_closure_gate_source_file_digest | provenance | none | false | false | false | 完整结果打包前复核的 gate 全部输入文件字节摘要映射稳定摘要。 |
| result_closure_gate_source_file_digest_ready | governance | none | false | false | false | Gate 报告声明的输入文件摘要映射 digest 是否可由映射本身重建。 |
| result_closure_gate_source_files_ready | governance | none | false | false | false | 完整结果打包前重新读取的全部 gate 输入文件 SHA-256 是否与门禁时完全一致。 |
| result_closure_gate_manifest_config_ready | governance | none | false | false | false | Gate manifest 的 config digest 是否绑定运行层级、FPR、样本规模、报告和输入文件摘要。 |
| result_closure_gate_code_version_ready | governance | none | false | false | false | Gate manifest 代码版本是否为输入锁绑定的同一 clean Git 提交。 |
| current_repository_code_version | provenance | none | false | false | false | 完整结果打包当下仓库的精确40位小写 clean Git 提交 SHA。 |
| current_repository_code_version_ready | governance | none | false | false | false | 完整结果打包当下仓库是否仍为输入锁绑定的同一 clean Git 提交且工作区无修改。 |
| observation_id | protocol | none | true | false | false | 分数分布表中一条仅图像检测 observation 的稳定标识。 |
| sample_scope | protocol | none | true | true | false | 连续检测统计所属的 test overall 或单一攻击条件范围。 |
| binary_label | metric | none | true | true | false | 由正式 sample role 映射得到的二分类真实标签, 阳性为1且阴性为0。 |
| score | metric | none | true | true | false | 与内容主判及冻结几何救回布尔决策等价的连续检测分数。 |
| curve_point_index | metric | none | true | true | false | 同一 sample scope 内按阈值从高到低排列的确定性曲线点序号。 |
| threshold_kind | protocol | none | true | true | false | 曲线阈值是正无穷端点、唯一观测分数还是负无穷端点。 |
| tp | metric | none | true | true | false | 当前阈值下真实阳性且预测阳性的记录数。 |
| fp | metric | none | true | true | false | 当前阈值下真实阴性但预测阳性的记录数。 |
| tn | metric | none | true | true | false | 当前阈值下真实阴性且预测阴性的记录数。 |
| fn | metric | none | true | true | false | 当前阈值下真实阳性但预测阴性的记录数。 |
| tpr | metric | none | true | true | false | 当前阈值下由 TP 和 FN 计算的真正率。 |
| fpr | metric | none | true | true | false | 当前阈值下由 FP 和 TN 计算的假正率。 |
| fnr | metric | none | true | true | false | 当前阈值下由 FN 和 TP 计算的假负率。 |
| sample_count | metric | none | true | true | false | 当前曲线 scope 同时参与统计的正负 observation 总数。 |
| score_distribution_row_count | metric | none | true | false | false | 当前仅图像运行写出的记录级连续分数分布行数。 |
| roc_curve_point_count | metric | none | true | false | false | 当前仅图像运行由完整唯一分数阈值 sweep 写出的 ROC 点数。 |
| det_curve_point_count | metric | none | true | false | false | 当前仅图像运行由完整唯一分数阈值 sweep 写出的 DET 点数。 |
| detection_curve_data_ready | governance | none | true | false | false | 分数分布、ROC 与 DET 三张实际数据表是否均非空生成。 |
| artifact_data_validation | governance | none | true | false | false | 论文证据 bundle 中实际表图数据验证报告的结构化内容。 |
| artifact_data_validation_ready | governance | none | true | false | false | 论文审计要求的11类实际数据文件及 ready 一致性是否全部通过。 |
| raw_image_only_detection_records_ready | governance | none | true | false | false | 仅图像盲检原始 JSONL 是否存在、非空且可作为分数分布、ROC 与 DET 的共同重建来源。 |
| frozen_evidence_protocol_ready | governance | none | true | false | false | 冻结 evidence protocol 文件是否存在且通过正式内容校验。 |
| test_detection_metrics_ready | governance | none | true | false | false | test detection metrics 是否可由原始记录重建并通过内容校验。 |
| score_distribution_table_ready | governance | none | true | false | false | 记录级连续分数分布表是否与原始仅图像检测记录完全一致。 |
| roc_curve_points_ready | governance | none | true | false | false | ROC 点表是否可由原始连续分数和标签精确重建。 |
| det_curve_points_ready | governance | none | true | false | false | DET 点表是否可由原始连续分数和标签精确重建。 |
| attack_family_metrics_ready | governance | none | true | false | false | 攻击族指标表是否存在且通过正式数值与身份校验。 |
| baseline_comparison_table_ready | governance | none | true | false | false | 主表 baseline 对比表是否存在且通过正式内容校验。 |
| mechanism_ablation_metrics_ready | governance | none | true | false | false | 机制消融指标表是否存在且通过正式内容校验。 |
| mechanism_pairwise_delta_ready | governance | none | true | false | false | 机制成对差值表是否存在且通过正式内容校验。 |
| dataset_quality_metrics_ready | governance | none | true | false | false | 正式 FID/KID 数据集质量表是否存在且通过内容校验。 |
| raw_image_only_detection_records_sha256 | provenance | none | true | false | false | 论文表图重建实际读取的仅图像盲检原始 JSONL 字节 SHA-256。 |
| artifact_data_check_count | metric | none | true | false | false | 实际论文表图数据验证执行的文件与跨摘要一致性检查总数。 |
| blocked_artifact_data_count | metric | none | true | false | false | 未通过实际内容验证的论文表图数据检查数量。 |
| blocked_artifact_data_ids | governance | none | true | false | false | 未通过实际内容验证的表图数据检查稳定标识集合。 |
| data_ready | governance | none | true | false | false | 单项实际数据检查是否通过文件、schema、数值和统计约束。 |
| row_count | metric | none | true | false | false | 单项实际数据检查读取并验证的记录行数。 |
| issues | governance | none | true | false | false | 单项实际数据检查发现的具体阻断原因集合。 |
| ready_flag_consistency_ready | governance | none | true | false | false | 上游关键 ready 标记是否均有对应实际数据检查支撑。 |
| evidence_source_file_sha256 | provenance | none | true | false | false | 论文证据审计实际读取的11类源文件路径到字节级 SHA-256 的映射。 |
| artifact_data_validation_digest | provenance | none | true | false | false | 对实际表图数据验证报告执行规范序列化得到的稳定摘要。 |
| official_reference_fidelity_record_id | protocol | none | true | false | false | 单个 official-reference 方法忠实度证据记录的稳定标识。 |
| official_reference_fidelity_record_digest | provenance | none | true | false | false | 单个 official-reference 方法忠实度证据记录核心内容的稳定摘要。 |
| reference_record_id | protocol | none | true | false | false | 单次 official-reference 复现记录的稳定标识。 |
| reference_record_digest | provenance | none | true | false | false | 单次 official-reference 复现记录全部受治理字段的稳定摘要。 |
| reference_protocol_name | protocol | none | true | false | false | official-reference 记录采用的固定补充证据协议名称。 |
| official_environment_profile | protocol | none | true | false | false | 官方原始实现实际运行所用的隔离 Python 与依赖环境身份。 |
| official_scientific_config | protocol | none | true | false | false | 官方参考科学单元、运行摘要与受治理记录共同绑定的规范科学配置。 |
| official_scientific_config_digest | provenance | none | true | false | false | 方法身份与 official_scientific_config 联合计算的稳定 SHA-256 摘要。 |
| scientific_config | protocol | none | true | false | false | 单个外部科学路线用于构造原子单元身份的规范配置。 |
| unit_identity | provenance | none | true | false | false | 原子科学单元绑定方法、配置、索引范围、源码和依赖的完整稳定身份。 |
| unit_identity_digest | provenance | none | true | false | false | unit_identity 排除路径搬迁因素后的稳定 SHA-256 摘要。 |
| unit_complete | governance | none | true | false | false | 原子科学单元是否已完整写出并可进入复验。 |
| scientific_unit_record_digest | provenance | none | true | false | false | 单个原子科学完成记录排除自指字段后的稳定 SHA-256 摘要。 |
| source_payload | provenance | none | true | false | false | 实际官方科学子进程发布的逐 Prompt 观测与来源对象。 |
| source_payload_digest | provenance | none | true | false | false | source_payload 排除自指字段后的稳定 SHA-256 摘要。 |
| observations | protocol | none | true | false | false | 一个官方参考科学单元实际产生的逐 Prompt 原始观测列表。 |
| observations_digest | provenance | none | true | false | false | observations 保持正式顺序计算的稳定 SHA-256 摘要。 |
| official_command_execution_evidence | provenance | none | true | false | false | 官方命令 argv、工作目录、解释器及 CUDA 检查的执行事实。 |
| official_command_canonical_identity | provenance | none | true | false | false | 从原始 argv 复算且排除 workspace 绝对路径的官方命令规范身份。 |
| official_command_canonical_identity_digest | provenance | none | true | false | false | 规范命令身份绑定科学参数、模型内容与 OpenCLIP 内容的稳定 SHA-256 摘要。 |
| official_command_scientific_parameters | protocol | none | true | false | false | 从实际 argv 提取并与规范科学配置逐项相等的 workspace-independent 科学参数。 |
| official_model_binding | provenance | none | true | false | false | 规范命令身份绑定的官方模型仓库、revision 与快照内容摘要。 |
| openclip_binding | provenance | none | true | false | false | 规范命令身份绑定的 OpenCLIP 模型名、checkpoint SHA 与快照内容摘要。 |
| workspace_independent | governance | none | true | false | false | 规范身份是否明确排除仅用于当前执行会话定位的绝对路径。 |
| entrypoint_filename | protocol | none | true | false | false | 规范命令身份绑定的官方方法入口文件名。 |
| canonical_identity | provenance | none | true | false | false | CPU 打包复验随原始批次 argv 读取的规范命令身份对象。 |
| cuda_inspection_report | provenance | none | true | false | false | 官方命令执行前对 PyTorch CUDA build 与实际设备的检查报告。 |
| official_unit_coverage_ready | governance | none | true | false | false | 官方参考原子批次是否无缺失、无重叠并完整覆盖预注册索引。 |
| official_unit_batch_size | protocol | none | true | false | false | 官方参考一次原子科学批次包含的 Prompt 数量。 |
| official_unit_expected_count | metric | none | true | false | false | 当前论文规模预注册的官方参考原子批次数量。 |
| official_unit_completed_count | metric | none | true | false | false | 聚合时通过完整复验的官方参考原子批次数量。 |
| official_unit_resumed_count | metric | none | true | false | false | 本次运行开始前已存在且通过复验的官方参考原子批次数量。 |
| official_unit_executed_count | metric | none | true | false | false | 本次运行实际补算的官方参考原子批次数量。 |
| official_unit_executed_ids | provenance | none | true | false | false | 本次运行实际补算的官方参考科学单元标识列表。 |
| official_unit_ids | provenance | none | true | false | false | 当前正式计划预注册的全部官方参考科学单元标识列表。 |
| official_unit_records_digest | provenance | none | true | false | false | 按索引范围排序的全部官方参考完成记录稳定摘要。 |
| official_unit_observations_digest | provenance | none | true | false | false | 全部官方参考逐 Prompt 观测保持全局顺序计算的稳定摘要。 |
| official_unit_observations | provenance | none | true | false | false | 从完整官方参考原子批次确定性拼接的逐 Prompt 原始观测。 |
| official_unit_command_identities | provenance | none | true | false | false | 按预注册索引顺序排列的全部 workspace-independent 规范命令身份。 |
| official_unit_command_identities_digest | provenance | none | true | false | false | 全部规范命令身份保持预注册顺序计算的稳定 SHA-256 摘要。 |
| official_unit_commands | provenance | none | true | false | false | 正式聚合实际引用的每批官方命令执行证据列表。 |
| stable_unit_identity | provenance | none | true | false | false | 归档复验从全部批次提取的单一稳定代码、源码、依赖与科学配置身份。 |
| prompt_seed_random | random | _random | true | false | false | 原子单元中实际控制单个 Prompt 随机流的整数 seed。 |
| no_w_metric | metric | none | true | false | false | Tree-Ring 官方参考单个 clean 样本的检测距离。 |
| w_metric | metric | none | true | false | false | Tree-Ring 官方参考单个 watermarked 样本的检测距离。 |
| w_no_sim | metric | none | true | false | false | Tree-Ring 官方参考单个 clean 样本与 Prompt 的 CLIP 相似度。 |
| w_sim | metric | none | true | false | false | Tree-Ring 官方参考单个 watermarked 样本与 Prompt 的 CLIP 相似度。 |
| detection_hit | metric | none | true | false | false | Gaussian Shading 官方参考单个样本是否达到检测阈值。 |
| traceability_hit | metric | none | true | false | false | Gaussian Shading 官方参考单个样本是否达到追踪阈值。 |
| random_material_digest_random | random | _random | true | false | false | Gaussian Shading 官方参考单元对 key、nonce 与 watermark 原文计算的不可逆摘要。 |
| no_w_metrics_none | metric | none | true | false | false | Shallow Diffuse 官方参考单个 clean 样本的无攻击检测指标。 |
| avg_metrics_none | metric | none | true | false | false | Shallow Diffuse 官方参考单个 watermarked 样本的无攻击检测指标。 |
| clip_scores_no_w | metric | none | true | false | false | Shallow Diffuse 官方参考单个 clean 样本的 CLIP 分数。 |
| clip_scores_avg | metric | none | true | false | false | Shallow Diffuse 官方参考单个 watermarked 样本的 CLIP 分数。 |
| source_unit_output_ready | governance | none | true | false | false | 动态补丁后的官方源码是否具备逐 Prompt 观测和原子 payload 发布能力。 |
| method_faithful_source_identity | provenance | none | true | false | false | Common-backbone adapter 绑定外部方法登记 commit 与实际适配实现摘要的源码身份。 |
| method_faithful_source_identity_digest | provenance | none | true | false | false | method_faithful_source_identity 排除自指字段后的稳定 SHA-256 摘要。 |
| stable_scientific_execution_identity | provenance | none | true | false | false | 跨 Colab 会话必须一致的代码锁、依赖锁、Python、PyTorch 与 CUDA build 身份。 |
| stable_scientific_execution_identity_digest | provenance | none | true | false | false | stable_scientific_execution_identity 的稳定 SHA-256 摘要。 |
| method_faithful_scientific_unit_count | metric | none | true | false | false | 当前 common-backbone 运行完整复验的 source pair 与攻击单元总数。 |
| method_faithful_scientific_unit_record_paths | artifact | none | true | false | false | 当前 common-backbone 运行全部原子完成记录的仓库相对路径列表。 |
| method_faithful_scientific_unit_records_digest | provenance | none | true | false | false | 按正式计划顺序排列的全部 common-backbone 原子记录稳定摘要。 |
| method_faithful_scientific_unit_resume_ready | governance | none | true | false | false | Common-backbone 完成单元集合是否可恢复且满足 Prompt 与攻击 exact set。 |
| method_faithful_source_prompt_unit_count | metric | none | true | false | false | Common-backbone 完整 source pair 单元覆盖的 Prompt 数量。 |
| method_faithful_formal_attack_unit_count | metric | none | true | false | false | Common-backbone 完整 test Prompt 攻击角色单元数量。 |
| method_faithful_run_identity_path | artifact | none | true | false | false | Common-backbone 原子运行稳定身份记录的仓库相对路径。 |
| method_faithful_run_identity_sha256 | provenance | none | true | false | false | method_faithful_run_identity_path 指向文件的字节 SHA-256。 |
| method_faithful_run_identity_digest | provenance | none | true | false | false | Common-backbone 原子运行身份对象排除自指字段后的稳定摘要。 |
| expected_formal_attack_unit_count | metric | none | true | false | false | 当前 test Prompt、攻击名称与阴阳角色笛卡尔积的预期单元数。 |
| source_registry_item_digest | provenance | none | true | false | false | 单个外部方法源码登记项的稳定 SHA-256 摘要。 |
| adapter_implementation_sha256 | provenance | none | true | false | false | Common-backbone adapter 与共享单元实现文件到字节 SHA-256 的映射。 |
| method_faithful_unit_digest | provenance | none | true | false | false | 单个 common-backbone 完成单元排除自指字段后的稳定摘要。 |
| unit_artifacts | artifact | none | true | false | false | 单个科学完成单元绑定的事实文件路径、大小与 SHA-256 列表。 |
| unit_data | artifact | none | true | false | false | 单个科学完成单元用于重建 observation 与派生 manifest 的规范数据。 |
| run_config | protocol | none | true | false | false | 原子科学运行排除传输路径后的完整方法与公平协议配置。 |
| run_config_digest | provenance | none | true | false | false | run_config 与稳定源码、代码锁和依赖身份联合计算的摘要。 |
| unit_contract_digest | provenance | none | true | false | false | T2SMark 完整原子单元契约排除自指字段后的稳定摘要。 |
| formal_unit_record_digest | provenance | none | true | false | false | 单个 T2SMark Prompt 完成记录排除自指字段后的稳定摘要。 |
| formal_unit_record_count | metric | none | true | false | false | T2SMark 完整聚合实际引用的 Prompt 单元数量。 |
| formal_unit_record_digests | provenance | none | true | false | false | 按全局 Prompt 索引排列的 T2SMark 单元摘要列表。 |
| formal_unit_records_digest | provenance | none | true | false | false | 按全局 Prompt 索引排列的全部 T2SMark 单元记录稳定摘要。 |
| formal_unit_aggregate_digest | provenance | none | true | false | false | T2SMark 单元覆盖、来源聚合与记录摘要对象的稳定摘要。 |
| formal_unit_set_complete | governance | none | true | false | false | T2SMark Prompt 单元是否完整覆盖当前正式计划。 |
| t2smark_formal_unit_set_ready | governance | none | true | false | false | T2SMark 运行摘要中的完整单元集合门禁。 |
| t2smark_formal_unit_record_count | metric | none | true | false | false | T2SMark 运行摘要绑定的 Prompt 完成单元数量。 |
| t2smark_formal_unit_records_digest | provenance | none | true | false | false | T2SMark 运行摘要绑定的全部 Prompt 单元记录摘要。 |
| t2smark_formal_unit_aggregate_digest | provenance | none | true | false | false | T2SMark 运行摘要绑定的单元来源聚合摘要。 |
| t2smark_formal_attack_expected_test_count | metric | none | true | false | false | T2SMark 正式攻击结果应覆盖的 test Prompt 数量。 |
| complete_result_set_ready | governance | none | true | false | false | Adapter 是否已读取连续无缺失的完整 Prompt 结果集合。 |
| calibration_result_count | metric | none | true | false | false | Adapter 用于 fixed-FPR 冻结的 calibration Prompt 结果数量。 |
| calibration_result_indices | protocol | none | true | false | false | Adapter fixed-FPR 校准引用的全局 Prompt 索引列表。 |
| calibration_result_digest | provenance | none | true | false | false | 按 calibration_result_indices 排列的原始结果稳定摘要。 |
| test_result_count | metric | none | true | false | false | Adapter 进入正式攻击聚合的 test Prompt 结果数量。 |
| artifact_sha256 | provenance | none | true | false | false | 单个科学单元内事实文件角色到字节 SHA-256 的映射。 |
| formal_reproduction_config | protocol | none | true | false | false | T2SMark 排除 Drive、checkout 与控制开关后的正式科学复现配置。 |
| paper_run_identity | protocol | none | true | false | false | T2SMark 单元契约绑定的论文层级、规模和 fixed-FPR 身份。 |
| prompt_rows | protocol | none | true | false | false | T2SMark 单元契约按全局索引保存的完整规范 Prompt 行列表。 |
| prompt_plan_digest | provenance | none | true | false | false | 完整 Prompt 行列表按正式顺序计算的稳定摘要。 |
| prompt_identity | protocol | none | true | false | false | 单个科学完成单元绑定的 Prompt id、索引、split、文本和摘要。 |
| source_identity | provenance | none | true | false | false | T2SMark 单元契约绑定的官方 commit、固定补丁与精确工作树身份。 |
| formal_unit_aggregate | provenance | none | true | false | false | T2SMark 完整 Prompt 单元的记录摘要与跨会话来源聚合对象。 |
| clean_base_latent_digest_random | random | _random | true | false | false | 对严格成对图像共用的水印前基础 Gaussian latent 计算的不可逆摘要。 |
| t2smark_secret_material_digest_random | random | _random | true | false | false | 对 T2SMark master key、Prompt key 与 message 计算的不可逆摘要。 |
| fixed_secret_material_digest_random | random | _random | true | false | false | T2SMark settings 对固定 master key 与 message 计算的不可逆摘要。 |
| watermark_seed_random | random | _random | true | false | false | 外部 baseline 原子单元实际用于构造水印随机材料的整数 seed。 |
| watermark_carrier_digest_random | random | _random | true | false | false | Tree-Ring 或 Shallow Diffuse 全局固定载体原文的不可逆摘要。 |
| gaussian_chacha_secret_material_digest_random | random | _random | true | false | false | Gaussian Shading 对 ChaCha20 key、nonce 与 watermark 原文计算的不可逆摘要。 |
| gaussian_chacha_message_digest_random | random | _random | true | false | false | Gaussian Shading 对 ChaCha20 加密后逐坐标符号 message 计算的不可逆摘要。 |
| strict_pair_shared_magnitude | method | none | true | false | false | Gaussian Shading clean 与 watermarked latent 是否逐坐标共享同一绝对幅值。 |
| message_encryption | method | none | true | false | false | Gaussian Shading 主路线使用的消息加密算子与 key / nonce 规格。 |
| acc_key | metric | none | true | false | false | T2SMark 单个 Prompt 的 key bit 恢复准确率。 |
| acc_msg | metric | none | true | false | false | T2SMark 单个 Prompt 的 message bit 恢复准确率。 |
| annotations | protocol | none | true | false | false | T2SMark 官方数据集输入中的规范 Prompt annotation 列表。 |
| artifact_root | artifact | none | false | false | false | Adapter 内部事实图像与原子记录的受限 outputs 根路径。 |
| attack_condition | protocol | none | true | false | false | Observation 对应的规范攻击条件名称。 |
| attack_execution_split | protocol | none | true | false | false | 正式攻击允许执行的唯一 Prompt split。 |
| attack_families | protocol | none | true | false | false | 当前外部 baseline 运行按顺序冻结的正式攻击名称列表。 |
| attacked_image_id | protocol | none | true | false | false | 单个攻击后图像的稳定标识。 |
| attacked_image_manifest_path | artifact | none | true | false | false | Adapter 攻击图像事实清单的仓库相对路径。 |
| attacked_record | artifact | none | true | false | false | 单个攻击单元用于重建 observation 与图像清单的规范事实对象。 |
| dependency_environment_ready | governance | none | true | false | false | 当前隔离科学依赖环境是否通过完整锁门禁。 |
| dependency_profile_digest | provenance | none | true | false | false | 科学依赖 profile 规范内容的稳定 SHA-256 摘要。 |
| dependency_profile_id | provenance | none | true | false | false | 当前科学执行使用的受治理依赖 profile 标识。 |
| explicit_threshold | protocol | none | true | false | false | 外部 adapter 是否由调用方显式提供检测阈值; 正式默认值为空。 |
| formal_attacks | protocol | none | true | false | false | T2SMark 单个 Prompt 按正式攻击名称组织的正负攻击结果映射。 |
| formal_unit_contract | protocol | none | true | false | false | T2SMark runner 返回并持久化的完整原子单元契约。 |
| gen_seed | protocol | none | true | false | false | 官方参考脚本用于生成逐 Prompt seed 的固定基值。 |
| generated_image_digest | provenance | none | true | false | false | 生成图像文件的字节 SHA-256。 |
| generated_image_path | artifact | none | true | false | false | 生成图像文件的仓库相对路径。 |
| image_id | protocol | none | true | false | false | 外部 baseline 图像样本的稳定标识。 |
| image_pair | artifact | none | true | false | false | 单个 source pair 单元的 clean / watermarked 图像事实对象。 |
| image_paths | artifact | none | false | false | false | 测试或归档构造中使用的事实图像路径集合。 |
| injection_mode | method | none | true | false | false | Shallow Diffuse 在 `edit_timestep` 分支位置使用的 patch 注入语义。 |
| input_access_mode | protocol | none | true | false | false | 检测器允许读取的输入访问模式; 正式外部 baseline 为 image_only。 |
| key | runtime | none | false | false | false | Tree-Ring 当前进程内检测所需的载体张量, 不得持久化。 |
| mask | runtime | none | false | false | false | Tree-Ring 或 Shallow Diffuse 当前进程内使用的载体区域 mask。 |
| metric_summary | metric | none | true | false | false | 测试或官方参考重建得到的方法指标汇总对象。 |
| metric_validation | governance | none | true | false | false | 官方参考逐 Prompt 指标重建是否通过完整性校验的报告。 |
| norm1_no_w | metric | none | true | false | false | T2SMark 对 clean 图像恢复噪声的 L1 检测分数。 |
| norm1_w | metric | none | true | false | false | T2SMark 对 watermarked 图像恢复噪声的 L1 检测分数。 |
| num_inference_steps | protocol | none | true | false | false | 图像生成使用的去噪采样步数。 |
| num_inversion_steps | protocol | none | true | false | false | 仅图像检测恢复初始 noise 使用的反演步数。 |
| observation_without_threshold | artifact | none | true | false | false | 单个攻击单元在 fixed-FPR 冻结前保存的连续分数 observation。 |
| observations_without_threshold | artifact | none | true | false | false | 单个 source pair 单元在 fixed-FPR 冻结前保存的正负连续分数列表。 |
| official_clip_scores_path | artifact | none | true | false | false | Shallow Diffuse 官方 CLIP 原始分数文件路径。 |
| official_command_execution_evidence_ready | governance | none | true | false | false | 官方命令是否成功且解释器、工作目录与 CUDA 证据均已绑定。 |
| official_metric_summary | metric | none | true | false | false | 从全部官方参考逐 Prompt 原始观测复算的最终指标对象。 |
| official_metric_text_path | artifact | none | true | false | false | Gaussian Shading 官方文本指标文件路径。 |
| official_model_id | protocol | none | true | false | false | 官方参考运行实际使用的冻结模型仓库标识。 |
| official_overall_scores_path | artifact | none | true | false | false | Shallow Diffuse 官方总体检测分数文件路径。 |
| official_run_name | protocol | none | true | false | false | 官方参考源码使用的运行目录语义名称。 |
| official_timestep_dir | artifact | none | true | false | false | Shallow Diffuse 官方 shallow timestep 输出目录。 |
| official_unit_contract | artifact | none | false | false | false | T2SMark outputs 路径映射中的原子单元契约文件。 |
| official_unit_dir | artifact | none | false | false | false | T2SMark outputs 路径映射中的逐 Prompt 单元目录。 |
| output_root | artifact | none | false | false | false | 测试或 adapter 路径构造使用的 outputs 根目录。 |
| pair_quality | metric | none | true | false | false | 单个 Prompt 严格 clean / watermarked 图像配对的事实与摘要对象。 |
| pair_quality_protocol | protocol | none | true | false | false | 图像质量记录采用的严格配对协议名称。 |
| patch | runtime | none | false | false | false | Shallow Diffuse 当前进程内注入与检测使用的固定 patch 张量。 |
| primary_baseline_id | protocol | none | true | false | false | 单条 common-backbone 运行唯一绑定的主 baseline 标识。 |
| protocol_profile | protocol | none | true | false | false | 当前论文运行层级使用的 fixed-FPR 协议 profile 名称。 |
| record_ready | governance | none | true | false | false | 官方参考受治理 record 是否已由完整单元和指标重建。 |
| robustness | metric | none | true | false | false | T2SMark 单个 Prompt 的连续检测与 bit 恢复指标对象。 |
| row | runtime | none | false | false | false | Adapter 当前进程内与图像状态关联的 Prompt 行对象。 |
| row_index | runtime | none | false | false | false | Adapter 当前进程内 Prompt 行的一基索引。 |
| run_dir | artifact | none | false | false | false | 测试或 runner 路径映射中的单方法独占运行目录。 |
| run_name | protocol | none | true | false | false | 正式 runner 输出目录与科学配置使用的运行名称。 |
| scientific_unit_dir | artifact | none | false | false | false | 官方参考 runner 保存预注册原子批次记录的目录。 |
| scientific_unit_workspace_root | artifact | none | false | false | false | 官方参考缺失批次执行时使用且聚合前必须删除的工作区根目录。 |
| source_random_identity_random | random | _random | false | false | false | Adapter 当前进程内关联 source 单元与攻击单元的随机来源对象。 |
| split_observations | artifact | none | false | false | false | 测试路径映射中的单 baseline 跨包 observation 文件。 |
| start_index | protocol | none | true | false | false | 官方参考当前正式计划的全局 Prompt 起始索引。 |
| stderr_path | artifact | none | true | false | false | 官方参考单元命令标准错误日志文件路径。 |
| stdout_path | artifact | none | true | false | false | 官方参考单元命令标准输出日志文件路径。 |
| strict_pair_quality_ready | governance | none | true | false | false | 当前 clean / watermarked 图像集合是否全部满足严格配对事实门禁。 |
| t2smark_results_path | artifact | none | true | false | false | T2SMark adapter 读取的官方结果仓库相对路径。 |
| transfer_manifest | artifact | none | false | false | false | 测试路径映射中的单 baseline 跨包传输 manifest。 |
| unit_evidence | provenance | none | false | false | false | Runner 内部从 transfer manifest 提取的原子单元证据对象。 |
| unit_records | artifact | none | false | false | false | 测试夹具或聚合函数输入的原子完成记录集合。 |
| w_channel | method | none | true | false | false | Tree-Ring 或 Shallow Diffuse 载体写入的 latent 通道选择。 |
| w_pattern | method | none | true | false | false | Tree-Ring 或 Shallow Diffuse 固定载体的 pattern 名称。 |
| w_radius | method | none | true | false | false | Tree-Ring 或 Shallow Diffuse mask 的外半径。 |
| w_seed | protocol | none | true | false | false | Tree-Ring 官方源码构造全局载体使用的固定 seed。 |
| watermark | runtime | none | false | false | false | Gaussian Shading 当前进程内检测使用的 watermark bit 张量。 |
| watermark_seed | protocol | none | true | false | false | Common-backbone adapter 构造水印载体随机流使用的固定 seed。 |
| with_tracking | protocol | none | true | false | false | Tree-Ring 官方源码是否启用实验追踪参数。 |
| official_source_ready | governance | none | true | false | false | 官方源码入口、精确 Git 身份、确定性补丁与数据来源是否共同通过。 |
| official_environment_report_ready | governance | none | true | false | false | official-reference 是否写出并绑定当前运行环境报告。 |
| official_execution_ready | governance | none | true | false | false | 当前 official-reference 命令是否被请求并以返回码0完成。 |
| official_command_requested | protocol | none | true | false | false | 当前运行是否真实请求 official-reference 命令。 |
| official_command_return_code | runtime | none | true | false | false | 当前 official-reference 子进程的实际返回码。 |
| official_command_succeeded | governance | none | true | false | false | 当前 official-reference 命令是否真实执行且返回码为0。 |
| required_metrics_ready | governance | none | true | false | false | 当前命令是否显式产生全部必需科学指标且数值域合法。 |
| official_result_summary_ready | governance | none | true | false | false | 当前命令的规范科学指标摘要是否完整并通过校验。 |
| governed_import_ready | governance | none | true | false | false | 当前 official-reference 记录是否通过补充证据 schema 校验。 |
| official_model_repository_id | provenance | none | true | false | false | official-reference 记录绑定的登记扩散模型仓库标识。 |
| auc | metric | none | true | false | false | official-reference 检测分数的 ROC 曲线下面积。 |
| accuracy | metric | none | true | false | false | official-reference 在其冻结判别规则下报告的分类准确率。 |
| true_positive_rate_at_one_percent_fpr | metric | none | true | false | false | official-reference 在 FPR=0.01 工作点报告的真正率。 |
| clip_score_mean | metric | none | true | false | false | Tree-Ring 或 Shallow Diffuse clean 图像与 Prompt 的平均 OpenCLIP 余弦相似度。 |
| watermarked_clip_score_mean | metric | none | true | false | false | Tree-Ring 或 Shallow Diffuse 水印图像与 Prompt 的平均 OpenCLIP 余弦相似度。 |
| detection_true_positive_rate | metric | none | true | false | false | Gaussian Shading official-reference 的水印检测真正率。 |
| traceability_true_positive_rate | metric | none | true | false | false | Gaussian Shading official-reference 的消息追踪真正率。 |
| mean_bit_accuracy | metric | none | true | false | false | Gaussian Shading official-reference 恢复消息 bit accuracy 的均值。 |
| std_bit_accuracy | metric | none | true | false | false | Gaussian Shading official-reference 恢复消息 bit accuracy 的标准差。 |
| mean_clip_score | metric | none | true | false | false | Gaussian Shading 生成图像与 Prompt 的平均 OpenCLIP 余弦相似度。 |
| std_clip_score | metric | none | true | false | false | Gaussian Shading 生成图像与 Prompt 的 OpenCLIP 余弦相似度标准差。 |
| supplemental_table_role | protocol | none | true | false | false | 官方参考环境复现记录的补充表职责; 当前方法忠实度证据固定为 supplemental_method_fidelity_reference。 |
| reference_import_ready | governance | none | true | false | false | 官方参考 受治理导入 是否接受全部输入记录且没有 schema issue。 |
| governed_reference_record_count | metric | none | true | false | false | 单个官方参考运行写出的受治理补充记录数量。 |
| main_table_eligible | governance | none | true | false | false | 当前记录或证据是否允许进入 common-backbone 主表; official-reference 忠实度证据必须为 false。 |
| official_reference_ready | governance | none | true | false | false | 当前官方参考环境复现 summary 是否通过方法自身 ready 门禁。 |
| records_nonempty_ready | governance | none | true | false | false | 当前 official-reference JSONL 是否包含至少一条有效 object 记录。 |
| records_baseline_identity_ready | governance | none | true | false | false | 当前 official-reference JSONL 的全部记录是否绑定同一预期 baseline 身份。 |
| validation_zero_rejection_ready | governance | none | true | false | false | 当前 official-reference validation 是否没有拒绝记录或 schema issue, 且 accepted records 与实际 JSONL 一致。 |
| run_manifest_ready | governance | none | true | false | false | 当前运行 manifest 的 artifact 身份、输出绑定、决策和补充证据边界是否通过。 |
| package_input_exact_set_ready | governance | none | true | false | false | Package input manifest 的动态 entry 是否精确等于当前 run 已物化文件集合, 且排除后写入的三类归档治理文件。 |
| package_input_digests_ready | governance | none | true | false | false | Package input manifest 的逐文件 SHA-256 是否可由当前已物化文件重算。 |
| package_governance_semantics_ready | governance | none | true | false | false | Package input、archive summary 与 archive manifest 是否符合 producer 的动态 entry 和后写入治理语义。 |
| source_code_version_consistent_ready | governance | none | true | false | false | 同一 official-reference family 的运行 manifest 与归档 manifest 是否共享同一 clean Git 提交标识。 |
| declared_package_entry_count | metric | none | true | false | false | 当前 official-reference package input manifest 声明的动态 entry 数量。 |
| official_reference_package_entry_digest | provenance | none | true | false | false | 当前 official-reference 动态 entry 路径及其 SHA-256 映射的稳定摘要。 |
| official_reference_source_paths | artifact | none | true | false | false | 单方法忠实度证据实际读取的 summary、manifest、records、validation 和归档治理路径映射。 |
| official_reference_source_artifact_digests | provenance | none | true | false | false | 单方法或三方法摘要实际读取的 official-reference 文件字节 SHA-256 映射。 |
| supports_main_table_superiority_claim | governance | none | true | false | false | 当前证据是否允许支持 common-backbone 主表优势结论; official-reference 忠实度证据必须为 false。 |
| supplemental_method_fidelity_evidence_ready | governance | none | true | true | false | 精确官方参考证据是否足以支持补充方法忠实度披露, 不表示主表优势。 |
| official_reference_fidelity_evidence_ready | governance | none | true | true | false | 三个固定 official-reference family 是否共同通过身份、validation、package 摘要、归档治理和 clean 代码版本门禁。 |
| expected_official_reference_baseline_ids | protocol | none | false | false | false | 方法忠实度审计唯一允许的 Tree-Ring、Gaussian Shading 和 Shallow Diffuse 身份序列。 |
| actual_official_reference_baseline_ids | protocol | none | false | false | false | 当前方法忠实度审计实际生成证据记录的方法身份序列。 |
| missing_official_reference_baseline_ids | governance | none | false | false | false | 当前方法忠实度证据相对固定三方法集合缺失的方法身份。 |
| unexpected_official_reference_baseline_ids | governance | none | false | false | false | 当前方法忠实度证据中不属于固定三方法集合的方法身份。 |
| duplicate_official_reference_baseline_ids | governance | none | false | false | false | 当前方法忠实度证据中重复出现的方法身份。 |
| official_reference_exact_set_ready | governance | none | false | false | false | 方法忠实度证据是否无缺失、无额外、无重复地覆盖固定三个 official-reference family。 |
| official_reference_fidelity_record_count | metric | none | false | false | false | 方法忠实度证据实际写出的逐方法记录数量, 闭合值为3。 |
| official_reference_fidelity_ready_count | metric | none | false | false | false | 方法忠实度证据中通过全部单方法门禁的记录数量, 闭合值为3。 |
| common_code_version_ready | governance | none | false | false | false | 三个 official-reference family 是否共享同一规范化 clean Git 提交标识。 |
| official_reference_fidelity_evidence_digest | provenance | none | false | false | false | 精确三方法忠实度证据记录列表的稳定摘要。 |
| method_threshold_digest | provenance | none | true | true | false | 单条正式论文结果记录绑定的当前方法冻结 fixed-FPR 阈值 SHA-256 摘要。 |
| method_threshold_digest_map | provenance | none | true | true | false | 方法身份到唯一冻结阈值摘要的精确映射; 正式结果必须覆盖 SLM-WM 与4个主表 baseline。 |
| method_threshold_digest_map_ready | governance | none | true | false | false | Formal import 是否为4个主表 baseline 建立无缺失、无额外且逐方法唯一的阈值摘要映射。 |
| result_record_set_digest | provenance | none | true | true | false | 对按方法、资源配置和攻击身份稳定排序后的完整正式结果记录集合计算的 SHA-256 摘要。 |
| primary_baseline_evidence_records_digest | provenance | none | true | true | false | 对精确4个 primary baseline 证据记录按方法身份排序后计算的稳定摘要。 |
| formal_evidence_paths | artifact | none | true | false | false | 单个 primary baseline 证据记录实际绑定的 observation、transfer、Prompt、adapter、execution 与 command 结果路径。 |
| primary_baseline_ids | protocol | none | true | true | false | Prompt 聚类配对优势统计唯一允许的4个主表 baseline 身份序列。 |
| proposed_decision | metric | none | true | true | false | 单个 Prompt 与攻击条件配对中 SLM-WM 的正式二元检测判定。 |
| baseline_decision | metric | none | true | true | false | 单个 Prompt 与攻击条件配对中主表 baseline 的正式二元检测判定。 |
| paired_difference | metric | none | true | true | false | 同一 Prompt 与攻击条件下 SLM-WM 判定减 baseline 判定的差值, 取值为 -1、0或1。 |
| paired_outcome_digest | provenance | none | true | true | false | 单条 Prompt x attack 配对结果核心内容的稳定摘要。 |
| proposed_method_threshold_digest | provenance | none | true | true | false | 单条配对 outcome 中 SLM-WM 实际使用的审计冻结阈值摘要。 |
| baseline_method_threshold_digest | provenance | none | true | true | false | 单条配对 outcome 中对应主表 baseline 实际使用的审计冻结阈值摘要。 |
| paired_prompt_count | metric | none | true | true | false | 单个主表 baseline 参与总体配对统计的唯一 test Prompt 数量。 |
| paired_attack_count | metric | none | true | true | false | 单个主表 baseline 对每个 Prompt 完整覆盖的攻击条件数量。 |
| paired_observation_count | metric | none | true | true | false | 单个主表 baseline 参与总体统计的 Prompt x attack 配对观测数量。 |
| mean_paired_true_positive_rate_difference | metric | none | true | true | false | 先在每个 Prompt 内跨攻击求均值, 再跨 Prompt 聚合得到的 SLM-WM 相对 baseline 平均二元检测差值。 |
| mean_paired_difference_ci_low | metric | none | true | true | false | Prompt-clustered bootstrap 对总体平均配对差值给出的 percentile CI 下界。 |
| mean_paired_difference_ci_high | metric | none | true | true | false | Prompt-clustered bootstrap 对总体平均配对差值给出的 percentile CI 上界。 |
| positive_prompt_cluster_count | metric | none | true | true | false | 平均配对差值大于0的 Prompt 聚类数量。 |
| negative_prompt_cluster_count | metric | none | true | true | false | 平均配对差值小于0的 Prompt 聚类数量。 |
| tied_prompt_cluster_count | metric | none | true | true | false | 平均配对差值等于0的 Prompt 聚类数量。 |
| one_sided_bounded_hoeffding_mean_p_value | metric | none | true | true | false | 对取值位于 [-1,1] 的 Prompt 聚类平均差执行单侧 bounded Hoeffding 均值零假设检验得到的 claim p 值。 |
| one_sided_exact_prompt_cluster_sign_flip_p_value | metric | none | true | true | false | 通过整数动态规划精确计算的 Prompt-cluster sign-flip sharp-null 诊断 p 值。 |
| exact_prompt_cluster_sign_flip_p_value_is_diagnostic | governance | none | true | false | false | 明确标识 exact sign-flip 仅检验 sharp null, 不用于均值优势 claim 门禁。 |
| sharp_null_diagnostic_method | protocol | none | true | false | false | sharp-null 诊断方法, 正式值为 exact_prompt_cluster_sign_flip_dp。 |
| claim_p_value_method | protocol | none | true | false | false | 配对优势 claim 使用的均值检验方法, 正式值为 bounded_hoeffding_prompt_cluster_mean; 结论范围限定于受治理 Prompt benchmark, 向未采样自然 Prompt 总体外推需要额外的独立性或可交换性论证。 |
| holm_adjusted_p_value | metric | none | true | true | false | 对4个主表 baseline 的单侧 bounded Hoeffding claim p 值执行 Holm 校正后的结果。 |
| bootstrap_resample_count | protocol | none | true | false | false | Prompt-clustered bootstrap 的重采样次数, 正式闭合固定为100000。 |
| bootstrap_seed_digest_random | random | _digest_random | true | false | false | 仅由固定分析 schema、baseline、规范 Prompt 集、攻击 registry、outcome 集、置信度和重采样次数确定的 bootstrap 随机源摘要。 |
| bootstrap_analysis_schema | protocol | none | true | false | false | bootstrap 固定分析规范, 正式值为 paired_prompt_cluster_bootstrap_v1。 |
| bootstrap_bit_generator | protocol | none | true | false | false | bootstrap 使用的 NumPy bit generator, 正式值为 PCG64。 |
| bootstrap_quantile_method | protocol | none | true | false | false | percentile CI 的 NumPy quantile 算法, 正式值为 linear。 |
| paired_attack_registry_digest | provenance | none | true | true | false | 配对 outcome 共同覆盖的正式攻击身份、资源档位与配置摘要 registry 的稳定摘要。 |
| protocol_digest | provenance | none | true | true | false | 单行配对统计绑定规范 threshold 行、审计报告与审计 manifest 配置摘要的统一 fixed-FPR 协议摘要。 |
| paired_superiority_ready | governance | none | true | true | false | 单个主表 baseline 是否同时满足正平均差值、正 CI 下界和 Holm 校正后显著性门禁。 |
| paired_superiority_row_count | metric | none | true | true | false | 配对总体优势表实际包含的主表 baseline 统计行数, 闭合值为4。 |
| paired_superiority_ready_ids | governance | none | true | true | false | 已通过单方法配对总体优势门禁的主表 baseline 身份序列。 |
| paired_superiority_exact_set_ready | governance | none | true | true | false | 配对总体优势统计是否无缺失、无额外且无重复地覆盖4个主表 baseline。 |
| overall_paired_superiority_ready | governance | none | true | true | false | 4个主表 baseline 是否全部通过 Prompt 聚类配对总体优势门禁。 |
| paired_superiority_rows_digest | provenance | none | true | true | false | 对按主表 baseline 稳定排序的4行总体配对统计计算的稳定摘要。 |
| paired_prompt_counts | metric | none | true | true | false | 配对优势 summary 中4个主表 baseline 的唯一 Prompt 数量集合。 |
| paired_attack_counts | metric | none | true | true | false | 配对优势 summary 中4个主表 baseline 的攻击条件数量集合。 |
| paired_outcome_count | metric | none | true | true | false | 4个主表 baseline 的全部 Prompt x attack 配对结果总数。 |
| paired_outcome_set_digest | provenance | none | true | true | false | 对全部主表 baseline Prompt x attack 配对结果计算的稳定摘要。 |
| paired_superiority_protocol_digest | provenance | none | true | true | false | 配对优势 summary、共同协议与结果分析共同绑定的统一 fixed-FPR 协议摘要。 |
| paired_test_prompt_count | metric | none | true | true | false | 4个主表 baseline 共享的规范 test Prompt 集合大小。 |
| paired_test_prompt_id_digest | provenance | none | true | true | false | 对4个主表 baseline 共享的规范 test Prompt 身份集合排序后计算的稳定摘要。 |
| expected_attack_count | protocol | none | true | false | false | 当前论文层级必须完整覆盖的正式攻击配置数量。 |
| method_observation_source_path_map | provenance | none | true | false | false | 五方法身份到 fixed-FPR 审计和配对统计实际读取的 observation 文件路径映射。 |
| paired_superiority_scale_ready | governance | none | true | true | false | 配对结果是否覆盖当前论文运行要求的完整 test Prompt 数量与同一非空攻击集合。 |
| point_estimate_effect_direction_ready | governance | none | true | true | false | 正式结果记录中 SLM-WM 的跨攻击平均 TPR 是否高于最强主表 baseline。 |
| official_reference_fidelity_digest | provenance | none | false | true | false | 结果闭合门禁对三方法 official-reference records、summary 与 manifest 计算的组合摘要。 |
| paired_superiority_digest | provenance | none | false | true | false | 结果闭合门禁对配对 outcomes、统计 rows、summary 与 manifest 计算的组合摘要。 |
| true_positive_rate_ci_low | metric | none | true | true | false | 正式 positive source 真正率置信区间下界。 |
| true_positive_rate_ci_high | metric | none | true | true | false | 正式 positive source 真正率置信区间上界。 |
| false_positive_rate_ci_low | metric | none | true | true | false | 正式总体假正率置信区间下界。 |
| false_positive_rate_ci_high | metric | none | true | true | false | 正式总体假正率置信区间上界。 |
| clean_false_positive_rate_ci_low | metric | none | true | true | false | clean negative 假正率置信区间下界。 |
| clean_false_positive_rate_ci_high | metric | none | true | true | false | clean negative 假正率置信区间上界。 |
| attacked_false_positive_rate_ci_low | metric | none | true | true | false | attacked negative 假正率置信区间下界。 |
| attacked_false_positive_rate_ci_high | metric | none | true | true | false | attacked negative 假正率置信区间上界。 |
| geometry_score_threshold | protocol | none | true | false | false | 由 calibration negative 几何分数冻结的几何诊断阈值。 |
| geometry_calibration_negative_count | metric | none | true | false | false | 参与几何诊断阈值校准的 negative 记录数量。 |
| geometry_calibration_exceedance_count | metric | none | true | false | false | calibration negative 中超过几何诊断阈值的记录数量。 |
| calibration_false_positive_count | metric | none | true | false | false | 冻结内容阈值下 calibration negative 的假正例数量。 |
| calibration_false_positive_rate | metric | none | true | false | false | 冻结内容阈值下 calibration negative 的假正率。 |
| positive_rate | metric | none | true | true | false | 指定样本角色在冻结阈值下的正判比例。 |
| content_score_mean | metric | none | true | true | false | 指定样本角色的正式连续内容分数均值。 |
| positive_rate_upper_95 | metric | none | true | true | false | 指定样本角色正判比例的单侧95%置信上界。 |
| fixed_fpr_upper_bound_ready | governance | none | true | true | false | negative 样本的假正率单侧95%上界是否不超过目标 FPR。 |
| false_positive_rate_upper_95 | metric | none | true | true | false | 攻击分组假正率的单侧95%置信上界。 |
| quality_ssim_mean | metric | none | true | true | false | 攻击分组内 source 与 attacked 图像之间的 SSIM 均值。 |
| quality_psnr_mean | metric | none | true | true | false | 攻击分组内 source 与 attacked 图像之间的有限 PSNR 均值。 |
| test_prompt_count | metric | none | true | true | false | 单个正式消融配置完整覆盖的唯一 test Prompt 数量。 |
| wrong_key_false_positive_rate | metric | none | true | true | false | 单个正式消融配置在 wrong-key negative 上的假正率。 |
| clean_true_positive_rate | metric | none | true | true | false | 单个正式消融配置在 clean positive 上的真正率。 |
| attacked_true_positive_rate | metric | none | true | true | false | 单个正式消融配置在 attacked positive 上的真正率。 |
| positive_content_score_mean | metric | none | true | true | false | 单个正式消融配置在 positive 样本上的连续内容分数均值。 |
| paired_ssim_mean | metric | none | true | true | false | 单个正式消融配置与完整方法逐 Prompt 配对图像的 SSIM 均值。 |
| frozen_threshold_digest | provenance | none | true | true | false | 正式消融记录绑定的冻结 fixed-FPR 阈值摘要。 |
| clean_true_positive_rate_delta | metric | none | true | true | false | 消融配置相对完整方法的 clean true positive rate 配对差值。 |
| attacked_true_positive_rate_delta | metric | none | true | true | false | 消融配置相对完整方法的 attacked true positive rate 配对差值。 |
| paired_ssim_delta | metric | none | true | true | false | 消融配置相对完整方法的逐 Prompt 配对 SSIM 差值。 |
| source_head_commit | provenance | none | true | false | false | 外部官方源码 checkout 实际核验得到的 HEAD 提交。 |
| source_remote_url | provenance | none | true | false | false | 外部官方源码 checkout 实际核验并归一化后的 origin 地址。 |
| source_base_worktree_clean | governance | none | true | false | false | 应用受治理补丁前的外部源码工作树是否恰好处于登记 clean commit。 |
| source_identity_ready | governance | none | true | false | false | 外部源码远端身份、HEAD 提交和基础工作树是否共同通过核验。 |
| source_modified_paths | provenance | none | true | false | false | 登记 commit 上应用确定性补丁后实际修改的相对文件路径集合。 |
| source_patch_sha256 | provenance | none | true | false | false | 相对登记 commit 的二进制 Git diff SHA-256。 |
| prompt_dataset_source | provenance | none | true | false | false | official-reference Prompt 数据集的登记来源完整记录。 |
| prompt_dataset_repository_id | provenance | none | true | false | false | official-reference 实际消费的 Prompt 数据集仓库标识。 |
| prompt_dataset_revision | provenance | none | true | false | false | official-reference 实际消费的 Prompt 数据集40位不可变提交。 |
| required_source_provenance_fields | protocol | none | true | false | false | official-reference governed record 必须包含的源码和 Prompt 数据来源字段集合。 |
| optim_utils_path | provenance | none | true | false | false | official-reference 数据加载与数学工具源码文件的受治理路径。 |
| dataset_revision | protocol | none | true | false | false | official-reference 配置绑定的 Prompt 数据集40位不可变提交。 |
| route_kind | runtime | none | true | false | false | GPU 服务器公开工作流进入科学会话、共享隔离工作流或官方参考编排器的稳定路由类别。 |
| shared_isolated_workflow_name | provenance | none | true | false | false | GPU 服务器路由调用的共享隔离科学工作流名称; 非共享路由为 null。 |
| official_reference_runner_name | provenance | none | true | false | false | GPU 服务器路由调用的官方参考编排器名称; 非官方参考路由为 null。 |
| workflow_summary | runtime | none | true | false | false | GPU 服务器内层隔离工作流返回的原始受治理摘要。 |
| archive_record | provenance | none | true | false | false | GPU 服务器共享或官方参考路由生成的结果包记录; 主方法会话为 null。 |
| orchestrator_dependency_environment | provenance | none | true | false | false | GPU 服务器 CPU 父解释器绑定的 workflow_orchestrator profile、完整锁和环境 inspection 证据。 |
| route_name | protocol | none | false | false | false | 外部 GPU workflow 持久化协议中的唯一路由名称; 三条 method-faithful 路由包含单 baseline 身份。 |
| configuration_identity_digest | provenance | none | false | false | false | 对论文运行配置、科学 profile 锁、影响结果的环境配置和正式执行锁计算的稳定摘要。 |
| configuration_environment_keys_intermediate | runtime | _intermediate | false | false | false | 进入配置身份摘要且不包含值的环境变量名称集合, 仅用于恢复诊断。 |
| checkpoint_state_intermediate | runtime | _intermediate | false | false | false | 定时 workflow 快照的运行态、中断态或完成态标记; 不表示论文证据资格。 |
| workflow_completed_intermediate | governance | _intermediate | false | false | false | 当前定时快照是否在 runner 完成后写出; 正式归档与闭合门禁仍需独立执行。 |
| checkpoint_file_records_intermediate | provenance | _intermediate | false | false | false | 定时快照保存的仓库相对路径、字节数、SHA-256 和扁平 payload 名称记录。 |
| unstable_files_intermediate | runtime | _intermediate | false | false | false | 定时复制期间仍在变化而未进入本 generation 的文件路径集合。 |
| checkpoint_digest_intermediate | provenance | _intermediate | false | false | false | 排除自指字段后对定时 checkpoint manifest 计算的稳定摘要。 |
| checkpoint_generation_intermediate | runtime | _intermediate | false | false | false | 原子 current 指针选中的不可变定时快照 generation 身份。 |
| payload_name_intermediate | runtime | _intermediate | false | false | false | checkpoint 内用于避免平台长路径问题的扁平 payload 文件名; 原路径仍由 entry record 绑定。 |
| checkpoint_persistence_configured | governance | none | false | false | false | 当前科学工作负载是否显式配置了服务器磁盘或挂载盘 checkpoint 根目录。 |
| checkpoint_persisted | governance | none | false | false | false | 摘要绑定的 checkpoint manifest 是否已完成原子发布。 |
| checkpoint_manifest_path | provenance | none | false | false | false | 已发布 checkpoint manifest 的持久化文件系统路径。 |
| checkpoint_manifest_digest | provenance | none | false | false | false | checkpoint manifest 排除自指字段后的稳定摘要。 |
| checkpoint_content_digest | provenance | none | false | false | false | checkpoint 全部 entry records 的稳定组合摘要。 |
| checkpoint_kind | protocol | none | false | false | false | checkpoint 成员职责, 区分进度记录、特征批次、完成科学单元和完成 workflow。 |
| checkpoint_id | protocol | none | false | false | false | 同一 artifact role 和 checkpoint kind 下的稳定快照身份。 |
| checkpoint_decision | governance | none | false | false | false | checkpoint 文件是否已完整发布的门禁结论; 不表示论文结论。 |
| evidence_eligibility | governance | none | true | false | false | 明确标识持久状态仅可用于续跑或诊断, 或已满足正式证据入口资格。 |
| entry_records | provenance | none | false | false | false | checkpoint manifest 中逐文件路径、大小、摘要和内部 payload 名称记录。 |
| restored_manifest_count | metric | none | false | false | false | 本次恢复通过执行锁、角色和摘要验证的 checkpoint manifest 数量。 |
| restored_file_count | metric | none | false | false | false | 本次恢复完成原子写回的普通文件数量。 |
| restored_entry_digest | provenance | none | false | false | false | 本次恢复全部仓库相对文件路径的稳定摘要。 |
| resume_checkpoint_dir | environment | none | false | false | false | 外层运行环境显式注入的 checkpoint 根目录; 未配置时通用服务器路径保持无操作。 |
| expected_feature_record_count | metric | none | false | false | false | 正式 Inception 特征续跑预期覆盖的 source 与 comparison 图像记录总数。 |
| completed_feature_record_count | metric | none | false | false | false | 已通过图像身份、提取器身份、维度和有限数值校验的特征记录数量。 |
| remaining_feature_record_count | metric | none | false | false | false | 尚未形成有效特征 shard 的图像记录数量。 |
| item_count | metric | none | false | false | false | 特征 checkpoint context 绑定的图像身份记录总数。 |
| item_identity_digest | provenance | none | false | false | false | 按确定顺序组合图像角色、路径和实际 SHA-256 后得到的稳定身份摘要。 |
| closed_archive_recovery_ready | governance | none | true | false | false | 当前请求的全部 artifact role 是否均从同一代码锁和科学依赖身份下的闭合包恢复。 |
| closed_archive_recovery | governance | none | true | false | false | 闭合包恢复诊断摘要; 部分角色命中只用于诊断, 不允许跳过当前科学执行。 |
| recovered_roles | artifact | none | true | false | false | 通过闭合包结构、代码锁、依赖身份和论文运行身份校验的 artifact role 集合。 |
| local_archives | artifact | none | true | false | false | 全角色恢复成功后原子复制到当前仓库 outputs 的闭合包路径映射。 |
| closed_archive_records | provenance | none | true | false | false | 已校验闭合包的角色、外部路径、本地路径、摘要和生成时间记录。 |
| all_expected_roles_recovered | governance | none | true | false | false | 本次请求要求的角色集合是否被有效闭合包精确覆盖。 |
| qualification_tool_lock_path | artifact | none | true | false | false | fresh Linux host 创建精确 orchestrator Python 前消费的固定 uv 单 wheel URL 与哈希锁路径。 |
| qualification_tool_lock_digest | provenance | none | true | false | false | host 资格化工具锁文件的实际 SHA-256; 该输入不属于六个运行 profile 完整锁。 |
| qualification_tool_wheel_url | provenance | none | true | false | false | 工具锁固定且实际下载的 PyPI Linux x86_64 uv wheel URL。 |
| qualification_tool_wheel_path | artifact | none | true | false | false | fresh-host 临时根目录中通过摘要门禁的 uv wheel 路径。 |
| qualification_tool_wheel_sha256 | provenance | none | true | false | false | 下载后重新计算并与工具锁比较的 uv wheel SHA-256。 |
| qualification_tool_wheel_member | provenance | none | true | false | false | 固定 wheel 内唯一被提取为资格化工具的 uv executable 成员路径。 |
| qualification_report_path | artifact | none | true | false | false | host launcher 写入 `outputs/dependency_lock_qualification/` 的资格化命令与 child 身份报告路径。 |
| accepted_reference_record_ids | provenance | none | true | true | false | official-reference 验证报告中通过全部来源、配置与指标门禁的记录身份集合。 |
| actual_formal_attack_unit_count | metric | none | true | false | false | 方法忠实外部基线实际完成并通过复验的 Prompt 攻击科学单元数量。 |
| artifact_path | artifact | none | true | false | false | 原子科学单元绑定的事实产物仓库相对路径。 |
| artifact_size | artifact | none | true | false | false | 原子科学单元绑定的事实产物字节数。 |
| base_seed_random | random | _random | true | false | false | T2SMark 逐 Prompt 随机种子计划使用的固定基础种子。 |
| channel | protocol | none | true | false | false | 水印载体参数中的 latent 通道索引或通道数量。 |
| clean_image | artifact | none | true | false | false | 严格配对单元中未嵌入水印的图像事实角色。 |
| dataset_path | artifact | none | true | false | false | 官方参考运行实际消费的数据集文件或目录路径。 |
| dependency_lock_digest | provenance | none | true | false | false | 科学单元实际使用的完整依赖锁稳定摘要。 |
| device | runtime | none | true | false | false | 科学单元实际执行设备的规范名称。 |
| effective_official_model_id | provenance | none | true | false | false | official-reference 命令解析后实际生效的官方模型身份。 |
| image_length | protocol | none | true | false | false | official-reference 生成与检测协议使用的方形图像边长。 |
| inner_radius | protocol | none | true | false | false | Tree-Ring 环形载体掩码的内半径。 |
| invalid_required_metric_fields | governance | none | true | false | false | official-reference 记录中存在但未通过类型或有限数值校验的必需指标字段集合。 |
| isolated_dependency_environment_report | provenance | none | true | false | false | 原子科学单元绑定的隔离依赖环境完整报告。 |
| isolated_dependency_environment_report_digest | provenance | none | true | false | false | 隔离依赖环境完整报告的稳定摘要。 |
| mask_shape | protocol | none | true | false | false | 水印载体掩码使用的几何形状。 |
| measured_score | metric | none | true | true | false | 单条方法忠实观测由真实检测算子计算的连续分数。 |
| missing_required_metric_fields | governance | none | true | false | false | official-reference 记录中缺失的必需指标字段集合。 |
| pattern | protocol | none | true | false | false | 水印载体在选定 latent 区域内使用的数值图案类型。 |
| primary_edit_timestep | protocol | none | true | false | false | Shallow Diffuse 正式检测采用的主编辑时间步。 |
| prompt_indices | protocol | none | true | false | false | official-reference 原子批次实际覆盖的规范 Prompt 索引序列。 |
| prompt_row | protocol | none | true | false | false | T2SMark 单元绑定的单条完整 Prompt 身份与 split 记录。 |
| prompt_seed_schedule_digest_random | random | _digest_random | true | false | false | official-reference 逐 Prompt 固定种子调度表的不可逆稳定摘要。 |
| protocol_binding | protocol | none | true | false | false | 原子科学单元绑定的攻击、检测、split 与 fixed-FPR 协议身份对象。 |
| radius | protocol | none | true | false | false | 水印载体掩码的外半径或单半径参数。 |
| random_identity_random | random | _random | true | false | false | 方法忠实单元中影响生成与载体构造的完整随机身份对象。 |
| raw_observations | artifact | none | true | false | false | 方法忠实科学单元聚合前重新读取并验证的原始观测记录集合。 |
| record_schema | protocol | none | true | false | false | 原子契约或记录采用的稳定 schema 名称。 |
| reference_model | provenance | none | true | false | false | official-reference 运行实际使用的参考检测模型身份。 |
| source_identity_digest | provenance | none | true | false | false | 方法忠实单元绑定的外部源码登记与适配实现组合摘要。 |
| source_prompt_count | metric | none | true | false | false | 方法忠实单元集合实际覆盖的唯一源 Prompt 数量。 |
| source_registry_sha256 | provenance | none | true | false | false | 外部 baseline 来源登记文件的实际 SHA-256。 |
| stable_execution_identity | provenance | none | true | false | false | 跨恢复会话必须完全一致的代码锁、依赖锁、源码与科学配置身份。 |
| torch | runtime | none | true | false | false | 依赖版本映射中实际加载的 PyTorch 版本。 |
| unit_batch_size | protocol | none | true | false | false | official-reference 原子科学单元的规范批次样本数。 |
| unit_end | protocol | none | true | false | false | official-reference 原子科学单元覆盖区间的排他结束索引。 |
| unit_kind | protocol | none | true | false | false | 原子科学单元的职责类型, 区分源图生成与攻击评估等任务。 |
| unit_parameters | protocol | none | true | false | false | 方法忠实原子科学单元完整绑定的 Prompt、攻击与随机参数对象。 |
| unit_rebuilt | governance | none | true | false | false | official-reference 归档记录是否由已复验原子单元重新构建。 |
| unit_start | protocol | none | true | false | false | official-reference 原子科学单元覆盖区间的包含式起始索引。 |
| use_chacha | protocol | none | true | false | false | Gaussian Shading 官方配置是否启用真实 ChaCha20 消息加密。 |
| user_number | protocol | none | true | false | false | Gaussian Shading 官方配置中的用户码字空间规模参数。 |
| watermarked_image | artifact | none | true | false | false | 严格配对单元中嵌入水印的图像事实角色。 |
| edit_timestep | method | none | true | false | false | Shallow Diffuse 在完整扩散步数中定义的浅层水印注入时间步。 |
| edit_schedule_index | method | none | true | false | false | Shallow Diffuse 完整 FlowMatch Euler 调度中与注入时间步对应的前段结束索引。 |
| pre_edit_guidance_scale | method | none | true | false | false | Shallow Diffuse 从初始噪声运行到浅层注入位置时使用的 Prompt guidance 强度。 |
| post_edit_guidance_scale | method | none | true | false | false | Shallow Diffuse clean 与 watermarked 分支在注入后共同使用的 guidance 强度, 正式值为1.0。 |
| detection_inversion_stop_timestep | method | none | true | false | false | Shallow Diffuse 仅图像检测反演停止并计算载体距离的生成同位时间步。 |
| detection_inversion_stop_schedule_index | method | none | true | false | false | Shallow Diffuse 检测反演停止时间步对应的完整 FlowMatch Euler 调度索引。 |
| channel_fusion | method | none | true | false | false | Shallow Diffuse 解码前保留水印通道并从 clean 分支恢复其他通道的融合规则。 |
| watermark_channel | method | none | true | false | false | Shallow Diffuse 注入、通道融合和检测共同使用的 latent 水印通道。 |
| conditioning | runtime | none | false | false | false | FlowMatch transformer 调用记录中用于核验条件分支选择的测试诊断值。 |
| dataset | protocol | none | true | false | false | official-reference 科学配置实际绑定的 Prompt 数据集仓库身份。 |
| latent_batch | runtime | none | false | false | false | FlowMatch transformer 单次调用接收的 latent batch 大小诊断值。 |
| patch_size | protocol | none | false | false | false | SD3 transformer 配置用于计算动态 shift 图像序列长度的 patch 边长。 |
| stochastic_sampling | protocol | none | false | false | false | FlowMatch scheduler 是否启用随机采样; 方法忠实正式路线要求为 false。 |
| use_dynamic_shifting | protocol | none | false | false | false | FlowMatch scheduler 是否依据图像序列长度计算动态 shift。 |
| result_path | provenance | none | false | false | false | 精确父编排入口写出的受治理 workflow 结果文件路径。 |
| orchestrator_bootstrap_identity | provenance | none | false | false | false | 标准库宿主入口实际创建的父 profile、完整锁、精确 Python 版本、解释器路径与解释器文件摘要。 |
| ablation_necessity_statistics_ready | governance | none | true | false | false | 7项机制必要性统计是否通过结构和摘要绑定门禁。 |
| adjusted_significance_ready | metric | none | true | true | false | Holm 校正后的单侧配对检验是否达到预注册显著性水平。 |
| all_mechanism_necessity_claims_supported | governance | none | true | true | false | 全部单机制必要性主张是否均满足预注册统计条件。 |
| clean_true_positive_mean_paired_effect | metric | none | true | true | false | 完整方法减去消融变体的 clean TPR Prompt 配对均值效应。 |
| clean_true_positive_mean_paired_effect_ci_low | metric | none | true | true | false | clean TPR Prompt 配对均值效应置信区间下界。 |
| clean_true_positive_mean_paired_effect_ci_high | metric | none | true | true | false | clean TPR Prompt 配对均值效应置信区间上界。 |
| confidence_interval_ready | metric | none | true | true | false | 主指标 bootstrap 置信区间下界是否超过最小效应阈值。 |
| effect_direction | protocol | none | true | true | false | 机制必要性主指标预注册的配对效应方向。 |
| effect_direction_ready | metric | none | true | true | false | 主指标实测效应是否符合预注册正方向。 |
| expected_paired_prompt_count | protocol | none | true | false | false | 当前 paper run test split 要求的精确 Prompt 配对数量。 |
| expected_variant_ablation_ids | protocol | none | true | false | false | 机制必要性统计必须精确覆盖的消融变体身份序列。 |
| input_record_digest | provenance | none | true | false | false | 机制必要性统计所消费规范化逐 Prompt 记录的稳定摘要。 |
| mean_paired_effect | metric | none | true | true | false | 完整方法减去消融变体的主指标 Prompt 配对均值效应。 |
| mean_paired_effect_ci_low | metric | none | true | true | false | 主指标 Prompt 配对均值效应 bootstrap 置信区间下界。 |
| mean_paired_effect_ci_high | metric | none | true | true | false | 主指标 Prompt 配对均值效应 bootstrap 置信区间上界。 |
| minimum_effect_ready | metric | none | true | true | false | 主指标均值效应是否达到预注册最小效应。 |
| minimum_effect_size | protocol | none | true | true | false | 机制必要性主张采用的预注册最小效应阈值。 |
| necessity_claim_decision | governance | none | true | true | false | 单机制或总机制必要性主张的实测支持决定。 |
| necessity_claim_supported | governance | none | true | true | false | 单机制必要性主张是否同时满足效应方向、最小效应、置信区间、Holm 校正显著性和 paired SSIM 质量非劣性。 |
| necessity_not_supported_ablation_ids | governance | none | true | true | false | 未达到必要性统计门槛的正式消融变体身份。 |
| necessity_statistic_row_count | metric | none | true | false | false | 独立机制必要性统计表的正式行数。 |
| necessity_statistic_rows_digest | provenance | none | true | false | false | 独立机制必要性统计行的稳定摘要。 |
| necessity_supported_ablation_ids | governance | none | true | true | false | 达到必要性统计门槛的正式消融变体身份。 |
| one_sided_paired_p_value | metric | none | true | true | false | 相对最小效应零假设的单侧 Prompt 配对 p 值。 |
| paired_p_value_method | protocol | none | true | true | false | 机制必要性单侧 Prompt 配对检验采用的方法身份。 |
| paired_prompt_id_digest | provenance | none | true | false | false | 机制必要性统计共同 test Prompt 身份集合的稳定摘要。 |
| paired_ssim_mean_paired_effect | metric | none | true | true | false | 完整方法减去消融变体的 paired SSIM Prompt 配对均值效应。 |
| paired_ssim_mean_paired_effect_ci_low | metric | none | true | true | false | paired SSIM Prompt 配对均值效应置信区间下界。 |
| paired_ssim_mean_paired_effect_ci_high | metric | none | true | true | false | paired SSIM Prompt 配对均值效应置信区间上界。 |
| paired_ssim_noninferiority_margin | protocol | none | true | true | false | 完整方法相对消融变体的 paired SSIM 预注册非劣界。 |
| paired_ssim_noninferiority_ready | metric | none | true | true | false | paired SSIM 置信区间是否满足预注册质量非劣界。 |
| primary_metric_name | protocol | none | true | true | false | 机制必要性统计预注册的主指标名称。 |
| prompt_file_sha256 | provenance | none | true | false | false | 规范 Prompt 文件的字节级 SHA-256 身份。 |
| significance_alpha | protocol | none | true | true | false | 机制必要性家族检验采用的预注册显著性水平。 |
| relation_sync_score | metric | none | true | true | false | 双边关系图配准后规范 attention 与密钥关系图的归一化一致性分数。 |
| registration_objective_margin | metric | none | true | true | false | 最优配准目标相对非近重复候选目标的差值。 |
| registration_candidate_count | protocol | none | true | false | false | 双边关系图配准实际评估的冻结仿射候选数量。 |
| sync_margin_duplicate_transform_tolerance | protocol | none | true | false | false | 计算配准目标差值时排除近重复仿射候选所用的冻结矩阵距离阈值。 |
| registration_geometry_reliable | metric | none | true | true | false | 仅依据覆盖率、唯一对应和重采样残差得到的结构配准可靠性。 |
| registration_confidence_threshold | protocol | none | true | true | false | calibration clean negatives 冻结的配准置信度门限。 |
| attention_sync_score_threshold | protocol | none | true | true | false | calibration clean negatives 冻结的对齐后真实 Q/K 同步分数门限。 |
| raw_attention_geometry_score | metric | none | true | true | false | 对待检图像未执行几何配准时计算的真实 Q/K 密钥关系分数。 |
| attention_sync_score | metric | none | true | true | false | 对齐图像重新提取真实 Q/K 后计算的密钥关系同步分数。 |
| attention_sync_source | provenance | none | true | false | false | 对齐后同步分数所使用的真实图像重提取 Q/K 数据来源。 |
| attention_content_base_score | metric | none | true | true | false | 固定加入 LF 与高斯幅值尾部更新后、加入 attention 更新前的真实 Q/K 分数。 |
| attention_final_combined_score | metric | none | true | true | false | 实际写回 LF、尾部与 attention 组合 latent 后复算的真实 Q/K 分数。 |
| scheduler_step_timestep | runtime | none | true | false | false | callback-on-step-end 当前刚完成采样步对应的 scheduler timestep。 |
| post_step_schedule_index | protocol | none | true | false | false | post-step latent 在公开 scheduler 序列中用于方法前向与盲检复现的下一索引。 |
| registration_calibration_negative_count | metric | none | true | true | false | 参与配准置信度门限冻结的 calibration clean negative 数量。 |
| registration_calibration_exceedance_count | metric | none | true | true | false | 在冻结配准置信度门限上达到或超过门限的 calibration clean negative 数量。 |
| sync_calibration_negative_count | metric | none | true | true | false | 参与对齐后 Q/K 同步门限冻结的 calibration clean negative 数量。 |
| sync_calibration_exceedance_count | metric | none | true | true | false | 在冻结同步门限上达到或超过门限的 calibration clean negative 数量。 |
| geometry_protocol_calibration_ready | governance | none | true | true | false | 关系分、配准置信度与对齐后同步分均具有完整 calibration 证据的门禁结果。 |
| frozen_geometry_score_threshold | protocol | none | true | true | false | 完整 evidence 协议在 calibration split 冻结的注册关系分门限。 |
| frozen_registration_confidence_threshold | protocol | none | true | true | false | 应用于正式 detection records 的冻结配准置信度门限。 |
| frozen_attention_sync_score_threshold | protocol | none | true | true | false | 应用于正式 detection records 的冻结对齐后 Q/K 同步门限。 |
| content_base_score | metric | none | true | true | false | 注意力回溯优化中固定内容载体基底的真实 Q/K 分数。 |
| optimization_base | method | none | true | false | false | attention 几何优化实际采用的固定基底更新身份。 |
| verified_candidate | method | none | true | false | false | 单调回溯最终复算并接受的真实候选 latent 角色。 |
| relation_transform | method | none | true | false | false | attention 双边关系图从观测参考系恢复到规范参考系的矩阵变换公式。 |
