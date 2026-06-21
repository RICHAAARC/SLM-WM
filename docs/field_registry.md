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
| code_version | artifact | none | false | false | false | 产物重建所用代码版本。 |
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
| fallback_path | governance | none | false | false | false | 当前阶段失败时采用的降级或停止路径。 |
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
| scenario_count | governance | none | false | false | false | synthetic smoke 场景数量。 |
| records_are_synthetic | governance | none | false | false | false | 标记本地 records 是否为 synthetic 阶段产物。 |
| carrier_digests | method | none | false | false | false | smoke metrics 中记录的 carrier 摘要集合。 |
| minimal_method_dependency | governance | none | false | false | false | minimal smoke 依赖的最小方法包入口。 |
| writes_persistent_output_by_default | governance | none | false | false | false | 脚本默认是否写出持久化输出。 |
| attention_runtime | method | none | false | false | false | attention 原语是否接入真实运行时的状态说明。 |
| event_name | method | none | false | false | false | synthetic 阶段事件名称。 |
| event_digest | method | none | false | false | false | 事件声明或 synthetic 事件的稳定摘要。 |
| branch | method | none | false | false | false | 载体派生分支名称, 例如 LF、HF 或 attention。 |
| field_length | method | none | false | false | false | synthetic 风险场或向量字段长度。 |
| eta_saliency | method | none | false | false | false | 语义风险场中 saliency 权重。 |
| eta_semantic | method | none | false | false | false | 语义风险场中 semantic 权重。 |
| eta_texture | method | none | false | false | false | 语义风险场中 texture 权重。 |
| eta_stability | method | none | false | false | false | 语义风险场中 stability 权重。 |
| budget_min | method | none | false | false | false | 语义承载预算下界。 |
| budget_max | method | none | false | false | false | 语义承载预算上界。 |
| budget_gain | method | none | false | false | false | 由低风险区域提升承载预算的增益。 |
| texture_threshold | method | none | false | false | false | LF/HF 路由的纹理阈值。 |
| risk_threshold | method | none | false | false | false | LF 路由的风险阈值。 |
| stability_threshold | method | none | false | false | false | HF 与 attention 路由的稳定性阈值。 |
| attention_threshold | method | none | false | false | false | attention synthetic stub 路由阈值。 |
| risk_values | method | none | false | false | false | 语义风险场逐位置风险值。 |
| budget_values | method | none | false | false | false | 语义承载预算逐位置数值。 |
| lf_mask | method | none | false | false | false | LF 主证据候选区域 mask。 |
| hf_mask | method | none | false | false | false | HF 补充证据候选区域 mask。 |
| attention_mask | method | none | false | false | false | attention 几何候选区域 mask。 |
| mask_values | method | none | false | false | false | 投影到 latent 长度后的 mask 数值。 |
| masked_latent_values | method | none | false | false | false | 应用 mask 后的 latent 数值。 |
| source_length | method | none | false | false | false | 投影前 mask 长度。 |
| target_length | method | none | false | false | false | 投影后目标 latent 长度。 |
| projection_digest | method | none | false | false | false | latent mask 投影结果摘要。 |
| safe_basis | method | none | false | false | false | 语义条件安全子空间 synthetic 基底。 |
| lf_basis | method | none | false | false | false | LF 路由投影后的子基底。 |
| hf_basis | method | none | false | false | false | HF 路由投影后的子基底。 |
| attention_basis | method | none | false | false | false | attention 路由投影后的子基底。 |
| selected_indices | method | none | false | false | false | synthetic 安全基底选中的 latent 位置。 |
| basis_rank | method | none | false | false | false | synthetic 安全基底最大秩。 |
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
| frequency_band | method | none | false | false | false | 水印载体使用的频带名称。 |
| key_digest | method | none | false | false | false | 载体派生时使用的密钥材料摘要。 |
| watermark_key_digest | method | none | true | false | false | 最小 latent injection 使用的水印密钥材料摘要, 不记录原始密钥。 |
| template_values | method | none | false | false | false | LF/HF/attention synthetic carrier 模板数值。 |
| update_values | method | none | false | false | false | 单个 carrier 生成的 latent update 数值。 |
| carrier_digest | method | none | true | false | false | 单个 carrier 的稳定摘要。 |
| carrier_source | method | none | true | false | false | 最小 latent injection 中 carrier 的来源机制。 |
| carrier_width | method | none | true | false | false | 从核心算法原语导出的 carrier 基础宽度。 |
| lf_carrier_digest | method | none | true | false | false | LF carrier 的稳定摘要。 |
| hf_carrier_digest | method | none | true | false | false | HF carrier 的稳定摘要。 |
| attention_carrier_digest | method | none | true | false | false | attention synthetic stub carrier 的稳定摘要。 |
| core_update_digest | method | none | true | false | false | LF/HF/attention carrier 合成后的核心 update 摘要。 |
| tail_threshold | method | none | false | false | false | HF tail truncation 使用的截断阈值。 |
| retained_fraction | method | none | false | false | false | HF tail truncation 后保留的模板比例。 |
| synthetic_stub | method | none | false | false | false | 标记 attention carrier 是否仅为 synthetic stub。 |
| tail_fraction | method | none | false | false | false | HF tail truncation 的目标尾部保留比例。 |
| embedding_strength | method | none | false | false | false | 水印嵌入强度。 |
| anchor_id | method | none | false | false | false | 注意力几何锚点对象的稳定标识。 |
| attention_layer | method | none | false | false | false | 注意力锚点对应的层或模块名称。 |
| anchor_digest | method | none | false | false | false | 注意力锚点的稳定摘要。 |
| lf_update_values | method | none | false | false | false | LF 分量 latent update 数值。 |
| hf_update_values | method | none | false | false | false | HF 分量 latent update 数值。 |
| attention_update_values | method | none | false | false | false | attention synthetic stub 分量 latent update 数值。 |
| combined_update_values | method | none | false | false | false | 三个分量相加后的 latent update 数值。 |
| update_digest | method | none | false | false | false | 组合 latent update 摘要。 |
| evidence_id | method | none | true | false | false | 检测证据对象的稳定标识。 |
| evidence_type | method | none | true | false | false | 检测证据类型, 例如 content、geometry 或 attention。 |
| score_name | method | none | true | false | false | 检测证据分数名称。 |
| score_value | method | none | true | false | false | 检测证据分数值。 |
| lf_score | method | none | true | false | false | LF 分支归一化相关分数。 |
| hf_score | method | none | true | false | false | HF 分支截断相关分数。 |
| content_score | method | none | true | false | false | LF/HF 按固定权重融合后的内容分数。 |
| lambda_lf | method | none | true | false | false | 内容分数中 LF 分支权重。 |
| lambda_hf | method | none | true | false | false | 内容分数中 HF 分支权重。 |
| used_independent_branch_vote | method | none | true | false | false | 是否使用 LF/HF 独立阈值投票。正式方法应为 false。 |
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
| evidence_ids | method | none | true | false | false | 融合决策引用的检测证据标识集合。 |
| raw_content_score | method | none | true | false | false | 几何 rescue 前的内容分数。 |
| aligned_content_score | method | none | true | false | false | 几何对齐后的内容分数。 |
| content_threshold | protocol | none | true | false | false | fixed-FPR 内容阈值。 |
| scenario_id | protocol | none | true | false | false | synthetic smoke 场景标识。 |
| scenario_role | protocol | none | true | false | false | synthetic smoke 场景角色。 |
| observed_digest | method | none | true | false | false | synthetic 观测 latent 的稳定摘要。 |
| score_margin | method | none | true | false | false | 内容分数相对内容阈值的边界余量。 |
| raw_content_margin | method | none | true | false | false | 原始内容分数相对内容阈值的边界余量。 |
| aligned_content_margin | method | none | true | false | false | 对齐后内容分数相对内容阈值的边界余量。 |
| fail_reason | method | none | true | false | false | 内容判定失败原因。 |
| rescue_margin_low | protocol | none | true | false | false | rescue 边界失败窗口下界。 |
| positive_by_content | method | none | true | false | false | 原始内容分支是否正判。 |
| rescue_eligible | method | none | true | false | false | 样本是否满足 rescue 条件。 |
| rescue_applied | method | none | true | false | false | 是否实际应用 rescue 并通过同阈值重判。 |
| evidence_level | method | none | true | false | false | evidence-level 判定结果。 |
| evidence_decision | method | none | true | false | false | smoke 场景中的 evidence-level 判定结果。 |
| attestation_pass | method | none | true | false | false | attestation 是否验证通过。 |
| final_level | method | none | true | false | false | final-level 判定结果。 |
| final_decision | method | none | true | false | false | smoke 场景中的 final-level 判定结果。 |
| final_label | method | none | true | false | false | final-level 判定标签。 |
| correct_key_score | method | none | true | false | false | synthetic 原语中正确 key 的内容分数。 |
| wrong_key_score | method | none | true | false | false | synthetic 原语中错误 key 的内容分数。 |
| key_separation_margin | method | none | true | false | false | 正确 key 与错误 key 内容分数差。 |
| score_margin_min | method | none | true | false | false | smoke 场景中的最小 score margin。 |
| rescue_trigger_rate | method | none | true | false | false | smoke 场景中 rescue_applied 的触发比例。 |
| wrong_key_over_threshold | method | none | true | false | false | 错误 key 是否超过内容阈值。 |
| geometry_unreliable_rescue_blocked | method | none | true | false | false | 几何可靠性不足时 rescue 是否被阻断。 |
| final_positive_count | method | none | true | false | false | smoke 场景 final positive 数量。 |
| evidence_positive_count | method | none | true | false | false | smoke 场景 evidence positive 数量。 |
| hf_tail_truncation_delta | method | none | true | false | false | HF tail truncation 前后分数差异。 |
| attestation_layering_pass | method | none | true | false | false | attestation 是否只影响 final-level 的检查结果。 |
| model_family | runtime | none | true | false | false | SD runtime adapter 使用的模型族。 |
| model_id | runtime | none | true | false | false | SD runtime adapter 使用的模型标识。 |
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
| latent_width | runtime | none | false | false | false | synthetic latent adapter 使用的 latent 向量宽度。 |
| generation_id | runtime | none | true | false | false | generation record 的稳定标识。 |
| latent_digest | runtime | none | true | false | false | latent 向量或最终 latent 的稳定摘要。 |
| image_digest | runtime | none | true | false | false | synthetic 或真实生成图像的稳定摘要。 |
| image_shape | runtime | none | true | false | false | generation record 记录的图像形状。 |
| quality_score | runtime | none | true | false | false | runtime probe 的轻量质量分数。 |
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
| attention_map_digest | runtime | none | true | false | false | attention map 或降级摘要 map 的稳定摘要。 |
| attention_shape | runtime | none | true | false | false | attention map 或降级摘要 map 的形状。 |
| attention_mean | runtime | none | true | false | false | attention map 或降级摘要 map 的均值。 |
| attention_entropy | runtime | none | true | false | false | attention map 或降级摘要 map 的平均熵。 |
| capture_backend | runtime | none | true | false | false | attention capture record 的捕获后端。 |
| capture_is_synthetic | runtime | none | false | false | false | 标记 attention capture 是否由 synthetic fallback 构造。 |
| supports_paper_claim | governance | none | true | false | false | 记录或摘要是否允许支持正式论文 claim。 |
| config_digests | artifact | none | false | false | false | runtime manifest 中记录的配置摘要集合。 |
| generation_record_count | runtime | none | false | false | false | runtime summary 中 generation record 数量。 |
| latent_trace_record_count | runtime | none | false | false | false | runtime summary 中 latent trace record 数量。 |
| attention_capture_record_count | runtime | none | false | false | false | runtime summary 中 attention capture record 数量。 |
| unsupported_reason_count | runtime | none | false | false | false | runtime summary 中 unsupported_reason 数量。 |
| reproducibility_digest | runtime | none | false | false | false | runtime summary 的复现摘要。 |
| mean_quality_score | runtime | none | false | false | false | generation records 的平均质量分数。 |
| model_priority | runtime | none | true | false | false | 真实 runtime 中模型作为主线、对照或兼容 fallback 的角色。 |
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
| dependency_mode | runtime | none | false | false | false | Colab runtime 依赖安装与解析模式。 |
| manual_version_pins | runtime | none | false | false | false | 记录 Notebook 是否手工 pin 具体依赖版本。 |
| pip_install_command | runtime | none | false | false | false | Colab runtime 中用于形成依赖组合的 pip 安装命令。 |
| python_version | runtime | none | false | false | false | Colab runtime 中的 Python 版本。 |
| python_executable | runtime | none | false | false | false | Colab runtime 中执行 Notebook 的 Python 解释器路径。 |
| platform | runtime | none | false | false | false | Colab runtime 的平台摘要。 |
| package_versions | runtime | none | false | false | false | Colab runtime 中关键 Python 包版本快照。 |
| cuda_available | runtime | none | false | false | false | Colab runtime 是否可用 CUDA。 |
| cuda_version | runtime | none | false | false | false | Colab runtime 中 torch 报告的 CUDA 版本。 |
| device_count | runtime | none | false | false | false | Colab runtime 中可见 GPU 设备数量。 |
| gpu_name | runtime | none | false | false | false | Colab runtime 中首个 GPU 设备名称。 |
| runtime_environment | runtime | none | false | false | false | 真实 runtime 结果 metadata 中嵌入的环境快照。 |
| environment_report_path | artifact | none | true | false | false | 指向完整 runtime environment report JSON 的受治理路径。 |
| geometry_manifest_digest | artifact | none | true | false | false | 真实 attention 捕获运行引用的几何 manifest 稳定摘要。 |
| elapsed_seconds | runtime | none | true | false | false | 真实推理耗时秒数。 |
| error_message | runtime | none | false | false | false | 真实后端不可用时的错误消息。 |
| image_path | runtime | none | true | false | false | 真实推理输出图像的受治理路径。 |
| archive_name | artifact | none | false | false | false | 真实 runtime 产物 zip 文件名。 |
| archive_path | artifact | none | true | false | false | 真实 runtime 产物 zip 在 outputs 下的受治理路径。 |
| archive_digest | artifact | none | true | false | false | 真实 runtime 产物 zip 的 SHA-256 摘要。 |
| archive_entry_count | artifact | none | true | false | false | 真实 runtime 产物 zip 中包含的文件数量。 |
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
| prompt_set | protocol | none | true | false | false | prompt 所属集合, 例如 probe、pilot 或 full。 |
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
| hf_basis_count | method | none | false | false | false | HF 路由投影后的基底行数。 |
| attention_basis_count | method | none | false | false | false | attention 路由投影后的基底行数。 |
| content_detection_record_id | method | none | true | false | false | 内容检测 record 的稳定标识。 |
| content_mode | method | none | true | false | false | LF/HF 内容载体机制开关名称。 |
| mechanism_scores | method | none | true | false | false | 同一观测样本在各内容机制开关下的统一内容分数集合。 |
| lf_enabled | method | none | true | false | false | 内容 update 组合时是否启用 LF 主证据分量。 |
| hf_enabled | method | none | true | false | false | 内容 update 组合时是否启用 HF 补充分量。 |
| tail_truncation_enabled | method | none | true | false | false | HF 内容载体是否启用 tail truncation。 |
| content_update_digest | method | none | true | false | false | LF/HF 内容 update 组合 payload 的稳定摘要。 |
| content_chain_digest | method | none | true | false | false | 内容载体链路组合后的稳定摘要。 |
| lf_content_carrier_digest | method | none | true | false | false | LF 内容载体 payload 的稳定摘要。 |
| hf_content_carrier_digest | method | none | true | false | false | HF 内容载体 payload 的稳定摘要。 |
| score_digest | method | none | true | false | false | 统一内容分数 payload 的稳定摘要。 |
| fixed_fpr_ready | method | none | true | false | false | 内容分数是否已保持可进入 fixed-FPR 校准的统计边界。 |
| content_detection_record_count | method | none | false | false | false | 内容检测 records 数量。 |
| content_modes | method | none | false | false | false | 当前内容载体产物支持的机制开关集合。 |
| score_count | method | none | true | false | false | 内容分数表或分布表中登记的分数数量。 |
| score_min | method | none | true | false | false | 内容分数集合的最小值。 |
| score_max | method | none | true | false | false | 内容分数集合的最大值。 |
| score_mean | method | none | true | false | false | 内容分数集合的均值。 |
| score_distribution_bin | method | none | true | false | false | 内容分数分布表使用的分数区间。 |
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
| real_attention_capture_count | method | none | false | false | false | 无 unsupported reason、metadata.capture_is_synthetic=false 且含有有界 attention_matrix_preview 的真实 attention capture records 数量。 |
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
| subspace_record_count | method | none | false | false | false | 本地重建中用于派生 attention update 的语义安全子空间记录数量。 |
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
| quality_proxy_drop | method | none | true | false | false | 本地重建中用于限制图像质量风险的轻量代理退化量。 |
| attention_update_strength | method | none | true | false | false | attention update 强度曲线中的实际强度值。 |
| attention_update_stable | method | none | true | false | false | 单条 carrier 或强度行是否满足 update 稳定性边界。 |
| fallback_mode | method | none | true | false | false | attention carrier 的执行模式, active_update 表示可执行 update, evidence_only 表示仅保留几何证据。 |
| attention_relative_carrier_digest | method | none | true | false | false | attention-relative carrier 的稳定摘要。 |
| quality_metric_count | method | none | false | false | false | attention update 质量代理指标数量。 |
| quality_metric_name | method | none | false | false | false | attention update 质量代理指标名称。 |
| quality_metric_value | method | none | false | false | false | attention update 质量代理指标数值。 |
| quality_metric_source | method | none | false | false | false | attention update 质量指标来源。 |
| image_quality_metrics_ready | method | none | false | false | false | 是否已经完成真实图像质量指标测量。 |
| full_method_claim_ready | method | none | false | false | false | 是否允许把 attention-relative update 写入 Full 方法主张。当前本地重建保持 false。 |
| selected_attention_carrier_id | method | none | true | false | false | 真实 attention latent injection 中选用的 active carrier 标识。 |
| attention_geometry_package_path | artifact | none | true | false | false | 真实 attention latent injection 使用的 geometry 输入包路径。 |
| method_manifest_path | artifact | none | true | false | false | 真实运行引用的 attention latent update 方法 manifest 路径。 |
| attention_runtime_strength | method | none | true | false | false | 真实 latent callback 中应用于 attention carrier tensor 的运行时强度。 |
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
| evidence_clean_fpr | metric | none | false | false | false | clean negative 上 rescue 后 evidence-level 误报率。 |
| evidence_attacked_fpr | metric | none | false | false | false | attacked negative 上 rescue 后 evidence-level 误报率。 |
| geo_direct_positive_audit_decision | method | none | false | false | false | 仅用于反例审计的几何直接判正风险指示, 不进入正式 evidence decision。 |
| geo_direct_positive_audit_rate | metric | none | false | false | false | clean negative 上几何直接判正反例审计触发率。 |
| geo_direct_positive_audit_formal_method | method | none | false | false | false | 几何直接判正反例审计是否进入正式方法, 正式方法中必须为 false。 |
| direct_positive_decision_used | method | none | false | false | false | 是否在正式 evidence decision 中使用几何直接判正。 |
| operating_point_id | metric | none | false | false | false | fixed-FPR operating point 的稳定标识。 |
| target_fpr | protocol | none | false | false | false | 阈值校准协议的目标误报率。 |
| calibrated_content_threshold | protocol | none | false | false | false | 由 calibration clean negative 冻结的内容阈值。 |
| calibration_negative_count | metric | none | false | false | false | 用于阈值冻结的 calibration clean negative 样本数。 |
| allowed_false_positive_count | metric | none | false | false | false | 目标 FPR 下允许的 false positive 数量。 |
| observed_false_positive_count | metric | none | false | false | false | 阈值冻结数据上的实际 false positive 数量。 |
| observed_fpr | metric | none | false | false | false | 阈值冻结数据上的实际 FPR。 |
| threshold_tie_count | metric | none | false | false | false | 与冻结阈值数值相同的样本数量。 |
| threshold_degenerate | metric | none | false | false | false | 阈值是否存在退化或并列导致的 FPR 风险。 |
| threshold_source | protocol | none | false | false | false | 阈值来源, 正式协议应为 calibration clean negative。 |
| rescue_window_frozen | protocol | none | false | false | false | rescue window 是否已冻结。 |
| fail_reason_gate_frozen | protocol | none | false | false | false | fail reason gate 是否已冻结。 |
| evidence_fpr_exceeds_target | metric | none | false | false | false | rescue 后 evidence-level FPR 是否超过目标 operating point。 |
| raw_content_claim_ready | claim | none | false | false | false | raw content 分支是否满足当前 fixed-FPR 口径。 |
| true_positive_rate | metric | none | false | false | false | positive source 上的 true positive rate。 |
| false_positive_rate | metric | none | false | false | false | ROC 曲线点中的 false positive rate。 |
| false_negative_rate | metric | none | false | false | false | DET 曲线点中的 false negative rate。 |
| raw_score_auc | metric | none | false | false | false | raw content score 对 positive / clean negative 的 AUC。 |
| aligned_score_auc | metric | none | false | false | false | aligned content score 对 positive / clean negative 的 AUC。 |
| rescue_applied_rate | metric | none | false | false | false | 阈值校准口径下 rescue_applied 的比例。 |
| metric_name | metric | none | false | false | false | 常规指标名称。 |
| metric_value | metric | none | false | false | false | 常规指标数值。 |
| metric_source | metric | none | false | false | false | 常规指标来源。 |
| metric_status | metric | none | false | false | false | 指标状态, 例如 measured、proxy 或 unsupported。 |
| roc_threshold | metric | none | false | false | false | ROC 曲线点对应阈值。 |
| det_threshold | metric | none | false | false | false | DET 曲线点对应阈值。 |
| det_false_positive_rate | metric | none | false | false | false | DET 曲线点中的 false positive rate。 |
| det_false_negative_rate | metric | none | false | false | false | DET 曲线点中的 false negative rate。 |
| fpr_exceeds_target | metric | none | false | false | false | 某一 FPR 审计口径是否超过目标 FPR。 |
| decision_scope | metric | none | false | false | false | FPR 审计中的判定范围。 |
| claim_id | claim | none | false | true | false | claim 审计表中的声明标识。 |
| evidence_path | claim | none | false | true | false | claim 绑定的证据路径。 |
| attack_record_id | protocol | none | true | false | false | 攻击检测 record 的稳定标识。 |
| attack_id | protocol | none | true | false | false | 单个攻击配置的稳定标识。 |
| attack_name | protocol | none | true | false | false | 攻击配置的语义名称。 |
| attack_strength | protocol | none | true | false | false | 攻击配置使用的归一化强度。 |
| resource_profile | protocol | none | true | false | false | 攻击矩阵运行使用的资源档位, 例如 probe、pilot、full_main 或 full_extra。 |
| requires_gpu | protocol | none | true | false | false | 攻击配置是否需要真实 GPU 推理或重生成能力。 |
| attack_parameters | protocol | none | true | false | false | 攻击配置的具体参数字典。 |
| attack_config_digest | artifact | none | true | false | false | 攻击配置 payload 的稳定摘要。 |
| attack_record_digest | artifact | none | true | false | false | 攻击检测 record payload 的稳定摘要。 |
| source_record_id | protocol | none | true | false | false | 攻击记录引用的源检测 record 标识。 |
| source_image_digest | artifact | none | true | false | false | 源图像或源记录代理的稳定摘要。 |
| source_image_digest_source | artifact | none | true | false | false | source image digest 的来源说明。 |
| attacked_image_digest | artifact | none | true | false | false | 攻击后图像或本地代理攻击结果的稳定摘要。 |
| attacked_image_digest_source | artifact | none | true | false | false | attacked image digest 的来源说明。 |
| attacked_image_available | artifact | none | true | false | false | 是否存在真实可读取的攻击后图像文件。 |
| attack_performed | protocol | none | true | false | false | 当前记录是否实际执行了本地可用攻击路径。 |
| real_attack_record_id | protocol | none | true | false | false | 真实图像级攻击闭环中单条记录的稳定标识。 |
| source_image_id | artifact | none | true | false | false | 真实 source image 文件的稳定标识。 |
| source_image_path | artifact | none | true | false | false | 真实 source image 文件的受治理路径。 |
| attacked_image_path | artifact | none | true | false | false | 真实 attacked image 文件的受治理路径。 |
| attack_implementation | protocol | none | true | false | false | 真实图像级攻击运行使用的具体 pipeline 或算子机制。 |
| detection_method | method | none | true | false | false | 攻击后重跑检测时使用的受治理检测方法名称。 |
| detection_threshold | protocol | none | true | false | false | 攻击后重跑检测使用的检测阈值。 |
| attacked_image_registry_path | artifact | none | false | false | false | 真实 attacked image 注册表 JSONL 文件路径。 |
| attack_family_metrics_path | artifact | none | false | false | false | 真实图像级攻击分组指标表路径。 |
| real_attack_record_count | metric | none | false | false | false | 真实图像级攻击检测记录数量。 |
| real_attacked_image_count | metric | none | false | false | false | 已生成并登记 digest 的真实 attacked image 文件数量。 |
| regeneration_attack_record_count | metric | none | false | false | false | 再扩散类攻击检测记录数量。 |
| required_regeneration_attack_count | metric | none | false | false | false | 当前证据门禁要求的再扩散攻击类型数量。 |
| measured_regeneration_attack_count | metric | none | false | false | false | 已在真实 GPU workflow 中完成测量的再扩散攻击类型数量。 |
| real_attacked_image_closed_loop_ready | metric | none | false | false | false | 真实 source / attacked image 文件、路径和 digest 是否完成闭环。 |
| regeneration_attack_gpu_validation_ready | metric | none | false | false | false | img2img、DDIM inversion、SDEdit 和 diffusion purification 是否已由真实 GPU workflow 生成并测量。 |
| attack_detection_rerun_ready | metric | none | false | false | false | 真实 attacked image 生成后是否已重跑攻击后检测记录。 |
| formal_attack_detection_ready | metric | none | false | false | false | 真实 attacked image 是否已经转换为 attack matrix 兼容正式检测记录。 |
| formal_records_path | artifact | none | false | false | false | 真实攻击闭环写出的 attack matrix 兼容检测记录 JSONL 路径。 |
| ddim_attack_model_id | runtime | none | false | false | false | 严格 DDIM inversion 攻击使用的 diffusion attacker 模型标识。 |
| ddim_inversion_steps | runtime | none | false | false | false | DDIMInverseScheduler inversion 循环步数。 |
| ddim_reconstruction_steps | runtime | none | false | false | false | DDIM inversion 后再生成循环步数。 |
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
| hf_score_retention | metric | none | true | false | false | HF 内容分数在攻击后的保持率。 |
| score_retention | metric | none | true | false | false | 统一内容分数在攻击后的保持率。 |
| quality_score_proxy | metric | none | true | false | false | 本地攻击代理估计的质量保持分数。 |
| attention_consistency_proxy | metric | none | true | false | false | 本地攻击代理估计的 attention 一致性保持分数。 |
| attack_record_count | metric | none | false | false | false | 攻击矩阵检测 records 数量。 |
| attack_config_count | metric | none | false | false | false | 攻击矩阵配置数量。 |
| attack_family_count | metric | none | false | false | false | 攻击矩阵中唯一攻击族数量。 |
| supported_record_count | metric | none | false | false | false | 已执行本地代理攻击并可进入表格统计的记录数量。 |
| unsupported_record_count | metric | none | false | false | false | 因真实资源或输入缺失而不能进入本地统计的记录数量。 |
| gpu_attack_unsupported_count | metric | none | false | false | false | 需要真实 GPU 但当前没有真实攻击产物的记录数量。 |
| attack_manifest_path | artifact | none | true | false | false | 攻击矩阵专用 manifest 路径。 |
| attack_metrics_ready | artifact | none | false | false | false | 攻击矩阵常规攻击本地统计是否可重建。 |
| clean_false_positive_rate | metric | none | false | false | false | clean negative 在攻击矩阵统计中的 false positive rate。 |
| attacked_false_positive_rate | metric | none | false | false | false | attacked negative 在攻击矩阵统计中的 false positive rate。 |
| score_retention_mean | metric | none | false | false | false | 攻击分组内 score retention 均值。 |
| quality_score_proxy_mean | metric | none | false | false | false | 攻击分组内 quality score proxy 均值。 |
| attention_consistency_proxy_mean | metric | none | false | false | false | 攻击分组内 attention consistency proxy 均值。 |
| geometry_reliable_rate | metric | none | false | false | false | 攻击分组内几何可靠记录比例。 |
| rescue_rate | metric | none | false | false | false | 攻击分组内 rescue_applied 比例。 |
| input_manifests | artifact | none | false | false | false | 攻击矩阵或重建 manifest 引用的输入 manifest 集合。 |
| input_records_path | artifact | none | true | false | false | 攻击矩阵重建读取的源 records 路径。 |
| input_thresholds_path | artifact | none | true | false | false | 攻击矩阵重建读取的 fixed-FPR 阈值文件路径。 |
| input_threshold_report_path | artifact | none | true | false | false | 攻击矩阵重建读取的阈值边界报告路径。 |
| attacked_images_dir | artifact | none | true | false | false | 攻击后图像或本地攻击代理登记目录。 |
| performed_attack_record_count | metric | none | false | false | false | 已执行本地攻击代理的记录数量。 |
| resource_profiles | protocol | none | false | false | false | 攻击矩阵中出现的资源档位集合。 |
| conventional_attack_names | protocol | none | false | false | false | 当前攻击矩阵登记的常规攻击名称集合。 |
| regeneration_attack_names | protocol | none | false | false | false | 当前攻击矩阵登记的再扩散攻击名称集合。 |
| evaluation_boundary | protocol | none | false | false | false | 攻击后检测复用的 fixed-FPR 与 rescue 统计边界。 |
| local_proxy_boundary | governance | none | false | false | false | 本地攻击矩阵代理实现的能力边界说明。 |
| regeneration_attack_status | governance | none | false | false | false | 再扩散攻击是否已有真实产物支持的状态说明。 |
| source_supports_paper_claim | claim | none | true | false | false | 源记录是否支持论文主张的继承状态。 |
| score_retention_min | metric | none | false | false | false | 攻击分组内 score retention 最小值。 |
| score_retention_max | metric | none | false | false | false | 攻击分组内 score retention 最大值。 |
| lf_score_retention_mean | metric | none | false | false | false | 攻击分组内 LF score retention 均值。 |
| hf_score_retention_mean | metric | none | false | false | false | 攻击分组内 HF score retention 均值。 |
| positive_count | metric | none | false | false | false | 攻击矩阵分组内 positive source 记录数量。 |
| negative_count | metric | none | false | false | false | 攻击矩阵分组内非 positive source 记录数量。 |
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
| method_role | protocol | none | true | false | false | 对比表中的方法角色, 例如当前方法本地代理或外部 baseline。 |
| comparison_scope | protocol | none | true | false | false | 对比表行对应的统计或协议范围。 |
| ablation_record_id | protocol | none | true | false | false | 内部消融 record 的稳定标识。 |
| ablation_record_digest | artifact | none | true | false | false | 内部消融 record payload 的稳定摘要。 |
| ablation_id | protocol | none | true | false | false | 内部机制消融配置的稳定标识。 |
| ablation_name | protocol | none | true | false | false | 内部机制消融配置的显示名称。 |
| mechanism_group | protocol | none | true | false | false | 消融配置所属机制组。 |
| ablated_mechanism | protocol | none | true | false | false | 被关闭或替换的具体机制。 |
| mechanism_change | protocol | none | true | false | false | 消融配置实际施加的机制改变。 |
| mechanism_change_digest | artifact | none | true | false | false | 消融机制改变配置的稳定摘要。 |
| baseline_evidence_decision | metric | none | true | false | false | 完整方法代理记录中的 evidence 判定。 |
| ablated_evidence_decision | metric | none | true | false | false | 消融后在相同 fixed-FPR 边界下的 evidence 判定。 |
| ablated_detection_decision | metric | none | true | false | false | 消融后同时考虑 attestation gate 的最终检测判定。 |
| baseline_score_retention | metric | none | true | false | false | 完整方法代理记录中的 score retention。 |
| ablated_score_retention | metric | none | true | false | false | 消融后重新计算的 score retention。 |
| baseline_lf_score_retention | metric | none | true | false | false | 完整方法代理记录中的 LF score retention。 |
| ablated_lf_score_retention | metric | none | true | false | false | 消融后重新计算的 LF score retention。 |
| baseline_hf_score_retention | metric | none | true | false | false | 完整方法代理记录中的 HF score retention。 |
| ablated_hf_score_retention | metric | none | true | false | false | 消融后重新计算的 HF score retention。 |
| baseline_quality_score_proxy | metric | none | true | false | false | 完整方法代理记录中的质量保持代理分数。 |
| ablated_quality_score_proxy | metric | none | true | false | false | 消融后重新计算的质量保持代理分数。 |
| baseline_attention_consistency_proxy | metric | none | true | false | false | 完整方法代理记录中的 attention 一致性代理分数。 |
| ablated_attention_consistency_proxy | metric | none | true | false | false | 消融后重新计算的 attention 一致性代理分数。 |
| baseline_geometry_reliable | metric | none | true | false | false | 完整方法代理记录中的几何可靠性。 |
| ablated_geometry_reliable | metric | none | true | false | false | 消融后重新计算的几何可靠性。 |
| baseline_rescue_applied | metric | none | true | false | false | 完整方法代理记录中的 rescue 触发状态。 |
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
| real_lf_score_before | metric | none | true | false | false | 对齐前真实 latent 投影上的 LF 内容分数。 |
| real_lf_score_after | metric | none | true | false | false | 对齐后真实 latent 投影上的 LF 内容分数。 |
| real_hf_score_before | metric | none | true | false | false | 对齐前真实 latent 投影上的 HF 内容分数。 |
| real_hf_score_after | metric | none | true | false | false | 对齐后真实 latent 投影上的 HF 内容分数。 |
| latent_projection_digest_before | artifact | none | true | false | false | 对齐前真实 latent 投影向量的稳定摘要。 |
| latent_projection_digest_after | artifact | none | true | false | false | 对齐后真实 latent 投影向量的稳定摘要。 |
| latent_projection_values_before | method | none | true | false | false | 对齐前真实 latent 投影到内容检测维度后的有界向量。 |
| latent_projection_values_after | method | none | true | false | false | 对齐后真实 latent 投影到内容检测维度后的有界向量。 |
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
| pair_perceptual_dependency_install_command | runtime | none | false | false | false | Colab 中安装 LPIPS 可选依赖的命令记录。 |
| perceptual_metric_device_name | runtime | none | false | false | false | LPIPS 与 CLIP pair-level 指标计算使用的设备名称。 |
| fid | metric | none | false | false | false | Fréchet Inception Distance 指标值或 unsupported 状态。 |
| fid_status | metric | none | false | false | false | FID 指标计算状态。 |
| kid | metric | none | false | false | false | Kernel Inception Distance 指标值或 unsupported 状态。 |
| kid_status | metric | none | false | false | false | KID 指标计算状态。 |
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
| artifact_builder_ready | artifact | none | false | false | false | 论文产物构建器是否至少具备可运行审计链路。 |
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
| profile_name | governance | none | false | false | false | release 抽取配置的稳定名称。 |
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
| dependency_profile | runtime | none | true | false | false | 外部 baseline 官方源码复现所需的依赖环境画像。 |
| recommended_python | runtime | none | true | false | false | 外部 baseline 官方源码建议使用的 Python 版本。 |
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
| baseline_command_results_path | artifact | none | false | false | false | 外部 baseline 命令执行结果 JSON 路径。 |
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
| missing_result_indices | metric | none | true | false | false | T2SMark 输入 image pair 中缺少官方结果的索引集合。 |
| adapter_digest | artifact | none | true | false | false | 外部 baseline adapter manifest 的稳定摘要。 |
| backend_placeholder | placeholder | _placeholder | true | false | true | Bootstrap 阶段的占位 backend 字段。 |
| example_digest_random | random | _digest_random | true | false | false | 可复现随机轨迹的 digest 字段。 |
| example_state_intermediate | intermediate | _intermediate | true | false | true | 跨步骤保存的示例中间状态字段, 正式产物生成前需要清理或迁移。 |
| example_artifact_temporary | temporary | _temporary | false | false | true | 可清理的示例临时产物标记。 |
| example_result_cache | cache | _cache | false | false | false | 可由输入、配置和代码重建的示例缓存标记。 |
| external_baseline_gpu_smoke_ready | artifact | none | false | false | false | 外部 baseline 真实 GPU smoke 链路是否已跑通并生成可审计结果包。|
| t2smark_real_gpu_smoke_ready | artifact | none | false | false | false | T2SMark 官方 SD3.5 Medium 真实 GPU 最小复现是否已生成或复用成功。|
| t2smark_official_result_generated | artifact | none | false | false | false | T2SMark 官方 results.json 是否由本次运行生成。|
| t2smark_official_result_reused | artifact | none | false | false | false | T2SMark 官方 results.json 是否来自本地或 Drive 历史结果复用。|
| t2smark_source_available | artifact | none | false | false | false | T2SMark 官方源码入口在当前工作区中是否可用。|
| t2smark_source_downloaded | artifact | none | false | false | false | T2SMark 官方源码缓存是否由本次冷启动流程下载。|
| source_available | artifact | none | false | false | false | 外部源码缓存入口文件是否存在且可用于后续命令。|
| source_downloaded | artifact | none | false | false | false | 外部源码缓存是否由本次命令补齐。|
| source_entry_path | artifact | none | false | false | false | 外部源码缓存入口脚本路径。|
| prior_package_reused | artifact | none | false | false | false | Google Drive 历史结果包是否被本次 workflow 复用。|
| prior_package_path | artifact | none | false | false | false | 被复用的 Google Drive 历史结果包路径。|
| prior_package_digest | artifact | none | false | false | false | 被复用的 Google Drive 历史结果包 SHA256 摘要。|
| extracted_entry_count | metric | none | false | false | false | 从历史结果包中解出的可复用文件数量。|
| extracted_entries | artifact | none | false | false | false | 从历史结果包中解出的可复用文件路径集合。|
| adapter_execution_ready | artifact | none | false | false | false | 外部 baseline adapter 命令计划是否执行并通过证据边界校验。|
| adapter_observation_count | metric | none | false | false | false | 外部 baseline adapter 输出的 observation 数量。|
| primary_baseline_adapter_ready | artifact | none | false | false | false | 四个主表 external baseline adapter 在同一 GPU smoke 命令计划中是否全部跑通。|
| primary_baseline_adapter_count | metric | none | false | false | false | 本次 GPU smoke 命令计划覆盖的主表 external baseline adapter 数量。|
| primary_baseline_observation_count | metric | none | false | false | false | 本次 GPU smoke 命令计划中主表 external baseline adapter 输出的 observation 总数。|
| primary_baseline_ids | protocol | none | false | false | false | 本次 GPU smoke 命令计划覆盖的主表 external baseline id 集合。|
| ready_primary_baseline_ids | protocol | none | false | false | false | 本次 GPU smoke 命令计划中已经成功输出 observation 的主表 external baseline id 集合。|
| primary_baseline_observation_count_by_id | metric | none | false | false | false | 按主表 external baseline id 聚合的 observation 数量。|
| primary_baseline_prompt_plan_path | artifact | none | false | false | false | 三类 latent smoke adapter 读取的最小 prompt 计划路径。|
| adapter_boundary | governance | none | false | false | false | adapter observation 或 manifest 对工程 smoke 与正式论文证据边界的说明。|
| execution_device | runtime | none | false | false | false | adapter 张量或诊断分数实际执行设备。|
| torch_available | runtime | none | false | false | false | adapter 运行环境中是否可导入 torch。|
| adapter_seed | runtime | none | false | false | false | adapter 为单条 prompt 派生的可复现整数种子。|
| score_metadata | runtime | none | false | false | false | adapter manifest 中按 prompt 记录的轻量分数计算元数据。|
| adapter_unsupported_reason | governance | none | false | false | false | 外部 baseline adapter 未能通过 smoke 链路时记录的边界原因。|
| official_result_generated | artifact | none | false | false | false | 外部 baseline 官方结果是否由本次命令生成。|
| official_result_reused | artifact | none | false | false | false | 外部 baseline 官方结果是否由本地或历史结果复用。|
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
