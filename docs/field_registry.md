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
| claim_id | claim | none | false | true | false | claim 审计表中的声明标识。 |
| evidence_path | claim | none | false | true | false | claim 绑定的证据路径。 |
| backend_placeholder | placeholder | _placeholder | true | false | true | Bootstrap 阶段的占位 backend 字段。 |
| example_digest_random | random | _digest_random | true | false | false | 可复现随机轨迹的 digest 字段。 |
| example_state_intermediate | intermediate | _intermediate | true | false | true | 跨步骤保存的示例中间状态字段, 正式产物生成前需要清理或迁移。 |
| example_artifact_temporary | temporary | _temporary | false | false | true | 可清理的示例临时产物标记。 |
| example_result_cache | cache | _cache | false | false | false | 可由输入、配置和代码重建的示例缓存标记。 |
