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
| project_stage | governance | none | true | false | false | 当前项目语义阶段。 |
| target_construction_phase | governance | none | true | false | false | 当前构建目标。 |
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
| stage_name | governance | none | false | false | false | 项目分阶段构建流程中的语义阶段名称。 |
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
| embedding_strength | method | none | false | false | false | 水印嵌入强度。 |
| anchor_id | method | none | false | false | false | 注意力几何锚点对象的稳定标识。 |
| attention_layer | method | none | false | false | false | 注意力锚点对应的层或模块名称。 |
| anchor_digest | method | none | false | false | false | 注意力锚点的稳定摘要。 |
| evidence_id | method | none | true | false | false | 检测证据对象的稳定标识。 |
| evidence_type | method | none | true | false | false | 检测证据类型, 例如 content、geometry 或 attention。 |
| score_name | method | none | true | false | false | 检测证据分数名称。 |
| score_value | method | none | true | false | false | 检测证据分数值。 |
| decision_id | method | none | true | false | false | 多证据融合检测决策的稳定标识。 |
| decision_label | method | none | true | false | false | 多证据融合检测决策标签。 |
| threshold_name | protocol | none | true | false | false | 检测或校准协议使用的阈值名称。 |
| threshold_value | protocol | none | true | false | false | 检测或校准协议使用的阈值数值。 |
| evidence_ids | method | none | true | false | false | 融合决策引用的检测证据标识集合。 |
| claim_id | claim | none | false | true | false | claim 审计表中的声明标识。 |
| evidence_path | claim | none | false | true | false | claim 绑定的证据路径。 |
| backend_placeholder | placeholder | _placeholder | true | false | true | Bootstrap 阶段的占位 backend 字段。 |
| example_digest_random | random | _digest_random | true | false | false | 可复现随机轨迹的 digest 字段。 |
| example_state_intermediate | intermediate | _intermediate | true | false | true | 跨步骤保存的示例中间状态字段, 正式产物生成前需要清理或迁移。 |
| example_artifact_temporary | temporary | _temporary | false | false | true | 可清理的示例临时产物标记。 |
| example_result_cache | cache | _cache | false | false | false | 可由输入、配置和代码重建的示例缓存标记。 |
