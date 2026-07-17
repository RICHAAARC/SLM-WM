# 文档索引

## 现行构建规范

- `builds/README.md`：现行构建规范的唯一清单、依赖关系和统一完成判据。
- `builds/algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md`：唯一算法真值。
- `builds/method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md`：唯一方法机制与接口真值。
- `builds/project_construction_state.md`：唯一项目状态、迁移清单和实施顺序真值。

其他文档只能链接或解释自己的局部职责，不得复制核心公式、接口 schema 或项目状态形成第二套规范。

## 实验与证据治理

- `builds/README.md`：三档同构、质量结论和论文主张规范的唯一目录清单。
- `builds/paper_profile_protocol_isomorphism.md`：7项核心攻击、10项补充攻击、5重复与三档同构的冻结协议。
- `paper_quality_evidence_governance.md`：四图质量证据与 GPU 资格化治理。
- `runtime/external_gpu_workflow_persistence.md`：外部 GPU 会话持久化和恢复。
- `artifact_rebuild.md`：由 records 重建论文产物的规则。

## 项目与发布治理

- `file_organization.md`：目录职责。
- `release_layer_boundary.md`：依赖层级。
- `release_boundary.md`：发布边界。
- `extraction_profiles.md`：三种抽离 profile。
- `core_method_package_readme.md`：最小方法包说明源。
- `field_registry.md`：当前运行字段登记。
- `naming_governance.md`：命名限制。
- `test_case_constraints.md`：测试分层。

## 迁移期兼容记录

`legacy/method_semantic_invariants.md` 与 `configs/method_semantic_registry.json` 只用于保证迁移前实现身份可识别。它们不是目标方法的权威来源，不得支持目标方法完成状态；其退出条件由 `builds/project_construction_state.md` 唯一登记。

已删除的旧方法章节、技术路线、科学算子映射、一致性报告和核心构建指南只存在于 Git 历史中，不再作为现行规范。
