"""验证方法语义规范追踪不能被实现自述或弱绑定绕过."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from main.methods.method_definition import (
    METHOD_DEFINITION_SCHEMA,
    semantic_conditioned_latent_method_definition_digest,
)
from tools.harness.lib.method_semantic_registry import (
    EXPECTED_CPU_PROPERTY_IDS,
    EXPECTED_CPU_PROPERTY_TEST_NODES,
    EXPECTED_INVARIANT_IDS,
    EXPECTED_METHOD_IMPLEMENTATION_SYMBOLS,
    EXPECTED_RUNTIME_BINDING_SYMBOLS,
    EXPECTED_SPECIFICATION_TEST_NODES,
    REGISTRY_SCOPE,
    load_method_semantic_registry,
    validate_method_semantic_registry,
)


ROOT = Path(__file__).resolve().parents[2]
INVARIANT_DOCUMENT = ROOT / "docs" / "builds" / "method_semantic_invariants.md"


def _violations(payload: dict[str, object]) -> list[dict[str, str]]:
    """使用当前方法定义身份验证指定登记内容."""

    return validate_method_semantic_registry(
        ROOT,
        payload,
        expected_method_definition_schema=METHOD_DEFINITION_SCHEMA,
        expected_method_definition_digest=(
            semantic_conditioned_latent_method_definition_digest()
        ),
    )


def _rules(payload: dict[str, object]) -> set[str]:
    """返回指定错误登记触发的规则集合."""

    return {violation["rule"] for violation in _violations(payload)}


def _invariants_by_id(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    """按冻结标识索引登记项."""

    return {
        str(item["invariant_id"]): item
        for item in payload["invariants"]
    }


@pytest.mark.constraint
def test_method_semantic_registry_has_exact_normative_trace_contract() -> None:
    """登记表必须覆盖全部独立不变量, 但只能声明规范追踪职责."""

    payload = load_method_semantic_registry(ROOT)
    invariants = _invariants_by_id(payload)

    assert _violations(payload) == []
    assert payload["registry_scope"] == REGISTRY_SCOPE
    assert tuple(invariants) == EXPECTED_INVARIANT_IDS
    assert {
        invariant_id: item["cpu_property_id"]
        for invariant_id, item in invariants.items()
    } == EXPECTED_CPU_PROPERTY_IDS
    assert {
        invariant_id: tuple(item["specification_test_nodes"])
        for invariant_id, item in invariants.items()
    } == EXPECTED_SPECIFICATION_TEST_NODES
    assert {
        invariant_id: tuple(item["cpu_property_test_nodes"])
        for invariant_id, item in invariants.items()
    } == EXPECTED_CPU_PROPERTY_TEST_NODES


@pytest.mark.constraint
def test_registry_separates_core_method_and_runtime_bindings() -> None:
    """核心方法符号与真实模型运行绑定必须使用不同字段和目录边界."""

    payload = load_method_semantic_registry(ROOT)
    invariants = _invariants_by_id(payload)

    for invariant_id, item in invariants.items():
        method_bindings = tuple(
            (binding["path"], binding["symbol"])
            for binding in item["method_implementation_symbols"]
        )
        runtime_bindings = tuple(
            (binding["path"], binding["symbol"])
            for binding in item["runtime_binding_symbols"]
        )
        assert method_bindings == EXPECTED_METHOD_IMPLEMENTATION_SYMBOLS[invariant_id]
        assert runtime_bindings == EXPECTED_RUNTIME_BINDING_SYMBOLS[invariant_id]
        assert all(path.startswith("main/") for path, _ in method_bindings)
        assert all(path.startswith("experiments/") for path, _ in runtime_bindings)

    assert invariants["three_branch_update_composition"][
        "method_implementation_symbols"
    ]
    assert invariants["actual_dtype_write_revalidation"][
        "method_implementation_symbols"
    ]


@pytest.mark.constraint
def test_registry_cannot_self_assert_conformance_with_status_aliases() -> None:
    """登记表不得用任何状态别名代替独立 CPU 或 GPU 测量."""

    for key in ("ready", "cpu_verified", "gpu_verified", "claim_status"):
        payload = deepcopy(load_method_semantic_registry(ROOT))
        payload["invariants"][0][key] = True

        assert "self_asserted_conformance" in _rules(payload)


@pytest.mark.constraint
def test_registry_rejects_incomplete_or_reordered_invariant_set() -> None:
    """删除或重排任一独立不变量必须破坏冻结集合门禁."""

    incomplete = deepcopy(load_method_semantic_registry(ROOT))
    incomplete["invariants"].pop()
    reordered = deepcopy(load_method_semantic_registry(ROOT))
    reordered["invariants"][0], reordered["invariants"][1] = (
        reordered["invariants"][1],
        reordered["invariants"][0],
    )

    assert "invariant_exact_set" in _rules(incomplete)
    assert "invariant_exact_set" in _rules(reordered)


@pytest.mark.constraint
def test_registry_separates_specification_nodes_from_cpu_property_nodes() -> None:
    """规范关联测试不得被改绑成独立 CPU 性质验证."""

    property_drift = deepcopy(load_method_semantic_registry(ROOT))
    property_drift["invariants"][0]["cpu_property_id"] = "always_true_property"
    missing_test = deepcopy(load_method_semantic_registry(ROOT))
    missing_test["invariants"][0]["specification_test_nodes"] = [
        "tests/functional/test_real_scientific_operators.py::test_not_present"
    ]
    circular_test = deepcopy(load_method_semantic_registry(ROOT))
    circular_test["invariants"][0]["specification_test_nodes"] = [
        "tests/constraints/test_method_semantic_registry_contract.py::"
        "test_method_semantic_registry_has_exact_normative_trace_contract"
    ]
    weak_property_claim = deepcopy(load_method_semantic_registry(ROOT))
    weak_property_claim["invariants"][0]["cpu_property_test_nodes"] = [
        "tests/constraints/test_method_definition_contract.py::"
        "test_machine_readable_method_definition_freezes_constructive_semantics"
    ]

    assert "cpu_property_id" in _rules(property_drift)
    assert "specification_test_nodes" in _rules(missing_test)
    assert "specification_test_nodes" in _rules(circular_test)
    assert "cpu_property_test_nodes" in _rules(weak_property_claim)


@pytest.mark.constraint
def test_registry_rejects_symbol_rebinding_and_path_escape() -> None:
    """实现绑定必须保持冻结层级、真实符号和仓库内路径."""

    unrelated_symbol = deepcopy(load_method_semantic_registry(ROOT))
    unrelated_symbol["invariants"][0]["method_implementation_symbols"][0][
        "symbol"
    ] = "semantic_conditioned_latent_method_definition_digest"
    wrong_layer = deepcopy(load_method_semantic_registry(ROOT))
    wrong_layer["invariants"][0]["runtime_binding_symbols"][0]["path"] = (
        "main/methods/method_definition.py"
    )
    path_escape = deepcopy(load_method_semantic_registry(ROOT))
    path_escape["invariants"][0]["method_implementation_symbols"][0]["path"] = (
        "../main/methods/method_definition.py"
    )

    assert "method_implementation_symbols" in _rules(unrelated_symbol)
    assert "runtime_binding_symbols" in _rules(wrong_layer)
    assert "method_implementation_symbols" in _rules(path_escape)


@pytest.mark.constraint
def test_registry_rejects_weak_definition_and_evidence_references() -> None:
    """定义标题必须唯一, 证据字段必须是字段登记表中的精确字段."""

    weak_pointer = deepcopy(load_method_semantic_registry(ROOT))
    weak_pointer["invariants"][0]["definition_pointer"] = (
        "docs/builds/method_semantic_invariants.md#not_the_invariant"
    )
    unregistered_field = deepcopy(load_method_semantic_registry(ROOT))
    unregistered_field["invariants"][0]["runtime_evidence_fields"][0] = (
        "unregistered_scientific_evidence"
    )

    assert "definition_anchor" in _rules(weak_pointer)
    assert "field_registry" in _rules(unregistered_field)


@pytest.mark.constraint
def test_registry_configuration_fields_resolve_exact_yaml_dot_paths() -> None:
    """配置追踪必须解析真实 YAML 键并拒绝抽象分支名或不存在字段."""

    payload = load_method_semantic_registry(ROOT)
    invariants = _invariants_by_id(payload)
    risk_fields = invariants["branch_risk_bounds_written_update"][
        "configuration_fields"
    ]

    assert "risk_neutral_texture_value" in risk_fields
    assert (
        "lf_content_risk_config.local_contrast_risk_weight" in risk_fields
    )
    assert "tail_robust_risk_config.texture_preference" in risk_fields
    assert (
        "attention_geometry_risk_config.attention_instability_weight"
        in risk_fields
    )
    assert all("{" not in field and "}" not in field for field in risk_fields)
    assert "method_numeric_epsilon" not in risk_fields
    assert "direction_activity_epsilon" not in risk_fields

    missing_field = deepcopy(payload)
    missing_field["invariants"][0]["configuration_fields"][0] = (
        "missing_method_configuration"
    )
    abstract_branch = deepcopy(payload)
    abstract_branch["invariants"][3]["configuration_fields"][0] = (
        "branch_risk_config.local_contrast_risk_weight"
    )
    missing_nested_leaf = deepcopy(payload)
    missing_nested_leaf["invariants"][3]["configuration_fields"][0] = (
        "lf_content_risk_config.missing_weight"
    )

    assert "configuration_fields" in _rules(missing_field)
    assert "configuration_fields" in _rules(abstract_branch)
    assert "configuration_fields" in _rules(missing_nested_leaf)


@pytest.mark.constraint
def test_registry_freezes_formula_failure_and_substitution_responsibilities() -> None:
    """每项追踪必须包含公式、配置、失败条件、禁用替代与 GPU 原子角色."""

    payload = load_method_semantic_registry(ROOT)

    for item in payload["invariants"]:
        assert item["formal_expression"]
        assert isinstance(item["configuration_fields"], list)
        assert item["fail_closed_conditions"]
        assert item["forbidden_substitutes"]
        assert item["specification_test_nodes"]
        assert isinstance(item["cpu_property_test_nodes"], list)
        assert item["gpu_atomic_roles"]
        assert item["gpu_observation_requirement"].strip()
        assert item["claim_boundary"].strip()

    formula_drift = deepcopy(payload)
    formula_drift["invariants"][0]["formal_expression"] = ["always_true"]
    forbidden_substitute_drift = deepcopy(payload)
    forbidden_substitute_drift["invariants"][0]["forbidden_substitutes"].pop()
    evidence_role_drift = deepcopy(payload)
    evidence_role_drift["invariants"][0]["runtime_evidence_fields"].pop()

    assert "normative_trace_digest" in _rules(formula_drift)
    assert "normative_trace_digest" in _rules(forbidden_substitute_drift)
    assert "normative_trace_digest" in _rules(evidence_role_drift)


@pytest.mark.constraint
def test_registry_covers_required_method_atomic_roles() -> None:
    """追踪必须覆盖风险、Null Space、量化写回、成图和联合内容身份."""

    invariants = _invariants_by_id(load_method_semantic_registry(ROOT))

    assert {
        "pipeline_class_name",
        "vae_class_name",
        "transformer_class_name",
        "scheduler_class_name",
        "vae_scaling_factor",
        "vae_shift_factor",
        "latent_torch_dtype",
        "vision_torch_dtype",
        "public_detection_schedule_index",
    }.issubset(
        invariants["frozen_model_operator_identity"]["configuration_fields"]
    )
    assert "formal_method_config_digest" in invariants[
        "frozen_model_operator_identity"
    ]["runtime_evidence_fields"]
    assert any(
        "formal_method_config_digest" in expression
        for expression in invariants["frozen_model_operator_identity"][
            "formal_expression"
        ]
    )

    assert "texture_risk_attention=risk_neutral_texture_value" in invariants[
        "branch_risk_bounds_written_update"
    ]["formal_expression"]
    assert {
        "semantic_risk_signal_content_sha256",
        "texture_risk_signal_content_sha256",
        "local_contrast_risk_signal_content_sha256",
        "adjacent_step_stability_signal_content_sha256",
        "attention_stability_signal_content_sha256",
    }.issubset(invariants["branch_signal_origin"]["runtime_evidence_fields"])
    assert {
        "effective_budget_values_content_sha256",
        "branch_nominal_strength",
        "branch_applied_strength",
        "branch_risk_scale_factor",
        "branch_budget_ceiling",
    }.issubset(
        invariants["branch_risk_bounds_written_update"][
            "runtime_evidence_fields"
        ]
    )
    assert {
        "projected_direction_matrix_content_sha256",
        "projected_direction_response_matrix_content_sha256",
    }.issubset(
        invariants["exact_jacobian_low_response_subspace"][
            "runtime_evidence_fields"
        ]
    )
    assert {
        "quantized_write_common_scale",
        "quantized_write_backtracking_step_count",
        "quantized_write_composition_order",
    }.issubset(
        invariants["actual_dtype_write_revalidation"][
            "runtime_evidence_fields"
        ]
    )
    assert {
        "clean_image_digest",
        "carrier_only_counterfactual_image_digest",
        "watermarked_image_digest",
        "final_image_qk_atomic_content_digest",
    }.issubset(
        invariants["final_image_attention_attribution"][
            "runtime_evidence_fields"
        ]
    )
    assert {
        "scientific_content_binding_schema",
        "scientific_content_binding_record",
        "scientific_content_binding_digest",
        "scientific_content_binding_digests",
        "scientific_content_binding_failure_count",
        "scientific_content_binding_gate_ready",
        "image_rgb_uint8_content_schema",
        "image_rgb_uint8_content_sha256",
        "full_update_content_bundle_digest",
        "carrier_only_update_content_bundle_digest",
        "detection_content_bundle_digest",
        "final_image_content_bundle_digest",
        "detection_qk_image_content_binding_digest",
        "final_image_qk_image_content_binding_digest",
        "public_detection_noise_evaluation_index",
        "public_detection_noise_evaluation_indices",
        "scientific_unit_config",
        "scientific_unit_config_digest",
        "manifest_path",
        "output_paths",
        "alignment_digest",
        "final_image_public_detection_noise_evidence_records",
        "final_image_public_detection_noise_evidence_digest",
        "final_image_public_detection_noise_content_sha256",
        "final_image_public_detection_noise_prg_identity_digest",
        "final_image_public_detection_noise_evidence_ready",
        "final_image_public_detection_noise_identity",
    }.issubset(
        invariants["scientific_content_binding"][
            "runtime_evidence_fields"
        ]
    )
    assert any(
        "final_images" in expression
        for expression in invariants["scientific_content_binding"][
            "formal_expression"
        ]
    )
    assert any(
        "dataset_scientific_content_binding_gate" in expression
        for expression in invariants["scientific_content_binding"][
            "formal_expression"
        ]
    )
    assert any(
        "public_detection_noise_evaluation_indices=range" in expression
        for expression in invariants["scientific_content_binding"][
            "formal_expression"
        ]
    )
    assert any(
        "for_each_unit_rebuild" in expression
        for expression in invariants["scientific_content_binding"][
            "formal_expression"
        ]
    )
    assert any(
        "carrier_only_artifact_binding" in expression
        for expression in invariants["scientific_content_binding"][
            "formal_expression"
        ]
    )
    assert any(
        "range(3,3+detection_qk_evaluation_count)" in expression
        for expression in invariants["scientific_content_binding"][
            "formal_expression"
        ]
    )
    assert any(
        "recompute_each_embedded" in expression
        for expression in invariants["scientific_content_binding"][
            "formal_expression"
        ]
    )
    assert any(
        "disjoint_union" in expression
        for expression in invariants["scientific_content_binding"][
            "formal_expression"
        ]
    )
    assert any(
        "final_image_public_noise_indices=(0,1,2)" in expression
        for expression in invariants["scientific_content_binding"][
            "formal_expression"
        ]
    )
    assert any(
        "final_detection_public_noise_identity=shared" in expression
        for expression in invariants["scientific_content_binding"][
            "formal_expression"
        ]
    )
    assert "omitted_unit_leaves_packaging" in invariants[
        "scientific_content_binding"
    ]["forbidden_substitutes"]
    assert {
        (
            "experiments/runtime/scientific_content_binding.py",
            "build_scientific_content_binding_record",
        ),
        (
            "experiments/runtime/scientific_content_binding.py",
            "_public_noise_evidence_identity",
        ),
        (
            "experiments/runners/semantic_watermark_runtime.py",
            "_carrier_only_counterfactual_artifact_binding_ready",
        ),
        (
            "experiments/runners/semantic_watermark_runtime.py",
            "_scientific_content_binding_validation_parameters",
        ),
        (
            "experiments/runners/semantic_watermark_runtime.py",
            "_scientific_content_binding_artifact_ready",
        ),
        (
            "experiments/runners/image_only_dataset_runtime.py",
            "run_image_only_dataset_runtime",
        ),
        (
            "experiments/runners/image_only_dataset_runtime.py",
            "package_image_only_dataset_runtime",
        ),
    }.issubset(
        {
            (binding["path"], binding["symbol"])
            for binding in invariants["scientific_content_binding"][
                "runtime_binding_symbols"
            ]
        }
    )
    assert {
        "public_detection_schedule_index",
        "public_detection_noise_prg_protocol",
        "public_detection_noise_domain",
        "public_detection_conditioning_protocol",
        "public_detection_condition_text",
    }.issubset(
        invariants["versioned_key_prg_reconstruction"]["configuration_fields"]
    )
    assert {
        "public_detection_noise_prg_identity",
        "public_detection_noise_prg_identity_digest",
    }.issubset(
        invariants["versioned_key_prg_reconstruction"][
            "runtime_evidence_fields"
        ]
    )
    assert "public_detection_noise_tensor" in invariants[
        "versioned_key_prg_reconstruction"
    ]["gpu_atomic_roles"]
    public_noise_fields = {
        "public_detection_noise_content_sha256",
        "public_detection_noise_prg_identity_digest",
        "public_detection_noise_evidence_digest",
        "public_detection_noise_evidence_records",
        "public_detection_noise_evidence_ready",
    }
    for invariant_id in (
        "image_only_detection_boundary",
        "same_threshold_geometry_rescue",
        "scientific_content_binding",
    ):
        assert public_noise_fields.issubset(
            invariants[invariant_id]["runtime_evidence_fields"]
        )
        assert "public_detection_noise_evidence_bundle" in invariants[
            invariant_id
        ]["gpu_atomic_roles"]
    assert any(
        "public_detection_noise_evidence" in expression
        for expression in invariants["scientific_content_binding"][
            "formal_expression"
        ]
    )


@pytest.mark.constraint
def test_registry_binds_preregistered_alignment_gate_evidence() -> None:
    """盲检和同阈值救回必须登记完整结构门禁及其核心实现符号."""

    invariants = _invariants_by_id(load_method_semantic_registry(ROOT))
    required_fields = {
        "attention_alignment_gate",
        "attention_anchor_count",
        "attention_residual_threshold",
        "attention_minimum_inlier_ratio",
        "alignment_digest",
    }
    required_symbols = {
        (
            "main/methods/geometry/attention_alignment.py",
            "attention_alignment_gate_record",
        ),
        (
            "main/methods/geometry/attention_alignment.py",
            "recover_attention_affine_alignment",
        ),
    }
    for invariant_id in (
        "image_only_detection_boundary",
        "same_threshold_geometry_rescue",
    ):
        evidence_fields = set(
            invariants[invariant_id]["runtime_evidence_fields"]
        )
        if invariant_id == "image_only_detection_boundary":
            assert required_fields - {"alignment_digest"} <= evidence_fields
        else:
            assert required_fields <= evidence_fields
        bindings = {
            (binding["path"], binding["symbol"])
            for binding in invariants[invariant_id][
                "method_implementation_symbols"
            ]
        }
        assert required_symbols <= bindings


@pytest.mark.constraint
def test_registry_formulas_freeze_risk_qk_null_and_actual_write_semantics() -> None:
    """规范公式必须显式覆盖关键项, 不能用宽泛算子名称代替."""

    invariants = _invariants_by_id(load_method_semantic_registry(ROOT))
    risk = invariants["branch_risk_bounds_written_update"]["formal_expression"]
    qk = invariants["direct_qk_four_component_relation"]["formal_expression"]
    null_space = invariants["exact_jacobian_low_response_subspace"][
        "formal_expression"
    ]
    actual_write = invariants["actual_dtype_write_revalidation"][
        "formal_expression"
    ]
    finite = invariants["finite_feature_preservation"]["formal_expression"]

    risk_formula = next(expression for expression in risk if expression.startswith("rho_b="))
    assert "local_contrast_risk" in risk_formula
    assert "semantic_risk" in risk_formula
    assert "texture_risk_b" in risk_formula
    assert "(1-adjacent_step_stability)" in risk_formula
    assert "(1-attention_stability)" in risk_formula
    assert {
        "texture_risk_lf=texture_signal",
        "texture_risk_tail=1-texture_signal",
        "texture_risk_attention=risk_neutral_texture_value",
    }.issubset(risk)

    for required_fragment in (
        "heads=explicit_positive_integer_attention_module_heads",
        "recorder_hidden_states=required_tensor_from_each_frozen_layer_hook",
        "probability_consumer_input=QKAttentionRelation",
        "qk_record_identity=(outer_layer_name==relation.metadata.record_layer_name)",
        "multilayer_qk_identity=unique",
        "keyed_sign=",
        "component_polarity=",
        "stable_selection=",
        "pair_weight=",
        "pair_weighted_RowCorr",
        "mean_layers",
        "s_A=",
    ):
        assert any(required_fragment in expression for expression in qk)

    assert any("right_upper_triangular_solve" in expression for expression in null_space)
    assert all("inverse(R)" not in expression for expression in null_space)
    assert all("method_numeric_epsilon" not in expression for expression in finite)
    assert any("null_space_numerical_epsilon" in expression for expression in finite)

    for required_fragment in (
        "sum_in_fixed_order",
        "common_scale_k=",
        "content_base_k=",
        "final_latent_k=",
        "attention_final_qk_revalidation=",
    ):
        assert any(required_fragment in expression for expression in actual_write)


@pytest.mark.constraint
def test_registry_freezes_fail_closed_model_and_qk_inputs() -> None:
    """CLIP、VAE 与 Q/K 核心输入不得通过兼容默认值或静默漏记补齐。"""

    invariants = _invariants_by_id(load_method_semantic_registry(ROOT))
    frozen_model = invariants["frozen_model_operator_identity"]
    complete_feature = invariants["complete_716_feature_jacobian"]
    direct_qk = invariants["direct_qk_four_component_relation"]

    assert {
        "vae_scaling_factor_missing",
        "vae_shift_factor_missing",
    }.issubset(frozen_model["fail_closed_conditions"])
    assert {
        "default_vae_scaling_factor_one",
        "default_vae_shift_factor_zero",
    }.issubset(frozen_model["forbidden_substitutes"])
    assert "clip_projected_image_embeds_missing" in complete_feature[
        "fail_closed_conditions"
    ]
    assert "clip_pooler_output_substitution" in complete_feature[
        "forbidden_substitutes"
    ]
    assert {
        "attention_heads_missing_or_invalid",
        "attention_hook_tensor_input_missing",
    }.issubset(direct_qk["fail_closed_conditions"])
    assert {
        "implicit_single_attention_head_default",
        "silent_qk_record_skip_on_missing_tensor_input",
    }.issubset(direct_qk["forbidden_substitutes"])
    assert {
        "qk_relation_type_missing",
        "qk_relation_source_mismatch",
        "qk_relation_operator_metadata_incomplete",
        "qk_relation_atom_content_incomplete",
    }.issubset(direct_qk["fail_closed_conditions"])
    assert {
        "bare_attention_probability_tensor",
        "non_direct_qk_relation_source",
        "qk_relation_without_operator_metadata",
        "qk_relation_without_atom_content_identity",
    }.issubset(direct_qk["forbidden_substitutes"])
    assert {
        "qk_outer_layer_name_mismatch",
        "qk_outer_token_indices_mismatch",
        "qk_multilayer_name_duplicate",
    }.issubset(direct_qk["fail_closed_conditions"])
    assert {
        "outer_qk_layer_renaming",
        "outer_qk_token_index_replacement",
        "same_internal_layer_clone_as_multilayer_evidence",
    }.issubset(direct_qk["forbidden_substitutes"])


@pytest.mark.constraint
def test_cpu_property_nodes_only_cover_independently_implemented_operators() -> None:
    """CPU 性质节点只能登记已有真实实现和独立正反例的算子。"""

    invariants = _invariants_by_id(load_method_semantic_registry(ROOT))

    assert all(item["specification_test_nodes"] for item in invariants.values())
    assert invariants["exact_jacobian_low_response_subspace"][
        "cpu_property_test_nodes"
    ]
    assert (
        "tests/functional/test_real_scientific_operators.py::"
        "test_exact_jacobian_linearization_satisfies_adjoint_identity"
        in invariants["exact_jacobian_low_response_subspace"][
            "cpu_property_test_nodes"
        ]
    )
    cpu_verified_invariants = {
        "branch_signal_origin",
        "branch_risk_bounds_written_update",
            "exact_jacobian_low_response_subspace",
            "versioned_key_prg_reconstruction",
            "spatial_low_pass_and_amplitude_tail_carriers",
        "direct_qk_monotonic_attention_update",
        "three_branch_update_composition",
        "actual_dtype_write_revalidation",
        "scientific_content_binding",
    }
    assert all(invariants[invariant_id]["cpu_property_test_nodes"] for invariant_id in cpu_verified_invariants)
    assert all(
        item["cpu_property_test_nodes"] == []
        for invariant_id, item in invariants.items()
        if invariant_id not in cpu_verified_invariants
    )


@pytest.mark.constraint
def test_authority_document_freezes_risk_and_null_space_counterexamples() -> None:
    """权威文档必须冻结最终写回包络与 QR 独立列参考."""

    text = INVARIANT_DOCUMENT.read_text(encoding="utf-8")

    assert r"|\Delta z_t^b(i)|\le E_b(i)" in text
    assert "禁止用当前样本的 `max(budget)`" in text
    assert r"\widetilde V_bR_b=V_b" in text
    assert "不得使用跨列共享 RMS" in text
    assert "SHA-256 本身不能重建 Tensor" in text
    assert "从磁盘重新读取更新 JSONL、检测 JSONL 和最终图像" in text
    assert "规范 RGB uint8 像素摘要" in text
    assert "不能证明外部数据来源真实" in text
    assert "禁止每进入一条 detection 就归零" in text
    assert "从数据集顶层 manifest" in text
    assert "读取 `runtime_results.jsonl` 中的配置摘要和紧凑随机化引用" in text
    assert "全部 JSONL、图像、结果记录和单元 manifest 叶子" in text
    assert "写出单元产物后和数据集打包前" in text
    assert "残留 attention 分数、更新、关系、pair 身份" in text
    assert "`range(3, 3+n)`" in text
    assert "`alignment_digest` 必须是完整 alignment 记录" in text
    assert "不允许用1或0补齐" in text
    assert "不允许用投影前 `pooler_output`" in text
    assert "必须显式公开非 bool 的正整数 `heads`" in text
    assert "没有 Tensor 输入时必须立即失败" in text
    assert "都必须接收 `QKAttentionRelation`" in text
    assert "裸概率 Tensor、错误 `relation_source`" in text
    assert "`layer_name=relation.metadata.record_layer_name`" in text
    assert "多层记录的层名必须唯一" in text
    assert "复制同一内部层冒充多层" in text
    assert "单元 manifest 自身" in text
    assert "配置摘要或随机化引用漂移" in text
    assert "重算每个内嵌 `scientific_content_binding_record`" in text
    assert "最终三图 Q/K 不能只记录 Q/K Tensor" in text
    assert "图像像素摘要、Q/K 原子摘要、公开噪声内容摘要" in text
    assert "最终三图与检测分别构造两套公开噪声" in text


@pytest.mark.constraint
def test_traceability_is_not_described_as_method_completion() -> None:
    """追踪通过不得被描述为 CPU、GPU 或论文证据完成."""

    text = INVARIANT_DOCUMENT.read_text(encoding="utf-8")

    assert "不允许自行声明任何不变量已经通过" in text
    assert "单个局部测试通过都不能单独证明方法完成" in text
    assert "在全部不变量达到 `cpu_verified` 前不得开始正式论文结果生产" in text
