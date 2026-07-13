"""解析并验证核心方法语义不变量追踪登记表."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

import yaml

from tools.harness.lib.field_rules import load_field_registry


REGISTRY_RELATIVE_PATH = PurePosixPath("configs/method_semantic_registry.json")
METHOD_CONFIG_RELATIVE_PATH = PurePosixPath("configs/model_sd35.yaml")
REGISTRY_SCHEMA = "slm_wm_method_semantic_registry_v1"
REGISTRY_SCOPE = "normative_traceability_without_scientific_conformance_decision"
EXPECTED_NORMATIVE_TRACE_DIGEST = (
    "6dd763343d71c3fc6803c9bfad60ff6ad2159d139a75682f2b3223c67112419c"
)
EXPECTED_INVARIANT_IDS = (
    "constructive_local_tangent_scope",
    "frozen_model_operator_identity",
    "branch_signal_origin",
    "branch_risk_bounds_written_update",
    "complete_716_feature_jacobian",
    "exact_jacobian_low_response_subspace",
    "versioned_key_prg_reconstruction",
    "spatial_low_pass_and_amplitude_tail_carriers",
    "direct_qk_four_component_relation",
    "direct_qk_monotonic_attention_update",
    "three_branch_update_composition",
    "actual_dtype_write_revalidation",
    "finite_feature_preservation",
    "final_image_attention_attribution",
    "image_only_detection_boundary",
    "same_threshold_geometry_rescue",
    "scientific_content_binding",
)
EXPECTED_CPU_PROPERTY_IDS = {
    "constructive_local_tangent_scope": (
        "local_level_set_scope_is_constructive_and_not_global_manifold"
    ),
    "frozen_model_operator_identity": (
        "exact_model_revisions_and_runtime_operator_classes_define_the_scientific_operator"
    ),
    "branch_signal_origin": (
        "branch_signals_use_frozen_analytic_ranges_and_real_adjacent_or_qk_sources"
    ),
    "branch_risk_bounds_written_update": (
        "risk_budget_monotonically_bounds_actual_dtype_write"
    ),
    "exact_jacobian_low_response_subspace": (
        "exact_jvp_vjp_projection_has_independent_column_residuals"
    ),
    "complete_716_feature_jacobian": (
        "jacobian_uses_all_512_clip_and_204_declared_structure_coordinates_without_compression"
    ),
    "spatial_low_pass_and_amplitude_tail_carriers": (
        "lf_is_two_dimensional_low_pass_and_tail_is_amplitude_order_statistic"
    ),
    "direct_qk_monotonic_attention_update": (
        "direct_to_q_to_k_probabilities_and_backtracking_are_formula_exact"
    ),
    "direct_qk_four_component_relation": (
        "direct_qk_subgraph_uses_the_frozen_four_component_relation_formula"
    ),
    "actual_dtype_write_revalidation": (
        "cast_sum_minus_original_is_the_only_write_gate_input"
    ),
    "finite_feature_preservation": (
        "actual_written_update_and_final_images_preserve_the_frozen_feature_coordinates"
    ),
    "final_image_attention_attribution": (
        "final_images_attribute_attention_by_a_shared_identity_carrier_only_counterfactual"
    ),
    "image_only_detection_boundary": (
        "detector_signature_and_runtime_reject_generation_private_state"
    ),
    "versioned_key_prg_reconstruction": (
        "sha256_counter_normal_icdf_table_bytes_are_platform_independent"
    ),
    "three_branch_update_composition": (
        "branch_updates_compose_once_and_preserve_joint_bounds"
    ),
    "same_threshold_geometry_rescue": (
        "geometry_rescue_reuses_the_calibrated_content_threshold_and_cannot_vote_positive_alone"
    ),
    "scientific_content_binding": (
        "all_scientific_tensors_and_ordered_roles_bind_dtype_shape_and_raw_bytes"
    ),
}
REQUIRED_INVARIANT_FIELDS = frozenset(
    {
        "invariant_id",
        "definition_pointer",
        "formal_expression",
        "configuration_fields",
        "method_implementation_symbols",
        "runtime_binding_symbols",
        "runtime_evidence_fields",
        "fail_closed_conditions",
        "forbidden_substitutes",
        "cpu_property_id",
        "specification_test_nodes",
        "cpu_property_test_nodes",
        "gpu_atomic_roles",
        "gpu_observation_requirement",
        "claim_boundary",
    }
)
FORBIDDEN_SELF_ASSERTION_KEYS = frozenset(
    {
        "pass",
        "passed",
        "ready",
        "verified",
        "decision",
        "status",
        "supports_method_claim",
        "supports_paper_claim",
        "cpu_verified",
        "gpu_verified",
        "evidence_closed",
    }
)
_FORBIDDEN_SELF_ASSERTION_PATTERN = re.compile(
    r"(?:^|_)(?:pass|passed|ready|verified|decision|status)(?:$|_)"
    r"|^supports_.*claim$"
)
_SNAKE_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_CONFIG_DOT_PATH_PATTERN = re.compile(
    r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*(?:\.[a-z][a-z0-9]*(?:_[a-z0-9]+)*)*$"
)
_TEST_NODE_PATTERN = re.compile(
    r"^(tests/[a-z0-9_/]*test_[a-z0-9_]+\.py)::(test_[a-z0-9_]+)$"
)
_TRACE_CONTRACT_TEST_PATH = (
    "tests/constraints/test_method_semantic_registry_contract.py"
)
_NORMATIVE_TRACE_FIELDS = (
    "invariant_id",
    "definition_pointer",
    "formal_expression",
    "configuration_fields",
    "runtime_evidence_fields",
    "fail_closed_conditions",
    "forbidden_substitutes",
    "gpu_atomic_roles",
    "gpu_observation_requirement",
    "claim_boundary",
)


def _bindings(*values: tuple[str, str]) -> tuple[tuple[str, str], ...]:
    """构造冻结的实现绑定集合."""

    return tuple(values)


EXPECTED_METHOD_IMPLEMENTATION_SYMBOLS = {
    "constructive_local_tangent_scope": _bindings(
        (
            "main/methods/method_definition.py",
            "semantic_conditioned_latent_method_definition",
        ),
    ),
    "frozen_model_operator_identity": _bindings(
        (
            "main/methods/method_definition.py",
            "semantic_conditioned_latent_method_definition",
        ),
    ),
    # 风险输入和完整特征仍由真实模型绑定层构造, 当前没有核心纯算子入口。
    "branch_signal_origin": (),
    "branch_risk_bounds_written_update": _bindings(
        ("main/methods/semantic/branch_risk.py", "build_branch_risk_fields"),
        (
            "main/methods/subspace/jacobian_nullspace.py",
            "solve_jacobian_null_space",
        ),
        (
            "main/methods/update_composition.py",
            "build_risk_bounded_update",
        ),
    ),
    "exact_jacobian_low_response_subspace": _bindings(
        (
            "main/methods/subspace/jacobian_nullspace.py",
            "build_exact_jacobian_linearization",
        ),
        (
            "main/methods/subspace/jacobian_nullspace.py",
            "solve_jacobian_null_space",
        ),
        (
            "main/methods/subspace/jacobian_nullspace.py",
            "recompute_jacobian_null_space_result_digest",
        ),
    ),
    "complete_716_feature_jacobian": (),
    "spatial_low_pass_and_amplitude_tail_carriers": _bindings(
        ("main/methods/carrier/keyed_tensor.py", "build_low_frequency_template"),
        ("main/methods/carrier/keyed_tensor.py", "LowFrequencyCarrierConfig"),
        (
            "main/methods/carrier/keyed_tensor.py",
            "validate_low_frequency_carrier_protocol_record",
        ),
        ("main/methods/carrier/keyed_tensor.py", "build_tail_robust_template"),
        (
            "main/methods/carrier/keyed_tensor.py",
            "validate_tail_robust_carrier_protocol_record",
        ),
        ("main/methods/carrier/keyed_tensor.py", "project_canonical_template"),
        ("main/methods/detection/image_only.py", "detect_image_only_watermark"),
    ),
    "direct_qk_four_component_relation": _bindings(
        (
            "main/methods/geometry/differentiable_attention.py",
            "qk_self_attention",
        ),
        (
            "main/methods/geometry/differentiable_attention.py",
            "build_attention_relation_descriptor",
        ),
        (
            "main/methods/geometry/differentiable_attention.py",
            "attention_relation_component_protocol",
        ),
        (
            "main/methods/geometry/differentiable_attention.py",
            "qk_operator_metadata_records_ready",
        ),
        (
            "main/methods/geometry/differentiable_attention.py",
            "qk_operator_metadata_records_digest",
        ),
    ),
    "direct_qk_monotonic_attention_update": _bindings(
        (
            "main/methods/geometry/differentiable_attention.py",
            "qk_self_attention",
        ),
        (
            "main/methods/geometry/differentiable_attention.py",
            "optimize_attention_geometry_update",
        ),
        (
            "main/methods/geometry/differentiable_attention.py",
            "DifferentiableAttentionRecorder",
        ),
    ),
    "actual_dtype_write_revalidation": _bindings(
        (
            "main/methods/update_composition.py",
            "build_quantized_composition_candidate",
        ),
        (
            "main/methods/update_composition.py",
            "iter_quantized_composition_candidates",
        ),
        (
            "main/methods/update_composition.py",
            "compose_ordered_float32_update_once",
        ),
    ),
    "finite_feature_preservation": (),
    "final_image_attention_attribution": (),
    "image_only_detection_boundary": _bindings(
        (
            "main/methods/geometry/attention_alignment.py",
            "attention_alignment_gate_record",
        ),
        (
            "main/methods/geometry/attention_alignment.py",
            "recover_attention_affine_alignment",
        ),
        (
            "main/methods/geometry/attention_alignment.py",
            "resample_attention_aligned_rgb_uint8",
        ),
        (
            "main/methods/detection/image_only.py",
            "select_image_only_alignment_candidate",
        ),
        ("main/methods/detection/image_only.py", "detect_image_only_watermark"),
    ),
    "versioned_key_prg_reconstruction": _bindings(
        ("main/core/keyed_prg.py", "build_keyed_gaussian_tensor"),
        ("main/core/keyed_prg.py", "build_keyed_uniform_tensor"),
        ("main/core/keyed_prg.py", "keyed_prg_protocol_record"),
        (
            "main/core/normal_quantile_table.py",
            "standard_normal_quantile_float32_table",
        ),
    ),
    "three_branch_update_composition": _bindings(
        (
            "main/methods/update_composition.py",
            "build_quantized_composition_candidate",
        ),
        (
            "main/methods/update_composition.py",
            "iter_quantized_composition_candidates",
        ),
        (
            "main/methods/update_composition.py",
            "compose_ordered_float32_update_once",
        ),
    ),
    "same_threshold_geometry_rescue": _bindings(
        (
            "main/methods/geometry/attention_alignment.py",
            "attention_alignment_gate_record",
        ),
        (
            "main/methods/geometry/attention_alignment.py",
            "recover_attention_affine_alignment",
        ),
        (
            "main/methods/geometry/attention_alignment.py",
            "resample_attention_aligned_rgb_uint8",
        ),
        (
            "main/methods/detection/image_only.py",
            "select_image_only_alignment_candidate",
        ),
        ("main/methods/detection/image_only.py", "detect_image_only_watermark"),
    ),
    "scientific_content_binding": _bindings(
        ("main/core/digest.py", "tensor_content_sha256"),
        ("main/core/digest.py", "build_stable_digest"),
    ),
}
EXPECTED_RUNTIME_BINDING_SYMBOLS = {
    invariant_id: _bindings(
        (
            "experiments/runners/semantic_watermark_runtime.py",
            "run_semantic_watermark_runtime",
        ),
    )
    for invariant_id in EXPECTED_INVARIANT_IDS
}
EXPECTED_RUNTIME_BINDING_SYMBOLS.update(
    {
        "frozen_model_operator_identity": _bindings(
            (
                "experiments/runtime/model_sources.py",
                "require_registered_model_reference",
            ),
            (
                "experiments/runtime/diffusion/sd3_pipeline_runtime.py",
                "load_pipeline",
            ),
            (
                "experiments/runners/semantic_watermark_runtime.py",
                "load_semantic_watermark_runtime_context",
            ),
        ),
        "branch_signal_origin": _bindings(
            (
                "experiments/runtime/diffusion/semantic_features.py",
                "DifferentiableSemanticFeatureRuntime",
            ),
            (
                "experiments/runners/semantic_watermark_runtime.py",
                "run_semantic_watermark_runtime",
            ),
        ),
        "complete_716_feature_jacobian": _bindings(
            (
                "experiments/runtime/diffusion/semantic_features.py",
                "DifferentiableSemanticFeatureRuntime",
            ),
            (
                "experiments/runners/semantic_watermark_runtime.py",
                "run_semantic_watermark_runtime",
            ),
        ),
        "actual_dtype_write_revalidation": _bindings(
            (
                "experiments/runners/semantic_watermark_runtime.py",
                "_quantized_write_jacobian_response_record",
            ),
        ),
        "spatial_low_pass_and_amplitude_tail_carriers": _bindings(
            (
                "experiments/runners/semantic_watermark_runtime.py",
                "run_semantic_watermark_runtime",
            ),
            (
                "experiments/runners/image_only_dataset_runtime.py",
                "validate_detection_content_carrier_protocol",
            ),
            (
                "experiments/protocol/detection_key_identity.py",
                "validate_detection_key_identity_record",
            ),
        ),
        "image_only_detection_boundary": _bindings(
            (
                "experiments/runners/semantic_watermark_runtime.py",
                "run_semantic_watermark_runtime",
            ),
            (
                "experiments/runners/image_only_dataset_runtime.py",
                "run_image_only_dataset_runtime",
            ),
        ),
        "finite_feature_preservation": _bindings(
            (
                "experiments/runners/semantic_watermark_runtime.py",
                "_combined_update_preservation_record",
            ),
            (
                "experiments/runners/semantic_watermark_runtime.py",
                "_final_image_preservation_record",
            ),
        ),
        "final_image_attention_attribution": _bindings(
            (
                "experiments/runners/semantic_watermark_runtime.py",
                "_final_image_attention_observability_record",
            ),
        ),
        "same_threshold_geometry_rescue": _bindings(
            ("experiments/protocol/calibration.py", "empirical_threshold_at_fpr"),
            (
                "experiments/runners/image_only_dataset_runtime.py",
                "run_image_only_dataset_runtime",
            ),
        ),
        "scientific_content_binding": _bindings(
            (
                "experiments/runtime/scientific_content_binding.py",
                "build_scientific_content_binding_record",
            ),
            (
                "experiments/runtime/scientific_content_binding.py",
                "read_canonical_rgb_uint8_content_record",
            ),
            (
                "experiments/runtime/scientific_content_binding.py",
                "recompute_scientific_content_binding_digest",
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
                "_scientific_content_binding_record_ready",
            ),
            (
                "experiments/runners/image_only_dataset_runtime.py",
                "run_image_only_dataset_runtime",
            ),
            (
                "experiments/runners/image_only_dataset_runtime.py",
                "package_image_only_dataset_runtime",
            ),
        ),
        "versioned_key_prg_reconstruction": _bindings(
            (
                "experiments/protocol/formal_randomization.py",
                "build_canonical_sd35_base_latent",
            ),
            (
                "experiments/runners/semantic_watermark_runtime.py",
                "_public_detection_noise_tensor",
            ),
        ),
    }
)
EXPECTED_SPECIFICATION_TEST_NODES = {
    "constructive_local_tangent_scope": (
        "tests/constraints/test_method_definition_contract.py::"
        "test_machine_readable_method_definition_freezes_constructive_semantics",
    ),
    "frozen_model_operator_identity": (
        "tests/functional/test_model_source_registry.py::"
        "test_primary_model_config_matches_immutable_source_registry",
        "tests/functional/test_model_source_registry.py::"
        "test_sd35_pipeline_forwards_registered_revision",
        "tests/functional/test_model_source_registry.py::"
        "test_clip_loader_forwards_registered_revision",
        "tests/functional/test_semantic_feature_conditions.py::"
        "test_latent_decoder_requires_explicit_vae_scaling_and_shift",
    ),
    "branch_signal_origin": (
        "tests/functional/test_semantic_feature_conditions.py::"
        "test_branch_signals_use_actual_adjacent_step_and_local_contrast",
        "tests/functional/test_real_scientific_operators.py::"
        "test_attention_stability_comes_from_multiple_real_qk_layers",
        "tests/functional/test_real_scientific_operators.py::"
        "test_attention_stability_and_selection_require_direct_qk_identity",
        "tests/functional/test_real_scientific_operators.py::"
        "test_attention_records_bind_outer_layer_and_token_identity",
        "tests/functional/test_real_scientific_operators.py::"
        "test_attention_stability_rejects_duplicate_layer_identity",
    ),
    "branch_risk_bounds_written_update": (
        "tests/functional/test_real_scientific_operators.py::"
        "test_branch_risk_fields_use_opposite_texture_preferences",
        "tests/functional/test_real_scientific_operators.py::"
        "test_risk_budget_is_explicit_in_full_jacobian_null_projection",
        "tests/functional/test_branch_risk_fail_closed.py::"
        "test_branch_risk_rejects_empty_frozen_eligibility_set",
    ),
    "exact_jacobian_low_response_subspace": (
        "tests/functional/test_real_scientific_operators.py::"
        "test_full_jacobian_constraint_projection_recovers_null_direction",
        "tests/functional/test_real_scientific_operators.py::"
        "test_null_projection_energy_retention_uses_squared_l2_ratio",
        "tests/functional/test_real_scientific_operators.py::"
        "test_undamped_psd_cg_reports_non_convergence_without_fallback",
        "tests/functional/test_real_scientific_operators.py::"
        "test_scientific_operator_gate_requires_all_real_operator_evidence",
    ),
    "complete_716_feature_jacobian": (
        "tests/functional/test_semantic_feature_conditions.py::"
        "test_formal_jacobian_keeps_clip_and_handcrafted_structure_coordinates",
        "tests/functional/test_semantic_feature_conditions.py::"
        "test_handcrafted_structure_vector_preserves_declared_coordinates",
        "tests/functional/test_semantic_feature_conditions.py::"
        "test_complete_feature_vector_supports_exact_jvp_and_vjp",
        "tests/functional/test_semantic_feature_conditions.py::"
        "test_semantic_features_require_projected_clip_image_embedding",
    ),
    "spatial_low_pass_and_amplitude_tail_carriers": (
        "tests/functional/test_real_scientific_operators.py::"
        "test_tail_robust_template_records_amplitude_tail_semantics",
        "tests/functional/test_real_scientific_operators.py::"
        "test_keyed_templates_use_versioned_device_independent_prg",
    ),
    "direct_qk_four_component_relation": (
        "tests/functional/test_real_scientific_operators.py::"
        "test_multihead_qk_relation_matches_independent_manual_calculation",
        "tests/functional/test_real_scientific_operators.py::"
        "test_each_attention_relation_component_changes_keyed_score",
        "tests/functional/test_real_scientific_operators.py::"
        "test_distance_modulated_probability_is_distinct_and_differentiable",
        "tests/functional/test_real_scientific_operators.py::"
        "test_image_alignment_uses_token_endpoint_coordinate_convention",
        "tests/functional/test_real_scientific_operators.py::"
        "test_scientific_operator_gate_requires_all_real_operator_evidence",
        "tests/functional/test_real_scientific_operators.py::"
        "test_qk_relation_requires_explicit_positive_integer_head_count",
        "tests/functional/test_real_scientific_operators.py::"
        "test_attention_recorder_rejects_missing_hidden_state_tensor",
        "tests/functional/test_real_scientific_operators.py::"
        "test_attention_stability_and_selection_require_direct_qk_identity",
        "tests/functional/test_real_scientific_operators.py::"
        "test_attention_records_bind_outer_layer_and_token_identity",
        "tests/functional/test_real_scientific_operators.py::"
        "test_attention_stability_rejects_duplicate_layer_identity",
    ),
    "direct_qk_monotonic_attention_update": (
        "tests/functional/test_real_scientific_operators.py::"
        "test_multihead_qk_relation_matches_independent_manual_calculation",
        "tests/functional/test_real_scientific_operators.py::"
        "test_attention_update_uses_real_qk_and_autograd",
        "tests/functional/test_real_scientific_operators.py::"
        "test_attention_update_verifies_actual_combined_latent",
        "tests/functional/test_real_scientific_operators.py::"
        "test_attention_modules_resolve_only_frozen_exact_layer_names",
        "tests/functional/test_real_scientific_operators.py::"
        "test_qk_relation_requires_explicit_positive_integer_head_count",
        "tests/functional/test_real_scientific_operators.py::"
        "test_attention_recorder_rejects_missing_hidden_state_tensor",
        "tests/functional/test_real_scientific_operators.py::"
        "test_attention_stability_and_selection_require_direct_qk_identity",
        "tests/functional/test_real_scientific_operators.py::"
        "test_attention_records_bind_outer_layer_and_token_identity",
        "tests/functional/test_real_scientific_operators.py::"
        "test_attention_stability_rejects_duplicate_layer_identity",
    ),
    "actual_dtype_write_revalidation": (
        "tests/functional/test_semantic_feature_conditions.py::"
        "test_quantized_write_jacobian_gate_rechecks_actual_float16_delta",
        "tests/functional/test_semantic_feature_conditions.py::"
        "test_quantized_write_jacobian_gate_rejects_update_lost_to_quantization",
    ),
    "finite_feature_preservation": (
        "tests/functional/test_semantic_feature_conditions.py::"
        "test_actual_combined_latent_uses_full_feature_preservation_gate",
        "tests/functional/test_semantic_feature_conditions.py::"
        "test_final_image_gate_checks_cumulative_clean_to_watermarked_drift",
    ),
    "final_image_attention_attribution": (
        "tests/functional/test_semantic_feature_conditions.py::"
        "test_final_image_attention_gate_uses_reencoded_real_qk_scores",
        "tests/functional/test_semantic_feature_conditions.py::"
        "test_attention_gate_rejects_blind_only_gain_without_frozen_pair_gain",
        "tests/functional/test_semantic_feature_conditions.py::"
        "test_carrier_only_counterfactual_binds_same_seed_and_scheduler",
    ),
    "image_only_detection_boundary": (
        "tests/functional/test_real_scientific_operators.py::"
        "test_image_only_detector_interface_and_positive_content_path",
        "tests/functional/test_attention_affine_protocol_geometry.py::"
        "test_cross_layer_alignment_selection_uses_frozen_lexicographic_rule",
        "tests/functional/test_real_scientific_operators.py::"
        "test_image_alignment_quantizes_fractional_rgb_with_floor",
        "tests/functional/test_attack_matrix.py::"
        "test_attack_matrix_rejects_non_blind_detection_records",
        "tests/functional/test_model_source_registry.py::"
        "test_runtime_detector_config_consumes_formal_alignment_gate",
    ),
    "versioned_key_prg_reconstruction": (
        "tests/functional/test_real_scientific_operators.py::"
        "test_keyed_templates_use_versioned_device_independent_prg",
    ),
    "three_branch_update_composition": (
        "tests/functional/test_real_scientific_operators.py::"
        "test_attention_update_verifies_actual_combined_latent",
        "tests/functional/test_semantic_feature_conditions.py::"
        "test_actual_combined_latent_uses_full_feature_preservation_gate",
    ),
    "same_threshold_geometry_rescue": (
        "tests/functional/test_real_scientific_operators.py::"
        "test_complete_evidence_calibration_includes_geometry_rescue",
        "tests/functional/test_attention_affine_protocol_geometry.py::"
        "test_cross_layer_alignment_selection_uses_frozen_lexicographic_rule",
        "tests/functional/test_real_scientific_operators.py::"
        "test_image_alignment_quantizes_fractional_rgb_with_floor",
        "tests/functional/test_real_scientific_operators.py::"
        "test_frozen_protocol_recomputes_threshold_dependent_failure_reason",
        "tests/functional/test_fixed_fpr_threshold_audit.py::"
        "test_main_method_threshold_audit_recomputes_complete_rescue_protocol",
        "tests/functional/test_real_scientific_operators.py::"
        "test_frozen_evidence_protocol_rejects_alignment_gate_drift",
    ),
    "scientific_content_binding": (
        "tests/functional/test_semantic_feature_conditions.py::"
        "test_tensor_content_sha256_binds_dtype_shape_and_raw_bytes",
        "tests/functional/test_real_scientific_operators.py::"
        "test_scientific_operator_gate_requires_all_real_operator_evidence",
        "tests/functional/test_scientific_content_binding.py::"
        "test_scientific_content_binding_digest_is_recomputable",
        "tests/functional/test_scientific_content_binding.py::"
        "test_detection_public_noise_uses_shared_global_evaluation_indices",
        "tests/functional/test_gpu_upstream_package_producers.py::"
        "test_primary_gpu_package_producers_pass_strict_closure_contract",
        "tests/functional/test_gpu_upstream_package_producers.py::"
        "test_image_runtime_package_requires_rebuilt_unit_content_binding",
        "tests/functional/test_scientific_content_binding.py::"
        "test_carrier_only_artifact_validator_accepts_persisted_config_payload",
        "tests/functional/test_scientific_content_binding.py::"
        "test_scientific_content_binding_artifact_validator_rejects_tampering",
    ),
}
EXPECTED_CPU_PROPERTY_TEST_NODES = {
    invariant_id: () for invariant_id in EXPECTED_INVARIANT_IDS
}
EXPECTED_CPU_PROPERTY_TEST_NODES["versioned_key_prg_reconstruction"] = (
    "tests/functional/test_normal_quantile_table.py::"
    "test_frozen_normal_quantile_table_has_exact_distribution_contract",
)
EXPECTED_CPU_PROPERTY_TEST_NODES["exact_jacobian_low_response_subspace"] = (
    "tests/functional/test_semantic_feature_conditions.py::"
    "test_complete_feature_vector_supports_exact_jvp_and_vjp",
    "tests/functional/test_real_scientific_operators.py::"
    "test_null_projection_energy_retention_uses_squared_l2_ratio",
    "tests/functional/test_real_scientific_operators.py::"
    "test_qr_basis_uses_independent_routed_candidate_references",
    "tests/functional/test_real_scientific_operators.py::"
    "test_exact_jacobian_linearization_satisfies_adjoint_identity",
)
EXPECTED_CPU_PROPERTY_TEST_NODES["branch_signal_origin"] = (
    "tests/functional/test_branch_risk_formula.py::"
    "test_constant_semantic_maps_keep_analytic_values_without_batch_mixing",
    "tests/functional/test_branch_risk_formula.py::"
    "test_image_risk_signals_match_analytic_texture_contrast_and_adjacent_formulas",
    "tests/functional/test_real_scientific_operators.py::"
    "test_attention_stability_comes_from_multiple_real_qk_layers",
    "tests/functional/test_real_scientific_operators.py::"
    "test_attention_stability_and_selection_require_direct_qk_identity",
    "tests/functional/test_real_scientific_operators.py::"
    "test_attention_records_bind_outer_layer_and_token_identity",
    "tests/functional/test_real_scientific_operators.py::"
    "test_attention_stability_rejects_duplicate_layer_identity",
)
EXPECTED_CPU_PROPERTY_TEST_NODES["branch_risk_bounds_written_update"] = (
    "tests/functional/test_branch_risk_formula.py::"
    "test_branch_risk_formula_uses_frozen_texture_directions_and_neutral_value",
    "tests/functional/test_branch_risk_formula.py::"
    "test_branch_risk_threshold_equality_is_ineligible",
    "tests/functional/test_risk_bounded_composition.py::"
    "test_risk_budget_monotonically_bounds_update_and_zero_support",
    "tests/functional/test_risk_bounded_composition.py::"
    "test_risk_bounded_update_is_batch_isolated_and_reuses_nchw_budget",
    "tests/functional/test_risk_bounded_composition.py::"
    "test_risk_bounded_update_separates_direction_and_numerical_epsilon",
    "tests/functional/test_risk_bounded_composition.py::"
    "test_risk_bounded_direction_support_is_invariant_to_input_scaling",
    "tests/functional/test_risk_bounded_composition.py::"
    "test_direction_epsilon_only_clears_zero_budget_leakage",
    "tests/functional/test_real_scientific_operators.py::"
    "test_post_risk_direction_reexecutes_independent_exact_jvp",
)
EXPECTED_CPU_PROPERTY_TEST_NODES[
    "spatial_low_pass_and_amplitude_tail_carriers"
] = (
    "tests/functional/test_risk_bounded_composition.py::"
    "test_low_frequency_template_consumes_frozen_spatial_pooling_parameters",
    "tests/functional/test_risk_bounded_composition.py::"
    "test_tail_template_preserves_exact_sparse_support_without_centering",
    "tests/functional/test_risk_bounded_composition.py::"
    "test_carrier_templates_remain_canonical_float32_for_float16_latent",
)
EXPECTED_CPU_PROPERTY_TEST_NODES["direct_qk_monotonic_attention_update"] = (
    "tests/functional/test_attention_risk_step.py::"
    "test_attention_risk_step_starts_at_maximum_and_consumes_factor",
    "tests/functional/test_attention_risk_step.py::"
    "test_attention_risk_step_uses_one_actual_dtype_cast",
    "tests/functional/test_attention_risk_step.py::"
    "test_attention_risk_step_rejects_non_monotonic_candidates_at_step_limit",
    "tests/functional/test_attention_risk_step.py::"
    "test_attention_optimizer_consumes_exact_risk_bounded_direction",
    "tests/functional/test_attention_risk_step.py::"
    "test_attention_optimizer_rejects_different_risk_direction",
)
EXPECTED_CPU_PROPERTY_TEST_NODES["three_branch_update_composition"] = (
    "tests/functional/test_risk_bounded_composition.py::"
    "test_quantized_composition_accepts_ordered_nonempty_branch_subset",
    "tests/functional/test_risk_bounded_composition.py::"
    "test_quantized_overshoot_requires_common_backtracking_candidate",
    "tests/functional/test_risk_bounded_composition.py::"
    "test_quantized_composition_evidence_binds_real_three_branch_tensors",
    "tests/functional/test_risk_bounded_composition.py::"
    "test_float32_composition_casts_once_instead_of_associative_latent_writes",
    "tests/functional/test_risk_bounded_composition.py::"
    "test_float32_composition_uses_frozen_role_order_not_mapping_order",
)
EXPECTED_CPU_PROPERTY_TEST_NODES["actual_dtype_write_revalidation"] = (
    "tests/functional/test_risk_bounded_composition.py::"
    "test_quantized_overshoot_requires_common_backtracking_candidate",
    "tests/functional/test_risk_bounded_composition.py::"
    "test_quantized_composition_evidence_binds_real_three_branch_tensors",
    "tests/functional/test_risk_bounded_composition.py::"
    "test_quantized_composition_rejects_inconsistent_backtracking_trace",
    "tests/functional/test_semantic_feature_conditions.py::"
    "test_quantized_write_jacobian_gate_rechecks_actual_float16_delta",
    "tests/functional/test_semantic_feature_conditions.py::"
    "test_quantized_write_jacobian_gate_rejects_update_lost_to_quantization",
)
EXPECTED_CPU_PROPERTY_TEST_NODES["scientific_content_binding"] = (
    "tests/functional/test_scientific_content_binding.py::"
    "test_scientific_content_binding_digest_is_recomputable",
    "tests/functional/test_scientific_content_binding.py::"
    "test_scientific_content_binding_digest_rejects_leaf_tampering",
    "tests/functional/test_scientific_content_binding.py::"
    "test_scientific_content_binding_artifact_validator_rejects_tampering",
    "tests/functional/test_scientific_content_binding.py::"
    "test_detection_public_noise_uses_shared_global_evaluation_indices",
    "tests/functional/test_gpu_upstream_package_producers.py::"
    "test_primary_gpu_package_producers_pass_strict_closure_contract",
    "tests/functional/test_gpu_upstream_package_producers.py::"
    "test_image_runtime_package_requires_rebuilt_unit_content_binding",
    "tests/functional/test_scientific_content_binding.py::"
    "test_carrier_only_artifact_validator_accepts_persisted_config_payload",
)


def load_method_semantic_registry(root: str | Path) -> dict[str, Any]:
    """从固定路径读取方法语义登记表."""

    path = Path(root) / Path(REGISTRY_RELATIVE_PATH)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("方法语义登记表顶层必须是 JSON object")
    return payload


def load_method_semantic_configuration(root: str | Path) -> dict[str, Any]:
    """读取方法唯一 YAML 配置供 dot path 追踪校验使用."""

    path = Path(root) / Path(METHOD_CONFIG_RELATIVE_PATH)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("方法 YAML 配置顶层必须是 mapping")
    return payload


def method_semantic_normative_trace_digest(payload: Mapping[str, Any]) -> str:
    """计算不含实现进度与测试绑定的规范语义摘要."""

    invariants = payload.get("invariants", [])
    canonical = {
        "registry_schema": payload.get("registry_schema"),
        "registry_scope": payload.get("registry_scope"),
        "invariants": [
            {field_name: item.get(field_name) for field_name in _NORMATIVE_TRACE_FIELDS}
            for item in invariants
            if isinstance(item, Mapping)
        ],
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _walk_keys(value: Any) -> tuple[str, ...]:
    """递归收集 JSON object 的全部字段名."""

    if isinstance(value, Mapping):
        nested = [str(key) for key in value]
        for item in value.values():
            nested.extend(_walk_keys(item))
        return tuple(nested)
    if isinstance(value, list):
        nested: list[str] = []
        for item in value:
            nested.extend(_walk_keys(item))
        return tuple(nested)
    return ()


def _repository_file(
    root: Path,
    path_text: str,
    *,
    required_root: str | None = None,
) -> Path | None:
    """解析仓库内规范路径并拒绝绝对路径与目录逃逸."""

    if not path_text or "\\" in path_text or ":" in path_text:
        return None
    pure_path = PurePosixPath(path_text)
    if pure_path.is_absolute() or any(part in {"", ".", ".."} for part in pure_path.parts):
        return None
    if required_root is not None and (
        not pure_path.parts or pure_path.parts[0] != required_root
    ):
        return None
    root_resolved = root.resolve()
    candidate = (root_resolved / Path(*pure_path.parts)).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        return None
    return candidate


def _top_level_symbols(path: Path) -> frozenset[str]:
    """读取 Python 文件中的顶层函数和类符号."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return frozenset(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def _validate_string_sequence(
    value: Any,
    *,
    invariant_id: str,
    field_name: str,
    add: Any,
    snake_case: bool,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """验证有序字符串集合并返回规范 tuple."""

    if not isinstance(value, list) or (not value and not allow_empty):
        add(field_name, f"{invariant_id} 的 {field_name} 必须是非空 JSON array")
        return ()
    if any(not isinstance(item, str) or not item.strip() for item in value):
        add(field_name, f"{invariant_id} 的 {field_name} 必须只包含非空字符串")
        return ()
    items = tuple(value)
    if len(items) != len(set(items)):
        add(field_name, f"{invariant_id} 的 {field_name} 不得重复")
    if snake_case:
        invalid = [item for item in items if not _SNAKE_CASE_PATTERN.fullmatch(item)]
        if invalid:
            add(field_name, f"{invariant_id} 的 {field_name} 包含无效标识")
    return items


def _configuration_value(
    configuration: Mapping[str, Any],
    dot_path: str,
) -> tuple[bool, Any]:
    """按完整 dot path 读取嵌套方法配置值."""

    current: Any = configuration
    for segment in dot_path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return False, None
        current = current[segment]
    return True, current


def _validate_configuration_fields(
    value: Any,
    *,
    invariant_id: str,
    configuration: Mapping[str, Any],
    add: Any,
) -> None:
    """验证配置字段使用 YAML 中真实存在的完整 dot path."""

    fields = _validate_string_sequence(
        value,
        invariant_id=invariant_id,
        field_name="configuration_fields",
        add=add,
        snake_case=False,
    )
    for field_path in fields:
        if not _CONFIG_DOT_PATH_PATTERN.fullmatch(field_path):
            add(
                "configuration_fields",
                f"{invariant_id} 的配置字段不是精确 dot path: {field_path}",
            )
            continue
        resolved, _ = _configuration_value(configuration, field_path)
        if not resolved:
            add(
                "configuration_fields",
                f"{invariant_id} 引用了不存在的方法配置字段: {field_path}",
            )


def _validate_symbol_bindings(
    root: Path,
    value: Any,
    *,
    invariant_id: str,
    field_name: str,
    required_root: str,
    expected_bindings: tuple[tuple[str, str], ...],
    symbol_cache: dict[Path, frozenset[str]],
    add: Any,
) -> None:
    """验证实现绑定的层级、精确集合与真实顶层符号."""

    if not isinstance(value, list):
        add(field_name, f"{invariant_id} 的 {field_name} 必须是 JSON array")
        return
    resolved_bindings: list[tuple[str, str]] = []
    for binding in value:
        if not isinstance(binding, Mapping) or set(binding) != {"path", "symbol"}:
            add(field_name, f"{invariant_id} 的 {field_name} 绑定格式无效")
            continue
        path_text = str(binding["path"])
        symbol = str(binding["symbol"])
        source_path = _repository_file(root, path_text, required_root=required_root)
        if source_path is None or source_path.suffix != ".py":
            add(field_name, f"{invariant_id} 的 {field_name} 路径越界: {path_text}")
            continue
        resolved_bindings.append((PurePosixPath(path_text).as_posix(), symbol))
        if not source_path.is_file():
            add(field_name, f"{invariant_id} 的实现文件不存在: {path_text}")
            continue
        if source_path not in symbol_cache:
            symbol_cache[source_path] = _top_level_symbols(source_path)
        if symbol not in symbol_cache[source_path]:
            add(
                field_name,
                f"{invariant_id} 的实现符号不存在: {path_text}:{symbol}",
            )
    if tuple(resolved_bindings) != expected_bindings:
        add(field_name, f"{invariant_id} 的 {field_name} 与冻结追踪集合不一致")
    if len(resolved_bindings) != len(set(resolved_bindings)):
        add(field_name, f"{invariant_id} 的 {field_name} 不得重复")


def _validate_test_nodes(
    root: Path,
    value: Any,
    *,
    invariant_id: str,
    field_name: str,
    expected_nodes: tuple[str, ...],
    allow_empty: bool,
    add: Any,
    test_cache: dict[Path, frozenset[str]],
) -> None:
    """验证测试节点指向真实测试函数而不是登记表自测."""

    nodes = _validate_string_sequence(
        value,
        invariant_id=invariant_id,
        field_name=field_name,
        add=add,
        snake_case=False,
        allow_empty=allow_empty,
    )
    if nodes != expected_nodes:
        add(field_name, f"{invariant_id} 的 {field_name} 集合发生漂移")
    for node in nodes:
        match = _TEST_NODE_PATTERN.fullmatch(node)
        if match is None:
            add(field_name, f"{invariant_id} 的测试节点格式无效: {node}")
            continue
        path_text, function_name = match.groups()
        if path_text == _TRACE_CONTRACT_TEST_PATH:
            add(field_name, f"{invariant_id} 不得引用追踪登记自测")
            continue
        if field_name == "cpu_property_test_nodes" and not path_text.startswith(
            "tests/functional/"
        ):
            add(field_name, f"{invariant_id} 的 CPU 性质节点必须位于 functional 测试层")
            continue
        test_path = _repository_file(root, path_text, required_root="tests")
        if test_path is None or not test_path.is_file():
            add(field_name, f"{invariant_id} 的测试文件不存在: {path_text}")
            continue
        if test_path not in test_cache:
            test_cache[test_path] = _top_level_symbols(test_path)
        if function_name not in test_cache[test_path]:
            add(
                field_name,
                f"{invariant_id} 的测试函数不存在: {node}",
            )


def validate_method_semantic_registry(
    root: str | Path,
    payload: Mapping[str, Any],
    *,
    expected_method_definition_schema: str,
    expected_method_definition_digest: str,
) -> list[dict[str, str]]:
    """验证规范追踪完整性, 不输出任何科学性质通过结论."""

    root_path = Path(root)
    violations: list[dict[str, str]] = []

    def add(rule: str, message: str) -> None:
        violations.append({"rule": rule, "message": message})

    expected_top_level_fields = {
        "registry_schema",
        "registry_scope",
        "method_definition_schema",
        "method_definition_digest",
        "invariants",
    }
    if set(payload) != expected_top_level_fields:
        add("registry_fields", "方法语义登记表顶层字段集合不精确")
    if payload.get("registry_schema") != REGISTRY_SCHEMA:
        add("registry_schema", "方法语义登记表 schema 不匹配")
    if payload.get("registry_scope") != REGISTRY_SCOPE:
        add("registry_scope", "登记表必须明确限定为规范追踪而非科学结论")
    if payload.get("method_definition_schema") != expected_method_definition_schema:
        add("method_definition_schema", "登记表绑定的方法定义 schema 不匹配")
    if payload.get("method_definition_digest") != expected_method_definition_digest:
        add("method_definition_digest", "登记表绑定的方法定义摘要不匹配")
    if method_semantic_normative_trace_digest(payload) != EXPECTED_NORMATIVE_TRACE_DIGEST:
        add("normative_trace_digest", "权威公式、失败条件或主张边界发生未登记漂移")

    forbidden_keys = sorted(
        {
            key
            for key in _walk_keys(payload)
            if key in FORBIDDEN_SELF_ASSERTION_KEYS
            or _FORBIDDEN_SELF_ASSERTION_PATTERN.search(key)
        }
    )
    if forbidden_keys:
        add(
            "self_asserted_conformance",
            "方法语义登记表不得自行声明科学验证结论: "
            + ", ".join(forbidden_keys),
        )

    invariants = payload.get("invariants")
    if not isinstance(invariants, list):
        add("invariants_type", "invariants 必须是 JSON array")
        return violations
    invariant_ids = [
        str(item.get("invariant_id", ""))
        for item in invariants
        if isinstance(item, Mapping)
    ]
    if tuple(invariant_ids) != EXPECTED_INVARIANT_IDS:
        add("invariant_exact_set", "方法语义不变量必须按冻结顺序精确登记")
    if len(invariant_ids) != len(set(invariant_ids)):
        add("invariant_unique", "方法语义不变量 ID 不得重复")

    registered_fields = load_field_registry(root_path)
    try:
        method_configuration = load_method_semantic_configuration(root_path)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as error:
        add("method_configuration", f"方法 YAML 配置无法解析: {type(error).__name__}")
        method_configuration = {}
    symbol_cache: dict[Path, frozenset[str]] = {}
    test_cache: dict[Path, frozenset[str]] = {}
    for item in invariants:
        if not isinstance(item, Mapping):
            add("invariant_type", "每个方法语义不变量必须是 JSON object")
            continue
        invariant_id = str(item.get("invariant_id", ""))
        if set(item) != REQUIRED_INVARIANT_FIELDS:
            add("invariant_fields", f"{invariant_id} 的字段集合不精确")
        if invariant_id not in EXPECTED_CPU_PROPERTY_IDS:
            add("invariant_id", f"{invariant_id} 不是冻结方法语义不变量")
            continue
        if not _SNAKE_CASE_PATTERN.fullmatch(invariant_id):
            add("invariant_id", f"{invariant_id} 不是稳定 snake_case 标识")

        definition_pointer = str(item.get("definition_pointer", ""))
        document_path_text, separator, anchor = definition_pointer.partition("#")
        document_path = _repository_file(root_path, document_path_text)
        if not separator or document_path is None or not document_path.is_file():
            add("definition_pointer", f"{invariant_id} 的定义指针无效")
        else:
            expected_heading = f"## `{invariant_id}`"
            heading_count = document_path.read_text(encoding="utf-8").splitlines().count(
                expected_heading
            )
            if anchor != invariant_id or heading_count != 1:
                add("definition_anchor", f"{invariant_id} 的文档标题必须精确且唯一")

        _validate_string_sequence(
            item.get("formal_expression"),
            invariant_id=invariant_id,
            field_name="formal_expression",
            add=add,
            snake_case=False,
        )
        _validate_configuration_fields(
            item.get("configuration_fields"),
            invariant_id=invariant_id,
            configuration=method_configuration,
            add=add,
        )
        _validate_symbol_bindings(
            root_path,
            item.get("method_implementation_symbols"),
            invariant_id=invariant_id,
            field_name="method_implementation_symbols",
            required_root="main",
            expected_bindings=EXPECTED_METHOD_IMPLEMENTATION_SYMBOLS[invariant_id],
            symbol_cache=symbol_cache,
            add=add,
        )
        _validate_symbol_bindings(
            root_path,
            item.get("runtime_binding_symbols"),
            invariant_id=invariant_id,
            field_name="runtime_binding_symbols",
            required_root="experiments",
            expected_bindings=EXPECTED_RUNTIME_BINDING_SYMBOLS[invariant_id],
            symbol_cache=symbol_cache,
            add=add,
        )

        evidence_fields = _validate_string_sequence(
            item.get("runtime_evidence_fields"),
            invariant_id=invariant_id,
            field_name="runtime_evidence_fields",
            add=add,
            snake_case=True,
        )
        for field_name in evidence_fields:
            if field_name not in registered_fields:
                add(
                    "field_registry",
                    f"{invariant_id} 的证据字段未登记: {field_name}",
                )

        _validate_string_sequence(
            item.get("fail_closed_conditions"),
            invariant_id=invariant_id,
            field_name="fail_closed_conditions",
            add=add,
            snake_case=True,
        )
        _validate_string_sequence(
            item.get("forbidden_substitutes"),
            invariant_id=invariant_id,
            field_name="forbidden_substitutes",
            add=add,
            snake_case=True,
        )
        property_id = str(item.get("cpu_property_id", ""))
        if property_id != EXPECTED_CPU_PROPERTY_IDS[invariant_id]:
            add("cpu_property_id", f"{invariant_id} 的 CPU 性质 ID 发生漂移")
        _validate_test_nodes(
            root_path,
            item.get("specification_test_nodes"),
            invariant_id=invariant_id,
            field_name="specification_test_nodes",
            expected_nodes=EXPECTED_SPECIFICATION_TEST_NODES[invariant_id],
            allow_empty=False,
            add=add,
            test_cache=test_cache,
        )
        _validate_test_nodes(
            root_path,
            item.get("cpu_property_test_nodes"),
            invariant_id=invariant_id,
            field_name="cpu_property_test_nodes",
            expected_nodes=EXPECTED_CPU_PROPERTY_TEST_NODES[invariant_id],
            allow_empty=True,
            add=add,
            test_cache=test_cache,
        )
        _validate_string_sequence(
            item.get("gpu_atomic_roles"),
            invariant_id=invariant_id,
            field_name="gpu_atomic_roles",
            add=add,
            snake_case=True,
        )
        if not str(item.get("gpu_observation_requirement", "")).strip():
            add("gpu_observation_requirement", f"{invariant_id} 缺少 GPU 观察要求")
        if not str(item.get("claim_boundary", "")).strip():
            add("claim_boundary", f"{invariant_id} 缺少主张边界")
    return violations


__all__ = [
    "EXPECTED_CPU_PROPERTY_IDS",
    "EXPECTED_CPU_PROPERTY_TEST_NODES",
    "EXPECTED_SPECIFICATION_TEST_NODES",
    "EXPECTED_INVARIANT_IDS",
    "EXPECTED_METHOD_IMPLEMENTATION_SYMBOLS",
    "EXPECTED_NORMATIVE_TRACE_DIGEST",
    "EXPECTED_RUNTIME_BINDING_SYMBOLS",
    "FORBIDDEN_SELF_ASSERTION_KEYS",
    "METHOD_CONFIG_RELATIVE_PATH",
    "REGISTRY_RELATIVE_PATH",
    "REGISTRY_SCHEMA",
    "REGISTRY_SCOPE",
    "load_method_semantic_registry",
    "load_method_semantic_configuration",
    "method_semantic_normative_trace_digest",
    "validate_method_semantic_registry",
]
