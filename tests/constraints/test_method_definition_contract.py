"""冻结构造式方法定义与“潜流形”术语边界。"""

from __future__ import annotations

from dataclasses import MISSING
import inspect
from pathlib import Path

import pytest

from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    semantic_watermark_runtime_config_payload,
)
from main.methods.carrier import build_low_frequency_template
from main.methods.detection import ImageOnlyDetectionConfig
from main.methods.geometry import recover_attention_affine_alignment
from main.methods.method_definition import (
    METHOD_DEFINITION_SCHEMA,
    semantic_conditioned_latent_method_definition,
    semantic_conditioned_latent_method_definition_digest,
)


ROOT = Path(__file__).resolve().parents[2]
METHOD_DOCUMENT = (
    ROOT
    / "docs"
    / "builds"
    / "method_section_semantic_conditioned_latent_manifold_watermark.md"
)
PRIMITIVE_DOCUMENT = (
    ROOT
    / "docs"
    / "builds"
    / "algorithm_primitives_semantic_conditioned_latent_manifold_watermark.md"
)
FIELD_REGISTRY = ROOT / "docs" / "field_registry.md"
EXPECTED_METHOD_DEFINITION = {
    "method_definition_schema": "slm_wm_constructive_local_tangent_v9",
    "method_name": "semantic_conditioned_latent_manifold_watermarking",
    "update_construction": {
        "semantics": "branchwise_constructive_safe_subspace_updates",
        "content_branch_rule": "project_template_then_risk_bounded_scale",
        "attention_branch_rule": (
            "project_direct_qk_gradient_then_risk_bounded_monotonic_backtracking"
        ),
        "composition_rule": "single_core_ordered_actual_dtype_branch_sum",
        "composition_implementation_layer": "main",
        "joint_argmax_solved": False,
    },
    "branch_risk": {
        "signal_calibration_rule": (
            "analytic_bounded_signals_without_per_sample_minmax"
        ),
        "neutral_texture_rule": "fixed_constant_risk_term",
        "neutral_texture_value": 0.5,
        "eligibility_rule": "strict_risk_less_than_threshold",
        "budget_broadcast_protocol": "per_sample_hw_repeat_channels_nchw_v1",
        "zero_support_rule": "exact_zero_direction_or_fail_closed",
        "effective_budget_rule": (
            "configured_budget_times_eligibility_indicator"
        ),
        "risk_bounded_scale_protocol": (
            "direction_peak_frozen_budget_ceiling_box_v1"
        ),
        "amplitude_envelope_rule": (
            "nominal_l2_times_unit_direction_linf_times_"
            "effective_budget_over_frozen_ceiling"
        ),
        "direction_scaling_rule": (
            "maximum_feasible_global_scalar_without_coordinate_clipping"
        ),
        "direction_ratio_epsilon": 1e-12,
        "direction_ratio_epsilon_role": (
            "zero_budget_leakage_and_active_coordinate_selection"
        ),
        "numerical_epsilon_role": (
            "nonzero_step_and_response_denominator_stability"
        ),
        "epsilon_roles_interchangeable": False,
    },
    "local_geometry": {
        "latent_manifold_term_scope": (
            "local_implicit_feature_level_set_tangent_interpretation"
        ),
        "numerical_object": "kernel_of_local_feature_jacobian",
        "feature_width": 716,
        "branch_risk_conditioned": True,
        "local_tangent_residual_gated": True,
        "global_nonlinear_manifold_constructed": False,
        "constant_rank_condition_verified": False,
        "chart_geodesic_or_retraction_used": False,
        "qr_column_reference_rule": (
            "routed_candidate_right_solve_upper_triangular_qr_factor"
        ),
        "qr_reference_solve_protocol": (
            "right_upper_triangular_solve_without_explicit_inverse_v1"
        ),
        "null_space_numerical_epsilon": 1e-12,
        "maximum_qr_condition_number": 1e6,
        "maximum_orthogonality_error": 1e-5,
        "shared_rms_column_reference_used": False,
        "explicit_qr_factor_inverse_used": False,
        "projection_energy_rule": "squared_l2_ratio",
        "post_risk_direction_jvp_rule": (
            "required_independently_for_each_active_branch"
        ),
        "post_risk_reference_direction_rule": (
            "unprojected_carrier_template_or_direct_qk_gradient"
        ),
    },
    "carrier_normalization": {
        "lf_carrier_protocol_schema": (
            "slm_wm_low_frequency_carrier_protocol_v1"
        ),
        "lf_kernel_size": 5,
        "lf_stride": 1,
        "lf_padding": 2,
        "lf_boundary_mode": "zero_padding",
        "lf_ceil_mode": False,
        "lf_count_include_pad": True,
        "lf_divisor_override": None,
        "lf_pooling_axes": "height_width_only",
        "lf_batch_channel_isolation": True,
        "lf_content_rule": "subtract_global_mean_then_l2_normalize",
        "lf_normalization_scope": "global_tensor",
        "lf_detection_score_weight": 0.70,
        "tail_robust_detection_score_weight": 0.30,
        "tail_robust_rule": (
            "amplitude_truncate_then_l2_normalize_without_mean_centering"
        ),
        "tail_carrier_protocol_schema": (
            "slm_wm_tail_robust_carrier_protocol_v1"
        ),
        "tail_selection_rule": (
            "descending_absolute_value_then_ascending_flat_index"
        ),
        "tail_nonselected_coordinate_rule": "exact_zero_after_normalization",
        "raw_aligned_template_identity_rule": (
            "same_shape_same_key_same_protocol_exact_content_digest"
        ),
        "fixed_fpr_carrier_identity_rule": (
            "lf_and_tail_protocol_digests_frozen_from_calibration"
        ),
    },
    "attention_geometry": {
        "relation_source": "direct_to_q_to_k_sampled_image_token_subgraph",
        "probability_inverse_relation_allowed": False,
        "relation_numerical_epsilon": 1e-12,
        "valid_row_energy_rule": (
            "both_centered_weighted_energies_strictly_above_epsilon_squared"
        ),
        "operator_metadata_evidence_rule": (
            "shared_full_record_validation_and_digest_recomputation"
        ),
        "risk_bounded_scale_is_backtracking_start": True,
        "acceptance_rule": (
            "actual_candidate_score_strictly_above_original_and_content_base"
        ),
        "full_joint_attention_all_tokens_optimized": False,
    },
    "image_only_alignment": {
        "attention_module_names": [
            "transformer_blocks.0.attn",
            "transformer_blocks.23.attn",
        ],
        "cross_layer_selection_rule": (
            "lexicographic_objective_observation_confidence_"
            "then_frozen_layer_order_v1"
        ),
        "cross_layer_tie_break_rule": (
            "earlier_frozen_attention_module_name"
        ),
        "anchor_selection_rule": (
            "evenly_spaced_over_sampled_token_index_range"
        ),
        "attention_anchor_count": 12,
        "inlier_ratio_denominator": "valid_covered_anchor_count",
        "attention_residual_threshold": 0.20,
        "attention_residual_coordinate_unit": (
            "normalized_xy_euclidean_distance"
        ),
        "attention_minimum_inlier_ratio": 0.50,
        "attention_coordinate_convention": (
            "normalized_xy_token_centers_corner_endpoints_v1"
        ),
        "attention_grid_align_corners": True,
        "image_resampling_mode": "bilinear",
        "image_padding_mode": "border",
        "image_quantization_protocol": (
            "clamp_0_1_multiply_255_floor_uint8_rgb_v1"
        ),
        "gate_parameter_source": (
            "preregistered_formal_method_configuration"
        ),
        "calibration_data_used_for_gate_parameters": False,
        "alignment_digest_binds_gate_parameters": True,
    },
    "write_validation": {
        "branch_amplitude_envelope_validation_rule": (
            "required_on_each_materialized_active_branch_update"
        ),
        "actual_dtype_composition_protocol": (
            "float32_ordered_branch_sum_add_float32_latent_single_cast_v1"
        ),
        "actual_dtype_composition_order": [
            "lf_content",
            "tail_robust",
            "attention_geometry",
        ],
        "combined_budget_envelope_rule": "sum_active_branch_envelopes",
        "quantized_budget_envelope_absolute_tolerance": 0.0,
        "quantized_budget_envelope_backtracking_factor": 0.5,
        "quantized_budget_envelope_backtracking_maximum_steps": 24,
        "quantized_envelope_recovery_rule": (
            "common_positive_scalar_backtracking_then_full_revalidation"
        ),
        "attention_post_composition_validation_rule": (
            "required_after_any_common_scalar_backtracking"
        ),
        "actual_dtype_budget_envelope_validation_rule": "required",
        "actual_dtype_update_jvp_validation_rule": "required",
        "finite_feature_change_validation_rule": "required",
    },
    "ablation_isolation": {
        "without_branch_risk_routing": (
            "unit_effective_budget_without_eligibility_filter"
        ),
        "without_jacobian_null_space": (
            "retain_risk_support_and_amplitude_envelope"
        ),
        "inactive_branch_envelope_rule": "exclude_from_combined_envelope_sum",
    },
    "keyed_prg": {
        "keyed_prg_version": (
            "sha256_counter_normal_icdf_table20_float32_v2"
        ),
        "keyed_prg_protocol_digest": (
            "a6266dc1fb4a59f8038062dcd120f145582153138b8176baae12013d5a22687b"
        ),
        "canonical_device": "cpu",
        "canonical_dtype": "float32",
        "uniform_output_rule": "direct_open_unit_interval_float32",
        "uniform_output_role": "attention_relation_signs",
        "uniform_uses_normal_transform": False,
        "gaussian_output_rule": (
            "20bit_msb_stream_index_to_frozen_midpoint_inverse_normal_cdf_float32"
        ),
        "gaussian_output_roles": [
            "content_carrier_templates",
            "jacobian_candidate_directions",
            "public_image_only_detection_noise",
        ],
        "public_detection_noise_key_role": (
            "deterministic_public_protocol_identity_not_secret_key"
        ),
        "gaussian_uses_frozen_normal_quantile_table": True,
        "uniform_and_gaussian_roles_interchangeable": False,
    },
    "branch_names": [
        "lf_content",
        "tail_robust",
        "attention_geometry",
    ],
}
EXPECTED_METHOD_DEFINITION_DIGEST = (
    "2b7ab51c952abf74d145fd23694790b9de71fc6712b319fb10b3f46c9db0a0fe"
)


@pytest.mark.constraint
def test_machine_readable_method_definition_freezes_constructive_semantics() -> None:
    """可机读记录必须拒绝联合优化和全局流形的过强解释."""

    definition = semantic_conditioned_latent_method_definition()

    assert METHOD_DEFINITION_SCHEMA == "slm_wm_constructive_local_tangent_v9"
    assert definition == EXPECTED_METHOD_DEFINITION
    assert definition["update_construction"]["joint_argmax_solved"] is False
    assert (
        definition["local_geometry"]["numerical_object"]
        == "kernel_of_local_feature_jacobian"
    )
    assert (
        definition["local_geometry"]["global_nonlinear_manifold_constructed"]
        is False
    )
    assert (
        definition["local_geometry"]["constant_rank_condition_verified"]
        is False
    )
    assert semantic_conditioned_latent_method_definition_digest() == (
        EXPECTED_METHOD_DEFINITION_DIGEST
    )


@pytest.mark.constraint
def test_method_documents_define_local_tangent_constructive_protocol() -> None:
    """方法文档必须描述当前构造协议和局部切空间边界."""

    for document in (METHOD_DOCUMENT, PRIMITIVE_DOCUMENT):
        text = document.read_text(encoding="utf-8")

        assert "构造式" in text
        assert "局部" in text and "特征水平集" in text and "切空间" in text
        assert "不构造全局非线性流形" in text
        assert "常秩" in text
        assert "\\arg\\max" not in text
        assert "\\beta_g" not in text
        assert "\\beta_s" not in text
        assert "\\beta_v" not in text
        assert "\\mathcal{M}_{\\mathrm{route}}" not in text


@pytest.mark.constraint
def test_field_registry_uses_numerical_basis_and_method_definition() -> None:
    """字段登记不得把数值 Null Space 基底误称为流形维度."""

    text = FIELD_REGISTRY.read_text(encoding="utf-8")

    assert "| basis_rank |" in text
    assert "| method_definition |" in text
    assert "| method_definition_digest |" in text
    assert "| manifold_dimension |" not in text


@pytest.mark.constraint
def test_runtime_config_identity_binds_method_definition() -> None:
    """单次科学运行身份必须绑定冻结的方法语义摘要."""

    payload = semantic_watermark_runtime_config_payload(
        SemanticWatermarkRuntimeConfig()
    )

    assert payload["method_definition"] == (
        semantic_conditioned_latent_method_definition()
    )
    assert payload["method_definition_digest"] == (
        EXPECTED_METHOD_DEFINITION_DIGEST
    )


@pytest.mark.constraint
def test_attention_alignment_gate_has_no_core_fallback_defaults() -> None:
    """核心检测和配准 API 必须要求调用方显式提供结构门禁."""

    signature = inspect.signature(recover_attention_affine_alignment)
    for parameter_name in (
        "anchor_count",
        "residual_threshold",
        "minimum_inlier_ratio",
    ):
        assert (
            signature.parameters[parameter_name].default
            is inspect.Parameter.empty
        )
    for field_name in (
        "attention_anchor_count",
        "attention_residual_threshold",
        "attention_minimum_inlier_ratio",
    ):
        field = ImageOnlyDetectionConfig.__dataclass_fields__[field_name]
        assert field.default is MISSING
        assert field.default_factory is MISSING


@pytest.mark.constraint
def test_low_frequency_carrier_has_no_core_fallback_defaults() -> None:
    """LF 构造和检测配置必须要求调用方显式提供完整协议."""

    signature = inspect.signature(build_low_frequency_template)
    assert (
        signature.parameters["low_frequency_config"].default
        is inspect.Parameter.empty
    )
    field = ImageOnlyDetectionConfig.__dataclass_fields__[
        "low_frequency_config"
    ]
    assert field.default is MISSING
    assert field.default_factory is MISSING
