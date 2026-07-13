"""冻结 SLM-WM 正式方法的可机读语义边界。"""

from __future__ import annotations

from typing import Any

from main.core.digest import build_stable_digest
from main.core.keyed_prg import KEYED_PRG_VERSION, keyed_prg_protocol_record
from main.methods.geometry.attention_alignment import (
    ATTENTION_ALIGNMENT_ANCHOR_COUNT,
    ATTENTION_ALIGNMENT_MINIMUM_INLIER_RATIO,
    ATTENTION_ALIGNMENT_RESIDUAL_THRESHOLD,
)
from main.methods.geometry.differentiable_attention import (
    ATTENTION_COORDINATE_CONVENTION,
    ATTENTION_GRID_ALIGN_CORNERS,
)


METHOD_DEFINITION_SCHEMA = "slm_wm_constructive_local_tangent_v6"


def semantic_conditioned_latent_method_definition() -> dict[str, Any]:
    """返回完整方法的构造协议与“潜流形”术语边界.

    该记录不保存单次运行参数. 它用于把论文方法名称绑定到必须执行的
    分支构造、局部线性几何和有限写回复验, 避免把构造式实现误写为
    未执行的联合优化或全局非线性流形求解. 该记录只表示规范身份,
    不表示当前实现已经通过 CPU、GPU 或论文证据验证.
    """

    return {
        "method_definition_schema": METHOD_DEFINITION_SCHEMA,
        "method_name": "semantic_conditioned_latent_manifold_watermarking",
        "update_construction": {
            "semantics": "branchwise_constructive_safe_subspace_updates",
            "content_branch_rule": (
                "project_template_then_risk_bounded_scale"
            ),
            "attention_branch_rule": (
                "project_direct_qk_gradient_then_risk_bounded_monotonic_backtracking"
            ),
            "composition_rule": (
                "single_core_ordered_actual_dtype_branch_sum"
            ),
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
            "budget_broadcast_protocol": (
                "per_sample_hw_repeat_channels_nchw_v1"
            ),
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
            "lf_content_rule": "subtract_global_mean_then_l2_normalize",
            "tail_robust_rule": (
                "amplitude_truncate_then_l2_normalize_without_mean_centering"
            ),
            "tail_nonselected_coordinate_rule": "exact_zero_after_normalization",
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
            "anchor_selection_rule": (
                "evenly_spaced_over_sampled_token_index_range"
            ),
            "attention_anchor_count": ATTENTION_ALIGNMENT_ANCHOR_COUNT,
            "inlier_ratio_denominator": "valid_covered_anchor_count",
            "attention_residual_threshold": (
                ATTENTION_ALIGNMENT_RESIDUAL_THRESHOLD
            ),
            "attention_residual_coordinate_unit": (
                "normalized_xy_euclidean_distance"
            ),
            "attention_minimum_inlier_ratio": (
                ATTENTION_ALIGNMENT_MINIMUM_INLIER_RATIO
            ),
            "attention_coordinate_convention": (
                ATTENTION_COORDINATE_CONVENTION
            ),
            "attention_grid_align_corners": (
                ATTENTION_GRID_ALIGN_CORNERS
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
            "combined_budget_envelope_rule": (
                "sum_active_branch_envelopes"
            ),
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
            "inactive_branch_envelope_rule": (
                "exclude_from_combined_envelope_sum"
            ),
        },
        "keyed_prg": {
            "keyed_prg_version": KEYED_PRG_VERSION,
            "keyed_prg_protocol_digest": keyed_prg_protocol_record(
                KEYED_PRG_VERSION
            )[
                "keyed_prg_protocol_digest"
            ],
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


def semantic_conditioned_latent_method_definition_digest() -> str:
    """返回正式方法语义记录的稳定 SHA-256 摘要."""

    return build_stable_digest(semantic_conditioned_latent_method_definition())


__all__ = [
    "METHOD_DEFINITION_SCHEMA",
    "semantic_conditioned_latent_method_definition",
    "semantic_conditioned_latent_method_definition_digest",
]
