"""冻结 SLM-WM 正式内容双链方法的可机读语义边界。"""

from __future__ import annotations

from typing import Any

from main.core.digest import build_stable_digest
from main.core.keyed_prg import KEYED_PRG_VERSION
from main.methods.geometry.differentiable_attention import (
    FROZEN_SD35_ATTENTION_MODULE_NAMES,
)


METHOD_DEFINITION_SCHEMA = "slm_wm_content_dual_chain_definition_v1"


def semantic_conditioned_latent_method_definition() -> dict[str, Any]:
    """返回正式 S/T/R/Q、双载体、Q/K 同步与盲检测方法身份。"""

    return {
        "method_definition_schema": METHOD_DEFINITION_SCHEMA,
        "method_name": "content_adaptive_dual_carrier_latent_watermark",
        "content_observations": {
            "ordered_observations": [
                "semantic_saliency",
                "texture_complexity",
                "adjacent_latent_response",
                "public_probe_local_sensitivity",
            ],
            "routing_rule": "formal_s_t_r_q_content_capacity_and_lf_hf_masks",
            "reference_source": "explicit_or_fixed_registry_fail_closed",
            "single_image_reference_recomputation_allowed": False,
        },
        "carrier_templates": {
            "low_frequency": "paired_2d_low_pass_center_per_sample_l2",
            "high_frequency_tail": (
                "paired_2d_high_pass_stable_topabs_one_fifth_then_l2"
            ),
            "independent_prg_domains": ["lf_content", "hf_tail_robust"],
            "keyed_prg_version": KEYED_PRG_VERSION,
            "relative_strengths": {
                "low_frequency": 0.0025,
                "high_frequency_tail": 0.0015,
                "attention_geometry": 0.0010,
            },
        },
        "generation_update": {
            "capture_index": 9,
            "write_index": 10,
            "write_count": 1,
            "composition_order": [
                "low_frequency",
                "high_frequency_tail",
                "attention_geometry",
            ],
            "common_backtracking": "gamma_j_equals_two_power_minus_j_j_0_to_24",
            "combined_relative_l2_limit": 0.0050,
            "actual_dtype_single_write": True,
            "legacy_multi_injection_allowed": False,
        },
        "attention_geometry": {
            "relation_source": "direct_qk",
            "attention_module_names": list(FROZEN_SD35_ATTENTION_MODULE_NAMES),
            "stable_token_fraction": 0.5,
            "unstable_pair_weight": 0.25,
            "post_write_score_rule": "strictly_improve_same_scoring_template",
            "jacobian_null_space_allowed": False,
            "jvp_vjp_allowed": False,
            "psd_cg_allowed": False,
        },
        "blind_detection": {
            "input_access": "image_key_public_model_only",
            "content_score": "0.70_lf_plus_0.30_hf_tail_by_method_role",
            "templates": "unmasked_formal_lf_and_hf_tail",
            "geometry_rescue": "near_threshold_alignment_then_same_threshold_retest",
            "geometry_score_can_directly_decide_positive": False,
        },
        "qualification_boundary": {
            "real_sd35_cuda_required": True,
            "registered_and_wrong_key_required": True,
            "reference_qualification_separate": True,
            "supports_paper_claim_without_formal_records": False,
        },
    }


def semantic_conditioned_latent_method_definition_digest() -> str:
    """返回正式方法定义的稳定摘要。"""

    return build_stable_digest(semantic_conditioned_latent_method_definition())


__all__ = [
    "METHOD_DEFINITION_SCHEMA",
    "semantic_conditioned_latent_method_definition",
    "semantic_conditioned_latent_method_definition_digest",
]
