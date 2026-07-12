"""冻结 SLM-WM 正式方法的可机读语义边界。"""

from __future__ import annotations

from typing import Any

from main.core.digest import build_stable_digest


METHOD_DEFINITION_SCHEMA = "slm_wm_constructive_local_tangent_v1"


def semantic_conditioned_latent_method_definition() -> dict[str, Any]:
    """返回完整方法的构造协议与“潜流形”术语边界.

    该记录不保存单次运行参数. 它用于把论文方法名称绑定到实际执行的
    分支构造、局部线性几何和有限写回复验, 避免把构造式实现误写为
    未执行的联合优化或全局非线性流形求解.
    """

    return {
        "method_definition_schema": METHOD_DEFINITION_SCHEMA,
        "method_name": "semantic_conditioned_latent_manifold_watermarking",
        "update_construction": {
            "semantics": "branchwise_constructive_safe_subspace_updates",
            "content_branch_rule": "project_template_then_l2_scale",
            "attention_branch_rule": (
                "project_direct_qk_gradient_then_monotonic_backtracking"
            ),
            "composition_rule": "sum_lf_tail_attention_updates",
            "joint_argmax_solved": False,
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
        },
        "write_validation": {
            "actual_dtype_update_jvp_revalidated": True,
            "finite_feature_change_revalidated": True,
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
