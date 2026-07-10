"""真实 Q/K 注意力几何方法。"""

from main.methods.geometry.attention_alignment import AttentionAlignmentResult, recover_attention_affine_alignment
from main.methods.geometry.differentiable_attention import (
    AttentionGeometryGradient,
    AttentionGeometryUpdate,
    DifferentiableAttentionRecorder,
    attention_geometry_score,
    attention_relation_stability_map,
    compute_attention_geometry_gradient,
    optimize_attention_geometry_update,
    qk_self_attention,
)

__all__ = [
    "AttentionAlignmentResult",
    "AttentionGeometryGradient",
    "AttentionGeometryUpdate",
    "DifferentiableAttentionRecorder",
    "attention_geometry_score",
    "attention_relation_stability_map",
    "compute_attention_geometry_gradient",
    "optimize_attention_geometry_update",
    "qk_self_attention",
    "recover_attention_affine_alignment",
]
