"""注意力相对几何方法模块。"""

from main.methods.geometry.attention_alignment import AttentionAlignmentResult, recover_attention_affine_alignment
from main.methods.geometry.attention_graph_types import AttentionGraphRecord, GeometryEvidenceRecord
from main.methods.geometry.differentiable_attention import (
    AttentionGeometryGradient,
    AttentionGeometryUpdate,
    DifferentiableAttentionRecorder,
    attention_geometry_score,
    compute_attention_geometry_gradient,
    optimize_attention_geometry_update,
    qk_self_attention,
)
from main.methods.geometry.recovery import (
    attention_from_query_key,
    build_attention_graph_record,
    build_geometry_evidence_record,
    normalize_attention_rows,
    recovered_sync_consistency,
    relation_consistency,
    relative_relation_values,
    stable_token_set,
)

__all__ = [
    "AttentionAlignmentResult",
    "AttentionGeometryGradient",
    "AttentionGeometryUpdate",
    "AttentionGraphRecord",
    "DifferentiableAttentionRecorder",
    "GeometryEvidenceRecord",
    "attention_from_query_key",
    "attention_geometry_score",
    "compute_attention_geometry_gradient",
    "build_attention_graph_record",
    "build_geometry_evidence_record",
    "normalize_attention_rows",
    "optimize_attention_geometry_update",
    "qk_self_attention",
    "recover_attention_affine_alignment",
    "recovered_sync_consistency",
    "relation_consistency",
    "relative_relation_values",
    "stable_token_set",
]
