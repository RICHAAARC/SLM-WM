"""SLM-WM 正式论文方法实现。"""

from main.methods.carrier import (
    BlindContentScore,
    KeyedTensorCarrier,
    build_low_frequency_template,
    build_tail_robust_template,
    compute_blind_content_score,
    project_canonical_template,
)
from main.methods.detection import (
    ImageOnlyDetectionConfig,
    ImageOnlyDetectionResult,
    detect_image_only_watermark,
)
from main.methods.geometry import (
    AttentionAlignmentResult,
    AttentionGeometryGradient,
    AttentionGeometryUpdate,
    DifferentiableAttentionRecorder,
    attention_geometry_score,
    compute_attention_geometry_gradient,
    optimize_attention_geometry_update,
    qk_self_attention,
    recover_attention_affine_alignment,
)
from main.methods.semantic import (
    BranchRiskConfig,
    BranchRiskFieldBundle,
    CarrierRiskField,
    build_branch_risk_fields,
)
from main.methods.subspace import (
    ExactJacobianLinearization,
    JacobianNullSpaceResult,
    PSDConjugateGradientResult,
    build_exact_jacobian_linearization,
    exact_jvp,
    generate_keyed_candidate_directions,
    solve_psd_conjugate_gradient,
    solve_jacobian_null_space,
)

__all__ = [
    "AttentionAlignmentResult",
    "AttentionGeometryGradient",
    "AttentionGeometryUpdate",
    "BlindContentScore",
    "BranchRiskConfig",
    "BranchRiskFieldBundle",
    "CarrierRiskField",
    "DifferentiableAttentionRecorder",
    "ExactJacobianLinearization",
    "ImageOnlyDetectionConfig",
    "ImageOnlyDetectionResult",
    "JacobianNullSpaceResult",
    "KeyedTensorCarrier",
    "PSDConjugateGradientResult",
    "attention_geometry_score",
    "build_branch_risk_fields",
    "build_exact_jacobian_linearization",
    "build_low_frequency_template",
    "build_tail_robust_template",
    "compute_attention_geometry_gradient",
    "compute_blind_content_score",
    "detect_image_only_watermark",
    "exact_jvp",
    "generate_keyed_candidate_directions",
    "optimize_attention_geometry_update",
    "project_canonical_template",
    "qk_self_attention",
    "recover_attention_affine_alignment",
    "solve_psd_conjugate_gradient",
    "solve_jacobian_null_space",
]
