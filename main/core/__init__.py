"""核心记录、manifest、schema、digest 与方法 typed object 能力。"""

from main.core.method_objects import (
    CORE_METHOD_OBJECT_NAMES,
    AttentionAnchorSpec,
    DetectionEvidenceSpec,
    FusionDecisionSpec,
    LatentSubspaceSpec,
    SemanticConditionSpec,
    WatermarkCarrierSpec,
)

__all__ = [
    "CORE_METHOD_OBJECT_NAMES",
    "AttentionAnchorSpec",
    "DetectionEvidenceSpec",
    "FusionDecisionSpec",
    "LatentSubspaceSpec",
    "SemanticConditionSpec",
    "WatermarkCarrierSpec",
]
