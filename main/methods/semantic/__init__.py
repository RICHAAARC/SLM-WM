"""分支语义风险场方法。"""

from main.methods.semantic.branch_risk import (
    BRANCH_NAMES,
    NEUTRAL_TEXTURE_RISK_VALUE,
    BranchRiskConfig,
    BranchRiskFieldBundle,
    CarrierRiskField,
    build_active_branch_risk_fields,
    build_branch_risk_fields,
)
from main.methods.semantic.feature_protocol import (
    HANDCRAFTED_STRUCTURE_FEATURE_SCHEMA,
    HANDCRAFTED_STRUCTURE_FEATURE_WIDTH,
    JOINT_FEATURE_WIDTH,
    SEMANTIC_FEATURE_PROTOCOL_SCHEMA,
    SEMANTIC_FEATURE_SCHEMA,
    SEMANTIC_FEATURE_WIDTH,
    semantic_feature_protocol_record,
)
from main.methods.semantic.runtime import (
    DifferentiableSemanticFeatureRuntime,
    freeze_module_parameters,
)

__all__ = [
    "BRANCH_NAMES",
    "NEUTRAL_TEXTURE_RISK_VALUE",
    "BranchRiskConfig",
    "BranchRiskFieldBundle",
    "CarrierRiskField",
    "DifferentiableSemanticFeatureRuntime",
    "HANDCRAFTED_STRUCTURE_FEATURE_SCHEMA",
    "HANDCRAFTED_STRUCTURE_FEATURE_WIDTH",
    "JOINT_FEATURE_WIDTH",
    "SEMANTIC_FEATURE_PROTOCOL_SCHEMA",
    "SEMANTIC_FEATURE_SCHEMA",
    "SEMANTIC_FEATURE_WIDTH",
    "build_active_branch_risk_fields",
    "build_branch_risk_fields",
    "freeze_module_parameters",
    "semantic_feature_protocol_record",
]
