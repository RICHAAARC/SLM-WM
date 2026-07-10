"""内容载体方法子包。"""

from main.methods.carrier.attention import AttentionRelativeCarrier, derive_attention_relative_carrier, simulate_attention_update_strengths
from main.methods.carrier.compose import CONTENT_MODES, ContentUpdate, compose_content_update
from main.methods.carrier.hf import HfContentCarrier, derive_hf_content_carrier
from main.methods.carrier.keyed_tensor import (
    BlindContentScore,
    KeyedTensorCarrier,
    build_low_frequency_template,
    build_tail_robust_template,
    compute_blind_content_score,
    project_canonical_template,
)
from main.methods.carrier.lf import LfContentCarrier, derive_lf_content_carrier

__all__ = [
    "CONTENT_MODES",
    "AttentionRelativeCarrier",
    "BlindContentScore",
    "ContentUpdate",
    "HfContentCarrier",
    "KeyedTensorCarrier",
    "LfContentCarrier",
    "build_low_frequency_template",
    "build_tail_robust_template",
    "compose_content_update",
    "compute_blind_content_score",
    "derive_attention_relative_carrier",
    "derive_hf_content_carrier",
    "derive_lf_content_carrier",
    "project_canonical_template",
    "simulate_attention_update_strengths",
]
