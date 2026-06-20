"""内容载体方法子包。"""

from main.methods.carrier.attention import AttentionRelativeCarrier, derive_attention_relative_carrier, simulate_attention_update_strengths
from main.methods.carrier.compose import CONTENT_MODES, ContentUpdate, compose_content_update
from main.methods.carrier.hf import HfContentCarrier, derive_hf_content_carrier
from main.methods.carrier.lf import LfContentCarrier, derive_lf_content_carrier

__all__ = [
    "CONTENT_MODES",
    "AttentionRelativeCarrier",
    "ContentUpdate",
    "HfContentCarrier",
    "LfContentCarrier",
    "compose_content_update",
    "derive_attention_relative_carrier",
    "derive_hf_content_carrier",
    "derive_lf_content_carrier",
    "simulate_attention_update_strengths",
]
