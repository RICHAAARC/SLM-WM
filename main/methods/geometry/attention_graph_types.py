"""注意力图与几何证据 typed object。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AttentionGraphRecord:
    """描述由 self-attention map 派生的锚点图。"""

    attention_graph_id: str
    capture_id: str
    attention_layer: str
    attention_map_digest: str
    attention_shape: tuple[int, int]
    stable_token_indices: tuple[int, ...]
    relative_relation_values: tuple[tuple[int, int, float], ...]
    attention_relation_consistency: float
    anchor_graph_digest: str
    unsupported_reason: str
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验注意力图的最小结构边界。"""
        if len(self.attention_shape) != 2:
            raise ValueError("attention_shape 必须包含两个维度")
        if not self.stable_token_indices:
            raise ValueError("stable_token_indices 不得为空")

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class GeometryEvidenceRecord:
    """描述注意力相对几何恢复统计证据。"""

    geometry_evidence_record_id: str
    attention_graph_id: str
    capture_id: str
    attention_relation_consistency: float
    anchor_inlier_ratio: float
    registration_confidence: float
    recovered_sync_consistency: float
    alignment_residual: float
    geometry_reliable: bool
    direct_positive_decision: bool
    unsupported_reason: str
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验几何证据分数的共同边界。"""
        for field_name in (
            "attention_relation_consistency",
            "anchor_inlier_ratio",
            "registration_confidence",
            "recovered_sync_consistency",
            "alignment_residual",
        ):
            value = getattr(self, field_name)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{field_name} 必须位于 [0, 1]")
        if self.direct_positive_decision:
            raise ValueError("几何证据不得直接给出 positive 判定")

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)
