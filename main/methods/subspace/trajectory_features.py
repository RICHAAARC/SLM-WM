"""latent 轨迹特征算子。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from main.core.digest import build_stable_digest
from main.methods.semantic.latent_mask import LatentMaskResult
from main.methods.semantic.risk_field import ensure_equal_length


@dataclass(frozen=True)
class TrajectoryFeatureSet:
    """由 latent 掩码作用后的轨迹特征。"""

    feature_values: tuple[float, ...]
    masked_feature_values: tuple[float, ...]
    feature_operator_digest: str
    trajectory_feature_digest: str
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验特征长度一致。"""
        ensure_equal_length(
            {
                "feature_values": self.feature_values,
                "masked_feature_values": self.masked_feature_values,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


def _normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    """将向量缩放到单位二范数, 零向量保持零值。"""
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return tuple(0.0 for _ in values)
    return tuple(value / norm for value in values)


def build_trajectory_features(latent_mask: LatentMaskResult) -> TrajectoryFeatureSet:
    """构造 P^T vec(Norm(M_z * z_t)) 的轻量特征算子。"""
    masked_feature_values = _normalize(latent_mask.masked_latent_values)
    feature_values = _normalize(latent_mask.latent_mask_values)
    operator_payload = {
        "latent_mask_digest": latent_mask.latent_mask_digest,
        "operator": "masked_latent_unit_norm",
    }
    feature_payload = {
        "feature_values": [round(value, 12) for value in feature_values],
        "masked_feature_values": [round(value, 12) for value in masked_feature_values],
    }
    return TrajectoryFeatureSet(
        feature_values=feature_values,
        masked_feature_values=masked_feature_values,
        feature_operator_digest=build_stable_digest(operator_payload),
        trajectory_feature_digest=build_stable_digest(feature_payload),
        supports_paper_claim=False,
        metadata={"feature_length": len(feature_values)},
    )
