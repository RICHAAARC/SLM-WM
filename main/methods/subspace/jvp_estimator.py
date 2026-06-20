"""近似 JVP 估计器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from main.core.digest import build_stable_digest
from main.methods.subspace.trajectory_features import TrajectoryFeatureSet


@dataclass(frozen=True)
class ApproximateJvpEstimate:
    """由轨迹特征派生的近似 JVP 摘要。"""

    approximate_jvp_values: tuple[float, ...]
    approximate_jvp_digest: str
    jvp_estimator_name: str
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


def estimate_approximate_jvp(features: TrajectoryFeatureSet) -> ApproximateJvpEstimate:
    """用相邻差分构造轻量近似 JVP。"""
    values = features.masked_feature_values
    if len(values) == 1:
        jvp_values = (abs(values[0]),)
    else:
        jvp_values = tuple(abs(values[index] - values[index - 1]) for index in range(len(values)))
    payload = {
        "trajectory_feature_digest": features.trajectory_feature_digest,
        "approximate_jvp_values": [round(value, 12) for value in jvp_values],
    }
    return ApproximateJvpEstimate(
        approximate_jvp_values=jvp_values,
        approximate_jvp_digest=build_stable_digest(payload),
        jvp_estimator_name="neighbor_difference_approximation",
        supports_paper_claim=False,
        metadata={"jvp_value_count": len(jvp_values)},
    )
