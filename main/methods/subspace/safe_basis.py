"""语义条件安全基底求解。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Sequence

from main.core.digest import build_stable_digest
from main.methods.semantic.risk_field import RiskFieldResult, ensure_equal_length
from main.methods.semantic.routing import SemanticRoute
from main.methods.subspace.jvp_estimator import ApproximateJvpEstimate
from main.methods.subspace.trajectory_features import TrajectoryFeatureSet


@dataclass(frozen=True)
class SafeBasisPlan:
    """安全子空间基底计划。"""

    safe_basis: tuple[tuple[float, ...], ...]
    selected_indices: tuple[int, ...]
    basis_digest: str
    basis_strategy: str
    semantic_mask_enabled: bool
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验基底行宽度一致。"""
        if self.safe_basis:
            ensure_equal_length({f"basis_row_{index}": row for index, row in enumerate(self.safe_basis)})

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


def _basis_from_indices(length: int, indices: Sequence[int], weights: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    """根据索引和权重生成 one-hot 基底。"""
    rows = []
    for index in indices:
        row = [0.0 for _ in range(length)]
        row[index] = weights[index]
        rows.append(tuple(row))
    return tuple(rows)


def _digest_basis(basis: tuple[tuple[float, ...], ...]) -> str:
    """生成基底稳定摘要。"""
    return build_stable_digest([[round(value, 12) for value in row] for row in basis])


def _stable_score(label: str, index: int) -> float:
    """根据标签和索引生成稳定诊断分数。"""
    digest = hashlib.sha256(f"{label}|{index}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def build_safe_basis_plan(
    features: TrajectoryFeatureSet,
    jvp_estimate: ApproximateJvpEstimate,
    risk_field: RiskFieldResult,
    semantic_route: SemanticRoute,
    basis_rank: int = 4,
    semantic_mask_enabled: bool = True,
    basis_strategy: str = "semantic_safe_basis",
) -> SafeBasisPlan:
    """根据语义路由、风险场和近似 JVP 构造安全基底。"""
    length = ensure_equal_length(
        {
            "feature_values": features.feature_values,
            "jvp_values": jvp_estimate.approximate_jvp_values,
            "risk_values": risk_field.risk_values,
        }
    )
    rank = max(1, min(basis_rank, length))
    if basis_strategy == "global_nullspace":
        scores = [(jvp_estimate.approximate_jvp_values[index], index) for index in range(length)]
        weights = tuple(1.0 for _ in range(length))
    elif basis_strategy == "diagnostic_basis":
        scores = [(_stable_score(semantic_route.route_digest, index), index) for index in range(length)]
        weights = tuple(1.0 for _ in range(length))
    else:
        semantic_bonus = set(semantic_route.lf_indices) | set(semantic_route.hf_indices) | set(semantic_route.attention_indices)
        scores = []
        weights = []
        for index in range(length):
            route_penalty = -0.15 if index in semantic_bonus else 0.0
            risk_term = risk_field.risk_values[index] if semantic_mask_enabled else 0.5
            mask_term = abs(features.feature_values[index]) if semantic_mask_enabled else 0.25
            response = risk_term + jvp_estimate.approximate_jvp_values[index] + mask_term + route_penalty
            scores.append((response, index))
            weights.append(risk_field.budget_values[index] if semantic_mask_enabled else 1.0)
    selected_indices = tuple(index for _, index in sorted(scores)[:rank])
    safe_basis = _basis_from_indices(length, selected_indices, weights)
    digest_payload = {
        "basis_strategy": basis_strategy,
        "semantic_mask_enabled": semantic_mask_enabled,
        "selected_indices": selected_indices,
        "basis": safe_basis,
        "route_digest": semantic_route.route_digest,
        "approximate_jvp_digest": jvp_estimate.approximate_jvp_digest,
    }
    return SafeBasisPlan(
        safe_basis=safe_basis,
        selected_indices=selected_indices,
        basis_digest=_digest_basis(safe_basis),
        basis_strategy=basis_strategy,
        semantic_mask_enabled=semantic_mask_enabled,
        supports_paper_claim=False,
        metadata={"basis_rank": rank, "plan_digest": build_stable_digest(digest_payload)},
    )
