"""Attention-relative latent carrier。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

from main.core.digest import build_stable_digest

Relation = tuple[int, int, float]


@dataclass(frozen=True)
class AttentionRelativeCarrier:
    """描述由 attention graph 派生的潜空间几何载体。"""

    carrier_id: str
    attention_graph_id: str
    capture_id: str
    basis_digest: str
    route_digest: str
    anchor_graph_digest: str
    stable_token_indices: tuple[int, ...]
    target_relation_values: tuple[Relation, ...]
    baseline_relation_values: tuple[float, ...]
    relation_gradient_values: tuple[float, ...]
    update_values: tuple[float, ...]
    embedding_strength: float
    relation_loss_before: float
    relation_loss_after: float
    relation_loss_delta: float
    relation_consistency_before: float
    relation_consistency_after: float
    projected_update_norm: float
    quality_proxy_drop: float
    attention_update_stable: bool
    fallback_mode: str
    unsupported_reason: str
    attention_relative_carrier_digest: str
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验载体向量和关系向量的最小一致性。"""
        if not self.update_values:
            raise ValueError("update_values 不得为空")
        if len(self.baseline_relation_values) != len(self.target_relation_values):
            raise ValueError("baseline_relation_values 与 target_relation_values 长度必须一致")
        if self.fallback_mode not in {"active_update", "evidence_only"}:
            raise ValueError("fallback_mode 必须是 active_update 或 evidence_only")

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """把数值限制到闭区间内。"""
    return max(lower, min(upper, value))


def l2_norm(values: Sequence[float]) -> float:
    """计算向量二范数。"""
    return math.sqrt(sum(float(value) * float(value) for value in values))


def mean_squared_error(left: Sequence[float], right: Sequence[float]) -> float:
    """计算两个等长向量之间的均方误差。"""
    if not left:
        return 0.0
    return sum((float(left_value) - float(right_value)) ** 2 for left_value, right_value in zip(left, right)) / len(left)


def normalize_vector(values: Sequence[float]) -> tuple[float, ...]:
    """把非零向量归一化, 零向量保持为零。"""
    norm = l2_norm(values)
    if norm <= 1e-12:
        return tuple(0.0 for _ in values)
    return tuple(float(value) / norm for value in values)


def baseline_relations(target_relations: Sequence[Relation], attention_width: int) -> tuple[float, ...]:
    """构造未注入几何 update 时的可审计关系近似。"""
    neutral = 1.0 / max(1, attention_width)
    return tuple(0.65 * neutral + 0.35 * float(weight) for _, _, weight in target_relations)


def projected_relation_gradient(
    target_relations: Sequence[Relation],
    baseline_values: Sequence[float],
    selected_indices: Sequence[int],
    vector_width: int,
) -> tuple[float, ...]:
    """把 attention 关系误差投影到语义安全子空间选择的轴上。"""
    update = [0.0 for _ in range(vector_width)]
    if not selected_indices:
        return tuple(update)
    selected = tuple(int(index) % vector_width for index in selected_indices)
    for relation_index, ((source, target, target_weight), baseline_weight) in enumerate(zip(target_relations, baseline_values)):
        error = float(target_weight) - float(baseline_weight)
        source_axis = selected[(int(source) + relation_index) % len(selected)]
        target_axis = selected[(int(target) + relation_index) % len(selected)]
        update[source_axis] += error
        update[target_axis] -= 0.5 * error
    return tuple(update)


def relation_after_update(
    baseline_values: Sequence[float],
    target_values: Sequence[float],
    embedding_strength: float,
    registration_confidence: float,
) -> tuple[float, ...]:
    """估计执行几何 update 后的 attention 相对关系。"""
    step = clamp(embedding_strength * max(0.0, registration_confidence) * 4.0)
    return tuple(float(base) + step * (float(target) - float(base)) for base, target in zip(baseline_values, target_values))


def zero_update(vector_width: int) -> tuple[float, ...]:
    """生成指定宽度的零 update。"""
    return tuple(0.0 for _ in range(vector_width))


def build_fallback_carrier(
    attention_graph: dict[str, Any],
    subspace_record: dict[str, Any],
    route_record: dict[str, Any],
    vector_width: int,
    embedding_strength: float,
    unsupported_reason: str,
) -> AttentionRelativeCarrier:
    """构造 evidence-only 载体, 用于保持边界而不执行几何 update。"""
    target_relations = tuple((int(source), int(target), float(weight)) for source, target, weight in attention_graph.get("relative_relation_values", ()))
    baseline = baseline_relations(target_relations, int(attention_graph.get("attention_shape", (vector_width, vector_width))[1]))
    target_values = tuple(weight for _, _, weight in target_relations)
    loss_before = mean_squared_error(baseline, target_values)
    payload = {
        "attention_graph_id": attention_graph.get("attention_graph_id", ""),
        "basis_digest": subspace_record.get("basis_digest", ""),
        "route_digest": route_record.get("route_digest", ""),
        "fallback_mode": "evidence_only",
        "unsupported_reason": unsupported_reason,
    }
    digest = build_stable_digest(payload)
    return AttentionRelativeCarrier(
        carrier_id=f"attention_carrier_{digest[:16]}",
        attention_graph_id=attention_graph.get("attention_graph_id", ""),
        capture_id=attention_graph.get("capture_id", ""),
        basis_digest=subspace_record.get("basis_digest", ""),
        route_digest=route_record.get("route_digest", ""),
        anchor_graph_digest=attention_graph.get("anchor_graph_digest", ""),
        stable_token_indices=tuple(int(value) for value in attention_graph.get("stable_token_indices", ())),
        target_relation_values=target_relations,
        baseline_relation_values=baseline,
        relation_gradient_values=zero_update(vector_width),
        update_values=zero_update(vector_width),
        embedding_strength=embedding_strength,
        relation_loss_before=loss_before,
        relation_loss_after=loss_before,
        relation_loss_delta=0.0,
        relation_consistency_before=float(attention_graph.get("attention_relation_consistency", 0.0)),
        relation_consistency_after=float(attention_graph.get("attention_relation_consistency", 0.0)),
        projected_update_norm=0.0,
        quality_proxy_drop=0.0,
        attention_update_stable=False,
        fallback_mode="evidence_only",
        unsupported_reason=unsupported_reason,
        attention_relative_carrier_digest=digest,
        supports_paper_claim=False,
        metadata={"carrier_family": "attention_relative_geometry", "supports_paper_claim": False},
    )


def derive_attention_relative_carrier(
    attention_graph: dict[str, Any],
    geometry_evidence: dict[str, Any],
    subspace_record: dict[str, Any],
    route_record: dict[str, Any],
    vector_width: int,
    embedding_strength: float = 0.08,
    max_quality_proxy_drop: float = 0.08,
) -> AttentionRelativeCarrier:
    """从真实 attention graph 和语义安全子空间派生几何 update。"""
    unsupported_reason = attention_graph.get("unsupported_reason") or geometry_evidence.get("unsupported_reason") or ""
    target_relations = tuple((int(source), int(target), float(weight)) for source, target, weight in attention_graph.get("relative_relation_values", ()))
    if unsupported_reason:
        return build_fallback_carrier(attention_graph, subspace_record, route_record, vector_width, embedding_strength, unsupported_reason)
    if not target_relations:
        return build_fallback_carrier(attention_graph, subspace_record, route_record, vector_width, embedding_strength, "missing_attention_relations")
    if not bool(geometry_evidence.get("geometry_reliable", False)):
        return build_fallback_carrier(attention_graph, subspace_record, route_record, vector_width, embedding_strength, "geometry_evidence_unreliable")

    selected_indices = tuple(int(value) for value in subspace_record.get("selected_indices", ()))
    attention_shape = tuple(int(value) for value in attention_graph.get("attention_shape", (vector_width, vector_width)))
    baseline = baseline_relations(target_relations, attention_shape[1] if len(attention_shape) == 2 else vector_width)
    target_values = tuple(weight for _, _, weight in target_relations)
    gradient = projected_relation_gradient(target_relations, baseline, selected_indices, vector_width)
    unit_gradient = normalize_vector(gradient)
    update = tuple(embedding_strength * value for value in unit_gradient)
    registration_confidence = float(geometry_evidence.get("registration_confidence", 0.0))
    after_values = relation_after_update(baseline, target_values, embedding_strength, registration_confidence)
    loss_before = mean_squared_error(baseline, target_values)
    loss_after = mean_squared_error(after_values, target_values)
    update_norm = l2_norm(update)
    quality_proxy_drop = clamp(update_norm * update_norm * (1.0 + max(0.0, 1.0 - registration_confidence)))
    stable = loss_after <= loss_before and quality_proxy_drop <= max_quality_proxy_drop and update_norm > 0.0
    if not stable:
        return build_fallback_carrier(attention_graph, subspace_record, route_record, vector_width, embedding_strength, "attention_update_unstable")

    consistency_before = float(geometry_evidence.get("attention_relation_consistency", attention_graph.get("attention_relation_consistency", 0.0)))
    consistency_after = clamp(consistency_before + registration_confidence * embedding_strength * (1.0 - consistency_before))
    payload = {
        "attention_graph_id": attention_graph["attention_graph_id"],
        "basis_digest": subspace_record["basis_digest"],
        "route_digest": route_record["route_digest"],
        "anchor_graph_digest": attention_graph["anchor_graph_digest"],
        "update_values": [round(value, 12) for value in update],
        "relation_loss_after": round(loss_after, 12),
        "embedding_strength": embedding_strength,
    }
    digest = build_stable_digest(payload)
    return AttentionRelativeCarrier(
        carrier_id=f"attention_carrier_{digest[:16]}",
        attention_graph_id=attention_graph["attention_graph_id"],
        capture_id=attention_graph["capture_id"],
        basis_digest=subspace_record["basis_digest"],
        route_digest=route_record["route_digest"],
        anchor_graph_digest=attention_graph["anchor_graph_digest"],
        stable_token_indices=tuple(int(value) for value in attention_graph["stable_token_indices"]),
        target_relation_values=target_relations,
        baseline_relation_values=baseline,
        relation_gradient_values=gradient,
        update_values=update,
        embedding_strength=embedding_strength,
        relation_loss_before=loss_before,
        relation_loss_after=loss_after,
        relation_loss_delta=loss_before - loss_after,
        relation_consistency_before=consistency_before,
        relation_consistency_after=consistency_after,
        projected_update_norm=update_norm,
        quality_proxy_drop=quality_proxy_drop,
        attention_update_stable=stable,
        fallback_mode="active_update",
        unsupported_reason="",
        attention_relative_carrier_digest=digest,
        supports_paper_claim=False,
        metadata={
            "carrier_family": "attention_relative_geometry",
            "projection_mode": "semantic_safe_axis_projection",
            "geometry_evidence_record_id": geometry_evidence.get("geometry_evidence_record_id", ""),
            "approximation_mode": "auditable_relation_gradient_proxy",
            "supports_paper_claim": False,
        },
    )


def simulate_attention_update_strengths(
    carrier: AttentionRelativeCarrier,
    strength_scales: Sequence[float],
    max_quality_proxy_drop: float = 0.08,
) -> tuple[dict[str, Any], ...]:
    """生成不同 update 强度下的关系稳定性曲线。"""
    rows: list[dict[str, Any]] = []
    for scale in strength_scales:
        bounded_scale = max(0.0, float(scale))
        if carrier.fallback_mode == "active_update":
            loss_after = carrier.relation_loss_before - bounded_scale * carrier.relation_loss_delta
            consistency_after = clamp(
                carrier.relation_consistency_before
                + bounded_scale * (carrier.relation_consistency_after - carrier.relation_consistency_before)
            )
            update_norm = bounded_scale * carrier.projected_update_norm
            quality_drop = clamp(carrier.quality_proxy_drop * bounded_scale * bounded_scale)
            stable = loss_after <= carrier.relation_loss_before and quality_drop <= max_quality_proxy_drop
        else:
            loss_after = carrier.relation_loss_before
            consistency_after = carrier.relation_consistency_before
            update_norm = 0.0
            quality_drop = 0.0
            stable = False
        rows.append(
            {
                "carrier_id": carrier.carrier_id,
                "attention_graph_id": carrier.attention_graph_id,
                "capture_id": carrier.capture_id,
                "attention_update_strength": bounded_scale * carrier.embedding_strength,
                "relation_loss_before": carrier.relation_loss_before,
                "relation_loss_after": loss_after,
                "relation_loss_delta": carrier.relation_loss_before - loss_after,
                "relation_consistency_before": carrier.relation_consistency_before,
                "relation_consistency_after": consistency_after,
                "projected_update_norm": update_norm,
                "quality_proxy_drop": quality_drop,
                "attention_update_stable": stable,
                "fallback_mode": carrier.fallback_mode,
                "unsupported_reason": carrier.unsupported_reason,
            }
        )
    return tuple(rows)
