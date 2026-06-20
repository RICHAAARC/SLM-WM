"""语义路由规则。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from main.core.digest import build_stable_digest
from main.methods.semantic.latent_mask import LatentMaskResult
from main.methods.semantic.risk_field import RiskFieldResult, ensure_equal_length


@dataclass(frozen=True)
class SemanticRoute:
    """LF、HF 和 attention 候选轴路由。"""

    route_id: str
    route_label: str
    risk_profile: str
    lf_indices: tuple[int, ...]
    hf_indices: tuple[int, ...]
    attention_indices: tuple[int, ...]
    route_digest: str
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


def _indices_by_score(scores: list[tuple[float, int]], count: int) -> tuple[int, ...]:
    """从排序分数中取稳定索引集合。"""
    bounded_count = max(1, min(count, len(scores)))
    return tuple(index for _, index in sorted(scores)[:bounded_count])


def build_semantic_route(
    prompt_id: str,
    risk_profile: str,
    risk_field: RiskFieldResult,
    latent_mask: LatentMaskResult,
    route_width: int = 4,
) -> SemanticRoute:
    """根据风险场和 latent 掩码生成语义路由。"""
    length = ensure_equal_length(
        {
            "risk_values": risk_field.risk_values,
            "budget_values": risk_field.budget_values,
            "latent_mask_values": latent_mask.latent_mask_values,
        }
    )
    lf_scores = []
    hf_scores = []
    attention_scores = []
    for index in range(length):
        low_risk = 1.0 - risk_field.risk_values[index]
        mask_value = latent_mask.latent_mask_values[index]
        budget = risk_field.budget_values[index]
        lf_scores.append((risk_field.risk_values[index] + abs(mask_value - 0.65), index))
        hf_scores.append((-(budget + mask_value * low_risk), index))
        attention_scores.append((abs(mask_value - low_risk), index))
    lf_indices = _indices_by_score(lf_scores, route_width)
    hf_indices = _indices_by_score(hf_scores, route_width)
    attention_indices = _indices_by_score(attention_scores, route_width)
    route_payload = {
        "prompt_id": prompt_id,
        "risk_profile": risk_profile,
        "lf_indices": lf_indices,
        "hf_indices": hf_indices,
        "attention_indices": attention_indices,
        "risk_field_digest": risk_field.risk_field_digest,
        "latent_mask_digest": latent_mask.latent_mask_digest,
    }
    route_digest = build_stable_digest(route_payload)
    return SemanticRoute(
        route_id=f"route_{route_digest[:16]}",
        route_label="semantic_conditioned_route",
        risk_profile=risk_profile,
        lf_indices=lf_indices,
        hf_indices=hf_indices,
        attention_indices=attention_indices,
        route_digest=route_digest,
        supports_paper_claim=False,
        metadata={"route_width": min(route_width, length)},
    )
