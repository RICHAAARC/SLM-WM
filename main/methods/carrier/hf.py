"""兼容历史记录的高斯幅值尾部截断鲁棒载体。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from main.core.digest import build_stable_digest
from main.methods.carrier.lf import stable_signed_template, update_from_indices


@dataclass(frozen=True)
class HfContentCarrier:
    """历史字段名保持 HF, 正式语义为尾部截断鲁棒载体。"""

    carrier_id: str
    basis_digest: str
    route_digest: str
    template_values: tuple[float, ...]
    update_values: tuple[float, ...]
    embedding_strength: float
    tail_fraction: float
    tail_threshold: float | None
    retained_fraction: float
    tail_truncation_enabled: bool
    hf_content_carrier_digest: str
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验 HF 模板边界。"""
        if not self.template_values:
            raise ValueError("template_values 不得为空")
        if not self.update_values:
            raise ValueError("update_values 不得为空")
        if self.retained_fraction < 0.0 or self.retained_fraction > 1.0:
            raise ValueError("retained_fraction 必须位于 [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


def tail_threshold_for(scores: tuple[float, ...], tail_fraction: float) -> float:
    """计算 HF tail truncation 阈值。"""
    ordered = sorted(scores)
    bounded = min(max(tail_fraction, 0.0), 1.0)
    if bounded >= 1.0:
        return ordered[0] - 1e-12
    if bounded <= 0.0:
        return ordered[-1] + 1e-12
    index = int((1.0 - bounded) * (len(ordered) - 1))
    return ordered[index]


def apply_tail_truncation(template: tuple[float, ...], tail_fraction: float) -> tuple[tuple[float, ...], float, float]:
    """保留 HF 模板尾部强响应。"""
    scores = tuple(abs(value) for value in template)
    threshold = tail_threshold_for(scores, tail_fraction)
    truncated = tuple(value if score >= threshold else 0.0 for value, score in zip(template, scores))
    retained_count = sum(1 for value in truncated if value != 0.0)
    if retained_count == 0:
        strongest = max(range(len(scores)), key=lambda index: scores[index])
        truncated = tuple(value if index == strongest else 0.0 for index, value in enumerate(template))
        retained_count = 1
    return truncated, threshold, retained_count / len(template)


def derive_hf_content_carrier(
    selected_indices: tuple[int, ...],
    basis_digest: str,
    route_digest: str,
    event_digest: str,
    key_material: str,
    vector_width: int,
    embedding_strength: float = 0.12,
    tail_fraction: float = 0.50,
    tail_truncation_enabled: bool = True,
) -> HfContentCarrier:
    """从安全基底和语义路由导出 HF 内容载体。"""
    template = stable_signed_template((key_material, event_digest, basis_digest, route_digest, "hf"), len(selected_indices))
    if tail_truncation_enabled:
        encoded_template, threshold, retained_fraction = apply_tail_truncation(template, tail_fraction)
    else:
        encoded_template, threshold, retained_fraction = template, None, 1.0
    update = update_from_indices(vector_width, selected_indices, encoded_template, embedding_strength)
    payload = {
        "basis_digest": basis_digest,
        "route_digest": route_digest,
        "template_values": [round(value, 12) for value in encoded_template],
        "update_values": [round(value, 12) for value in update],
        "embedding_strength": embedding_strength,
        "tail_truncation_enabled": tail_truncation_enabled,
        "tail_threshold": None if threshold is None else round(threshold, 12),
    }
    carrier_digest = build_stable_digest(payload)
    return HfContentCarrier(
        carrier_id=f"hf_content_{carrier_digest[:16]}",
        basis_digest=basis_digest,
        route_digest=route_digest,
        template_values=encoded_template,
        update_values=update,
        embedding_strength=embedding_strength,
        tail_fraction=tail_fraction,
        tail_threshold=threshold,
        retained_fraction=retained_fraction,
        tail_truncation_enabled=tail_truncation_enabled,
        hf_content_carrier_digest=carrier_digest,
        supports_paper_claim=False,
        metadata={
            "carrier_family": "gaussian_tail_robust",
            "branch_semantics": "amplitude_tail_not_spatial_high_frequency",
            "legacy_field_prefix": "hf",
        },
    )
