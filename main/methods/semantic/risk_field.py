"""语义风险场构造。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any, Iterable, Sequence

from main.core.digest import build_stable_digest

NumberLike = int | float
VectorInput = NumberLike | Sequence["VectorInput"]


def _is_sequence(value: object) -> bool:
    """判断对象是否为可展开数值序列。"""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def as_float_vector(values: VectorInput | Iterable[NumberLike], field_name: str) -> tuple[float, ...]:
    """将数值或嵌套数值序列展平为 float 向量。"""
    flattened: list[float] = []

    def visit(candidate: object) -> None:
        if isinstance(candidate, Real):
            flattened.append(float(candidate))
            return
        if _is_sequence(candidate):
            for item in candidate:
                visit(item)
            return
        raise TypeError(f"{field_name} 必须是数值或数值序列")

    visit(values)
    if not flattened:
        raise ValueError(f"{field_name} 不得为空")
    if any(not math.isfinite(value) for value in flattened):
        raise ValueError(f"{field_name} 必须只包含有限数值")
    return tuple(flattened)


def ensure_equal_length(named_vectors: dict[str, Sequence[Any]]) -> int:
    """集中校验多个向量长度一致。"""
    lengths = {name: len(values) for name, values in named_vectors.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"字段长度不一致: {lengths}")
    return unique_lengths.pop()


def clip_unit(value: float) -> float:
    """将数值裁剪到 [0, 1] 区间。"""
    return min(max(value, 0.0), 1.0)


@dataclass(frozen=True)
class RiskFieldConfig:
    """语义风险场权重和预算配置。"""

    saliency_weight: float = 0.34
    semantic_weight: float = 0.26
    texture_weight: float = 0.20
    instability_weight: float = 0.20
    budget_floor: float = 0.05
    budget_ceiling: float = 1.0
    budget_gain: float = 0.55

    def __post_init__(self) -> None:
        """集中校验配置边界。"""
        weights = (self.saliency_weight, self.semantic_weight, self.texture_weight, self.instability_weight)
        if any(value < 0.0 for value in weights):
            raise ValueError("风险场权重不得为负")
        if self.budget_floor < 0.0 or self.budget_ceiling <= self.budget_floor:
            raise ValueError("预算上下界必须形成有效区间")


@dataclass(frozen=True)
class RiskFieldResult:
    """语义风险场结果。"""

    risk_values: tuple[float, ...]
    budget_values: tuple[float, ...]
    risk_field_digest: str
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验风险值和预算值长度一致。"""
        ensure_equal_length({"risk_values": self.risk_values, "budget_values": self.budget_values})

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


def build_risk_field(
    semantic_values: VectorInput | Iterable[NumberLike],
    texture_values: VectorInput | Iterable[NumberLike],
    stability_values: VectorInput | Iterable[NumberLike],
    saliency_values: VectorInput | Iterable[NumberLike],
    config: RiskFieldConfig | None = None,
) -> RiskFieldResult:
    """由标准化语义、纹理、稳定性和显著性向量构造风险场。"""
    risk_config = config or RiskFieldConfig()
    semantic_vector = tuple(clip_unit(value) for value in as_float_vector(semantic_values, "semantic_values"))
    texture_vector = tuple(clip_unit(value) for value in as_float_vector(texture_values, "texture_values"))
    stability_vector = tuple(clip_unit(value) for value in as_float_vector(stability_values, "stability_values"))
    saliency_vector = tuple(clip_unit(value) for value in as_float_vector(saliency_values, "saliency_values"))
    length = ensure_equal_length(
        {
            "semantic_values": semantic_vector,
            "texture_values": texture_vector,
            "stability_values": stability_vector,
            "saliency_values": saliency_vector,
        }
    )
    weight_sum = (
        risk_config.saliency_weight
        + risk_config.semantic_weight
        + risk_config.texture_weight
        + risk_config.instability_weight
    )
    risk_values = []
    budget_values = []
    for index in range(length):
        instability = 1.0 - stability_vector[index]
        raw_risk = (
            risk_config.saliency_weight * saliency_vector[index]
            + risk_config.semantic_weight * semantic_vector[index]
            + risk_config.texture_weight * texture_vector[index]
            + risk_config.instability_weight * instability
        ) / weight_sum
        risk = clip_unit(raw_risk)
        budget = min(
            risk_config.budget_ceiling,
            risk_config.budget_floor + risk_config.budget_gain * (1.0 - risk),
        )
        risk_values.append(risk)
        budget_values.append(budget)
    digest_payload = {
        "risk_values": [round(value, 12) for value in risk_values],
        "budget_values": [round(value, 12) for value in budget_values],
    }
    return RiskFieldResult(
        risk_values=tuple(risk_values),
        budget_values=tuple(budget_values),
        risk_field_digest=build_stable_digest(digest_payload),
        supports_paper_claim=False,
        metadata={"risk_field_length": length},
    )
