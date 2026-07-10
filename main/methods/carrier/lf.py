"""LF 内容载体。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Sequence

from main.core.digest import build_stable_digest


@dataclass(frozen=True)
class LfContentCarrier:
    """LF 主证据载体。"""

    carrier_id: str
    basis_digest: str
    route_digest: str
    template_values: tuple[float, ...]
    update_values: tuple[float, ...]
    embedding_strength: float
    lf_content_carrier_digest: str
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验模板和 update 的最小长度。"""
        if not self.template_values:
            raise ValueError("template_values 不得为空")
        if not self.update_values:
            raise ValueError("update_values 不得为空")

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


def stable_signed_template(seed_parts: Sequence[str], count: int) -> tuple[float, ...]:
    """根据稳定种子材料生成 [-1, 1] 模板。"""
    if count <= 0:
        raise ValueError("count 必须为正数")
    values = []
    for index in range(count):
        digest = hashlib.sha256("|".join((*seed_parts, str(index))).encode("utf-8")).hexdigest()
        unit = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
        values.append(unit * 2.0 - 1.0)
    return tuple(values)


def smooth_template(values: tuple[float, ...]) -> tuple[float, ...]:
    """对 LF 模板执行邻域平滑, 形成低频主证据。"""
    if len(values) == 1:
        return values
    return tuple(
        (values[max(index - 1, 0)] + values[index] + values[min(index + 1, len(values) - 1)]) / 3.0
        for index in range(len(values))
    )


def update_from_indices(
    vector_width: int,
    selected_indices: tuple[int, ...],
    coefficients: tuple[float, ...],
    strength: float,
) -> tuple[float, ...]:
    """把稀疏轴系数写入 latent update 向量。"""
    if vector_width <= 0:
        raise ValueError("vector_width 必须为正数")
    if len(selected_indices) != len(coefficients):
        raise ValueError("selected_indices 与 coefficients 长度必须一致")
    update = [0.0 for _ in range(vector_width)]
    for axis, coefficient in zip(selected_indices, coefficients):
        update[axis % vector_width] += strength * coefficient
    return tuple(update)


def derive_lf_content_carrier(
    selected_indices: tuple[int, ...],
    basis_digest: str,
    route_digest: str,
    event_digest: str,
    key_material: str,
    vector_width: int,
    embedding_strength: float = 0.20,
) -> LfContentCarrier:
    """从安全基底和语义路由导出 LF 内容载体。"""
    template = smooth_template(stable_signed_template((key_material, event_digest, basis_digest, route_digest, "lf"), len(selected_indices)))
    update = update_from_indices(vector_width, selected_indices, template, embedding_strength)
    payload = {
        "basis_digest": basis_digest,
        "route_digest": route_digest,
        "template_values": [round(value, 12) for value in template],
        "update_values": [round(value, 12) for value in update],
        "embedding_strength": embedding_strength,
    }
    carrier_digest = build_stable_digest(payload)
    return LfContentCarrier(
        carrier_id=f"lf_content_{carrier_digest[:16]}",
        basis_digest=basis_digest,
        route_digest=route_digest,
        template_values=template,
        update_values=update,
        embedding_strength=embedding_strength,
        lf_content_carrier_digest=carrier_digest,
        supports_paper_claim=False,
        metadata={
            "carrier_family": "legacy_vector_low_pass",
            "frequency_band": "diagnostic_vector_index_low_pass",
            "formal_runtime_implementation": "main.methods.carrier.keyed_tensor.build_low_frequency_template",
        },
    )
