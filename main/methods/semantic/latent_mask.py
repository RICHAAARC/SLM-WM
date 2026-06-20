"""图像域语义掩码到 latent 掩码的投影。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from main.core.digest import build_stable_digest
from main.methods.semantic.risk_field import NumberLike, VectorInput, as_float_vector, clip_unit, ensure_equal_length


@dataclass(frozen=True)
class LatentMaskResult:
    """投影后的 latent 掩码和被掩码 latent。"""

    mask_source: str
    mask_values: tuple[float, ...]
    latent_mask_values: tuple[float, ...]
    masked_latent_values: tuple[float, ...]
    source_length: int
    target_length: int
    latent_mask_digest: str
    mask_source_digest: str
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验投影后掩码和 latent 长度一致。"""
        ensure_equal_length(
            {
                "latent_mask_values": self.latent_mask_values,
                "masked_latent_values": self.masked_latent_values,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


def resample_vector(values: tuple[float, ...], target_length: int) -> tuple[float, ...]:
    """使用最近邻策略将一维向量投影到目标长度。"""
    if target_length <= 0:
        raise ValueError("target_length 必须为正数")
    if len(values) == target_length:
        return values
    if len(values) == 1:
        return tuple(values[0] for _ in range(target_length))
    source_last = len(values) - 1
    target_last = target_length - 1
    return tuple(values[round(index * source_last / target_last)] for index in range(target_length))


def project_mask_to_latent(
    latent_values: VectorInput | Iterable[NumberLike],
    mask_values: VectorInput | Iterable[NumberLike],
    mask_source: str,
) -> LatentMaskResult:
    """将标准化语义掩码投影到 latent 长度并应用到 latent。"""
    latent_vector = as_float_vector(latent_values, "latent_values")
    source_mask = tuple(clip_unit(value) for value in as_float_vector(mask_values, "mask_values"))
    latent_mask = resample_vector(source_mask, len(latent_vector))
    masked_latent = tuple(value * mask for value, mask in zip(latent_vector, latent_mask))
    mask_payload = {
        "mask_source": mask_source,
        "latent_mask_values": [round(value, 12) for value in latent_mask],
        "masked_latent_values": [round(value, 12) for value in masked_latent],
    }
    return LatentMaskResult(
        mask_source=mask_source,
        mask_values=source_mask,
        latent_mask_values=latent_mask,
        masked_latent_values=masked_latent,
        source_length=len(source_mask),
        target_length=len(latent_vector),
        latent_mask_digest=build_stable_digest(mask_payload),
        mask_source_digest=build_stable_digest({"mask_source": mask_source, "mask_values": source_mask}),
        supports_paper_claim=False,
        metadata={"projection_operator": "nearest_neighbor_1d"},
    )
