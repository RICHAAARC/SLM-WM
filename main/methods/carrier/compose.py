"""组合 LF/HF 内容载体。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from main.core.digest import build_stable_digest
from main.methods.carrier.hf import HfContentCarrier
from main.methods.carrier.lf import LfContentCarrier

CONTENT_MODES = ("full_content_chain", "lf_only", "hf_only", "no_hf", "no_tail_truncation", "no_lf")


@dataclass(frozen=True)
class ContentUpdate:
    """LF/HF 内容 update 组合结果。"""

    content_mode: str
    lf_enabled: bool
    hf_enabled: bool
    tail_truncation_enabled: bool
    lf_update_values: tuple[float, ...]
    hf_update_values: tuple[float, ...]
    combined_update_values: tuple[float, ...]
    content_update_digest: str
    content_chain_digest: str
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验 update 分量长度一致。"""
        lengths = {len(self.lf_update_values), len(self.hf_update_values), len(self.combined_update_values)}
        if len(lengths) != 1:
            raise ValueError("内容 update 分量长度必须一致")

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


def zero_like(values: tuple[float, ...]) -> tuple[float, ...]:
    """生成等长零向量。"""
    return tuple(0.0 for _ in values)


def compose_content_update(lf_carrier: LfContentCarrier, hf_carrier: HfContentCarrier, content_mode: str) -> ContentUpdate:
    """根据机制开关组合 LF 和 HF update。"""
    if content_mode not in CONTENT_MODES:
        raise ValueError("未知内容载体模式")
    if len(lf_carrier.update_values) != len(hf_carrier.update_values):
        raise ValueError("LF 与 HF update 长度必须一致")
    lf_enabled = content_mode not in {"hf_only", "no_lf"}
    hf_enabled = content_mode not in {"lf_only", "no_hf"}
    lf_update = lf_carrier.update_values if lf_enabled else zero_like(lf_carrier.update_values)
    hf_update = hf_carrier.update_values if hf_enabled else zero_like(hf_carrier.update_values)
    combined = tuple(left + right for left, right in zip(lf_update, hf_update))
    payload = {
        "content_mode": content_mode,
        "lf_digest": lf_carrier.lf_content_carrier_digest,
        "hf_digest": hf_carrier.hf_content_carrier_digest,
        "combined_update_values": [round(value, 12) for value in combined],
    }
    update_digest = build_stable_digest(payload)
    return ContentUpdate(
        content_mode=content_mode,
        lf_enabled=lf_enabled,
        hf_enabled=hf_enabled,
        tail_truncation_enabled=hf_carrier.tail_truncation_enabled,
        lf_update_values=lf_update,
        hf_update_values=hf_update,
        combined_update_values=combined,
        content_update_digest=update_digest,
        content_chain_digest=build_stable_digest({"content_update_digest": update_digest, "mode": content_mode}),
        supports_paper_claim=False,
        metadata={"content_mode": content_mode},
    )
