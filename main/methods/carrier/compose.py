"""组合 LF 与高斯幅值尾部截断内容载体。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from main.core.digest import build_stable_digest
from main.methods.carrier.tail import TailContentCarrier
from main.methods.carrier.lf import LfContentCarrier

CONTENT_MODES = ("full_content_chain", "lf_only", "tail_only", "no_tail", "no_tail_truncation", "no_lf")


@dataclass(frozen=True)
class ContentUpdate:
    """LF 与高斯幅值尾部截断内容 update 的组合结果。"""

    content_mode: str
    lf_enabled: bool
    tail_enabled: bool
    tail_truncation_enabled: bool
    lf_update_values: tuple[float, ...]
    tail_update_values: tuple[float, ...]
    combined_update_values: tuple[float, ...]
    content_update_digest: str
    content_chain_digest: str
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """校验 update 分量长度一致。"""
        lengths = {len(self.lf_update_values), len(self.tail_update_values), len(self.combined_update_values)}
        if len(lengths) != 1:
            raise ValueError("内容 update 分量长度必须一致")

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return asdict(self)


def zero_like(values: tuple[float, ...]) -> tuple[float, ...]:
    """生成等长零向量。"""
    return tuple(0.0 for _ in values)


def compose_content_update(lf_carrier: LfContentCarrier, tail_carrier: TailContentCarrier, content_mode: str) -> ContentUpdate:
    """根据机制开关组合 LF 和高斯幅值尾部截断 update。"""
    if content_mode not in CONTENT_MODES:
        raise ValueError("未知内容载体模式")
    if len(lf_carrier.update_values) != len(tail_carrier.update_values):
        raise ValueError("LF 与高斯幅值尾部截断 update 长度必须一致")
    lf_enabled = content_mode not in {"tail_only", "no_lf"}
    tail_enabled = content_mode not in {"lf_only", "no_tail"}
    lf_update = lf_carrier.update_values if lf_enabled else zero_like(lf_carrier.update_values)
    tail_update = tail_carrier.update_values if tail_enabled else zero_like(tail_carrier.update_values)
    combined = tuple(left + right for left, right in zip(lf_update, tail_update))
    payload = {
        "content_mode": content_mode,
        "lf_digest": lf_carrier.lf_content_carrier_digest,
        "tail_digest": tail_carrier.tail_content_carrier_digest,
        "combined_update_values": [round(value, 12) for value in combined],
    }
    update_digest = build_stable_digest(payload)
    return ContentUpdate(
        content_mode=content_mode,
        lf_enabled=lf_enabled,
        tail_enabled=tail_enabled,
        tail_truncation_enabled=tail_carrier.tail_truncation_enabled,
        lf_update_values=lf_update,
        tail_update_values=tail_update,
        combined_update_values=combined,
        content_update_digest=update_digest,
        content_chain_digest=build_stable_digest({"content_update_digest": update_digest, "mode": content_mode}),
        supports_paper_claim=False,
        metadata={"content_mode": content_mode},
    )
