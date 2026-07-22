"""在扩散结束后构造固定能量的 LF/HF-tail 密钥载体更新。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from main.methods.carrier.high_frequency_tail import HighFrequencyTailCarrierTemplate
from main.methods.carrier.low_frequency import LowFrequencyCarrierTemplate
from main.methods.content.routing import ContentRoutingResult


TERMINAL_CONTENT_STRENGTH_MULTIPLIERS = (4.0, 8.0)
_LF_RELATIVE_STRENGTH = 0.0025
_HF_TAIL_RELATIVE_STRENGTH = 0.0015


@dataclass(frozen=True)
class TerminalContentCarrierUpdate:
    """保存 terminal latent 上一次内容载体写入的实际结果。"""

    written_latent: Any
    lf_direction: Any
    hf_tail_direction: Any
    lf_update: Any
    hf_tail_update: Any
    routing_mode: str
    carrier_mode: str
    strength_multiplier: float
    lf_effective_l2: float
    hf_tail_effective_l2: float
    combined_effective_l2: float
    combined_relative_l2: float


def _unit_direction(value: Any, *, label: str) -> Any:
    import torch

    norm = torch.linalg.vector_norm(value.reshape(-1))
    if not bool(torch.isfinite(norm)) or not bool(norm > 0.0):
        raise ValueError(f"{label} 必须具有正有限 L2 能量")
    return value / norm


def build_terminal_content_carrier_update(
    terminal_latent: Any,
    routing: ContentRoutingResult,
    lf_template: LowFrequencyCarrierTemplate,
    hf_tail_template: HighFrequencyTailCarrierTemplate,
    *,
    routing_mode: Literal["semantic_unit_energy", "uniform"],
    carrier_mode: Literal["lf_only", "hf_only", "dual"],
    strength_multiplier: float,
) -> TerminalContentCarrierUpdate:
    """在 pre-VAE terminal latent 上按固定相对 L2 一次写入密钥载体。"""

    import torch

    if routing_mode not in ("semantic_unit_energy", "uniform"):
        raise ValueError("routing_mode 必须为 semantic_unit_energy 或 uniform")
    if carrier_mode not in ("lf_only", "hf_only", "dual"):
        raise ValueError("carrier_mode 必须为 lf_only、hf_only 或 dual")
    if strength_multiplier not in TERMINAL_CONTENT_STRENGTH_MULTIPLIERS:
        raise ValueError("strength_multiplier 必须为登记的 terminal 强度倍率")
    if not isinstance(terminal_latent, torch.Tensor) or terminal_latent.ndim != 4:
        raise TypeError("terminal_latent 必须为四维 Tensor")
    if not terminal_latent.dtype.is_floating_point or not bool(
        torch.isfinite(terminal_latent).all()
    ):
        raise ValueError("terminal_latent 必须为有限浮点 Tensor")
    if type(routing) is not ContentRoutingResult:
        raise TypeError("routing 必须为 ContentRoutingResult")
    if type(lf_template) is not LowFrequencyCarrierTemplate:
        raise TypeError("lf_template 必须为 LowFrequencyCarrierTemplate")
    if type(hf_tail_template) is not HighFrequencyTailCarrierTemplate:
        raise TypeError("hf_tail_template 必须为 HighFrequencyTailCarrierTemplate")

    shape = tuple(int(value) for value in terminal_latent.shape)
    if lf_template.latent_shape != shape or hf_tail_template.latent_shape != shape:
        raise ValueError("terminal 模板形状必须与 terminal_latent 一致")
    if lf_template.scoring_key_identity_digest != hf_tail_template.scoring_key_identity_digest:
        raise ValueError("LF 与 HF-tail 模板必须属于同一评分密钥")

    latent_float32 = terminal_latent.detach().to(dtype=torch.float32)
    latent_l2 = torch.linalg.vector_norm(latent_float32.reshape(-1))
    if not bool(torch.isfinite(latent_l2)) or not bool(latent_l2 > 0.0):
        raise ValueError("terminal_latent 必须具有正有限 L2 能量")
    if routing_mode == "uniform":
        lf_raw = lf_template.template.detach()
        hf_raw = hf_tail_template.template.detach()
    else:
        lf_raw = routing.lf_mask.detach() * lf_template.template.detach()
        hf_raw = routing.hf_tail_mask.detach() * hf_tail_template.template.detach()

    lf_active = carrier_mode != "hf_only"
    hf_tail_active = carrier_mode != "lf_only"
    lf_direction = (
        _unit_direction(lf_raw, label="LF terminal 方向")
        if lf_active
        else torch.zeros_like(latent_float32)
    )
    hf_tail_direction = (
        _unit_direction(hf_raw, label="HF-tail terminal 方向")
        if hf_tail_active
        else torch.zeros_like(latent_float32)
    )
    multiplier = latent_float32.new_tensor(strength_multiplier)
    lf_update = (
        lf_direction * latent_l2 * _LF_RELATIVE_STRENGTH * multiplier
        if lf_active
        else torch.zeros_like(latent_float32)
    )
    hf_tail_update = (
        hf_tail_direction * latent_l2 * _HF_TAIL_RELATIVE_STRENGTH * multiplier
        if hf_tail_active
        else torch.zeros_like(latent_float32)
    )
    combined = lf_update + hf_tail_update
    written = (latent_float32 + combined).to(dtype=terminal_latent.dtype)
    actual = written.detach().to(dtype=torch.float32) - latent_float32
    actual_l2 = torch.linalg.vector_norm(actual.reshape(-1))
    if not bool(torch.isfinite(actual_l2)) or not bool(actual_l2 > 0.0):
        raise RuntimeError("terminal 内容载体没有产生实际 dtype 写入")
    return TerminalContentCarrierUpdate(
        written_latent=written,
        lf_direction=lf_direction,
        hf_tail_direction=hf_tail_direction,
        lf_update=lf_update,
        hf_tail_update=hf_tail_update,
        routing_mode=routing_mode,
        carrier_mode=carrier_mode,
        strength_multiplier=float(strength_multiplier),
        lf_effective_l2=float(torch.linalg.vector_norm(lf_update.reshape(-1)).item()),
        hf_tail_effective_l2=float(
            torch.linalg.vector_norm(hf_tail_update.reshape(-1)).item()
        ),
        combined_effective_l2=float(actual_l2.item()),
        combined_relative_l2=float((actual_l2 / latent_l2).item()),
    )


__all__ = [
    "TERMINAL_CONTENT_STRENGTH_MULTIPLIERS",
    "TerminalContentCarrierUpdate",
    "build_terminal_content_carrier_update",
]
