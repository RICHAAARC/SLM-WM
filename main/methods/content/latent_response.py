"""从相邻 scheduler latent 构造响应不稳定度图。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import torch

from main.core.digest import tensor_content_sha256


@dataclass(frozen=True)
class LatentResponseResult:
    """保存相邻 latent 响应图及其输入、输出身份。"""

    response_map: Any
    reference_response: float
    previous_latent_digest: str
    current_latent_digest: str
    response_map_digest: str


def _validate_latent_metadata(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a real floating Tensor")
    if value.ndim != 4 or value.shape[0] != 1:
        raise ValueError(f"{name} must have B=1 NCHW shape")
    if any(int(size) <= 0 for size in value.shape):
        raise ValueError(f"{name} dimensions must be positive")
    return value


def _validate_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def _stable_channel_rms(name: str, value: torch.Tensor) -> torch.Tensor:
    scale = torch.amax(torch.abs(value), dim=1, keepdim=True)
    safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    normalized = value / safe_scale
    _validate_finite_tensor(f"{name} normalized values", normalized)
    rms = scale * torch.sqrt(torch.mean(torch.square(normalized), dim=1, keepdim=True))
    _validate_finite_tensor(name, rms)
    return rms


def _validate_reference_response(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("reference_response must be a real scalar")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("reference_response must be finite and strictly positive")
    return resolved


def _measure_adjacent_latent_relative_response(
    previous_float: torch.Tensor,
    current_float: torch.Tensor,
) -> torch.Tensor:
    """测量未按共享 reference 归一化的相邻 latent 相对响应。"""

    latent_difference = current_float - previous_float
    _validate_finite_tensor("latent_difference", latent_difference)
    difference_rms = _stable_channel_rms("difference_rms", latent_difference)
    current_rms = _stable_channel_rms("current_rms", current_float)
    previous_rms = _stable_channel_rms("previous_rms", previous_float)
    response_denominator = current_rms + previous_rms + 1.0e-12
    _validate_finite_tensor("response_denominator", response_denominator)
    relative_response = difference_rms / response_denominator
    _validate_finite_tensor("relative_response", relative_response)
    return relative_response


def build_adjacent_latent_response_map(
    previous_scheduler_latent: Any,
    current_scheduler_latent: Any,
    reference_response: float,
) -> LatentResponseResult:
    """按冻结跨通道 RMS 相对变化公式构造 ``[1, 1, H, W]`` 响应图。"""

    previous = _validate_latent_metadata(
        "previous_scheduler_latent", previous_scheduler_latent
    )
    current = _validate_latent_metadata(
        "current_scheduler_latent", current_scheduler_latent
    )
    reference = _validate_reference_response(reference_response)
    if previous.shape != current.shape:
        raise ValueError("scheduler latent shapes must match")
    if previous.device != current.device:
        raise ValueError("scheduler latents must use the same device")

    _validate_finite_tensor("previous_scheduler_latent", previous)
    _validate_finite_tensor("current_scheduler_latent", current)
    previous_float = previous.to(dtype=torch.float32)
    current_float = current.to(dtype=torch.float32)
    _validate_finite_tensor("previous_scheduler_latent after float32 cast", previous_float)
    _validate_finite_tensor("current_scheduler_latent after float32 cast", current_float)

    relative_response = _measure_adjacent_latent_relative_response(
        previous_float,
        current_float,
    )
    reference_normalized_response = relative_response / reference
    _validate_finite_tensor(
        "reference_normalized_response", reference_normalized_response
    )
    response_map = torch.clamp(reference_normalized_response, min=0.0, max=1.0)
    _validate_finite_tensor("response_map", response_map)

    return LatentResponseResult(
        response_map=response_map,
        reference_response=reference,
        previous_latent_digest=tensor_content_sha256(previous),
        current_latent_digest=tensor_content_sha256(current),
        response_map_digest=tensor_content_sha256(response_map),
    )
