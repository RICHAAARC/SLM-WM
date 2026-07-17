"""以冻结几何平均公式构造内容载体路由。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from main.core.digest import build_stable_digest, tensor_content_sha256


_ROUTING_SCHEMA_VERSION = "content_routing_result_v1"
_ROUTING_FORMULA_VERSION = "semantic_saliency_adaptive_content_routing_v1"


@dataclass(frozen=True)
class ContentRoutingResult:
    """保存可写容量图、两个内容掩码及其联合身份。"""

    writable_capacity_map: Any
    lf_mask: Any
    hf_tail_mask: Any
    routing_identity_digest: str


def _validate_routing_metadata(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a real floating Tensor")
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [1, 1, H, W]")
    if any(int(size) <= 0 for size in value.shape):
        raise ValueError(f"{name} dimensions must be positive")
    return value


def _validate_routing_contents(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    if bool(((value < 0.0) | (value > 1.0)).any().item()):
        raise ValueError(f"{name} values must lie in [0, 1]")


def _resize_content_map_to_latent(
    content_map: Any,
    latent_spatial_shape: Any,
) -> torch.Tensor:
    """按唯一内容图协议将单通道空间图映射到 latent H×W。"""

    resolved_map = _validate_routing_metadata("content_map", content_map)
    if (
        not isinstance(latent_spatial_shape, (tuple, list))
        or len(latent_spatial_shape) != 2
        or any(type(size) is not int or size <= 0 for size in latent_spatial_shape)
    ):
        raise ValueError(
            "latent_spatial_shape must contain exactly two positive integers"
        )
    _validate_routing_contents("content_map", resolved_map)
    content_float = resolved_map.to(dtype=torch.float32)
    _validate_routing_contents("content_map after float32 cast", content_float)
    target_shape = tuple(latent_spatial_shape)
    resized = functional.interpolate(
        content_float,
        size=target_shape,
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    if resized.shape != (1, 1, *target_shape):
        raise ValueError("resized content map does not match latent_spatial_shape")
    _validate_routing_contents("resized content_map", resized)
    return resized


def route_content_carriers(
    saliency_map: Any,
    texture_map: Any,
    response_map: Any,
    local_sensitivity_map: Any,
) -> ContentRoutingResult:
    """构造容量 ``A`` 以及互补的 LF 与 HF-tail 空间掩码。"""

    named_maps = (
        ("saliency_map", _validate_routing_metadata("saliency_map", saliency_map)),
        ("texture_map", _validate_routing_metadata("texture_map", texture_map)),
        ("response_map", _validate_routing_metadata("response_map", response_map)),
        (
            "local_sensitivity_map",
            _validate_routing_metadata(
                "local_sensitivity_map", local_sensitivity_map
            ),
        ),
    )
    expected_shape = named_maps[0][1].shape
    expected_device = named_maps[0][1].device
    if any(value.shape != expected_shape for _, value in named_maps[1:]):
        raise ValueError("content routing map shapes must match")
    if any(value.device != expected_device for _, value in named_maps[1:]):
        raise ValueError("content routing maps must use the same device")

    for name, value in named_maps:
        _validate_routing_contents(name, value)
    float_maps = tuple(value.to(dtype=torch.float32) for _, value in named_maps)
    for (name, _), value in zip(named_maps, float_maps, strict=True):
        _validate_routing_contents(f"{name} after float32 cast", value)
    saliency, texture, response, local_sensitivity = float_maps

    writable_capacity_map = torch.pow(
        (1.0 - saliency) * (1.0 - response) * (1.0 - local_sensitivity),
        1.0 / 3.0,
    )
    lf_mask = writable_capacity_map * (1.0 - texture)
    hf_tail_mask = writable_capacity_map * texture
    for name, value in (
        ("writable_capacity_map", writable_capacity_map),
        ("lf_mask", lf_mask),
        ("hf_tail_mask", hf_tail_mask),
    ):
        _validate_routing_contents(name, value)

    input_digests = {
        name: tensor_content_sha256(value) for name, value in named_maps
    }
    output_digests = {
        "writable_capacity_map": tensor_content_sha256(writable_capacity_map),
        "lf_mask": tensor_content_sha256(lf_mask),
        "hf_tail_mask": tensor_content_sha256(hf_tail_mask),
    }
    routing_identity_digest = build_stable_digest(
        {
            "schema_version": _ROUTING_SCHEMA_VERSION,
            "formula_version": _ROUTING_FORMULA_VERSION,
            "input_tensor_digests": input_digests,
            "output_tensor_digests": output_digests,
        }
    )
    return ContentRoutingResult(
        writable_capacity_map=writable_capacity_map,
        lf_mask=lf_mask,
        hf_tail_mask=hf_tail_mask,
        routing_identity_digest=routing_identity_digest,
    )
