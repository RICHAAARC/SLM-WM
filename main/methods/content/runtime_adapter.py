"""编排单次 Prompt 条件内容观测并映射到 latent 路由。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import struct
from typing import Any

import torch

from main.methods.content.latent_response import (
    LatentResponseResult,
    build_adjacent_latent_response_map,
)
from main.methods.content.local_sensitivity import (
    LocalSensitivityResult,
    build_public_probe_local_sensitivity_map,
)
from main.methods.content.routing import (
    ContentRoutingResult,
    _route_content_observations_to_latent,
)
from main.methods.content.saliency import (
    SemanticSaliencyResult,
    build_prompt_conditioned_semantic_saliency,
)
from main.methods.content.texture import (
    TextureResult,
    build_texture_complexity_map,
)


__all__ = [
    "ContentObservationRuntimeResult",
    "build_content_observation_routing",
]


@dataclass(frozen=True)
class ContentObservationRuntimeResult:
    """保存同一次内容观测的四项原语结果及其 latent 路由。"""

    semantic_saliency: SemanticSaliencyResult
    texture: TextureResult
    latent_response: LatentResponseResult
    local_sensitivity: LocalSensitivityResult
    routing: ContentRoutingResult


def _validate_latent_metadata(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a real floating Tensor")
    if value.ndim != 4 or value.shape[0] != 1:
        raise ValueError(f"{name} must have B=1 NCHW shape")
    if any(int(size) <= 0 for size in value.shape):
        raise ValueError(f"{name} dimensions must be positive")
    if value.device.type == "meta":
        raise ValueError(f"{name} must be materialized on a real device")
    return value


def _validate_image_metadata(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("decoded_current_image must be a Tensor")
    if not torch.is_floating_point(value):
        raise TypeError("decoded_current_image must be a real floating Tensor")
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 3:
        raise ValueError("decoded_current_image must have B=1 RGB NCHW shape")
    if any(int(size) <= 0 for size in value.shape):
        raise ValueError("decoded_current_image dimensions must be positive")
    if value.device.type == "meta":
        raise ValueError(
            "decoded_current_image must be materialized on a real device"
        )
    return value


def _validate_reference(name: str, value: Any) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    try:
        binary32_value = struct.unpack(">f", struct.pack(">f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"{name} must be exactly representable as binary32") from exc
    if binary32_value != value:
        raise ValueError(f"{name} must be exactly representable as binary32")
    return value


def _validate_runtime_callability(runtime: Any) -> None:
    if not callable(getattr(runtime, "encode_image_patch_features", None)):
        raise TypeError(
            "saliency_runtime must provide callable encode_image_patch_features"
        )
    if not callable(getattr(runtime, "encode_prompt_feature", None)):
        raise TypeError(
            "saliency_runtime must provide callable encode_prompt_feature"
        )


def _preflight_content_observations(
    *,
    previous_scheduler_latent: Any,
    current_scheduler_latent: Any,
    decoded_current_image: Any,
    prompt: Any,
    saliency_runtime: Any,
    vae_decoder: Any,
    reference_gradient: Any,
    reference_response: Any,
    reference_sensitivity: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
    """在任何 CLIP 或 VAE 调用前闭合跨组件静态边界。"""

    if type(prompt) is not str:
        raise TypeError("prompt must be an exact string")
    gradient = _validate_reference("reference_gradient", reference_gradient)
    response = _validate_reference("reference_response", reference_response)
    sensitivity = _validate_reference(
        "reference_sensitivity", reference_sensitivity
    )
    _validate_runtime_callability(saliency_runtime)
    if not callable(vae_decoder):
        raise TypeError("vae_decoder must be callable")

    previous = _validate_latent_metadata(
        "previous_scheduler_latent", previous_scheduler_latent
    )
    current = _validate_latent_metadata(
        "current_scheduler_latent", current_scheduler_latent
    )
    image = _validate_image_metadata(decoded_current_image)
    if previous.shape != current.shape:
        raise ValueError("scheduler latent shapes must match")
    if previous.device != current.device or current.device != image.device:
        raise ValueError(
            "scheduler latents and decoded_current_image must use the same device"
        )
    return previous, current, image, gradient, response, sensitivity


def build_content_observation_routing(
    *,
    previous_scheduler_latent: Any,
    current_scheduler_latent: Any,
    decoded_current_image: Any,
    prompt: str,
    saliency_runtime: Any,
    vae_decoder: Callable[[Any], Any],
    public_probe_identity: Any,
    reference_gradient: float,
    reference_response: float,
    reference_sensitivity: float,
) -> ContentObservationRuntimeResult:
    """以固定 R→T→S→Q 顺序构造一次内容观测及 latent 路由。"""

    previous, current, image, gradient, response, sensitivity = (
        _preflight_content_observations(
            previous_scheduler_latent=previous_scheduler_latent,
            current_scheduler_latent=current_scheduler_latent,
            decoded_current_image=decoded_current_image,
            prompt=prompt,
            saliency_runtime=saliency_runtime,
            vae_decoder=vae_decoder,
            reference_gradient=reference_gradient,
            reference_response=reference_response,
            reference_sensitivity=reference_sensitivity,
        )
    )

    latent_response = build_adjacent_latent_response_map(
        previous,
        current,
        response,
    )
    texture = build_texture_complexity_map(image, gradient)
    semantic_saliency = build_prompt_conditioned_semantic_saliency(
        image,
        prompt,
        saliency_runtime,
    )
    local_sensitivity = build_public_probe_local_sensitivity_map(
        current,
        image,
        vae_decoder,
        public_probe_identity,
        sensitivity,
    )
    routing = _route_content_observations_to_latent(
        semantic_saliency.saliency_map,
        texture.texture_map,
        latent_response.response_map,
        local_sensitivity.local_sensitivity_map,
    )
    return ContentObservationRuntimeResult(
        semantic_saliency=semantic_saliency,
        texture=texture,
        latent_response=latent_response,
        local_sensitivity=local_sensitivity,
        routing=routing,
    )
