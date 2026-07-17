"""使用冻结 Sobel 协议构造纹理复杂度图。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import torch
import torch.nn.functional as functional

from main.core.digest import build_stable_digest, tensor_content_sha256


_TEXTURE_RESULT_SCHEMA_VERSION = "texture_result_v1"
_TEXTURE_FORMULA_PROTOCOL_VERSION = "frozen_rgb_sobel_texture_complexity_v1"
_TEXTURE_FORMULA_PROTOCOL = {
    "input_color_space": "rgb_0_1",
    "luminance_weights": [0.299, 0.587, 0.114],
    "sobel_kernel_x": [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ],
    "sobel_kernel_y": [
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ],
    "padding_mode": "replicate",
    "padding_pixels": 1,
    "stride": 1,
    "gradient_magnitude": "sqrt_gx_squared_plus_gy_squared",
    "normalization": "clip_gradient_magnitude_div_reference_gradient_0_1",
    "computation_dtype": "float32",
    "output_resolution": "input_rgb_spatial_resolution",
}


@dataclass(frozen=True)
class TextureResult:
    """保存原始 RGB 分辨率纹理图、冻结尺度和协议摘要。"""

    texture_map: Any
    reference_gradient: float
    texture_map_digest: str


def _validate_image_metadata(image: Any) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a Tensor")
    if not torch.is_floating_point(image):
        raise TypeError("image must be a real floating Tensor")
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("image must have B=1 RGB NCHW shape")
    if any(int(size) <= 0 for size in image.shape):
        raise ValueError("image dimensions must be positive")
    return image


def _validate_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_unit_interval(name: str, value: torch.Tensor) -> None:
    _validate_finite_tensor(name, value)
    if bool(((value < 0.0) | (value > 1.0)).any().item()):
        raise ValueError(f"{name} values must lie in [0, 1]")


def _validate_reference_gradient(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("reference_gradient must be a real scalar")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("reference_gradient must be finite and strictly positive")
    return resolved


def build_texture_complexity_map(
    image: Any,
    reference_gradient: float,
) -> TextureResult:
    """按冻结亮度、Sobel 和参考梯度公式构造原始分辨率纹理图。"""

    resolved_image = _validate_image_metadata(image)
    reference = _validate_reference_gradient(reference_gradient)
    _validate_unit_interval("image", resolved_image)

    image_float = resolved_image.to(dtype=torch.float32)
    _validate_unit_interval("image after float32 cast", image_float)
    reference_tensor = torch.tensor(
        reference,
        dtype=torch.float32,
        device=image_float.device,
    )
    if (
        not bool(torch.isfinite(reference_tensor).item())
        or not bool((reference_tensor > 0.0).item())
    ):
        raise ValueError(
            "reference_gradient must remain finite and strictly positive in float32"
        )

    luminance_weights = image_float.new_tensor(
        _TEXTURE_FORMULA_PROTOCOL["luminance_weights"]
    ).view(1, 3, 1, 1)
    luminance = torch.sum(image_float * luminance_weights, dim=1, keepdim=True)
    _validate_finite_tensor("luminance", luminance)

    kernel_x = image_float.new_tensor(
        _TEXTURE_FORMULA_PROTOCOL["sobel_kernel_x"]
    ).view(1, 1, 3, 3)
    kernel_y = image_float.new_tensor(
        _TEXTURE_FORMULA_PROTOCOL["sobel_kernel_y"]
    ).view(1, 1, 3, 3)
    padded = functional.pad(luminance, (1, 1, 1, 1), mode="replicate")
    gradient_x = functional.conv2d(padded, kernel_x, stride=1)
    gradient_y = functional.conv2d(padded, kernel_y, stride=1)
    _validate_finite_tensor("gradient_x", gradient_x)
    _validate_finite_tensor("gradient_y", gradient_y)

    gradient_squared = torch.square(gradient_x) + torch.square(gradient_y)
    _validate_finite_tensor("gradient_squared", gradient_squared)
    gradient_magnitude = torch.sqrt(gradient_squared)
    _validate_finite_tensor("gradient_magnitude", gradient_magnitude)
    reference_normalized_gradient = gradient_magnitude / reference_tensor
    _validate_finite_tensor(
        "reference_normalized_gradient", reference_normalized_gradient
    )
    texture_map = torch.clamp(reference_normalized_gradient, min=0.0, max=1.0)
    _validate_unit_interval("texture_map", texture_map)

    texture_map_digest = build_stable_digest(
        {
            "schema_version": _TEXTURE_RESULT_SCHEMA_VERSION,
            "formula_protocol_version": _TEXTURE_FORMULA_PROTOCOL_VERSION,
            "formula_protocol": _TEXTURE_FORMULA_PROTOCOL,
            "reference_gradient": reference,
            "texture_map_content_sha256": tensor_content_sha256(texture_map),
        }
    )
    return TextureResult(
        texture_map=texture_map,
        reference_gradient=reference,
        texture_map_digest=texture_map_digest,
    )
