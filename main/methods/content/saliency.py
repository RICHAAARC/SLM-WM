"""从冻结 CLIP patch-text 特征构造 Prompt 条件语义显著图。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from main.core.digest import tensor_content_sha256


_PATCH_COUNT = 49
_PATCH_GRID_SIZE = 7
_PROJECTION_DIMENSION = 512
_UNIT_NORM_RTOL = 1.0e-5
_UNIT_NORM_ATOL = 1.0e-6


@dataclass(frozen=True)
class SemanticSaliencyResult:
    """保存原始 patch 网格显著图、相关性及分离的输入身份。"""

    saliency_map: Any
    patch_relevance: Any
    image_feature_digest: str
    prompt_feature_digest: str
    saliency_map_digest: str
    model_identity_digest: str


def _validate_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_image_boundary(image: Any) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a Tensor")
    if not torch.is_floating_point(image):
        raise TypeError("image must be a real floating Tensor")
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("image must have B=1 RGB NCHW shape")
    if any(int(size) <= 0 for size in image.shape):
        raise ValueError("image dimensions must be positive")
    _validate_finite_tensor("image", image)
    if bool(((image < 0.0) | (image > 1.0)).any().item()):
        raise ValueError("image values must lie in [0, 1]")
    return image


def _validate_model_identity_digest(value: Any) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "runtime model_identity_digest must be a lowercase SHA-256 digest"
        )
    return value


def _validate_runtime_capabilities(runtime: Any) -> tuple[Any, Any, str]:
    image_encoder = getattr(runtime, "encode_image_patch_features", None)
    prompt_encoder = getattr(runtime, "encode_prompt_feature", None)
    if not callable(image_encoder):
        raise TypeError("runtime must provide callable encode_image_patch_features")
    if not callable(prompt_encoder):
        raise TypeError("runtime must provide callable encode_prompt_feature")
    model_identity_digest = _validate_model_identity_digest(
        getattr(runtime, "model_identity_digest", None)
    )
    return image_encoder, prompt_encoder, model_identity_digest


def _validate_feature_metadata(
    name: str,
    value: Any,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use float32")
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} has an invalid shape")
    return value


def _validate_unit_feature_contents(name: str, value: torch.Tensor) -> None:
    _validate_finite_tensor(name, value)
    norms = torch.linalg.vector_norm(value, dim=-1)
    _validate_finite_tensor(f"{name} L2 norm", norms)
    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        rtol=_UNIT_NORM_RTOL,
        atol=_UNIT_NORM_ATOL,
    ):
        raise ValueError(
            f"{name} must have unit L2 norm within the frozen tolerance"
        )


def build_prompt_conditioned_semantic_saliency(
    image: Any,
    prompt: str,
    runtime: Any,
) -> SemanticSaliencyResult:
    """从真实图像和 Prompt 的 CLIP patch-text 相关性构造空间显著图。"""

    resolved_image = _validate_image_boundary(image)
    if type(prompt) is not str:
        raise TypeError("prompt must be an exact string")
    image_encoder, prompt_encoder, model_identity_digest = (
        _validate_runtime_capabilities(runtime)
    )

    image_features = _validate_feature_metadata(
        "image_feature",
        image_encoder(resolved_image),
        (1, _PATCH_COUNT, _PROJECTION_DIMENSION),
    )
    _validate_unit_feature_contents("image_feature", image_features)

    prompt_features = _validate_feature_metadata(
        "prompt_feature",
        prompt_encoder(prompt),
        (1, _PROJECTION_DIMENSION),
    )
    if prompt_features.device != image_features.device:
        raise ValueError("image_feature and prompt_feature must use the same device")
    _validate_unit_feature_contents("prompt_feature", prompt_features)

    patch_products = image_features * prompt_features.unsqueeze(1)
    _validate_finite_tensor("patch_products", patch_products)
    patch_relevance = torch.sum(patch_products, dim=-1)
    _validate_finite_tensor("patch_relevance", patch_relevance)

    saliency_preclip = (
        patch_relevance.reshape(1, 1, _PATCH_GRID_SIZE, _PATCH_GRID_SIZE) + 1.0
    ) / 2.0
    _validate_finite_tensor("saliency_preclip", saliency_preclip)
    saliency_map = torch.clamp(saliency_preclip, min=0.0, max=1.0)
    _validate_finite_tensor("saliency_map", saliency_map)

    return SemanticSaliencyResult(
        saliency_map=saliency_map,
        patch_relevance=patch_relevance,
        image_feature_digest=tensor_content_sha256(image_features),
        prompt_feature_digest=tensor_content_sha256(prompt_features),
        saliency_map_digest=tensor_content_sha256(saliency_map),
        model_identity_digest=model_identity_digest,
    )
