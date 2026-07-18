"""提供冻结 CLIP 图文特征的内部语义显著性运行边界。"""

from __future__ import annotations

from typing import Any

import torch


_IMAGE_SIZE = 224
_PATCH_COUNT = 49
_PROJECTION_DIMENSION = 512


def _validate_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_unit_interval(name: str, value: torch.Tensor) -> None:
    _validate_finite_tensor(name, value)
    if bool(((value < 0.0) | (value > 1.0)).any().item()):
        raise ValueError(f"{name} values must lie in [0, 1]")


def _validate_image(image: Any) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a Tensor")
    if not torch.is_floating_point(image):
        raise TypeError("image must be a real floating Tensor")
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("image must have B=1 RGB NCHW shape")
    if any(int(size) <= 0 for size in image.shape):
        raise ValueError("image dimensions must be positive")
    _validate_unit_interval("image", image)
    canonical = image.detach().to(device="cpu", dtype=torch.float32).contiguous()
    _validate_unit_interval("image after canonical float32 cast", canonical)
    return canonical


def _validate_feature_tensor(
    name: str,
    value: Any,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    resolved = value.to(dtype=torch.float32)
    if tuple(resolved.shape) != expected_shape:
        raise ValueError(f"{name} has an invalid shape")
    _validate_finite_tensor(name, resolved)
    norms = torch.linalg.vector_norm(resolved, dim=-1, keepdim=True)
    _validate_finite_tensor(f"{name} L2 norm", norms)
    if bool((norms <= 0.0).any().item()):
        raise ValueError(f"{name} must have non-zero L2 norm")
    normalized = resolved / norms
    _validate_finite_tensor(f"normalized {name}", normalized)
    return normalized


class _PromptSaliencyClipRuntime:
    """封装已验证的冻结 CLIP processor、tokenizer 与双塔模型。"""

    def __init__(
        self,
        *,
        model: Any,
        image_processor: Any,
        tokenizer: Any,
        device_name: str,
        model_identity_digest: str,
    ) -> None:
        if type(device_name) is not str or not device_name:
            raise ValueError("device_name must be a non-empty exact string")
        if (
            type(model_identity_digest) is not str
            or len(model_identity_digest) != 64
        ):
            raise ValueError("model_identity_digest must be a SHA-256 digest")
        self._model = model
        self._image_processor = image_processor
        self._tokenizer = tokenizer
        self._device = torch.device(device_name)
        self._model_identity_digest = model_identity_digest

    @property
    def model_identity_digest(self) -> str:
        """返回模型、依赖锁、预处理与特征协议的联合摘要。"""

        return self._model_identity_digest

    def prepare_image_pixels(self, image: Any) -> torch.Tensor:
        """按唯一 ``[0,1]`` range bridge 构造 CLIP pixel values。"""

        canonical = _validate_image(image)
        processed = self._image_processor(
            images=canonical[0],
            input_data_format="channels_first",
            return_tensors="pt",
            do_rescale=False,
        )
        try:
            pixel_values = processed["pixel_values"]
        except (KeyError, TypeError) as exc:
            raise ValueError("image processor must return pixel_values") from exc
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError("pixel_values must be a Tensor")
        if pixel_values.dtype != torch.float32:
            raise ValueError("pixel_values must use float32")
        if tuple(pixel_values.shape) != (1, 3, _IMAGE_SIZE, _IMAGE_SIZE):
            raise ValueError("pixel_values must have shape [1, 3, 224, 224]")
        _validate_finite_tensor("pixel_values", pixel_values)
        placed = pixel_values.to(device=self._device, dtype=torch.float32)
        _validate_finite_tensor("placed pixel_values", placed)
        return placed

    def encode_image_patch_features(self, image: Any) -> torch.Tensor:
        """返回去 CLS 后逐 token post-layernorm 与视觉投影的 patch 特征。"""

        pixel_values = self.prepare_image_pixels(image)
        with torch.inference_mode():
            outputs = self._model.vision_model(
                pixel_values=pixel_values,
                return_dict=True,
            )
            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            if not isinstance(last_hidden_state, torch.Tensor):
                raise TypeError("vision_model must return last_hidden_state")
            if (
                last_hidden_state.ndim != 3
                or last_hidden_state.shape[0] != 1
                or last_hidden_state.shape[1] != _PATCH_COUNT + 1
            ):
                raise ValueError("vision last_hidden_state must contain CLS plus 49 patches")
            _validate_finite_tensor("vision last_hidden_state", last_hidden_state)
            patch_tokens = last_hidden_state[:, 1:, :]
            post_layernorm_tokens = self._model.vision_model.post_layernorm(
                patch_tokens
            )
            _validate_finite_tensor(
                "post-layernorm patch tokens", post_layernorm_tokens
            )
            projected = self._model.visual_projection(post_layernorm_tokens)
        return _validate_feature_tensor(
            "projected patch features",
            projected,
            (1, _PATCH_COUNT, _PROJECTION_DIMENSION),
        )

    def tokenize_prompt(self, prompt: Any) -> dict[str, torch.Tensor]:
        """按冻结 tokenizer 调用构造精确77 token的文本输入。"""

        if type(prompt) is not str:
            raise TypeError("prompt must be an exact string")
        encoded = self._tokenizer(
            prompt,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        try:
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "tokenizer must return input_ids and attention_mask"
            ) from exc
        if not isinstance(input_ids, torch.Tensor) or not isinstance(
            attention_mask, torch.Tensor
        ):
            raise TypeError("tokenizer outputs must be Tensors")
        if tuple(input_ids.shape) != (1, 77) or tuple(attention_mask.shape) != (
            1,
            77,
        ):
            raise ValueError("tokenizer outputs must have shape [1, 77]")
        if input_ids.dtype != torch.int64 or attention_mask.dtype != torch.int64:
            raise TypeError("tokenizer outputs must use int64")
        return {
            "input_ids": input_ids.to(device=self._device),
            "attention_mask": attention_mask.to(device=self._device),
        }

    def encode_prompt_feature(self, prompt: Any) -> torch.Tensor:
        """返回标准 text-model pooler output 经 text projection 的特征。"""

        encoded = self.tokenize_prompt(prompt)
        with torch.inference_mode():
            outputs = self._model.text_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                return_dict=True,
            )
            pooled = getattr(outputs, "pooler_output", None)
            if not isinstance(pooled, torch.Tensor):
                raise TypeError("text_model must return pooler_output")
            if tuple(pooled.shape[:1]) != (1,) or pooled.ndim != 2:
                raise ValueError("text pooler_output must have B=1 matrix shape")
            _validate_finite_tensor("text pooler_output", pooled)
            projected = self._model.text_projection(pooled)
        return _validate_feature_tensor(
            "projected prompt feature",
            projected,
            (1, _PROJECTION_DIMENSION),
        )
