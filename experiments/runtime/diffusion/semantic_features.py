"""为真实 SD3/SD3.5 latent 构造可微语义与视觉特征。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from experiments.runtime.model_sources import require_registered_model_reference


def load_clip_vision_model(
    model_id: str,
    model_revision: str,
    device_name: str,
    torch_dtype: str = "float32",
) -> Any:
    """加载冻结的 CLIP 图像编码器, 用作真实语义特征映射。"""

    import torch
    from transformers import CLIPVisionModelWithProjection

    require_registered_model_reference(
        model_id,
        model_revision,
        required_usage_role="semantic_condition_encoder",
    )
    dtype = getattr(torch, torch_dtype)
    model = CLIPVisionModelWithProjection.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    model = model.to(device_name)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def freeze_module_parameters(module: Any) -> None:
    """冻结模块参数但保留输入梯度。"""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


@dataclass
class DifferentiableSemanticFeatureRuntime:
    """封装 VAE 解码、CLIP 语义特征和视觉质量特征。"""

    vae: Any
    vision_model: Any
    vision_input_size: int = 224

    def __post_init__(self) -> None:
        """冻结模型权重, 后续 autograd 只计算 latent 的方向导数。"""

        freeze_module_parameters(self.vae)
        freeze_module_parameters(self.vision_model)

    def decode_latent(self, latent: Any) -> Any:
        """按照 diffusers VAE 缩放约定将 latent 解码为 [0, 1] 图像 tensor。"""

        scaling_factor = float(getattr(self.vae.config, "scaling_factor", 1.0))
        shift_factor = float(getattr(self.vae.config, "shift_factor", 0.0) or 0.0)
        vae_dtype = next(self.vae.parameters()).dtype
        scaled_latent = (latent / scaling_factor + shift_factor).to(dtype=vae_dtype)
        decoded = self.vae.decode(scaled_latent, return_dict=False)[0]
        return (decoded.float() / 2.0 + 0.5).clamp(0.0, 1.0)

    def clip_pixels(self, image: Any) -> Any:
        """用可微插值和官方 CLIP 归一化常量准备图像输入。"""

        import torch
        import torch.nn.functional as functional

        resized = functional.interpolate(
            image,
            size=(self.vision_input_size, self.vision_input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=image.device).view(1, 3, 1, 1)
        std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=image.device).view(1, 3, 1, 1)
        normalized = (resized - mean) / std
        model_dtype = next(self.vision_model.parameters()).dtype
        return normalized.to(dtype=model_dtype)

    def vision_outputs(self, latent: Any) -> Any:
        """从 latent 端到端计算 CLIP 图像编码输出。"""

        image = self.decode_latent(latent)
        return self.vision_model(pixel_values=self.clip_pixels(image), output_hidden_states=True)

    def semantic_features(self, latent: Any) -> Any:
        """返回归一化 CLIP 全局语义嵌入, 作为语义 Jacobian 输出。"""

        import torch.nn.functional as functional

        outputs = self.vision_outputs(latent)
        image_embeds = getattr(outputs, "image_embeds", None)
        if image_embeds is None:
            image_embeds = outputs.pooler_output
        return functional.normalize(image_embeds.float(), dim=-1)

    def _semantic_features_from_image(self, image: Any) -> Any:
        """从已解码图像计算语义特征, 供联合 JVP 避免重复 VAE 解码。"""

        import torch.nn.functional as functional

        outputs = self.vision_model(pixel_values=self.clip_pixels(image), output_hidden_states=True)
        image_embeds = getattr(outputs, "image_embeds", None)
        if image_embeds is None:
            image_embeds = outputs.pooler_output
        return functional.normalize(image_embeds.float(), dim=-1)

    def visual_features(self, latent: Any) -> Any:
        """返回亮度、对比度、边缘和多尺度结构特征。"""

        image = self.decode_latent(latent)
        return self._visual_features_from_image(image)

    def _visual_features_from_image(self, image: Any) -> Any:
        """从已解码图像计算视觉质量特征。"""

        import torch
        import torch.nn.functional as functional

        horizontal = image[:, :, :, 1:] - image[:, :, :, :-1]
        vertical = image[:, :, 1:, :] - image[:, :, :-1, :]
        pooled = functional.adaptive_avg_pool2d(image, (8, 8))
        channel_mean = image.mean(dim=(-1, -2))
        channel_std = image.std(dim=(-1, -2), unbiased=False)
        return torch.cat(
            (
                channel_mean.reshape(-1),
                channel_std.reshape(-1),
                horizontal.abs().mean(dim=(-1, -2)).reshape(-1),
                vertical.abs().mean(dim=(-1, -2)).reshape(-1),
                pooled.reshape(-1),
            )
        )

    def joint_features(self, latent: Any) -> tuple[Any, Any]:
        """一次 VAE 解码同时返回语义与视觉特征, 用于块状真实 JVP。"""

        image = self.decode_latent(latent)
        return self._semantic_features_from_image(image), self._visual_features_from_image(image)

    @staticmethod
    def _condition_blocks(features: Any, block_count: int = 8) -> Any:
        """把真实特征向量汇总为固定数量的可微语义条件。"""

        import torch

        flattened = features.reshape(-1)
        if flattened.numel() < block_count:
            raise ValueError("特征宽度不足以构造语义条件分块")
        return torch.stack(tuple(block.mean() for block in torch.tensor_split(flattened, block_count)))

    def semantic_condition_features(self, latent: Any) -> Any:
        """返回 8 个 CLIP 嵌入分块均值, 作为显式语义保持条件。"""

        return self._condition_blocks(self.semantic_features(latent))

    def visual_condition_features(self, latent: Any) -> Any:
        """返回 8 个真实视觉统计分块均值, 作为质量保持条件。"""

        return self._condition_blocks(self.visual_features(latent))

    def joint_condition_features(self, latent: Any) -> tuple[Any, Any]:
        """一次 VAE 解码返回语义与视觉条件, 供精确 JVP 复用。"""

        semantic, visual = self.joint_features(latent)
        return self._condition_blocks(semantic), self._condition_blocks(visual)

    @staticmethod
    def _normalize_map(values: Any) -> Any:
        """逐样本把空间图归一化到 [0, 1]。"""

        flattened = values.flatten(1)
        minimum = flattened.min(dim=1).values.view(-1, 1, 1)
        maximum = flattened.max(dim=1).values.view(-1, 1, 1)
        return (values - minimum) / (maximum - minimum).clamp_min(1e-6)

    def branch_signal_maps(self, latent: Any, previous_latent: Any | None = None) -> dict[str, Any]:
        """从真实解码图像和 CLIP patch token 构造分支风险输入图。

        semantic 图表示 patch 与全局 CLS 语义的一致性; texture 图表示局部梯度;
        stability 图表示相邻注入时刻的图像稳定性。若没有上一时刻, 使用局部平滑
        一致性作为可复现的首时刻稳定性定义。
        """

        import torch.nn.functional as functional

        image = self.decode_latent(latent)
        outputs = self.vision_model(pixel_values=self.clip_pixels(image), output_hidden_states=True)
        tokens = outputs.last_hidden_state.float()
        patch_tokens = tokens[:, 1:, :]
        cls_token = functional.normalize(tokens[:, :1, :], dim=-1)
        normalized_patches = functional.normalize(patch_tokens, dim=-1)
        semantic_patch = (normalized_patches * cls_token).sum(dim=-1)
        patch_side = int(round(math.sqrt(semantic_patch.shape[1])))
        if patch_side * patch_side != semantic_patch.shape[1]:
            raise RuntimeError("CLIP patch token 数量无法还原为方形空间网格")
        semantic_map = semantic_patch.reshape(image.shape[0], 1, patch_side, patch_side)
        semantic_map = functional.interpolate(
            semantic_map,
            size=latent.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        semantic_map = self._normalize_map(semantic_map)

        gray = image.mean(dim=1, keepdim=True)
        horizontal = functional.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        vertical = functional.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        texture_map = functional.interpolate(
            horizontal + vertical,
            size=latent.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        texture_map = self._normalize_map(texture_map)

        local_mean = functional.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        saliency_map = functional.interpolate(
            (gray - local_mean).abs(),
            size=latent.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        saliency_map = self._normalize_map(saliency_map)

        if previous_latent is None:
            stability_map = 1.0 - self._normalize_map(
                functional.interpolate(
                    (gray - local_mean).abs(),
                    size=latent.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
            )
        else:
            previous_image = self.decode_latent(previous_latent)
            difference = (image - previous_image).abs().mean(dim=1, keepdim=True)
            instability_map = functional.interpolate(
                difference,
                size=latent.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            stability_map = 1.0 - self._normalize_map(instability_map)
        return {
            "semantic": semantic_map.detach(),
            "texture": texture_map.detach(),
            "stability": stability_map.detach(),
            "saliency": saliency_map.detach(),
        }
