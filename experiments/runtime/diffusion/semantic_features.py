"""为真实 SD3/SD3.5 latent 构造可微语义与视觉特征。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from experiments.runtime.model_sources import require_registered_model_reference


SEMANTIC_FEATURE_SCHEMA = "full_normalized_clip_embedding"
VISUAL_FEATURE_SCHEMA = "full_global_gradient_multiscale_visual_vector"
SEMANTIC_FEATURE_WIDTH = 512
VISUAL_FEATURE_WIDTH = 204
JOINT_FEATURE_WIDTH = SEMANTIC_FEATURE_WIDTH + VISUAL_FEATURE_WIDTH


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

    @staticmethod
    def _visual_features_from_image(image: Any) -> Any:
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
        """一次 VAE 解码同时返回未压缩语义与完整视觉特征。"""

        image = self.decode_latent(latent)
        return self.joint_image_features(image)

    def joint_image_features(self, image: Any) -> tuple[Any, Any]:
        """从 [0, 1] 图像 tensor 返回未压缩语义与完整视觉特征。"""

        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("完整图像特征要求 [batch, 3, height, width] tensor")
        return (
            self._semantic_features_from_image(image),
            self._visual_features_from_image(image),
        )

    def full_joint_feature_vector(self, latent: Any) -> Any:
        """返回进入正式 Jacobian 的716维完整特征向量。"""

        import torch

        semantic, visual = self.joint_features(latent)
        vector = torch.cat((semantic.reshape(-1).float(), visual.reshape(-1).float()))
        if latent.shape[0] == 1 and vector.numel() != JOINT_FEATURE_WIDTH:
            raise RuntimeError("完整 Jacobian 特征宽度与冻结 CLIP/视觉 schema 不一致")
        return vector

    def feature_schema_record(self) -> dict[str, Any]:
        """返回正式完整特征的模型身份、宽度与 schema。"""

        model_config = getattr(self.vision_model, "config", None)
        semantic_width = int(getattr(model_config, "projection_dim", 0) or 0)
        if semantic_width <= 0:
            projection = getattr(self.vision_model, "visual_projection", None)
            semantic_width = int(getattr(projection, "out_features", 0) or 0)
        if semantic_width != SEMANTIC_FEATURE_WIDTH:
            raise ValueError("冻结 CLIP 模型 projection_dim 与正式完整特征宽度不一致")
        return {
            "semantic_feature_schema": SEMANTIC_FEATURE_SCHEMA,
            "semantic_feature_width": semantic_width,
            "visual_feature_schema": VISUAL_FEATURE_SCHEMA,
            "visual_feature_width": VISUAL_FEATURE_WIDTH,
            "joint_feature_width": semantic_width + VISUAL_FEATURE_WIDTH,
            "feature_compression_applied": False,
        }

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
