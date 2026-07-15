"""为生成模型 latent 构造可微语义与手工结构约束特征。

该模块属于核心方法层。调用方负责注入已经加载的 VAE 与视觉编码器,
因此该实现不依赖实验配置、模型注册表或具体运行环境。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from main.methods.semantic.feature_protocol import (
    CLIP_CLS_TOKEN_INDEX,
    CLIP_PATCH_TOKEN_START_INDEX,
    CLIP_PROJECTED_EMBEDDING_SOURCE,
    CLIP_TOKEN_SEQUENCE_SOURCE,
    CLIP_VISION_CHANNEL_MEAN,
    CLIP_VISION_CHANNEL_STD,
    CLIP_VISION_INPUT_SIZE,
    CLIP_VISION_RESIZE_ALIGN_CORNERS,
    CLIP_VISION_RESIZE_ANTIALIAS,
    CLIP_VISION_RESIZE_MODE,
    HANDCRAFTED_STRUCTURE_FEATURE_SCHEMA,
    HANDCRAFTED_STRUCTURE_FEATURE_WIDTH,
    JOINT_FEATURE_WIDTH,
    SEMANTIC_FEATURE_PROTOCOL_SCHEMA,
    SEMANTIC_FEATURE_SCHEMA,
    SEMANTIC_FEATURE_WIDTH,
    STRUCTURE_POOL_HEIGHT,
    STRUCTURE_POOL_WIDTH,
    semantic_feature_protocol_record,
)


def freeze_module_parameters(module: Any) -> None:
    """冻结模块参数但保留输入梯度。"""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


@dataclass
class DifferentiableSemanticFeatureRuntime:
    """封装 VAE 解码、CLIP 语义特征和手工结构统计约束."""

    vae: Any
    vision_model: Any

    def __post_init__(self) -> None:
        """冻结模型权重, 后续 autograd 只计算 latent 的方向导数。"""

        freeze_module_parameters(self.vae)
        freeze_module_parameters(self.vision_model)

    def decode_latent(self, latent: Any) -> Any:
        """按照 diffusers VAE 缩放约定将 latent 解码为 [0, 1] 图像 tensor。"""

        import torch

        if not bool(torch.isfinite(latent).all()):
            raise RuntimeError("VAE 解码输入 latent 必须全部有限")
        scaling_factor = float(self.vae.config.scaling_factor)
        shift_factor = float(self.vae.config.shift_factor)
        vae_dtype = next(self.vae.parameters()).dtype
        scaled_latent = (latent / scaling_factor + shift_factor).to(dtype=vae_dtype)
        if not bool(torch.isfinite(scaled_latent).all()):
            raise RuntimeError("VAE 解码缩放后的 latent 必须全部有限")
        decoded = self.vae.decode(scaled_latent, return_dict=False)[0]
        image = (decoded.float() / 2.0 + 0.5).clamp(0.0, 1.0)
        if not bool(torch.isfinite(image).all()):
            raise RuntimeError("VAE 解码图像必须全部有限")
        return image

    def clip_pixels(self, image: Any) -> Any:
        """用可微插值和官方 CLIP 归一化常量准备图像输入。"""

        import torch
        import torch.nn.functional as functional

        resized = functional.interpolate(
            image,
            size=(CLIP_VISION_INPUT_SIZE, CLIP_VISION_INPUT_SIZE),
            mode=CLIP_VISION_RESIZE_MODE,
            align_corners=CLIP_VISION_RESIZE_ALIGN_CORNERS,
            antialias=CLIP_VISION_RESIZE_ANTIALIAS,
        )
        mean = torch.tensor(
            CLIP_VISION_CHANNEL_MEAN,
            device=image.device,
            dtype=resized.dtype,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            CLIP_VISION_CHANNEL_STD,
            device=image.device,
            dtype=resized.dtype,
        ).view(1, 3, 1, 1)
        normalized = (resized - mean) / std
        model_dtype = next(self.vision_model.parameters()).dtype
        return normalized.to(dtype=model_dtype)

    def vision_outputs(self, latent: Any) -> Any:
        """从 latent 端到端计算 CLIP 图像编码输出。"""

        image = self.decode_latent(latent)
        return self.vision_model(
            pixel_values=self.clip_pixels(image),
            output_hidden_states=True,
        )

    def semantic_features(self, latent: Any) -> Any:
        """返回归一化 CLIP 全局语义嵌入, 作为语义 Jacobian 输出。"""

        outputs = self.vision_outputs(latent)
        image_embeds = getattr(outputs, CLIP_PROJECTED_EMBEDDING_SOURCE, None)
        if image_embeds is None:
            raise RuntimeError("冻结 CLIP 图像模型没有返回投影后的 image_embeds")
        return self._normalize_projected_embedding(image_embeds)

    def _semantic_features_from_image(self, image: Any) -> Any:
        """从已解码图像计算语义特征, 供联合 JVP 避免重复 VAE 解码。"""

        outputs = self.vision_model(
            pixel_values=self.clip_pixels(image),
            output_hidden_states=True,
        )
        image_embeds = getattr(outputs, CLIP_PROJECTED_EMBEDDING_SOURCE, None)
        if image_embeds is None:
            raise RuntimeError("冻结 CLIP 图像模型没有返回投影后的 image_embeds")
        return self._normalize_projected_embedding(image_embeds)

    @staticmethod
    def _normalize_projected_embedding(image_embeds: Any) -> Any:
        """拒绝退化向量后执行最后一维 L2 归一化。"""

        import torch

        values = image_embeds.float()
        if values.ndim != 2 or int(values.shape[-1]) != SEMANTIC_FEATURE_WIDTH:
            raise RuntimeError("投影后 CLIP image_embeds 必须具有冻结的二维宽度")
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("投影后 CLIP image_embeds 必须全部有限")
        norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
        if not bool(torch.isfinite(norms).all()) or bool((norms == 0.0).any()):
            raise RuntimeError("投影后 CLIP image_embeds 必须具有有限非零能量")
        normalized = values / norms
        if not bool(torch.isfinite(normalized).all()):
            raise RuntimeError("投影后 CLIP image_embeds 归一化结果必须全部有限")
        return normalized

    def handcrafted_structure_features(self, latent: Any) -> Any:
        """返回固定204维 RGB 统计、梯度和8x8池化结构向量."""

        image = self.decode_latent(latent)
        return self._handcrafted_structure_features_from_image(image)

    @staticmethod
    def _handcrafted_structure_features_from_image(image: Any) -> Any:
        """从已解码图像计算范围明确的手工结构统计向量."""

        import torch
        import torch.nn.functional as functional

        if (
            image.ndim != 4
            or int(image.shape[0]) != 1
            or int(image.shape[1]) != 3
        ):
            raise ValueError("204维结构坐标要求单样本 RGB 图像 tensor")
        if not bool(torch.isfinite(image).all()):
            raise RuntimeError("204维结构坐标输入图像必须全部有限")
        horizontal = image[:, :, :, 1:] - image[:, :, :, :-1]
        vertical = image[:, :, 1:, :] - image[:, :, :-1, :]
        pooled = functional.adaptive_avg_pool2d(
            image,
            (STRUCTURE_POOL_HEIGHT, STRUCTURE_POOL_WIDTH),
        )
        channel_mean = image.mean(dim=(-1, -2))
        channel_std = image.std(dim=(-1, -2), unbiased=False)
        features = torch.cat(
            (
                channel_mean.reshape(-1),
                channel_std.reshape(-1),
                horizontal.abs().mean(dim=(-1, -2)).reshape(-1),
                vertical.abs().mean(dim=(-1, -2)).reshape(-1),
                pooled.reshape(-1),
            )
        )
        if (
            features.numel() != HANDCRAFTED_STRUCTURE_FEATURE_WIDTH
            or not bool(torch.isfinite(features).all())
        ):
            raise RuntimeError("204维结构坐标宽度或有限性不满足冻结协议")
        return features

    def joint_features(self, latent: Any) -> tuple[Any, Any]:
        """一次 VAE 解码同时返回 CLIP 语义与手工结构统计."""

        image = self.decode_latent(latent)
        return self.joint_image_features(image)

    def joint_image_features(self, image: Any) -> tuple[Any, Any]:
        """从 [0, 1] 图像 tensor 返回语义与手工结构统计."""

        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError("完整图像特征要求 [1, 3, height, width] tensor")
        return (
            self._semantic_features_from_image(image),
            self._handcrafted_structure_features_from_image(image),
        )

    def full_joint_feature_vector(self, latent: Any) -> Any:
        """返回进入正式 Jacobian 的716维完整特征向量。"""

        import torch

        semantic, structure = self.joint_features(latent)
        vector = torch.cat(
            (
                semantic.reshape(-1).float(),
                structure.reshape(-1).float(),
            )
        )
        if latent.ndim != 4 or latent.shape[0] != 1:
            raise ValueError("完整 Jacobian 特征算子只接受单样本 latent")
        if vector.numel() != JOINT_FEATURE_WIDTH:
            raise RuntimeError(
                "完整 Jacobian 特征宽度与冻结 CLIP/手工结构 schema 不一致"
            )
        if not bool(torch.isfinite(vector).all()):
            raise RuntimeError("完整 Jacobian 特征向量必须全部有限")
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
        protocol = semantic_feature_protocol_record()
        return {
            "semantic_feature_protocol_schema": (
                SEMANTIC_FEATURE_PROTOCOL_SCHEMA
            ),
            "semantic_feature_protocol_digest": protocol[
                "semantic_feature_protocol_digest"
            ],
            "semantic_feature_schema": SEMANTIC_FEATURE_SCHEMA,
            "semantic_feature_width": semantic_width,
            "handcrafted_structure_feature_schema": HANDCRAFTED_STRUCTURE_FEATURE_SCHEMA,
            "handcrafted_structure_feature_width": HANDCRAFTED_STRUCTURE_FEATURE_WIDTH,
            "joint_feature_width": semantic_width + HANDCRAFTED_STRUCTURE_FEATURE_WIDTH,
            "feature_compression_applied": False,
        }

    def branch_signal_maps(
        self,
        latent: Any,
        previous_step_latent: Any,
    ) -> dict[str, Any]:
        """从真实解码图像和 CLIP patch token 构造分支风险输入图。

        semantic 图表示 patch 与全局 CLS 语义的一致性; texture 图表示局部梯度;
        local_contrast_risk 图表示灰度相对5x5局部均值的绝对偏离;
        adjacent_step_stability 图表示相邻 scheduler 步解码 RGB 的真实稳定性.
        """

        import torch.nn.functional as functional

        if previous_step_latent is None:
            raise ValueError(
                "adjacent_step_stability 要求上一 scheduler 步的真实 latent"
            )
        image = self.decode_latent(latent)
        outputs = self.vision_model(
            pixel_values=self.clip_pixels(image),
            output_hidden_states=True,
        )
        tokens = getattr(outputs, CLIP_TOKEN_SEQUENCE_SOURCE, None)
        if tokens is None:
            raise RuntimeError("冻结 CLIP 图像模型没有返回风险信号所需 token 序列")
        tokens = tokens.float()
        patch_tokens = tokens[:, CLIP_PATCH_TOKEN_START_INDEX:, :]
        cls_token = functional.normalize(
            tokens[
                :,
                CLIP_CLS_TOKEN_INDEX : CLIP_CLS_TOKEN_INDEX + 1,
                :,
            ],
            dim=-1,
        )
        normalized_patches = functional.normalize(patch_tokens, dim=-1)
        semantic_patch = (
            (normalized_patches * cls_token).sum(dim=-1) + 1.0
        ) * 0.5
        patch_side = int(round(math.sqrt(semantic_patch.shape[1])))
        if patch_side * patch_side != semantic_patch.shape[1]:
            raise RuntimeError("CLIP patch token 数量无法还原为方形空间网格")
        semantic_map = semantic_patch.reshape(image.shape[0], 1, patch_side, patch_side)
        semantic_map = functional.interpolate(
            semantic_map,
            size=latent.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0].clamp(0.0, 1.0)

        gray = image.mean(dim=1, keepdim=True)
        horizontal = functional.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        vertical = functional.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        texture_map = functional.interpolate(
            (horizontal + vertical) * 0.5,
            size=latent.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0].clamp(0.0, 1.0)

        local_mean = functional.avg_pool2d(
            functional.pad(gray, (2, 2, 2, 2), mode="reflect"),
            kernel_size=5,
            stride=1,
        )
        local_contrast_risk_map = functional.interpolate(
            (gray - local_mean).abs(),
            size=latent.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0].clamp(0.0, 1.0)

        previous_image = self.decode_latent(previous_step_latent)
        difference = (image - previous_image).abs().mean(
            dim=1,
            keepdim=True,
        )
        adjacent_step_instability_map = functional.interpolate(
            difference,
            size=latent.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0].clamp(0.0, 1.0)
        adjacent_step_stability_map = 1.0 - adjacent_step_instability_map
        return {
            "semantic": semantic_map.detach(),
            "texture": texture_map.detach(),
            "adjacent_step_stability": adjacent_step_stability_map.detach(),
            "local_contrast_risk": local_contrast_risk_map.detach(),
            "current_decoded_rgb": image.detach(),
            "previous_step_decoded_rgb": previous_image.detach(),
            "clip_patch_tokens": patch_tokens.detach(),
            "clip_cls_token": tokens[
                :,
                CLIP_CLS_TOKEN_INDEX : CLIP_CLS_TOKEN_INDEX + 1,
                :,
            ].detach(),
        }
