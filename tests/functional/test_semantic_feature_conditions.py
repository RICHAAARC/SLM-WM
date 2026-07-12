"""验证完整特征 Jacobian 与累计成图保持门禁。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from experiments.runtime.diffusion.semantic_features import (
    JOINT_FEATURE_WIDTH,
    SEMANTIC_FEATURE_SCHEMA,
    SEMANTIC_FEATURE_WIDTH,
    VISUAL_FEATURE_SCHEMA,
    VISUAL_FEATURE_WIDTH,
    DifferentiableSemanticFeatureRuntime,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    _combined_update_preservation_record,
    _final_image_preservation_record,
)
from main.methods.subspace import build_exact_jacobian_linearization


@pytest.mark.quick
def test_formal_jacobian_vector_keeps_every_clip_and_visual_coordinate() -> None:
    """正式 Jacobian 输入必须直接连接512维 CLIP 与204维视觉特征。"""

    semantic = torch.arange(SEMANTIC_FEATURE_WIDTH, dtype=torch.float32).reshape(1, -1)
    visual = torch.arange(VISUAL_FEATURE_WIDTH, dtype=torch.float32)
    runtime = SimpleNamespace(joint_features=lambda _latent: (semantic, visual))

    vector = DifferentiableSemanticFeatureRuntime.full_joint_feature_vector(
        runtime,
        torch.zeros(1, 1, 1, 1),
    )

    assert vector.shape == (JOINT_FEATURE_WIDTH,)
    assert torch.equal(vector[:SEMANTIC_FEATURE_WIDTH], semantic.reshape(-1))
    assert torch.equal(vector[SEMANTIC_FEATURE_WIDTH:], visual)


@pytest.mark.quick
def test_full_feature_schema_declares_no_compression() -> None:
    """完整特征 schema 必须冻结宽度并显式声明未压缩。"""

    runtime = object.__new__(DifferentiableSemanticFeatureRuntime)
    object.__setattr__(
        runtime,
        "vision_model",
        SimpleNamespace(config=SimpleNamespace(projection_dim=SEMANTIC_FEATURE_WIDTH)),
    )

    record = runtime.feature_schema_record()

    assert record == {
        "semantic_feature_schema": SEMANTIC_FEATURE_SCHEMA,
        "semantic_feature_width": SEMANTIC_FEATURE_WIDTH,
        "visual_feature_schema": VISUAL_FEATURE_SCHEMA,
        "visual_feature_width": VISUAL_FEATURE_WIDTH,
        "joint_feature_width": JOINT_FEATURE_WIDTH,
        "feature_compression_applied": False,
    }


@pytest.mark.quick
def test_full_visual_vector_preserves_spatial_and_gradient_information() -> None:
    """完整视觉向量必须保留通道统计、梯度与8x8空间池化。"""

    smooth = torch.full((1, 3, 16, 16), 0.5)
    checker = smooth.clone()
    checker[:, :, ::2, ::2] = 1.0
    checker[:, :, 1::2, 1::2] = 0.0
    spatial = smooth.clone()
    spatial[:, :, :, :8] = 0.25
    spatial[:, :, :, 8:] = 0.75

    smooth_features = DifferentiableSemanticFeatureRuntime._visual_features_from_image(
        smooth
    )
    checker_features = DifferentiableSemanticFeatureRuntime._visual_features_from_image(
        checker
    )
    spatial_features = DifferentiableSemanticFeatureRuntime._visual_features_from_image(
        spatial
    )

    assert smooth_features.shape == (VISUAL_FEATURE_WIDTH,)
    assert checker_features.shape == (VISUAL_FEATURE_WIDTH,)
    assert checker_features[3:6].mean() > smooth_features[3:6].mean()
    assert checker_features[6:9].mean() > smooth_features[6:9].mean()
    assert checker_features[9:12].mean() > smooth_features[9:12].mean()
    assert not torch.equal(
        spatial_features[12:],
        smooth_features[12:],
    )


@pytest.mark.quick
def test_complete_feature_vector_supports_exact_jvp_and_vjp() -> None:
    """716维完整输出必须同时支持精确 JVP 与 VJP。"""

    latent = torch.linspace(-1.0, 1.0, 16)
    projection = torch.arange(
        JOINT_FEATURE_WIDTH * latent.numel(),
        dtype=torch.float32,
    ).reshape(JOINT_FEATURE_WIDTH, latent.numel())
    projection = projection.remainder(17.0) / 17.0

    def full_features(values: torch.Tensor) -> torch.Tensor:
        return projection @ values

    linearization = build_exact_jacobian_linearization(full_features, latent)
    tangent = linearization.apply(torch.ones_like(latent))
    cotangent = linearization.transpose_apply(torch.ones(JOINT_FEATURE_WIDTH))

    assert linearization.output_width == JOINT_FEATURE_WIDTH
    assert tangent.shape == (JOINT_FEATURE_WIDTH,)
    assert cotangent.shape == latent.shape
    assert torch.allclose(tangent, projection @ torch.ones_like(latent))
    assert torch.allclose(cotangent, projection.transpose(0, 1) @ torch.ones(JOINT_FEATURE_WIDTH))


@pytest.mark.quick
def test_actual_combined_latent_uses_full_feature_preservation_gate() -> None:
    """有限更新门禁必须检查完整特征, 而不只信局部 Jacobian 残差。"""

    class _FeatureRuntime:
        """提供可精确控制的完整语义与视觉特征。"""

        @staticmethod
        def joint_features(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            values = latent.reshape(-1).float()
            semantic = torch.nn.functional.normalize(values[:2], dim=0)
            visual = torch.stack((values[0], values[1], values.square().mean()))
            return semantic, visual

    config = SemanticWatermarkRuntimeConfig(
        minimum_semantic_preservation_cosine=0.99,
        maximum_visual_feature_relative_drift=0.05,
    )
    latent = torch.tensor([1.0, 0.0, 0.5])
    accepted = _combined_update_preservation_record(
        _FeatureRuntime(),
        latent,
        torch.tensor([1.0, 0.001, 0.5]),
        config,
    )
    rejected = _combined_update_preservation_record(
        _FeatureRuntime(),
        latent,
        torch.tensor([0.0, 1.0, 0.5]),
        config,
    )

    assert accepted["semantic_preservation_gate_ready"] is True
    assert rejected["semantic_preservation_gate_ready"] is False


@pytest.mark.quick
def test_final_image_gate_checks_cumulative_clean_to_watermarked_drift() -> None:
    """最终门禁必须直接比较 clean 与 watermarked 成图的累计变化。"""

    class _ImageProcessor:
        """模拟正式预处理器返回 [-1, 1] tensor。"""

        @staticmethod
        def preprocess(image: torch.Tensor) -> torch.Tensor:
            return image * 2.0 - 1.0

    class _ImageFeatureRuntime:
        """从最终图像 tensor 生成可控制的语义与视觉特征。"""

        @staticmethod
        def joint_image_features(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            flat = image.reshape(-1).float()
            semantic = torch.nn.functional.normalize(flat[:3], dim=0)
            return semantic, flat

    pipeline = SimpleNamespace(
        _execution_device="cpu",
        image_processor=_ImageProcessor(),
    )
    config = SemanticWatermarkRuntimeConfig(
        minimum_semantic_preservation_cosine=0.99,
        maximum_visual_feature_relative_drift=0.05,
    )
    clean = torch.tensor([[[[1.0, 0.5], [0.25, 0.75]]]])
    close = clean + 0.001
    changed = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])

    accepted = _final_image_preservation_record(
        pipeline,
        _ImageFeatureRuntime(),
        clean,
        close,
        config,
    )
    rejected = _final_image_preservation_record(
        pipeline,
        _ImageFeatureRuntime(),
        clean,
        changed,
        config,
    )

    assert accepted["final_image_preservation_gate_ready"] is True
    assert rejected["final_image_preservation_gate_ready"] is False
