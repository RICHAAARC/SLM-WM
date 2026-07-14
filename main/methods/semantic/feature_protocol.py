"""冻结语义条件 Jacobian 与风险信号共享的图像特征协议。"""

from __future__ import annotations

from typing import Any

from main.core.digest import build_stable_digest


SEMANTIC_FEATURE_PROTOCOL_SCHEMA = (
    "semantic_conditioned_latent_complete_feature_operator"
)
SEMANTIC_FEATURE_SCHEMA = "normalized_projected_clip_image_embedding"
HANDCRAFTED_STRUCTURE_FEATURE_SCHEMA = (
    "explicit_rgb_statistics_gradient_spatial_pool_structure_vector"
)
SEMANTIC_FEATURE_WIDTH = 512
HANDCRAFTED_STRUCTURE_FEATURE_WIDTH = 204
JOINT_FEATURE_WIDTH = SEMANTIC_FEATURE_WIDTH + HANDCRAFTED_STRUCTURE_FEATURE_WIDTH

CLIP_VISION_INPUT_SIZE = 224
CLIP_VISION_RESIZE_MODE = "bicubic"
CLIP_VISION_RESIZE_ALIGN_CORNERS = False
CLIP_VISION_RESIZE_ANTIALIAS = True
CLIP_VISION_CHANNEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_VISION_CHANNEL_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_PROJECTED_EMBEDDING_SOURCE = "image_embeds"
CLIP_TOKEN_SEQUENCE_SOURCE = "last_hidden_state"
CLIP_CLS_TOKEN_INDEX = 0
CLIP_PATCH_TOKEN_START_INDEX = 1
STRUCTURE_POOL_HEIGHT = 8
STRUCTURE_POOL_WIDTH = 8
STRUCTURE_FEATURE_COMPONENTS = (
    ("rgb_channel_mean", 3),
    ("rgb_channel_population_standard_deviation", 3),
    ("rgb_horizontal_absolute_gradient_mean", 3),
    ("rgb_vertical_absolute_gradient_mean", 3),
    ("rgb_adaptive_average_pool_channel_row_column", 192),
)


def semantic_feature_protocol_record() -> dict[str, Any]:
    """返回可摘要的完整716维特征算子定义。

    该记录只包含纯数据常量，因此 `main/` 的最小方法包可以独立发布。
    GPU 运行层必须消费这些常量，而不能在外层重新声明另一套预处理或坐标顺序。
    """

    payload = {
        "semantic_feature_protocol_schema": SEMANTIC_FEATURE_PROTOCOL_SCHEMA,
        "vae_latent_transform": (
            "latent_divide_scaling_factor_then_add_shift_factor"
        ),
        "vae_decoded_image_transform": (
            "decoded_divide_two_add_half_then_clamp_zero_one"
        ),
        "clip_vision_input_size": CLIP_VISION_INPUT_SIZE,
        "clip_vision_resize_mode": CLIP_VISION_RESIZE_MODE,
        "clip_vision_resize_align_corners": (
            CLIP_VISION_RESIZE_ALIGN_CORNERS
        ),
        "clip_vision_resize_antialias": CLIP_VISION_RESIZE_ANTIALIAS,
        "clip_vision_channel_mean": list(CLIP_VISION_CHANNEL_MEAN),
        "clip_vision_channel_std": list(CLIP_VISION_CHANNEL_STD),
        "clip_projected_embedding_source": CLIP_PROJECTED_EMBEDDING_SOURCE,
        "clip_projected_embedding_normalization": (
            "l2_normalize_last_dimension"
        ),
        "clip_token_sequence_source": CLIP_TOKEN_SEQUENCE_SOURCE,
        "clip_cls_token_index": CLIP_CLS_TOKEN_INDEX,
        "clip_patch_token_start_index": CLIP_PATCH_TOKEN_START_INDEX,
        "semantic_feature_schema": SEMANTIC_FEATURE_SCHEMA,
        "semantic_feature_width": SEMANTIC_FEATURE_WIDTH,
        "handcrafted_structure_feature_schema": (
            HANDCRAFTED_STRUCTURE_FEATURE_SCHEMA
        ),
        "handcrafted_structure_feature_width": (
            HANDCRAFTED_STRUCTURE_FEATURE_WIDTH
        ),
        "handcrafted_structure_feature_components": [
            {"component_name": component_name, "coordinate_count": width}
            for component_name, width in STRUCTURE_FEATURE_COMPONENTS
        ],
        "structure_pool_shape": [
            STRUCTURE_POOL_HEIGHT,
            STRUCTURE_POOL_WIDTH,
        ],
        "structure_tensor_flatten_order": (
            "batch_channel_row_column_with_single_sample"
        ),
        "joint_feature_concatenation_order": [
            "normalized_projected_clip_image_embedding",
            "explicit_structure_coordinates",
        ],
        "joint_feature_width": JOINT_FEATURE_WIDTH,
        "feature_compression_applied": False,
    }
    return {
        **payload,
        "semantic_feature_protocol_digest": build_stable_digest(payload),
    }


__all__ = [
    "CLIP_CLS_TOKEN_INDEX",
    "CLIP_PATCH_TOKEN_START_INDEX",
    "CLIP_PROJECTED_EMBEDDING_SOURCE",
    "CLIP_TOKEN_SEQUENCE_SOURCE",
    "CLIP_VISION_CHANNEL_MEAN",
    "CLIP_VISION_CHANNEL_STD",
    "CLIP_VISION_INPUT_SIZE",
    "CLIP_VISION_RESIZE_ALIGN_CORNERS",
    "CLIP_VISION_RESIZE_ANTIALIAS",
    "CLIP_VISION_RESIZE_MODE",
    "HANDCRAFTED_STRUCTURE_FEATURE_SCHEMA",
    "HANDCRAFTED_STRUCTURE_FEATURE_WIDTH",
    "JOINT_FEATURE_WIDTH",
    "SEMANTIC_FEATURE_PROTOCOL_SCHEMA",
    "SEMANTIC_FEATURE_SCHEMA",
    "SEMANTIC_FEATURE_WIDTH",
    "STRUCTURE_FEATURE_COMPONENTS",
    "STRUCTURE_POOL_HEIGHT",
    "STRUCTURE_POOL_WIDTH",
    "semantic_feature_protocol_record",
]
