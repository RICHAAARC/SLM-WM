"""验证仅图像盲检测量配置身份及摘要门禁。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from main.core.keyed_prg import KEYED_PRG_VERSION
from main.methods.carrier import (
    LOW_FREQUENCY_BOUNDARY_MODE,
    LOW_FREQUENCY_CEIL_MODE,
    LOW_FREQUENCY_COUNT_INCLUDE_PAD,
    LOW_FREQUENCY_DIVISOR_OVERRIDE,
    LOW_FREQUENCY_KERNEL_SIZE,
    LOW_FREQUENCY_PADDING,
    LOW_FREQUENCY_STRIDE,
    LowFrequencyCarrierConfig,
)
from main.methods.detection import (
    ImageOnlyMeasurementConfig,
    measure_image_only_watermark,
    image_only_measurement_config_identity_record,
    validate_image_only_measurement_digest_record,
)
from main.methods.geometry import FROZEN_SD35_ATTENTION_MODULE_NAMES


pytestmark = pytest.mark.quick


def _config() -> ImageOnlyMeasurementConfig:
    """构造显式覆盖每个正式字段的阈值无关测量配置。"""

    return ImageOnlyMeasurementConfig(
        model_id="detector-config-model",
        model_revision="1" * 40,
        vae_class_name="AutoencoderKL",
        transformer_class_name="SD3Transformer2DModel",
        scheduler_class_name="FlowMatchEulerDiscreteScheduler",
        vae_scaling_factor=1.5305,
        vae_shift_factor=0.0609,
        latent_torch_dtype="float16",
        width=512,
        height=512,
        inference_steps=28,
        public_detection_schedule_index=14,
        public_detection_noise_prg_protocol=KEYED_PRG_VERSION,
        public_detection_noise_domain="slm_wm_public_detection_noise_v1",
        public_detection_conditioning_protocol="sd3_three_encoder_empty_text_v1",
        public_detection_condition_text="",
        max_attention_tokens=1024,
        attention_coordinate_convention="normalized_xy_token_centers_corner_endpoints_v1",
        attention_grid_align_corners=True,
        attention_module_names=FROZEN_SD35_ATTENTION_MODULE_NAMES,
        attention_anchor_count=12,
        attention_residual_threshold=0.20,
        attention_minimum_inlier_ratio=0.50,
        low_frequency_config=LowFrequencyCarrierConfig(
            kernel_size=LOW_FREQUENCY_KERNEL_SIZE,
            stride=LOW_FREQUENCY_STRIDE,
            padding=LOW_FREQUENCY_PADDING,
            boundary_mode=LOW_FREQUENCY_BOUNDARY_MODE,
            ceil_mode=LOW_FREQUENCY_CEIL_MODE,
            count_include_pad=LOW_FREQUENCY_COUNT_INCLUDE_PAD,
            divisor_override=LOW_FREQUENCY_DIVISOR_OVERRIDE,
        ),
        lf_weight=0.70,
        tail_robust_weight=0.30,
        tail_fraction=0.20,
        keyed_prg_version=KEYED_PRG_VERSION,
        attention_stable_token_fraction=0.50,
        attention_unstable_pair_weight=0.25,
        attention_relation_component_weights=(0.25, 0.25, 0.25, 0.25),
    )


@pytest.mark.parametrize(
    ("changes"),
    (
        {"model_id": "detector-config-model-2"},
        {"model_revision": "2" * 40},
        {"vae_class_name": "DifferentVae"},
        {"transformer_class_name": "DifferentTransformer"},
        {"scheduler_class_name": "DifferentScheduler"},
        {"vae_scaling_factor": 1.5306},
        {"vae_shift_factor": 0.0610},
        {"latent_torch_dtype": "bfloat16"},
        {"width": 520},
        {"height": 520},
        {"inference_steps": 29},
        {"public_detection_schedule_index": 15},
        {"public_detection_noise_domain": "different_public_noise_domain"},
        {"public_detection_conditioning_protocol": "different_conditioning"},
        {"public_detection_condition_text": "different"},
        {"max_attention_tokens": 1000},
        {"attention_coordinate_convention": "different_coordinates"},
        {"attention_grid_align_corners": False},
        {"attention_anchor_count": 13},
        {"attention_residual_threshold": 0.21},
        {"attention_minimum_inlier_ratio": 0.51},
        {"lf_weight": 0.60, "tail_robust_weight": 0.40},
        {"tail_fraction": 0.21},
        {"attention_stable_token_fraction": 0.51},
        {"attention_unstable_pair_weight": 0.26},
        {
            "attention_relation_component_weights": (
                1.0 / 3.0,
                0.0,
                1.0 / 3.0,
                1.0 / 3.0,
            )
        },
    ),
)
def test_measurement_config_digest_binds_every_runtime_parameter(
    changes: dict[str, object],
) -> None:
    """任一有效测量参数变化都必须产生不同的配置摘要。"""

    baseline = image_only_measurement_config_identity_record(
        _config(),
        attention_geometry_enabled=True,
        image_alignment_enabled=True,
    )
    changed = image_only_measurement_config_identity_record(
        replace(_config(), **changes),
        attention_geometry_enabled=True,
        image_alignment_enabled=True,
    )

    assert changed["image_only_measurement_config_digest"] != baseline[
        "image_only_measurement_config_digest"
    ]


@pytest.mark.parametrize(
    ("attention_geometry_enabled", "image_alignment_enabled"),
    ((False, True), (True, False), (False, False)),
)
def test_measurement_config_digest_binds_mechanism_switches(
    attention_geometry_enabled: bool,
    image_alignment_enabled: bool,
) -> None:
    """消融机制开关必须属于测量身份而不是外部说明。"""

    baseline = image_only_measurement_config_identity_record(
        _config(),
        attention_geometry_enabled=True,
        image_alignment_enabled=True,
    )
    changed = image_only_measurement_config_identity_record(
        _config(),
        attention_geometry_enabled=attention_geometry_enabled,
        image_alignment_enabled=image_alignment_enabled,
    )

    assert changed["image_only_measurement_config_digest"] != baseline[
        "image_only_measurement_config_digest"
    ]


def test_measurement_record_rejects_embedded_calibration_parameter() -> None:
    """原始测量记录不得重新携带 calibration 决策参数。"""

    torch = pytest.importorskip("torch")
    image = torch.zeros(1, 2, 8, 8)
    record = measure_image_only_watermark(
        image=image,
        key_material="detector-config-key",
        config=_config(),
        image_latent_encoder=lambda value: value,
    ).to_record()
    validate_image_only_measurement_digest_record(record)

    drifted = deepcopy(record)
    drifted["metadata"]["content_threshold"] = 0.12
    with pytest.raises(ValueError, match="calibration 参数"):
        validate_image_only_measurement_digest_record(drifted)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("attention_residual_threshold", 1),
        ("attention_minimum_inlier_ratio", 1),
        ("attention_relation_component_weights", [0.25] * 4),
    ),
)
def test_measurement_config_rejects_implicit_numeric_or_container_types(
    field_name: str,
    invalid_value: object,
) -> None:
    """正式配置拒绝会掩盖序列化漂移的隐式类型转换。"""

    with pytest.raises(ValueError, match="精确 float"):
        replace(_config(), **{field_name: invalid_value})


def test_measurement_config_rejects_attention_layer_order_drift() -> None:
    """盲检配置不得交换或替换冻结的 SD3.5 注意力层顺序."""

    with pytest.raises(ValueError, match="冻结 SD3.5 层顺序"):
        replace(
            _config(),
            attention_module_names=tuple(
                reversed(FROZEN_SD35_ATTENTION_MODULE_NAMES)
            ),
        )
