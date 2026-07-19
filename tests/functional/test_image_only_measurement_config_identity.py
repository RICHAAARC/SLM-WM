"""验证仅图像盲检测量配置身份及摘要门禁。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    _build_image_only_measurement_config,
)
from main.core.digest import build_stable_digest
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
    build_high_frequency_tail_template,
    build_low_frequency_template,
)
from main.methods.detection import (
    ImageOnlyMeasurementConfig,
    measure_image_only_watermark,
    image_only_measurement_config_identity_record,
    validate_image_only_measurement_digest_record,
)
from main.methods.geometry import (
    ATTENTION_COORDINATE_CONVENTION,
    ATTENTION_GRID_ALIGN_CORNERS,
    ATTENTION_OPERATOR_SCHEDULE_INDEX,
    FROZEN_SD35_ATTENTION_MODULE_NAMES,
)


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
        public_detection_schedule_index=ATTENTION_OPERATOR_SCHEDULE_INDEX,
        public_detection_noise_prg_protocol=KEYED_PRG_VERSION,
        public_detection_noise_domain="slm_wm_public_detection_noise",
        public_detection_conditioning_protocol="sd3_three_encoder_empty_text",
        public_detection_condition_text="",
        max_attention_tokens=1024,
        attention_coordinate_convention=ATTENTION_COORDINATE_CONVENTION,
        attention_grid_align_corners=ATTENTION_GRID_ALIGN_CORNERS,
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
        method_role="full_dual_chain",
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
        {"public_detection_noise_domain": "different_public_noise_domain"},
        {"public_detection_conditioning_protocol": "different_conditioning"},
        {"public_detection_condition_text": "different"},
        {"max_attention_tokens": 1000},
        {"attention_anchor_count": 13},
        {"attention_residual_threshold": 0.21},
        {"attention_minimum_inlier_ratio": 0.51},
        {
            "method_role": "lf_only_content",
            "lf_weight": 1.0,
            "tail_robust_weight": 0.0,
        },
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
    ((True, False), (False, False)),
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


def test_measurement_identity_rejects_alignment_without_attention_geometry() -> None:
    """仅启用配准不具备可计算的真实注意力几何输入。"""

    with pytest.raises(ValueError, match="真实注意力几何"):
        image_only_measurement_config_identity_record(
            _config(),
            attention_geometry_enabled=False,
        image_alignment_enabled=True,
    )


def test_generation_and_detection_rebuild_identical_formal_carriers() -> None:
    """嵌入与盲检必须从同一model/revision身份逐值重建LF/HF。"""

    runtime_config = SemanticWatermarkRuntimeConfig(key_material="identity-key")
    measurement_config = _build_image_only_measurement_config(runtime_config)
    generation_identity = build_stable_digest(
        {
            "model_id": runtime_config.model_id,
            "model_revision": runtime_config.model_revision,
        }
    )
    detection_identity = build_stable_digest(
        {
            "model_id": measurement_config.model_id,
            "model_revision": measurement_config.model_revision,
        }
    )
    reference = torch.arange(30, dtype=torch.float32).reshape(1, 2, 3, 5)

    assert measurement_config.model_id == runtime_config.model_id
    assert measurement_config.model_revision == runtime_config.model_revision
    assert detection_identity == generation_identity
    for builder in (
        build_low_frequency_template,
        build_high_frequency_tail_template,
    ):
        embedded = builder(
            reference,
            runtime_config.key_material,
            generation_identity,
            prg_version=runtime_config.keyed_prg_version,
        )
        detected = builder(
            reference,
            runtime_config.key_material,
            detection_identity,
            prg_version=measurement_config.keyed_prg_version,
        )
        assert embedded.template_digest == detected.template_digest
        assert torch.equal(embedded.template, detected.template)


@pytest.mark.parametrize(
    "changes",
    (
        {"public_detection_schedule_index": 8},
        {"attention_coordinate_convention": "different_coordinates"},
        {"attention_grid_align_corners": False},
    ),
)
def test_measurement_config_rejects_frozen_geometry_identity_drift(
    changes: dict[str, object],
) -> None:
    """检测时刻、token 坐标与图像采样约定不得成为可变参数。"""

    with pytest.raises(ValueError, match="冻结|注意力坐标"):
        replace(_config(), **changes)


def test_measurement_record_rejects_embedded_calibration_parameter() -> None:
    """原始测量记录不得重新携带 calibration 决策参数。"""

    torch = pytest.importorskip("torch")
    image = torch.linspace(-1.0, 1.0, steps=128).reshape(1, 2, 8, 8)
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
