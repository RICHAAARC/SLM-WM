"""冻结正式内容双链方法身份与核心接口边界。"""

from __future__ import annotations

from dataclasses import MISSING
import inspect
from pathlib import Path

import pytest

from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    semantic_watermark_runtime_config_payload,
)
from main.methods.carrier import build_low_frequency_template
from main.methods.detection import ImageOnlyMeasurementConfig
from main.methods.geometry import (
    FROZEN_SD35_ATTENTION_MODULE_NAMES,
    recover_attention_affine_alignment,
)
from main.methods.method_definition import (
    METHOD_DEFINITION_SCHEMA,
    semantic_conditioned_latent_method_definition,
    semantic_conditioned_latent_method_definition_digest,
)


ROOT = Path(__file__).resolve().parents[2]
METHOD_DOCUMENT = ROOT / "docs/builds/method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md"
PRIMITIVE_DOCUMENT = ROOT / "docs/builds/algorithm_primitives_content_adaptive_dual_carrier_latent_watermark.md"
EXPECTED_METHOD_DEFINITION_DIGEST = (
    "f2f69ad790dd1249063cca95733b890349cfdccbbef4ec22aa1df9fbe59cfcd7"
)


@pytest.mark.constraint
def test_machine_readable_method_definition_freezes_formal_dual_chain() -> None:
    """机器身份必须绑定正式内容观测、单写回、Q/K 与盲检测。"""

    definition = semantic_conditioned_latent_method_definition()
    assert METHOD_DEFINITION_SCHEMA == "slm_wm_content_dual_chain_definition_v1"
    assert definition["method_definition_schema"] == METHOD_DEFINITION_SCHEMA
    assert definition["method_name"] == (
        "content_adaptive_dual_carrier_latent_watermark"
    )
    assert definition["content_observations"]["ordered_observations"] == [
        "semantic_saliency",
        "texture_complexity",
        "adjacent_latent_response",
        "public_probe_local_sensitivity",
    ]
    update = definition["generation_update"]
    assert update["capture_index"] == 9
    assert update["write_index"] == 10
    assert update["write_count"] == 1
    assert update["actual_dtype_single_write"] is True
    assert update["legacy_multi_injection_allowed"] is False
    geometry = definition["attention_geometry"]
    assert geometry["relation_source"] == "direct_qk"
    assert geometry["attention_module_names"] == list(
        FROZEN_SD35_ATTENTION_MODULE_NAMES
    )
    assert geometry["jacobian_null_space_allowed"] is False
    assert geometry["jvp_vjp_allowed"] is False
    assert geometry["psd_cg_allowed"] is False
    detection = definition["blind_detection"]
    assert detection["templates"] == "unmasked_formal_lf_and_hf_tail"
    assert detection["geometry_score_can_directly_decide_positive"] is False
    assert semantic_conditioned_latent_method_definition_digest() == (
        EXPECTED_METHOD_DEFINITION_DIGEST
    )


@pytest.mark.constraint
def test_method_documents_define_formal_content_dual_chain() -> None:
    """两份无状态规范必须描述同一正式单写回内容双链。"""

    method_text = METHOD_DOCUMENT.read_text(encoding="utf-8")
    primitive_text = PRIMITIVE_DOCUMENT.read_text(encoding="utf-8")
    assert "唯一算法原语权威来源" in primitive_text
    assert "callback 索引10" in primitive_text
    assert "索引9只保存" in primitive_text
    assert "不得调用 JVP/VJP" in primitive_text
    assert "仅图像检测" in method_text
    assert "共同回溯" in method_text
    assert "GitNexus 影响面" not in method_text


@pytest.mark.constraint
def test_runtime_config_identity_binds_method_definition() -> None:
    """单次科学运行身份必须绑定当前正式方法摘要。"""

    payload = semantic_watermark_runtime_config_payload(
        SemanticWatermarkRuntimeConfig()
    )
    assert payload["method_definition"] == (
        semantic_conditioned_latent_method_definition()
    )
    assert payload["method_definition_digest"] == (
        EXPECTED_METHOD_DEFINITION_DIGEST
    )


@pytest.mark.constraint
def test_attention_alignment_gate_has_no_core_fallback_defaults() -> None:
    """核心检测和配准 API 必须要求调用方显式提供结构门禁。"""

    signature = inspect.signature(recover_attention_affine_alignment)
    for parameter_name in (
        "anchor_count",
        "residual_threshold",
        "minimum_inlier_ratio",
    ):
        assert signature.parameters[parameter_name].default is inspect.Parameter.empty
    for field_name in (
        "attention_anchor_count",
        "attention_residual_threshold",
        "attention_minimum_inlier_ratio",
    ):
        field = ImageOnlyMeasurementConfig.__dataclass_fields__[field_name]
        assert field.default is MISSING
        assert field.default_factory is MISSING


@pytest.mark.constraint
def test_low_frequency_carrier_has_no_core_fallback_defaults() -> None:
    """正式 LF 构造与检测配置必须显式接收完整协议。"""

    signature = inspect.signature(build_low_frequency_template)
    assert list(signature.parameters) == [
        "reference_latent",
        "key_material",
        "model_identity_digest",
        "prg_version",
    ]
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    assert (
        signature.parameters["prg_version"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    field = ImageOnlyMeasurementConfig.__dataclass_fields__["low_frequency_config"]
    assert field.default is MISSING
    assert field.default_factory is MISSING
