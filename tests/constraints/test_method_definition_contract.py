"""冻结构造式方法定义与“潜流形”术语边界。"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    semantic_watermark_runtime_config_payload,
)
from main.methods.method_definition import (
    METHOD_DEFINITION_SCHEMA,
    semantic_conditioned_latent_method_definition,
    semantic_conditioned_latent_method_definition_digest,
)


ROOT = Path(__file__).resolve().parents[2]
METHOD_DOCUMENT = (
    ROOT
    / "docs"
    / "builds"
    / "method_section_semantic_conditioned_latent_manifold_watermark.md"
)
PRIMITIVE_DOCUMENT = (
    ROOT
    / "docs"
    / "builds"
    / "algorithm_primitives_semantic_conditioned_latent_manifold_watermark.md"
)
FIELD_REGISTRY = ROOT / "docs" / "field_registry.md"
EXPECTED_METHOD_DEFINITION_DIGEST = (
    "c0ff8777eb21c97ccca4725c3be30f73ed2948ab98143fdcd1a1d17a916618c3"
)


@pytest.mark.constraint
def test_machine_readable_method_definition_freezes_constructive_semantics() -> None:
    """可机读记录必须拒绝联合优化和全局流形的过强解释."""

    definition = semantic_conditioned_latent_method_definition()

    assert definition["method_definition_schema"] == METHOD_DEFINITION_SCHEMA
    assert definition["update_construction"]["joint_argmax_solved"] is False
    assert (
        definition["local_geometry"]["numerical_object"]
        == "kernel_of_local_feature_jacobian"
    )
    assert (
        definition["local_geometry"]["global_nonlinear_manifold_constructed"]
        is False
    )
    assert (
        definition["local_geometry"]["constant_rank_condition_verified"]
        is False
    )
    assert semantic_conditioned_latent_method_definition_digest() == (
        EXPECTED_METHOD_DEFINITION_DIGEST
    )


@pytest.mark.constraint
def test_method_documents_define_local_tangent_constructive_protocol() -> None:
    """方法文档必须描述当前构造协议和局部切空间边界."""

    for document in (METHOD_DOCUMENT, PRIMITIVE_DOCUMENT):
        text = document.read_text(encoding="utf-8")

        assert "构造式" in text
        assert "局部" in text and "特征水平集" in text and "切空间" in text
        assert "不构造全局非线性流形" in text
        assert "常秩" in text
        assert "\\arg\\max" not in text
        assert "\\beta_g" not in text
        assert "\\beta_s" not in text
        assert "\\beta_v" not in text
        assert "\\mathcal{M}_{\\mathrm{route}}" not in text


@pytest.mark.constraint
def test_field_registry_uses_numerical_basis_and_method_definition() -> None:
    """字段登记不得把数值 Null Space 基底误称为流形维度."""

    text = FIELD_REGISTRY.read_text(encoding="utf-8")

    assert "| basis_rank |" in text
    assert "| method_definition |" in text
    assert "| method_definition_digest |" in text
    assert "| manifold_dimension |" not in text


@pytest.mark.constraint
def test_runtime_config_identity_binds_method_definition() -> None:
    """单次科学运行身份必须绑定冻结的方法语义摘要."""

    payload = semantic_watermark_runtime_config_payload(
        SemanticWatermarkRuntimeConfig()
    )

    assert payload["method_definition"] == (
        semantic_conditioned_latent_method_definition()
    )
    assert payload["method_definition_digest"] == (
        EXPECTED_METHOD_DEFINITION_DIGEST
    )
