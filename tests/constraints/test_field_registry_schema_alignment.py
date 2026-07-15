"""验证正式持久化 schema 与字段注册表保持同步."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from experiments.artifacts.detection_score_curves import (
    CURVE_POINT_FIELDNAMES,
    SCORE_DISTRIBUTION_FIELDNAMES,
)
from experiments.protocol.paper_fixed_fpr import (
    PAPER_REQUIRED_METRIC_FIELDS,
    PAPER_REQUIRED_SOURCE_FIELDS,
)
from experiments.runners.semantic_watermark_runtime import (
    SemanticWatermarkRuntimeConfig,
    SemanticWatermarkRuntimeResult,
)
from paper_experiments.analysis.paper_artifact_data_validation import (
    ABLATION_DELTA_FIELDS,
    ABLATION_METRIC_FIELDS,
    ATTACK_METRIC_FIELDS,
    BASELINE_COMPARISON_FIELDS,
    DATASET_QUALITY_FIELDS,
    FROZEN_PROTOCOL_FIELDS,
    TEST_METRIC_FIELDS,
)
from paper_experiments.baselines.primary_evidence import (
    PrimaryBaselineEvidenceRecord,
)
from paper_experiments.baselines.gaussian_shading_official_reference import (
    GaussianShadingOfficialReferenceRecord,
)
from paper_experiments.baselines.shallow_diffuse_official_reference import (
    ShallowDiffuseOfficialReferenceRecord,
)
from paper_experiments.baselines.tree_ring_official_reference import (
    TreeRingOfficialReferenceRecord,
)


pytestmark = pytest.mark.quick

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DATA_CHECK_FIELDS = {
    "frozen_evidence_protocol_ready",
    "raw_image_only_detection_records_ready",
    "test_detection_metrics_ready",
    "score_distribution_table_ready",
    "roc_curve_points_ready",
    "det_curve_points_ready",
    "attack_family_metrics_ready",
    "baseline_comparison_table_ready",
    "mechanism_ablation_metrics_ready",
    "mechanism_pairwise_delta_ready",
    "dataset_quality_metrics_ready",
    "ready_flag_consistency_ready",
}
FORMAL_PERSISTED_SCHEMA_FIELDS = set().union(
    PAPER_REQUIRED_METRIC_FIELDS,
    PAPER_REQUIRED_SOURCE_FIELDS,
    FROZEN_PROTOCOL_FIELDS,
    TEST_METRIC_FIELDS,
    ATTACK_METRIC_FIELDS,
    BASELINE_COMPARISON_FIELDS,
    ABLATION_METRIC_FIELDS,
    ABLATION_DELTA_FIELDS,
    DATASET_QUALITY_FIELDS,
    SCORE_DISTRIBUTION_FIELDNAMES,
    CURVE_POINT_FIELDNAMES,
)
SEMANTIC_WATERMARK_RUNTIME_DERIVED_CONFIG_FIELDS = {
    "key_material_digest_random",
    "method_definition",
    "method_definition_digest",
}


def _registered_fields() -> set[str]:
    """读取 Markdown 字段表首列并返回已登记名称集合."""

    result = set()
    for line in (ROOT / "docs" / "field_registry.md").read_text(
        encoding="utf-8"
    ).splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) >= 3 and cells[1] and cells[1] != "field_name":
            result.add(cells[1])
    return result


def test_primary_baseline_evidence_fields_are_registered() -> None:
    """主表 baseline 持久化记录不得出现未登记字段."""

    registered = _registered_fields()
    persisted = {field.name for field in fields(PrimaryBaselineEvidenceRecord)}

    assert persisted <= registered
    assert not {
        "adapter_smoke_ready",
        "adapter_smoke_ready_count",
        "adapter_smoke_ready_ids",
        "adapter_smoke_observation_count",
        "adapter_smoke_execution_devices",
        "adapter_smoke_sample_roles",
        "adapter_smoke_latent_shapes",
    } & registered


def test_official_reference_record_fields_are_registered() -> None:
    """三套官方参考受治理记录不得持久化未登记字段。"""

    persisted = set().union(
        *(set(field.name for field in fields(record_type)) for record_type in (
            TreeRingOfficialReferenceRecord,
            GaussianShadingOfficialReferenceRecord,
            ShallowDiffuseOfficialReferenceRecord,
        ))
    )

    assert persisted <= _registered_fields()


def test_artifact_data_check_fields_are_registered() -> None:
    """12个固定表图内容检查字段必须全部登记."""

    assert ARTIFACT_DATA_CHECK_FIELDS <= _registered_fields()


def test_formal_persisted_schema_fields_are_registered() -> None:
    """正式协议与论文表格持久化 schema 不得出现未登记字段."""

    assert FORMAL_PERSISTED_SCHEMA_FIELDS <= _registered_fields()


def test_semantic_watermark_runtime_public_fields_are_registered() -> None:
    """方法运行持久化配置和结果字段必须与字段注册表同步."""

    config_fields = {
        field.name
        for field in fields(SemanticWatermarkRuntimeConfig)
        if field.name != "key_material"
    }
    persisted_config_fields = (
        config_fields | SEMANTIC_WATERMARK_RUNTIME_DERIVED_CONFIG_FIELDS
    )
    result_fields = {
        field.name for field in fields(SemanticWatermarkRuntimeResult)
    }

    registered = _registered_fields()
    assert "key_material" not in persisted_config_fields
    assert persisted_config_fields <= registered
    assert result_fields <= registered
