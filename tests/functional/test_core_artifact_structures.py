"""验证 records 和 manifests 的最小结构。"""

from __future__ import annotations

import pytest

from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.method_objects import (
    AttentionAnchorSpec,
    DetectionEvidenceSpec,
    FusionDecisionSpec,
    LatentSubspaceSpec,
    SemanticConditionSpec,
    WatermarkCarrierSpec,
)
from main.core.records import ExperimentRecord, validate_record


@pytest.mark.quick
def test_experiment_record_contains_required_fields() -> None:
    """实验 record 必须包含产物重建所需的最小字段。"""
    record = ExperimentRecord(
        record_id="record_example",
        run_id="run_example",
        split="validation",
        method_name="example_method",
        metric_name="accuracy",
        metric_value=0.9,
        metadata={},
    )
    assert validate_record(record.to_dict()) == []


@pytest.mark.quick
def test_artifact_manifest_records_rebuild_provenance() -> None:
    """产物 manifest 必须记录输入、输出、配置摘要和重建命令。"""
    manifest = build_artifact_manifest(
        artifact_id="table_example",
        artifact_type="table",
        input_paths=("outputs/records/example.jsonl",),
        output_paths=("outputs/tables/example.csv",),
        config={"metric_name": "accuracy"},
        code_version="uncommitted_template",
        rebuild_command="python scripts/rebuild_example_artifacts.py",
    )
    manifest_dict = manifest.to_dict()
    assert manifest_dict["artifact_id"] == "table_example"
    assert manifest_dict["config_digest"]
    assert manifest_dict["rebuild_command"]


@pytest.mark.quick
def test_core_method_objects_are_serializable_without_runtime_state() -> None:
    """核心方法 typed object 必须能在不携带运行时状态的情况下序列化。"""
    condition = SemanticConditionSpec(
        condition_id="condition_unit",
        semantic_digest="semantic_digest_unit",
        semantic_tags=("object", "scene"),
        risk_policy="default_risk_policy",
        metadata={},
    )
    subspace = LatentSubspaceSpec(
        subspace_id="subspace_unit",
        basis_digest="basis_digest_unit",
        manifold_dimension=4,
        safe_axes=("axis_low_frequency", "axis_high_frequency"),
        metadata={},
    )
    carrier = WatermarkCarrierSpec(
        carrier_id="carrier_unit",
        carrier_family="latent_frequency",
        frequency_band="low_frequency",
        embedding_strength=0.1,
        metadata={},
    )
    anchor = AttentionAnchorSpec(
        anchor_id="anchor_unit",
        attention_layer="self_attention_mid",
        anchor_digest="anchor_digest_unit",
        metadata={},
    )
    evidence = DetectionEvidenceSpec(
        evidence_id="evidence_unit",
        evidence_type="geometry",
        score_name="alignment_score",
        score_value=0.8,
        metadata={},
    )
    decision = FusionDecisionSpec(
        decision_id="decision_unit",
        decision_label="watermarked",
        threshold_name="fixed_fpr_threshold",
        threshold_value=0.5,
        evidence_ids=(evidence.evidence_id,),
        metadata={},
    )

    assert condition.to_dict()["semantic_digest"] == "semantic_digest_unit"
    assert subspace.to_dict()["manifold_dimension"] == 4
    assert carrier.to_dict()["embedding_strength"] == 0.1
    assert anchor.to_dict()["attention_layer"] == "self_attention_mid"
    assert decision.to_dict()["evidence_ids"] == (evidence.evidence_id,)
