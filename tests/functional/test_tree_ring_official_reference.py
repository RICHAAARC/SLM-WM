"""验证 Tree-Ring 官方原始环境补充表 governed import 协议。"""

from __future__ import annotations

import pytest

from experiments.baselines import (
    TREE_RING_OFFICIAL_REFERENCE_PROTOCOL_NAME,
    build_tree_ring_official_reference_record,
    build_tree_ring_official_reference_schema,
    validate_tree_ring_official_reference_records,
)


@pytest.mark.quick
def test_tree_ring_official_reference_record_validates_when_all_boundaries_ready() -> None:
    """官方 legacy 复现记录满足证据边界时应通过补充表导入校验。"""

    record = build_tree_ring_official_reference_record(
        official_entrypoint="external_baseline/primary/tree_ring/source/run_tree_ring_watermark.py",
        official_repository_commit="3015283d9cf82e90b628f02ad2121bd37408ca9a",
        official_environment_profile="python3.8_diffusers0.11.1_legacy_ddim",
        baseline_result_source="outputs/tree_ring_official_reference/summary.json",
        baseline_result_source_digest="digest",
        evidence_paths=["outputs/tree_ring_official_reference/summary.json"],
        metric_values={
            "sample_count": 10,
            "positive_count": 10,
            "negative_count": 10,
            "auc": 0.9,
            "accuracy": 0.8,
            "true_positive_rate_at_one_percent_fpr": 0.7,
            "clip_score_mean": 0.3,
            "watermarked_clip_score_mean": 0.29,
        },
        ready_flags={
            "official_source_ready": True,
            "official_environment_report_ready": True,
            "official_result_summary_ready": True,
            "governed_import_ready": True,
        },
    )

    report = validate_tree_ring_official_reference_records([record])
    schema = build_tree_ring_official_reference_schema()

    assert schema["reference_protocol_name"] == TREE_RING_OFFICIAL_REFERENCE_PROTOCOL_NAME
    assert record["supplemental_table_role"] == "supplemental_method_fidelity_reference"
    assert record["main_table_eligible"] is False
    assert report["reference_import_ready"] is True
    assert report["accepted_reference_record_count"] == 1


@pytest.mark.quick
def test_tree_ring_official_reference_rejects_main_table_eligibility() -> None:
    """官方 legacy 参考记录不得伪装为主表同协议结果。"""

    record = build_tree_ring_official_reference_record(
        official_entrypoint="external_baseline/primary/tree_ring/source/run_tree_ring_watermark.py",
        official_repository_commit="3015283d9cf82e90b628f02ad2121bd37408ca9a",
        official_environment_profile="python3.8_diffusers0.11.1_legacy_ddim",
        baseline_result_source="outputs/tree_ring_official_reference/summary.json",
        baseline_result_source_digest="digest",
        evidence_paths=["outputs/tree_ring_official_reference/summary.json"],
        metric_values={
            "sample_count": 10,
            "positive_count": 10,
            "negative_count": 10,
            "auc": 0.9,
            "accuracy": 0.8,
            "true_positive_rate_at_one_percent_fpr": 0.7,
            "clip_score_mean": 0.3,
            "watermarked_clip_score_mean": 0.29,
        },
        ready_flags={
            "official_source_ready": True,
            "official_environment_report_ready": True,
            "official_result_summary_ready": True,
            "governed_import_ready": True,
        },
    )
    record["main_table_eligible"] = True

    report = validate_tree_ring_official_reference_records([record])
    reasons = {issue["reason"] for issue in report["issues"]}

    assert report["reference_import_ready"] is False
    assert "legacy_reference_must_not_enter_main_table" in reasons
