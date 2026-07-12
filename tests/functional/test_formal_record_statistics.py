"""验证原始正式记录到论文统计的独立重建门禁."""

from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.ablations.necessity_statistics import (
    build_ablation_necessity_statistics,
)
from experiments.protocol.dataset_quality import (
    FORMAL_FEATURE_BACKEND,
    FORMAL_FEATURE_EXTRACTOR_ID,
    rebuild_formal_fid_kid_metric_rows,
)
from paper_experiments.analysis.formal_record_statistics import (
    FormalRecordStatisticsError,
    rebuild_and_validate_ablation_necessity_statistics,
    rebuild_and_validate_formal_fid_kid_metrics,
)


def ablation_evidence() -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    dict[str, object],
    tuple[str, ...],
]:
    """构造包含任意扩展变体身份的配对原始记录."""

    expected_ids = (
        "complete_method",
        "without_semantic_routing",
        "shared_global_risk_routing",
        "tail_robust_only",
    )
    prompt_ids = tuple(f"prompt_{index:02d}" for index in range(6))
    records = tuple(
        {
            "ablation_id": ablation_id,
            "prompt_id": prompt_id,
            "split": "test",
            "formal_attack_coverage_ready": True,
            "attacked_positive_rate": (
                1.0 if ablation_id == "complete_method" else 0.0
            ),
            "positive_source_positive": ablation_id == "complete_method",
            "paired_ssim": 0.95,
        }
        for ablation_id in expected_ids
        for prompt_id in prompt_ids
    )
    rows, summary = build_ablation_necessity_statistics(
        records,
        expected_ablation_ids=expected_ids[1:],
        expected_paired_prompt_count=len(prompt_ids),
    )
    return records, tuple(rows), summary, expected_ids


def feature_evidence() -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
]:
    """构造无需模型推理即可重算的小维正式特征记录."""

    pair_count = 5
    records = tuple(
        record
        for pair_index in range(pair_count)
        for record in (
            {
                "dataset_quality_record_id": f"quality_{pair_index:02d}",
                "dataset_quality_image_role": "source",
                "feature_backend": FORMAL_FEATURE_BACKEND,
                "feature_extractor_id": FORMAL_FEATURE_EXTRACTOR_ID,
                "feature_dimension": 3,
                "feature_vector": [float(pair_index), 0.2, -0.1],
                "supports_paper_claim": False,
            },
            {
                "dataset_quality_record_id": f"quality_{pair_index:02d}",
                "dataset_quality_image_role": "comparison",
                "feature_backend": FORMAL_FEATURE_BACKEND,
                "feature_extractor_id": FORMAL_FEATURE_EXTRACTOR_ID,
                "feature_dimension": 3,
                "feature_vector": [float(pair_index) + 0.1, 0.3, -0.15],
                "supports_paper_claim": False,
            },
        )
    )
    source = [
        row["feature_vector"]
        for row in records
        if row["dataset_quality_image_role"] == "source"
    ]
    comparison = [
        row["feature_vector"]
        for row in records
        if row["dataset_quality_image_role"] == "comparison"
    ]
    rows = tuple(
        rebuild_formal_fid_kid_metric_rows(
            source,
            comparison,
            sample_pair_count=pair_count,
        )
    )
    return records, rows


@pytest.mark.quick
def test_ablation_statistics_are_rebuilt_for_declared_variant_set() -> None:
    """重建器应按声明身份工作,而不是绑定当前固定变体数量."""

    records, rows, summary, expected_ids = ablation_evidence()

    report = rebuild_and_validate_ablation_necessity_statistics(
        records,
        rows,
        summary,
        {**summary, "protocol_decision": "pass"},
        expected_ablation_ids=expected_ids,
        expected_paired_prompt_count=6,
    )

    assert report["ablation_statistics_rebuild_ready"] is True
    assert report["ablation_raw_record_count"] == len(records)


@pytest.mark.quick
def test_ablation_statistics_reject_raw_record_and_csv_drift() -> None:
    """原始配对结果或任一派生字段漂移时都必须 fail-closed."""

    records, rows, summary, expected_ids = ablation_evidence()
    changed_records = [dict(record) for record in records]
    changed_records[0]["attacked_positive_rate"] = 0.5
    with pytest.raises(FormalRecordStatisticsError, match="重建值不一致"):
        rebuild_and_validate_ablation_necessity_statistics(
            changed_records,
            rows,
            summary,
            summary,
            expected_ablation_ids=expected_ids,
            expected_paired_prompt_count=6,
        )

    changed_rows = [dict(row) for row in rows]
    changed_rows[0]["mean_paired_effect"] = 0.5
    with pytest.raises(FormalRecordStatisticsError, match="field=mean_paired_effect"):
        rebuild_and_validate_ablation_necessity_statistics(
            records,
            changed_rows,
            summary,
            summary,
            expected_ablation_ids=expected_ids,
            expected_paired_prompt_count=6,
        )


@pytest.mark.quick
def test_ablation_protocol_closure_allows_measured_not_supported_result() -> None:
    """协议闭合不得把未获支持的单机制结论伪造成统计重建失败."""

    records, _, _, expected_ids = ablation_evidence()
    changed_records = [dict(record) for record in records]
    for record in changed_records:
        if record["ablation_id"] == "tail_robust_only":
            record["attacked_positive_rate"] = 1.0
            record["positive_source_positive"] = True
    rows, summary = build_ablation_necessity_statistics(
        changed_records,
        expected_ablation_ids=expected_ids[1:],
        expected_paired_prompt_count=6,
    )

    report = rebuild_and_validate_ablation_necessity_statistics(
        changed_records,
        rows,
        summary,
        {**summary, "supports_paper_claim": True},
        expected_ablation_ids=expected_ids,
        expected_paired_prompt_count=6,
    )

    assert report["ablation_statistics_rebuild_ready"] is True
    assert summary["all_mechanism_necessity_claims_supported"] is False


@pytest.mark.quick
def test_fid_kid_are_rebuilt_from_feature_records() -> None:
    """正式特征的精确 source/comparison 配对应独立生成两行指标."""

    records, rows = feature_evidence()

    report = rebuild_and_validate_formal_fid_kid_metrics(
        records,
        rows,
        expected_pair_count=5,
    )

    assert report["dataset_quality_metric_rebuild_ready"] is True
    assert report["dataset_quality_feature_record_count"] == 10


@pytest.mark.quick
def test_fid_kid_rebuild_rejects_feature_and_metric_drift() -> None:
    """特征向量、角色配对或指标表任一漂移均不得通过."""

    records, rows = feature_evidence()
    changed_records = [deepcopy(record) for record in records]
    changed_records[0]["feature_vector"][0] += 0.5
    with pytest.raises(FormalRecordStatisticsError, match="独立重算结果不一致"):
        rebuild_and_validate_formal_fid_kid_metrics(
            changed_records,
            rows,
            expected_pair_count=5,
        )

    changed_rows = [dict(row) for row in rows]
    changed_rows[1]["feature_backend"] = "unregistered_backend"
    with pytest.raises(FormalRecordStatisticsError, match="field=feature_backend"):
        rebuild_and_validate_formal_fid_kid_metrics(
            records,
            changed_rows,
            expected_pair_count=5,
        )

    duplicate_records = [dict(record) for record in records]
    duplicate_records[1]["dataset_quality_image_role"] = "source"
    with pytest.raises(FormalRecordStatisticsError, match="重复身份角色"):
        rebuild_and_validate_formal_fid_kid_metrics(
            duplicate_records,
            rows,
            expected_pair_count=5,
        )
