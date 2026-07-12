"""从正式逐样本记录独立重建论文统计并核对派生表.

该模块位于完整论文实验层,而不是核心方法层.它不信任上游 summary 或 CSV
中的 ready 标记,而是重新消费逐 Prompt 消融记录和逐图像 Inception 特征
记录,调用冻结的实验统计算子重算结果,再逐字段比较持久化派生产物.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping

import numpy as np

from experiments.ablations.necessity_statistics import (
    ABLATION_NECESSITY_BOOTSTRAP_RESAMPLE_COUNT,
    ABLATION_NECESSITY_FIELDNAMES,
    build_ablation_necessity_statistics,
    canonicalize_ablation_necessity_rows,
)
from experiments.protocol.dataset_quality import (
    FORMAL_FEATURE_BACKEND,
    FORMAL_FEATURE_EXTRACTOR_ID,
    rebuild_formal_fid_kid_metric_rows,
)
from main.core.digest import build_stable_digest


DATASET_QUALITY_METRIC_FIELDNAMES = (
    "quality_metric_name",
    "quality_metric_value",
    "metric_status",
    "paper_metric_name",
    "feature_backend",
    "source_image_count",
    "comparison_image_count",
    "sample_pair_count",
    "supports_paper_claim",
)
FORMAL_METRIC_RELATIVE_TOLERANCE = 1e-8
FORMAL_METRIC_ABSOLUTE_TOLERANCE = 1e-10


class FormalRecordStatisticsError(ValueError):
    """表示原始正式记录无法唯一重建或与派生证据不一致."""


def _strict_bool(value: Any) -> bool:
    """读取 JSON 或 CSV 中无歧义的布尔表示."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise FormalRecordStatisticsError(f"字段不是严格布尔值: {value!r}")


def _positive_int(value: Any, field_name: str) -> int:
    """读取必须为正整数的计数字段."""

    if isinstance(value, bool):
        raise FormalRecordStatisticsError(f"{field_name} 不能使用布尔值")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise FormalRecordStatisticsError(f"{field_name} 不是整数") from exc
    if str(value).strip() not in {str(resolved), f"{resolved}.0"} or resolved <= 0:
        raise FormalRecordStatisticsError(f"{field_name} 必须为正整数")
    return resolved


def rebuild_and_validate_ablation_necessity_statistics(
    raw_records: Iterable[Mapping[str, Any]],
    reported_rows: Iterable[Mapping[str, Any]],
    reported_summary: Mapping[str, Any],
    claim_summary: Mapping[str, Any],
    *,
    expected_ablation_ids: Iterable[str],
    expected_paired_prompt_count: int,
) -> dict[str, Any]:
    """从逐 Prompt rerun records 重建必要性统计并逐字段比对.

    ``expected_ablation_ids`` 来自当前正式 manifest/summary 契约,因此该函数
    不绑定当前变体数量.列表必须包含一次 ``complete_method``,其余身份按
    声明顺序生成统计行;未来扩展消融集合时可直接复用.
    """

    declared_ids = tuple(str(value) for value in expected_ablation_ids)
    if (
        not declared_ids
        or declared_ids.count("complete_method") != 1
        or len(set(declared_ids)) != len(declared_ids)
    ):
        raise FormalRecordStatisticsError("正式消融身份必须唯一包含 complete_method")
    variant_ids = tuple(
        ablation_id
        for ablation_id in declared_ids
        if ablation_id != "complete_method"
    )
    if not variant_ids:
        raise FormalRecordStatisticsError("正式消融至少需要一个机制变体")

    materialized_records = tuple(dict(record) for record in raw_records)
    rebuilt_rows, rebuilt_summary = build_ablation_necessity_statistics(
        materialized_records,
        expected_ablation_ids=variant_ids,
        expected_paired_prompt_count=expected_paired_prompt_count,
        bootstrap_resample_count=ABLATION_NECESSITY_BOOTSTRAP_RESAMPLE_COUNT,
    )
    materialized_reported_rows = tuple(dict(row) for row in reported_rows)
    if len(materialized_reported_rows) != len(rebuilt_rows):
        raise FormalRecordStatisticsError("机制必要性统计行数与原始记录重建结果不一致")
    if any(set(row) != set(ABLATION_NECESSITY_FIELDNAMES) for row in materialized_reported_rows):
        raise FormalRecordStatisticsError("机制必要性统计字段集合不符合冻结 schema")
    rebuilt_canonical_rows = canonicalize_ablation_necessity_rows(rebuilt_rows)
    reported_canonical_rows = canonicalize_ablation_necessity_rows(
        materialized_reported_rows
    )
    for row_index, (reported, rebuilt) in enumerate(
        zip(reported_canonical_rows, rebuilt_canonical_rows, strict=True)
    ):
        for field_name in ABLATION_NECESSITY_FIELDNAMES:
            if reported[field_name] != rebuilt[field_name]:
                raise FormalRecordStatisticsError(
                    "机制必要性统计与 raw rerun records 重建值不一致: "
                    f"row={row_index}, field={field_name}"
                )

    for field_name, rebuilt_value in rebuilt_summary.items():
        if reported_summary.get(field_name) != rebuilt_value:
            raise FormalRecordStatisticsError(
                f"机制必要性 summary 与 raw rerun records 重建值不一致: {field_name}"
            )
        # ablation claim summary 的 supports_paper_claim 表示协议闭合,而
        # necessity summary 的同名字段表示全部单机制必要性主张均获支持.
        # 两者语义不同,不得强制相等;其余必要性字段必须逐项绑定.
        if (
            field_name != "supports_paper_claim"
            and claim_summary.get(field_name) != rebuilt_value
        ):
            raise FormalRecordStatisticsError(
                f"消融 claim summary 与 raw rerun records 重建值不一致: {field_name}"
            )
    return {
        "ablation_raw_record_count": len(materialized_records),
        "ablation_raw_records_digest": build_stable_digest(materialized_records),
        "ablation_statistics_rebuilt_rows_digest": build_stable_digest(
            rebuilt_canonical_rows
        ),
        "ablation_statistics_rebuild_ready": True,
    }


def _formal_feature_arrays(
    feature_records: Iterable[Mapping[str, Any]],
    *,
    expected_pair_count: int,
) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, Any], ...]]:
    """验证 feature records 的身份与角色并构造确定顺序的两组矩阵."""

    materialized = tuple(dict(record) for record in feature_records)
    if len(materialized) != expected_pair_count * 2:
        raise FormalRecordStatisticsError("正式 feature record 数量不是样本对数量的2倍")
    grouped: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    feature_dimension: int | None = None
    for row_index, record in enumerate(materialized):
        record_id = str(record.get("dataset_quality_record_id", "")).strip()
        role = str(record.get("dataset_quality_image_role", "")).strip()
        if not record_id or role not in {"source", "comparison"}:
            raise FormalRecordStatisticsError(
                f"正式 feature record 身份或角色无效: row={row_index}"
            )
        if role in grouped[record_id]:
            raise FormalRecordStatisticsError(
                f"正式 feature record 出现重复身份角色: {record_id}/{role}"
            )
        if str(record.get("feature_backend", "")) != FORMAL_FEATURE_BACKEND:
            raise FormalRecordStatisticsError("正式 feature record 后端不符合冻结协议")
        if str(record.get("feature_extractor_id", "")) != FORMAL_FEATURE_EXTRACTOR_ID:
            raise FormalRecordStatisticsError("正式 feature record 提取器身份不符合冻结协议")
        if _strict_bool(record.get("supports_paper_claim")):
            raise FormalRecordStatisticsError("原始 feature record 不得直接声明论文结论")
        vector = np.asarray(record.get("feature_vector"), dtype=np.float64)
        if vector.ndim != 1 or vector.size <= 0 or not np.isfinite(vector).all():
            raise FormalRecordStatisticsError("正式 feature vector 必须是一维非空有限向量")
        declared_dimension = _positive_int(
            record.get("feature_dimension"),
            "feature_dimension",
        )
        if declared_dimension != int(vector.size):
            raise FormalRecordStatisticsError("feature_dimension 与向量长度不一致")
        if feature_dimension is None:
            feature_dimension = declared_dimension
        elif feature_dimension != declared_dimension:
            raise FormalRecordStatisticsError("正式 feature records 的向量维度不一致")
        grouped[record_id][role] = vector

    if len(grouped) != expected_pair_count or any(
        set(role_map) != {"source", "comparison"}
        for role_map in grouped.values()
    ):
        raise FormalRecordStatisticsError("正式 feature records 未形成精确 source/comparison 配对")
    ordered_ids = tuple(sorted(grouped))
    source_features = np.stack(
        [grouped[record_id]["source"] for record_id in ordered_ids]
    )
    comparison_features = np.stack(
        [grouped[record_id]["comparison"] for record_id in ordered_ids]
    )
    return source_features, comparison_features, materialized


def _normalized_dataset_metric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """把 CSV/JSON 指标行规整为可逐字段核对的统一类型."""

    if set(row) != set(DATASET_QUALITY_METRIC_FIELDNAMES):
        raise FormalRecordStatisticsError("FID/KID 指标字段集合不符合冻结 schema")
    try:
        value = float(row["quality_metric_value"])
    except (TypeError, ValueError) as exc:
        raise FormalRecordStatisticsError("FID/KID 指标值不是有限数值") from exc
    if not math.isfinite(value):
        raise FormalRecordStatisticsError("FID/KID 指标值不是有限数值")
    return {
        "quality_metric_name": str(row["quality_metric_name"]),
        "quality_metric_value": value,
        "metric_status": str(row["metric_status"]),
        "paper_metric_name": str(row["paper_metric_name"]),
        "feature_backend": str(row["feature_backend"]),
        "source_image_count": _positive_int(
            row["source_image_count"], "source_image_count"
        ),
        "comparison_image_count": _positive_int(
            row["comparison_image_count"], "comparison_image_count"
        ),
        "sample_pair_count": _positive_int(
            row["sample_pair_count"], "sample_pair_count"
        ),
        "supports_paper_claim": _strict_bool(row["supports_paper_claim"]),
    }


def rebuild_and_validate_formal_fid_kid_metrics(
    feature_records: Iterable[Mapping[str, Any]],
    reported_rows: Iterable[Mapping[str, Any]],
    *,
    expected_pair_count: int,
) -> dict[str, Any]:
    """从正式 feature records 重算 FID/KID 并逐字段核对指标表."""

    source_features, comparison_features, materialized_records = (
        _formal_feature_arrays(
            feature_records,
            expected_pair_count=expected_pair_count,
        )
    )
    rebuilt_rows = rebuild_formal_fid_kid_metric_rows(
        source_features,
        comparison_features,
        sample_pair_count=expected_pair_count,
    )
    normalized_rebuilt = tuple(
        _normalized_dataset_metric_row(row) for row in rebuilt_rows
    )
    normalized_reported = tuple(
        _normalized_dataset_metric_row(row) for row in reported_rows
    )
    if len(normalized_reported) != len(normalized_rebuilt):
        raise FormalRecordStatisticsError("FID/KID 指标行数与 feature records 重建结果不一致")
    for row_index, (reported, rebuilt) in enumerate(
        zip(normalized_reported, normalized_rebuilt, strict=True)
    ):
        for field_name in DATASET_QUALITY_METRIC_FIELDNAMES:
            if field_name == "quality_metric_value":
                if not math.isclose(
                    float(reported[field_name]),
                    float(rebuilt[field_name]),
                    rel_tol=FORMAL_METRIC_RELATIVE_TOLERANCE,
                    abs_tol=FORMAL_METRIC_ABSOLUTE_TOLERANCE,
                ):
                    raise FormalRecordStatisticsError(
                        "FID/KID 指标值与 feature records 独立重算结果不一致: "
                        f"row={row_index}"
                    )
            elif reported[field_name] != rebuilt[field_name]:
                raise FormalRecordStatisticsError(
                    "FID/KID 指标字段与 feature records 重建结果不一致: "
                    f"row={row_index}, field={field_name}"
                )
    return {
        "dataset_quality_feature_record_count": len(materialized_records),
        "dataset_quality_rebuilt_metric_rows_digest": build_stable_digest(
            normalized_rebuilt
        ),
        "dataset_quality_metric_relative_tolerance": (
            FORMAL_METRIC_RELATIVE_TOLERANCE
        ),
        "dataset_quality_metric_absolute_tolerance": (
            FORMAL_METRIC_ABSOLUTE_TOLERANCE
        ),
        "dataset_quality_metric_rebuild_ready": True,
    }
