"""从注册9重复的原始 Inception 特征重建正式 FID/KID.

该模块只计算跨重复数据集质量统计. 每个注册 repeat 必须精确覆盖同一受治理
Prompt 集合, 每个质量记录必须具有唯一的 source/comparison 特征对. FID 和
KID 直接从9份原始特征联合重算, 不读取或平均任何单重复派生指标表.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from experiments.protocol.dataset_quality import (
    FORMAL_DATASET_QUALITY_METRIC_NAMES,
    FORMAL_FEATURE_BACKEND,
    formal_dataset_quality_metric_protocol,
)
from experiments.protocol.formal_randomization import (
    formal_randomization_repeat_ids,
)
from experiments.protocol.paper_run_config import (
    RUN_EXPECTED_PROMPT_COUNTS,
    normalize_paper_run_name,
    validate_frozen_paper_run_target_fpr,
)
from main.core.digest import build_stable_digest
from paper_experiments.analysis.formal_record_statistics import (
    FORMAL_FEATURE_DIMENSION,
    FormalRecordStatisticsError,
    rebuild_formal_fid_kid_metric_rows_from_feature_records,
)


RANDOMIZATION_DATASET_QUALITY_SUMMARY_SCHEMA = (
    "randomization_dataset_quality_summary_v1"
)
RANDOMIZATION_DATASET_QUALITY_METRIC_PROTOCOL_SCHEMA = (
    "registered_repeat_joint_inception_distribution_v1"
)
RANDOMIZATION_DATASET_QUALITY_MEMBERSHIP_FIELDNAMES = (
    "randomization_repeat_id",
    "prompt_id",
    "dataset_quality_record_id",
    "dataset_quality_record_digest",
    "source_image_digest",
    "comparison_image_digest",
)


class RandomizationDatasetQualityError(ValueError):
    """表示跨重复质量特征不能形成精确且可重建的正式统计."""


@dataclass(frozen=True)
class RandomizationDatasetQualityStatistics:
    """保存规范成员关系, FID/KID 三行指标和重建摘要."""

    membership_records: tuple[Mapping[str, Any], ...]
    metric_rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]


def randomization_dataset_quality_metric_protocol() -> dict[str, Any]:
    """冻结9重复联合特征总体与实际 KID 子集大小."""

    base_protocol = formal_dataset_quality_metric_protocol()
    repeat_count = len(formal_randomization_repeat_ids())
    sample_pair_count_by_run = {
        run_name: repeat_count * prompt_count
        for run_name, prompt_count in RUN_EXPECTED_PROMPT_COUNTS.items()
    }
    kid_subset_size = int(base_protocol["kid_subset_size"])
    payload = {
        "protocol_schema": (
            RANDOMIZATION_DATASET_QUALITY_METRIC_PROTOCOL_SCHEMA
        ),
        "base_formal_metric_protocol_digest": base_protocol[
            "formal_metric_protocol_digest"
        ],
        "registered_repeat_count": repeat_count,
        "feature_population_rule": (
            "joint_raw_feature_rows_across_registered_repeats"
        ),
        "prompt_weighting_rule": (
            "equal_prompt_multiplicity_from_exact_repeat_cartesian_product"
        ),
        "aggregate_sample_pair_count_by_paper_run": (
            sample_pair_count_by_run
        ),
        "randomization_kid_effective_subset_size_by_paper_run": {
            run_name: min(kid_subset_size, pair_count)
            for run_name, pair_count in sample_pair_count_by_run.items()
        },
    }
    payload["randomization_dataset_quality_metric_protocol_digest"] = (
        build_stable_digest(payload)
    )
    return payload


def _is_sha256(value: Any) -> bool:
    """判断值是否为规范小写 SHA-256."""

    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _canonical_membership_records(
    membership_records: Iterable[Mapping[str, Any]],
    *,
    expected_prompt_ids: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    """验证9重复与 Prompt 的笛卡尔积并返回规范顺序."""

    repeat_ids = formal_randomization_repeat_ids()
    repeat_order = {
        repeat_id: index for index, repeat_id in enumerate(repeat_ids)
    }
    prompt_order = {
        prompt_id: index for index, prompt_id in enumerate(expected_prompt_ids)
    }
    expected_keys = {
        (repeat_id, prompt_id)
        for repeat_id in repeat_ids
        for prompt_id in expected_prompt_ids
    }
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    record_ids: set[str] = set()
    for row_index, raw_record in enumerate(membership_records):
        record = dict(raw_record)
        if set(record) != set(
            RANDOMIZATION_DATASET_QUALITY_MEMBERSHIP_FIELDNAMES
        ):
            raise RandomizationDatasetQualityError(
                f"质量特征成员字段集合无效: row={row_index}"
            )
        repeat_id = str(record["randomization_repeat_id"])
        prompt_id = str(record["prompt_id"])
        record_id = str(record["dataset_quality_record_id"])
        record_digest = str(record["dataset_quality_record_digest"])
        source_image_digest = str(record["source_image_digest"])
        comparison_image_digest = str(record["comparison_image_digest"])
        key = (repeat_id, prompt_id)
        if (
            key not in expected_keys
            or key in by_key
            or not record_id
            or record_id in record_ids
            or not _is_sha256(record_digest)
            or record_id != f"dataset_quality_record_{record_digest[:16]}"
            or not _is_sha256(source_image_digest)
            or not _is_sha256(comparison_image_digest)
        ):
            raise RandomizationDatasetQualityError(
                f"质量特征成员身份重复, 缺失或无效: row={row_index}"
            )
        by_key[key] = {
            "randomization_repeat_id": repeat_id,
            "prompt_id": prompt_id,
            "dataset_quality_record_id": record_id,
            "dataset_quality_record_digest": record_digest,
            "source_image_digest": source_image_digest,
            "comparison_image_digest": comparison_image_digest,
        }
        record_ids.add(record_id)
    if set(by_key) != expected_keys:
        raise RandomizationDatasetQualityError(
            "质量特征成员未精确覆盖9重复与完整 Prompt 集合"
        )
    return tuple(
        by_key[key]
        for key in sorted(
            by_key,
            key=lambda value: (
                repeat_order[value[0]],
                prompt_order[value[1]],
            ),
        )
    )


def _canonical_feature_records(
    feature_records: Iterable[Mapping[str, Any]],
    *,
    membership_records: tuple[Mapping[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """把原始特征绑定到成员图像摘要并返回规范角色顺序."""

    membership_by_record_id = {
        str(record["dataset_quality_record_id"]): record
        for record in membership_records
    }
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row_index, raw_record in enumerate(feature_records):
        record = dict(raw_record)
        record_id = str(record.get("dataset_quality_record_id", ""))
        role = str(record.get("dataset_quality_image_role", ""))
        membership = membership_by_record_id.get(record_id)
        key = (record_id, role)
        expected_image_digest = (
            str(membership[f"{role}_image_digest"])
            if membership is not None and role in {"source", "comparison"}
            else ""
        )
        declared_repeat_id = record.get("randomization_repeat_id")
        if (
            membership is None
            or role not in {"source", "comparison"}
            or key in by_key
            or str(record.get("image_digest", ""))
            != expected_image_digest
            or str(record.get("feature_backend", ""))
            != FORMAL_FEATURE_BACKEND
            or type(record.get("feature_dimension")) is not int
            or int(record["feature_dimension"]) != FORMAL_FEATURE_DIMENSION
            or (
                declared_repeat_id not in (None, "")
                and str(declared_repeat_id)
                != str(membership["randomization_repeat_id"])
            )
        ):
            raise RandomizationDatasetQualityError(
                f"质量特征角色, 图像身份或正式维度无效: row={row_index}"
            )
        by_key[key] = record
    expected_keys = {
        (record_id, role)
        for record_id in membership_by_record_id
        for role in ("source", "comparison")
    }
    if set(by_key) != expected_keys:
        raise RandomizationDatasetQualityError(
            "质量 feature records 未精确覆盖全部成员与两个图像角色"
        )
    return tuple(
        by_key[(str(membership["dataset_quality_record_id"]), role)]
        for membership in membership_records
        for role in ("source", "comparison")
    )


def rebuild_randomization_dataset_quality_statistics(
    feature_records: Iterable[Mapping[str, Any]],
    membership_records: Iterable[Mapping[str, Any]],
    *,
    paper_run_name: str,
    target_fpr: float,
    expected_prompt_ids: Iterable[str],
) -> RandomizationDatasetQualityStatistics:
    """从9重复原始特征联合重建一次正式 FID/KID 结果."""

    run_name = normalize_paper_run_name(paper_run_name)
    resolved_target_fpr = validate_frozen_paper_run_target_fpr(
        run_name,
        target_fpr,
    )
    prompt_ids = tuple(str(prompt_id) for prompt_id in expected_prompt_ids)
    expected_prompt_count = RUN_EXPECTED_PROMPT_COUNTS[run_name]
    if (
        len(prompt_ids) != expected_prompt_count
        or len(set(prompt_ids)) != expected_prompt_count
        or any(not prompt_id for prompt_id in prompt_ids)
    ):
        raise RandomizationDatasetQualityError(
            "受治理 Prompt 身份未匹配论文运行层级的精确数量"
        )
    canonical_membership = _canonical_membership_records(
        membership_records,
        expected_prompt_ids=prompt_ids,
    )
    canonical_features = _canonical_feature_records(
        feature_records,
        membership_records=canonical_membership,
    )
    aggregate_pair_count = (
        len(formal_randomization_repeat_ids()) * expected_prompt_count
    )
    try:
        metric_rows = (
            rebuild_formal_fid_kid_metric_rows_from_feature_records(
                canonical_features,
                expected_pair_count=aggregate_pair_count,
            )
        )
    except (FormalRecordStatisticsError, TypeError, ValueError) as exc:
        raise RandomizationDatasetQualityError(
            "9重复原始特征不能完成正式 FID/KID 数值重建"
        ) from exc
    if (
        tuple(row["quality_metric_name"] for row in metric_rows)
        != tuple(FORMAL_DATASET_QUALITY_METRIC_NAMES)
        or any(
            row["metric_status"] != "measured"
            or row["feature_backend"] != FORMAL_FEATURE_BACKEND
            or row["source_image_count"] != aggregate_pair_count
            or row["comparison_image_count"] != aggregate_pair_count
            or row["sample_pair_count"] != aggregate_pair_count
            or row["supports_paper_claim"] is not False
            for row in metric_rows
        )
    ):
        raise RandomizationDatasetQualityError(
            "9重复 FID/KID 三行指标未形成完整 measured 结果"
        )

    metric_protocol = randomization_dataset_quality_metric_protocol()
    summary: dict[str, Any] = {
        "summary_schema": RANDOMIZATION_DATASET_QUALITY_SUMMARY_SCHEMA,
        "paper_claim_scale": run_name,
        "target_fpr": resolved_target_fpr,
        "randomization_repeat_ids": list(
            formal_randomization_repeat_ids()
        ),
        "randomization_repeat_count": len(
            formal_randomization_repeat_ids()
        ),
        "prompt_count_per_repeat": expected_prompt_count,
        "aggregate_quality_pair_count": aggregate_pair_count,
        "aggregate_feature_record_count": len(canonical_features),
        "prompt_id_set_digest": build_stable_digest(sorted(prompt_ids)),
        "quality_feature_membership_digest": build_stable_digest(
            canonical_membership
        ),
        "quality_feature_records_digest": build_stable_digest(
            canonical_features
        ),
        "randomization_dataset_quality_metric_protocol_digest": (
            metric_protocol[
                "randomization_dataset_quality_metric_protocol_digest"
            ]
        ),
        "fid_kid_metric_rows_digest": build_stable_digest(metric_rows),
        "quality_metric_names": list(FORMAL_DATASET_QUALITY_METRIC_NAMES),
        "quality_metric_status": "measured",
        "randomization_dataset_quality_statistics_ready": True,
        "conclusion_decision": "measured_evidence_component",
        "supports_paper_claim": False,
    }
    summary["randomization_dataset_quality_summary_digest"] = (
        build_stable_digest(summary)
    )
    return RandomizationDatasetQualityStatistics(
        membership_records=canonical_membership,
        metric_rows=tuple(dict(row) for row in metric_rows),
        summary=summary,
    )


__all__ = [
    "RANDOMIZATION_DATASET_QUALITY_MEMBERSHIP_FIELDNAMES",
    "RANDOMIZATION_DATASET_QUALITY_METRIC_PROTOCOL_SCHEMA",
    "RANDOMIZATION_DATASET_QUALITY_SUMMARY_SCHEMA",
    "RandomizationDatasetQualityError",
    "RandomizationDatasetQualityStatistics",
    "randomization_dataset_quality_metric_protocol",
    "rebuild_randomization_dataset_quality_statistics",
]
