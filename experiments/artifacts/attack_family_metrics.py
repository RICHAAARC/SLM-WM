"""从正式攻击检测记录重建逐攻击指标表."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable

from experiments.protocol.calibration import binomial_rate_upper_confidence_bound


ATTACK_FAMILY_METRIC_FIELDS = (
    "attack_id",
    "attack_family",
    "attack_name",
    "resource_profile",
    "attack_config_digest",
    "metric_status",
    "attack_record_count",
    "supported_record_count",
    "unsupported_record_count",
    "positive_count",
    "negative_count",
    "true_positive_rate",
    "false_positive_rate",
    "clean_false_positive_rate",
    "attacked_false_positive_rate",
    "false_positive_rate_upper_95",
    "target_fpr",
    "fixed_fpr_upper_bound_ready",
    "quality_score_mean",
    "quality_ssim_mean",
    "quality_psnr_mean",
    "attacked_positive_source_to_attacked_ssim_mean",
    "score_retention_mean",
    "lf_score_retention_mean",
    "tail_score_retention_mean",
    "geometry_reliable_rate",
    "rescue_rate",
    "supports_paper_claim",
)


def _finite_mean(values: Iterable[Any], field_name: str) -> float:
    """要求每个正式样本都提供有限测量值并计算均值."""

    materialized = tuple(values)
    if (
        not materialized
        or any(isinstance(value, bool) for value in materialized)
        or any(not isinstance(value, int | float) for value in materialized)
    ):
        raise ValueError(f"{field_name} 必须覆盖全部正式样本")
    resolved = tuple(float(value) for value in materialized)
    if any(not math.isfinite(value) for value in resolved):
        raise ValueError(f"{field_name} 必须是有限数值")
    return sum(resolved) / len(resolved)


def _boolean_rate(
    records: Iterable[dict[str, Any]],
    field_name: str,
) -> float:
    """要求每条记录显式携带布尔字段并计算比例."""

    rows = tuple(records)
    if not rows or any(not isinstance(row.get(field_name), bool) for row in rows):
        raise ValueError(f"{field_name} 必须覆盖全部正式攻击记录")
    return sum(row[field_name] for row in rows) / len(rows)


def build_attack_family_metrics(
    records: Iterable[dict[str, Any]],
    target_fpr: float,
    supports_paper_claim: bool,
) -> tuple[dict[str, Any], ...]:
    """按正式攻击身份重建 TPR、FPR、质量和方法内分数保持指标."""

    grouped: dict[
        tuple[str, str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for raw_record in records:
        record = dict(raw_record)
        key = (
            str(record.get("attack_id", "")),
            str(record.get("attack_family", "")),
            str(record.get("attack_name", "")),
            str(record.get("resource_profile", "")),
            str(record.get("attack_config_digest", "")),
        )
        if not all(key):
            raise ValueError("正式攻击指标记录缺少完整攻击身份")
        grouped[key].append(record)

    rows: list[dict[str, Any]] = []
    for (
        attack_id,
        family,
        name,
        profile,
        config_digest,
    ), group in sorted(grouped.items()):
        positives = [
            row for row in group if row.get("sample_role") == "positive_source"
        ]
        negatives = [
            row for row in group if row.get("sample_role") == "clean_negative"
        ]
        if not positives or not negatives:
            raise ValueError("每个正式攻击必须同时覆盖正样本和 clean negative")
        if any(
            not isinstance(row.get("formal_evidence_positive"), bool)
            for row in group
        ):
            raise ValueError("formal_evidence_positive 必须是布尔值")
        true_positive_count = sum(
            row["formal_evidence_positive"] for row in positives
        )
        false_positive_count = sum(
            row["formal_evidence_positive"] for row in negatives
        )
        false_positive_upper = binomial_rate_upper_confidence_bound(
            false_positive_count,
            len(negatives),
            0.95,
        )
        rows.append(
            {
                "attack_id": attack_id,
                "attack_family": family,
                "attack_name": name,
                "resource_profile": profile,
                "attack_config_digest": config_digest,
                "metric_status": (
                    "measured_real_attacked_image_image_only_detection"
                ),
                "attack_record_count": len(group),
                "supported_record_count": (
                    len(group) if supports_paper_claim else 0
                ),
                "unsupported_record_count": (
                    0 if supports_paper_claim else len(group)
                ),
                "positive_count": len(positives),
                "negative_count": len(negatives),
                "true_positive_rate": true_positive_count / len(positives),
                "false_positive_rate": false_positive_count / len(negatives),
                "clean_false_positive_rate": (
                    false_positive_count / len(negatives)
                ),
                "attacked_false_positive_rate": (
                    false_positive_count / len(negatives)
                ),
                "false_positive_rate_upper_95": false_positive_upper,
                "target_fpr": float(target_fpr),
                "fixed_fpr_upper_bound_ready": (
                    false_positive_upper <= float(target_fpr)
                ),
                "quality_score_mean": _finite_mean(
                    (row.get("quality_score") for row in positives),
                    "quality_score",
                ),
                "quality_ssim_mean": _finite_mean(
                    (row.get("quality_ssim") for row in positives),
                    "quality_ssim",
                ),
                "quality_psnr_mean": _finite_mean(
                    (row.get("quality_psnr") for row in positives),
                    "quality_psnr",
                ),
                "attacked_positive_source_to_attacked_ssim_mean": (
                    _finite_mean(
                        (row.get("quality_ssim") for row in positives),
                        "quality_ssim",
                    )
                ),
                "score_retention_mean": _finite_mean(
                    (row.get("score_retention") for row in positives),
                    "score_retention",
                ),
                "lf_score_retention_mean": _finite_mean(
                    (row.get("lf_score_retention") for row in positives),
                    "lf_score_retention",
                ),
                "tail_score_retention_mean": _finite_mean(
                    (row.get("tail_score_retention") for row in positives),
                    "tail_score_retention",
                ),
                "geometry_reliable_rate": _boolean_rate(
                    group,
                    "geometry_reliable",
                ),
                "rescue_rate": _boolean_rate(
                    group,
                    "formal_rescue_applied",
                ),
                "supports_paper_claim": bool(supports_paper_claim),
            }
        )
    return tuple(rows)
