"""从仅图像检测原始记录构造可重建的测试指标行."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable

from experiments.protocol.calibration import binomial_rate_upper_confidence_bound


def _strict_boolean(value: Any, field_name: str) -> bool:
    """只接受 JSON 布尔值, 避免非空字符串被误解释为真."""

    if not isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是布尔值")
    return value


def _finite_float(value: Any, field_name: str) -> float:
    """读取有限浮点值并在原始证据非法时立即阻断."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是有限数值, 不得使用布尔值")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是有限数值") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{field_name} 必须是有限数值")
    return resolved


def _complete_finite_values(
    records: Iterable[dict[str, Any]],
    field_name: str,
    measurement_name: str,
) -> list[float]:
    """要求每条正式记录都携带同一有限测量, 禁止选择性聚合子集."""

    values: list[float] = []
    for record in records:
        if field_name not in record or record[field_name] is None:
            raise ValueError(
                f"每条 test detection record 必须包含实测 {measurement_name}"
            )
        values.append(_finite_float(record[field_name], field_name))
    return values


def _psnr_value(record: dict[str, Any]) -> float:
    """读取完整 PSNR 证据, 仅为完全相同图像保留正无穷语义."""

    raw_value = record.get("source_to_evaluated_psnr")
    if isinstance(raw_value, bool):
        raise ValueError("source_to_evaluated_psnr 不得使用布尔值")
    try:
        resolved = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "每条 test detection record 必须包含实测 PSNR"
        ) from exc
    if math.isnan(resolved) or resolved == -math.inf:
        raise ValueError("PSNR 必须有限或为有效正无穷")
    if resolved == math.inf:
        mse = _finite_float(
            record.get("source_to_evaluated_mse"),
            "source_to_evaluated_mse",
        )
        if mse != 0.0:
            raise ValueError("正无穷 PSNR 仅允许对应图像的实测 MSE 为0")
    return resolved


def build_image_only_test_metric_rows(
    records: Iterable[dict[str, Any]],
    target_fpr: float,
) -> tuple[dict[str, Any], ...]:
    """按攻击身份和样本角色聚合 test split 的原始检测事实.

    该函数属于可复用的证据重建原语.运行器、结果记录构造器和最终门禁
    可以消费同一实现, 从而避免各自维护一套计数与质量聚合规则.
    """

    resolved_target_fpr = _finite_float(target_fpr, "target_fpr")
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw_record in records:
        record = dict(raw_record)
        if record.get("split") != "test":
            continue
        attack_family = str(record.get("attack_family", "clean"))
        attack_name = str(record.get("attack_name", "none"))
        resource_profile = str(record.get("resource_profile", "clean"))
        sample_role = str(record.get("sample_role", ""))
        if not sample_role:
            raise ValueError("test detection record 缺少 sample_role")
        grouped[(attack_family, attack_name, resource_profile, sample_role)].append(
            record
        )

    rows: list[dict[str, Any]] = []
    for (
        attack_family,
        attack_name,
        resource_profile,
        sample_role,
    ), group_records in sorted(grouped.items()):
        positive_count = sum(
            _strict_boolean(
                record.get("formal_evidence_positive"),
                "formal_evidence_positive",
            )
            for record in group_records
        )
        rate = positive_count / len(group_records)
        upper = binomial_rate_upper_confidence_bound(
            positive_count,
            len(group_records),
            0.95,
        )
        quality_ssim_values = _complete_finite_values(
            group_records,
            "source_to_evaluated_ssim",
            "SSIM",
        )
        quality_psnr_values = [_psnr_value(record) for record in group_records]
        infinite_psnr_count = sum(
            value == math.inf for value in quality_psnr_values
        )
        if infinite_psnr_count not in {0, len(quality_psnr_values)}:
            raise ValueError(
                "同一 test 指标聚合组不得混合有限 PSNR 与正无穷 PSNR"
            )
        quality_psnr_mean = (
            math.inf
            if infinite_psnr_count
            else sum(quality_psnr_values) / len(quality_psnr_values)
        )
        content_scores = _complete_finite_values(
            group_records,
            "content_score",
            "content_score",
        )
        rows.append(
            {
                "attack_family": attack_family,
                "attack_name": attack_name,
                "resource_profile": resource_profile,
                "sample_role": sample_role,
                "record_count": len(group_records),
                "positive_count": positive_count,
                "positive_rate": rate,
                "content_score_mean": sum(content_scores) / len(content_scores),
                "source_to_evaluated_ssim_mean": (
                    sum(quality_ssim_values) / len(quality_ssim_values)
                ),
                "source_to_evaluated_psnr_mean": quality_psnr_mean,
                "positive_rate_upper_95": upper,
                "target_fpr": resolved_target_fpr,
                "fixed_fpr_upper_bound_ready": (
                    sample_role in {"clean_negative", "wrong_key_negative"}
                    and upper <= resolved_target_fpr
                ),
                "metric_status": "measured_image_only_detection",
                "supports_paper_claim": False,
            }
        )
    return tuple(rows)
