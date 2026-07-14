"""校验 observation 是否真正共享由 calibration clean negative 冻结的 fixed-FPR 阈值。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from experiments.protocol.image_only_evidence import (
    CALIBRATION_PARTITION_PROTOCOL,
    partition_calibration_prompt_ids,
)
from experiments.protocol.calibration import is_clean_unattacked_negative
from main.core.digest import build_stable_digest


FORMAL_THRESHOLD_SOURCE = "nested_calibration_threshold_freeze_conformal"


@dataclass(frozen=True)
class FixedFprObservationAudit:
    """保存 observation 级阈值冻结审计结果。

    该对象属于通用工程写法: 统一校验阈值数值、阈值来源和逐条判定，避免
    不同 baseline 导入路径仅凭布尔 ready 字段或来源文本声明正式就绪。
    """

    observation_count: int
    calibration_source_negative_count: int
    expected_calibration_source_negative_count: int
    threshold_freeze_negative_count: int
    expected_threshold_freeze_negative_count: int
    calibration_partition_digest: str
    threshold_freeze_prompt_id_digest: str
    calibration_false_positive_count: int
    frozen_threshold: float | None
    expected_frozen_threshold: float | None
    calibration_partition_ready: bool
    observation_fields_ready: bool
    threshold_source_ready: bool
    threshold_value_ready: bool
    detection_decision_ready: bool
    empirical_fpr_ready: bool
    fixed_fpr_ready: bool
    threshold_digest: str


def conformal_threshold_from_clean_negative_scores(
    clean_negative_scores: Iterable[float],
    target_fpr: float,
) -> float:
    """按照 baseline adapter 的 conformal 规则冻结 fixed-FPR 阈值。

    该函数只消费 calibration clean negative 分数。它可以被不同 baseline
    adapter、结果导入器和审计器共同复用，确保阈值生成与阈值核验采用同一规则。
    """

    if not 0.0 < float(target_fpr) < 1.0:
        raise ValueError("target_fpr 必须位于 (0, 1)")
    scores = tuple(float(score) for score in clean_negative_scores)
    if not scores or not all(math.isfinite(score) for score in scores):
        raise ValueError("calibration clean negative 分数必须是非空有限数值集合")
    allowed_false_positives = max(
        0,
        math.floor(float(target_fpr) * (len(scores) + 1)) - 1,
    )
    for threshold in sorted({math.nextafter(score, math.inf) for score in scores}):
        if sum(score >= threshold for score in scores) <= allowed_false_positives:
            return threshold
    raise RuntimeError("无法从 calibration clean negative 冻结 fixed-FPR 阈值")


def _finite_float(row: Mapping[str, Any], field_name: str) -> float | None:
    """读取有限浮点字段，非法或缺失时返回空值。"""

    if field_name not in row:
        return None
    try:
        value = float(row[field_name])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _boolean_value(row: Mapping[str, Any], field_name: str) -> bool | None:
    """读取严格布尔字段，并兼容 JSON 或 CSV 的常见布尔文本。"""

    if field_name not in row:
        return None
    value = row[field_name]
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def audit_fixed_fpr_observation_threshold(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_fpr: float,
    expected_calibration_source_negative_count: int,
    threshold_source: str = FORMAL_THRESHOLD_SOURCE,
) -> FixedFprObservationAudit:
    """核验 observation 是否使用共享 threshold-freeze 子集冻结阈值。

    此处同时重算阈值和逐条 detection decision。这样即使外部记录伪造
    `threshold_source`，或直接写入与 score 不一致的 decision，也不能通过正式门禁。
    window-fit、test 与 dev 分数只参与共享阈值一致性检查，不参与阈值重算。
    """

    if expected_calibration_source_negative_count < 3:
        raise ValueError(
            "expected_calibration_source_negative_count 必须至少为3"
        )
    materialized_rows = tuple(dict(row) for row in rows)
    calibration_source_rows = tuple(
        row
        for row in materialized_rows
        if is_clean_unattacked_negative(row, split="calibration")
    )
    try:
        _, threshold_freeze_prompt_ids, calibration_partition_digest = (
            partition_calibration_prompt_ids(
                str(row.get("prompt_id", ""))
                for row in calibration_source_rows
            )
        )
    except ValueError:
        threshold_freeze_prompt_ids = ()
        calibration_partition_digest = ""
    threshold_freeze_prompt_id_set = set(threshold_freeze_prompt_ids)
    threshold_freeze_rows = tuple(
        row
        for row in calibration_source_rows
        if str(row.get("prompt_id", "")) in threshold_freeze_prompt_id_set
    )
    calibration_scores = tuple(
        _finite_float(row, "score") for row in threshold_freeze_rows
    )
    calibration_scores_ready = bool(threshold_freeze_rows) and all(
        score is not None for score in calibration_scores
    )
    expected_threshold = (
        conformal_threshold_from_clean_negative_scores(
            (float(score) for score in calibration_scores if score is not None),
            target_fpr,
        )
        if calibration_scores_ready
        else None
    )

    row_scores = tuple(_finite_float(row, "score") for row in materialized_rows)
    row_thresholds = tuple(_finite_float(row, "threshold") for row in materialized_rows)
    row_decisions = tuple(_boolean_value(row, "detection_decision") for row in materialized_rows)
    observation_fields_ready = bool(materialized_rows) and all(
        score is not None and threshold is not None and decision is not None
        for score, threshold, decision in zip(row_scores, row_thresholds, row_decisions)
    )
    frozen_threshold = row_thresholds[0] if observation_fields_ready else None
    shared_threshold_ready = bool(
        frozen_threshold is not None
        and all(
            math.isclose(float(threshold), float(frozen_threshold), rel_tol=0.0, abs_tol=1e-12)
            for threshold in row_thresholds
            if threshold is not None
        )
    )
    threshold_value_ready = bool(
        shared_threshold_ready
        and expected_threshold is not None
        and frozen_threshold is not None
        and math.isclose(float(frozen_threshold), float(expected_threshold), rel_tol=0.0, abs_tol=1e-12)
    )
    threshold_source_ready = bool(materialized_rows) and all(
        str(row.get("threshold_source", "")) == threshold_source for row in materialized_rows
    )
    detection_decision_ready = bool(
        observation_fields_ready
        and all(
            bool(decision) == (float(score) >= float(threshold))
            for score, threshold, decision in zip(row_scores, row_thresholds, row_decisions)
            if score is not None and threshold is not None and decision is not None
        )
    )
    calibration_false_positive_count = (
        sum(float(score) >= float(frozen_threshold) for score in calibration_scores if score is not None)
        if frozen_threshold is not None
        else 0
    )
    empirical_fpr_ready = bool(
        threshold_freeze_rows
        and calibration_false_positive_count / len(threshold_freeze_rows)
        <= float(target_fpr)
    )
    expected_threshold_freeze_count = (
        int(expected_calibration_source_negative_count)
        - int(expected_calibration_source_negative_count) // 3
    )
    calibration_partition_ready = bool(
        calibration_partition_digest
        and len(calibration_source_rows)
        == int(expected_calibration_source_negative_count)
        and len(threshold_freeze_rows) == expected_threshold_freeze_count
        and len(threshold_freeze_prompt_ids) == expected_threshold_freeze_count
    )
    fixed_fpr_ready = all(
        (
            calibration_partition_ready,
            calibration_scores_ready,
            observation_fields_ready,
            threshold_source_ready,
            threshold_value_ready,
            detection_decision_ready,
            empirical_fpr_ready,
        )
    )
    threshold_digest = (
        build_stable_digest(
            {
                "target_fpr": float(target_fpr),
                "threshold_source": threshold_source,
                "calibration_partition_protocol": (
                    CALIBRATION_PARTITION_PROTOCOL
                ),
                "calibration_partition_digest": (
                    calibration_partition_digest
                ),
                "calibration_source_negative_count": len(
                    calibration_source_rows
                ),
                "threshold_freeze_negative_count": len(
                    threshold_freeze_rows
                ),
                "threshold_freeze_prompt_id_digest": build_stable_digest(
                    list(threshold_freeze_prompt_ids)
                ),
                "calibration_scores": sorted(
                    (
                        str(row.get("prompt_id", "")),
                        str(row.get("event_id", "")),
                        float(score),
                    )
                    for row, score in zip(
                        threshold_freeze_rows,
                        calibration_scores,
                    )
                    if score is not None
                ),
                "frozen_threshold": float(frozen_threshold),
            }
        )
        if fixed_fpr_ready and frozen_threshold is not None
        else ""
    )
    return FixedFprObservationAudit(
        observation_count=len(materialized_rows),
        calibration_source_negative_count=len(calibration_source_rows),
        expected_calibration_source_negative_count=int(
            expected_calibration_source_negative_count
        ),
        threshold_freeze_negative_count=len(threshold_freeze_rows),
        expected_threshold_freeze_negative_count=(
            expected_threshold_freeze_count
        ),
        calibration_partition_digest=calibration_partition_digest,
        threshold_freeze_prompt_id_digest=(
            build_stable_digest(list(threshold_freeze_prompt_ids))
            if threshold_freeze_prompt_ids
            else ""
        ),
        calibration_false_positive_count=calibration_false_positive_count,
        frozen_threshold=frozen_threshold,
        expected_frozen_threshold=expected_threshold,
        calibration_partition_ready=calibration_partition_ready,
        observation_fields_ready=observation_fields_ready,
        threshold_source_ready=threshold_source_ready,
        threshold_value_ready=threshold_value_ready,
        detection_decision_ready=detection_decision_ready,
        empirical_fpr_ready=empirical_fpr_ready,
        fixed_fpr_ready=fixed_fpr_ready,
        threshold_digest=threshold_digest,
    )
