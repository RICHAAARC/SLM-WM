"""实验协议的 split、fixed-FPR 阈值与统计校验。"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

from experiments.protocol.events import EventProtocolRecord
from experiments.protocol.pilot_paper_fixed_fpr import PILOT_PAPER_FIXED_FPR
from experiments.protocol.prompts import PromptProtocolRecord
from experiments.protocol.splits import assert_disjoint_calibration_and_test, group_prompt_ids_by_split


def build_prompt_statistics(
    prompt_records: Iterable[PromptProtocolRecord],
    event_records: Iterable[EventProtocolRecord],
) -> dict[str, Any]:
    """构造 prompt 与事件协议统计。"""
    prompt_tuple = tuple(prompt_records)
    event_tuple = tuple(event_records)
    split_groups = group_prompt_ids_by_split(prompt_tuple)
    split_counts = {name: len(ids) for name, ids in split_groups.items()}
    sample_role_counts = Counter(event.sample_role for event in event_tuple)
    prompt_set_counts = Counter(prompt.prompt_set for prompt in prompt_tuple)
    calibration_test_disjoint = assert_disjoint_calibration_and_test(split_groups)
    return {
        "protocol_decision": "pass" if calibration_test_disjoint else "fail",
        "prompt_count": len(prompt_tuple),
        "event_count": len(event_tuple),
        "split_counts": split_counts,
        "sample_role_counts": dict(sorted(sample_role_counts.items())),
        "prompt_set_counts": dict(sorted(prompt_set_counts.items())),
        "calibration_test_disjoint": calibration_test_disjoint,
        "supports_paper_claim": False,
    }


@dataclass(frozen=True)
class FixedFprCalibrationConfig:
    """描述 fixed-FPR 阈值校准协议。

    该对象属于通用工程写法: 将目标 FPR、样本角色和 rescue 窗口集中
    到配置对象, 让统计函数只关注阈值和指标计算。
    """

    target_fpr: float = PILOT_PAPER_FIXED_FPR
    calibration_split: str = "calibration"
    positive_role: str = "positive_source"
    clean_negative_role: str = "clean_negative"
    attacked_negative_role: str = "attacked_negative"
    rescue_margin_low: float = -0.05
    allowed_fail_reasons: tuple[str, ...] = ("geometry_suspected", "low_confidence")

    def __post_init__(self) -> None:
        """集中校验校准协议边界。"""
        if not 0.0 < self.target_fpr < 1.0:
            raise ValueError("target_fpr 必须位于 (0, 1)")
        if self.rescue_margin_low >= 0.0:
            raise ValueError("rescue_margin_low 必须小于 0")

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


@dataclass(frozen=True)
class FixedFprThreshold:
    """记录由 calibration clean negative 分数冻结的内容阈值。"""

    threshold_name: str
    threshold_value: float
    target_fpr: float
    observed_fpr: float
    calibration_negative_count: int
    allowed_false_positive_count: int
    observed_false_positive_count: int
    threshold_tie_count: int
    threshold_degenerate: bool
    supports_paper_claim: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 兼容字典。"""
        return asdict(self)


def empirical_threshold_at_fpr(clean_negative_scores: Iterable[float], config: FixedFprCalibrationConfig) -> FixedFprThreshold:
    """从 calibration clean negative 分数中冻结 fixed-FPR 内容阈值。

    该函数只使用 clean negative calibration split, 不读取 test split,
    因此可复用于后续正式 split 协议。
    """
    scores = tuple(float(score) for score in clean_negative_scores)
    if not scores:
        raise ValueError("clean_negative_scores 不得为空")
    ordered_desc = tuple(sorted(scores, reverse=True))
    allowed_count = math.floor(config.target_fpr * len(scores))
    if allowed_count <= 0:
        threshold = max(scores) + 1e-12
    else:
        threshold = ordered_desc[allowed_count - 1]
    observed_count = sum(1 for score in scores if score >= threshold)
    observed_fpr = observed_count / len(scores)
    threshold_tie_count = sum(1 for score in scores if math.isclose(score, threshold, rel_tol=0.0, abs_tol=1e-12))
    threshold_degenerate = observed_fpr > config.target_fpr or threshold_tie_count > max(1, allowed_count)
    return FixedFprThreshold(
        threshold_name="content_threshold_fixed_fpr",
        threshold_value=threshold,
        target_fpr=config.target_fpr,
        observed_fpr=observed_fpr,
        calibration_negative_count=len(scores),
        allowed_false_positive_count=allowed_count,
        observed_false_positive_count=observed_count,
        threshold_tie_count=threshold_tie_count,
        threshold_degenerate=threshold_degenerate,
        supports_paper_claim=False,
        metadata={
            "calibration_split": config.calibration_split,
            "threshold_source": "calibration_clean_negative",
        },
    )


def split_role(records: Iterable[dict[str, Any]], split: str, sample_role: str) -> tuple[dict[str, Any], ...]:
    """按 split 和样本角色筛选记录。"""
    return tuple(record for record in records if record.get("split") == split and record.get("sample_role") == sample_role)


def binary_rate(records: Iterable[dict[str, Any]], field_name: str) -> float:
    """计算布尔字段触发率。"""
    record_tuple = tuple(records)
    if not record_tuple:
        return 0.0
    return sum(1 for record in record_tuple if bool(record.get(field_name, False))) / len(record_tuple)


def mean_value(records: Iterable[dict[str, Any]], field_name: str) -> float:
    """计算数值字段均值。"""
    values = [float(record.get(field_name, 0.0)) for record in records]
    return sum(values) / len(values) if values else 0.0


def score_auc(positive_scores: Iterable[float], negative_scores: Iterable[float]) -> float:
    """计算二分类 AUC, 分数越高越偏向 positive。"""
    positives = tuple(float(score) for score in positive_scores)
    negatives = tuple(float(score) for score in negative_scores)
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif math.isclose(positive, negative, rel_tol=0.0, abs_tol=1e-12):
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def decision_fields_at_threshold(
    record: dict[str, Any],
    threshold: FixedFprThreshold,
    config: FixedFprCalibrationConfig,
) -> dict[str, Any]:
    """在冻结阈值下重算 raw content 与 rescue 后 evidence 判定。"""
    raw_score = float(record["raw_content_score"])
    aligned_score = float(record["aligned_content_score"])
    raw_margin = raw_score - threshold.threshold_value
    aligned_margin = aligned_score - threshold.threshold_value
    positive_by_content = raw_margin >= 0.0
    rescue_eligible = (
        config.rescue_margin_low <= raw_margin < 0.0
        and bool(record.get("geometry_reliable", False))
        and str(record.get("fail_reason", "")) in config.allowed_fail_reasons
        and record.get("rescue_ablation_mode") == "full_rescue"
    )
    rescue_applied = rescue_eligible and aligned_margin >= 0.0
    evidence_decision = positive_by_content or rescue_applied
    return {
        "raw_content_margin": raw_margin,
        "aligned_content_margin": aligned_margin,
        "positive_by_content": positive_by_content,
        "rescue_eligible": rescue_eligible,
        "rescue_applied": rescue_applied,
        "evidence_decision": evidence_decision,
    }


def calibrated_records(
    records: Iterable[dict[str, Any]],
    threshold: FixedFprThreshold,
    config: FixedFprCalibrationConfig,
) -> tuple[dict[str, Any], ...]:
    """用冻结阈值重建校准后记录。"""
    return tuple({**record, **decision_fields_at_threshold(record, threshold, config)} for record in records)


def operating_point_metrics(
    records: Iterable[dict[str, Any]],
    threshold: FixedFprThreshold,
    config: FixedFprCalibrationConfig,
) -> dict[str, Any]:
    """计算 fixed-FPR operating point 下的核心统计。"""
    calibrated = tuple(records)
    positives = tuple(record for record in calibrated if record["sample_role"] == config.positive_role)
    clean_negatives = tuple(record for record in calibrated if record["sample_role"] == config.clean_negative_role)
    attacked_negatives = tuple(record for record in calibrated if record["sample_role"] == config.attacked_negative_role)
    return {
        "operating_point_id": f"fixed_fpr_{config.target_fpr:g}",
        "target_fpr": config.target_fpr,
        "calibrated_content_threshold": threshold.threshold_value,
        "threshold_degenerate": threshold.threshold_degenerate,
        "positive_count": len(positives),
        "clean_negative_count": len(clean_negatives),
        "attacked_negative_count": len(attacked_negatives),
        "true_positive_rate": binary_rate(positives, "evidence_decision"),
        "raw_content_clean_fpr": binary_rate(clean_negatives, "positive_by_content"),
        "evidence_clean_fpr": binary_rate(clean_negatives, "evidence_decision"),
        "evidence_attacked_fpr": binary_rate(attacked_negatives, "evidence_decision"),
        "rescue_applied_rate": binary_rate(calibrated, "rescue_applied"),
        "aligned_score_gain_mean": mean_value(calibrated, "rescue_score_gain"),
        "raw_score_auc": score_auc(
            (record["raw_content_score"] for record in positives),
            (record["raw_content_score"] for record in clean_negatives),
        ),
        "aligned_score_auc": score_auc(
            (record["aligned_content_score"] for record in positives),
            (record["aligned_content_score"] for record in clean_negatives),
        ),
        "full_method_claim_ready": False,
        "supports_paper_claim": False,
    }


def threshold_grid(records: Iterable[dict[str, Any]], score_field: str = "raw_content_score", max_points: int = 21) -> tuple[float, ...]:
    """构造 ROC / DET 曲线使用的阈值网格。"""
    scores = sorted({float(record[score_field]) for record in records})
    if not scores:
        return tuple()
    if len(scores) <= max_points:
        return tuple(reversed(scores))
    indices = [round(index * (len(scores) - 1) / (max_points - 1)) for index in range(max_points)]
    return tuple(reversed([scores[index] for index in indices]))


def curve_rows(records: Iterable[dict[str, Any]], config: FixedFprCalibrationConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """生成 ROC 与 DET 曲线点。"""
    record_tuple = tuple(records)
    positives = tuple(record for record in record_tuple if record["sample_role"] == config.positive_role)
    clean_negatives = tuple(record for record in record_tuple if record["sample_role"] == config.clean_negative_role)
    roc_rows: list[dict[str, Any]] = []
    det_rows: list[dict[str, Any]] = []
    for threshold in threshold_grid(record_tuple):
        true_positive_rate = (
            sum(1 for record in positives if float(record["raw_content_score"]) >= threshold) / len(positives)
            if positives
            else 0.0
        )
        false_positive_rate = (
            sum(1 for record in clean_negatives if float(record["raw_content_score"]) >= threshold) / len(clean_negatives)
            if clean_negatives
            else 0.0
        )
        false_negative_rate = 1.0 - true_positive_rate
        roc_rows.append(
            {
                "roc_threshold": threshold,
                "true_positive_rate": true_positive_rate,
                "false_positive_rate": false_positive_rate,
                "supports_paper_claim": False,
            }
        )
        det_rows.append(
            {
                "det_threshold": threshold,
                "det_false_positive_rate": false_positive_rate,
                "det_false_negative_rate": false_negative_rate,
                "supports_paper_claim": False,
            }
        )
    return roc_rows, det_rows


def score_distribution_rows(records: Iterable[dict[str, Any]], bin_count: int = 10) -> list[dict[str, Any]]:
    """生成分数分布表。"""
    record_tuple = tuple(records)
    scores = [float(record["raw_content_score"]) for record in record_tuple]
    if not scores:
        return []
    lower_bound = min(scores)
    upper_bound = max(scores)
    width = (upper_bound - lower_bound) / bin_count if upper_bound > lower_bound else 1.0
    rows: list[dict[str, Any]] = []
    for sample_role in sorted({str(record["sample_role"]) for record in record_tuple}):
        role_records = [record for record in record_tuple if record["sample_role"] == sample_role]
        for index in range(bin_count):
            lower = lower_bound + width * index
            upper = lower_bound + width * (index + 1)
            count = sum(
                1
                for record in role_records
                if lower <= float(record["raw_content_score"]) < upper
                or (index == bin_count - 1 and float(record["raw_content_score"]) <= upper)
            )
            rows.append(
                {
                    "sample_role": sample_role,
                    "score_distribution_bin": f"[{lower:.6f},{upper:.6f}]",
                    "score_count": count,
                    "score_min": min(float(record["raw_content_score"]) for record in role_records),
                    "score_max": max(float(record["raw_content_score"]) for record in role_records),
                    "score_mean": mean_value(role_records, "raw_content_score"),
                    "supports_paper_claim": False,
                }
            )
    return rows
