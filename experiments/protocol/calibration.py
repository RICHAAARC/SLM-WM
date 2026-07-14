"""实验协议的 split、fixed-FPR 阈值与统计校验。"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping

from experiments.protocol.events import EventProtocolRecord
from experiments.protocol.prompts import PromptProtocolRecord
from experiments.protocol.splits import assert_disjoint_calibration_and_test, group_prompt_ids_by_split


_CLEAN_ATTACK_FAMILIES = frozenset({"", "none", "clean"})
_CLEAN_ATTACK_NAMES = frozenset({"", "none", "clean", "clean_none"})


def is_clean_unattacked_negative(
    record: Mapping[str, Any],
    *,
    split: str | None = None,
    expected_detection_key_role: str | None = None,
) -> bool:
    """判断记录是否为可用于 fixed-FPR 的未攻击 clean negative。

    该谓词同时检查攻击 ID、执行标志以及 family/name/condition,
    防止只清空 `attack_id` 的攻击记录混入 calibration 或 test clean 计数。
    """

    if not isinstance(record, Mapping):
        return False
    if split is not None and record.get("split") != split:
        return False
    if record.get("sample_role") != "clean_negative":
        return False
    if expected_detection_key_role is not None and record.get(
        "detection_key_role"
    ) != expected_detection_key_role:
        return False
    return bool(
        not str(record.get("attack_id", "")).strip()
        and record.get("attack_performed") is not True
        and str(record.get("attack_family", "")).strip().lower()
        in _CLEAN_ATTACK_FAMILIES
        and str(record.get("attack_name", "")).strip().lower()
        in _CLEAN_ATTACK_NAMES
        and str(record.get("attack_condition", "")).strip().lower()
        in _CLEAN_ATTACK_NAMES
    )


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

    该对象属于通用工程写法: 将目标 FPR 与样本角色集中到配置对象,
    让不含几何救回的 baseline 统计函数只关注阈值和指标计算。
    """

    target_fpr: float
    calibration_split: str = "calibration"
    evaluation_split: str = "test"
    positive_role: str = "positive_source"
    clean_negative_role: str = "clean_negative"
    attacked_negative_role: str = "attacked_negative"
    confidence_level: float = 0.95
    false_positive_budget_mode: str = "empirical"

    def __post_init__(self) -> None:
        """集中校验校准协议边界。"""
        if not 0.0 < self.target_fpr < 1.0:
            raise ValueError("target_fpr 必须位于 (0, 1)")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level 必须位于 (0, 1)")
        if self.false_positive_budget_mode not in {"empirical", "confidence_controlled"}:
            raise ValueError("false_positive_budget_mode 必须为 empirical 或 confidence_controlled")

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
    nominal_allowed_count = math.floor(config.target_fpr * len(scores))
    confidence_allowed_count = confidence_controlled_false_positive_budget(
        negative_count=len(scores),
        target_fpr=config.target_fpr,
        confidence_level=config.confidence_level,
    )
    allowed_count = (
        confidence_allowed_count
        if config.false_positive_budget_mode == "confidence_controlled"
        else nominal_allowed_count
    )
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
            "false_positive_budget_mode": config.false_positive_budget_mode,
            "nominal_allowed_false_positive_count": nominal_allowed_count,
            "confidence_controlled_false_positive_count": confidence_allowed_count,
            "confidence_level": config.confidence_level,
            "calibration_fpr_confidence_upper_bound": binomial_rate_upper_confidence_bound(
                observed_count,
                len(scores),
                config.confidence_level,
            ),
        },
    )


def binomial_rate_upper_confidence_bound(
    false_positive_count: int,
    negative_count: int,
    confidence_level: float,
) -> float:
    """计算 false positive rate 的单侧置信上界。

    该函数属于统计协议层: 三个论文运行层级使用同一上界计算方式, 但分别
    对应 FPR=0.1、FPR=0.01和 FPR=0.001。仅报告 observed FPR 仍不足以
    支撑正式论文结论, 因此这里提供一个无需外部依赖的 Wilson 单侧上界,
    用于选择更保守的 calibration false positive 预算。
    """

    if negative_count <= 0:
        return 1.0
    phat = max(0.0, min(1.0, false_positive_count / negative_count))
    z_value = normal_quantile_for_confidence(confidence_level)
    denominator = 1.0 + z_value * z_value / negative_count
    center = phat + z_value * z_value / (2.0 * negative_count)
    margin = z_value * math.sqrt((phat * (1.0 - phat) + z_value * z_value / (4.0 * negative_count)) / negative_count)
    return min(1.0, (center + margin) / denominator)


def normal_quantile_for_confidence(confidence_level: float) -> float:
    """返回常用单侧置信水平对应的正态分位近似。"""

    if confidence_level >= 0.999:
        return 3.090232306167813
    if confidence_level >= 0.99:
        return 2.3263478740408408
    if confidence_level >= 0.975:
        return 1.959963984540054
    if confidence_level >= 0.95:
        return 1.6448536269514722
    if confidence_level >= 0.90:
        return 1.2815515655446004
    return 1.0


def confidence_controlled_false_positive_budget(
    negative_count: int,
    target_fpr: float,
    confidence_level: float,
) -> int:
    """选择满足置信上界约束的 false positive 预算。

    返回值不会超过 nominal `floor(target_fpr * negative_count)`。如果样本量太小,
    即使 0 个 false positive 的置信上界也可能高于目标 FPR, 此时仍返回 0,
    但 downstream report 会通过置信上界字段明确说明论文声明边界尚未闭合。
    """

    if negative_count <= 0:
        return 0
    nominal_allowed_count = math.floor(target_fpr * negative_count)
    for false_positive_count in range(nominal_allowed_count, -1, -1):
        upper_bound = binomial_rate_upper_confidence_bound(
            false_positive_count,
            negative_count,
            confidence_level,
        )
        if upper_bound <= target_fpr:
            return false_positive_count
    return 0


def split_role(records: Iterable[dict[str, Any]], split: str, sample_role: str) -> tuple[dict[str, Any], ...]:
    """按 split 和样本角色筛选记录。"""
    return tuple(record for record in records if record.get("split") == split and record.get("sample_role") == sample_role)


def split_records(records: Iterable[dict[str, Any]], split: str) -> tuple[dict[str, Any], ...]:
    """按 split 筛选记录。

    该函数属于通用协议工具: 阈值冻结使用 calibration split, 论文指标报告使用
    evaluation split, 二者必须显式分离, 避免把 calibration 或 dev 样本混入正式表格。
    """

    return tuple(record for record in records if record.get("split") == split)


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


def threshold_score_field(threshold: FixedFprThreshold) -> str:
    """读取当前 fixed-FPR 正式判定使用的分数字段。"""

    return str(threshold.metadata.get("threshold_score_field", "raw_content_score"))


def decision_fields_at_threshold(
    record: dict[str, Any],
    threshold: FixedFprThreshold,
    config: FixedFprCalibrationConfig,
) -> dict[str, Any]:
    """在冻结阈值下重算不含几何救回的通用分数判定。"""
    raw_score = float(record["raw_content_score"])
    aligned_score = float(record["aligned_content_score"])
    formal_score_field = threshold_score_field(threshold)
    formal_score = float(
        record.get(
            formal_score_field,
            aligned_score if formal_score_field in {"aligned_content_score", "formal_detection_score"} else raw_score,
        )
    )
    raw_margin = raw_score - threshold.threshold_value
    aligned_margin = aligned_score - threshold.threshold_value
    formal_margin = formal_score - threshold.threshold_value
    positive_by_content = raw_margin >= 0.0
    formal_detection_decision = formal_margin >= 0.0
    formal_score_is_raw = formal_score_field == "raw_content_score"
    rescue_eligible = False
    rescue_applied = False
    evidence_decision = formal_detection_decision
    return {
        "raw_content_margin": raw_margin,
        "aligned_content_margin": aligned_margin,
        "formal_detection_score": formal_score,
        "formal_detection_margin": formal_margin,
        "threshold_score_field": formal_score_field,
        "positive_by_content": positive_by_content,
        "formal_detection_decision": formal_detection_decision,
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
    formal_score_field = threshold_score_field(threshold)
    return {
        "operating_point_id": f"fixed_fpr_{config.target_fpr:g}",
        "target_fpr": config.target_fpr,
        "calibrated_content_threshold": threshold.threshold_value,
        "calibrated_detection_threshold": threshold.threshold_value,
        "threshold_score_field": formal_score_field,
        "threshold_degenerate": threshold.threshold_degenerate,
        "positive_count": len(positives),
        "clean_negative_count": len(clean_negatives),
        "attacked_negative_count": len(attacked_negatives),
        "true_positive_rate": binary_rate(positives, "evidence_decision"),
        "raw_content_clean_fpr": binary_rate(clean_negatives, "positive_by_content"),
        "formal_detection_score_clean_fpr": binary_rate(clean_negatives, "formal_detection_decision"),
        "formal_detection_score_attacked_fpr": binary_rate(attacked_negatives, "formal_detection_decision"),
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
        "full_method_component_ready": False,
        "supports_paper_claim": False,
    }


def fixed_threshold_rate(records: Iterable[dict[str, Any]], score_field: str, threshold_value: float) -> float:
    """按指定分数字段计算固定阈值通过率。"""
    record_tuple = tuple(records)
    if not record_tuple:
        return 0.0
    return sum(1 for record in record_tuple if float(record[score_field]) >= threshold_value) / len(record_tuple)


def score_mode_operating_point_rows(
    records: Iterable[dict[str, Any]],
    threshold: FixedFprThreshold,
    config: FixedFprCalibrationConfig,
) -> list[dict[str, Any]]:
    """按 raw、aligned 和 evidence 三种判定模式输出 fixed-FPR 诊断行。

    该函数属于协议诊断层: 它不改变正式阈值, 只显式展示 raw score、
    aligned score 与 rescue 后 evidence decision 各自的 TPR/FPR, 用于排查
    clean negative 高尾、aligned score 增益不足或 rescue 窗口过窄等问题。
    """

    calibrated = tuple(records)
    positives = tuple(record for record in calibrated if record["sample_role"] == config.positive_role)
    clean_negatives = tuple(record for record in calibrated if record["sample_role"] == config.clean_negative_role)
    attacked_negatives = tuple(record for record in calibrated if record["sample_role"] == config.attacked_negative_role)
    formal_score_field = threshold_score_field(threshold)
    mode_specs = (
        ("raw_content_threshold", "raw_content_score", "raw_score_auc", formal_score_field == "raw_content_score"),
        (
            "aligned_content_threshold",
            "aligned_content_score",
            "aligned_score_auc",
            formal_score_field in {"aligned_content_score", "formal_detection_score"},
        ),
    )
    rows = [
        {
            "decision_mode": decision_mode,
            "score_field": score_field,
            "target_fpr": config.target_fpr,
            "threshold_value": threshold.threshold_value,
            "positive_count": len(positives),
            "clean_negative_count": len(clean_negatives),
            "attacked_negative_count": len(attacked_negatives),
            "true_positive_rate": fixed_threshold_rate(positives, score_field, threshold.threshold_value),
            "clean_false_positive_rate": fixed_threshold_rate(clean_negatives, score_field, threshold.threshold_value),
            "attacked_false_positive_rate": fixed_threshold_rate(attacked_negatives, score_field, threshold.threshold_value),
            "score_auc": score_auc(
                (record[score_field] for record in positives),
                (record[score_field] for record in clean_negatives),
            ),
            "governs_fixed_fpr": governs_fixed_fpr,
            "supports_paper_claim": False,
        }
        for decision_mode, score_field, _, governs_fixed_fpr in mode_specs
    ]
    rows.append(
        {
            "decision_mode": "formal_detection_threshold",
            "score_field": formal_score_field,
            "target_fpr": config.target_fpr,
            "threshold_value": threshold.threshold_value,
            "positive_count": len(positives),
            "clean_negative_count": len(clean_negatives),
            "attacked_negative_count": len(attacked_negatives),
            "true_positive_rate": binary_rate(positives, "formal_detection_decision"),
            "clean_false_positive_rate": binary_rate(clean_negatives, "formal_detection_decision"),
            "attacked_false_positive_rate": binary_rate(attacked_negatives, "formal_detection_decision"),
            "score_auc": score_auc(
                (record["formal_detection_score"] for record in positives),
                (record["formal_detection_score"] for record in clean_negatives),
            ),
            "governs_fixed_fpr": True,
            "supports_paper_claim": False,
        }
    )
    rows.append(
        {
            "decision_mode": "evidence_after_rescue",
            "score_field": "evidence_decision",
            "target_fpr": config.target_fpr,
            "threshold_value": threshold.threshold_value,
            "positive_count": len(positives),
            "clean_negative_count": len(clean_negatives),
            "attacked_negative_count": len(attacked_negatives),
            "true_positive_rate": binary_rate(positives, "evidence_decision"),
            "clean_false_positive_rate": binary_rate(clean_negatives, "evidence_decision"),
            "attacked_false_positive_rate": binary_rate(attacked_negatives, "evidence_decision"),
            "score_auc": 0.0,
            "governs_fixed_fpr": True,
            "supports_paper_claim": False,
        }
    )
    return rows
