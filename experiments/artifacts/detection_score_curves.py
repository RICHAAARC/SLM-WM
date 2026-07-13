"""从仅图像检测连续分数重建真实 ROC,DET 与分数分布表."""

from __future__ import annotations

from collections import defaultdict
import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


POSITIVE_SAMPLE_ROLES = frozenset({"positive_source"})
NEGATIVE_SAMPLE_ROLES = frozenset({"clean_negative", "wrong_key_negative"})

SCORE_DISTRIBUTION_FIELDNAMES = (
    "sample_scope",
    "attack_family",
    "attack_name",
    "resource_profile",
    "observation_id",
    "prompt_id",
    "sample_role",
    "binary_label",
    "score",
    "raw_content_score",
    "aligned_content_score",
    "threshold",
    "tp",
    "fp",
    "tn",
    "fn",
    "tpr",
    "fpr",
    "fnr",
    "sample_count",
    "positive_count",
    "negative_count",
    "threshold_digest",
    "metric_status",
)

CURVE_POINT_FIELDNAMES = (
    "sample_scope",
    "attack_family",
    "attack_name",
    "resource_profile",
    "curve_point_index",
    "threshold_kind",
    "threshold",
    "tp",
    "fp",
    "tn",
    "fn",
    "tpr",
    "fpr",
    "fnr",
    "sample_count",
    "positive_count",
    "negative_count",
    "threshold_digest",
    "metric_status",
)


def _finite_float(value: Any, field_name: str) -> float:
    """把外部记录字段转为有限浮点数, 避免静默生成无效曲线."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是数值")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field_name} 必须是有限数值")
    return resolved


def _binary_label(sample_role: Any) -> int:
    """把正式 sample role 映射为二分类真实标签."""

    role = str(sample_role)
    if role in POSITIVE_SAMPLE_ROLES:
        return 1
    if role in NEGATIVE_SAMPLE_ROLES:
        return 0
    raise ValueError(f"不支持用于检测曲线的 sample_role: {role}")


def _strict_boolean(value: Any, field_name: str) -> bool:
    """只接受 JSON 布尔值, 避免文本 truthiness 改写正式判定."""

    if not isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是布尔值")
    return value


def decision_equivalent_score(
    record: Mapping[str, Any],
    *,
    rescue_margin_low: float,
    geometry_score_threshold: float,
    registration_confidence_threshold: float,
    attention_sync_score_threshold: float,
) -> float:
    """把内容主判与几何救回写成等价的连续检测分数.

    对几何可靠记录, 救回只允许在 ``rescue_margin_low <= raw-threshold < 0``
    的冻结带宽内使用 aligned score.因此能够保持阳性判定的最大阈值为
    ``min(aligned_score, raw_score-rescue_margin_low)``.该值与 raw score 取最大值后,
    对任意阈值执行 ``score >= threshold`` 与正式布尔协议完全等价.
    """

    raw_score = _finite_float(record.get("content_score"), "content_score")
    resolved_rescue_margin = _finite_float(rescue_margin_low, "rescue_margin_low")
    if resolved_rescue_margin >= 0.0:
        raise ValueError("rescue_margin_low 必须小于0")
    resolved_geometry_threshold = _finite_float(
        geometry_score_threshold,
        "geometry_score_threshold",
    )
    resolved_registration_threshold = _finite_float(
        registration_confidence_threshold,
        "registration_confidence_threshold",
    )
    resolved_sync_threshold = _finite_float(
        attention_sync_score_threshold,
        "attention_sync_score_threshold",
    )
    aligned_score_value = record.get("aligned_content_score")
    if aligned_score_value is None:
        return raw_score
    aligned_score = _finite_float(aligned_score_value, "aligned_content_score")
    alignment = record.get("alignment")
    alignment_reliable = bool(
        _strict_boolean(
            alignment.get("geometry_reliable"),
            "alignment.geometry_reliable",
        )
        if isinstance(alignment, Mapping)
        else False
    )

    def finite_at_least(value: Any, threshold: float) -> bool:
        """复现冻结布尔协议对可选几何数值的有限性门禁."""

        return bool(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= threshold
        )

    geometry_reliable = (
        alignment_reliable
        and finite_at_least(
            record.get("attention_geometry_score"),
            resolved_geometry_threshold,
        )
        and finite_at_least(
            record.get("registration_confidence"),
            resolved_registration_threshold,
        )
        and finite_at_least(
            record.get("attention_sync_score"),
            resolved_sync_threshold,
        )
    )
    if not geometry_reliable:
        return raw_score
    rescue_upper_threshold = min(
        aligned_score,
        raw_score - resolved_rescue_margin,
    )
    return max(raw_score, rescue_upper_threshold)


def _scope_key(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """构造单一攻击条件的稳定分组键."""

    return (
        "test_attack_condition",
        str(record.get("attack_family", "none")),
        str(record.get("attack_name", "none")),
        str(record.get("resource_profile", "clean")),
    )


def _confusion(
    observations: Iterable[Mapping[str, Any]],
    threshold: float,
) -> dict[str, int | float]:
    """按一个明确阈值计算完整二分类混淆矩阵."""

    values = tuple(observations)
    positive_count = sum(int(row["binary_label"]) == 1 for row in values)
    negative_count = len(values) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("每个检测曲线 scope 必须同时包含阳性与阴性记录")
    tp = sum(int(row["binary_label"]) == 1 and float(row["score"]) >= threshold for row in values)
    fp = sum(int(row["binary_label"]) == 0 and float(row["score"]) >= threshold for row in values)
    fn = positive_count - tp
    tn = negative_count - fp
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tp / positive_count,
        "fpr": fp / negative_count,
        "fnr": fn / positive_count,
        "sample_count": len(values),
        "positive_count": positive_count,
        "negative_count": negative_count,
    }


def _confusion_from_counts(
    *,
    tp: int,
    fp: int,
    positive_count: int,
    negative_count: int,
) -> dict[str, int | float]:
    """由累计计数构造一个曲线点的完整混淆矩阵."""

    fn = positive_count - tp
    tn = negative_count - fp
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tp / positive_count,
        "fpr": fp / negative_count,
        "fnr": fn / positive_count,
        "sample_count": positive_count + negative_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
    }


def _complete_threshold_sweep(
    observations: Iterable[Mapping[str, Any]],
) -> tuple[tuple[float, dict[str, int | float]], ...]:
    """以 O(n log n) 构造两端点与全部唯一观测阈值的混淆矩阵."""

    ordered = sorted(
        ((float(row["score"]), int(row["binary_label"])) for row in observations),
        reverse=True,
    )
    positive_count = sum(label == 1 for _score, label in ordered)
    negative_count = len(ordered) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("每个检测曲线 scope 必须同时包含阳性与阴性记录")
    points: list[tuple[float, dict[str, int | float]]] = [
        (
            math.inf,
            _confusion_from_counts(
                tp=0,
                fp=0,
                positive_count=positive_count,
                negative_count=negative_count,
            ),
        )
    ]
    tp = 0
    fp = 0
    row_index = 0
    while row_index < len(ordered):
        score = ordered[row_index][0]
        while row_index < len(ordered) and ordered[row_index][0] == score:
            if ordered[row_index][1] == 1:
                tp += 1
            else:
                fp += 1
            row_index += 1
        points.append(
            (
                score,
                _confusion_from_counts(
                    tp=tp,
                    fp=fp,
                    positive_count=positive_count,
                    negative_count=negative_count,
                ),
            )
        )
    points.append(
        (
            -math.inf,
            _confusion_from_counts(
                tp=tp,
                fp=fp,
                positive_count=positive_count,
                negative_count=negative_count,
            ),
        )
    )
    return tuple(points)


def build_detection_score_tables(
    records: Iterable[Mapping[str, Any]],
    frozen_protocol: Mapping[str, Any],
) -> dict[str, tuple[dict[str, Any], ...]]:
    """从 test split 正式记录构造分数分布,ROC 和 DET 原始表.

    该函数属于通用实验写法: 输入连续分数与真实标签, 对总体 test scope 和每个
    同时具有正负样本的攻击条件分别枚举全部唯一分数.它不接收聚合 operating
    point, 因而无法由单点指标伪造曲线.
    """

    threshold = _finite_float(frozen_protocol.get("content_threshold"), "content_threshold")
    rescue_margin_low = _finite_float(frozen_protocol.get("rescue_margin_low"), "rescue_margin_low")
    geometry_score_threshold = _finite_float(
        frozen_protocol.get("geometry_score_threshold"),
        "geometry_score_threshold",
    )
    registration_confidence_threshold = _finite_float(
        frozen_protocol.get("registration_confidence_threshold"),
        "registration_confidence_threshold",
    )
    attention_sync_score_threshold = _finite_float(
        frozen_protocol.get("attention_sync_score_threshold"),
        "attention_sync_score_threshold",
    )
    threshold_digest = str(frozen_protocol.get("threshold_digest", ""))
    if not threshold_digest:
        raise ValueError("frozen protocol 必须提供 threshold_digest")

    observations: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        if record.get("split") != "test":
            continue
        label = _binary_label(record.get("sample_role"))
        score = decision_equivalent_score(
            record,
            rescue_margin_low=rescue_margin_low,
            geometry_score_threshold=geometry_score_threshold,
            registration_confidence_threshold=(
                registration_confidence_threshold
            ),
            attention_sync_score_threshold=(
                attention_sync_score_threshold
            ),
        )
        formal_decision = record.get("formal_evidence_positive")
        if formal_decision is not None and _strict_boolean(
            formal_decision,
            "formal_evidence_positive",
        ) != (score >= threshold):
            raise ValueError("连续检测分数与冻结协议正式判定不一致")
        aligned_value = record.get("aligned_content_score")
        observation_id = str(record.get("detector_digest") or record.get("record_id") or "")
        if not observation_id:
            observation_id = (
                f"{record.get('run_id', '')}:{record.get('prompt_id', '')}:"
                f"{record.get('sample_role', '')}:{record.get('attack_id', '')}:{record_index}"
            )
        observations.append(
            {
                "observation_id": observation_id,
                "prompt_id": str(record.get("prompt_id", "")),
                "sample_role": str(record.get("sample_role", "")),
                "binary_label": label,
                "score": score,
                "raw_content_score": _finite_float(record.get("content_score"), "content_score"),
                "aligned_content_score": (
                    "" if aligned_value is None else _finite_float(aligned_value, "aligned_content_score")
                ),
                "condition_key": _scope_key(record),
            }
        )
    if not observations:
        raise ValueError("正式检测曲线缺少 test split 记录")

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped[("test_overall", "all", "all", "all")].extend(observations)
    for observation in observations:
        grouped[observation["condition_key"]].append(observation)

    valid_groups = {
        key: group
        for key, group in grouped.items()
        if {int(row["binary_label"]) for row in group} == {0, 1}
    }
    if ("test_overall", "all", "all", "all") not in valid_groups:
        raise ValueError("test overall scope 必须同时包含阳性与阴性记录")

    distribution_rows: list[dict[str, Any]] = []
    roc_rows: list[dict[str, Any]] = []
    det_rows: list[dict[str, Any]] = []
    for scope_key, group in sorted(valid_groups.items()):
        sample_scope, attack_family, attack_name, resource_profile = scope_key
        operating_point = _confusion(group, threshold)
        for observation in sorted(
            group,
            key=lambda row: (
                -float(row["score"]),
                int(row["binary_label"]),
                str(row["observation_id"]),
            ),
        ):
            distribution_rows.append(
                {
                    "sample_scope": sample_scope,
                    "attack_family": attack_family,
                    "attack_name": attack_name,
                    "resource_profile": resource_profile,
                    **{name: observation[name] for name in (
                        "observation_id",
                        "prompt_id",
                        "sample_role",
                        "binary_label",
                        "score",
                        "raw_content_score",
                        "aligned_content_score",
                    )},
                    "threshold": threshold,
                    **operating_point,
                    "threshold_digest": threshold_digest,
                    "metric_status": "measured_continuous_image_only_detection_score",
                }
            )
        threshold_sweep = _complete_threshold_sweep(group)
        for point_index, (point_threshold, confusion) in enumerate(threshold_sweep):
            if point_index == 0:
                threshold_kind = "positive_infinity_endpoint"
            elif point_index == len(threshold_sweep) - 1:
                threshold_kind = "negative_infinity_endpoint"
            else:
                threshold_kind = "observed_score"
            point = {
                "sample_scope": sample_scope,
                "attack_family": attack_family,
                "attack_name": attack_name,
                "resource_profile": resource_profile,
                "curve_point_index": point_index,
                "threshold_kind": threshold_kind,
                "threshold": point_threshold,
                **confusion,
                "threshold_digest": threshold_digest,
                "metric_status": "measured_complete_threshold_sweep",
            }
            roc_rows.append(point)
            det_rows.append(dict(point))

    return {
        "score_distribution_table": tuple(distribution_rows),
        "roc_curve_points": tuple(roc_rows),
        "det_curve_points": tuple(det_rows),
    }


def write_detection_score_tables(
    output_dir: Path,
    tables: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, Path]:
    """以固定列顺序写出三张可重建论文图数据表."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "score_distribution_table": output_dir / "score_distribution_table.csv",
        "roc_curve_points": output_dir / "roc_curve_points.csv",
        "det_curve_points": output_dir / "det_curve_points.csv",
    }
    for table_name, path in paths.items():
        rows = tuple(dict(row) for row in tables[table_name])
        if not rows:
            raise ValueError(f"{table_name} 不得为空")
        fieldnames = (
            SCORE_DISTRIBUTION_FIELDNAMES
            if table_name == "score_distribution_table"
            else CURVE_POINT_FIELDNAMES
        )
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return paths
