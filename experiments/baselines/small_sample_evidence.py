"""主表 external baseline 小样本证据边界。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from main.core.digest import build_stable_digest

PRIMARY_BASELINE_IDS = ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
SMALL_SAMPLE_BOUNDARY = "small_sample_only"
PAPER_CLAIM_BOUNDARY = "not_full_paper_claim"
SMALL_SAMPLE_OPERATING_POINT = "fixed_fpr_0.05"
EXCLUDED_OPERATING_POINTS = ("tpr_at_fpr_0_01", "tpr_at_fpr_0_001")


@dataclass(frozen=True)
class SmallSampleEvidenceRecord:
    """记录单个 baseline 候选结果在小样本证据边界下的可审计状态。

    该对象属于通用工程写法: 它把候选记录、schema 校验问题和论文 claim 边界集中到一个
    records 层对象中, 下游表格可以复用该对象, 但不能把它解释为正式 full paper 指标。
    """

    small_sample_evidence_id: str
    small_sample_evidence_digest: str
    baseline_id: str
    baseline_result_record_id: str
    baseline_result_digest: str
    resource_profile: str
    comparable_operating_point: str
    attack_family: str
    attack_name: str
    metric_status: str
    small_sample_boundary: str
    paper_claim_boundary: str
    positive_count: int
    negative_count: int
    supported_record_count: int
    attack_record_count: int
    true_positive_rate: float
    false_positive_rate: float
    clean_false_positive_rate: float
    attacked_false_positive_rate: float
    quality_score_proxy_mean: float
    score_retention_mean: float
    evidence_path_count: int
    small_sample_evidence_ready: bool
    small_sample_fixed_fpr_boundary_ready: bool
    small_sample_attack_detection_ready: bool
    small_sample_common_protocol_ready: bool
    formal_import_ready: bool
    formal_import_blocking_reasons: tuple[str, ...]
    excluded_operating_points: tuple[str, ...]
    supports_paper_claim: bool

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)


def _str_field(row: Mapping[str, Any], field_name: str) -> str:
    """读取字符串字段, 缺失时返回空字符串。"""

    return str(row.get(field_name, "") or "")


def _int_field(row: Mapping[str, Any], field_name: str) -> int:
    """读取计数字段, 缺失时返回 0。"""

    return int(float(row.get(field_name, 0) or 0))


def _bool_field(row: Mapping[str, Any], field_name: str) -> bool:
    """读取布尔字段, 兼容 JSON 布尔值与文本布尔值。"""

    value = row.get(field_name)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _float_field(row: Mapping[str, Any], field_name: str) -> float:
    """读取指标字段, 缺失时返回 0.0 以保持小样本表格可重建。"""

    return float(row.get(field_name, 0.0) or 0.0)


def _list_field(row: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    """读取路径或原因列表字段。"""

    value = row.get(field_name, ())
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value if str(item).strip())
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(";") if part.strip())
    return ()


def _issues_by_row(validation_report: Mapping[str, Any]) -> dict[int, tuple[str, ...]]:
    """按候选记录序号聚合 schema validator 拒绝原因。"""

    grouped: dict[int, list[str]] = {}
    for issue in validation_report.get("issues", []):
        if not isinstance(issue, Mapping):
            continue
        row_index = int(issue.get("row_index", -1))
        reason = str(issue.get("reason", "") or "")
        if row_index >= 0 and reason:
            grouped.setdefault(row_index, []).append(reason)
    return {row_index: tuple(sorted(set(reasons))) for row_index, reasons in grouped.items()}


def build_primary_baseline_small_sample_evidence_records(
    candidate_rows: Iterable[Mapping[str, Any]],
    validation_report: Mapping[str, Any],
) -> tuple[SmallSampleEvidenceRecord, ...]:
    """把正式导入候选记录转换为小样本证据记录。

    该函数属于项目特定写法: 它显式允许小样本证据进入工程审计, 同时强制保留
    `supports_paper_claim=false`, 避免当前结果被误用为 TPR@FPR=0.01 或 TPR@FPR=0.001 的正式结论。
    """

    grouped_issues = _issues_by_row(validation_report)
    records: list[SmallSampleEvidenceRecord] = []
    for row_index, row in enumerate(candidate_rows):
        baseline_id = _str_field(row, "baseline_id")
        if baseline_id not in PRIMARY_BASELINE_IDS:
            continue
        evidence_paths = _list_field(row, "evidence_paths")
        positive_count = _int_field(row, "positive_count")
        negative_count = _int_field(row, "negative_count")
        supported_count = _int_field(row, "supported_record_count")
        attack_count = _int_field(row, "attack_record_count")
        blocking_reasons = grouped_issues.get(row_index, ())
        comparable_operating_point = _str_field(row, "comparable_operating_point")
        attack_family = _str_field(row, "attack_family")
        attack_name = _str_field(row, "attack_name")
        fixed_fpr_boundary_ready = comparable_operating_point == SMALL_SAMPLE_OPERATING_POINT
        attack_detection_ready = bool(attack_family) and bool(attack_name) and attack_count >= supported_count > 0
        evidence_ready = (
            bool(evidence_paths)
            and _bool_field(row, "formal_evidence_paths_ready")
            and positive_count > 0
            and negative_count > 0
            and supported_count > 0
        )
        common_protocol_ready = evidence_ready and fixed_fpr_boundary_ready and attack_detection_ready
        payload = {
            "baseline_id": baseline_id,
            "baseline_result_record_id": _str_field(row, "baseline_result_record_id"),
            "baseline_result_digest": _str_field(row, "baseline_result_digest"),
            "resource_profile": _str_field(row, "resource_profile"),
            "comparable_operating_point": comparable_operating_point,
            "attack_family": attack_family,
            "attack_name": attack_name,
            "metric_status": _str_field(row, "metric_status") or "not_recorded",
            "small_sample_boundary": SMALL_SAMPLE_BOUNDARY,
            "paper_claim_boundary": PAPER_CLAIM_BOUNDARY,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "supported_record_count": supported_count,
            "attack_record_count": attack_count,
            "true_positive_rate": _float_field(row, "true_positive_rate"),
            "false_positive_rate": _float_field(row, "false_positive_rate"),
            "clean_false_positive_rate": _float_field(row, "clean_false_positive_rate"),
            "attacked_false_positive_rate": _float_field(row, "attacked_false_positive_rate"),
            "quality_score_proxy_mean": _float_field(row, "quality_score_proxy_mean"),
            "score_retention_mean": _float_field(row, "score_retention_mean"),
            "evidence_path_count": len(evidence_paths),
            "small_sample_evidence_ready": evidence_ready,
            "small_sample_fixed_fpr_boundary_ready": fixed_fpr_boundary_ready,
            "small_sample_attack_detection_ready": attack_detection_ready,
            "small_sample_common_protocol_ready": common_protocol_ready,
            "formal_import_ready": not blocking_reasons,
            "formal_import_blocking_reasons": blocking_reasons,
            "excluded_operating_points": EXCLUDED_OPERATING_POINTS,
            "supports_paper_claim": False,
        }
        digest = build_stable_digest(payload)
        records.append(
            SmallSampleEvidenceRecord(
                small_sample_evidence_id=f"primary_baseline_small_sample_evidence_{digest[:16]}",
                small_sample_evidence_digest=digest,
                baseline_id=payload["baseline_id"],
                baseline_result_record_id=payload["baseline_result_record_id"],
                baseline_result_digest=payload["baseline_result_digest"],
                resource_profile=payload["resource_profile"],
                comparable_operating_point=payload["comparable_operating_point"],
                attack_family=payload["attack_family"],
                attack_name=payload["attack_name"],
                metric_status=payload["metric_status"],
                small_sample_boundary=payload["small_sample_boundary"],
                paper_claim_boundary=payload["paper_claim_boundary"],
                positive_count=payload["positive_count"],
                negative_count=payload["negative_count"],
                supported_record_count=payload["supported_record_count"],
                attack_record_count=payload["attack_record_count"],
                true_positive_rate=payload["true_positive_rate"],
                false_positive_rate=payload["false_positive_rate"],
                clean_false_positive_rate=payload["clean_false_positive_rate"],
                attacked_false_positive_rate=payload["attacked_false_positive_rate"],
                quality_score_proxy_mean=payload["quality_score_proxy_mean"],
                score_retention_mean=payload["score_retention_mean"],
                evidence_path_count=payload["evidence_path_count"],
                small_sample_evidence_ready=payload["small_sample_evidence_ready"],
                small_sample_fixed_fpr_boundary_ready=payload["small_sample_fixed_fpr_boundary_ready"],
                small_sample_attack_detection_ready=payload["small_sample_attack_detection_ready"],
                small_sample_common_protocol_ready=payload["small_sample_common_protocol_ready"],
                formal_import_ready=payload["formal_import_ready"],
                formal_import_blocking_reasons=payload["formal_import_blocking_reasons"],
                excluded_operating_points=payload["excluded_operating_points"],
                supports_paper_claim=False,
            )
        )
    return tuple(records)


def build_primary_baseline_small_sample_comparison_rows(
    records: Iterable[SmallSampleEvidenceRecord],
) -> list[dict[str, Any]]:
    """把小样本 evidence records 转换为可读共同协议对比表行。

    该表属于工程审计表: 它允许查看小样本链路下的指标和证据边界, 但每一行都保持
    `supports_paper_claim=false`, 不能替代正式 full paper 统计表。
    """

    rows = []
    for record in sorted(records, key=lambda item: item.baseline_id):
        rows.append(
            {
                "baseline_id": record.baseline_id,
                "comparison_scope": "small_sample_common_protocol",
                "resource_profile": record.resource_profile,
                "comparable_operating_point": record.comparable_operating_point,
                "attack_family": record.attack_family,
                "attack_name": record.attack_name,
                "metric_status": record.metric_status,
                "true_positive_rate": record.true_positive_rate,
                "false_positive_rate": record.false_positive_rate,
                "clean_false_positive_rate": record.clean_false_positive_rate,
                "attacked_false_positive_rate": record.attacked_false_positive_rate,
                "quality_score_proxy_mean": record.quality_score_proxy_mean,
                "score_retention_mean": record.score_retention_mean,
                "positive_count": record.positive_count,
                "negative_count": record.negative_count,
                "supported_record_count": record.supported_record_count,
                "attack_record_count": record.attack_record_count,
                "small_sample_evidence_ready": record.small_sample_evidence_ready,
                "small_sample_common_protocol_ready": record.small_sample_common_protocol_ready,
                "formal_import_ready": record.formal_import_ready,
                "paper_claim_boundary": record.paper_claim_boundary,
                "excluded_operating_points": ";".join(record.excluded_operating_points),
                "supports_paper_claim": False,
            }
        )
    return rows


def build_primary_baseline_small_sample_evidence_summary(
    records: Iterable[SmallSampleEvidenceRecord],
) -> dict[str, Any]:
    """聚合小样本 baseline 证据摘要。"""

    record_values = tuple(records)
    ready_records = tuple(record for record in record_values if record.small_sample_evidence_ready)
    fixed_fpr_ready_records = tuple(record for record in record_values if record.small_sample_fixed_fpr_boundary_ready)
    attack_ready_records = tuple(record for record in record_values if record.small_sample_attack_detection_ready)
    common_protocol_ready_records = tuple(record for record in record_values if record.small_sample_common_protocol_ready)
    covered_ids = tuple(sorted({record.baseline_id for record in ready_records}))
    missing_ids = tuple(baseline_id for baseline_id in PRIMARY_BASELINE_IDS if baseline_id not in covered_ids)
    formal_ready_ids = tuple(sorted({record.baseline_id for record in record_values if record.formal_import_ready}))
    return {
        "construction_unit_name": "primary_baseline_small_sample_evidence",
        "small_sample_evidence_record_count": len(record_values),
        "small_sample_evidence_ready_count": len(ready_records),
        "small_sample_fixed_fpr_boundary_ready_count": len(fixed_fpr_ready_records),
        "small_sample_attack_detection_ready_count": len(attack_ready_records),
        "small_sample_common_protocol_ready_count": len(common_protocol_ready_records),
        "covered_primary_baseline_count": len(covered_ids),
        "covered_primary_baseline_ids": list(covered_ids),
        "missing_primary_baseline_ids": list(missing_ids),
        "formal_import_ready_count": len(formal_ready_ids),
        "formal_import_ready_ids": list(formal_ready_ids),
        "formal_full_paper_run_requested": False,
        "formal_full_paper_run_permitted": False,
        "excluded_operating_points": list(EXCLUDED_OPERATING_POINTS),
        "small_sample_evidence_ready": not missing_ids and len(record_values) > 0,
        "small_sample_common_protocol_ready": len(common_protocol_ready_records) == len(PRIMARY_BASELINE_IDS),
        "paper_claim_boundary": PAPER_CLAIM_BOUNDARY,
        "paper_claim_ready": False,
        "supports_paper_claim": False,
    }
