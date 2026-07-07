"""Tree-Ring 官方原始环境复现的补充表 governed import 协议。

该模块服务补充表忠实度审计, 不参与主表 SD3.5 common-backbone 对比。官方原始环境通常使用
legacy Stable Diffusion、旧版 diffusers 和 DDIM inversion, 因此其结果必须和 SD3.5 方法忠实
adapter 主表结果分开记录。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from main.core.digest import build_stable_digest

TREE_RING_OFFICIAL_REFERENCE_PROTOCOL_NAME = "tree_ring_official_legacy_reference_protocol"
TREE_RING_SUPPLEMENTAL_TABLE_ROLE = "supplemental_method_fidelity_reference"
REQUIRED_READY_FLAGS = (
    "official_source_ready",
    "official_environment_report_ready",
    "official_result_summary_ready",
    "governed_import_ready",
)
REQUIRED_METRIC_FIELDS = (
    "sample_count",
    "positive_count",
    "negative_count",
    "auc",
    "accuracy",
    "true_positive_rate_at_one_percent_fpr",
    "clip_score_mean",
    "watermarked_clip_score_mean",
)


@dataclass(frozen=True)
class TreeRingOfficialReferenceIssue:
    """记录 Tree-Ring 官方参考导入中的单个 schema 问题。"""

    row_index: int
    field_name: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)


@dataclass(frozen=True)
class TreeRingOfficialReferenceRecord:
    """记录 Tree-Ring 官方 legacy 复现结果的补充表候选记录。"""

    reference_record_id: str
    reference_record_digest: str
    baseline_id: str
    reference_protocol_name: str
    supplemental_table_role: str
    result_source_type: str
    official_entrypoint: str
    official_repository_commit: str
    official_environment_profile: str
    baseline_result_source: str
    baseline_result_source_digest: str
    evidence_paths: tuple[str, ...]
    sample_count: int
    positive_count: int
    negative_count: int
    auc: float
    accuracy: float
    true_positive_rate_at_one_percent_fpr: float
    clip_score_mean: float
    watermarked_clip_score_mean: float
    official_source_ready: bool
    official_environment_report_ready: bool
    official_result_summary_ready: bool
    governed_import_ready: bool
    main_table_eligible: bool
    supports_paper_claim: bool

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        data = asdict(self)
        data["evidence_paths"] = list(self.evidence_paths)
        return data


def _str_field(row: Mapping[str, Any], field_name: str) -> str:
    """读取字符串字段。"""

    return str(row.get(field_name, "") or "")


def _bool_field(row: Mapping[str, Any], field_name: str) -> bool:
    """读取布尔字段, 兼容 JSON 与文本表示。"""

    value = row.get(field_name)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _int_field(row: Mapping[str, Any], field_name: str) -> int:
    """读取非负整数指标。"""

    return int(float(row.get(field_name, 0) or 0))


def _float_field(row: Mapping[str, Any], field_name: str) -> float:
    """读取有限浮点指标。"""

    value = float(row.get(field_name, 0.0) or 0.0)
    if value != value or value in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} 必须是有限数值")
    return value


def build_tree_ring_official_reference_schema() -> dict[str, Any]:
    """构造 Tree-Ring 官方参考导入 schema 描述。"""

    return {
        "reference_protocol_name": TREE_RING_OFFICIAL_REFERENCE_PROTOCOL_NAME,
        "baseline_id": "tree_ring",
        "supplemental_table_role": TREE_RING_SUPPLEMENTAL_TABLE_ROLE,
        "required_ready_flags": list(REQUIRED_READY_FLAGS),
        "required_metric_fields": list(REQUIRED_METRIC_FIELDS),
        "main_table_eligible": False,
        "supports_paper_claim": False,
    }


def build_tree_ring_official_reference_record(
    *,
    official_entrypoint: str,
    official_repository_commit: str,
    official_environment_profile: str,
    baseline_result_source: str,
    baseline_result_source_digest: str,
    evidence_paths: Iterable[str],
    metric_values: Mapping[str, Any],
    ready_flags: Mapping[str, bool],
) -> dict[str, Any]:
    """构造 Tree-Ring 官方 legacy 复现的补充表 governed import 记录。

    该记录只表达补充表忠实度参考, 因 legacy backbone 与 SD3.5 主线不同, 不允许进入主表正式对比。
    """

    payload = {
        "baseline_id": "tree_ring",
        "reference_protocol_name": TREE_RING_OFFICIAL_REFERENCE_PROTOCOL_NAME,
        "supplemental_table_role": TREE_RING_SUPPLEMENTAL_TABLE_ROLE,
        "result_source_type": "official_reproduction",
        "official_entrypoint": official_entrypoint,
        "official_repository_commit": official_repository_commit,
        "official_environment_profile": official_environment_profile,
        "baseline_result_source": baseline_result_source,
        "baseline_result_source_digest": baseline_result_source_digest,
        "evidence_paths": tuple(evidence_paths),
        "sample_count": _int_field(metric_values, "sample_count"),
        "positive_count": _int_field(metric_values, "positive_count"),
        "negative_count": _int_field(metric_values, "negative_count"),
        "auc": _float_field(metric_values, "auc"),
        "accuracy": _float_field(metric_values, "accuracy"),
        "true_positive_rate_at_one_percent_fpr": _float_field(metric_values, "true_positive_rate_at_one_percent_fpr"),
        "clip_score_mean": _float_field(metric_values, "clip_score_mean"),
        "watermarked_clip_score_mean": _float_field(metric_values, "watermarked_clip_score_mean"),
        "official_source_ready": bool(ready_flags.get("official_source_ready", False)),
        "official_environment_report_ready": bool(ready_flags.get("official_environment_report_ready", False)),
        "official_result_summary_ready": bool(ready_flags.get("official_result_summary_ready", False)),
        "governed_import_ready": bool(ready_flags.get("governed_import_ready", False)),
        "main_table_eligible": False,
        "supports_paper_claim": False,
    }
    digest = build_stable_digest(payload)
    record = TreeRingOfficialReferenceRecord(
        reference_record_id=f"tree_ring_official_reference_{digest[:16]}",
        reference_record_digest=digest,
        **payload,
    )
    return record.to_dict()


def validate_tree_ring_official_reference_records(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """校验 Tree-Ring 官方 legacy 参考记录是否满足补充表 governed import 协议。"""

    materialized_rows = [dict(row) for row in rows]
    issues: list[TreeRingOfficialReferenceIssue] = []
    accepted: list[dict[str, Any]] = []
    for row_index, row in enumerate(materialized_rows):
        row_issues: list[TreeRingOfficialReferenceIssue] = []
        if _str_field(row, "baseline_id") != "tree_ring":
            row_issues.append(TreeRingOfficialReferenceIssue(row_index, "baseline_id", "tree_ring_baseline_required"))
        if _str_field(row, "reference_protocol_name") != TREE_RING_OFFICIAL_REFERENCE_PROTOCOL_NAME:
            row_issues.append(TreeRingOfficialReferenceIssue(row_index, "reference_protocol_name", "official_reference_protocol_required"))
        if _str_field(row, "supplemental_table_role") != TREE_RING_SUPPLEMENTAL_TABLE_ROLE:
            row_issues.append(TreeRingOfficialReferenceIssue(row_index, "supplemental_table_role", "supplemental_table_role_required"))
        if _bool_field(row, "main_table_eligible"):
            row_issues.append(TreeRingOfficialReferenceIssue(row_index, "main_table_eligible", "legacy_reference_must_not_enter_main_table"))
        for flag_name in REQUIRED_READY_FLAGS:
            if not _bool_field(row, flag_name):
                row_issues.append(TreeRingOfficialReferenceIssue(row_index, flag_name, f"{flag_name}_required"))
        if not _str_field(row, "baseline_result_source_digest"):
            row_issues.append(TreeRingOfficialReferenceIssue(row_index, "baseline_result_source_digest", "result_source_digest_required"))
        if not row.get("evidence_paths"):
            row_issues.append(TreeRingOfficialReferenceIssue(row_index, "evidence_paths", "evidence_paths_required"))
        for field_name in REQUIRED_METRIC_FIELDS:
            value = _float_field(row, field_name)
            if field_name.endswith("count") and value <= 0:
                row_issues.append(TreeRingOfficialReferenceIssue(row_index, field_name, "positive_count_value_required"))
            if field_name in {"auc", "accuracy", "true_positive_rate_at_one_percent_fpr", "clip_score_mean", "watermarked_clip_score_mean"}:
                if not 0.0 <= value <= 1.0:
                    row_issues.append(TreeRingOfficialReferenceIssue(row_index, field_name, "metric_rate_must_be_in_unit_interval"))
        if row_issues:
            issues.extend(row_issues)
        else:
            accepted.append(row)
    return {
        "reference_protocol_name": TREE_RING_OFFICIAL_REFERENCE_PROTOCOL_NAME,
        "input_record_count": len(materialized_rows),
        "accepted_reference_record_count": len(accepted),
        "rejected_reference_record_count": len(materialized_rows) - len(accepted),
        "reference_issue_count": len(issues),
        "reference_import_ready": bool(materialized_rows) and not issues,
        "accepted_records": accepted,
        "issues": [issue.to_dict() for issue in issues],
        "main_table_eligible": False,
        "supports_paper_claim": False,
    }
