"""Gaussian Shading 官方原始环境复现的补充表 governed import 协议。

该模块服务方法忠实度审计。Gaussian Shading 官方实现面向 Stable Diffusion 1.x/2.x
的 4-channel latent、truncated Gaussian message 与 DDIM inversion, 因此官方原始环境结果
必须与 SD3.5 Medium common-backbone 主表结果分开记录。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from main.core.digest import build_stable_digest

GAUSSIAN_SHADING_OFFICIAL_REFERENCE_PROTOCOL_NAME = "gaussian_shading_official_legacy_reference_protocol"
GAUSSIAN_SHADING_SUPPLEMENTAL_TABLE_ROLE = "supplemental_method_fidelity_reference"
REQUIRED_READY_FLAGS = (
    "official_source_ready",
    "official_environment_report_ready",
    "official_result_summary_ready",
    "governed_import_ready",
)
REQUIRED_METRIC_FIELDS = (
    "sample_count",
    "positive_count",
    "detection_true_positive_rate",
    "traceability_true_positive_rate",
    "mean_bit_accuracy",
    "std_bit_accuracy",
    "mean_clip_score",
    "std_clip_score",
)


@dataclass(frozen=True)
class GaussianShadingOfficialReferenceIssue:
    """记录 Gaussian Shading 官方参考导入中的单个 schema 问题。"""

    row_index: int
    field_name: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)


@dataclass(frozen=True)
class GaussianShadingOfficialReferenceRecord:
    """记录 Gaussian Shading 官方 legacy 复现结果的补充表候选记录。"""

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
    detection_true_positive_rate: float
    traceability_true_positive_rate: float
    mean_bit_accuracy: float
    std_bit_accuracy: float
    mean_clip_score: float
    std_clip_score: float
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


def build_gaussian_shading_official_reference_schema() -> dict[str, Any]:
    """构造 Gaussian Shading 官方参考导入 schema 描述。"""

    return {
        "reference_protocol_name": GAUSSIAN_SHADING_OFFICIAL_REFERENCE_PROTOCOL_NAME,
        "baseline_id": "gaussian_shading",
        "supplemental_table_role": GAUSSIAN_SHADING_SUPPLEMENTAL_TABLE_ROLE,
        "required_ready_flags": list(REQUIRED_READY_FLAGS),
        "required_metric_fields": list(REQUIRED_METRIC_FIELDS),
        "main_table_eligible": False,
        "supports_paper_claim": False,
    }


def build_gaussian_shading_official_reference_record(
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
    """构造 Gaussian Shading 官方 legacy 复现的补充表 governed import 记录。

    该记录只表达补充表方法忠实度参考。由于 legacy backbone 与 SD3.5 主线不同,
    该记录不得进入主表正式对比。
    """

    payload = {
        "baseline_id": "gaussian_shading",
        "reference_protocol_name": GAUSSIAN_SHADING_OFFICIAL_REFERENCE_PROTOCOL_NAME,
        "supplemental_table_role": GAUSSIAN_SHADING_SUPPLEMENTAL_TABLE_ROLE,
        "result_source_type": "official_reproduction",
        "official_entrypoint": official_entrypoint,
        "official_repository_commit": official_repository_commit,
        "official_environment_profile": official_environment_profile,
        "baseline_result_source": baseline_result_source,
        "baseline_result_source_digest": baseline_result_source_digest,
        "evidence_paths": tuple(evidence_paths),
        "sample_count": _int_field(metric_values, "sample_count"),
        "positive_count": _int_field(metric_values, "positive_count"),
        "detection_true_positive_rate": _float_field(metric_values, "detection_true_positive_rate"),
        "traceability_true_positive_rate": _float_field(metric_values, "traceability_true_positive_rate"),
        "mean_bit_accuracy": _float_field(metric_values, "mean_bit_accuracy"),
        "std_bit_accuracy": _float_field(metric_values, "std_bit_accuracy"),
        "mean_clip_score": _float_field(metric_values, "mean_clip_score"),
        "std_clip_score": _float_field(metric_values, "std_clip_score"),
        "official_source_ready": bool(ready_flags.get("official_source_ready", False)),
        "official_environment_report_ready": bool(ready_flags.get("official_environment_report_ready", False)),
        "official_result_summary_ready": bool(ready_flags.get("official_result_summary_ready", False)),
        "governed_import_ready": bool(ready_flags.get("governed_import_ready", False)),
        "main_table_eligible": False,
        "supports_paper_claim": False,
    }
    digest = build_stable_digest(payload)
    record = GaussianShadingOfficialReferenceRecord(
        reference_record_id=f"gaussian_shading_official_reference_{digest[:16]}",
        reference_record_digest=digest,
        **payload,
    )
    return record.to_dict()


def validate_gaussian_shading_official_reference_records(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """校验 Gaussian Shading 官方 legacy 参考记录是否满足补充表 governed import 协议。"""

    materialized_rows = [dict(row) for row in rows]
    issues: list[GaussianShadingOfficialReferenceIssue] = []
    accepted: list[dict[str, Any]] = []
    for row_index, row in enumerate(materialized_rows):
        row_issues: list[GaussianShadingOfficialReferenceIssue] = []
        if _str_field(row, "baseline_id") != "gaussian_shading":
            row_issues.append(
                GaussianShadingOfficialReferenceIssue(row_index, "baseline_id", "gaussian_shading_baseline_required")
            )
        if _str_field(row, "reference_protocol_name") != GAUSSIAN_SHADING_OFFICIAL_REFERENCE_PROTOCOL_NAME:
            row_issues.append(
                GaussianShadingOfficialReferenceIssue(
                    row_index, "reference_protocol_name", "official_reference_protocol_required"
                )
            )
        if _str_field(row, "supplemental_table_role") != GAUSSIAN_SHADING_SUPPLEMENTAL_TABLE_ROLE:
            row_issues.append(
                GaussianShadingOfficialReferenceIssue(
                    row_index, "supplemental_table_role", "supplemental_table_role_required"
                )
            )
        if _bool_field(row, "main_table_eligible"):
            row_issues.append(
                GaussianShadingOfficialReferenceIssue(
                    row_index, "main_table_eligible", "legacy_reference_must_not_enter_main_table"
                )
            )
        for flag_name in REQUIRED_READY_FLAGS:
            if not _bool_field(row, flag_name):
                row_issues.append(
                    GaussianShadingOfficialReferenceIssue(row_index, flag_name, f"{flag_name}_required")
                )
        if not _str_field(row, "baseline_result_source_digest"):
            row_issues.append(
                GaussianShadingOfficialReferenceIssue(row_index, "baseline_result_source_digest", "result_source_digest_required")
            )
        if not row.get("evidence_paths"):
            row_issues.append(GaussianShadingOfficialReferenceIssue(row_index, "evidence_paths", "evidence_paths_required"))
        for field_name in REQUIRED_METRIC_FIELDS:
            value = _float_field(row, field_name)
            if field_name.endswith("count") and value <= 0:
                row_issues.append(
                    GaussianShadingOfficialReferenceIssue(row_index, field_name, "positive_count_value_required")
                )
            if field_name in {
                "detection_true_positive_rate",
                "traceability_true_positive_rate",
                "mean_bit_accuracy",
                "std_bit_accuracy",
                "mean_clip_score",
                "std_clip_score",
            } and not 0.0 <= value <= 1.0:
                row_issues.append(
                    GaussianShadingOfficialReferenceIssue(row_index, field_name, "metric_rate_must_be_in_unit_interval")
                )
        if row_issues:
            issues.extend(row_issues)
        else:
            accepted.append(row)
    return {
        "reference_protocol_name": GAUSSIAN_SHADING_OFFICIAL_REFERENCE_PROTOCOL_NAME,
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
