"""Gaussian Shading 官方参考环境复现的补充表 受治理导入 协议。

该模块服务方法忠实度审计。Gaussian Shading 官方实现面向 Stable Diffusion 1.x/2.x
的 4-channel latent、truncated Gaussian message 与 DDIM inversion, 因此官方参考环境结果
必须与 SD3.5 Medium common-backbone 主表结果分开记录。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Iterable, Mapping

from experiments.runtime.model_sources import get_model_source
from main.core.digest import build_stable_digest

GAUSSIAN_SHADING_OFFICIAL_REFERENCE_PROTOCOL_NAME = "gaussian_shading_official_reference_protocol"
GAUSSIAN_SHADING_SUPPLEMENTAL_TABLE_ROLE = "supplemental_method_fidelity_reference"
REQUIRED_READY_FLAGS = (
    "official_command_succeeded",
    "official_source_ready",
    "source_identity_ready",
    "source_worktree_exact",
    "official_environment_report_ready",
    "official_result_summary_ready",
    "model_source_ready",
    "openclip_source_ready",
    "governed_import_ready",
)
REQUIRED_SOURCE_PROVENANCE_FIELDS = (
    "official_repository_commit",
    "source_worktree_digest",
    "source_patch_sha256",
    "prompt_dataset_repository_id",
    "prompt_dataset_revision",
    "official_model_repository_id",
    "official_model_revision",
    "model_snapshot_content_digest",
    "openclip_source_name",
    "openclip_usage_role",
    "openclip_model_name",
    "openclip_repository_id",
    "openclip_revision",
    "openclip_checkpoint_filename",
    "openclip_checkpoint_sha256",
    "openclip_checkpoint_size_bytes",
    "openclip_snapshot_content_digest",
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

_HEX_DIGEST_PATTERN = re.compile(r"^[0-9a-f]+$")
_OFFICIAL_MODEL_SOURCE = get_model_source("manojb_stable_diffusion_2_1_base")
_PROMPT_DATASET_SOURCE = get_model_source("gustavosta_stable_diffusion_prompts")
_OPENCLIP_SOURCE = get_model_source("laion_clip_vit_g14")
if len(_OPENCLIP_SOURCE.required_files) != 1:
    raise RuntimeError("Gaussian Shading OpenCLIP 登记源必须声明唯一 checkpoint")
_OPENCLIP_REQUIRED_FILE = _OPENCLIP_SOURCE.required_files[0]
EXPECTED_OPENCLIP_PROVENANCE = {
    "openclip_source_name": "laion_clip_vit_g14",
    "openclip_usage_role": "official_reference_openclip_encoder",
    "openclip_model_name": "ViT-g-14",
    "openclip_repository_id": _OPENCLIP_SOURCE.repository_id,
    "openclip_revision": _OPENCLIP_SOURCE.revision,
    "openclip_checkpoint_filename": _OPENCLIP_REQUIRED_FILE.path,
    "openclip_checkpoint_sha256": _OPENCLIP_REQUIRED_FILE.sha256,
    "openclip_checkpoint_size_bytes": _OPENCLIP_REQUIRED_FILE.size_bytes,
}


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
    """记录 Gaussian Shading 官方固定 profile 复现结果的补充表候选记录。"""

    reference_record_id: str
    reference_record_digest: str
    baseline_id: str
    reference_protocol_name: str
    supplemental_table_role: str
    result_source_type: str
    official_command_requested: bool
    official_command_return_code: int
    official_entrypoint: str
    official_repository_commit: str
    source_worktree_digest: str
    source_patch_sha256: str
    prompt_dataset_repository_id: str
    prompt_dataset_revision: str
    official_model_repository_id: str
    official_model_revision: str
    model_snapshot_content_digest: str
    openclip_source_name: str
    openclip_usage_role: str
    openclip_model_name: str
    openclip_repository_id: str
    openclip_revision: str
    openclip_checkpoint_filename: str
    openclip_checkpoint_sha256: str
    openclip_checkpoint_size_bytes: int
    openclip_snapshot_content_digest: str
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
    official_command_succeeded: bool
    official_source_ready: bool
    source_identity_ready: bool
    source_worktree_exact: bool
    official_environment_report_ready: bool
    official_result_summary_ready: bool
    model_source_ready: bool
    openclip_source_ready: bool
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
    """读取必需整数, 不为缺失字段构造默认值."""

    if field_name not in row or row[field_name] is None or row[field_name] == "":
        raise ValueError(f"{field_name} 缺失")
    value = row[field_name]
    if isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是整数")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{field_name} 必须是有限整数")
    return int(numeric)


def _float_field(row: Mapping[str, Any], field_name: str) -> float:
    """读取必需有限浮点指标, 不把缺失值解释为 0."""

    if field_name not in row or row[field_name] is None or row[field_name] == "":
        raise ValueError(f"{field_name} 缺失")
    value = float(row[field_name])
    if not math.isfinite(value):
        raise ValueError(f"{field_name} 必须是有限数值")
    return value


def _exact_digest(value: Any, length: int) -> bool:
    """判断值是否为指定长度的小写十六进制摘要."""

    text = str(value or "")
    return len(text) == length and _HEX_DIGEST_PATTERN.fullmatch(text) is not None


def build_gaussian_shading_official_reference_schema() -> dict[str, Any]:
    """构造 Gaussian Shading 官方参考导入 schema 描述。"""

    return {
        "reference_protocol_name": GAUSSIAN_SHADING_OFFICIAL_REFERENCE_PROTOCOL_NAME,
        "baseline_id": "gaussian_shading",
        "supplemental_table_role": GAUSSIAN_SHADING_SUPPLEMENTAL_TABLE_ROLE,
        "required_ready_flags": list(REQUIRED_READY_FLAGS),
        "required_source_provenance_fields": list(REQUIRED_SOURCE_PROVENANCE_FIELDS),
        "required_metric_fields": list(REQUIRED_METRIC_FIELDS),
        "expected_official_model_repository_id": _OFFICIAL_MODEL_SOURCE.repository_id,
        "expected_official_model_revision": _OFFICIAL_MODEL_SOURCE.revision,
        "expected_prompt_dataset_repository_id": _PROMPT_DATASET_SOURCE.repository_id,
        "expected_prompt_dataset_revision": _PROMPT_DATASET_SOURCE.revision,
        "expected_openclip_provenance": dict(EXPECTED_OPENCLIP_PROVENANCE),
        "main_table_eligible": False,
        "supports_paper_claim": False,
    }


def build_gaussian_shading_official_reference_record(
    *,
    official_command_requested: bool,
    official_command_return_code: int,
    official_entrypoint: str,
    official_repository_commit: str,
    official_environment_profile: str,
    baseline_result_source: str,
    baseline_result_source_digest: str,
    evidence_paths: Iterable[str],
    source_provenance: Mapping[str, Any],
    metric_values: Mapping[str, Any],
    ready_flags: Mapping[str, bool],
) -> dict[str, Any]:
    """构造 Gaussian Shading 官方固定 profile 复现的补充表 受治理导入 记录。

    该记录只表达补充表方法忠实度参考。由于 Stable Diffusion 2.1 backbone 与 SD3.5 主线不同,
    该记录不得进入主表正式对比。
    """

    payload = {
        "baseline_id": "gaussian_shading",
        "reference_protocol_name": GAUSSIAN_SHADING_OFFICIAL_REFERENCE_PROTOCOL_NAME,
        "supplemental_table_role": GAUSSIAN_SHADING_SUPPLEMENTAL_TABLE_ROLE,
        "result_source_type": "official_reproduction",
        "official_command_requested": bool(official_command_requested),
        "official_command_return_code": int(official_command_return_code),
        "official_entrypoint": official_entrypoint,
        "official_repository_commit": official_repository_commit,
        "source_worktree_digest": _str_field(source_provenance, "source_worktree_digest"),
        "source_patch_sha256": _str_field(source_provenance, "source_patch_sha256"),
        "prompt_dataset_repository_id": _str_field(source_provenance, "prompt_dataset_repository_id"),
        "prompt_dataset_revision": _str_field(source_provenance, "prompt_dataset_revision"),
        "official_model_repository_id": _str_field(source_provenance, "official_model_repository_id"),
        "official_model_revision": _str_field(source_provenance, "official_model_revision"),
        "model_snapshot_content_digest": _str_field(source_provenance, "model_snapshot_content_digest"),
        "openclip_source_name": _str_field(source_provenance, "openclip_source_name"),
        "openclip_usage_role": _str_field(source_provenance, "openclip_usage_role"),
        "openclip_model_name": _str_field(source_provenance, "openclip_model_name"),
        "openclip_repository_id": _str_field(source_provenance, "openclip_repository_id"),
        "openclip_revision": _str_field(source_provenance, "openclip_revision"),
        "openclip_checkpoint_filename": _str_field(source_provenance, "openclip_checkpoint_filename"),
        "openclip_checkpoint_sha256": _str_field(source_provenance, "openclip_checkpoint_sha256"),
        "openclip_checkpoint_size_bytes": _int_field(source_provenance, "openclip_checkpoint_size_bytes"),
        "openclip_snapshot_content_digest": _str_field(source_provenance, "openclip_snapshot_content_digest"),
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
        "official_command_succeeded": bool(ready_flags.get("official_command_succeeded", False)),
        "official_source_ready": bool(ready_flags.get("official_source_ready", False)),
        "source_identity_ready": bool(ready_flags.get("source_identity_ready", False)),
        "source_worktree_exact": bool(ready_flags.get("source_worktree_exact", False)),
        "official_environment_report_ready": bool(ready_flags.get("official_environment_report_ready", False)),
        "official_result_summary_ready": bool(ready_flags.get("official_result_summary_ready", False)),
        "model_source_ready": bool(ready_flags.get("model_source_ready", False)),
        "openclip_source_ready": bool(ready_flags.get("openclip_source_ready", False)),
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
    """校验 Gaussian Shading 官方固定 profile 参考记录是否满足补充表 受治理导入 协议。"""

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
        if not _bool_field(row, "official_command_requested"):
            row_issues.append(
                GaussianShadingOfficialReferenceIssue(
                    row_index, "official_command_requested", "current_official_command_required"
                )
            )
        try:
            official_return_code = _int_field(row, "official_command_return_code")
        except (TypeError, ValueError):
            official_return_code = -1
            row_issues.append(
                GaussianShadingOfficialReferenceIssue(
                    row_index, "official_command_return_code", "official_command_return_code_required"
                )
            )
        if official_return_code != 0:
            row_issues.append(
                GaussianShadingOfficialReferenceIssue(
                    row_index, "official_command_return_code", "current_official_command_must_succeed"
                )
            )
        if _bool_field(row, "main_table_eligible"):
            row_issues.append(
                GaussianShadingOfficialReferenceIssue(
                row_index, "main_table_eligible", "official_reference_must_not_enter_main_table"
                )
            )
        for flag_name in REQUIRED_READY_FLAGS:
            if not _bool_field(row, flag_name):
                row_issues.append(
                    GaussianShadingOfficialReferenceIssue(row_index, flag_name, f"{flag_name}_required")
                )
        for field_name in REQUIRED_SOURCE_PROVENANCE_FIELDS:
            if not _str_field(row, field_name):
                row_issues.append(
                    GaussianShadingOfficialReferenceIssue(
                        row_index,
                        field_name,
                        f"{field_name}_required",
                    )
                )
        for field_name, expected_length in (
            ("official_repository_commit", 40),
            ("source_worktree_digest", 64),
            ("source_patch_sha256", 64),
            ("prompt_dataset_revision", 40),
            ("official_model_revision", 40),
            ("model_snapshot_content_digest", 64),
            ("openclip_revision", 40),
            ("openclip_checkpoint_sha256", 64),
            ("openclip_snapshot_content_digest", 64),
        ):
            if not _exact_digest(row.get(field_name), expected_length):
                row_issues.append(
                    GaussianShadingOfficialReferenceIssue(
                        row_index,
                        field_name,
                        f"{field_name}_exact_digest_required",
                    )
                )
        if not _exact_digest(row.get("baseline_result_source_digest"), 64):
            row_issues.append(
                GaussianShadingOfficialReferenceIssue(row_index, "baseline_result_source_digest", "result_source_digest_required")
            )
        if not row.get("evidence_paths"):
            row_issues.append(GaussianShadingOfficialReferenceIssue(row_index, "evidence_paths", "evidence_paths_required"))
        expected_source_values = {
            "prompt_dataset_repository_id": _PROMPT_DATASET_SOURCE.repository_id,
            "prompt_dataset_revision": _PROMPT_DATASET_SOURCE.revision,
            "official_model_repository_id": _OFFICIAL_MODEL_SOURCE.repository_id,
            "official_model_revision": _OFFICIAL_MODEL_SOURCE.revision,
            **EXPECTED_OPENCLIP_PROVENANCE,
        }
        for field_name, expected_value in expected_source_values.items():
            actual_value = row.get(field_name)
            if actual_value != expected_value:
                row_issues.append(
                    GaussianShadingOfficialReferenceIssue(
                        row_index, field_name, f"{field_name}_registered_value_required"
                    )
                )
        try:
            checkpoint_size = _int_field(row, "openclip_checkpoint_size_bytes")
        except (TypeError, ValueError):
            checkpoint_size = -1
        if checkpoint_size != _OPENCLIP_REQUIRED_FILE.size_bytes:
            row_issues.append(
                GaussianShadingOfficialReferenceIssue(
                    row_index,
                    "openclip_checkpoint_size_bytes",
                    "openclip_checkpoint_size_bytes_registered_value_required",
                )
            )
        metric_values: dict[str, float] = {}
        for field_name in REQUIRED_METRIC_FIELDS:
            try:
                value = _float_field(row, field_name)
            except (TypeError, ValueError):
                row_issues.append(
                    GaussianShadingOfficialReferenceIssue(
                        row_index, field_name, f"{field_name}_measured_value_required"
                    )
                )
                continue
            metric_values[field_name] = value
            if field_name.endswith("count") and value <= 0:
                row_issues.append(
                    GaussianShadingOfficialReferenceIssue(row_index, field_name, "positive_count_value_required")
                )
            if field_name in {
                "detection_true_positive_rate",
                "traceability_true_positive_rate",
                "mean_bit_accuracy",
                "std_bit_accuracy",
                "std_clip_score",
            } and not 0.0 <= value <= 1.0:
                row_issues.append(
                    GaussianShadingOfficialReferenceIssue(row_index, field_name, "metric_rate_must_be_in_unit_interval")
                )
            if field_name == "mean_clip_score" and not -1.0 <= value <= 1.0:
                row_issues.append(
                    GaussianShadingOfficialReferenceIssue(
                        row_index,
                        field_name,
                        "clip_cosine_similarity_must_be_in_signed_unit_interval",
                    )
                )
        if (
            "sample_count" in metric_values
            and "positive_count" in metric_values
            and metric_values["sample_count"] != metric_values["positive_count"]
        ):
            row_issues.append(
                GaussianShadingOfficialReferenceIssue(
                    row_index, "positive_count", "positive_count_must_equal_official_sample_count"
                )
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
