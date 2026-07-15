"""主表 external baseline 正式结果导入协议与 schema 校验。"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.protocol.attacks import (
    attack_config_digest,
    resolve_formal_attack_config,
)
from experiments.protocol.paper_fixed_fpr import (
    PAPER_ATTACK_RESOURCE_PROFILES,
    prompt_protocol_name_for_run,
)
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.protocol.fixed_fpr_observation_audit import (
    FORMAL_THRESHOLD_SOURCE,
    audit_fixed_fpr_observation_threshold,
)
from experiments.protocol.splits import build_group_split_counts
from experiments.protocol.calibration import binomial_rate_upper_confidence_bound
from main.core.digest import build_stable_digest

PRIMARY_BASELINE_IDS = ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
PRIMARY_BASELINE_FORMAL_PROTOCOL_NAME = "primary_baseline_formal_import_protocol"
FORMAL_OPERATING_POINT_PREFIX = "fixed_fpr"
REJECTED_ADAPTER_BOUNDARIES = (
    "method_faithful_not_full_external_baseline_comparison",
    "sd35_method_faithful_adapter_not_formal_external_baseline_evidence",
)
ALLOWED_ADAPTER_BOUNDARIES = (
    "sd35_medium_native_official_reproduction",
    "method_faithful_sd35_adapter_reproduction",
    "governed_image_level_baseline_import",
)
ALLOWED_RESULT_SOURCE_TYPES = ("official_reproduction", "governed_import")
REQUIRED_READY_FLAGS = (
    "method_faithful_adapter_ready",
    "paper_run_prompt_protocol_ready",
    "fixed_fpr_baseline_calibration_ready",
    "attack_matrix_baseline_detection_ready",
    "formal_evidence_paths_ready",
)
REQUIRED_METRIC_FIELDS = (
    "positive_count",
    "negative_count",
    "attacked_negative_count",
    "attack_record_count",
    "supported_record_count",
    "true_positive_rate",
    "false_positive_rate",
    "clean_false_positive_rate",
    "attacked_false_positive_rate",
    "quality_score_mean",
)
REQUIRED_SOURCE_FIELDS = (
    "attack_id",
    "attack_config_digest",
    "baseline_result_source",
    "baseline_result_source_digest",
    "result_protocol_name",
    "result_source_type",
    "adapter_boundary",
    "evidence_paths",
    "prompt_protocol_name",
    "prompt_protocol_digest",
)
REQUIRED_THRESHOLD_FIELDS = (
    "evaluation_split",
    "target_fpr",
    "calibrated_detection_threshold",
    "threshold_source",
    "calibration_clean_negative_count",
    "test_clean_negative_count",
    "threshold_digest",
    "fixed_fpr_observation_evidence_path",
    "fixed_fpr_observation_evidence_digest",
)
METHOD_FAITHFUL_ADAPTER_BOUNDARY = "method_faithful_sd35_adapter_reproduction"
FORMAL_READINESS_BLOCKING_FLAG_GROUPS = {
    "missing_formal_attack_resource_profile": {"resource_profile"},
    "missing_paper_run_prompt_protocol": {"prompt_protocol_name", "paper_run_prompt_protocol_ready"},
    "missing_fixed_fpr_baseline_calibration": {"comparable_operating_point", "fixed_fpr_baseline_calibration_ready"},
    "missing_attack_matrix_baseline_detection": {"attack_matrix_baseline_detection_ready"},
    "missing_formal_evidence_paths": {"evidence_paths", "formal_evidence_paths_ready"},
}


@dataclass(frozen=True)
class FormalImportIssue:
    """记录正式结果导入校验中的单个问题。

    该对象属于通用 schema validator 写法: 它把错误边界集中在导入层, 避免下游表格构建函数反复维护相同的字段校验逻辑。
    """

    row_index: int
    baseline_id: str
    field_name: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)


@dataclass(frozen=True)
class FormalImportValidationReport:
    """记录一批主表 baseline 正式导入结果的校验摘要。"""

    protocol_name: str
    target_fpr: float
    expected_operating_point: str
    input_record_count: int
    accepted_formal_import_count: int
    rejected_formal_import_count: int
    formal_import_issue_count: int
    formal_import_validation_ready: bool
    accepted_records: tuple[dict[str, Any], ...]
    issues: tuple[FormalImportIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return {
            "protocol_name": self.protocol_name,
            "target_fpr": self.target_fpr,
            "expected_operating_point": self.expected_operating_point,
            "input_record_count": self.input_record_count,
            "accepted_formal_import_count": self.accepted_formal_import_count,
            "rejected_formal_import_count": self.rejected_formal_import_count,
            "formal_import_issue_count": self.formal_import_issue_count,
            "formal_import_validation_ready": self.formal_import_validation_ready,
            "overall_decision": "pass" if self.formal_import_issue_count == 0 else "fail",
            "accepted_records": list(self.accepted_records),
            "issues": [issue.to_dict() for issue in self.issues],
            "supports_paper_claim": False,
        }


def _str_field(row: Mapping[str, Any], field_name: str) -> str:
    """读取字符串字段, 缺失时返回空字符串。"""

    return str(row.get(field_name, "") or "")


def _bool_field(row: Mapping[str, Any], field_name: str) -> bool:
    """读取布尔字段, 兼容 JSON 与 CSV 常见文本表示。"""

    value = row.get(field_name)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _int_field(row: Mapping[str, Any], field_name: str) -> int:
    """读取非负计数字段。"""

    return int(float(row.get(field_name, 0) or 0))


def _float_field(row: Mapping[str, Any], field_name: str) -> float:
    """读取有限浮点指标字段。"""

    value = float(row.get(field_name, 0.0) or 0.0)
    if value != value or value in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} 必须是有限数值")
    return value


def _list_field(row: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    """读取路径列表字段, 兼容列表和分号分隔字符串。"""

    value = row.get(field_name, ())
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value if str(item).strip())
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(";") if part.strip())
    return ()


def build_fixed_fpr_operating_point(target_fpr: float) -> str:
    """把目标 FPR 映射为共同协议中的 operating point 名称。"""

    return f"{FORMAL_OPERATING_POINT_PREFIX}_{float(target_fpr):g}"


def _normalized_search_roots(evidence_root: Path, evidence_search_roots: Iterable[str | Path]) -> tuple[Path, ...]:
    """把外部 evidence 搜索根目录解析为绝对路径集合."""

    roots: list[Path] = []
    for search_root in evidence_search_roots:
        candidate = Path(search_root).expanduser()
        resolved = candidate.resolve() if candidate.is_absolute() else (evidence_root / candidate).resolve()
        if resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


def _candidate_relative_suffix(candidate: Path) -> Path | None:
    """提取可用于外部镜像目录的相对路径后缀.

    该函数属于路径 schema 解析层, 用于把 `outputs/<name>` 形式的本地记录映射到 Google Drive `SLM/<name>` 镜像布局。
    """

    parts = candidate.parts
    if not parts or parts[0] != "outputs" or len(parts) <= 1:
        return None
    return Path(*parts[1:])


def _package_family_dirname(basename: str) -> str:
    """从标准 package 文件名中提取镜像目录名."""

    marker = "_package_"
    if marker not in basename:
        return ""
    return basename.split(marker, maxsplit=1)[0]


@lru_cache(maxsize=256)
def _find_evidence_by_mirror_layout(search_roots: tuple[Path, ...], candidate: Path) -> Path | None:
    """在显式搜索根目录中按受治理镜像布局查找 evidence 文件.

    该查找只在 schema / provenance 层使用。它优先尝试固定镜像布局, 避免对 Google Drive 目录执行大范围递归扫描。
    """

    basename = candidate.name
    if not basename:
        return None
    suffix = _candidate_relative_suffix(candidate)
    package_dirname = _package_family_dirname(basename)
    for search_root in search_roots:
        if not search_root.exists():
            continue
        path_candidates = [search_root / basename]
        if suffix is not None:
            path_candidates.append(search_root / suffix)
        if package_dirname:
            path_candidates.append(search_root / package_dirname / basename)
        for path_candidate in path_candidates:
            if path_candidate.is_file():
                return path_candidate
        for child in sorted(search_root.iterdir()):
            child_candidate = child / basename
            if child.is_dir() and child_candidate.is_file():
                return child_candidate
    return None


def _resolve_evidence_path(
    evidence_root: Path,
    value: str,
    evidence_search_roots: Iterable[str | Path] = (),
) -> Path:
    """把证据路径解析为可检查的本地路径.

    优先使用记录中的原始路径; 若原始路径在本地 outputs 副本中不可解析, 则只在显式传入的搜索根目录中按文件名查找。
    """

    candidate = Path(value)
    direct_path = candidate if candidate.is_absolute() else evidence_root / candidate
    if direct_path.is_file():
        return direct_path
    search_roots = _normalized_search_roots(evidence_root, evidence_search_roots)
    mirror_path = _find_evidence_by_mirror_layout(search_roots, candidate)
    return mirror_path if mirror_path is not None else direct_path


def _issue(row_index: int, row: Mapping[str, Any], field_name: str, reason: str) -> FormalImportIssue:
    """构造统一校验问题对象。"""

    return FormalImportIssue(
        row_index=row_index,
        baseline_id=_str_field(row, "baseline_id"),
        field_name=field_name,
        reason=reason,
    )


def resolve_paper_run_prompt_protocol_name(root: str | Path = ".") -> str:
    """解析当前论文运行层级对应的主表 prompt 协议名称。"""

    return prompt_protocol_name_for_run(build_paper_run_config(root).run_name)


def build_primary_baseline_formal_import_schema(
    target_fpr: float,
    *,
    root: str | Path = ".",
) -> dict[str, Any]:
    """构造主表 baseline 正式结果导入 schema 的可落盘描述。"""

    paper_run = build_paper_run_config(root)
    if not math.isclose(float(target_fpr), paper_run.target_fpr, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("baseline schema target_fpr 必须与当前论文运行层级一致")
    prompt_protocol_name = prompt_protocol_name_for_run(paper_run.run_name)
    return {
        "protocol_name": PRIMARY_BASELINE_FORMAL_PROTOCOL_NAME,
        "primary_baseline_ids": list(PRIMARY_BASELINE_IDS),
        "expected_operating_point": build_fixed_fpr_operating_point(target_fpr),
        "required_ready_flags": list(REQUIRED_READY_FLAGS),
        "required_metric_fields": list(REQUIRED_METRIC_FIELDS),
        "required_source_fields": list(REQUIRED_SOURCE_FIELDS),
        "required_threshold_fields": list(REQUIRED_THRESHOLD_FIELDS),
        "allowed_result_source_types": list(ALLOWED_RESULT_SOURCE_TYPES),
        "allowed_adapter_boundaries": list(ALLOWED_ADAPTER_BOUNDARIES),
        "allowed_resource_profiles": list(PAPER_ATTACK_RESOURCE_PROFILES),
        "rejected_adapter_boundaries": list(REJECTED_ADAPTER_BOUNDARIES),
        "prompt_protocol_name": prompt_protocol_name,
        "paper_claim_scale": paper_run.run_name,
        "minimum_result_positive_count": paper_run.minimum_clean_negative_count,
        "minimum_result_negative_count": paper_run.minimum_clean_negative_count,
        "supports_paper_claim": False,
    }


def build_primary_baseline_formal_evidence_path_summary(
    rows: Iterable[Mapping[str, Any]],
    *,
    evidence_root: str | Path = ".",
    evidence_search_roots: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """汇总正式导入候选记录中的证据路径可解析状态.

    该函数属于 schema / provenance 层边界: 它只说明 evidence paths 在当前工作区或挂载目录下是否可解析,
    不改变正式导入 validator 的接受规则, 也不会把小样本候选记录升级为论文级 baseline 结论。
    """

    evidence_root_path = Path(evidence_root).resolve()
    search_roots = _normalized_search_roots(evidence_root_path, evidence_search_roots)
    materialized_rows = [dict(row) for row in rows]
    reference_count = 0
    existing_count = 0
    direct_count = 0
    search_resolved_count = 0
    missing_count = 0
    missing_baseline_ids: set[str] = set()
    missing_paths: list[str] = []
    resolved_paths: list[str] = []
    for row in materialized_rows:
        baseline_id = _str_field(row, "baseline_id")
        for evidence_path in _list_field(row, "evidence_paths"):
            reference_count += 1
            candidate = Path(evidence_path)
            direct_path = candidate if candidate.is_absolute() else evidence_root_path / candidate
            resolved_path = _resolve_evidence_path(evidence_root_path, evidence_path, search_roots)
            if resolved_path.is_file():
                existing_count += 1
                resolved_paths.append(resolved_path.as_posix())
                if direct_path.is_file():
                    direct_count += 1
                else:
                    search_resolved_count += 1
            else:
                missing_count += 1
                if baseline_id:
                    missing_baseline_ids.add(baseline_id)
                missing_paths.append(evidence_path)
    return {
        "construction_unit_name": "primary_baseline_formal_evidence_path_resolution",
        "candidate_record_count": len(materialized_rows),
        "formal_evidence_path_reference_count": reference_count,
        "existing_formal_evidence_path_count": existing_count,
        "direct_formal_evidence_path_count": direct_count,
        "search_resolved_formal_evidence_path_count": search_resolved_count,
        "missing_formal_evidence_path_count": missing_count,
        "formal_evidence_path_resolution_ready": bool(materialized_rows) and reference_count > 0 and missing_count == 0,
        "evidence_search_roots": [path.as_posix() for path in search_roots],
        "resolved_formal_evidence_paths": sorted(set(resolved_paths)),
        "formal_evidence_path_missing_baseline_ids": sorted(missing_baseline_ids),
        "missing_formal_evidence_paths": sorted(set(missing_paths)),
        "supports_paper_claim": False,
    }


def _validate_metric_fields(
    row: Mapping[str, Any],
    row_index: int,
    *,
    expected_test_count: int,
    target_fpr: float,
) -> list[FormalImportIssue]:
    """校验正式导入记录中的完整 test 计数和率值边界。"""

    issues: list[FormalImportIssue] = []
    positive_count = _int_field(row, "positive_count")
    negative_count = _int_field(row, "negative_count")
    attacked_negative_count = _int_field(row, "attacked_negative_count")
    supported_count = _int_field(row, "supported_record_count")
    attack_count = _int_field(row, "attack_record_count")
    if positive_count != expected_test_count:
        issues.append(_issue(row_index, row, "positive_count", "complete_test_positive_count_required"))
    if negative_count != expected_test_count:
        issues.append(_issue(row_index, row, "negative_count", "complete_test_negative_count_required"))
    if attacked_negative_count != expected_test_count:
        issues.append(
            _issue(
                row_index,
                row,
                "attacked_negative_count",
                "complete_test_attacked_negative_count_required",
            )
        )
    if supported_count != expected_test_count:
        issues.append(
            _issue(row_index, row, "supported_record_count", "complete_test_supported_record_count_required")
        )
    expected_attack_count = 2 * expected_test_count
    if attack_count != expected_attack_count:
        issues.append(_issue(row_index, row, "attack_record_count", "complete_test_attack_record_count_required"))
    for field_name in (
        "true_positive_rate",
        "false_positive_rate",
        "clean_false_positive_rate",
        "attacked_false_positive_rate",
    ):
        value = _float_field(row, field_name)
        if not 0.0 <= value <= 1.0:
            issues.append(_issue(row_index, row, field_name, "metric_rate_must_be_in_unit_interval"))
    quality_value = _float_field(row, "quality_score_mean")
    if not -1.0 <= quality_value <= 1.0:
        issues.append(
            _issue(
                row_index,
                row,
                "quality_score_mean",
                "ssim_quality_must_be_in_signed_unit_interval",
            )
        )
    clean_fpr = _float_field(row, "clean_false_positive_rate")
    clean_false_positive_value = clean_fpr * negative_count
    clean_false_positive_count = round(clean_false_positive_value)
    if not math.isclose(clean_false_positive_value, clean_false_positive_count, rel_tol=0.0, abs_tol=1e-8):
        issues.append(_issue(row_index, row, "clean_false_positive_rate", "clean_fpr_count_inconsistent"))
    elif negative_count > 0:
        upper_bound = binomial_rate_upper_confidence_bound(
            clean_false_positive_count,
            negative_count,
            0.95,
        )
        if clean_fpr > target_fpr:
            issues.append(_issue(row_index, row, "clean_false_positive_rate", "clean_fpr_exceeds_target"))
        if upper_bound > target_fpr:
            issues.append(
                _issue(
                    row_index,
                    row,
                    "clean_false_positive_rate",
                    "clean_fpr_confidence_upper_bound_exceeds_target",
                )
            )
    return issues


def _read_fixed_fpr_observation_evidence(path: Path) -> tuple[dict[str, Any], ...]:
    """读取绑定的 observation evidence，并拒绝非结构化或非逐条记录输入。"""

    if path.suffix.lower() == ".jsonl":
        payload: Any = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("observations")
    if not isinstance(payload, list) or not payload or not all(isinstance(item, Mapping) for item in payload):
        raise ValueError("fixed-FPR observation evidence 必须是非空 observation 列表")
    return tuple(dict(item) for item in payload)


def _observation_evidence_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    """计算 observation 列表的规范摘要，使文件排版不影响 provenance。"""

    return build_stable_digest([dict(row) for row in rows])


def _observation_attack_name(row: Mapping[str, Any]) -> str:
    """统一读取 observation 使用的攻击名称字段。"""

    return _str_field(row, "attack_name") or _str_field(row, "attack_condition") or "clean_none"


def _formal_attack_identity_from_observations(
    rows: Iterable[Mapping[str, Any]],
    *,
    attack_family: str,
    attack_name: str,
) -> dict[str, str]:
    """从执行端 attacked observations 提取并核验唯一攻击身份."""

    config = resolve_formal_attack_config(
        attack_family=attack_family,
        attack_name=attack_name,
    )
    expected = {
        "attack_id": config.attack_id,
        "resource_profile": config.resource_profile,
        "attack_config_digest": attack_config_digest(config),
    }
    attacked_rows = tuple(
        row
        for row in rows
        if _str_field(row, "sample_role")
        in {"attacked_negative", "attacked_positive"}
    )
    if not attacked_rows:
        raise ValueError(
            f"正式攻击缺少 attacked observation: {attack_family}/{attack_name}"
        )
    for row in attacked_rows:
        for field_name, expected_value in expected.items():
            if _str_field(row, field_name) != expected_value:
                raise ValueError(
                    "正式 observation 攻击身份与 AttackConfig 不一致: "
                    f"{attack_family}/{attack_name}/{field_name}"
                )
    return expected


def _validate_formal_attack_identity(
    row: Mapping[str, Any],
    row_index: int,
) -> list[FormalImportIssue]:
    """校验候选记录直接继承的正式攻击身份."""

    try:
        config = resolve_formal_attack_config(
            attack_family=_str_field(row, "attack_family"),
            attack_name=_str_field(row, "attack_name"),
            resource_profile=_str_field(row, "resource_profile"),
        )
    except ValueError:
        return [
            _issue(
                row_index,
                row,
                "attack_id",
                "registered_formal_attack_identity_required",
            )
        ]
    expected = {
        "attack_id": config.attack_id,
        "resource_profile": config.resource_profile,
        "attack_config_digest": attack_config_digest(config),
    }
    return [
        _issue(
            row_index,
            row,
            field_name,
            "formal_attack_identity_must_match_attack_config",
        )
        for field_name, expected_value in expected.items()
        if _str_field(row, field_name) != expected_value
    ]


def _validate_exact_prompt_coverage(
    row: Mapping[str, Any],
    row_index: int,
    *,
    observation_rows: tuple[dict[str, Any], ...],
    expected_test_count: int,
) -> list[FormalImportIssue]:
    """校验单个正式攻击的正负样本是否一一覆盖独立 test Prompt。"""

    issues: list[FormalImportIssue] = []
    clean_negative_rows = tuple(
        observation
        for observation in observation_rows
        if _str_field(observation, "split") == "test"
        and _str_field(observation, "attack_family") == "clean"
        and _str_field(observation, "sample_role") == "clean_negative"
    )
    clean_prompt_ids = tuple(_str_field(observation, "prompt_id") for observation in clean_negative_rows)
    if len(clean_negative_rows) != expected_test_count:
        issues.append(
            _issue(
                row_index,
                row,
                "fixed_fpr_observation_evidence_path",
                "complete_test_clean_negative_observations_required",
            )
        )
    if any(not prompt_id for prompt_id in clean_prompt_ids) or len(set(clean_prompt_ids)) != len(clean_prompt_ids):
        issues.append(
            _issue(
                row_index,
                row,
                "fixed_fpr_observation_evidence_path",
                "unique_test_clean_negative_prompt_coverage_required",
            )
        )

    attack_family = _str_field(row, "attack_family")
    attack_name = _str_field(row, "attack_name")
    attack_rows = tuple(
        observation
        for observation in observation_rows
        if _str_field(observation, "split") == "test"
        and _str_field(observation, "attack_family") == attack_family
        and _observation_attack_name(observation) == attack_name
    )
    if attack_family != "clean" and any(
        _str_field(observation, "sample_role") == "positive_source" for observation in attack_rows
    ):
        issues.append(
            _issue(
                row_index,
                row,
                "positive_count",
                "non_clean_attack_requires_attacked_positive_role",
            )
        )

    expected_prompt_set = set(clean_prompt_ids)
    for sample_role, field_name in (
        ("attacked_positive", "positive_count"),
        ("attacked_negative", "attacked_negative_count"),
    ):
        role_rows = tuple(
            observation for observation in attack_rows if _str_field(observation, "sample_role") == sample_role
        )
        prompt_ids = tuple(_str_field(observation, "prompt_id") for observation in role_rows)
        if len(role_rows) != expected_test_count:
            issues.append(_issue(row_index, row, field_name, f"complete_test_{sample_role}_observations_required"))
        if (
            any(not prompt_id for prompt_id in prompt_ids)
            or len(set(prompt_ids)) != len(prompt_ids)
            or set(prompt_ids) != expected_prompt_set
        ):
            issues.append(_issue(row_index, row, field_name, f"exact_test_prompt_coverage_required_for_{sample_role}"))
    return issues


def _validate_observation_derived_metrics(
    row: Mapping[str, Any],
    row_index: int,
    *,
    observation_rows: tuple[dict[str, Any], ...],
) -> list[FormalImportIssue]:
    """从同一 observation evidence 重算正式计数、率值和 SSIM 均值。"""

    try:
        attack_family = _str_field(row, "attack_family")
        attack_name = _str_field(row, "attack_name")
        attack_rows = tuple(
            observation
            for observation in observation_rows
            if _str_field(observation, "attack_family") == attack_family
            and _observation_attack_name(observation) == attack_name
        )
        expected_values = build_primary_baseline_observation_metric_values(
            all_observations=observation_rows,
            attack_rows=attack_rows,
            attack_family=attack_family,
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return [
            _issue(
                row_index,
                row,
                "fixed_fpr_observation_evidence_path",
                "formal_metrics_cannot_be_recomputed_from_observations",
            )
        ]
    issues: list[FormalImportIssue] = []
    for field_name in (
        "positive_count",
        "negative_count",
        "attacked_negative_count",
        "attack_record_count",
        "supported_record_count",
    ):
        if _int_field(row, field_name) != int(expected_values[field_name]):
            issues.append(
                _issue(
                    row_index,
                    row,
                    field_name,
                    "formal_metric_must_match_governed_observations",
                )
            )
    for field_name in (
        "true_positive_rate",
        "false_positive_rate",
        "clean_false_positive_rate",
        "attacked_false_positive_rate",
        "quality_score_mean",
    ):
        if not math.isclose(
            _float_field(row, field_name),
            float(expected_values[field_name]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            issues.append(
                _issue(
                    row_index,
                    row,
                    field_name,
                    "formal_metric_must_match_governed_observations",
                )
            )
    return issues


def _validate_threshold_provenance(
    row: Mapping[str, Any],
    row_index: int,
    *,
    expected_calibration_count: int,
    expected_test_count: int,
    target_fpr: float,
    evidence_root: Path,
    evidence_search_roots: Iterable[str | Path],
    observation_evidence_cache: dict[Path, tuple[tuple[dict[str, Any], ...], str]],
) -> list[FormalImportIssue]:
    """从绑定 observation evidence 重算 fixed-FPR 阈值和 Prompt 覆盖。"""

    issues: list[FormalImportIssue] = []
    missing = [field for field in REQUIRED_THRESHOLD_FIELDS if field not in row]
    for field_name in missing:
        issues.append(_issue(row_index, row, field_name, "threshold_provenance_field_required"))
    if missing:
        return issues
    if _str_field(row, "evaluation_split") != "test":
        issues.append(_issue(row_index, row, "evaluation_split", "formal_metrics_must_use_test_split"))
    if not math.isclose(
        _float_field(row, "target_fpr"),
        target_fpr,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        issues.append(
            _issue(
                row_index,
                row,
                "target_fpr",
                "frozen_target_fpr_required",
            )
        )
    try:
        threshold = float(row["calibrated_detection_threshold"])
    except (TypeError, ValueError):
        threshold = math.nan
    if not math.isfinite(threshold):
        issues.append(_issue(row_index, row, "calibrated_detection_threshold", "finite_threshold_required"))
    if _str_field(row, "threshold_source") != FORMAL_THRESHOLD_SOURCE:
        issues.append(_issue(row_index, row, "threshold_source", "calibration_conformal_source_required"))
    if _int_field(row, "calibration_clean_negative_count") != expected_calibration_count:
        issues.append(_issue(row_index, row, "calibration_clean_negative_count", "complete_calibration_count_required"))
    if _int_field(row, "test_clean_negative_count") != expected_test_count:
        issues.append(_issue(row_index, row, "test_clean_negative_count", "complete_test_count_required"))
    if _int_field(row, "test_clean_negative_count") != _int_field(row, "negative_count"):
        issues.append(_issue(row_index, row, "test_clean_negative_count", "test_count_must_equal_negative_count"))

    evidence_path_value = _str_field(row, "fixed_fpr_observation_evidence_path")
    evidence_digest = _str_field(row, "fixed_fpr_observation_evidence_digest")
    if not evidence_path_value:
        issues.append(
            _issue(
                row_index,
                row,
                "fixed_fpr_observation_evidence_path",
                "governed_threshold_observation_evidence_path_required",
            )
        )
        return issues
    if evidence_path_value not in _list_field(row, "evidence_paths"):
        issues.append(
            _issue(
                row_index,
                row,
                "fixed_fpr_observation_evidence_path",
                "threshold_observation_evidence_must_be_bound",
            )
        )
    resolved_path = _resolve_evidence_path(evidence_root, evidence_path_value, evidence_search_roots)
    if not resolved_path.is_file():
        issues.append(
            _issue(
                row_index,
                row,
                "fixed_fpr_observation_evidence_path",
                "threshold_observation_evidence_missing",
            )
        )
        return issues
    try:
        if resolved_path not in observation_evidence_cache:
            observations = _read_fixed_fpr_observation_evidence(resolved_path)
            observation_evidence_cache[resolved_path] = (
                observations,
                _observation_evidence_digest(observations),
            )
        observations, actual_evidence_digest = observation_evidence_cache[resolved_path]
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        issues.append(
            _issue(
                row_index,
                row,
                "fixed_fpr_observation_evidence_path",
                "threshold_observation_evidence_invalid",
            )
        )
        return issues
    if len(evidence_digest) != 64 or evidence_digest != actual_evidence_digest:
        issues.append(
            _issue(
                row_index,
                row,
                "fixed_fpr_observation_evidence_digest",
                "threshold_observation_evidence_digest_mismatch",
            )
        )
    baseline_id = _str_field(row, "baseline_id")
    if any(_str_field(observation, "baseline_id") != baseline_id for observation in observations):
        issues.append(
            _issue(
                row_index,
                row,
                "fixed_fpr_observation_evidence_path",
                "threshold_observation_baseline_mismatch",
            )
        )
    attack_rows = tuple(
        observation
        for observation in observations
        if _str_field(observation, "attack_family")
        == _str_field(row, "attack_family")
        and _observation_attack_name(observation) == _str_field(row, "attack_name")
    )
    try:
        observation_identity = _formal_attack_identity_from_observations(
            attack_rows,
            attack_family=_str_field(row, "attack_family"),
            attack_name=_str_field(row, "attack_name"),
        )
    except ValueError:
        issues.append(
            _issue(
                row_index,
                row,
                "attack_config_digest",
                "observation_attack_identity_required",
            )
        )
    else:
        for field_name, expected_value in observation_identity.items():
            if _str_field(row, field_name) != expected_value:
                issues.append(
                    _issue(
                        row_index,
                        row,
                        field_name,
                        "result_attack_identity_must_match_observations",
                    )
                )

    threshold_audit = audit_fixed_fpr_observation_threshold(
        observations,
        target_fpr=target_fpr,
        expected_calibration_source_negative_count=expected_calibration_count,
    )
    if not threshold_audit.fixed_fpr_ready:
        issues.append(
            _issue(
                row_index,
                row,
                "fixed_fpr_observation_evidence_path",
                "governed_threshold_observation_audit_required",
            )
        )
    if threshold_audit.frozen_threshold is None or not math.isclose(
        threshold,
        threshold_audit.frozen_threshold,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        issues.append(
            _issue(
                row_index,
                row,
                "calibrated_detection_threshold",
                "calibrated_threshold_must_match_governed_observations",
            )
        )
    if _str_field(row, "threshold_digest") != threshold_audit.threshold_digest:
        issues.append(
            _issue(
                row_index,
                row,
                "threshold_digest",
                "threshold_digest_must_match_governed_observations",
            )
        )
    issues.extend(
        _validate_exact_prompt_coverage(
            row,
            row_index,
            observation_rows=observations,
            expected_test_count=expected_test_count,
        )
    )
    issues.extend(
        _validate_observation_derived_metrics(
            row,
            row_index,
            observation_rows=observations,
        )
    )
    return issues


def _validate_evidence_paths(
    row: Mapping[str, Any],
    row_index: int,
    evidence_root: Path,
    require_existing_evidence: bool,
    evidence_search_roots: Iterable[str | Path] = (),
) -> list[FormalImportIssue]:
    """校验证据路径是否非空并可在当前工作区或挂载目录中解析。"""

    evidence_paths = _list_field(row, "evidence_paths")
    issues: list[FormalImportIssue] = []
    if not evidence_paths:
        return [_issue(row_index, row, "evidence_paths", "evidence_paths_required")]
    if require_existing_evidence:
        for evidence_path in evidence_paths:
            if not _resolve_evidence_path(evidence_root, evidence_path, evidence_search_roots).is_file():
                issues.append(_issue(row_index, row, "evidence_paths", "evidence_path_missing"))
    return issues


def _validate_result_source_digest(
    row: Mapping[str, Any],
    row_index: int,
    evidence_root: Path,
    evidence_search_roots: Iterable[str | Path],
) -> list[FormalImportIssue]:
    """校验正式结果事实来源属于 evidence 集合且摘要匹配实际文件。"""

    source_value = _str_field(row, "baseline_result_source")
    source_digest = _str_field(row, "baseline_result_source_digest")
    issues: list[FormalImportIssue] = []
    if source_value not in _list_field(row, "evidence_paths"):
        issues.append(
            _issue(
                row_index,
                row,
                "baseline_result_source",
                "baseline_result_source_must_be_bound_to_evidence_paths",
            )
        )
    if len(source_digest) != 64:
        issues.append(
            _issue(
                row_index,
                row,
                "baseline_result_source_digest",
                "baseline_result_source_sha256_required",
            )
        )
        return issues
    resolved_path = _resolve_evidence_path(
        evidence_root,
        source_value,
        evidence_search_roots,
    )
    if resolved_path.is_file():
        digest = hashlib.sha256()
        with resolved_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_digest = digest.hexdigest()
        if actual_digest != source_digest:
            issues.append(
                _issue(
                    row_index,
                    row,
                    "baseline_result_source_digest",
                    "baseline_result_source_sha256_mismatch",
                )
            )
    return issues


def validate_primary_baseline_formal_import_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    evidence_root: str | Path = ".",
    target_fpr: float,
    require_existing_evidence: bool = True,
    evidence_search_roots: Iterable[str | Path] = (),
    prompt_protocol_name: str | None = None,
    allowed_resource_profiles: Iterable[str] = PAPER_ATTACK_RESOURCE_PROFILES,
) -> dict[str, Any]:
    """校验主表 baseline 正式结果导入记录, 并返回仅包含通过记录的报告。

    该函数是 schema 层边界: 下游 comparison builder 只消费 accepted_records, 因而不会把 method-faithful observation 或缺少 fixed-FPR 边界的记录误纳入正式主表统计。
    """

    evidence_root_path = Path(evidence_root).resolve()
    paper_run = build_paper_run_config(evidence_root_path)
    if not math.isclose(float(target_fpr), paper_run.target_fpr, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("baseline target_fpr 必须与当前论文运行层级一致")
    search_roots = _normalized_search_roots(evidence_root_path, evidence_search_roots)
    expected_operating_point = build_fixed_fpr_operating_point(target_fpr)
    expected_prompt_protocol_name = prompt_protocol_name or resolve_paper_run_prompt_protocol_name(evidence_root_path)
    expected_split_counts = build_group_split_counts(paper_run.prompt_count)
    resource_profiles = tuple(str(value) for value in allowed_resource_profiles if str(value).strip())
    resource_profile_issue_reason = (
        "full_main_resource_profile_required"
        if resource_profiles == ("full_main",)
        else "allowed_resource_profile_required"
    )
    accepted: list[dict[str, Any]] = []
    issues: list[FormalImportIssue] = []
    materialized_rows = [dict(row) for row in rows]
    seen_template_keys: set[tuple[str, str, str, str, str]] = set()
    observation_evidence_cache: dict[Path, tuple[tuple[dict[str, Any], ...], str]] = {}
    for row_index, row in enumerate(materialized_rows):
        row_issues: list[FormalImportIssue] = []
        template_key = _formal_template_key(row)
        if template_key in seen_template_keys:
            row_issues.append(_issue(row_index, row, "baseline_id", "duplicate_formal_template_key"))
        else:
            seen_template_keys.add(template_key)
        baseline_id = _str_field(row, "baseline_id")
        if baseline_id not in PRIMARY_BASELINE_IDS:
            row_issues.append(_issue(row_index, row, "baseline_id", "primary_baseline_id_required"))
        if _str_field(row, "result_protocol_name") != PRIMARY_BASELINE_FORMAL_PROTOCOL_NAME:
            row_issues.append(_issue(row_index, row, "result_protocol_name", "formal_result_protocol_required"))
        if _str_field(row, "result_source_type") not in ALLOWED_RESULT_SOURCE_TYPES:
            row_issues.append(_issue(row_index, row, "result_source_type", "allowed_result_source_type_required"))
        if _str_field(row, "resource_profile") not in resource_profiles:
            row_issues.append(_issue(row_index, row, "resource_profile", resource_profile_issue_reason))
        row_issues.extend(_validate_formal_attack_identity(row, row_index))
        if _str_field(row, "comparable_operating_point") != expected_operating_point:
            row_issues.append(_issue(row_index, row, "comparable_operating_point", "fixed_fpr_operating_point_required"))
        if _str_field(row, "prompt_protocol_name") != expected_prompt_protocol_name:
            row_issues.append(_issue(row_index, row, "prompt_protocol_name", "paper_run_prompt_protocol_required"))
        if not _str_field(row, "prompt_protocol_digest"):
            row_issues.append(_issue(row_index, row, "prompt_protocol_digest", "prompt_protocol_digest_required"))
        if not _str_field(row, "baseline_result_source"):
            row_issues.append(_issue(row_index, row, "baseline_result_source", "baseline_result_source_required"))
        if not _str_field(row, "baseline_result_source_digest"):
            row_issues.append(_issue(row_index, row, "baseline_result_source_digest", "baseline_result_source_digest_required"))
        adapter_boundary = _str_field(row, "adapter_boundary")
        if adapter_boundary in REJECTED_ADAPTER_BOUNDARIES:
            row_issues.append(_issue(row_index, row, "adapter_boundary", "adapter_boundary_not_formal"))
        elif adapter_boundary not in ALLOWED_ADAPTER_BOUNDARIES:
            row_issues.append(_issue(row_index, row, "adapter_boundary", "formal_adapter_boundary_required"))
        if _str_field(row, "metric_status") != "measured":
            row_issues.append(_issue(row_index, row, "metric_status", "measured_metric_status_required"))
        for flag_name in REQUIRED_READY_FLAGS:
            if not _bool_field(row, flag_name):
                row_issues.append(_issue(row_index, row, flag_name, f"{flag_name}_required"))
        row_issues.extend(
            _validate_metric_fields(
                row,
                row_index,
                expected_test_count=expected_split_counts["test"],
                target_fpr=target_fpr,
            )
        )
        row_issues.extend(
            _validate_threshold_provenance(
                row,
                row_index,
                expected_calibration_count=expected_split_counts["calibration"],
                expected_test_count=expected_split_counts["test"],
                target_fpr=target_fpr,
                evidence_root=evidence_root_path,
                evidence_search_roots=search_roots,
                observation_evidence_cache=observation_evidence_cache,
            )
        )
        row_issues.extend(
            _validate_evidence_paths(row, row_index, evidence_root_path, require_existing_evidence, search_roots)
        )
        row_issues.extend(
            _validate_result_source_digest(
                row,
                row_index,
                evidence_root_path,
                search_roots,
            )
        )
        if row_issues:
            issues.extend(row_issues)
        else:
            accepted.append(row)
    report = FormalImportValidationReport(
        protocol_name=PRIMARY_BASELINE_FORMAL_PROTOCOL_NAME,
        target_fpr=float(target_fpr),
        expected_operating_point=expected_operating_point,
        input_record_count=len(materialized_rows),
        accepted_formal_import_count=len(accepted),
        rejected_formal_import_count=len(materialized_rows) - len(accepted),
        formal_import_issue_count=len(issues),
        formal_import_validation_ready=bool(materialized_rows) and not issues,
        accepted_records=tuple(accepted),
        issues=tuple(issues),
    )
    return report.to_dict()


def _issue_rows_by_baseline(validation_report: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    """把正式导入校验问题按 baseline id 聚合。"""

    grouped = {baseline_id: [] for baseline_id in PRIMARY_BASELINE_IDS}
    for issue_row in validation_report.get("issues", ()):
        baseline_id = _str_field(issue_row, "baseline_id")
        if baseline_id in grouped:
            grouped[baseline_id].append(issue_row)
    return grouped


def _record_counts_by_baseline(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """统计每个主表 baseline 的记录数量。"""

    counts = {baseline_id: 0 for baseline_id in PRIMARY_BASELINE_IDS}
    for row in rows:
        baseline_id = _str_field(row, "baseline_id")
        if baseline_id in counts:
            counts[baseline_id] += 1
    return counts


def _candidate_flag_ready(rows: Iterable[Mapping[str, Any]], baseline_id: str, flag_name: str) -> bool:
    """检查同一 baseline 的任一候选记录是否声明给定 ready 标志。"""

    return any(_str_field(row, "baseline_id") == baseline_id and _bool_field(row, flag_name) for row in rows)


def _issue_fields(issues: Iterable[Mapping[str, Any]]) -> set[str]:
    """提取一组校验问题涉及的字段名。"""

    return {_str_field(issue, "field_name") for issue in issues}


def _issue_reasons(issues: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    """提取一组校验问题涉及的原因码。"""

    return tuple(sorted({_str_field(issue, "reason") for issue in issues if _str_field(issue, "reason")}))


def build_primary_baseline_formal_import_readiness_rows(
    candidate_rows: Iterable[Mapping[str, Any]],
    validation_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """构造主表 baseline 正式共同协议导入 readiness 行。

    该函数属于 schema 汇总层: 它把候选记录的拒绝原因集中整理为每个主表 baseline 一行,
    供文档审计和下游对比报告读取, 但不会把小样本或 method-faithful 候选提升为论文级正式结果。
    """

    materialized_candidates = [dict(row) for row in candidate_rows]
    candidate_counts = _record_counts_by_baseline(materialized_candidates)
    accepted_counts = _record_counts_by_baseline(validation_report.get("accepted_records", ()))
    issues_by_baseline = _issue_rows_by_baseline(validation_report)
    rows: list[dict[str, Any]] = []
    for baseline_id in PRIMARY_BASELINE_IDS:
        candidate_count = candidate_counts[baseline_id]
        accepted_count = accepted_counts[baseline_id]
        rejected_count = max(candidate_count - accepted_count, 0)
        issues = issues_by_baseline[baseline_id]
        reason_values = _issue_reasons(issues)
        if candidate_count == 0:
            reason_values = tuple(sorted((*reason_values, "candidate_record_missing")))
        field_values = _issue_fields(issues)
        formal_result_ready = bool(candidate_count) and accepted_count == candidate_count and not reason_values
        row = {
            "baseline_id": baseline_id,
            "candidate_record_count": candidate_count,
            "accepted_formal_import_count": accepted_count,
            "rejected_formal_import_count": rejected_count,
            "formal_import_issue_count": len(issues),
            "formal_result_ready": formal_result_ready,
            "blocking_reason_count": len(reason_values),
            "blocking_reasons": ";".join(reason_values),
            "missing_formal_attack_resource_profile": bool(
                candidate_count == 0
                or field_values & FORMAL_READINESS_BLOCKING_FLAG_GROUPS["missing_formal_attack_resource_profile"]
            ),
            "missing_paper_run_prompt_protocol": bool(
                candidate_count == 0
                or field_values & FORMAL_READINESS_BLOCKING_FLAG_GROUPS["missing_paper_run_prompt_protocol"]
            ),
            "missing_fixed_fpr_baseline_calibration": bool(
                candidate_count == 0
                or field_values & FORMAL_READINESS_BLOCKING_FLAG_GROUPS["missing_fixed_fpr_baseline_calibration"]
            ),
            "missing_attack_matrix_baseline_detection": bool(
                candidate_count == 0
                or field_values & FORMAL_READINESS_BLOCKING_FLAG_GROUPS["missing_attack_matrix_baseline_detection"]
            ),
            "formal_evidence_paths_ready": _candidate_flag_ready(
                materialized_candidates,
                baseline_id,
                "formal_evidence_paths_ready",
            )
            and "evidence_paths" not in field_values,
            "supports_paper_claim": False,
        }
        rows.append(row)
    return rows


def build_primary_baseline_formal_import_readiness_summary(
    readiness_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """构造主表 baseline 正式共同协议导入 readiness 摘要。"""

    rows = [dict(row) for row in readiness_rows]
    ready_ids = tuple(row["baseline_id"] for row in rows if _bool_field(row, "formal_result_ready"))
    blocked_ids = tuple(row["baseline_id"] for row in rows if not _bool_field(row, "formal_result_ready"))
    reason_counts: dict[str, int] = {}
    for row in rows:
        for reason in _list_field(row, "blocking_reasons"):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    dominant_reasons = tuple(
        reason
        for reason, _ in sorted(
            reason_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    return {
        "primary_baseline_count": len(rows),
        "formal_result_ready_count": len(ready_ids),
        "formal_result_ready_ids": list(ready_ids),
        "blocked_primary_baseline_ids": list(blocked_ids),
        "primary_baseline_formal_ready": len(ready_ids) == len(PRIMARY_BASELINE_IDS) and len(rows) == len(PRIMARY_BASELINE_IDS),
        "dominant_blocking_reasons": list(dominant_reasons),
        "formal_import_issue_count": sum(_int_field(row, "formal_import_issue_count") for row in rows),
        "supports_paper_claim": False,
    }


def _formal_template_key(
    row: Mapping[str, Any],
) -> tuple[str, str, str, str, str, str, str]:
    """生成正式共同协议模板覆盖检查使用的稳定键。"""

    return (
        _str_field(row, "baseline_id"),
        _str_field(row, "attack_id"),
        _str_field(row, "attack_family"),
        _str_field(row, "attack_name"),
        _str_field(row, "resource_profile"),
        _str_field(row, "attack_config_digest"),
        _str_field(row, "comparable_operating_point"),
    )


def build_primary_baseline_formal_template_coverage_rows(
    template_rows: Iterable[Mapping[str, Any]],
    candidate_rows: Iterable[Mapping[str, Any]],
    validation_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """构造主表 baseline 正式模板覆盖行。

    该函数属于项目特定写法: row-level validator 只能说明单条记录是否符合 schema,
    template coverage 进一步说明每个主表 baseline 是否覆盖了正式共同协议要求的全部攻击模板。
    """

    templates = [dict(row) for row in template_rows]
    candidates = [dict(row) for row in candidate_rows]
    accepted_records = [dict(row) for row in validation_report.get("accepted_records", ())]
    rows: list[dict[str, Any]] = []
    for baseline_id in PRIMARY_BASELINE_IDS:
        baseline_templates = [row for row in templates if _str_field(row, "baseline_id") == baseline_id]
        template_keys = {_formal_template_key(row) for row in baseline_templates}
        candidate_keys = [
            _formal_template_key(row) for row in candidates if _str_field(row, "baseline_id") == baseline_id
        ]
        accepted_keys = [
            _formal_template_key(row) for row in accepted_records if _str_field(row, "baseline_id") == baseline_id
        ]
        candidate_key_counts = Counter(candidate_keys)
        accepted_key_counts = Counter(accepted_keys)
        candidate_match_count = sum(1 for key in template_keys if key in candidate_key_counts)
        accepted_match_count = sum(1 for key in template_keys if key in accepted_key_counts)
        missing_count = max(len(template_keys) - accepted_match_count, 0)
        unexpected_candidate_count = sum(
            count for key, count in candidate_key_counts.items() if key not in template_keys
        )
        unexpected_accepted_count = sum(
            count for key, count in accepted_key_counts.items() if key not in template_keys
        )
        duplicate_candidate_count = sum(max(count - 1, 0) for count in candidate_key_counts.values())
        duplicate_accepted_count = sum(max(count - 1, 0) for count in accepted_key_counts.values())
        rows.append(
            {
                "baseline_id": baseline_id,
                "expected_formal_template_count": len(template_keys),
                "candidate_template_match_count": candidate_match_count,
                "accepted_template_match_count": accepted_match_count,
                "missing_formal_template_count": missing_count,
                "unexpected_candidate_record_count": unexpected_candidate_count,
                "unexpected_accepted_record_count": unexpected_accepted_count,
                "duplicate_candidate_template_count": duplicate_candidate_count,
                "duplicate_accepted_template_count": duplicate_accepted_count,
                "formal_template_coverage_ready": bool(template_keys)
                and missing_count == 0
                and unexpected_accepted_count == 0
                and duplicate_accepted_count == 0,
                "supports_paper_claim": False,
            }
        )
    return rows


def build_primary_baseline_formal_template_coverage_summary(
    coverage_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """构造主表 baseline 正式模板覆盖摘要。"""

    rows = [dict(row) for row in coverage_rows]
    ready_ids = tuple(row["baseline_id"] for row in rows if _bool_field(row, "formal_template_coverage_ready"))
    blocked_ids = tuple(row["baseline_id"] for row in rows if not _bool_field(row, "formal_template_coverage_ready"))
    formal_template_count = sum(_int_field(row, "expected_formal_template_count") for row in rows)
    candidate_match_count = sum(_int_field(row, "candidate_template_match_count") for row in rows)
    accepted_match_count = sum(_int_field(row, "accepted_template_match_count") for row in rows)
    unexpected_candidate_count = sum(_int_field(row, "unexpected_candidate_record_count") for row in rows)
    unexpected_accepted_count = sum(_int_field(row, "unexpected_accepted_record_count") for row in rows)
    duplicate_candidate_count = sum(_int_field(row, "duplicate_candidate_template_count") for row in rows)
    duplicate_accepted_count = sum(_int_field(row, "duplicate_accepted_template_count") for row in rows)
    return {
        "primary_baseline_count": len(rows),
        "formal_template_record_count": formal_template_count,
        "candidate_template_match_count": candidate_match_count,
        "accepted_template_match_count": accepted_match_count,
        "formal_template_coverage_ready_count": len(ready_ids),
        "formal_template_coverage_ready_ids": list(ready_ids),
        "blocked_primary_baseline_ids": list(blocked_ids),
        "primary_baseline_formal_template_coverage_ready": len(ready_ids) == len(PRIMARY_BASELINE_IDS)
        and len(rows) == len(PRIMARY_BASELINE_IDS),
        "missing_candidate_template_count": max(formal_template_count - candidate_match_count, 0),
        "missing_formal_template_count": sum(_int_field(row, "missing_formal_template_count") for row in rows),
        "unexpected_candidate_record_count": unexpected_candidate_count,
        "unexpected_accepted_record_count": unexpected_accepted_count,
        "duplicate_candidate_template_count": duplicate_candidate_count,
        "duplicate_accepted_template_count": duplicate_accepted_count,
        "supports_paper_claim": False,
    }


def build_primary_baseline_formal_evidence_collection_rows(
    template_rows: Iterable[Mapping[str, Any]],
    candidate_rows: Iterable[Mapping[str, Any]],
    validation_report: Mapping[str, Any],
    *,
    paper_run_name: str,
) -> list[dict[str, Any]]:
    """构造主表 baseline 正式证据收集计划行。

    该函数属于项目特定写法: 它不生成或伪造 baseline 结果, 只把缺失的正式模板
    转换为当前 paper run 独占的证据收集任务, 便于后续 Colab 或受治理导入流程
    逐项补齐且避免跨层级结果混用。
    """

    normalized_paper_run_name = str(paper_run_name).strip()
    if not normalized_paper_run_name:
        raise ValueError("paper_run_name 不能为空")
    candidates = [dict(row) for row in candidate_rows]
    accepted_records = [dict(row) for row in validation_report.get("accepted_records", ())]
    candidate_keys = [_formal_template_key(row) for row in candidates]
    accepted_keys = [_formal_template_key(row) for row in accepted_records]
    rows: list[dict[str, Any]] = []
    for template_row in template_rows:
        template = dict(template_row)
        template_key = _formal_template_key(template)
        candidate_match_count = candidate_keys.count(template_key)
        accepted_match_count = accepted_keys.count(template_key)
        actions: list[str] = []
        if candidate_match_count == 0:
            actions.append("generate_paper_baseline_result_record")
        if accepted_match_count == 0:
            actions.extend(
                [
                    "run_paper_prompt_protocol",
                    "calibrate_fixed_fpr_baseline",
                    "run_attack_matrix_baseline_detection",
                    "attach_formal_evidence_paths",
                ]
            )
        ready = accepted_match_count > 0
        payload = {
            "baseline_id": _str_field(template, "baseline_id"),
            "attack_id": _str_field(template, "attack_id"),
            "attack_family": _str_field(template, "attack_family"),
            "attack_name": _str_field(template, "attack_name"),
            "resource_profile": _str_field(template, "resource_profile"),
            "attack_config_digest": _str_field(
                template,
                "attack_config_digest",
            ),
            "comparable_operating_point": _str_field(template, "comparable_operating_point"),
            "candidate_template_match_count": candidate_match_count,
            "accepted_template_match_count": accepted_match_count,
            "formal_evidence_collection_ready": ready,
            "required_collection_actions": actions,
            "required_metric_fields": list(template.get("required_metric_fields", ())),
            "required_source_fields": list(template.get("required_source_fields", ())),
            "required_result_record_path": (
                "outputs/external_baseline_results/"
                f"{normalized_paper_run_name}/baseline_result_records.jsonl"
            ),
            "supports_paper_claim": False,
        }
        digest = build_stable_digest(payload)
        payload["formal_evidence_collection_id"] = f"primary_baseline_formal_evidence_collection_{digest[:16]}"
        payload["formal_evidence_collection_digest"] = digest
        rows.append(payload)
    return rows


def build_primary_baseline_formal_evidence_collection_summary(
    collection_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """构造主表 baseline 正式证据收集计划摘要。"""

    rows = [dict(row) for row in collection_rows]
    ready_count = sum(1 for row in rows if _bool_field(row, "formal_evidence_collection_ready"))
    missing_count = max(len(rows) - ready_count, 0)
    return {
        "formal_evidence_collection_task_count": len(rows),
        "ready_formal_evidence_collection_task_count": ready_count,
        "missing_formal_evidence_collection_task_count": missing_count,
        "primary_baseline_formal_evidence_collection_ready": bool(rows) and missing_count == 0,
        "supports_paper_claim": False,
    }


def _group_observations_by_attack(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    """按攻击族与攻击名称聚合 observation, 便于生成导入候选。"""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        attack_family = _str_field(row, "attack_family") or "clean"
        attack_name = _str_field(row, "attack_name") or _str_field(row, "attack_condition") or "clean_none"
        grouped.setdefault((attack_family, attack_name), []).append(row)
    return grouped


def build_primary_baseline_formal_result_record(
    *,
    baseline_id: str,
    attack_id: str,
    attack_family: str,
    attack_name: str,
    resource_profile: str,
    attack_config_digest_value: str,
    target_fpr: float,
    result_source_type: str,
    baseline_result_source: str,
    baseline_result_source_digest: str,
    evidence_paths: Iterable[str],
    prompt_protocol_digest: str,
    prompt_protocol_name: str | None = None,
    adapter_boundary: str,
    metric_values: Mapping[str, Any],
    ready_flags: Mapping[str, bool],
    threshold_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造一条稳定的主表 baseline 正式导入候选记录。"""

    payload = {
        "baseline_id": baseline_id,
        "attack_id": attack_id,
        "attack_family": attack_family,
        "attack_name": attack_name,
        "resource_profile": resource_profile,
        "attack_config_digest": attack_config_digest_value,
        "comparable_operating_point": build_fixed_fpr_operating_point(target_fpr),
        "target_fpr": float(target_fpr),
        "result_protocol_name": PRIMARY_BASELINE_FORMAL_PROTOCOL_NAME,
        "result_source_type": result_source_type,
        "baseline_result_source": baseline_result_source,
        "baseline_result_source_digest": baseline_result_source_digest,
        "metric_status": "measured",
        "prompt_protocol_name": prompt_protocol_name or resolve_paper_run_prompt_protocol_name(),
        "prompt_protocol_digest": prompt_protocol_digest,
        "adapter_boundary": adapter_boundary,
        "evidence_paths": list(evidence_paths),
        "supports_paper_claim": False,
        **{field_name: bool(ready_flags.get(field_name, False)) for field_name in REQUIRED_READY_FLAGS},
        **{field_name: metric_values[field_name] for field_name in REQUIRED_METRIC_FIELDS},
        **dict(threshold_values or {}),
    }
    digest = build_stable_digest(payload)
    payload["baseline_result_record_id"] = f"primary_baseline_formal_result_{digest[:16]}"
    payload["baseline_result_digest"] = digest
    return payload


def _select_fixed_fpr_observation_evidence_path(
    baseline_result_source: str,
    evidence_paths: Iterable[str],
) -> str:
    """从显式 evidence 集合中选择 observation 级 fixed-FPR 事实来源。"""

    candidates = (baseline_result_source, *tuple(evidence_paths))
    for candidate in dict.fromkeys(str(value) for value in candidates if str(value).strip()):
        path = Path(candidate)
        if "observation" in path.stem.lower() and path.suffix.lower() in {".json", ".jsonl"}:
            return candidate
    return ""


def _threshold_provenance_values(
    observations: tuple[dict[str, Any], ...],
    target_fpr: float,
    *,
    observation_evidence_path: str,
) -> dict[str, Any]:
    """从 observation 重算并绑定 fixed-FPR 阈值来源。"""

    calibration_count = sum(
        row.get("split") == "calibration"
        and row.get("sample_role") == "clean_negative"
        and row.get("attack_family") == "clean"
        for row in observations
    )
    test_count = sum(
        row.get("split") == "test"
        and row.get("sample_role") == "clean_negative"
        and row.get("attack_family") == "clean"
        for row in observations
    )
    if calibration_count <= 0:
        return {
            "evaluation_split": "test",
            "calibrated_detection_threshold": None,
            "threshold_source": "",
            "calibration_clean_negative_count": 0,
            "test_clean_negative_count": test_count,
            "threshold_digest": "",
            "fixed_fpr_observation_evidence_path": observation_evidence_path,
            "fixed_fpr_observation_evidence_digest": (
                _observation_evidence_digest(observations) if observation_evidence_path else ""
            ),
        }
    audit = audit_fixed_fpr_observation_threshold(
        observations,
        target_fpr=target_fpr,
        expected_calibration_source_negative_count=calibration_count,
    )
    return {
        "evaluation_split": "test",
        "calibrated_detection_threshold": audit.frozen_threshold,
        "threshold_source": FORMAL_THRESHOLD_SOURCE if audit.threshold_source_ready else "",
        "calibration_clean_negative_count": calibration_count,
        "test_clean_negative_count": test_count,
        "threshold_digest": audit.threshold_digest,
        "fixed_fpr_observation_evidence_path": observation_evidence_path,
        "fixed_fpr_observation_evidence_digest": (
            _observation_evidence_digest(observations) if observation_evidence_path else ""
        ),
    }


def build_primary_baseline_method_threshold_digest_map(
    candidate_rows: Iterable[Mapping[str, Any]],
    *,
    require_exact: bool = True,
) -> dict[str, str]:
    """构造四个主表 baseline 的精确阈值摘要映射.

    同一方法的逐攻击候选必须绑定同一个冻结阈值摘要.缺少方法,混入额外
    方法或同一方法出现多个摘要时直接拒绝, 避免 formal import 只报告整体
    ready 而丢失可交叉核验的阈值来源.
    """

    values_by_method: dict[str, set[str]] = {}
    for row in candidate_rows:
        baseline_id = _str_field(row, "baseline_id")
        threshold_digest = _str_field(row, "threshold_digest")
        values_by_method.setdefault(baseline_id, set()).add(threshold_digest)
    expected_ids = set(PRIMARY_BASELINE_IDS)
    if not set(values_by_method).issubset(expected_ids):
        raise ValueError("formal import 阈值摘要包含非主表 baseline")
    if require_exact and set(values_by_method) != expected_ids:
        raise ValueError("formal import 阈值摘要必须精确覆盖四个主表 baseline")
    result: dict[str, str] = {}
    for baseline_id, digests in sorted(values_by_method.items()):
        if len(digests) != 1:
            raise ValueError(f"{baseline_id} formal import 出现多个阈值摘要")
        digest = next(iter(digests))
        if len(digest) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in digest
        ):
            raise ValueError(f"{baseline_id} formal import 缺少 SHA-256 阈值摘要")
        result[baseline_id] = digest
    return result


def build_t2smark_formal_candidate_records(
    *,
    observation_rows: Iterable[Mapping[str, Any]],
    target_fpr: float,
    baseline_result_source: str,
    baseline_result_source_digest: str,
    evidence_paths: Iterable[str],
    prompt_protocol_digest: str,
    paper_run_prompt_protocol_ready: bool,
    fixed_fpr_baseline_calibration_ready: bool,
    attack_matrix_baseline_detection_ready: bool,
) -> tuple[dict[str, Any], ...]:
    """把 T2SMark formal observations 聚合为正式导入候选记录。

    该函数只负责候选记录聚合。是否可进入正式比较由 validate_primary_baseline_formal_import_rows 决定。
    """

    materialized_observations = tuple(dict(row) for row in observation_rows)
    evidence_path_values = tuple(evidence_paths)
    observation_evidence_path = _select_fixed_fpr_observation_evidence_path(
        baseline_result_source,
        evidence_path_values,
    )
    threshold_values = _threshold_provenance_values(
        materialized_observations,
        target_fpr,
        observation_evidence_path=observation_evidence_path,
    )
    records: list[dict[str, Any]] = []
    for (attack_family, attack_name), group in _group_observations_by_attack(materialized_observations).items():
        if attack_family == "clean":
            continue
        attack_identity = _formal_attack_identity_from_observations(
            group,
            attack_family=attack_family,
            attack_name=attack_name,
        )
        metric_values = build_primary_baseline_observation_metric_values(
            all_observations=materialized_observations,
            attack_rows=group,
            attack_family=attack_family,
        )
        ready_flags = {
            "method_faithful_adapter_ready": True,
            "paper_run_prompt_protocol_ready": paper_run_prompt_protocol_ready,
            "fixed_fpr_baseline_calibration_ready": fixed_fpr_baseline_calibration_ready,
            "attack_matrix_baseline_detection_ready": attack_matrix_baseline_detection_ready,
            "formal_evidence_paths_ready": bool(evidence_path_values),
        }
        records.append(
            build_primary_baseline_formal_result_record(
                baseline_id="t2smark",
                attack_id=attack_identity["attack_id"],
                attack_family=attack_family,
                attack_name=attack_name,
                resource_profile=attack_identity["resource_profile"],
                attack_config_digest_value=attack_identity[
                    "attack_config_digest"
                ],
                target_fpr=target_fpr,
                result_source_type="official_reproduction",
                baseline_result_source=baseline_result_source,
                baseline_result_source_digest=baseline_result_source_digest,
                evidence_paths=evidence_path_values,
                prompt_protocol_digest=prompt_protocol_digest,
                adapter_boundary="sd35_medium_native_official_reproduction",
                metric_values=metric_values,
                ready_flags=ready_flags,
                threshold_values=threshold_values,
            )
        )
    return tuple(records)


def _decision_field(row: Mapping[str, Any]) -> bool:
    """读取不同正式 adapter 的 JSON 布尔检测判定."""

    field_name = (
        "detection_decision"
        if "detection_decision" in row
        else "final_decision"
    )
    value = row.get(field_name)
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是布尔值")
    return value


def _mean_measured_rate(rows: Iterable[Mapping[str, Any]], field_name: str) -> float:
    """读取每条 observation 的实测 rate 字段并求均值。"""

    materialized = tuple(rows)
    if not materialized or any(
        field_name not in row
        or row.get(field_name) is None
        or isinstance(row.get(field_name), bool)
        for row in materialized
    ):
        raise ValueError(f"{field_name} 必须覆盖全部正式 observation")
    values = [float(row[field_name]) for row in materialized]
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{field_name} 必须覆盖全部 observation 且为有限值")
    return sum(values) / len(values)


def build_primary_baseline_observation_metric_values(
    *,
    all_observations: Iterable[Mapping[str, Any]],
    attack_rows: Iterable[Mapping[str, Any]],
    attack_family: str,
) -> dict[str, Any]:
    """按冻结 clean negative operating point 聚合单个攻击设置的指标。"""

    all_rows = tuple(
        row for row in all_observations if _str_field(row, "split") == "test"
    )
    group = tuple(row for row in attack_rows if _str_field(row, "split") == "test")
    clean_negative_rows = tuple(
        row
        for row in all_rows
        if _str_field(row, "sample_role") == "clean_negative"
        and _str_field(row, "attack_family") == "clean"
    )
    positive_rows = tuple(row for row in group if _str_field(row, "sample_role") == "attacked_positive")
    attacked_negative_rows = tuple(
        row for row in group if _str_field(row, "sample_role") == "attacked_negative"
    )
    true_positive = sum(1 for row in positive_rows if _decision_field(row))
    clean_false_positive = sum(1 for row in clean_negative_rows if _decision_field(row))
    attacked_false_positive = sum(1 for row in attacked_negative_rows if _decision_field(row))
    return {
        "positive_count": len(positive_rows),
        "negative_count": len(clean_negative_rows),
        "attacked_negative_count": len(attacked_negative_rows),
        # attack_record_count 只计入当前攻击实际产生的正、负样本记录.
        "attack_record_count": len(positive_rows) + len(attacked_negative_rows),
        # 质量与分数保持统计只由具备实测值的 attacked positive 支持.
        "supported_record_count": len(positive_rows),
        "true_positive_rate": true_positive / len(positive_rows) if positive_rows else 0.0,
        "false_positive_rate": clean_false_positive / len(clean_negative_rows) if clean_negative_rows else 0.0,
        "clean_false_positive_rate": (
            clean_false_positive / len(clean_negative_rows) if clean_negative_rows else 0.0
        ),
        "attacked_false_positive_rate": (
            attacked_false_positive / len(attacked_negative_rows) if attacked_negative_rows else 0.0
        ),
        "quality_score_mean": _mean_measured_rate(positive_rows, "quality_score"),
    }


def build_method_faithful_baseline_candidate_records(
    *,
    baseline_id: str,
    observation_rows: Iterable[Mapping[str, Any]],
    target_fpr: float,
    baseline_result_source: str,
    baseline_result_source_digest: str,
    evidence_paths: Iterable[str],
    prompt_protocol_digest: str,
    paper_run_prompt_protocol_ready: bool,
    fixed_fpr_baseline_calibration_ready: bool,
    attack_matrix_baseline_detection_ready: bool,
    result_source_type: str = "governed_import",
    adapter_boundary: str = METHOD_FAITHFUL_ADAPTER_BOUNDARY,
) -> tuple[dict[str, Any], ...]:
    """把方法忠实 SD3.5 baseline observations 聚合为正式导入候选记录。

    该函数属于通用 schema 前置聚合层。它只把已落盘 observation 映射到共同 fixed-FPR
    结果记录, 不负责把候选结果升级为论文结论。正式可用性仍由
    validate_primary_baseline_formal_import_rows 统一校验。
    """

    materialized_observations = tuple(dict(row) for row in observation_rows)
    records: list[dict[str, Any]] = []
    evidence_path_values = tuple(evidence_paths)
    observation_evidence_path = _select_fixed_fpr_observation_evidence_path(
        baseline_result_source,
        evidence_path_values,
    )
    threshold_values = _threshold_provenance_values(
        materialized_observations,
        target_fpr,
        observation_evidence_path=observation_evidence_path,
    )
    for (attack_family, attack_name), group in _group_observations_by_attack(materialized_observations).items():
        if attack_family == "clean":
            continue
        attack_identity = _formal_attack_identity_from_observations(
            group,
            attack_family=attack_family,
            attack_name=attack_name,
        )
        metric_values = build_primary_baseline_observation_metric_values(
            all_observations=materialized_observations,
            attack_rows=group,
            attack_family=attack_family,
        )
        ready_flags = {
            "method_faithful_adapter_ready": True,
            "paper_run_prompt_protocol_ready": paper_run_prompt_protocol_ready,
            "fixed_fpr_baseline_calibration_ready": fixed_fpr_baseline_calibration_ready,
            "attack_matrix_baseline_detection_ready": attack_matrix_baseline_detection_ready,
            "formal_evidence_paths_ready": bool(evidence_path_values),
        }
        records.append(
            build_primary_baseline_formal_result_record(
                baseline_id=baseline_id,
                attack_id=attack_identity["attack_id"],
                attack_family=attack_family,
                attack_name=attack_name,
                resource_profile=attack_identity["resource_profile"],
                attack_config_digest_value=attack_identity[
                    "attack_config_digest"
                ],
                target_fpr=target_fpr,
                result_source_type=result_source_type,
                baseline_result_source=baseline_result_source,
                baseline_result_source_digest=baseline_result_source_digest,
                evidence_paths=evidence_path_values,
                prompt_protocol_digest=prompt_protocol_digest,
                adapter_boundary=adapter_boundary,
                metric_values=metric_values,
                ready_flags=ready_flags,
                threshold_values=threshold_values,
            )
        )
    return tuple(records)


def build_tree_ring_method_faithful_candidate_records(
    *,
    observation_rows: Iterable[Mapping[str, Any]],
    target_fpr: float,
    baseline_result_source: str,
    baseline_result_source_digest: str,
    evidence_paths: Iterable[str],
    prompt_protocol_digest: str,
    paper_run_prompt_protocol_ready: bool,
    fixed_fpr_baseline_calibration_ready: bool,
    attack_matrix_baseline_detection_ready: bool,
) -> tuple[dict[str, Any], ...]:
    """把 Tree-Ring SD3.5 方法忠实 observation 聚合为主表正式导入候选记录。"""

    return build_method_faithful_baseline_candidate_records(
        baseline_id="tree_ring",
        observation_rows=observation_rows,
        target_fpr=target_fpr,
        baseline_result_source=baseline_result_source,
        baseline_result_source_digest=baseline_result_source_digest,
        evidence_paths=evidence_paths,
        prompt_protocol_digest=prompt_protocol_digest,
        paper_run_prompt_protocol_ready=paper_run_prompt_protocol_ready,
        fixed_fpr_baseline_calibration_ready=fixed_fpr_baseline_calibration_ready,
        attack_matrix_baseline_detection_ready=attack_matrix_baseline_detection_ready,
    )
