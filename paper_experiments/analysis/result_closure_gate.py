"""对完整论文结果闭合所需的受治理证据执行语义级门禁。"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import math
from pathlib import Path
import string
from typing import Any, Iterable, Mapping

from experiments.ablations.runtime_rerun import (
    FORMAL_RUNTIME_RERUN_ABLATION_IDS,
    FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
)
from experiments.artifacts.image_only_detection_metrics import (
    build_image_only_test_metric_rows,
)
from experiments.artifacts.attack_family_metrics import (
    ATTACK_FAMILY_METRIC_FIELDS,
    build_attack_family_metrics,
)
from experiments.protocol.attacks import (
    attack_config_digest,
    build_attack_record_digest,
    build_attack_matrix_manifest_config,
    default_attack_configs,
)
from experiments.protocol.pilot_paper_fixed_fpr import (
    PILOT_PAPER_METRIC_BOUNDS,
    PilotPaperFixedFprConfig,
    bounded_hoeffding_confidence_interval,
    bounded_metric_value,
    build_attack_matrix_digest,
    build_fixed_fpr_protocol_digest,
    build_pilot_paper_attack_matrix_rows,
    build_pilot_paper_result_import_schema,
    build_pilot_paper_result_record_set_digest,
    build_pilot_paper_result_records_manifest_config,
    clamp_unit_interval,
    prompt_protocol_name_for_run,
    result_claim_scope_for_run,
    result_protocol_name_for_run,
    result_scope_for_run,
    validate_pilot_paper_result_import_rows,
)
from experiments.protocol.paper_run_config import RUN_DEFAULTS
from experiments.protocol.splits import build_group_split_counts
from experiments.runtime.image_metrics import measured_score_retention
from main.core.digest import build_stable_digest
from paper_experiments.analysis.paired_superiority import (
    BOOTSTRAP_ANALYSIS_SCHEMA,
    BOOTSTRAP_BIT_GENERATOR,
    BOOTSTRAP_QUANTILE_METHOD,
    CLAIM_P_VALUE_METHOD,
    DEFAULT_BOOTSTRAP_RESAMPLE_COUNT,
    DEFAULT_CONFIDENCE_LEVEL,
    PRIMARY_BASELINE_IDS,
    SHARP_NULL_DIAGNOSTIC_METHOD,
    PairedSuperiorityError,
    build_paired_outcomes,
    build_paired_superiority_manifest_config,
    build_paired_superiority_protocol_digest,
    build_paired_superiority_rows,
    build_paired_superiority_summary,
    build_threshold_audit_binding_maps,
    canonical_attack_registry_rows,
    canonical_threshold_audit_rows,
)
from paper_experiments.analysis.fixed_fpr_threshold_audit import (
    build_fixed_fpr_threshold_manifest_config,
)
from paper_experiments.analysis.paper_evidence_audit import (
    AuditInputBundle,
    build_evidence_audit_manifest_config,
    build_evidence_audit_materialization,
)
from paper_experiments.baselines.formal_import import (
    build_primary_baseline_observation_metric_values,
)


def _recorded_source_path(path: Path, repository_root: Path) -> str:
    """优先以仓库相对路径记录门禁实际读取文件."""

    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_source_file_sha256_map(
    input_paths: Iterable[str | Path],
    *,
    root: str | Path,
) -> dict[str, str]:
    """构造门禁全部实际输入文件的路径到字节摘要映射.

    该映射属于通用工程写法. 最终打包器可以重新读取这些文件并比对
    SHA-256, 从而发现语义门禁通过后发生的任何源文件字节变化.
    """

    repository_root = Path(root).resolve()
    result: dict[str, str] = {}
    for raw_path in input_paths:
        candidate = Path(raw_path).expanduser()
        path = candidate.resolve() if candidate.is_absolute() else (repository_root / candidate).resolve()
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"结果闭合门禁输入必须是普通文件: {path.as_posix()}")
        recorded_path = _recorded_source_path(path, repository_root)
        if recorded_path in result:
            raise ValueError(f"结果闭合门禁输入路径重复: {recorded_path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[recorded_path] = digest.hexdigest()
    return dict(sorted(result.items()))


@dataclass(frozen=True)
class ResultClosureGateInput:
    """集中保存论文结果闭合门禁需要的只读证据。

    该对象属于通用工程写法: 文件读取由外层脚本负责, 本模块只表达跨产物
    语义一致性和 fail-closed 判定, 因而可以在 CPU 环境中稳定复用和测试。
    """

    expected_paper_claim_scale: str
    expected_target_fpr: float
    expected_prompt_count: int
    expected_test_count: int
    expected_prompt_split_digest: str
    expected_prompt_id_digest: str
    expected_calibration_prompt_id_digest: str
    expected_test_prompt_id_digest: str
    source_file_sha256: dict[str, str]
    attack_report: dict[str, Any]
    attack_detection_records: tuple[dict[str, Any], ...]
    attack_family_metrics: tuple[dict[str, Any], ...]
    attacked_image_registry: tuple[dict[str, Any], ...]
    attack_manifest: dict[str, Any]
    threshold_audit_report: dict[str, Any]
    threshold_audit_rows: tuple[dict[str, Any], ...]
    threshold_audit_manifest: dict[str, Any]
    closure_input_lock: dict[str, Any]
    closure_input_lock_manifest: dict[str, Any]
    official_reference_fidelity_records: tuple[dict[str, Any], ...]
    official_reference_fidelity_summary: dict[str, Any]
    official_reference_fidelity_manifest: dict[str, Any]
    primary_baseline_evidence_records: tuple[dict[str, Any], ...]
    primary_baseline_evidence_summary: dict[str, Any]
    primary_baseline_evidence_manifest: dict[str, Any]
    baseline_report: dict[str, Any]
    baseline_manifest: dict[str, Any]
    result_records: tuple[dict[str, Any], ...]
    result_record_validation_report: dict[str, Any]
    result_record_template_coverage: tuple[dict[str, Any], ...]
    result_record_summary: dict[str, Any]
    result_record_manifest: dict[str, Any]
    common_protocol_summary: dict[str, Any]
    common_protocol_schema: dict[str, Any]
    common_protocol_manifest: dict[str, Any]
    result_analysis_summary: dict[str, Any]
    result_analysis_manifest: dict[str, Any]
    paired_observation_records_by_method: dict[
        str,
        tuple[dict[str, Any], ...],
    ]
    paired_outcomes: tuple[dict[str, Any], ...]
    paired_superiority_rows: tuple[dict[str, Any], ...]
    paired_superiority_summary: dict[str, Any]
    paired_superiority_manifest: dict[str, Any]
    ablation_summary: dict[str, Any]
    ablation_manifest: dict[str, Any]
    dataset_quality_summary: dict[str, Any]
    dataset_quality_feature_report: dict[str, Any]
    dataset_quality_metrics: tuple[dict[str, Any], ...]
    dataset_quality_feature_records_sha256: str
    dataset_quality_manifest: dict[str, Any]
    evidence_builder_report: dict[str, Any]
    evidence_blocker_report: dict[str, Any]
    evidence_audit_runtime_report: dict[str, Any]
    evidence_audit_runtime_manifest: dict[str, Any]
    evidence_audit_source_path_map: dict[str, str]
    artifact_data_validation_report: dict[str, Any]
    recomputed_artifact_data_validation_report: dict[str, Any]
    evidence_audit_manifest: dict[str, Any]
    submission_readiness_report: dict[str, Any]
    submission_readiness_manifest: dict[str, Any]
    entry_review_report: dict[str, Any]
    entry_review_manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为紧凑字典, 大型原始记录由 source SHA 与 manifest 单独绑定."""

        payload = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name
            not in {
                "paired_observation_records_by_method",
                "attack_detection_records",
                "attacked_image_registry",
            }
        }
        return payload


def _strict_bool(value: Any) -> bool:
    """仅把明确的布尔值或常见布尔文本解释为真。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and value in {0, 1}:
        return bool(value)
    return str(value).strip().lower() in {"true", "1"}


def _float_value(value: Any) -> float | None:
    """读取有限浮点值, 缺失或非法时返回空值。"""

    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def _int_value(value: Any) -> int | None:
    """读取整数值, 非整数数值不得通过计数门禁。"""

    try:
        resolved = int(value)
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(numeric) and numeric == resolved else None


def _same_float(value: Any, expected: float) -> bool:
    """用固定绝对容差比较协议浮点值。"""

    resolved = _float_value(value)
    return resolved is not None and math.isclose(resolved, float(expected), rel_tol=0.0, abs_tol=1e-12)


def _is_sha256(value: Any) -> bool:
    """判断字段是否为完整的小写或大写 SHA-256 十六进制摘要。"""

    text = str(value)
    return len(text) == 64 and all(character in string.hexdigits for character in text)


def _all_true(payload: Mapping[str, Any], field_names: Iterable[str]) -> bool:
    """要求给定正向 readiness 字段全部明确为真。"""

    names = tuple(field_names)
    return bool(names) and all(field_name in payload and _strict_bool(payload[field_name]) for field_name in names)


def _all_zero(payload: Mapping[str, Any], field_names: Iterable[str]) -> bool:
    """要求给定阻断计数字段全部存在且等于0。"""

    names = tuple(field_names)
    return bool(names) and all(_int_value(payload.get(field_name)) == 0 for field_name in names)


def _path_present(paths: Iterable[Any], required_suffix: str) -> bool:
    """判断 manifest 路径列表是否包含指定受治理路径。"""

    normalized_suffix = required_suffix.replace("\\", "/").lstrip("./")
    return any(
        str(path).replace("\\", "/").lstrip("./").endswith(normalized_suffix)
        for path in paths
    )


def _declared_source_digests_ready(
    source_paths: Any,
    declared_digests: Any,
    actual_digests: Mapping[str, str],
) -> bool:
    """核验路径角色,声明摘要与门禁即时重算摘要完全一致."""

    if (
        not isinstance(source_paths, Mapping)
        or not source_paths
        or not isinstance(declared_digests, Mapping)
        or set(source_paths) != set(declared_digests)
    ):
        return False
    return all(
        _is_sha256(declared_digests.get(role, ""))
        and actual_digests.get(str(path)) == str(declared_digests.get(role, ""))
        for role, path in source_paths.items()
    )


def _manifest_ready(
    manifest: Mapping[str, Any],
    *,
    artifact_id: str,
    required_output_suffixes: Iterable[str],
) -> bool:
    """核验 manifest 身份、配置摘要、重建命令和关键输出路径。"""

    outputs = manifest.get("output_paths", ())
    return (
        str(manifest.get("artifact_id", "")) == artifact_id
        and bool(str(manifest.get("artifact_type", "")))
        and _is_sha256(manifest.get("config_digest", ""))
        and bool(str(manifest.get("code_version", "")))
        and bool(str(manifest.get("rebuild_command", "")))
        and all(_path_present(outputs, suffix) for suffix in required_output_suffixes)
    )


def _metadata_matches(
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    field_names: Iterable[str],
) -> bool:
    """核验 manifest metadata 与对应 report 的关键状态一致。"""

    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return False
    names = tuple(field_names)
    return bool(names) and all(field_name in report and metadata.get(field_name) == report.get(field_name) for field_name in names)


def _record_digest_ready(record: Mapping[str, Any]) -> bool:
    """重算单条正式结果记录摘要, 防止记录正文与 id 分离。"""

    digest = str(record.get("pilot_paper_result_record_digest", ""))
    record_id = str(record.get("pilot_paper_result_record_id", ""))
    payload = {
        key: value
        for key, value in record.items()
        if key not in {"pilot_paper_result_record_digest", "pilot_paper_result_record_id"}
    }
    return (
        _is_sha256(digest)
        and build_stable_digest(payload) == digest
        and record_id == f"pilot_paper_result_record_{digest[:16]}"
    )


def _normalized_method_id(value: Any) -> str:
    """把阈值审计中的主方法身份规范为正式结果记录身份."""

    method_id = str(value)
    return "slm_wm_current" if method_id == "slm_wm" else method_id


def _threshold_digest_map_from_audit(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, str] | None:
    """从五方法阈值审计行重建精确方法到阈值摘要映射."""

    expected_ids = {
        "slm_wm_current",
        "tree_ring",
        "gaussian_shading",
        "shallow_diffuse",
        "t2smark",
    }
    result: dict[str, str] = {}
    for row in rows:
        method_id = _normalized_method_id(row.get("method_id", ""))
        threshold_digest = str(row.get("threshold_digest", ""))
        if method_id in result or not _is_sha256(threshold_digest):
            return None
        result[method_id] = threshold_digest
    return dict(sorted(result.items())) if set(result) == expected_ids else None


def _primary_baseline_record_digest_ready(record: Mapping[str, Any]) -> bool:
    """复算 primary evidence 单条记录当前 builder 使用的身份摘要."""

    payload = {
        "baseline_id": str(record.get("baseline_id", "")),
        "source_status": str(record.get("source_status", "")),
        "official_repository_commit": str(record.get("official_repository_commit", "")),
        "adapter_status": str(record.get("adapter_status", "")),
        "adapter_run_ready": _strict_bool(record.get("adapter_run_ready")),
        "adapter_run_observation_count": _int_value(
            record.get("adapter_run_observation_count")
        ),
        "method_faithful_adapter_ready": _strict_bool(
            record.get("method_faithful_adapter_ready")
        ),
        "blocking_reasons": tuple(record.get("blocking_reasons", ())),
    }
    digest = str(record.get("primary_baseline_evidence_digest", ""))
    return (
        _is_sha256(digest)
        and build_stable_digest(payload) == digest
        and str(record.get("primary_baseline_evidence_id", ""))
        == f"primary_baseline_evidence_{digest[:16]}"
    )


def _official_reference_record_digest_ready(record: Mapping[str, Any]) -> bool:
    """复算官方参考方法忠实度单条证据记录摘要."""

    payload = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "official_reference_fidelity_record_id",
            "official_reference_fidelity_record_digest",
        }
    }
    digest = str(record.get("official_reference_fidelity_record_digest", ""))
    baseline_id = str(record.get("baseline_id", ""))
    return (
        _is_sha256(digest)
        and build_stable_digest(payload) == digest
        and str(record.get("official_reference_fidelity_record_id", ""))
        == f"{baseline_id}_official_reference_fidelity_{digest[:16]}"
    )


def _paired_outcome_digest_ready(record: Mapping[str, Any]) -> bool:
    """复算单个 Prompt 与攻击条件的二元配对结果摘要."""

    proposed = record.get("proposed_decision")
    baseline = record.get("baseline_decision")
    if not isinstance(proposed, bool) or not isinstance(baseline, bool):
        return False
    payload = {
        key: value for key, value in record.items() if key != "paired_outcome_digest"
    }
    return (
        _is_sha256(record.get("paired_outcome_digest", ""))
        and _int_value(record.get("paired_difference"))
        == int(proposed) - int(baseline)
        and build_stable_digest(payload) == str(record.get("paired_outcome_digest"))
    )


def _normalized_paired_statistical_row(
    row: Mapping[str, Any],
) -> dict[str, Any] | None:
    """把 CSV 配对统计行恢复为 builder 写出前的稳定类型."""

    integer_fields = (
        "paired_prompt_count",
        "paired_attack_count",
        "paired_observation_count",
        "positive_prompt_cluster_count",
        "negative_prompt_cluster_count",
        "tied_prompt_cluster_count",
        "bootstrap_resample_count",
    )
    float_fields = (
        "mean_paired_true_positive_rate_difference",
        "mean_paired_difference_ci_low",
        "mean_paired_difference_ci_high",
        "one_sided_bounded_hoeffding_mean_p_value",
        "one_sided_exact_prompt_cluster_sign_flip_p_value",
        "holm_adjusted_p_value",
        "confidence_level",
    )
    text_fields = (
        "baseline_id",
        "bootstrap_seed_digest_random",
        "sharp_null_diagnostic_method",
        "claim_p_value_method",
        "bootstrap_analysis_schema",
        "bootstrap_bit_generator",
        "bootstrap_quantile_method",
        "proposed_method_threshold_digest",
        "baseline_method_threshold_digest",
        "paired_attack_registry_digest",
        "paired_test_prompt_id_digest",
        "paired_outcome_set_digest",
        "protocol_digest",
    )
    boolean_fields = (
        "exact_prompt_cluster_sign_flip_p_value_is_diagnostic",
        "paired_superiority_ready",
        "supports_paper_claim",
    )
    if set(row) != {*integer_fields, *float_fields, *text_fields, *boolean_fields}:
        return None
    integers = {field_name: _int_value(row.get(field_name)) for field_name in integer_fields}
    floats = {field_name: _float_value(row.get(field_name)) for field_name in float_fields}
    if any(value is None for value in (*integers.values(), *floats.values())):
        return None
    return {
        "baseline_id": str(row.get("baseline_id", "")),
        **{field_name: int(value) for field_name, value in integers.items()},
        **{field_name: float(value) for field_name, value in floats.items()},
        "bootstrap_seed_digest_random": str(
            row.get("bootstrap_seed_digest_random", "")
        ),
        "exact_prompt_cluster_sign_flip_p_value_is_diagnostic": _strict_bool(
            row.get("exact_prompt_cluster_sign_flip_p_value_is_diagnostic")
        ),
        "sharp_null_diagnostic_method": str(
            row.get("sharp_null_diagnostic_method", "")
        ),
        "claim_p_value_method": str(row.get("claim_p_value_method", "")),
        "bootstrap_analysis_schema": str(
            row.get("bootstrap_analysis_schema", "")
        ),
        "bootstrap_bit_generator": str(row.get("bootstrap_bit_generator", "")),
        "bootstrap_quantile_method": str(
            row.get("bootstrap_quantile_method", "")
        ),
        "proposed_method_threshold_digest": str(
            row.get("proposed_method_threshold_digest", "")
        ),
        "baseline_method_threshold_digest": str(
            row.get("baseline_method_threshold_digest", "")
        ),
        "paired_attack_registry_digest": str(
            row.get("paired_attack_registry_digest", "")
        ),
        "paired_test_prompt_id_digest": str(
            row.get("paired_test_prompt_id_digest", "")
        ),
        "paired_outcome_set_digest": str(
            row.get("paired_outcome_set_digest", "")
        ),
        "protocol_digest": str(row.get("protocol_digest", "")),
        "paired_superiority_ready": _strict_bool(
            row.get("paired_superiority_ready")
        ),
        "supports_paper_claim": _strict_bool(row.get("supports_paper_claim")),
    }


def _check(
    check_id: str,
    check_area: str,
    ready: bool,
    source_artifacts: Iterable[str],
    blocker_reason: str,
) -> dict[str, Any]:
    """构造稳定的门禁检查行。"""

    return {
        "check_id": check_id,
        "check_area": check_area,
        "check_status": "pass" if ready else "blocked",
        "source_artifacts": list(source_artifacts),
        "blocker_reason": "" if ready else blocker_reason,
        "supports_paper_claim": False,
    }


def _scope_ready(bundle: ResultClosureGateInput) -> bool:
    """核验所有显式声明运行层级的产物均属于当前 run。"""

    expected = bundle.expected_paper_claim_scale
    values = (
        bundle.attack_report.get("paper_run_name"),
        bundle.threshold_audit_report.get("paper_claim_scale"),
        bundle.official_reference_fidelity_summary.get("paper_claim_scale"),
        bundle.primary_baseline_evidence_summary.get("paper_claim_scale"),
        bundle.result_record_summary.get("paper_claim_scale"),
        bundle.common_protocol_summary.get("paper_claim_scale"),
        bundle.common_protocol_schema.get("paper_claim_scale"),
        bundle.result_analysis_summary.get("paper_claim_scale"),
        bundle.paired_superiority_summary.get("paper_claim_scale"),
        bundle.ablation_summary.get("paper_claim_scale", bundle.ablation_summary.get("paper_run_name")),
        bundle.dataset_quality_summary.get(
            "paper_claim_scale",
            bundle.dataset_quality_summary.get("paper_run_name"),
        ),
        *(row.get("paper_claim_scale") for row in bundle.result_records),
    )
    optional_values = tuple(
        value
        for payload in (
            bundle.baseline_report,
            bundle.evidence_builder_report,
            bundle.evidence_blocker_report,
            bundle.submission_readiness_report,
            bundle.entry_review_report,
        )
        for value in (
            payload.get("paper_claim_scale"),
            payload.get("paper_run_name"),
        )
        if value is not None
    )
    return bool(values) and all(str(value) == expected for value in (*values, *optional_values))


def _target_fpr_ready(bundle: ResultClosureGateInput) -> bool:
    """核验跨攻击、阈值、baseline、质量、结果与消融的目标 FPR 一致。"""

    expected = bundle.expected_target_fpr
    attack_boundary = bundle.attack_report.get("evaluation_boundary", {})
    values = (
        attack_boundary.get("target_fpr") if isinstance(attack_boundary, Mapping) else None,
        bundle.threshold_audit_report.get("target_fpr"),
        bundle.official_reference_fidelity_summary.get("target_fpr"),
        bundle.primary_baseline_evidence_summary.get("target_fpr"),
        bundle.baseline_report.get("target_fpr"),
        bundle.common_protocol_summary.get("paper_target_fpr"),
        bundle.common_protocol_summary.get("expected_target_fpr"),
        bundle.common_protocol_schema.get("target_fpr"),
        bundle.result_analysis_summary.get("target_fpr"),
        bundle.paired_superiority_summary.get("target_fpr"),
        bundle.ablation_summary.get("target_fpr"),
        bundle.dataset_quality_summary.get("target_fpr"),
        *(row.get("target_fpr") for row in bundle.threshold_audit_rows),
        *(row.get("target_fpr") for row in bundle.result_records),
    )
    return bool(values) and all(_same_float(value, expected) for value in values)


def _attack_record_counts_ready(bundle: ResultClosureGateInput) -> bool:
    """核验攻击、样本角色和物化记录数之间的精确笛卡尔积关系。"""

    expected_ids = tuple(str(value) for value in bundle.attack_report.get("expected_attack_ids", ()))
    actual_ids = tuple(str(value) for value in bundle.attack_report.get("actual_attack_ids", ()))
    expected_role_count = _int_value(bundle.attack_report.get("expected_attack_split_role_count"))
    role_counts = bundle.attack_report.get("attack_split_role_counts", {})
    expected_role_keys = {
        f"{attack_id}|{sample_role}"
        for attack_id in expected_ids
        for sample_role in ("positive_source", "clean_negative")
    }
    if (
        not expected_ids
        or len(set(expected_ids)) != len(expected_ids)
        or len(set(actual_ids)) != len(actual_ids)
        or set(actual_ids) != set(expected_ids)
        or expected_role_count is None
        or expected_role_count <= 0
        or not isinstance(role_counts, Mapping)
        or set(str(key) for key in role_counts) != expected_role_keys
        or not all(_int_value(value) == expected_role_count for value in role_counts.values())
    ):
        return False
    expected_record_count = len(expected_role_keys) * expected_role_count
    count_fields = (
        "attack_record_count",
        "performed_attack_record_count",
        "formal_real_attack_record_count",
        "formal_image_attack_record_count",
        "real_attacked_image_count",
    )
    return all(_int_value(bundle.attack_report.get(field_name)) == expected_record_count for field_name in count_fields)


def _test_count_ready(bundle: ResultClosureGateInput) -> bool:
    """核验统一阈值、结果记录、共同协议与消融使用完整 test split。"""

    expected = bundle.expected_test_count
    ablation_splits = bundle.ablation_summary.get("split_counts", {})
    common_counts = (
        bundle.common_protocol_summary.get("pilot_paper_negative_count_minimum_required"),
        bundle.common_protocol_summary.get("minimum_result_positive_count"),
        bundle.common_protocol_summary.get("minimum_result_negative_count"),
        bundle.common_protocol_summary.get("minimum_result_attacked_negative_count"),
    )
    threshold_counts = tuple(row.get("test_clean_negative_count") for row in bundle.threshold_audit_rows)
    result_counts = tuple(
        value
        for row in bundle.result_records
        for value in (
            row.get("positive_count"),
            row.get("negative_count"),
            row.get("attacked_negative_count"),
        )
    )
    attack_counts_ready = (
        _attack_record_counts_ready(bundle)
        and _int_value(bundle.attack_report.get("expected_attack_split_role_count")) == expected
    )
    values = (*common_counts, *threshold_counts, *result_counts, ablation_splits.get("test"))
    return (
        attack_counts_ready
        and bool(bundle.threshold_audit_rows)
        and bool(bundle.result_records)
        and all(_int_value(value) == expected for value in values)
    )


def _threshold_digest_ready(bundle: ResultClosureGateInput) -> bool:
    """核验五方法冻结阈值摘要贯穿审计,正式记录与共同协议."""

    expected_map = _threshold_digest_map_from_audit(bundle.threshold_audit_rows)
    if expected_map is None:
        return False
    attack_boundary = bundle.attack_report.get("evaluation_boundary", {})
    attack_digest = attack_boundary.get("threshold_digest", "") if isinstance(attack_boundary, Mapping) else ""
    record_method_ids = {
        _normalized_method_id(row.get("method_id", "")) for row in bundle.result_records
    }
    result_metadata = bundle.result_record_manifest.get("metadata", {})
    common_metadata = bundle.common_protocol_manifest.get("metadata", {})
    return (
        str(attack_digest) == expected_map["slm_wm_current"]
        and record_method_ids == set(expected_map)
        and all(
            str(row.get("method_threshold_digest", ""))
            == expected_map.get(_normalized_method_id(row.get("method_id", "")), "")
            for row in bundle.result_records
        )
        and bundle.result_record_summary.get("method_threshold_digest_map") == expected_map
        and isinstance(result_metadata, Mapping)
        and result_metadata.get("method_threshold_digest_map") == expected_map
        and bundle.common_protocol_summary.get("method_threshold_digest_map") == expected_map
        and bundle.common_protocol_schema.get("method_threshold_digest_map") == expected_map
        and isinstance(common_metadata, Mapping)
        and common_metadata.get("method_threshold_digest_map") == expected_map
    )


def _common_protocol_digest_ready(bundle: ResultClosureGateInput) -> bool:
    """核验所有正式结果记录共享共同协议 schema 的三个关键摘要."""

    digest_fields = ("prompt_split_digest", "attack_matrix_digest", "fixed_fpr_protocol_digest")
    return bool(bundle.result_records) and all(
        _is_sha256(bundle.common_protocol_schema.get(field_name, ""))
        and all(
            str(row.get(field_name, "")) == str(bundle.common_protocol_schema.get(field_name, ""))
            for row in bundle.result_records
        )
        for field_name in digest_fields
    )


def _expected_fixed_fpr_config(
    bundle: ResultClosureGateInput,
) -> PilotPaperFixedFprConfig | None:
    """根据门禁显式期望值重建当前论文运行配置."""

    defaults = RUN_DEFAULTS.get(bundle.expected_paper_claim_scale)
    if defaults is None:
        return None
    try:
        return PilotPaperFixedFprConfig(
            paper_run_name=bundle.expected_paper_claim_scale,
            prompt_set=bundle.expected_paper_claim_scale,
            prompt_file=str(defaults["prompt_file"]),
            prompt_protocol_name=prompt_protocol_name_for_run(
                bundle.expected_paper_claim_scale
            ),
            result_protocol_name=result_protocol_name_for_run(
                bundle.expected_paper_claim_scale
            ),
            result_scope=result_scope_for_run(bundle.expected_paper_claim_scale),
            result_claim_scope=result_claim_scope_for_run(
                bundle.expected_paper_claim_scale
            ),
            target_fpr=bundle.expected_target_fpr,
            minimum_clean_negative_count=bundle.expected_test_count,
        )
    except (TypeError, ValueError):
        return None


def _expected_common_protocol_schema(
    bundle: ResultClosureGateInput,
) -> dict[str, Any] | None:
    """从规范配置、攻击表、阈值和配对统计重建共同协议 schema."""

    config = _expected_fixed_fpr_config(bundle)
    threshold_map = _threshold_digest_map_from_audit(bundle.threshold_audit_rows)
    if config is None or threshold_map is None:
        return None
    attack_rows = build_pilot_paper_attack_matrix_rows(
        default_attack_configs(),
        config,
    )
    attack_matrix_digest = build_attack_matrix_digest(attack_rows)
    fixed_fpr_protocol_digest = build_fixed_fpr_protocol_digest(config)
    schema = build_pilot_paper_result_import_schema(
        prompt_split_digest=bundle.expected_prompt_split_digest,
        attack_matrix_digest=attack_matrix_digest,
        fixed_fpr_protocol_digest=fixed_fpr_protocol_digest,
        config=config,
    )
    paired_summary = bundle.paired_superiority_summary
    schema.update(
        {
            "result_record_set_digest": (
                build_pilot_paper_result_record_set_digest(bundle.result_records)
            ),
            "calibration_prompt_id_digest": (
                bundle.expected_calibration_prompt_id_digest
            ),
            "test_prompt_id_digest": bundle.expected_test_prompt_id_digest,
            "method_threshold_digest_map": threshold_map,
            "closure_input_lock_digest": str(
                bundle.closure_input_lock.get("closure_input_lock_digest", "")
            ),
            "common_code_version": str(
                bundle.closure_input_lock.get("common_code_version", "")
            ),
            "paired_superiority_ready": paired_summary.get(
                "overall_paired_superiority_ready",
                False,
            ),
            "overall_paired_superiority_ready": paired_summary.get(
                "overall_paired_superiority_ready",
                False,
            ),
            "paired_superiority_protocol_digest": paired_summary.get(
                "paired_superiority_protocol_digest",
                "",
            ),
            "paired_superiority_rows_digest": paired_summary.get(
                "paired_superiority_rows_digest",
                "",
            ),
            "paired_outcome_set_digest": paired_summary.get(
                "paired_outcome_set_digest",
                "",
            ),
            "paired_test_prompt_count": paired_summary.get(
                "paired_test_prompt_count",
                0,
            ),
            "paired_test_prompt_id_digest": paired_summary.get(
                "paired_test_prompt_id_digest",
                "",
            ),
            "paired_attack_registry_digest": paired_summary.get(
                "paired_attack_registry_digest",
                "",
            ),
            "method_observation_source_sha256_map": paired_summary.get(
                "method_observation_source_sha256_map",
                {},
            ),
            "threshold_audit_rows_digest": paired_summary.get(
                "threshold_audit_rows_digest",
                "",
            ),
            "claim_p_value_method": paired_summary.get(
                "claim_p_value_method",
                "",
            ),
            "sharp_null_diagnostic_method": paired_summary.get(
                "sharp_null_diagnostic_method",
                "",
            ),
            "bootstrap_analysis_schema": paired_summary.get(
                "bootstrap_analysis_schema",
                "",
            ),
            "bootstrap_bit_generator": paired_summary.get(
                "bootstrap_bit_generator",
                "",
            ),
            "bootstrap_quantile_method": paired_summary.get(
                "bootstrap_quantile_method",
                "",
            ),
            "bootstrap_resample_count": paired_summary.get(
                "bootstrap_resample_count",
                0,
            ),
            "confidence_level": paired_summary.get("confidence_level", 0.0),
        }
    )
    return schema


def _result_record_set_provenance_ready(bundle: ResultClosureGateInput) -> bool:
    """复算正式记录稳定有序集合摘要并核验全部下游登记."""

    expected_digest = build_pilot_paper_result_record_set_digest(bundle.result_records)
    result_metadata = bundle.result_record_manifest.get("metadata", {})
    common_metadata = bundle.common_protocol_manifest.get("metadata", {})
    analysis_metadata = bundle.result_analysis_manifest.get("metadata", {})
    values = (
        bundle.result_record_summary.get("result_record_set_digest"),
        result_metadata.get("result_record_set_digest") if isinstance(result_metadata, Mapping) else None,
        bundle.common_protocol_summary.get("result_record_set_digest"),
        bundle.common_protocol_schema.get("result_record_set_digest"),
        common_metadata.get("result_record_set_digest") if isinstance(common_metadata, Mapping) else None,
        bundle.result_analysis_summary.get("result_record_set_digest"),
        analysis_metadata.get("result_record_set_digest") if isinstance(analysis_metadata, Mapping) else None,
    )
    return _is_sha256(expected_digest) and all(
        str(value) == expected_digest for value in values
    )


def _formal_attack_registry_rows(
    bundle: ResultClosureGateInput,
) -> tuple[dict[str, str], ...] | None:
    """重建项目登记的完整正式攻击 registry, 并核验报告未删减模板."""

    formal_configs = tuple(
        config
        for config in default_attack_configs()
        if config.enabled and config.resource_profile in {"full_main", "full_extra"}
    )
    expected_ids = {
        str(value) for value in bundle.attack_report.get("expected_attack_ids", ())
    }
    if expected_ids != {config.attack_id for config in formal_configs}:
        return None
    try:
        return canonical_attack_registry_rows(
            {
                "attack_id": config.attack_id,
                "attack_family": config.attack_family,
                "attack_name": config.attack_name,
                "resource_profile": config.resource_profile,
                "attack_config_digest": attack_config_digest(config),
            }
            for config in formal_configs
        )
    except PairedSuperiorityError:
        return None


def _threshold_audit_bindings(
    bundle: ResultClosureGateInput,
) -> tuple[tuple[dict[str, Any], ...], dict[str, str], dict[str, str]] | None:
    """规范化五方法阈值行并返回阈值与 observation 摘要映射."""

    try:
        canonical_rows = canonical_threshold_audit_rows(bundle.threshold_audit_rows)
        threshold_map, observation_map = build_threshold_audit_binding_maps(
            canonical_rows
        )
    except (PairedSuperiorityError, TypeError, ValueError):
        return None
    return canonical_rows, threshold_map, observation_map


def _expected_result_metric_fields(
    *,
    positive_count: int,
    negative_count: int,
    attacked_negative_count: int,
    attack_record_count: int,
    supported_record_count: int,
    true_positive_rate: float,
    false_positive_rate: float,
    clean_false_positive_rate: float,
    attacked_false_positive_rate: float,
    quality_score_mean: float,
    score_retention_mean: float,
    confidence_level: float,
) -> dict[str, int | float]:
    """从原始计数和均值重建 result record 的完整指标字段."""

    counts = {
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "attacked_negative_count": int(attacked_negative_count),
        "attack_record_count": int(attack_record_count),
        "supported_record_count": int(supported_record_count),
    }
    if any(value <= 0 for value in counts.values()):
        raise ValueError("正式结果指标计数必须为正整数")
    quality_lower, quality_upper = PILOT_PAPER_METRIC_BOUNDS[
        "quality_score_mean"
    ]
    rates = {
        "true_positive_rate": clamp_unit_interval(true_positive_rate),
        "false_positive_rate": clamp_unit_interval(false_positive_rate),
        "clean_false_positive_rate": clamp_unit_interval(
            clean_false_positive_rate
        ),
        "attacked_false_positive_rate": clamp_unit_interval(
            attacked_false_positive_rate
        ),
        "quality_score_mean": bounded_metric_value(
            quality_score_mean,
            lower_bound=quality_lower,
            upper_bound=quality_upper,
        ),
        "score_retention_mean": clamp_unit_interval(score_retention_mean),
    }
    ci_fields = {
        "true_positive_rate": (
            counts["positive_count"],
            "true_positive_rate_ci_low",
            "true_positive_rate_ci_high",
            0.0,
            1.0,
        ),
        "false_positive_rate": (
            counts["negative_count"],
            "false_positive_rate_ci_low",
            "false_positive_rate_ci_high",
            0.0,
            1.0,
        ),
        "clean_false_positive_rate": (
            counts["negative_count"],
            "clean_false_positive_rate_ci_low",
            "clean_false_positive_rate_ci_high",
            0.0,
            1.0,
        ),
        "attacked_false_positive_rate": (
            counts["attacked_negative_count"],
            "attacked_false_positive_rate_ci_low",
            "attacked_false_positive_rate_ci_high",
            0.0,
            1.0,
        ),
        # 与生产器一致, 质量与分数保持区间只使用 attacked positive 样本数.
        "quality_score_mean": (
            counts["positive_count"],
            "quality_score_ci_low",
            "quality_score_ci_high",
            quality_lower,
            quality_upper,
        ),
        "score_retention_mean": (
            counts["positive_count"],
            "score_retention_ci_low",
            "score_retention_ci_high",
            0.0,
            1.0,
        ),
    }
    result: dict[str, int | float] = {**counts, **rates}
    for metric_name, (
        sample_count,
        low_name,
        high_name,
        lower_bound,
        upper_bound,
    ) in ci_fields.items():
        low, high = bounded_hoeffding_confidence_interval(
            rates[metric_name],
            sample_count,
            confidence_level,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        result[low_name] = low
        result[high_name] = high
    return result


def _single_metric_row(
    rows: Iterable[Mapping[str, Any]],
    *,
    attack_family: str,
    attack_name: str,
    resource_profile: str,
    sample_role: str,
) -> Mapping[str, Any]:
    """按完整攻击与样本角色身份提取唯一指标行."""

    matches = tuple(
        row
        for row in rows
        if str(row.get("attack_family", "")) == attack_family
        and str(row.get("attack_name", "")) == attack_name
        and str(row.get("resource_profile", "")) == resource_profile
        and str(row.get("sample_role", "")) == sample_role
    )
    if len(matches) != 1:
        raise ValueError("原始 observation 未生成唯一指标行")
    return matches[0]


def _prompt_group_exact(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str,
    sample_role: str,
    expected_count: int,
    expected_prompt_id_digest: str,
    attack: Mapping[str, str] | None = None,
) -> bool:
    """核验一个 split x role x attack 分组精确覆盖规范 Prompt 集合."""

    matches = []
    for row in rows:
        if (
            str(row.get("split", "")) != split
            or str(row.get("sample_role", "")) != sample_role
        ):
            continue
        if attack is None:
            if (
                str(row.get("attack_family", "clean")) != "clean"
                or str(row.get("attack_id", ""))
            ):
                continue
        elif any(
            str(row.get(field_name, "")) != str(attack[field_name])
            for field_name in (
                "attack_id",
                "attack_family",
                "attack_name",
                "resource_profile",
                "attack_config_digest",
            )
        ):
            continue
        matches.append(row)
    prompt_ids = [str(row.get("prompt_id", "")) for row in matches]
    return (
        len(prompt_ids) == expected_count
        and len(set(prompt_ids)) == expected_count
        and all(prompt_ids)
        and build_stable_digest(sorted(prompt_ids))
        == expected_prompt_id_digest
    )


def _observation_prompt_coverage_ready(
    bundle: ResultClosureGateInput,
    attack_registry: tuple[dict[str, str], ...],
) -> bool:
    """核验五方法 calibration/test 的每个正式角色精确覆盖规范 Prompt."""

    observations = bundle.paired_observation_records_by_method
    split_counts = build_group_split_counts(bundle.expected_prompt_count)
    calibration_count = split_counts["calibration"]
    if split_counts["test"] != bundle.expected_test_count:
        return False
    main_rows = observations.get("slm_wm", ())
    main_groups_ready = all(
        (
            _prompt_group_exact(
                main_rows,
                split="calibration",
                sample_role="clean_negative",
                expected_count=calibration_count,
                expected_prompt_id_digest=(
                    bundle.expected_calibration_prompt_id_digest
                ),
            ),
            *(
                _prompt_group_exact(
                    main_rows,
                    split="test",
                    sample_role=sample_role,
                    expected_count=bundle.expected_test_count,
                    expected_prompt_id_digest=(
                        bundle.expected_test_prompt_id_digest
                    ),
                )
                for sample_role in (
                    "clean_negative",
                    "wrong_key_negative",
                    "positive_source",
                )
            ),
            *(
                _prompt_group_exact(
                    main_rows,
                    split="test",
                    sample_role=sample_role,
                    expected_count=bundle.expected_test_count,
                    expected_prompt_id_digest=(
                        bundle.expected_test_prompt_id_digest
                    ),
                    attack=attack,
                )
                for attack in attack_registry
                for sample_role in ("clean_negative", "positive_source")
            ),
        )
    )
    baseline_groups_ready = all(
        _prompt_group_exact(
            observations.get(baseline_id, ()),
            split="calibration",
            sample_role="clean_negative",
            expected_count=calibration_count,
            expected_prompt_id_digest=(
                bundle.expected_calibration_prompt_id_digest
            ),
        )
        and _prompt_group_exact(
            observations.get(baseline_id, ()),
            split="test",
            sample_role="clean_negative",
            expected_count=bundle.expected_test_count,
            expected_prompt_id_digest=bundle.expected_test_prompt_id_digest,
        )
        and all(
            _prompt_group_exact(
                observations.get(baseline_id, ()),
                split="test",
                sample_role=sample_role,
                expected_count=bundle.expected_test_count,
                expected_prompt_id_digest=(
                    bundle.expected_test_prompt_id_digest
                ),
                attack=attack,
            )
            for attack in attack_registry
            for sample_role in ("attacked_positive", "attacked_negative")
        )
        for baseline_id in PRIMARY_BASELINE_IDS
    )
    return main_groups_ready and baseline_groups_ready


def _result_record_metric_expectations(
    bundle: ResultClosureGateInput,
    attack_registry: tuple[dict[str, str], ...],
) -> dict[tuple[str, str], dict[str, int | float]] | None:
    """从五方法原始 observation 重建逐攻击正式指标."""

    observations = bundle.paired_observation_records_by_method
    confidence_level = _float_value(
        bundle.common_protocol_schema.get("confidence_level")
    )
    if (
        set(observations) != {"slm_wm", *PRIMARY_BASELINE_IDS}
        or confidence_level is None
        or not _observation_prompt_coverage_ready(bundle, attack_registry)
    ):
        return None
    try:
        main_rows = build_image_only_test_metric_rows(
            observations["slm_wm"],
            bundle.expected_target_fpr,
        )
        clean_negative = _single_metric_row(
            main_rows,
            attack_family="clean",
            attack_name="none",
            resource_profile="clean",
            sample_role="clean_negative",
        )
        clean_positive = _single_metric_row(
            main_rows,
            attack_family="clean",
            attack_name="none",
            resource_profile="clean",
            sample_role="positive_source",
        )
        clean_positive_score = float(clean_positive["content_score_mean"])
        expectations: dict[tuple[str, str], dict[str, int | float]] = {}
        for attack in attack_registry:
            positive = _single_metric_row(
                main_rows,
                attack_family=attack["attack_family"],
                attack_name=attack["attack_name"],
                resource_profile=attack["resource_profile"],
                sample_role="positive_source",
            )
            attacked_negative = _single_metric_row(
                main_rows,
                attack_family=attack["attack_family"],
                attack_name=attack["attack_name"],
                resource_profile=attack["resource_profile"],
                sample_role="clean_negative",
            )
            quality_value = positive.get("source_to_evaluated_ssim_mean")
            if quality_value is None:
                raise ValueError("主方法正式攻击记录缺少 SSIM")
            expectations[("slm_wm_current", attack["attack_id"])] = (
                _expected_result_metric_fields(
                    positive_count=int(positive["record_count"]),
                    negative_count=int(clean_negative["record_count"]),
                    attacked_negative_count=int(attacked_negative["record_count"]),
                    attack_record_count=(
                        int(positive["record_count"])
                        + int(attacked_negative["record_count"])
                    ),
                    supported_record_count=int(positive["record_count"]),
                    true_positive_rate=float(positive["positive_rate"]),
                    false_positive_rate=float(clean_negative["positive_rate"]),
                    clean_false_positive_rate=float(
                        clean_negative["positive_rate"]
                    ),
                    attacked_false_positive_rate=float(
                        attacked_negative["positive_rate"]
                    ),
                    quality_score_mean=float(quality_value),
                    score_retention_mean=measured_score_retention(
                        clean_positive_score,
                        float(positive["content_score_mean"]),
                    ),
                    confidence_level=confidence_level,
                )
            )

        attack_by_key = {
            (row["attack_family"], row["attack_name"]): row
            for row in attack_registry
        }
        for baseline_id in PRIMARY_BASELINE_IDS:
            baseline_rows = observations[baseline_id]
            if any(
                str(row.get("baseline_id", "")) != baseline_id
                for row in baseline_rows
            ):
                raise ValueError("baseline observation 身份不一致")
            for attack_key, attack in attack_by_key.items():
                group = tuple(
                    row
                    for row in baseline_rows
                    if str(row.get("attack_family", "")) == attack_key[0]
                    and str(row.get("attack_name", "")) == attack_key[1]
                )
                for row in group:
                    descriptor = {
                        field_name: str(row.get(field_name, ""))
                        for field_name in (
                            "attack_id",
                            "attack_family",
                            "attack_name",
                            "resource_profile",
                            "attack_config_digest",
                        )
                    }
                    if descriptor != attack:
                        raise ValueError("baseline observation 攻击身份不一致")
                values = build_primary_baseline_observation_metric_values(
                    all_observations=baseline_rows,
                    attack_rows=group,
                    attack_family=attack["attack_family"],
                )
                expectations[(baseline_id, attack["attack_id"])] = (
                    _expected_result_metric_fields(
                        positive_count=int(values["positive_count"]),
                        negative_count=int(values["negative_count"]),
                        attacked_negative_count=int(
                            values["attacked_negative_count"]
                        ),
                        attack_record_count=int(values["attack_record_count"]),
                        supported_record_count=int(
                            values["supported_record_count"]
                        ),
                        true_positive_rate=float(values["true_positive_rate"]),
                        false_positive_rate=float(values["false_positive_rate"]),
                        clean_false_positive_rate=float(
                            values["clean_false_positive_rate"]
                        ),
                        attacked_false_positive_rate=float(
                            values["attacked_false_positive_rate"]
                        ),
                        quality_score_mean=float(values["quality_score_mean"]),
                        score_retention_mean=0.0,
                        confidence_level=confidence_level,
                    )
                )
    except (KeyError, TypeError, ValueError):
        return None
    return expectations


def _record_metrics_match(
    record: Mapping[str, Any],
    expected: Mapping[str, int | float],
) -> bool:
    """逐字段核验计数、率值、质量值和全部 Hoeffding 区间."""

    for field_name, expected_value in expected.items():
        if isinstance(expected_value, int):
            if _int_value(record.get(field_name)) != expected_value:
                return False
        elif not _same_float(record.get(field_name), expected_value):
            return False
    return True


def _paired_result_record_consistency_ready(
    bundle: ResultClosureGateInput,
    outcomes: tuple[dict[str, Any], ...],
    attack_registry: tuple[dict[str, str], ...],
) -> bool:
    """核验配对二元判定与正式结果记录的逐攻击计数和均值一致."""

    expected_method_ids = ("slm_wm_current", *PRIMARY_BASELINE_IDS)
    metric_expectations = _result_record_metric_expectations(
        bundle,
        attack_registry,
    )
    if metric_expectations is None:
        return False
    expected_record_keys = {
        (
            method_id,
            attack["attack_id"],
            attack["attack_family"],
            attack["attack_name"],
            attack["resource_profile"],
            attack["attack_config_digest"],
        )
        for method_id in expected_method_ids
        for attack in attack_registry
    }
    record_index: dict[tuple[str, str, str, str, str, str], Mapping[str, Any]] = {}
    for row in bundle.result_records:
        key = (
            str(row.get("method_id", "")),
            str(row.get("attack_id", "")),
            str(row.get("attack_family", "")),
            str(row.get("attack_name", "")),
            str(row.get("resource_profile", "")),
            str(row.get("attack_config_digest", "")),
        )
        if key in record_index:
            return False
        record_index[key] = row
    if set(record_index) != expected_record_keys:
        return False
    if set(metric_expectations) != {
        (method_id, attack["attack_id"])
        for method_id in expected_method_ids
        for attack in attack_registry
    }:
        return False
    if any(
        not _record_metrics_match(
            record,
            metric_expectations[
                (
                    str(record.get("method_id", "")),
                    str(record.get("attack_id", "")),
                )
            ],
        )
        for record in record_index.values()
    ):
        return False

    expected_prompt_count = bundle.expected_test_count
    attack_by_id = {row["attack_id"]: row for row in attack_registry}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in outcomes:
        grouped.setdefault(
            (str(row.get("baseline_id", "")), str(row.get("attack_id", ""))),
            [],
        ).append(row)

    proposed_decisions_by_attack: dict[str, dict[str, bool]] = {}
    method_attack_tprs: dict[str, list[float]] = {
        method_id: [] for method_id in expected_method_ids
    }
    for baseline_id in PRIMARY_BASELINE_IDS:
        for attack_id, attack in attack_by_id.items():
            paired_rows = grouped.get((baseline_id, attack_id), [])
            if len(paired_rows) != expected_prompt_count:
                return False
            prompt_decisions = {
                str(row.get("prompt_id", "")): bool(row.get("proposed_decision"))
                for row in paired_rows
            }
            if len(prompt_decisions) != expected_prompt_count:
                return False
            previous = proposed_decisions_by_attack.setdefault(
                attack_id,
                prompt_decisions,
            )
            if previous != prompt_decisions:
                return False
            attack_key = (
                attack["attack_id"],
                attack["attack_family"],
                attack["attack_name"],
                attack["resource_profile"],
                attack["attack_config_digest"],
            )
            baseline_record = record_index[(baseline_id, *attack_key)]
            baseline_positive_count = sum(
                bool(row.get("baseline_decision")) for row in paired_rows
            )
            if (
                _int_value(baseline_record.get("positive_count"))
                != expected_prompt_count
                or not _same_float(
                    baseline_record.get("true_positive_rate"),
                    baseline_positive_count / expected_prompt_count,
                )
            ):
                return False
            method_attack_tprs[baseline_id].append(
                baseline_positive_count / expected_prompt_count
            )

    for attack_id, attack in attack_by_id.items():
        decisions = proposed_decisions_by_attack.get(attack_id, {})
        proposed_positive_count = sum(decisions.values())
        attack_key = (
            attack["attack_id"],
            attack["attack_family"],
            attack["attack_name"],
            attack["resource_profile"],
            attack["attack_config_digest"],
        )
        proposed_record = record_index[("slm_wm_current", *attack_key)]
        proposed_tpr = proposed_positive_count / expected_prompt_count
        if (
            len(decisions) != expected_prompt_count
            or _int_value(proposed_record.get("positive_count"))
            != expected_prompt_count
            or not _same_float(proposed_record.get("true_positive_rate"), proposed_tpr)
        ):
            return False
        method_attack_tprs["slm_wm_current"].append(proposed_tpr)

    mean_tprs = {
        method_id: sum(values) / len(values)
        for method_id, values in method_attack_tprs.items()
        if values
    }
    baseline_mean_tprs = {
        baseline_id: mean_tprs[baseline_id] for baseline_id in PRIMARY_BASELINE_IDS
    }
    best_baseline_id = max(
        baseline_mean_tprs,
        key=baseline_mean_tprs.get,
    )
    slm_wm_mean_fpr = sum(
        float(
            metric_expectations[("slm_wm_current", attack["attack_id"])][
                "false_positive_rate"
            ]
        )
        for attack in attack_registry
    ) / len(attack_registry)
    return (
        set(mean_tprs) == set(expected_method_ids)
        and _same_float(
            bundle.common_protocol_summary.get("slm_wm_mean_true_positive_rate"),
            mean_tprs["slm_wm_current"],
        )
        and str(bundle.common_protocol_summary.get("best_baseline_method_id", ""))
        == best_baseline_id
        and _same_float(
            bundle.common_protocol_summary.get(
                "best_baseline_mean_true_positive_rate"
            ),
            baseline_mean_tprs[best_baseline_id],
        )
        and _same_float(
            bundle.common_protocol_summary.get("slm_wm_mean_false_positive_rate"),
            slm_wm_mean_fpr,
        )
        and slm_wm_mean_fpr <= bundle.expected_target_fpr
    )


def _paired_superiority_ready(bundle: ResultClosureGateInput) -> bool:
    """复算 Prompt 聚类统计并核验阈值, 攻击, 结果与来源证据."""

    threshold_bindings = _threshold_audit_bindings(bundle)
    attack_registry = _formal_attack_registry_rows(bundle)
    if threshold_bindings is None or attack_registry is None:
        return False
    canonical_threshold_rows, threshold_map, observation_sha256_map = (
        threshold_bindings
    )
    expected_baseline_ids = set(PRIMARY_BASELINE_IDS)
    expected_attack_by_id = {row["attack_id"]: row for row in attack_registry}
    expected_attack_count = len(attack_registry)
    expected_prompt_count = bundle.expected_test_count
    expected_outcome_count = (
        len(PRIMARY_BASELINE_IDS) * expected_prompt_count * expected_attack_count
    )
    expected_threshold_rows_digest = build_stable_digest(
        list(canonical_threshold_rows)
    )
    if (
        bundle.threshold_audit_report.get("method_threshold_digest_map")
        != threshold_map
        or bundle.threshold_audit_report.get(
            "method_observation_source_sha256_map"
        )
        != observation_sha256_map
        or str(bundle.threshold_audit_report.get("threshold_audit_rows_digest", ""))
        != expected_threshold_rows_digest
    ):
        return False
    try:
        protocol_digest = build_paired_superiority_protocol_digest(
            bundle.threshold_audit_report,
            canonical_threshold_rows,
            str(bundle.threshold_audit_manifest.get("config_digest", "")),
        )
    except PairedSuperiorityError:
        return False

    outcomes = tuple(dict(row) for row in bundle.paired_outcomes)
    raw_observations = bundle.paired_observation_records_by_method
    if set(raw_observations) != {"slm_wm", *PRIMARY_BASELINE_IDS}:
        return False
    try:
        rebuilt_outcomes = tuple(
            outcome
            for baseline_id in PRIMARY_BASELINE_IDS
            for outcome in build_paired_outcomes(
                raw_observations["slm_wm"],
                raw_observations[baseline_id],
                baseline_id=baseline_id,
                proposed_method_threshold_digest=threshold_map["slm_wm"],
                baseline_method_threshold_digest=threshold_map[baseline_id],
                attack_registry_rows=attack_registry,
            )
        )
    except (PairedSuperiorityError, TypeError, ValueError):
        return False
    if outcomes != rebuilt_outcomes:
        return False
    outcome_ids = {str(row.get("baseline_id", "")) for row in outcomes}
    keys_by_baseline: dict[str, set[tuple[str, str]]] = {
        baseline_id: set() for baseline_id in PRIMARY_BASELINE_IDS
    }
    outcomes_ready = (
        len(outcomes) == expected_outcome_count
        and outcome_ids == expected_baseline_ids
    )
    for row in outcomes:
        baseline_id = str(row.get("baseline_id", ""))
        prompt_id = str(row.get("prompt_id", ""))
        attack_id = str(row.get("attack_id", ""))
        attack = expected_attack_by_id.get(attack_id)
        key = (prompt_id, attack_id)
        descriptor = (
            {
                field_name: str(row.get(field_name, ""))
                for field_name in (
                    "attack_id",
                    "attack_family",
                    "attack_name",
                    "resource_profile",
                    "attack_config_digest",
                )
            }
            if attack is not None
            else {}
        )
        if (
            baseline_id not in expected_baseline_ids
            or not prompt_id
            or attack is None
            or descriptor != attack
            or key in keys_by_baseline[baseline_id]
            or str(row.get("proposed_method_threshold_digest", ""))
            != threshold_map["slm_wm"]
            or str(row.get("baseline_method_threshold_digest", ""))
            != threshold_map[baseline_id]
            or not _paired_outcome_digest_ready(row)
        ):
            outcomes_ready = False
            break
        keys_by_baseline[baseline_id].add(key)
    if outcomes_ready:
        expected_attack_ids = set(expected_attack_by_id)
        key_sets = tuple(keys_by_baseline.values())
        prompt_sets = {
            frozenset(prompt_id for prompt_id, _attack_id in keys)
            for keys in key_sets
        }
        outcomes_ready = bool(
            len({frozenset(keys) for keys in key_sets}) == 1
            and len(prompt_sets) == 1
            and len(next(iter(prompt_sets))) == expected_prompt_count
            and all(
                {
                    attack_id
                    for candidate_prompt_id, attack_id in keys
                    if candidate_prompt_id == prompt_id
                }
                == expected_attack_ids
                for keys in key_sets
                for prompt_id in {value[0] for value in keys}
            )
        )
    if not outcomes_ready:
        return False

    normalized_rows = tuple(
        _normalized_paired_statistical_row(row)
        for row in bundle.paired_superiority_rows
    )
    if any(row is None for row in normalized_rows):
        return False
    statistical_rows = tuple(
        dict(row) for row in normalized_rows if isinstance(row, Mapping)
    )
    try:
        recomputed_rows = tuple(
            build_paired_superiority_rows(
                outcomes,
                protocol_digest=protocol_digest,
                confidence_level=DEFAULT_CONFIDENCE_LEVEL,
                bootstrap_resample_count=DEFAULT_BOOTSTRAP_RESAMPLE_COUNT,
            )
        )
        recomputed_summary = build_paired_superiority_summary(
            recomputed_rows,
            paired_outcomes=outcomes,
        )
    except (PairedSuperiorityError, TypeError, ValueError):
        return False
    if statistical_rows != recomputed_rows:
        return False

    expected_attack_registry_digest = build_stable_digest(list(attack_registry))
    summary = bundle.paired_superiority_summary
    summary_fields = (
        "primary_baseline_ids",
        "paired_superiority_row_count",
        "paired_superiority_ready_ids",
        "paired_superiority_exact_set_ready",
        "overall_paired_superiority_ready",
        "paired_superiority_rows_digest",
        "paired_test_prompt_count",
        "paired_test_prompt_id_digest",
        "supports_paper_claim",
    )
    summary_ready = bool(
        all(summary.get(field_name) == recomputed_summary.get(field_name)
            for field_name in summary_fields)
        and summary.get("paper_claim_scale") == bundle.expected_paper_claim_scale
        and _same_float(summary.get("target_fpr"), bundle.expected_target_fpr)
        and _int_value(summary.get("expected_test_count")) == expected_prompt_count
        and _int_value(summary.get("expected_attack_count")) == expected_attack_count
        and summary.get("paired_prompt_counts") == [expected_prompt_count]
        and summary.get("paired_attack_counts") == [expected_attack_count]
        and _int_value(summary.get("paired_outcome_count")) == expected_outcome_count
        and str(summary.get("paired_outcome_set_digest", ""))
        == build_stable_digest(outcomes)
        and str(summary.get("paired_superiority_protocol_digest", ""))
        == protocol_digest
        and summary.get("method_threshold_digest_map") == threshold_map
        and summary.get("method_observation_source_sha256_map")
        == observation_sha256_map
        and str(summary.get("threshold_audit_rows_digest", ""))
        == expected_threshold_rows_digest
        and str(summary.get("paired_attack_registry_digest", ""))
        == expected_attack_registry_digest
        and summary.get("paired_superiority_scale_ready") is True
        and str(summary.get("paired_test_prompt_id_digest", ""))
        == bundle.expected_test_prompt_id_digest
        and _int_value(summary.get("paired_test_prompt_count"))
        == bundle.expected_test_count
        and summary.get("claim_p_value_method") == CLAIM_P_VALUE_METHOD
        and summary.get("sharp_null_diagnostic_method")
        == SHARP_NULL_DIAGNOSTIC_METHOD
        and summary.get("bootstrap_analysis_schema")
        == BOOTSTRAP_ANALYSIS_SCHEMA
        and summary.get("bootstrap_bit_generator") == BOOTSTRAP_BIT_GENERATOR
        and summary.get("bootstrap_quantile_method")
        == BOOTSTRAP_QUANTILE_METHOD
        and _int_value(summary.get("bootstrap_resample_count"))
        == DEFAULT_BOOTSTRAP_RESAMPLE_COUNT
        and _same_float(summary.get("confidence_level"), DEFAULT_CONFIDENCE_LEVEL)
    )

    observation_path_map = summary.get("method_observation_source_path_map", {})
    manifest = bundle.paired_superiority_manifest
    manifest_inputs = manifest.get("input_paths", ())
    metadata = manifest.get("metadata", {})
    source_binding_ready = bool(
        isinstance(observation_path_map, Mapping)
        and set(observation_path_map) == set(observation_sha256_map)
        and len(set(str(path) for path in observation_path_map.values()))
        == len(observation_path_map)
        and all(
            bundle.source_file_sha256.get(str(observation_path_map[method_id]))
            == digest
            and _path_present(manifest_inputs, str(observation_path_map[method_id]))
            for method_id, digest in observation_sha256_map.items()
        )
        and all(
            _path_present(manifest_inputs, suffix)
            for suffix in (
                f"outputs/fixed_fpr_threshold_audit/{bundle.expected_paper_claim_scale}/threshold_audit_rows.csv",
                f"outputs/fixed_fpr_threshold_audit/{bundle.expected_paper_claim_scale}/threshold_audit_report.json",
                f"outputs/fixed_fpr_threshold_audit/{bundle.expected_paper_claim_scale}/manifest.local.json",
            )
        )
    )
    propagated_fields = (
        "paired_outcome_set_digest",
        "paired_superiority_rows_digest",
        "paired_superiority_protocol_digest",
        "paired_test_prompt_count",
        "paired_test_prompt_id_digest",
        "paired_attack_registry_digest",
        "method_observation_source_sha256_map",
        "threshold_audit_rows_digest",
        "claim_p_value_method",
        "sharp_null_diagnostic_method",
        "bootstrap_analysis_schema",
        "bootstrap_bit_generator",
        "bootstrap_quantile_method",
        "bootstrap_resample_count",
        "confidence_level",
        "overall_paired_superiority_ready",
    )
    common_metadata = bundle.common_protocol_manifest.get("metadata", {})
    downstream_ready = all(
        payload.get(field_name) == summary.get(field_name)
        for payload in (
            bundle.common_protocol_summary,
            bundle.common_protocol_schema,
            common_metadata if isinstance(common_metadata, Mapping) else {},
        )
        for field_name in propagated_fields
    )
    manifest_ready = bool(
        manifest.get("code_version")
        == bundle.closure_input_lock.get("common_code_version")
        and _manifest_ready(
            manifest,
            artifact_id="paired_superiority_analysis_manifest",
            required_output_suffixes=(
                f"outputs/paired_superiority_analysis/{bundle.expected_paper_claim_scale}/paired_outcomes.jsonl",
                f"outputs/paired_superiority_analysis/{bundle.expected_paper_claim_scale}/paired_superiority_table.csv",
                f"outputs/paired_superiority_analysis/{bundle.expected_paper_claim_scale}/paired_superiority_summary.json",
                f"outputs/paired_superiority_analysis/{bundle.expected_paper_claim_scale}/manifest.local.json",
            ),
        )
        and isinstance(metadata, Mapping)
        and all(metadata.get(field_name) == summary.get(field_name)
                for field_name in (*propagated_fields, "method_threshold_digest_map", "method_observation_source_path_map"))
        and str(manifest.get("config_digest", ""))
        == build_stable_digest(
            build_paired_superiority_manifest_config(summary)
        )
    )
    return bool(
        summary_ready
        and source_binding_ready
        and downstream_ready
        and manifest_ready
        and _paired_result_record_consistency_ready(
            bundle,
            outcomes,
            attack_registry,
        )
    )

def _closure_input_provenance_ready(bundle: ResultClosureGateInput) -> bool:
    """核验输入锁身份并要求最终结果与共同协议传播同一来源."""

    lock_payload = bundle.closure_input_lock
    lock_manifest = bundle.closure_input_lock_manifest
    records = lock_payload.get("closure_input_packages", ())
    if not isinstance(records, list):
        return False
    digest_payload = dict(lock_payload)
    declared_digest = str(digest_payload.pop("closure_input_lock_digest", ""))
    common_code_version = str(lock_payload.get("common_code_version", ""))
    metadata = lock_manifest.get("metadata", {})
    result_metadata = bundle.result_record_manifest.get("metadata", {})
    common_metadata = bundle.common_protocol_manifest.get("metadata", {})
    propagated_pairs = (
        (
            bundle.result_record_summary.get("closure_input_lock_digest"),
            bundle.result_record_summary.get("common_code_version"),
        ),
        (
            result_metadata.get("closure_input_lock_digest") if isinstance(result_metadata, Mapping) else None,
            result_metadata.get("common_code_version") if isinstance(result_metadata, Mapping) else None,
        ),
        (
            bundle.common_protocol_summary.get("closure_input_lock_digest"),
            bundle.common_protocol_summary.get("common_code_version"),
        ),
        (
            bundle.common_protocol_schema.get("closure_input_lock_digest"),
            bundle.common_protocol_schema.get("common_code_version"),
        ),
        (
            common_metadata.get("closure_input_lock_digest") if isinstance(common_metadata, Mapping) else None,
            common_metadata.get("common_code_version") if isinstance(common_metadata, Mapping) else None,
        ),
    )
    return (
        _is_sha256(declared_digest)
        and build_stable_digest(digest_payload) == declared_digest
        and str(lock_payload.get("paper_run_name", ""))
        == bundle.expected_paper_claim_scale
        and _same_float(lock_payload.get("target_fpr"), bundle.expected_target_fpr)
        and _int_value(lock_payload.get("closure_input_package_count")) == 10
        and len(records) == 10
        and len({str(row.get("package_family", "")) for row in records}) == 10
        and bool(common_code_version)
        and all(
            isinstance(row, Mapping)
            and str(row.get("paper_run_name", "")) == bundle.expected_paper_claim_scale
            and _same_float(row.get("target_fpr"), bundle.expected_target_fpr)
            and str(row.get("code_version", "")) == common_code_version
            and _is_sha256(row.get("package_sha256", ""))
            for row in records
        )
        and str(lock_manifest.get("artifact_id", ""))
        == f"{bundle.expected_paper_claim_scale}_closure_input_lock_manifest"
        and isinstance(metadata, Mapping)
        and _strict_bool(metadata.get("closure_input_lock_ready"))
        and str(metadata.get("closure_input_lock_digest", "")) == declared_digest
        and str(metadata.get("common_code_version", "")) == common_code_version
        and all(
            str(digest) == declared_digest and str(version) == common_code_version
            for digest, version in propagated_pairs
        )
    )


_ATTACK_FAMILY_METRIC_INTEGER_FIELDS = frozenset(
    {
        "attack_record_count",
        "supported_record_count",
        "unsupported_record_count",
        "positive_count",
        "negative_count",
    }
)
_ATTACK_FAMILY_METRIC_BOOLEAN_FIELDS = frozenset(
    {
        "fixed_fpr_upper_bound_ready",
        "supports_paper_claim",
    }
)
_ATTACK_FAMILY_METRIC_TEXT_FIELDS = frozenset(
    {
        "attack_id",
        "attack_family",
        "attack_name",
        "resource_profile",
        "attack_config_digest",
        "metric_status",
    }
)


def _csv_boolean_value(value: Any) -> bool | None:
    """严格解析正式 CSV writer 产生的布尔值."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def _normalized_attack_family_metrics(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...] | None:
    """按正式字段类型规范化持久化攻击指标, 供原始记录逐字段复算."""

    expected_fields = set(ATTACK_FAMILY_METRIC_FIELDS)
    normalized_rows: list[dict[str, Any]] = []
    for raw_row in rows:
        if not isinstance(raw_row, Mapping) or set(raw_row) != expected_fields:
            return None
        row: dict[str, Any] = {}
        for field_name in ATTACK_FAMILY_METRIC_FIELDS:
            value = raw_row[field_name]
            if field_name in _ATTACK_FAMILY_METRIC_TEXT_FIELDS:
                if not isinstance(value, str) or not value:
                    return None
                row[field_name] = value
            elif field_name in _ATTACK_FAMILY_METRIC_INTEGER_FIELDS:
                if isinstance(value, bool):
                    return None
                resolved_integer = _int_value(value)
                if resolved_integer is None:
                    return None
                row[field_name] = resolved_integer
            elif field_name in _ATTACK_FAMILY_METRIC_BOOLEAN_FIELDS:
                resolved_boolean = _csv_boolean_value(value)
                if resolved_boolean is None:
                    return None
                row[field_name] = resolved_boolean
            else:
                if isinstance(value, bool):
                    return None
                resolved_float = _float_value(value)
                if resolved_float is None:
                    return None
                row[field_name] = resolved_float
        normalized_rows.append(row)
    return tuple(normalized_rows)


def _attack_records_ready(
    bundle: ResultClosureGateInput,
    formal_configs: tuple[Any, ...],
) -> bool:
    """从攻击 detection records 重算身份、记录摘要、registry 与 manifest."""

    records = bundle.attack_detection_records
    registry = bundle.attacked_image_registry
    config_by_id = {config.attack_id: config for config in formal_configs}
    if not records or len(records) != len(registry):
        return False
    evaluation_boundary = bundle.attack_report.get("evaluation_boundary", {})
    if not isinstance(evaluation_boundary, Mapping):
        return False
    expected_registry_fields = (
        "attack_record_id",
        "run_id",
        "prompt_id",
        "split",
        "sample_role",
        "attack_id",
        "attack_family",
        "attack_name",
        "resource_profile",
        "source_image_path",
        "source_image_digest",
        "attacked_image_path",
        "attacked_image_digest",
        "attack_config_digest",
        "metric_status",
        "supports_paper_claim",
    )
    expected_registry: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    prompt_ids_by_attack_role: dict[tuple[str, str], list[str]] = {}
    threshold_digest = str(evaluation_boundary.get("threshold_digest", ""))
    for record in records:
        config = config_by_id.get(str(record.get("attack_id", "")))
        if config is None:
            return False
        expected_identity = {
            "attack_id": config.attack_id,
            "attack_family": config.attack_family,
            "attack_name": config.attack_name,
            "resource_profile": config.resource_profile,
            "attack_config_digest": attack_config_digest(config),
            "attack_parameters": config.attack_parameters,
        }
        actual_identity = {
            field_name: record.get(field_name)
            for field_name in expected_identity
        }
        record_digest = build_attack_record_digest(record)
        record_id = str(record.get("attack_record_id", ""))
        if (
            actual_identity != expected_identity
            or not str(record.get("run_id", ""))
            or not str(record.get("prompt_id", ""))
            or record.get("split") != "test"
            or record.get("sample_role")
            not in {"positive_source", "clean_negative"}
            or not isinstance(record.get("attack_performed"), bool)
            or record.get("attack_performed") is not True
            or not isinstance(record.get("formal_evidence_positive"), bool)
            or not isinstance(record.get("evidence_decision"), bool)
            or record.get("evidence_decision")
            != record.get("formal_evidence_positive")
            or record.get("formal_metric_status")
            != "measured_image_only_detection"
            or record.get("metric_status")
            != "measured_real_attacked_image_image_only_detection"
            or record.get("attacked_image_available") is not True
            or record.get("supports_paper_claim") is not True
            or record.get("requires_gpu") is not config.requires_gpu
            or str(record.get("frozen_threshold_digest", ""))
            != threshold_digest
            or not _is_sha256(record.get("source_image_digest", ""))
            or not _is_sha256(record.get("attacked_image_digest", ""))
            or str(record.get("attack_record_digest", "")) != record_digest
            or record_id != f"attack_{record_digest[:24]}"
            or record_id in seen_record_ids
        ):
            return False
        seen_record_ids.add(record_id)
        attack_role_key = (
            config.attack_id,
            str(record.get("sample_role", "")),
        )
        prompt_ids_by_attack_role.setdefault(attack_role_key, []).append(
            str(record["prompt_id"])
        )
        expected_registry.append(
            {
                field_name: record.get(field_name)
                for field_name in expected_registry_fields
            }
        )
    expected_attack_role_keys = {
        (config.attack_id, sample_role)
        for config in formal_configs
        for sample_role in ("positive_source", "clean_negative")
    }
    if set(prompt_ids_by_attack_role) != expected_attack_role_keys:
        return False
    for prompt_ids in prompt_ids_by_attack_role.values():
        unique_prompt_ids = set(prompt_ids)
        if (
            len(prompt_ids) != bundle.expected_test_count
            or len(unique_prompt_ids) != bundle.expected_test_count
            or build_stable_digest(sorted(unique_prompt_ids))
            != bundle.expected_test_prompt_id_digest
        ):
            return False
    if tuple(expected_registry) != registry:
        return False

    normalized_family_metrics = _normalized_attack_family_metrics(
        bundle.attack_family_metrics
    )
    if normalized_family_metrics is None:
        return False
    try:
        rebuilt_family_metrics = build_attack_family_metrics(
            records,
            bundle.expected_target_fpr,
            True,
        )
    except (TypeError, ValueError):
        return False
    expected_attack_ids = {config.attack_id for config in formal_configs}
    persisted_attack_ids = {
        str(row["attack_id"]) for row in normalized_family_metrics
    }
    if (
        persisted_attack_ids != expected_attack_ids
        or normalized_family_metrics != rebuilt_family_metrics
    ):
        return False

    manifest_inputs = bundle.attack_manifest.get("input_paths", ())
    if not isinstance(manifest_inputs, list | tuple):
        return False
    input_records_path = bundle.attack_report.get("input_records_path")
    if (
        _declared_source_digest(
            bundle.source_file_sha256,
            input_records_path,
        )
        is None
        or not _path_present(manifest_inputs, str(input_records_path))
        or not all(
            _declared_source_digest(bundle.source_file_sha256, path) is not None
            for path in manifest_inputs
        )
    ):
        return False
    expected_manifest_config = build_attack_matrix_manifest_config(
        paper_run_name=bundle.expected_paper_claim_scale,
        evaluation_boundary=evaluation_boundary,
        attack_configs=formal_configs,
        attack_records=records,
    )
    return (
        str(bundle.attack_manifest.get("config_digest", ""))
        == build_stable_digest(expected_manifest_config)
        and str(bundle.attack_manifest.get("code_version", ""))
        == str(bundle.closure_input_lock.get("common_code_version", ""))
    )


def _attack_ready(bundle: ResultClosureGateInput) -> bool:
    """核验真实攻击矩阵覆盖、文件证据链和 GPU 攻击均已闭合。"""

    scale = bundle.expected_paper_claim_scale
    attack_registry = _formal_attack_registry_rows(bundle)
    formal_configs = tuple(
        config
        for config in default_attack_configs()
        if config.enabled and config.resource_profile in {"full_main", "full_extra"}
    )
    expected_gpu_attack_count = sum(
        config.requires_gpu and config.resource_profile == "full_extra"
        for config in formal_configs
    )
    ready_fields = (
        "real_attacked_image_closed_loop_ready",
        "formal_attack_detection_ready",
        "attack_metrics_ready",
        "attack_record_coverage_ready",
        "real_gpu_attack_validation_ready",
        "full_method_claim_ready",
        "supports_paper_claim",
    )
    required_gpu_count = _int_value(bundle.attack_report.get("required_real_gpu_attack_count"))
    measured_gpu_count = _int_value(bundle.attack_report.get("measured_real_gpu_attack_count"))
    return (
        _all_true(bundle.attack_report, ready_fields)
        and attack_registry is not None
        and len(attack_registry) == len(formal_configs)
        and _int_value(bundle.attack_report.get("attack_config_count"))
        == len(formal_configs)
        and _int_value(bundle.attack_report.get("attack_family_count"))
        == len({config.attack_family for config in formal_configs})
        and set(bundle.attack_report.get("resource_profiles", ()))
        == {config.resource_profile for config in formal_configs}
        and _attack_record_counts_ready(bundle)
        and _attack_records_ready(bundle, formal_configs)
        and required_gpu_count is not None
        and required_gpu_count == expected_gpu_attack_count
        and measured_gpu_count is not None
        and measured_gpu_count >= required_gpu_count
        and _int_value(bundle.attack_report.get("gpu_attack_real_measurement_missing_count")) == 0
        and list(bundle.attack_report.get("missing_attack_ids", ())) == []
        and list(bundle.attack_report.get("unexpected_attack_ids", ())) == []
        and set(bundle.attack_report.get("actual_attack_ids", ()))
        == set(bundle.attack_report.get("expected_attack_ids", ()))
        and bool(bundle.attack_report.get("expected_attack_ids"))
        and _manifest_ready(
            bundle.attack_manifest,
            artifact_id=f"{bundle.expected_paper_claim_scale}_attack_matrix_manifest",
            required_output_suffixes=(
                f"outputs/attack_matrix/{scale}/attack_manifest.json",
                f"outputs/attack_matrix/{scale}/attack_detection_records.jsonl",
                f"outputs/attack_matrix/{scale}/attacked_image_registry.jsonl",
                f"outputs/attack_matrix/{scale}/attack_family_metrics.csv",
                f"outputs/attack_matrix/{scale}/manifest.local.json",
            ),
        )
        and _metadata_matches(
            bundle.attack_manifest,
            bundle.attack_report,
            ("full_method_claim_ready", "supports_paper_claim"),
        )
        and str(bundle.attack_manifest.get("metadata", {}).get("protocol_decision", "")) == "pass"
    )


def _threshold_audit_ready(bundle: ResultClosureGateInput) -> bool:
    """核验五方法统一阈值审计的逐方法结论和 manifest。"""

    expected_ids = {"slm_wm", "tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark"}
    scale = bundle.expected_paper_claim_scale
    bindings = _threshold_audit_bindings(bundle)
    if bindings is None:
        return False
    canonical_rows, threshold_map, observation_map = bindings
    actual_ids = [str(row.get("method_id", "")) for row in canonical_rows]
    rows_digest = build_stable_digest(list(canonical_rows))
    metadata = bundle.threshold_audit_manifest.get("metadata", {})
    expected_manifest_config_digest = build_stable_digest(
        build_fixed_fpr_threshold_manifest_config(bundle.threshold_audit_report)
    )
    row_ready_fields = (
        "protocol_target_ready",
        "protocol_value_ready",
        "detection_decision_ready",
        "split_count_ready",
        "fixed_fpr_threshold_ready",
    )
    return (
        _all_true(
            bundle.threshold_audit_report,
            (
                "method_identity_ready",
                "all_method_thresholds_ready",
                "threshold_observation_binding_ready",
                "fixed_fpr_threshold_audit_ready",
                "supports_paper_claim",
            ),
        )
        and len(actual_ids) == len(expected_ids)
        and len(set(actual_ids)) == len(actual_ids)
        and set(actual_ids) == expected_ids
        and bundle.threshold_audit_report.get("method_threshold_digest_map")
        == threshold_map
        and bundle.threshold_audit_report.get(
            "method_observation_source_sha256_map"
        )
        == observation_map
        and str(bundle.threshold_audit_report.get("threshold_audit_rows_digest", ""))
        == rows_digest
        and all(_all_true(row, row_ready_fields) for row in canonical_rows)
        and str(bundle.threshold_audit_manifest.get("config_digest", ""))
        == expected_manifest_config_digest
        and _manifest_ready(
            bundle.threshold_audit_manifest,
            artifact_id="fixed_fpr_threshold_audit_manifest",
            required_output_suffixes=(
                f"outputs/fixed_fpr_threshold_audit/{scale}/threshold_audit_rows.csv",
                f"outputs/fixed_fpr_threshold_audit/{scale}/threshold_audit_report.json",
                f"outputs/fixed_fpr_threshold_audit/{scale}/manifest.local.json",
            ),
        )
        and _metadata_matches(
            bundle.threshold_audit_manifest,
            bundle.threshold_audit_report,
            (
                "paper_claim_scale",
                "target_fpr",
                "method_threshold_digest_map",
                "method_observation_source_sha256_map",
                "threshold_audit_rows_digest",
                "threshold_observation_binding_ready",
                "fixed_fpr_threshold_audit_ready",
                "supports_paper_claim",
            ),
        )
        and isinstance(metadata, Mapping)
        and metadata.get("method_threshold_digest_map") == threshold_map
        and metadata.get("method_observation_source_sha256_map")
        == observation_map
        and str(metadata.get("threshold_audit_rows_digest", "")) == rows_digest
    )


def _official_reference_fidelity_ready(bundle: ResultClosureGateInput) -> bool:
    """核验三个官方原始环境复现仅作为补充方法忠实度证据."""

    expected_ids = {"tree_ring", "gaussian_shading", "shallow_diffuse"}
    records = tuple(bundle.official_reference_fidelity_records)
    summary = bundle.official_reference_fidelity_summary
    manifest = bundle.official_reference_fidelity_manifest
    metadata = manifest.get("metadata", {})
    record_ids = [str(row.get("baseline_id", "")) for row in records]
    records_digest = build_stable_digest(list(records))
    common_code_version = str(summary.get("common_code_version", ""))
    record_ready_fields = (
        "official_reference_ready",
        "reference_import_ready",
        "records_nonempty_ready",
        "records_baseline_identity_ready",
        "validation_zero_rejection_ready",
        "run_manifest_ready",
        "package_input_exact_set_ready",
        "package_input_digests_ready",
        "package_governance_semantics_ready",
        "source_code_version_consistent_ready",
        "supplemental_method_fidelity_evidence_ready",
        "official_reference_fidelity_evidence_ready",
    )
    records_ready = (
        len(record_ids) == len(expected_ids)
        and len(set(record_ids)) == len(record_ids)
        and set(record_ids) == expected_ids
        and all(
            _official_reference_record_digest_ready(row)
            and row.get("paper_claim_scale") == bundle.expected_paper_claim_scale
            and _same_float(row.get("target_fpr"), bundle.expected_target_fpr)
            and row.get("run_decision") == "pass"
            and row.get("supplemental_table_role")
            == "supplemental_method_fidelity_reference"
            and _all_true(row, record_ready_fields)
            and row.get("main_table_eligible") is False
            and row.get("supports_main_table_superiority_claim") is False
            and row.get("supports_paper_claim") is not True
            and str(row.get("code_version", "")) == common_code_version
            and _declared_source_digests_ready(
                row.get("official_reference_source_paths"),
                row.get("official_reference_source_artifact_digests"),
                bundle.source_file_sha256,
            )
            for row in records
        )
    )
    scale = bundle.expected_paper_claim_scale
    summary_ready = (
        summary.get("expected_official_reference_baseline_ids")
        == ["tree_ring", "gaussian_shading", "shallow_diffuse"]
        and set(summary.get("actual_official_reference_baseline_ids", ()))
        == expected_ids
        and summary.get("missing_official_reference_baseline_ids") == []
        and summary.get("unexpected_official_reference_baseline_ids") == []
        and summary.get("duplicate_official_reference_baseline_ids") == []
        and _all_true(
            summary,
            (
                "official_reference_exact_set_ready",
                "common_code_version_ready",
                "supplemental_method_fidelity_evidence_ready",
                "official_reference_fidelity_evidence_ready",
            ),
        )
        and _int_value(summary.get("official_reference_fidelity_record_count"))
        == len(expected_ids)
        and _int_value(summary.get("official_reference_fidelity_ready_count"))
        == len(expected_ids)
        and str(summary.get("official_reference_fidelity_evidence_digest", ""))
        == records_digest
        and summary.get("main_table_eligible") is False
        and summary.get("supports_main_table_superiority_claim") is False
        and common_code_version
        == str(bundle.closure_input_lock.get("common_code_version", ""))
    )
    return (
        records_ready
        and summary_ready
        and manifest.get("code_version") == common_code_version
        and _manifest_ready(
            manifest,
            artifact_id="official_reference_fidelity_evidence_manifest",
            required_output_suffixes=(
                f"outputs/official_reference_fidelity_evidence/{scale}/official_reference_fidelity_evidence_records.jsonl",
                f"outputs/official_reference_fidelity_evidence/{scale}/official_reference_fidelity_evidence_summary.json",
                f"outputs/official_reference_fidelity_evidence/{scale}/manifest.local.json",
            ),
        )
        and isinstance(metadata, Mapping)
        and all(
            metadata.get(field_name) == summary.get(field_name)
            for field_name in (
                "paper_claim_scale",
                "target_fpr",
                "common_code_version",
                "official_reference_exact_set_ready",
                "official_reference_fidelity_evidence_digest",
                "official_reference_fidelity_evidence_ready",
                "supports_main_table_superiority_claim",
            )
        )
    )


def _baseline_ready(bundle: ResultClosureGateInput) -> bool:
    """核验外部 baseline 的正式导入、模板覆盖和证据路径均通过。"""

    scale = bundle.expected_paper_claim_scale
    ready_fields = (
        "comparison_protocol_ready",
        "comparison_table_supports_paper_claim",
        "primary_baseline_formal_ready",
        "primary_baseline_results_ready",
        "primary_baseline_formal_template_coverage_ready",
        "primary_baseline_formal_evidence_collection_ready",
        "formal_import_validation_ready",
        "formal_evidence_path_resolution_ready",
        "baseline_source_registry_ready",
        "supports_paper_claim",
    )
    zero_fields = (
        "rejected_formal_import_count",
        "formal_import_issue_count",
        "missing_candidate_template_count",
        "missing_formal_template_count",
        "unexpected_candidate_record_count",
        "unexpected_accepted_record_count",
        "duplicate_candidate_template_count",
        "duplicate_accepted_template_count",
        "missing_formal_evidence_collection_task_count",
        "missing_formal_evidence_path_count",
    )
    return (
        _all_true(bundle.baseline_report, ready_fields)
        and _all_zero(bundle.baseline_report, zero_fields)
        and (_int_value(bundle.baseline_report.get("accepted_formal_import_count")) or 0) > 0
        and _manifest_ready(
            bundle.baseline_manifest,
            artifact_id="external_baseline_comparison_manifest",
            required_output_suffixes=(
                f"outputs/external_baseline_comparison/{scale}/baseline_runtime_report.json",
                f"outputs/external_baseline_comparison/{scale}/manifest.local.json",
            ),
        )
        and _metadata_matches(
            bundle.baseline_manifest,
            bundle.baseline_report,
            ("baseline_results_ready", "comparison_table_supports_paper_claim", "supports_paper_claim"),
        )
    )


def _primary_baseline_evidence_ready(bundle: ResultClosureGateInput) -> bool:
    """核验四个主表 baseline 均有独立、完整且无阻断的正式证据。"""

    summary = bundle.primary_baseline_evidence_summary
    scale = bundle.expected_paper_claim_scale
    expected_ids = {"tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark"}
    records = tuple(bundle.primary_baseline_evidence_records)
    record_ids = [str(row.get("baseline_id", "")) for row in records]
    canonical_records = sorted(records, key=lambda row: str(row.get("baseline_id", "")))
    records_digest = build_stable_digest(canonical_records)
    manifest_metadata = bundle.primary_baseline_evidence_manifest.get("metadata", {})
    adapter_ids = [str(value) for value in summary.get("adapter_run_ready_ids", ())]
    formal_ids = [str(value) for value in summary.get("formal_result_ready_ids", ())]
    input_ids = [str(value) for value in summary.get("input_baseline_ids", ())]
    identities_ready = all(
        len(values) == len(expected_ids)
        and len(set(values)) == len(values)
        and set(values) == expected_ids
        for values in (adapter_ids, formal_ids, input_ids)
    )
    record_identity_ready = (
        len(record_ids) == len(expected_ids)
        and len(set(record_ids)) == len(record_ids)
        and set(record_ids) == expected_ids
        and all(
            str(row.get("comparison_group", "")) == "primary"
            and str(row.get("source_status", "")) == "downloaded"
            and bool(str(row.get("source_dir", "")))
            and 7 <= len(str(row.get("official_repository_commit", ""))) <= 40
            and all(
                character in string.hexdigits
                for character in str(row.get("official_repository_commit", ""))
            )
            and bool(str(row.get("adapter_status", "")))
            and _strict_bool(row.get("adapter_run_ready"))
            and (_int_value(row.get("adapter_run_observation_count")) or 0) > 0
            and _strict_bool(row.get("method_faithful_adapter_ready"))
            and _strict_bool(row.get("paper_run_prompt_protocol_ready"))
            and _strict_bool(row.get("fixed_fpr_baseline_calibration_ready"))
            and _strict_bool(row.get("attack_matrix_baseline_detection_ready"))
            and _strict_bool(row.get("formal_evidence_paths_ready"))
            and isinstance(row.get("formal_evidence_paths"), list)
            and bool(row.get("formal_evidence_paths"))
            and all(bool(str(path)) for path in row.get("formal_evidence_paths", ()))
            and _strict_bool(row.get("formal_result_ready"))
            and row.get("blocking_reasons") == []
            and _primary_baseline_record_digest_ready(row)
            for row in records
        )
    )
    return (
        str(summary.get("paper_claim_scale", "")) == scale
        and _same_float(summary.get("target_fpr"), bundle.expected_target_fpr)
        and _int_value(summary.get("primary_baseline_count")) == len(expected_ids)
        and _int_value(summary.get("adapter_run_ready_count")) == len(expected_ids)
        and _int_value(summary.get("formal_result_ready_count")) == len(expected_ids)
        and _strict_bool(summary.get("primary_baseline_formal_ready"))
        and identities_ready
        and isinstance(summary.get("blocking_reasons"), list)
        and summary.get("blocking_reasons") == []
        and (_int_value(summary.get("input_observation_count")) or 0) > 0
        and (_int_value(summary.get("input_command_result_count")) or 0) == len(expected_ids)
        and _is_sha256(summary.get("t2smark_formal_evidence_digest", ""))
        and record_identity_ready
        and str(summary.get("primary_baseline_evidence_records_digest", ""))
        == records_digest
        and isinstance(manifest_metadata, Mapping)
        and str(
            manifest_metadata.get("primary_baseline_evidence_records_digest", "")
        )
        == records_digest
        and _manifest_ready(
            bundle.primary_baseline_evidence_manifest,
            artifact_id="primary_baseline_evidence_manifest",
            required_output_suffixes=(
                f"outputs/primary_baseline_evidence/{scale}/primary_baseline_evidence_records.jsonl",
                f"outputs/primary_baseline_evidence/{scale}/primary_baseline_evidence_summary.json",
                f"outputs/primary_baseline_evidence/{scale}/manifest.local.json",
            ),
        )
        and _metadata_matches(
            bundle.primary_baseline_evidence_manifest,
            summary,
            (
                "paper_claim_scale",
                "target_fpr",
                "primary_baseline_formal_ready",
                "primary_baseline_evidence_records_digest",
            ),
        )
    )


def _normalized_declared_path(value: Any) -> str:
    """统一比较 manifest 和记录中的相对或绝对路径文本."""

    return str(value).replace("\\", "/").removeprefix("./")


def _declared_source_digest(
    source_file_sha256: Mapping[str, str],
    declared_path: Any,
) -> str | None:
    """按规范化路径从门禁即时 SHA 映射提取唯一摘要."""

    normalized = _normalized_declared_path(declared_path)
    matches = {
        str(digest)
        for path, digest in source_file_sha256.items()
        if _normalized_declared_path(path) == normalized
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _result_record_sources_ready(bundle: ResultClosureGateInput) -> bool:
    """核验 result records 的来源摘要和证据路径均被即时读取."""

    manifest_inputs = bundle.result_record_manifest.get("input_paths", ())
    if not isinstance(manifest_inputs, list | tuple):
        return False
    normalized_manifest_inputs = {
        _normalized_declared_path(path) for path in manifest_inputs
    }
    for record in bundle.result_records:
        source_path = record.get("baseline_result_source")
        source_digest = _declared_source_digest(
            bundle.source_file_sha256,
            source_path,
        )
        evidence_paths = record.get("evidence_paths")
        if (
            source_digest is None
            or source_digest != str(record.get("baseline_result_source_digest", ""))
            or _normalized_declared_path(source_path)
            not in normalized_manifest_inputs
            or not isinstance(evidence_paths, list | tuple)
            or not evidence_paths
        ):
            return False
        for evidence_path in evidence_paths:
            if (
                _declared_source_digest(
                    bundle.source_file_sha256,
                    evidence_path,
                )
                is None
                or _normalized_declared_path(evidence_path)
                not in normalized_manifest_inputs
            ):
                return False
    return all(
        _declared_source_digest(bundle.source_file_sha256, path) is not None
        for path in manifest_inputs
    )


def _expected_result_record_coverage(
    bundle: ResultClosureGateInput,
) -> tuple[dict[str, Any], ...] | None:
    """按规范 method x attack 顺序重建完整模板覆盖表."""

    config = _expected_fixed_fpr_config(bundle)
    if config is None:
        return None
    attack_rows = build_pilot_paper_attack_matrix_rows(
        default_attack_configs(),
        config,
    )
    rows = []
    for method_id in ("slm_wm_current", *PRIMARY_BASELINE_IDS):
        for attack in attack_rows:
            rows.append(
                {
                    "method_id": method_id,
                    "attack_id": str(attack["attack_id"]),
                    "attack_family": str(attack["attack_family"]),
                    "attack_name": str(attack["attack_name"]),
                    "resource_profile": str(attack["resource_profile"]),
                    "attack_config_digest": str(
                        attack["attack_config_digest"]
                    ),
                    "template_covered": True,
                    "supports_paper_claim": False,
                }
            )
    return tuple(rows)


def _normalized_result_record_coverage(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...] | None:
    """把 CSV 覆盖行恢复为 manifest 构造前的稳定类型."""

    fields = (
        "method_id",
        "attack_id",
        "attack_family",
        "attack_name",
        "resource_profile",
        "attack_config_digest",
        "template_covered",
        "supports_paper_claim",
    )
    normalized = []
    for row in rows:
        if set(row) != set(fields):
            return None
        normalized.append(
            {
                "method_id": str(row.get("method_id", "")),
                "attack_id": str(row.get("attack_id", "")),
                "attack_family": str(row.get("attack_family", "")),
                "attack_name": str(row.get("attack_name", "")),
                "resource_profile": str(row.get("resource_profile", "")),
                "attack_config_digest": str(
                    row.get("attack_config_digest", "")
                ),
                "template_covered": _strict_bool(
                    row.get("template_covered")
                ),
                "supports_paper_claim": _strict_bool(
                    row.get("supports_paper_claim")
                ),
            }
        )
    return tuple(normalized)


def _result_record_manifest_config_ready(
    bundle: ResultClosureGateInput,
) -> bool:
    """精确重建 result records manifest 配置摘要并绑定代码版本."""

    require_existing_evidence = bundle.result_record_summary.get(
        "require_existing_evidence"
    )
    if not isinstance(require_existing_evidence, bool):
        return False
    normalized_coverage = _normalized_result_record_coverage(
        bundle.result_record_template_coverage
    )
    if normalized_coverage is None:
        return False
    expected_config = build_pilot_paper_result_records_manifest_config(
        result_records=bundle.result_records,
        method_threshold_digest_map=(
            bundle.result_record_summary.get("method_threshold_digest_map", {})
        ),
        closure_input_lock_digest=str(
            bundle.closure_input_lock.get("closure_input_lock_digest", "")
        ),
        common_code_version=str(
            bundle.closure_input_lock.get("common_code_version", "")
        ),
        validation_report=bundle.result_record_validation_report,
        template_coverage_rows=normalized_coverage,
        summary=bundle.result_record_summary,
        require_existing_evidence=require_existing_evidence,
    )
    common_code_version = str(
        bundle.closure_input_lock.get("common_code_version", "")
    )
    return (
        str(bundle.result_record_manifest.get("config_digest", ""))
        == build_stable_digest(expected_config)
        and str(bundle.result_record_manifest.get("code_version", ""))
        == common_code_version
    )


def _result_records_ready(bundle: ResultClosureGateInput) -> bool:
    """核验正式 records 完整覆盖模板且每条记录正文摘要有效。"""

    scale = bundle.expected_paper_claim_scale
    summary_ready_fields = (
        "pilot_paper_template_coverage_ready",
        "pilot_paper_result_import_ready",
        "pilot_paper_claim_record_ready",
        "supports_paper_claim",
    )
    expected_schema = _expected_common_protocol_schema(bundle)
    expected_coverage = _expected_result_record_coverage(bundle)
    normalized_coverage = _normalized_result_record_coverage(
        bundle.result_record_template_coverage
    )
    if (
        expected_schema is None
        or expected_coverage is None
        or normalized_coverage is None
    ):
        return False
    try:
        validation_report = validate_pilot_paper_result_import_rows(
            bundle.result_records,
            expected_schema,
            require_existing_evidence=False,
        )
    except (TypeError, ValueError):
        return False
    expected_record_count = len(expected_coverage)
    accepted_records = validation_report.get("accepted_records")
    validation_report_ready = (
        validation_report.get("pilot_paper_result_import_ready") is True
        and validation_report.get("pilot_paper_claim_record_ready") is True
        and validation_report.get("supports_paper_claim") is True
        and _int_value(validation_report.get("input_record_count"))
        == expected_record_count
        and _int_value(validation_report.get("accepted_pilot_paper_import_count"))
        == expected_record_count
        and _int_value(
            validation_report.get("accepted_pilot_paper_claim_record_count")
        )
        == expected_record_count
        and _int_value(validation_report.get("rejected_pilot_paper_import_count"))
        == 0
        and _int_value(validation_report.get("pilot_paper_import_issue_count"))
        == 0
        and isinstance(accepted_records, list)
        and len(accepted_records) == expected_record_count
        and validation_report.get("issues") == []
    )
    return (
        bool(bundle.result_records)
        and len(bundle.result_records) == expected_record_count
        and bundle.common_protocol_schema == expected_schema
        and validation_report_ready
        and validation_report == bundle.result_record_validation_report
        and normalized_coverage == expected_coverage
        and _all_true(bundle.result_record_summary, summary_ready_fields)
        and _int_value(bundle.result_record_summary.get("pilot_paper_result_record_count")) == len(bundle.result_records)
        and _int_value(bundle.result_record_summary.get("pilot_paper_template_record_count")) == len(bundle.result_records)
        and _int_value(bundle.result_record_summary.get("pilot_paper_template_covered_count")) == len(bundle.result_records)
        and _int_value(bundle.result_record_summary.get("accepted_pilot_paper_import_count")) == len(bundle.result_records)
        and _int_value(bundle.result_record_summary.get("accepted_pilot_paper_claim_record_count"))
        == len(bundle.result_records)
        and _int_value(bundle.result_record_summary.get("pilot_paper_template_missing_count")) == 0
        and all(
            _all_true(row, ("strict_formal_result_ready", "supports_paper_claim"))
            and _record_digest_ready(row)
            for row in bundle.result_records
        )
        and _result_record_sources_ready(bundle)
        and _result_record_manifest_config_ready(bundle)
        and _manifest_ready(
            bundle.result_record_manifest,
            artifact_id="pilot_paper_fixed_fpr_result_records_manifest",
            required_output_suffixes=(
                f"outputs/pilot_paper_fixed_fpr_results/{scale}/pilot_paper_result_records.jsonl",
                f"outputs/pilot_paper_fixed_fpr_results/{scale}/pilot_paper_result_import_validation_report.json",
                f"outputs/pilot_paper_fixed_fpr_results/{scale}/pilot_paper_result_template_coverage.csv",
                f"outputs/pilot_paper_fixed_fpr_results/{scale}/pilot_paper_result_record_summary.json",
                f"outputs/pilot_paper_fixed_fpr_results/{scale}/manifest.local.json",
            ),
        )
        and _metadata_matches(
            bundle.result_record_manifest,
            bundle.result_record_summary,
            ("paper_claim_scale", "pilot_paper_result_import_ready", "supports_paper_claim"),
        )
    )


def _common_protocol_ready(bundle: ResultClosureGateInput) -> bool:
    """核验共同协议导入、完整模板与优势性门禁全部为真。"""

    scale = bundle.expected_paper_claim_scale
    ready_fields = (
        "paper_run_allows_paper_claim",
        "strict_formal_evidence_required",
        "pilot_paper_common_protocol_ready",
        "paper_run_workflow_validation_ready",
        "pilot_paper_prompt_split_ready",
        "paper_prompt_split_ready",
        "pilot_paper_result_import_ready",
        "pilot_paper_claim_record_ready",
        "paper_run_result_import_coverage_ready",
        "paper_run_template_registry_unique",
        "pilot_paper_evidence_coverage_ready",
        "point_estimate_effect_direction_ready",
        "paired_superiority_ready",
        "paired_superiority_exact_set_ready",
        "overall_paired_superiority_ready",
        "pilot_paper_effectiveness_gate_ready",
        "slm_wm_fixed_fpr_boundary_ready",
        "paper_run_claim_ready",
        "paper_run_supports_superiority_claim",
        "paper_claim_ready",
    )
    zero_fields = (
        "paper_run_result_missing_template_count",
        "paper_run_result_unexpected_template_count",
        "paper_run_result_duplicate_template_count",
    )
    return (
        _all_true(bundle.common_protocol_summary, ready_fields)
        and _all_zero(bundle.common_protocol_summary, zero_fields)
        and str(
            bundle.common_protocol_summary.get(
                "calibration_prompt_id_digest",
                "",
            )
        )
        == bundle.expected_calibration_prompt_id_digest
        and str(bundle.common_protocol_summary.get("test_prompt_id_digest", ""))
        == bundle.expected_test_prompt_id_digest
        and _int_value(bundle.common_protocol_summary.get("paper_prompt_count")) == bundle.expected_prompt_count
        and _int_value(bundle.common_protocol_summary.get("pilot_paper_import_template_count"))
        == len(bundle.result_records)
        and _int_value(bundle.common_protocol_summary.get("accepted_pilot_paper_import_count"))
        == len(bundle.result_records)
        and _int_value(bundle.common_protocol_summary.get("accepted_pilot_paper_claim_record_count"))
        == len(bundle.result_records)
        and _manifest_ready(
            bundle.common_protocol_manifest,
            artifact_id="pilot_paper_fixed_fpr_common_protocol_manifest",
            required_output_suffixes=(
                f"outputs/pilot_paper_fixed_fpr_common_protocol/{scale}/pilot_paper_result_import_schema.json",
                f"outputs/pilot_paper_fixed_fpr_common_protocol/{scale}/pilot_paper_common_protocol_summary.json",
                f"outputs/pilot_paper_fixed_fpr_common_protocol/{scale}/manifest.local.json",
            ),
        )
        and _metadata_matches(
            bundle.common_protocol_manifest,
            bundle.common_protocol_summary,
            ("paper_claim_scale", "paper_run_claim_ready", "paper_claim_ready"),
        )
    )


def _result_analysis_ready(bundle: ResultClosureGateInput) -> bool:
    """核验完整 CI 与逐攻击披露, 不把所有攻击显著胜出设为闭合前提。"""

    scale = bundle.expected_paper_claim_scale
    result_record_count = len(bundle.result_records)
    attack_registry = _formal_attack_registry_rows(bundle)
    expected_result_count = _int_value(bundle.result_analysis_summary.get("expected_result_record_count"))
    expected_superiority_count = _int_value(
        bundle.result_analysis_summary.get("expected_superiority_row_count")
    )
    superiority_row_count = _int_value(
        bundle.result_analysis_summary.get("per_attack_superiority_row_count")
    )
    superiority_ready_count = _int_value(
        bundle.result_analysis_summary.get("superiority_claim_ready_count")
    )
    analysis_inputs = bundle.result_analysis_manifest.get("input_paths", ())
    return (
        _all_true(
            bundle.result_analysis_summary,
            (
                "failure_case_figure_ready",
                "result_template_coverage_ready",
                "per_attack_ci_coverage_ready",
                "per_attack_superiority_evaluation_ready",
                "paired_superiority_ready",
                "overall_paired_superiority_ready",
                "supports_paper_claim",
            ),
        )
        and result_record_count > 0
        and attack_registry is not None
        and result_record_count
        == len(attack_registry) * (len(PRIMARY_BASELINE_IDS) + 1)
        and expected_result_count == result_record_count
        and _int_value(bundle.result_analysis_summary.get("result_record_count")) == result_record_count
        and _int_value(bundle.result_analysis_summary.get("actual_result_record_count")) == result_record_count
        and _int_value(bundle.result_analysis_summary.get("unique_result_record_key_count")) == result_record_count
        and _int_value(bundle.result_analysis_summary.get("confidence_interval_row_count")) == result_record_count
        and expected_superiority_count is not None
        and expected_superiority_count == len(attack_registry)
        and superiority_row_count == expected_superiority_count
        and superiority_ready_count is not None
        and 0 <= superiority_ready_count <= superiority_row_count
        and _int_value(bundle.result_analysis_summary.get("paired_superiority_row_count"))
        == 4
        and all(
            str(bundle.result_analysis_summary.get(field_name, ""))
            == str(bundle.paired_superiority_summary.get(field_name, ""))
            for field_name in (
                "paired_outcome_set_digest",
                "paired_superiority_rows_digest",
                "paired_superiority_protocol_digest",
                "paired_test_prompt_count",
                "paired_test_prompt_id_digest",
                "paired_attack_registry_digest",
                "method_observation_source_sha256_map",
                "threshold_audit_rows_digest",
                "claim_p_value_method",
                "sharp_null_diagnostic_method",
                "bootstrap_analysis_schema",
                "bootstrap_bit_generator",
                "bootstrap_quantile_method",
                "bootstrap_resample_count",
                "confidence_level",
                "overall_paired_superiority_ready",
            )
        )
        and _int_value(bundle.result_analysis_summary.get("duplicate_result_record_count")) == 0
        and _int_value(bundle.result_analysis_summary.get("missing_result_record_count")) == 0
        and _int_value(bundle.result_analysis_summary.get("unexpected_result_record_count")) == 0
        and all(
            _path_present(analysis_inputs, suffix)
            for suffix in (
                f"outputs/paired_superiority_analysis/{scale}/paired_superiority_summary.json",
                f"outputs/paired_superiority_analysis/{scale}/paired_superiority_table.csv",
                f"outputs/paired_superiority_analysis/{scale}/manifest.local.json",
            )
        )
        and _manifest_ready(
            bundle.result_analysis_manifest,
            artifact_id="pilot_paper_result_analysis_manifest",
            required_output_suffixes=(
                f"outputs/pilot_paper_result_analysis/{scale}/confidence_interval_table.csv",
                f"outputs/pilot_paper_result_analysis/{scale}/per_attack_superiority_table.csv",
                f"outputs/pilot_paper_result_analysis/{scale}/failure_case_records.jsonl",
                f"outputs/pilot_paper_result_analysis/{scale}/failure_case_figure.svg",
                f"outputs/pilot_paper_result_analysis/{scale}/result_analysis_summary.json",
                f"outputs/pilot_paper_result_analysis/{scale}/manifest.local.json",
            ),
        )
        and _metadata_matches(
            bundle.result_analysis_manifest,
            bundle.result_analysis_summary,
            (
                "paper_claim_scale",
                "target_fpr",
                "result_template_coverage_ready",
                "per_attack_ci_coverage_ready",
                "per_attack_superiority_evaluation_ready",
                "universal_per_attack_superiority_claim_ready",
                "paired_superiority_ready",
                "overall_paired_superiority_ready",
                "paired_superiority_rows_digest",
                "paired_test_prompt_count",
                "paired_test_prompt_id_digest",
                "paired_attack_registry_digest",
                "method_observation_source_sha256_map",
                "threshold_audit_rows_digest",
                "claim_p_value_method",
                "sharp_null_diagnostic_method",
                "bootstrap_analysis_schema",
                "bootstrap_bit_generator",
                "bootstrap_quantile_method",
                "bootstrap_resample_count",
                "confidence_level",
                "supports_paper_claim",
            ),
        )
    )


def _ablation_ready(bundle: ResultClosureGateInput) -> bool:
    """核验正式消融由真实重运行产生并通过逐配置校准门禁。"""

    scale = bundle.expected_paper_claim_scale
    record_count = _int_value(bundle.ablation_summary.get("record_count"))
    ablation_count = _int_value(bundle.ablation_summary.get("ablation_count"))
    expected_ids = list(FORMAL_RUNTIME_RERUN_ABLATION_IDS)
    manifest_metadata = bundle.ablation_manifest.get("metadata", {})
    return (
        _all_true(bundle.ablation_summary, ("ablation_claim_gate_ready", "supports_paper_claim"))
        and str(bundle.ablation_summary.get("protocol_decision", "")) == "pass"
        and bundle.ablation_summary.get("expected_ablation_ids") == expected_ids
        and bundle.ablation_summary.get("actual_ablation_ids") == expected_ids
        and bundle.ablation_summary.get("ablation_spec_digest")
        == FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST
        and _strict_bool(bundle.ablation_summary.get("ablation_exact_set_ready"))
        and record_count is not None
        and record_count > 0
        and _int_value(bundle.ablation_summary.get("prompt_count")) == bundle.expected_prompt_count
        and ablation_count == len(expected_ids)
        and record_count == bundle.expected_prompt_count * ablation_count
        and _int_value(bundle.ablation_summary.get("per_ablation_calibration_count")) == ablation_count
        and _int_value(bundle.ablation_summary.get("generation_rerun_count")) == record_count
        and _int_value(bundle.ablation_summary.get("attack_and_detection_rerun_count")) == record_count
        and _manifest_ready(
            bundle.ablation_manifest,
            artifact_id="formal_mechanism_ablation_manifest",
            required_output_suffixes=(
                f"outputs/formal_mechanism_ablation/{scale}/per_ablation_frozen_protocols.json",
                f"outputs/formal_mechanism_ablation/{scale}/mechanism_ablation_metrics.csv",
                f"outputs/formal_mechanism_ablation/{scale}/ablation_claim_summary.json",
                f"outputs/formal_mechanism_ablation/{scale}/manifest.local.json",
            ),
        )
        and _metadata_matches(
            bundle.ablation_manifest,
            bundle.ablation_summary,
            (
                "protocol_decision",
                "expected_ablation_ids",
                "actual_ablation_ids",
                "ablation_spec_digest",
                "ablation_exact_set_ready",
                "supports_paper_claim",
            ),
        )
        and isinstance(manifest_metadata, Mapping)
        and _strict_bool(bundle.ablation_manifest.get("metadata", {}).get("generation_rerun_required"))
        and _strict_bool(bundle.ablation_manifest.get("metadata", {}).get("per_ablation_calibration_required"))
    )


def _dataset_quality_ready(bundle: ResultClosureGateInput) -> bool:
    """核验正式 FID/KID 使用规范 Inception 特征和当前 run 的完整样本。"""

    scale = bundle.expected_paper_claim_scale
    ready_fields = (
        "formal_fid_kid_ready",
        "formal_fid_kid_metric_names_ready",
        "formal_feature_backend_ready",
        "formal_sample_scale_ready",
        "canonical_formal_feature_extractor_ready",
        "formal_fid_kid_claim_gate_ready",
    )
    sample_pair_count = _int_value(bundle.dataset_quality_summary.get("sample_pair_count"))
    source_count = _int_value(bundle.dataset_quality_summary.get("source_image_count"))
    comparison_count = _int_value(bundle.dataset_quality_summary.get("comparison_image_count"))
    expected_prompt_count = bundle.expected_prompt_count
    feature_report = bundle.dataset_quality_feature_report
    metric_rows = bundle.dataset_quality_metrics
    metric_names = [str(row.get("quality_metric_name", "")) for row in metric_rows]
    metric_contract_ready = (
        len(metric_rows) == 2
        and len(set(metric_names)) == 2
        and set(metric_names) == {"fid", "kid"}
        and all(str(row.get("metric_status", "")) == "measured" for row in metric_rows)
        and all(_float_value(row.get("quality_metric_value")) is not None for row in metric_rows)
        and all(
            _int_value(row.get(field_name)) == expected_prompt_count
            for row in metric_rows
            for field_name in (
                "source_image_count",
                "comparison_image_count",
                "sample_pair_count",
            )
        )
    )
    coverage_fields = (
        "accepted_feature_pair_count",
        "missing_feature_pair_count",
        "feature_issue_count",
        "formal_feature_record_count",
        "formal_feature_records_sha256",
        "canonical_prompt_id_digest",
        "registry_prompt_id_digest",
        "prompt_registry_exact_set_ready",
    )
    return (
        _all_true(bundle.dataset_quality_summary, ready_fields)
        and _strict_bool(bundle.dataset_quality_summary.get("prompt_registry_exact_set_ready"))
        and sample_pair_count == expected_prompt_count
        and source_count == sample_pair_count
        and comparison_count == sample_pair_count
        and _int_value(bundle.dataset_quality_summary.get("expected_prompt_count"))
        == expected_prompt_count
        and _int_value(bundle.dataset_quality_summary.get("registry_prompt_count"))
        == expected_prompt_count
        and _int_value(bundle.dataset_quality_summary.get("duplicate_registry_prompt_id_count")) == 0
        and _int_value(bundle.dataset_quality_summary.get("missing_registry_prompt_id_count")) == 0
        and _int_value(bundle.dataset_quality_summary.get("unexpected_registry_prompt_id_count")) == 0
        and str(bundle.dataset_quality_summary.get("canonical_prompt_id_digest", ""))
        == bundle.expected_prompt_id_digest
        and str(bundle.dataset_quality_summary.get("registry_prompt_id_digest", ""))
        == bundle.expected_prompt_id_digest
        and _int_value(bundle.dataset_quality_summary.get("accepted_feature_pair_count"))
        == expected_prompt_count
        and _int_value(bundle.dataset_quality_summary.get("missing_feature_pair_count")) == 0
        and _int_value(bundle.dataset_quality_summary.get("feature_issue_count")) == 0
        and _int_value(bundle.dataset_quality_summary.get("formal_feature_record_count"))
        == expected_prompt_count * 2
        and _is_sha256(bundle.dataset_quality_summary.get("formal_feature_records_sha256", ""))
        and bundle.dataset_quality_feature_records_sha256
        == str(bundle.dataset_quality_summary.get("formal_feature_records_sha256", ""))
        and str(feature_report.get("paper_run_name", "")) == bundle.expected_paper_claim_scale
        and _same_float(feature_report.get("target_fpr"), bundle.expected_target_fpr)
        and _strict_bool(feature_report.get("prompt_registry_exact_set_ready"))
        and _int_value(feature_report.get("expected_feature_pair_count")) == expected_prompt_count
        and _int_value(feature_report.get("accepted_feature_pair_count")) == expected_prompt_count
        and _int_value(feature_report.get("missing_feature_pair_count")) == 0
        and _int_value(feature_report.get("feature_issue_count")) == 0
        and _int_value(feature_report.get("formal_feature_record_count")) == expected_prompt_count * 2
        and str(feature_report.get("canonical_prompt_id_digest", ""))
        == bundle.expected_prompt_id_digest
        and str(feature_report.get("registry_prompt_id_digest", ""))
        == bundle.expected_prompt_id_digest
        and all(
            bundle.dataset_quality_summary.get(field_name) == feature_report.get(field_name)
            for field_name in coverage_fields
        )
        and metric_contract_ready
        and _manifest_ready(
            bundle.dataset_quality_manifest,
            artifact_id="dataset_level_quality_manifest",
            required_output_suffixes=(
                f"outputs/dataset_level_quality/{scale}/dataset_quality_formal_feature_records.jsonl",
                f"outputs/dataset_level_quality/{scale}/dataset_quality_formal_feature_import_report.json",
                f"outputs/dataset_level_quality/{scale}/dataset_quality_metrics.csv",
                f"outputs/dataset_level_quality/{scale}/dataset_quality_summary.json",
                f"outputs/dataset_level_quality/{scale}/manifest.local.json",
            ),
        )
        and _metadata_matches(
            bundle.dataset_quality_manifest,
            bundle.dataset_quality_summary,
            (
                "formal_fid_kid_ready",
                "formal_sample_scale_ready",
                "formal_fid_kid_claim_gate_ready",
                *coverage_fields,
            ),
        )
    )


def _rebuild_evidence_audit(
    bundle: ResultClosureGateInput,
) -> tuple[dict[str, Any], dict[str, str]]:
    """从最终闭合输入精确重建证据审计报告与 manifest 配置.

    该函数不访问文件系统, 因而保持结果闭合门禁为纯函数.实际文件读取和
    11类数据 validator 的再次执行由外层 writer 完成, 本函数只核验注入的
    即时重算报告是否能唯一重建持久化审计产物与 ``config_digest``.
    """

    audit_bundle = AuditInputBundle(
        threshold_report=bundle.evidence_audit_runtime_report,
        threshold_manifest=bundle.evidence_audit_runtime_manifest,
        threshold_audit_report=bundle.threshold_audit_report,
        threshold_audit_manifest=bundle.threshold_audit_manifest,
        attack_manifest=bundle.attack_report,
        attack_matrix_manifest=bundle.attack_manifest,
        baseline_manifest=bundle.baseline_manifest,
        baseline_runtime_report=bundle.baseline_report,
        dataset_quality_manifest=bundle.dataset_quality_manifest,
        dataset_quality_summary=bundle.dataset_quality_summary,
        ablation_manifest=bundle.ablation_manifest,
        ablation_claim_summary=bundle.ablation_summary,
        source_path_map=bundle.evidence_audit_source_path_map,
        artifact_data_validation=(
            bundle.recomputed_artifact_data_validation_report
        ),
    )
    materialization = build_evidence_audit_materialization(audit_bundle)
    manifest_config = build_evidence_audit_manifest_config(
        audit_bundle,
        materialization,
    )
    return materialization, manifest_config


def _evidence_audit_ready(bundle: ResultClosureGateInput) -> bool:
    """核验 claim 到论文表图的证据审计没有缺口或阻断项。"""

    rebuilt, expected_manifest_config = _rebuild_evidence_audit(bundle)
    reports_rebuilt_ready = (
        bundle.artifact_data_validation_report
        == bundle.recomputed_artifact_data_validation_report
        and rebuilt["builder_report"] == bundle.evidence_builder_report
        and rebuilt["blocker_report"] == bundle.evidence_blocker_report
    )

    builder_ready = _all_true(
        bundle.evidence_builder_report,
        ("artifact_builder_ready", "paper_artifact_claim_ready", "paper_artifact_audit_ready"),
    ) and _all_zero(bundle.evidence_builder_report, ("blocked_artifact_count",))
    blocker_ready = (
        _all_true(
            bundle.evidence_blocker_report,
            (
                "submission_ready",
                "artifact_builder_ready",
                "paper_artifact_claim_ready",
                "paper_artifact_audit_ready",
                "full_method_claim_ready",
                "supports_paper_claim",
            ),
        )
        and _all_zero(
            bundle.evidence_blocker_report,
            ("blocking_claim_count", "critical_gap_count", "gap_count"),
        )
    )
    scale = bundle.expected_paper_claim_scale
    inputs = bundle.evidence_audit_manifest.get("input_paths", ())
    data_report = bundle.recomputed_artifact_data_validation_report
    expected_data_check_ids = {
        "frozen_evidence_protocol_ready",
        "raw_image_only_detection_records_ready",
        "test_detection_metrics_ready",
        "score_distribution_table_ready",
        "roc_curve_points_ready",
        "det_curve_points_ready",
        "attack_family_metrics_ready",
        "baseline_comparison_table_ready",
        "mechanism_ablation_metrics_ready",
        "mechanism_pairwise_delta_ready",
        "dataset_quality_metrics_ready",
        "ready_flag_consistency_ready",
    }
    source_paths = data_report.get("source_paths", {})
    source_sha256 = data_report.get("evidence_source_file_sha256", {})
    checks = data_report.get("checks", {})
    data_ready = (
        data_report.get("artifact_data_validation_ready") is True
        and _int_value(data_report.get("artifact_data_check_count"))
        == len(expected_data_check_ids)
        and _int_value(data_report.get("blocked_artifact_data_count")) == 0
        and data_report.get("blocked_artifact_data_ids") == []
        and isinstance(checks, Mapping)
        and set(checks) == expected_data_check_ids
        and all(
            isinstance(check, Mapping)
            and check.get("data_ready") is True
            and check.get("issues") == []
            for check in checks.values()
        )
        and isinstance(source_paths, Mapping)
        and len(source_paths) == 11
        and set(source_paths) == expected_data_check_ids - {"ready_flag_consistency_ready"}
        and isinstance(source_sha256, Mapping)
        and set(source_sha256) == set(source_paths.values())
        and all(_is_sha256(value) for value in source_sha256.values())
        and all(
            bundle.source_file_sha256.get(str(path)) == str(digest)
            for path, digest in source_sha256.items()
        )
        and _is_sha256(data_report.get("raw_image_only_detection_records_sha256", ""))
        and str(data_report.get("raw_image_only_detection_records_sha256", ""))
        == str(
            source_sha256.get(
                str(source_paths.get("raw_image_only_detection_records_ready", "")),
                "",
            )
        )
        and all(_path_present(inputs, str(path)) for path in source_paths.values())
    )
    manifest_metadata = bundle.evidence_audit_manifest.get("metadata", {})
    manifest_config_ready = (
        set(expected_manifest_config)
        == {
            "summary_digest",
            "input_bundle_digest",
            "artifact_data_validation_digest",
        }
        and all(_is_sha256(value) for value in expected_manifest_config.values())
        and expected_manifest_config["artifact_data_validation_digest"]
        == build_stable_digest(data_report)
        and str(bundle.evidence_audit_manifest.get("config_digest", ""))
        == build_stable_digest(expected_manifest_config)
    )
    provenance_ready = all(
        _path_present(inputs, suffix)
        for suffix in (
            f"outputs/image_only_dataset_runtime/{scale}/dataset_runtime_summary.json",
            f"outputs/formal_mechanism_ablation/{scale}/ablation_claim_summary.json",
            f"outputs/dataset_level_quality/{scale}/dataset_quality_summary.json",
        )
    )
    return (
        reports_rebuilt_ready
        and builder_ready
        and blocker_ready
        and data_ready
        and provenance_ready
        and manifest_config_ready
        and _manifest_ready(
            bundle.evidence_audit_manifest,
            artifact_id="paper_artifact_evidence_audit_manifest",
            required_output_suffixes=(
                f"outputs/paper_artifact_evidence_audit/{scale}/artifact_builder_readiness_report.json",
                f"outputs/paper_artifact_evidence_audit/{scale}/submission_blocker_report.json",
                f"outputs/paper_artifact_evidence_audit/{scale}/artifact_data_validation_report.json",
                f"outputs/paper_artifact_evidence_audit/{scale}/manifest.local.json",
            ),
        )
        and isinstance(manifest_metadata, Mapping)
        and manifest_metadata.get("artifact_data_validation_ready") is True
        and manifest_metadata.get("blocked_artifact_data_ids") == []
        and manifest_metadata.get("evidence_source_file_sha256") == source_sha256
        and _metadata_matches(
            bundle.evidence_audit_manifest,
            bundle.evidence_blocker_report,
            ("submission_ready", "paper_artifact_claim_ready", "supports_paper_claim"),
        )
    )


def _submission_ready(bundle: ResultClosureGateInput) -> bool:
    """核验投稿就绪门禁允许冻结且不存在待补证据。"""

    scale = bundle.expected_paper_claim_scale
    ready_fields = (
        "submission_ready",
        "package_freeze_allowed",
        "artifact_builder_ready",
        "paper_artifact_claim_ready",
        "release_dry_run_ready",
    )
    return (
        _all_true(bundle.submission_readiness_report, ready_fields)
        and str(bundle.submission_readiness_report.get("readiness_decision", "")) == "ready"
        and _all_zero(
            bundle.submission_readiness_report,
            ("required_input_count", "critical_required_input_count", "blocking_claim_count"),
        )
        and _manifest_ready(
            bundle.submission_readiness_manifest,
            artifact_id="submission_readiness_manifest",
            required_output_suffixes=(
                f"outputs/submission_readiness/{scale}/readiness_blocker_report.json",
                f"outputs/submission_readiness/{scale}/submission_readiness_manifest.local.json",
            ),
        )
        and _metadata_matches(
            bundle.submission_readiness_manifest,
            bundle.submission_readiness_report,
            ("readiness_decision", "submission_ready", "package_freeze_allowed"),
        )
    )


def _entry_review_ready(bundle: ResultClosureGateInput) -> bool:
    """核验证据闭合入口明确允许进入闭合, 而非仅生成可审计报告。"""

    ready_fields = (
        "entry_review_ready",
        "evidence_closure_allowed",
        "primary_baseline_results_ready",
        "formal_import_validation_ready",
        "formal_evidence_path_resolution_ready",
        "formal_fid_kid_ready",
        "formal_sample_scale_ready",
        "formal_feature_backend_ready",
    )
    scale = bundle.expected_paper_claim_scale
    inputs = bundle.entry_review_manifest.get("input_paths", ())
    return (
        _all_true(bundle.entry_review_report, ready_fields)
        and str(bundle.entry_review_report.get("entry_review_decision", ""))
        == "ready_for_evidence_closure"
        and _all_zero(
            bundle.entry_review_report,
            (
                "blocked_review_item_count",
                "required_input_count",
                "critical_required_input_count",
                "blocking_claim_count",
            ),
        )
        and (_int_value(bundle.entry_review_report.get("accepted_formal_import_count")) or 0) > 0
        and _path_present(inputs, f"outputs/dataset_level_quality/{scale}/dataset_quality_summary.json")
        and _manifest_ready(
            bundle.entry_review_manifest,
            artifact_id="evidence_closure_entry_review_manifest",
            required_output_suffixes=(
                f"outputs/evidence_closure_entry_review/{scale}/entry_review_report.json",
                f"outputs/evidence_closure_entry_review/{scale}/manifest.local.json",
            ),
        )
        and _metadata_matches(
            bundle.entry_review_manifest,
            bundle.entry_review_report,
            ("entry_review_decision", "evidence_closure_allowed"),
        )
    )


def build_result_closure_gate_checks(bundle: ResultClosureGateInput) -> list[dict[str, Any]]:
    """构造完整论文结果闭合的语义级检查清单。"""

    return [
        _check(
            "current_run_scope_consistent",
            "paper_run_scope",
            _scope_ready(bundle),
            ("attack_matrix", "threshold_audit", "result_records", "common_protocol", "result_analysis"),
            "paper_claim_scale_inconsistent",
        ),
        _check(
            "target_fpr_consistent",
            "fixed_fpr_protocol",
            _target_fpr_ready(bundle),
            (
                "attack_matrix",
                "threshold_audit",
                "primary_baseline_evidence",
                "baseline_comparison",
                "result_records",
                "common_protocol",
                "result_analysis",
                "formal_ablation",
                "dataset_quality",
            ),
            "target_fpr_inconsistent",
        ),
        _check(
            "test_split_count_consistent",
            "sample_scale",
            _test_count_ready(bundle),
            ("attack_matrix", "threshold_audit", "result_records", "common_protocol", "formal_ablation"),
            "test_split_count_inconsistent",
        ),
        _check(
            "threshold_digest_consistent",
            "fixed_fpr_protocol",
            _threshold_digest_ready(bundle),
            ("threshold_audit", "attack_matrix", "result_records", "common_protocol"),
            "method_threshold_digest_inconsistent",
        ),
        _check(
            "closure_input_provenance_consistent",
            "input_provenance",
            _closure_input_provenance_ready(bundle),
            ("closure_input_lock", "result_records", "common_protocol"),
            "closure_input_provenance_inconsistent",
        ),
        _check(
            "common_protocol_digests_consistent",
            "result_protocol",
            _common_protocol_digest_ready(bundle),
            ("common_protocol_schema", "result_records"),
            "result_protocol_digest_inconsistent",
        ),
        _check(
            "result_record_set_provenance_consistent",
            "result_provenance",
            _result_record_set_provenance_ready(bundle),
            ("result_records", "common_protocol", "result_analysis"),
            "result_record_set_digest_inconsistent",
        ),
        _check(
            "paired_superiority_ready",
            "paired_statistical_inference",
            _paired_superiority_ready(bundle),
            (
                "paired_outcomes",
                "paired_superiority_rows",
                "paired_superiority_summary",
                "paired_superiority_manifest",
                "common_protocol",
            ),
            "paired_superiority_evidence_not_ready",
        ),
        _check(
            "attack_matrix_ready",
            "attack_robustness",
            _attack_ready(bundle),
            ("attack_report", "attack_manifest"),
            "formal_attack_matrix_not_ready",
        ),
        _check(
            "fixed_fpr_threshold_audit_ready",
            "fixed_fpr_protocol",
            _threshold_audit_ready(bundle),
            ("threshold_audit_report", "threshold_audit_rows", "threshold_audit_manifest"),
            "unified_threshold_audit_not_ready",
        ),
        _check(
            "official_reference_fidelity_evidence_ready",
            "supplemental_method_fidelity",
            _official_reference_fidelity_ready(bundle),
            (
                "official_reference_fidelity_records",
                "official_reference_fidelity_summary",
                "official_reference_fidelity_manifest",
            ),
            "official_reference_fidelity_evidence_not_ready",
        ),
        _check(
            "primary_baseline_evidence_ready",
            "baseline_evidence",
            _primary_baseline_evidence_ready(bundle),
            (
                "primary_baseline_evidence_records",
                "primary_baseline_evidence_summary",
                "primary_baseline_evidence_manifest",
            ),
            "primary_baseline_evidence_not_ready",
        ),
        _check(
            "baseline_comparison_ready",
            "baseline_comparison",
            _baseline_ready(bundle),
            ("baseline_report", "baseline_manifest"),
            "formal_baseline_comparison_not_ready",
        ),
        _check(
            "result_records_ready",
            "result_records",
            _result_records_ready(bundle),
            ("result_records", "result_record_summary", "result_record_manifest"),
            "formal_result_records_not_ready",
        ),
        _check(
            "common_protocol_ready",
            "result_protocol",
            _common_protocol_ready(bundle),
            ("common_protocol_summary", "common_protocol_schema", "common_protocol_manifest"),
            "common_protocol_not_ready",
        ),
        _check(
            "result_analysis_ready",
            "result_analysis",
            _result_analysis_ready(bundle),
            ("result_analysis_summary", "result_analysis_manifest"),
            "paper_result_analysis_not_ready",
        ),
        _check(
            "formal_ablation_ready",
            "mechanism_ablation",
            _ablation_ready(bundle),
            ("ablation_summary", "ablation_manifest"),
            "formal_mechanism_ablation_not_ready",
        ),
        _check(
            "formal_fid_kid_ready",
            "dataset_quality",
            _dataset_quality_ready(bundle),
            ("dataset_quality_summary", "dataset_quality_manifest"),
            "formal_fid_kid_not_ready",
        ),
        _check(
            "paper_evidence_audit_ready",
            "claim_evidence",
            _evidence_audit_ready(bundle),
            ("evidence_builder_report", "evidence_blocker_report", "evidence_audit_manifest"),
            "paper_evidence_audit_not_ready",
        ),
        _check(
            "submission_readiness_ready",
            "submission_readiness",
            _submission_ready(bundle),
            ("submission_readiness_report", "submission_readiness_manifest"),
            "submission_readiness_not_ready",
        ),
        _check(
            "evidence_closure_entry_ready",
            "evidence_closure",
            _entry_review_ready(bundle),
            ("entry_review_report", "entry_review_manifest"),
            "evidence_closure_not_allowed",
        ),
    ]


def build_result_closure_gate_report(
    bundle: ResultClosureGateInput,
    checks: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """汇总语义检查并给出唯一的论文结果闭合判定。"""

    materialized = [dict(row) for row in checks]
    blocked = [row for row in materialized if str(row.get("check_status", "")) != "pass"]
    entry_allowed = _strict_bool(bundle.entry_review_report.get("evidence_closure_allowed"))
    ready = bool(materialized) and not blocked and entry_allowed
    source_digests = {
        "attack_matrix_digest": build_stable_digest(
            {"report": bundle.attack_report, "manifest": bundle.attack_manifest}
        ),
        "threshold_audit_digest": build_stable_digest(
            {
                "report": bundle.threshold_audit_report,
                "rows": bundle.threshold_audit_rows,
                "manifest": bundle.threshold_audit_manifest,
            }
        ),
        "closure_input_lock_digest": build_stable_digest(
            {
                "lock": bundle.closure_input_lock,
                "manifest": bundle.closure_input_lock_manifest,
            }
        ),
        "official_reference_fidelity_digest": build_stable_digest(
            {
                "records": bundle.official_reference_fidelity_records,
                "summary": bundle.official_reference_fidelity_summary,
                "manifest": bundle.official_reference_fidelity_manifest,
            }
        ),
        "baseline_comparison_digest": build_stable_digest(
            {"report": bundle.baseline_report, "manifest": bundle.baseline_manifest}
        ),
        "primary_baseline_evidence_digest": build_stable_digest(
            {
                "summary": bundle.primary_baseline_evidence_summary,
                "records": bundle.primary_baseline_evidence_records,
                "manifest": bundle.primary_baseline_evidence_manifest,
            }
        ),
        "result_records_digest": build_stable_digest(bundle.result_records),
        "common_protocol_digest": build_stable_digest(
            {"summary": bundle.common_protocol_summary, "schema": bundle.common_protocol_schema}
        ),
        "result_analysis_digest": build_stable_digest(bundle.result_analysis_summary),
        "paired_superiority_digest": build_stable_digest(
            {
                "outcomes": bundle.paired_outcomes,
                "rows": bundle.paired_superiority_rows,
                "summary": bundle.paired_superiority_summary,
                "manifest": bundle.paired_superiority_manifest,
            }
        ),
        "formal_ablation_digest": build_stable_digest(
            {"summary": bundle.ablation_summary, "manifest": bundle.ablation_manifest}
        ),
        "formal_fid_kid_digest": build_stable_digest(
            {
                "summary": bundle.dataset_quality_summary,
                "feature_report": bundle.dataset_quality_feature_report,
                "metrics": bundle.dataset_quality_metrics,
                "manifest": bundle.dataset_quality_manifest,
            }
        ),
        "paper_evidence_audit_digest": build_stable_digest(
            {
                "builder": bundle.evidence_builder_report,
                "blocker": bundle.evidence_blocker_report,
                "persisted_artifact_data_validation": (
                    bundle.artifact_data_validation_report
                ),
                "recomputed_artifact_data_validation": (
                    bundle.recomputed_artifact_data_validation_report
                ),
                "manifest": bundle.evidence_audit_manifest,
            }
        ),
        "submission_readiness_digest": build_stable_digest(bundle.submission_readiness_report),
        "entry_review_digest": build_stable_digest(bundle.entry_review_report),
    }
    return {
        "construction_unit_name": "paper_result_closure_gate",
        "paper_claim_scale": bundle.expected_paper_claim_scale,
        "target_fpr": bundle.expected_target_fpr,
        "expected_prompt_count": bundle.expected_prompt_count,
        "expected_test_count": bundle.expected_test_count,
        "expected_prompt_id_digest": bundle.expected_prompt_id_digest,
        "expected_test_prompt_id_digest": bundle.expected_test_prompt_id_digest,
        "closure_check_count": len(materialized),
        "blocked_check_count": len(blocked),
        "blocked_check_ids": [str(row.get("check_id", "")) for row in blocked],
        "checks": materialized,
        "source_artifact_digests": source_digests,
        "evidence_closure_allowed": ready,
        "result_closure_ready": ready,
        "closure_decision": "pass" if ready else "blocked",
        "supports_paper_claim": ready,
    }
