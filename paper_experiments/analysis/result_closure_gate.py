"""对完整论文结果闭合所需的受治理证据执行语义级门禁。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import string
from typing import Any, Iterable, Mapping

from experiments.ablations.runtime_rerun import (
    FORMAL_RUNTIME_RERUN_ABLATION_IDS,
    FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST,
)
from main.core.digest import build_stable_digest


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
    expected_prompt_id_digest: str
    attack_report: dict[str, Any]
    attack_manifest: dict[str, Any]
    threshold_audit_report: dict[str, Any]
    threshold_audit_rows: tuple[dict[str, Any], ...]
    threshold_audit_manifest: dict[str, Any]
    primary_baseline_evidence_summary: dict[str, Any]
    primary_baseline_evidence_manifest: dict[str, Any]
    baseline_report: dict[str, Any]
    baseline_manifest: dict[str, Any]
    result_records: tuple[dict[str, Any], ...]
    result_record_summary: dict[str, Any]
    result_record_manifest: dict[str, Any]
    common_protocol_summary: dict[str, Any]
    common_protocol_schema: dict[str, Any]
    common_protocol_manifest: dict[str, Any]
    result_analysis_summary: dict[str, Any]
    result_analysis_manifest: dict[str, Any]
    ablation_summary: dict[str, Any]
    ablation_manifest: dict[str, Any]
    dataset_quality_summary: dict[str, Any]
    dataset_quality_feature_report: dict[str, Any]
    dataset_quality_metrics: tuple[dict[str, Any], ...]
    dataset_quality_feature_records_sha256: str
    dataset_quality_manifest: dict[str, Any]
    evidence_builder_report: dict[str, Any]
    evidence_blocker_report: dict[str, Any]
    evidence_audit_manifest: dict[str, Any]
    submission_readiness_report: dict[str, Any]
    submission_readiness_manifest: dict[str, Any]
    entry_review_report: dict[str, Any]
    entry_review_manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典, 用于生成稳定输入摘要。"""

        return asdict(self)


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
        bundle.primary_baseline_evidence_summary.get("paper_claim_scale"),
        bundle.result_record_summary.get("paper_claim_scale"),
        bundle.common_protocol_summary.get("paper_claim_scale"),
        bundle.common_protocol_schema.get("paper_claim_scale"),
        bundle.result_analysis_summary.get("paper_claim_scale"),
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
        bundle.primary_baseline_evidence_summary.get("target_fpr"),
        bundle.baseline_report.get("target_fpr"),
        bundle.common_protocol_summary.get("paper_target_fpr"),
        bundle.common_protocol_summary.get("expected_target_fpr"),
        bundle.common_protocol_schema.get("target_fpr"),
        bundle.result_analysis_summary.get("target_fpr"),
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
    """核验主方法冻结阈值摘要贯穿统一审计与真实攻击矩阵。"""

    main_rows = [row for row in bundle.threshold_audit_rows if str(row.get("method_id", "")) == "slm_wm"]
    attack_boundary = bundle.attack_report.get("evaluation_boundary", {})
    attack_digest = attack_boundary.get("threshold_digest", "") if isinstance(attack_boundary, Mapping) else ""
    return (
        len(main_rows) == 1
        and _is_sha256(main_rows[0].get("threshold_digest", ""))
        and str(main_rows[0].get("threshold_digest", "")) == str(attack_digest)
        and all(_is_sha256(row.get("threshold_digest", "")) for row in bundle.threshold_audit_rows)
    )


def _common_protocol_digest_ready(bundle: ResultClosureGateInput) -> bool:
    """核验所有正式结果记录共享共同协议 schema 的三个关键摘要。"""

    digest_fields = ("prompt_split_digest", "attack_matrix_digest", "fixed_fpr_protocol_digest")
    return bool(bundle.result_records) and all(
        _is_sha256(bundle.common_protocol_schema.get(field_name, ""))
        and all(
            str(row.get(field_name, "")) == str(bundle.common_protocol_schema.get(field_name, ""))
            for row in bundle.result_records
        )
        for field_name in digest_fields
    )


def _attack_ready(bundle: ResultClosureGateInput) -> bool:
    """核验真实攻击矩阵覆盖、文件证据链和 GPU 攻击均已闭合。"""

    scale = bundle.expected_paper_claim_scale
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
        and _attack_record_counts_ready(bundle)
        and required_gpu_count is not None
        and required_gpu_count > 0
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
    actual_ids = [str(row.get("method_id", "")) for row in bundle.threshold_audit_rows]
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
            ("method_identity_ready", "all_method_thresholds_ready", "fixed_fpr_threshold_audit_ready", "supports_paper_claim"),
        )
        and len(actual_ids) == len(expected_ids)
        and len(set(actual_ids)) == len(actual_ids)
        and set(actual_ids) == expected_ids
        and all(_all_true(row, row_ready_fields) for row in bundle.threshold_audit_rows)
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
            ("paper_claim_scale", "target_fpr", "fixed_fpr_threshold_audit_ready", "supports_paper_claim"),
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
    adapter_ids = [str(value) for value in summary.get("adapter_run_ready_ids", ())]
    formal_ids = [str(value) for value in summary.get("formal_result_ready_ids", ())]
    input_ids = [str(value) for value in summary.get("input_baseline_ids", ())]
    identities_ready = all(
        len(values) == len(expected_ids)
        and len(set(values)) == len(values)
        and set(values) == expected_ids
        for values in (adapter_ids, formal_ids, input_ids)
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
            ("paper_claim_scale", "target_fpr", "primary_baseline_formal_ready"),
        )
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
    return (
        bool(bundle.result_records)
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
        and _manifest_ready(
            bundle.result_record_manifest,
            artifact_id="pilot_paper_fixed_fpr_result_records_manifest",
            required_output_suffixes=(
                f"outputs/pilot_paper_fixed_fpr_results/{scale}/pilot_paper_result_records.jsonl",
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
    return (
        _all_true(
            bundle.result_analysis_summary,
            (
                "failure_case_figure_ready",
                "result_template_coverage_ready",
                "per_attack_ci_coverage_ready",
                "per_attack_superiority_evaluation_ready",
                "supports_paper_claim",
            ),
        )
        and result_record_count > 0
        and expected_result_count == result_record_count
        and _int_value(bundle.result_analysis_summary.get("result_record_count")) == result_record_count
        and _int_value(bundle.result_analysis_summary.get("actual_result_record_count")) == result_record_count
        and _int_value(bundle.result_analysis_summary.get("unique_result_record_key_count")) == result_record_count
        and _int_value(bundle.result_analysis_summary.get("confidence_interval_row_count")) == result_record_count
        and expected_superiority_count is not None
        and expected_superiority_count > 0
        and superiority_row_count == expected_superiority_count
        and superiority_ready_count is not None
        and 0 <= superiority_ready_count <= superiority_row_count
        and _int_value(bundle.result_analysis_summary.get("duplicate_result_record_count")) == 0
        and _int_value(bundle.result_analysis_summary.get("missing_result_record_count")) == 0
        and _int_value(bundle.result_analysis_summary.get("unexpected_result_record_count")) == 0
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
    manifest_config = bundle.ablation_manifest.get("config", {})
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
        and isinstance(manifest_config, Mapping)
        and manifest_config.get("expected_ablation_ids") == expected_ids
        and manifest_config.get("actual_ablation_ids") == expected_ids
        and manifest_config.get("ablation_spec_digest")
        == FORMAL_RUNTIME_RERUN_ABLATION_SPEC_DIGEST
        and _strict_bool(manifest_config.get("ablation_exact_set_ready"))
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
    manifest_config = bundle.dataset_quality_manifest.get("config", {})
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
        and isinstance(manifest_config, Mapping)
        and str(manifest_config.get("canonical_prompt_id_digest", ""))
        == bundle.expected_prompt_id_digest
        and str(manifest_config.get("registry_prompt_id_digest", ""))
        == bundle.expected_prompt_id_digest
        and _strict_bool(manifest_config.get("prompt_registry_exact_set_ready"))
        and _int_value(manifest_config.get("accepted_feature_pair_count"))
        == expected_prompt_count
        and _int_value(manifest_config.get("missing_feature_pair_count")) == 0
        and _int_value(manifest_config.get("feature_issue_count")) == 0
        and _int_value(manifest_config.get("formal_feature_record_count"))
        == expected_prompt_count * 2
        and str(manifest_config.get("formal_feature_records_sha256", ""))
        == str(feature_report.get("formal_feature_records_sha256", ""))
    )


def _evidence_audit_ready(bundle: ResultClosureGateInput) -> bool:
    """核验 claim 到论文表图的证据审计没有缺口或阻断项。"""

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
    provenance_ready = all(
        _path_present(inputs, suffix)
        for suffix in (
            f"outputs/image_only_dataset_runtime/{scale}/dataset_runtime_summary.json",
            f"outputs/formal_mechanism_ablation/{scale}/ablation_claim_summary.json",
            f"outputs/dataset_level_quality/{scale}/dataset_quality_summary.json",
        )
    )
    return (
        builder_ready
        and blocker_ready
        and provenance_ready
        and _manifest_ready(
            bundle.evidence_audit_manifest,
            artifact_id="paper_artifact_evidence_audit_manifest",
            required_output_suffixes=(
                f"outputs/paper_artifact_evidence_audit/{scale}/artifact_builder_readiness_report.json",
                f"outputs/paper_artifact_evidence_audit/{scale}/submission_blocker_report.json",
                f"outputs/paper_artifact_evidence_audit/{scale}/manifest.local.json",
            ),
        )
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
        "user_audit_required",
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
        and str(bundle.entry_review_report.get("entry_review_decision", "")) == "ready_for_user_audit"
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
            ("threshold_audit", "attack_matrix"),
            "main_threshold_digest_inconsistent",
        ),
        _check(
            "common_protocol_digests_consistent",
            "result_protocol",
            _common_protocol_digest_ready(bundle),
            ("common_protocol_schema", "result_records"),
            "result_protocol_digest_inconsistent",
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
            "primary_baseline_evidence_ready",
            "baseline_evidence",
            _primary_baseline_evidence_ready(bundle),
            ("primary_baseline_evidence_summary", "primary_baseline_evidence_manifest"),
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
        "baseline_comparison_digest": build_stable_digest(
            {"report": bundle.baseline_report, "manifest": bundle.baseline_manifest}
        ),
        "primary_baseline_evidence_digest": build_stable_digest(
            {
                "summary": bundle.primary_baseline_evidence_summary,
                "manifest": bundle.primary_baseline_evidence_manifest,
            }
        ),
        "result_records_digest": build_stable_digest(bundle.result_records),
        "common_protocol_digest": build_stable_digest(
            {"summary": bundle.common_protocol_summary, "schema": bundle.common_protocol_schema}
        ),
        "result_analysis_digest": build_stable_digest(bundle.result_analysis_summary),
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
            {"builder": bundle.evidence_builder_report, "blocker": bundle.evidence_blocker_report}
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
