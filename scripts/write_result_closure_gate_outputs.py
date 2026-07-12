"""写出当前论文运行层级的 CPU 结果闭合语义门禁。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.artifacts.dataset_level_quality_outputs import (
    canonical_prompt_ids_for_paper_run,
)
from experiments.protocol.paper_run_config import (
    PaperRunPromptContract,
    build_paper_run_config,
)
from experiments.protocol.pilot_paper_fixed_fpr import (
    build_paper_fixed_fpr_config_from_paper_run,
    build_pilot_paper_prompt_split_summary,
)
from experiments.protocol.prompts import build_prompt_records, read_prompt_file
from experiments.protocol.splits import build_group_split_counts, group_prompt_ids_by_split
from experiments.runtime.repository_environment import resolve_code_version
from main.core.digest import build_stable_digest
from paper_experiments.analysis.result_closure_gate import (
    ResultClosureGateInput,
    build_result_closure_gate_checks,
    build_result_closure_gate_report,
    build_source_file_sha256_map,
)
from paper_experiments.analysis.result_analysis_payload import (
    build_governed_paper_payload_path_map,
)
from paper_experiments.runners.closure_package_selection import (
    validate_closure_input_lock_payloads,
)
from scripts.write_paper_artifact_evidence_audit_outputs import build_input_bundle


DEFAULT_OUTPUT_ROOT = Path("outputs/result_closure_gate")


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _resolve_path(root_path: Path, path: str | Path) -> Path:
    """把输入路径解析为绝对路径。"""

    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def _relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录相对仓库根目录的受治理路径。"""

    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    """读取必须存在的 JSON 对象。"""

    if not path.is_file():
        raise FileNotFoundError(f"结果闭合门禁缺少 JSON 输入: {path.as_posix()}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"结果闭合门禁输入必须是 JSON 对象: {path.as_posix()}")
    return dict(payload)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """读取必须存在且非空的 JSONL 正式记录。"""

    if not path.is_file():
        raise FileNotFoundError(f"结果闭合门禁缺少 JSONL 输入: {path.as_posix()}")
    rows = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"结果闭合门禁要求非空 JSONL 对象序列: {path.as_posix()}")
    return tuple(dict(row) for row in rows)


def _read_jsonl_allow_empty(path: Path) -> tuple[dict[str, Any], ...]:
    """读取允许零行这一真实负结果的 JSONL 对象序列。"""

    if not path.is_file():
        raise FileNotFoundError(f"结果闭合门禁缺少 JSONL 输入: {path.as_posix()}")
    rows = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    )
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"结果闭合门禁 JSONL 行必须是对象: {path.as_posix()}")
    return tuple(dict(row) for row in rows)


def _read_json_array(path: Path) -> tuple[dict[str, Any], ...]:
    """读取必须存在且非空的 JSON 对象数组."""

    if not path.is_file():
        raise FileNotFoundError(f"结果闭合门禁缺少 JSON 数组输入: {path.as_posix()}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(payload, list)
        or not payload
        or any(not isinstance(row, dict) for row in payload)
    ):
        raise ValueError(f"结果闭合门禁要求非空 JSON 对象数组: {path.as_posix()}")
    return tuple(dict(row) for row in payload)


def _read_csv(path: Path) -> tuple[dict[str, Any], ...]:
    """读取必须存在且非空的 CSV 审计行。"""

    if not path.is_file():
        raise FileNotFoundError(f"结果闭合门禁缺少 CSV 输入: {path.as_posix()}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = tuple(dict(row) for row in csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果闭合门禁要求非空 CSV 记录: {path.as_posix()}")
    return rows


def _ensure_run_output_dir(root_path: Path, output_root: str | Path, paper_run_name: str) -> Path:
    """构造 outputs/ 下按论文运行层级隔离的门禁目录。"""

    resolved_root = _resolve_path(root_path, output_root)
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved_root.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("结果闭合门禁输出根目录必须位于 outputs/ 下") from exc
    resolved = resolved_root / paper_run_name
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _per_run_path(
    root_path: Path,
    path: str | Path | None,
    *,
    artifact_root: str,
    paper_run_name: str,
    file_name: str,
) -> Path:
    """解析显式路径, 未提供时使用统一的当前 run 产物路径。"""

    if path is not None:
        return _resolve_path(root_path, path)
    return (root_path / "outputs" / artifact_root / paper_run_name / file_name).resolve()


def write_result_closure_gate_outputs(
    *,
    root: str | Path = ".",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    attack_report_path: str | Path | None = None,
    attack_manifest_path: str | Path | None = None,
    attack_family_metrics_path: str | Path | None = None,
    threshold_audit_report_path: str | Path | None = None,
    threshold_audit_rows_path: str | Path | None = None,
    threshold_audit_manifest_path: str | Path | None = None,
    dataset_runtime_summary_path: str | Path | None = None,
    dataset_runtime_manifest_path: str | Path | None = None,
    closure_input_lock_path: str | Path | None = None,
    closure_input_lock_manifest_path: str | Path | None = None,
    official_reference_fidelity_records_path: str | Path | None = None,
    official_reference_fidelity_summary_path: str | Path | None = None,
    official_reference_fidelity_manifest_path: str | Path | None = None,
    primary_baseline_evidence_records_path: str | Path | None = None,
    primary_baseline_evidence_summary_path: str | Path | None = None,
    primary_baseline_evidence_manifest_path: str | Path | None = None,
    baseline_report_path: str | Path | None = None,
    baseline_manifest_path: str | Path | None = None,
    result_records_path: str | Path | None = None,
    result_record_summary_path: str | Path | None = None,
    result_record_manifest_path: str | Path | None = None,
    common_protocol_summary_path: str | Path | None = None,
    common_protocol_schema_path: str | Path | None = None,
    common_protocol_manifest_path: str | Path | None = None,
    result_analysis_summary_path: str | Path | None = None,
    result_analysis_manifest_path: str | Path | None = None,
    paired_outcomes_path: str | Path | None = None,
    paired_superiority_rows_path: str | Path | None = None,
    paired_superiority_summary_path: str | Path | None = None,
    paired_superiority_manifest_path: str | Path | None = None,
    ablation_runtime_records_path: str | Path | None = None,
    ablation_detection_records_path: str | Path | None = None,
    ablation_frozen_protocols_path: str | Path | None = None,
    ablation_summary_path: str | Path | None = None,
    ablation_manifest_path: str | Path | None = None,
    dataset_quality_summary_path: str | Path | None = None,
    dataset_quality_image_records_path: str | Path | None = None,
    dataset_quality_image_resolution_records_path: str | Path | None = None,
    dataset_quality_feature_records_path: str | Path | None = None,
    dataset_quality_feature_report_path: str | Path | None = None,
    dataset_quality_metrics_path: str | Path | None = None,
    dataset_quality_manifest_path: str | Path | None = None,
    evidence_builder_report_path: str | Path | None = None,
    evidence_blocker_report_path: str | Path | None = None,
    artifact_data_validation_report_path: str | Path | None = None,
    evidence_audit_manifest_path: str | Path | None = None,
    submission_readiness_report_path: str | Path | None = None,
    submission_readiness_manifest_path: str | Path | None = None,
    entry_review_report_path: str | Path | None = None,
    entry_review_manifest_path: str | Path | None = None,
    prompt_contract: PaperRunPromptContract | None = None,
) -> dict[str, Any]:
    """读取所有正式证据并写出当前 run 的结果闭合 report 与 manifest。"""

    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(
        root_path,
        prompt_contract=prompt_contract,
    )
    split_counts = build_group_split_counts(paper_run.prompt_count)
    output_dir = _ensure_run_output_dir(root_path, output_root, paper_run.run_name)
    requested_prompt_path = _resolve_path(root_path, paper_run.prompt_file)
    packaged_prompt_path = (ROOT / paper_run.prompt_file).resolve()
    canonical_prompt_path = (
        requested_prompt_path if requested_prompt_path.is_file() else packaged_prompt_path
    )

    resolved_paths = {
        "canonical_prompt_file": canonical_prompt_path,
        "attack_report": _per_run_path(
            root_path,
            attack_report_path,
            artifact_root="attack_matrix",
            paper_run_name=paper_run.run_name,
            file_name="attack_manifest.json",
        ),
        "attack_manifest": _per_run_path(
            root_path,
            attack_manifest_path,
            artifact_root="attack_matrix",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "attack_family_metrics": _per_run_path(
            root_path,
            attack_family_metrics_path,
            artifact_root="attack_matrix",
            paper_run_name=paper_run.run_name,
            file_name="attack_family_metrics.csv",
        ),
        "threshold_audit_report": _per_run_path(
            root_path,
            threshold_audit_report_path,
            artifact_root="fixed_fpr_threshold_audit",
            paper_run_name=paper_run.run_name,
            file_name="threshold_audit_report.json",
        ),
        "threshold_audit_rows": _per_run_path(
            root_path,
            threshold_audit_rows_path,
            artifact_root="fixed_fpr_threshold_audit",
            paper_run_name=paper_run.run_name,
            file_name="threshold_audit_rows.csv",
        ),
        "threshold_audit_manifest": _per_run_path(
            root_path,
            threshold_audit_manifest_path,
            artifact_root="fixed_fpr_threshold_audit",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "dataset_runtime_summary": _per_run_path(
            root_path,
            dataset_runtime_summary_path,
            artifact_root="image_only_dataset_runtime",
            paper_run_name=paper_run.run_name,
            file_name="dataset_runtime_summary.json",
        ),
        "dataset_runtime_manifest": _per_run_path(
            root_path,
            dataset_runtime_manifest_path,
            artifact_root="image_only_dataset_runtime",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "attack_detection_records": _per_run_path(
            root_path,
            None,
            artifact_root="attack_matrix",
            paper_run_name=paper_run.run_name,
            file_name="attack_detection_records.jsonl",
        ),
        "attacked_image_registry": _per_run_path(
            root_path,
            None,
            artifact_root="attack_matrix",
            paper_run_name=paper_run.run_name,
            file_name="attacked_image_registry.jsonl",
        ),
        "closure_input_lock": _per_run_path(
            root_path,
            closure_input_lock_path,
            artifact_root="paper_result_closure",
            paper_run_name=paper_run.run_name,
            file_name="closure_input_lock.json",
        ),
        "closure_input_lock_manifest": _per_run_path(
            root_path,
            closure_input_lock_manifest_path,
            artifact_root="paper_result_closure",
            paper_run_name=paper_run.run_name,
            file_name="input_lock_manifest.local.json",
        ),
        "official_reference_fidelity_records": _per_run_path(
            root_path,
            official_reference_fidelity_records_path,
            artifact_root="official_reference_fidelity_evidence",
            paper_run_name=paper_run.run_name,
            file_name="official_reference_fidelity_evidence_records.jsonl",
        ),
        "official_reference_fidelity_summary": _per_run_path(
            root_path,
            official_reference_fidelity_summary_path,
            artifact_root="official_reference_fidelity_evidence",
            paper_run_name=paper_run.run_name,
            file_name="official_reference_fidelity_evidence_summary.json",
        ),
        "official_reference_fidelity_manifest": _per_run_path(
            root_path,
            official_reference_fidelity_manifest_path,
            artifact_root="official_reference_fidelity_evidence",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "primary_baseline_evidence_records": _per_run_path(
            root_path,
            primary_baseline_evidence_records_path,
            artifact_root="primary_baseline_evidence",
            paper_run_name=paper_run.run_name,
            file_name="primary_baseline_evidence_records.jsonl",
        ),
        "primary_baseline_evidence_summary": _per_run_path(
            root_path,
            primary_baseline_evidence_summary_path,
            artifact_root="primary_baseline_evidence",
            paper_run_name=paper_run.run_name,
            file_name="primary_baseline_evidence_summary.json",
        ),
        "primary_baseline_evidence_manifest": _per_run_path(
            root_path,
            primary_baseline_evidence_manifest_path,
            artifact_root="primary_baseline_evidence",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "baseline_report": _per_run_path(
            root_path,
            baseline_report_path,
            artifact_root="external_baseline_comparison",
            paper_run_name=paper_run.run_name,
            file_name="baseline_runtime_report.json",
        ),
        "baseline_manifest": _per_run_path(
            root_path,
            baseline_manifest_path,
            artifact_root="external_baseline_comparison",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "result_records": _per_run_path(
            root_path,
            result_records_path,
            artifact_root="pilot_paper_fixed_fpr_results",
            paper_run_name=paper_run.run_name,
            file_name="pilot_paper_result_records.jsonl",
        ),
        "result_record_summary": _per_run_path(
            root_path,
            result_record_summary_path,
            artifact_root="pilot_paper_fixed_fpr_results",
            paper_run_name=paper_run.run_name,
            file_name="pilot_paper_result_record_summary.json",
        ),
        "result_record_validation_report": _per_run_path(
            root_path,
            None,
            artifact_root="pilot_paper_fixed_fpr_results",
            paper_run_name=paper_run.run_name,
            file_name="pilot_paper_result_import_validation_report.json",
        ),
        "result_record_template_coverage": _per_run_path(
            root_path,
            None,
            artifact_root="pilot_paper_fixed_fpr_results",
            paper_run_name=paper_run.run_name,
            file_name="pilot_paper_result_template_coverage.csv",
        ),
        "result_record_manifest": _per_run_path(
            root_path,
            result_record_manifest_path,
            artifact_root="pilot_paper_fixed_fpr_results",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "common_protocol_summary": _per_run_path(
            root_path,
            common_protocol_summary_path,
            artifact_root="pilot_paper_fixed_fpr_common_protocol",
            paper_run_name=paper_run.run_name,
            file_name="pilot_paper_common_protocol_summary.json",
        ),
        "common_protocol_schema": _per_run_path(
            root_path,
            common_protocol_schema_path,
            artifact_root="pilot_paper_fixed_fpr_common_protocol",
            paper_run_name=paper_run.run_name,
            file_name="pilot_paper_result_import_schema.json",
        ),
        "common_protocol_manifest": _per_run_path(
            root_path,
            common_protocol_manifest_path,
            artifact_root="pilot_paper_fixed_fpr_common_protocol",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "result_analysis_summary": _per_run_path(
            root_path,
            result_analysis_summary_path,
            artifact_root="pilot_paper_result_analysis",
            paper_run_name=paper_run.run_name,
            file_name="result_analysis_summary.json",
        ),
        "result_analysis_manifest": _per_run_path(
            root_path,
            result_analysis_manifest_path,
            artifact_root="pilot_paper_result_analysis",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "baseline_comparison_table": _per_run_path(
            root_path,
            None,
            artifact_root="external_baseline_comparison",
            paper_run_name=paper_run.run_name,
            file_name="baseline_comparison_table.csv",
        ),
        "result_analysis_confidence_interval_table": _per_run_path(
            root_path,
            None,
            artifact_root="pilot_paper_result_analysis",
            paper_run_name=paper_run.run_name,
            file_name="confidence_interval_table.csv",
        ),
        "result_analysis_per_attack_superiority_table": _per_run_path(
            root_path,
            None,
            artifact_root="pilot_paper_result_analysis",
            paper_run_name=paper_run.run_name,
            file_name="per_attack_superiority_table.csv",
        ),
        "result_analysis_failure_case_records": _per_run_path(
            root_path,
            None,
            artifact_root="pilot_paper_result_analysis",
            paper_run_name=paper_run.run_name,
            file_name="failure_case_records.jsonl",
        ),
        "result_analysis_failure_case_figure": _per_run_path(
            root_path,
            None,
            artifact_root="pilot_paper_result_analysis",
            paper_run_name=paper_run.run_name,
            file_name="failure_case_figure.svg",
        ),
        "paired_outcomes": _per_run_path(
            root_path,
            paired_outcomes_path,
            artifact_root="paired_superiority_analysis",
            paper_run_name=paper_run.run_name,
            file_name="paired_outcomes.jsonl",
        ),
        "paired_superiority_rows": _per_run_path(
            root_path,
            paired_superiority_rows_path,
            artifact_root="paired_superiority_analysis",
            paper_run_name=paper_run.run_name,
            file_name="paired_superiority_table.csv",
        ),
        "paired_superiority_summary": _per_run_path(
            root_path,
            paired_superiority_summary_path,
            artifact_root="paired_superiority_analysis",
            paper_run_name=paper_run.run_name,
            file_name="paired_superiority_summary.json",
        ),
        "paired_superiority_manifest": _per_run_path(
            root_path,
            paired_superiority_manifest_path,
            artifact_root="paired_superiority_analysis",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "ablation_summary": _per_run_path(
            root_path,
            ablation_summary_path,
            artifact_root="formal_mechanism_ablation",
            paper_run_name=paper_run.run_name,
            file_name="ablation_claim_summary.json",
        ),
        "ablation_runtime_records": _per_run_path(
            root_path,
            ablation_runtime_records_path,
            artifact_root="formal_mechanism_ablation",
            paper_run_name=paper_run.run_name,
            file_name="runtime_rerun_records.jsonl",
        ),
        "ablation_detection_records": _per_run_path(
            root_path,
            ablation_detection_records_path,
            artifact_root="formal_mechanism_ablation",
            paper_run_name=paper_run.run_name,
            file_name="formal_detection_records.jsonl",
        ),
        "ablation_frozen_protocols": _per_run_path(
            root_path,
            ablation_frozen_protocols_path,
            artifact_root="formal_mechanism_ablation",
            paper_run_name=paper_run.run_name,
            file_name="per_ablation_frozen_protocols.json",
        ),
        "ablation_manifest": _per_run_path(
            root_path,
            ablation_manifest_path,
            artifact_root="formal_mechanism_ablation",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "ablation_necessity_rows": _per_run_path(
            root_path,
            None,
            artifact_root="formal_mechanism_ablation",
            paper_run_name=paper_run.run_name,
            file_name="mechanism_necessity_statistics.csv",
        ),
        "ablation_necessity_summary": _per_run_path(
            root_path,
            None,
            artifact_root="formal_mechanism_ablation",
            paper_run_name=paper_run.run_name,
            file_name="mechanism_necessity_summary.json",
        ),
        "dataset_quality_summary": _per_run_path(
            root_path,
            dataset_quality_summary_path,
            artifact_root="dataset_level_quality",
            paper_run_name=paper_run.run_name,
            file_name="dataset_quality_summary.json",
        ),
        "dataset_quality_image_records": _per_run_path(
            root_path,
            dataset_quality_image_records_path,
            artifact_root="dataset_level_quality",
            paper_run_name=paper_run.run_name,
            file_name="dataset_quality_image_records.jsonl",
        ),
        "dataset_quality_image_resolution_records": _per_run_path(
            root_path,
            dataset_quality_image_resolution_records_path,
            artifact_root="dataset_level_quality",
            paper_run_name=paper_run.run_name,
            file_name="dataset_quality_image_resolution_records.jsonl",
        ),
        "dataset_quality_feature_report": _per_run_path(
            root_path,
            dataset_quality_feature_report_path,
            artifact_root="dataset_level_quality",
            paper_run_name=paper_run.run_name,
            file_name="dataset_quality_formal_feature_import_report.json",
        ),
        "dataset_quality_feature_records": _per_run_path(
            root_path,
            dataset_quality_feature_records_path,
            artifact_root="dataset_level_quality",
            paper_run_name=paper_run.run_name,
            file_name="dataset_quality_formal_feature_records.jsonl",
        ),
        "dataset_quality_metrics": _per_run_path(
            root_path,
            dataset_quality_metrics_path,
            artifact_root="dataset_level_quality",
            paper_run_name=paper_run.run_name,
            file_name="dataset_quality_metrics.csv",
        ),
        "dataset_quality_manifest": _per_run_path(
            root_path,
            dataset_quality_manifest_path,
            artifact_root="dataset_level_quality",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "evidence_builder_report": _per_run_path(
            root_path,
            evidence_builder_report_path,
            artifact_root="paper_artifact_evidence_audit",
            paper_run_name=paper_run.run_name,
            file_name="artifact_builder_readiness_report.json",
        ),
        "evidence_blocker_report": _per_run_path(
            root_path,
            evidence_blocker_report_path,
            artifact_root="paper_artifact_evidence_audit",
            paper_run_name=paper_run.run_name,
            file_name="submission_blocker_report.json",
        ),
        "artifact_data_validation_report": _per_run_path(
            root_path,
            artifact_data_validation_report_path,
            artifact_root="paper_artifact_evidence_audit",
            paper_run_name=paper_run.run_name,
            file_name="artifact_data_validation_report.json",
        ),
        "evidence_audit_manifest": _per_run_path(
            root_path,
            evidence_audit_manifest_path,
            artifact_root="paper_artifact_evidence_audit",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "submission_readiness_report": _per_run_path(
            root_path,
            submission_readiness_report_path,
            artifact_root="submission_readiness",
            paper_run_name=paper_run.run_name,
            file_name="readiness_blocker_report.json",
        ),
        "submission_readiness_manifest": _per_run_path(
            root_path,
            submission_readiness_manifest_path,
            artifact_root="submission_readiness",
            paper_run_name=paper_run.run_name,
            file_name="submission_readiness_manifest.local.json",
        ),
        "entry_review_report": _per_run_path(
            root_path,
            entry_review_report_path,
            artifact_root="evidence_closure_entry_review",
            paper_run_name=paper_run.run_name,
            file_name="entry_review_report.json",
        ),
        "entry_review_manifest": _per_run_path(
            root_path,
            entry_review_manifest_path,
            artifact_root="evidence_closure_entry_review",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
    }
    recomputed_evidence_audit_bundle = build_input_bundle(
        root_path,
        resolved_paths["dataset_runtime_summary"],
        resolved_paths["dataset_runtime_manifest"],
        resolved_paths["threshold_audit_report"],
        resolved_paths["threshold_audit_manifest"],
        resolved_paths["attack_report"],
        resolved_paths["attack_manifest"],
        resolved_paths["baseline_manifest"],
        resolved_paths["baseline_report"],
        resolved_paths["dataset_quality_manifest"],
        resolved_paths["dataset_quality_summary"],
        resolved_paths["ablation_manifest"],
        resolved_paths["ablation_summary"],
    )
    recomputed_artifact_data_validation_report = (
        recomputed_evidence_audit_bundle.artifact_data_validation
    )
    official_reference_fidelity_records = _read_jsonl(
        resolved_paths["official_reference_fidelity_records"]
    )
    dataset_quality_image_resolution_records = _read_jsonl(
        resolved_paths["dataset_quality_image_resolution_records"]
    )
    artifact_data_validation_report = _read_json(
        resolved_paths["artifact_data_validation_report"]
    )
    paired_superiority_manifest = _read_json(
        resolved_paths["paired_superiority_manifest"]
    )
    attack_manifest = _read_json(resolved_paths["attack_manifest"])
    result_records = _read_jsonl(resolved_paths["result_records"])
    result_record_manifest = _read_json(resolved_paths["result_record_manifest"])
    paired_superiority_summary = _read_json(
        resolved_paths["paired_superiority_summary"]
    )
    observation_path_map = paired_superiority_summary.get(
        "method_observation_source_path_map",
        {},
    )
    if not isinstance(observation_path_map, dict):
        raise TypeError("配对优势 summary 的 observation path map 必须是 JSON 对象")
    paired_observation_records_by_method = {
        str(method_id): (
            _read_jsonl(_resolve_path(root_path, str(source_path)))
            if Path(str(source_path)).suffix.lower() == ".jsonl"
            else _read_json_array(_resolve_path(root_path, str(source_path)))
        )
        for method_id, source_path in sorted(observation_path_map.items())
    }
    existing_source_paths = {path.resolve() for path in resolved_paths.values()}
    nested_source_paths: dict[str, Path] = {}
    artifact_source_paths = recomputed_artifact_data_validation_report.get(
        "source_paths",
        {},
    )
    if not isinstance(artifact_source_paths, dict):
        raise TypeError("论文表图数据验证报告的 source_paths 必须是 JSON 对象")
    for source_id, source_path in sorted(artifact_source_paths.items()):
        resolved_source = _resolve_path(root_path, str(source_path))
        if resolved_source.resolve() not in existing_source_paths:
            nested_source_paths[f"artifact_data_source::{source_id}"] = resolved_source
            existing_source_paths.add(resolved_source.resolve())
    for record in official_reference_fidelity_records:
        baseline_id = str(record.get("baseline_id", ""))
        declared_paths = record.get("official_reference_source_paths", {})
        if not isinstance(declared_paths, dict):
            raise TypeError("官方参考方法忠实度记录的 source paths 必须是 JSON 对象")
        for source_role, source_path in sorted(declared_paths.items()):
            resolved_source = _resolve_path(root_path, str(source_path))
            if resolved_source.resolve() not in existing_source_paths:
                nested_source_paths[
                    f"official_reference_source::{baseline_id}::{source_role}"
                ] = resolved_source
                existing_source_paths.add(resolved_source.resolve())
    paired_input_paths = paired_superiority_manifest.get("input_paths", ())
    if not isinstance(paired_input_paths, list | tuple):
        raise TypeError("配对优势 manifest 的 input_paths 必须是数组")
    for source_index, source_path in enumerate(paired_input_paths):
        resolved_source = _resolve_path(root_path, str(source_path))
        if resolved_source.resolve() not in existing_source_paths:
            nested_source_paths[
                f"paired_superiority_source::{source_index:02d}"
            ] = resolved_source
            existing_source_paths.add(resolved_source.resolve())
    result_record_input_paths = result_record_manifest.get("input_paths", ())
    if not isinstance(result_record_input_paths, list | tuple):
        raise TypeError("正式 result records manifest 的 input_paths 必须是数组")
    result_record_declared_paths = [
        *result_record_input_paths,
        *(
            str(record.get("baseline_result_source", ""))
            for record in result_records
        ),
        *(
            str(evidence_path)
            for record in result_records
            for evidence_path in record.get("evidence_paths", ())
        ),
    ]
    for source_index, source_path in enumerate(result_record_declared_paths):
        if not str(source_path).strip():
            raise ValueError("正式 result record 来源路径不得为空")
        resolved_source = _resolve_path(root_path, str(source_path))
        if resolved_source.resolve() not in existing_source_paths:
            nested_source_paths[
                f"result_record_source::{source_index:04d}"
            ] = resolved_source
            existing_source_paths.add(resolved_source.resolve())
    attack_input_paths = attack_manifest.get("input_paths", ())
    if not isinstance(attack_input_paths, list | tuple):
        raise TypeError("攻击矩阵 manifest 的 input_paths 必须是数组")
    for source_index, source_path in enumerate(attack_input_paths):
        resolved_source = _resolve_path(root_path, str(source_path))
        if resolved_source.resolve() not in existing_source_paths:
            nested_source_paths[
                f"attack_matrix_source::{source_index:02d}"
            ] = resolved_source
            existing_source_paths.add(resolved_source.resolve())
    dataset_quality_resolved_identities: set[Path] = set()
    for source_index, resolution in enumerate(
        dataset_quality_image_resolution_records
    ):
        resolved_image_path = str(resolution.get("resolved_image_path", ""))
        if not resolved_image_path:
            raise ValueError("数据集质量图像解析记录缺少实际图像路径")
        resolved_source = _resolve_path(root_path, resolved_image_path)
        resolved_identity = resolved_source.resolve()
        if resolved_identity in dataset_quality_resolved_identities:
            raise ValueError(
                "数据集质量图像解析记录重复引用同一实际文件"
            )
        dataset_quality_resolved_identities.add(resolved_identity)
        if resolved_source.resolve() not in existing_source_paths:
            nested_source_paths[
                f"dataset_quality_image::{source_index:05d}"
            ] = resolved_source
            existing_source_paths.add(resolved_source.resolve())
    resolved_paths.update(nested_source_paths)

    closure_input_lock = _read_json(resolved_paths["closure_input_lock"])
    closure_input_lock_manifest = _read_json(
        resolved_paths["closure_input_lock_manifest"]
    )
    validate_closure_input_lock_payloads(
        closure_input_lock,
        closure_input_lock_manifest,
        paper_run_name=paper_run.run_name,
        target_fpr=paper_run.target_fpr,
    )
    closure_source_file_sha256 = build_source_file_sha256_map(
        resolved_paths.values(),
        root=root_path,
    )
    closure_source_file_digest = build_stable_digest(closure_source_file_sha256)
    canonical_prompt_ids = canonical_prompt_ids_for_paper_run(
        root_path=root_path,
        prompt_set=paper_run.prompt_set,
        prompt_file=paper_run.prompt_file,
    )
    canonical_prompt_records = build_prompt_records(
        paper_run.prompt_set,
        read_prompt_file(canonical_prompt_path),
    )
    expected_prompt_digest_by_id = {
        record.prompt_id: record.prompt_digest for record in canonical_prompt_records
    }
    canonical_split_prompt_ids = group_prompt_ids_by_split(
        canonical_prompt_records
    )
    expected_prompt_split_by_id = {
        prompt_id: split
        for split, prompt_ids in canonical_split_prompt_ids.items()
        for prompt_id in prompt_ids
    }
    canonical_calibration_prompt_ids = canonical_split_prompt_ids[
        "calibration"
    ]
    canonical_test_prompt_ids = canonical_split_prompt_ids["test"]
    prompt_split_summary = build_pilot_paper_prompt_split_summary(
        canonical_prompt_records,
        build_paper_fixed_fpr_config_from_paper_run(paper_run),
    )
    governed_payload_path_map = {
        "main_comparison_table": _relative_or_absolute(
            resolved_paths["baseline_comparison_table"], root_path
        ),
        "attack_table": _relative_or_absolute(
            resolved_paths["attack_family_metrics"], root_path
        ),
        "quality_table": _relative_or_absolute(
            resolved_paths["dataset_quality_metrics"], root_path
        ),
        "main_confidence_interval_table": _relative_or_absolute(
            resolved_paths["result_analysis_confidence_interval_table"],
            root_path,
        ),
        "per_attack_superiority_table": _relative_or_absolute(
            resolved_paths["result_analysis_per_attack_superiority_table"],
            root_path,
        ),
        "failure_case_records": _relative_or_absolute(
            resolved_paths["result_analysis_failure_case_records"],
            root_path,
        ),
        "failure_case_figure": _relative_or_absolute(
            resolved_paths["result_analysis_failure_case_figure"],
            root_path,
        ),
    }
    if governed_payload_path_map != build_governed_paper_payload_path_map(
        paper_run.run_name
    ):
        raise ValueError("结果闭合读取的论文 payload 路径不是规范仓库相对路径")
    bundle = ResultClosureGateInput(
        expected_paper_claim_scale=paper_run.run_name,
        expected_target_fpr=paper_run.target_fpr,
        expected_prompt_count=paper_run.prompt_count,
        expected_test_count=int(split_counts["test"]),
        expected_prompt_split_digest=str(
            prompt_split_summary["prompt_split_digest"]
        ),
        expected_prompt_id_digest=build_stable_digest(sorted(canonical_prompt_ids)),
        expected_calibration_prompt_id_digest=build_stable_digest(
            sorted(canonical_calibration_prompt_ids)
        ),
        expected_test_prompt_id_digest=build_stable_digest(
            sorted(canonical_test_prompt_ids)
        ),
        expected_prompt_split_by_id=expected_prompt_split_by_id,
        expected_prompt_digest_by_id=expected_prompt_digest_by_id,
        source_file_sha256=closure_source_file_sha256,
        attack_report=_read_json(resolved_paths["attack_report"]),
        attack_detection_records=_read_jsonl(
            resolved_paths["attack_detection_records"]
        ),
        attack_family_metrics=_read_csv(
            resolved_paths["attack_family_metrics"]
        ),
        attacked_image_registry=_read_jsonl(
            resolved_paths["attacked_image_registry"]
        ),
        attack_manifest=attack_manifest,
        threshold_audit_report=_read_json(resolved_paths["threshold_audit_report"]),
        threshold_audit_rows=_read_csv(resolved_paths["threshold_audit_rows"]),
        threshold_audit_manifest=_read_json(resolved_paths["threshold_audit_manifest"]),
        closure_input_lock=closure_input_lock,
        closure_input_lock_manifest=closure_input_lock_manifest,
        official_reference_fidelity_records=official_reference_fidelity_records,
        official_reference_fidelity_summary=_read_json(
            resolved_paths["official_reference_fidelity_summary"]
        ),
        official_reference_fidelity_manifest=_read_json(
            resolved_paths["official_reference_fidelity_manifest"]
        ),
        primary_baseline_evidence_records=_read_jsonl(
            resolved_paths["primary_baseline_evidence_records"]
        ),
        primary_baseline_evidence_summary=_read_json(
            resolved_paths["primary_baseline_evidence_summary"]
        ),
        primary_baseline_evidence_manifest=_read_json(
            resolved_paths["primary_baseline_evidence_manifest"]
        ),
        baseline_report=_read_json(resolved_paths["baseline_report"]),
        baseline_manifest=_read_json(resolved_paths["baseline_manifest"]),
        result_records=result_records,
        result_record_validation_report=_read_json(
            resolved_paths["result_record_validation_report"]
        ),
        result_record_template_coverage=_read_csv(
            resolved_paths["result_record_template_coverage"]
        ),
        result_record_summary=_read_json(resolved_paths["result_record_summary"]),
        result_record_manifest=result_record_manifest,
        common_protocol_summary=_read_json(resolved_paths["common_protocol_summary"]),
        common_protocol_schema=_read_json(resolved_paths["common_protocol_schema"]),
        common_protocol_manifest=_read_json(resolved_paths["common_protocol_manifest"]),
        result_analysis_summary=_read_json(resolved_paths["result_analysis_summary"]),
        result_analysis_manifest=_read_json(resolved_paths["result_analysis_manifest"]),
        paired_observation_records_by_method=(
            paired_observation_records_by_method
        ),
        paired_outcomes=_read_jsonl(resolved_paths["paired_outcomes"]),
        paired_superiority_rows=_read_csv(resolved_paths["paired_superiority_rows"]),
        paired_superiority_summary=paired_superiority_summary,
        paired_superiority_manifest=paired_superiority_manifest,
        ablation_summary=_read_json(resolved_paths["ablation_summary"]),
        ablation_manifest=_read_json(resolved_paths["ablation_manifest"]),
        ablation_runtime_records=_read_jsonl(
            resolved_paths["ablation_runtime_records"]
        ),
        ablation_detection_records=_read_jsonl(
            resolved_paths["ablation_detection_records"]
        ),
        ablation_frozen_protocols=_read_json(
            resolved_paths["ablation_frozen_protocols"]
        ),
        ablation_necessity_rows=_read_csv(
            resolved_paths["ablation_necessity_rows"]
        ),
        ablation_necessity_summary=_read_json(
            resolved_paths["ablation_necessity_summary"]
        ),
        dataset_quality_summary=_read_json(resolved_paths["dataset_quality_summary"]),
        dataset_quality_image_records=_read_jsonl(
            resolved_paths["dataset_quality_image_records"]
        ),
        dataset_quality_image_resolution_records=(
            dataset_quality_image_resolution_records
        ),
        dataset_quality_feature_report=_read_json(
            resolved_paths["dataset_quality_feature_report"]
        ),
        dataset_quality_metrics=_read_csv(resolved_paths["dataset_quality_metrics"]),
        dataset_quality_feature_records=_read_jsonl(
            resolved_paths["dataset_quality_feature_records"]
        ),
        dataset_quality_feature_records_sha256=closure_source_file_sha256[
            _relative_or_absolute(
                resolved_paths["dataset_quality_feature_records"],
                root_path,
            )
        ],
        dataset_quality_manifest=_read_json(resolved_paths["dataset_quality_manifest"]),
        evidence_builder_report=_read_json(resolved_paths["evidence_builder_report"]),
        evidence_blocker_report=_read_json(resolved_paths["evidence_blocker_report"]),
        evidence_audit_runtime_report=(
            recomputed_evidence_audit_bundle.threshold_report
        ),
        evidence_audit_runtime_manifest=(
            recomputed_evidence_audit_bundle.threshold_manifest
        ),
        evidence_audit_source_path_map=dict(
            recomputed_evidence_audit_bundle.source_path_map
        ),
        artifact_data_validation_report=artifact_data_validation_report,
        recomputed_artifact_data_validation_report=(
            recomputed_artifact_data_validation_report
        ),
        evidence_audit_manifest=_read_json(resolved_paths["evidence_audit_manifest"]),
        submission_readiness_report=_read_json(resolved_paths["submission_readiness_report"]),
        submission_readiness_manifest=_read_json(resolved_paths["submission_readiness_manifest"]),
        entry_review_report=_read_json(resolved_paths["entry_review_report"]),
        entry_review_manifest=_read_json(resolved_paths["entry_review_manifest"]),
        result_analysis_governed_payload_path_map=(
            governed_payload_path_map
        ),
        result_analysis_baseline_comparison_rows=_read_csv(
            resolved_paths["baseline_comparison_table"]
        ),
        result_analysis_confidence_interval_rows=_read_csv(
            resolved_paths["result_analysis_confidence_interval_table"]
        ),
        result_analysis_per_attack_superiority_rows=_read_csv(
            resolved_paths["result_analysis_per_attack_superiority_table"]
        ),
        result_analysis_failure_case_rows=_read_jsonl_allow_empty(
            resolved_paths["result_analysis_failure_case_records"]
        ),
        result_analysis_failure_case_svg_text=(
            resolved_paths["result_analysis_failure_case_figure"].read_text(
                encoding="utf-8-sig"
            )
        ),
        result_analysis_failure_figure_path=governed_payload_path_map[
            "failure_case_figure"
        ],
    )
    checks = build_result_closure_gate_checks(bundle)
    report = build_result_closure_gate_report(bundle, checks)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["closure_source_file_sha256"] = closure_source_file_sha256
    report["closure_source_file_digest"] = closure_source_file_digest

    report_path = output_dir / "result_closure_gate_report.json"
    manifest_path = output_dir / "manifest.local.json"
    report_path.write_text(stable_json_text(report), encoding="utf-8")
    input_bundle_digest = build_stable_digest(bundle.to_dict())
    report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    manifest_config = {
        "paper_claim_scale": paper_run.run_name,
        "target_fpr": paper_run.target_fpr,
        "expected_prompt_count": paper_run.prompt_count,
        "expected_test_count": int(split_counts["test"]),
        "expected_prompt_id_digest": bundle.expected_prompt_id_digest,
        "expected_test_prompt_id_digest": bundle.expected_test_prompt_id_digest,
        "input_bundle_digest": input_bundle_digest,
        "report_digest": report_digest,
        "source_artifact_digests": report["source_artifact_digests"],
        "closure_source_file_sha256": closure_source_file_sha256,
        "closure_source_file_digest": closure_source_file_digest,
    }
    manifest = build_artifact_manifest(
        artifact_id=f"{paper_run.run_name}_result_closure_gate_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(_relative_or_absolute(path, root_path) for path in resolved_paths.values()),
        output_paths=(
            _relative_or_absolute(report_path, root_path),
            _relative_or_absolute(manifest_path, root_path),
        ),
        config=manifest_config,
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_result_closure_gate_outputs.py --require-pass",
        metadata={
            "paper_claim_scale": report["paper_claim_scale"],
            "target_fpr": report["target_fpr"],
            "blocked_check_count": report["blocked_check_count"],
            "evidence_closure_allowed": report["evidence_closure_allowed"],
            "result_closure_ready": report["result_closure_ready"],
            "closure_decision": report["closure_decision"],
            "supports_paper_claim": report["supports_paper_claim"],
            "generated_at": report["generated_at"],
            "expected_prompt_id_digest": report["expected_prompt_id_digest"],
            "expected_test_prompt_id_digest": report[
                "expected_test_prompt_id_digest"
            ],
            "input_bundle_digest": input_bundle_digest,
            "report_digest": report_digest,
            "source_artifact_digests": report["source_artifact_digests"],
            "closure_source_file_sha256": closure_source_file_sha256,
            "closure_source_file_digest": closure_source_file_digest,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="写出当前 run 的论文结果闭合语义门禁。")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--attack-report-path", default=None)
    parser.add_argument("--attack-manifest-path", default=None)
    parser.add_argument("--attack-family-metrics-path", default=None)
    parser.add_argument("--threshold-audit-report-path", default=None)
    parser.add_argument("--threshold-audit-rows-path", default=None)
    parser.add_argument("--threshold-audit-manifest-path", default=None)
    parser.add_argument("--dataset-runtime-summary-path", default=None)
    parser.add_argument("--dataset-runtime-manifest-path", default=None)
    parser.add_argument("--closure-input-lock-path", default=None)
    parser.add_argument("--closure-input-lock-manifest-path", default=None)
    parser.add_argument("--official-reference-fidelity-records-path", default=None)
    parser.add_argument("--official-reference-fidelity-summary-path", default=None)
    parser.add_argument("--official-reference-fidelity-manifest-path", default=None)
    parser.add_argument("--primary-baseline-evidence-records-path", default=None)
    parser.add_argument("--primary-baseline-evidence-summary-path", default=None)
    parser.add_argument("--primary-baseline-evidence-manifest-path", default=None)
    parser.add_argument("--baseline-report-path", default=None)
    parser.add_argument("--baseline-manifest-path", default=None)
    parser.add_argument("--result-records-path", default=None)
    parser.add_argument("--result-record-summary-path", default=None)
    parser.add_argument("--result-record-manifest-path", default=None)
    parser.add_argument("--common-protocol-summary-path", default=None)
    parser.add_argument("--common-protocol-schema-path", default=None)
    parser.add_argument("--common-protocol-manifest-path", default=None)
    parser.add_argument("--result-analysis-summary-path", default=None)
    parser.add_argument("--result-analysis-manifest-path", default=None)
    parser.add_argument("--paired-outcomes-path", default=None)
    parser.add_argument("--paired-superiority-rows-path", default=None)
    parser.add_argument("--paired-superiority-summary-path", default=None)
    parser.add_argument("--paired-superiority-manifest-path", default=None)
    parser.add_argument("--ablation-runtime-records-path", default=None)
    parser.add_argument("--ablation-detection-records-path", default=None)
    parser.add_argument("--ablation-frozen-protocols-path", default=None)
    parser.add_argument("--ablation-summary-path", default=None)
    parser.add_argument("--ablation-manifest-path", default=None)
    parser.add_argument("--dataset-quality-summary-path", default=None)
    parser.add_argument("--dataset-quality-image-records-path", default=None)
    parser.add_argument(
        "--dataset-quality-image-resolution-records-path",
        default=None,
    )
    parser.add_argument("--dataset-quality-feature-records-path", default=None)
    parser.add_argument("--dataset-quality-feature-report-path", default=None)
    parser.add_argument("--dataset-quality-metrics-path", default=None)
    parser.add_argument("--dataset-quality-manifest-path", default=None)
    parser.add_argument("--evidence-builder-report-path", default=None)
    parser.add_argument("--evidence-blocker-report-path", default=None)
    parser.add_argument("--artifact-data-validation-report-path", default=None)
    parser.add_argument("--evidence-audit-manifest-path", default=None)
    parser.add_argument("--submission-readiness-report-path", default=None)
    parser.add_argument("--submission-readiness-manifest-path", default=None)
    parser.add_argument("--entry-review-report-path", default=None)
    parser.add_argument("--entry-review-manifest-path", default=None)
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main() -> None:
    """命令行入口, `--require-pass` 在阻断状态下返回非零。"""

    args = build_parser().parse_args()
    report = write_result_closure_gate_outputs(
        root=args.root,
        output_root=args.output_root,
        attack_report_path=args.attack_report_path,
        attack_manifest_path=args.attack_manifest_path,
        attack_family_metrics_path=args.attack_family_metrics_path,
        threshold_audit_report_path=args.threshold_audit_report_path,
        threshold_audit_rows_path=args.threshold_audit_rows_path,
        threshold_audit_manifest_path=args.threshold_audit_manifest_path,
        dataset_runtime_summary_path=args.dataset_runtime_summary_path,
        dataset_runtime_manifest_path=args.dataset_runtime_manifest_path,
        closure_input_lock_path=args.closure_input_lock_path,
        closure_input_lock_manifest_path=args.closure_input_lock_manifest_path,
        official_reference_fidelity_records_path=args.official_reference_fidelity_records_path,
        official_reference_fidelity_summary_path=args.official_reference_fidelity_summary_path,
        official_reference_fidelity_manifest_path=args.official_reference_fidelity_manifest_path,
        primary_baseline_evidence_records_path=args.primary_baseline_evidence_records_path,
        primary_baseline_evidence_summary_path=args.primary_baseline_evidence_summary_path,
        primary_baseline_evidence_manifest_path=args.primary_baseline_evidence_manifest_path,
        baseline_report_path=args.baseline_report_path,
        baseline_manifest_path=args.baseline_manifest_path,
        result_records_path=args.result_records_path,
        result_record_summary_path=args.result_record_summary_path,
        result_record_manifest_path=args.result_record_manifest_path,
        common_protocol_summary_path=args.common_protocol_summary_path,
        common_protocol_schema_path=args.common_protocol_schema_path,
        common_protocol_manifest_path=args.common_protocol_manifest_path,
        result_analysis_summary_path=args.result_analysis_summary_path,
        result_analysis_manifest_path=args.result_analysis_manifest_path,
        paired_outcomes_path=args.paired_outcomes_path,
        paired_superiority_rows_path=args.paired_superiority_rows_path,
        paired_superiority_summary_path=args.paired_superiority_summary_path,
        paired_superiority_manifest_path=args.paired_superiority_manifest_path,
        ablation_runtime_records_path=args.ablation_runtime_records_path,
        ablation_detection_records_path=args.ablation_detection_records_path,
        ablation_frozen_protocols_path=args.ablation_frozen_protocols_path,
        ablation_summary_path=args.ablation_summary_path,
        ablation_manifest_path=args.ablation_manifest_path,
        dataset_quality_summary_path=args.dataset_quality_summary_path,
        dataset_quality_image_records_path=args.dataset_quality_image_records_path,
        dataset_quality_image_resolution_records_path=(
            args.dataset_quality_image_resolution_records_path
        ),
        dataset_quality_feature_records_path=args.dataset_quality_feature_records_path,
        dataset_quality_feature_report_path=args.dataset_quality_feature_report_path,
        dataset_quality_metrics_path=args.dataset_quality_metrics_path,
        dataset_quality_manifest_path=args.dataset_quality_manifest_path,
        evidence_builder_report_path=args.evidence_builder_report_path,
        evidence_blocker_report_path=args.evidence_blocker_report_path,
        artifact_data_validation_report_path=args.artifact_data_validation_report_path,
        evidence_audit_manifest_path=args.evidence_audit_manifest_path,
        submission_readiness_report_path=args.submission_readiness_report_path,
        submission_readiness_manifest_path=args.submission_readiness_manifest_path,
        entry_review_report_path=args.entry_review_report_path,
        entry_review_manifest_path=args.entry_review_manifest_path,
    )
    print(stable_json_text(report), end="")
    if args.require_pass and not report["result_closure_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
