"""写出外部 baseline 共同协议对比产物。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.baselines import (
    aggregate_baseline_metrics,
    aggregate_slm_metrics,
    build_baseline_observations,
    build_comparison_rows,
    build_primary_baseline_formal_evidence_path_summary,
    build_primary_baseline_formal_evidence_collection_rows,
    build_primary_baseline_formal_evidence_collection_summary,
    build_primary_baseline_formal_import_readiness_rows,
    build_primary_baseline_formal_import_readiness_summary,
    build_primary_baseline_formal_template_coverage_rows,
    build_primary_baseline_formal_template_coverage_summary,
    build_primary_baseline_execution_plans,
    build_primary_result_templates,
    default_baseline_specs,
    load_baseline_source_registry,
    normalize_baseline_result_record,
    overlay_specs_with_source_registry,
    validate_primary_baseline_formal_import_rows,
)
from experiments.protocol.pilot_paper_fixed_fpr import PILOT_PAPER_FIXED_FPR
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest

CONSTRUCTION_UNIT_NAME = "external_baseline_comparison"
DEFAULT_OUTPUT_DIR = Path("outputs/external_baseline_comparison")
DEFAULT_ATTACK_MANIFEST_PATH = Path("outputs/attack_matrix/attack_manifest.json")
DEFAULT_ATTACK_FAMILY_METRICS_PATH = Path("outputs/attack_matrix/attack_family_metrics.csv")
DEFAULT_ATTACK_MATRIX_MANIFEST_PATH = Path("outputs/attack_matrix/manifest.local.json")
DEFAULT_THRESHOLD_REPORT_PATH = Path("outputs/threshold_calibration/threshold_degeneracy_report.json")
DEFAULT_BASELINE_RESULT_RECORDS_PATH = Path("outputs/external_baseline_results/baseline_result_records.jsonl")
DEFAULT_BASELINE_SOURCE_REGISTRY_PATH = Path("external_baseline/source_registry.json")
DEFAULT_EVIDENCE_SEARCH_ROOTS_ENV = "SLM_WM_EVIDENCE_SEARCH_ROOTS"


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: dict[str, Any]) -> str:
    """把字典转换为 JSONL 单行。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """读取 CSV 表格。"""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 记录; 文件缺失时返回空记录集合。"""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def parse_evidence_search_roots(values: Iterable[str | Path] | None = None) -> tuple[str, ...]:
    """解析外部 evidence 镜像根目录.

    该函数属于配置解析层: 它把命令行参数或环境变量统一收敛为路径列表, 避免在正式比较逻辑中硬编码 Google Drive 等机器相关目录。
    """

    if values is not None:
        return tuple(str(value) for value in values if str(value).strip())
    raw_value = os.environ.get(DEFAULT_EVIDENCE_SEARCH_ROOTS_ENV, "")
    return tuple(part.strip() for part in raw_value.split(";") if part.strip())


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写出 CSV 表格。"""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_code_version(root_path: Path) -> str:
    """读取 Git 提交标识, 工作区有变更时附加 dirty 标记。"""
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--short"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "git_version_unavailable"
    commit_id = commit_result.stdout.strip()
    if not commit_id:
        return "git_version_unavailable"
    return f"{commit_id}-dirty" if status_result.stdout.strip() else commit_id


def ensure_output_dir_under_outputs(root_path: Path, output_dir: Path) -> Path:
    """确保持久化输出目录位于 outputs 下。"""
    resolved_output_dir = (root_path / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved_output_dir.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("外部 baseline 对比输出目录必须位于 outputs/ 下") from exc
    return resolved_output_dir


def resolve_input_path(root_path: Path, path: str | Path) -> Path:
    """解析输入路径。"""
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """将路径尽量转换为相对仓库根目录的字符串。"""
    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_runtime_report(
    attack_manifest: dict[str, Any],
    threshold_report: dict[str, Any],
    baseline_metric_rows: list[dict[str, Any]],
    observations: tuple[dict[str, Any], ...],
    source_registry: dict[str, Any],
    imported_result_count: int,
    formal_import_validation: dict[str, Any],
    formal_import_readiness_summary: dict[str, Any],
    formal_template_coverage_summary: dict[str, Any],
    formal_evidence_collection_summary: dict[str, Any],
    formal_evidence_path_summary: dict[str, Any],
    formal_evidence_path_summary_path: str,
) -> dict[str, Any]:
    """构造外部 baseline 对比运行摘要。"""
    baseline_count = len(baseline_metric_rows)
    ready_count = sum(1 for row in baseline_metric_rows if row["metric_status"] != "unsupported")
    primary_rows = tuple(row for row in baseline_metric_rows if row.get("comparison_group") == "primary")
    supplemental_rows = tuple(row for row in baseline_metric_rows if row.get("comparison_group") == "supplemental")
    primary_ready_count = sum(1 for row in primary_rows if row["metric_status"] != "unsupported")
    supplemental_ready_count = sum(1 for row in supplemental_rows if row["metric_status"] != "unsupported")
    official_source_ready_count = sum(1 for row in baseline_metric_rows if row["baseline_official_code_ready"])
    protocol_compatible_count = sum(1 for row in baseline_metric_rows if row["baseline_protocol_compatible"])
    unsupported_reasons = sorted({row["unsupported_reason"] for row in baseline_metric_rows if row["unsupported_reason"]})
    primary_formal_ready = (
        bool(formal_import_readiness_summary.get("primary_baseline_formal_ready", False))
        and bool(formal_template_coverage_summary.get("primary_baseline_formal_template_coverage_ready", False))
        and bool(formal_evidence_collection_summary.get("primary_baseline_formal_evidence_collection_ready", False))
    )
    comparison_protocol_ready = (
        bool(attack_manifest.get("attack_metrics_ready"))
        and bool(attack_manifest.get("supports_paper_claim", False))
        and not threshold_report.get("threshold_degenerate", True)
    )
    primary_baseline_results_ready = bool(primary_rows) and all(
        row["metric_status"] != "unsupported"
        and int(row["baseline_result_ready_count"]) == int(row["baseline_observation_count"])
        for row in primary_rows
    )
    comparison_table_supports_paper_claim = (
        comparison_protocol_ready
        and primary_formal_ready
        and primary_baseline_results_ready
        and bool(formal_import_validation.get("formal_import_validation_ready", False))
        and bool(formal_evidence_path_summary.get("formal_evidence_path_resolution_ready", False))
    )
    return {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_count": baseline_count,
        "primary_baseline_count": len(primary_rows),
        "supplemental_baseline_count": len(supplemental_rows),
        "baseline_observation_count": len(observations),
        "comparable_baseline_count": protocol_compatible_count,
        "official_source_ready_count": official_source_ready_count,
        "imported_baseline_result_count": imported_result_count,
        "formal_import_input_record_count": int(formal_import_validation.get("input_record_count", 0)),
        "accepted_formal_import_count": int(formal_import_validation.get("accepted_formal_import_count", 0)),
        "rejected_formal_import_count": int(formal_import_validation.get("rejected_formal_import_count", 0)),
        "formal_import_issue_count": int(formal_import_validation.get("formal_import_issue_count", 0)),
        "formal_import_validation_ready": bool(formal_import_validation.get("formal_import_validation_ready", False)),
        "primary_baseline_formal_ready": primary_formal_ready,
        "formal_result_ready_count": int(formal_import_readiness_summary.get("formal_result_ready_count", 0)),
        "blocked_primary_baseline_ids": list(formal_import_readiness_summary.get("blocked_primary_baseline_ids", ())),
        "dominant_formal_import_blocking_reasons": list(
            formal_import_readiness_summary.get("dominant_blocking_reasons", ())
        ),
        "formal_template_record_count": int(formal_template_coverage_summary.get("formal_template_record_count", 0)),
        "candidate_template_match_count": int(formal_template_coverage_summary.get("candidate_template_match_count", 0)),
        "accepted_template_match_count": int(formal_template_coverage_summary.get("accepted_template_match_count", 0)),
        "formal_template_coverage_ready_count": int(
            formal_template_coverage_summary.get("formal_template_coverage_ready_count", 0)
        ),
        "missing_candidate_template_count": int(
            formal_template_coverage_summary.get("missing_candidate_template_count", 0)
        ),
        "missing_formal_template_count": int(formal_template_coverage_summary.get("missing_formal_template_count", 0)),
        "unexpected_candidate_record_count": int(
            formal_template_coverage_summary.get("unexpected_candidate_record_count", 0)
        ),
        "unexpected_accepted_record_count": int(
            formal_template_coverage_summary.get("unexpected_accepted_record_count", 0)
        ),
        "duplicate_candidate_template_count": int(
            formal_template_coverage_summary.get("duplicate_candidate_template_count", 0)
        ),
        "duplicate_accepted_template_count": int(
            formal_template_coverage_summary.get("duplicate_accepted_template_count", 0)
        ),
        "primary_baseline_formal_template_coverage_ready": bool(
            formal_template_coverage_summary.get("primary_baseline_formal_template_coverage_ready", False)
        ),
        "formal_evidence_collection_task_count": int(
            formal_evidence_collection_summary.get("formal_evidence_collection_task_count", 0)
        ),
        "ready_formal_evidence_collection_task_count": int(
            formal_evidence_collection_summary.get("ready_formal_evidence_collection_task_count", 0)
        ),
        "missing_formal_evidence_collection_task_count": int(
            formal_evidence_collection_summary.get("missing_formal_evidence_collection_task_count", 0)
        ),
        "primary_baseline_formal_evidence_collection_ready": bool(
            formal_evidence_collection_summary.get("primary_baseline_formal_evidence_collection_ready", False)
        ),
        "formal_evidence_path_resolution_report_path": formal_evidence_path_summary_path,
        "formal_evidence_path_reference_count": int(
            formal_evidence_path_summary.get("formal_evidence_path_reference_count", 0)
        ),
        "existing_formal_evidence_path_count": int(
            formal_evidence_path_summary.get("existing_formal_evidence_path_count", 0)
        ),
        "direct_formal_evidence_path_count": int(
            formal_evidence_path_summary.get("direct_formal_evidence_path_count", 0)
        ),
        "search_resolved_formal_evidence_path_count": int(
            formal_evidence_path_summary.get("search_resolved_formal_evidence_path_count", 0)
        ),
        "missing_formal_evidence_path_count": int(
            formal_evidence_path_summary.get("missing_formal_evidence_path_count", 0)
        ),
        "formal_evidence_path_resolution_ready": bool(
            formal_evidence_path_summary.get("formal_evidence_path_resolution_ready", False)
        ),
        "evidence_search_roots": list(formal_evidence_path_summary.get("evidence_search_roots", ())),
        "formal_evidence_path_missing_baseline_ids": list(
            formal_evidence_path_summary.get("formal_evidence_path_missing_baseline_ids", ())
        ),
        "baseline_result_ready_count": ready_count,
        "primary_baseline_result_ready_count": primary_ready_count,
        "supplemental_baseline_result_ready_count": supplemental_ready_count,
        "primary_baseline_results_ready": primary_baseline_results_ready,
        "supplemental_baseline_results_ready": bool(supplemental_rows) and supplemental_ready_count == len(supplemental_rows),
        "comparison_protocol_ready": comparison_protocol_ready,
        "baseline_results_ready": ready_count == baseline_count and baseline_count > 0,
        "comparison_table_supports_paper_claim": comparison_table_supports_paper_claim,
        "baseline_source_registry_ready": bool(source_registry.get("baseline_sources")),
        "unsupported_reasons": unsupported_reasons,
        "attack_manifest_supports_paper_claim": bool(attack_manifest.get("supports_paper_claim", False)),
        "full_method_claim_ready": False,
        "supports_paper_claim": comparison_table_supports_paper_claim,
    }


def align_comparison_table_claim_scope(
    baseline_metric_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    runtime_report: dict[str, Any],
) -> None:
    """统一外部 baseline 对比表的 claim 标记口径。

    该函数只在外部 baseline 的主表结果、共同攻击协议、fixed-FPR 协议和 evidence
    路径解析全部通过后, 把主表 comparison 行标记为可支撑论文主张。补充 baseline
    若缺失正式结果, 仍保持 unsupported, 避免把补充表缺口误计入主表结论。
    """

    comparison_claim_ready = bool(runtime_report.get("comparison_table_supports_paper_claim", False))
    for row in baseline_metric_rows:
        row["supports_paper_claim"] = (
            comparison_claim_ready
            and row.get("comparison_group") == "primary"
            and row.get("metric_status") != "unsupported"
        )
    for row in comparison_rows:
        method_id = row.get("method_id", "")
        method_role = row.get("method_role", "")
        measured = row.get("metric_status") != "unsupported"
        if method_id == "slm_wm_current":
            row["method_role"] = "proposed_method_governed_result" if comparison_claim_ready else method_role
            row["comparison_scope"] = "common_protocol_governed_result" if comparison_claim_ready else row[
                "comparison_scope"
            ]
            row["metric_status"] = (
                "measured_from_attack_matrix_formal_records" if comparison_claim_ready else row["metric_status"]
            )
            row["supports_paper_claim"] = comparison_claim_ready and measured
        elif method_role == "external_baseline_primary":
            row["supports_paper_claim"] = comparison_claim_ready and measured
        else:
            row["supports_paper_claim"] = False


def write_external_baseline_comparison_outputs(
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    attack_manifest_path: str | Path = DEFAULT_ATTACK_MANIFEST_PATH,
    attack_family_metrics_path: str | Path = DEFAULT_ATTACK_FAMILY_METRICS_PATH,
    attack_matrix_manifest_path: str | Path = DEFAULT_ATTACK_MATRIX_MANIFEST_PATH,
    threshold_report_path: str | Path = DEFAULT_THRESHOLD_REPORT_PATH,
    baseline_result_records_path: str | Path = DEFAULT_BASELINE_RESULT_RECORDS_PATH,
    baseline_source_registry_path: str | Path = DEFAULT_BASELINE_SOURCE_REGISTRY_PATH,
    evidence_search_roots: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """写出外部 baseline 对比 records, 表格, 运行报告与 manifest。"""
    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, Path(output_dir))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_attack_manifest_path = resolve_input_path(root_path, attack_manifest_path)
    resolved_attack_family_metrics_path = resolve_input_path(root_path, attack_family_metrics_path)
    resolved_attack_matrix_manifest_path = resolve_input_path(root_path, attack_matrix_manifest_path)
    resolved_threshold_report_path = resolve_input_path(root_path, threshold_report_path)
    resolved_baseline_result_records_path = resolve_input_path(root_path, baseline_result_records_path)
    resolved_baseline_source_registry_path = resolve_input_path(root_path, baseline_source_registry_path)
    resolved_evidence_search_roots = parse_evidence_search_roots(evidence_search_roots)

    attack_manifest = read_json(resolved_attack_manifest_path)
    attack_matrix_manifest = read_json(resolved_attack_matrix_manifest_path)
    threshold_report = read_json(resolved_threshold_report_path)
    attack_rows = read_csv_rows(resolved_attack_family_metrics_path)
    baseline_source_registry = load_baseline_source_registry(resolved_baseline_source_registry_path)
    baseline_specs = overlay_specs_with_source_registry(
        default_baseline_specs(),
        baseline_source_registry,
        root=root_path,
    )
    baseline_result_rows = read_jsonl_rows(resolved_baseline_result_records_path)
    boundary = attack_manifest.get("evaluation_boundary", {})
    target_fpr = float(boundary.get("target_fpr", PILOT_PAPER_FIXED_FPR))
    formal_import_validation = validate_primary_baseline_formal_import_rows(
        baseline_result_rows,
        evidence_root=root_path,
        target_fpr=target_fpr,
        require_existing_evidence=True,
        evidence_search_roots=resolved_evidence_search_roots,
        allowed_resource_profiles=("full_main", "full_extra"),
    )
    baseline_result_records = [
        normalize_baseline_result_record(row) for row in formal_import_validation.get("accepted_records", [])
    ]
    execution_plans = build_primary_baseline_execution_plans(baseline_source_registry, root=root_path)
    formal_template_rows = build_primary_result_templates(execution_plans, attack_rows, boundary)
    formal_import_readiness_rows = build_primary_baseline_formal_import_readiness_rows(
        baseline_result_rows,
        formal_import_validation,
    )
    formal_import_readiness_summary = build_primary_baseline_formal_import_readiness_summary(
        formal_import_readiness_rows
    )
    formal_template_coverage_rows = build_primary_baseline_formal_template_coverage_rows(
        formal_template_rows,
        baseline_result_rows,
        formal_import_validation,
    )
    formal_template_coverage_summary = build_primary_baseline_formal_template_coverage_summary(
        formal_template_coverage_rows
    )
    formal_evidence_collection_rows = build_primary_baseline_formal_evidence_collection_rows(
        formal_template_rows,
        baseline_result_rows,
        formal_import_validation,
    )
    formal_evidence_collection_summary = build_primary_baseline_formal_evidence_collection_summary(
        formal_evidence_collection_rows
    )
    formal_evidence_path_summary = build_primary_baseline_formal_evidence_path_summary(
        baseline_result_rows,
        evidence_root=root_path,
        evidence_search_roots=resolved_evidence_search_roots,
    )

    observations = build_baseline_observations(baseline_specs, attack_rows, boundary, baseline_result_records)
    baseline_metric_rows = aggregate_baseline_metrics(observations)
    slm_metrics = aggregate_slm_metrics(attack_rows)
    comparison_rows = build_comparison_rows(slm_metrics, baseline_metric_rows)
    formal_evidence_path_summary_path = (
        resolved_output_dir / "baseline_formal_evidence_path_resolution_report.json"
    )
    runtime_report = build_runtime_report(
        attack_manifest,
        threshold_report,
        baseline_metric_rows,
        observations,
        baseline_source_registry,
        len(baseline_result_records),
        formal_import_validation,
        formal_import_readiness_summary,
        formal_template_coverage_summary,
        formal_evidence_collection_summary,
        formal_evidence_path_summary,
        relative_or_absolute(formal_evidence_path_summary_path, root_path),
    )
    align_comparison_table_claim_scope(baseline_metric_rows, comparison_rows, runtime_report)

    observations_path = resolved_output_dir / "baseline_observations.jsonl"
    imported_records_path = resolved_output_dir / "baseline_result_records.jsonl"
    formal_import_validation_path = resolved_output_dir / "baseline_formal_import_validation_report.json"
    metrics_path = resolved_output_dir / "baseline_metrics.csv"
    comparison_path = resolved_output_dir / "baseline_comparison_table.csv"
    runtime_report_path = resolved_output_dir / "baseline_runtime_report.json"
    manifest_path = resolved_output_dir / "manifest.local.json"

    observations_path.write_text("".join(json_line(row) for row in observations), encoding="utf-8")
    imported_records_path.write_text(
        "".join(json_line(record.to_dict()) for record in baseline_result_records),
        encoding="utf-8",
    )
    write_csv(
        metrics_path,
        baseline_metric_rows,
        [
            "baseline_id",
            "baseline_family",
            "baseline_name",
            "comparison_group",
            "baseline_adapter_ready",
            "baseline_official_code_ready",
            "baseline_reproduced_result_ready",
            "baseline_imported_result_ready",
            "baseline_result_source",
            "baseline_protocol_compatible",
            "baseline_requires_gpu",
            "baseline_requires_training",
            "baseline_observation_count",
            "baseline_result_ready_count",
            "unsupported_record_count",
            "metric_status",
            "true_positive_rate",
            "false_positive_rate",
            "clean_false_positive_rate",
            "attacked_false_positive_rate",
            "quality_score_mean",
            "score_retention_mean",
            "unsupported_reason",
            "supports_paper_claim",
        ],
    )
    write_csv(
        comparison_path,
        comparison_rows,
        [
            "method_id",
            "method_role",
            "comparison_scope",
            "common_prompt_protocol_ready",
            "common_attack_protocol_ready",
            "common_threshold_protocol_ready",
            "metric_status",
            "true_positive_rate",
            "false_positive_rate",
            "clean_false_positive_rate",
            "attacked_false_positive_rate",
            "quality_score_mean",
            "score_retention_mean",
            "supports_paper_claim",
        ],
    )
    runtime_report_path.write_text(stable_json_text(runtime_report), encoding="utf-8")
    formal_import_validation_path.write_text(stable_json_text(formal_import_validation), encoding="utf-8")
    formal_evidence_path_summary_path.write_text(stable_json_text(formal_evidence_path_summary), encoding="utf-8")

    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            observations_path,
            imported_records_path,
            metrics_path,
            comparison_path,
            runtime_report_path,
            formal_import_validation_path,
            formal_evidence_path_summary_path,
            manifest_path,
        )
    )
    input_path_candidates = [
        relative_or_absolute(resolved_attack_manifest_path, root_path),
        relative_or_absolute(resolved_attack_family_metrics_path, root_path),
        relative_or_absolute(resolved_attack_matrix_manifest_path, root_path),
        relative_or_absolute(resolved_threshold_report_path, root_path),
    ]
    if resolved_baseline_result_records_path.exists():
        input_path_candidates.append(relative_or_absolute(resolved_baseline_result_records_path, root_path))
    if resolved_baseline_source_registry_path.exists():
        input_path_candidates.append(relative_or_absolute(resolved_baseline_source_registry_path, root_path))
    input_paths = tuple(input_path_candidates)
    summary = {
        "runtime_report": runtime_report,
        "baseline_metrics": baseline_metric_rows,
        "comparison_rows": comparison_rows,
        "attack_matrix_manifest_digest": attack_matrix_manifest.get("config_digest", ""),
        "source_registry_digest": build_stable_digest(baseline_source_registry) if baseline_source_registry else "",
        "imported_result_digest": build_stable_digest([record.to_dict() for record in baseline_result_records])
        if baseline_result_records
        else "",
        "formal_import_validation_digest": build_stable_digest(formal_import_validation),
        "formal_import_readiness_digest": build_stable_digest(formal_import_readiness_rows),
        "formal_template_coverage_digest": build_stable_digest(formal_template_coverage_rows),
        "formal_evidence_collection_digest": build_stable_digest(formal_evidence_collection_rows),
        "formal_evidence_path_summary_digest": build_stable_digest(formal_evidence_path_summary),
    }
    manifest = build_artifact_manifest(
        artifact_id="external_baseline_comparison_manifest",
        artifact_type="local_manifest",
        input_paths=input_paths,
        output_paths=output_paths,
        config={
            "baseline_spec_digest": build_stable_digest([spec.to_dict() for spec in baseline_specs]),
            "summary_digest": build_stable_digest(summary),
            "baseline_source_registry_path": relative_or_absolute(resolved_baseline_source_registry_path, root_path),
            "baseline_result_records_path": relative_or_absolute(resolved_baseline_result_records_path, root_path),
            "formal_import_validation_report_path": relative_or_absolute(formal_import_validation_path, root_path),
            "formal_evidence_path_resolution_report_path": relative_or_absolute(
                formal_evidence_path_summary_path,
                root_path,
            ),
            "evidence_search_roots": list(resolved_evidence_search_roots),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_external_baseline_comparison_outputs.py",
        metadata={
            **runtime_report,
            "baseline_result_ready": runtime_report["baseline_results_ready"],
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="写出外部 baseline 共同协议对比产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--attack-manifest-path", default=str(DEFAULT_ATTACK_MANIFEST_PATH), help="攻击矩阵 manifest 路径。")
    parser.add_argument(
        "--attack-family-metrics-path",
        default=str(DEFAULT_ATTACK_FAMILY_METRICS_PATH),
        help="攻击矩阵 family metrics 表路径。",
    )
    parser.add_argument(
        "--attack-matrix-manifest-path",
        default=str(DEFAULT_ATTACK_MATRIX_MANIFEST_PATH),
        help="攻击矩阵产物 manifest 路径。",
    )
    parser.add_argument("--threshold-report-path", default=str(DEFAULT_THRESHOLD_REPORT_PATH), help="fixed-FPR 边界报告路径。")
    parser.add_argument(
        "--baseline-result-records-path",
        default=str(DEFAULT_BASELINE_RESULT_RECORDS_PATH),
        help="受治理外部 baseline 结果 JSONL 路径; 缺失时保持 unsupported 状态。",
    )
    parser.add_argument(
        "--baseline-source-registry-path",
        default=str(DEFAULT_BASELINE_SOURCE_REGISTRY_PATH),
        help="外部 baseline 官方源码登记 JSON 路径; 缺失时仅使用默认 spec。",
    )
    parser.add_argument(
        "--evidence-search-root",
        action="append",
        default=None,
        help="用于解析外部 baseline evidence paths 的额外镜像根目录; 可重复传入。",
    )
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    manifest = write_external_baseline_comparison_outputs(
        root=args.root,
        output_dir=args.output_dir,
        attack_manifest_path=args.attack_manifest_path,
        attack_family_metrics_path=args.attack_family_metrics_path,
        attack_matrix_manifest_path=args.attack_matrix_manifest_path,
        threshold_report_path=args.threshold_report_path,
        baseline_result_records_path=args.baseline_result_records_path,
        baseline_source_registry_path=args.baseline_source_registry_path,
        evidence_search_roots=args.evidence_search_root,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()

