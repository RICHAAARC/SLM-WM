"""写出外部 baseline 共同协议对比产物。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.runners.paper_claim_provenance import (
    require_exact9_randomization_aggregate_provenance,
)
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
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.runtime.repository_environment import resolve_code_version
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest

CONSTRUCTION_UNIT_NAME = "external_baseline_comparison"
DEFAULT_OUTPUT_ROOT = Path("outputs/external_baseline_comparison")
DEFAULT_ATTACK_MATRIX_ROOT = Path("outputs/attack_matrix")
DEFAULT_THRESHOLD_AUDIT_ROOT = Path("outputs/fixed_fpr_threshold_audit")
DEFAULT_BASELINE_RESULT_ROOT = Path("outputs/external_baseline_results")
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
    boundary = attack_manifest.get("evaluation_boundary", {})
    target_fpr = boundary.get("target_fpr") if isinstance(boundary, dict) else None
    threshold_audit_ready = (
        threshold_report.get("fixed_fpr_threshold_audit_ready") is True
        and threshold_report.get("all_method_thresholds_ready") is True
        and threshold_report.get("supports_paper_claim") is True
        and target_fpr is not None
        and math.isclose(
            float(threshold_report.get("target_fpr", math.nan)),
            float(target_fpr),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    comparison_protocol_ready = (
        bool(attack_manifest.get("attack_metrics_ready"))
        and bool(attack_manifest.get("supports_paper_claim", False))
        and threshold_audit_ready
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
        "paper_claim_scale": str(attack_manifest.get("paper_run_name", "")),
        "target_fpr": target_fpr,
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
        "fixed_fpr_threshold_audit_ready": threshold_audit_ready,
        "baseline_results_ready": ready_count == baseline_count and baseline_count > 0,
        "comparison_table_supports_paper_claim": comparison_table_supports_paper_claim,
        "baseline_source_registry_ready": bool(source_registry.get("baseline_sources")),
        "unsupported_reasons": unsupported_reasons,
        "attack_manifest_supports_paper_claim": bool(attack_manifest.get("supports_paper_claim", False)),
        "full_method_component_ready": False,
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


def write_external_baseline_comparison_outputs(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """在精确9重复原始记录重算 Writer 就绪前拒绝正式结论物化."""

    require_exact9_randomization_aggregate_provenance()


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="写出外部 baseline 共同协议对比产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录; 默认写入当前论文运行子目录, 且必须位于 outputs/ 下。",
    )
    parser.add_argument(
        "--attack-manifest-path",
        default=None,
        help="攻击矩阵 manifest 路径; 默认读取当前论文运行子目录。",
    )
    parser.add_argument(
        "--attack-family-metrics-path",
        default=None,
        help="攻击矩阵 family metrics 表路径; 默认读取当前论文运行子目录。",
    )
    parser.add_argument(
        "--attack-matrix-manifest-path",
        default=None,
        help="攻击矩阵产物 manifest 路径; 默认读取当前论文运行子目录。",
    )
    parser.add_argument(
        "--threshold-report-path",
        default=None,
        help="五方法统一 fixed-FPR 阈值审计报告; 默认读取当前论文运行子目录。",
    )
    parser.add_argument(
        "--baseline-result-records-path",
        default=None,
        help="受治理外部 baseline 结果 JSONL 路径; 默认读取当前论文运行子目录。",
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

