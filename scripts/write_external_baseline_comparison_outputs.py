"""写出外部 baseline 共同协议对比产物。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.baselines import (
    aggregate_baseline_metrics,
    aggregate_slm_proxy_metrics,
    build_baseline_observations,
    build_comparison_rows,
    default_baseline_specs,
    load_baseline_source_registry,
    normalize_baseline_result_record,
    overlay_specs_with_source_registry,
    validate_primary_baseline_formal_import_rows,
)
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest

CONSTRUCTION_UNIT_NAME = "external_baseline_comparison"
DEFAULT_OUTPUT_DIR = Path("outputs/external_baseline_comparison")
DEFAULT_ATTACK_MANIFEST_PATH = Path("outputs/attack_matrix/attack_manifest.json")
DEFAULT_ATTACK_FAMILY_METRICS_PATH = Path("outputs/attack_matrix/attack_family_metrics.csv")
DEFAULT_ATTACK_MATRIX_MANIFEST_PATH = Path("outputs/attack_matrix/manifest.local.json")
DEFAULT_THRESHOLD_REPORT_PATH = Path("outputs/threshold_calibration/threshold_degeneracy_report.json")
DEFAULT_BASELINE_RESULT_RECORDS_PATH = Path("outputs/external_baseline_results/baseline_result_records.jsonl")
DEFAULT_FORMAL_IMPORT_READINESS_SUMMARY_PATH = Path(
    "outputs/external_baseline_results/baseline_formal_import_readiness_summary.json"
)
DEFAULT_BASELINE_SOURCE_REGISTRY_PATH = Path("external_baseline/source_registry.json")


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: dict[str, Any]) -> str:
    """把字典转换为 JSONL 单行。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def read_optional_json(path: Path) -> dict[str, Any]:
    """读取可选 JSON 文件; 文件缺失时返回空字典。"""

    if not path.exists():
        return {}
    return read_json(path)


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
    formal_import_readiness_summary_path: str,
) -> dict[str, Any]:
    """构造外部 baseline 对比运行摘要。"""
    baseline_count = len(baseline_metric_rows)
    ready_count = sum(1 for row in baseline_metric_rows if row["metric_status"] != "unsupported")
    official_source_ready_count = sum(1 for row in baseline_metric_rows if row["baseline_official_code_ready"])
    protocol_compatible_count = sum(1 for row in baseline_metric_rows if row["baseline_protocol_compatible"])
    unsupported_reasons = sorted({row["unsupported_reason"] for row in baseline_metric_rows if row["unsupported_reason"]})
    primary_formal_ready = bool(formal_import_readiness_summary.get("primary_baseline_formal_ready", False))
    return {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_count": baseline_count,
        "baseline_observation_count": len(observations),
        "comparable_baseline_count": protocol_compatible_count,
        "official_source_ready_count": official_source_ready_count,
        "imported_baseline_result_count": imported_result_count,
        "formal_import_input_record_count": int(formal_import_validation.get("input_record_count", 0)),
        "accepted_formal_import_count": int(formal_import_validation.get("accepted_formal_import_count", 0)),
        "rejected_formal_import_count": int(formal_import_validation.get("rejected_formal_import_count", 0)),
        "formal_import_issue_count": int(formal_import_validation.get("formal_import_issue_count", 0)),
        "formal_import_validation_ready": bool(formal_import_validation.get("formal_import_validation_ready", False)),
        "formal_import_readiness_summary_path": formal_import_readiness_summary_path,
        "primary_baseline_formal_ready": primary_formal_ready,
        "formal_result_ready_count": int(formal_import_readiness_summary.get("formal_result_ready_count", 0)),
        "blocked_primary_baseline_ids": list(formal_import_readiness_summary.get("blocked_primary_baseline_ids", ())),
        "dominant_formal_import_blocking_reasons": list(
            formal_import_readiness_summary.get("dominant_blocking_reasons", ())
        ),
        "baseline_result_ready_count": ready_count,
        "comparison_protocol_ready": bool(attack_manifest.get("attack_metrics_ready"))
        and not threshold_report.get("threshold_degenerate", True),
        "baseline_results_ready": ready_count == baseline_count and baseline_count > 0,
        "baseline_source_registry_ready": bool(source_registry.get("baseline_sources")),
        "unsupported_reasons": unsupported_reasons,
        "attack_manifest_supports_paper_claim": bool(attack_manifest.get("supports_paper_claim", False)),
        "full_method_claim_ready": False,
        "supports_paper_claim": False,
    }


def write_external_baseline_comparison_outputs(
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    attack_manifest_path: str | Path = DEFAULT_ATTACK_MANIFEST_PATH,
    attack_family_metrics_path: str | Path = DEFAULT_ATTACK_FAMILY_METRICS_PATH,
    attack_matrix_manifest_path: str | Path = DEFAULT_ATTACK_MATRIX_MANIFEST_PATH,
    threshold_report_path: str | Path = DEFAULT_THRESHOLD_REPORT_PATH,
    baseline_result_records_path: str | Path = DEFAULT_BASELINE_RESULT_RECORDS_PATH,
    formal_import_readiness_summary_path: str | Path = DEFAULT_FORMAL_IMPORT_READINESS_SUMMARY_PATH,
    baseline_source_registry_path: str | Path = DEFAULT_BASELINE_SOURCE_REGISTRY_PATH,
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
    resolved_formal_import_readiness_summary_path = resolve_input_path(root_path, formal_import_readiness_summary_path)
    resolved_baseline_source_registry_path = resolve_input_path(root_path, baseline_source_registry_path)

    attack_manifest = read_json(resolved_attack_manifest_path)
    attack_matrix_manifest = read_json(resolved_attack_matrix_manifest_path)
    threshold_report = read_json(resolved_threshold_report_path)
    attack_rows = read_csv_rows(resolved_attack_family_metrics_path)
    formal_import_readiness_summary = read_optional_json(resolved_formal_import_readiness_summary_path)
    baseline_source_registry = load_baseline_source_registry(resolved_baseline_source_registry_path)
    baseline_specs = overlay_specs_with_source_registry(
        default_baseline_specs(),
        baseline_source_registry,
        root=root_path,
    )
    baseline_result_rows = read_jsonl_rows(resolved_baseline_result_records_path)
    boundary = attack_manifest.get("evaluation_boundary", {})
    target_fpr = float(boundary.get("target_fpr", 0.05))
    formal_import_validation = validate_primary_baseline_formal_import_rows(
        baseline_result_rows,
        evidence_root=root_path,
        target_fpr=target_fpr,
        require_existing_evidence=True,
    )
    baseline_result_records = [
        normalize_baseline_result_record(row) for row in formal_import_validation.get("accepted_records", [])
    ]

    observations = build_baseline_observations(baseline_specs, attack_rows, boundary, baseline_result_records)
    baseline_metric_rows = aggregate_baseline_metrics(observations)
    slm_proxy_metrics = aggregate_slm_proxy_metrics(attack_rows)
    comparison_rows = build_comparison_rows(slm_proxy_metrics, baseline_metric_rows)
    runtime_report = build_runtime_report(
        attack_manifest,
        threshold_report,
        baseline_metric_rows,
        observations,
        baseline_source_registry,
        len(baseline_result_records),
        formal_import_validation,
        formal_import_readiness_summary,
        relative_or_absolute(resolved_formal_import_readiness_summary_path, root_path)
        if resolved_formal_import_readiness_summary_path.exists()
        else "",
    )

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
            "quality_score_proxy_mean",
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
            "quality_score_proxy_mean",
            "score_retention_mean",
            "supports_paper_claim",
        ],
    )
    runtime_report_path.write_text(stable_json_text(runtime_report), encoding="utf-8")
    formal_import_validation_path.write_text(stable_json_text(formal_import_validation), encoding="utf-8")

    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            observations_path,
            imported_records_path,
            metrics_path,
            comparison_path,
            runtime_report_path,
            formal_import_validation_path,
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
    if resolved_formal_import_readiness_summary_path.exists():
        input_path_candidates.append(relative_or_absolute(resolved_formal_import_readiness_summary_path, root_path))
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
            "formal_import_readiness_summary_path": relative_or_absolute(
                resolved_formal_import_readiness_summary_path,
                root_path,
            ),
            "formal_import_validation_report_path": relative_or_absolute(formal_import_validation_path, root_path),
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
        "--formal-import-readiness-summary-path",
        default=str(DEFAULT_FORMAL_IMPORT_READINESS_SUMMARY_PATH),
        help="主表 baseline 正式导入 readiness 摘要路径; 缺失时只使用 validator 摘要。",
    )
    parser.add_argument(
        "--baseline-source-registry-path",
        default=str(DEFAULT_BASELINE_SOURCE_REGISTRY_PATH),
        help="外部 baseline 官方源码登记 JSON 路径; 缺失时仅使用默认 spec。",
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
        formal_import_readiness_summary_path=args.formal_import_readiness_summary_path,
        baseline_source_registry_path=args.baseline_source_registry_path,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
