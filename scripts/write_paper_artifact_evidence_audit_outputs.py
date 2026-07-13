"""写出论文图表与声明证据审计产物。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.runners.paper_claim_provenance import (
    require_exact9_randomization_aggregate_provenance,
)
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.runtime.repository_environment import resolve_code_version
from paper_experiments.analysis.paper_evidence_audit import (
    AuditInputBundle,
    build_evidence_audit_manifest_config,
    build_evidence_audit_materialization,
)
from paper_experiments.analysis.paper_artifact_data_validation import (
    validate_paper_artifact_source_data,
)
from experiments.protocol.paper_run_config import (
    PaperRunPromptContract,
    build_paper_run_config,
)

CONSTRUCTION_UNIT_NAME = "paper_artifact_evidence_audit"
DEFAULT_OUTPUT_ROOT = Path("outputs/paper_artifact_evidence_audit")
DEFAULT_RUNTIME_ROOT = Path("outputs/image_only_dataset_runtime")
DEFAULT_THRESHOLD_AUDIT_ROOT = Path("outputs/fixed_fpr_threshold_audit")
DEFAULT_ATTACK_MATRIX_ROOT = Path("outputs/attack_matrix")
DEFAULT_BASELINE_COMPARISON_ROOT = Path("outputs/external_baseline_comparison")
DEFAULT_DATASET_QUALITY_ROOT = Path("outputs/dataset_level_quality")
DEFAULT_ABLATION_ROOT = Path("outputs/formal_mechanism_ablation")


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定文本。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> tuple[dict[str, Any], ...]:
    """读取受治理 CSV 的全部数据行。"""

    with path.open(encoding="utf-8-sig", newline="") as stream:
        return tuple(dict(row) for row in csv.DictReader(stream))


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
        raise ValueError("论文证据审计输出目录必须位于 outputs/ 下") from exc
    return resolved_output_dir


def resolve_input_path(root_path: Path, path: str | Path) -> Path:
    """解析输入路径。"""
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """将路径尽量转为相对仓库根目录的字符串。"""
    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_input_bundle(
    root_path: Path,
    threshold_report_path: Path,
    threshold_manifest_path: Path,
    threshold_audit_report_path: Path,
    threshold_audit_manifest_path: Path,
    attack_manifest_path: Path,
    attack_matrix_manifest_path: Path,
    baseline_manifest_path: Path,
    baseline_runtime_report_path: Path,
    dataset_quality_manifest_path: Path,
    dataset_quality_summary_path: Path,
    ablation_manifest_path: Path,
    ablation_component_summary_path: Path,
) -> AuditInputBundle:
    """读取上游产物并构造审计输入包。"""
    runtime_dir = threshold_report_path.parent
    threshold_audit_dir = threshold_audit_report_path.parent
    attack_dir = attack_manifest_path.parent
    baseline_dir = baseline_runtime_report_path.parent
    ablation_dir = ablation_component_summary_path.parent
    dataset_quality_dir = dataset_quality_summary_path.parent

    def governed_path(path: Path) -> str:
        """把输入同目录的证据文件写成可迁移路径。"""

        return relative_or_absolute(path, root_path)

    threshold_report = read_json(threshold_report_path)
    threshold_manifest = read_json(threshold_manifest_path)
    threshold_audit_report = read_json(threshold_audit_report_path)
    threshold_audit_manifest = read_json(threshold_audit_manifest_path)
    attack_manifest = read_json(attack_manifest_path)
    attack_matrix_manifest = read_json(attack_matrix_manifest_path)
    baseline_manifest = read_json(baseline_manifest_path)
    baseline_runtime_report = read_json(baseline_runtime_report_path)
    dataset_quality_manifest = read_json(dataset_quality_manifest_path)
    dataset_quality_summary = read_json(dataset_quality_summary_path)
    ablation_manifest = read_json(ablation_manifest_path)
    ablation_component_summary = read_json(ablation_component_summary_path)
    ablation_necessity_path = ablation_dir / "mechanism_necessity_statistics.csv"
    ablation_necessity_summary_path = (
        ablation_dir / "mechanism_necessity_summary.json"
    )
    ablation_necessity_rows = (
        read_csv_rows(ablation_necessity_path)
        if ablation_necessity_path.is_file()
        else ()
    )
    ablation_necessity_summary = (
        read_json(ablation_necessity_summary_path)
        if ablation_necessity_summary_path.is_file()
        else {}
    )
    source_data_paths = {
        "frozen_evidence_protocol_ready": runtime_dir / "frozen_evidence_protocol.json",
        "raw_image_only_detection_records_ready": (
            runtime_dir / "image_only_detection_records.jsonl"
        ),
        "test_detection_metrics_ready": runtime_dir / "test_detection_metrics.csv",
        "score_distribution_table_ready": runtime_dir / "score_distribution_table.csv",
        "roc_curve_points_ready": runtime_dir / "roc_curve_points.csv",
        "det_curve_points_ready": runtime_dir / "det_curve_points.csv",
        "attack_family_metrics_ready": attack_dir / "attack_family_metrics.csv",
        "baseline_comparison_table_ready": baseline_dir / "baseline_comparison_table.csv",
        "mechanism_ablation_metrics_ready": ablation_dir / "mechanism_ablation_metrics.csv",
        "mechanism_pairwise_delta_ready": ablation_dir / "mechanism_pairwise_delta.csv",
        "mechanism_necessity_statistics_ready": ablation_necessity_path,
        "dataset_quality_metrics_ready": dataset_quality_dir / "dataset_quality_metrics.csv",
    }
    artifact_data_validation = validate_paper_artifact_source_data(
        root_path=root_path,
        source_paths=source_data_paths,
        threshold_report=threshold_report,
        attack_manifest=attack_manifest,
        baseline_runtime_report=baseline_runtime_report,
        dataset_quality_summary=dataset_quality_summary,
        ablation_component_summary=ablation_component_summary,
    )

    return AuditInputBundle(
        threshold_report=threshold_report,
        threshold_manifest=threshold_manifest,
        threshold_audit_report=threshold_audit_report,
        threshold_audit_manifest=threshold_audit_manifest,
        attack_manifest=attack_manifest,
        attack_matrix_manifest=attack_matrix_manifest,
        baseline_manifest=baseline_manifest,
        baseline_runtime_report=baseline_runtime_report,
        dataset_quality_manifest=dataset_quality_manifest,
        dataset_quality_summary=dataset_quality_summary,
        ablation_manifest=ablation_manifest,
        ablation_component_summary=ablation_component_summary,
        source_path_map={
            "threshold_report": governed_path(threshold_report_path),
            "threshold_audit_report": governed_path(threshold_audit_report_path),
            "threshold_audit_rows": governed_path(threshold_audit_dir / "threshold_audit_rows.csv"),
            "fixed_fpr_operating_points": governed_path(runtime_dir / "frozen_evidence_protocol.json"),
            "standard_watermark_metrics": governed_path(runtime_dir / "test_detection_metrics.csv"),
            "quality_metrics_summary": governed_path(runtime_dir / "runtime_results.jsonl"),
            "raw_image_only_detection_records": governed_path(
                runtime_dir / "image_only_detection_records.jsonl"
            ),
            "dataset_quality_summary": governed_path(dataset_quality_summary_path),
            "dataset_quality_metrics": governed_path(dataset_quality_dir / "dataset_quality_metrics.csv"),
            "score_distribution_table": governed_path(runtime_dir / "score_distribution_table.csv"),
            "roc_curve_points": governed_path(runtime_dir / "roc_curve_points.csv"),
            "det_curve_points": governed_path(runtime_dir / "det_curve_points.csv"),
            "attack_manifest": governed_path(attack_manifest_path),
            "attack_family_metrics": governed_path(attack_dir / "attack_family_metrics.csv"),
            "attack_strength_curve": governed_path(attack_dir / "attack_strength_curve.csv"),
            "score_retention_by_attack": governed_path(attack_dir / "score_retention_by_attack.csv"),
            "attacked_image_root": governed_path(runtime_dir / "runs"),
            "attacked_image_registry": governed_path(attack_dir / "attacked_image_registry.jsonl"),
            "baseline_runtime_report": governed_path(baseline_runtime_report_path),
            "baseline_comparison_table": governed_path(baseline_dir / "baseline_comparison_table.csv"),
            "ablation_component_summary": governed_path(ablation_component_summary_path),
            "mechanism_ablation_table": governed_path(ablation_dir / "mechanism_ablation_metrics.csv"),
            "method_pairwise_delta_table": governed_path(ablation_dir / "mechanism_pairwise_delta.csv"),
            "mechanism_necessity_statistics": governed_path(
                ablation_necessity_path
            ),
            "mechanism_necessity_summary": governed_path(
                ablation_necessity_summary_path
            ),
        },
        artifact_data_validation=artifact_data_validation,
        ablation_necessity_rows=ablation_necessity_rows,
        ablation_necessity_summary=ablation_necessity_summary,
    )


def write_paper_artifact_evidence_audit_outputs(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """在精确9重复原始记录重算 Writer 就绪前拒绝正式结论物化."""

    require_exact9_randomization_aggregate_provenance()


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="写出论文图表与声明证据审计产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录; 默认写入当前论文运行子目录, 且必须位于 outputs/ 下。",
    )
    parser.add_argument(
        "--threshold-report-path",
        default=None,
        help="阈值边界报告路径; 默认读取当前论文运行子目录。",
    )
    parser.add_argument(
        "--threshold-manifest-path",
        default=None,
        help="阈值校准 manifest 路径; 默认读取当前论文运行子目录。",
    )
    parser.add_argument(
        "--threshold-audit-report-path",
        default=None,
        help="五方法统一 fixed-FPR 阈值审计报告路径; 默认读取当前论文运行子目录。",
    )
    parser.add_argument(
        "--threshold-audit-manifest-path",
        default=None,
        help="五方法统一 fixed-FPR 阈值审计 manifest 路径; 默认读取当前论文运行子目录。",
    )
    parser.add_argument("--attack-manifest-path", default=None, help="攻击矩阵专用 manifest; 默认读取当前论文运行子目录。")
    parser.add_argument("--attack-matrix-manifest-path", default=None, help="攻击矩阵产物 manifest; 默认读取当前论文运行子目录。")
    parser.add_argument("--baseline-manifest-path", default=None, help="外部 baseline 产物 manifest; 默认读取当前论文运行子目录。")
    parser.add_argument("--baseline-runtime-report-path", default=None, help="外部 baseline runtime report; 默认读取当前论文运行子目录。")
    parser.add_argument("--dataset-quality-manifest-path", default=None, help="数据集级质量 manifest; 默认读取当前论文运行子目录。")
    parser.add_argument("--dataset-quality-summary-path", default=None, help="数据集级质量 summary; 默认读取当前论文运行子目录。")
    parser.add_argument("--ablation-manifest-path", default=None, help="内部消融 manifest; 默认读取当前论文运行子目录。")
    parser.add_argument("--ablation-claim-summary-path", default=None, help="内部消融 claim summary; 默认读取当前论文运行子目录。")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    manifest = write_paper_artifact_evidence_audit_outputs(
        root=args.root,
        output_dir=args.output_dir,
        threshold_report_path=args.threshold_report_path,
        threshold_manifest_path=args.threshold_manifest_path,
        threshold_audit_report_path=args.threshold_audit_report_path,
        threshold_audit_manifest_path=args.threshold_audit_manifest_path,
        attack_manifest_path=args.attack_manifest_path,
        attack_matrix_manifest_path=args.attack_matrix_manifest_path,
        baseline_manifest_path=args.baseline_manifest_path,
        baseline_runtime_report_path=args.baseline_runtime_report_path,
        dataset_quality_manifest_path=args.dataset_quality_manifest_path,
        dataset_quality_summary_path=args.dataset_quality_summary_path,
        ablation_manifest_path=args.ablation_manifest_path,
        ablation_component_summary_path=args.ablation_component_summary_path,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
