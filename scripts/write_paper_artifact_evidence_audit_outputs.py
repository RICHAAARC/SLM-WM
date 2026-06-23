"""写出论文图表与声明证据审计产物。"""

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

from main.analysis.artifact_manifest import build_artifact_manifest
from main.analysis.paper_evidence_audit import (
    AuditInputBundle,
    build_builder_readiness_report,
    build_claim_audit_rows,
    build_evidence_gap_rows,
    build_figure_readiness_rows,
    build_submission_blocker_report,
    build_table_readiness_rows,
)
from main.core.digest import build_stable_digest

CONSTRUCTION_UNIT_NAME = "paper_artifact_evidence_audit"
DEFAULT_OUTPUT_DIR = Path("outputs/paper_artifact_evidence_audit")
DEFAULT_THRESHOLD_REPORT_PATH = Path("outputs/threshold_calibration/threshold_degeneracy_report.json")
DEFAULT_THRESHOLD_MANIFEST_PATH = Path("outputs/threshold_calibration/manifest.local.json")
DEFAULT_ATTACK_MANIFEST_PATH = Path("outputs/attack_matrix/attack_manifest.json")
DEFAULT_ATTACK_MATRIX_MANIFEST_PATH = Path("outputs/attack_matrix/manifest.local.json")
DEFAULT_BASELINE_MANIFEST_PATH = Path("outputs/external_baseline_comparison/manifest.local.json")
DEFAULT_BASELINE_RUNTIME_REPORT_PATH = Path("outputs/external_baseline_comparison/baseline_runtime_report.json")
DEFAULT_BASELINE_SMALL_SAMPLE_MANIFEST_PATH = Path("outputs/primary_baseline_small_sample_evidence/manifest.local.json")
DEFAULT_BASELINE_SMALL_SAMPLE_SUMMARY_PATH = Path(
    "outputs/primary_baseline_small_sample_evidence/primary_baseline_small_sample_evidence_summary.json"
)
DEFAULT_DATASET_QUALITY_MANIFEST_PATH = Path("outputs/dataset_level_quality/manifest.local.json")
DEFAULT_DATASET_QUALITY_SUMMARY_PATH = Path("outputs/dataset_level_quality/dataset_quality_summary.json")
DEFAULT_ABLATION_MANIFEST_PATH = Path("outputs/internal_ablation_evidence/manifest.local.json")
DEFAULT_ABLATION_CLAIM_SUMMARY_PATH = Path("outputs/internal_ablation_evidence/ablation_claim_summary.json")


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定文本。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


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
    threshold_report_path: Path,
    threshold_manifest_path: Path,
    attack_manifest_path: Path,
    attack_matrix_manifest_path: Path,
    baseline_manifest_path: Path,
    baseline_runtime_report_path: Path,
    baseline_small_sample_manifest_path: Path,
    baseline_small_sample_summary_path: Path,
    dataset_quality_manifest_path: Path,
    dataset_quality_summary_path: Path,
    ablation_manifest_path: Path,
    ablation_claim_summary_path: Path,
) -> AuditInputBundle:
    """读取上游产物并构造审计输入包。"""
    return AuditInputBundle(
        threshold_report=read_json(threshold_report_path),
        threshold_manifest=read_json(threshold_manifest_path),
        attack_manifest=read_json(attack_manifest_path),
        attack_matrix_manifest=read_json(attack_matrix_manifest_path),
        baseline_manifest=read_json(baseline_manifest_path),
        baseline_runtime_report=read_json(baseline_runtime_report_path),
        baseline_small_sample_manifest=read_json(baseline_small_sample_manifest_path),
        baseline_small_sample_summary=read_json(baseline_small_sample_summary_path),
        dataset_quality_manifest=read_json(dataset_quality_manifest_path),
        dataset_quality_summary=read_json(dataset_quality_summary_path),
        ablation_manifest=read_json(ablation_manifest_path),
        ablation_claim_summary=read_json(ablation_claim_summary_path),
        source_path_map={
            "threshold_report": "outputs/threshold_calibration/threshold_degeneracy_report.json",
            "fixed_fpr_operating_points": "outputs/threshold_calibration/fixed_fpr_operating_points.csv",
            "standard_watermark_metrics": "outputs/threshold_calibration/standard_watermark_metrics.csv",
            "quality_metrics_summary": "outputs/threshold_calibration/quality_metrics_summary.csv",
            "dataset_quality_summary": "outputs/dataset_level_quality/dataset_quality_summary.json",
            "dataset_quality_metrics": "outputs/dataset_level_quality/dataset_quality_metrics.csv",
            "score_distribution_table": "outputs/threshold_calibration/score_distribution_table.csv",
            "roc_curve_points": "outputs/threshold_calibration/roc_curve_points.csv",
            "det_curve_points": "outputs/threshold_calibration/det_curve_points.csv",
            "attack_manifest": "outputs/attack_matrix/attack_manifest.json",
            "attack_family_metrics": "outputs/attack_matrix/attack_family_metrics.csv",
            "attack_strength_curve": "outputs/attack_matrix/attack_strength_curve.csv",
            "score_retention_by_attack": "outputs/attack_matrix/score_retention_by_attack.csv",
            "attacked_image_root": "outputs/attack_matrix/attacked_images",
            "attacked_image_registry": "outputs/attack_matrix/attacked_image_registry.jsonl",
            "baseline_runtime_report": "outputs/external_baseline_comparison/baseline_runtime_report.json",
            "baseline_comparison_table": "outputs/external_baseline_comparison/baseline_comparison_table.csv",
            "baseline_small_sample_summary": "outputs/primary_baseline_small_sample_evidence/primary_baseline_small_sample_evidence_summary.json",
            "baseline_small_sample_records": "outputs/primary_baseline_small_sample_evidence/primary_baseline_small_sample_evidence_records.jsonl",
            "baseline_small_sample_comparison_table": "outputs/primary_baseline_small_sample_evidence/primary_baseline_small_sample_comparison_table.csv",
            "ablation_claim_summary": "outputs/internal_ablation_evidence/ablation_claim_summary.json",
            "mechanism_ablation_table": "outputs/internal_ablation_evidence/mechanism_ablation_table.csv",
            "method_pairwise_delta_table": "outputs/internal_ablation_evidence/method_pairwise_delta_table.csv",
        },
    )


def write_paper_artifact_evidence_audit_outputs(
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    threshold_report_path: str | Path = DEFAULT_THRESHOLD_REPORT_PATH,
    threshold_manifest_path: str | Path = DEFAULT_THRESHOLD_MANIFEST_PATH,
    attack_manifest_path: str | Path = DEFAULT_ATTACK_MANIFEST_PATH,
    attack_matrix_manifest_path: str | Path = DEFAULT_ATTACK_MATRIX_MANIFEST_PATH,
    baseline_manifest_path: str | Path = DEFAULT_BASELINE_MANIFEST_PATH,
    baseline_runtime_report_path: str | Path = DEFAULT_BASELINE_RUNTIME_REPORT_PATH,
    baseline_small_sample_manifest_path: str | Path = DEFAULT_BASELINE_SMALL_SAMPLE_MANIFEST_PATH,
    baseline_small_sample_summary_path: str | Path = DEFAULT_BASELINE_SMALL_SAMPLE_SUMMARY_PATH,
    dataset_quality_manifest_path: str | Path = DEFAULT_DATASET_QUALITY_MANIFEST_PATH,
    dataset_quality_summary_path: str | Path = DEFAULT_DATASET_QUALITY_SUMMARY_PATH,
    ablation_manifest_path: str | Path = DEFAULT_ABLATION_MANIFEST_PATH,
    ablation_claim_summary_path: str | Path = DEFAULT_ABLATION_CLAIM_SUMMARY_PATH,
) -> dict[str, Any]:
    """写出论文证据审计表、readiness 表、gap 清单、阻断报告与 manifest。"""
    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, Path(output_dir))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_threshold_report_path = resolve_input_path(root_path, threshold_report_path)
    resolved_threshold_manifest_path = resolve_input_path(root_path, threshold_manifest_path)
    resolved_attack_manifest_path = resolve_input_path(root_path, attack_manifest_path)
    resolved_attack_matrix_manifest_path = resolve_input_path(root_path, attack_matrix_manifest_path)
    resolved_baseline_manifest_path = resolve_input_path(root_path, baseline_manifest_path)
    resolved_baseline_runtime_report_path = resolve_input_path(root_path, baseline_runtime_report_path)
    resolved_baseline_small_sample_manifest_path = resolve_input_path(root_path, baseline_small_sample_manifest_path)
    resolved_baseline_small_sample_summary_path = resolve_input_path(root_path, baseline_small_sample_summary_path)
    resolved_dataset_quality_manifest_path = resolve_input_path(root_path, dataset_quality_manifest_path)
    resolved_dataset_quality_summary_path = resolve_input_path(root_path, dataset_quality_summary_path)
    resolved_ablation_manifest_path = resolve_input_path(root_path, ablation_manifest_path)
    resolved_ablation_claim_summary_path = resolve_input_path(root_path, ablation_claim_summary_path)

    bundle = build_input_bundle(
        resolved_threshold_report_path,
        resolved_threshold_manifest_path,
        resolved_attack_manifest_path,
        resolved_attack_matrix_manifest_path,
        resolved_baseline_manifest_path,
        resolved_baseline_runtime_report_path,
        resolved_baseline_small_sample_manifest_path,
        resolved_baseline_small_sample_summary_path,
        resolved_dataset_quality_manifest_path,
        resolved_dataset_quality_summary_path,
        resolved_ablation_manifest_path,
        resolved_ablation_claim_summary_path,
    )
    claim_rows = build_claim_audit_rows(bundle)
    table_rows = build_table_readiness_rows(bundle)
    figure_rows = build_figure_readiness_rows(bundle)
    gap_rows = build_evidence_gap_rows(bundle)
    builder_report = build_builder_readiness_report(claim_rows, table_rows, figure_rows)
    blocker_report = build_submission_blocker_report(claim_rows, gap_rows, builder_report)
    dry_run_report = {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run_decision": "pass" if builder_report["artifact_builder_ready"] else "fail",
        "artifact_builder_readiness_report": builder_report,
        "submission_blocker_report": blocker_report,
        "supports_paper_claim": False,
    }

    claim_audit_path = resolved_output_dir / "claim_audit_table.csv"
    table_readiness_path = resolved_output_dir / "paper_table_readiness.csv"
    figure_readiness_path = resolved_output_dir / "paper_figure_readiness.csv"
    gap_list_path = resolved_output_dir / "evidence_gap_list.csv"
    builder_report_path = resolved_output_dir / "artifact_builder_readiness_report.json"
    dry_run_report_path = resolved_output_dir / "evidence_audit_dry_run.json"
    blocker_report_path = resolved_output_dir / "submission_blocker_report.json"
    manifest_path = resolved_output_dir / "manifest.local.json"

    write_csv(
        claim_audit_path,
        claim_rows,
        [
            "claim_id",
            "claim_scope",
            "claim_text",
            "claim_decision",
            "evidence_path",
            "blocker_count",
            "primary_blocker",
            "paper_claim_supported",
            "supports_paper_claim",
        ],
    )
    write_csv(
        table_readiness_path,
        table_rows,
        ["audit_item_id", "artifact_kind", "artifact_name", "source_paths", "builder_status", "paper_ready", "blocker_count", "primary_blocker", "supports_paper_claim"],
    )
    write_csv(
        figure_readiness_path,
        figure_rows,
        ["audit_item_id", "artifact_kind", "artifact_name", "source_paths", "builder_status", "paper_ready", "blocker_count", "primary_blocker", "supports_paper_claim"],
    )
    write_csv(
        gap_list_path,
        gap_rows,
        ["gap_id", "gap_area", "blocker_severity", "required_action", "related_artifacts", "closes_claim_ids", "recommended_order", "supports_paper_claim"],
    )
    builder_report_path.write_text(stable_json_text(builder_report), encoding="utf-8")
    dry_run_report_path.write_text(stable_json_text(dry_run_report), encoding="utf-8")
    blocker_report_path.write_text(stable_json_text(blocker_report), encoding="utf-8")

    input_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            resolved_threshold_report_path,
            resolved_threshold_manifest_path,
            resolved_attack_manifest_path,
            resolved_attack_matrix_manifest_path,
            resolved_baseline_manifest_path,
            resolved_baseline_runtime_report_path,
            resolved_baseline_small_sample_manifest_path,
            resolved_baseline_small_sample_summary_path,
            resolved_dataset_quality_manifest_path,
            resolved_dataset_quality_summary_path,
            resolved_ablation_manifest_path,
            resolved_ablation_claim_summary_path,
        )
    )
    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            claim_audit_path,
            table_readiness_path,
            figure_readiness_path,
            gap_list_path,
            builder_report_path,
            dry_run_report_path,
            blocker_report_path,
            manifest_path,
        )
    )
    summary = {
        "claim_rows": claim_rows,
        "table_rows": table_rows,
        "figure_rows": figure_rows,
        "gap_rows": gap_rows,
        "builder_report": builder_report,
        "blocker_report": blocker_report,
    }
    manifest = build_artifact_manifest(
        artifact_id="paper_artifact_evidence_audit_manifest",
        artifact_type="local_manifest",
        input_paths=input_paths,
        output_paths=output_paths,
        config={
            "summary_digest": build_stable_digest(summary),
            "input_bundle_digest": build_stable_digest(bundle.to_dict()),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_paper_artifact_evidence_audit_outputs.py",
        metadata={
            **blocker_report,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="写出论文图表与声明证据审计产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--threshold-report-path", default=str(DEFAULT_THRESHOLD_REPORT_PATH), help="阈值边界报告路径。")
    parser.add_argument("--threshold-manifest-path", default=str(DEFAULT_THRESHOLD_MANIFEST_PATH), help="阈值校准 manifest 路径。")
    parser.add_argument("--attack-manifest-path", default=str(DEFAULT_ATTACK_MANIFEST_PATH), help="攻击矩阵专用 manifest 路径。")
    parser.add_argument("--attack-matrix-manifest-path", default=str(DEFAULT_ATTACK_MATRIX_MANIFEST_PATH), help="攻击矩阵产物 manifest 路径。")
    parser.add_argument("--baseline-manifest-path", default=str(DEFAULT_BASELINE_MANIFEST_PATH), help="外部 baseline 产物 manifest 路径。")
    parser.add_argument("--baseline-runtime-report-path", default=str(DEFAULT_BASELINE_RUNTIME_REPORT_PATH), help="外部 baseline runtime report 路径。")
    parser.add_argument(
        "--baseline-small-sample-manifest-path",
        default=str(DEFAULT_BASELINE_SMALL_SAMPLE_MANIFEST_PATH),
        help="外部 baseline 小样本证据 manifest 路径。",
    )
    parser.add_argument(
        "--baseline-small-sample-summary-path",
        default=str(DEFAULT_BASELINE_SMALL_SAMPLE_SUMMARY_PATH),
        help="外部 baseline 小样本证据 summary 路径。",
    )
    parser.add_argument("--dataset-quality-manifest-path", default=str(DEFAULT_DATASET_QUALITY_MANIFEST_PATH), help="数据集级质量 manifest 路径。")
    parser.add_argument("--dataset-quality-summary-path", default=str(DEFAULT_DATASET_QUALITY_SUMMARY_PATH), help="数据集级质量 summary 路径。")
    parser.add_argument("--ablation-manifest-path", default=str(DEFAULT_ABLATION_MANIFEST_PATH), help="内部消融 manifest 路径。")
    parser.add_argument("--ablation-claim-summary-path", default=str(DEFAULT_ABLATION_CLAIM_SUMMARY_PATH), help="内部消融 claim summary 路径。")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    manifest = write_paper_artifact_evidence_audit_outputs(
        root=args.root,
        output_dir=args.output_dir,
        threshold_report_path=args.threshold_report_path,
        threshold_manifest_path=args.threshold_manifest_path,
        attack_manifest_path=args.attack_manifest_path,
        attack_matrix_manifest_path=args.attack_matrix_manifest_path,
        baseline_manifest_path=args.baseline_manifest_path,
        baseline_runtime_report_path=args.baseline_runtime_report_path,
        baseline_small_sample_manifest_path=args.baseline_small_sample_manifest_path,
        baseline_small_sample_summary_path=args.baseline_small_sample_summary_path,
        dataset_quality_manifest_path=args.dataset_quality_manifest_path,
        dataset_quality_summary_path=args.dataset_quality_summary_path,
        ablation_manifest_path=args.ablation_manifest_path,
        ablation_claim_summary_path=args.ablation_claim_summary_path,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
