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
    ablation_claim_summary_path: Path,
) -> AuditInputBundle:
    """读取上游产物并构造审计输入包。"""
    runtime_dir = threshold_report_path.parent
    threshold_audit_dir = threshold_audit_report_path.parent
    attack_dir = attack_manifest_path.parent
    baseline_dir = baseline_runtime_report_path.parent
    ablation_dir = ablation_claim_summary_path.parent
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
    ablation_claim_summary = read_json(ablation_claim_summary_path)
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
        ablation_claim_summary=ablation_claim_summary,
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
        ablation_claim_summary=ablation_claim_summary,
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
            "ablation_claim_summary": governed_path(ablation_claim_summary_path),
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


def write_paper_artifact_evidence_audit_outputs(
    root: str | Path = ".",
    output_dir: str | Path | None = None,
    threshold_report_path: str | Path | None = None,
    threshold_manifest_path: str | Path | None = None,
    threshold_audit_report_path: str | Path | None = None,
    threshold_audit_manifest_path: str | Path | None = None,
    attack_manifest_path: str | Path | None = None,
    attack_matrix_manifest_path: str | Path | None = None,
    baseline_manifest_path: str | Path | None = None,
    baseline_runtime_report_path: str | Path | None = None,
    dataset_quality_manifest_path: str | Path | None = None,
    dataset_quality_summary_path: str | Path | None = None,
    ablation_manifest_path: str | Path | None = None,
    ablation_claim_summary_path: str | Path | None = None,
    prompt_contract: PaperRunPromptContract | None = None,
) -> dict[str, Any]:
    """写出论文证据审计表、readiness 表、gap 清单、阻断报告与 manifest。"""
    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(
        root_path,
        prompt_contract=prompt_contract,
    )
    resolved_output_dir = ensure_output_dir_under_outputs(
        root_path,
        Path(output_dir or DEFAULT_OUTPUT_ROOT / paper_run.run_name),
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    runtime_dir = DEFAULT_RUNTIME_ROOT / paper_run.run_name
    threshold_audit_dir = DEFAULT_THRESHOLD_AUDIT_ROOT / paper_run.run_name
    formal_ablation_dir = DEFAULT_ABLATION_ROOT / paper_run.run_name
    dataset_quality_dir = DEFAULT_DATASET_QUALITY_ROOT / paper_run.run_name
    attack_matrix_dir = DEFAULT_ATTACK_MATRIX_ROOT / paper_run.run_name
    baseline_comparison_dir = DEFAULT_BASELINE_COMPARISON_ROOT / paper_run.run_name
    threshold_report_path = threshold_report_path or runtime_dir / "dataset_runtime_summary.json"
    threshold_manifest_path = threshold_manifest_path or runtime_dir / "manifest.local.json"
    threshold_audit_report_path = threshold_audit_report_path or threshold_audit_dir / "threshold_audit_report.json"
    threshold_audit_manifest_path = threshold_audit_manifest_path or threshold_audit_dir / "manifest.local.json"
    attack_manifest_path = attack_manifest_path or attack_matrix_dir / "attack_manifest.json"
    attack_matrix_manifest_path = attack_matrix_manifest_path or attack_matrix_dir / "manifest.local.json"
    baseline_manifest_path = baseline_manifest_path or baseline_comparison_dir / "manifest.local.json"
    baseline_runtime_report_path = (
        baseline_runtime_report_path or baseline_comparison_dir / "baseline_runtime_report.json"
    )
    ablation_manifest_path = ablation_manifest_path or formal_ablation_dir / "manifest.local.json"
    ablation_claim_summary_path = (
        ablation_claim_summary_path or formal_ablation_dir / "ablation_claim_summary.json"
    )
    dataset_quality_manifest_path = dataset_quality_manifest_path or dataset_quality_dir / "manifest.local.json"
    dataset_quality_summary_path = (
        dataset_quality_summary_path or dataset_quality_dir / "dataset_quality_summary.json"
    )

    resolved_threshold_report_path = resolve_input_path(root_path, threshold_report_path)
    resolved_threshold_manifest_path = resolve_input_path(root_path, threshold_manifest_path)
    resolved_threshold_audit_report_path = resolve_input_path(root_path, threshold_audit_report_path)
    resolved_threshold_audit_manifest_path = resolve_input_path(root_path, threshold_audit_manifest_path)
    resolved_attack_manifest_path = resolve_input_path(root_path, attack_manifest_path)
    resolved_attack_matrix_manifest_path = resolve_input_path(root_path, attack_matrix_manifest_path)
    resolved_baseline_manifest_path = resolve_input_path(root_path, baseline_manifest_path)
    resolved_baseline_runtime_report_path = resolve_input_path(root_path, baseline_runtime_report_path)
    resolved_dataset_quality_manifest_path = resolve_input_path(root_path, dataset_quality_manifest_path)
    resolved_dataset_quality_summary_path = resolve_input_path(root_path, dataset_quality_summary_path)
    resolved_ablation_manifest_path = resolve_input_path(root_path, ablation_manifest_path)
    resolved_ablation_claim_summary_path = resolve_input_path(root_path, ablation_claim_summary_path)

    bundle = build_input_bundle(
        root_path,
        resolved_threshold_report_path,
        resolved_threshold_manifest_path,
        resolved_threshold_audit_report_path,
        resolved_threshold_audit_manifest_path,
        resolved_attack_manifest_path,
        resolved_attack_matrix_manifest_path,
        resolved_baseline_manifest_path,
        resolved_baseline_runtime_report_path,
        resolved_dataset_quality_manifest_path,
        resolved_dataset_quality_summary_path,
        resolved_ablation_manifest_path,
        resolved_ablation_claim_summary_path,
    )
    materialization = build_evidence_audit_materialization(bundle)
    claim_rows = materialization["claim_rows"]
    table_rows = materialization["table_rows"]
    figure_rows = materialization["figure_rows"]
    gap_rows = materialization["gap_rows"]
    builder_report = materialization["builder_report"]
    blocker_report = materialization["blocker_report"]
    dry_run_report = {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run_decision": (
            "pass"
            if builder_report["artifact_builder_ready"]
            and bundle.artifact_data_validation["artifact_data_validation_ready"]
            else "fail"
        ),
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
    artifact_data_validation_path = resolved_output_dir / "artifact_data_validation_report.json"
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
    artifact_data_validation_path.write_text(
        stable_json_text(bundle.artifact_data_validation),
        encoding="utf-8",
    )

    primary_input_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            resolved_threshold_report_path,
            resolved_threshold_manifest_path,
            resolved_threshold_audit_report_path,
            resolved_threshold_audit_manifest_path,
            resolved_attack_manifest_path,
            resolved_attack_matrix_manifest_path,
            resolved_baseline_manifest_path,
            resolved_baseline_runtime_report_path,
            resolved_dataset_quality_manifest_path,
            resolved_dataset_quality_summary_path,
            resolved_ablation_manifest_path,
            resolved_ablation_claim_summary_path,
        )
    )
    input_paths = tuple(
        dict.fromkeys(
            (
                *primary_input_paths,
                *bundle.artifact_data_validation["source_paths"].values(),
            )
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
            artifact_data_validation_path,
            manifest_path,
        )
    )
    manifest = build_artifact_manifest(
        artifact_id="paper_artifact_evidence_audit_manifest",
        artifact_type="local_manifest",
        input_paths=input_paths,
        output_paths=output_paths,
        config=build_evidence_audit_manifest_config(bundle, materialization),
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_paper_artifact_evidence_audit_outputs.py",
        metadata={
            **blocker_report,
            "artifact_data_validation_ready": bundle.artifact_data_validation[
                "artifact_data_validation_ready"
            ],
            "blocked_artifact_data_ids": bundle.artifact_data_validation[
                "blocked_artifact_data_ids"
            ],
            "evidence_source_file_sha256": bundle.artifact_data_validation[
                "evidence_source_file_sha256"
            ],
            "raw_image_only_detection_records_ready": bundle.artifact_data_validation[
                "raw_image_only_detection_records_ready"
            ],
            "raw_image_only_detection_records_sha256": bundle.artifact_data_validation[
                "raw_image_only_detection_records_sha256"
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


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
        ablation_claim_summary_path=args.ablation_claim_summary_path,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
