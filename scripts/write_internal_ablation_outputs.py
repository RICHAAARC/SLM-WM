"""写出内部机制消融证据产物。"""

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

from experiments.ablations import (
    aggregate_ablation_by_attack_family,
    aggregate_mechanism_ablation_table,
    build_ablation_claim_summary,
    build_ablation_records,
    build_pairwise_delta_rows,
    default_ablation_specs,
    filter_ablation_claim_input_records,
)
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest

CONSTRUCTION_UNIT_NAME = "internal_ablation_evidence"
DEFAULT_OUTPUT_DIR = Path("outputs/internal_ablation_evidence")
DEFAULT_ATTACK_RECORDS_PATH = Path("outputs/attack_matrix/attack_detection_records.jsonl")
DEFAULT_ATTACK_MANIFEST_PATH = Path("outputs/attack_matrix/attack_manifest.json")
DEFAULT_ATTACK_MATRIX_MANIFEST_PATH = Path("outputs/attack_matrix/manifest.local.json")
DEFAULT_THRESHOLD_REPORT_PATH = Path("outputs/threshold_calibration/threshold_degeneracy_report.json")
DEFAULT_BASELINE_MANIFEST_PATH = Path("outputs/external_baseline_comparison/manifest.local.json")


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定文本。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: dict[str, Any]) -> str:
    """把字典转为 JSONL 单行。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """读取 JSONL 文件。"""
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


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
        raise ValueError("内部消融输出目录必须位于 outputs/ 下") from exc
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


def _threshold_value(attack_manifest: dict[str, Any], threshold_report: dict[str, Any]) -> float:
    """读取内部消融复用的 fixed-FPR 内容阈值。"""
    boundary = attack_manifest.get("evaluation_boundary", {})
    return float(threshold_report.get("calibrated_content_threshold", boundary.get("calibrated_content_threshold", 0.5)))


def write_internal_ablation_outputs(
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    attack_records_path: str | Path = DEFAULT_ATTACK_RECORDS_PATH,
    attack_manifest_path: str | Path = DEFAULT_ATTACK_MANIFEST_PATH,
    attack_matrix_manifest_path: str | Path = DEFAULT_ATTACK_MATRIX_MANIFEST_PATH,
    threshold_report_path: str | Path = DEFAULT_THRESHOLD_REPORT_PATH,
    baseline_manifest_path: str | Path = DEFAULT_BASELINE_MANIFEST_PATH,
) -> dict[str, Any]:
    """写出内部消融 records、表格、声明摘要与 manifest。"""
    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, Path(output_dir))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_attack_records_path = resolve_input_path(root_path, attack_records_path)
    resolved_attack_manifest_path = resolve_input_path(root_path, attack_manifest_path)
    resolved_attack_matrix_manifest_path = resolve_input_path(root_path, attack_matrix_manifest_path)
    resolved_threshold_report_path = resolve_input_path(root_path, threshold_report_path)
    resolved_baseline_manifest_path = resolve_input_path(root_path, baseline_manifest_path)

    attack_records = read_jsonl(resolved_attack_records_path)
    attack_manifest = read_json(resolved_attack_manifest_path)
    attack_matrix_manifest = read_json(resolved_attack_matrix_manifest_path)
    threshold_report = read_json(resolved_threshold_report_path)
    baseline_manifest = read_json(resolved_baseline_manifest_path)
    ablation_specs = default_ablation_specs()
    threshold = _threshold_value(attack_manifest, threshold_report)

    claim_input_records, claim_input_report = filter_ablation_claim_input_records(attack_records)
    ablation_records = build_ablation_records(claim_input_records, ablation_specs, threshold)
    mechanism_rows = aggregate_mechanism_ablation_table(ablation_records)
    pairwise_rows = build_pairwise_delta_rows(mechanism_rows)
    family_rows = aggregate_ablation_by_attack_family(ablation_records)
    claim_summary = {
        **build_ablation_claim_summary(ablation_specs, ablation_records, mechanism_rows, attack_manifest, baseline_manifest),
        "ablation_claim_input_filter": claim_input_report,
        "ablation_claim_total_source_record_count": claim_input_report["ablation_claim_total_source_record_count"],
        "ablation_claim_input_record_count": claim_input_report["ablation_claim_input_record_count"],
        "ablation_claim_excluded_record_count": claim_input_report["ablation_claim_excluded_record_count"],
        "ablation_claim_excluded_proxy_record_count": claim_input_report["ablation_claim_excluded_proxy_record_count"],
        "ablation_claim_excluded_record_examples": claim_input_report["ablation_claim_excluded_record_examples"],
    }

    records_path = resolved_output_dir / "ablation_records.jsonl"
    mechanism_table_path = resolved_output_dir / "mechanism_ablation_table.csv"
    pairwise_delta_path = resolved_output_dir / "method_pairwise_delta_table.csv"
    family_table_path = resolved_output_dir / "ablation_by_attack_family.csv"
    claim_input_report_path = resolved_output_dir / "ablation_claim_input_report.json"
    claim_summary_path = resolved_output_dir / "ablation_claim_summary.json"
    manifest_path = resolved_output_dir / "manifest.local.json"

    records_path.write_text("".join(json_line(row) for row in ablation_records), encoding="utf-8")
    write_csv(
        mechanism_table_path,
        mechanism_rows,
        [
            "ablation_id",
            "ablation_name",
            "mechanism_group",
            "ablated_mechanism",
            "mechanism_change_digest",
            "mechanism_explanation",
            "metric_status",
            "ablation_record_count",
            "supported_record_count",
            "unsupported_record_count",
            "positive_count",
            "negative_count",
            "true_positive_rate",
            "false_positive_rate",
            "clean_false_positive_rate",
            "attacked_false_positive_rate",
            "score_retention_mean",
            "quality_score_proxy_mean",
            "attention_consistency_proxy_mean",
            "geometry_reliable_rate",
            "rescue_rate",
            "attestation_available_rate",
            "true_positive_delta_from_full",
            "false_positive_delta_from_full",
            "score_retention_delta_from_full",
            "quality_delta_from_full",
            "degradation_chain_rank",
            "supports_paper_claim",
        ],
    )
    write_csv(
        pairwise_delta_path,
        pairwise_rows,
        [
            "ablation_id",
            "compared_to_ablation_id",
            "metric_name",
            "full_metric_value",
            "ablated_metric_value",
            "delta_value",
            "degradation_direction",
            "mechanism_interpretation",
            "supports_paper_claim",
        ],
    )
    write_csv(
        family_table_path,
        family_rows,
        [
            "ablation_id",
            "ablation_name",
            "mechanism_group",
            "attack_family",
            "metric_status",
            "ablation_record_count",
            "supported_record_count",
            "unsupported_record_count",
            "positive_count",
            "negative_count",
            "true_positive_rate",
            "false_positive_rate",
            "clean_false_positive_rate",
            "attacked_false_positive_rate",
            "score_retention_mean",
            "quality_score_proxy_mean",
            "attention_consistency_proxy_mean",
            "geometry_reliable_rate",
            "rescue_rate",
            "attestation_available_rate",
            "supports_paper_claim",
        ],
    )
    claim_input_report_path.write_text(stable_json_text(claim_input_report), encoding="utf-8")
    claim_summary_path.write_text(stable_json_text(claim_summary), encoding="utf-8")

    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            records_path,
            mechanism_table_path,
            pairwise_delta_path,
            family_table_path,
            claim_input_report_path,
            claim_summary_path,
            manifest_path,
        )
    )
    input_paths = (
        relative_or_absolute(resolved_attack_records_path, root_path),
        relative_or_absolute(resolved_attack_manifest_path, root_path),
        relative_or_absolute(resolved_attack_matrix_manifest_path, root_path),
        relative_or_absolute(resolved_threshold_report_path, root_path),
        relative_or_absolute(resolved_baseline_manifest_path, root_path),
    )
    summary = {
        "claim_summary": claim_summary,
        "mechanism_rows": mechanism_rows,
        "pairwise_rows": pairwise_rows,
        "family_rows": family_rows,
        "attack_matrix_manifest_digest": attack_matrix_manifest.get("config_digest", ""),
    }
    manifest = build_artifact_manifest(
        artifact_id="internal_ablation_evidence_manifest",
        artifact_type="local_manifest",
        input_paths=input_paths,
        output_paths=output_paths,
        config={
            "ablation_spec_digest": build_stable_digest([spec.to_dict() for spec in ablation_specs]),
            "summary_digest": build_stable_digest(summary),
            "threshold_value": threshold,
            "claim_input_digest": build_stable_digest(claim_input_report),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_internal_ablation_outputs.py",
        metadata={
            **claim_summary,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="写出内部机制消融证据产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--attack-records-path", default=str(DEFAULT_ATTACK_RECORDS_PATH), help="攻击检测 records 路径。")
    parser.add_argument("--attack-manifest-path", default=str(DEFAULT_ATTACK_MANIFEST_PATH), help="攻击矩阵 manifest 路径。")
    parser.add_argument(
        "--attack-matrix-manifest-path",
        default=str(DEFAULT_ATTACK_MATRIX_MANIFEST_PATH),
        help="攻击矩阵产物 manifest 路径。",
    )
    parser.add_argument("--threshold-report-path", default=str(DEFAULT_THRESHOLD_REPORT_PATH), help="fixed-FPR 边界报告路径。")
    parser.add_argument("--baseline-manifest-path", default=str(DEFAULT_BASELINE_MANIFEST_PATH), help="外部 baseline 对比 manifest 路径。")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    manifest = write_internal_ablation_outputs(
        root=args.root,
        output_dir=args.output_dir,
        attack_records_path=args.attack_records_path,
        attack_manifest_path=args.attack_manifest_path,
        attack_matrix_manifest_path=args.attack_matrix_manifest_path,
        threshold_report_path=args.threshold_report_path,
        baseline_manifest_path=args.baseline_manifest_path,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
