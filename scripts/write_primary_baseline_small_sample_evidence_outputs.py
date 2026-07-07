"""写出主表 external baseline 小样本证据边界产物。"""

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

from paper_experiments.baselines.small_sample_evidence import (
    build_primary_baseline_small_sample_comparison_rows,
    build_primary_baseline_small_sample_evidence_records,
    build_primary_baseline_small_sample_evidence_summary,
)
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest

DEFAULT_OUTPUT_DIR = Path("outputs/primary_baseline_small_sample_evidence")
DEFAULT_CANDIDATE_RECORDS_PATH = Path("outputs/external_baseline_results/baseline_result_records.jsonl")
DEFAULT_VALIDATION_REPORT_PATH = Path("outputs/external_baseline_results/baseline_result_candidate_validation_report.json")


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: dict[str, Any]) -> str:
    """把单条记录写成稳定 JSONL 行。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写出小样本共同协议表格, 供审计和下游文档引用。"""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_path(root_path: Path, path: str | Path) -> Path:
    """把输入路径解析为绝对路径。"""

    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录相对仓库根目录的路径。"""

    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def ensure_output_dir_under_outputs(root_path: Path, output_dir: str | Path) -> Path:
    """确保小样本证据产物输出目录位于 outputs/ 下。"""

    resolved = resolve_path(root_path, output_dir)
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("主表 baseline 小样本证据输出目录必须位于 outputs/ 下。") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_code_version(root_path: Path) -> str:
    """读取当前 Git 短提交标识, 工作区有变更时附加 dirty 标记。"""

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


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""

    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 候选记录。"""

    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_primary_baseline_small_sample_evidence_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    candidate_records_path: str | Path = DEFAULT_CANDIDATE_RECORDS_PATH,
    validation_report_path: str | Path = DEFAULT_VALIDATION_REPORT_PATH,
) -> dict[str, Any]:
    """写出小样本证据 records、summary 和 manifest。

    该脚本只记录小样本工程证据, 不触发 TPR@FPR=0.01 或 TPR@FPR=0.001 的正式 full paper 运行。
    """

    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, output_dir)
    resolved_candidate_records_path = resolve_path(root_path, candidate_records_path)
    resolved_validation_report_path = resolve_path(root_path, validation_report_path)

    candidate_rows = read_jsonl_rows(resolved_candidate_records_path)
    validation_report = read_json(resolved_validation_report_path) if resolved_validation_report_path.is_file() else {}
    records = build_primary_baseline_small_sample_evidence_records(candidate_rows, validation_report)
    comparison_rows = build_primary_baseline_small_sample_comparison_rows(records)
    records_path = resolved_output_dir / "primary_baseline_small_sample_evidence_records.jsonl"
    comparison_table_path = resolved_output_dir / "primary_baseline_small_sample_comparison_table.csv"
    summary_path = resolved_output_dir / "primary_baseline_small_sample_evidence_summary.json"
    manifest_path = resolved_output_dir / "manifest.local.json"
    summary = build_primary_baseline_small_sample_evidence_summary(records)
    summary = {
        **summary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_records_path": relative_or_absolute(resolved_candidate_records_path, root_path),
        "validation_report_path": relative_or_absolute(resolved_validation_report_path, root_path),
        "small_sample_comparison_table_path": relative_or_absolute(comparison_table_path, root_path),
    }

    records_path.write_text("".join(json_line(record.to_dict()) for record in records), encoding="utf-8")
    write_csv(
        comparison_table_path,
        comparison_rows,
        [
            "baseline_id",
            "comparison_scope",
            "resource_profile",
            "comparable_operating_point",
            "attack_family",
            "attack_name",
            "metric_status",
            "true_positive_rate",
            "false_positive_rate",
            "clean_false_positive_rate",
            "attacked_false_positive_rate",
            "quality_score_proxy_mean",
            "score_retention_mean",
            "positive_count",
            "negative_count",
            "supported_record_count",
            "attack_record_count",
            "small_sample_evidence_ready",
            "small_sample_common_protocol_ready",
            "formal_import_ready",
            "paper_claim_boundary",
            "excluded_operating_points",
            "supports_paper_claim",
        ],
    )
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")

    input_paths = []
    for path in (resolved_candidate_records_path, resolved_validation_report_path):
        if path.exists():
            input_paths.append(relative_or_absolute(path, root_path))
    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (records_path, comparison_table_path, summary_path, manifest_path)
    )
    manifest = build_artifact_manifest(
        artifact_id="primary_baseline_small_sample_evidence_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(input_paths),
        output_paths=output_paths,
        config={
            "records_digest": build_stable_digest([record.to_dict() for record in records]),
            "comparison_table_digest": build_stable_digest(comparison_rows),
            "summary_digest": build_stable_digest(summary),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_primary_baseline_small_sample_evidence_outputs.py",
        metadata=summary,
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="写出主表 external baseline 小样本证据边界产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--candidate-records-path", default=str(DEFAULT_CANDIDATE_RECORDS_PATH), help="候选结果 JSONL 路径。")
    parser.add_argument("--validation-report-path", default=str(DEFAULT_VALIDATION_REPORT_PATH), help="候选结果校验报告路径。")
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    manifest = write_primary_baseline_small_sample_evidence_outputs(
        root=args.root,
        output_dir=args.output_dir,
        candidate_records_path=args.candidate_records_path,
        validation_report_path=args.validation_report_path,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()

