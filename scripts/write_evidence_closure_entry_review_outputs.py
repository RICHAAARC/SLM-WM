"""写出论文投稿级证据闭合入口审计产物。"""

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

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from paper_experiments.analysis.evidence_closure_entry_review import (
    EvidenceClosureEntryInput,
    build_evidence_closure_entry_checklist,
    build_evidence_closure_entry_review_report,
)
from main.core.digest import build_stable_digest

CONSTRUCTION_UNIT_NAME = "evidence_closure_entry_review"
DEFAULT_OUTPUT_DIR = Path("outputs/evidence_closure_entry_review")
DEFAULT_SUBMISSION_READINESS_REPORT_PATH = Path("outputs/submission_readiness/readiness_blocker_report.json")
DEFAULT_REQUIRED_EVIDENCE_INPUTS_PATH = Path("outputs/submission_readiness/required_evidence_inputs.csv")
DEFAULT_PAPER_BLOCKER_REPORT_PATH = Path("outputs/paper_artifact_evidence_audit/submission_blocker_report.json")
DEFAULT_BASELINE_RUNTIME_REPORT_PATH = Path("outputs/external_baseline_comparison/baseline_runtime_report.json")
DEFAULT_DATASET_QUALITY_SUMMARY_PATH = Path("outputs/dataset_level_quality/dataset_quality_summary.json")
DEFAULT_BASELINE_SMALL_SAMPLE_SUMMARY_PATH = Path(
    "outputs/primary_baseline_small_sample_evidence/primary_baseline_small_sample_evidence_summary.json"
)


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""

    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, Any]]:
    """读取 CSV 文件并返回字典行。"""

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    """确保持久化输出目录位于 outputs/ 下。"""

    resolved_output_dir = (root_path / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved_output_dir.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("证据闭合入口审计输出目录必须位于 outputs/ 下。") from exc
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


def build_input_bundle(
    submission_readiness_report_path: Path,
    required_evidence_inputs_path: Path,
    paper_blocker_report_path: Path,
    baseline_runtime_report_path: Path,
    dataset_quality_summary_path: Path,
    baseline_small_sample_summary_path: Path,
) -> EvidenceClosureEntryInput:
    """读取受治理输入并构造证据闭合入口审计对象。"""

    return EvidenceClosureEntryInput(
        submission_readiness_report=read_json(submission_readiness_report_path),
        required_evidence_rows=tuple(read_csv(required_evidence_inputs_path)),
        paper_blocker_report=read_json(paper_blocker_report_path),
        baseline_runtime_report=read_json(baseline_runtime_report_path),
        dataset_quality_summary=read_json(dataset_quality_summary_path),
        baseline_small_sample_summary=read_json(baseline_small_sample_summary_path),
    )


def write_evidence_closure_entry_review_outputs(
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    submission_readiness_report_path: str | Path = DEFAULT_SUBMISSION_READINESS_REPORT_PATH,
    required_evidence_inputs_path: str | Path = DEFAULT_REQUIRED_EVIDENCE_INPUTS_PATH,
    paper_blocker_report_path: str | Path = DEFAULT_PAPER_BLOCKER_REPORT_PATH,
    baseline_runtime_report_path: str | Path = DEFAULT_BASELINE_RUNTIME_REPORT_PATH,
    dataset_quality_summary_path: str | Path = DEFAULT_DATASET_QUALITY_SUMMARY_PATH,
    baseline_small_sample_summary_path: str | Path = DEFAULT_BASELINE_SMALL_SAMPLE_SUMMARY_PATH,
) -> dict[str, Any]:
    """写出证据闭合入口审计报告、审计清单和 manifest。"""

    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, Path(output_dir))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_submission_readiness_report_path = resolve_input_path(root_path, submission_readiness_report_path)
    resolved_required_evidence_inputs_path = resolve_input_path(root_path, required_evidence_inputs_path)
    resolved_paper_blocker_report_path = resolve_input_path(root_path, paper_blocker_report_path)
    resolved_baseline_runtime_report_path = resolve_input_path(root_path, baseline_runtime_report_path)
    resolved_dataset_quality_summary_path = resolve_input_path(root_path, dataset_quality_summary_path)
    resolved_baseline_small_sample_summary_path = resolve_input_path(root_path, baseline_small_sample_summary_path)

    bundle = build_input_bundle(
        resolved_submission_readiness_report_path,
        resolved_required_evidence_inputs_path,
        resolved_paper_blocker_report_path,
        resolved_baseline_runtime_report_path,
        resolved_dataset_quality_summary_path,
        resolved_baseline_small_sample_summary_path,
    )
    checklist_rows = build_evidence_closure_entry_checklist(bundle)
    review_report = build_evidence_closure_entry_review_report(bundle, checklist_rows)

    review_report_path = resolved_output_dir / "entry_review_report.json"
    checklist_path = resolved_output_dir / "entry_review_checklist.csv"
    manifest_path = resolved_output_dir / "manifest.local.json"
    review_report_path.write_text(stable_json_text(review_report), encoding="utf-8")
    write_csv(
        checklist_path,
        checklist_rows,
        [
            "review_item_id",
            "review_area",
            "review_status",
            "source_artifact",
            "blocker_reason",
            "user_audit_note",
            "supports_paper_claim",
        ],
    )

    input_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            resolved_submission_readiness_report_path,
            resolved_required_evidence_inputs_path,
            resolved_paper_blocker_report_path,
            resolved_baseline_runtime_report_path,
            resolved_dataset_quality_summary_path,
            resolved_baseline_small_sample_summary_path,
        )
    )
    output_paths = tuple(
        relative_or_absolute(path, root_path) for path in (review_report_path, checklist_path, manifest_path)
    )
    summary = {
        "review_report": review_report,
        "checklist_rows": checklist_rows,
    }
    manifest = build_artifact_manifest(
        artifact_id="evidence_closure_entry_review_manifest",
        artifact_type="local_manifest",
        input_paths=input_paths,
        output_paths=output_paths,
        config={
            "summary_digest": build_stable_digest(summary),
            "input_bundle_digest": build_stable_digest(bundle.to_dict()),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_evidence_closure_entry_review_outputs.py",
        metadata={
            **review_report,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="写出论文投稿级证据闭合入口审计产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument(
        "--submission-readiness-report-path",
        default=str(DEFAULT_SUBMISSION_READINESS_REPORT_PATH),
        help="投稿就绪门禁报告路径。",
    )
    parser.add_argument(
        "--required-evidence-inputs-path",
        default=str(DEFAULT_REQUIRED_EVIDENCE_INPUTS_PATH),
        help="仍需补齐的证据输入清单路径。",
    )
    parser.add_argument(
        "--paper-blocker-report-path",
        default=str(DEFAULT_PAPER_BLOCKER_REPORT_PATH),
        help="论文产物证据审计阻断报告路径。",
    )
    parser.add_argument(
        "--baseline-runtime-report-path",
        default=str(DEFAULT_BASELINE_RUNTIME_REPORT_PATH),
        help="外部 baseline 运行报告路径。",
    )
    parser.add_argument(
        "--dataset-quality-summary-path",
        default=str(DEFAULT_DATASET_QUALITY_SUMMARY_PATH),
        help="数据集级质量摘要路径。",
    )
    parser.add_argument(
        "--baseline-small-sample-summary-path",
        default=str(DEFAULT_BASELINE_SMALL_SAMPLE_SUMMARY_PATH),
        help="主表 baseline 小样本证据摘要路径。",
    )
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    manifest = write_evidence_closure_entry_review_outputs(
        root=args.root,
        output_dir=args.output_dir,
        submission_readiness_report_path=args.submission_readiness_report_path,
        required_evidence_inputs_path=args.required_evidence_inputs_path,
        paper_blocker_report_path=args.paper_blocker_report_path,
        baseline_runtime_report_path=args.baseline_runtime_report_path,
        dataset_quality_summary_path=args.dataset_quality_summary_path,
        baseline_small_sample_summary_path=args.baseline_small_sample_summary_path,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
