"""写出论文投稿级证据闭合入口审计产物。"""

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
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.runtime.repository_environment import resolve_code_version
from paper_experiments.analysis.evidence_closure_entry_review import (
    EvidenceClosureEntryInput,
    build_evidence_closure_entry_checklist,
    build_evidence_closure_entry_review_report,
)
from main.core.digest import build_stable_digest

CONSTRUCTION_UNIT_NAME = "evidence_closure_entry_review"
DEFAULT_OUTPUT_ROOT = Path("outputs/evidence_closure_entry_review")
DEFAULT_SUBMISSION_READINESS_ROOT = Path("outputs/submission_readiness")
DEFAULT_EVIDENCE_AUDIT_ROOT = Path("outputs/paper_artifact_evidence_audit")
DEFAULT_BASELINE_COMPARISON_ROOT = Path("outputs/external_baseline_comparison")
DEFAULT_DATASET_QUALITY_ROOT = Path("outputs/dataset_level_quality")


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
) -> EvidenceClosureEntryInput:
    """读取受治理输入并构造证据闭合入口审计对象。"""

    return EvidenceClosureEntryInput(
        submission_readiness_report=read_json(submission_readiness_report_path),
        required_evidence_rows=tuple(read_csv(required_evidence_inputs_path)),
        paper_blocker_report=read_json(paper_blocker_report_path),
        baseline_runtime_report=read_json(baseline_runtime_report_path),
        dataset_quality_summary=read_json(dataset_quality_summary_path),
    )


def write_evidence_closure_entry_review_outputs(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """在精确9重复原始记录重算 Writer 就绪前拒绝正式结论物化."""

    require_exact9_randomization_aggregate_provenance()


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="写出论文投稿级证据闭合入口审计产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录; 默认写入当前论文运行子目录, 且必须位于 outputs/ 下。",
    )
    parser.add_argument(
        "--submission-readiness-report-path",
        default=None,
        help="投稿就绪门禁报告路径; 默认读取当前论文运行子目录。",
    )
    parser.add_argument(
        "--required-evidence-inputs-path",
        default=None,
        help="仍需补齐的证据输入清单路径; 默认读取当前论文运行子目录。",
    )
    parser.add_argument(
        "--paper-blocker-report-path",
        default=None,
        help="论文产物证据审计阻断报告路径; 默认读取当前论文运行子目录。",
    )
    parser.add_argument(
        "--baseline-runtime-report-path",
        default=None,
        help="外部 baseline 运行报告路径; 默认读取当前论文运行子目录。",
    )
    parser.add_argument(
        "--dataset-quality-summary-path",
        default=None,
        help="数据集级质量摘要路径; 默认读取当前论文运行子目录。",
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
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
