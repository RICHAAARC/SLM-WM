"""写出投稿就绪门禁的本地审计产物。"""

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
from paper_experiments.analysis.submission_readiness import (
    SubmissionReadinessInput,
    build_release_profile_rows,
    build_required_evidence_rows,
    build_submission_readiness_report,
)
from main.core.digest import build_stable_digest
from scripts.extract_release_package import PROFILES, extract_profile

CONSTRUCTION_UNIT_NAME = "submission_readiness_gate"
DEFAULT_OUTPUT_ROOT = Path("outputs/submission_readiness")
DEFAULT_EVIDENCE_AUDIT_ROOT = Path("outputs/paper_artifact_evidence_audit")


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
    """确保持久输出目录位于 outputs 下。"""
    resolved_output_dir = (root_path / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved_output_dir.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("投稿就绪门禁输出目录必须位于 outputs/ 下") from exc
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


def build_release_profile_dry_runs(root_path: Path, output_dir: Path) -> tuple[dict[str, Any], ...]:
    """执行 release profile dry-run, 只记录文件清单, 不生成发布包。"""
    dry_run_root = output_dir / "release_profile_dry_runs"
    profiles: list[dict[str, Any]] = []
    for profile_name in sorted(PROFILES):
        profiles.append(
            extract_profile(
                root=root_path,
                output=dry_run_root / profile_name,
                profile_name=profile_name,
                dry_run=True,
            )
        )
    return tuple(profiles)


def build_input_bundle(
    root_path: Path,
    output_dir: Path,
    evidence_manifest_path: Path,
    builder_report_path: Path,
    blocker_report_path: Path,
    gap_list_path: Path,
) -> SubmissionReadinessInput:
    """读取受治理输入并构造投稿就绪门禁输入对象。"""
    return SubmissionReadinessInput(
        evidence_manifest=read_json(evidence_manifest_path),
        builder_report=read_json(builder_report_path),
        blocker_report=read_json(blocker_report_path),
        evidence_gaps=tuple(read_csv(gap_list_path)),
        release_profiles=build_release_profile_dry_runs(root_path, output_dir),
    )


def write_submission_readiness_outputs(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """在精确9重复原始记录重算 Writer 就绪前拒绝正式结论物化."""

    require_exact9_randomization_aggregate_provenance()


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="写出投稿就绪门禁本地审计产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录; 默认写入当前论文运行子目录, 且必须位于 outputs/ 下。",
    )
    parser.add_argument("--evidence-manifest-path", default=None, help="证据审计 manifest; 默认读取当前论文运行子目录。")
    parser.add_argument("--builder-report-path", default=None, help="产物构建器 readiness report; 默认读取当前论文运行子目录。")
    parser.add_argument("--blocker-report-path", default=None, help="投稿阻断 report; 默认读取当前论文运行子目录。")
    parser.add_argument("--gap-list-path", default=None, help="证据缺口列表; 默认读取当前论文运行子目录。")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    manifest = write_submission_readiness_outputs(
        root=args.root,
        output_dir=args.output_dir,
        evidence_manifest_path=args.evidence_manifest_path,
        builder_report_path=args.builder_report_path,
        blocker_report_path=args.blocker_report_path,
        gap_list_path=args.gap_list_path,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
