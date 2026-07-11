"""写出投稿就绪门禁的本地审计产物。"""

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
from paper_experiments.analysis.submission_readiness import (
    SubmissionReadinessInput,
    build_release_profile_rows,
    build_required_evidence_rows,
    build_submission_readiness_report,
)
from main.core.digest import build_stable_digest
from scripts.extract_minimal_paper_package import PROFILES, extract_profile

CONSTRUCTION_UNIT_NAME = "submission_readiness_gate"
DEFAULT_OUTPUT_DIR = Path("outputs/submission_readiness")
DEFAULT_EVIDENCE_MANIFEST_PATH = Path("outputs/paper_artifact_evidence_audit/manifest.local.json")
DEFAULT_BUILDER_REPORT_PATH = Path("outputs/paper_artifact_evidence_audit/artifact_builder_readiness_report.json")
DEFAULT_BLOCKER_REPORT_PATH = Path("outputs/paper_artifact_evidence_audit/submission_blocker_report.json")
DEFAULT_GAP_LIST_PATH = Path("outputs/paper_artifact_evidence_audit/evidence_gap_list.csv")


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


def write_submission_readiness_outputs(
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    evidence_manifest_path: str | Path = DEFAULT_EVIDENCE_MANIFEST_PATH,
    builder_report_path: str | Path = DEFAULT_BUILDER_REPORT_PATH,
    blocker_report_path: str | Path = DEFAULT_BLOCKER_REPORT_PATH,
    gap_list_path: str | Path = DEFAULT_GAP_LIST_PATH,
) -> dict[str, Any]:
    """写出投稿就绪阻断报告、所需输入清单、release dry-run 表和 manifest。"""
    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, Path(output_dir))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_evidence_manifest_path = resolve_input_path(root_path, evidence_manifest_path)
    resolved_builder_report_path = resolve_input_path(root_path, builder_report_path)
    resolved_blocker_report_path = resolve_input_path(root_path, blocker_report_path)
    resolved_gap_list_path = resolve_input_path(root_path, gap_list_path)

    bundle = build_input_bundle(
        root_path,
        resolved_output_dir,
        resolved_evidence_manifest_path,
        resolved_builder_report_path,
        resolved_blocker_report_path,
        resolved_gap_list_path,
    )
    required_rows = build_required_evidence_rows(bundle)
    release_rows = build_release_profile_rows(bundle)
    readiness_report = build_submission_readiness_report(bundle, required_rows, release_rows)

    blocker_report_path_out = resolved_output_dir / "readiness_blocker_report.json"
    required_inputs_path = resolved_output_dir / "required_evidence_inputs.csv"
    release_profile_path = resolved_output_dir / "release_profile_dry_run.csv"
    manifest_path = resolved_output_dir / "submission_readiness_manifest.local.json"

    blocker_report_path_out.write_text(stable_json_text(readiness_report), encoding="utf-8")
    write_csv(
        required_inputs_path,
        required_rows,
        [
            "required_input_id",
            "required_input_area",
            "required_input_severity",
            "required_action",
            "related_artifacts",
            "closes_claim_ids",
            "recommended_order",
            "input_ready",
            "supports_paper_claim",
        ],
    )
    write_csv(
        release_profile_path,
        release_rows,
        [
            "release_profile_name",
            "release_profile_file_count",
            "release_profile_missing_count",
            "release_dry_run_ready",
            "release_package_allowed",
            "package_freeze_allowed",
            "release_scope",
            "supports_paper_claim",
        ],
    )

    input_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            resolved_evidence_manifest_path,
            resolved_builder_report_path,
            resolved_blocker_report_path,
            resolved_gap_list_path,
            root_path / "docs" / "extraction_profiles.md",
            root_path / "docs" / "release_boundary.md",
        )
    )
    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            blocker_report_path_out,
            required_inputs_path,
            release_profile_path,
            manifest_path,
        )
    )
    summary = {
        "readiness_report": readiness_report,
        "required_rows": required_rows,
        "release_rows": release_rows,
    }
    manifest = build_artifact_manifest(
        artifact_id="submission_readiness_manifest",
        artifact_type="local_manifest",
        input_paths=input_paths,
        output_paths=output_paths,
        config={
            "summary_digest": build_stable_digest(summary),
            "input_bundle_digest": build_stable_digest(bundle.to_dict()),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_submission_readiness_outputs.py",
        metadata={
            **readiness_report,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="写出投稿就绪门禁本地审计产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--evidence-manifest-path", default=str(DEFAULT_EVIDENCE_MANIFEST_PATH), help="证据审计 manifest 路径。")
    parser.add_argument("--builder-report-path", default=str(DEFAULT_BUILDER_REPORT_PATH), help="产物构建器 readiness report 路径。")
    parser.add_argument("--blocker-report-path", default=str(DEFAULT_BLOCKER_REPORT_PATH), help="投稿阻断 report 路径。")
    parser.add_argument("--gap-list-path", default=str(DEFAULT_GAP_LIST_PATH), help="证据缺口列表路径。")
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
