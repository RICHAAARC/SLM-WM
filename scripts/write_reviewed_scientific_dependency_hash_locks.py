"""原子复验并写入五个科学执行环境的完整依赖哈希锁.

该入口只接收由同一 clean detached Git 提交生成的五个审查包. 所有候选
在任何锁文件写入前都会重新核验 manifest、文件摘要、pip resolver report
和规范锁文本; 任一候选失败时不会留下部分锁. 成功结果仍只表示依赖锁
可以提交, 不表示 CUDA 可用或论文实验已经通过.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runtime.dependency_profiles import (  # noqa: E402
    REQUIRED_DEPENDENCY_PROFILE_NAMES,
    WORKFLOW_ORCHESTRATOR_PROFILE_ID,
    get_dependency_profile,
)
from scripts import write_reviewed_dependency_hash_lock as single_writer  # noqa: E402
from scripts import write_dependency_lock_review_bundle as review_bundle  # noqa: E402


SCIENTIFIC_PROFILE_IDS = tuple(
    profile_id
    for profile_id in REQUIRED_DEPENDENCY_PROFILE_NAMES
    if profile_id != WORKFLOW_ORCHESTRATOR_PROFILE_ID
)
REPORT_SCHEMA = "reviewed_scientific_dependency_hash_locks_write_report"
REPORT_SCHEMA_VERSION = 1
SUCCESS_DECISION = "scientific_dependency_locks_written_for_commit"
REPORT_RELATIVE_PATH = Path(
    "outputs/dependency_lock_acceptance/scientific_profiles/"
    "reviewed_scientific_dependency_hash_locks_write_report.json"
)

GitCommandRunner = Callable[[Sequence[str], Path], Any]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """稳定写出批量接收报告."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _report_path(repository_root: Path) -> Path:
    """返回受 outputs 边界约束的批量接收报告路径."""

    outputs_root = (repository_root / "outputs").resolve()
    path = (repository_root / REPORT_RELATIVE_PATH).resolve()
    try:
        path.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("科学依赖锁接收报告必须位于 outputs/ 下") from exc
    return path


def _report_skeleton(
    repository_root: Path,
    review_bundle_root: Path,
) -> tuple[dict[str, Any], Path]:
    """构造批量接收成功和失败共享的报告."""

    report_path = _report_path(repository_root)
    return {
        "report_schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "operation_kind": "reviewed_scientific_dependency_hash_locks_write",
        "review_bundle_root": str(review_bundle_root),
        "approved_profile_ids": [],
        "formal_execution_lock": {},
        "formal_execution_commit": "",
        "formal_execution_lock_digest": "",
        "profile_records": [],
        "command_results": [],
        "decision": "fail",
        "failure_reasons": [],
        "diagnostic_message": None,
        "supports_paper_claim": False,
    }, report_path


def _write_failure(
    report: dict[str, Any],
    report_path: Path,
    reason: str,
    diagnostic_message: str,
) -> tuple[dict[str, Any], Path]:
    """持久化批量接收失败, 且保持论文结论关闭."""

    report["decision"] = "fail"
    report["failure_reasons"] = [reason]
    report["diagnostic_message"] = diagnostic_message
    report["supports_paper_claim"] = False
    _write_json(report_path, report)
    return report, report_path


def _validate_bundle_root(review_bundle_root: Path) -> dict[str, Path]:
    """要求根目录恰好包含五个真实目录, 每个目录对应一个科学 profile."""

    if not review_bundle_root.is_dir() or review_bundle_root.is_symlink():
        raise ValueError("科学审查包根路径必须是实际目录且不得是符号链接")
    entries = tuple(review_bundle_root.iterdir())
    actual_names = {entry.name for entry in entries}
    if actual_names != set(SCIENTIFIC_PROFILE_IDS):
        raise ValueError("科学审查包根目录必须恰好包含五个登记 profile")
    if any(not entry.is_dir() or entry.is_symlink() for entry in entries):
        raise ValueError("每个科学审查包必须是实际目录且不得是符号链接")
    return {
        profile_id: (review_bundle_root / profile_id).resolve()
        for profile_id in SCIENTIFIC_PROFILE_IDS
    }


def write_reviewed_scientific_dependency_hash_locks(
    review_bundle_root: str | Path,
    approved_profile_ids: Sequence[str],
    *,
    repository_root: str | Path = ROOT,
    git_command_runner: GitCommandRunner = single_writer._run_git_command,
) -> tuple[dict[str, Any], Path]:
    """复验同一提交的五个审查包并原子写入 registry 锁目标."""

    root = Path(repository_root).resolve()
    bundle_root = Path(review_bundle_root).expanduser().resolve()
    report, report_path = _report_skeleton(root, bundle_root)
    approvals = tuple(approved_profile_ids)
    report["approved_profile_ids"] = list(approvals)
    if approvals != SCIENTIFIC_PROFILE_IDS:
        return _write_failure(
            report,
            report_path,
            "scientific_profile_approval_mismatch",
            "必须按 registry 顺序逐项批准全部五个科学 profile.",
        )

    try:
        bundle_directories = _validate_bundle_root(bundle_root)
        current_head = single_writer._read_clean_repository_head(
            root,
            git_command_runner=git_command_runner,
            report=report,
        )
        validated_candidates: list[dict[str, Any]] = []
        common_formal_lock: dict[str, Any] | None = None
        for profile_id in SCIENTIFIC_PROFILE_IDS:
            profile = get_dependency_profile(
                profile_id,
                root / "configs/dependency_profile_registry.json",
            )
            if profile.complete_hash_lock_present or profile.formal_ready:
                raise ValueError(f"目标完整哈希锁已经存在: {profile_id}")
            if profile.readiness_blockers != ("complete_hash_lock_missing",):
                raise ValueError(f"目标 profile 不是仅缺完整锁的状态: {profile_id}")

            bundle_dir = bundle_directories[profile_id]
            manifest_path = bundle_dir / review_bundle.BUNDLE_MANIFEST_FILE_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            formal_lock = single_writer._validate_bundle_identity(
                profile,
                manifest=manifest,
                current_head=current_head,
            )
            if common_formal_lock is None:
                common_formal_lock = formal_lock
            elif formal_lock != common_formal_lock:
                raise ValueError("五个科学审查包的正式执行锁不一致")
            paths = single_writer._bundle_files(bundle_dir, manifest)
            candidate_bytes, logical_digest, dependency_count = (
                single_writer._validate_candidate_closure(
                    profile,
                    paths=paths,
                    formal_execution_lock=formal_lock,
                )
            )
            validated_candidates.append(
                {
                    "profile": profile,
                    "bundle_dir": bundle_dir,
                    "candidate_bytes": candidate_bytes,
                    "logical_digest": logical_digest,
                    "dependency_count": dependency_count,
                }
            )

        if common_formal_lock is None:
            raise ValueError("科学审查包集合不得为空")
        pre_write_head = single_writer._read_clean_repository_head(
            root,
            git_command_runner=git_command_runner,
            report=report,
        )
        if pre_write_head != current_head:
            raise ValueError("审查复验期间仓库 HEAD 已发生变化")
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        return _write_failure(
            report,
            report_path,
            "scientific_review_bundle_validation_failed",
            str(exc),
        )

    report["formal_execution_lock"] = common_formal_lock
    report["formal_execution_commit"] = common_formal_lock[
        "formal_execution_commit"
    ]
    report["formal_execution_lock_digest"] = common_formal_lock[
        "formal_execution_lock_digest"
    ]

    written_paths: list[Path] = []
    try:
        for candidate in validated_candidates:
            profile = candidate["profile"]
            target_path = (root / profile.complete_hash_lock_path).resolve()
            target_path.relative_to(root)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("xb") as handle:
                handle.write(candidate["candidate_bytes"])
            written_paths.append(target_path)

        records: list[dict[str, Any]] = []
        for candidate in validated_candidates:
            profile = candidate["profile"]
            accepted = get_dependency_profile(
                profile.profile_name,
                root / "configs/dependency_profile_registry.json",
            )
            if not accepted.formal_ready or accepted.readiness_blockers:
                raise ValueError(
                    f"写入后的完整哈希锁未通过 registry 门禁: {profile.profile_name}"
                )
            if accepted.complete_hash_lock_digest != candidate["logical_digest"]:
                raise ValueError(
                    f"写入后的锁摘要与审查候选不一致: {profile.profile_name}"
                )
            if (
                accepted.complete_hash_lock_dependency_count
                != candidate["dependency_count"]
            ):
                raise ValueError(
                    f"写入后的依赖数量与审查候选不一致: {profile.profile_name}"
                )
            records.append(
                {
                    "profile_id": profile.profile_name,
                    "profile_digest": profile.profile_digest,
                    "review_bundle_dir": str(candidate["bundle_dir"]),
                    "complete_hash_lock_path": profile.complete_hash_lock_path,
                    "complete_hash_lock_digest": candidate["logical_digest"],
                    "complete_hash_lock_dependency_count": candidate[
                        "dependency_count"
                    ],
                    "formal_ready": True,
                }
            )
    except (FileExistsError, OSError, ValueError) as exc:
        for written_path in reversed(written_paths):
            written_path.unlink(missing_ok=True)
        return _write_failure(
            report,
            report_path,
            "scientific_dependency_lock_write_failed",
            str(exc),
        )

    report["profile_records"] = records
    report["decision"] = SUCCESS_DECISION
    report["failure_reasons"] = []
    report["diagnostic_message"] = None
    report["supports_paper_claim"] = False
    _write_json(report_path, report)
    return report, report_path


def build_parser() -> argparse.ArgumentParser:
    """构造五个科学 profile 的显式批准 CLI."""

    parser = argparse.ArgumentParser(
        description="复验五个科学依赖锁审查包并原子写入可提交锁."
    )
    parser.add_argument("--review-bundle-root", required=True)
    parser.add_argument(
        "--approve-profile",
        action="append",
        required=True,
        choices=SCIENTIFIC_PROFILE_IDS,
        help="必须按 registry 顺序重复五次, 显式批准每个科学 profile.",
    )
    parser.add_argument("--root", default=".")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行批量接收, 并用退出码表达是否形成五个可提交锁."""

    arguments = build_parser().parse_args(argv)
    requested_root = Path(arguments.root).resolve()
    if requested_root != ROOT.resolve():
        print(
            json.dumps(
                {
                    "decision": "fail",
                    "failure_reasons": ["receiver_code_root_mismatch"],
                    "diagnostic_message": (
                        "批量锁接收 CLI 必须由目标仓库 checkout 内的同一份脚本执行."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    report, report_path = write_reviewed_scientific_dependency_hash_locks(
        arguments.review_bundle_root,
        arguments.approve_profile,
        repository_root=requested_root,
    )
    print(
        json.dumps(
            {
                "report_path": str(report_path),
                "decision": report["decision"],
                "failure_reasons": report["failure_reasons"],
                "profile_count": len(report["profile_records"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["decision"] == SUCCESS_DECISION else 1


if __name__ == "__main__":
    raise SystemExit(main())
