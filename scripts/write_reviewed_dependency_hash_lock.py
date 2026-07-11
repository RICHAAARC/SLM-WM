"""复验人工批准的审查包并写入可提交完整哈希锁.

该命令不会提交 Git, 也不会把候选直接转换为论文证据. 它要求当前仓库在
候选生成提交上保持 clean, 对审查包的身份、文件摘要和 pip 解析闭包重新
计算后, 才把规范候选写入 registry 登记的缺失锁路径.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runtime.dependency_profiles import (  # noqa: E402
    REQUIRED_DEPENDENCY_PROFILE_NAMES,
    DependencyProfile,
    get_dependency_profile,
)
from experiments.runtime import repository_environment  # noqa: E402
from scripts import (  # noqa: E402
    materialize_dependency_lock_candidate as candidate_materializer,
)
from scripts import write_dependency_lock_review_bundle as review_bundle  # noqa: E402


REPORT_SCHEMA = "reviewed_dependency_hash_lock_write_report"
REPORT_SCHEMA_VERSION = 1
SUCCESS_DECISION = "lock_written_for_commit"
REPORT_RELATIVE_ROOT = Path("outputs/dependency_lock_acceptance")

GitCommandRunner = Callable[[Sequence[str], Path], Any]


def _sha256(path: Path) -> str:
    """计算实际审查文件或写入锁文件的 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """把锁接收报告稳定写入 ``outputs/``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_git_command(command: Sequence[str], repository_root: Path) -> dict[str, Any]:
    """执行只读 Git 身份查询并返回标准进程记录."""

    completed = subprocess.run(
        list(command),
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    return {
        "return_code": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _normalize_command_result(value: Any) -> dict[str, Any]:
    """统一测试 runner 与 subprocess 的 Git 查询结果."""

    if isinstance(value, subprocess.CompletedProcess):
        return {
            "return_code": int(value.returncode),
            "stdout": str(value.stdout or ""),
            "stderr": str(value.stderr or ""),
        }
    if isinstance(value, Mapping):
        return_code = value.get("return_code")
        if isinstance(return_code, bool) or not isinstance(return_code, int):
            raise ValueError("Git runner 必须返回整数 return_code")
        return {
            "return_code": return_code,
            "stdout": str(value.get("stdout") or ""),
            "stderr": str(value.get("stderr") or ""),
        }
    raise TypeError("Git runner 返回类型不受支持")


def _report_path(repository_root: Path, profile_id: str) -> Path:
    """返回单个 profile 的锁接收报告路径并阻止越界."""

    outputs_root = (repository_root / "outputs").resolve()
    path = (
        repository_root
        / REPORT_RELATIVE_ROOT
        / profile_id
        / "reviewed_dependency_hash_lock_write_report.json"
    ).resolve()
    try:
        path.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("依赖锁接收报告必须位于 outputs/ 下") from exc
    return path


def _report_skeleton(
    profile: DependencyProfile,
    *,
    repository_root: Path,
    bundle_dir: Path,
) -> tuple[dict[str, Any], Path]:
    """构造成功与失败共享的可提交锁写入报告."""

    report_path = _report_path(repository_root, profile.profile_name)
    return {
        "report_schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "operation_kind": "reviewed_dependency_hash_lock_write",
        "profile_id": profile.profile_name,
        "profile_digest": profile.profile_digest,
        "source_path": str(bundle_dir),
        "manifest_path": str(
            bundle_dir / review_bundle.BUNDLE_MANIFEST_FILE_NAME
        ),
        "candidate_lock_path": str(
            bundle_dir / candidate_materializer.CANDIDATE_LOCK_FILE_NAME
        ),
        "complete_hash_lock_path": profile.complete_hash_lock_path,
        "complete_hash_lock_digest": None,
        "complete_hash_lock_dependency_count": 0,
        "formal_execution_lock": {},
        "formal_execution_commit": "",
        "formal_execution_lock_digest": "",
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
    """持久化首个不可恢复的锁接收失败."""

    report["decision"] = "fail"
    report["failure_reasons"] = [reason]
    report["diagnostic_message"] = diagnostic_message
    report["supports_paper_claim"] = False
    _write_json(report_path, report)
    return report, report_path


def _read_clean_repository_head(
    repository_root: Path,
    *,
    git_command_runner: GitCommandRunner,
    report: dict[str, Any],
) -> str:
    """要求接收前 HEAD 等于候选提交且工作树完全 clean."""

    command_plan = (
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
    )
    results = []
    for command in command_plan:
        normalized = _normalize_command_result(
            git_command_runner(command, repository_root)
        )
        record = {
            "operation": (
                "repository_head"
                if len(results) == 0
                else "repository_status"
            ),
            "argv": command,
            "working_directory": str(repository_root),
            "environment_overrides": {},
            **normalized,
        }
        report["command_results"].append(record)
        results.append(normalized)
        if normalized["return_code"] != 0:
            raise ValueError("无法读取锁接收所需的 Git 仓库状态")
    head = repository_environment.normalize_formal_git_commit(
        results[0]["stdout"].strip()
    )
    if results[1]["stdout"]:
        raise ValueError("写入可提交锁前 Git 工作树必须完全 clean")
    return head


def _bundle_files(
    bundle_dir: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Path]:
    """核验审查包精确文件集合、逐文件摘要和大小."""

    expected_names = {
        "candidate_lock": candidate_materializer.CANDIDATE_LOCK_FILE_NAME,
        "pip_resolver_report": candidate_materializer.PIP_REPORT_FILE_NAME,
        "candidate_provenance": candidate_materializer.PROVENANCE_FILE_NAME,
    }
    entries = tuple(bundle_dir.iterdir())
    actual_names = {path.name for path in entries}
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise ValueError("审查包目录不得包含子目录或符号链接")
    if actual_names != {
        *expected_names.values(),
        review_bundle.BUNDLE_MANIFEST_FILE_NAME,
    }:
        raise ValueError("审查包目录必须恰好包含三个候选文件和 manifest")
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != len(expected_names):
        raise ValueError("审查包 manifest 文件记录不完整")
    paths: dict[str, Path] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("审查包 manifest 文件记录必须是对象")
        artifact_role = record.get("artifact_role")
        file_name = record.get("file_name")
        if (
            artifact_role not in expected_names
            or file_name != expected_names[artifact_role]
            or artifact_role in paths
        ):
            raise ValueError("审查包 manifest 文件身份不一致")
        path = (bundle_dir / str(file_name)).resolve()
        if path.parent != bundle_dir or not path.is_file() or path.is_symlink():
            raise ValueError("审查包候选文件路径无效")
        if record.get("sha256") != _sha256(path):
            raise ValueError("审查包候选文件 SHA-256 不一致")
        if record.get("size_bytes") != path.stat().st_size:
            raise ValueError("审查包候选文件大小不一致")
        paths[str(artifact_role)] = path
    if set(paths) != set(expected_names):
        raise ValueError("审查包候选文件职责集合不一致")
    return paths


def _validate_bundle_identity(
    profile: DependencyProfile,
    *,
    manifest: Any,
    current_head: str,
) -> dict[str, Any]:
    """验证 manifest、正式代码锁和 profile 输入身份."""

    if not isinstance(manifest, dict):
        raise ValueError("审查包 manifest 必须是对象")
    formal_execution_lock = (
        repository_environment.validate_formal_execution_lock_record(
            manifest.get("formal_execution_lock")
        )
    )
    expected = {
        "manifest_schema": review_bundle.BUNDLE_MANIFEST_SCHEMA,
        "schema_version": review_bundle.BUNDLE_MANIFEST_SCHEMA_VERSION,
        "profile_id": profile.profile_name,
        "profile_digest": profile.profile_digest,
        "direct_requirements_digest": profile.direct_requirements_digest,
        "formal_execution_lock": formal_execution_lock,
        "formal_execution_commit": formal_execution_lock[
            "formal_execution_commit"
        ],
        "formal_execution_lock_digest": formal_execution_lock[
            "formal_execution_lock_digest"
        ],
        "decision": review_bundle.SUCCESS_DECISION,
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    if any(
        manifest.get(field_name) != expected_value
        for field_name, expected_value in expected.items()
    ):
        raise ValueError("审查包 manifest 身份门禁未通过")
    if formal_execution_lock["formal_execution_commit"] != current_head:
        raise ValueError("当前 HEAD 与审查包候选生成提交不一致")
    return formal_execution_lock


def _validate_candidate_closure(
    profile: DependencyProfile,
    *,
    paths: Mapping[str, Path],
    formal_execution_lock: Mapping[str, Any],
) -> tuple[bytes, str, int]:
    """从 pip 报告重建规范候选文本、逻辑摘要和依赖数量."""

    try:
        provenance = json.loads(
            paths["candidate_provenance"].read_text(encoding="utf-8-sig")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("候选 provenance 无法读取") from exc
    provenance_errors = review_bundle._validate_candidate_provenance(
        provenance,
        profile=profile,
        formal_execution_lock=dict(formal_execution_lock),
    )
    if provenance_errors:
        raise ValueError(
            "候选 provenance 身份门禁未通过: " + ",".join(provenance_errors)
        )
    pip_version = provenance.get("pip_version")
    if not isinstance(pip_version, str) or not pip_version:
        raise ValueError("候选 provenance 缺少 pip 版本")
    wheels, report_pip_version = candidate_materializer.load_resolved_wheels(
        paths["pip_resolver_report"],
        profile,
        expected_pip_version=pip_version,
    )
    if report_pip_version != pip_version:
        raise ValueError("pip 报告版本与候选 provenance 不一致")
    canonical_text = candidate_materializer.candidate_lock_text(wheels)
    try:
        candidate_bytes = paths["candidate_lock"].read_bytes()
        candidate_text = candidate_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("候选锁不是规范 UTF-8 文本") from exc
    if candidate_text != canonical_text:
        raise ValueError("候选锁文本与 pip 报告重建结果不一致")
    logical_digest = candidate_materializer.candidate_lock_logical_digest(wheels)
    if provenance.get("candidate_lock_logical_digest") != logical_digest:
        raise ValueError("候选锁逻辑摘要与 provenance 不一致")
    if provenance.get("candidate_lock_dependency_count") != len(wheels):
        raise ValueError("候选锁依赖数量与 provenance 不一致")
    return candidate_bytes, logical_digest, len(wheels)


def write_reviewed_dependency_hash_lock(
    profile_id: str,
    review_bundle_dir: str | Path,
    approval_profile_id: str,
    *,
    repository_root: str | Path = ROOT,
    git_command_runner: GitCommandRunner = _run_git_command,
) -> tuple[dict[str, Any], Path]:
    """显式确认 profile 后复验审查包并写入 registry 锁目标."""

    root = Path(repository_root).resolve()
    registry_path = root / "configs/dependency_profile_registry.json"
    profile = get_dependency_profile(profile_id, registry_path)
    bundle_dir = Path(review_bundle_dir).expanduser().resolve()
    report, report_path = _report_skeleton(
        profile,
        repository_root=root,
        bundle_dir=bundle_dir,
    )
    if approval_profile_id != profile.profile_name:
        return _write_failure(
            report,
            report_path,
            "approval_profile_mismatch",
            "显式批准的 profile 与目标 profile 不一致.",
        )
    if profile.complete_hash_lock_present or profile.formal_ready:
        return _write_failure(
            report,
            report_path,
            "complete_hash_lock_already_present",
            "目标完整哈希锁已经存在, 接收入口不允许覆盖.",
        )
    if profile.readiness_blockers != ("complete_hash_lock_missing",):
        return _write_failure(
            report,
            report_path,
            "profile_not_ready_for_lock_acceptance",
            "目标 profile 不是仅缺完整哈希锁的闭锁状态.",
        )
    if not bundle_dir.is_dir() or bundle_dir.is_symlink():
        return _write_failure(
            report,
            report_path,
            "review_bundle_directory_invalid",
            "审查包必须是实际目录且不得是符号链接.",
        )

    try:
        current_head = _read_clean_repository_head(
            root,
            git_command_runner=git_command_runner,
            report=report,
        )
        manifest_path = bundle_dir / review_bundle.BUNDLE_MANIFEST_FILE_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        formal_execution_lock = _validate_bundle_identity(
            profile,
            manifest=manifest,
            current_head=current_head,
        )
        paths = _bundle_files(bundle_dir, manifest)
        candidate_bytes, logical_digest, dependency_count = (
            _validate_candidate_closure(
                profile,
                paths=paths,
                formal_execution_lock=formal_execution_lock,
            )
        )
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
            "review_bundle_validation_failed",
            str(exc),
        )

    report["formal_execution_lock"] = formal_execution_lock
    report["formal_execution_commit"] = formal_execution_lock[
        "formal_execution_commit"
    ]
    report["formal_execution_lock_digest"] = formal_execution_lock[
        "formal_execution_lock_digest"
    ]
    target_path = (root / profile.complete_hash_lock_path).resolve()
    try:
        target_path.relative_to(root)
    except ValueError:
        return _write_failure(
            report,
            report_path,
            "complete_hash_lock_path_invalid",
            "registry 完整哈希锁路径越过仓库根目录.",
        )
    target_written = False
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("xb") as handle:
            handle.write(candidate_bytes)
        target_written = True
        accepted_profile = get_dependency_profile(profile.profile_name, registry_path)
        if not accepted_profile.formal_ready or accepted_profile.readiness_blockers:
            raise ValueError("写入后的完整哈希锁未通过 registry 正式门禁")
        if accepted_profile.complete_hash_lock_digest != logical_digest:
            raise ValueError("写入后的完整哈希锁摘要与审查候选不一致")
        if accepted_profile.complete_hash_lock_dependency_count != dependency_count:
            raise ValueError("写入后的完整哈希锁依赖数量与审查候选不一致")
    except (FileExistsError, OSError, ValueError) as exc:
        if target_written:
            target_path.unlink(missing_ok=True)
        return _write_failure(
            report,
            report_path,
            "complete_hash_lock_write_failed",
            str(exc),
        )

    report["complete_hash_lock_path"] = profile.complete_hash_lock_path
    report["complete_hash_lock_digest"] = logical_digest
    report["complete_hash_lock_dependency_count"] = dependency_count
    report["decision"] = SUCCESS_DECISION
    report["failure_reasons"] = []
    report["diagnostic_message"] = None
    report["supports_paper_claim"] = False
    _write_json(report_path, report)
    return report, report_path


def build_parser() -> argparse.ArgumentParser:
    """构造人工批准后的锁接收 CLI."""

    parser = argparse.ArgumentParser(
        description="复验依赖锁审查包并写入可提交完整哈希锁."
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=REQUIRED_DEPENDENCY_PROFILE_NAMES,
    )
    parser.add_argument("--review-bundle-dir", required=True)
    parser.add_argument(
        "--approve-profile",
        required=True,
        choices=REQUIRED_DEPENDENCY_PROFILE_NAMES,
        help="必须与 --profile 完全相同的显式人工批准值.",
    )
    parser.add_argument("--root", default=".")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """运行锁接收并用退出码表达是否已形成可提交锁."""

    arguments = build_parser().parse_args(argv)
    try:
        report, report_path = write_reviewed_dependency_hash_lock(
            arguments.profile,
            arguments.review_bundle_dir,
            arguments.approve_profile,
            repository_root=arguments.root,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "profile_id": arguments.profile,
                    "decision": "fail",
                    "failure_reasons": [
                        f"dependency_hash_lock_write_error:{type(exc).__name__}"
                    ],
                    "diagnostic_message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "profile_id": report["profile_id"],
                "report_path": str(report_path),
                "complete_hash_lock_path": report["complete_hash_lock_path"],
                "decision": report["decision"],
                "failure_reasons": report["failure_reasons"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["decision"] == SUCCESS_DECISION else 1


if __name__ == "__main__":
    raise SystemExit(main())
