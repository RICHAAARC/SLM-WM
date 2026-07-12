"""在脱离开发仓库的目录中验证论文代码包完整性与可执行入口."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runtime.dependency_profiles import (  # noqa: E402
    REQUIRED_DEPENDENCY_PROFILE_NAMES,
    load_dependency_profile_registry,
)


MANIFEST_FILE_NAME = "extraction_manifest.json"
MANIFEST_SCHEMA = "release_package_extraction_manifest"
MANIFEST_SCHEMA_VERSION = 2
SUPPORTED_STANDALONE_PROFILES = frozenset(
    {
        "paper_artifact_rebuild_package",
        "paper_experiment_execution_package",
    }
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    """流式计算抽离文件的实际 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """执行只读 Git 核验并在失败时提供稳定诊断."""

    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"Git 核验失败: {' '.join(arguments)}: {diagnostic}")
    return completed


def _normalized_relative_path(value: Any) -> str:
    """解析 manifest 路径并拒绝越界或平台相关写法."""

    if not isinstance(value, str):
        raise ValueError("抽离文件路径必须是字符串")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError(f"抽离文件路径无效: {value}")
    return value


def _load_manifest(root: Path) -> dict[str, Any]:
    """读取并核验独立代码包 manifest 的固定顶层契约."""

    manifest_path = root / MANIFEST_FILE_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("无法读取抽离 manifest") from exc
    if not isinstance(manifest, dict):
        raise ValueError("抽离 manifest 必须是 JSON 对象")
    expected_fields = {
        "extraction_manifest_schema",
        "schema_version",
        "profile_name",
        "source_repository_commit",
        "copied_files",
        "copied_file_records",
        "missing_paths",
        "excluded_parts",
        "standalone_repository",
        "complete_dependency_locks_required",
        "required_entrypoints",
        "dry_run",
        "supports_paper_claim",
    }
    if set(manifest) != expected_fields:
        raise ValueError("抽离 manifest 顶层字段集合不一致")
    if manifest["extraction_manifest_schema"] != MANIFEST_SCHEMA:
        raise ValueError("抽离 manifest schema 不受支持")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("抽离 manifest schema 版本不受支持")
    if manifest["profile_name"] not in SUPPORTED_STANDALONE_PROFILES:
        raise ValueError("抽离 profile 不是可独立执行类型")
    if manifest["standalone_repository"] is not True:
        raise ValueError("抽离 manifest 未声明独立 Git 仓库")
    if manifest["complete_dependency_locks_required"] is not True:
        raise ValueError("独立代码包未要求完整依赖锁")
    if manifest["dry_run"] is not False:
        raise ValueError("dry-run manifest 不能作为独立代码包")
    if manifest["supports_paper_claim"] is not False:
        raise ValueError("代码抽离本身不得支持论文结论")
    source_commit = manifest["source_repository_commit"]
    if (
        not isinstance(source_commit, str)
        or _GIT_COMMIT_PATTERN.fullmatch(source_commit) is None
    ):
        raise ValueError("抽离 manifest 缺少精确源仓库提交")
    if manifest["missing_paths"] != []:
        raise ValueError("独立代码包仍包含缺失输入")
    return manifest


def _validate_file_records(
    root: Path,
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """逐文件复算摘要、大小和唯一路径集合."""

    copied_files = manifest["copied_files"]
    records = manifest["copied_file_records"]
    if not isinstance(copied_files, list) or not isinstance(records, list):
        raise ValueError("抽离文件清单必须是列表")
    normalized_files = tuple(_normalized_relative_path(value) for value in copied_files)
    if list(normalized_files) != sorted(set(normalized_files)):
        raise ValueError("抽离文件清单必须排序且不得重复")
    if len(records) != len(normalized_files):
        raise ValueError("抽离文件记录数量与清单不一致")

    record_paths: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("抽离文件记录字段集合不一致")
        relative = _normalized_relative_path(record["path"])
        digest = record["sha256"]
        size_bytes = record["size_bytes"]
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"抽离文件 SHA-256 无效: {relative}")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError(f"抽离文件大小无效: {relative}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"抽离文件越过代码包根目录: {relative}") from exc
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"抽离文件不存在或不是普通文件: {relative}")
        if path.stat().st_size != size_bytes or _sha256(path) != digest:
            raise ValueError(f"抽离文件字节身份不一致: {relative}")
        record_paths.append(relative)
    if tuple(record_paths) != normalized_files:
        raise ValueError("抽离文件记录顺序或路径集合不一致")
    return normalized_files


def _tracked_files(root: Path) -> set[str]:
    """读取独立 Git 根提交跟踪的精确文件集合."""

    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
        shell=False,
    )
    if completed.returncode != 0:
        raise ValueError("无法读取独立代码包跟踪文件")
    return {
        entry.decode("utf-8")
        for entry in completed.stdout.split(b"\0")
        if entry
    }


def _validate_git_identity(root: Path, copied_files: tuple[str, ...]) -> str:
    """要求代码包是自包含、clean、detached 且文件集合精确的 Git 根."""

    top_level = Path(
        _run_git(root, ["rev-parse", "--show-toplevel"]).stdout.strip()
    ).resolve()
    if top_level != root:
        raise ValueError("独立代码包 Git 根与验证根目录不一致")
    commit = _run_git(root, ["rev-parse", "--verify", "HEAD^{commit}"]).stdout.strip()
    if _GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError("独立代码包 HEAD 不是精确40位小写 Git SHA")
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if symbolic.returncode != 1:
        raise ValueError("独立代码包 HEAD 必须 detached")
    status = _run_git(
        root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
    ).stdout
    if status:
        raise ValueError("独立代码包工作树必须 clean")
    expected_tracked = {*copied_files, MANIFEST_FILE_NAME}
    if _tracked_files(root) != expected_tracked:
        raise ValueError("独立代码包跟踪文件与抽离 manifest 不一致")
    return commit


def _validate_dependency_locks(root: Path) -> list[dict[str, Any]]:
    """使用运行时真实 parser 复验六个完整哈希锁均处于 ready."""

    profiles = load_dependency_profile_registry(
        root / "configs/dependency_profile_registry.json"
    )
    if tuple(profiles) != REQUIRED_DEPENDENCY_PROFILE_NAMES:
        raise ValueError("独立代码包依赖 profile 集合或顺序不一致")
    records: list[dict[str, Any]] = []
    for profile in profiles.values():
        if not profile.formal_ready or profile.readiness_blockers:
            raise ValueError(f"独立代码包依赖锁未 ready: {profile.profile_name}")
        records.append(
            {
                "profile_id": profile.profile_name,
                "profile_digest": profile.profile_digest,
                "complete_hash_lock_path": profile.complete_hash_lock_path,
                "complete_hash_lock_digest": profile.complete_hash_lock_digest,
                "complete_hash_lock_dependency_count": (
                    profile.complete_hash_lock_dependency_count
                ),
                "formal_ready": True,
            }
        )
    return records


def _validate_entrypoints(
    root: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """在不继承开发仓库 ``PYTHONPATH`` 的子进程中执行入口帮助门禁."""

    entrypoints = manifest["required_entrypoints"]
    if not isinstance(entrypoints, list) or not entrypoints:
        raise ValueError("独立代码包缺少必需入口清单")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    records: list[dict[str, Any]] = []
    for value in entrypoints:
        relative = _normalized_relative_path(value)
        path = root / relative
        if not path.is_file():
            raise ValueError(f"独立代码包入口不存在: {relative}")
        completed = subprocess.run(
            [sys.executable, "-I", str(path), "--help"],
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
        if completed.returncode != 0:
            diagnostic = completed.stderr.strip() or completed.stdout.strip()
            raise ValueError(f"独立代码包入口无法启动: {relative}: {diagnostic}")
        records.append(
            {
                "entrypoint": relative,
                "return_code": completed.returncode,
            }
        )
    return records


def validate_extracted_package(root: str | Path) -> dict[str, Any]:
    """完整验证一个已经抽离并初始化 Git 的论文代码包."""

    root_path = Path(root).resolve()
    report: dict[str, Any] = {
        "report_schema": "extracted_package_validation_report",
        "schema_version": 1,
        "profile_name": None,
        "source_repository_commit": None,
        "package_repository_commit": None,
        "dependency_profile_records": [],
        "entrypoint_records": [],
        "paper_workflow_excluded": False,
        "decision": "fail",
        "failure_reasons": [],
        "diagnostic_message": None,
        "supports_paper_claim": False,
    }
    try:
        manifest = _load_manifest(root_path)
        report["profile_name"] = manifest["profile_name"]
        report["source_repository_commit"] = manifest[
            "source_repository_commit"
        ]
        copied_files = _validate_file_records(root_path, manifest)
        report["package_repository_commit"] = _validate_git_identity(
            root_path,
            copied_files,
        )
        if (root_path / "paper_workflow").exists():
            raise ValueError("独立代码包不得包含 Colab 或 Notebook 外层")
        report["paper_workflow_excluded"] = True
        report["dependency_profile_records"] = _validate_dependency_locks(root_path)
        report["entrypoint_records"] = _validate_entrypoints(root_path, manifest)
        status = _run_git(
            root_path,
            ["status", "--porcelain=v1", "--untracked-files=all"],
        ).stdout
        if status:
            raise ValueError("入口验证后独立代码包工作树不再 clean")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        report["failure_reasons"] = [
            f"extracted_package_validation_failed:{type(exc).__name__}"
        ]
        report["diagnostic_message"] = str(exc)
        return report

    report["decision"] = "pass"
    report["failure_reasons"] = []
    report["diagnostic_message"] = None
    return report


def build_parser() -> argparse.ArgumentParser:
    """构造独立代码包验证 CLI."""

    parser = argparse.ArgumentParser(description="验证已抽离论文代码包.")
    parser.add_argument("--root", default=".", help="已抽离代码包根目录.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """运行验证并把报告输出到标准输出, 不在代码包内产生持久文件."""

    arguments = build_parser().parse_args(argv)
    report = validate_extracted_package(arguments.root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
