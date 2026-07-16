"""按受治理 profile 生成可发布或可独立执行的论文代码包."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runtime.dependency_profiles import (  # noqa: E402
    load_dependency_profile_registry,
)
from scripts.build_specification_inventory import (  # noqa: E402
    BUILD_SPECIFICATION_PATHS,
    CORE_METHOD_SPECIFICATION_PATHS,
)


EXTRACTION_MANIFEST_SCHEMA = "release_package_extraction_manifest"
EXTRACTION_MANIFEST_SCHEMA_VERSION = 3
EXTRACTION_MANIFEST_FILE_NAME = "extraction_manifest.json"
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ExtractionProfile:
    """表示一个可执行的论文代码包抽离契约."""

    profile_name: str
    include_paths: tuple[str, ...]
    exclude_parts: tuple[str, ...]
    standalone_repository: bool = False
    complete_dependency_locks_required: bool = False
    required_entrypoints: tuple[str, ...] = ()
    mapped_files: tuple[tuple[str, str], ...] = ()


COMMON_EXCLUDED_PARTS = (
    ".codex",
    "tools",
    "paper_workflow",
    "audit_reports",
    "outputs",
    "__pycache__",
    ".pytest_cache",
)


PROFILES = {
    "minimal_method_package": ExtractionProfile(
        profile_name="minimal_method_package",
        include_paths=(
            "main",
            "configs/core_method_dependency_identity.json",
            "configs/model_sd35.yaml",
            "configs/model_source_registry.json",
            "pyproject.toml",
            *CORE_METHOD_SPECIFICATION_PATHS,
        ),
        exclude_parts=(
            *COMMON_EXCLUDED_PARTS,
            "tests",
            "experiments",
            "paper_experiments",
            "external_baseline",
            "scripts",
        ),
        standalone_repository=True,
        complete_dependency_locks_required=False,
        required_entrypoints=("validate_core_method_package.py",),
        mapped_files=(
            ("docs/core_method_package_readme.md", "README.md"),
            (
                "scripts/validate_core_method_package.py",
                "validate_core_method_package.py",
            ),
        ),
    ),
    "paper_artifact_rebuild_package": ExtractionProfile(
        profile_name="paper_artifact_rebuild_package",
        include_paths=(
            "main",
            "configs",
            "experiments",
            "paper_experiments",
            "scripts",
            "docs/artifact_rebuild.md",
            "docs/field_registry.md",
            "docs/file_organization.md",
            "docs/release_boundary.md",
            "docs/extraction_profiles.md",
            "docs/intermediate_state_governance.md",
            "docs/placeholder_random_governance.md",
            "docs/paper_quality_evidence_governance.md",
            "docs/core_method_package_readme.md",
            "docs/release_layer_boundary.md",
            "docs/legacy/method_semantic_invariants.md",
            *BUILD_SPECIFICATION_PATHS,
            ".gitignore",
            ".gitattributes",
            "README.md",
            "pyproject.toml",
        ),
        exclude_parts=(
            *COMMON_EXCLUDED_PARTS,
            "tests",
            "external_baseline",
        ),
        standalone_repository=True,
        complete_dependency_locks_required=True,
        required_entrypoints=(
            "scripts/validate_extracted_package.py",
            "scripts/run_gpu_server_result_closure.py",
            "scripts/write_paper_profile_protocol_isomorphism_report.py",
        ),
    ),
    "paper_experiment_execution_package": ExtractionProfile(
        profile_name="paper_experiment_execution_package",
        include_paths=(
            "main",
            "configs",
            "experiments",
            "paper_experiments",
            "scripts",
            "external_baseline/README.md",
            "external_baseline/source_registry.json",
            "external_baseline/primary",
            "docs/artifact_rebuild.md",
            "docs/field_registry.md",
            "docs/file_organization.md",
            "docs/release_boundary.md",
            "docs/release_layer_boundary.md",
            "docs/extraction_profiles.md",
            "docs/intermediate_state_governance.md",
            "docs/placeholder_random_governance.md",
            "docs/paper_quality_evidence_governance.md",
            "docs/core_method_package_readme.md",
            "docs/legacy/method_semantic_invariants.md",
            *BUILD_SPECIFICATION_PATHS,
            ".gitignore",
            ".gitattributes",
            "README.md",
            "pyproject.toml",
        ),
        exclude_parts=(
            *COMMON_EXCLUDED_PARTS,
            "tests",
            "source",
        ),
        standalone_repository=True,
        complete_dependency_locks_required=True,
        required_entrypoints=(
            "scripts/validate_extracted_package.py",
            "scripts/run_formal_workflow_host.py",
            "scripts/formal_workflow_entry.py",
            "scripts/run_gpu_server_workflow.py",
            "scripts/run_gpu_server_result_closure.py",
            "scripts/run_gpu_method_qualification.py",
            "scripts/write_paper_profile_protocol_isomorphism_report.py",
        ),
    ),
}


def _sha256(path: Path) -> str:
    """流式计算文件 SHA-256, 供复制前后身份复验与 manifest 记录复用."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(
    repository_root: Path,
    arguments: Sequence[str],
    *,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """执行抽离身份所需的 Git 命令, 并统一保留标准输出和错误."""

    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Git 命令失败: {' '.join(arguments)}: {diagnostic}")
    return completed


def _source_repository_commit(root_path: Path, *, require_clean: bool) -> str:
    """读取精确源提交, 正式抽离时同时要求源工作树 clean."""

    commit = _run_git(
        root_path,
        ["rev-parse", "--verify", "HEAD^{commit}"],
    ).stdout.strip()
    if _GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError("抽离源仓库 HEAD 必须是精确40位小写 Git SHA")
    if require_clean:
        status = _run_git(
            root_path,
            ["status", "--porcelain=v1", "--untracked-files=all"],
        ).stdout
        if status:
            raise ValueError("独立执行包只能从 clean 源工作树抽离")
    return commit


def should_skip(relative_path: Path, exclude_parts: Iterable[str]) -> bool:
    """判断相对路径是否应从抽离包中排除."""

    normalized = relative_path.as_posix()
    parts = set(relative_path.parts)
    for excluded in exclude_parts:
        excluded_normalized = excluded.strip("/").replace("\\", "/")
        if (
            excluded_normalized in parts
            or normalized == excluded_normalized
            or normalized.startswith(f"{excluded_normalized}/")
        ):
            return True
    return False


def iter_copy_candidates(
    root_path: Path,
    include_path: str,
    profile: ExtractionProfile,
) -> Iterable[Path]:
    """遍历某个 include path 下允许复制的普通文件."""

    source = root_path / include_path
    if not source.exists():
        return
    if source.is_file():
        relative = source.relative_to(root_path)
        if not should_skip(relative, profile.exclude_parts):
            yield source
        return
    for path in source.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root_path)
        if not should_skip(relative, profile.exclude_parts):
            yield path


def _validate_relative_file_path(value: str) -> str:
    """拒绝绝对路径、反斜杠和父目录跳转, 保持 manifest 可移植."""

    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError(f"抽离文件路径不是规范 POSIX 相对路径: {value}")
    return value


def _initialize_standalone_repository(
    output_path: Path,
    profile: ExtractionProfile,
) -> str:
    """为独立执行包创建确定性 SHA-1 Git 根并切换到 clean detached HEAD.

    正式工作流以 clean detached commit 作为代码身份. 抽离包不能依赖原仓库
    的 ``.git`` 目录, 因而在包内对已复制文件和 manifest 创建新的根提交.
    manifest 已记录源提交和逐文件摘要, 可将发布提交映射回开发仓库.
    """

    _run_git(
        output_path,
        ["init", "--quiet", "--initial-branch=main", "--object-format=sha1"],
    )
    _run_git(output_path, ["config", "core.autocrlf", "false"])
    _run_git(output_path, ["config", "core.filemode", "false"])
    _run_git(output_path, ["config", "user.name", "SLM-WM Release"])
    _run_git(output_path, ["config", "user.email", "release@slm-wm.invalid"])
    _run_git(output_path, ["add", "--all"])
    commit_environment = dict(os.environ)
    commit_environment.update(
        {
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
        }
    )
    _run_git(
        output_path,
        [
            "commit",
            "--quiet",
            "--no-gpg-sign",
            "-m",
            f"构建独立{profile.profile_name}代码包",
        ],
        environment=commit_environment,
    )
    commit = _run_git(
        output_path,
        ["rev-parse", "--verify", "HEAD^{commit}"],
    ).stdout.strip()
    if _GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError("独立执行包提交不是精确40位小写 Git SHA")
    _run_git(output_path, ["checkout", "--quiet", "--detach", commit])
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=output_path,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if symbolic.returncode != 1:
        raise RuntimeError("独立执行包未进入 detached HEAD")
    status = _run_git(
        output_path,
        ["status", "--porcelain=v1", "--untracked-files=all"],
    ).stdout
    if status:
        raise RuntimeError("独立执行包初始化后工作树不 clean")
    return commit


def _copy_file_records(
    root_path: Path,
    output_path: Path,
    copied_file_mappings: Sequence[tuple[str, str]],
    *,
    dry_run: bool,
) -> list[dict[str, object]]:
    """复制唯一文件集合, 并记录复制后实际字节摘要和大小."""

    records: list[dict[str, object]] = []
    for source_text, target_text in copied_file_mappings:
        source_file = root_path / Path(source_text)
        source_digest = _sha256(source_file)
        if not dry_run:
            target_file = output_path / Path(target_text)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
            if _sha256(target_file) != source_digest:
                raise RuntimeError(f"抽离文件复制后 SHA-256 不一致: {target_text}")
        records.append(
            {
                "path": target_text,
                "sha256": source_digest,
                "size_bytes": source_file.stat().st_size,
                "source_path": source_text,
            }
        )
    return records


def _missing_complete_dependency_locks(root_path: Path) -> list[str]:
    """返回 registry 中缺失或未通过完整锁门禁的目标路径."""

    profiles = load_dependency_profile_registry(
        root_path / "configs/dependency_profile_registry.json"
    )
    return sorted(
        profile.complete_hash_lock_path
        for profile in profiles.values()
        if not profile.formal_ready or profile.readiness_blockers
    )


def extract_profile(
    root: str | Path,
    output: str | Path,
    profile_name: str,
    dry_run: bool = False,
) -> dict[str, object]:
    """按指定 profile 复制文件并返回可重建的稳定抽离 manifest."""

    root_path = Path(root).resolve()
    output_path = Path(output).resolve()
    if profile_name not in PROFILES:
        raise ValueError(f"不支持的抽离 profile: {profile_name}")
    profile = PROFILES[profile_name]
    if not dry_run and output_path.exists() and any(output_path.iterdir()):
        raise ValueError("抽离输出目录必须不存在或为空, 以阻止陈旧文件混入")

    copied_file_sources: dict[str, str] = {}
    missing_paths: list[str] = []
    for include_path in profile.include_paths:
        source = root_path / include_path
        if not source.exists():
            missing_paths.append(include_path)
            continue
        for source_file in iter_copy_candidates(root_path, include_path, profile):
            relative_text = _validate_relative_file_path(
                source_file.relative_to(root_path).as_posix()
            )
            copied_file_sources[relative_text] = relative_text
    for source_text, target_text in profile.mapped_files:
        source_relative = _validate_relative_file_path(source_text)
        target_relative = _validate_relative_file_path(target_text)
        source = root_path / source_relative
        if not source.is_file() or source.is_symlink():
            missing_paths.append(source_relative)
            continue
        existing_source = copied_file_sources.get(target_relative)
        if existing_source is not None and existing_source != source_relative:
            raise ValueError(
                "抽离 profile 多个源文件映射到同一包路径: "
                f"{existing_source}, {source_relative} -> {target_relative}"
            )
        copied_file_sources[target_relative] = source_relative
    for entrypoint in profile.required_entrypoints:
        if entrypoint not in copied_file_sources and entrypoint not in missing_paths:
            missing_paths.append(entrypoint)
    if profile.complete_dependency_locks_required:
        try:
            missing_lock_paths = _missing_complete_dependency_locks(root_path)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            missing_lock_paths = ["configs/dependency_profile_registry.json"]
        for lock_path in missing_lock_paths:
            if lock_path not in missing_paths:
                missing_paths.append(lock_path)
    copied_files = sorted(copied_file_sources)
    copied_file_mappings = [
        (copied_file_sources[target], target)
        for target in copied_files
    ]
    if not copied_files:
        raise ValueError("抽离 profile 未产生任何文件")
    if not dry_run and missing_paths:
        raise FileNotFoundError(
            "正式抽离缺少登记输入: " + ", ".join(missing_paths)
        )
    try:
        source_commit = _source_repository_commit(
            root_path,
            require_clean=not dry_run and profile.standalone_repository,
        )
    except RuntimeError:
        if not dry_run:
            raise
        source_commit = ""

    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    copied_file_records = _copy_file_records(
        root_path,
        output_path,
        copied_file_mappings,
        dry_run=dry_run,
    )
    manifest: dict[str, object] = {
        "extraction_manifest_schema": EXTRACTION_MANIFEST_SCHEMA,
        "schema_version": EXTRACTION_MANIFEST_SCHEMA_VERSION,
        "profile_name": profile.profile_name,
        "source_repository_commit": source_commit,
        "copied_files": copied_files,
        "copied_file_records": copied_file_records,
        "missing_paths": missing_paths,
        "excluded_parts": list(profile.exclude_parts),
        "standalone_repository": profile.standalone_repository,
        "complete_dependency_locks_required": (
            profile.complete_dependency_locks_required
        ),
        "required_entrypoints": list(profile.required_entrypoints),
        "dry_run": dry_run,
        "supports_paper_claim": False,
    }
    if not dry_run:
        manifest_path = output_path / EXTRACTION_MANIFEST_FILE_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        if profile.standalone_repository:
            _initialize_standalone_repository(output_path, profile)
    return manifest


def _require_cli_output_under_outputs(root: Path, output: Path) -> None:
    """要求仓库命令的持久化抽离结果写在 ``outputs/`` 内."""

    outputs_root = (root / "outputs").resolve()
    try:
        output.resolve().relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("抽离命令输出必须位于仓库 outputs/ 下") from exc


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器."""

    parser = argparse.ArgumentParser(description="按治理 profile 抽离论文代码包.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="minimal_method_package",
        help="选择抽离 profile.",
    )
    parser.add_argument("--root", default=".", help="仓库根目录.")
    parser.add_argument(
        "--output",
        required=True,
        help="位于仓库 outputs/ 下的目标目录.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出文件与摘要清单, 不写入代码包.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行抽离并输出稳定 manifest."""

    arguments = build_parser().parse_args(argv)
    root = Path(arguments.root).resolve()
    output = Path(arguments.output).resolve()
    if not arguments.dry_run:
        _require_cli_output_under_outputs(root, output)
    manifest = extract_profile(
        root,
        output,
        arguments.profile,
        arguments.dry_run,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
