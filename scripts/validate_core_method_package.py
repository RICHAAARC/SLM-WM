"""在脱离开发仓库的目录中验证最小核心方法包。"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import pkgutil
import re
import subprocess
import sys
import tomllib
from typing import Any, Mapping, Sequence


MANIFEST_FILE_NAME = "extraction_manifest.json"
MANIFEST_SCHEMA = "release_package_extraction_manifest"
MANIFEST_SCHEMA_VERSION = 3
PROFILE_NAME = "minimal_method_package"
DEPENDENCY_IDENTITY_PATH = "configs/core_method_dependency_identity.json"
VALIDATION_ENTRYPOINT = "validate_core_method_package.py"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_TOP_LEVEL_PATHS = frozenset(
    {
        ".codex",
        "audit_reports",
        "experiments",
        "external_baseline",
        "outputs",
        "paper_experiments",
        "paper_workflow",
        "scripts",
        "tests",
        "tools",
    }
)


def _sha256(path: Path) -> str:
    """流式计算普通文件的 SHA-256, 避免把文件整体读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_relative_path(value: Any) -> str:
    """拒绝绝对路径、父目录跳转和平台相关路径写法。"""

    if not isinstance(value, str):
        raise ValueError("文件路径必须是字符串")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError(f"文件路径不是规范 POSIX 相对路径: {value}")
    return value


def _run_git(root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """执行只读 Git 命令, 并把失败收敛为稳定验证错误。"""

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


def _load_manifest(root: Path) -> dict[str, Any]:
    """读取最小包抽离 manifest 并核验固定顶层契约。"""

    try:
        manifest = json.loads(
            (root / MANIFEST_FILE_NAME).read_text(encoding="utf-8-sig")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("无法读取抽离 manifest") from exc
    expected_fields = {
        "complete_dependency_locks_required",
        "copied_file_records",
        "copied_files",
        "dry_run",
        "excluded_parts",
        "extraction_manifest_schema",
        "missing_paths",
        "profile_name",
        "required_entrypoints",
        "schema_version",
        "source_repository_commit",
        "standalone_repository",
        "supports_paper_claim",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        raise ValueError("抽离 manifest 顶层字段集合不一致")
    if manifest["extraction_manifest_schema"] != MANIFEST_SCHEMA:
        raise ValueError("抽离 manifest schema 不受支持")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("抽离 manifest schema 版本不受支持")
    if manifest["profile_name"] != PROFILE_NAME:
        raise ValueError("抽离 manifest 不是最小核心方法 profile")
    if manifest["standalone_repository"] is not True:
        raise ValueError("最小核心方法包未声明独立 Git 仓库")
    if manifest["complete_dependency_locks_required"] is not False:
        raise ValueError("最小核心方法包不得要求论文实验依赖锁")
    if manifest["required_entrypoints"] != [VALIDATION_ENTRYPOINT]:
        raise ValueError("最小核心方法包验证入口清单不一致")
    if manifest["missing_paths"] != [] or manifest["dry_run"] is not False:
        raise ValueError("最小核心方法包存在缺失输入或来自 dry-run")
    if manifest["supports_paper_claim"] is not False:
        raise ValueError("最小核心方法包抽离不得支持论文结论")
    source_commit = manifest["source_repository_commit"]
    if (
        not isinstance(source_commit, str)
        or _GIT_COMMIT_PATTERN.fullmatch(source_commit) is None
    ):
        raise ValueError("抽离 manifest 缺少精确源仓库提交")
    return manifest


def _validate_file_records(
    root: Path,
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """逐文件复算大小与摘要, 并核验源路径到包路径的映射。"""

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
    source_paths: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
            "source_path",
        }:
            raise ValueError("抽离文件记录字段集合不一致")
        relative = _normalized_relative_path(record["path"])
        source_relative = _normalized_relative_path(record["source_path"])
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
        source_paths[relative] = source_relative
    if tuple(record_paths) != normalized_files:
        raise ValueError("抽离文件记录顺序或路径集合不一致")
    if source_paths.get("README.md") != "docs/core_method_package_readme.md":
        raise ValueError("最小核心方法包 README 未绑定专用源文件")
    if source_paths.get(VALIDATION_ENTRYPOINT) != (
        "scripts/validate_core_method_package.py"
    ):
        raise ValueError("最小核心方法包验证入口源映射不一致")
    return normalized_files


def _tracked_files(root: Path) -> set[str]:
    """读取独立 Git 根提交跟踪的精确文件集合。"""

    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
        shell=False,
    )
    if completed.returncode != 0:
        raise ValueError("无法读取独立核心方法包跟踪文件")
    return {
        entry.decode("utf-8")
        for entry in completed.stdout.split(b"\0")
        if entry
    }


def _validate_git_identity(root: Path, copied_files: tuple[str, ...]) -> str:
    """要求最小包是自包含、clean、detached 且文件集合精确的 Git 根。"""

    top_level = Path(
        _run_git(root, ["rev-parse", "--show-toplevel"]).stdout.strip()
    ).resolve()
    if top_level != root:
        raise ValueError("独立核心方法包 Git 根与验证根目录不一致")
    commit = _run_git(root, ["rev-parse", "--verify", "HEAD^{commit}"]).stdout.strip()
    if _GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError("独立核心方法包 HEAD 不是精确40位小写 Git SHA")
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if symbolic.returncode != 1:
        raise ValueError("独立核心方法包 HEAD 必须 detached")
    if _run_git(root, ["status", "--porcelain=v1", "--untracked-files=all"]).stdout:
        raise ValueError("独立核心方法包工作树必须 clean")
    expected = {*copied_files, MANIFEST_FILE_NAME}
    if _tracked_files(root) != expected:
        raise ValueError("独立核心方法包跟踪文件与 manifest 不一致")
    return commit


def _validate_release_boundary(root: Path, copied_files: tuple[str, ...]) -> None:
    """拒绝外层目录, 并要求最小包只保留约定的根文件和路径。"""

    required_files = {
        "README.md",
        "configs/core_method_dependency_identity.json",
        "configs/model_sd35.yaml",
        "configs/model_source_registry.json",
        "main/__init__.py",
        "pyproject.toml",
        VALIDATION_ENTRYPOINT,
    }
    missing = sorted(required_files - set(copied_files))
    if missing:
        raise ValueError("最小核心方法包缺少必需文件: " + ", ".join(missing))
    allowed_config_files = {
        "configs/core_method_dependency_identity.json",
        "configs/model_sd35.yaml",
        "configs/model_source_registry.json",
    }
    allowed_root_files = {
        "README.md",
        "pyproject.toml",
        VALIDATION_ENTRYPOINT,
    }
    for relative in copied_files:
        top_level = PurePosixPath(relative).parts[0]
        if top_level in _FORBIDDEN_TOP_LEVEL_PATHS:
            raise ValueError(f"最小核心方法包包含外层路径: {relative}")
        if top_level == "configs" and relative not in allowed_config_files:
            raise ValueError(f"最小核心方法包包含未登记配置: {relative}")
        if top_level not in {"configs", "main"} and relative not in allowed_root_files:
            raise ValueError(f"最小核心方法包包含未登记根路径: {relative}")
    if not (root / "README.md").read_text(encoding="utf-8-sig").startswith(
        "# SLM-WM 核心方法包"
    ):
        raise ValueError("最小核心方法包未使用专用根 README")


def _load_dependency_identity(root: Path) -> dict[str, Any]:
    """读取核心依赖身份, 该协议只约束最小包而不消费论文依赖锁。"""

    try:
        identity = json.loads(
            (root / DEPENDENCY_IDENTITY_PATH).read_text(encoding="utf-8-sig")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("无法读取核心依赖身份") from exc
    expected = {
        "build_backend": "setuptools.build_meta",
        "build_requirements": ["setuptools>=69"],
        "dependency_identity_schema": "slm_wm_core_dependency_identity",
        "distribution_name": "slm-wm-core",
        "included_packages": ["main"],
        "requires_python": ">=3.11",
        "runtime_dependencies": ["torch>=2.11,<2.12"],
        "schema_version": 1,
    }
    if identity != expected:
        raise ValueError("核心依赖身份字段或取值不一致")
    return identity


def _validate_pyproject(root: Path, identity: Mapping[str, Any]) -> None:
    """核验标准构建元数据与核心依赖身份逐项一致。"""

    try:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("无法读取 pyproject.toml") from exc
    build_system = project.get("build-system")
    metadata = project.get("project")
    setuptools_config = project.get("tool", {}).get("setuptools", {})
    package_find = setuptools_config.get("packages", {}).get("find", {})
    if build_system != {
        "requires": identity["build_requirements"],
        "build-backend": identity["build_backend"],
    }:
        raise ValueError("pyproject 构建后端与核心依赖身份不一致")
    if not isinstance(metadata, dict):
        raise ValueError("pyproject 缺少标准 project 元数据")
    if metadata.get("name") != identity["distribution_name"]:
        raise ValueError("pyproject distribution 名称与核心依赖身份不一致")
    if metadata.get("requires-python") != identity["requires_python"]:
        raise ValueError("pyproject Python 约束与核心依赖身份不一致")
    if metadata.get("dependencies") != identity["runtime_dependencies"]:
        raise ValueError("pyproject 运行依赖与核心依赖身份不一致")
    if setuptools_config.get("include-package-data") is not False:
        raise ValueError("pyproject 必须关闭隐式 package data 收集")
    if package_find != {
        "where": ["."],
        "include": ["main", "main.*"],
        "namespaces": False,
    }:
        raise ValueError("pyproject 必须只发现 main 及其子包")


def _import_core_modules(root: Path) -> list[str]:
    """在显式包根中导入全部核心模块, 证明不依赖开发仓库路径。"""

    if sys.flags.isolated != 1:
        raise ValueError("核心包验证必须使用 python -I 启动")
    sys.dont_write_bytecode = True
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    package = importlib.import_module("main")
    module_names = ["main"]
    for module_info in pkgutil.walk_packages(
        package.__path__,
        prefix="main.",
    ):
        importlib.import_module(module_info.name)
        module_names.append(module_info.name)
    return sorted(module_names)


def validate_core_method_package(root: str | Path) -> dict[str, Any]:
    """完整验证已抽离的最小核心方法包, 不产生持久化文件。"""

    root_path = Path(root).resolve()
    report: dict[str, Any] = {
        "decision": "fail",
        "dependency_identity_schema": None,
        "diagnostic_message": None,
        "failure_reasons": [],
        "imported_modules": [],
        "package_repository_commit": None,
        "profile_name": PROFILE_NAME,
        "report_schema": "core_method_package_validation_report",
        "schema_version": 1,
        "source_repository_commit": None,
        "supports_paper_claim": False,
    }
    try:
        manifest = _load_manifest(root_path)
        report["source_repository_commit"] = manifest["source_repository_commit"]
        copied_files = _validate_file_records(root_path, manifest)
        report["package_repository_commit"] = _validate_git_identity(
            root_path,
            copied_files,
        )
        _validate_release_boundary(root_path, copied_files)
        identity = _load_dependency_identity(root_path)
        report["dependency_identity_schema"] = identity[
            "dependency_identity_schema"
        ]
        _validate_pyproject(root_path, identity)
        report["imported_modules"] = _import_core_modules(root_path)
        if _run_git(
            root_path,
            ["status", "--porcelain=v1", "--untracked-files=all"],
        ).stdout:
            raise ValueError("导入验证后独立核心方法包工作树不再 clean")
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        report["failure_reasons"] = [
            f"core_method_package_validation_failed:{type(exc).__name__}"
        ]
        report["diagnostic_message"] = str(exc)
        return report
    report["decision"] = "pass"
    return report


def build_parser() -> argparse.ArgumentParser:
    """构造独立核心方法包验证命令行。"""

    parser = argparse.ArgumentParser(description="验证已抽离的最小核心方法包.")
    parser.add_argument("--root", default=".", help="已抽离代码包根目录.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行验证并将结构化报告写到标准输出。"""

    arguments = build_parser().parse_args(argv)
    report = validate_core_method_package(arguments.root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
