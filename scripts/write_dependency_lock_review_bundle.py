"""生成单个依赖 profile 的候选锁审查包并可选镜像到 Drive.

该入口只编排已有候选物化器和 isolated Python 解释器创建 API. 它不解析
依赖、不维护包列表、不写入 ``configs/``, 也不把候选审查包作为论文证据.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runtime.dependency_preparation import (  # noqa: E402
    prepare_dependency_profile,
)
from experiments.runtime.dependency_profiles import (  # noqa: E402
    DependencyProfile,
    REQUIRED_DEPENDENCY_PROFILE_NAMES,
    WORKFLOW_ORCHESTRATOR_PROFILE_ID,
    get_dependency_profile,
)
from experiments.runtime.isolated_dependency_environment import (  # noqa: E402
    UV_DISTRIBUTION_VERSION,
    provision_isolated_dependency_python,
)
from experiments.runtime import repository_environment  # noqa: E402
from scripts import materialize_dependency_lock_candidate as candidate_materializer  # noqa: E402


LOCAL_BUNDLE_RELATIVE_ROOT = Path("outputs/dependency_lock_review_bundles")
BUNDLE_MANIFEST_FILE_NAME = "dependency_lock_review_bundle_manifest.json"
BUNDLE_MANIFEST_SCHEMA = "dependency_lock_review_bundle_manifest"
BUNDLE_MANIFEST_SCHEMA_VERSION = 1
SUCCESS_DECISION = "review_bundle_written"
QUALIFICATION_REPORT_SCHEMA = "dependency_lock_qualification_launcher_report"
QUALIFICATION_REPORT_SCHEMA_VERSION = 1
QUALIFICATION_SUCCESS_DECISION = "qualification_complete"
QUALIFICATION_TOOL_LOCK_RELATIVE_PATH = Path(
    "configs/dependency_profiles/dependency_qualification_uv_linux_x86_64_lock.txt"
)
DEFAULT_QUALIFICATION_RUNTIME_ROOT = (
    Path(tempfile.gettempdir()) / "slm_wm_dependency_lock_qualification"
)
QUALIFICATION_REPORT_RELATIVE_ROOT = Path(
    "outputs/dependency_lock_qualification"
)
QUALIFICATION_CHILD_ENVIRONMENT_KEY = (
    "SLM_WM_DEPENDENCY_LOCK_QUALIFICATION_CHILD"
)
QUALIFICATION_PYTHON_ENVIRONMENT_KEY = (
    "SLM_WM_DEPENDENCY_LOCK_QUALIFICATION_PYTHON"
)
QUALIFICATION_PYTHON_DIGEST_ENVIRONMENT_KEY = (
    "SLM_WM_DEPENDENCY_LOCK_QUALIFICATION_PYTHON_SHA256"
)
_QUALIFICATION_TOOL_LOCK_PATTERN = re.compile(
    r"^uv==(?P<version>[0-9]+\.[0-9]+\.[0-9]+) "
    r"--hash=sha256:(?P<digest>[0-9a-f]{64})$"
)

CURRENT_INTERPRETER_PROFILE_ID = WORKFLOW_ORCHESTRATOR_PROFILE_ID
ISOLATED_PYTHON_PROFILE_IDS = tuple(
    profile_id
    for profile_id in REQUIRED_DEPENDENCY_PROFILE_NAMES
    if profile_id != CURRENT_INTERPRETER_PROFILE_ID
)

ChildCommandRunner = Callable[[Sequence[str], Path], Any]
QualificationCommandRunner = Callable[
    [Sequence[str], Path, Mapping[str, str]],
    Any,
]


def _run_child_command(command: Sequence[str], working_directory: Path) -> dict[str, Any]:
    """执行 isolated Python 子解释器命令并捕获可审计诊断."""

    completed = subprocess.run(
        list(command),
        cwd=working_directory,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return {
        "return_code": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _run_qualification_command(
    command: Sequence[str],
    working_directory: Path,
    environment_overrides: Mapping[str, str],
) -> dict[str, Any]:
    """执行 host 资格化命令, 并让精确子解释器继承正式代码锁."""

    environment = dict(os.environ)
    environment.update(environment_overrides)
    completed = subprocess.run(
        list(command),
        cwd=working_directory,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return {
        "return_code": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _normalize_child_result(result: Any) -> dict[str, Any]:
    """把测试 runner 与 subprocess 结果收敛为统一命令记录."""

    if isinstance(result, int) and not isinstance(result, bool):
        return {"return_code": result, "stdout": "", "stderr": ""}
    if isinstance(result, subprocess.CompletedProcess):
        return {
            "return_code": int(result.returncode),
            "stdout": str(result.stdout or ""),
            "stderr": str(result.stderr or ""),
        }
    if isinstance(result, Mapping):
        return_code = result.get("return_code")
        if isinstance(return_code, bool) or not isinstance(return_code, int):
            raise ValueError("子解释器 runner 必须返回整数 return_code")
        return {
            "return_code": return_code,
            "stdout": str(result.get("stdout") or ""),
            "stderr": str(result.get("stderr") or ""),
        }
    raise TypeError("子解释器 runner 返回类型不受支持")


def _sha256(path: Path) -> str:
    """计算审查包实际文件的 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """以稳定排版写入本地审查包 manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _local_bundle_dir(repository_root: Path, profile_id: str) -> Path:
    """返回 ``outputs/`` 下单个 profile 的审查包目录."""

    outputs_root = (repository_root / "outputs").resolve()
    bundle_dir = (
        repository_root / LOCAL_BUNDLE_RELATIVE_ROOT / profile_id
    ).resolve()
    try:
        bundle_dir.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("本地依赖锁审查包必须位于 outputs/ 下") from exc
    return bundle_dir


def _resolve_drive_bundle_dir(
    repository_root: Path,
    drive_output_dir: str | Path | None,
    profile_id: str,
) -> Path | None:
    """仅在显式提供 Drive 根目录时解析 profile 镜像目录."""

    if drive_output_dir is None:
        return None
    drive_root = Path(drive_output_dir).expanduser()
    resolved_root = (
        drive_root.resolve()
        if drive_root.is_absolute()
        else (repository_root / drive_root).resolve()
    )
    return resolved_root / profile_id


def _candidate_source_paths(repository_root: Path, profile_id: str) -> dict[str, Path]:
    """返回已有候选物化器的三个受治理输出路径."""

    candidate_root = (
        repository_root / candidate_materializer.OUTPUT_RELATIVE_ROOT / profile_id
    ).resolve()
    return {
        "candidate_lock": candidate_root / candidate_materializer.CANDIDATE_LOCK_FILE_NAME,
        "pip_resolver_report": candidate_root / candidate_materializer.PIP_REPORT_FILE_NAME,
        "candidate_provenance": candidate_root / candidate_materializer.PROVENANCE_FILE_NAME,
    }


def _relative_or_absolute(repository_root: Path, path: Path) -> str:
    """仓库内路径写为相对 POSIX 形式, 外部 Drive 路径保留绝对值."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _build_manifest(
    profile: DependencyProfile,
    *,
    repository_root: Path,
    local_bundle_dir: Path,
    drive_bundle_dir: Path | None,
    formal_execution_lock: dict[str, Any] | None,
) -> dict[str, Any]:
    """构造成功与失败路径共享的审查包 manifest schema."""

    resolved_execution_lock = formal_execution_lock or {}
    return {
        "manifest_schema": BUNDLE_MANIFEST_SCHEMA,
        "schema_version": BUNDLE_MANIFEST_SCHEMA_VERSION,
        "profile_id": profile.profile_name,
        "profile_digest": profile.profile_digest,
        "direct_requirements_digest": profile.direct_requirements_digest,
        "review_execution_mode": (
            "isolated_python"
            if profile.profile_name in ISOLATED_PYTHON_PROFILE_IDS
            else "orchestrator_interpreter"
        ),
        "formal_execution_lock": resolved_execution_lock,
        "formal_execution_commit": resolved_execution_lock.get(
            "formal_execution_commit", ""
        ),
        "formal_execution_lock_digest": resolved_execution_lock.get(
            "formal_execution_lock_digest", ""
        ),
        "local_bundle_dir": _relative_or_absolute(repository_root, local_bundle_dir),
        "drive_bundle_dir": (
            None
            if drive_bundle_dir is None
            else _relative_or_absolute(repository_root, drive_bundle_dir)
        ),
        "drive_copy_performed": False,
        "orchestrator_preparation": None,
        "isolated_python_provision": None,
        "candidate_materialization": None,
        "files": [],
        "decision": "fail",
        "failure_reasons": [],
        "diagnostic_message": None,
        "supports_paper_claim": False,
    }


def _write_failure_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    reason: str,
    diagnostic_message: str,
) -> tuple[dict[str, Any], Path]:
    """持久化首个不可恢复的资格化失败."""

    manifest["decision"] = "fail"
    manifest["failure_reasons"] = [reason]
    manifest["diagnostic_message"] = diagnostic_message
    manifest["supports_paper_claim"] = False
    _write_json(manifest_path, manifest)
    return manifest, manifest_path


def _normalize_machine(machine: str) -> str:
    """统一 Linux x86_64 常见别名, 供资格化 host 身份门禁复用."""

    normalized = machine.strip().lower()
    if normalized in {"amd64", "x64", "x86-64"}:
        return "x86_64"
    return normalized


def _qualification_report_path(
    repository_root: Path,
    profile_id: str,
) -> Path:
    """返回 ``outputs/`` 下单个 profile 的资格化启动报告路径."""

    outputs_root = (repository_root / "outputs").resolve()
    report_path = (
        repository_root
        / QUALIFICATION_REPORT_RELATIVE_ROOT
        / profile_id
        / "dependency_lock_qualification_report.json"
    ).resolve()
    try:
        report_path.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("依赖锁资格化报告必须位于 outputs/ 下") from exc
    return report_path


def _qualification_report_skeleton(
    profile: DependencyProfile,
    *,
    repository_root: Path,
    qualification_runtime_root: Path,
    qualification_tool_lock_path: Path,
) -> tuple[dict[str, Any], Path]:
    """构造 host 启动器成功与失败路径共享的稳定报告."""

    report_path = _qualification_report_path(
        repository_root,
        profile.profile_name,
    )
    return {
        "report_schema": QUALIFICATION_REPORT_SCHEMA,
        "schema_version": QUALIFICATION_REPORT_SCHEMA_VERSION,
        "operation_kind": "dependency_lock_candidate_qualification",
        "profile_id": profile.profile_name,
        "profile_digest": profile.profile_digest,
        "formal_execution_lock": {},
        "qualification_tool_lock_path": _relative_or_absolute(
            repository_root,
            qualification_tool_lock_path,
        ),
        "qualification_tool_lock_digest": None,
        "environment_root": str(qualification_runtime_root),
        "managed_python_root": str(
            qualification_runtime_root / "managed_pythons"
        ),
        "isolated_environment_path": str(
            qualification_runtime_root / WORKFLOW_ORCHESTRATOR_PROFILE_ID
        ),
        "uv_distribution_version": UV_DISTRIBUTION_VERSION,
        "uv_executable_path": None,
        "uv_executable_sha256": None,
        "python_executable_path": None,
        "python_executable_sha256": None,
        "manifest_path": None,
        "command_results": [],
        "decision": "fail",
        "failure_reasons": [],
        "diagnostic_message": None,
        "supports_paper_claim": False,
    }, report_path


def _write_qualification_failure(
    report: dict[str, Any],
    report_path: Path,
    reason: str,
    diagnostic_message: str,
) -> tuple[dict[str, Any], Path]:
    """写出首个 host 资格化失败, 不产生表面可审查候选."""

    report["decision"] = "fail"
    report["failure_reasons"] = [reason]
    report["diagnostic_message"] = diagnostic_message
    report["supports_paper_claim"] = False
    _write_json(report_path, report)
    return report, report_path


def _read_qualification_tool_lock(path: Path) -> tuple[str, str]:
    """读取只允许固定 uv wheel 哈希的 host 工具锁."""

    try:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    except FileNotFoundError as exc:
        raise ValueError("依赖资格化工具锁不存在") from exc
    if len(lines) != 1:
        raise ValueError("依赖资格化工具锁必须只包含一个 uv wheel 规格")
    match = _QUALIFICATION_TOOL_LOCK_PATTERN.fullmatch(lines[0])
    if match is None or match.group("version") != UV_DISTRIBUTION_VERSION:
        raise ValueError("依赖资格化工具锁必须匹配固定 uv 版本和 SHA-256")
    return match.group("version"), match.group("digest")


def _execute_qualification_operation(
    operation: str,
    command: Sequence[str],
    *,
    repository_root: Path,
    environment_overrides: Mapping[str, str],
    command_runner: QualificationCommandRunner,
    report: dict[str, Any],
) -> dict[str, Any]:
    """执行并记录一条 host 或精确子解释器资格化命令."""

    argv = [str(token) for token in command]
    result = _normalize_child_result(
        command_runner(argv, repository_root, environment_overrides)
    )
    record = {
        "operation": operation,
        "argv": argv,
        "working_directory": str(repository_root),
        "environment_overrides": dict(sorted(environment_overrides.items())),
        **result,
    }
    report["command_results"].append(record)
    return record


def _remove_qualification_runtime_root(runtime_root: Path) -> None:
    """清理单次资格化临时环境, 防止复用不可审计的 host 工具状态."""

    resolved = runtime_root.resolve()
    if resolved == Path(resolved.anchor) or resolved.parent == resolved:
        raise ValueError("依赖锁资格化临时根目录无效")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _validate_written_review_bundle(
    profile: DependencyProfile,
    *,
    repository_root: Path,
    formal_execution_lock: Mapping[str, Any],
    drive_output_dir: str | Path | None,
) -> tuple[dict[str, Any], Path]:
    """重新读取 child manifest 与三个本地文件, 拒绝仅返回退出码0."""

    manifest_path = (
        _local_bundle_dir(repository_root, profile.profile_name)
        / BUNDLE_MANIFEST_FILE_NAME
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("资格化 child 未写出有效审查包 manifest") from exc
    if not isinstance(manifest, dict):
        raise ValueError("资格化 child 审查包 manifest 必须是对象")
    expected_drive_bundle_dir = _resolve_drive_bundle_dir(
        repository_root,
        drive_output_dir,
        profile.profile_name,
    )
    expected = {
        "manifest_schema": BUNDLE_MANIFEST_SCHEMA,
        "schema_version": BUNDLE_MANIFEST_SCHEMA_VERSION,
        "profile_id": profile.profile_name,
        "profile_digest": profile.profile_digest,
        "direct_requirements_digest": profile.direct_requirements_digest,
        "review_execution_mode": (
            "isolated_python"
            if profile.profile_name in ISOLATED_PYTHON_PROFILE_IDS
            else "orchestrator_interpreter"
        ),
        "formal_execution_lock": dict(formal_execution_lock),
        "formal_execution_commit": formal_execution_lock[
            "formal_execution_commit"
        ],
        "formal_execution_lock_digest": formal_execution_lock[
            "formal_execution_lock_digest"
        ],
        "local_bundle_dir": _relative_or_absolute(
            repository_root,
            manifest_path.parent,
        ),
        "drive_bundle_dir": (
            None
            if expected_drive_bundle_dir is None
            else _relative_or_absolute(
                repository_root,
                expected_drive_bundle_dir,
            )
        ),
        "decision": SUCCESS_DECISION,
        "failure_reasons": [],
        "supports_paper_claim": False,
        "drive_copy_performed": drive_output_dir is not None,
    }
    if any(
        manifest.get(field_name) != expected_value
        for field_name, expected_value in expected.items()
    ):
        raise ValueError("资格化 child 审查包 manifest 身份门禁未通过")
    files = manifest.get("files")
    expected_file_names = {
        candidate_materializer.CANDIDATE_LOCK_FILE_NAME,
        candidate_materializer.PIP_REPORT_FILE_NAME,
        candidate_materializer.PROVENANCE_FILE_NAME,
    }
    if not isinstance(files, list) or len(files) != len(expected_file_names):
        raise ValueError("资格化 child 审查包文件记录不完整")
    local_bundle_dir = manifest_path.parent.resolve()
    observed_file_names: set[str] = set()
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("资格化 child 审查包文件记录必须是对象")
        file_name = record.get("file_name")
        if not isinstance(file_name, str) or file_name not in expected_file_names:
            raise ValueError("资格化 child 审查包包含未登记文件")
        bundle_path = Path(str(record.get("bundle_path", "")))
        if not bundle_path.is_absolute():
            bundle_path = repository_root / bundle_path
        resolved_bundle_path = bundle_path.resolve()
        if (
            resolved_bundle_path.parent != local_bundle_dir
            or resolved_bundle_path.name != file_name
            or not resolved_bundle_path.is_file()
        ):
            raise ValueError("资格化 child 审查包文件路径无效")
        if (
            record.get("sha256") != _sha256(resolved_bundle_path)
            or record.get("size_bytes") != resolved_bundle_path.stat().st_size
        ):
            raise ValueError("资格化 child 审查包文件摘要或大小不一致")
        observed_file_names.add(file_name)
    if observed_file_names != expected_file_names:
        raise ValueError("资格化 child 审查包文件集合不一致")
    return manifest, manifest_path


def launch_dependency_lock_qualification(
    profile_id: str,
    *,
    repository_root: str | Path = ROOT,
    drive_output_dir: str | Path | None = None,
    qualification_runtime_root: str | Path = DEFAULT_QUALIFICATION_RUNTIME_ROOT,
    command_runner: QualificationCommandRunner = _run_qualification_command,
) -> tuple[dict[str, Any], Path]:
    """从任意 Linux host 创建精确 orchestrator 并启动唯一审查脚本.

    host 只安装受单 wheel SHA-256 约束的固定 ``uv`` 工具. 候选解析始终在
    registry 登记的 CPython 3.12.13 子解释器中运行; 五个科学 profile 仍由
    该子解释器内的正式 orchestrator 锁门禁创建各自隔离解释器.
    """

    root = Path(repository_root).resolve()
    registry_path = root / "configs/dependency_profile_registry.json"
    profile = get_dependency_profile(profile_id, registry_path)
    orchestrator = get_dependency_profile(
        WORKFLOW_ORCHESTRATOR_PROFILE_ID,
        registry_path,
    )
    runtime_root = Path(qualification_runtime_root).expanduser().resolve()
    tool_lock_path = (root / QUALIFICATION_TOOL_LOCK_RELATIVE_PATH).resolve()
    report, report_path = _qualification_report_skeleton(
        profile,
        repository_root=root,
        qualification_runtime_root=runtime_root,
        qualification_tool_lock_path=tool_lock_path,
    )

    try:
        formal_execution_lock = (
            repository_environment.require_published_formal_execution_lock(root)
        )
    except repository_environment.FormalExecutionLockError as exc:
        return _write_qualification_failure(
            report,
            report_path,
            "formal_execution_lock_unavailable",
            str(exc),
        )
    report["formal_execution_lock"] = formal_execution_lock

    host_identity = {
        "operating_system": platform.system().strip().lower(),
        "machine": _normalize_machine(platform.machine()),
    }
    if host_identity != {
        "operating_system": orchestrator.operating_system,
        "machine": orchestrator.machine,
    }:
        return _write_qualification_failure(
            report,
            report_path,
            "qualification_host_platform_mismatch",
            "资格化 host 必须匹配登记的 Linux x86_64 平台.",
        )
    try:
        _read_qualification_tool_lock(tool_lock_path)
    except ValueError as exc:
        return _write_qualification_failure(
            report,
            report_path,
            "qualification_tool_lock_invalid",
            str(exc),
        )
    report["qualification_tool_lock_digest"] = _sha256(tool_lock_path)

    try:
        _remove_qualification_runtime_root(runtime_root)
    except (OSError, ValueError) as exc:
        return _write_qualification_failure(
            report,
            report_path,
            "qualification_runtime_root_unavailable",
            str(exc),
        )

    bootstrap_environment = runtime_root / "uv_tool_environment"
    bootstrap_python = bootstrap_environment / "bin" / "python"
    uv_executable = bootstrap_environment / "bin" / "uv"
    managed_python_root = runtime_root / "managed_pythons"
    orchestrator_environment = runtime_root / WORKFLOW_ORCHESTRATOR_PROFILE_ID
    orchestrator_python = orchestrator_environment / "bin" / "python"
    uv_environment = {"UV_PYTHON_INSTALL_DIR": str(managed_python_root)}

    command_plan = (
        (
            "qualification_tool_environment_create",
            [
                sys.executable,
                "-m",
                "venv",
                "--clear",
                str(bootstrap_environment),
            ],
            {},
        ),
        (
            "qualification_tool_lock_install",
            [
                str(bootstrap_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--require-hashes",
                "--only-binary=:all:",
                "--no-deps",
                "-r",
                str(tool_lock_path),
            ],
            {},
        ),
        (
            "qualification_uv_version",
            [str(uv_executable), "--version"],
            {},
        ),
        (
            "qualification_python_install",
            [
                str(uv_executable),
                "python",
                "install",
                orchestrator.python_version,
                "--install-dir",
                str(managed_python_root),
            ],
            uv_environment,
        ),
        (
            "qualification_orchestrator_venv",
            [
                str(uv_executable),
                "venv",
                "--clear",
                "--python",
                orchestrator.python_version,
                "--managed-python",
                str(orchestrator_environment),
            ],
            uv_environment,
        ),
        (
            "qualification_orchestrator_ensurepip",
            [str(orchestrator_python), "-m", "ensurepip"],
            {},
        ),
        (
            "qualification_orchestrator_python_inspection",
            [
                str(orchestrator_python),
                "-c",
                "import platform; print(platform.python_version())",
            ],
            {},
        ),
    )
    for operation, command, environment_overrides in command_plan:
        try:
            record = _execute_qualification_operation(
                operation,
                command,
                repository_root=root,
                environment_overrides=environment_overrides,
                command_runner=command_runner,
                report=report,
            )
        except (OSError, TypeError, ValueError) as exc:
            return _write_qualification_failure(
                report,
                report_path,
                f"{operation}_launch_failed",
                str(exc),
            )
        if record["return_code"] != 0:
            return _write_qualification_failure(
                report,
                report_path,
                f"{operation}_failed",
                "资格化命令返回非零退出码.",
            )
        if operation == "qualification_tool_environment_create":
            if not bootstrap_python.is_file():
                return _write_qualification_failure(
                    report,
                    report_path,
                    "qualification_tool_python_missing",
                    "host 工具环境未生成 Python executable.",
                )
        elif operation == "qualification_tool_lock_install":
            if not uv_executable.is_file():
                return _write_qualification_failure(
                    report,
                    report_path,
                    "qualification_uv_executable_missing",
                    "固定哈希工具锁未生成 uv executable.",
                )
            report["uv_executable_path"] = str(uv_executable)
            report["uv_executable_sha256"] = _sha256(uv_executable)
        elif operation == "qualification_uv_version":
            if record["stdout"].strip() != f"uv {UV_DISTRIBUTION_VERSION}":
                return _write_qualification_failure(
                    report,
                    report_path,
                    "qualification_uv_version_mismatch",
                    "uv executable 报告版本与固定工具锁不一致.",
                )
        elif operation == "qualification_orchestrator_venv":
            if not orchestrator_python.is_file():
                return _write_qualification_failure(
                    report,
                    report_path,
                    "qualification_orchestrator_python_missing",
                    "精确 orchestrator Python executable 不存在.",
                )
        elif operation == "qualification_orchestrator_python_inspection":
            if record["stdout"].strip() != orchestrator.python_version:
                return _write_qualification_failure(
                    report,
                    report_path,
                    "qualification_orchestrator_python_version_mismatch",
                    "精确 orchestrator Python patch 核验失败.",
                )

    if not orchestrator_python.is_file():
        return _write_qualification_failure(
            report,
            report_path,
            "qualification_orchestrator_python_missing",
            "精确 orchestrator Python executable 不存在.",
        )
    orchestrator_python_digest = _sha256(orchestrator_python)
    report["python_executable_path"] = str(orchestrator_python)
    report["python_executable_sha256"] = orchestrator_python_digest
    child_environment = {
        QUALIFICATION_CHILD_ENVIRONMENT_KEY: "1",
        QUALIFICATION_PYTHON_ENVIRONMENT_KEY: str(orchestrator_python),
        QUALIFICATION_PYTHON_DIGEST_ENVIRONMENT_KEY: orchestrator_python_digest,
        "PATH": (
            str(orchestrator_environment / "bin")
            + os.pathsep
            + os.environ.get("PATH", "")
        ),
    }
    review_command = [
        str(orchestrator_python),
        str(root / "scripts/write_dependency_lock_review_bundle.py"),
        "--profile",
        profile.profile_name,
    ]
    if drive_output_dir is not None:
        review_command.extend(
            ["--drive-output-dir", str(Path(drive_output_dir).expanduser())]
        )
    try:
        review_record = _execute_qualification_operation(
            "dependency_lock_review_bundle",
            review_command,
            repository_root=root,
            environment_overrides=child_environment,
            command_runner=command_runner,
            report=report,
        )
    except (OSError, TypeError, ValueError) as exc:
        return _write_qualification_failure(
            report,
            report_path,
            "dependency_lock_review_bundle_launch_failed",
            str(exc),
        )
    if review_record["return_code"] != 0:
        return _write_qualification_failure(
            report,
            report_path,
            "dependency_lock_review_bundle_failed",
            "精确 orchestrator 子解释器未生成完整审查包.",
        )
    try:
        _, manifest_path = _validate_written_review_bundle(
            profile,
            repository_root=root,
            formal_execution_lock=formal_execution_lock,
            drive_output_dir=drive_output_dir,
        )
    except ValueError as exc:
        return _write_qualification_failure(
            report,
            report_path,
            "dependency_lock_review_bundle_validation_failed",
            str(exc),
        )
    report["manifest_path"] = _relative_or_absolute(root, manifest_path)

    report["decision"] = QUALIFICATION_SUCCESS_DECISION
    report["failure_reasons"] = []
    report["diagnostic_message"] = None
    report["supports_paper_claim"] = False
    _write_json(report_path, report)
    return report, report_path


def _require_qualification_child_interpreter(repository_root: Path) -> None:
    """阻止伪造 child 标记绕过精确 orchestrator 解释器身份."""

    registry_path = repository_root / "configs/dependency_profile_registry.json"
    orchestrator = get_dependency_profile(
        WORKFLOW_ORCHESTRATOR_PROFILE_ID,
        registry_path,
    )
    expected_python_text = os.environ.get(
        QUALIFICATION_PYTHON_ENVIRONMENT_KEY,
        "",
    )
    expected_digest = os.environ.get(
        QUALIFICATION_PYTHON_DIGEST_ENVIRONMENT_KEY,
        "",
    )
    current_python = Path(sys.executable).resolve()
    try:
        expected_python = Path(expected_python_text).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise RuntimeError("资格化 child Python 路径无效") from exc
    identity_ready = all(
        (
            expected_python == current_python,
            current_python.is_file(),
            re.fullmatch(r"[0-9a-f]{64}", expected_digest) is not None,
            _sha256(current_python) == expected_digest
            if current_python.is_file()
            else False,
            platform.python_implementation() == orchestrator.python_implementation,
            platform.python_version() == orchestrator.python_version,
            platform.system().strip().lower() == orchestrator.operating_system,
            _normalize_machine(platform.machine()) == orchestrator.machine,
        )
    )
    if not identity_ready:
        raise RuntimeError("资格化 child 未通过精确 orchestrator 解释器门禁")


def _validate_candidate_provenance(
    provenance: Any,
    *,
    profile: DependencyProfile,
    formal_execution_lock: dict[str, Any],
) -> list[str]:
    """校验候选产物身份、代码锁和非论文证据边界."""

    if not isinstance(provenance, dict):
        return ["candidate_provenance_not_object"]
    expected = {
        "report_schema": candidate_materializer.PROVENANCE_SCHEMA,
        "schema_version": candidate_materializer.PROVENANCE_SCHEMA_VERSION,
        "profile_id": profile.profile_name,
        "profile_digest": profile.profile_digest,
        "cuda_version": profile.cuda_version,
        "pytorch_index_url": profile.pytorch_index_url,
        "torch_version": profile.torch_version,
        "torchvision_version": profile.torchvision_version,
        "direct_requirements_path": profile.direct_requirements_path,
        "direct_requirements_digest": profile.direct_requirements_digest,
        "formal_execution_lock": formal_execution_lock,
        "formal_execution_commit": formal_execution_lock["formal_execution_commit"],
        "formal_execution_lock_digest": formal_execution_lock[
            "formal_execution_lock_digest"
        ],
        "decision": "candidate_ready_for_review",
        "failure_reasons": [],
        "supports_paper_claim": False,
        "resolver_return_code": 0,
        "candidate_hash_source": (
            "pip_install_report.download_info.archive_info.hashes.sha256"
        ),
    }
    errors = [
        f"candidate_provenance_{field_name}_mismatch"
        for field_name, expected_value in expected.items()
        if provenance.get(field_name) != expected_value
    ]
    if not isinstance(provenance.get("candidate_lock_dependency_count"), int) or int(
        provenance.get("candidate_lock_dependency_count", 0)
    ) <= 0:
        errors.append("candidate_dependency_count_invalid")
    digest = provenance.get("candidate_lock_logical_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        errors.append("candidate_logical_digest_invalid")
    pip_version = provenance.get("pip_version")
    if not isinstance(pip_version, str) or not pip_version:
        errors.append("candidate_pip_version_invalid")
    return errors


def _validate_candidate_artifact_closure(
    source_paths: dict[str, Path],
    *,
    profile: DependencyProfile,
    provenance: dict[str, Any],
    repository_root: Path,
) -> list[str]:
    """从实际 pip 报告重建候选锁, 拒绝生成后的任一语义篡改."""

    expected_relative_paths = {
        "pip_resolver_report_path": _relative_or_absolute(
            repository_root,
            source_paths["pip_resolver_report"],
        ),
        "candidate_lock_path": _relative_or_absolute(
            repository_root,
            source_paths["candidate_lock"],
        ),
    }
    errors = [
        f"candidate_provenance_{field_name}_mismatch"
        for field_name, expected_value in expected_relative_paths.items()
        if provenance.get(field_name) != expected_value
    ]
    pip_version = provenance.get("pip_version")
    if not isinstance(pip_version, str) or not pip_version:
        return [*errors, "candidate_pip_version_invalid"]
    try:
        wheels, report_pip_version = candidate_materializer.load_resolved_wheels(
            source_paths["pip_resolver_report"],
            profile,
            expected_pip_version=pip_version,
        )
    except ValueError:
        return [*errors, "candidate_pip_report_revalidation_failed"]

    canonical_text = candidate_materializer.candidate_lock_text(wheels)
    try:
        actual_candidate_text = source_paths["candidate_lock"].read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeDecodeError):
        return [*errors, "candidate_lock_not_canonical_utf8"]
    if actual_candidate_text != canonical_text:
        errors.append("candidate_lock_text_mismatch")
    if report_pip_version != pip_version:
        errors.append("candidate_pip_version_mismatch")
    if provenance.get("candidate_lock_dependency_count") != len(wheels):
        errors.append("candidate_lock_dependency_count_mismatch")
    logical_digest = candidate_materializer.candidate_lock_logical_digest(wheels)
    if provenance.get("candidate_lock_logical_digest") != logical_digest:
        errors.append("candidate_lock_logical_digest_mismatch")
    return errors


def _run_orchestrator_interpreter_materializer(
    profile_id: str,
    repository_root: Path,
) -> tuple[dict[str, Any], Path]:
    """仅在当前父编排解释器内直接调用 orchestrator 候选物化器."""

    return candidate_materializer.materialize_dependency_lock_candidate(
        profile_id,
        repository_root=repository_root,
    )


def _run_isolated_python_materializer(
    profile: DependencyProfile,
    *,
    repository_root: Path,
    child_command_runner: ChildCommandRunner,
    manifest: dict[str, Any],
) -> tuple[bool, str]:
    """准备 orchestrator 后创建 isolated Python 并执行候选物化器."""

    orchestrator_report, orchestrator_report_path = prepare_dependency_profile(
        "workflow_orchestrator",
        repository_root=repository_root,
    )
    manifest["orchestrator_preparation"] = {
        "report_path": _relative_or_absolute(repository_root, orchestrator_report_path),
        "report_sha256": (
            _sha256(orchestrator_report_path)
            if orchestrator_report_path.is_file()
            else None
        ),
        "decision": orchestrator_report.get("decision"),
        "failure_reasons": list(orchestrator_report.get("failure_reasons", [])),
    }
    if orchestrator_report.get("decision") != "pass":
        return False, "请先生成、审查并提交 workflow_orchestrator 完整锁."

    provision_report, provision_report_path = provision_isolated_dependency_python(
        profile.profile_name,
        repository_root=repository_root,
    )
    manifest["isolated_python_provision"] = {
        "report_path": _relative_or_absolute(repository_root, provision_report_path),
        "report_sha256": (
            _sha256(provision_report_path) if provision_report_path.is_file() else None
        ),
        "decision": provision_report.get("decision"),
        "failure_reasons": list(provision_report.get("failure_reasons", [])),
        "python_executable_path": provision_report.get("python_executable_path"),
        "python_executable_sha256": provision_report.get("python_executable_sha256"),
    }
    if (
        provision_report.get("decision") != "provisioned"
        or provision_report.get("provisioned") is not True
    ):
        return False, "isolated Python 解释器创建未通过."
    python_executable = Path(str(provision_report.get("python_executable_path", "")))
    if not python_executable.is_file():
        return False, "isolated Python executable 不存在."

    command = [
        str(python_executable),
        str(repository_root / "scripts/materialize_dependency_lock_candidate.py"),
        "--profile",
        profile.profile_name,
    ]
    command_result = _normalize_child_result(
        child_command_runner(command, repository_root)
    )
    manifest["candidate_materialization"] = {
        "python_executable": str(python_executable),
        "command": command,
        **command_result,
    }
    if command_result["return_code"] != 0:
        return False, "isolated Python 候选物化器执行失败."
    return True, ""


def write_dependency_lock_review_bundle(
    profile_id: str,
    *,
    repository_root: str | Path = ROOT,
    drive_output_dir: str | Path | None = None,
    child_command_runner: ChildCommandRunner = _run_child_command,
) -> tuple[dict[str, Any], Path]:
    """为一个 profile 生成本地审查包并按需复制到显式 Drive 目录."""

    root = Path(repository_root).resolve()
    registry_path = root / "configs/dependency_profile_registry.json"
    profile = get_dependency_profile(profile_id, registry_path)
    local_bundle_dir = _local_bundle_dir(root, profile.profile_name)
    drive_bundle_dir = _resolve_drive_bundle_dir(
        root,
        drive_output_dir,
        profile.profile_name,
    )
    local_bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = local_bundle_dir / BUNDLE_MANIFEST_FILE_NAME
    for file_name in (
        candidate_materializer.CANDIDATE_LOCK_FILE_NAME,
        candidate_materializer.PIP_REPORT_FILE_NAME,
        candidate_materializer.PROVENANCE_FILE_NAME,
        BUNDLE_MANIFEST_FILE_NAME,
    ):
        (local_bundle_dir / file_name).unlink(missing_ok=True)

    manifest = _build_manifest(
        profile,
        repository_root=root,
        local_bundle_dir=local_bundle_dir,
        drive_bundle_dir=drive_bundle_dir,
        formal_execution_lock=None,
    )
    try:
        formal_execution_lock = (
            repository_environment.require_published_formal_execution_lock(root)
        )
    except repository_environment.FormalExecutionLockError as exc:
        return _write_failure_manifest(
            manifest,
            manifest_path,
            "formal_execution_lock_unavailable",
            str(exc),
        )
    manifest["formal_execution_lock"] = formal_execution_lock
    manifest["formal_execution_commit"] = formal_execution_lock[
        "formal_execution_commit"
    ]
    manifest["formal_execution_lock_digest"] = formal_execution_lock[
        "formal_execution_lock_digest"
    ]

    if profile.profile_name in ISOLATED_PYTHON_PROFILE_IDS:
        materialized, diagnostic = _run_isolated_python_materializer(
            profile,
            repository_root=root,
            child_command_runner=child_command_runner,
            manifest=manifest,
        )
        if not materialized:
            return _write_failure_manifest(
                manifest,
                manifest_path,
                "isolated_python_candidate_materialization_failed",
                diagnostic,
            )
    elif profile.profile_name == CURRENT_INTERPRETER_PROFILE_ID:
        candidate_report, candidate_report_path = _run_orchestrator_interpreter_materializer(
            profile.profile_name,
            root,
        )
        manifest["candidate_materialization"] = {
            "provenance_path": _relative_or_absolute(root, candidate_report_path),
            "decision": candidate_report.get("decision"),
            "failure_reasons": list(candidate_report.get("failure_reasons", [])),
        }
        if candidate_report.get("decision") != "candidate_ready_for_review":
            return _write_failure_manifest(
                manifest,
                manifest_path,
                "candidate_materialization_failed",
                "候选物化器未生成可供审查的候选.",
            )
    else:
        return _write_failure_manifest(
            manifest,
            manifest_path,
            "unsupported_dependency_profile",
            "依赖 profile 未登记资格化路径.",
        )

    source_paths = _candidate_source_paths(root, profile.profile_name)
    missing_paths = [path for path in source_paths.values() if not path.is_file()]
    if missing_paths:
        return _write_failure_manifest(
            manifest,
            manifest_path,
            "candidate_artifact_missing",
            "候选物化输出不完整.",
        )
    try:
        provenance = json.loads(
            source_paths["candidate_provenance"].read_text(encoding="utf-8-sig")
        )
    except (OSError, json.JSONDecodeError) as exc:
        return _write_failure_manifest(
            manifest,
            manifest_path,
            "candidate_provenance_invalid",
            f"候选 provenance 无法读取: {exc}",
        )
    provenance_errors = _validate_candidate_provenance(
        provenance,
        profile=profile,
        formal_execution_lock=formal_execution_lock,
    )
    if provenance_errors:
        manifest["failure_reasons"] = provenance_errors
        manifest["diagnostic_message"] = "候选 provenance 身份门禁未通过."
        _write_json(manifest_path, manifest)
        return manifest, manifest_path
    artifact_closure_errors = _validate_candidate_artifact_closure(
        source_paths,
        profile=profile,
        provenance=provenance,
        repository_root=root,
    )
    if artifact_closure_errors:
        manifest["failure_reasons"] = artifact_closure_errors
        manifest["diagnostic_message"] = "候选锁与 pip resolver 报告闭包不一致."
        _write_json(manifest_path, manifest)
        return manifest, manifest_path

    file_records: list[dict[str, Any]] = []
    for artifact_role, source_path in source_paths.items():
        destination = local_bundle_dir / source_path.name
        shutil.copy2(source_path, destination)
        source_digest = _sha256(source_path)
        destination_digest = _sha256(destination)
        if destination_digest != source_digest:
            return _write_failure_manifest(
                manifest,
                manifest_path,
                "local_bundle_copy_digest_mismatch",
                f"本地审查包复制摘要不一致: {artifact_role}",
            )
        file_records.append(
            {
                "artifact_role": artifact_role,
                "file_name": destination.name,
                "source_path": _relative_or_absolute(root, source_path),
                "bundle_path": _relative_or_absolute(root, destination),
                "sha256": destination_digest,
                "size_bytes": destination.stat().st_size,
            }
        )
    manifest["files"] = file_records

    if drive_bundle_dir is not None:
        try:
            drive_bundle_dir.mkdir(parents=True, exist_ok=True)
            for record in file_records:
                local_path = local_bundle_dir / record["file_name"]
                drive_path = drive_bundle_dir / record["file_name"]
                shutil.copy2(local_path, drive_path)
                if _sha256(drive_path) != record["sha256"]:
                    raise RuntimeError(f"Drive 文件摘要不一致: {record['file_name']}")
                record["drive_path"] = str(drive_path.resolve())
        except (OSError, RuntimeError) as exc:
            return _write_failure_manifest(
                manifest,
                manifest_path,
                "drive_bundle_copy_failed",
                str(exc),
            )
        manifest["drive_copy_performed"] = True

    manifest["decision"] = SUCCESS_DECISION
    manifest["failure_reasons"] = []
    manifest["diagnostic_message"] = None
    _write_json(manifest_path, manifest)
    if drive_bundle_dir is not None:
        drive_manifest_path = drive_bundle_dir / BUNDLE_MANIFEST_FILE_NAME
        try:
            shutil.copy2(manifest_path, drive_manifest_path)
            manifest_digest_matches = _sha256(drive_manifest_path) == _sha256(
                manifest_path
            )
        except OSError as exc:
            return _write_failure_manifest(
                manifest,
                manifest_path,
                "drive_manifest_copy_failed",
                str(exc),
            )
        if not manifest_digest_matches:
            return _write_failure_manifest(
                manifest,
                manifest_path,
                "drive_manifest_copy_digest_mismatch",
                "Drive manifest 复制摘要不一致.",
            )
    return manifest, manifest_path


def build_parser() -> argparse.ArgumentParser:
    """构造依赖锁审查包 CLI 参数解析器."""

    parser = argparse.ArgumentParser(description="生成单个依赖 profile 的候选锁审查包.")
    parser.add_argument(
        "--profile",
        required=True,
        choices=REQUIRED_DEPENDENCY_PROFILE_NAMES,
        help="需要资格化的依赖 profile id.",
    )
    parser.add_argument(
        "--drive-output-dir",
        default=None,
        help="可选 Drive 根目录; 未提供时只生成本地 outputs 审查包.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """从普通 host 启动精确环境, child 中仅写完整审查包."""

    arguments = build_parser().parse_args(argv)
    is_qualification_child = (
        os.environ.get(QUALIFICATION_CHILD_ENVIRONMENT_KEY) == "1"
    )
    try:
        if is_qualification_child:
            _require_qualification_child_interpreter(ROOT)
            manifest, manifest_path = write_dependency_lock_review_bundle(
                arguments.profile,
                drive_output_dir=arguments.drive_output_dir,
            )
            result = {
                "profile_id": manifest["profile_id"],
                "manifest_path": str(manifest_path),
                "drive_bundle_dir": manifest["drive_bundle_dir"],
                "decision": manifest["decision"],
                "failure_reasons": manifest["failure_reasons"],
            }
            success = manifest["decision"] == SUCCESS_DECISION
        else:
            qualification, qualification_path = (
                launch_dependency_lock_qualification(
                    arguments.profile,
                    drive_output_dir=arguments.drive_output_dir,
                )
            )
            result = {
                "profile_id": qualification["profile_id"],
                "qualification_report_path": str(qualification_path),
                "decision": qualification["decision"],
                "failure_reasons": qualification["failure_reasons"],
            }
            success = (
                qualification["decision"]
                == QUALIFICATION_SUCCESS_DECISION
            )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "profile_id": arguments.profile,
                    "decision": "fail",
                    "failure_reasons": [f"review_bundle_error:{type(exc).__name__}"],
                    "diagnostic_message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
