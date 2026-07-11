"""使用受治理 uv 构建 隔离 Python profile 的确定性隔离 Python 环境.

本模块区分两个边界. ``provision_isolated_dependency_python`` 只创建精确
CPython 与 venv, 可供目标 profile 的完整锁物化流程使用;
``prepare_isolated_dependency_environment`` 进一步要求目标完整锁 ready, 并在
隔离解释器中调用内层 dependency preparation 形成正式环境报告.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

from experiments.runtime.dependency_preparation import (
    REPORT_SCHEMA as DEPENDENCY_PREPARATION_REPORT_SCHEMA,
    REPORT_SCHEMA_VERSION as DEPENDENCY_PREPARATION_REPORT_SCHEMA_VERSION,
)
from experiments.runtime.dependency_profiles import (
    DependencyProfile,
    WORKFLOW_ORCHESTRATOR_PROFILE_ID,
    get_dependency_profile,
    inspect_dependency_profile_environment,
    require_dependency_profile_ready,
)
from experiments.runtime.repository_environment import (
    require_published_formal_execution_lock,
)


ROOT = Path(__file__).resolve().parents[2]
ISOLATED_DEPENDENCY_PROFILE_IDS = (
    "sd35_method_runtime_gpu",
    "t2smark_sd35_gpu",
    "tree_ring_official_py39_cu117",
    "gaussian_shading_official_py38_cu117",
    "shallow_diffuse_official_py39_cu117",
)
UV_DISTRIBUTION_NAME = "uv"
UV_DISTRIBUTION_VERSION = "0.11.28"
UV_PYTHON_INSTALL_DIR_ENV_NAME = "UV_PYTHON_INSTALL_DIR"
DEFAULT_ENVIRONMENT_ROOT = Path(tempfile.gettempdir()) / "slm_wm_dependency_envs"
DEFAULT_MANAGED_PYTHON_ROOT = Path(tempfile.gettempdir()) / "slm_wm_dependency_pythons"
PROVISION_REPORT_SCHEMA = "isolated_dependency_python_provision_report"
FORMAL_REPORT_SCHEMA = "isolated_dependency_environment_preparation_report"
REPORT_SCHEMA_VERSION = 1
PROVISION_REPORT_FILE_NAME = "isolated_python_provision_report.json"
FORMAL_REPORT_FILE_NAME = "isolated_dependency_environment_report.json"
DEPENDENCY_PREPARATION_REPORT_FILE_NAME = "dependency_profile_report.json"

CommandRunner = Callable[[Sequence[str], Path, Mapping[str, str]], Any]


def _run_command(
    command: Sequence[str],
    working_directory: Path,
    environment_overrides: Mapping[str, str],
) -> dict[str, Any]:
    """以 argv 方式执行命令并捕获可审计输出, 不经过 shell."""

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
    return {
        "return_code": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _normalize_command_result(result: Any) -> dict[str, Any]:
    """把测试 runner 与 subprocess 结果收敛为同一结构."""

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
            raise ValueError("command runner 结果必须包含整数 return_code")
        return {
            "return_code": return_code,
            "stdout": str(result.get("stdout") or ""),
            "stderr": str(result.get("stderr") or ""),
        }
    raise TypeError("command runner 必须返回 int、CompletedProcess 或 mapping")


def _execute_command(
    operation: str,
    argv: Sequence[str],
    *,
    repository_root: Path,
    environment_overrides: Mapping[str, str],
    command_runner: CommandRunner,
) -> dict[str, Any]:
    """执行单条 argv 命令并把完整调用身份写入结果."""

    normalized_argv = [str(token) for token in argv]
    result = _normalize_command_result(
        command_runner(normalized_argv, repository_root, environment_overrides)
    )
    return {
        "operation": operation,
        "argv": normalized_argv,
        "environment_overrides": dict(sorted(environment_overrides.items())),
        **result,
    }


def _file_sha256(path: Path) -> str:
    """计算实际可执行文件或报告文件的 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_text(payload: Any) -> str:
    """生成稳定且便于审计的 JSON 文本."""

    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write_report(path: Path, report: dict[str, Any]) -> None:
    """把报告持久化到受治理 outputs 路径."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_stable_json_text(report), encoding="utf-8")


def _profile_output_dir(repository_root: Path, profile_id: str) -> Path:
    """解析单个 profile 的 outputs 目录并阻止路径越界."""

    outputs_root = (repository_root / "outputs").resolve()
    output_dir = (outputs_root / "dependency_profiles" / profile_id).resolve()
    try:
        output_dir.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("隔离依赖环境报告必须位于 outputs/ 下") from exc
    return output_dir


def isolated_environment_path(
    profile_id: str,
    *,
    environment_root: str | Path = DEFAULT_ENVIRONMENT_ROOT,
) -> Path:
    """返回可由服务器或 CLI root 覆盖的固定 profile venv 路径."""

    if profile_id not in ISOLATED_DEPENDENCY_PROFILE_IDS:
        raise ValueError(f"隔离 Python 只支持五个科学执行 profile: {profile_id}")
    source_text = str(environment_root).replace("\\", "/")
    root_path = Path(environment_root).expanduser()
    preserve_posix_root = source_text.startswith("/")
    root = root_path if preserve_posix_root else root_path.resolve()
    environment_path = root / profile_id
    if not preserve_posix_root:
        environment_path = environment_path.resolve()
    try:
        environment_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("隔离环境路径不得越过 environment_root") from exc
    return environment_path


def isolated_python_executable_path(
    profile_id: str,
    *,
    environment_root: str | Path = DEFAULT_ENVIRONMENT_ROOT,
) -> Path:
    """返回固定 venv 中的 Linux Python executable 路径."""

    return isolated_environment_path(profile_id, environment_root=environment_root) / "bin" / "python"


def _resolve_uv_executable(explicit_path: str | Path | None) -> Path:
    """解析实际 uv executable 并要求其为可摘要文件."""

    candidate = str(explicit_path) if explicit_path is not None else shutil.which("uv")
    if not candidate:
        raise FileNotFoundError("uv executable 不存在")
    path = Path(candidate).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"uv executable 不是文件: {path}")
    return path


def _read_uv_distribution_version() -> str | None:
    """读取当前解释器安装的 uv distribution 精确版本."""

    try:
        return importlib_metadata.version(UV_DISTRIBUTION_NAME)
    except importlib_metadata.PackageNotFoundError:
        return None


def _inspect_uv_executable_distribution_source(uv_executable: Path) -> dict[str, str]:
    """确认 uv executable 由当前解释器安装的固定 distribution RECORD 管理."""

    try:
        distribution = importlib_metadata.distribution(UV_DISTRIBUTION_NAME)
    except importlib_metadata.PackageNotFoundError as exc:
        raise ValueError("uv distribution 不存在") from exc
    if distribution.version != UV_DISTRIBUTION_VERSION:
        raise ValueError("uv distribution 版本不匹配")
    distribution_files = distribution.files
    if not distribution_files:
        raise ValueError("uv distribution 缺少 RECORD 文件清单")

    resolved_executable = uv_executable.resolve()
    executable_record = None
    record_path = None
    for distribution_path in distribution_files:
        located_path = Path(distribution.locate_file(distribution_path)).resolve()
        if located_path == resolved_executable:
            executable_record = distribution_path
        if (
            distribution_path.name == "RECORD"
            and distribution_path.parent.name.endswith(".dist-info")
        ):
            record_path = located_path
    if (
        executable_record is None
        or resolved_executable.name.lower() not in {"uv", "uv.exe"}
    ):
        raise ValueError("uv executable 不属于当前解释器的 uv distribution")
    record_hash = executable_record.hash
    if record_hash is None or record_hash.mode != "sha256":
        raise ValueError("uv executable 缺少 distribution RECORD SHA-256")
    executable_digest_hex = _file_sha256(resolved_executable)
    executable_digest_bytes = bytes.fromhex(executable_digest_hex)
    executable_record_digest = base64.urlsafe_b64encode(
        executable_digest_bytes
    ).decode("ascii").rstrip("=")
    if executable_record_digest != record_hash.value:
        raise ValueError("uv executable 与 distribution RECORD SHA-256 不一致")
    if record_path is None or not record_path.is_file():
        raise ValueError("uv distribution RECORD 文件不存在")
    return {
        "uv_distribution_record_path": str(record_path),
        "uv_distribution_record_sha256": _file_sha256(record_path),
        "uv_distribution_executable_record_path": executable_record.as_posix(),
        "uv_distribution_executable_record_sha256": executable_digest_hex,
    }


def _validate_isolated_profile(profile_id: str, registry_path: Path) -> DependencyProfile:
    """查询并限制五个项目登记的隔离 Python profile."""

    if profile_id not in ISOLATED_DEPENDENCY_PROFILE_IDS:
        raise ValueError(f"未登记的 隔离 Python profile: {profile_id}")
    return get_dependency_profile(profile_id, registry_path)


def _validate_orchestrator_environment(registry_path: Path) -> tuple[DependencyProfile, dict[str, Any]]:
    """要求父解释器的 orchestrator profile 已锁定且 inspection 通过."""

    orchestrator = require_dependency_profile_ready(
        WORKFLOW_ORCHESTRATOR_PROFILE_ID,
        registry_path,
    )
    if f"uv=={UV_DISTRIBUTION_VERSION}" not in orchestrator.direct_requirements:
        raise ValueError("workflow_orchestrator 未登记固定 uv distribution")
    inspection = inspect_dependency_profile_environment(
        WORKFLOW_ORCHESTRATOR_PROFILE_ID,
        path=registry_path,
    )
    if inspection.get("decision") != "pass" or inspection.get("environment_match") is not True:
        raise RuntimeError("workflow_orchestrator_inspection_failed")
    if inspection.get("profile_digest") != orchestrator.profile_digest:
        raise ValueError("workflow_orchestrator inspection profile_digest 不一致")
    if inspection.get("complete_hash_lock_digest") != orchestrator.complete_hash_lock_digest:
        raise ValueError("workflow_orchestrator inspection lock digest 不一致")
    return orchestrator, inspection


def _provision_report_skeleton(
    profile: DependencyProfile,
    *,
    repository_root: Path,
    environment_root: Path,
    managed_python_root: Path,
) -> tuple[dict[str, Any], Path]:
    """构造只证明 Python provision 的报告骨架."""

    report_path = _profile_output_dir(repository_root, profile.profile_name) / PROVISION_REPORT_FILE_NAME
    report = {
        "report_schema": PROVISION_REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "operation_kind": "isolated_python_provision",
        "profile_id": profile.profile_name,
        "profile_digest": profile.profile_digest,
        "python_version": profile.python_version,
        "target_complete_hash_lock_ready": profile.formal_ready,
        "environment_root": str(environment_root),
        "managed_python_root": str(managed_python_root),
        "isolated_environment_path": str(environment_root / profile.profile_name),
        "uv_distribution_version": None,
        "uv_distribution_record_path": None,
        "uv_distribution_record_sha256": None,
        "uv_distribution_executable_record_path": None,
        "uv_distribution_executable_record_sha256": None,
        "uv_executable_path": None,
        "uv_executable_sha256": None,
        "uv_reported_version": None,
        "python_executable_path": None,
        "python_executable_sha256": None,
        "command_results": [],
        "uv_commands": [],
        "orchestrator_profile_digest": None,
        "orchestrator_complete_hash_lock_digest": None,
        "orchestrator_inspection": None,
        "formal_execution_lock": {},
        "formal_execution_commit": None,
        "formal_execution_lock_digest": None,
        "formal_execution_lock_ready": False,
        "provisioned": False,
        "formal_ready": False,
        "decision": "fail",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    return report, report_path


def _fail_report(
    report: dict[str, Any],
    report_path: Path,
    reason: str,
) -> tuple[dict[str, Any], Path]:
    """在首个不可恢复边界写出 fail-closed 报告."""

    report["failure_reasons"] = [reason]
    report["decision"] = "fail"
    report["formal_ready"] = False
    report["supports_paper_claim"] = False
    _write_report(report_path, report)
    return report, report_path


def provision_isolated_dependency_python(
    profile_id: str,
    *,
    repository_root: str | Path = ROOT,
    environment_root: str | Path = DEFAULT_ENVIRONMENT_ROOT,
    managed_python_root: str | Path = DEFAULT_MANAGED_PYTHON_ROOT,
    uv_executable_path: str | Path | None = None,
    command_runner: CommandRunner = _run_command,
) -> tuple[dict[str, Any], Path]:
    """创建精确 隔离 Python venv, 不要求目标 profile 的完整锁.

    此 API 只形成 lock materializer 可使用的解释器, 不执行目标依赖安装, 也不
    产生 formal-ready 结论.
    """

    root = Path(repository_root).resolve()
    registry_path = root / "configs/dependency_profile_registry.json"
    profile = _validate_isolated_profile(profile_id, registry_path)
    resolved_environment_root = Path(environment_root).expanduser().resolve()
    resolved_managed_python_root = Path(managed_python_root).expanduser().resolve()
    environment_path = isolated_environment_path(
        profile_id,
        environment_root=resolved_environment_root,
    )
    python_executable = isolated_python_executable_path(
        profile_id,
        environment_root=resolved_environment_root,
    )
    report, report_path = _provision_report_skeleton(
        profile,
        repository_root=root,
        environment_root=resolved_environment_root,
        managed_python_root=resolved_managed_python_root,
    )

    try:
        formal_execution_lock = require_published_formal_execution_lock(root)
    except ValueError as exc:
        return _fail_report(
            report,
            report_path,
            f"formal_execution_lock_not_ready:{type(exc).__name__}",
        )
    report["formal_execution_lock"] = formal_execution_lock
    report["formal_execution_commit"] = formal_execution_lock[
        "formal_execution_commit"
    ]
    report["formal_execution_lock_digest"] = formal_execution_lock[
        "formal_execution_lock_digest"
    ]
    report["formal_execution_lock_ready"] = True

    try:
        orchestrator, orchestrator_inspection = _validate_orchestrator_environment(registry_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return _fail_report(report, report_path, f"orchestrator_not_ready:{type(exc).__name__}")
    report["orchestrator_profile_digest"] = orchestrator.profile_digest
    report["orchestrator_complete_hash_lock_digest"] = orchestrator.complete_hash_lock_digest
    report["orchestrator_inspection"] = orchestrator_inspection

    uv_distribution_version = _read_uv_distribution_version()
    report["uv_distribution_version"] = uv_distribution_version
    if uv_distribution_version != UV_DISTRIBUTION_VERSION:
        return _fail_report(report, report_path, "uv_distribution_version_mismatch")
    try:
        uv_executable = _resolve_uv_executable(uv_executable_path)
    except FileNotFoundError:
        return _fail_report(report, report_path, "uv_executable_missing")
    report["uv_executable_path"] = str(uv_executable)
    report["uv_executable_sha256"] = _file_sha256(uv_executable)
    try:
        distribution_source = _inspect_uv_executable_distribution_source(
            uv_executable
        )
    except ValueError:
        return _fail_report(
            report,
            report_path,
            "uv_executable_distribution_source_mismatch",
        )
    report.update(distribution_source)

    resolved_environment_root.mkdir(parents=True, exist_ok=True)
    resolved_managed_python_root.mkdir(parents=True, exist_ok=True)
    command_environment = {
        UV_PYTHON_INSTALL_DIR_ENV_NAME: str(resolved_managed_python_root),
    }
    command_plan = (
        (
            "uv_version",
            [str(uv_executable), "--version"],
            True,
        ),
        (
            "uv_python_install",
            [
                str(uv_executable),
                "python",
                "install",
                profile.python_version,
                "--install-dir",
                str(resolved_managed_python_root),
            ],
            True,
        ),
        (
            "uv_venv",
            [
                str(uv_executable),
                "venv",
                "--clear",
                "--python",
                profile.python_version,
                "--managed-python",
                str(environment_path),
            ],
            True,
        ),
        (
            "python_ensurepip",
            [str(python_executable), "-m", "ensurepip"],
            False,
        ),
        (
            "python_patch_inspection",
            [
                str(python_executable),
                "-c",
                "import platform; print(platform.python_version())",
            ],
            False,
        ),
    )
    for operation, argv, is_uv_command in command_plan:
        result = _execute_command(
            operation,
            argv,
            repository_root=root,
            environment_overrides=command_environment,
            command_runner=command_runner,
        )
        report["command_results"].append(result)
        if is_uv_command:
            report["uv_commands"].append(result)
        if result["return_code"] != 0:
            return _fail_report(report, report_path, f"{operation}_failed")
        if operation == "uv_version":
            version_tokens = result["stdout"].strip().split()
            reported_version = version_tokens[1] if len(version_tokens) >= 2 and version_tokens[0] == "uv" else ""
            report["uv_reported_version"] = reported_version
            if reported_version != UV_DISTRIBUTION_VERSION:
                return _fail_report(report, report_path, "uv_executable_version_mismatch")
        if operation == "uv_venv" and not python_executable.is_file():
            return _fail_report(report, report_path, "isolated_python_executable_missing")
        if operation == "python_patch_inspection" and result["stdout"].strip() != profile.python_version:
            return _fail_report(report, report_path, "isolated_python_version_mismatch")

    if not python_executable.is_file():
        return _fail_report(report, report_path, "isolated_python_executable_missing")
    report["python_executable_path"] = str(python_executable)
    report["python_executable_sha256"] = _file_sha256(python_executable)
    report["provisioned"] = True
    report["formal_ready"] = False
    report["decision"] = "provisioned"
    report["failure_reasons"] = []
    _write_report(report_path, report)
    return report, report_path


def _formal_report_skeleton(
    profile: DependencyProfile,
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], Path]:
    """构造要求目标完整锁的正式隔离环境报告骨架."""

    report_path = _profile_output_dir(repository_root, profile.profile_name) / FORMAL_REPORT_FILE_NAME
    report = {
        "report_schema": FORMAL_REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "operation_kind": "formal_dependency_environment_preparation",
        "profile_id": profile.profile_name,
        "profile_digest": profile.profile_digest,
        "direct_requirements_digest": profile.direct_requirements_digest,
        "complete_hash_lock_digest": profile.complete_hash_lock_digest,
        "complete_hash_lock_dependency_count": profile.complete_hash_lock_dependency_count,
        "provision_report_path": None,
        "provision_report_digest": None,
        "provision_report": None,
        "uv_distribution_version": None,
        "uv_distribution_record_path": None,
        "uv_distribution_record_sha256": None,
        "uv_distribution_executable_record_path": None,
        "uv_distribution_executable_record_sha256": None,
        "uv_executable_path": None,
        "uv_executable_sha256": None,
        "uv_commands": [],
        "python_executable_path": None,
        "python_executable_sha256": None,
        "python_executable_sha256_after_preparation": None,
        "dependency_preparation_command": None,
        "dependency_preparation_report_path": None,
        "dependency_preparation_report_digest": None,
        "dependency_preparation_report": None,
        "command_results": [],
        "formal_execution_lock": {},
        "formal_execution_commit": None,
        "formal_execution_lock_digest": None,
        "formal_execution_lock_ready": False,
        "provisioned": False,
        "formal_preparation_completed": False,
        "formal_ready": False,
        "decision": "fail",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    return report, report_path


def _validate_dependency_preparation_report(
    report: Any,
    *,
    profile: DependencyProfile,
    python_executable: Path,
    formal_execution_lock: dict[str, Any],
) -> list[str]:
    """严格验证子解释器 dependency preparation 的身份与通过结论."""

    if not isinstance(report, dict):
        return ["dependency_preparation_report_not_object"]
    expected_values = {
        "report_schema": DEPENDENCY_PREPARATION_REPORT_SCHEMA,
        "schema_version": DEPENDENCY_PREPARATION_REPORT_SCHEMA_VERSION,
        "profile_id": profile.profile_name,
        "python_executable": str(python_executable),
        "profile_digest": profile.profile_digest,
        "direct_requirements_digest": profile.direct_requirements_digest,
        "complete_hash_lock_digest": profile.complete_hash_lock_digest,
        "complete_hash_lock_dependency_count": profile.complete_hash_lock_dependency_count,
        "formal_ready": True,
        "decision": "pass",
        "failure_reasons": [],
        "supports_paper_claim": False,
        "formal_execution_lock": formal_execution_lock,
        "formal_execution_commit": formal_execution_lock["formal_execution_commit"],
        "formal_execution_lock_digest": formal_execution_lock[
            "formal_execution_lock_digest"
        ],
        "formal_execution_lock_ready": True,
    }
    errors = [
        f"dependency_preparation_{field_name}_mismatch"
        for field_name, expected_value in expected_values.items()
        if report.get(field_name) != expected_value
    ]
    repository_commit_state = report.get("repository_commit_state")
    if not isinstance(repository_commit_state, dict) or repository_commit_state.get("all_committed") is not True:
        errors.append("dependency_preparation_inputs_not_committed")
    installation = report.get("installation")
    if (
        not isinstance(installation, dict)
        or installation.get("attempted") is not True
        or installation.get("return_code") != 0
    ):
        errors.append("dependency_preparation_installation_invalid")
    pip_check = report.get("pip_check")
    if (
        not isinstance(pip_check, dict)
        or pip_check.get("compatibility_check_required") is not True
        or pip_check.get("attempted") is not True
        or pip_check.get("return_code") != 0
        or pip_check.get("decision") != "pass"
    ):
        errors.append("dependency_preparation_pip_check_invalid")
    runtime_comparison = report.get("runtime_comparison")
    if (
        not isinstance(runtime_comparison, dict)
        or runtime_comparison.get("decision") != "pass"
        or runtime_comparison.get("profile_digest") != profile.profile_digest
        or runtime_comparison.get("complete_hash_lock_digest") != profile.complete_hash_lock_digest
    ):
        errors.append("dependency_preparation_runtime_comparison_invalid")
    return errors


def prepare_isolated_dependency_environment(
    profile_id: str,
    *,
    repository_root: str | Path = ROOT,
    environment_root: str | Path = DEFAULT_ENVIRONMENT_ROOT,
    managed_python_root: str | Path = DEFAULT_MANAGED_PYTHON_ROOT,
    uv_executable_path: str | Path | None = None,
    command_runner: CommandRunner = _run_command,
) -> tuple[dict[str, Any], Path]:
    """为目标 隔离 Python profile 创建并正式准备完整隔离环境."""

    root = Path(repository_root).resolve()
    registry_path = root / "configs/dependency_profile_registry.json"
    profile = _validate_isolated_profile(profile_id, registry_path)
    report, report_path = _formal_report_skeleton(profile, repository_root=root)
    try:
        formal_execution_lock = require_published_formal_execution_lock(root)
    except ValueError as exc:
        return _fail_report(
            report,
            report_path,
            f"formal_execution_lock_not_ready:{type(exc).__name__}",
        )
    report["formal_execution_lock"] = formal_execution_lock
    report["formal_execution_commit"] = formal_execution_lock[
        "formal_execution_commit"
    ]
    report["formal_execution_lock_digest"] = formal_execution_lock[
        "formal_execution_lock_digest"
    ]
    report["formal_execution_lock_ready"] = True
    try:
        ready_profile = require_dependency_profile_ready(profile_id, registry_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return _fail_report(report, report_path, f"target_profile_not_ready:{type(exc).__name__}")

    provision_report, provision_report_path = provision_isolated_dependency_python(
        profile_id,
        repository_root=root,
        environment_root=environment_root,
        managed_python_root=managed_python_root,
        uv_executable_path=uv_executable_path,
        command_runner=command_runner,
    )
    report["provision_report_path"] = str(provision_report_path)
    report["provision_report_digest"] = _file_sha256(provision_report_path)
    report["provision_report"] = provision_report
    report["command_results"] = list(provision_report.get("command_results", []))
    report["uv_commands"] = list(provision_report.get("uv_commands", []))
    report["uv_distribution_version"] = provision_report.get("uv_distribution_version")
    report["uv_distribution_record_path"] = provision_report.get(
        "uv_distribution_record_path"
    )
    report["uv_distribution_record_sha256"] = provision_report.get(
        "uv_distribution_record_sha256"
    )
    report["uv_distribution_executable_record_path"] = provision_report.get(
        "uv_distribution_executable_record_path"
    )
    report["uv_distribution_executable_record_sha256"] = provision_report.get(
        "uv_distribution_executable_record_sha256"
    )
    report["uv_executable_path"] = provision_report.get("uv_executable_path")
    report["uv_executable_sha256"] = provision_report.get("uv_executable_sha256")
    report["python_executable_path"] = provision_report.get("python_executable_path")
    report["python_executable_sha256"] = provision_report.get("python_executable_sha256")
    if (
        provision_report.get("decision") != "provisioned"
        or provision_report.get("provisioned") is not True
        or provision_report.get("formal_ready") is not False
        or provision_report.get("profile_digest") != ready_profile.profile_digest
        or provision_report.get("formal_execution_lock") != formal_execution_lock
        or provision_report.get("uv_distribution_executable_record_sha256")
        != provision_report.get("uv_executable_sha256")
        or not isinstance(
            provision_report.get("uv_distribution_record_sha256"),
            str,
        )
        or len(provision_report.get("uv_distribution_record_sha256", "")) != 64
    ):
        return _fail_report(report, report_path, "isolated_python_provision_failed")
    report["provisioned"] = True

    python_executable = Path(str(provision_report["python_executable_path"]))
    if not python_executable.is_file():
        return _fail_report(report, report_path, "isolated_python_executable_missing")
    dependency_report_path = (
        _profile_output_dir(root, profile_id) / DEPENDENCY_PREPARATION_REPORT_FILE_NAME
    )
    dependency_report_path.unlink(missing_ok=True)
    command_environment = {
        UV_PYTHON_INSTALL_DIR_ENV_NAME: str(Path(managed_python_root).expanduser().resolve()),
    }
    prepare_argv = [
        str(python_executable),
        "-m",
        "experiments.runtime.dependency_preparation",
        "--profile",
        profile_id,
    ]
    prepare_result = _execute_command(
        "dependency_profile_preparation",
        prepare_argv,
        repository_root=root,
        environment_overrides=command_environment,
        command_runner=command_runner,
    )
    report["dependency_preparation_command"] = prepare_result
    report["command_results"].append(prepare_result)
    report["dependency_preparation_report_path"] = str(dependency_report_path)
    if prepare_result["return_code"] != 0:
        return _fail_report(report, report_path, "dependency_profile_preparation_failed")
    if not dependency_report_path.is_file():
        return _fail_report(report, report_path, "dependency_preparation_report_missing")
    try:
        dependency_report = json.loads(dependency_report_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return _fail_report(report, report_path, "dependency_preparation_report_invalid")
    report["dependency_preparation_report"] = dependency_report
    report["dependency_preparation_report_digest"] = _file_sha256(dependency_report_path)
    validation_errors = _validate_dependency_preparation_report(
        dependency_report,
        profile=ready_profile,
        python_executable=python_executable,
        formal_execution_lock=formal_execution_lock,
    )
    if validation_errors:
        report["failure_reasons"] = validation_errors
        report["decision"] = "fail"
        _write_report(report_path, report)
        return report, report_path

    python_digest_after = _file_sha256(python_executable)
    report["python_executable_sha256_after_preparation"] = python_digest_after
    if python_digest_after != report["python_executable_sha256"]:
        return _fail_report(report, report_path, "python_executable_digest_drift")
    report["formal_preparation_completed"] = True
    report["formal_ready"] = True
    report["decision"] = "pass"
    report["failure_reasons"] = []
    _write_report(report_path, report)
    return report, report_path


def build_parser() -> argparse.ArgumentParser:
    """构造独立 GPU 服务器可调用的隔离环境 CLI."""

    parser = argparse.ArgumentParser(description="准备受治理隔离 Python 依赖环境.")
    parser.add_argument("--profile", required=True, choices=ISOLATED_DEPENDENCY_PROFILE_IDS)
    parser.add_argument("--mode", choices=("provision", "prepare"), default="prepare")
    parser.add_argument("--environment-root", default=str(DEFAULT_ENVIRONMENT_ROOT))
    parser.add_argument("--managed-python-root", default=str(DEFAULT_MANAGED_PYTHON_ROOT))
    parser.add_argument("--uv-executable", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行 provision 或正式 preparation 并输出报告路径."""

    arguments = build_parser().parse_args(argv)
    operation = (
        provision_isolated_dependency_python
        if arguments.mode == "provision"
        else prepare_isolated_dependency_environment
    )
    try:
        report, report_path = operation(
            arguments.profile,
            environment_root=arguments.environment_root,
            managed_python_root=arguments.managed_python_root,
            uv_executable_path=arguments.uv_executable,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "profile_id": arguments.profile,
                    "operation_kind": arguments.mode,
                    "decision": "fail",
                    "failure_reasons": [f"isolated_environment_error:{type(exc).__name__}"],
                    "error": str(exc),
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
                "operation_kind": report["operation_kind"],
                "report_path": str(report_path),
                "decision": report["decision"],
                "failure_reasons": report["failure_reasons"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["decision"] in {"provisioned", "pass"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
