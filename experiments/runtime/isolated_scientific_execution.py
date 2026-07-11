"""在受治理隔离 Python 中执行单个科学子命令.

该模块位于实验运行时层, 只负责验证隔离依赖环境、启动子解释器和保存执行证据.
调用方仍负责选择具体科学 runner 及其参数, 因而本模块可以被主方法、外部 baseline
或独立 GPU 服务器复用, 且不需要引用 ``scripts`` 或 ``paper_workflow``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence, Union

from experiments.runtime.dependency_profiles import get_dependency_profile
from experiments.runtime.isolated_dependency_environment import (
    DEFAULT_ENVIRONMENT_ROOT,
    DEFAULT_MANAGED_PYTHON_ROOT,
    FORMAL_REPORT_FILE_NAME as DEPENDENCY_ENVIRONMENT_REPORT_FILE_NAME,
    FORMAL_REPORT_SCHEMA as DEPENDENCY_ENVIRONMENT_REPORT_SCHEMA,
    ISOLATED_DEPENDENCY_PROFILE_IDS,
    REPORT_SCHEMA_VERSION as DEPENDENCY_ENVIRONMENT_REPORT_SCHEMA_VERSION,
    isolated_python_executable_path,
    prepare_isolated_dependency_environment,
)
from experiments.runtime.repository_environment import (
    FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY,
    FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY,
    require_published_formal_execution_lock,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_SCHEMA = "isolated_scientific_execution_report"
REPORT_SCHEMA_VERSION = 1
DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY = (
    "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH"
)
DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY = (
    "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST"
)

PathLike = Union[str, Path]
CommandRunner = Callable[[Sequence[str], Path, Mapping[str, str]], Any]


def _run_command(
    command: Sequence[str],
    working_directory: Path,
    environment_overrides: Mapping[str, str],
) -> Mapping[str, Any]:
    """执行隔离科学子命令并完整捕获标准输出与标准错误."""

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


def _normalize_command_result(result: Any) -> Mapping[str, Any]:
    """把可注入 runner 的返回值收敛为稳定进程证据."""

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


def _file_sha256(path: Path) -> str:
    """计算实际环境报告或解释器文件的 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    """判断一个值是否是规范小写 SHA-256 文本."""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_execution_report_path(repository_root: Path, report_path: PathLike) -> Path:
    """解析调用方报告路径并限制其位于当前仓库 ``outputs/`` 下."""

    raw_path = Path(report_path).expanduser()
    resolved = (
        raw_path.resolve()
        if raw_path.is_absolute()
        else (repository_root / raw_path).resolve()
    )
    outputs_root = (repository_root / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("隔离科学执行报告必须位于当前仓库 outputs/ 下") from exc
    if resolved.suffix.lower() != ".json":
        raise ValueError("隔离科学执行报告必须使用 .json 文件名")
    return resolved


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    """以稳定 JSON 排版持久化成功或失败报告."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _report_skeleton(
    profile_id: str,
    child_argv_tail: Sequence[str],
    repository_root: Path,
    report_path: Path,
) -> dict[str, Any]:
    """构造所有执行分支共享的固定报告 schema."""

    return {
        "report_schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "operation_kind": "isolated_scientific_execution",
        "profile_id": profile_id,
        "profile_digest": None,
        "direct_requirements_digest": None,
        "complete_hash_lock_digest": None,
        "complete_hash_lock_dependency_count": 0,
        "dependency_environment_report_path": None,
        "dependency_environment_report_digest": None,
        "dependency_environment_report_valid": False,
        "dependency_environment_validation_errors": [],
        "python_executable_path": None,
        "python_executable_sha256": None,
        "python_executable_revalidated_before_child": False,
        "python_executable_revalidated_after_child": False,
        "dependency_environment_report_revalidated_before_child": False,
        "dependency_environment_report_revalidated_after_child": False,
        "formal_execution_lock": {},
        "formal_execution_commit": None,
        "formal_execution_lock_digest": None,
        "formal_execution_lock_ready": False,
        "formal_execution_lock_revalidated_before_child": False,
        "formal_execution_lock_revalidated_after_child": False,
        "child_argv_tail": list(child_argv_tail),
        "execution": {
            "attempted": False,
            "argv": [],
            "working_directory": str(repository_root),
            "environment_overrides": {},
            "return_code": None,
            "stdout": "",
            "stderr": "",
        },
        "execution_report_path": str(report_path),
        "execution_completed": False,
        "decision": "fail",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }


def _fail_report(
    report: dict[str, Any],
    report_path: Path,
    reason: str,
) -> tuple[dict[str, Any], Path]:
    """记录首个不可恢复边界并保持非论文证据状态."""

    report["decision"] = "fail"
    report["failure_reasons"] = [reason]
    report["supports_paper_claim"] = False
    _write_report(report_path, report)
    return report, report_path


def _expected_dependency_environment_report_path(
    repository_root: Path,
    profile_id: str,
) -> Path:
    """返回共享环境准备 API 的唯一正式报告路径."""

    return (
        repository_root
        / "outputs"
        / "dependency_profiles"
        / profile_id
        / DEPENDENCY_ENVIRONMENT_REPORT_FILE_NAME
    ).resolve()


def _validate_dependency_environment_report(
    report: Any,
    *,
    persisted_report: Any,
    profile: Any,
    expected_python_executable: Path,
    formal_execution_lock: Mapping[str, Any],
) -> list[str]:
    """严格核对隔离环境报告、解释器身份、锁摘要和执行锁."""

    if not isinstance(report, dict):
        return ["dependency_environment_report_not_object"]
    errors = []
    expected_values = {
        "report_schema": DEPENDENCY_ENVIRONMENT_REPORT_SCHEMA,
        "schema_version": DEPENDENCY_ENVIRONMENT_REPORT_SCHEMA_VERSION,
        "operation_kind": "formal_dependency_environment_preparation",
        "profile_id": profile.profile_name,
        "profile_digest": profile.profile_digest,
        "direct_requirements_digest": profile.direct_requirements_digest,
        "complete_hash_lock_digest": profile.complete_hash_lock_digest,
        "complete_hash_lock_dependency_count": profile.complete_hash_lock_dependency_count,
        "formal_execution_lock": dict(formal_execution_lock),
        "formal_execution_commit": formal_execution_lock["formal_execution_commit"],
        "formal_execution_lock_digest": formal_execution_lock[
            "formal_execution_lock_digest"
        ],
        "formal_execution_lock_ready": True,
        "provisioned": True,
        "formal_preparation_completed": True,
        "formal_ready": True,
        "decision": "pass",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    for field_name, expected_value in expected_values.items():
        if report.get(field_name) != expected_value:
            errors.append("dependency_environment_{0}_mismatch".format(field_name))
    if profile.formal_ready is not True or profile.readiness_blockers:
        errors.append("registered_dependency_profile_not_ready")
    if not _is_sha256(profile.profile_digest):
        errors.append("registered_dependency_profile_digest_invalid")
    if not _is_sha256(profile.direct_requirements_digest):
        errors.append("registered_direct_requirements_digest_invalid")
    if not _is_sha256(profile.complete_hash_lock_digest):
        errors.append("registered_complete_hash_lock_digest_invalid")
    if profile.complete_hash_lock_dependency_count <= 0:
        errors.append("registered_complete_hash_lock_dependency_count_invalid")
    if persisted_report != report:
        errors.append("dependency_environment_persisted_report_mismatch")

    python_path_value = report.get("python_executable_path")
    python_path = Path(str(python_path_value)) if python_path_value else Path()
    if (
        not python_path_value
        or not python_path.is_absolute()
        or not python_path.is_file()
        or python_path.resolve() != expected_python_executable.resolve()
    ):
        errors.append("dependency_environment_python_executable_invalid")
        actual_python_digest = None
    else:
        actual_python_digest = _file_sha256(python_path)
    registered_python_digest = report.get("python_executable_sha256")
    registered_python_digest_after = report.get(
        "python_executable_sha256_after_preparation"
    )
    if (
        not _is_sha256(registered_python_digest)
        or registered_python_digest_after != registered_python_digest
        or actual_python_digest != registered_python_digest
    ):
        errors.append("dependency_environment_python_executable_digest_invalid")

    dependency_preparation_report = report.get("dependency_preparation_report")
    if not isinstance(dependency_preparation_report, dict):
        errors.append("dependency_preparation_report_not_object")
    else:
        nested_expected = {
            "profile_id": profile.profile_name,
            "profile_digest": profile.profile_digest,
            "complete_hash_lock_digest": profile.complete_hash_lock_digest,
            "python_executable": str(python_path),
            "formal_execution_lock": dict(formal_execution_lock),
            "formal_ready": True,
            "decision": "pass",
            "failure_reasons": [],
            "supports_paper_claim": False,
        }
        for field_name, expected_value in nested_expected.items():
            if dependency_preparation_report.get(field_name) != expected_value:
                errors.append(
                    "dependency_preparation_{0}_mismatch".format(field_name)
                )
    return list(dict.fromkeys(errors))


def execute_isolated_scientific_command(
    profile_id: str,
    child_argv_tail: Sequence[str],
    *,
    execution_report_path: PathLike,
    repository_root: PathLike = ROOT,
    environment_root: PathLike = DEFAULT_ENVIRONMENT_ROOT,
    managed_python_root: PathLike = DEFAULT_MANAGED_PYTHON_ROOT,
    uv_executable_path: Union[PathLike, None] = None,
    command_runner: CommandRunner = _run_command,
) -> tuple[dict[str, Any], Path]:
    """准备隔离环境并用其受验证 Python 执行调用方提供的 argv 尾部.

    ``child_argv_tail`` 不包含 Python executable. 本函数只在隔离环境报告、解释器
    文件摘要、完整依赖锁和正式执行锁全部一致时自动前置解释器并启动子进程.
    """

    root = Path(repository_root).resolve()
    report_path = _resolve_execution_report_path(root, execution_report_path)
    normalized_profile_id = str(profile_id)
    tail_is_sequence = not isinstance(child_argv_tail, (str, bytes))
    normalized_tail = (
        tuple(str(token) for token in child_argv_tail)
        if tail_is_sequence
        else ()
    )
    report = _report_skeleton(
        normalized_profile_id,
        normalized_tail,
        root,
        report_path,
    )
    if normalized_profile_id not in ISOLATED_DEPENDENCY_PROFILE_IDS:
        return _fail_report(report, report_path, "scientific_profile_not_allowed")
    if not tail_is_sequence or not normalized_tail or not normalized_tail[0]:
        return _fail_report(report, report_path, "scientific_child_argv_empty")
    expected_dependency_report_path = _expected_dependency_environment_report_path(
        root,
        normalized_profile_id,
    )
    if report_path == expected_dependency_report_path:
        raise ValueError("隔离科学执行报告不得覆盖依赖环境报告")

    registry_path = root / "configs" / "dependency_profile_registry.json"
    try:
        profile = get_dependency_profile(normalized_profile_id, registry_path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        report["dependency_environment_validation_errors"] = [
            "dependency_profile_load_failed:{0}".format(type(exc).__name__)
        ]
        return _fail_report(report, report_path, "dependency_profile_unavailable")
    if profile.profile_name != normalized_profile_id:
        report["dependency_environment_validation_errors"] = [
            "dependency_profile_identity_mismatch"
        ]
        return _fail_report(report, report_path, "dependency_profile_unavailable")
    report["profile_digest"] = profile.profile_digest
    report["direct_requirements_digest"] = profile.direct_requirements_digest
    report["complete_hash_lock_digest"] = profile.complete_hash_lock_digest
    report["complete_hash_lock_dependency_count"] = (
        profile.complete_hash_lock_dependency_count
    )

    try:
        formal_execution_lock = require_published_formal_execution_lock(root)
    except ValueError as exc:
        report["dependency_environment_validation_errors"] = [
            "formal_execution_lock_unavailable:{0}".format(type(exc).__name__)
        ]
        return _fail_report(report, report_path, "formal_execution_lock_not_ready")
    report["formal_execution_lock"] = formal_execution_lock
    report["formal_execution_commit"] = formal_execution_lock[
        "formal_execution_commit"
    ]
    report["formal_execution_lock_digest"] = formal_execution_lock[
        "formal_execution_lock_digest"
    ]
    report["formal_execution_lock_ready"] = True

    try:
        dependency_environment_report, dependency_environment_report_path = (
            prepare_isolated_dependency_environment(
                normalized_profile_id,
                repository_root=root,
                environment_root=environment_root,
                managed_python_root=managed_python_root,
                uv_executable_path=uv_executable_path,
            )
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        report["dependency_environment_validation_errors"] = [
            "dependency_environment_preparation_failed:{0}".format(type(exc).__name__)
        ]
        return _fail_report(
            report,
            report_path,
            "dependency_environment_preparation_failed",
        )

    resolved_dependency_report_path = Path(dependency_environment_report_path).resolve()
    report["dependency_environment_report_path"] = str(
        resolved_dependency_report_path
    )
    if (
        resolved_dependency_report_path != expected_dependency_report_path
        or not resolved_dependency_report_path.is_file()
    ):
        report["dependency_environment_validation_errors"] = [
            "dependency_environment_report_path_invalid"
        ]
        return _fail_report(
            report,
            report_path,
            "dependency_environment_report_rejected",
        )
    try:
        persisted_dependency_report = json.loads(
            resolved_dependency_report_path.read_text(encoding="utf-8-sig")
        )
    except (OSError, json.JSONDecodeError) as exc:
        report["dependency_environment_validation_errors"] = [
            "dependency_environment_report_read_failed:{0}".format(type(exc).__name__)
        ]
        return _fail_report(
            report,
            report_path,
            "dependency_environment_report_rejected",
        )
    dependency_report_digest = _file_sha256(resolved_dependency_report_path)
    report["dependency_environment_report_digest"] = dependency_report_digest
    expected_python_executable = isolated_python_executable_path(
        normalized_profile_id,
        environment_root=environment_root,
    )
    validation_errors = _validate_dependency_environment_report(
        dependency_environment_report,
        persisted_report=persisted_dependency_report,
        profile=profile,
        expected_python_executable=expected_python_executable,
        formal_execution_lock=formal_execution_lock,
    )
    report["dependency_environment_validation_errors"] = validation_errors
    if validation_errors:
        return _fail_report(
            report,
            report_path,
            "dependency_environment_report_rejected",
        )
    report["dependency_environment_report_valid"] = True

    python_executable = Path(dependency_environment_report["python_executable_path"])
    python_digest = str(dependency_environment_report["python_executable_sha256"])
    report["python_executable_path"] = str(python_executable)
    report["python_executable_sha256"] = python_digest
    try:
        lock_before_child = require_published_formal_execution_lock(root)
    except ValueError:
        return _fail_report(
            report,
            report_path,
            "formal_execution_lock_drift_before_child",
        )
    if lock_before_child != formal_execution_lock:
        return _fail_report(
            report,
            report_path,
            "formal_execution_lock_drift_before_child",
        )
    report["formal_execution_lock_revalidated_before_child"] = True
    if (
        not python_executable.is_file()
        or _file_sha256(python_executable) != python_digest
    ):
        return _fail_report(
            report,
            report_path,
            "python_executable_drift_before_child",
        )
    report["python_executable_revalidated_before_child"] = True
    if (
        not resolved_dependency_report_path.is_file()
        or _file_sha256(resolved_dependency_report_path) != dependency_report_digest
    ):
        return _fail_report(
            report,
            report_path,
            "dependency_environment_report_drift_before_child",
        )
    report["dependency_environment_report_revalidated_before_child"] = True

    environment_overrides = {
        FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY: formal_execution_lock[
            "formal_execution_commit"
        ],
        FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY: formal_execution_lock[
            "formal_execution_lock_digest"
        ],
        DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY: str(
            resolved_dependency_report_path
        ),
        DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY: dependency_report_digest,
    }
    command = [str(python_executable), *normalized_tail]
    report["execution"] = {
        "attempted": True,
        "argv": command,
        "working_directory": str(root),
        "environment_overrides": dict(sorted(environment_overrides.items())),
        "return_code": None,
        "stdout": "",
        "stderr": "",
    }
    try:
        raw_execution_result = command_runner(command, root, environment_overrides)
    except OSError as exc:
        report["execution"]["stderr"] = str(exc)
        return _fail_report(
            report,
            report_path,
            "scientific_child_command_launch_failed",
        )
    try:
        execution_result = _normalize_command_result(raw_execution_result)
    except (TypeError, ValueError) as exc:
        report["execution"]["stderr"] = str(exc)
        return _fail_report(
            report,
            report_path,
            "scientific_child_command_result_invalid",
        )
    report["execution"].update(execution_result)

    try:
        lock_after_child = require_published_formal_execution_lock(root)
    except ValueError:
        return _fail_report(
            report,
            report_path,
            "formal_execution_lock_drift_after_child",
        )
    if lock_after_child != formal_execution_lock:
        return _fail_report(
            report,
            report_path,
            "formal_execution_lock_drift_after_child",
        )
    report["formal_execution_lock_revalidated_after_child"] = True
    if (
        not python_executable.is_file()
        or _file_sha256(python_executable) != python_digest
    ):
        return _fail_report(
            report,
            report_path,
            "python_executable_drift_after_child",
        )
    report["python_executable_revalidated_after_child"] = True
    if (
        not resolved_dependency_report_path.is_file()
        or _file_sha256(resolved_dependency_report_path) != dependency_report_digest
    ):
        return _fail_report(
            report,
            report_path,
            "dependency_environment_report_drift_after_child",
        )
    report["dependency_environment_report_revalidated_after_child"] = True
    if execution_result["return_code"] != 0:
        return _fail_report(
            report,
            report_path,
            "scientific_child_command_failed",
        )

    report["execution_completed"] = True
    report["decision"] = "pass"
    report["failure_reasons"] = []
    report["supports_paper_claim"] = False
    _write_report(report_path, report)
    return report, report_path
