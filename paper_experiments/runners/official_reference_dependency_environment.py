"""为官方参考复现统一准备并核验受治理隔离依赖环境."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from experiments.runtime.dependency_profiles import (
    DependencyProfile,
    get_dependency_profile,
    require_dependency_profile_ready,
)
from experiments.runtime.isolated_dependency_environment import (
    FORMAL_REPORT_SCHEMA as ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_SCHEMA,
    REPORT_SCHEMA_VERSION as ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_SCHEMA_VERSION,
    prepare_isolated_dependency_environment,
)
from experiments.runtime.progress import emit_progress_status
from experiments.runtime import repository_environment


@dataclass(frozen=True)
class OfficialReferenceDependencyEnvironmentValidation:
    """保存隔离环境报告验证结果及后续命令所需的可信解释器."""

    validation_errors: tuple[str, ...]
    dependency_python_executable: str
    dependency_installation_performed: bool
    isolated_dependency_environment_report_digest: str

    @property
    def passed(self) -> bool:
        """仅当所有身份、安装和运行时边界均闭合时返回 ``True``."""

        return not self.validation_errors


def _stable_json_text(value: Any) -> str:
    """使用与三个 official-reference runner 相同的稳定 JSON 排版."""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    """写出共享依赖环境报告并创建父目录."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_stable_json_text(payload), encoding="utf-8")


def _read_json(path: Path) -> Any:
    """读取共享验证所需的 JSON 报告."""

    return json.loads(path.read_text(encoding="utf-8-sig"))


def _relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录相对仓库根目录的路径."""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def validate_official_reference_dependency_environment_report(
    isolated_report: Any,
    isolated_report_path: str | Path,
    ready_profile: DependencyProfile,
    *,
    expected_formal_execution_lock: Mapping[str, Any] | None = None,
) -> OfficialReferenceDependencyEnvironmentValidation:
    """统一验证隔离环境正式报告、嵌套安装证据和解释器身份.

    该函数属于通用工程写法.三个官方参考 runner 共享完全相同的依赖环境
    协议, 因此由此处集中维护字段集合、错误码和 fail-closed 判定.各 runner
    只负责传入自身固定的 profile ID 和结果文件路径.
    """

    report_path = Path(isolated_report_path)
    validation_errors: list[str] = []
    if not report_path.is_file():
        validation_errors.append("isolated_environment_report_missing")
    else:
        try:
            persisted_isolated_report = _read_json(report_path)
        except Exception:
            validation_errors.append("isolated_environment_report_unreadable")
        else:
            if persisted_isolated_report != isolated_report:
                validation_errors.append(
                    "isolated_environment_report_content_mismatch"
                )

    dependency_python_text = ""
    dependency_installation_performed = False
    if not isinstance(isolated_report, dict):
        validation_errors.append("isolated_environment_report_not_object")
    else:
        expected_values = {
            "report_schema": ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_SCHEMA,
            "schema_version": (
                ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_SCHEMA_VERSION
            ),
            "profile_id": ready_profile.profile_name,
            "profile_digest": ready_profile.profile_digest,
            "direct_requirements_digest": ready_profile.direct_requirements_digest,
            "complete_hash_lock_digest": ready_profile.complete_hash_lock_digest,
            "complete_hash_lock_dependency_count": (
                ready_profile.complete_hash_lock_dependency_count
            ),
            "provisioned": True,
            "formal_preparation_completed": True,
            "formal_ready": True,
            "decision": "pass",
            "failure_reasons": [],
            "supports_paper_claim": False,
        }
        if expected_formal_execution_lock is not None:
            expected_values.update(
                {
                    "formal_execution_lock": dict(expected_formal_execution_lock),
                    "formal_execution_commit": expected_formal_execution_lock[
                        "formal_execution_commit"
                    ],
                    "formal_execution_lock_digest": expected_formal_execution_lock[
                        "formal_execution_lock_digest"
                    ],
                    "formal_execution_lock_ready": True,
                }
            )
        validation_errors.extend(
            f"{field_name}_mismatch"
            for field_name, expected_value in expected_values.items()
            if isolated_report.get(field_name) != expected_value
        )

        dependency_python_text = str(
            isolated_report.get("python_executable_path", "")
        )
        dependency_python = (
            Path(dependency_python_text) if dependency_python_text else None
        )
        if dependency_python is None or not dependency_python.is_file():
            validation_errors.append("dependency_python_executable_missing")
        python_digest = str(
            isolated_report.get("python_executable_sha256", "")
        )
        python_digest_after = str(
            isolated_report.get(
                "python_executable_sha256_after_preparation",
                "",
            )
        )
        if (
            re.fullmatch(r"[0-9a-f]{64}", python_digest) is None
            or python_digest_after != python_digest
        ):
            validation_errors.append(
                "dependency_python_executable_digest_invalid"
            )
        elif (
            dependency_python is not None
            and dependency_python.is_file()
            and repository_environment.file_digest(dependency_python)
            != python_digest
        ):
            validation_errors.append(
                "dependency_python_executable_content_mismatch"
            )

        dependency_preparation_report = isolated_report.get(
            "dependency_preparation_report",
            {},
        )
        if not isinstance(dependency_preparation_report, dict):
            validation_errors.append("dependency_preparation_report_not_object")
            dependency_preparation_report = {}
        else:
            if (
                dependency_preparation_report.get("profile_id")
                != ready_profile.profile_name
            ):
                validation_errors.append(
                    "dependency_preparation_profile_id_mismatch"
                )
            if (
                dependency_preparation_report.get("profile_digest")
                != ready_profile.profile_digest
            ):
                validation_errors.append(
                    "dependency_preparation_profile_digest_mismatch"
                )
            if (
                dependency_preparation_report.get("complete_hash_lock_digest")
                != ready_profile.complete_hash_lock_digest
            ):
                validation_errors.append(
                    "dependency_preparation_lock_digest_mismatch"
                )
            if dependency_preparation_report.get("decision") != "pass":
                validation_errors.append(
                    "dependency_preparation_decision_not_pass"
                )
            repository_commit_state = dependency_preparation_report.get(
                "repository_commit_state"
            )
            if (
                not isinstance(repository_commit_state, dict)
                or repository_commit_state.get("all_committed") is not True
            ):
                validation_errors.append(
                    "dependency_profile_inputs_not_committed"
                )
            runtime_comparison = dependency_preparation_report.get(
                "runtime_comparison"
            )
            if (
                not isinstance(runtime_comparison, dict)
                or runtime_comparison.get("decision") != "pass"
                or runtime_comparison.get("environment_match") is not True
                or runtime_comparison.get("mismatches") != []
            ):
                validation_errors.append("dependency_runtime_not_matched")
            if expected_formal_execution_lock is not None and (
                dependency_preparation_report.get("formal_execution_lock")
                != dict(expected_formal_execution_lock)
            ):
                validation_errors.append(
                    "dependency_preparation_formal_execution_lock_mismatch"
                )

        installation = dependency_preparation_report.get("installation", {})
        dependency_installation_performed = bool(
            isinstance(installation, dict)
            and installation.get("attempted") is True
        )
        if (
            not isinstance(installation, dict)
            or installation.get("attempted") is not True
            or installation.get("return_code") != 0
        ):
            validation_errors.append("dependency_installation_not_completed")

        provision_report = isolated_report.get("provision_report", {})
        if (
            not isinstance(provision_report, dict)
            or provision_report.get("decision") != "provisioned"
            or provision_report.get("provisioned") is not True
            or provision_report.get("profile_digest")
            != ready_profile.profile_digest
        ):
            validation_errors.append("dependency_python_provision_invalid")
        if (
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(
                    isolated_report.get(
                        "dependency_preparation_report_digest",
                        "",
                    )
                ),
            )
            is None
        ):
            validation_errors.append(
                "dependency_preparation_report_digest_invalid"
            )
        else:
            dependency_report_path_text = str(
                isolated_report.get("dependency_preparation_report_path", "")
            )
            dependency_report_path = (
                Path(dependency_report_path_text)
                if dependency_report_path_text
                else None
            )
            dependency_report_digest = str(
                isolated_report.get("dependency_preparation_report_digest", "")
            )
            if (
                dependency_report_path is None
                or not dependency_report_path.is_file()
            ):
                validation_errors.append(
                    "dependency_preparation_report_file_missing"
                )
            else:
                try:
                    persisted_dependency_report = _read_json(
                        dependency_report_path
                    )
                except Exception:
                    validation_errors.append(
                        "dependency_preparation_report_file_unreadable"
                    )
                else:
                    if persisted_dependency_report != dependency_preparation_report:
                        validation_errors.append(
                            "dependency_preparation_report_content_mismatch"
                        )
                    if (
                        repository_environment.file_digest(
                            dependency_report_path
                        )
                        != dependency_report_digest
                    ):
                        validation_errors.append(
                            "dependency_preparation_report_file_digest_mismatch"
                        )

        provision_report_path_text = str(
            isolated_report.get("provision_report_path", "")
        )
        provision_report_path = (
            Path(provision_report_path_text)
            if provision_report_path_text
            else None
        )
        provision_report_digest = str(
            isolated_report.get("provision_report_digest", "")
        )
        if provision_report_path is None or not provision_report_path.is_file():
            validation_errors.append("dependency_provision_report_file_missing")
        else:
            try:
                persisted_provision_report = _read_json(provision_report_path)
            except Exception:
                validation_errors.append(
                    "dependency_provision_report_file_unreadable"
                )
            else:
                if persisted_provision_report != provision_report:
                    validation_errors.append(
                        "dependency_provision_report_content_mismatch"
                    )
                if (
                    repository_environment.file_digest(provision_report_path)
                    != provision_report_digest
                ):
                    validation_errors.append(
                        "dependency_provision_report_file_digest_mismatch"
                    )
        if expected_formal_execution_lock is not None and (
            not isinstance(provision_report, dict)
            or provision_report.get("formal_execution_lock")
            != dict(expected_formal_execution_lock)
        ):
            validation_errors.append(
                "dependency_provision_formal_execution_lock_mismatch"
            )

    report_digest = ""
    if not validation_errors and report_path.is_file():
        report_digest = repository_environment.file_digest(report_path)
    return OfficialReferenceDependencyEnvironmentValidation(
        validation_errors=tuple(validation_errors),
        dependency_python_executable=dependency_python_text,
        dependency_installation_performed=dependency_installation_performed,
        isolated_dependency_environment_report_digest=report_digest,
    )


def prepare_official_reference_dependency_environment(
    root_path: Path,
    dependency_profile_id: str,
    result_path: Path,
    progress: object | None = None,
) -> dict[str, Any]:
    """准备一个固定 official-reference profile 并写出统一诊断报告."""

    report: dict[str, Any] = {
        "dependency_environment_requested": True,
        "dependency_environment_ready": False,
        "dependency_environment_materialized": False,
        "dependency_environment_profile_id": "",
        "dependency_python_executable": "",
        "dependency_profile_id": dependency_profile_id,
        "dependency_profile_ready": False,
        "dependency_lock_ready": False,
        "dependency_environment_report_valid": False,
        "dependency_installation_performed": False,
        "command_results": [],
    }

    def persist_report() -> dict[str, Any]:
        """持久化当前诊断状态并返回同一报告对象."""

        _write_json(result_path, report)
        return report

    registry_path = root_path / "configs" / "dependency_profile_registry.json"
    try:
        profile = get_dependency_profile(dependency_profile_id, registry_path)
    except Exception as error:
        report.update(
            {
                "dependency_environment_failure_reason": "dependency_profile_invalid",
                "dependency_profile_error": f"{type(error).__name__}:{error}",
            }
        )
        return persist_report()

    report.update(
        {
            "dependency_profile": profile.to_dict(),
            "dependency_profile_digest": profile.profile_digest,
            "dependency_lock_path": profile.complete_hash_lock_path,
            "dependency_lock_digest": profile.complete_hash_lock_digest or "",
            "dependency_readiness_blockers": list(profile.readiness_blockers),
        }
    )
    try:
        ready_profile = require_dependency_profile_ready(
            dependency_profile_id,
            registry_path,
        )
    except Exception as error:
        report.update(
            {
                "dependency_environment_failure_reason": (
                    "dependency_hash_lock_not_ready"
                ),
                "dependency_profile_error": f"{type(error).__name__}:{error}",
            }
        )
        return persist_report()

    report["dependency_profile_ready"] = ready_profile.formal_ready
    report["dependency_lock_ready"] = ready_profile.complete_hash_lock_present
    emit_progress_status(
        progress,
        profile="operation=prepare_isolated_dependency_environment status=running",
    )
    try:
        isolated_report, isolated_report_path = (
            prepare_isolated_dependency_environment(
                dependency_profile_id,
                repository_root=root_path,
            )
        )
    except Exception as error:
        report.update(
            {
                "dependency_environment_failure_reason": (
                    "isolated_dependency_environment_prepare_failed"
                ),
                "dependency_environment_prepare_error": (
                    f"{type(error).__name__}:{error}"
                ),
            }
        )
        return persist_report()
    emit_progress_status(
        progress,
        profile="operation=prepare_isolated_dependency_environment status=completed",
    )

    report.update(
        {
            "isolated_dependency_environment_report_path": _relative_or_absolute(
                Path(isolated_report_path),
                root_path,
            ),
            "isolated_dependency_environment_report": isolated_report,
            "command_results": (
                list(isolated_report.get("command_results", []))
                if isinstance(isolated_report, dict)
                else []
            ),
        }
    )
    try:
        formal_execution_lock = (
            repository_environment.require_published_formal_execution_lock(
                root_path
            )
        )
    except Exception as error:
        report.update(
            {
                "dependency_environment_failure_reason": (
                    "formal_execution_lock_not_ready"
                ),
                "formal_execution_lock_error": f"{type(error).__name__}:{error}",
            }
        )
        return persist_report()
    validation = validate_official_reference_dependency_environment_report(
        isolated_report,
        isolated_report_path,
        ready_profile,
        expected_formal_execution_lock=formal_execution_lock,
    )
    report["dependency_installation_performed"] = (
        validation.dependency_installation_performed
    )
    if not validation.passed:
        report.update(
            {
                "dependency_environment_failure_reason": (
                    "isolated_dependency_environment_report_invalid"
                ),
                "dependency_environment_validation_errors": list(
                    validation.validation_errors
                ),
            }
        )
        return persist_report()

    report.update(
        {
            "dependency_environment_ready": True,
            "dependency_environment_materialized": True,
            "dependency_environment_profile_id": dependency_profile_id,
            "dependency_python_executable": (
                validation.dependency_python_executable
            ),
            "dependency_environment_report_valid": True,
            "dependency_environment_failure_reason": "",
            "isolated_dependency_environment_report_digest": (
                validation.isolated_dependency_environment_report_digest
            ),
        }
    )
    return persist_report()
