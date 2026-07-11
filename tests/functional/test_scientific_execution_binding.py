"""验证科学执行证据的本地化绑定与离线审计语义."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runtime.scientific_execution_binding import (
    DEPENDENCY_REPORT_FILE_NAME,
    EXECUTION_REPORT_FILE_NAME,
    file_sha256,
    stable_json_file_digest,
    validate_scientific_command_context_snapshot,
    validate_scientific_execution_binding,
    write_scientific_execution_binding,
)
from tests.helpers.formal_execution_lock import build_test_formal_execution_lock


pytestmark = pytest.mark.quick
PROFILE_ID = "sd35_method_runtime_gpu"


def _write_source_session(
    root: Path,
    *,
    session_name: str,
    dependency_bytes: bytes,
) -> tuple[Path, Path, Path]:
    """构造满足 live 重验证门禁的最小隔离执行会话证据."""

    source_dir = root / "outputs" / "source_sessions" / session_name
    source_dir.mkdir(parents=True)
    dependency_path = (
        root
        / "outputs"
        / "dependency_profiles"
        / PROFILE_ID
        / "isolated_dependency_environment_report.json"
    )
    dependency_path.parent.mkdir(parents=True, exist_ok=True)
    python_path = source_dir / "python"
    python_path.write_bytes((session_name + "-python").encode("utf-8"))
    execution_path = source_dir / "execution.json"
    formal_lock = build_test_formal_execution_lock("1" * 40)
    python_digest = file_sha256(python_path)
    python_executable = str(python_path.resolve())
    working_directory = str(root.resolve())
    complete_hash_lock_path = "configs/dependency_profiles/test_complete.lock"
    complete_hash_lock_absolute_path = str(
        (root / complete_hash_lock_path).resolve()
    )
    pytorch_index_url = "https://download.pytorch.org/whl/cu128"
    managed_python_root = str((root / "temporary_managed_python").resolve())
    isolated_environment_path = str((root / "temporary_environment").resolve())
    uv_executable_path = str((root / "temporary_uv" / "uv").resolve())
    command_environment = {"UV_PYTHON_INSTALL_DIR": managed_python_root}

    def command_record(operation: str, argv: list[str]) -> dict[str, object]:
        """构造完整的测试依赖命令记录."""

        return {
            "operation": operation,
            "argv": argv,
            "working_directory": working_directory,
            "environment_overrides": command_environment,
            "return_code": 0,
            "stdout": "",
            "stderr": "",
        }

    provision_commands = [
        command_record("uv_version", [uv_executable_path, "--version"]),
        command_record(
            "uv_python_install",
            [
                uv_executable_path,
                "python",
                "install",
                "3.11.0",
                "--install-dir",
                managed_python_root,
            ],
        ),
        command_record(
            "uv_venv",
            [
                uv_executable_path,
                "venv",
                "--clear",
                "--python",
                "3.11.0",
                "--managed-python",
                isolated_environment_path,
            ],
        ),
        command_record(
            "python_ensurepip",
            [python_executable, "-m", "ensurepip"],
        ),
        command_record(
            "python_patch_inspection",
            [
                python_executable,
                "-c",
                "import platform; print(platform.python_version())",
            ],
        ),
    ]
    dependency_preparation_command = command_record(
        "dependency_profile_preparation",
        [
            python_executable,
            "-m",
            "experiments.runtime.dependency_preparation",
            "--profile",
            PROFILE_ID,
        ],
    )
    dependency_preparation = {
        "report_schema": "dependency_profile_preparation_report",
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "python_executable": python_executable,
        "working_directory": working_directory,
        "profile_digest": "3" * 64,
        "direct_requirements_digest": "4" * 64,
        "complete_hash_lock_digest": "5" * 64,
        "complete_hash_lock_dependency_count": 1,
        "complete_hash_lock_path": complete_hash_lock_path,
        "pytorch_index_url": pytorch_index_url,
        "formal_ready": True,
        "readiness_blockers": [],
        "formal_execution_lock": formal_lock,
        "formal_execution_commit": formal_lock["formal_execution_commit"],
        "formal_execution_lock_digest": formal_lock[
            "formal_execution_lock_digest"
        ],
        "formal_execution_lock_ready": True,
        "repository_commit_state": {"all_committed": True},
        "installation": {
            "attempted": True,
            "command": [
                python_executable,
                "-m",
                "pip",
                "install",
                "--require-hashes",
                "--only-binary=:all:",
                "--extra-index-url",
                pytorch_index_url,
                "-r",
                complete_hash_lock_absolute_path,
            ],
            "working_directory": working_directory,
            "return_code": 0,
        },
        "pip_check": {
            "compatibility_check_required": True,
            "attempted": True,
            "command": [python_executable, "-m", "pip", "check"],
            "working_directory": working_directory,
            "return_code": 0,
            "decision": "pass",
        },
        "runtime_comparison": {
            "decision": "pass",
            "profile_digest": "3" * 64,
            "complete_hash_lock_digest": "5" * 64,
        },
        "decision": "pass",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    provision_report = {
        "report_schema": "isolated_dependency_python_provision_report",
        "schema_version": 1,
        "operation_kind": "isolated_python_provision",
        "profile_id": PROFILE_ID,
        "profile_digest": "3" * 64,
        "python_version": "3.11.0",
        "managed_python_root": managed_python_root,
        "working_directory": working_directory,
        "isolated_environment_path": isolated_environment_path,
        "uv_executable_path": uv_executable_path,
        "python_executable_path": python_executable,
        "command_results": provision_commands,
        "uv_commands": provision_commands[:3],
        "target_complete_hash_lock_ready": True,
        "python_executable_sha256": python_digest,
        "formal_execution_lock": formal_lock,
        "formal_execution_commit": formal_lock["formal_execution_commit"],
        "formal_execution_lock_digest": formal_lock[
            "formal_execution_lock_digest"
        ],
        "formal_execution_lock_ready": True,
        "provisioned": True,
        "formal_ready": False,
        "decision": "provisioned",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    dependency_report = {
        "report_schema": "isolated_dependency_environment_preparation_report",
        "schema_version": 1,
        "operation_kind": "formal_dependency_environment_preparation",
        "profile_id": PROFILE_ID,
        "profile_digest": "3" * 64,
        "direct_requirements_digest": "4" * 64,
        "complete_hash_lock_digest": "5" * 64,
        "complete_hash_lock_dependency_count": 1,
        "working_directory": working_directory,
        "python_executable_path": python_executable,
        "python_executable_sha256": python_digest,
        "python_executable_sha256_after_preparation": python_digest,
        "dependency_preparation_report": dependency_preparation,
        "dependency_preparation_report_digest": stable_json_file_digest(
            dependency_preparation
        ),
        "provision_report": provision_report,
        "provision_report_digest": stable_json_file_digest(provision_report),
        "dependency_preparation_command": dependency_preparation_command,
        "command_results": [*provision_commands, dependency_preparation_command],
        "uv_commands": provision_commands[:3],
        "provisioned": True,
        "formal_execution_lock": formal_lock,
        "formal_execution_commit": formal_lock["formal_execution_commit"],
        "formal_execution_lock_digest": formal_lock[
            "formal_execution_lock_digest"
        ],
        "formal_execution_lock_ready": True,
        "formal_preparation_completed": True,
        "formal_ready": True,
        "decision": "pass",
        "failure_reasons": [],
        "supports_paper_claim": False,
        "session_token": dependency_bytes.decode("utf-8"),
    }
    dependency_path.write_text(
        json.dumps(dependency_report, sort_keys=True),
        encoding="utf-8",
    )
    execution_report = {
        "report_schema": "isolated_scientific_execution_report",
        "schema_version": 1,
        "operation_kind": "isolated_scientific_execution",
        "repository_root": str(root.resolve()),
        "profile_id": PROFILE_ID,
        "profile_digest": "3" * 64,
        "direct_requirements_digest": "4" * 64,
        "complete_hash_lock_digest": "5" * 64,
        "complete_hash_lock_dependency_count": 1,
        "dependency_environment_report_path": str(dependency_path.resolve()),
        "source_dependency_environment_report_path": str(dependency_path.resolve()),
        "dependency_environment_report_digest": file_sha256(dependency_path),
        "dependency_environment_report_valid": True,
        "python_executable_path": str(python_path.resolve()),
        "python_executable_sha256": file_sha256(python_path),
        "python_executable_revalidated_before_child": True,
        "python_executable_revalidated_after_child": True,
        "dependency_environment_report_revalidated_before_child": True,
        "dependency_environment_report_revalidated_after_child": True,
        "formal_execution_lock": formal_lock,
        "formal_execution_commit": formal_lock["formal_execution_commit"],
        "formal_execution_lock_digest": formal_lock[
            "formal_execution_lock_digest"
        ],
        "formal_execution_lock_ready": True,
        "formal_execution_lock_revalidated_before_child": True,
        "formal_execution_lock_revalidated_after_child": True,
        "child_argv_tail": ["-m", "experiments.runtime.test_scientific_child"],
        "execution": {
            "attempted": True,
            "argv": [
                str(python_path.resolve()),
                "-m",
                "experiments.runtime.test_scientific_child",
            ],
            "working_directory": str(root.resolve()),
            "environment_overrides": {
                "SLM_WM_FORMAL_EXECUTION_COMMIT": formal_lock[
                    "formal_execution_commit"
                ],
                "SLM_WM_FORMAL_EXECUTION_LOCK_DIGEST": formal_lock[
                    "formal_execution_lock_digest"
                ],
                "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH": str(
                    dependency_path.resolve()
                ),
                "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST": (
                    file_sha256(dependency_path)
                ),
            },
            "return_code": 0,
        },
        "execution_report_path": str(execution_path.resolve()),
        "execution_completed": True,
        "decision": "pass",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    execution_path.write_text(
        json.dumps(execution_report, sort_keys=True),
        encoding="utf-8",
    )
    dispatch_path = source_dir / "dispatch.json"
    dispatch_path.write_text(
        json.dumps(
            {
                "report_schema": "scientific_command_dispatch_report",
                "schema_version": 1,
                "paper_run_name": "probe_paper",
                "decision": "pass",
                "failure_reasons": [],
                "supports_paper_claim": False,
            }
        ),
        encoding="utf-8",
    )
    return execution_path, dispatch_path, python_path


def _artifact_dir(root: Path) -> Path:
    """创建绑定所需的最小正式产物目录."""

    artifact_dir = root / "outputs" / "image_only_dataset_runtime" / "probe_paper"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "summary.json").write_text('{"decision":"pass"}', encoding="utf-8")
    (artifact_dir / "manifest.local.json").write_text(
        '{"artifact_id":"fixture"}',
        encoding="utf-8",
    )
    return artifact_dir


def _write_binding(
    root: Path,
    artifact_dir: Path,
    execution_path: Path,
    dispatch_path: Path,
) -> Path:
    """以固定测试身份写入一次科学执行绑定."""

    _, binding_path = write_scientific_execution_binding(
        artifact_dir,
        artifact_role="image_only_dataset_runtime",
        paper_run_name="probe_paper",
        summary_file_name="summary.json",
        manifest_file_name="manifest.local.json",
        execution_report_path=execution_path,
        dispatch_report_path=dispatch_path,
        expected_profile_id=PROFILE_ID,
        repository_root=root,
    )
    return binding_path


def test_binding_remains_valid_after_live_session_disappears(tmp_path: Path) -> None:
    """临时 venv 与共享报告消失后, 产物内快照仍必须可离线核验."""

    artifact_dir = _artifact_dir(tmp_path)
    execution_path, dispatch_path, python_path = _write_source_session(
        tmp_path,
        session_name="session_a",
        dependency_bytes=b"dependency-a",
    )
    source_dependency_path = (
        tmp_path
        / "outputs"
        / "dependency_profiles"
        / PROFILE_ID
        / "isolated_dependency_environment_report.json"
    )
    binding_path = _write_binding(
        tmp_path,
        artifact_dir,
        execution_path,
        dispatch_path,
    )

    localized_report = json.loads(
        (artifact_dir / EXECUTION_REPORT_FILE_NAME).read_text(encoding="utf-8")
    )
    assert localized_report["dependency_environment_report_path"] == DEPENDENCY_REPORT_FILE_NAME
    assert localized_report["execution_report_path"] == EXECUTION_REPORT_FILE_NAME

    python_path.unlink()
    source_dependency_path.unlink()
    execution_path.unlink()
    validated = validate_scientific_execution_binding(
        binding_path,
        expected_artifact_role="image_only_dataset_runtime",
        expected_paper_run_name="probe_paper",
        repository_root=tmp_path,
    )
    assert validated["decision"] == "pass"


def test_scientific_command_context_rejects_shadow_working_directory(
    tmp_path: Path,
) -> None:
    """工作目录和依赖报告路径不得脱离执行报告记录的仓库根."""

    execution_path, _, _ = _write_source_session(
        tmp_path,
        session_name="context_check",
        dependency_bytes=b"dependency-context",
    )
    report = json.loads(execution_path.read_text(encoding="utf-8"))
    report["execution"]["working_directory"] = str(
        (tmp_path / "attacker_shadow_root").resolve()
    )
    report["execution"]["environment_overrides"][
        "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH"
    ] = str((tmp_path / "attacker" / "unrelated.json").resolve())

    with pytest.raises(RuntimeError, match="上下文身份无效"):
        validate_scientific_command_context_snapshot(
            report,
            expected_profile_id=PROFILE_ID,
        )


def test_new_session_overwrites_local_snapshots_as_one_consistent_binding(
    tmp_path: Path,
) -> None:
    """同一产物再次完成时, 四个快照应整体切换到新会话而非混用旧证据."""

    artifact_dir = _artifact_dir(tmp_path)
    first_execution, first_dispatch, _ = _write_source_session(
        tmp_path,
        session_name="session_a",
        dependency_bytes=b"dependency-a",
    )
    binding_path = _write_binding(
        tmp_path,
        artifact_dir,
        first_execution,
        first_dispatch,
    )
    first_binding_digest = file_sha256(binding_path)

    second_execution, second_dispatch, second_python = _write_source_session(
        tmp_path,
        session_name="session_b",
        dependency_bytes=b"dependency-b",
    )
    binding_path = _write_binding(
        tmp_path,
        artifact_dir,
        second_execution,
        second_dispatch,
    )

    assert file_sha256(binding_path) != first_binding_digest
    localized_dependency = json.loads(
        (artifact_dir / DEPENDENCY_REPORT_FILE_NAME).read_text(encoding="utf-8")
    )
    assert localized_dependency["session_token"] == "dependency-b"
    second_python.unlink()
    (
        tmp_path
        / "outputs"
        / "dependency_profiles"
        / PROFILE_ID
        / "isolated_dependency_environment_report.json"
    ).unlink()
    second_execution.unlink()
    validate_scientific_execution_binding(
        binding_path,
        expected_artifact_role="image_only_dataset_runtime",
        expected_paper_run_name="probe_paper",
        repository_root=tmp_path,
    )
