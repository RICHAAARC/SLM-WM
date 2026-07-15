"""构造离线结果包测试使用的完整科学执行绑定."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.runtime.scientific_execution_binding import (
    BOUND_MANIFEST_DIGEST_SCOPE,
    scientific_manifest_payload_digest,
    stable_json_file_digest,
)


PROFILE_DIGEST = "1" * 64
DIRECT_REQUIREMENTS_DIGEST = "2" * 64
COMPLETE_HASH_LOCK_DIGEST = "3" * 64
PYTHON_EXECUTABLE_DIGEST = "4" * 64


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """使用稳定排版写出测试 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    """计算测试产物的 SHA-256."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_test_scientific_execution_binding(
    *,
    repository_root: Path,
    artifact_dir: Path,
    artifact_role: str,
    paper_run_name: str,
    profile_id: str,
    summary_file_name: str,
    manifest_file_name: str,
    formal_execution_lock: Mapping[str, Any],
    execution_route: str = "generic_test",
    baseline_id: str | None = None,
) -> None:
    """写出可由 package selector 离线严格复核的最小真实 schema 夹具."""

    root = repository_root.resolve()
    artifact_root = artifact_dir.resolve()
    artifact_root.relative_to((root / "outputs").resolve())
    summary_path = artifact_root / summary_file_name
    manifest_path = artifact_root / manifest_file_name
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("测试科学执行绑定要求 summary 与 manifest 已存在")

    dependency_path = artifact_root / "isolated_dependency_environment_report.json"
    execution_path = artifact_root / "isolated_scientific_execution_report.json"
    dispatch_path = artifact_root / "scientific_command_dispatch_report.json"
    binding_path = artifact_root / "scientific_execution_binding.json"
    envelope_path = (
        artifact_root
        / "scientific_execution"
        / "scientific_workflow_result_envelope.json"
    )
    formal_commit = formal_execution_lock["formal_execution_commit"]
    formal_digest = formal_execution_lock["formal_execution_lock_digest"]
    recorded_python_path = str(
        (root / "temporary_scientific_python" / "python").resolve()
    )
    working_directory = str(root)
    complete_hash_lock_path = "configs/dependency_profiles/test_complete.lock"
    complete_hash_lock_absolute_path = str(
        (root / complete_hash_lock_path).resolve()
    )
    pytorch_index_url = "https://download.pytorch.org/whl/cu128"
    managed_python_root = str((root / "temporary_managed_python").resolve())
    isolated_environment_path = str(
        (root / "temporary_scientific_python").resolve()
    )
    uv_executable_path = str((root / "temporary_uv" / "uv").resolve())
    command_environment = {"UV_PYTHON_INSTALL_DIR": managed_python_root}

    def command_record(operation: str, argv: list[str]) -> dict[str, Any]:
        """构造具备完整 argv 和工作目录的测试命令记录."""

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
            [recorded_python_path, "-m", "ensurepip"],
        ),
        command_record(
            "python_patch_inspection",
            [
                recorded_python_path,
                "-c",
                "import platform; print(platform.python_version())",
            ],
        ),
    ]
    dependency_preparation_command = command_record(
        "dependency_profile_preparation",
        [
            recorded_python_path,
            "-m",
            "experiments.runtime.dependency_preparation",
            "--profile",
            profile_id,
        ],
    )
    dependency_preparation = {
        "report_schema": "dependency_profile_preparation_report",
        "schema_version": 1,
        "profile_id": profile_id,
        "python_executable": recorded_python_path,
        "working_directory": working_directory,
        "profile_digest": PROFILE_DIGEST,
        "direct_requirements_digest": DIRECT_REQUIREMENTS_DIGEST,
        "complete_hash_lock_digest": COMPLETE_HASH_LOCK_DIGEST,
        "complete_hash_lock_dependency_count": 7,
        "complete_hash_lock_path": complete_hash_lock_path,
        "pytorch_index_url": pytorch_index_url,
        "formal_ready": True,
        "readiness_blockers": [],
        "formal_execution_lock": dict(formal_execution_lock),
        "formal_execution_commit": formal_commit,
        "formal_execution_lock_digest": formal_digest,
        "formal_execution_lock_ready": True,
        "repository_commit_state": {"all_committed": True},
        "installation": {
            "attempted": True,
            "command": [
                recorded_python_path,
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
            "command": [recorded_python_path, "-m", "pip", "check"],
            "working_directory": working_directory,
            "return_code": 0,
            "decision": "pass",
        },
        "runtime_comparison": {
            "decision": "pass",
            "profile_digest": PROFILE_DIGEST,
            "complete_hash_lock_digest": COMPLETE_HASH_LOCK_DIGEST,
        },
        "decision": "pass",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    provision_report = {
        "report_schema": "isolated_dependency_python_provision_report",
        "schema_version": 1,
        "operation_kind": "isolated_python_provision",
        "profile_id": profile_id,
        "profile_digest": PROFILE_DIGEST,
        "python_version": "3.11.0",
        "managed_python_root": managed_python_root,
        "working_directory": working_directory,
        "isolated_environment_path": isolated_environment_path,
        "uv_executable_path": uv_executable_path,
        "python_executable_path": recorded_python_path,
        "command_results": provision_commands,
        "uv_commands": provision_commands[:3],
        "target_complete_hash_lock_ready": True,
        "python_executable_sha256": PYTHON_EXECUTABLE_DIGEST,
        "formal_execution_lock": dict(formal_execution_lock),
        "formal_execution_commit": formal_commit,
        "formal_execution_lock_digest": formal_digest,
        "formal_execution_lock_ready": True,
        "provisioned": True,
        "formal_ready": False,
        "decision": "provisioned",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    dependency = {
        "report_schema": "isolated_dependency_environment_preparation_report",
        "schema_version": 1,
        "operation_kind": "formal_dependency_environment_preparation",
        "profile_id": profile_id,
        "profile_digest": PROFILE_DIGEST,
        "direct_requirements_digest": DIRECT_REQUIREMENTS_DIGEST,
        "complete_hash_lock_digest": COMPLETE_HASH_LOCK_DIGEST,
        "complete_hash_lock_dependency_count": 7,
        "working_directory": working_directory,
        "python_executable_path": recorded_python_path,
        "python_executable_sha256": PYTHON_EXECUTABLE_DIGEST,
        "python_executable_sha256_after_preparation": PYTHON_EXECUTABLE_DIGEST,
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
        "formal_preparation_completed": True,
        "formal_ready": True,
        "formal_execution_lock_ready": True,
        "formal_execution_lock": dict(formal_execution_lock),
        "formal_execution_commit": formal_commit,
        "formal_execution_lock_digest": formal_digest,
        "decision": "pass",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    _write_json(dependency_path, dependency)
    source_dependency_path = str(
        (
            root
            / "outputs"
            / "dependency_profiles"
            / profile_id
            / "isolated_dependency_environment_report.json"
        ).resolve()
    )
    if execution_route == "semantic_watermark_ablation_session":
        child_argv_tail = [
            "-m",
            "experiments.runtime.semantic_watermark_scientific_session",
            "--run-formal-ablation",
        ]
    elif execution_route == "semantic_watermark_session":
        child_argv_tail = [
            "-m",
            "experiments.runtime.semantic_watermark_scientific_session",
        ]
    elif execution_route in {
        "isolated_method_faithful_workflow",
        "isolated_t2smark_workflow",
    }:
        workflow_name = (
            "external_baseline_method_faithful"
            if execution_route == "isolated_method_faithful_workflow"
            else "official_reference_t2smark"
        )
        child_argv_tail = [
            "-m",
            "paper_experiments.runners.isolated_scientific_workflow",
            "--child-workflow",
            workflow_name,
            "--root",
            str(root),
            "--result-envelope",
            str(envelope_path.resolve()),
        ]
    else:
        child_argv_tail = ["scripts/test_scientific_child.py"]
    execution = {
        "report_schema": "isolated_scientific_execution_report",
        "schema_version": 1,
        "operation_kind": "isolated_scientific_execution",
        "repository_root": str(root.resolve()),
        "profile_id": profile_id,
        "profile_digest": PROFILE_DIGEST,
        "direct_requirements_digest": DIRECT_REQUIREMENTS_DIGEST,
        "complete_hash_lock_digest": COMPLETE_HASH_LOCK_DIGEST,
        "complete_hash_lock_dependency_count": 7,
        "dependency_environment_report_path": dependency_path.name,
        "source_dependency_environment_report_path": source_dependency_path,
        "dependency_environment_report_digest": _file_sha256(dependency_path),
        "dependency_environment_report_valid": True,
        "python_executable_path": recorded_python_path,
        "python_executable_sha256": PYTHON_EXECUTABLE_DIGEST,
        "python_executable_revalidated_before_child": True,
        "python_executable_revalidated_after_child": True,
        "dependency_environment_report_revalidated_before_child": True,
        "dependency_environment_report_revalidated_after_child": True,
        "formal_execution_lock": dict(formal_execution_lock),
        "formal_execution_commit": formal_commit,
        "formal_execution_lock_digest": formal_digest,
        "formal_execution_lock_ready": True,
        "formal_execution_lock_revalidated_before_child": True,
        "formal_execution_lock_revalidated_after_child": True,
        "child_argv_tail": child_argv_tail,
        "execution": {
            "attempted": True,
            "argv": [recorded_python_path, *child_argv_tail],
            "working_directory": str(root),
            "environment_overrides": {
                "SLM_WM_FORMAL_EXECUTION_COMMIT": formal_commit,
                "SLM_WM_FORMAL_EXECUTION_LOCK_DIGEST": formal_digest,
                "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH": str(
                    source_dependency_path
                ),
                "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST": (
                    _file_sha256(dependency_path)
                ),
            },
            "return_code": 0,
            "stdout": "",
            "stderr": "",
        },
        "execution_report_path": execution_path.name,
        "execution_completed": True,
        "decision": "pass",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    _write_json(execution_path, execution)
    dispatch = {
        "report_schema": "scientific_command_dispatch_report",
        "schema_version": 1,
        "paper_run_name": paper_run_name,
        "decision": "pass",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    if execution_route in {
        "semantic_watermark_session",
        "semantic_watermark_ablation_session",
    }:
        command_specs = [
            (
                "image_only_dataset_runtime",
                [
                    recorded_python_path,
                    "-m",
                    "experiments.runners.image_only_dataset_workload",
                ],
            )
        ]
        if execution_route == "semantic_watermark_ablation_session":
            command_specs.extend(
                (
                    (
                        "runtime_rerun_ablation",
                        [
                            recorded_python_path,
                            "-m",
                            "experiments.ablations.mechanism_ablation_workload",
                        ],
                    ),
                    (
                        "branch_risk_parameter_sensitivity",
                        [
                            recorded_python_path,
                            "-m",
                            "experiments.ablations.branch_risk_sensitivity_workload",
                        ],
                    ),
                )
            )
        artifact_specs = {
            "image_only_dataset_runtime": (
                f"outputs/image_only_dataset_runtime/{paper_run_name}/dataset_runtime_summary.json",
                f"outputs/image_only_dataset_runtime/{paper_run_name}/manifest.local.json",
            ),
            "dataset_level_quality": (
                f"outputs/dataset_level_quality/{paper_run_name}/dataset_quality_summary.json",
                f"outputs/dataset_level_quality/{paper_run_name}/manifest.local.json",
            ),
            "runtime_rerun_ablation": (
                f"outputs/formal_mechanism_ablation/{paper_run_name}/ablation_component_summary.json",
                f"outputs/formal_mechanism_ablation/{paper_run_name}/manifest.local.json",
            ),
            "branch_risk_parameter_sensitivity": (
                f"outputs/formal_branch_risk_sensitivity/{paper_run_name}/parameter_sensitivity_summary.json",
                f"outputs/formal_branch_risk_sensitivity/{paper_run_name}/manifest.local.json",
            ),
        }
        artifact_roles = [
            "image_only_dataset_runtime",
            "dataset_level_quality",
        ]
        if execution_route == "semantic_watermark_ablation_session":
            artifact_roles.extend(
                (
                    "runtime_rerun_ablation",
                    "branch_risk_parameter_sensitivity",
                )
            )
        artifact_records = []
        for role in artifact_roles:
            summary_relative, manifest_relative = artifact_specs[role]
            is_bound_role = role == artifact_role
            artifact_records.append(
                {
                    "artifact_role": role,
                    "summary_path": summary_relative,
                    "summary_sha256": (
                        _file_sha256(summary_path) if is_bound_role else "7" * 64
                    ),
                    "manifest_path": manifest_relative,
                    "manifest_sha256_at_session": (
                        _file_sha256(manifest_path) if is_bound_role else "8" * 64
                    ),
                    "manifest_scientific_digest": (
                        scientific_manifest_payload_digest(
                            json.loads(manifest_path.read_text(encoding="utf-8"))
                        )
                        if is_bound_role
                        else "9" * 64
                    ),
                    "formal_execution_run_lock": dict(formal_execution_lock),
                    "summary_protocol_decision": "pass",
                }
            )
        dispatch.update(
            {
                "formal_ablation_requested": (
                    execution_route == "semantic_watermark_ablation_session"
                ),
                "packaging_deferred": True,
                "session_execution_decision": "pass",
                "workflow_completion_state": "repeat_component_complete",
                "paper_run_closed": False,
                "result_closure_ready": False,
                "python_executable": recorded_python_path,
                "artifact_validation_mode": (
                    "completed_or_revalidated_in_current_session"
                ),
                "artifact_records": artifact_records,
                "commands": [
                    {
                        "command_role": role,
                        "argv": argv,
                        "return_code": 0,
                        "packaging_deferred": True,
                    }
                    for role, argv in command_specs
                ],
            }
        )
    elif execution_route in {
        "isolated_method_faithful_workflow",
        "isolated_t2smark_workflow",
    }:
        dispatch.update(
            {
                "operation_kind": "isolated_scientific_workflow",
                "workflow_name": workflow_name,
                "profile_id": profile_id,
                "formal_execution_lock": dict(formal_execution_lock),
                "formal_execution_commit": formal_commit,
                "formal_execution_lock_digest": formal_digest,
                "summary_path": summary_path.relative_to(root).as_posix(),
                "summary_sha256": _file_sha256(summary_path),
                "manifest_path": manifest_path.relative_to(root).as_posix(),
                "manifest_sha256": _file_sha256(manifest_path),
                "dependency_environment_report_digest": _file_sha256(
                    dependency_path
                ),
                "baseline_id": baseline_id,
            }
        )
    _write_json(dispatch_path, dispatch)
    _write_json(envelope_path, dispatch)

    def relative(path: Path) -> str:
        """把测试证据路径转换为仓库相对路径."""

        return path.relative_to(root).as_posix()

    binding = {
        "report_schema": "scientific_execution_binding",
        "schema_version": 2,
        "artifact_role": artifact_role,
        "paper_run_name": paper_run_name,
        "profile_id": profile_id,
        "profile_digest": PROFILE_DIGEST,
        "direct_requirements_digest": DIRECT_REQUIREMENTS_DIGEST,
        "complete_hash_lock_digest": COMPLETE_HASH_LOCK_DIGEST,
        "scientific_execution_report_path": relative(execution_path),
        "scientific_execution_report_digest": _file_sha256(execution_path),
        "dependency_environment_report_path": relative(dependency_path),
        "dependency_environment_report_digest": _file_sha256(dependency_path),
        "scientific_command_dispatch_report_path": relative(dispatch_path),
        "scientific_command_dispatch_report_digest": _file_sha256(dispatch_path),
        "bound_summary_path": relative(summary_path),
        "bound_summary_digest": _file_sha256(summary_path),
        "bound_manifest_path": relative(manifest_path),
        "bound_manifest_scientific_digest": scientific_manifest_payload_digest(
            json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        ),
        "bound_manifest_digest_scope": BOUND_MANIFEST_DIGEST_SCOPE,
        "formal_execution_lock": dict(formal_execution_lock),
        "formal_execution_commit": formal_commit,
        "formal_execution_lock_digest": formal_digest,
        "decision": "pass",
        "supports_paper_claim": False,
    }
    _write_json(binding_path, binding)
