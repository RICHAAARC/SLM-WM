from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from zipfile import ZipFile

import pytest

from paper_experiments.runners import closure_package_selection as selection_module
from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    LOCK_FILENAME,
    LOCK_MANIFEST_FILENAME,
    LOCK_OUTPUT_ROOT,
    ClosurePackageFamilySpec,
    ClosurePackageSelectionError,
    JsonFieldSource,
    build_closure_input_selection_report,
    inspect_closure_package,
    select_and_lock_closure_input_packages,
)
from experiments.runtime.scientific_execution_binding import (
    BOUND_MANIFEST_DIGEST_SCOPE,
    scientific_manifest_payload_digest,
)
from tests.helpers.formal_execution_lock import build_test_formal_execution_lock


pytestmark = pytest.mark.quick


PAPER_RUN_NAME = "probe_paper"
TARGET_FPR = 0.1
CODE_VERSION = "a" * 40
GENERATED_AT = "2026-07-11T08:00:00+00:00"


@pytest.fixture(autouse=True)
def _ready_repository_dependency_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """把选择器锚定到与归档 fixture 相同的正式 profile 身份."""

    def ready_profile(profile_id: str, registry_path: Path) -> SimpleNamespace:
        assert registry_path.name == "dependency_profile_registry.json"
        return SimpleNamespace(
            profile_name=profile_id,
            profile_digest="3" * 64,
            direct_requirements_digest="4" * 64,
            complete_hash_lock_digest="5" * 64,
            complete_hash_lock_dependency_count=7,
            complete_hash_lock_present=True,
            formal_ready=True,
            readiness_blockers=(),
        )

    monkeypatch.setattr(
        selection_module,
        "require_dependency_profile_ready",
        ready_profile,
    )
    monkeypatch.setattr(
        selection_module,
        "resolve_code_version",
        lambda root: CODE_VERSION,
    )


def _render(template: str, spec: ClosurePackageFamilySpec, paper_run_name: str) -> str:
    return template.format(
        paper_run=paper_run_name,
        baseline=spec.baseline_id or "",
    )


def _assign(payload: dict[str, Any], field_path: tuple[str, ...], value: Any) -> None:
    current = payload
    for field_name in field_path[:-1]:
        nested = current.setdefault(field_name, {})
        assert isinstance(nested, dict)
        current = nested
    current[field_path[-1]] = value


def _assign_source(
    documents: dict[str, dict[str, Any]],
    spec: ClosurePackageFamilySpec,
    source: JsonFieldSource,
    value: Any,
    *,
    paper_run_name: str,
) -> None:
    member_name = _render(source.member_template, spec, paper_run_name)
    payload = documents.setdefault(member_name, {})
    _assign(payload, source.field_path, value)


def _assign_execution_locks(
    documents: dict[str, dict[str, Any]],
    spec: ClosurePackageFamilySpec,
    commit: str,
    *,
    paper_run_name: str,
) -> None:
    """让测试 manifest 的运行锁和打包锁共同绑定指定 commit."""

    manifest_name = _render(
        spec.manifest_member_template,
        spec,
        paper_run_name,
    )
    manifest = documents.setdefault(manifest_name, {})
    manifest["formal_execution_run_lock"] = build_test_formal_execution_lock(commit)
    manifest["formal_execution_package_lock"] = build_test_formal_execution_lock(commit)


def _dependency_environment_document(
    *,
    profile_id: str,
    profile_digest: str,
    direct_requirements_digest: str,
    complete_hash_lock_digest: str,
    python_digest: str,
    formal_lock: dict[str, Any],
    working_directory: str,
    python_path: str,
    report_file_digest: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    """构造包含完整真实 argv 的隔离依赖环境 fixture."""

    complete_hash_lock_path = "configs/dependency_profiles/test_complete.lock"
    complete_hash_lock_absolute_path = str(
        (Path(working_directory) / complete_hash_lock_path).resolve()
    )
    pytorch_index_url = "https://download.pytorch.org/whl/cu128"
    managed_python_root = str(
        (Path(working_directory) / "outputs/test_managed_python").resolve()
    )
    isolated_environment_path = str(
        (Path(working_directory) / "outputs/test_isolated_environment").resolve()
    )
    uv_executable_path = str(
        (Path(working_directory) / "outputs/test_uv/uv").resolve()
    )
    command_environment = {"UV_PYTHON_INSTALL_DIR": managed_python_root}

    def command_record(operation: str, argv: list[str]) -> dict[str, Any]:
        """构造一条完整依赖命令记录."""

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
        command_record("python_ensurepip", [python_path, "-m", "ensurepip"]),
        command_record(
            "python_patch_inspection",
            [
                python_path,
                "-c",
                "import platform; print(platform.python_version())",
            ],
        ),
    ]
    dependency_preparation_command = command_record(
        "dependency_profile_preparation",
        [
            python_path,
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
        "python_executable": python_path,
        "working_directory": working_directory,
        "profile_digest": profile_digest,
        "direct_requirements_digest": direct_requirements_digest,
        "complete_hash_lock_path": complete_hash_lock_path,
        "complete_hash_lock_digest": complete_hash_lock_digest,
        "complete_hash_lock_dependency_count": 7,
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
                python_path,
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
            "command": [python_path, "-m", "pip", "check"],
            "working_directory": working_directory,
            "return_code": 0,
            "decision": "pass",
        },
        "runtime_comparison": {
            "decision": "pass",
            "profile_digest": profile_digest,
            "complete_hash_lock_digest": complete_hash_lock_digest,
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
        "profile_digest": profile_digest,
        "python_version": "3.11.0",
        "managed_python_root": managed_python_root,
        "working_directory": working_directory,
        "isolated_environment_path": isolated_environment_path,
        "uv_executable_path": uv_executable_path,
        "python_executable_path": python_path,
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
    return {
        "report_schema": "isolated_dependency_environment_preparation_report",
        "schema_version": 1,
        "operation_kind": "formal_dependency_environment_preparation",
        "profile_id": profile_id,
        "profile_digest": profile_digest,
        "direct_requirements_digest": direct_requirements_digest,
        "complete_hash_lock_digest": complete_hash_lock_digest,
        "complete_hash_lock_dependency_count": 7,
        "working_directory": working_directory,
        "python_executable_path": python_path,
        "python_executable_sha256": python_digest,
        "python_executable_sha256_after_preparation": python_digest,
        "dependency_preparation_report": dependency_preparation,
        "dependency_preparation_report_digest": report_file_digest(
            dependency_preparation
        ),
        "provision_report": provision_report,
        "provision_report_digest": report_file_digest(provision_report),
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
    }


def _official_reference_command_config(
    baseline_id: str,
    profile_id: str,
) -> dict[str, Any]:
    """构造3个官方入口可重建精确 argv 的运行配置."""

    common = {
        "output_dir": f"outputs/{baseline_id}_official_reference",
        "source_dir": f"external_baseline/primary/{baseline_id}/source",
        "sample_count": 34,
        "official_model_id": "/models/stable-diffusion-2-1-base",
        "reference_model": "ViT-g-14",
        "reference_model_checkpoint_path": "/models/openclip.bin",
        "dependency_profile_id": profile_id,
        "require_cuda": True,
    }
    if baseline_id == "tree_ring":
        common.update(
            {
                "run_name": "tree_ring_official_reference",
                "dataset": "Gustavosta/Stable-Diffusion-Prompts",
                "start_index": 0,
            }
        )
    elif baseline_id == "gaussian_shading":
        common.update(
            {
                "official_output_subdir": "official_outputs",
                "fpr": 0.000001,
                "channel_copy": 1,
                "hw_copy": 8,
                "user_number": 1000000,
                "gen_seed": 0,
                "image_length": 512,
                "guidance_scale": 7.5,
                "num_inference_steps": 50,
                "num_inversion_steps": 50,
                "dataset_path": "Gustavosta/Stable-Diffusion-Prompts",
                "use_chacha": True,
            }
        )
    elif baseline_id == "shallow_diffuse":
        common.update(
            {
                "run_name": "shallow_diffuse_official_reference",
                "dataset": "Gustavosta/Stable-Diffusion-Prompts",
                "start_index": 0,
                "image_length": 512,
                "guidance_scale": 7.5,
                "num_inference_steps": 50,
                "w_seed": 42,
                "w_channel": 3,
                "w_pattern": "complex2_ring",
                "w_mask_shape": "circle",
                "w_radius": 10,
                "w_measurement": "l1_complex2",
                "w_injection": "complex2",
                "edit_time_list": "0.3,0.5,0.7",
            }
        )
    else:
        raise AssertionError("测试官方 baseline 身份无效")
    return common


def _valid_member_payloads(
    spec: ClosurePackageFamilySpec,
    *,
    paper_run_name: str,
    target_fpr: float,
    generated_at: str,
    scientific_python_digest: str = "6" * 64,
    mutate: Callable[[dict[str, dict[str, Any]], ClosurePackageFamilySpec], None] | None = None,
) -> dict[str, bytes]:
    required_members = {
        _render(template, spec, paper_run_name)
        for template in spec.required_member_templates
    }
    documents: dict[str, dict[str, Any]] = {
        member_name: {}
        for member_name in required_members
        if member_name.endswith(".json")
    }
    manifest_member = _render(spec.manifest_member_template, spec, paper_run_name)
    manifest = documents.setdefault(manifest_member, {})
    manifest["artifact_id"] = _render(
        spec.manifest_artifact_id_template,
        spec,
        paper_run_name,
    )
    manifest["artifact_type"] = "local_manifest"
    _assign_execution_locks(
        documents,
        spec,
        CODE_VERSION,
        paper_run_name=paper_run_name,
    )
    for source in spec.paper_run_sources:
        _assign_source(
            documents,
            spec,
            source,
            paper_run_name,
            paper_run_name=paper_run_name,
        )
    for source in spec.target_fpr_sources:
        _assign_source(
            documents,
            spec,
            source,
            target_fpr,
            paper_run_name=paper_run_name,
        )
    for source in spec.baseline_sources:
        _assign_source(
            documents,
            spec,
            source,
            spec.baseline_id,
            paper_run_name=paper_run_name,
        )
    for source in spec.code_version_sources:
        _assign_source(
            documents,
            spec,
            source,
            CODE_VERSION,
            paper_run_name=paper_run_name,
        )
    _assign_source(
        documents,
        spec,
        spec.generated_at_source,
        generated_at,
        paper_run_name=paper_run_name,
    )
    for requirement in spec.value_requirements:
        _assign_source(
            documents,
            spec,
            requirement.source,
            requirement.expected_value,
            paper_run_name=paper_run_name,
        )
    if mutate is not None:
        mutate(documents, spec)

    scientific_member_payloads: dict[str, bytes] = {}

    def serialized(document: dict[str, Any]) -> bytes:
        """使用与 fixture ZIP 相同的稳定编码计算成员摘要."""

        return (
            json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")

    def report_file_digest(document: dict[str, Any]) -> str:
        """按正式 runtime 报告排版计算文件摘要."""

        payload = (
            json.dumps(
                document,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    scientific_contract = spec.scientific_execution_binding
    if scientific_contract is not None:
        binding_member = _render(
            scientific_contract.binding_member_template,
            spec,
            paper_run_name,
        )
        execution_member = _render(
            scientific_contract.execution_report_member_template,
            spec,
            paper_run_name,
        )
        dependency_member = _render(
            scientific_contract.dependency_report_member_template,
            spec,
            paper_run_name,
        )
        dispatch_member = _render(
            scientific_contract.dispatch_report_member_template,
            spec,
            paper_run_name,
        )
        summary_member = _render(
            scientific_contract.summary_member_template,
            spec,
            paper_run_name,
        )
        bound_manifest_member = _render(
            scientific_contract.manifest_member_template,
            spec,
            paper_run_name,
        )
        formal_lock = documents[bound_manifest_member].get(
            "formal_execution_run_lock"
        )
        if formal_lock is None:
            formal_lock = documents[manifest_member].get(
                "formal_execution_run_lock"
            )
        if formal_lock is None:
            formal_lock = build_test_formal_execution_lock(CODE_VERSION)
        if bound_manifest_member != manifest_member:
            documents[bound_manifest_member][
                "formal_execution_run_lock"
            ] = formal_lock
            documents[bound_manifest_member]["code_version"] = formal_lock[
                "formal_execution_commit"
            ]
        profile_digest = "3" * 64
        direct_requirements_digest = "4" * 64
        complete_hash_lock_digest = "5" * 64
        python_digest = scientific_python_digest
        working_directory = str(Path.cwd().resolve())
        python_path = str(
            (Path.cwd() / "outputs" / "test_scientific_python").resolve()
        )
        dependency_document = _dependency_environment_document(
            profile_id=scientific_contract.profile_id,
            profile_digest=profile_digest,
            direct_requirements_digest=direct_requirements_digest,
            complete_hash_lock_digest=complete_hash_lock_digest,
            python_digest=python_digest,
            formal_lock=formal_lock,
            working_directory=working_directory,
            python_path=python_path,
            report_file_digest=report_file_digest,
        )
        dependency_source_path = str(
            (
                Path(working_directory)
                / "outputs"
                / "dependency_profiles"
                / scientific_contract.profile_id
                / "isolated_dependency_environment_report.json"
            ).resolve()
        )
        if scientific_contract.execution_route in {
            "semantic_watermark_session",
            "semantic_watermark_ablation_session",
        }:
            child_argv_tail = [
                "-m",
                "experiments.runtime.semantic_watermark_scientific_session",
                "--run-formal-ablation",
            ]
        else:
            workflow_name = (
                "external_baseline_method_faithful"
                if scientific_contract.execution_route
                == "isolated_method_faithful_workflow"
                else "official_reference_t2smark"
            )
            envelope_member = (
                Path(working_directory)
                / (
                    "outputs/external_baseline_method_faithful/"
                    f"{paper_run_name}/run_records/{spec.baseline_id}/"
                    "scientific_execution/scientific_workflow_result_envelope.json"
                    if workflow_name == "external_baseline_method_faithful"
                    else (
                        f"outputs/t2smark_formal_reproduction/{paper_run_name}/"
                        "scientific_execution/scientific_workflow_result_envelope.json"
                    )
                )
            ).resolve()
            child_argv_tail = [
                "-m",
                "paper_experiments.runners.isolated_scientific_workflow",
                "--child-workflow",
                workflow_name,
                "--root",
                working_directory,
                "--result-envelope",
                str(envelope_member),
            ]
        execution_document = {
            "report_schema": "isolated_scientific_execution_report",
            "schema_version": 1,
            "operation_kind": "isolated_scientific_execution",
            "repository_root": working_directory,
            "profile_id": scientific_contract.profile_id,
            "profile_digest": profile_digest,
            "direct_requirements_digest": direct_requirements_digest,
            "complete_hash_lock_digest": complete_hash_lock_digest,
            "complete_hash_lock_dependency_count": 7,
            "dependency_environment_report_path": Path(dependency_member).name,
            "source_dependency_environment_report_path": dependency_source_path,
            "dependency_environment_report_digest": hashlib.sha256(
                serialized(dependency_document)
            ).hexdigest(),
            "dependency_environment_report_valid": True,
            "python_executable_path": python_path,
            "python_executable_sha256": python_digest,
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
            "child_argv_tail": child_argv_tail,
            "execution": {
                "attempted": True,
                "argv": [python_path, *child_argv_tail],
                "working_directory": working_directory,
                "environment_overrides": {
                    "SLM_WM_FORMAL_EXECUTION_COMMIT": formal_lock[
                        "formal_execution_commit"
                    ],
                    "SLM_WM_FORMAL_EXECUTION_LOCK_DIGEST": formal_lock[
                        "formal_execution_lock_digest"
                    ],
                    "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH": (
                        dependency_source_path
                    ),
                    "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST": (
                        hashlib.sha256(serialized(dependency_document)).hexdigest()
                    ),
                },
                "return_code": 0,
            },
            "execution_report_path": Path(execution_member).name,
            "execution_completed": True,
            "decision": "pass",
            "failure_reasons": [],
            "supports_paper_claim": False,
        }
        dispatch_document = {
            "report_schema": "scientific_command_dispatch_report",
            "schema_version": 1,
            "paper_run_name": paper_run_name,
            "decision": "pass",
            "failure_reasons": [],
            "supports_paper_claim": False,
        }
        if scientific_contract.execution_route in {
            "semantic_watermark_session",
            "semantic_watermark_ablation_session",
        }:
            command_specs = [
                (
                    "image_only_dataset_runtime",
                    [
                        python_path,
                        "-m",
                        "experiments.runners.image_only_dataset_workload",
                    ],
                )
            ]
            include_ablation_session = (
                child_argv_tail[-1] == "--run-formal-ablation"
            )
            if include_ablation_session:
                command_specs.append(
                    (
                        "runtime_rerun_ablation",
                        [
                            python_path,
                            "-m",
                            "experiments.ablations.mechanism_ablation_workload",
                        ],
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
                    f"outputs/formal_mechanism_ablation/{paper_run_name}/ablation_claim_summary.json",
                    f"outputs/formal_mechanism_ablation/{paper_run_name}/manifest.local.json",
                ),
            }
            artifact_roles = [
                "image_only_dataset_runtime",
                "dataset_level_quality",
            ]
            if include_ablation_session:
                artifact_roles.append("runtime_rerun_ablation")
            artifact_records = []
            for role in artifact_roles:
                summary_path, manifest_path = artifact_specs[role]
                is_bound_role = role == scientific_contract.artifact_role
                artifact_records.append(
                    {
                        "artifact_role": role,
                        "summary_path": summary_path,
                        "summary_sha256": (
                            hashlib.sha256(
                                serialized(documents[summary_member])
                            ).hexdigest()
                            if is_bound_role
                            else "7" * 64
                        ),
                        "manifest_path": manifest_path,
                        "manifest_sha256_at_session": (
                            hashlib.sha256(
                                serialized(documents[bound_manifest_member])
                            ).hexdigest()
                            if is_bound_role
                            else "8" * 64
                        ),
                        "manifest_scientific_digest": (
                            scientific_manifest_payload_digest(
                                documents[bound_manifest_member]
                            )
                            if is_bound_role
                            else "9" * 64
                        ),
                        "formal_execution_run_lock": formal_lock,
                        "summary_protocol_decision": "pass",
                    }
                )
            dispatch_document.update(
                {
                    "formal_ablation_requested": (
                        include_ablation_session
                    ),
                    "packaging_deferred": True,
                    "python_executable": python_path,
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
        else:
            dispatch_document.update(
                {
                    "operation_kind": "isolated_scientific_workflow",
                    "workflow_name": workflow_name,
                    "profile_id": scientific_contract.profile_id,
                    "formal_execution_lock": formal_lock,
                    "formal_execution_commit": formal_lock[
                        "formal_execution_commit"
                    ],
                    "formal_execution_lock_digest": formal_lock[
                        "formal_execution_lock_digest"
                    ],
                    "summary_path": summary_member,
                    "summary_sha256": hashlib.sha256(
                        serialized(documents[summary_member])
                    ).hexdigest(),
                    "manifest_path": bound_manifest_member,
                    "manifest_sha256": hashlib.sha256(
                        serialized(documents[bound_manifest_member])
                    ).hexdigest(),
                    "dependency_environment_report_digest": hashlib.sha256(
                        serialized(dependency_document)
                    ).hexdigest(),
                }
            )
        binding_document = {
            "report_schema": "scientific_execution_binding",
            "schema_version": 2,
            "artifact_role": scientific_contract.artifact_role,
            "paper_run_name": paper_run_name,
            "profile_id": scientific_contract.profile_id,
            "profile_digest": profile_digest,
            "direct_requirements_digest": direct_requirements_digest,
            "complete_hash_lock_digest": complete_hash_lock_digest,
            "scientific_execution_report_path": execution_member,
            "scientific_execution_report_digest": hashlib.sha256(
                serialized(execution_document)
            ).hexdigest(),
            "dependency_environment_report_path": dependency_member,
            "dependency_environment_report_digest": hashlib.sha256(
                serialized(dependency_document)
            ).hexdigest(),
            "scientific_command_dispatch_report_path": dispatch_member,
            "scientific_command_dispatch_report_digest": hashlib.sha256(
                serialized(dispatch_document)
            ).hexdigest(),
            "bound_summary_path": summary_member,
            "bound_summary_digest": hashlib.sha256(
                serialized(documents[summary_member])
            ).hexdigest(),
            "bound_manifest_path": bound_manifest_member,
            "bound_manifest_scientific_digest": scientific_manifest_payload_digest(
                documents[bound_manifest_member]
            ),
            "bound_manifest_digest_scope": BOUND_MANIFEST_DIGEST_SCOPE,
            "formal_execution_lock": formal_lock,
            "formal_execution_commit": formal_lock["formal_execution_commit"],
            "formal_execution_lock_digest": formal_lock[
                "formal_execution_lock_digest"
            ],
            "decision": "pass",
            "supports_paper_claim": False,
        }
        scientific_member_payloads = {
            dependency_member: serialized(dependency_document),
            execution_member: serialized(execution_document),
            dispatch_member: serialized(dispatch_document),
            binding_member: serialized(binding_document),
        }
    dependency_contract = spec.dependency_environment_evidence
    if dependency_contract is not None:
        report_member = _render(
            dependency_contract.report_member_template,
            spec,
            paper_run_name,
        )
        formal_lock = documents[manifest_member]["formal_execution_run_lock"]
        profile_digest = "3" * 64
        direct_requirements_digest = "4" * 64
        complete_hash_lock_digest = "5" * 64
        python_digest = "6" * 64
        working_directory = str(Path.cwd().resolve())
        python_path = str(
            (Path.cwd() / "outputs" / "test_scientific_python").resolve()
        )
        embedded_report = _dependency_environment_document(
            profile_id=dependency_contract.profile_id,
            profile_digest=profile_digest,
            direct_requirements_digest=direct_requirements_digest,
            complete_hash_lock_digest=complete_hash_lock_digest,
            python_digest=python_digest,
            formal_lock=formal_lock,
            working_directory=working_directory,
            python_path=python_path,
            report_file_digest=report_file_digest,
        )
        outer_report = {
            "dependency_environment_requested": True,
            "dependency_environment_ready": True,
            "dependency_environment_materialized": True,
            "dependency_environment_profile_id": dependency_contract.profile_id,
            "dependency_python_executable": python_path,
            "dependency_profile_id": dependency_contract.profile_id,
            "dependency_profile_ready": True,
            "dependency_lock_ready": True,
            "dependency_environment_report_valid": True,
            "dependency_installation_performed": True,
            "dependency_environment_failure_reason": "",
            "dependency_profile_digest": profile_digest,
            "dependency_lock_digest": complete_hash_lock_digest,
            "isolated_dependency_environment_report_digest": report_file_digest(
                embedded_report
            ),
            "isolated_dependency_environment_report": embedded_report,
        }
        scientific_member_payloads[report_member] = serialized(outer_report)
        run_manifest_member = _render(
            dependency_contract.run_manifest_member_template,
            spec,
            paper_run_name,
        )
        run_manifest = documents[run_manifest_member]
        run_config = run_manifest.setdefault(
            "config",
            _official_reference_command_config(
                str(spec.baseline_id),
                dependency_contract.profile_id,
            ),
        )
        assert isinstance(run_config, dict)
        normalized_repository_root = (
            selection_module.normalize_scientific_absolute_path(
                working_directory
            )
        )
        official_command, _, official_working_directory = (
            selection_module._expected_official_reference_command(
                baseline_id=str(spec.baseline_id),
                config=run_config,
                repository_root=normalized_repository_root,
                paper_run_name=paper_run_name,
                python_executable=python_path,
            )
        )
        cuda_payload = {
            "python_executable": python_path,
            "torch_available": True,
            "cuda_available": True,
            "device": "cuda",
            "torch_version": "2.7.1+cu128",
            "torch_cuda_version": "12.8",
            "device_count": 1,
            "gpu_name": "Test GPU",
        }
        device_report = {
            "command": [
                python_path,
                "-c",
                selection_module.CUDA_INSPECTION_PROGRAM,
                "1",
            ],
            "working_directory": working_directory,
            "return_code": 0,
            "stdout": json.dumps(
                cuda_payload,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            "stderr": "",
            "python_executable": python_path,
            "python_executable_sha256": python_digest,
            **{key: value for key, value in cuda_payload.items() if key != "python_executable"},
            "decision": "pass",
            "failure_reasons": [],
            "supports_paper_claim": False,
        }
        command_result = {
            "report_schema": "official_reference_command_execution_report",
            "schema_version": 1,
            "baseline_id": spec.baseline_id,
            "official_command_requested": True,
            "official_command": official_command,
            "official_command_working_directory": official_working_directory,
            "dependency_python_executable": python_path,
            "dependency_python_executable_sha256": python_digest,
            "cuda_inspection_report_digest": (
                selection_module.stable_json_payload_digest(device_report)
            ),
            "official_command_execution_evidence_ready": True,
            "return_code": 0,
            "failure_reasons": [],
            "supports_paper_claim": False,
        }
        environment_report = {
            f"{spec.baseline_id}_official_reference_device_report": device_report,
            f"{spec.baseline_id}_official_reference_dependency_environment_report": (
                outer_report
            ),
        }
        command_result_member = _render(
            dependency_contract.command_result_member_template,
            spec,
            paper_run_name,
        )
        environment_report_member = _render(
            dependency_contract.environment_report_member_template,
            spec,
            paper_run_name,
        )
        scientific_member_payloads[command_result_member] = serialized(
            command_result
        )
        scientific_member_payloads[environment_report_member] = serialized(
            environment_report
        )

    member_payloads: dict[str, bytes] = {}
    baseline_rows_members = {
        _render(source.member_template, spec, paper_run_name)
        for source in spec.baseline_rows_sources
    }
    for member_name in required_members:
        if member_name in scientific_member_payloads:
            member_payloads[member_name] = scientific_member_payloads[member_name]
        elif member_name in baseline_rows_members:
            row = {"baseline_id": spec.baseline_id}
            member_payloads[member_name] = (
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8")
        elif member_name in documents:
            member_payloads[member_name] = (
                json.dumps(documents[member_name], ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8")
        elif member_name.endswith(".jsonl"):
            member_payloads[member_name] = b'{"record_id":"fixture"}\n'
        elif member_name.endswith(".csv"):
            member_payloads[member_name] = b"metric_name,metric_value\nfixture,1\n"
        else:
            member_payloads[member_name] = b"fixture\n"
    package_input_template = spec.package_input_manifest_template
    if package_input_template is not None:
        package_input_member = _render(
            package_input_template,
            spec,
            paper_run_name,
        )
        package_input = json.loads(
            member_payloads[package_input_member].decode("utf-8")
        )
        declared_members = sorted(
            member_name
            for member_name in member_payloads
            if member_name != package_input_member
        )
        package_input["entry_count"] = len(declared_members)
        package_input["entry_paths"] = declared_members
        package_input["entry_sha256"] = {
            member_name: hashlib.sha256(member_payloads[member_name]).hexdigest()
            for member_name in declared_members
        }
        package_manifest = json.loads(
            member_payloads[manifest_member].decode("utf-8")
        )
        for lock_field in (
            "formal_execution_run_lock",
            "formal_execution_package_lock",
        ):
            if lock_field in package_manifest:
                package_input[lock_field] = package_manifest[lock_field]
        member_payloads[package_input_member] = (
            json.dumps(package_input, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
    return member_payloads


def _write_family_package(
    package_root: Path,
    spec: ClosurePackageFamilySpec,
    *,
    token: str,
    generated_at: str = GENERATED_AT,
    scientific_python_digest: str = "6" * 64,
    mutate: Callable[[dict[str, dict[str, Any]], ClosurePackageFamilySpec], None] | None = None,
    extra_members: dict[str, bytes] | None = None,
) -> Path:
    package_path = package_root / spec.filename_pattern.replace("*", token)
    package_path.parent.mkdir(parents=True, exist_ok=True)
    members = _valid_member_payloads(
        spec,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        generated_at=generated_at,
        scientific_python_digest=scientific_python_digest,
        mutate=mutate,
    )
    members.update(extra_members or {})
    with ZipFile(package_path, "w") as archive:
        for member_name, payload in sorted(members.items()):
            archive.writestr(member_name, payload)
    return package_path


def _rewrite_package_json_members(
    package_path: Path,
    mutate: Callable[[dict[str, dict[str, Any]]], None],
) -> None:
    """重写指定 JSON 成员, 用于构造摘要自洽的恶意官方证据."""

    with ZipFile(package_path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    json_members = {
        name: json.loads(payload.decode("utf-8-sig"))
        for name, payload in members.items()
        if name.endswith(".json")
    }
    mutate(json_members)
    for name, document in json_members.items():
        members[name] = (
            json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
    for name, document in json_members.items():
        declared_paths = document.get("entry_paths")
        if not isinstance(declared_paths, list) or not isinstance(
            document.get("entry_sha256"),
            dict,
        ):
            continue
        document["entry_sha256"] = {
            member_name: hashlib.sha256(members[member_name]).hexdigest()
            for member_name in declared_paths
        }
        members[name] = (
            json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
    with ZipFile(package_path, "w") as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(name, payload)


def _rebind_package_input_member_digests(
    members: dict[str, bytes],
    spec: ClosurePackageFamilySpec,
) -> None:
    """重算自洽篡改测试的 package-input 精确成员摘要."""

    template = spec.package_input_manifest_template
    assert template is not None
    package_input_member = _render(template, spec, PAPER_RUN_NAME)
    package_input = json.loads(members[package_input_member].decode("utf-8"))
    declared_paths = package_input["entry_paths"]
    assert isinstance(declared_paths, list)
    package_input["entry_count"] = len(declared_paths)
    package_input["entry_sha256"] = {
        member_name: hashlib.sha256(members[member_name]).hexdigest()
        for member_name in declared_paths
    }
    members[package_input_member] = (
        json.dumps(package_input, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_all_family_packages(package_root: Path, *, token: str = "current") -> list[Path]:
    return [
        _write_family_package(package_root, spec, token=token)
        for spec in CLOSURE_PACKAGE_FAMILY_SPECS
    ]


@pytest.mark.parametrize(
    "spec",
    CLOSURE_PACKAGE_FAMILY_SPECS[6:9],
    ids=lambda spec: spec.package_family,
)
@pytest.mark.parametrize(
    "tamper_kind",
    ("argv", "cwd", "python", "cuda_program"),
)
def test_official_closure_rejects_command_and_cuda_identity_drift(
    tmp_path: Path,
    spec: ClosurePackageFamilySpec,
    tamper_kind: str,
) -> None:
    """3条官方参考路径必须精确绑定 argv, cwd, Python 与 CUDA 探针."""

    package_path = _write_family_package(tmp_path, spec, token=tamper_kind)
    contract = spec.dependency_environment_evidence
    assert contract is not None
    command_member = _render(
        contract.command_result_member_template,
        spec,
        PAPER_RUN_NAME,
    )
    environment_member = _render(
        contract.environment_report_member_template,
        spec,
        PAPER_RUN_NAME,
    )
    device_field = f"{spec.baseline_id}_official_reference_device_report"

    def mutate(documents: dict[str, dict[str, Any]]) -> None:
        command_result = documents[command_member]
        if tamper_kind == "argv":
            command_result["official_command"][2] = "--forged-option"
        elif tamper_kind == "cwd":
            command_result["official_command_working_directory"] = "/forged/cwd"
        elif tamper_kind == "python":
            command_result["official_command"][0] = "/forged/python"
            command_result["dependency_python_executable"] = "/forged/python"
        else:
            device_report = documents[environment_member][device_field]
            device_report["command"][2] = "print('forged cuda inspection')"
            command_result["cuda_inspection_report_digest"] = (
                selection_module.stable_json_payload_digest(device_report)
            )

    _rewrite_package_json_members(package_path, mutate)

    with pytest.raises(ClosurePackageSelectionError):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def _rebind_scientific_dependency_member(
    members: dict[str, bytes],
    spec: ClosurePackageFamilySpec,
    dependency: dict[str, Any],
) -> None:
    """重算依赖,执行和 binding 摘要, 构造自洽篡改 fixture."""

    contract = spec.scientific_execution_binding
    assert contract is not None
    dependency_member = _render(
        contract.dependency_report_member_template,
        spec,
        PAPER_RUN_NAME,
    )
    execution_member = _render(
        contract.execution_report_member_template,
        spec,
        PAPER_RUN_NAME,
    )
    binding_member = _render(
        contract.binding_member_template,
        spec,
        PAPER_RUN_NAME,
    )
    members[dependency_member] = (
        json.dumps(dependency, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    dependency_digest = hashlib.sha256(members[dependency_member]).hexdigest()
    execution = json.loads(members[execution_member].decode("utf-8"))
    execution["dependency_environment_report_digest"] = dependency_digest
    execution["execution"]["environment_overrides"][
        "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST"
    ] = dependency_digest
    members[execution_member] = (
        json.dumps(execution, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    binding = json.loads(members[binding_member].decode("utf-8"))
    binding["dependency_environment_report_digest"] = dependency_digest
    binding["scientific_execution_report_digest"] = hashlib.sha256(
        members[execution_member]
    ).hexdigest()
    members[binding_member] = (
        json.dumps(binding, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def test_dry_run_selects_exact_ten_families_without_mixing_unrelated_archives(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "drive"
    expected_paths = _write_all_family_packages(package_root)
    with ZipFile(package_root / "probe_paper_complete_result_package_prior.zip", "w") as archive:
        archive.writestr("outputs/paper_result/summary.json", "{}")
    with ZipFile(package_root / "unrelated_evidence.zip", "w") as archive:
        archive.writestr("outputs/unrelated/value.json", "{}")
    matching_directory = package_root / CLOSURE_PACKAGE_FAMILY_SPECS[0].filename_pattern.replace(
        "*",
        "directory",
    )
    matching_directory.mkdir()

    report = build_closure_input_selection_report(
        package_root,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        root=tmp_path,
    )

    assert report["closure_input_selection_ready"] is True
    assert report["closure_input_lock_written"] is False
    assert report["closure_input_package_count"] == 10
    assert len(report["selected_package_paths"]) == 10
    assert set(report["selected_package_paths"]) == {
        path.resolve().as_posix() for path in expected_paths
    }
    assert not (tmp_path / LOCK_OUTPUT_ROOT / PAPER_RUN_NAME).exists()


def test_formal_selection_writes_run_scoped_lock_and_independent_manifest(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "drive"
    _write_all_family_packages(package_root)

    selected_paths = select_and_lock_closure_input_packages(
        package_root,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        root=tmp_path,
    )

    output_dir = tmp_path / LOCK_OUTPUT_ROOT / PAPER_RUN_NAME
    lock_path = output_dir / LOCK_FILENAME
    manifest_path = output_dir / LOCK_MANIFEST_FILENAME
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(selected_paths) == 10
    assert lock_payload["closure_input_package_count"] == 10
    assert lock_payload["paper_run_name"] == PAPER_RUN_NAME
    assert lock_payload["target_fpr"] == TARGET_FPR
    assert lock_payload["common_code_version"] == CODE_VERSION
    assert [row["package_path"] for row in lock_payload["closure_input_packages"]] == list(
        selected_paths
    )
    assert all(len(row["package_sha256"]) == 64 for row in lock_payload["closure_input_packages"])
    assert all(
        len(row["formal_execution_run_lock_digest"]) == 64
        and len(row["formal_execution_package_lock_digest"]) == 64
        for row in lock_payload["closure_input_packages"]
    )
    assert lock_payload["formal_execution_run_lock_digests"] == {
        row["package_family"]: row["formal_execution_run_lock_digest"]
        for row in lock_payload["closure_input_packages"]
    }
    assert lock_payload["formal_execution_package_lock_digests"] == {
        row["package_family"]: row["formal_execution_package_lock_digest"]
        for row in lock_payload["closure_input_packages"]
    }
    scientific_rows = [
        row
        for row in lock_payload["closure_input_packages"]
        if row["scientific_profile_id"]
    ]
    assert len(scientific_rows) == 10
    assert all(
        len(row["scientific_profile_digest"]) == 64
        and len(row["scientific_direct_requirements_digest"]) == 64
        and len(row["scientific_complete_hash_lock_digest"]) == 64
        and row["scientific_complete_hash_lock_dependency_count"] > 0
        and len(row["scientific_python_executable_digest"]) == 64
        and len(row["scientific_dependency_evidence_digest"]) == 64
        for row in scientific_rows
    )
    assert sum(
        len(row["scientific_execution_binding_digest"]) == 64
        for row in scientific_rows
    ) == 7
    digest_payload = dict(lock_payload)
    stored_digest = digest_payload.pop("closure_input_lock_digest")
    canonical = json.dumps(
        digest_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert stored_digest == hashlib.sha256(canonical).hexdigest()
    assert manifest["artifact_id"] == f"{PAPER_RUN_NAME}_closure_input_lock_manifest"
    assert manifest["metadata"]["closure_input_lock_ready"] is True
    assert manifest["metadata"]["closure_input_package_count"] == 10
    assert manifest["metadata"]["closure_input_packages"] == lock_payload[
        "closure_input_packages"
    ]
    assert manifest["metadata"]["closure_input_lock_digest"] == stored_digest
    assert manifest["metadata"]["common_code_version"] == CODE_VERSION
    assert manifest["output_paths"] == [
        f"outputs/paper_result_closure/{PAPER_RUN_NAME}/{LOCK_FILENAME}",
        f"outputs/paper_result_closure/{PAPER_RUN_NAME}/{LOCK_MANIFEST_FILENAME}",
    ]


def test_selection_rejects_main_packages_from_different_scientific_sessions(
    tmp_path: Path,
) -> None:
    """主方法、质量和消融包不得来自三个自洽但不同的科学解释器会话."""

    package_root = tmp_path / "drive"
    distinct_digests = {
        "image_only_dataset_runtime": "6" * 64,
        "dataset_level_quality": "7" * 64,
        "runtime_rerun_ablation": "8" * 64,
    }
    for spec in CLOSURE_PACKAGE_FAMILY_SPECS:
        _write_family_package(
            package_root,
            spec,
            token="mixed_session",
            scientific_python_digest=distinct_digests.get(
                spec.package_family,
                "6" * 64,
            ),
        )

    with pytest.raises(
        ClosurePackageSelectionError,
        match="不属于同一科学会话",
    ):
        build_closure_input_selection_report(
            package_root,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
        )


def test_multiple_candidates_use_governed_generated_at_instead_of_path_name(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "drive"
    for spec in CLOSURE_PACKAGE_FAMILY_SPECS[1:]:
        _write_family_package(package_root, spec, token="current")
    selected_spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]
    _write_family_package(
        package_root,
        selected_spec,
        token="zzz",
        generated_at="2026-07-10T08:00:00+00:00",
    )
    latest_path = _write_family_package(
        package_root,
        selected_spec,
        token="aaa",
        generated_at="2026-07-12T08:00:00+00:00",
    )

    report = build_closure_input_selection_report(
        package_root,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        root=tmp_path,
    )

    runtime_record = next(
        row
        for row in report["closure_input_packages"]
        if row["package_family"] == selected_spec.package_family
    )
    assert runtime_record["package_path"] == latest_path.resolve().as_posix()
    assert runtime_record["generated_at"] == "2026-07-12T08:00:00+00:00"


def test_internal_identity_rejects_wrong_run_fpr_baseline_and_ready_flag(
    tmp_path: Path,
) -> None:
    cases: list[
        tuple[
            ClosurePackageFamilySpec,
            Callable[[dict[str, dict[str, Any]], ClosurePackageFamilySpec], None],
        ]
    ] = []

    runtime_spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]

    def wrong_run(documents: dict[str, dict[str, Any]], spec: ClosurePackageFamilySpec) -> None:
        _assign_source(
            documents,
            spec,
            spec.paper_run_sources[0],
            "pilot_paper",
            paper_run_name=PAPER_RUN_NAME,
        )

    cases.append((runtime_spec, wrong_run))
    ablation_spec = CLOSURE_PACKAGE_FAMILY_SPECS[1]

    def wrong_fpr(documents: dict[str, dict[str, Any]], spec: ClosurePackageFamilySpec) -> None:
        _assign_source(
            documents,
            spec,
            spec.target_fpr_sources[0],
            0.01,
            paper_run_name=PAPER_RUN_NAME,
        )

    cases.append((ablation_spec, wrong_fpr))
    baseline_spec = CLOSURE_PACKAGE_FAMILY_SPECS[3]

    def wrong_baseline(
        documents: dict[str, dict[str, Any]],
        spec: ClosurePackageFamilySpec,
    ) -> None:
        _assign_source(
            documents,
            spec,
            spec.baseline_sources[0],
            "gaussian_shading",
            paper_run_name=PAPER_RUN_NAME,
        )

    cases.append((baseline_spec, wrong_baseline))
    official_spec = CLOSURE_PACKAGE_FAMILY_SPECS[6]

    def ready_false(documents: dict[str, dict[str, Any]], spec: ClosurePackageFamilySpec) -> None:
        requirement = next(
            requirement
            for requirement in spec.value_requirements
            if requirement.expected_value is True
        )
        _assign_source(
            documents,
            spec,
            requirement.source,
            False,
            paper_run_name=PAPER_RUN_NAME,
        )

    cases.append((official_spec, ready_false))

    for case_index, (spec, mutate) in enumerate(cases):
        package_path = _write_family_package(
            tmp_path / f"case_{case_index}",
            spec,
            token="invalid",
            mutate=mutate,
        )
        with pytest.raises(ClosurePackageSelectionError):
            inspect_closure_package(
                package_path,
                spec=spec,
                paper_run_name=PAPER_RUN_NAME,
                target_fpr=TARGET_FPR,
            )


def test_package_rejects_tampered_scientific_execution_binding(
    tmp_path: Path,
) -> None:
    """科学绑定被替换后, 即使业务摘要仍为 pass 也不得进入闭合."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]
    contract = spec.scientific_execution_binding
    assert contract is not None
    binding_member = _render(
        contract.binding_member_template,
        spec,
        PAPER_RUN_NAME,
    )
    package_path = _write_family_package(
        tmp_path,
        spec,
        token="tampered_scientific_binding",
        extra_members={binding_member: b'{}\n'},
    )
    with pytest.raises(ClosurePackageSelectionError, match="科学执行绑定"):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_package_rejects_self_consistent_wrong_scientific_route(
    tmp_path: Path,
) -> None:
    """重算摘要也不能把科学执行入口替换为未登记命令."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]
    contract = spec.scientific_execution_binding
    assert contract is not None
    members = _valid_member_payloads(
        spec,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        generated_at=GENERATED_AT,
    )
    execution_member = _render(
        contract.execution_report_member_template,
        spec,
        PAPER_RUN_NAME,
    )
    binding_member = _render(
        contract.binding_member_template,
        spec,
        PAPER_RUN_NAME,
    )
    execution = json.loads(members[execution_member].decode("utf-8"))
    execution["child_argv_tail"] = ["-m", "unregistered.scientific_entry"]
    execution["execution"]["argv"] = [
        execution["python_executable_path"],
        *execution["child_argv_tail"],
    ]
    members[execution_member] = (
        json.dumps(execution, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    binding = json.loads(members[binding_member].decode("utf-8"))
    binding["scientific_execution_report_digest"] = hashlib.sha256(
        members[execution_member]
    ).hexdigest()
    members[binding_member] = (
        json.dumps(binding, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    package_path = tmp_path / "wrong_route.zip"
    with ZipFile(package_path, "w") as archive:
        for member_name, payload in sorted(members.items()):
            archive.writestr(member_name, payload)

    with pytest.raises(ClosurePackageSelectionError, match="route 不匹配"):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_package_rejects_self_consistent_dependency_command_tampering(
    tmp_path: Path,
) -> None:
    """重算全部重复摘要也不能把依赖安装或外层准备命令改成其他 argv."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]
    contract = spec.scientific_execution_binding
    assert contract is not None
    members = _valid_member_payloads(
        spec,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        generated_at=GENERATED_AT,
    )
    dependency_member = _render(
        contract.dependency_report_member_template,
        spec,
        PAPER_RUN_NAME,
    )
    dependency = json.loads(members[dependency_member].decode("utf-8"))
    dependency_preparation = dependency["dependency_preparation_report"]
    dependency_preparation["installation"]["command"][3] = "download"
    dependency["dependency_preparation_report_digest"] = hashlib.sha256(
        (
            json.dumps(
                dependency_preparation,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    outer_command = dependency["dependency_preparation_command"]
    outer_command["argv"][2] = "experiments.runtime.repository_environment"
    dependency["command_results"][-1] = outer_command
    _rebind_scientific_dependency_member(members, spec, dependency)
    package_path = tmp_path / "self_consistent_dependency_command.zip"
    with ZipFile(package_path, "w") as archive:
        for member_name, payload in sorted(members.items()):
            archive.writestr(member_name, payload)

    with pytest.raises(ClosurePackageSelectionError, match="依赖嵌套证据"):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_selection_rejects_self_consistent_dependency_count_tampering(
    tmp_path: Path,
) -> None:
    """execution,顶层和内层 count 同时改写后仍必须匹配仓库完整锁."""

    package_root = tmp_path / "drive"
    package_paths = _write_all_family_packages(package_root)
    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]
    contract = spec.scientific_execution_binding
    assert contract is not None
    members = _valid_member_payloads(
        spec,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        generated_at=GENERATED_AT,
    )
    dependency_member = _render(
        contract.dependency_report_member_template,
        spec,
        PAPER_RUN_NAME,
    )
    execution_member = _render(
        contract.execution_report_member_template,
        spec,
        PAPER_RUN_NAME,
    )
    binding_member = _render(
        contract.binding_member_template,
        spec,
        PAPER_RUN_NAME,
    )
    dependency = json.loads(members[dependency_member].decode("utf-8"))
    dependency["complete_hash_lock_dependency_count"] = 8
    dependency["dependency_preparation_report"][
        "complete_hash_lock_dependency_count"
    ] = 8
    dependency["dependency_preparation_report_digest"] = hashlib.sha256(
        (
            json.dumps(
                dependency["dependency_preparation_report"],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    _rebind_scientific_dependency_member(members, spec, dependency)
    execution = json.loads(members[execution_member].decode("utf-8"))
    execution["complete_hash_lock_dependency_count"] = 8
    members[execution_member] = (
        json.dumps(execution, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    binding = json.loads(members[binding_member].decode("utf-8"))
    binding["scientific_execution_report_digest"] = hashlib.sha256(
        members[execution_member]
    ).hexdigest()
    members[binding_member] = (
        json.dumps(binding, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    _rebind_package_input_member_digests(members, spec)
    with ZipFile(package_paths[0], "w") as archive:
        for member_name, payload in sorted(members.items()):
            archive.writestr(member_name, payload)
    candidate = inspect_closure_package(
        package_paths[0],
        spec=spec,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
    )
    assert candidate.scientific_complete_hash_lock_dependency_count == 8

    with pytest.raises(ClosurePackageSelectionError, match="仓库正式 profile 不一致"):
        build_closure_input_selection_report(
            package_root,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
        )


def test_package_rejects_bound_scientific_manifest_lock_drift(
    tmp_path: Path,
) -> None:
    """bound manifest 自洽重算后仍必须与 archive 和 binding 的锁一致."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[3]
    contract = spec.scientific_execution_binding
    assert contract is not None
    members = _valid_member_payloads(
        spec,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        generated_at=GENERATED_AT,
    )
    manifest_member = _render(
        contract.manifest_member_template,
        spec,
        PAPER_RUN_NAME,
    )
    binding_member = _render(
        contract.binding_member_template,
        spec,
        PAPER_RUN_NAME,
    )
    dispatch_member = _render(
        contract.dispatch_report_member_template,
        spec,
        PAPER_RUN_NAME,
    )
    manifest = json.loads(members[manifest_member].decode("utf-8"))
    manifest["formal_execution_run_lock"] = build_test_formal_execution_lock(
        "b" * 40
    )
    manifest["code_version"] = "b" * 40
    members[manifest_member] = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    dispatch = json.loads(members[dispatch_member].decode("utf-8"))
    dispatch["manifest_sha256"] = hashlib.sha256(
        members[manifest_member]
    ).hexdigest()
    members[dispatch_member] = (
        json.dumps(dispatch, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    binding = json.loads(members[binding_member].decode("utf-8"))
    binding["bound_manifest_scientific_digest"] = (
        scientific_manifest_payload_digest(manifest)
    )
    binding["scientific_command_dispatch_report_digest"] = hashlib.sha256(
        members[dispatch_member]
    ).hexdigest()
    members[binding_member] = (
        json.dumps(binding, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    package_path = tmp_path / "bound_manifest_lock_drift.zip"
    with ZipFile(package_path, "w") as archive:
        for member_name, payload in sorted(members.items()):
            archive.writestr(member_name, payload)

    with pytest.raises(ClosurePackageSelectionError, match="code_version 不一致"):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_official_package_rejects_tampered_dependency_environment_evidence(
    tmp_path: Path,
) -> None:
    """官方参考结果不得用业务 ready 字段替代隔离依赖证据."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[6]
    contract = spec.dependency_environment_evidence
    assert contract is not None
    report_member = _render(
        contract.report_member_template,
        spec,
        PAPER_RUN_NAME,
    )
    package_path = _write_family_package(
        tmp_path,
        spec,
        token="tampered_dependency_evidence",
        extra_members={report_member: b'{}\n'},
    )
    with pytest.raises(ClosurePackageSelectionError):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_official_package_rejects_unbound_embedded_environment_digest(
    tmp_path: Path,
) -> None:
    """official 外层报告必须绑定内嵌正式环境报告的真实文件摘要."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[6]
    contract = spec.dependency_environment_evidence
    assert contract is not None
    members = _valid_member_payloads(
        spec,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        generated_at=GENERATED_AT,
    )
    report_member = _render(
        contract.report_member_template,
        spec,
        PAPER_RUN_NAME,
    )
    report = json.loads(members[report_member].decode("utf-8"))
    report["isolated_dependency_environment_report_digest"] = "7" * 64
    members[report_member] = (
        json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    package_path = tmp_path / "official_unbound_environment.zip"
    with ZipFile(package_path, "w") as archive:
        for member_name, payload in sorted(members.items()):
            archive.writestr(member_name, payload)

    with pytest.raises(ClosurePackageSelectionError, match="内嵌隔离依赖环境报告摘要"):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


@pytest.mark.parametrize(
    "missing_lock_field",
    ("formal_execution_run_lock", "formal_execution_package_lock"),
)
def test_package_rejects_missing_formal_execution_lock(
    tmp_path: Path,
    missing_lock_field: str,
) -> None:
    """每个闭合包 manifest 必须同时携带运行锁和打包锁."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]

    def mutate(
        documents: dict[str, dict[str, Any]],
        current_spec: ClosurePackageFamilySpec,
    ) -> None:
        manifest_name = _render(
            current_spec.manifest_member_template,
            current_spec,
            PAPER_RUN_NAME,
        )
        documents[manifest_name].pop(missing_lock_field)

    package_path = _write_family_package(
        tmp_path,
        spec,
        token=f"missing_{missing_lock_field}",
        mutate=mutate,
    )
    with pytest.raises(ClosurePackageSelectionError, match="缺少 formal_execution"):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_package_rejects_forged_formal_execution_lock_digest(tmp_path: Path) -> None:
    """任一执行锁摘要被篡改后都不得进入正式闭合."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]

    def mutate(
        documents: dict[str, dict[str, Any]],
        current_spec: ClosurePackageFamilySpec,
    ) -> None:
        manifest_name = _render(
            current_spec.manifest_member_template,
            current_spec,
            PAPER_RUN_NAME,
        )
        package_lock = documents[manifest_name]["formal_execution_package_lock"]
        assert isinstance(package_lock, dict)
        package_lock["formal_execution_lock_digest"] = "0" * 64

    package_path = _write_family_package(
        tmp_path,
        spec,
        token="forged_lock_digest",
        mutate=mutate,
    )
    with pytest.raises(ClosurePackageSelectionError, match="严格复验"):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_package_rejects_run_and_package_lock_commit_drift(tmp_path: Path) -> None:
    """运行锁与打包锁即使各自有效也必须绑定同一个 commit."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]

    def mutate(
        documents: dict[str, dict[str, Any]],
        current_spec: ClosurePackageFamilySpec,
    ) -> None:
        manifest_name = _render(
            current_spec.manifest_member_template,
            current_spec,
            PAPER_RUN_NAME,
        )
        documents[manifest_name]["formal_execution_package_lock"] = (
            build_test_formal_execution_lock("b" * 40)
        )

    package_path = _write_family_package(
        tmp_path,
        spec,
        token="lock_commit_drift",
        mutate=mutate,
    )
    with pytest.raises(ClosurePackageSelectionError, match="运行锁与打包锁 commit 不一致"):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_package_rejects_execution_lock_and_code_version_drift(tmp_path: Path) -> None:
    """两个执行锁一致时仍必须与全部 code_version 来源完全一致."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]

    def mutate(
        documents: dict[str, dict[str, Any]],
        current_spec: ClosurePackageFamilySpec,
    ) -> None:
        manifest_name = _render(
            current_spec.manifest_member_template,
            current_spec,
            PAPER_RUN_NAME,
        )
        drifted_lock = build_test_formal_execution_lock("b" * 40)
        documents[manifest_name]["formal_execution_run_lock"] = drifted_lock
        documents[manifest_name]["formal_execution_package_lock"] = drifted_lock

    package_path = _write_family_package(
        tmp_path,
        spec,
        token="lock_code_version_drift",
        mutate=mutate,
    )
    with pytest.raises(ClosurePackageSelectionError, match="code_version 来源不一致"):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


@pytest.mark.parametrize(
    "invalid_code_version",
    (
        f"{CODE_VERSION}-dirty",
        "git_version_unavailable",
        "main",
        "abc1234",
        "A" * 40,
    ),
)
def test_package_rejects_non_clean_git_code_version(
    tmp_path: Path,
    invalid_code_version: str,
) -> None:
    """单包必须拒绝 dirty, 降级值, 分支名, 短 SHA 与大写 SHA."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]

    def mutate(documents: dict[str, dict[str, Any]], current_spec: ClosurePackageFamilySpec) -> None:
        for source in current_spec.code_version_sources:
            _assign_source(
                documents,
                current_spec,
                source,
                invalid_code_version,
                paper_run_name=PAPER_RUN_NAME,
            )

    package_path = _write_family_package(tmp_path, spec, token="invalid_code", mutate=mutate)
    with pytest.raises(ClosurePackageSelectionError, match="code_version"):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_selection_requires_one_common_code_version_across_all_families(tmp_path: Path) -> None:
    """即使各包内部自洽, 10个 family 的 clean Git 提交也必须完全相同."""

    package_root = tmp_path / "drive"
    for spec in CLOSURE_PACKAGE_FAMILY_SPECS[1:]:
        _write_family_package(package_root, spec, token="current")
    selected_spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]

    def mutate(documents: dict[str, dict[str, Any]], current_spec: ClosurePackageFamilySpec) -> None:
        for source in current_spec.code_version_sources:
            _assign_source(
                documents,
                current_spec,
                source,
                "b" * 40,
                paper_run_name=PAPER_RUN_NAME,
            )
        _assign_execution_locks(
            documents,
            current_spec,
            "b" * 40,
            paper_run_name=PAPER_RUN_NAME,
        )

    _write_family_package(package_root, selected_spec, token="different", mutate=mutate)
    with pytest.raises(ClosurePackageSelectionError, match="同一科学会话|共享同一"):
        build_closure_input_selection_report(
            package_root,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
        )


def test_full_lowercase_clean_commit_is_accepted(tmp_path: Path) -> None:
    """精确40位小写 clean 提交可以进入闭合输入候选."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]
    full_commit = "b" * 40

    def mutate(documents: dict[str, dict[str, Any]], current_spec: ClosurePackageFamilySpec) -> None:
        for source in current_spec.code_version_sources:
            _assign_source(
                documents,
                current_spec,
                source,
                full_commit,
                paper_run_name=PAPER_RUN_NAME,
            )
        _assign_execution_locks(
            documents,
            current_spec,
            full_commit,
            paper_run_name=PAPER_RUN_NAME,
        )

    package_path = _write_family_package(tmp_path, spec, token="full_commit", mutate=mutate)
    candidate = inspect_closure_package(
        package_path,
        spec=spec,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
    )
    assert candidate.code_version == full_commit
    expected_lock_digest = build_test_formal_execution_lock(full_commit)[
        "formal_execution_lock_digest"
    ]
    assert candidate.formal_execution_run_lock_digest == expected_lock_digest
    assert candidate.formal_execution_package_lock_digest == expected_lock_digest


def test_selection_rejects_package_profile_not_anchored_to_repository_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """自洽结果包的 profile 摘要仍必须匹配当前提交的正式 registry."""

    package_root = tmp_path / "drive"
    _write_all_family_packages(package_root)

    def mismatched_profile(profile_id: str, registry_path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            profile_name=profile_id,
            profile_digest="8" * 64,
            direct_requirements_digest="4" * 64,
            complete_hash_lock_digest="5" * 64,
            complete_hash_lock_dependency_count=7,
            complete_hash_lock_present=True,
            formal_ready=True,
            readiness_blockers=(),
        )

    monkeypatch.setattr(
        selection_module,
        "require_dependency_profile_ready",
        mismatched_profile,
    )
    with pytest.raises(ClosurePackageSelectionError, match="仓库正式 profile 不一致"):
        build_closure_input_selection_report(
            package_root,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
        )


def test_selection_rejects_packages_from_another_repository_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """10类包即使彼此一致, 只要不是当前提交就必须在闭合入口立即阻断."""

    package_root = tmp_path / "drive"
    _write_all_family_packages(package_root)
    monkeypatch.setattr(
        selection_module,
        "resolve_code_version",
        lambda root: "b" * 40,
    )

    with pytest.raises(ClosurePackageSelectionError, match="当前 clean 仓库提交"):
        build_closure_input_selection_report(
            package_root,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
            root=tmp_path,
        )


def test_extended_clean_short_commit_is_rejected(tmp_path: Path) -> None:
    """任何不足40位的 clean 提交前缀都不得进入正式闭合."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]
    extended_short_commit = "abc12345"

    def mutate(documents: dict[str, dict[str, Any]], current_spec: ClosurePackageFamilySpec) -> None:
        for source in current_spec.code_version_sources:
            _assign_source(
                documents,
                current_spec,
                source,
                extended_short_commit,
                paper_run_name=PAPER_RUN_NAME,
            )

    package_path = _write_family_package(tmp_path, spec, token="extended_short", mutate=mutate)
    with pytest.raises(ClosurePackageSelectionError, match="code_version"):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_matching_filename_cannot_replace_internal_family_identity(tmp_path: Path) -> None:
    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]

    def wrong_artifact(
        documents: dict[str, dict[str, Any]],
        current_spec: ClosurePackageFamilySpec,
    ) -> None:
        manifest_name = _render(
            current_spec.manifest_member_template,
            current_spec,
            PAPER_RUN_NAME,
        )
        documents[manifest_name]["artifact_id"] = "unrelated_artifact"

    package_path = _write_family_package(
        tmp_path,
        spec,
        token="masquerading",
        mutate=wrong_artifact,
    )
    with pytest.raises(ClosurePackageSelectionError):
        inspect_closure_package(
            package_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )


def test_rejects_directory_empty_damaged_traversal_and_non_output_members(
    tmp_path: Path,
) -> None:
    spec = CLOSURE_PACKAGE_FAMILY_SPECS[0]
    candidate_paths: list[Path] = []

    directory_path = tmp_path / "directory.zip"
    directory_path.mkdir()
    candidate_paths.append(directory_path)

    empty_file_path = tmp_path / "empty_file.zip"
    empty_file_path.write_bytes(b"")
    candidate_paths.append(empty_file_path)

    empty_zip_path = tmp_path / "empty_archive.zip"
    with ZipFile(empty_zip_path, "w"):
        pass
    candidate_paths.append(empty_zip_path)

    damaged_path = tmp_path / "damaged.zip"
    damaged_path.write_bytes(b"not-a-zip")
    candidate_paths.append(damaged_path)

    traversal_path = tmp_path / "traversal.zip"
    with ZipFile(traversal_path, "w") as archive:
        archive.writestr("../escape.json", "{}")
    candidate_paths.append(traversal_path)

    non_output_path = tmp_path / "non_output.zip"
    with ZipFile(non_output_path, "w") as archive:
        archive.writestr("README.md", "not allowed")
    candidate_paths.append(non_output_path)

    for candidate_path in candidate_paths:
        with pytest.raises(ClosurePackageSelectionError):
            inspect_closure_package(
                candidate_path,
                spec=spec,
                paper_run_name=PAPER_RUN_NAME,
                target_fpr=TARGET_FPR,
            )


def test_package_input_member_digests_bind_declared_bytes_and_exact_member_set(
    tmp_path: Path,
) -> None:
    """存在逐成员摘要声明时, 内容变化或额外成员都必须被拒绝."""

    spec = CLOSURE_PACKAGE_FAMILY_SPECS[3]
    assert spec.package_input_manifest_template is not None
    members = _valid_member_payloads(
        spec,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        generated_at=GENERATED_AT,
    )
    package_input_member = _render(spec.package_input_manifest_template, spec, PAPER_RUN_NAME)
    declared_member = next(
        name
        for name in sorted(members)
        if name != package_input_member and name.endswith("_summary.json")
    )
    package_input = json.loads(members[package_input_member].decode("utf-8"))
    package_input["entry_count"] = 1
    package_input["entry_paths"] = [declared_member]
    package_input["entry_sha256"] = {
        declared_member: hashlib.sha256(members[declared_member]).hexdigest()
    }
    members[package_input_member] = (
        json.dumps(package_input, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")

    valid_path = tmp_path / "declared_valid.zip"
    with ZipFile(valid_path, "w") as archive:
        for member_name, payload in sorted(members.items()):
            archive.writestr(member_name, payload)
    inspect_closure_package(
        valid_path,
        spec=spec,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
    )

    changed_path = tmp_path / "declared_changed.zip"
    with ZipFile(changed_path, "w") as archive:
        for member_name, payload in sorted(members.items()):
            archive.writestr(
                member_name,
                b"changed\n" if member_name == declared_member else payload,
            )
    with pytest.raises(ClosurePackageSelectionError, match="成员摘要不匹配"):
        inspect_closure_package(
            changed_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )

    extra_path = tmp_path / "declared_extra.zip"
    allowed_extra_member = _render(
        spec.allowed_output_prefix_templates[0], spec, PAPER_RUN_NAME
    ) + "undeclared_extra.json"
    with ZipFile(extra_path, "w") as archive:
        for member_name, payload in sorted(members.items()):
            archive.writestr(member_name, payload)
        archive.writestr(allowed_extra_member, b"{}\n")
    with pytest.raises(ClosurePackageSelectionError, match="精确成员集合不一致"):
        inspect_closure_package(
            extra_path,
            spec=spec,
            paper_run_name=PAPER_RUN_NAME,
            target_fpr=TARGET_FPR,
        )
