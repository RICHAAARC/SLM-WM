"""验证 method-faithful 与 T2SMark 的共享隔离科学调度."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.runtime.scientific_execution_binding import (
    file_sha256,
    stable_json_file_digest,
)
from paper_experiments.runners import isolated_scientific_workflow as workflow
from paper_workflow.notebook_utils import notebook_entrypoint
from paper_workflow.notebook_utils import workflow_archive_naming
from tests.helpers.formal_execution_lock import build_test_formal_execution_lock


FORMAL_LOCK = build_test_formal_execution_lock("c" * 40)


@pytest.mark.quick
def test_notebook_workflow_archive_naming_owns_baseline_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """外层命名器必须独立解析 baseline 角色并复用内层通用身份原语。"""

    monkeypatch.setattr(
        workflow_archive_naming,
        "utc_archive_token",
        lambda: "20260713t000000z",
    )
    monkeypatch.setattr(
        workflow_archive_naming,
        "resolve_short_commit",
        lambda root: "00b967c",
    )

    assert workflow_archive_naming.build_workflow_archive_name(
        "external_baseline_method_faithful",
        baseline_id="tree_ring",
    ) == (
        "external_baseline_method_faithful_package_tree_ring_"
        "20260713t000000z_00b967c.zip"
    )
    with pytest.raises(ValueError, match="唯一受支持 baseline_id"):
        workflow_archive_naming.build_workflow_archive_name(
            "external_baseline_method_faithful",
            baseline_id="unsupported",
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """稳定写出测试需要的 JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _install_fake_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workflow_name: str,
    *,
    tamper_summary_digest: bool = False,
) -> dict[str, Any]:
    """注入一个不创建真实 venv 的完整跨进程成功证据."""

    paths = workflow.scientific_artifact_paths(tmp_path, workflow_name)
    profile_id = workflow.WORKFLOW_PROFILE_IDS[workflow_name]
    if workflow_name == "external_baseline_method_faithful":
        summary = {
            "run_decision": "pass",
            "external_baseline_method_faithful_ready": True,
            "primary_baseline_id": "tree_ring",
        }
    else:
        summary = {
            "run_decision": "pass",
            "t2smark_formal_reproduction_ready": True,
            "baseline_id": "t2smark",
        }
    manifest = {
        "formal_execution_run_lock": FORMAL_LOCK,
        "code_version": FORMAL_LOCK["formal_execution_commit"],
    }
    dependency_report_path = (
        tmp_path
        / "outputs"
        / "dependency_profiles"
        / profile_id
        / "isolated_dependency_environment_report.json"
    )
    python_path = tmp_path / "managed_python" / profile_id / "python"
    dependency_report_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_bytes(b"isolated-python")
    python_executable = str(python_path.resolve())
    working_directory = str(tmp_path.resolve())
    complete_hash_lock_path = "configs/dependency_profiles/test_complete.lock"
    complete_hash_lock_absolute_path = str(
        (tmp_path / complete_hash_lock_path).resolve()
    )
    pytorch_index_url = "https://download.pytorch.org/whl/cu128"
    managed_python_root = str((tmp_path / "managed_pythons").resolve())
    isolated_environment_path = str((tmp_path / "dependency_environment").resolve())
    uv_executable_path = str((tmp_path / "bin" / "uv").resolve())
    command_environment = {"UV_PYTHON_INSTALL_DIR": managed_python_root}

    def command_record(operation: str, argv: list[str]) -> dict[str, Any]:
        """构造完整依赖命令 fixture."""

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
        command_record("python_ensurepip", [python_executable, "-m", "ensurepip"]),
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
            profile_id,
        ],
    )
    dependency_preparation = {
        "report_schema": "dependency_profile_preparation_report",
        "schema_version": 1,
        "profile_id": profile_id,
        "python_executable": python_executable,
        "working_directory": working_directory,
        "profile_digest": "1" * 64,
        "direct_requirements_digest": "2" * 64,
        "complete_hash_lock_digest": "3" * 64,
        "complete_hash_lock_dependency_count": 7,
        "complete_hash_lock_path": complete_hash_lock_path,
        "pytorch_index_url": pytorch_index_url,
        "formal_ready": True,
        "readiness_blockers": [],
        "formal_execution_lock": FORMAL_LOCK,
        "formal_execution_commit": FORMAL_LOCK["formal_execution_commit"],
        "formal_execution_lock_digest": FORMAL_LOCK[
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
            "profile_digest": "1" * 64,
            "complete_hash_lock_digest": "3" * 64,
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
        "profile_digest": "1" * 64,
        "python_version": "3.11.0",
        "managed_python_root": managed_python_root,
        "working_directory": working_directory,
        "isolated_environment_path": isolated_environment_path,
        "uv_executable_path": uv_executable_path,
        "python_executable_path": python_executable,
        "command_results": provision_commands,
        "uv_commands": provision_commands[:3],
        "target_complete_hash_lock_ready": True,
        "python_executable_sha256": file_sha256(python_path),
        "formal_execution_lock": FORMAL_LOCK,
        "formal_execution_commit": FORMAL_LOCK["formal_execution_commit"],
        "formal_execution_lock_digest": FORMAL_LOCK[
            "formal_execution_lock_digest"
        ],
        "formal_execution_lock_ready": True,
        "provisioned": True,
        "formal_ready": False,
        "decision": "provisioned",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    _write_json(
        dependency_report_path,
        {
            "report_schema": "isolated_dependency_environment_preparation_report",
            "schema_version": 1,
            "operation_kind": "formal_dependency_environment_preparation",
            "profile_id": profile_id,
            "profile_digest": "1" * 64,
            "direct_requirements_digest": "2" * 64,
            "complete_hash_lock_digest": "3" * 64,
            "complete_hash_lock_dependency_count": 7,
            "working_directory": working_directory,
            "python_executable_path": python_executable,
            "python_executable_sha256": file_sha256(python_path),
            "python_executable_sha256_after_preparation": file_sha256(python_path),
            "dependency_preparation_report": dependency_preparation,
            "dependency_preparation_report_digest": stable_json_file_digest(
                dependency_preparation
            ),
            "provision_report": provision_report,
            "provision_report_digest": stable_json_file_digest(
                provision_report
            ),
            "dependency_preparation_command": dependency_preparation_command,
            "command_results": [
                *provision_commands,
                dependency_preparation_command,
            ],
            "uv_commands": provision_commands[:3],
            "provisioned": True,
            "formal_execution_lock": FORMAL_LOCK,
            "formal_execution_commit": FORMAL_LOCK["formal_execution_commit"],
            "formal_execution_lock_digest": FORMAL_LOCK[
                "formal_execution_lock_digest"
            ],
            "formal_execution_lock_ready": True,
            "formal_preparation_completed": True,
            "formal_ready": True,
            "decision": "pass",
            "failure_reasons": [],
            "supports_paper_claim": False,
        },
    )

    def fake_execute(
        supplied_profile_id: str,
        child_argv_tail: Any,
        *,
        execution_report_path: Any,
        repository_root: Any,
    ) -> Any:
        """模拟 runtime API, 同时物化科学子进程唯一 envelope."""

        assert supplied_profile_id == profile_id
        assert Path(repository_root).resolve() == tmp_path.resolve()
        _write_json(paths["summary"], summary)
        _write_json(paths["manifest"], manifest)
        envelope = {
            "report_schema": "scientific_command_dispatch_report",
            "result_schema": workflow.RESULT_ENVELOPE_SCHEMA,
            "schema_version": workflow.RESULT_ENVELOPE_SCHEMA_VERSION,
            "operation_kind": "isolated_scientific_workflow",
            "workflow_name": workflow_name,
            "paper_run_name": "probe_paper",
            "profile_id": profile_id,
            "decision": "pass",
            "child_decision": "pass",
            "summary": summary,
            "summary_path": paths["summary"].relative_to(tmp_path).as_posix(),
            "summary_sha256": (
                "0" * 64 if tamper_summary_digest else file_sha256(paths["summary"])
            ),
            "manifest_path": paths["manifest"].relative_to(tmp_path).as_posix(),
            "manifest_sha256": file_sha256(paths["manifest"]),
            "dependency_environment_report_digest": file_sha256(dependency_report_path),
            "formal_execution_lock": FORMAL_LOCK,
            "formal_execution_commit": FORMAL_LOCK["formal_execution_commit"],
            "formal_execution_lock_digest": FORMAL_LOCK["formal_execution_lock_digest"],
            "supports_paper_claim": False,
        }
        _write_json(paths["result_envelope"], envelope)
        resolved_report_path = Path(execution_report_path).resolve()
        report = {
            "report_schema": "isolated_scientific_execution_report",
            "schema_version": 1,
            "operation_kind": "isolated_scientific_execution",
            "repository_root": str(tmp_path.resolve()),
            "profile_id": profile_id,
            "profile_digest": "1" * 64,
            "direct_requirements_digest": "2" * 64,
            "complete_hash_lock_digest": "3" * 64,
            "complete_hash_lock_dependency_count": 7,
            "dependency_environment_report_path": str(dependency_report_path.resolve()),
            "source_dependency_environment_report_path": str(
                dependency_report_path.resolve()
            ),
            "dependency_environment_report_digest": file_sha256(dependency_report_path),
            "dependency_environment_report_valid": True,
            "dependency_environment_validation_errors": [],
            "python_executable_path": str(python_path.resolve()),
            "python_executable_sha256": file_sha256(python_path),
            "python_executable_revalidated_before_child": True,
            "python_executable_revalidated_after_child": True,
            "dependency_environment_report_revalidated_before_child": True,
            "dependency_environment_report_revalidated_after_child": True,
            "formal_execution_lock": FORMAL_LOCK,
            "formal_execution_commit": FORMAL_LOCK["formal_execution_commit"],
            "formal_execution_lock_digest": FORMAL_LOCK["formal_execution_lock_digest"],
            "formal_execution_lock_ready": True,
            "formal_execution_lock_revalidated_before_child": True,
            "formal_execution_lock_revalidated_after_child": True,
            "child_argv_tail": list(child_argv_tail),
            "execution": {
                "attempted": True,
                "argv": [str(python_path), *child_argv_tail],
                "working_directory": str(tmp_path.resolve()),
                "environment_overrides": {
                    "SLM_WM_FORMAL_EXECUTION_COMMIT": FORMAL_LOCK[
                        "formal_execution_commit"
                    ],
                    "SLM_WM_FORMAL_EXECUTION_LOCK_DIGEST": FORMAL_LOCK[
                        "formal_execution_lock_digest"
                    ],
                    "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH": str(
                        dependency_report_path.resolve()
                    ),
                    "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST": file_sha256(
                        dependency_report_path
                    ),
                },
                "return_code": 0,
                "stdout": "",
                "stderr": "",
            },
            "execution_report_path": str(resolved_report_path),
            "execution_completed": True,
            "decision": "pass",
            "failure_reasons": [],
            "supports_paper_claim": False,
        }
        _write_json(resolved_report_path, report)
        return report, resolved_report_path

    monkeypatch.setattr(workflow, "execute_isolated_scientific_command", fake_execute)
    return summary


@pytest.mark.quick
@pytest.mark.parametrize(
    "workflow_name",
    ("external_baseline_method_faithful", "official_reference_t2smark"),
)
def test_parent_dispatch_binds_isolated_execution_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflow_name: str,
) -> None:
    """父进程必须验证 envelope 并把执行证据复制到 workflow 输出范围."""

    monkeypatch.setenv("SLM_WM_PRIMARY_BASELINE_ID", "tree_ring")
    monkeypatch.setattr(
        workflow,
        "build_paper_run_config",
        lambda root: SimpleNamespace(run_name="probe_paper"),
    )
    expected_summary = _install_fake_execution(monkeypatch, tmp_path, workflow_name)

    actual_summary = workflow.run_isolated_scientific_workflow(
        root=tmp_path,
        workflow_name=workflow_name,
    )

    paths = workflow.scientific_artifact_paths(tmp_path, workflow_name)
    binding = json.loads(paths["execution_binding"].read_text(encoding="utf-8"))
    localized_report = json.loads(paths["execution_report"].read_text(encoding="utf-8"))
    assert actual_summary == expected_summary
    assert binding["profile_id"] == workflow.WORKFLOW_PROFILE_IDS[workflow_name]
    assert binding["scientific_execution_report_digest"] == file_sha256(paths["execution_report"])
    assert binding["dependency_environment_report_digest"] == file_sha256(paths["dependency_report"])
    assert localized_report["dependency_environment_report_path"] == paths["dependency_report"].name
    assert localized_report["execution_report_path"] == paths["execution_report"].name
    assert not paths["source_execution_report"].exists()


@pytest.mark.quick
def test_parent_dispatch_rejects_tampered_result_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """子进程 envelope 摘要失配时不得写出科学执行 binding."""

    monkeypatch.setenv("SLM_WM_PRIMARY_BASELINE_ID", "tree_ring")
    monkeypatch.setattr(
        workflow,
        "build_paper_run_config",
        lambda root: SimpleNamespace(run_name="probe_paper"),
    )
    _install_fake_execution(
        monkeypatch,
        tmp_path,
        "external_baseline_method_faithful",
        tamper_summary_digest=True,
    )

    with pytest.raises(RuntimeError, match="产物摘要不一致"):
        workflow.run_isolated_scientific_workflow(
            root=tmp_path,
            workflow_name="external_baseline_method_faithful",
        )

    paths = workflow.scientific_artifact_paths(
        tmp_path,
        "external_baseline_method_faithful",
    )
    assert not paths["execution_binding"].exists()


@pytest.mark.quick
def test_scientific_child_writes_one_summary_bound_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """科学子进程必须调用 runner 并写出唯一的 summary / manifest 绑定结果."""

    monkeypatch.setenv("SLM_WM_PRIMARY_BASELINE_ID", "tree_ring")
    monkeypatch.setattr(
        workflow,
        "build_paper_run_config",
        lambda root: SimpleNamespace(run_name="probe_paper"),
    )
    paths = workflow.scientific_artifact_paths(
        tmp_path,
        "external_baseline_method_faithful",
    )
    summary = {
        "run_decision": "pass",
        "external_baseline_method_faithful_ready": True,
        "primary_baseline_id": "tree_ring",
    }
    manifest = {"formal_execution_run_lock": FORMAL_LOCK}

    def fake_scientific_runner(root: Path, workflow_name: str) -> dict[str, Any]:
        assert root == tmp_path.resolve()
        assert workflow_name == "external_baseline_method_faithful"
        _write_json(paths["summary"], summary)
        _write_json(paths["manifest"], manifest)
        return summary

    dependency_path = tmp_path / "outputs" / "dependency_profiles" / "report.json"
    dependency_path.parent.mkdir(parents=True, exist_ok=True)
    dependency_path.write_text('{"decision":"pass"}\n', encoding="utf-8")
    monkeypatch.setenv(
        "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH",
        str(dependency_path.resolve()),
    )
    monkeypatch.setenv(
        "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST",
        file_sha256(dependency_path),
    )
    monkeypatch.setattr(workflow, "_run_child_scientific_workflow", fake_scientific_runner)
    monkeypatch.setattr(
        workflow,
        "require_published_formal_execution_lock",
        lambda root: dict(FORMAL_LOCK),
    )

    return_code = workflow.run_scientific_child(
        root=tmp_path,
        workflow_name="external_baseline_method_faithful",
        result_envelope_path=paths["result_envelope"],
    )

    envelope_files = tuple(paths["scientific_dir"].glob("*result_envelope.json"))
    envelope = json.loads(paths["result_envelope"].read_text(encoding="utf-8"))
    assert return_code == 0
    assert envelope_files == (paths["result_envelope"],)
    assert envelope["summary"] == summary
    assert envelope["summary_sha256"] == file_sha256(paths["summary"])
    assert envelope["manifest_sha256"] == file_sha256(paths["manifest"])
    assert envelope["decision"] == "pass"


@pytest.mark.quick
def test_notebook_entrypoint_calls_shared_isolated_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notebook helper 对两个 workflow 均只调用共享隔离调度函数."""

    calls = []

    def fake_dispatch(*, root: Any, workflow_name: str) -> dict[str, str]:
        calls.append((root, workflow_name))
        return {"workflow_name": workflow_name}

    monkeypatch.setattr(workflow, "run_isolated_scientific_workflow", fake_dispatch)

    method_summary = notebook_entrypoint.run_workflow(
        root="repository",
        workflow_name="external_baseline_method_faithful",
    )
    t2smark_summary = notebook_entrypoint.run_workflow(
        root="repository",
        workflow_name="official_reference_t2smark",
    )

    assert method_summary["workflow_name"] == "external_baseline_method_faithful"
    assert t2smark_summary["workflow_name"] == "official_reference_t2smark"
    assert calls == [
        ("repository", "external_baseline_method_faithful"),
        ("repository", "official_reference_t2smark"),
    ]


@pytest.mark.quick
def test_t2smark_packaging_reuses_bound_scientific_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2SMark 打包必须复用原科学 Python, 并复验本地与镜像归档."""

    monkeypatch.setattr(
        workflow,
        "build_paper_run_config",
        lambda root: SimpleNamespace(run_name="probe_paper"),
    )
    paths = workflow.scientific_artifact_paths(
        tmp_path,
        "official_reference_t2smark",
    )
    python_path = tmp_path / "managed_python" / "t2smark" / "python"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_bytes(b"bound-scientific-python")
    _write_json(paths["dependency_report"], {"decision": "pass"})
    dependency_digest = file_sha256(paths["dependency_report"])
    _write_json(
        paths["execution_report"],
        {
            "python_executable_path": str(python_path.resolve()),
            "python_executable_sha256": file_sha256(python_path),
            "formal_execution_lock": FORMAL_LOCK,
        },
    )

    validation_calls = []

    def fake_validate_binding(
        binding_path: Any,
        *,
        expected_artifact_role: str,
        expected_paper_run_name: str,
        repository_root: Any,
    ) -> dict[str, str]:
        """返回测试需要的已验证依赖报告摘要."""

        validation_calls.append(
            (
                Path(binding_path),
                expected_artifact_role,
                expected_paper_run_name,
                Path(repository_root),
            )
        )
        return {"dependency_environment_report_digest": dependency_digest}

    monkeypatch.setattr(
        workflow,
        "validate_scientific_execution_binding",
        fake_validate_binding,
    )
    monkeypatch.setattr(
        workflow,
        "require_published_formal_execution_lock",
        lambda root: dict(FORMAL_LOCK),
    )
    drive_dir = tmp_path / "drive"
    local_archive = paths["output_scope"] / "bound-t2smark.zip"
    drive_archive = drive_dir / "bound-t2smark.zip"

    def fake_run(
        argv: Any,
        *,
        cwd: Any,
        env: dict[str, str],
        check: bool,
        capture_output: bool,
        text: bool,
        shell: bool,
    ) -> Any:
        """模拟相同科学解释器中的打包子进程."""

        assert argv[0] == str(python_path)
        assert argv[1:5] == [
            "-m",
            "paper_experiments.runners.isolated_scientific_workflow",
            "--package-workflow",
            "official_reference_t2smark",
        ]
        assert argv[-2:] == ["--archive-name", "bound-t2smark.zip"]
        assert Path(cwd) == tmp_path.resolve()
        assert check is False
        assert capture_output is True
        assert text is True
        assert shell is False
        assert env[
            workflow.DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY
        ] == str(paths["dependency_report"].resolve())
        assert env[
            workflow.DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY
        ] == dependency_digest
        assert env[workflow.FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY] == FORMAL_LOCK[
            "formal_execution_commit"
        ]
        assert env[
            workflow.FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY
        ] == FORMAL_LOCK["formal_execution_lock_digest"]
        local_archive.parent.mkdir(parents=True, exist_ok=True)
        drive_archive.parent.mkdir(parents=True, exist_ok=True)
        local_archive.write_bytes(b"identical-bound-archive")
        drive_archive.write_bytes(local_archive.read_bytes())
        archive_digest = file_sha256(local_archive)
        record = {
            "archive_path": local_archive.relative_to(tmp_path).as_posix(),
            "archive_digest": archive_digest,
            "archive_entry_count": 5,
            "drive_archive_path": str(drive_archive.resolve()),
            "drive_archive_digest": archive_digest,
            "metadata": {"workflow_name": "official_reference_t2smark"},
        }
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "decision": "pass",
                    "archive_record": record,
                    "supports_paper_claim": False,
                }
            )
            + "\n",
            stderr="",
        )

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)

    record = workflow.package_isolated_scientific_workflow_outputs(
        root=tmp_path,
        workflow_name="official_reference_t2smark",
        drive_output_dir=str(drive_dir),
        archive_name="bound-t2smark.zip",
    )

    assert validation_calls == [
        (
            paths["execution_binding"],
            "t2smark_formal_reproduction",
            "probe_paper",
            tmp_path.resolve(),
        )
    ]
    assert record.archive_path == local_archive.relative_to(tmp_path).as_posix()
    assert record.archive_digest == file_sha256(local_archive)
    assert record.drive_archive_digest == file_sha256(drive_archive)
    assert record.archive_entry_count == 5


@pytest.mark.quick
def test_notebook_t2smark_packaging_calls_bound_scientific_packager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notebook 的 T2SMark 打包入口不得在父解释器导入科学 runner."""

    calls = []
    marker = object()

    def fake_package(**kwargs: Any) -> object:
        """记录 Notebook 传给隔离打包器的稳定参数."""

        calls.append(dict(kwargs))
        return marker

    monkeypatch.setattr(
        workflow,
        "package_isolated_scientific_workflow_outputs",
        fake_package,
    )
    monkeypatch.setattr(
        notebook_entrypoint,
        "build_workflow_archive_name",
        lambda workflow_name, root, baseline_id: "notebook-t2smark.zip",
    )
    monkeypatch.setattr(
        notebook_entrypoint,
        "build_paper_run_config",
        lambda root: SimpleNamespace(run_name="probe_paper"),
    )
    monkeypatch.setattr(
        notebook_entrypoint,
        "write_notebook_runtime_report",
        lambda **kwargs: {},
    )
    drive_dir = tmp_path / "drive"

    result = notebook_entrypoint.package_workflow_outputs(
        root=tmp_path,
        workflow_name="official_reference_t2smark",
        drive_output_dir=str(drive_dir),
    )

    assert result is marker
    assert calls == [
        {
            "root": tmp_path,
            "workflow_name": "official_reference_t2smark",
            "drive_output_dir": str(drive_dir),
            "archive_name": "notebook-t2smark.zip",
        }
    ]


@pytest.mark.quick
def test_t2smark_parent_routes_do_not_import_formal_runner() -> None:
    """服务器与 Notebook 父路由只能调用隔离打包器."""

    root = Path(__file__).resolve().parents[2]
    for relative_path in (
        "scripts/run_gpu_server_workflow.py",
        "paper_workflow/notebook_utils/notebook_entrypoint.py",
    ):
        source = (root / relative_path).read_text(encoding="utf-8")
        assert (
            "from paper_experiments.runners.t2smark_formal_reproduction" not in source
        )
