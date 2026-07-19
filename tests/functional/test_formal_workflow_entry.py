"""验证精确父解释器内部入口只调用现有正式 workflow."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import formal_workflow_entry as entry


BOOTSTRAP_IDENTITY = {
    "profile_id": "workflow_orchestrator",
    "python_version": "3.12.13",
    "complete_hash_lock_digest": "b" * 64,
    "python_executable": "/runtime/workflow_orchestrator/bin/python",
    "python_executable_sha256": "c" * 64,
}


@pytest.mark.quick
def test_gpu_entry_delegates_to_independent_server_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """内部入口应把 GPU 配置与持久化解析交给独立服务器 workflow."""

    captured: dict[str, object] = {}

    def fake_run_workflow(
        *arguments: object,
        **keywords: object,
    ) -> dict[str, object]:
        captured["run_arguments"] = arguments
        captured["run_keywords"] = keywords
        return {
            "workflow_summary": {"run_decision": "pass"},
            "archive_record": {},
        }

    monkeypatch.setattr(entry, "run_workflow", fake_run_workflow)
    arguments = argparse.Namespace(
        workflow="external_baseline_tree_ring",
        paper_run_name="probe_paper",
        persistent_output_dir="",
        repository_commit="a" * 40,
        randomization_repeat_id="seed_00_key_00",
        calibration_only=False,
        expected_reference_registry_digest="1" * 64,
        expected_reference_registry_file_sha256="2" * 64,
    )
    result = entry._gpu_result(arguments, tmp_path)

    assert captured["run_arguments"] == (
        "external_baseline_tree_ring",
        "probe_paper",
        "a" * 40,
        tmp_path,
        None,
    )
    assert captured["run_keywords"] == {
        "randomization_repeat_id": "seed_00_key_00",
        "calibration_only": False,
        "expected_reference_registry_digest": "1" * 64,
        "expected_reference_registry_file_sha256": "2" * 64,
    }
    assert result["workflow_summary"] == {"run_decision": "pass"}


@pytest.mark.quick
def test_result_writer_rejects_path_outside_outputs(tmp_path: Path) -> None:
    """Notebook 可读结果不得越过统一 outputs 根目录."""

    with pytest.raises(ValueError, match="outputs"):
        entry._write_result(tmp_path, tmp_path / "outside.json", {"decision": "pass"})

    path = entry._write_result(
        tmp_path,
        "outputs/formal_workflow_execution/probe_paper/result.json",
        {"decision": "pass"},
    )
    assert path.is_file()
    assert not path.with_name(path.name + ".partial").exists()


@pytest.mark.quick
def test_bootstrap_identity_revalidates_current_python_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """内部入口必须复算当前精确解释器摘要, 不能只信任宿主参数."""

    executable = tmp_path / "bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"exact-cpython")
    executable_digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    monkeypatch.setattr(entry.sys, "executable", str(executable))
    monkeypatch.setattr(entry.sys, "flags", SimpleNamespace(isolated=1))
    monkeypatch.setattr(entry.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(entry.platform, "python_version", lambda: "3.12.13")
    monkeypatch.setattr(entry.platform, "system", lambda: "Linux")
    monkeypatch.setattr(entry.platform, "machine", lambda: "x86_64")
    arguments = argparse.Namespace(
        orchestrator_profile_id="workflow_orchestrator",
        orchestrator_python_version="3.12.13",
        orchestrator_lock_digest="b" * 64,
        orchestrator_python_executable=str(executable),
        orchestrator_python_executable_sha256=executable_digest,
    )

    assert entry._validate_bootstrap_identity(arguments) == {
        **BOOTSTRAP_IDENTITY,
        "python_executable": executable.as_posix(),
        "python_executable_sha256": executable_digest,
    }
    arguments.orchestrator_python_executable_sha256 = "d" * 64
    with pytest.raises(RuntimeError, match="摘要不一致"):
        entry._validate_bootstrap_identity(arguments)


@pytest.mark.quick
def test_execute_records_bootstrap_and_verified_orchestrator_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """受治理结果必须同时记录宿主引导身份与父环境复验证据."""

    execution_lock = {"formal_execution_commit": "a" * 40}
    orchestrator_environment = {
        "profile_id": "workflow_orchestrator",
        "complete_hash_lock_digest": "b" * 64,
    }
    monkeypatch.setattr(
        entry,
        "_validate_bootstrap_identity",
        lambda arguments: dict(BOOTSTRAP_IDENTITY),
    )
    monkeypatch.setattr(
        entry,
        "build_formal_execution_lock",
        lambda root, commit: execution_lock,
    )
    monkeypatch.setattr(entry, "publish_formal_execution_lock", lambda lock: lock)
    monkeypatch.setattr(
        entry,
        "_gpu_result",
        lambda arguments, root: {
            "workflow_summary": {"workflow_decision": "complete"},
            "workflow_environment": {"persistent_output_dir": "/persistent"},
            "archive_record": {"archive_ready": True},
            "orchestrator_dependency_environment": orchestrator_environment,
            "randomization_scope": "active_repeat_component",
            "randomization_repeat_id": "seed_00_key_00",
        },
    )
    arguments = argparse.Namespace(
        root=str(tmp_path),
        operation="gpu",
        workflow="image_only_dataset",
        paper_run_name="probe_paper",
        repository_commit="a" * 40,
        randomization_repeat_id="seed_00_key_00",
    )

    result = entry.execute(arguments)

    assert result["orchestrator_bootstrap_identity"] == BOOTSTRAP_IDENTITY
    assert result["orchestrator_dependency_environment"] == orchestrator_environment
    assert result["formal_execution_lock"] == execution_lock
    assert result["workflow_environment"] == {
        "persistent_output_dir": "/persistent"
    }


@pytest.mark.quick
def test_content_runtime_smoke_is_a_named_orchestrator_operation() -> None:
    """The smoke operation carries all three explicit references through parsing."""

    arguments = entry.build_parser().parse_args(
        [
            "content_runtime_smoke",
            "--root", "/repository",
            "--repository-commit", "a" * 40,
            "--paper-run-name", "probe_paper",
            "--result-path", "outputs/smoke.json",
            "--orchestrator-profile-id", "workflow_orchestrator",
            "--orchestrator-python-version", "3.12.13",
            "--orchestrator-lock-digest", "b" * 64,
            "--orchestrator-python-executable", "/managed/python",
            "--orchestrator-python-executable-sha256", "c" * 64,
            "--prompt-id", "probe_prompt_0001",
            "--reference-gradient", "1.0",
            "--reference-response", "0.5",
            "--reference-sensitivity", "0.25",
        ]
    )
    assert arguments.operation == "content_runtime_smoke"
    assert arguments.reference_gradient == 1.0
    assert arguments.smoke_output_root == "outputs/content_runtime_smoke"
