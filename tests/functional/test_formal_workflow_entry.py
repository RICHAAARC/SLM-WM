"""验证精确父解释器内部入口只调用现有正式 workflow."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_workflow.cli import formal_workflow_entry as entry


BOOTSTRAP_IDENTITY = {
    "profile_id": "workflow_orchestrator",
    "python_version": "3.12.13",
    "complete_hash_lock_digest": "b" * 64,
    "python_executable": "/runtime/workflow_orchestrator/bin/python",
    "python_executable_sha256": "c" * 64,
}


@pytest.mark.quick
def test_gpu_entry_configures_route_and_uses_drive_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """内部入口应配置唯一 baseline 并把 Drive 目录交给服务器 workflow."""

    captured: dict[str, object] = {}

    def fake_configure(
        workflow_name: str,
        *,
        baseline_id: str,
        repository_root: Path,
    ) -> dict[str, object]:
        captured.update(
            workflow_name=workflow_name,
            baseline_id=baseline_id,
            repository_root=repository_root,
        )
        monkeypatch.setenv(
            "SLM_WM_EXTERNAL_BASELINE_DRIVE_OUTPUT_DIR",
            "/drive/method_faithful",
        )
        return {}

    def fake_run_workflow(*arguments: object) -> dict[str, object]:
        captured["run_arguments"] = arguments
        return {"workflow_summary": {"run_decision": "pass"}, "archive_record": {}}

    monkeypatch.setattr(entry, "configure_paper_run_environment", fake_configure)
    monkeypatch.setattr(entry, "run_workflow", fake_run_workflow)
    arguments = argparse.Namespace(
        workflow="external_baseline_tree_ring",
        paper_run_name="probe_paper",
        persistent_output_dir="",
        repository_commit="a" * 40,
    )
    result = entry._gpu_result(arguments, tmp_path)

    assert captured["workflow_name"] == "external_baseline_method_faithful"
    assert captured["baseline_id"] == "tree_ring"
    assert captured["run_arguments"][-1] == "/drive/method_faithful"
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
            "archive_record": {"archive_ready": True},
            "orchestrator_dependency_environment": orchestrator_environment,
        },
    )
    arguments = argparse.Namespace(
        root=str(tmp_path),
        operation="gpu",
        workflow="image_only_dataset",
        paper_run_name="probe_paper",
        repository_commit="a" * 40,
    )

    result = entry.execute(arguments)

    assert result["orchestrator_bootstrap_identity"] == BOOTSTRAP_IDENTITY
    assert result["orchestrator_dependency_environment"] == orchestrator_environment
    assert result["formal_execution_lock"] == execution_lock
