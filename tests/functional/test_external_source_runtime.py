"""验证外部官方源码的不可变 Git 身份和补丁工作树证据。"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from paper_experiments.runners.external_source_runtime import (
    bind_successful_official_command_execution_evidence,
    build_registered_source_patch_evidence,
    inspect_cuda_with_python_executable,
    prepare_registered_source_checkout,
)


def _run_git(source_dir: Path, *arguments: str) -> str:
    """在测试 Git 仓库执行命令并返回标准输出。"""

    completed = subprocess.run(
        ["git", *arguments],
        cwd=source_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_source_registry(root_path: Path, commit: str) -> None:
    """写入只包含测试 baseline 的固定源码登记项。"""

    registry_path = root_path / "external_baseline" / "source_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "baseline_sources": [
                    {
                        "baseline_id": "source_identity_test",
                        "official_repository_url": "git@github.com:example/source-identity-test.git",
                        "official_repository_commit": commit,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.quick
def test_registered_source_checkout_restores_exact_commit_and_builds_patch_evidence(
    tmp_path: Path,
) -> None:
    """正式源码缓存应先恢复登记提交, 再只接受登记文件集合内的补丁。"""

    source_dir = tmp_path / "external_baseline" / "primary" / "source_identity_test" / "source"
    source_dir.mkdir(parents=True)
    _run_git(source_dir, "init")
    _run_git(source_dir, "config", "user.email", "source-test@example.com")
    _run_git(source_dir, "config", "user.name", "Source Test")
    source_file = source_dir / "runtime.py"
    source_file.write_text("value = 1\n", encoding="utf-8")
    _run_git(source_dir, "add", "runtime.py")
    _run_git(source_dir, "commit", "-m", "建立测试源码提交")
    commit = _run_git(source_dir, "rev-parse", "HEAD")
    _run_git(
        source_dir,
        "remote",
        "add",
        "origin",
        "https://github.com/example/source-identity-test.git",
    )
    _write_source_registry(tmp_path, commit)

    source_file.write_text("value = 999\n", encoding="utf-8")
    (source_dir / "untracked.txt").write_text("不得保留\n", encoding="utf-8")
    identity = prepare_registered_source_checkout(
        tmp_path,
        "source_identity_test",
        source_dir,
    )

    assert identity["source_identity_ready"] is True
    assert identity["source_head_commit"] == commit
    assert identity["source_base_worktree_clean"] is True
    assert source_file.read_text(encoding="utf-8") == "value = 1\n"
    assert not (source_dir / "untracked.txt").exists()

    source_file.write_text("value = 2\n", encoding="utf-8")
    evidence = build_registered_source_patch_evidence(
        tmp_path,
        "source_identity_test",
        source_dir,
        ("runtime.py",),
    )

    assert evidence["source_worktree_exact"] is True
    assert evidence["source_modified_paths"] == ["runtime.py"]
    assert len(evidence["source_patch_sha256"]) == 64
    assert len(evidence["source_worktree_digest"]) == 64
    assert len(evidence["patched_source_sha256"]["runtime.py"]) == 64


@pytest.mark.quick
def test_registered_source_checkout_rejects_non_git_directory(tmp_path: Path) -> None:
    """普通目录不能伪装成已核验的官方源码 checkout。"""

    source_dir = tmp_path / "external_baseline" / "primary" / "source_identity_test" / "source"
    source_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="不是可验证的 Git checkout"):
        prepare_registered_source_checkout(
            tmp_path,
            "source_identity_test",
            source_dir,
        )


@pytest.mark.quick
def test_cuda_inspection_uses_supplied_isolated_python_and_records_process_evidence(
    tmp_path: Path,
) -> None:
    """CUDA 检查必须前置隔离 Python, 并保留解释器摘要与子进程证据."""

    python_path = tmp_path / "dependency_environment" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_bytes(b"isolated-python")
    captured: dict[str, object] = {}

    def fake_runner(command: list[str], **kwargs: object) -> dict[str, object]:
        captured["command"] = command
        captured.update(kwargs)
        payload = {
            "python_executable": str(python_path.absolute()),
            "torch_available": True,
            "cuda_available": True,
            "device": "cuda",
            "torch_version": "2.5.0+cu124",
            "torch_cuda_version": "12.4",
            "device_count": 1,
            "gpu_name": "Test GPU",
        }
        return {
            "return_code": 0,
            "stdout": json.dumps(payload) + "\n",
            "stderr": "",
        }

    report = inspect_cuda_with_python_executable(
        python_path,
        require_cuda=True,
        cwd=tmp_path,
        command_runner=fake_runner,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[0] == str(python_path.absolute())
    assert command[1] == "-c"
    assert command[-1] == "1"
    assert captured["cwd"] == tmp_path
    assert report["python_executable"] == str(python_path.absolute())
    assert len(report["python_executable_sha256"]) == 64
    assert report["torch_available"] is True
    assert report["cuda_available"] is True
    assert report["decision"] == "pass"
    assert report["failure_reasons"] == []


@pytest.mark.quick
def test_successful_official_command_binds_cuda_interpreter_and_cwd(
    tmp_path: Path,
) -> None:
    """成功官方命令必须绑定同一隔离解释器, CUDA 报告和工作目录."""

    python_path = (tmp_path / "environment" / "python").absolute()
    command = [str(python_path), str((tmp_path / "source" / "run.py").absolute())]
    device_report = {
        "python_executable": str(python_path),
        "python_executable_sha256": "a" * 64,
        "decision": "pass",
        "failure_reasons": [],
        "return_code": 0,
        "torch_available": True,
        "cuda_available": True,
        "device": "cuda",
        "device_count": 1,
        "gpu_name": "Test GPU",
        "torch_version": "2.7.1+cu128",
        "supports_paper_claim": False,
    }
    result = bind_successful_official_command_execution_evidence(
        {
            "official_command_requested": True,
            "official_command": command,
            "return_code": 0,
        },
        baseline_id="tree_ring",
        command=command,
        working_directory=tmp_path / "source",
        dependency_python_executable=python_path,
        cuda_inspection_report=device_report,
    )

    assert result["report_schema"] == "official_reference_command_execution_report"
    assert result["baseline_id"] == "tree_ring"
    assert result["official_command_working_directory"] == str(
        (tmp_path / "source").absolute()
    )
    assert result["dependency_python_executable"] == str(python_path)
    assert result["dependency_python_executable_sha256"] == "a" * 64
    assert len(result["cuda_inspection_report_digest"]) == 64
    assert result["official_command_execution_evidence_ready"] is True


@pytest.mark.quick
def test_successful_official_command_rejects_cuda_python_identity_drift(
    tmp_path: Path,
) -> None:
    """CUDA 探针使用不同解释器时不得形成官方执行证据."""

    python_path = (tmp_path / "environment" / "python").absolute()
    command = [str(python_path), "run.py"]
    with pytest.raises(RuntimeError, match="无法绑定"):
        bind_successful_official_command_execution_evidence(
            {
                "official_command_requested": True,
                "official_command": command,
                "return_code": 0,
            },
            baseline_id="tree_ring",
            command=command,
            working_directory=tmp_path,
            dependency_python_executable=python_path,
            cuda_inspection_report={
                "python_executable": str(
                    (tmp_path / "other_environment" / "python").absolute()
                ),
                "python_executable_sha256": "a" * 64,
                "decision": "pass",
                "failure_reasons": [],
                "return_code": 0,
                "torch_available": True,
                "cuda_available": True,
                "device": "cuda",
                "device_count": 1,
                "gpu_name": "Test GPU",
                "torch_version": "2.7.1+cu128",
                "supports_paper_claim": False,
            },
        )


@pytest.mark.quick
def test_cuda_inspection_rejects_required_cuda_absence(tmp_path: Path) -> None:
    """隔离解释器可导入 torch 但无 CUDA 时必须保留失败证据."""

    python_path = tmp_path / "dependency_environment" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_bytes(b"isolated-python")

    def fake_runner(command: list[str], **kwargs: object) -> dict[str, object]:
        return {
            "return_code": 3,
            "stdout": json.dumps(
                {
                    "python_executable": str(python_path.absolute()),
                    "torch_available": True,
                    "cuda_available": False,
                    "device": "cpu",
                    "torch_version": "2.5.0+cu124",
                    "torch_cuda_version": "12.4",
                    "device_count": 0,
                    "gpu_name": "",
                }
            ),
            "stderr": "",
        }

    report = inspect_cuda_with_python_executable(
        python_path,
        require_cuda=True,
        cwd=tmp_path,
        command_runner=fake_runner,
    )

    assert report["decision"] == "fail"
    assert report["failure_reasons"] == ["cuda_required_but_unavailable"]


@pytest.mark.quick
def test_official_reference_parents_do_not_call_current_python_cuda_check() -> None:
    """三个 official-reference 父 runner 只能调用隔离 Python CUDA 检查."""

    runner_paths = (
        Path("paper_experiments/runners/tree_ring_official_reference.py"),
        Path("paper_experiments/runners/gaussian_shading_official_reference.py"),
        Path("paper_experiments/runners/shallow_diffuse_official_reference.py"),
    )
    for runner_path in runner_paths:
        source = runner_path.read_text(encoding="utf-8")
        assert "ensure_cuda_if_requested" not in source
        assert "inspect_cuda_with_python_executable" in source
