"""验证精确父编排宿主入口的 fail-closed 边界."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_formal_workflow_host as host_launcher


UV_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/75/2e/"
    "62273ee6c9fbebccd8248c153b44870f81ebf5267c31edf4c095d78537fb/"
    "uv-0.11.28-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
)
UV_WHEEL_SHA256 = "49fe42df9f42056037473f3876adec1615709b57d3470ed39178ff420f3afb9f"


def _git(root: Path, *arguments: str) -> str:
    """执行测试仓库 Git 命令并返回标准输出."""

    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _detached_repository(tmp_path: Path) -> tuple[Path, str]:
    """创建带单个 detached 提交的最小仓库."""

    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "测试")
    (root / "tracked.txt").write_text("content\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-m", "测试提交")
    commit = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "--detach", commit)
    return root, commit


def _bootstrap_repository(tmp_path: Path) -> tuple[Path, Path]:
    """创建包含父 profile、完整锁和固定 uv 工具锁的 detached 仓库."""

    root = tmp_path / "bootstrap_repository"
    profile_root = root / "configs/dependency_profiles"
    profile_root.mkdir(parents=True)
    registry = {
        "profiles": {
            "workflow_orchestrator": {
                "accelerator": {"runtime": "cpu"},
                "complete_hash_lock_path": (
                    "configs/dependency_profiles/workflow_orchestrator_lock.txt"
                ),
                "execution_role": "workflow_orchestration",
                "platform": {
                    "machine": "x86_64",
                    "operating_system": "linux",
                },
                "python": {
                    "implementation": "CPython",
                    "version": "3.12.13",
                },
            }
        }
    }
    (root / "configs/dependency_profile_registry.json").write_text(
        json.dumps(registry),
        encoding="utf-8",
    )
    lock_path = profile_root / "workflow_orchestrator_lock.txt"
    lock_path.write_text(
        "packaging==25.0 --hash=sha256:" + "1" * 64 + "\n",
        encoding="utf-8",
    )
    (profile_root / "dependency_qualification_uv_linux_x86_64_lock.txt").write_text(
        f"uv==0.11.28 --wheel-url={UV_WHEEL_URL} "
        f"--hash=sha256:{UV_WHEEL_SHA256}\n",
        encoding="utf-8",
    )
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "测试")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "父编排锁")
    commit = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "--detach", commit)
    return root, lock_path


@pytest.mark.quick
def test_clean_detached_checkout_gate_rejects_attached_and_dirty_state(tmp_path: Path) -> None:
    """宿主入口只接受请求提交对应的 clean detached checkout."""

    root, commit = _detached_repository(tmp_path)
    assert host_launcher.validate_clean_detached_checkout(root, commit) == root.resolve()
    (root / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(host_launcher.FormalWorkflowHostError, match="clean Git"):
        host_launcher.validate_clean_detached_checkout(root, commit)
    (root / "untracked.txt").unlink()
    _git(root, "checkout", "master")
    with pytest.raises(host_launcher.FormalWorkflowHostError, match="detached HEAD"):
        host_launcher.validate_clean_detached_checkout(root, commit)


@pytest.mark.quick
def test_child_command_covers_gpu_and_cpu_closure_routes(tmp_path: Path) -> None:
    """同一精确父解释器入口应覆盖 GPU workflow 与 CPU 闭合."""

    python_executable = tmp_path / "python"
    common = {
        "root": ".",
        "repository_commit": "a" * 40,
        "paper_run_name": "probe_paper",
        "result_path": "outputs/result.json",
        "persistent_output_dir": "",
        "package_search_root": "",
        "complete_output_dir": "",
        "dry_run": False,
    }
    bootstrap_identity = {
        "profile_id": "workflow_orchestrator",
        "python_version": "3.12.13",
        "complete_hash_lock_digest": "b" * 64,
        "python_executable": str(python_executable),
        "python_executable_sha256": "c" * 64,
    }
    gpu_arguments = argparse.Namespace(
        **common,
        operation="gpu",
        workflow="external_baseline_tree_ring",
    )
    gpu_command = host_launcher.build_child_command(
        gpu_arguments,
        python_executable,
        Path("/repository"),
        bootstrap_identity,
    )
    assert gpu_command[:3] == [
        str(python_executable),
        "-I",
        str(Path("/repository/scripts/formal_workflow_entry.py")),
    ]
    assert gpu_command[gpu_command.index("--workflow") + 1] == "external_baseline_tree_ring"

    closure_arguments = argparse.Namespace(
        **{
            **common,
            "package_search_root": "/drive/probe_paper_results",
            "complete_output_dir": "/drive/probe_paper_results/complete_result_package",
        },
        operation="closure",
        workflow=None,
    )
    closure_command = host_launcher.build_child_command(
        closure_arguments,
        python_executable,
        Path("/repository"),
        bootstrap_identity,
    )
    assert "--package-search-root" in closure_command
    assert "--complete-output-dir" in closure_command


@pytest.mark.quick
def test_prepare_exact_orchestrator_binds_fixed_bootstrap_and_all_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全 mocked 引导仍必须固定工具锁、Python patch、哈希锁和复验命令."""

    root, lock_path = _bootstrap_repository(tmp_path)
    captured_tool: dict[str, Any] = {}
    commands: list[tuple[str, list[str], dict[str, str]]] = []

    def fake_materialize_tool(**kwargs: Any) -> tuple[Path, Path, str]:
        captured_tool.update(kwargs)
        runtime_root = Path(kwargs["runtime_root"])
        wheel_path = runtime_root / "tool_downloads/uv.whl"
        executable = runtime_root / "uv_tool/bin/uv"
        wheel_path.parent.mkdir(parents=True, exist_ok=True)
        executable.parent.mkdir(parents=True, exist_ok=True)
        wheel_path.write_bytes(b"fixed-wheel")
        executable.write_bytes(b"fixed-uv")
        return wheel_path, executable, "uv"

    def fake_execute(
        command: list[str],
        *,
        root: Path,
        environment: dict[str, str],
        operation: str,
    ) -> None:
        commands.append((operation, list(command), dict(environment)))
        if operation == "orchestrator_venv":
            python_executable = (
                tmp_path
                / "runtime/workflow_orchestrator/bin/python"
            )
            python_executable.parent.mkdir(parents=True, exist_ok=True)
            python_executable.write_bytes(b"exact-cpython-3.12.13")

    monkeypatch.setattr(host_launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(host_launcher.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        host_launcher,
        "_materialize_qualification_uv_tool",
        fake_materialize_tool,
    )
    monkeypatch.setattr(host_launcher, "_execute_checked", fake_execute)

    python_executable, identity = host_launcher.prepare_exact_orchestrator(
        root=root,
        runtime_root=tmp_path / "runtime",
    )

    assert captured_tool["wheel_url"] == UV_WHEEL_URL
    assert captured_tool["expected_wheel_digest"] == UV_WHEEL_SHA256
    assert captured_tool["downloader"] is host_launcher._download_qualification_tool_wheel
    by_operation = {operation: command for operation, command, _ in commands}
    assert list(by_operation) == [
        "uv_version",
        "python_install",
        "orchestrator_venv",
        "orchestrator_ensurepip",
        "orchestrator_hash_install",
        "orchestrator_pip_check",
        "orchestrator_python_inspection",
    ]
    assert "3.12.13" in by_operation["python_install"]
    hash_install = by_operation["orchestrator_hash_install"]
    assert hash_install[:4] == [
        str(python_executable),
        "-m",
        "pip",
        "install",
    ]
    assert "--require-hashes" in hash_install
    assert "--only-binary=:all:" in hash_install
    assert hash_install[hash_install.index("-r") + 1] == str(lock_path.resolve())
    assert by_operation["orchestrator_pip_check"] == [
        str(python_executable),
        "-m",
        "pip",
        "check",
    ]
    assert by_operation["orchestrator_python_inspection"][:3] == [
        str(python_executable),
        "-I",
        "-c",
    ]
    assert identity == {
        "profile_id": "workflow_orchestrator",
        "python_version": "3.12.13",
        "complete_hash_lock_digest": host_launcher._complete_hash_lock_digest(
            host_launcher._load_complete_hash_lock(lock_path)
        ),
        "python_executable": str(python_executable),
        "python_executable_sha256": hashlib.sha256(
            python_executable.read_bytes()
        ).hexdigest(),
    }

    lock_path.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(host_launcher.FormalWorkflowHostError, match="当前 HEAD"):
        host_launcher.prepare_exact_orchestrator(
            root=root,
            runtime_root=tmp_path / "second_runtime",
        )


@pytest.mark.quick
@pytest.mark.parametrize(
    "operation",
    (
        "uv_version",
        "python_install",
        "orchestrator_venv",
        "orchestrator_ensurepip",
        "orchestrator_hash_install",
        "orchestrator_pip_check",
        "orchestrator_python_inspection",
    ),
)
def test_each_bootstrap_command_nonzero_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """任一引导操作返回非零状态都必须变为统一宿主门禁失败."""

    monkeypatch.setattr(
        host_launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=17),
    )
    with pytest.raises(host_launcher.FormalWorkflowHostError, match=operation):
        host_launcher._execute_checked(
            ["command"],
            root=tmp_path,
            environment={},
            operation=operation,
        )


@pytest.mark.quick
def test_host_main_requires_python_isolated_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """未使用 python -I 时必须在任何下载或环境创建前闭锁."""

    monkeypatch.setattr(host_launcher.sys, "flags", argparse.Namespace(isolated=0))
    assert host_launcher.main(
        [
            "--repository-commit",
            "a" * 40,
            "gpu",
            "--workflow",
            "image_only_dataset",
            "--paper-run-name",
            "probe_paper",
            "--result-path",
            "outputs/result.json",
        ]
    ) == 2
