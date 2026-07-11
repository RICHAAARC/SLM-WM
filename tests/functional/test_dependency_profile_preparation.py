"""验证统一依赖 profile CLI 的哈希锁安装和精确环境门禁."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from experiments.runtime import dependency_preparation as preparation
from experiments.runtime.dependency_profiles import (
    DependencyProfile,
    get_dependency_profile,
    parse_exact_requirement_spec,
)


def _summary(profile: DependencyProfile) -> dict[str, Any]:
    """构造与共享 API 同形的稳定测试摘要."""

    return {
        **profile.to_dict(),
        "summary_digest": "b" * 64,
    }


def _formal_execution_lock() -> dict[str, Any]:
    """构造内层 preparation 必须传播的正式执行锁."""

    return {
        "formal_execution_lock_schema": "formal_execution_lock",
        "formal_execution_commit": "a" * 40,
        "formal_execution_head_detached": True,
        "formal_execution_worktree_clean": True,
        "formal_execution_lock_ready": True,
        "formal_execution_lock_digest": "f" * 64,
    }


def _ready_profile(profile_id: str = "workflow_orchestrator") -> DependencyProfile:
    """由真实 registry 记录派生带完整哈希锁状态的轻量 fixture."""

    profile = get_dependency_profile(profile_id)
    return replace(
        profile,
        complete_hash_lock_present=True,
        complete_hash_lock_digest="a" * 64,
        complete_hash_lock_dependency_count=len(profile.direct_requirements) + 3,
        formal_ready=True,
        readiness_blockers=(),
    )


def _matching_inspection(profile: DependencyProfile) -> dict[str, Any]:
    """返回共享 inspection API 的完全匹配记录."""

    direct_dependencies = {
        dependency.normalized_name: dependency.version
        for dependency in (
            parse_exact_requirement_spec(specification)
            for specification in profile.direct_requirements
        )
    }
    expected_environment = {
        "python_implementation": profile.python_implementation,
        "python_version": profile.python_version,
        "operating_system": profile.operating_system,
        "machine": profile.machine,
        "accelerator_runtime": profile.accelerator_runtime,
        "cuda_version": profile.cuda_version,
        "torch_version": profile.torch_version,
        "torchvision_version": profile.torchvision_version,
        "direct_dependencies": direct_dependencies,
    }
    observed_environment = {
        "python_implementation": profile.python_implementation,
        "python_version": profile.python_version,
        "operating_system": profile.operating_system,
        "machine": profile.machine,
        "torch_module_available": (
            True if profile.accelerator_runtime == "cuda" else None
        ),
        "torch_module_version": profile.torch_version,
        "torch_cuda_version": profile.cuda_version,
        "cuda_available": True if profile.accelerator_runtime == "cuda" else None,
        "direct_dependencies": direct_dependencies,
    }
    return {
        "profile_name": profile.profile_name,
        "profile_digest": profile.profile_digest,
        "complete_hash_lock_digest": profile.complete_hash_lock_digest,
        "profile_formal_ready": True,
        "expected_environment": expected_environment,
        "observed_environment": observed_environment,
        "environment_match": True,
        "mismatches": [],
        "readiness_blockers": [],
        "decision": "pass",
        "inspection_digest": "c" * 64,
    }


def _bind_profile_api(
    monkeypatch: pytest.MonkeyPatch,
    profile: DependencyProfile,
) -> None:
    """把 CLI 绑定到同一个共享 profile API 测试记录."""

    monkeypatch.setattr(
        preparation,
        "get_dependency_profile",
        lambda profile_id, path: profile,
    )
    monkeypatch.setattr(
        preparation,
        "build_dependency_profile_summary",
        lambda profile_id, path: _summary(profile),
    )
    monkeypatch.setattr(
        preparation,
        "require_dependency_profile_ready",
        lambda profile_id, path: profile,
    )
    monkeypatch.setattr(
        preparation,
        "require_published_formal_execution_lock",
        lambda root: _formal_execution_lock(),
    )


@pytest.mark.quick
def test_ready_profile_uses_committed_hash_lock_and_records_exact_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全部 profile 都必须由当前解释器通过完整哈希锁执行精确安装."""

    profile = _ready_profile()
    assert profile.accelerator_runtime == "cpu"
    assert profile.pytorch_index_url is None
    _bind_profile_api(monkeypatch, profile)
    monkeypatch.setattr(
        preparation,
        "_inspect_dependency_files_commit_state",
        lambda root, selected_profile: {
            "files": {
                "registry": {"is_committed": True},
                "direct_requirements": {"is_committed": True},
                "complete_hash_lock": {"is_committed": True},
            },
            "all_committed": True,
        },
    )
    monkeypatch.setattr(
        preparation,
        "inspect_dependency_profile_environment",
        lambda profile_id, *, path, torch_module=None: _matching_inspection(profile),
    )
    commands: list[tuple[list[str], Path]] = []

    def record_command(command: list[str], working_directory: Path) -> int:
        commands.append((list(command), working_directory))
        return 0

    report, report_path = preparation.prepare_dependency_profile(
        profile.profile_name,
        repository_root=tmp_path,
        command_runner=record_command,
    )

    expected_lock_path = (tmp_path / profile.complete_hash_lock_path).resolve()
    assert commands == [
        (
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--require-hashes",
                "--only-binary=:all:",
                "-r",
                str(expected_lock_path),
            ],
            tmp_path.resolve(),
        ),
    ]
    assert report_path == (
        tmp_path
        / "outputs/dependency_profiles/workflow_orchestrator/dependency_profile_report.json"
    ).resolve()
    assert report["decision"] == "pass"
    assert report["python_executable"] == sys.executable
    assert report["profile_digest"] == profile.profile_digest
    assert report["direct_requirements_path"] == profile.direct_requirements_path
    assert report["direct_requirements_digest"] == profile.direct_requirements_digest
    assert report["complete_hash_lock_digest"] == "a" * 64
    assert report["runtime_comparison"]["environment_match"] is True
    assert report["pip_check"] == {
        "compatibility_check_required": False,
        "attempted": False,
        "command": [],
        "return_code": None,
        "stdout": "",
        "stderr": "",
        "decision": "not_applicable_to_orchestrator",
    }
    assert set(report["repository_commit_state"]["files"]) == {
        "registry",
        "direct_requirements",
        "complete_hash_lock",
    }
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


@pytest.mark.quick
def test_missing_complete_hash_lock_fails_before_install_and_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """精确直接输入不能替代完整锁, 缺锁时必须持久化失败并停止安装."""

    profile = get_dependency_profile("workflow_orchestrator")
    api_paths: list[Path] = []

    def read_profile(profile_id: str, path: Path) -> DependencyProfile:
        api_paths.append(Path(path))
        return profile

    def read_summary(profile_id: str, path: Path) -> dict[str, Any]:
        api_paths.append(Path(path))
        return _summary(profile)

    monkeypatch.setattr(
        preparation,
        "get_dependency_profile",
        read_profile,
    )
    monkeypatch.setattr(
        preparation,
        "build_dependency_profile_summary",
        read_summary,
    )
    monkeypatch.setattr(
        preparation,
        "require_published_formal_execution_lock",
        lambda root: _formal_execution_lock(),
    )

    def reject_not_ready(profile_id: str, path: Path) -> DependencyProfile:
        api_paths.append(Path(path))
        raise RuntimeError("complete_hash_lock_missing")

    monkeypatch.setattr(
        preparation,
        "require_dependency_profile_ready",
        reject_not_ready,
    )
    monkeypatch.setattr(
        preparation,
        "_inspect_dependency_files_commit_state",
        lambda root, selected_profile: {
            "files": {
                "registry": {"is_committed": True},
                "direct_requirements": {"is_committed": True},
                "complete_hash_lock": {"is_committed": False},
            },
            "all_committed": False,
        },
    )

    def reject_command(command: list[str], working_directory: Path) -> int:
        raise AssertionError("缺少完整哈希锁时不得调用 pip")

    report, report_path = preparation.prepare_dependency_profile(
        profile.profile_name,
        repository_root=tmp_path,
        command_runner=reject_command,
    )

    assert report["decision"] == "fail"
    assert report["failure_reasons"] == ["complete_hash_lock_missing"]
    assert report["complete_hash_lock_digest"] is None
    assert report["installation"] == {
        "attempted": False,
        "command": [],
        "return_code": None,
        "stdout": "",
        "stderr": "",
    }
    expected_registry_path = (
        tmp_path / "configs/dependency_profile_registry.json"
    ).resolve()
    assert api_paths == [expected_registry_path] * 3
    assert set(report["repository_commit_state"]["files"]) == {
        "registry",
        "direct_requirements",
        "complete_hash_lock",
    }
    assert report_path.is_file()


@pytest.mark.quick
def test_formal_execution_lock_is_required_before_dependency_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """普通分支或 dirty 工作树不得形成依赖 preparation 通过报告."""

    profile = _ready_profile()
    monkeypatch.setattr(
        preparation,
        "get_dependency_profile",
        lambda profile_id, path: profile,
    )
    monkeypatch.setattr(
        preparation,
        "build_dependency_profile_summary",
        lambda profile_id, path: _summary(profile),
    )
    monkeypatch.setattr(
        preparation,
        "require_published_formal_execution_lock",
        lambda root: (_ for _ in ()).throw(ValueError("dirty worktree")),
    )

    def reject_command(command: list[str], working_directory: Path) -> int:
        raise AssertionError("正式执行锁缺失时不得安装依赖")

    report, report_path = preparation.prepare_dependency_profile(
        profile.profile_name,
        repository_root=tmp_path,
        command_runner=reject_command,
    )

    assert report["decision"] == "fail"
    assert report["formal_execution_lock_ready"] is False
    assert report["formal_execution_lock"] == {}
    assert report["failure_reasons"] == [
        "formal_execution_lock_not_ready:ValueError"
    ]
    assert report["installation"]["attempted"] is False
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


@pytest.mark.quick
def test_uncommitted_hash_lock_is_rejected_before_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """格式有效但尚未提交或工作树已漂移的锁不能开启正式安装."""

    profile = _ready_profile()
    _bind_profile_api(monkeypatch, profile)
    monkeypatch.setattr(
        preparation,
        "_inspect_dependency_files_commit_state",
        lambda root, selected_profile: {
            "files": {
                "registry": {"is_committed": True},
                "direct_requirements": {"is_committed": True},
                "complete_hash_lock": {"is_committed": False},
            },
            "all_committed": False,
        },
    )

    def reject_command(command: list[str], working_directory: Path) -> int:
        raise AssertionError("未提交完整哈希锁时不得调用 pip")

    report, _ = preparation.prepare_dependency_profile(
        profile.profile_name,
        repository_root=tmp_path,
        command_runner=reject_command,
    )

    assert report["decision"] == "fail"
    assert report["failure_reasons"] == ["dependency_profile_inputs_not_committed"]
    assert report["repository_commit_state"]["all_committed"] is False
    assert report["installation"]["attempted"] is False


@pytest.mark.quick
def test_wrong_python_interpreter_fails_before_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """隔离 profile 必须由目标 Python 调用, 不得向错误解释器安装."""

    profile = _ready_profile()
    _bind_profile_api(monkeypatch, profile)
    monkeypatch.setattr(
        preparation,
        "_inspect_dependency_files_commit_state",
        lambda root, selected_profile: {
            "files": {
                "registry": {"is_committed": True},
                "direct_requirements": {"is_committed": True},
                "complete_hash_lock": {"is_committed": True},
            },
            "all_committed": True,
        },
    )
    mismatch = _matching_inspection(profile)
    mismatch["observed_environment"]["python_version"] = "3.9.19"
    mismatch["environment_match"] = False
    mismatch["mismatches"] = ["python_version_mismatch"]
    mismatch["readiness_blockers"] = ["python_version_mismatch"]
    mismatch["decision"] = "blocked"
    monkeypatch.setattr(
        preparation,
        "inspect_dependency_profile_environment",
        lambda profile_id, *, path, torch_module=None: mismatch,
    )

    def reject_command(command: list[str], working_directory: Path) -> int:
        raise AssertionError("错误 Python 解释器不得调用 pip")

    report, _ = preparation.prepare_dependency_profile(
        profile.profile_name,
        repository_root=tmp_path,
        command_runner=reject_command,
    )

    assert report["decision"] == "fail"
    assert report["failure_reasons"] == ["python_version_mismatch"]
    assert (
        report["runtime_comparison"]["observed_environment"]["python_version"]
        == "3.9.19"
    )
    assert report["installation"]["attempted"] is False


@pytest.mark.quick
def test_post_install_version_drift_fails_with_exact_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pip 成功后仍必须逐包精确比对, 任何版本漂移都关闭门禁."""

    profile = _ready_profile()
    _bind_profile_api(monkeypatch, profile)
    monkeypatch.setattr(
        preparation,
        "_inspect_dependency_files_commit_state",
        lambda root, selected_profile: {
            "files": {
                "registry": {"is_committed": True},
                "direct_requirements": {"is_committed": True},
                "complete_hash_lock": {"is_committed": True},
            },
            "all_committed": True,
        },
    )
    before_install = _matching_inspection(profile)
    comparison = _matching_inspection(profile)
    package_name = next(
        iter(comparison["expected_environment"]["direct_dependencies"])
    )
    comparison["observed_environment"]["direct_dependencies"][package_name] = "0.0.0"
    comparison["environment_match"] = False
    comparison["mismatches"] = [
        f"direct_dependency_version_mismatch:{package_name}"
    ]
    comparison["readiness_blockers"] = list(comparison["mismatches"])
    comparison["decision"] = "blocked"
    inspections = iter((before_install, comparison))
    monkeypatch.setattr(
        preparation,
        "inspect_dependency_profile_environment",
        lambda profile_id, *, path, torch_module=None: next(inspections),
    )

    report, report_path = preparation.prepare_dependency_profile(
        profile.profile_name,
        repository_root=tmp_path,
        command_runner=lambda command, root: 0,
    )

    assert report["decision"] == "fail"
    assert report["failure_reasons"] == [
        f"direct_dependency_version_mismatch:{package_name}"
    ]
    assert (
        report["runtime_comparison"]["observed_environment"]["direct_dependencies"][
            package_name
        ]
        == "0.0.0"
    )
    assert json.loads(report_path.read_text(encoding="utf-8"))["decision"] == "fail"


@pytest.mark.quick
def test_pip_failure_remains_closed_even_when_existing_versions_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """哈希锁安装命令失败时不得以原环境恰好匹配为由放行."""

    profile = _ready_profile()
    _bind_profile_api(monkeypatch, profile)
    monkeypatch.setattr(
        preparation,
        "_inspect_dependency_files_commit_state",
        lambda root, selected_profile: {
            "files": {
                "registry": {"is_committed": True},
                "direct_requirements": {"is_committed": True},
                "complete_hash_lock": {"is_committed": True},
            },
            "all_committed": True,
        },
    )
    monkeypatch.setattr(
        preparation,
        "inspect_dependency_profile_environment",
        lambda profile_id, *, path, torch_module=None: _matching_inspection(profile),
    )

    report, _ = preparation.prepare_dependency_profile(
        profile.profile_name,
        repository_root=tmp_path,
        command_runner=lambda command, root: 17,
    )

    assert report["decision"] == "fail"
    assert report["failure_reasons"] == ["pip_install_failed"]
    assert report["installation"]["return_code"] == 17


@pytest.mark.quick
def test_pip_check_failure_blocks_formal_preparation_and_records_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整锁安装后存在 distribution 冲突时必须保留 pip check 诊断并阻断."""

    profile = _ready_profile("sd35_method_runtime_gpu")
    _bind_profile_api(monkeypatch, profile)
    monkeypatch.setattr(
        preparation,
        "_inspect_dependency_files_commit_state",
        lambda root, selected_profile: {
            "files": {
                "registry": {"is_committed": True},
                "direct_requirements": {"is_committed": True},
                "complete_hash_lock": {"is_committed": True},
            },
            "all_committed": True,
        },
    )
    monkeypatch.setattr(
        preparation,
        "inspect_dependency_profile_environment",
        lambda profile_id, *, path, torch_module=None: _matching_inspection(profile),
    )

    def run(command: list[str], root: Path) -> dict[str, Any]:
        if command[-2:] == ["pip", "check"]:
            return {
                "return_code": 1,
                "stdout": "conflicting-package 1.0\n",
                "stderr": "dependency conflict\n",
            }
        return {"return_code": 0, "stdout": "installed\n", "stderr": ""}

    report, report_path = preparation.prepare_dependency_profile(
        profile.profile_name,
        repository_root=tmp_path,
        command_runner=run,
    )

    assert report["decision"] == "fail"
    assert report["failure_reasons"] == ["pip_check_failed"]
    assert report["installation"]["command"][
        report["installation"]["command"].index("--extra-index-url") + 1
    ] == profile.pytorch_index_url
    assert report["pip_check"]["command"] == [sys.executable, "-m", "pip", "check"]
    assert report["pip_check"]["return_code"] == 1
    assert report["pip_check"]["stdout"] == "conflicting-package 1.0\n"
    assert report["pip_check"]["stderr"] == "dependency conflict\n"
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
