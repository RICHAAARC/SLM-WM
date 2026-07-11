"""验证 uv 驱动的 隔离精确 Python provision 与正式环境准备."""

from __future__ import annotations

import ast
import base64
from dataclasses import replace
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from experiments.runtime import isolated_dependency_environment as isolated
from experiments.runtime.dependency_profiles import DependencyProfile, get_dependency_profile


def _ready_profile(profile_id: str, digest_token: str) -> DependencyProfile:
    """由真实 registry profile 派生带完整锁的测试记录."""

    profile = get_dependency_profile(profile_id)
    return replace(
        profile,
        complete_hash_lock_present=True,
        complete_hash_lock_digest=digest_token * 64,
        complete_hash_lock_dependency_count=len(profile.direct_requirements) + 7,
        formal_ready=True,
        readiness_blockers=(),
    )


def _orchestrator_inspection(profile: DependencyProfile) -> dict[str, Any]:
    """构造父解释器完全匹配的 inspection 记录."""

    return {
        "profile_name": profile.profile_name,
        "profile_digest": profile.profile_digest,
        "complete_hash_lock_digest": profile.complete_hash_lock_digest,
        "profile_formal_ready": True,
        "environment_match": True,
        "mismatches": [],
        "readiness_blockers": [],
        "decision": "pass",
        "inspection_digest": "i" * 64,
    }


def _formal_execution_lock() -> dict[str, Any]:
    """构造隔离环境报告必须传播的正式执行锁."""

    return {
        "formal_execution_lock_schema": "formal_execution_lock",
        "formal_execution_commit": "a" * 40,
        "formal_execution_head_detached": True,
        "formal_execution_worktree_clean": True,
        "formal_execution_lock_ready": True,
        "formal_execution_lock_digest": "f" * 64,
    }


def _dependency_preparation_report(
    profile: DependencyProfile,
    python_executable: Path,
) -> dict[str, Any]:
    """构造内层 dependency preparation 的严格通过报告."""

    return {
        "report_schema": "dependency_profile_preparation_report",
        "schema_version": 1,
        "profile_id": profile.profile_name,
        "python_executable": str(python_executable),
        "profile_digest": profile.profile_digest,
        "direct_requirements_digest": profile.direct_requirements_digest,
        "complete_hash_lock_digest": profile.complete_hash_lock_digest,
        "complete_hash_lock_dependency_count": profile.complete_hash_lock_dependency_count,
        "formal_ready": True,
        "repository_commit_state": {"all_committed": True},
        "installation": {"attempted": True, "return_code": 0},
        "pip_check": {
            "compatibility_check_required": True,
            "attempted": True,
            "return_code": 0,
            "decision": "pass",
        },
        "runtime_comparison": {
            "decision": "pass",
            "profile_digest": profile.profile_digest,
            "complete_hash_lock_digest": profile.complete_hash_lock_digest,
        },
        "decision": "pass",
        "failure_reasons": [],
        "supports_paper_claim": False,
        "formal_execution_lock": _formal_execution_lock(),
        "formal_execution_commit": _formal_execution_lock()["formal_execution_commit"],
        "formal_execution_lock_digest": _formal_execution_lock()[
            "formal_execution_lock_digest"
        ],
        "formal_execution_lock_ready": True,
    }


def _bind_provision_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target_profile: DependencyProfile,
    orchestrator_profile: DependencyProfile,
) -> None:
    """绑定 profile 查询、父环境 inspection 和 uv distribution 版本."""

    monkeypatch.setattr(
        isolated,
        "get_dependency_profile",
        lambda profile_id, path: target_profile,
    )
    monkeypatch.setattr(
        isolated,
        "_validate_orchestrator_environment",
        lambda path: (
            orchestrator_profile,
            _orchestrator_inspection(orchestrator_profile),
        ),
    )
    monkeypatch.setattr(
        isolated,
        "_read_uv_distribution_version",
        lambda: isolated.UV_DISTRIBUTION_VERSION,
    )
    monkeypatch.setattr(
        isolated,
        "_inspect_uv_executable_distribution_source",
        lambda path: {
            "uv_distribution_record_path": "/fixture/uv.dist-info/RECORD",
            "uv_distribution_record_sha256": "r" * 64,
            "uv_distribution_executable_record_path": "../../../bin/uv",
            "uv_distribution_executable_record_sha256": isolated._file_sha256(
                Path(path)
            ),
        },
    )
    monkeypatch.setattr(
        isolated,
        "require_published_formal_execution_lock",
        lambda root: _formal_execution_lock(),
    )


def _provision_runner(
    *,
    profile: DependencyProfile,
    python_executable: Path,
    commands: list[tuple[list[str], Path, dict[str, str]]],
) -> isolated.CommandRunner:
    """返回可物化测试 Python 文件的 argv runner."""

    def run(
        argv: Sequence[str],
        working_directory: Path,
        environment_overrides: Mapping[str, str],
    ) -> dict[str, Any]:
        normalized = [str(token) for token in argv]
        commands.append((normalized, working_directory, dict(environment_overrides)))
        if normalized[1:] == ["--version"]:
            return {
                "return_code": 0,
                "stdout": f"uv {isolated.UV_DISTRIBUTION_VERSION}\n",
                "stderr": "",
            }
        if len(normalized) >= 2 and normalized[1] == "venv":
            python_executable.parent.mkdir(parents=True, exist_ok=True)
            python_executable.write_bytes(b"deterministic-test-python")
        if normalized[:2] == [str(python_executable), "-c"]:
            return {
                "return_code": 0,
                "stdout": profile.python_version + "\n",
                "stderr": "",
            }
        return {"return_code": 0, "stdout": "", "stderr": ""}

    return run


@pytest.mark.quick
def test_provision_creates_exact_python_without_requiring_target_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """provision 只形成候选锁物化解释器, 不得宣称正式环境 ready."""

    target_profile = get_dependency_profile("tree_ring_official_py39_cu117")
    orchestrator_profile = _ready_profile("workflow_orchestrator", "o")
    _bind_provision_prerequisites(
        monkeypatch,
        target_profile=target_profile,
        orchestrator_profile=orchestrator_profile,
    )
    environment_root = tmp_path / "dependency_envs"
    managed_python_root = tmp_path / "managed_pythons"
    python_executable = (
        environment_root / target_profile.profile_name / "bin" / "python"
    ).resolve()
    uv_executable = tmp_path / "bin/uv"
    uv_executable.parent.mkdir(parents=True)
    uv_executable.write_bytes(b"pinned-uv-executable")
    commands: list[tuple[list[str], Path, dict[str, str]]] = []

    report, report_path = isolated.provision_isolated_dependency_python(
        target_profile.profile_name,
        repository_root=tmp_path,
        environment_root=environment_root,
        managed_python_root=managed_python_root,
        uv_executable_path=uv_executable,
        command_runner=_provision_runner(
            profile=target_profile,
            python_executable=python_executable,
            commands=commands,
        ),
    )

    resolved_uv = uv_executable.resolve()
    assert [command[0] for command in commands] == [
        [str(resolved_uv), "--version"],
        [
            str(resolved_uv),
            "python",
            "install",
            "3.9.19",
            "--install-dir",
            str(managed_python_root.resolve()),
        ],
        [
            str(resolved_uv),
            "venv",
            "--clear",
            "--python",
            "3.9.19",
            "--managed-python",
            str((environment_root / target_profile.profile_name).resolve()),
        ],
        [str(python_executable), "-m", "ensurepip"],
        [
            str(python_executable),
            "-c",
            "import platform; print(platform.python_version())",
        ],
    ]
    assert all(
        environment == {
            "UV_PYTHON_INSTALL_DIR": str(managed_python_root.resolve())
        }
        for _, _, environment in commands
    )
    assert report["decision"] == "provisioned"
    assert report["provisioned"] is True
    assert report["formal_ready"] is False
    assert report["target_complete_hash_lock_ready"] is False
    assert report["supports_paper_claim"] is False
    assert report["formal_execution_lock"] == _formal_execution_lock()
    assert report["formal_execution_lock_ready"] is True
    assert report["uv_distribution_version"] == "0.11.28"
    assert report["uv_reported_version"] == "0.11.28"
    assert report["uv_executable_path"] == str(resolved_uv)
    assert report["uv_executable_sha256"] == hashlib.sha256(
        b"pinned-uv-executable"
    ).hexdigest()
    assert report["uv_distribution_record_sha256"] == "r" * 64
    assert report["uv_distribution_executable_record_path"] == "../../../bin/uv"
    assert (
        report["uv_distribution_executable_record_sha256"]
        == report["uv_executable_sha256"]
    )
    assert report["python_executable_path"] == str(python_executable)
    assert report["python_executable_sha256"] == hashlib.sha256(
        b"deterministic-test-python"
    ).hexdigest()
    assert len(report["uv_commands"]) == 3
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


@pytest.mark.quick
def test_formal_prepare_requires_target_lock_and_validates_child_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式 API 必须调用隔离解释器内层模块并闭合全部摘要与通过结论."""

    target_profile = _ready_profile("gaussian_shading_official_py38_cu117", "t")
    orchestrator_profile = _ready_profile("workflow_orchestrator", "o")
    _bind_provision_prerequisites(
        monkeypatch,
        target_profile=target_profile,
        orchestrator_profile=orchestrator_profile,
    )
    monkeypatch.setattr(
        isolated,
        "require_dependency_profile_ready",
        lambda profile_id, path: (
            target_profile if profile_id == target_profile.profile_name else orchestrator_profile
        ),
    )
    environment_root = tmp_path / "dependency_envs"
    managed_python_root = tmp_path / "managed_pythons"
    python_executable = (
        environment_root / target_profile.profile_name / "bin" / "python"
    ).resolve()
    uv_executable = tmp_path / "bin/uv"
    uv_executable.parent.mkdir(parents=True)
    uv_executable.write_bytes(b"pinned-uv-executable")
    commands: list[tuple[list[str], Path, dict[str, str]]] = []
    base_runner = _provision_runner(
        profile=target_profile,
        python_executable=python_executable,
        commands=commands,
    )

    def run(
        argv: Sequence[str],
        working_directory: Path,
        environment_overrides: Mapping[str, str],
    ) -> dict[str, Any]:
        normalized = [str(token) for token in argv]
        if "experiments.runtime.dependency_preparation" in normalized:
            commands.append((normalized, working_directory, dict(environment_overrides)))
            dependency_report_path = (
                tmp_path
                / "outputs/dependency_profiles"
                / target_profile.profile_name
                / "dependency_profile_report.json"
            )
            dependency_report_path.parent.mkdir(parents=True, exist_ok=True)
            dependency_report_path.write_text(
                json.dumps(
                    _dependency_preparation_report(target_profile, python_executable),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            return {"return_code": 0, "stdout": "", "stderr": ""}
        return base_runner(normalized, working_directory, environment_overrides)

    report, report_path = isolated.prepare_isolated_dependency_environment(
        target_profile.profile_name,
        repository_root=tmp_path,
        environment_root=environment_root,
        managed_python_root=managed_python_root,
        uv_executable_path=uv_executable,
        command_runner=run,
    )

    expected_prepare_argv = [
        str(python_executable),
        "-m",
        "experiments.runtime.dependency_preparation",
        "--profile",
        target_profile.profile_name,
    ]
    assert commands[-1][0] == expected_prepare_argv
    assert report["decision"] == "pass"
    assert report["provisioned"] is True
    assert report["formal_preparation_completed"] is True
    assert report["formal_ready"] is True
    assert report["supports_paper_claim"] is False
    assert report["profile_digest"] == target_profile.profile_digest
    assert report["complete_hash_lock_digest"] == target_profile.complete_hash_lock_digest
    assert report["dependency_preparation_report"]["decision"] == "pass"
    assert report["python_executable_sha256_after_preparation"] == report[
        "python_executable_sha256"
    ]
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


@pytest.mark.quick
def test_formal_prepare_blocks_before_provision_when_target_lock_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式准备缺少目标锁时不得创建 Python 或执行 uv."""

    target_profile = get_dependency_profile("shallow_diffuse_official_py39_cu117")
    monkeypatch.setattr(
        isolated,
        "get_dependency_profile",
        lambda profile_id, path: target_profile,
    )
    monkeypatch.setattr(
        isolated,
        "require_published_formal_execution_lock",
        lambda root: _formal_execution_lock(),
    )
    monkeypatch.setattr(
        isolated,
        "require_dependency_profile_ready",
        lambda profile_id, path: (_ for _ in ()).throw(
            RuntimeError("complete_hash_lock_missing")
        ),
    )

    def reject_runner(
        argv: Sequence[str],
        working_directory: Path,
        environment_overrides: Mapping[str, str],
    ) -> int:
        raise AssertionError("目标完整锁缺失时不得执行任何命令")

    report, report_path = isolated.prepare_isolated_dependency_environment(
        target_profile.profile_name,
        repository_root=tmp_path,
        environment_root=tmp_path / "dependency_envs",
        managed_python_root=tmp_path / "managed_pythons",
        command_runner=reject_runner,
    )

    assert report["decision"] == "fail"
    assert report["provisioned"] is False
    assert report["formal_ready"] is False
    assert report["failure_reasons"] == ["target_profile_not_ready:RuntimeError"]
    assert report_path.is_file()


@pytest.mark.quick
def test_provision_rejects_uv_distribution_drift_before_any_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """当前 uv distribution 不是固定版本时不得信任可执行文件或创建 venv."""

    target_profile = get_dependency_profile("tree_ring_official_py39_cu117")
    orchestrator_profile = _ready_profile("workflow_orchestrator", "o")
    _bind_provision_prerequisites(
        monkeypatch,
        target_profile=target_profile,
        orchestrator_profile=orchestrator_profile,
    )
    monkeypatch.setattr(isolated, "_read_uv_distribution_version", lambda: "0.11.27")

    def reject_runner(
        argv: Sequence[str],
        working_directory: Path,
        environment_overrides: Mapping[str, str],
    ) -> int:
        raise AssertionError("uv distribution 漂移时不得执行命令")

    report, _ = isolated.provision_isolated_dependency_python(
        target_profile.profile_name,
        repository_root=tmp_path,
        environment_root=tmp_path / "dependency_envs",
        managed_python_root=tmp_path / "managed_pythons",
        command_runner=reject_runner,
    )

    assert report["decision"] == "fail"
    assert report["failure_reasons"] == ["uv_distribution_version_mismatch"]
    assert report["command_results"] == []


@pytest.mark.quick
def test_provision_rejects_same_version_uv_path_outside_distribution_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PATH 中伪造同版本 uv 时必须在任何子环境命令前失败闭合."""

    target_profile = get_dependency_profile("sd35_method_runtime_gpu")
    orchestrator_profile = _ready_profile("workflow_orchestrator", "o")
    _bind_provision_prerequisites(
        monkeypatch,
        target_profile=target_profile,
        orchestrator_profile=orchestrator_profile,
    )
    fake_uv = tmp_path / "spoofed_path/uv"
    fake_uv.parent.mkdir(parents=True)
    fake_uv.write_bytes(b"same-version-path-spoof")
    monkeypatch.setattr(
        isolated,
        "_inspect_uv_executable_distribution_source",
        lambda path: (_ for _ in ()).throw(
            ValueError("不在 uv distribution RECORD 中")
        ),
    )

    def reject_runner(
        argv: Sequence[str],
        working_directory: Path,
        environment_overrides: Mapping[str, str],
    ) -> int:
        raise AssertionError("伪造 uv 在来源门禁失败后不得执行")

    report, report_path = isolated.provision_isolated_dependency_python(
        target_profile.profile_name,
        repository_root=tmp_path,
        uv_executable_path=fake_uv,
        command_runner=reject_runner,
    )

    assert report["decision"] == "fail"
    assert report["failure_reasons"] == [
        "uv_executable_distribution_source_mismatch"
    ]
    assert report["uv_executable_path"] == str(fake_uv.resolve())
    assert report["command_results"] == []
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


@pytest.mark.quick
def test_uv_distribution_source_inspection_matches_recorded_executable_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """来源检查必须同时验证 RECORD 路径成员关系与记录的文件摘要."""

    uv_executable = tmp_path / "bin/uv"
    uv_executable.parent.mkdir(parents=True)
    uv_executable.write_bytes(b"record-managed-uv")
    record_path = tmp_path / "site-packages/uv-0.11.28.dist-info/RECORD"
    record_path.parent.mkdir(parents=True)
    record_path.write_text("fixture RECORD\n", encoding="utf-8")
    digest_bytes = hashlib.sha256(uv_executable.read_bytes()).digest()
    record_digest = base64.urlsafe_b64encode(digest_bytes).decode("ascii").rstrip("=")

    class DistributionPath:
        """提供 importlib metadata PackagePath 所需的最小只读接口."""

        def __init__(self, value: str, hash_record: Any) -> None:
            self.value = PurePosixPath(value)
            self.hash = hash_record

        @property
        def name(self) -> str:
            return self.value.name

        @property
        def parent(self) -> PurePosixPath:
            return self.value.parent

        def as_posix(self) -> str:
            return self.value.as_posix()

    executable_record = DistributionPath(
        "../../../bin/uv",
        SimpleNamespace(mode="sha256", value=record_digest),
    )
    metadata_record = DistributionPath(
        "uv-0.11.28.dist-info/RECORD",
        None,
    )
    located_paths = {
        executable_record.as_posix(): uv_executable,
        metadata_record.as_posix(): record_path,
    }
    distribution = SimpleNamespace(
        version=isolated.UV_DISTRIBUTION_VERSION,
        files=(executable_record, metadata_record),
        locate_file=lambda path: located_paths[path.as_posix()],
    )
    monkeypatch.setattr(
        isolated.importlib_metadata,
        "distribution",
        lambda name: distribution,
    )

    source = isolated._inspect_uv_executable_distribution_source(uv_executable)

    assert source["uv_distribution_record_path"] == str(record_path.resolve())
    assert source["uv_distribution_executable_record_path"] == "../../../bin/uv"
    assert source["uv_distribution_executable_record_sha256"] == hashlib.sha256(
        uv_executable.read_bytes()
    ).hexdigest()
    spoofed_uv = tmp_path / "other/uv"
    spoofed_uv.parent.mkdir()
    spoofed_uv.write_bytes(uv_executable.read_bytes())
    with pytest.raises(ValueError, match="不属于当前解释器"):
        isolated._inspect_uv_executable_distribution_source(spoofed_uv)


@pytest.mark.quick
def test_provision_requires_formal_execution_lock_before_uv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """隔离 Python provision 也必须绑定 clean detached commit."""

    target_profile = get_dependency_profile("sd35_method_runtime_gpu")
    monkeypatch.setattr(
        isolated,
        "get_dependency_profile",
        lambda profile_id, path: target_profile,
    )
    monkeypatch.setattr(
        isolated,
        "require_published_formal_execution_lock",
        lambda root: (_ for _ in ()).throw(ValueError("branch checkout")),
    )

    def reject_runner(
        argv: Sequence[str],
        working_directory: Path,
        environment_overrides: Mapping[str, str],
    ) -> int:
        raise AssertionError("正式执行锁缺失时不得调用 uv")

    report, report_path = isolated.provision_isolated_dependency_python(
        target_profile.profile_name,
        repository_root=tmp_path,
        environment_root=tmp_path / "dependency_envs",
        managed_python_root=tmp_path / "managed_pythons",
        command_runner=reject_runner,
    )

    assert report["decision"] == "fail"
    assert report["formal_execution_lock_ready"] is False
    assert report["failure_reasons"] == [
        "formal_execution_lock_not_ready:ValueError"
    ]
    assert report["command_results"] == []
    assert report_path.is_file()


@pytest.mark.quick
def test_isolated_environment_paths_use_cross_server_temporary_root() -> None:
    """默认路径不得在 experiments 层内置 Colab 专用目录."""

    for profile_id in isolated.ISOLATED_DEPENDENCY_PROFILE_IDS:
        assert isolated.isolated_python_executable_path(profile_id).as_posix() == (
            (isolated.DEFAULT_ENVIRONMENT_ROOT / profile_id / "bin/python").as_posix()
        )
    assert "/content/" not in isolated.DEFAULT_ENVIRONMENT_ROOT.as_posix()


@pytest.mark.quick
def test_scripts_are_thin_forwarders_to_experiments_runtime() -> None:
    """scripts 层只能转发 CLI, 不得保存安装或隔离环境业务实现."""

    dependency_script = Path("scripts/prepare_dependency_profile.py").read_text(
        encoding="utf-8"
    )
    isolated_script = Path(
        "scripts/prepare_isolated_dependency_environment.py"
    ).read_text(encoding="utf-8")

    assert "experiments.runtime.dependency_preparation import main" in dependency_script
    assert "experiments.runtime.isolated_dependency_environment import main" in isolated_script
    for source in (dependency_script, isolated_script):
        assert "subprocess" not in source
        assert "pip" not in source
        assert "uv venv" not in source


@pytest.mark.quick
def test_isolated_child_runtime_sources_parse_with_python38_grammar() -> None:
    """Python 3.8 子环境会导入的共享模块不得使用更新版本专属语法."""

    repository_root = Path(__file__).resolve().parents[2]
    runtime_sources = (
        "experiments/runtime/dependency_profiles.py",
        "experiments/runtime/dependency_preparation.py",
        "experiments/runtime/repository_environment.py",
        "main/core/digest.py",
    )
    for relative_path in runtime_sources:
        source = (repository_root / relative_path).read_text(encoding="utf-8-sig")
        ast.parse(source, filename=relative_path, feature_version=(3, 8))
