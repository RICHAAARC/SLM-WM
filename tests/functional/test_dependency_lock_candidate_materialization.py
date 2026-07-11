"""验证完整 wheel 锁候选的离线解析、诊断和输出边界."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable
from urllib.parse import quote

import pytest

from experiments.runtime.dependency_profiles import (
    DependencyProfile,
    get_dependency_profile,
    parse_exact_requirement_spec,
)
import scripts.materialize_dependency_lock_candidate as materialization


PIP_VERSION = "25.1.1"
FORMAL_EXECUTION_LOCK = {
    "formal_execution_lock_schema": "clean_detached_git_commit_v1",
    "formal_execution_commit": "a" * 40,
    "formal_execution_head_detached": True,
    "formal_execution_worktree_clean": True,
    "formal_execution_lock_ready": True,
    "formal_execution_lock_digest": "b" * 64,
}


def _profile() -> DependencyProfile:
    """复用真实 registry 中经过集中校验的直接依赖身份."""

    return get_dependency_profile("workflow_orchestrator")


def _matching_interpreter(profile: DependencyProfile) -> dict[str, Any]:
    """返回与 profile 精确匹配的解释器和平台记录."""

    identity = {
        "python_implementation": profile.python_implementation,
        "python_version": profile.python_version,
        "operating_system": profile.operating_system,
        "machine": profile.machine,
    }
    return {
        "expected": dict(identity),
        "observed": dict(identity),
        "matches": True,
        "mismatches": [],
    }


def _wheel_item(
    package_name: str,
    version: str,
    *,
    requested: bool,
) -> dict[str, Any]:
    """构造与 pip installation report version 1 同形的 wheel 条目."""

    normalized_name = parse_exact_requirement_spec(
        f"{package_name}=={version}"
    ).normalized_name
    wheel_distribution = normalized_name.replace("-", "_")
    wheel_version = quote(version, safe=".!_")
    digest = hashlib.sha256(
        f"wheel-fixture:{normalized_name}=={version}".encode("utf-8")
    ).hexdigest()
    return {
        "download_info": {
            "url": (
                "https://packages.example.test/wheels/"
                f"{wheel_distribution}-{wheel_version}-py3-none-any.whl"
            ),
            "archive_info": {"hashes": {"sha256": digest}},
        },
        "is_direct": False,
        "is_yanked": False,
        "requested": requested,
        "metadata": {
            "name": package_name,
            "version": version,
        },
    }


def _pip_report(profile: DependencyProfile) -> dict[str, Any]:
    """构造覆盖全部直接依赖并包含一个传递依赖的真实结构报告."""

    install = []
    for specification in profile.direct_requirements:
        dependency = parse_exact_requirement_spec(specification)
        install.append(
            _wheel_item(
                dependency.package_name,
                dependency.version,
                requested=True,
            )
        )
    install.append(_wheel_item("typing_extensions", "4.12.2", requested=False))
    return {
        "version": "1",
        "pip_version": PIP_VERSION,
        "install": install,
        "environment": {
            "implementation_name": "cpython",
            "implementation_version": profile.python_version,
            "os_name": "posix",
            "platform_machine": profile.machine,
            "platform_python_implementation": profile.python_implementation,
            "python_full_version": profile.python_version,
            "python_version": ".".join(profile.python_version.split(".")[:2]),
            "sys_platform": profile.operating_system,
        },
    }


def _bind_profile(
    monkeypatch: pytest.MonkeyPatch,
    profile: DependencyProfile,
    observed_registry_paths: list[Path] | None = None,
) -> None:
    """让 CLI 仅通过共享 profile API 取得同一测试记录."""

    def read_profile(profile_id: str, path: Path) -> DependencyProfile:
        if observed_registry_paths is not None:
            observed_registry_paths.append(Path(path))
        return profile

    monkeypatch.setattr(materialization, "get_dependency_profile", read_profile)
    monkeypatch.setattr(
        materialization,
        "_inspect_current_interpreter",
        _matching_interpreter,
    )
    monkeypatch.setattr(
        materialization,
        "_read_current_pip_version",
        lambda: PIP_VERSION,
    )
    monkeypatch.setattr(
        materialization.repository_environment,
        "require_published_formal_execution_lock",
        lambda root: dict(FORMAL_EXECUTION_LOCK),
    )


def _report_writing_runner(
    report: dict[str, Any] | str,
    observed_commands: list[list[str]] | None = None,
    *,
    return_code: int = 0,
    stdout: str = "解析完成.\n",
    stderr: str = "",
) -> materialization.CommandRunner:
    """返回不访问网络且按命令 ``--report`` 路径写入 fixture 的 runner."""

    def run(command: list[str], working_directory: Path) -> materialization.CommandExecution:
        command_list = list(command)
        if observed_commands is not None:
            observed_commands.append(command_list)
        report_path = Path(command_list[command_list.index("--report") + 1])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(report, str):
            report_path.write_text(report, encoding="utf-8")
        else:
            report_path.write_text(json.dumps(report), encoding="utf-8")
        return materialization.CommandExecution(
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
        )

    return run


def _candidate_rows(report: dict[str, Any]) -> list[dict[str, str]]:
    """独立生成预期的候选摘要记录, 避免测试复述被测摘要函数."""

    rows = []
    for item in report["install"]:
        dependency = parse_exact_requirement_spec(
            f"{item['metadata']['name']}=={item['metadata']['version']}"
        )
        rows.append(
            {
                "package_name": dependency.normalized_name,
                "version": dependency.version,
                "sha256_digests": [
                    item["download_info"]["archive_info"]["hashes"]["sha256"]
                ],
            }
        )
    return sorted(rows, key=lambda row: row["package_name"])


@pytest.mark.quick
def test_matching_profile_materializes_sorted_candidate_from_pip_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """匹配环境应使用精确解析命令并只在 outputs 中生成审查候选."""

    profile = _profile()
    assert profile.accelerator_runtime == "cpu"
    assert profile.pytorch_index_url is None
    registry_paths: list[Path] = []
    _bind_profile(monkeypatch, profile, registry_paths)
    pip_report = _pip_report(profile)
    commands: list[list[str]] = []

    provenance, provenance_path = (
        materialization.materialize_dependency_lock_candidate(
            profile.profile_name,
            repository_root=tmp_path,
            command_runner=_report_writing_runner(pip_report, commands),
        )
    )

    output_root = (
        tmp_path / "outputs/dependency_lock_candidates/workflow_orchestrator"
    ).resolve()
    pip_report_path = output_root / "pip_resolver_report.json"
    candidate_path = output_root / "dependency_lock_candidate.txt"
    direct_path = (tmp_path / profile.direct_requirements_path).resolve()
    assert commands == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--only-binary=:all:",
            "--report",
            str(pip_report_path),
            "-r",
            str(direct_path),
        ]
    ]
    assert registry_paths == [
        (tmp_path / "configs/dependency_profile_registry.json").resolve()
    ]
    expected_rows = _candidate_rows(pip_report)
    expected_lines = [
        f"{row['package_name']}=={row['version']} "
        f"--hash=sha256:{row['sha256_digests'][0]}"
        for row in expected_rows
    ]
    assert candidate_path.read_text(encoding="utf-8").splitlines() == expected_lines
    expected_digest = hashlib.sha256(
        json.dumps(
            expected_rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert provenance_path == output_root / "dependency_lock_candidate_provenance.json"
    assert provenance["decision"] == "candidate_ready_for_review"
    assert provenance["candidate_lock_dependency_count"] == len(expected_rows)
    assert provenance["candidate_lock_logical_digest"] == expected_digest
    assert provenance["profile_digest"] == profile.profile_digest
    assert provenance["direct_requirements_digest"] == profile.direct_requirements_digest
    assert provenance["formal_execution_lock"] == FORMAL_EXECUTION_LOCK
    assert provenance["formal_execution_commit"] == "a" * 40
    assert provenance["formal_execution_lock_digest"] == "b" * 64
    assert provenance["pip_version"] == PIP_VERSION
    assert provenance["supports_paper_claim"] is False
    assert not (tmp_path / profile.complete_hash_lock_path).exists()
    assert json.loads(provenance_path.read_text(encoding="utf-8")) == provenance


@pytest.mark.quick
def test_unpublished_execution_lock_blocks_resolver_and_keeps_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未发布 clean detached 代码身份锁时不得启动 pip 解析."""

    profile = _profile()
    _bind_profile(monkeypatch, profile)
    observed_roots: list[Path] = []

    def reject_execution_lock(root: Path) -> dict[str, Any]:
        observed_roots.append(Path(root))
        raise materialization.repository_environment.FormalExecutionLockError(
            "没有已发布执行锁."
        )

    monkeypatch.setattr(
        materialization.repository_environment,
        "require_published_formal_execution_lock",
        reject_execution_lock,
    )

    def reject_command(command: list[str], working_directory: Path) -> materialization.CommandExecution:
        raise AssertionError("代码身份锁未通过时不得调用 pip")

    provenance, provenance_path = (
        materialization.materialize_dependency_lock_candidate(
            profile.profile_name,
            repository_root=tmp_path,
            command_runner=reject_command,
        )
    )

    assert observed_roots == [tmp_path.resolve()]
    assert provenance["decision"] == "fail"
    assert provenance["failure_reasons"] == ["formal_execution_lock_unavailable"]
    assert provenance["diagnostic_message"] == "没有已发布执行锁."
    assert provenance["formal_execution_lock"] == {}
    assert provenance["formal_execution_commit"] == ""
    assert provenance["formal_execution_lock_digest"] == ""
    assert provenance["resolver_return_code"] is None
    assert json.loads(provenance_path.read_text(encoding="utf-8")) == provenance
    assert not provenance_path.with_name("dependency_lock_candidate.txt").exists()


@pytest.mark.quick
def test_interpreter_mismatch_fails_before_resolver_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python patch 或平台身份不匹配时不得启动 pip 解析."""

    profile = _profile()
    _bind_profile(monkeypatch, profile)
    mismatch = _matching_interpreter(profile)
    mismatch["observed"]["python_version"] = "3.12.12"
    mismatch["matches"] = False
    mismatch["mismatches"] = ["python_version_mismatch"]
    monkeypatch.setattr(
        materialization,
        "_inspect_current_interpreter",
        lambda selected_profile: mismatch,
    )

    def reject_command(command: list[str], working_directory: Path) -> materialization.CommandExecution:
        raise AssertionError("解释器身份不匹配时不得调用 pip")

    provenance, provenance_path = (
        materialization.materialize_dependency_lock_candidate(
            profile.profile_name,
            repository_root=tmp_path,
            command_runner=reject_command,
        )
    )

    assert provenance["decision"] == "fail"
    assert provenance["failure_reasons"] == ["python_version_mismatch"]
    assert provenance["resolver_return_code"] is None
    assert provenance_path.is_file()
    assert not provenance_path.with_name("dependency_lock_candidate.txt").exists()


@pytest.mark.quick
def test_resolver_failure_preserves_console_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """网络或依赖解析失败必须返回失败并持久化 pip 诊断."""

    profile = _profile()
    _bind_profile(monkeypatch, profile)
    provenance, provenance_path = (
        materialization.materialize_dependency_lock_candidate(
            profile.profile_name,
            repository_root=tmp_path,
            command_runner=_report_writing_runner(
                _pip_report(profile),
                return_code=1,
                stdout="正在解析.\n",
                stderr="网络索引不可用.\n",
            ),
        )
    )

    assert provenance["decision"] == "fail"
    assert provenance["failure_reasons"] == ["resolver_command_failed"]
    assert provenance["resolver_return_code"] == 1
    assert provenance["resolver_stdout"] == "正在解析.\n"
    assert provenance["resolver_stderr"] == "网络索引不可用.\n"
    assert json.loads(provenance_path.read_text(encoding="utf-8")) == provenance
    assert not provenance_path.with_name("dependency_lock_candidate.txt").exists()


def _mutate_sdist(report: dict[str, Any]) -> None:
    """把首个 wheel URL 改成 sdist URL."""

    report["install"][0]["download_info"]["url"] = (
        "https://packages.example.test/source/package-1.0.tar.gz"
    )


def _mutate_vcs(report: dict[str, Any]) -> None:
    """向首个条目注入 VCS 来源."""

    report["install"][0]["download_info"]["vcs_info"] = {
        "vcs": "git",
        "commit_id": "a" * 40,
    }


def _mutate_missing_hash(report: dict[str, Any]) -> None:
    """删除首个 wheel 的 SHA-256."""

    report["install"][0]["download_info"]["archive_info"]["hashes"] = {}


@pytest.mark.quick
@pytest.mark.parametrize(
    "mutator",
    (_mutate_sdist, _mutate_vcs, _mutate_missing_hash),
    ids=("sdist", "vcs", "missing_sha256"),
)
def test_non_wheel_or_unhashed_report_item_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    """sdist、VCS 和缺少真实 SHA-256 的条目均不得进入候选锁."""

    profile = _profile()
    _bind_profile(monkeypatch, profile)
    pip_report = _pip_report(profile)
    mutator(pip_report)

    provenance, provenance_path = (
        materialization.materialize_dependency_lock_candidate(
            profile.profile_name,
            repository_root=tmp_path,
            command_runner=_report_writing_runner(pip_report),
        )
    )

    assert provenance["decision"] == "fail"
    assert provenance["failure_reasons"] == ["pip_resolver_report_rejected"]
    assert provenance["diagnostic_message"]
    assert provenance_path.with_name("pip_resolver_report.json").is_file()
    assert not provenance_path.with_name("dependency_lock_candidate.txt").exists()


@pytest.mark.quick
@pytest.mark.parametrize("report_defect", ("duplicate", "direct_missing"))
def test_duplicate_or_incomplete_resolution_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_defect: str,
) -> None:
    """重复分发名或未覆盖直接输入的解析结果必须失败闭合."""

    profile = _profile()
    _bind_profile(monkeypatch, profile)
    pip_report = _pip_report(profile)
    if report_defect == "duplicate":
        pip_report["install"].append(dict(pip_report["install"][0]))
    else:
        pip_report["install"].pop(0)

    provenance, provenance_path = (
        materialization.materialize_dependency_lock_candidate(
            profile.profile_name,
            repository_root=tmp_path,
            command_runner=_report_writing_runner(pip_report),
        )
    )

    assert provenance["decision"] == "fail"
    assert provenance["failure_reasons"] == ["pip_resolver_report_rejected"]
    assert not provenance_path.with_name("dependency_lock_candidate.txt").exists()


@pytest.mark.quick
def test_malformed_pip_report_is_preserved_but_not_materialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """损坏的 pip JSON 报告应保留供诊断, 但不得产生候选锁."""

    profile = _profile()
    _bind_profile(monkeypatch, profile)
    provenance, provenance_path = (
        materialization.materialize_dependency_lock_candidate(
            profile.profile_name,
            repository_root=tmp_path,
            command_runner=_report_writing_runner("{not-json"),
        )
    )

    assert provenance["decision"] == "fail"
    assert provenance["failure_reasons"] == ["pip_resolver_report_rejected"]
    assert "不是有效 JSON" in provenance["diagnostic_message"]
    assert provenance_path.with_name("pip_resolver_report.json").read_text(
        encoding="utf-8"
    ) == "{not-json"
    assert not provenance_path.with_name("dependency_lock_candidate.txt").exists()
