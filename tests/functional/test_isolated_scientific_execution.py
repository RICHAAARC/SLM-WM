"""验证隔离科学子解释器的环境门禁、命令前缀和执行证据."""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from experiments.runtime import isolated_scientific_execution as execution
from experiments.runtime import repository_environment
from experiments.runtime.dependency_profiles import DependencyProfile, get_dependency_profile


def _formal_execution_lock(digest_character: str = "f") -> dict[str, Any]:
    """构造可在三次实时复核中传播的正式执行锁."""

    return {
        "formal_execution_lock_schema": "formal_execution_lock",
        "formal_execution_commit": "a" * 40,
        "formal_execution_head_detached": True,
        "formal_execution_worktree_clean": True,
        "formal_execution_lock_ready": True,
        "formal_execution_lock_digest": digest_character * 64,
    }


def _ready_profile(profile_id: str = "sd35_method_runtime_gpu") -> DependencyProfile:
    """由真实 registry 记录派生带完整哈希锁的测试 profile."""

    profile = get_dependency_profile(profile_id)
    return replace(
        profile,
        complete_hash_lock_present=True,
        complete_hash_lock_digest="b" * 64,
        complete_hash_lock_dependency_count=len(profile.direct_requirements) + 9,
        formal_ready=True,
        readiness_blockers=(),
    )


def _environment_report(
    profile: DependencyProfile,
    python_executable: Path,
    formal_execution_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """构造共享隔离环境 API 的严格通过报告."""

    python_digest = hashlib.sha256(python_executable.read_bytes()).hexdigest()
    dependency_preparation_report = {
        "profile_id": profile.profile_name,
        "profile_digest": profile.profile_digest,
        "complete_hash_lock_digest": profile.complete_hash_lock_digest,
        "python_executable": str(python_executable),
        "formal_execution_lock": dict(formal_execution_lock),
        "formal_ready": True,
        "decision": "pass",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    return {
        "report_schema": execution.DEPENDENCY_ENVIRONMENT_REPORT_SCHEMA,
        "schema_version": execution.DEPENDENCY_ENVIRONMENT_REPORT_SCHEMA_VERSION,
        "operation_kind": "formal_dependency_environment_preparation",
        "profile_id": profile.profile_name,
        "profile_digest": profile.profile_digest,
        "direct_requirements_digest": profile.direct_requirements_digest,
        "complete_hash_lock_digest": profile.complete_hash_lock_digest,
        "complete_hash_lock_dependency_count": profile.complete_hash_lock_dependency_count,
        "formal_execution_lock": dict(formal_execution_lock),
        "formal_execution_commit": formal_execution_lock["formal_execution_commit"],
        "formal_execution_lock_digest": formal_execution_lock[
            "formal_execution_lock_digest"
        ],
        "formal_execution_lock_ready": True,
        "provisioned": True,
        "formal_preparation_completed": True,
        "formal_ready": True,
        "decision": "pass",
        "failure_reasons": [],
        "supports_paper_claim": False,
        "python_executable_path": str(python_executable),
        "python_executable_sha256": python_digest,
        "python_executable_sha256_after_preparation": python_digest,
        "dependency_preparation_report": dependency_preparation_report,
    }


def _write_environment_report(
    repository_root: Path,
    profile: DependencyProfile,
    environment_root: Path,
    formal_execution_lock: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    """物化测试解释器与隔离环境报告."""

    python_executable = execution.isolated_python_executable_path(
        profile.profile_name,
        environment_root=environment_root,
    )
    python_executable.parent.mkdir(parents=True, exist_ok=True)
    python_executable.write_bytes(b"isolated-scientific-python")
    report = _environment_report(
        profile,
        python_executable,
        formal_execution_lock,
    )
    report_path = (
        repository_root
        / "outputs"
        / "dependency_profiles"
        / profile.profile_name
        / execution.DEPENDENCY_ENVIRONMENT_REPORT_FILE_NAME
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report, report_path, python_executable


def _bind_ready_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: DependencyProfile,
    environment_report: dict[str, Any],
    environment_report_path: Path,
    formal_execution_locks: Sequence[Mapping[str, Any]],
) -> None:
    """绑定 registry、环境准备和连续正式执行锁复核结果."""

    lock_iterator = iter(formal_execution_locks)
    monkeypatch.setattr(
        execution,
        "get_dependency_profile",
        lambda profile_id, path: profile,
    )
    monkeypatch.setattr(
        execution,
        "prepare_isolated_dependency_environment",
        lambda profile_id, **kwargs: (environment_report, environment_report_path),
    )
    monkeypatch.setattr(
        execution,
        "require_published_formal_execution_lock",
        lambda root: dict(next(lock_iterator)),
    )


@pytest.mark.quick
def test_execution_prefixes_validated_python_and_injects_environment_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """通过报告必须使用受验证 Python 并传播环境报告与执行锁."""

    profile = _ready_profile()
    formal_lock = _formal_execution_lock()
    environment_root = tmp_path / "dependency_envs"
    environment_report, environment_report_path, python_executable = (
        _write_environment_report(
            tmp_path,
            profile,
            environment_root,
            formal_lock,
        )
    )
    _bind_ready_environment(
        monkeypatch,
        profile=profile,
        environment_report=environment_report,
        environment_report_path=environment_report_path,
        formal_execution_locks=(formal_lock, formal_lock, formal_lock),
    )
    calls: list[tuple[list[str], Path, dict[str, str]]] = []

    def run_child(
        argv: Sequence[str],
        working_directory: Path,
        environment_overrides: Mapping[str, str],
    ) -> dict[str, Any]:
        calls.append(
            (
                [str(token) for token in argv],
                working_directory,
                dict(environment_overrides),
            )
        )
        return {"return_code": 0, "stdout": "scientific-ok\n", "stderr": ""}

    report, report_path = execution.execute_isolated_scientific_command(
        profile.profile_name,
        ("-m", "experiments.runners.semantic_watermark_runtime", "--help"),
        execution_report_path="outputs/scientific_execution/report.json",
        repository_root=tmp_path,
        environment_root=environment_root,
        managed_python_root=tmp_path / "managed_pythons",
        command_runner=run_child,
    )

    dependency_report_digest = hashlib.sha256(
        environment_report_path.read_bytes()
    ).hexdigest()
    assert calls == [
        (
            [
                str(python_executable),
                "-m",
                "experiments.runners.semantic_watermark_runtime",
                "--help",
            ],
            tmp_path.resolve(),
            {
                execution.FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY: "a" * 40,
                execution.FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY: "f" * 64,
                execution.DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY: str(
                    environment_report_path.resolve()
                ),
                execution.DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY: (
                    dependency_report_digest
                ),
            },
        )
    ]
    assert report["decision"] == "pass"
    assert report["execution_completed"] is True
    assert report["supports_paper_claim"] is False
    assert report["dependency_environment_report_valid"] is True
    assert report["profile_digest"] == profile.profile_digest
    assert report["complete_hash_lock_digest"] == profile.complete_hash_lock_digest
    assert report["python_executable_path"] == str(python_executable)
    assert report["python_executable_sha256"] == hashlib.sha256(
        b"isolated-scientific-python"
    ).hexdigest()
    assert report["formal_execution_lock"] == formal_lock
    assert report["formal_execution_lock_revalidated_before_child"] is True
    assert report["formal_execution_lock_revalidated_after_child"] is True
    assert report["python_executable_revalidated_before_child"] is True
    assert report["python_executable_revalidated_after_child"] is True
    assert report["dependency_environment_report_revalidated_before_child"] is True
    assert report["dependency_environment_report_revalidated_after_child"] is True
    assert report["execution"]["return_code"] == 0
    assert report["execution"]["stdout"] == "scientific-ok\n"
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


@pytest.mark.quick
def test_execution_rejects_non_scientific_profile_before_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """父编排 profile 不得进入科学子解释器执行路径."""

    preparation_called = False

    def prepare(*args: Any, **kwargs: Any) -> Any:
        nonlocal preparation_called
        preparation_called = True
        raise AssertionError("不应调用环境准备")

    monkeypatch.setattr(execution, "prepare_isolated_dependency_environment", prepare)
    report, report_path = execution.execute_isolated_scientific_command(
        "workflow_orchestrator",
        ("-m", "experiments.runners.semantic_watermark_runtime"),
        execution_report_path="outputs/scientific_execution/rejected.json",
        repository_root=tmp_path,
    )

    assert preparation_called is False
    assert report["decision"] == "fail"
    assert report["failure_reasons"] == ["scientific_profile_not_allowed"]
    assert report["execution"]["attempted"] is False
    assert report["supports_paper_claim"] is False
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


@pytest.mark.quick
def test_execution_rejects_environment_report_with_python_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """解释器文件摘要不一致时不得启动科学命令."""

    profile = _ready_profile()
    formal_lock = _formal_execution_lock()
    environment_root = tmp_path / "dependency_envs"
    environment_report, environment_report_path, _ = _write_environment_report(
        tmp_path,
        profile,
        environment_root,
        formal_lock,
    )
    environment_report["python_executable_sha256"] = "c" * 64
    environment_report["python_executable_sha256_after_preparation"] = "c" * 64
    environment_report_path.write_text(
        json.dumps(environment_report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _bind_ready_environment(
        monkeypatch,
        profile=profile,
        environment_report=environment_report,
        environment_report_path=environment_report_path,
        formal_execution_locks=(formal_lock,),
    )
    child_called = False

    def run_child(*args: Any, **kwargs: Any) -> Any:
        nonlocal child_called
        child_called = True
        return 0

    report, _ = execution.execute_isolated_scientific_command(
        profile.profile_name,
        ("-m", "experiments.runners.semantic_watermark_runtime"),
        execution_report_path="outputs/scientific_execution/digest_rejected.json",
        repository_root=tmp_path,
        environment_root=environment_root,
        command_runner=run_child,
    )

    assert child_called is False
    assert report["decision"] == "fail"
    assert report["failure_reasons"] == ["dependency_environment_report_rejected"]
    assert (
        "dependency_environment_python_executable_digest_invalid"
        in report["dependency_environment_validation_errors"]
    )


@pytest.mark.quick
def test_execution_persists_nonzero_child_result_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """科学子命令返回非零时必须保留诊断并关闭通过结论."""

    profile = _ready_profile()
    formal_lock = _formal_execution_lock()
    environment_root = tmp_path / "dependency_envs"
    environment_report, environment_report_path, _ = _write_environment_report(
        tmp_path,
        profile,
        environment_root,
        formal_lock,
    )
    _bind_ready_environment(
        monkeypatch,
        profile=profile,
        environment_report=environment_report,
        environment_report_path=environment_report_path,
        formal_execution_locks=(formal_lock, formal_lock, formal_lock),
    )

    report, report_path = execution.execute_isolated_scientific_command(
        profile.profile_name,
        ("-m", "experiments.runners.semantic_watermark_runtime"),
        execution_report_path="outputs/scientific_execution/child_failed.json",
        repository_root=tmp_path,
        environment_root=environment_root,
        command_runner=lambda argv, cwd, env: {
            "return_code": 7,
            "stdout": "partial\n",
            "stderr": "scientific-error\n",
        },
    )

    assert report["decision"] == "fail"
    assert report["failure_reasons"] == ["scientific_child_command_failed"]
    assert report["execution_completed"] is False
    assert report["execution"]["return_code"] == 7
    assert report["execution"]["stdout"] == "partial\n"
    assert report["execution"]["stderr"] == "scientific-error\n"
    assert report["formal_execution_lock_revalidated_after_child"] is True
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


@pytest.mark.quick
def test_execution_rejects_formal_lock_drift_after_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """子命令结束后的正式执行锁漂移必须覆盖进程成功退出."""

    profile = _ready_profile()
    formal_lock = _formal_execution_lock()
    environment_root = tmp_path / "dependency_envs"
    environment_report, environment_report_path, _ = _write_environment_report(
        tmp_path,
        profile,
        environment_root,
        formal_lock,
    )
    _bind_ready_environment(
        monkeypatch,
        profile=profile,
        environment_report=environment_report,
        environment_report_path=environment_report_path,
        formal_execution_locks=(
            formal_lock,
            formal_lock,
            _formal_execution_lock("e"),
        ),
    )

    report, _ = execution.execute_isolated_scientific_command(
        profile.profile_name,
        ("-m", "experiments.runners.semantic_watermark_runtime"),
        execution_report_path="outputs/scientific_execution/lock_drift.json",
        repository_root=tmp_path,
        environment_root=environment_root,
        command_runner=lambda argv, cwd, env: 0,
    )

    assert report["decision"] == "fail"
    assert report["failure_reasons"] == [
        "formal_execution_lock_drift_after_child"
    ]
    assert report["execution"]["return_code"] == 0
    assert report["formal_execution_lock_revalidated_after_child"] is False


@pytest.mark.quick
def test_execution_report_path_must_stay_under_outputs(tmp_path: Path) -> None:
    """调用方不得把持久报告写入仓库输出根之外."""

    with pytest.raises(ValueError, match="outputs/"):
        execution.execute_isolated_scientific_command(
            "sd35_method_runtime_gpu",
            ("-m", "experiments.runners.semantic_watermark_runtime"),
            execution_report_path=tmp_path / "outside.json",
            repository_root=tmp_path,
        )


@pytest.mark.quick
def test_execution_module_is_python38_parseable_and_has_no_outer_imports() -> None:
    """通用执行原语必须保持 Python 3.8 语法与由外向内依赖方向."""

    source_path = Path(execution.__file__)
    source = source_path.read_text(encoding="utf-8-sig")
    syntax_tree = ast.parse(source, feature_version=(3, 8))
    imported_roots = set()
    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert "scripts" not in imported_roots
    assert "paper_workflow" not in imported_roots
    assert execution.DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY == (
        repository_environment.ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY
    )
    assert execution.DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY == (
        repository_environment.ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY
    )
