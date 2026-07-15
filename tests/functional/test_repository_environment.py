"""正式 Git 执行锁与完整提交身份的轻量功能测试."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from experiments.runtime import repository_environment
from experiments.runtime.repository_environment import (
    FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY,
    FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY,
    FORMAL_EXECUTION_LOCK_SCHEMA,
    ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY,
    ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_FILE_NAME,
    ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY,
    FormalExecutionLockError,
    build_formal_execution_lock,
    build_runtime_environment_report,
    flatten_environment_versions,
    normalize_formal_git_commit,
    publish_formal_execution_lock,
    require_published_formal_execution_lock,
    resolve_code_version,
    validate_formal_execution_lock_pair,
    validate_formal_execution_lock_record,
    verify_formal_execution_lock_code_version,
)
from experiments.runtime.dependency_profiles import parse_exact_requirement_spec
from main.core.digest import build_stable_digest
from paper_workflow.colab_utils.paper_run_environment import (
    configure_paper_run_environment,
)


pytestmark = pytest.mark.quick


class _CudaRuntime:
    """提供运行环境报告所需的最小 CUDA 查询接口."""

    @staticmethod
    def device_count() -> int:
        """返回单个测试设备."""

        return 1

    @staticmethod
    def get_device_name(index: int) -> str:
        """返回稳定测试设备名."""

        assert index == 0
        return "test-gpu"


class _TorchRuntime:
    """提供运行环境报告所需的最小 torch 外形."""

    cuda = _CudaRuntime()


def test_environment_version_flattening_uses_canonical_distribution_name() -> None:
    """运行版本摘要必须读取依赖检查实际产生的规范化包名."""

    normalized_name = parse_exact_requirement_spec(
        "huggingface_hub==1.20.1"
    ).normalized_name
    assert normalized_name == "huggingface-hub"
    package_versions = {
        "accelerate": "1.14.0",
        "diffusers": "0.38.0",
        normalized_name: "1.20.1",
        "numpy": "2.0.2",
        "pillow": "11.3.0",
        "protobuf": "7.35.1",
        "safetensors": "0.8.0",
        "sentencepiece": "0.2.1",
        "tokenizers": "0.22.2",
        "torch": "2.11.0+cu128",
        "transformers": "5.12.1",
    }

    flattened = flatten_environment_versions(
        {"package_versions": package_versions}
    )

    assert flattened["huggingface_hub_version"] == "1.20.1"
    legacy_package_versions = {
        name: version
        for name, version in package_versions.items()
        if name != normalized_name
    }
    legacy_package_versions["huggingface_hub"] = package_versions[normalized_name]
    with pytest.raises(KeyError, match="huggingface-hub"):
        flatten_environment_versions(
            {"package_versions": legacy_package_versions}
        )


def _git(repository: Path, *arguments: str) -> str:
    """执行测试仓库 Git 命令并返回去除行尾的标准输出."""

    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _build_repository(root: Path) -> tuple[str, str]:
    """构造包含两个提交且当前附着在 main 的 clean 测试仓库."""

    root.mkdir()
    _git(root, "init", "--initial-branch=main")
    _git(root, "config", "user.name", "SLM-WM Test")
    _git(root, "config", "user.email", "slm-wm-test@example.invalid")
    _git(root, "config", "core.autocrlf", "false")
    tracked_path = root / "tracked.txt"
    tracked_path.write_text("first\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-m", "首次提交")
    first_commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
    tracked_path.write_text("second\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-m", "第二次提交")
    second_commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
    return first_commit, second_commit


def _runtime_formal_execution_lock(commit_character: str = "a") -> dict[str, Any]:
    """构造无需临时 Git 仓库即可验证的规范执行锁."""

    payload = {
        "formal_execution_lock_schema": FORMAL_EXECUTION_LOCK_SCHEMA,
        "formal_execution_commit": commit_character * 40,
        "formal_execution_head_detached": True,
        "formal_execution_worktree_clean": True,
        "formal_execution_lock_ready": True,
    }
    return {
        **payload,
        "formal_execution_lock_digest": build_stable_digest(payload),
    }


def _ready_runtime_profile_summary(
    profile_id: str = "sd35_method_runtime_gpu",
) -> dict[str, Any]:
    """构造具有完整锁的运行环境 profile 摘要."""

    return {
        "profile_name": profile_id,
        "profile_digest": "1" * 64,
        "summary_digest": "2" * 64,
        "direct_requirements_path": (
            "configs/dependency_profiles/{0}_direct.txt".format(profile_id)
        ),
        "direct_requirements_digest": "3" * 64,
        "complete_hash_lock_path": (
            "configs/dependency_profiles/{0}_lock.txt".format(profile_id)
        ),
        "complete_hash_lock_digest": "4" * 64,
        "complete_hash_lock_present": True,
        "complete_hash_lock_dependency_count": 37,
        "formal_ready": True,
    }


def _matching_runtime_inspection(profile_id: str) -> dict[str, Any]:
    """构造当前解释器依赖与 CUDA identity 全部匹配的 inspection."""

    return {
        "profile_name": profile_id,
        "observed_environment": {
            "python_version": "3.12.13",
            "direct_dependencies": {},
            "cuda_available": True,
            "torch_cuda_version": "12.8",
        },
        "decision": "pass",
        "readiness_blockers": [],
    }


def _bind_runtime_profile(
    monkeypatch: pytest.MonkeyPatch,
    profile_summary: dict[str, Any],
) -> None:
    """把运行环境构造器绑定到同一组 profile 与 inspection 测试事实."""

    monkeypatch.setattr(
        repository_environment,
        "build_dependency_profile_summary",
        lambda profile_id: profile_summary,
    )
    monkeypatch.setattr(
        repository_environment,
        "inspect_dependency_profile_environment",
        lambda profile_id, torch_module=None: _matching_runtime_inspection(profile_id),
    )


def _write_isolated_environment_context(
    repository_root: Path,
    profile_summary: dict[str, Any],
    python_executable: Path,
    formal_execution_lock: dict[str, Any],
) -> tuple[dict[str, Any], Path, str]:
    """写出科学子进程必须继承的隔离环境报告."""

    python_digest = repository_environment.file_digest(python_executable)
    nested_report = {
        "profile_id": profile_summary["profile_name"],
        "profile_digest": profile_summary["profile_digest"],
        "direct_requirements_digest": profile_summary["direct_requirements_digest"],
        "complete_hash_lock_digest": profile_summary["complete_hash_lock_digest"],
        "complete_hash_lock_dependency_count": profile_summary[
            "complete_hash_lock_dependency_count"
        ],
        "python_executable": str(python_executable),
        "formal_execution_lock": formal_execution_lock,
        "formal_ready": True,
        "decision": "pass",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    report = {
        "report_schema": (
            repository_environment.ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_SCHEMA
        ),
        "schema_version": (
            repository_environment.ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_SCHEMA_VERSION
        ),
        "operation_kind": "formal_dependency_environment_preparation",
        "profile_id": profile_summary["profile_name"],
        "profile_digest": profile_summary["profile_digest"],
        "direct_requirements_digest": profile_summary["direct_requirements_digest"],
        "complete_hash_lock_digest": profile_summary["complete_hash_lock_digest"],
        "complete_hash_lock_dependency_count": profile_summary[
            "complete_hash_lock_dependency_count"
        ],
        "formal_execution_lock": formal_execution_lock,
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
        "dependency_preparation_report": nested_report,
    }
    report_path = (
        repository_root
        / "outputs"
        / "dependency_profiles"
        / profile_summary["profile_name"]
        / ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_FILE_NAME
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report, report_path, repository_environment.file_digest(report_path)


@pytest.mark.parametrize("invalid_commit", ("main", "abc1234", "A" * 40))
def test_formal_git_commit_rejects_branch_short_and_uppercase(
    invalid_commit: str,
) -> None:
    """分支名、短 SHA 与大写 SHA 均不得被规范化为正式提交身份."""

    with pytest.raises(FormalExecutionLockError, match="40位小写"):
        normalize_formal_git_commit(invalid_commit)


def test_formal_execution_lock_accepts_exact_detached_clean_commit(
    tmp_path: Path,
) -> None:
    """精确40位提交、detached HEAD 与 clean 工作树同时满足时返回稳定锁."""

    repository = tmp_path / "repository"
    _, commit = _build_repository(repository)
    _git(repository, "checkout", "--detach", commit)

    record = build_formal_execution_lock(repository, commit)
    digest_payload = {
        key: value
        for key, value in record.items()
        if key != "formal_execution_lock_digest"
    }

    assert record["formal_execution_lock_schema"] == FORMAL_EXECUTION_LOCK_SCHEMA
    assert record["formal_execution_commit"] == commit
    assert record["formal_execution_head_detached"] is True
    assert record["formal_execution_worktree_clean"] is True
    assert record["formal_execution_lock_ready"] is True
    assert record["formal_execution_lock_digest"] == build_stable_digest(
        digest_payload
    )
    assert resolve_code_version(repository) == commit


def test_formal_execution_lock_record_rejects_extra_field_and_forged_digest(
    tmp_path: Path,
) -> None:
    """锁记录只能使用唯一字段集合且摘要必须由规范载荷重建."""

    repository = tmp_path / "repository"
    _, commit = _build_repository(repository)
    _git(repository, "checkout", "--detach", commit)
    record = build_formal_execution_lock(repository, commit)

    with pytest.raises(FormalExecutionLockError, match="字段集合"):
        validate_formal_execution_lock_record({**record, "extra": "forged"})
    with pytest.raises(FormalExecutionLockError, match="摘要与规范载荷"):
        validate_formal_execution_lock_record(
            {**record, "formal_execution_lock_digest": "0" * 64}
        )


def test_published_formal_execution_lock_is_reverified_successfully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """统一发布函数写出的身份必须能够通过仓库状态实时复验."""

    repository = tmp_path / "repository"
    _, commit = _build_repository(repository)
    _git(repository, "checkout", "--detach", commit)
    record = build_formal_execution_lock(repository, commit)
    monkeypatch.setenv(FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY, "")
    monkeypatch.setenv(FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY, "")

    published_record = publish_formal_execution_lock(record)

    assert require_published_formal_execution_lock(repository) == published_record


def test_paper_run_environment_rejects_forged_published_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notebook 配置层不得把格式正确但伪造的环境摘要视为执行锁."""

    repository = tmp_path / "repository"
    _, commit = _build_repository(repository)
    _git(repository, "checkout", "--detach", commit)
    monkeypatch.setenv(FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY, commit)
    monkeypatch.setenv(FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY, "0" * 64)

    with pytest.raises(FormalExecutionLockError, match="摘要与当前仓库状态不一致"):
        configure_paper_run_environment(
            "attention_geometry",
            repository_root=repository,
        )


def test_published_formal_execution_lock_rejects_later_worktree_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """锁发布后发生的未跟踪文件变更必须在业务入口前被实时拒绝."""

    repository = tmp_path / "repository"
    _, commit = _build_repository(repository)
    _git(repository, "checkout", "--detach", commit)
    monkeypatch.setenv(FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY, "")
    monkeypatch.setenv(FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY, "")
    publish_formal_execution_lock(build_formal_execution_lock(repository, commit))
    (repository / "changed_after_publish.txt").write_text(
        "changed\n",
        encoding="utf-8",
    )

    with pytest.raises(FormalExecutionLockError, match="clean"):
        require_published_formal_execution_lock(repository)


def test_runtime_environment_report_does_not_trust_environment_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """运行环境报告未收到已验证记录时必须保持执行锁未就绪."""

    monkeypatch.setenv(FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY, "a" * 40)
    monkeypatch.setenv(FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY, "b" * 64)

    report = build_runtime_environment_report("sd35_method_runtime_gpu")

    assert report["formal_execution_commit"] == ""
    assert report["formal_execution_lock_digest"] == ""
    assert report["formal_execution_lock_ready"] is False


def test_runtime_environment_report_accepts_validated_record(
    tmp_path: Path,
) -> None:
    """调用方显式传入规范执行锁时环境报告可以传播已验证身份."""

    repository = tmp_path / "repository"
    _, commit = _build_repository(repository)
    _git(repository, "checkout", "--detach", commit)
    record = build_formal_execution_lock(repository, commit)

    report = build_runtime_environment_report(
        "sd35_method_runtime_gpu",
        verified_formal_execution_lock=record,
    )

    assert report["formal_execution_commit"] == commit
    assert report["formal_execution_lock_digest"] == record[
        "formal_execution_lock_digest"
    ]
    assert report["formal_execution_lock_ready"] is True


def test_scientific_runtime_requires_injected_isolated_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """科学 profile 缺少父执行原语注入时必须关闭环境 readiness."""

    profile_summary = _ready_runtime_profile_summary()
    _bind_runtime_profile(monkeypatch, profile_summary)
    monkeypatch.delenv(
        ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY,
        raising=False,
    )
    monkeypatch.delenv(
        ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY,
        raising=False,
    )

    report = build_runtime_environment_report(
        profile_summary["profile_name"],
        torch_module=_TorchRuntime(),
        verified_formal_execution_lock=_runtime_formal_execution_lock(),
        repository_root=tmp_path,
    )

    assert report["dependency_environment_ready"] is False
    assert report["isolated_scientific_context_required"] is True
    assert report["isolated_scientific_context_ready"] is False
    assert report["dependency_readiness_blockers"] == [
        "isolated_context_environment_report_path_missing"
    ]


def test_scientific_runtime_accepts_strict_isolated_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """环境、解释器、完整锁和执行锁一致时科学上下文通过."""

    profile_summary = _ready_runtime_profile_summary()
    formal_execution_lock = _runtime_formal_execution_lock()
    python_executable = tmp_path / "dependency_envs" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_bytes(b"isolated-runtime-python")
    _, report_path, report_digest = _write_isolated_environment_context(
        tmp_path,
        profile_summary,
        python_executable,
        formal_execution_lock,
    )
    _bind_runtime_profile(monkeypatch, profile_summary)
    monkeypatch.setattr(repository_environment.sys, "executable", str(python_executable))
    monkeypatch.setenv(
        ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY,
        str(report_path.absolute()),
    )
    monkeypatch.setenv(
        ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY,
        report_digest,
    )

    report = build_runtime_environment_report(
        profile_summary["profile_name"],
        torch_module=_TorchRuntime(),
        verified_formal_execution_lock=formal_execution_lock,
        repository_root=tmp_path,
    )

    assert report["dependency_environment_ready"] is True
    assert report["dependency_readiness_blockers"] == []
    assert report["isolated_scientific_context_required"] is True
    assert report["isolated_scientific_context_ready"] is True
    context = report["isolated_scientific_context"]
    assert context["decision"] == "pass"
    assert context["dependency_environment_report_digest"] == report_digest
    assert context["dependency_environment_report_actual_digest"] == report_digest
    assert context["reported_profile_digest"] == profile_summary["profile_digest"]
    assert context["reported_complete_hash_lock_digest"] == profile_summary[
        "complete_hash_lock_digest"
    ]
    assert context["reported_formal_execution_lock_digest"] == formal_execution_lock[
        "formal_execution_lock_digest"
    ]
    assert context["current_python_executable"] == str(python_executable)
    assert context["current_python_executable_sha256"] == (
        repository_environment.file_digest(python_executable)
    )


@pytest.mark.parametrize(
    ("mutation", "expected_blocker"),
    (
        (
            "profile",
            "isolated_context_profile_identity_mismatch",
        ),
        (
            "lock",
            "isolated_context_complete_hash_lock_mismatch",
        ),
        (
            "formal_execution_lock",
            "isolated_context_formal_execution_lock_mismatch",
        ),
        (
            "python_path",
            "isolated_context_python_executable_mismatch",
        ),
        (
            "python_digest",
            "isolated_context_python_executable_digest_mismatch",
        ),
    ),
)
def test_scientific_runtime_rejects_context_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_blocker: str,
) -> None:
    """任一隔离身份字段漂移都必须形成稳定 blocker."""

    profile_summary = _ready_runtime_profile_summary()
    formal_execution_lock = _runtime_formal_execution_lock()
    python_executable = tmp_path / "dependency_envs" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_bytes(b"isolated-runtime-python")
    environment_report, report_path, _ = _write_isolated_environment_context(
        tmp_path,
        profile_summary,
        python_executable,
        formal_execution_lock,
    )
    if mutation == "profile":
        environment_report["profile_digest"] = "5" * 64
    elif mutation == "lock":
        environment_report["complete_hash_lock_digest"] = "5" * 64
    elif mutation == "formal_execution_lock":
        environment_report["formal_execution_lock"] = (
            _runtime_formal_execution_lock("b")
        )
    elif mutation == "python_path":
        environment_report["python_executable_path"] = str(
            tmp_path / "different" / "python"
        )
    elif mutation == "python_digest":
        environment_report["python_executable_sha256"] = "5" * 64
        environment_report["python_executable_sha256_after_preparation"] = "5" * 64
    report_path.write_text(
        json.dumps(environment_report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _bind_runtime_profile(monkeypatch, profile_summary)
    monkeypatch.setattr(repository_environment.sys, "executable", str(python_executable))
    monkeypatch.setenv(
        ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY,
        str(report_path.absolute()),
    )
    monkeypatch.setenv(
        ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY,
        repository_environment.file_digest(report_path),
    )

    report = build_runtime_environment_report(
        profile_summary["profile_name"],
        torch_module=_TorchRuntime(),
        verified_formal_execution_lock=formal_execution_lock,
        repository_root=tmp_path,
    )

    assert report["dependency_environment_ready"] is False
    assert expected_blocker in report["dependency_readiness_blockers"]


def test_scientific_runtime_rejects_injected_report_digest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """注入摘要与环境报告文件不一致时不得读取其中身份."""

    profile_summary = _ready_runtime_profile_summary()
    formal_execution_lock = _runtime_formal_execution_lock()
    python_executable = tmp_path / "dependency_envs" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_bytes(b"isolated-runtime-python")
    _, report_path, _ = _write_isolated_environment_context(
        tmp_path,
        profile_summary,
        python_executable,
        formal_execution_lock,
    )
    _bind_runtime_profile(monkeypatch, profile_summary)
    monkeypatch.setattr(repository_environment.sys, "executable", str(python_executable))
    monkeypatch.setenv(
        ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY,
        str(report_path.absolute()),
    )
    monkeypatch.setenv(
        ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY,
        "0" * 64,
    )

    report = build_runtime_environment_report(
        profile_summary["profile_name"],
        torch_module=_TorchRuntime(),
        verified_formal_execution_lock=formal_execution_lock,
        repository_root=tmp_path,
    )

    assert report["dependency_environment_ready"] is False
    assert report["dependency_readiness_blockers"] == [
        "isolated_context_environment_report_digest_mismatch"
    ]


def test_workflow_orchestrator_does_not_require_isolated_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """父编排 profile 保持当前解释器运行, 不消费科学上下文环境键."""

    profile_summary = _ready_runtime_profile_summary("workflow_orchestrator")
    _bind_runtime_profile(monkeypatch, profile_summary)
    monkeypatch.delenv(
        ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY,
        raising=False,
    )
    monkeypatch.delenv(
        ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY,
        raising=False,
    )

    report = build_runtime_environment_report(
        "workflow_orchestrator",
        torch_module=_TorchRuntime(),
        repository_root=tmp_path,
    )

    assert report["dependency_environment_ready"] is True
    assert report["isolated_scientific_context_required"] is False
    assert report["isolated_scientific_context_ready"] is True
    assert report["isolated_scientific_context"]["decision"] == "not_required"
    assert report["dependency_readiness_blockers"] == []


def test_formal_execution_lock_code_version_and_pair_are_strict(
    tmp_path: Path,
) -> None:
    """单锁与双锁辅助函数都必须拒绝短版本或跨提交组合."""

    repository = tmp_path / "repository"
    base_commit, head_commit = _build_repository(repository)
    _git(repository, "checkout", "--detach", head_commit)
    head_lock = build_formal_execution_lock(repository, head_commit)
    _git(repository, "checkout", "--detach", base_commit)
    base_lock = build_formal_execution_lock(repository, base_commit)

    assert (
        verify_formal_execution_lock_code_version(head_lock, head_commit)
        == head_lock
    )
    assert (
        validate_formal_execution_lock_pair(
            head_lock,
            head_lock,
            head_commit,
        )
        == head_lock
    )
    with pytest.raises(FormalExecutionLockError, match="40位小写"):
        verify_formal_execution_lock_code_version(head_lock, head_commit[:7])
    with pytest.raises(FormalExecutionLockError, match="不一致"):
        validate_formal_execution_lock_pair(
            head_lock,
            base_lock,
            head_commit,
        )


def test_formal_execution_lock_rejects_attached_main(tmp_path: Path) -> None:
    """即使提交与工作树有效, 附着在 main 的 HEAD 也不得正式执行."""

    repository = tmp_path / "repository"
    _, commit = _build_repository(repository)

    with pytest.raises(FormalExecutionLockError, match="detached"):
        build_formal_execution_lock(repository, commit)


def test_formal_execution_lock_rejects_dirty_worktree(tmp_path: Path) -> None:
    """detached HEAD 下任一未跟踪文件也必须使正式执行锁失败."""

    repository = tmp_path / "repository"
    _, commit = _build_repository(repository)
    _git(repository, "checkout", "--detach", commit)
    (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(FormalExecutionLockError, match="clean"):
        build_formal_execution_lock(repository, commit)


def test_formal_execution_lock_rejects_expected_commit_mismatch(
    tmp_path: Path,
) -> None:
    """当前 detached HEAD 与调用方冻结提交不一致时必须立即失败."""

    repository = tmp_path / "repository"
    first_commit, second_commit = _build_repository(repository)
    _git(repository, "checkout", "--detach", second_commit)

    with pytest.raises(FormalExecutionLockError, match="不一致"):
        build_formal_execution_lock(repository, first_commit)
