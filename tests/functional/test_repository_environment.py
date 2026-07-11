"""正式 Git 执行锁与完整提交身份的轻量功能测试."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from experiments.runtime.repository_environment import (
    FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY,
    FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY,
    FORMAL_EXECUTION_LOCK_SCHEMA,
    FormalExecutionLockError,
    build_formal_execution_lock,
    build_runtime_environment_report,
    normalize_formal_git_commit,
    publish_formal_execution_lock,
    require_published_formal_execution_lock,
    resolve_code_version,
    validate_formal_execution_lock_pair,
    validate_formal_execution_lock_record,
    verify_formal_execution_lock_code_version,
)
from main.core.digest import build_stable_digest
from paper_workflow.colab_utils.paper_run_environment import (
    configure_paper_run_environment,
)


pytestmark = pytest.mark.quick


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

    report = build_runtime_environment_report()

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
        verified_formal_execution_lock=record,
    )

    assert report["formal_execution_commit"] == commit
    assert report["formal_execution_lock_digest"] == record[
        "formal_execution_lock_digest"
    ]
    assert report["formal_execution_lock_ready"] is True


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
