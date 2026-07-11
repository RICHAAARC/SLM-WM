"""验证外部官方源码的不可变 Git 身份和补丁工作树证据。"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from paper_experiments.runners.external_source_runtime import (
    build_registered_source_patch_evidence,
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
