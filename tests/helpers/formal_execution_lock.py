"""构造功能测试使用的确定性正式执行锁记录."""

from __future__ import annotations

from typing import Any

from main.core.digest import build_stable_digest


def build_test_formal_execution_lock(
    commit: str = "b" * 40,
) -> dict[str, Any]:
    """构造字段完整且摘要可重算的 clean detached 测试锁."""

    payload = {
        "formal_execution_lock_schema": "clean_detached_git_commit_v1",
        "formal_execution_commit": commit,
        "formal_execution_head_detached": True,
        "formal_execution_worktree_clean": True,
        "formal_execution_lock_ready": True,
    }
    return {
        **payload,
        "formal_execution_lock_digest": build_stable_digest(payload),
    }

