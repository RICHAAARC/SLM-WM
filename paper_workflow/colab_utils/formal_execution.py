"""为 Colab 入口发布不可变 Git 正式执行身份."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.runtime.repository_environment import (
    build_formal_execution_lock,
    publish_formal_execution_lock,
)


def verify_and_publish_formal_execution(
    root: str | Path,
    expected_commit: str,
) -> dict[str, Any]:
    """验证仓库身份并把同一执行锁发布给后续 repository runner.

    这一实现属于最外层 Colab 包装. 真正的 Git 状态校验位于
    ``experiments.runtime.repository_environment`` 中, 因而 GPU 服务器脚本
    可以绕过 Notebook 直接复用同一契约.
    """

    execution_lock = build_formal_execution_lock(root, expected_commit)
    return publish_formal_execution_lock(execution_lock)
