"""提供不感知具体 workflow 的通用归档时间与提交身份原语。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from experiments.runtime.repository_environment import (
    FORMAL_GIT_COMMIT_PATTERN,
    resolve_code_version,
)

def resolve_short_commit(root: str | Path = ".") -> str:
    """从完整仓库提交身份显式截取7位归档文件名摘要."""

    code_version = resolve_code_version(Path(root))
    dirty = code_version.endswith("-dirty")
    commit = code_version.removesuffix("-dirty")
    if FORMAL_GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        return "git_unknown"
    short_commit = commit[:7]
    return f"{short_commit}-dirty" if dirty else short_commit


def utc_archive_token() -> str:
    """生成归档共用的 UTC 时间后缀。"""

    current_time = datetime.now(timezone.utc)
    return f"{current_time:%Y%m%d}t{current_time:%H%M%S}z"
