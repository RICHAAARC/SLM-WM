"""pytest 运行环境配置。"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTEST_TEMP_ROOT = REPOSITORY_ROOT / "outputs" / "pytest_work"
PYTEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(PYTEST_TEMP_ROOT)

_ORIGINAL_MKDIR = os.mkdir


def _mkdir_with_workspace_access(path: str | bytes | os.PathLike[str] | os.PathLike[bytes], mode: int = 0o777, *args: Any, **kwargs: Any) -> None:
    """让 pytest 临时目录在 Windows sandbox 中保持可读写。

    pytest 在 Windows 上会用 0o700 创建 basetemp, 当前受限执行环境会把该权限解释为不可访问。
    该补丁仅作用于测试进程, 使 tmp_path 仍位于受治理的 outputs 临时根目录下, 不影响业务代码路径。
    """
    return _ORIGINAL_MKDIR(path, 0o777, *args, **kwargs)


os.mkdir = _mkdir_with_workspace_access
