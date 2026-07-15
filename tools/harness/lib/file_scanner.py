"""提供受治理文本文件扫描能力。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

SKIP_DIRECTORY_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".pytest_work",
    ".tmp",
    ".venv",
    "outputs",
    "audit_reports",
    "dist",
    "build",
}

EXTERNAL_BASELINE_VENDOR_DIRECTORY_NAMES = {"source", "artifacts"}

BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".exe",
    ".dll",
    ".so",
    ".pyc",
    ".pyd",
}

DEFAULT_GOVERNED_SCAN_ROOTS = (
    "AGENTS.md",
    "README.md",
    ".gitattributes",
    ".gitignore",
    "pyproject.toml",
    ".codex",
    "configs",
    "docs",
    "main",
    "tools",
    "tests",
    "scripts",
    "experiments",
    "paper_experiments",
    "paper_workflow",
    "external_baseline",
)


def _parts(path: str | Path) -> tuple[str, ...]:
    """把路径转换为可比较的语义片段。"""

    return tuple(Path(path).parts)


def _is_external_baseline_vendor_path(path: str | Path) -> bool:
    """判断路径是否位于外部 baseline 的第三方源码或运行产物边界内。

    该判断属于项目特定写法: `external_baseline/` 下的 adapter 和登记文件需要接受 harness 审计,
    但第三方官方源码快照不属于本项目实现, 因此只跳过 `source/` 和 `artifacts/` 子树。
    """

    parts = _parts(path)
    if "external_baseline" not in parts:
        return False
    root_index = parts.index("external_baseline")
    return any(part in EXTERNAL_BASELINE_VENDOR_DIRECTORY_NAMES for part in parts[root_index + 1 :])


def should_skip_path(path: str | Path) -> bool:
    """判断路径是否属于缓存、输出、构建产物或外部第三方快照。"""

    parts = _parts(path)
    if any(part in SKIP_DIRECTORY_NAMES for part in parts):
        return True
    return _is_external_baseline_vendor_path(path)


def iter_text_files(root: str | Path) -> Iterator[Path]:
    """遍历目录下的文本候选文件。"""

    root_path = Path(root)
    if not root_path.exists():
        return
    for path in root_path.rglob("*"):
        if not path.is_file() or should_skip_path(path):
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        yield path


def iter_governed_paths(root: str | Path) -> Iterator[Path]:
    """遍历项目契约明确登记的受治理路径。

    该函数采用正向根目录登记, 使未跟踪的新项目文件仍接受审计, 同时把
    `.claude/`、`.gitnexus/` 和 `CLAUDE.md` 等外部工具元数据留在项目命名
    契约之外。新增正式项目根目录时必须同步扩展登记和约束测试, 不能通过
    临时排除具体违规文件绕过审计。
    """

    root_path = Path(root)
    for relative_root in DEFAULT_GOVERNED_SCAN_ROOTS:
        candidate = root_path / relative_root
        if not candidate.exists() or should_skip_path(relative_root):
            continue
        yield candidate
        if not candidate.is_dir():
            continue
        for path in candidate.rglob("*"):
            relative = path.relative_to(root_path)
            if should_skip_path(relative):
                continue
            yield path


def iter_governed_text_files(root: str | Path) -> Iterator[Path]:
    """按默认受治理根目录遍历文本文件。"""

    for path in iter_governed_paths(root):
        if not path.is_file() or path.suffix.lower() in BINARY_SUFFIXES:
            continue
        yield path
