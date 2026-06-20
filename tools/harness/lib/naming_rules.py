"""提供通用命名治理规则。"""

from __future__ import annotations

import re
from pathlib import Path

SNAKE_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
FORBIDDEN_WEAK_TOKEN_PATTERN = re.compile(r"(?:^|[_-])(new|old|best|final)(?:$|[_-])", re.IGNORECASE)
RESERVED_PROGRESS_MARKERS = ("sta" + "ge", "pha" + "se", "\u9636\u6bb5")
ALLOWED_LITERAL_FILE_NAMES = {"README.md", "AGENTS.md", ".gitignore", "pyproject.toml", "__init__.py"}
ALLOWED_DIRECTORY_NAMES = {".codex", ".git", ".pytest_cache", "__pycache__"}
ALLOWED_FILE_SUFFIXES = {".md", ".py", ".json", ".toml", ".txt", ".yml", ".yaml"}


def is_snake_case_name(name: str) -> bool:
    """判断名称是否为 snake_case。"""
    return bool(SNAKE_CASE_PATTERN.fullmatch(name))


def is_allowed_directory_name(name: str) -> bool:
    """判断目录名是否满足正式命名规则。"""
    return name in ALLOWED_DIRECTORY_NAMES or is_snake_case_name(name)


def is_allowed_file_name(name: str) -> bool:
    """判断文件名是否满足正式命名规则。"""
    if name in ALLOWED_LITERAL_FILE_NAMES:
        return True
    if name.endswith(".skill.md"):
        return is_snake_case_name(name[: -len(".skill.md")])
    path = Path(name)
    return path.suffix in ALLOWED_FILE_SUFFIXES and is_snake_case_name(path.stem)


def has_weak_semantic_token(name: str) -> bool:
    """判断名称是否包含弱语义词。"""
    return bool(FORBIDDEN_WEAK_TOKEN_PATTERN.search(name))


def has_reserved_progress_marker(text: str) -> bool:
    """判断文本是否包含只允许写入 docs 的过程标记词。"""
    lowered_text = text.lower()
    return any(marker in lowered_text for marker in RESERVED_PROGRESS_MARKERS)
