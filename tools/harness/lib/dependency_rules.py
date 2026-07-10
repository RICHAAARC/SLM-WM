"""提供方法可抽离性的导入边界规则。"""

from __future__ import annotations

import ast
from pathlib import Path


FORBIDDEN_IMPORT_PREFIXES_BY_ROOT = {
    "main/core": (
        "experiments",
        "paper_experiments",
        "external_baseline",
        "scripts",
        "tests",
        "tools",
        "paper_workflow",
    ),
    "main/methods": (
        "experiments",
        "paper_experiments",
        "external_baseline",
        "scripts",
        "tests",
        "tools",
        "paper_workflow",
    ),
    "experiments": (
        "paper_experiments",
        "external_baseline",
        "scripts",
        "tests",
        "tools",
        "paper_workflow",
    ),
    "paper_experiments": (
        "scripts",
        "tests",
        "tools",
        "paper_workflow",
    ),
    "scripts": (
        "paper_workflow",
    ),
}


def extract_imported_modules(path: Path) -> list[str]:
    """从 Python 文件中提取顶层导入模块名。"""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def get_boundary_root(relative_path: Path) -> str | None:
    """判断文件属于哪个受约束的依赖边界根。"""
    normalized = relative_path.as_posix()
    for boundary_root in FORBIDDEN_IMPORT_PREFIXES_BY_ROOT:
        if normalized.startswith(f"{boundary_root}/") or normalized == boundary_root:
            return boundary_root
    return None


def is_forbidden_import(module_name: str, forbidden_prefixes: tuple[str, ...]) -> bool:
    """判断导入模块是否命中禁止前缀。"""
    return any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in forbidden_prefixes)
