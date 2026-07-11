"""提供方法可抽离性的静态依赖边界规则."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable


FORBIDDEN_IMPORT_PREFIXES_BY_ROOT = {
    "main": (
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

# 内层代码不应通过字符串绕过静态导入边界. `scripts/` 可能保存发布排除清单,
# 因此该层只审计实际调用参数, 避免把治理元数据误判为运行时依赖.
STRICT_LITERAL_BOUNDARY_ROOTS = frozenset({"main", "experiments", "paper_experiments"})

_DOCSTRING_PARENT_TYPES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
_SHELL_DEPENDENCY_TOKEN_PATTERN = re.compile(
    r"(?:^|[\s'\"=])(?:\./|\.\\)?(?P<root>[A-Za-z_][A-Za-z0-9_]*)(?:[./\\]|$)"
)
_DYNAMIC_DEPENDENCY_CALL_NAMES = frozenset(
    {
        "__import__",
        "call",
        "check_call",
        "check_output",
        "dispatch",
        "execv",
        "execve",
        "execute",
        "execute_command",
        "find_spec",
        "import_module",
        "invoke",
        "launch",
        "open",
        "path",
        "popen",
        "purepath",
        "run",
        "run_child",
        "run_module",
        "run_path",
        "run_scientific_child",
        "spawnv",
        "spawnve",
        "startfile",
        "system",
    }
)


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


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """返回模块、类和函数文档字符串节点的对象标识."""
    node_ids: set[int] = set()
    for parent in ast.walk(tree):
        if not isinstance(parent, _DOCSTRING_PARENT_TYPES) or not parent.body:
            continue
        first_statement = parent.body[0]
        if (
            isinstance(first_statement, ast.Expr)
            and isinstance(first_statement.value, ast.Constant)
            and isinstance(first_statement.value.value, str)
        ):
            node_ids.add(id(first_statement.value))
    return node_ids


def _literal_forbidden_prefix(value: str, forbidden_prefixes: tuple[str, ...]) -> str | None:
    """识别完整字符串是否表示受禁止模块名或仓库相对路径."""
    candidate = value.strip()
    while candidate.startswith(("./", ".\\")):
        candidate = candidate[2:]
    for prefix in forbidden_prefixes:
        reference_starts = (f"{prefix}.", f"{prefix}/", f"{prefix}\\")
        if candidate == prefix or candidate.startswith(reference_starts):
            return prefix
    return None


def _is_dotted_module_literal(value: str, prefix: str) -> bool:
    """判断字符串是否明确表示带成员路径的 Python 模块名."""
    return value.strip().startswith(f"{prefix}.")


def _embedded_call_forbidden_prefix(value: str, forbidden_prefixes: tuple[str, ...]) -> str | None:
    """识别命令字符串中作为独立参数出现的受禁止模块或路径根."""
    for match in _SHELL_DEPENDENCY_TOKEN_PATTERN.finditer(value):
        root = match.group("root")
        if root in forbidden_prefixes:
            return root
    return None


def _iter_call_string_nodes(node: ast.Call) -> Iterable[ast.Constant]:
    """遍历调用参数中的字符串节点, 同时覆盖列表、元组与 f-string 静态片段."""
    for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
        for child in ast.walk(argument):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                yield child


def _call_leaf_name(node: ast.Call) -> str | None:
    """返回调用目标最末端的函数名."""
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _is_dynamic_dependency_call(node: ast.Call) -> bool:
    """判断调用是否具有模块加载、路径读取或子进程执行语义."""
    leaf_name = _call_leaf_name(node)
    if leaf_name is None:
        return False
    return leaf_name.lstrip("_").lower() in _DYNAMIC_DEPENDENCY_CALL_NAMES


def _looks_like_python_command(value: str) -> bool:
    """判断字符串是否是可执行 Python 命令而不是普通说明文本."""
    candidate = value.strip().lower()
    return candidate.startswith(("python ", "python3 ", "python.exe ", "py ", "uv run "))


def extract_dynamic_dependency_references(
    path: Path,
    forbidden_prefixes: tuple[str, ...],
    *,
    strict_literals: bool,
) -> list[dict[str, str | int]]:
    """提取字符串模块名、仓库路径和调用参数中的动态外层依赖."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    docstring_node_ids = _docstring_node_ids(tree)
    references: list[dict[str, str | int]] = []
    seen: set[tuple[int, int, str]] = set()

    def append_reference(node: ast.Constant, prefix: str, kind: str) -> None:
        key = (node.lineno, node.col_offset, prefix)
        if key in seen:
            return
        seen.add(key)
        references.append(
            {
                "reference": node.value,
                "forbidden_prefix": prefix,
                "reference_kind": kind,
                "line": node.lineno,
                "column": node.col_offset,
            }
        )

    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Constant)
            or not isinstance(node.value, str)
            or id(node) in docstring_node_ids
        ):
            continue
        prefix = _literal_forbidden_prefix(node.value, forbidden_prefixes)
        kind = "module_or_path_literal"
        if prefix is not None and (strict_literals or _is_dotted_module_literal(node.value, prefix)):
            append_reference(node, prefix, kind)
            continue
        if strict_literals and _looks_like_python_command(node.value):
            prefix = _embedded_call_forbidden_prefix(node.value, forbidden_prefixes)
            if prefix is not None:
                append_reference(node, prefix, "command_literal")

    for call in (
        node for node in ast.walk(tree) if isinstance(node, ast.Call) and _is_dynamic_dependency_call(node)
    ):
        for string_node in _iter_call_string_nodes(call):
            prefix = _literal_forbidden_prefix(string_node.value, forbidden_prefixes)
            kind = "call_argument_literal"
            if prefix is None:
                prefix = _embedded_call_forbidden_prefix(string_node.value, forbidden_prefixes)
                kind = "embedded_call_argument"
            if prefix is not None:
                append_reference(string_node, prefix, kind)
    return references


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
