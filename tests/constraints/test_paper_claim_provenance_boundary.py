"""验证正式论文结论 Writer 统一受精确9重复聚合来源门禁保护."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import re
from typing import Callable

import pytest

from paper_experiments.runners.paper_claim_provenance import (
    PaperClaimAggregateRequiredError,
)
pytestmark = pytest.mark.quick

ROOT = Path(__file__).resolve().parents[2]
PAPER_CLAIM_PROVENANCE_MODULE = (
    "paper_experiments.runners.paper_claim_provenance"
)
PRIVATE_WRITER_BYPASS_PATTERN = re.compile(
    r"^_write_[a-z0-9_]+_from_validated_aggregate$"
)
PRIVATE_SERIALIZATION_HELPERS_BY_MODULE = {
    "scripts/write_fixed_fpr_threshold_audit_outputs.py": frozenset(
        {"_write_json", "_write_rows_csv"}
    ),
    "scripts/write_paired_superiority_outputs.py": frozenset(
        {"_write_csv"}
    ),
}
GOVERNED_PYTHON_SOURCE_ROOTS = (
    "main",
    "experiments",
    "paper_experiments",
    "scripts",
    "paper_workflow",
    "tools",
    "tests",
)


def _read_python_tree(path: Path) -> ast.Module:
    """读取 Python 源码并返回可供约束检查的语法树."""

    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _imports_paper_claim_provenance(tree: ast.Module) -> bool:
    """判断脚本是否显式依赖统一论文结论来源门禁."""

    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == PAPER_CLAIM_PROVENANCE_MODULE
        for node in tree.body
    )


def _static_string_value(node: ast.AST) -> str | None:
    """返回不依赖运行时变量的字符串表达式值."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string_value(node.left)
        right = _static_string_value(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _dynamic_private_writer_name(node: ast.AST) -> str | None:
    """识别 getattr 或下标表达式中的静态私有 Writer 名称."""

    candidate: str | None = None
    if isinstance(node, ast.Call):
        call_name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        if call_name in {"getattr", "__getattribute__"} and len(node.args) >= 2:
            candidate = _static_string_value(node.args[1])
    elif isinstance(node, ast.Subscript):
        candidate = _static_string_value(node.slice)
    if candidate is None or PRIVATE_WRITER_BYPASS_PATTERN.fullmatch(
        candidate
    ) is None:
        return None
    return candidate


def _discover_formal_claim_writers() -> tuple[Callable[..., object], ...]:
    """从门禁依赖和公开 Writer 命名自动发现正式结论入口."""

    discovered: list[Callable[..., object]] = []
    for source_path in sorted((ROOT / "scripts").glob("*.py")):
        tree = _read_python_tree(source_path)
        if not _imports_paper_claim_provenance(tree):
            continue
        module_name = f"scripts.{source_path.stem}"
        module = importlib.import_module(module_name)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("write_") or not node.name.endswith(
                "_outputs"
            ):
                continue
            writer = getattr(module, node.name)
            if callable(writer):
                discovered.append(writer)
    if not discovered:
        raise AssertionError("没有自动发现任何正式论文结论 Writer")
    return tuple(discovered)


def _writer_test_id(writer: Callable[..., object]) -> str:
    """为自动发现的 Writer 构造稳定测试标识."""

    return f"{writer.__module__}.{writer.__name__}"


FORMAL_CLAIM_WRITERS = _discover_formal_claim_writers()


@pytest.mark.parametrize(
    "writer",
    FORMAL_CLAIM_WRITERS,
    ids=_writer_test_id,
)
def test_formal_claim_writer_fails_before_any_output_without_exact9_aggregate(
    tmp_path: Path,
    writer: Callable[..., object],
) -> None:
    """缺少版本化聚合证据时, 正式 Writer 不得读取历史输入或创建输出."""

    with pytest.raises(
        PaperClaimAggregateRequiredError,
        match="版本化精确9重复聚合证据",
    ):
        writer(root=tmp_path)

    assert not (tmp_path / "outputs").exists()


def test_private_validated_aggregate_writer_bypass_is_absent() -> None:
    """生产代码和测试不得静态或动态引用私有聚合 Writer."""

    violations: list[str] = []
    for source_root_name in GOVERNED_PYTHON_SOURCE_ROOTS:
        source_root = ROOT / source_root_name
        if not source_root.is_dir():
            continue
        for source_path in sorted(source_root.rglob("*.py")):
            tree = _read_python_tree(source_path)
            relative_path = source_path.relative_to(ROOT).as_posix()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if PRIVATE_WRITER_BYPASS_PATTERN.fullmatch(node.name):
                        violations.append(
                            f"{relative_path}:{node.lineno}:定义:{node.name}"
                        )
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if PRIVATE_WRITER_BYPASS_PATTERN.fullmatch(alias.name):
                            violations.append(
                                f"{relative_path}:{node.lineno}:导入:{alias.name}"
                            )
                elif isinstance(node, ast.Attribute):
                    if PRIVATE_WRITER_BYPASS_PATTERN.fullmatch(node.attr):
                        violations.append(
                            f"{relative_path}:{node.lineno}:属性调用:{node.attr}"
                        )
                dynamic_name = _dynamic_private_writer_name(node)
                if dynamic_name is not None:
                    violations.append(
                        f"{relative_path}:{node.lineno}:动态引用:{dynamic_name}"
                    )

    assert not violations, (
        "发现可绕过不可变 aggregate 来源复验的私有 I/O Writer:\n"
        + "\n".join(violations)
    )


def test_formal_claim_modules_only_keep_leaf_private_serializers() -> None:
    """正式结论脚本不得用新的私有 Writer 重新建立旁路入口."""

    violations: list[str] = []
    for source_path in sorted((ROOT / "scripts").glob("*.py")):
        tree = _read_python_tree(source_path)
        if not _imports_paper_claim_provenance(tree):
            continue
        relative_path = source_path.relative_to(ROOT).as_posix()
        allowed_helpers = PRIVATE_SERIALIZATION_HELPERS_BY_MODULE.get(
            relative_path,
            frozenset(),
        )
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if (
                node.name.startswith("_write_")
                and node.name not in allowed_helpers
            ):
                violations.append(
                    f"{relative_path}:{node.lineno}:私有 Writer:{node.name}"
                )

    assert not violations, (
        "正式论文结论脚本只允许叶级 CSV/JSON 序列化 helper, "
        "不得新增私有结论 Writer:\n"
        + "\n".join(violations)
    )

def test_result_gate_rejects_before_reading_explicit_aggregate_path(
    tmp_path: Path,
) -> None:
    """跨重复 Writer 未接入时, 显式路径也不得触发文件读取或输出."""

    from scripts.write_result_closure_gate_outputs import (
        write_result_closure_gate_outputs,
    )

    missing_path = tmp_path / "not_read.zip"
    with pytest.raises(
        PaperClaimAggregateRequiredError,
        match="版本化精确9重复聚合证据",
    ):
        write_result_closure_gate_outputs(
            root=tmp_path,
            randomization_aggregate_package_path=missing_path,
        )

    assert not missing_path.exists()
    assert not (tmp_path / "outputs").exists()
