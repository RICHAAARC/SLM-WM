"""验证正式运行、打包和闭合入口不可绕过 Git 执行锁."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
FORMAL_RUN_AND_PACKAGE_FUNCTIONS = {
    "experiments/runners/image_only_dataset_runtime.py": (
        "run_image_only_dataset_runtime",
        "package_image_only_dataset_runtime",
    ),
    "experiments/ablations/runtime_rerun.py": (
        "run_runtime_rerun_ablations",
        "package_runtime_rerun_ablations",
    ),
    "experiments/artifacts/dataset_level_quality_outputs.py": (
        "write_dataset_level_quality_outputs",
        "package_dataset_level_quality_outputs",
    ),
    "paper_experiments/runners/external_baseline_method_faithful.py": (
        "write_external_baseline_method_faithful_outputs",
        "package_external_baseline_method_faithful_outputs",
    ),
    "paper_experiments/runners/tree_ring_official_reference.py": (
        "write_tree_ring_official_reference_outputs",
        "package_tree_ring_official_reference_outputs",
    ),
    "paper_experiments/runners/gaussian_shading_official_reference.py": (
        "write_gaussian_shading_official_reference_outputs",
        "package_gaussian_shading_official_reference_outputs",
    ),
    "paper_experiments/runners/shallow_diffuse_official_reference.py": (
        "write_shallow_diffuse_official_reference_outputs",
        "package_shallow_diffuse_official_reference_outputs",
    ),
    "paper_experiments/runners/t2smark_formal_reproduction.py": (
        "write_t2smark_formal_reproduction_outputs",
        "package_t2smark_formal_reproduction_outputs",
    ),
}


def _function_node(path: str, function_name: str) -> ast.FunctionDef:
    """解析指定模块中的顶层函数节点."""

    module = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def _called_attributes(function: ast.FunctionDef) -> tuple[str, ...]:
    """收集函数体调用的属性名称, 用于检查统一锁原语调用."""

    return tuple(
        node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    )


def _called_names(function: ast.FunctionDef) -> tuple[str, ...]:
    """收集函数体直接调用的模块级函数名称."""

    return tuple(
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )


def _string_literals(function: ast.FunctionDef) -> set[str]:
    """收集函数体字符串常量, 用于检查 manifest 锁字段."""

    return {
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


@pytest.mark.quick
@pytest.mark.parametrize(
    ("module_path", "function_names"),
    tuple(FORMAL_RUN_AND_PACKAGE_FUNCTIONS.items()),
)
def test_formal_run_and_package_functions_revalidate_execution_lock(
    module_path: str,
    function_names: tuple[str, str],
) -> None:
    """8组底层运行与打包函数必须分别执行起止锁复验."""

    run_function = _function_node(module_path, function_names[0])
    package_function = _function_node(module_path, function_names[1])
    run_calls = _called_attributes(run_function)
    package_calls = _called_attributes(package_function)

    assert run_calls.count("require_published_formal_execution_lock") >= 2
    assert "validate_formal_execution_lock_pair" in run_calls
    assert "formal_execution_run_lock" in _string_literals(run_function)

    assert package_calls.count("require_published_formal_execution_lock") >= 2
    assert package_calls.count("validate_formal_execution_lock_pair") >= 2
    package_literals = _string_literals(package_function)
    assert "formal_execution_run_lock" in package_literals
    assert "formal_execution_package_lock" in package_literals


@pytest.mark.quick
def test_cpu_closure_and_complete_package_require_exact9_aggregate_first() -> None:
    """聚合闭合未启用时, 闭合与完整包入口都必须立即 fail-closed."""

    closure_function = _function_node(
        "scripts/paper_result_closure.py",
        "run_paper_result_closure_commands",
    )
    package_function = _function_node(
        "scripts/write_pilot_paper_complete_result_package.py",
        "write_pilot_paper_complete_result_package_outputs",
    )

    closure_literals = _string_literals(closure_function)
    assert any("精确9重复聚合证据" in value for value in closure_literals)
    assert _called_names(package_function) == (
        "require_exact9_randomization_aggregate_provenance",
    )


@pytest.mark.quick
def test_sd35_pipeline_requires_live_execution_lock_before_model_load() -> None:
    """真实 SD3.5 pipeline 加载前必须实时复验 Git 执行锁."""

    load_function = _function_node(
        "experiments/runtime/diffusion/sd3_pipeline_runtime.py",
        "load_pipeline",
    )
    calls = tuple(
        node
        for node in ast.walk(load_function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    )
    lock_call = next(
        node
        for node in calls
        if node.func.attr == "require_published_formal_execution_lock"
    )
    model_load_call = next(
        node for node in calls if node.func.attr == "from_pretrained"
    )

    assert lock_call.lineno < model_load_call.lineno
