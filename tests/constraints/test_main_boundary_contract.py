"""验证冻结后的 `main/` 核心包边界。"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from scripts.write_core_package_boundary_outputs import build_core_boundary_report


@pytest.mark.constraint
def test_main_package_imports_without_outer_runtime() -> None:
    """核心包必须可以独立导入, 不依赖外层实验、Notebook 或治理运行时。"""
    imported_module = importlib.import_module("main")
    assert imported_module.__name__ == "main"


@pytest.mark.constraint
def test_main_package_boundary_rejects_forbidden_outer_imports() -> None:
    """核心包不得导入外层 workflow、脚本、测试、harness 或外部 baseline。"""
    report = build_core_boundary_report(Path.cwd())
    assert report["decision"] == "pass", report["violations"]


@pytest.mark.constraint
def test_main_package_contains_core_minimal_layout() -> None:
    """核心包必须包含最小语义子包。"""
    required_paths = (
        Path("main/core"),
        Path("main/methods"),
        Path("main/protocol"),
        Path("main/analysis"),
        Path("main/cli"),
    )
    assert all(path.is_dir() for path in required_paths)
