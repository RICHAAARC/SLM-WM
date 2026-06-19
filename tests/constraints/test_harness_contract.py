"""验证模板仓库的 harness 基础契约。"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.harness.run_all_audits import run_all_audits


@pytest.mark.constraint
def test_project_contract_exists() -> None:
    """项目契约必须存在, 作为所有修改前置依据。"""
    assert Path(".codex/project_contract.md").exists()


@pytest.mark.constraint
def test_harness_audits_pass_for_template() -> None:
    """模板仓库自身必须通过内置 harness 审计。"""
    summary = run_all_audits(Path.cwd())
    assert summary["overall_decision"] == "pass"


@pytest.mark.constraint
def test_main_core_package_exists() -> None:
    """论文研究项目模板必须使用 main 作为核心包目录。"""
    assert Path("main/__init__.py").exists()
    assert not Path("src").exists()
