"""验证模板仓库的 harness 基础契约。"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.harness.lib.naming_rules import (
    has_reserved_progress_marker,
    is_allowed_file_name,
)
from tools.harness.audits.audit_naming_conventions import (
    run_audit as run_naming_audit,
)
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
def test_reserved_progress_marker_detection() -> None:
    """过程标记词必须被统一检测, 语义名称不得误报。"""
    blocked_tokens = ("sta" + "ge", "pha" + "se", "\u9636\u6bb5")
    assert all(has_reserved_progress_marker(token) for token in blocked_tokens)
    assert not has_reserved_progress_marker("core_package_boundary_freeze")


@pytest.mark.constraint
def test_git_attributes_is_an_allowed_repository_control_file() -> None:
    """逐字节治理输入需要允许使用 Git 行尾属性固定 LF."""

    assert is_allowed_file_name(".gitattributes")
    assert is_allowed_file_name("prompt_selection_manifest.jsonl")


@pytest.mark.constraint
def test_naming_audit_separates_external_tool_metadata(
    tmp_path: Path,
) -> None:
    """外部工具元数据不受项目命名约束, 正式项目根仍必须完整审计。"""

    (tmp_path / ".claude" / "tool-owned-name").mkdir(parents=True)
    (tmp_path / ".gitnexus" / "parse-cache").mkdir(parents=True)
    (tmp_path / "CLAUDE.md").write_text("外部工具入口\n", encoding="utf-8")
    (tmp_path / "paper_experiments" / "bad-name").mkdir(parents=True)

    report = run_naming_audit(tmp_path)

    assert report["decision"] == "fail"
    assert report["violations"] == [
        {
            "path": str(Path("paper_experiments") / "bad-name"),
            "reason": "directory_name_not_snake_case",
        }
    ]


@pytest.mark.constraint
def test_main_core_package_exists() -> None:
    """论文研究项目模板必须使用 main 作为核心包目录。"""
    assert Path("main/__init__.py").exists()
    assert not Path("src").exists()
