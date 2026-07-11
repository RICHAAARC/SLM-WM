"""验证静态与动态依赖边界审计."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.harness.audits.audit_dependency_boundaries import run_audit


def _write_boundary_fixture(root: Path, files: dict[str, str]) -> Path:
    """构造包含全部受治理层的最小审计仓库."""
    for directory in ("main", "experiments", "paper_experiments", "scripts"):
        package_root = root / directory
        package_root.mkdir(parents=True, exist_ok=True)
        (package_root / "__init__.py").write_text("", encoding="utf-8")
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


@pytest.mark.constraint
def test_dependency_boundary_audit_rejects_dynamic_outer_layer_references(tmp_path: Path) -> None:
    """内层不得用模块字符串、脚本路径或命令字符串绕过导入审计."""
    root = _write_boundary_fixture(
        tmp_path,
        {
            "main/runtime.py": 'SESSION_MODULE = "paper_workflow.session"\n',
            "experiments/session.py": 'command = run_child(("scripts/run_method.py",))\n',
            "experiments/shell.py": 'result = execute("python -m paper_experiments.runner")\n',
            "paper_experiments/runner.py": 'SCRIPT_PATH = f"scripts/{entry_name}"\n',
            "scripts/entry.py": 'result = launch(["python", r"paper_workflow\\entry.py"])\n',
            "scripts/module.py": 'ENTRY_MODULE = "paper_workflow.entry"\n',
        },
    )

    report = run_audit(root)

    assert report["decision"] == "fail"
    dynamic_violations = [
        item
        for item in report["violations"]
        if item["reason"] == "forbidden_dynamic_dependency_for_extraction_boundary"
    ]
    assert {item["path"] for item in dynamic_violations} == {
        "main/runtime.py",
        "experiments/session.py",
        "experiments/shell.py",
        "paper_experiments/runner.py",
        "scripts/entry.py",
        "scripts/module.py",
    }
    assert all(item["line"] == 1 for item in dynamic_violations)
    assert all(item["forbidden_prefix"] in item["reference"] for item in dynamic_violations)


@pytest.mark.constraint
def test_dependency_boundary_audit_ignores_docs_and_script_exclusion_metadata(tmp_path: Path) -> None:
    """文档字符串和独立发布排除清单不应被误判为运行时依赖."""
    root = _write_boundary_fixture(
        tmp_path,
        {
            "main/runtime.py": '"""说明 scripts/run_method.py 不属于核心方法包."""\n',
            "experiments/runtime.py": '"""说明 paper_workflow/ 仅提供外层入口."""\n',
            "paper_experiments/source.py": 'SOURCE_PATH = "external_baseline/source/model.py"\n',
            "scripts/extraction.py": 'EXCLUDED_PATHS = ("paper_workflow", "paper_workflow/")\n',
        },
    )

    report = run_audit(root)

    assert report["decision"] == "pass"


@pytest.mark.constraint
def test_dependency_boundary_audit_covers_python_files_directly_under_main(tmp_path: Path) -> None:
    """直接位于 `main/` 的 Python 文件也必须服从核心方法边界."""
    root = _write_boundary_fixture(
        tmp_path,
        {"main/runtime.py": "from experiments.runtime import loader\n"},
    )

    report = run_audit(root)

    assert report["decision"] == "fail"
    assert any(
        item["path"] == "main/runtime.py"
        and item["reason"] == "forbidden_import_for_extraction_boundary"
        and item["imported_module"] == "experiments.runtime"
        for item in report["violations"]
    )
