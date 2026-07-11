"""约束依赖锁资格化 Notebook 与单一脚本入口边界."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = (
    ROOT / "paper_workflow/notebooks/dependency_lock_review_run.ipynb"
)
README_PATH = ROOT / "paper_workflow/notebooks/README.md"
SCRIPT_PATH = ROOT / "scripts/write_dependency_lock_review_bundle.py"


def _notebook() -> dict[str, object]:
    """读取资格化 Notebook JSON."""

    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _source(cell: dict[str, object]) -> str:
    """合并单个 Notebook cell source."""

    value = cell.get("source", [])
    return "".join(value) if isinstance(value, list) else str(value)


def _all_source() -> str:
    """合并全部 Markdown 与代码, 便于检查职责越界文本."""

    return "\n".join(_source(cell) for cell in _notebook()["cells"])


def _code_source() -> str:
    """只合并代码 cell, 避免 Markdown 示例影响代码约束."""

    return "\n".join(
        _source(cell)
        for cell in _notebook()["cells"]
        if cell.get("cell_type") == "code"
    )


@pytest.mark.constraint
def test_review_notebook_is_valid_thin_colab_entrypoint() -> None:
    """Notebook 必须是带唯一 cell id 的有效 v4 薄入口."""

    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert int(notebook["nbformat_minor"]) >= 5
    cells = notebook["cells"]
    assert isinstance(cells, list) and cells
    cell_ids = [cell.get("id") for cell in cells]
    assert all(isinstance(cell_id, str) and cell_id for cell_id in cell_ids)
    assert len(cell_ids) == len(set(cell_ids))
    for cell in cells:
        if cell.get("cell_type") == "code":
            compile(_source(cell), str(NOTEBOOK_PATH), "exec")


@pytest.mark.constraint
def test_review_notebook_publishes_exact_code_lock_before_single_script() -> None:
    """入口必须先挂载 Drive 和发布精确代码锁, 再调用唯一资格化脚本."""

    source = _all_source()
    code = _code_source()
    assert code.count('PROFILE_ID = "workflow_orchestrator"') == 1
    assert code.count("scripts/write_dependency_lock_review_bundle.py") == 1
    assert "drive.mount(" in code
    assert 're.fullmatch(r"[0-9a-f]{40}", repository_commit)' in code
    assert '["git", "checkout", "--detach", repository_commit]' in code
    assert "verify_and_publish_formal_execution(workspace_dir, repository_commit)" in code
    assert '"--profile",\n    PROFILE_ID' in code
    assert '"--drive-output-dir"' in code
    assert source.index("drive.mount(") < source.index("git\", \"checkout")
    assert source.index("verify_and_publish_formal_execution") < source.index(
        "scripts/write_dependency_lock_review_bundle.py"
    )
    assert "五个科学 profile" in source
    assert "只有 `workflow_orchestrator` 候选允许在 Notebook 当前解释器中生成" in source


@pytest.mark.constraint
def test_review_notebook_contains_no_dependency_or_copy_implementation() -> None:
    """Notebook 不得内嵌解析、包列表、解释器创建或复制实现."""

    code = _code_source()
    forbidden_tokens = (
        "%pip",
        "pip install",
        "--upgrade",
        "requirements.txt",
        "materialize_dependency_lock_candidate",
        "provision_isolated_dependency_python",
        "prepare_dependency_profile",
        "shutil",
        "copy2",
        "from experiments",
        "import experiments",
        "from main",
        "import main",
    )
    for token in forbidden_tokens:
        assert token not in code
    assert re.search(r"^\s*(def|class)\s+", code, flags=re.MULTILINE) is None
    package_names = (
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "open_clip_torch",
        "scikit-learn",
    )
    for line in code.splitlines():
        assert sum(name in line.lower() for name in package_names) <= 1


@pytest.mark.constraint
def test_review_bundle_cli_exposes_only_profile_and_optional_drive_arguments() -> None:
    """外层 CLI 不得暴露第二套解析、解释器创建或复制配置."""

    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    argument_names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        first_argument = node.args[0]
        if isinstance(first_argument, ast.Constant) and isinstance(
            first_argument.value, str
        ):
            argument_names.append(first_argument.value)
    assert argument_names == ["--profile", "--drive-output-dir"]


@pytest.mark.constraint
def test_readme_defines_orchestrator_first_isolated_python_order() -> None:
    """文档必须固定 orchestrator 优先和五个科学子环境顺序."""

    readme = README_PATH.read_text(encoding="utf-8")
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "dependency_lock_review_run.ipynb" in readme
    assert "write_dependency_lock_review_bundle.py" in readme
    orchestrator_index = readme.index("workflow_orchestrator")
    isolated_index = readme.index("五个科学 profile")
    assert orchestrator_index < isolated_index
    assert (
        "CURRENT_INTERPRETER_PROFILE_ID = WORKFLOW_ORCHESTRATOR_PROFILE_ID"
        in script
    )
    assert "if profile_id != CURRENT_INTERPRETER_PROFILE_ID" in script
    assert "sd35_method_runtime_gpu" in readme
    assert "t2smark_sd35_gpu" in readme
    assert "仅在显式提供 `--drive-output-dir` 时才复制到 Drive" in readme
    assert "supports_paper_claim=false" in readme
