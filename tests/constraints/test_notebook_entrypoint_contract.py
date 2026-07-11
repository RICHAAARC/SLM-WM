"""验证当前 Colab Notebook 只承担环境准备与仓库入口调用。"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

NOTEBOOK_DIR = Path("paper_workflow/notebooks")
REQUIRED_NOTEBOOKS = {
    "semantic_watermark_image_only_run.ipynb",
    "paper_result_closure_run.ipynb",
    "external_baseline_tree_ring_run.ipynb",
    "external_baseline_gaussian_shading_run.ipynb",
    "external_baseline_shallow_diffuse_run.ipynb",
    "official_reference_t2smark_run.ipynb",
}
FORBIDDEN_COMPONENT_NOTEBOOKS = {
    "aligned_rescoring_run.ipynb",
    "attention_geometry_capture_run.ipynb",
    "attention_latent_injection_run.ipynb",
    "conventional_geometric_attack_evaluation_run.ipynb",
    "real_attack_evaluation_run.ipynb",
    "runtime_method_precheck_run.ipynb",
    "threshold_calibration_run.ipynb",
    "dataset_level_quality_run.ipynb",
}
PAPER_RUN_DEFAULT_PATTERN = re.compile(
    r'^\s*SLM_WM_PAPER_RUN_NAME\s*=\s*["\']([^"\']+)["\']\s*$',
    re.MULTILINE,
)


def _code_source(path: Path) -> str:
    """连接 Notebook 的代码单元, 供入口边界静态检查。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", ()))
        for cell in payload.get("cells", ())
        if cell.get("cell_type") == "code"
    )


def _paper_run_defaults(path: Path) -> tuple[str, ...]:
    """解析 Notebook 中显式发布给环境 helper 的运行层级默认值."""

    return tuple(PAPER_RUN_DEFAULT_PATTERN.findall(_code_source(path)))


@pytest.mark.quick
def test_current_notebook_set_contains_formal_entrypoints_only() -> None:
    """正式入口必须存在, 已移除的分量诊断入口不得重新出现。"""

    names = {path.name for path in NOTEBOOK_DIR.glob("*.ipynb")}
    assert REQUIRED_NOTEBOOKS <= names
    assert not (FORBIDDEN_COMPONENT_NOTEBOOKS & names)


@pytest.mark.quick
@pytest.mark.parametrize(
    "notebook_path",
    tuple(sorted(NOTEBOOK_DIR.glob("*.ipynb"))),
    ids=lambda path: path.name,
)
def test_all_notebook_entrypoints_have_one_probe_paper_default(
    notebook_path: Path,
) -> None:
    """解析全部 Notebook, 正式入口只允许唯一的 probe_paper 默认值."""

    defaults = _paper_run_defaults(notebook_path)

    assert defaults == ("probe_paper",)
    assert "pilot_paper" not in defaults


@pytest.mark.quick
@pytest.mark.parametrize(
    "notebook_path",
    tuple(sorted(NOTEBOOK_DIR.glob("*.ipynb"))),
    ids=lambda path: path.name,
)
def test_all_notebooks_require_clean_detached_full_commit(
    notebook_path: Path,
) -> None:
    """全部 Colab 入口必须在运行前锁定完整 detached Git 提交."""

    source = _code_source(notebook_path)

    assert "SLM_WM_REPOSITORY_COMMIT" in source
    assert "SLM_WM_REPOSITORY_REF" not in source
    assert "verify_and_publish_formal_execution" in source
    assert '["git", "checkout", "--detach", repository_commit]' in source
    assert 'os.environ.get("SLM_WM_REPOSITORY_COMMIT", "").strip()' not in source
    assert 'repository_commit + "^{commit}"' in source
    assert 'r"[0-9a-f]{40}"' in source


@pytest.mark.quick
@pytest.mark.parametrize(
    "notebook_path",
    tuple(sorted(NOTEBOOK_DIR.glob("*.ipynb"))),
    ids=lambda path: path.name,
)
def test_notebooks_lock_repository_before_dependency_install(
    notebook_path: Path,
) -> None:
    """Notebook 必须先验证代码身份, 再安装会改变正式运行环境的依赖."""

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    lock_cell_index = next(
        index
        for index, source in enumerate(code_cells)
        if "verify_and_publish_formal_execution" in source
    )
    install_cell_indexes = [
        index
        for index, source in enumerate(code_cells)
        if "%pip install" in source
    ]

    assert all(lock_cell_index < install_index for install_index in install_cell_indexes)
    if install_cell_indexes:
        configure_cell_index = next(
            index
            for index, source in enumerate(code_cells)
            if "configure_paper_run_environment" in source
        )
        dependency_report_cell_index = next(
            index
            for index, source in enumerate(code_cells)
            if "build_notebook_dependency_report" in source
        )
        assert all(
            install_index < configure_cell_index
            for install_index in install_cell_indexes
        )
        assert all(
            install_index < dependency_report_cell_index
            for install_index in install_cell_indexes
        )


@pytest.mark.quick
@pytest.mark.parametrize("notebook_name", sorted(REQUIRED_NOTEBOOKS))
def test_notebook_contains_no_method_or_experiment_definition(notebook_name: str) -> None:
    """Notebook 不得定义函数、类或直接实现水印与实验算法。"""

    source = _code_source(NOTEBOOK_DIR / notebook_name)
    assert "def " not in source
    assert "class " not in source
    assert "from main." not in source
    assert "from experiments." not in source
    assert "from paper_experiments." not in source
    assert "torch.autograd" not in source
    assert "pipeline(" not in source


@pytest.mark.quick
def test_semantic_watermark_notebook_delegates_to_workflow_helper() -> None:
    """主方法 Notebook 必须调用外层 helper, 不能承载项目代码。"""

    source = _code_source(NOTEBOOK_DIR / "semantic_watermark_image_only_run.ipynb")
    assert "paper_workflow.colab_utils.semantic_watermark_image_only" in source
