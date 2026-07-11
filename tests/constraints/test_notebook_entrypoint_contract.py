"""验证当前 Colab Notebook 只承担环境准备与仓库入口调用。"""

from __future__ import annotations

import json
from pathlib import Path

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


def _code_source(path: Path) -> str:
    """连接 Notebook 的代码单元, 供入口边界静态检查。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", ()))
        for cell in payload.get("cells", ())
        if cell.get("cell_type") == "code"
    )


@pytest.mark.quick
def test_current_notebook_set_contains_formal_entrypoints_only() -> None:
    """正式入口必须存在, 已移除的分量诊断入口不得重新出现。"""

    names = {path.name for path in NOTEBOOK_DIR.glob("*.ipynb")}
    assert REQUIRED_NOTEBOOKS <= names
    assert not (FORBIDDEN_COMPONENT_NOTEBOOKS & names)


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
