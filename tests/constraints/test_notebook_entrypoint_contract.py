"""验证当前 Colab Notebook 只承担参数、挂载、CLI 调用和结果读取."""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest


NOTEBOOK_DIR = Path("paper_workflow/notebooks")
REQUIRED_NOTEBOOKS = {
    "colab_drive_cold_start_smoke.ipynb",
    "dependency_lock_review_run.ipynb",
    "semantic_watermark_image_only_run.ipynb",
    "randomization_repeat_evidence_run.ipynb",
    "external_baseline_tree_ring_run.ipynb",
    "external_baseline_gaussian_shading_run.ipynb",
    "external_baseline_shallow_diffuse_run.ipynb",
    "official_reference_t2smark_run.ipynb",
    "official_reference_tree_ring_run.ipynb",
    "official_reference_gaussian_shading_run.ipynb",
    "official_reference_shallow_diffuse_run.ipynb",
}
HOST_LAUNCHER_NOTEBOOKS = REQUIRED_NOTEBOOKS - {
    "dependency_lock_review_run.ipynb",
    "colab_drive_cold_start_smoke.ipynb",
}
ACTIVE_REPEAT_GPU_NOTEBOOKS = {
    "semantic_watermark_image_only_run.ipynb",
    "external_baseline_tree_ring_run.ipynb",
    "external_baseline_gaussian_shading_run.ipynb",
    "external_baseline_shallow_diffuse_run.ipynb",
    "official_reference_t2smark_run.ipynb",
}
OFFICIAL_REFERENCE_NOTEBOOKS = {
    "official_reference_t2smark_run.ipynb",
    "official_reference_tree_ring_run.ipynb",
    "official_reference_gaussian_shading_run.ipynb",
    "official_reference_shallow_diffuse_run.ipynb",
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
    """连接 Notebook 代码单元, 供入口边界静态检查."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", ()))
        for cell in payload.get("cells", ())
        if cell.get("cell_type") == "code"
    )


def _all_cell_source(path: Path) -> str:
    """连接全部单元格, 防止 Markdown 隐藏安装逻辑."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", ())) for cell in payload.get("cells", ()))


@pytest.mark.quick
def test_current_notebook_set_contains_formal_entrypoints_only() -> None:
    """正式入口必须存在, 已移除的分量入口不得重新出现."""

    names = {path.name for path in NOTEBOOK_DIR.glob("*.ipynb")}
    assert REQUIRED_NOTEBOOKS <= names
    assert not (FORBIDDEN_COMPONENT_NOTEBOOKS & names)


@pytest.mark.quick
@pytest.mark.parametrize(
    "notebook_name",
    sorted(OFFICIAL_REFERENCE_NOTEBOOKS),
)
def test_official_reference_notebooks_describe_one_fixed_fpr_working_point(
    notebook_name: str,
) -> None:
    """外层入口说明不得为三个论文层级发布不同 FPR 身份."""

    source = _all_cell_source(NOTEBOOK_DIR / notebook_name)
    assert "FPR=0.1" in source
    assert "FPR=0.01" not in source
    assert "FPR=0.001" not in source


@pytest.mark.quick
@pytest.mark.parametrize("notebook_name", sorted(REQUIRED_NOTEBOOKS))
def test_all_notebooks_have_one_probe_default_and_exact_checkout(notebook_name: str) -> None:
    """全部入口必须默认 probe_paper 并检出精确 detached 提交."""

    source = _code_source(NOTEBOOK_DIR / notebook_name)
    expected_defaults = () if notebook_name == "dependency_lock_review_run.ipynb" else ("probe_paper",)
    assert tuple(PAPER_RUN_DEFAULT_PATTERN.findall(source)) == expected_defaults
    assert "SLM_WM_REPOSITORY_COMMIT" in source
    assert 'r"[0-9a-f]{40}"' in source
    assert 'repository_commit + "^{commit}"' in source
    assert '["git", "checkout", "--detach", repository_commit]' in source


@pytest.mark.quick
@pytest.mark.parametrize("notebook_name", sorted(HOST_LAUNCHER_NOTEBOOKS))
def test_formal_result_notebooks_use_only_exact_host_launcher(notebook_name: str) -> None:
    """正式结果入口不得在宿主解释器导入或准备 repository 环境."""

    source = _code_source(NOTEBOOK_DIR / notebook_name)
    assert source.count("scripts/run_formal_workflow_host.py") == 1
    assert "scripts/prepare_dependency_profile.py" not in source
    assert "build_notebook_dependency_report" not in source
    assert "verify_and_publish_formal_execution" not in source
    assert "configure_paper_run_environment" not in source
    assert "from paper_workflow" not in source
    assert "import paper_workflow" not in source
    assert '"-I"' in source
    assert '"--repository-commit"' in source
    assert '"--result-path"' in source
    assert "workflow_result.json" in source


@pytest.mark.quick
@pytest.mark.parametrize("notebook_name", sorted(REQUIRED_NOTEBOOKS))
def test_notebooks_contain_no_local_dependency_or_method_logic(notebook_name: str) -> None:
    """Notebook 不得维护包清单、安装命令或论文方法实现."""

    source = _all_cell_source(NOTEBOOK_DIR / notebook_name).lower()
    forbidden_snippets = (
        "%pip",
        "pip install",
        "python -m pip",
        "conda install",
        "mamba install",
        "uv pip",
        "poetry add",
        "--upgrade",
        "micromamba",
        "colab_sd35_runtime_constraints.txt",
        "from main.",
        "from experiments.",
        "from paper_experiments.",
        "torch.autograd",
    )
    assert all(snippet not in source for snippet in forbidden_snippets)
    assert re.search(r"^\s*(def|class)\s+", source, flags=re.MULTILINE) is None


@pytest.mark.quick
def test_probe_routes_cover_nine_gpu_workflows_and_repeat_evidence() -> None:
    """正式 Notebook 集合必须覆盖9条 GPU route 和单 repeat 证据封装."""

    sources = {
        name: _code_source(NOTEBOOK_DIR / name) for name in HOST_LAUNCHER_NOTEBOOKS
    }
    combined = "\n".join(sources.values())
    expected_routes = {
        "image_only_dataset",
        "mechanism_ablation",
        "external_baseline_tree_ring",
        "external_baseline_gaussian_shading",
        "external_baseline_shallow_diffuse",
        "official_reference_t2smark",
        "official_reference_tree_ring",
        "official_reference_gaussian_shading",
        "official_reference_shallow_diffuse",
    }
    assert all(route in combined for route in expected_routes)
    assert '"repeat_evidence"' in sources[
        "randomization_repeat_evidence_run.ipynb"
    ]
    assert "SLM_WM_RANDOMIZATION_REPEAT_ID" in sources[
        "randomization_repeat_evidence_run.ipynb"
    ]
    assert '"--randomization-repeat-id"' in sources[
        "randomization_repeat_evidence_run.ipynb"
    ]
    assert '"gpu"' in sources["semantic_watermark_image_only_run.ipynb"]


@pytest.mark.quick
@pytest.mark.parametrize(
    "notebook_name",
    sorted(ACTIVE_REPEAT_GPU_NOTEBOOKS),
)
def test_active_repeat_gpu_notebooks_forward_explicit_repeat_identity(
    notebook_name: str,
) -> None:
    """活动随机化 GPU Notebook 必须把同一 repeat 传到环境与 host CLI."""

    source = _code_source(NOTEBOOK_DIR / notebook_name)
    assert source.count("SLM_WM_RANDOMIZATION_REPEAT_ID") >= 4
    assert 'os.environ["SLM_WM_RANDOMIZATION_REPEAT_ID"]' in source
    assert '"--randomization-repeat-id"' in source
    assert "/ SLM_WM_RANDOMIZATION_REPEAT_ID /" in source


@pytest.mark.quick
@pytest.mark.parametrize(
    "notebook_name",
    sorted(
        {
            "official_reference_tree_ring_run.ipynb",
            "official_reference_gaussian_shading_run.ipynb",
            "official_reference_shallow_diffuse_run.ipynb",
        }
    ),
)
def test_cross_repeat_invariant_notebooks_do_not_bind_active_repeat(
    notebook_name: str,
) -> None:
    """跨 repeat 不变官方参考入口不得被复制为9份活动随机化运行."""

    source = _code_source(NOTEBOOK_DIR / notebook_name)
    assert "SLM_WM_RANDOMIZATION_REPEAT_ID" not in source
    assert '"--randomization-repeat-id"' not in source
