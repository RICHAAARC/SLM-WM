"""验证当前 Colab Notebook 只承担环境准备与仓库入口调用。"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

NOTEBOOK_DIR = Path("paper_workflow/notebooks")
REQUIRED_NOTEBOOKS = {
    "colab_drive_cold_start_smoke.ipynb",
    "semantic_watermark_image_only_run.ipynb",
    "paper_result_closure_run.ipynb",
    "external_baseline_tree_ring_run.ipynb",
    "external_baseline_gaussian_shading_run.ipynb",
    "external_baseline_shallow_diffuse_run.ipynb",
    "official_reference_t2smark_run.ipynb",
    "official_reference_tree_ring_run.ipynb",
    "official_reference_gaussian_shading_run.ipynb",
    "official_reference_shallow_diffuse_run.ipynb",
    "official_reference_tree_ring_run.ipynb",
    "official_reference_gaussian_shading_run.ipynb",
    "official_reference_shallow_diffuse_run.ipynb",
}
NOTEBOOK_DEPENDENCY_PROFILES = {
    "semantic_watermark_image_only_run.ipynb": "workflow_orchestrator",
    "external_baseline_tree_ring_run.ipynb": "workflow_orchestrator",
    "external_baseline_gaussian_shading_run.ipynb": "workflow_orchestrator",
    "external_baseline_shallow_diffuse_run.ipynb": "workflow_orchestrator",
    "official_reference_t2smark_run.ipynb": "workflow_orchestrator",
    "official_reference_tree_ring_run.ipynb": "workflow_orchestrator",
    "official_reference_gaussian_shading_run.ipynb": "workflow_orchestrator",
    "official_reference_shallow_diffuse_run.ipynb": "workflow_orchestrator",
    "paper_result_closure_run.ipynb": "workflow_orchestrator",
    "colab_drive_cold_start_smoke.ipynb": "workflow_orchestrator",
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
FORMAL_NOTEBOOK_PATHS = tuple(
    NOTEBOOK_DIR / notebook_name for notebook_name in sorted(REQUIRED_NOTEBOOKS)
)
ISOLATED_SCIENTIFIC_NOTEBOOKS = (
    "external_baseline_tree_ring_run.ipynb",
    "external_baseline_gaussian_shading_run.ipynb",
    "external_baseline_shallow_diffuse_run.ipynb",
    "official_reference_t2smark_run.ipynb",
)


def _code_source(path: Path) -> str:
    """连接 Notebook 的代码单元, 供入口边界静态检查。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", ()))
        for cell in payload.get("cells", ())
        if cell.get("cell_type") == "code"
    )


def _all_cell_source(path: Path) -> str:
    """连接 Notebook 的全部单元格, 防止在 Markdown 中隐藏依赖命令。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", ())) for cell in payload.get("cells", ()))


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
    FORMAL_NOTEBOOK_PATHS,
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
    FORMAL_NOTEBOOK_PATHS,
    ids=lambda path: path.name,
)
def test_notebooks_lock_repository_before_dependency_profile_preparation(
    notebook_path: Path,
) -> None:
    """Notebook 必须先验证代码身份, 再调用仓库统一依赖 profile CLI。"""

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
    profile_cell_indexes = [
        index
        for index, source in enumerate(code_cells)
        if "scripts/prepare_dependency_profile.py" in source
    ]

    assert len(profile_cell_indexes) == 1
    profile_cell_index = profile_cell_indexes[0]
    assert lock_cell_index < profile_cell_index
    configure_cell_indexes = [
        index
        for index, source in enumerate(code_cells)
        if "configure_paper_run_environment" in source
    ]
    assert all(profile_cell_index < index for index in configure_cell_indexes)


@pytest.mark.quick
@pytest.mark.parametrize(
    ("notebook_name", "expected_profile"),
    tuple(sorted(NOTEBOOK_DEPENDENCY_PROFILES.items())),
)
def test_notebooks_only_select_registered_dependency_profile(
    notebook_name: str,
    expected_profile: str,
) -> None:
    """每个正式 Notebook 只能选择一个 profile, 不得维护第二套安装协议。"""

    source = _code_source(NOTEBOOK_DIR / notebook_name)
    profile_pattern = re.compile(r'^DEPENDENCY_PROFILE_ID = "([a-z0-9_]+)"$', re.MULTILINE)

    assert tuple(profile_pattern.findall(source)) == (expected_profile,)
    assert source.count("scripts/prepare_dependency_profile.py") == 1
    assert source.count("build_notebook_dependency_report(") == 1
    assert "dependency_report = build_notebook_dependency_report(DEPENDENCY_PROFILE_ID)" in source
    assert (
        '["python", "scripts/prepare_dependency_profile.py", "--profile", DEPENDENCY_PROFILE_ID]'
        in source
    )


@pytest.mark.quick
@pytest.mark.parametrize(
    "notebook_path",
    tuple(sorted(NOTEBOOK_DIR.glob("*.ipynb"))),
    ids=lambda path: path.name,
)
def test_notebooks_contain_no_local_dependency_install_logic(notebook_path: Path) -> None:
    """依赖求解、包清单和浮动下载必须全部收敛到 repository CLI。"""

    source = _all_cell_source(notebook_path).lower()
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
        "requirements.txt",
        "colab_sd35_runtime_constraints.txt",
        "diffusers transformers accelerate",
        "packaging huggingface_hub",
    )

    assert all(snippet not in source for snippet in forbidden_snippets)
    assert re.search(r'["\']pip["\']\s*,\s*["\']install["\']', source) is None
    package_names = (
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "open_clip_torch",
        "scikit-learn",
        "datasets",
    )
    for line in source.splitlines():
        listed_names = [name for name in package_names if name in line]
        assert len(listed_names) <= 1, f"Notebook 包清单未收敛: {line}"


@pytest.mark.quick
@pytest.mark.parametrize("notebook_name", ISOLATED_SCIENTIFIC_NOTEBOOKS)
def test_scientific_notebooks_prepare_cpu_orchestrator_only(notebook_name: str) -> None:
    """method-faithful 与 T2SMark Notebook 不得在父解释器准备科学 profile."""

    source = _code_source(NOTEBOOK_DIR / notebook_name)

    assert 'DEPENDENCY_PROFILE_ID = "workflow_orchestrator"' in source
    assert 'DEPENDENCY_PROFILE_ID = "sd35_method_runtime_gpu"' not in source
    assert 'DEPENDENCY_PROFILE_ID = "t2smark_sd35_gpu"' not in source
    assert "import torch" not in source
    assert "from huggingface_hub import login" not in source


@pytest.mark.quick
def test_notebook_entrypoint_routes_two_workflows_through_shared_isolated_dispatch() -> None:
    """最外层入口只选择共享 dispatch, 不直接导入两个科学 runner."""

    source = Path("paper_workflow/notebook_utils/notebook_entrypoint.py").read_text(
        encoding="utf-8"
    )

    assert "run_isolated_scientific_workflow" in source
    assert "run_default_external_baseline_method_faithful_plan" not in source
    assert "run_default_t2smark_formal_reproduction_plan" not in source


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
    assert "import torch" not in source
    assert "torch.autograd" not in source
    assert "pipeline(" not in source


@pytest.mark.quick
def test_semantic_watermark_notebook_delegates_to_workflow_helper() -> None:
    """主方法 Notebook 必须调用外层 helper, 不能承载项目代码。"""

    source = _code_source(NOTEBOOK_DIR / "semantic_watermark_image_only_run.ipynb")
    assert "paper_workflow.colab_utils.semantic_watermark_image_only" in source
    assert 'DEPENDENCY_PROFILE_ID = "workflow_orchestrator"' in source
    assert "sd35_method_runtime_gpu" not in source
    assert "import torch" not in source
    assert "nvidia-smi" in source
