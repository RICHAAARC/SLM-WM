"""验证 Colab Notebook 入口契约."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from paper_workflow.colab_utils.minimal_latent_injection import package_injection_outputs
from paper_workflow.colab_utils.aligned_rescoring import package_aligned_rescoring_outputs
from paper_workflow.colab_utils.attention_latent_injection import package_attention_latent_injection_outputs
from paper_workflow.colab_utils.attention_geometry_capture import package_attention_geometry_outputs
from paper_workflow.colab_utils.external_baseline_gpu_smoke import package_external_baseline_gpu_smoke_outputs
from paper_workflow.colab_utils.real_attack_evaluation import package_real_attack_evaluation_outputs
from paper_workflow.colab_utils.dataset_level_quality import package_dataset_level_quality_outputs
from paper_workflow.colab_utils.threshold_calibration import package_threshold_calibration_outputs
from paper_workflow.colab_utils.sd_runtime_cold_start import package_probe_outputs
from tools.harness.lib.naming_rules import is_allowed_file_name


RUNTIME_METHOD_PRECHECK_NOTEBOOK_PATH = Path("paper_workflow/runtime_method_precheck_run.ipynb")
DRIVE_COLD_START_NOTEBOOK_PATH = Path("paper_workflow/colab_drive_cold_start_smoke.ipynb")
ATTENTION_GEOMETRY_NOTEBOOK_PATH = Path("paper_workflow/attention_geometry_capture_run.ipynb")
ATTENTION_LATENT_INJECTION_NOTEBOOK_PATH = Path("paper_workflow/attention_latent_injection_run.ipynb")
ALIGNED_RESCORING_NOTEBOOK_PATH = Path("paper_workflow/aligned_rescoring_run.ipynb")
THRESHOLD_CALIBRATION_NOTEBOOK_PATH = Path("paper_workflow/threshold_calibration_run.ipynb")
REAL_ATTACK_EVALUATION_NOTEBOOK_PATH = Path("paper_workflow/real_attack_evaluation_run.ipynb")
EXTERNAL_BASELINE_GPU_SMOKE_NOTEBOOK_PATH = Path("paper_workflow/external_baseline_gpu_smoke_run.ipynb")
DATASET_LEVEL_QUALITY_NOTEBOOK_PATH = Path("paper_workflow/dataset_level_quality_run.ipynb")
T2SMARK_OFFICIAL_REPRODUCTION_NOTEBOOK_PATH = Path("paper_workflow/t2smark_full_main_reproduction_run.ipynb")
TREE_RING_OFFICIAL_REFERENCE_NOTEBOOK_PATH = Path("paper_workflow/tree_ring_official_reference_run.ipynb")
GAUSSIAN_SHADING_OFFICIAL_REFERENCE_NOTEBOOK_PATH = Path("paper_workflow/gaussian_shading_official_reference_run.ipynb")
SHALLOW_DIFFUSE_OFFICIAL_REFERENCE_NOTEBOOK_PATH = Path("paper_workflow/shallow_diffuse_official_reference_run.ipynb")
PILOT_PAPER_RESULT_CLOSURE_NOTEBOOK_PATH = Path("paper_workflow/pilot_paper_result_closure_run.ipynb")
NOTEBOOK_PATHS = (
    RUNTIME_METHOD_PRECHECK_NOTEBOOK_PATH,
    DRIVE_COLD_START_NOTEBOOK_PATH,
    ATTENTION_GEOMETRY_NOTEBOOK_PATH,
    ATTENTION_LATENT_INJECTION_NOTEBOOK_PATH,
    ALIGNED_RESCORING_NOTEBOOK_PATH,
    THRESHOLD_CALIBRATION_NOTEBOOK_PATH,
    REAL_ATTACK_EVALUATION_NOTEBOOK_PATH,
    EXTERNAL_BASELINE_GPU_SMOKE_NOTEBOOK_PATH,
    DATASET_LEVEL_QUALITY_NOTEBOOK_PATH,
    T2SMARK_OFFICIAL_REPRODUCTION_NOTEBOOK_PATH,
    TREE_RING_OFFICIAL_REFERENCE_NOTEBOOK_PATH,
    GAUSSIAN_SHADING_OFFICIAL_REFERENCE_NOTEBOOK_PATH,
    SHALLOW_DIFFUSE_OFFICIAL_REFERENCE_NOTEBOOK_PATH,
    PILOT_PAPER_RESULT_CLOSURE_NOTEBOOK_PATH,
)
COLAB_RUNTIME_CONSTRAINTS_PATH = Path("configs/colab_sd35_runtime_constraints.txt")
COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND = (
    "%pip install -q --upgrade diffusers transformers accelerate safetensors sentencepiece protobuf huggingface_hub"
)
PAIR_PERCEPTUAL_DEPENDENCY_INSTALL_COMMAND = "%pip install -q --upgrade lpips"
EXTERNAL_BASELINE_DEPENDENCY_INSTALL_COMMAND = (
    "%pip install -q --upgrade diffusers transformers accelerate safetensors sentencepiece protobuf "
    "huggingface_hub open_clip_torch scikit-learn scipy pandas datasets tqdm"
)


@pytest.mark.constraint
def test_ipynb_names_are_allowed_when_semantic() -> None:
    """语义化 Notebook 文件名应被命名治理接受."""
    assert all(is_allowed_file_name(notebook_path.name) for notebook_path in NOTEBOOK_PATHS)


@pytest.mark.constraint
def test_colab_notebooks_have_no_stored_outputs() -> None:
    """Colab 入口不应提交已执行输出."""
    for notebook_path in NOTEBOOK_PATHS:
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]

        assert code_cells
        assert all(cell.get("execution_count") is None for cell in code_cells)
        assert all(cell.get("outputs") == [] for cell in code_cells)


@pytest.mark.constraint
def test_colab_runtime_constraints_document_known_working_environment() -> None:
    """Colab 依赖约束记录应保存已验证组合, 但不能强制安装平台提供的 torch."""
    text = COLAB_RUNTIME_CONSTRAINTS_PATH.read_text(encoding="utf-8")
    requirement_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND in text
    assert PAIR_PERCEPTUAL_DEPENDENCY_INSTALL_COMMAND in text
    assert "diffusers==0.38.0" in requirement_lines
    assert "transformers==5.12.1" in requirement_lines
    assert "accelerate==1.14.0" in requirement_lines
    assert "huggingface_hub==1.20.1" in requirement_lines
    assert "numpy==2.0.2" in requirement_lines
    assert all(not line.startswith("torch==") for line in requirement_lines)


@pytest.mark.constraint
def test_colab_notebook_delegates_runtime_and_method_precheck_logic_to_helpers() -> None:
    """合并后的预检 Notebook 必须同时调度运行时诊断和最小机制 helper。"""
    payload = json.loads(RUNTIME_METHOD_PRECHECK_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    joined_source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    first_code_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
    first_code_source = "".join(first_code_cell.get("source", []))

    assert "paper_workflow.colab_utils.sd_runtime_cold_start" in joined_source
    assert "run_default_model_plan" in joined_source
    assert "package_probe_outputs" in joined_source
    assert "paper_workflow.colab_utils.minimal_latent_injection" in joined_source
    assert "run_default_injection_plan" in joined_source
    assert "package_injection_outputs" in joined_source
    assert "SLM_WM_RUNTIME_MODEL_SELECTION', 'auto'" in joined_source
    assert "SLM_WM_INJECTION_MODEL_SELECTION', 'auto'" in joined_source
    assert "/content/drive/MyDrive/SLM/runtime_method_precheck" in joined_source
    assert "real_sd_runtime_probe_package_" in joined_source
    assert "minimal_latent_injection_package_" in joined_source
    assert "drive.mount('/content/drive')" in first_code_source
    assert "datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%sz')" in joined_source
    assert "['git', 'rev-parse', '--short', 'HEAD']" in joined_source
    assert COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND in joined_source
    assert "--force-reinstall" not in joined_source
    assert "numpy pillow" not in joined_source
    assert "del sys.modules" not in joined_source
    assert "module_name == 'numpy'" not in joined_source
    assert '"diffusers==' not in joined_source
    assert '"transformers==' not in joined_source
    assert '"accelerate==' not in joined_source


@pytest.mark.constraint
def test_colab_drive_notebook_delegates_workflow_logic_to_helper() -> None:
    """Drive workflow Notebook 必须调用 repository helper, 不直接拼接受治理清单。"""
    cold_start_payload = json.loads(DRIVE_COLD_START_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cold_start_source = "\n".join("".join(cell.get("source", [])) for cell in cold_start_payload["cells"])
    first_cold_start_code = next(cell for cell in cold_start_payload["cells"] if cell["cell_type"] == "code")

    assert "paper_workflow.colab_utils.drive_workflow" in cold_start_source
    assert "run_colab_drive_workflow" in cold_start_source
    assert "reload_smoke_record.jsonl" in cold_start_source
    assert "drive.mount('/content/drive')" in "".join(first_cold_start_code.get("source", []))
    assert "/content/drive/MyDrive/SLM" in cold_start_source
    assert "manifest.json" not in cold_start_source
    assert "json.dumps" not in cold_start_source


@pytest.mark.constraint
def test_colab_notebook_delegates_attention_geometry_logic_to_helper() -> None:
    """Notebook 必须复用 repository helper 执行真实 attention 捕获与几何重建。"""
    payload = json.loads(ATTENTION_GEOMETRY_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    joined_source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    first_code_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
    first_code_source = "".join(first_code_cell.get("source", []))

    assert "paper_workflow.colab_utils.attention_geometry_capture" in joined_source
    assert "run_default_attention_geometry_plan" in joined_source
    assert "package_attention_geometry_outputs" in joined_source
    assert "pilot_paper_fixed_fpr_0_01" in joined_source
    assert "configs/paper_main_pilot_paper_prompts.txt" in joined_source
    assert "drive.mount('/content/drive')" in first_code_source
    assert "/content/drive/MyDrive/SLM/pilot_paper_results/attention_geometry" in joined_source
    assert "attention_geometry_ready" in joined_source
    assert "SLM_WM_ATTENTION_CAPTURE_COUNT', '16'" in joined_source
    assert "SLM_WM_ATTENTION_TOKEN_COUNT', '32'" in joined_source
    assert "datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%sz')" in joined_source
    assert "['git', 'rev-parse', '--short', 'HEAD']" in joined_source
    assert "archive_name=archive_name" in joined_source
    assert COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND in joined_source
    assert "--force-reinstall" not in joined_source
    assert "numpy pillow" not in joined_source
    assert "del sys.modules" not in joined_source
    assert '"diffusers==' not in joined_source
    assert '"transformers==' not in joined_source


@pytest.mark.constraint
def test_colab_notebook_delegates_attention_latent_injection_logic_to_helper() -> None:
    """Notebook 必须复用 repository helper 执行 attention-relative latent update。"""
    payload = json.loads(ATTENTION_LATENT_INJECTION_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    joined_source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    first_code_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
    first_code_source = "".join(first_code_cell.get("source", []))

    assert "paper_workflow.colab_utils.attention_latent_injection" in joined_source
    assert "run_default_attention_latent_injection_plan" in joined_source
    assert "package_attention_latent_injection_outputs" in joined_source
    assert "pilot_paper_fixed_fpr_0_01" in joined_source
    assert "configs/paper_main_pilot_paper_prompts.txt" in joined_source
    assert "drive.mount('/content/drive')" in first_code_source
    assert "/content/drive/MyDrive/SLM/pilot_paper_results/attention_latent_injection" in joined_source
    assert "/content/drive/MyDrive/SLM/pilot_paper_results/attention_geometry" in joined_source
    assert "SLM_WM_ATTENTION_SUBSPACE_RECORDS', '128'" in joined_source
    assert "SLM_WM_ATTENTION_RUNTIME_STRENGTH', '0.025'" in joined_source
    assert "datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%sz')" in joined_source
    assert "['git', 'rev-parse', '--short', 'HEAD']" in joined_source
    assert "archive_name=archive_name" in joined_source
    assert COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND in joined_source
    assert "--force-reinstall" not in joined_source
    assert "numpy pillow" not in joined_source
    assert "del sys.modules" not in joined_source
    assert '"diffusers==' not in joined_source
    assert '"transformers==' not in joined_source


@pytest.mark.constraint
def test_colab_notebook_delegates_aligned_rescoring_logic_to_helper() -> None:
    """Notebook 必须复用 repository helper 执行真实 aligned rescoring。"""
    payload = json.loads(ALIGNED_RESCORING_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    joined_source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    first_code_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
    first_code_source = "".join(first_code_cell.get("source", []))

    assert "paper_workflow.colab_utils.aligned_rescoring" in joined_source
    assert "run_default_aligned_rescoring_plan" in joined_source
    assert "package_aligned_rescoring_outputs" in joined_source
    assert "pilot_paper_fixed_fpr_0_01" in joined_source
    assert "configs/paper_main_pilot_paper_prompts.txt" in joined_source
    assert "drive.mount('/content/drive')" in first_code_source
    assert "/content/drive/MyDrive/SLM/pilot_paper_results/aligned_rescoring" in joined_source
    assert "/content/drive/MyDrive/SLM/pilot_paper_results/attention_geometry" in joined_source
    assert "real_aligned_rescore_count" in joined_source
    assert "SLM_WM_ALIGNED_RESCORING_SUBSPACE_RECORDS', '128'" in joined_source
    assert "SLM_WM_ALIGNED_RESCORING_CARRIER_COUNT', '120'" in joined_source
    assert "SLM_WM_ENABLE_PAIR_PERCEPTUAL_METRICS', '1'" in joined_source
    assert "SLM_WM_REQUIRE_PAIR_PERCEPTUAL_METRICS', '1'" in joined_source
    assert "openai/clip-vit-base-patch32" in joined_source
    assert "SLM_WM_LPIPS_NETWORK', 'alex'" in joined_source
    assert "SLM_WM_PERCEPTUAL_METRIC_DEVICE', 'cpu'" in joined_source
    assert "SLM_WM_ENABLE_PIPELINE_PROGRESS_BAR', '0'" in joined_source
    assert "SLM_WM_ENABLE_CARRIER_PROGRESS_BAR', '1'" in joined_source
    assert "perceptual_metrics_ready" in joined_source
    assert "datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%sz')" in joined_source
    assert "['git', 'rev-parse', '--short', 'HEAD']" in joined_source
    assert "archive_name=archive_name" in joined_source
    assert COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND in joined_source
    assert PAIR_PERCEPTUAL_DEPENDENCY_INSTALL_COMMAND in joined_source
    assert "--force-reinstall" not in joined_source
    assert "numpy pillow" not in joined_source
    assert "del sys.modules" not in joined_source
    assert '"diffusers==' not in joined_source
    assert '"transformers==' not in joined_source


@pytest.mark.constraint
def test_colab_notebook_delegates_threshold_calibration_logic_to_helper() -> None:
    """Notebook 必须复用 repository helper 生成 threshold calibration 结果包。"""
    payload = json.loads(THRESHOLD_CALIBRATION_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    joined_source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    first_code_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
    first_code_source = "".join(first_code_cell.get("source", []))

    assert "paper_workflow.colab_utils.threshold_calibration" in joined_source
    assert "run_default_threshold_calibration_from_drive_plan" in joined_source
    assert "package_threshold_calibration_outputs" in joined_source
    assert "pilot_paper_fixed_fpr_0_01" in joined_source
    assert "configs/paper_main_pilot_paper_prompts.txt" in joined_source
    assert "drive.mount('/content/drive')" in first_code_source
    assert "/content/drive/MyDrive/SLM/pilot_paper_results/threshold_calibration" in joined_source
    assert "/content/drive/MyDrive/SLM/pilot_paper_results/attention_latent_injection" in joined_source
    assert "/content/drive/MyDrive/SLM/pilot_paper_results/aligned_rescoring" in joined_source
    assert "attention_latent_injection_package_*.zip" in joined_source
    assert "aligned_rescoring_package_*.zip" in joined_source
    assert "threshold_calibration_ready" in joined_source
    assert "SLM_WM_THRESHOLD_TARGET_FPR', '0.01'" in joined_source
    assert "geometric_rescue_ready" in joined_source
    assert "datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%sz')" in joined_source
    assert "['git', 'rev-parse', '--short', 'HEAD']" in joined_source
    assert "archive_name=archive_name" in joined_source
    assert COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND in joined_source
    assert "--force-reinstall" not in joined_source
    assert "numpy pillow" not in joined_source
    assert "del sys.modules" not in joined_source
    assert '"diffusers==' not in joined_source
    assert '"transformers==' not in joined_source


@pytest.mark.constraint
def test_colab_notebook_delegates_real_attack_evaluation_logic_to_helper() -> None:
    """Notebook 必须复用 repository helper 执行真实再扩散攻击闭环。"""
    payload = json.loads(REAL_ATTACK_EVALUATION_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    joined_source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    first_code_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
    first_code_source = "".join(first_code_cell.get("source", []))

    assert "paper_workflow.colab_utils.real_attack_evaluation" in joined_source
    assert "run_default_real_attack_evaluation_from_drive_plan" in joined_source
    assert "package_real_attack_evaluation_outputs" in joined_source
    assert "pilot_paper_fixed_fpr_0_01" in joined_source
    assert "configs/paper_main_pilot_paper_prompts.txt" in joined_source
    assert "drive.mount('/content/drive')" in first_code_source
    assert "/content/drive/MyDrive/SLM/pilot_paper_results/real_attack_evaluation" in joined_source
    assert "/content/drive/MyDrive/SLM/pilot_paper_results/aligned_rescoring" in joined_source
    assert "/content/drive/MyDrive/SLM/pilot_paper_results/threshold_calibration" in joined_source
    assert "aligned_rescoring_package_*.zip" in joined_source
    assert "real_attacked_image_closed_loop_ready" in joined_source
    assert "regeneration_attack_gpu_validation_ready" in joined_source
    assert "attack_detection_rerun_ready" in joined_source
    assert "formal_attack_detection_ready" in joined_source
    assert "SLM_WM_REAL_ATTACK_SOURCE_COUNT', '120'" in joined_source
    assert "runwayml/stable-diffusion-v1-5" in joined_source
    assert "datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%sz')" in joined_source
    assert "['git', 'rev-parse', '--short', 'HEAD']" in joined_source
    assert "archive_name=archive_name" in joined_source
    assert COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND in joined_source
    assert "--force-reinstall" not in joined_source
    assert "numpy pillow" not in joined_source
    assert "del sys.modules" not in joined_source
    assert '"diffusers==' not in joined_source
    assert '"transformers==' not in joined_source


@pytest.mark.constraint
def test_colab_notebook_delegates_external_baseline_gpu_smoke_logic_to_helper() -> None:
    """Notebook 必须复用 repository helper 执行外部 baseline 真实 GPU smoke。"""
    payload = json.loads(EXTERNAL_BASELINE_GPU_SMOKE_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    joined_source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    first_code_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
    first_code_source = "".join(first_code_cell.get("source", []))

    assert "paper_workflow.colab_utils.external_baseline_gpu_smoke" in joined_source
    assert "run_default_external_baseline_gpu_smoke_plan" in joined_source
    assert "package_external_baseline_gpu_smoke_outputs" in joined_source
    assert "pilot_paper_fixed_fpr_0_01" in joined_source
    assert "configs/paper_main_pilot_paper_prompts.txt" in joined_source
    assert "drive.mount('/content/drive')" in first_code_source
    assert "/content/drive/MyDrive/SLM/pilot_paper_results/external_baseline_gpu_smoke" in joined_source
    assert "external_baseline/source_registry.json" in joined_source
    assert "stabilityai/stable-diffusion-3.5-medium" in joined_source
    assert "external_baseline_gpu_smoke_ready" in joined_source
    assert "t2smark_real_gpu_smoke_ready" in joined_source
    assert "adapter_observation_count" in joined_source
    assert "primary_baseline_adapter_ready" in joined_source
    assert "primary_baseline_observation_count" in joined_source
    assert "SLM_WM_T2SMARK_ROBUST_TEST_NUM', '120'" in joined_source
    assert "SLM_WM_PRIMARY_BASELINE_MAX_SAMPLES', '120'" in joined_source
    assert "5 个数字样本条目" not in joined_source
    assert "新的 5 样本真实 GPU 结果" not in joined_source
    assert "默认共享样本数为 5" not in joined_source
    assert "expected_sample_count = int(os.environ['SLM_WM_PRIMARY_BASELINE_MAX_SAMPLES'])" in joined_source
    assert "tree_ring" in joined_source
    assert "gaussian_shading" in joined_source
    assert "shallow_diffuse" in joined_source
    assert "datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%sz')" in joined_source
    assert "['git', 'rev-parse', '--short', 'HEAD']" in joined_source
    assert "archive_name=archive_name" in joined_source
    assert EXTERNAL_BASELINE_DEPENDENCY_INSTALL_COMMAND in joined_source
    assert "--force-reinstall" not in joined_source
    assert "numpy pillow" not in joined_source
    assert "del sys.modules" not in joined_source
    assert '"diffusers==' not in joined_source
    assert '"transformers==' not in joined_source


@pytest.mark.constraint
def test_colab_notebook_delegates_dataset_level_quality_logic_to_helper() -> None:
    """Notebook 必须复用 repository helper 执行数据集级质量特征导入."""
    payload = json.loads(DATASET_LEVEL_QUALITY_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    joined_source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    first_code_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
    first_code_source = "".join(first_code_cell.get("source", []))

    assert "paper_workflow.colab_utils.dataset_level_quality" in joined_source
    assert "run_default_dataset_level_quality_from_drive_plan" in joined_source
    assert "package_dataset_level_quality_outputs" in joined_source
    assert "pilot_paper_fixed_fpr_0_01" in joined_source
    assert "configs/paper_main_pilot_paper_prompts.txt" in joined_source
    assert "drive.mount('/content/drive')" in first_code_source
    assert "/content/drive/MyDrive/SLM/pilot_paper_results/dataset_level_quality" in joined_source
    assert "/content/drive/MyDrive/SLM/pilot_paper_results/real_attack_evaluation" in joined_source
    assert "/content/drive/MyDrive/SLM/pilot_paper_results/aligned_rescoring" in joined_source
    assert "real_attack_evaluation_package_*.zip" in joined_source
    assert "aligned_rescoring_package_*.zip" in joined_source
    assert "formal_feature_backend_ready" in joined_source
    assert "formal_fid_kid_ready" in joined_source
    assert "SLM_WM_FORMAL_MIN_SAMPLE_COUNT', '100'" in joined_source
    assert "datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%sz')" in joined_source
    assert "['git', 'rev-parse', '--short', 'HEAD']" in joined_source
    assert "archive_name=archive_name" in joined_source
    assert "--force-reinstall" not in joined_source
    assert "numpy pillow" not in joined_source
    assert "del sys.modules" not in joined_source
    assert '"torch==' not in joined_source
    assert '"torchvision==' not in joined_source


@pytest.mark.constraint
def test_official_baseline_notebooks_default_to_pilot_paper_outputs() -> None:
    """四个官方 baseline 复现入口应默认写入 pilot_paper 结果目录并使用 pilot_paper 规模。"""

    expectations = {
        T2SMARK_OFFICIAL_REPRODUCTION_NOTEBOOK_PATH: (
            "/content/drive/MyDrive/SLM/pilot_paper_results/t2smark_full_main_reproduction",
            "SLM_WM_T2SMARK_FULL_MAIN_PROMPT_LIMIT', '120'",
            "SLM_WM_T2SMARK_FULL_MAIN_TARGET_FPR', '0.01'",
        ),
        TREE_RING_OFFICIAL_REFERENCE_NOTEBOOK_PATH: (
            "/content/drive/MyDrive/SLM/pilot_paper_results/tree_ring_official_reference",
            "SLM_WM_TREE_RING_OFFICIAL_SAMPLE_COUNT', '120'",
            "configs/paper_main_pilot_paper_prompts.txt",
        ),
        GAUSSIAN_SHADING_OFFICIAL_REFERENCE_NOTEBOOK_PATH: (
            "/content/drive/MyDrive/SLM/pilot_paper_results/gaussian_shading_official_reference",
            "SLM_WM_GAUSSIAN_SHADING_OFFICIAL_SAMPLE_COUNT', '120'",
            "configs/paper_main_pilot_paper_prompts.txt",
        ),
        SHALLOW_DIFFUSE_OFFICIAL_REFERENCE_NOTEBOOK_PATH: (
            "/content/drive/MyDrive/SLM/pilot_paper_results/shallow_diffuse_official_reference",
            "SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_SAMPLE_COUNT', '120'",
            "configs/paper_main_pilot_paper_prompts.txt",
        ),
    }
    for notebook_path, required_texts in expectations.items():
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
        joined_source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
        first_code_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
        first_code_source = "".join(first_code_cell.get("source", []))

        assert "drive.mount('/content/drive')" in first_code_source
        assert "pilot_paper_fixed_fpr_0_01" in joined_source
        assert "SLM_WM_PROMPT_SET', 'pilot_paper'" in joined_source
        assert "默认样本数为 5" not in joined_source
        assert "sample_count'] == 5" not in joined_source
        assert "configs/paper_main_full_paper_prompts.txt" not in joined_source
        for required_text in required_texts:
            assert required_text in joined_source

    official_sample_assertions = {
        TREE_RING_OFFICIAL_REFERENCE_NOTEBOOK_PATH: "expected_sample_count = int(os.environ['SLM_WM_TREE_RING_OFFICIAL_SAMPLE_COUNT'])",
        GAUSSIAN_SHADING_OFFICIAL_REFERENCE_NOTEBOOK_PATH: (
            "expected_sample_count = int(os.environ['SLM_WM_GAUSSIAN_SHADING_OFFICIAL_SAMPLE_COUNT'])"
        ),
        SHALLOW_DIFFUSE_OFFICIAL_REFERENCE_NOTEBOOK_PATH: (
            "expected_sample_count = int(os.environ['SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_SAMPLE_COUNT'])"
        ),
    }
    for notebook_path, required_assertion in official_sample_assertions.items():
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
        joined_source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
        assert required_assertion in joined_source
        assert "['sample_count'] == expected_sample_count" in joined_source


@pytest.mark.constraint
def test_pilot_paper_result_closure_notebook_delegates_to_repository_commands() -> None:
    """pilot_paper 结果闭合入口必须只调度 repository commands, 不直接拼写正式产物。"""
    payload = json.loads(PILOT_PAPER_RESULT_CLOSURE_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    joined_source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    first_code_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
    first_code_source = "".join(first_code_cell.get("source", []))

    assert "drive.mount('/content/drive')" in first_code_source
    assert "pilot_paper_fixed_fpr_0_01" in joined_source
    assert "configs/paper_main_pilot_paper_prompts.txt" in joined_source
    assert "/content/drive/MyDrive/SLM/pilot_paper_results" in joined_source
    assert "/content/drive/MyDrive/SLM/pilot_paper_results/complete_result_package" in joined_source
    assert "scripts/write_pilot_paper_result_records.py" in joined_source
    assert "--materialize-only" in joined_source
    assert "scripts/write_attack_matrix_outputs.py" in joined_source
    assert "scripts/write_primary_baseline_result_candidates.py" in joined_source
    assert "--target-fpr-override" in joined_source
    assert "scripts/write_primary_baseline_formal_import_protocol.py" in joined_source
    assert "scripts/write_external_baseline_comparison_outputs.py" in joined_source
    assert "scripts/write_internal_ablation_outputs.py" in joined_source
    assert "scripts/write_pilot_paper_fixed_fpr_common_protocol_outputs.py" in joined_source
    assert "scripts/write_pilot_paper_complete_result_package.py" in joined_source
    assert "json.dumps" not in joined_source
    assert "write_text(" not in joined_source


@pytest.mark.constraint
def test_probe_outputs_can_be_packaged_and_mirrored(tmp_path: Path) -> None:
    """真实 runtime 产物应能打包, 并可镜像到外部同步目录."""
    output_dir = tmp_path / "outputs" / "real_sd_runtime_probe"
    output_dir.mkdir(parents=True)
    (output_dir / "sample_runtime_summary.json").write_text('{"probe_decision":"pass"}\n', encoding="utf-8")
    (output_dir / "sample_latent_trajectory_records.jsonl").write_text('{"trajectory_index":0}\n', encoding="utf-8")

    drive_dir = tmp_path / "drive_mirror"
    record = package_probe_outputs(root=tmp_path, drive_output_dir=str(drive_dir))
    archive_path = tmp_path / record.archive_path

    assert archive_path.exists()
    assert (drive_dir / "real_sd_runtime_probe_package.zip").exists()
    assert record.archive_digest == record.drive_archive_digest
    assert record.archive_entry_count == 2
    assert (output_dir / "real_sd_runtime_probe_archive_summary.json").exists()
    assert (output_dir / "real_sd_runtime_probe_archive_manifest.local.json").exists()

    with ZipFile(archive_path) as archive:
        assert sorted(archive.namelist()) == [
            "sample_latent_trajectory_records.jsonl",
            "sample_runtime_summary.json",
        ]


@pytest.mark.constraint
def test_injection_outputs_can_be_packaged_and_mirrored(tmp_path: Path) -> None:
    """最小 latent injection 产物应能打包, 并可镜像到外部同步目录."""
    output_dir = tmp_path / "outputs" / "minimal_diffusion_latent_injection"
    output_dir.mkdir(parents=True)
    (output_dir / "sample_injection_result.json").write_text('{"run_decision":"pass"}\n', encoding="utf-8")
    (output_dir / "sample_latent_update_records.jsonl").write_text('{"trajectory_index":0}\n', encoding="utf-8")
    (output_dir / "sample_paired_quality_metrics.csv").write_text("injection_id,psnr\nsample,inf\n", encoding="utf-8")

    drive_dir = tmp_path / "drive_mirror"
    record = package_injection_outputs(root=tmp_path, drive_output_dir=str(drive_dir))
    archive_path = tmp_path / record.archive_path

    assert archive_path.exists()
    assert (drive_dir / "minimal_latent_injection_package.zip").exists()
    assert record.archive_digest == record.drive_archive_digest
    assert record.archive_entry_count == 3
    assert (output_dir / "minimal_latent_injection_archive_summary.json").exists()
    assert (output_dir / "minimal_latent_injection_archive_manifest.local.json").exists()

    with ZipFile(archive_path) as archive:
        assert sorted(archive.namelist()) == [
            "sample_injection_result.json",
            "sample_latent_update_records.jsonl",
            "sample_paired_quality_metrics.csv",
        ]


@pytest.mark.constraint
def test_attention_geometry_outputs_can_be_packaged_and_mirrored(tmp_path: Path) -> None:
    """真实 attention 几何产物应能打包, 且包含关键核对文件。"""
    capture_dir = tmp_path / "outputs" / "real_attention_geometry"
    geometry_dir = tmp_path / "outputs" / "attention_geometry"
    capture_dir.mkdir(parents=True)
    geometry_dir.mkdir(parents=True)
    (capture_dir / "real_attention_capture_records.jsonl").write_text('{"capture_id":"sample"}\n', encoding="utf-8")
    (capture_dir / "real_attention_capture_summary.json").write_text('{"attention_geometry_ready":true}\n', encoding="utf-8")
    (capture_dir / "real_attention_environment_report.json").write_text('{"cuda_available":true}\n', encoding="utf-8")
    (capture_dir / "real_attention_manifest.local.json").write_text('{"artifact_id":"real_attention_geometry_manifest"}\n', encoding="utf-8")
    (geometry_dir / "attention_graph_records.jsonl").write_text('{"capture_id":"sample"}\n', encoding="utf-8")
    (geometry_dir / "geometry_evidence_records.jsonl").write_text('{"capture_id":"sample"}\n', encoding="utf-8")
    (geometry_dir / "attention_relation_consistency.csv").write_text("capture_id,attention_relation_consistency\nsample,1.0\n", encoding="utf-8")
    (geometry_dir / "geometry_evidence_summary.json").write_text('{"attention_geometry_ready":true}\n', encoding="utf-8")
    (geometry_dir / "manifest.local.json").write_text('{"artifact_id":"attention_geometry_manifest"}\n', encoding="utf-8")

    drive_dir = tmp_path / "drive_mirror"
    record = package_attention_geometry_outputs(root=tmp_path, drive_output_dir=str(drive_dir))
    archive_path = tmp_path / record.archive_path

    assert archive_path.exists()
    assert (drive_dir / "attention_geometry_package.zip").exists()
    assert record.archive_digest == record.drive_archive_digest
    assert record.archive_entry_count >= 5
    assert (capture_dir / "attention_geometry_archive_summary.json").exists()
    assert (capture_dir / "attention_geometry_archive_manifest.local.json").exists()

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "outputs/real_attention_geometry/real_attention_capture_records.jsonl" in names
        assert "outputs/real_attention_geometry/real_attention_capture_summary.json" in names
        assert "outputs/real_attention_geometry/real_attention_environment_report.json" in names
        assert "outputs/real_attention_geometry/real_attention_manifest.local.json" in names
        assert "outputs/attention_geometry/attention_graph_records.jsonl" in names
        assert "outputs/attention_geometry/geometry_evidence_records.jsonl" in names
        assert "outputs/attention_geometry/attention_relation_consistency.csv" in names
        assert "outputs/attention_geometry/geometry_evidence_summary.json" in names
        assert "outputs/attention_geometry/manifest.local.json" in names
        assert "outputs/real_attention_geometry/attention_geometry_package_input_manifest.json" in names


@pytest.mark.constraint
def test_attention_latent_injection_outputs_can_be_packaged_and_mirrored(tmp_path: Path) -> None:
    """真实 attention latent injection 产物应能打包, 且包含方法与运行核对文件。"""
    injection_dir = tmp_path / "outputs" / "attention_latent_injection"
    method_dir = tmp_path / "outputs" / "attention_latent_update"
    injection_dir.mkdir(parents=True)
    method_dir.mkdir(parents=True)
    (injection_dir / "attention_latent_injection_result.json").write_text('{"run_decision":"pass"}\n', encoding="utf-8")
    (injection_dir / "attention_latent_update_records.jsonl").write_text('{"trajectory_index":0}\n', encoding="utf-8")
    (injection_dir / "attention_paired_quality_metrics.csv").write_text("injection_id,psnr\nsample,inf\n", encoding="utf-8")
    (injection_dir / "attention_injection_environment_report.json").write_text('{"cuda_available":true}\n', encoding="utf-8")
    (injection_dir / "attention_latent_injection_manifest.local.json").write_text('{"artifact_id":"attention_latent_injection_manifest"}\n', encoding="utf-8")
    (method_dir / "attention_carrier_records.jsonl").write_text('{"carrier_id":"sample"}\n', encoding="utf-8")
    (method_dir / "attention_update_summary.json").write_text('{"active_update_count":1}\n', encoding="utf-8")
    (method_dir / "manifest.local.json").write_text('{"artifact_id":"attention_latent_update_manifest"}\n', encoding="utf-8")

    drive_dir = tmp_path / "drive_mirror"
    record = package_attention_latent_injection_outputs(root=tmp_path, drive_output_dir=str(drive_dir))
    archive_path = tmp_path / record.archive_path

    assert archive_path.exists()
    assert (drive_dir / "attention_latent_injection_package.zip").exists()
    assert record.archive_digest == record.drive_archive_digest
    assert record.archive_entry_count >= 9
    assert (injection_dir / "attention_latent_injection_archive_summary.json").exists()
    assert (injection_dir / "attention_latent_injection_archive_manifest.local.json").exists()

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "outputs/attention_latent_injection/attention_latent_injection_result.json" in names
        assert "outputs/attention_latent_injection/attention_latent_update_records.jsonl" in names
        assert "outputs/attention_latent_injection/attention_paired_quality_metrics.csv" in names
        assert "outputs/attention_latent_injection/attention_injection_environment_report.json" in names
        assert "outputs/attention_latent_injection/attention_latent_injection_manifest.local.json" in names
        assert "outputs/attention_latent_update/attention_carrier_records.jsonl" in names
        assert "outputs/attention_latent_update/attention_update_summary.json" in names
        assert "outputs/attention_latent_update/manifest.local.json" in names
        assert "outputs/attention_latent_injection/attention_latent_injection_package_input_manifest.json" in names


@pytest.mark.constraint
def test_aligned_rescoring_outputs_can_be_packaged_and_mirrored(tmp_path: Path) -> None:
    """真实 aligned rescoring 产物应能打包, 且包含重打分与方法核对文件。"""
    rescoring_dir = tmp_path / "outputs" / "aligned_rescoring"
    method_dir = tmp_path / "outputs" / "attention_latent_update"
    rescoring_dir.mkdir(parents=True)
    method_dir.mkdir(parents=True)
    (rescoring_dir / "aligned_rescoring_result.json").write_text('{"run_decision":"pass"}\n', encoding="utf-8")
    (rescoring_dir / "aligned_rescoring_records.jsonl").write_text('{"aligned_rescoring_ready":true}\n', encoding="utf-8")
    (rescoring_dir / "aligned_rescoring_quality_metrics.csv").write_text("carrier_id,psnr\nsample,35.0\n", encoding="utf-8")
    (rescoring_dir / "aligned_rescoring_environment_report.json").write_text('{"cuda_available":true}\n', encoding="utf-8")
    (rescoring_dir / "aligned_rescoring_manifest.local.json").write_text('{"artifact_id":"aligned_rescoring_manifest"}\n', encoding="utf-8")
    (method_dir / "attention_carrier_records.jsonl").write_text('{"carrier_id":"sample"}\n', encoding="utf-8")
    (method_dir / "attention_update_summary.json").write_text('{"active_update_count":1}\n', encoding="utf-8")
    (method_dir / "manifest.local.json").write_text('{"artifact_id":"attention_latent_update_manifest"}\n', encoding="utf-8")

    drive_dir = tmp_path / "drive_mirror"
    record = package_aligned_rescoring_outputs(root=tmp_path, drive_output_dir=str(drive_dir))
    archive_path = tmp_path / record.archive_path

    assert archive_path.exists()
    assert (drive_dir / "aligned_rescoring_package.zip").exists()
    assert record.archive_digest == record.drive_archive_digest
    assert record.archive_entry_count >= 9
    assert (rescoring_dir / "aligned_rescoring_archive_summary.json").exists()
    assert (rescoring_dir / "aligned_rescoring_archive_manifest.local.json").exists()

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "outputs/aligned_rescoring/aligned_rescoring_result.json" in names
        assert "outputs/aligned_rescoring/aligned_rescoring_records.jsonl" in names
        assert "outputs/aligned_rescoring/aligned_rescoring_quality_metrics.csv" in names
        assert "outputs/aligned_rescoring/aligned_rescoring_environment_report.json" in names
        assert "outputs/aligned_rescoring/aligned_rescoring_manifest.local.json" in names
        assert "outputs/attention_latent_update/attention_carrier_records.jsonl" in names
        assert "outputs/attention_latent_update/attention_update_summary.json" in names
        assert "outputs/attention_latent_update/manifest.local.json" in names
        assert "outputs/aligned_rescoring/aligned_rescoring_package_input_manifest.json" in names


@pytest.mark.constraint
def test_threshold_calibration_outputs_can_be_packaged_and_mirrored(tmp_path: Path) -> None:
    """Threshold calibration 产物应能打包, 且包含几何恢复核对文件。"""
    threshold_dir = tmp_path / "outputs" / "threshold_calibration"
    rescue_dir = tmp_path / "outputs" / "geometric_rescue"
    content_dir = tmp_path / "outputs" / "content_carriers"
    threshold_dir.mkdir(parents=True)
    rescue_dir.mkdir(parents=True)
    content_dir.mkdir(parents=True)
    (threshold_dir / "threshold_calibration_result.json").write_text('{"run_decision":"pass"}\n', encoding="utf-8")
    (threshold_dir / "calibration_thresholds.json").write_text('{"threshold_value":0.75}\n', encoding="utf-8")
    (threshold_dir / "threshold_degeneracy_report.json").write_text('{"supports_paper_claim":false}\n', encoding="utf-8")
    (threshold_dir / "fixed_fpr_operating_points.csv").write_text("target_fpr,raw_content_clean_fpr\n0.05,0.01\n", encoding="utf-8")
    (threshold_dir / "manifest.local.json").write_text('{"artifact_id":"threshold_calibration_manifest"}\n', encoding="utf-8")
    (rescue_dir / "aligned_detection_records.jsonl").write_text('{"rescue_ablation_mode":"full_rescue"}\n', encoding="utf-8")
    (rescue_dir / "geometry_rescue_audit.json").write_text('{"protocol_decision":"pass"}\n', encoding="utf-8")
    (rescue_dir / "manifest.local.json").write_text('{"artifact_id":"geometric_rescue_manifest"}\n', encoding="utf-8")
    (content_dir / "content_detection_records.jsonl").write_text('{"content_detection_record_id":"sample"}\n', encoding="utf-8")

    drive_dir = tmp_path / "drive_mirror"
    record = package_threshold_calibration_outputs(root=tmp_path, drive_output_dir=str(drive_dir))
    archive_path = tmp_path / record.archive_path

    assert archive_path.exists()
    assert (drive_dir / "threshold_calibration_package.zip").exists()
    assert record.archive_digest == record.drive_archive_digest
    assert record.archive_entry_count >= 10
    assert (threshold_dir / "threshold_calibration_archive_summary.json").exists()
    assert (threshold_dir / "threshold_calibration_archive_manifest.local.json").exists()

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "outputs/threshold_calibration/threshold_calibration_result.json" in names
        assert "outputs/threshold_calibration/calibration_thresholds.json" in names
        assert "outputs/threshold_calibration/threshold_degeneracy_report.json" in names
        assert "outputs/threshold_calibration/fixed_fpr_operating_points.csv" in names
        assert "outputs/threshold_calibration/manifest.local.json" in names
        assert "outputs/geometric_rescue/aligned_detection_records.jsonl" in names
        assert "outputs/geometric_rescue/geometry_rescue_audit.json" in names
        assert "outputs/geometric_rescue/manifest.local.json" in names
        assert "outputs/content_carriers/content_detection_records.jsonl" in names
        assert "outputs/threshold_calibration/threshold_calibration_package_input_manifest.json" in names
        assert "outputs/threshold_calibration/threshold_calibration_archive_summary.json" in names
        assert "outputs/threshold_calibration/threshold_calibration_archive_manifest.local.json" in names


@pytest.mark.constraint
def test_real_attack_evaluation_outputs_can_be_packaged_and_mirrored(tmp_path: Path) -> None:
    """真实攻击闭环产物应能打包, 且包含 attacked image 与 digest 注册表。"""
    attack_dir = tmp_path / "outputs" / "real_attack_evaluation"
    image_dir = attack_dir / "attacked_images"
    image_dir.mkdir(parents=True)
    (attack_dir / "real_attack_run_summary.json").write_text('{"run_decision":"pass"}\n', encoding="utf-8")
    (attack_dir / "real_attack_detection_records.jsonl").write_text('{"attack_performed":true}\n', encoding="utf-8")
    (attack_dir / "formal_attack_detection_records.jsonl").write_text('{"attack_performed":true}\n', encoding="utf-8")
    (attack_dir / "real_attacked_image_registry.jsonl").write_text('{"attacked_image_digest":"abc"}\n', encoding="utf-8")
    (attack_dir / "real_attack_family_metrics.csv").write_text("attack_name,measured_record_count\nimg2img_regeneration,1\n", encoding="utf-8")
    (attack_dir / "real_attack_environment_report.json").write_text('{"cuda_available":true}\n', encoding="utf-8")
    (attack_dir / "real_attack_manifest.local.json").write_text('{"artifact_id":"real_attack_evaluation_manifest"}\n', encoding="utf-8")
    (image_dir / "sample_attacked.png").write_bytes(b"fake_png_bytes")

    drive_dir = tmp_path / "drive_mirror"
    record = package_real_attack_evaluation_outputs(root=tmp_path, drive_output_dir=str(drive_dir))
    archive_path = tmp_path / record.archive_path

    assert archive_path.exists()
    assert (drive_dir / "real_attack_evaluation_package.zip").exists()
    assert record.archive_digest == record.drive_archive_digest
    assert record.archive_entry_count >= 8
    assert (attack_dir / "real_attack_archive_summary.json").exists()
    assert (attack_dir / "real_attack_archive_manifest.local.json").exists()

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "outputs/real_attack_evaluation/real_attack_run_summary.json" in names
        assert "outputs/real_attack_evaluation/real_attack_detection_records.jsonl" in names
        assert "outputs/real_attack_evaluation/formal_attack_detection_records.jsonl" in names
        assert "outputs/real_attack_evaluation/real_attacked_image_registry.jsonl" in names
        assert "outputs/real_attack_evaluation/real_attack_family_metrics.csv" in names
        assert "outputs/real_attack_evaluation/real_attack_environment_report.json" in names
        assert "outputs/real_attack_evaluation/real_attack_manifest.local.json" in names
        assert "outputs/real_attack_evaluation/attacked_images/sample_attacked.png" in names
        assert "outputs/real_attack_evaluation/real_attack_package_input_manifest.json" in names
        assert "outputs/real_attack_evaluation/real_attack_archive_summary.json" in names
        assert "outputs/real_attack_evaluation/real_attack_archive_manifest.local.json" in names


@pytest.mark.constraint
def test_external_baseline_gpu_smoke_outputs_can_be_packaged_and_mirrored(tmp_path: Path) -> None:
    """外部 baseline 真实 GPU smoke 产物应能打包, 且包含官方结果与 adapter 观测。"""
    smoke_dir = tmp_path / "outputs" / "external_baseline_gpu_smoke"
    official_dir = smoke_dir / "t2smark_official" / "t2smark_sd35_medium_gpu_smoke"
    execution_dir = smoke_dir / "execution"
    image_dir = official_dir / "images"
    image_dir.mkdir(parents=True)
    execution_dir.mkdir(parents=True)
    (official_dir / "results.json").write_text('{"0":{"robustness":{"acc_msg":1.0}},"bit_accuracy":1.0}\n', encoding="utf-8")
    (official_dir / "settings.json").write_text('{"model_key":"stabilityai/stable-diffusion-3.5-medium"}\n', encoding="utf-8")
    (image_dir / "00000.png").write_bytes(b"fake_png_bytes")
    (smoke_dir / "t2smark_smoke_prompts.json").write_text(
        '{"annotations":[{"caption":"a small ceramic fox"}]}\n',
        encoding="utf-8",
    )
    (smoke_dir / "t2smark_image_pairs.json").write_text('[{"image_id":"t2smark_00000"}]\n', encoding="utf-8")
    (smoke_dir / "baseline_command_plan.json").write_text('{"command_count":1}\n', encoding="utf-8")
    (execution_dir / "baseline_execution_manifest.json").write_text('{"observation_count":1}\n', encoding="utf-8")
    (execution_dir / "baseline_observations.json").write_text('[{"baseline_id":"t2smark"}]\n', encoding="utf-8")
    (smoke_dir / "external_baseline_gpu_smoke_summary.json").write_text(
        '{"run_decision":"pass","external_baseline_gpu_smoke_ready":true}\n',
        encoding="utf-8",
    )
    (smoke_dir / "external_baseline_gpu_smoke_environment_report.json").write_text('{"cuda_available":true}\n', encoding="utf-8")
    (smoke_dir / "external_baseline_gpu_smoke_manifest.local.json").write_text(
        '{"artifact_id":"external_baseline_gpu_smoke_manifest"}\n',
        encoding="utf-8",
    )

    drive_dir = tmp_path / "drive_mirror"
    record = package_external_baseline_gpu_smoke_outputs(root=tmp_path, drive_output_dir=str(drive_dir))
    archive_path = tmp_path / record.archive_path

    assert archive_path.exists()
    assert (drive_dir / "external_baseline_gpu_smoke_package.zip").exists()
    assert record.archive_digest == record.drive_archive_digest
    assert record.archive_entry_count >= 11
    assert (smoke_dir / "external_baseline_gpu_smoke_archive_summary.json").exists()
    assert (smoke_dir / "external_baseline_gpu_smoke_archive_manifest.local.json").exists()

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "outputs/external_baseline_gpu_smoke/t2smark_official/t2smark_sd35_medium_gpu_smoke/results.json" in names
        assert "outputs/external_baseline_gpu_smoke/t2smark_official/t2smark_sd35_medium_gpu_smoke/settings.json" in names
        assert "outputs/external_baseline_gpu_smoke/t2smark_official/t2smark_sd35_medium_gpu_smoke/images/00000.png" in names
        assert "outputs/external_baseline_gpu_smoke/t2smark_smoke_prompts.json" in names
        assert "outputs/external_baseline_gpu_smoke/t2smark_image_pairs.json" in names
        assert "outputs/external_baseline_gpu_smoke/baseline_command_plan.json" in names
        assert "outputs/external_baseline_gpu_smoke/execution/baseline_execution_manifest.json" in names
        assert "outputs/external_baseline_gpu_smoke/execution/baseline_observations.json" in names
        assert "outputs/external_baseline_gpu_smoke/external_baseline_gpu_smoke_summary.json" in names
        assert "outputs/external_baseline_gpu_smoke/external_baseline_gpu_smoke_environment_report.json" in names
        assert "outputs/external_baseline_gpu_smoke/external_baseline_gpu_smoke_manifest.local.json" in names
        assert "outputs/external_baseline_gpu_smoke/external_baseline_gpu_smoke_package_input_manifest.json" in names
        assert "outputs/external_baseline_gpu_smoke/external_baseline_gpu_smoke_archive_summary.json" in names
        assert "outputs/external_baseline_gpu_smoke/external_baseline_gpu_smoke_archive_manifest.local.json" in names


@pytest.mark.constraint
def test_dataset_level_quality_outputs_can_be_packaged_and_mirrored(tmp_path: Path) -> None:
    """数据集级质量产物应能打包, 且包含正式特征导入核对文件."""
    quality_dir = tmp_path / "outputs" / "dataset_level_quality"
    quality_dir.mkdir(parents=True)
    (quality_dir / "dataset_level_quality_result.json").write_text('{"run_decision":"pass"}\n', encoding="utf-8")
    (quality_dir / "dataset_quality_image_records.jsonl").write_text('{"dataset_quality_record_id":"sample"}\n', encoding="utf-8")
    (quality_dir / "dataset_quality_image_resolution_records.jsonl").write_text(
        '{"image_resolution_record_id":"sample"}\n',
        encoding="utf-8",
    )
    (quality_dir / "dataset_quality_formal_feature_records.jsonl").write_text(
        '{"dataset_quality_record_id":"sample","dataset_quality_image_role":"source"}\n',
        encoding="utf-8",
    )
    (quality_dir / "dataset_quality_formal_feature_import_report.json").write_text(
        '{"formal_feature_backend_ready":true}\n',
        encoding="utf-8",
    )
    (quality_dir / "dataset_quality_metrics.csv").write_text("quality_metric_name,metric_status\nfid,unsupported\n", encoding="utf-8")
    (quality_dir / "dataset_quality_summary.json").write_text('{"formal_fid_kid_ready":false}\n', encoding="utf-8")
    (quality_dir / "dataset_level_quality_environment_report.json").write_text('{"cuda_available":true}\n', encoding="utf-8")
    (quality_dir / "manifest.local.json").write_text('{"artifact_id":"dataset_level_quality_manifest"}\n', encoding="utf-8")
    (quality_dir / "dataset_level_quality_colab_manifest.local.json").write_text(
        '{"artifact_id":"dataset_level_quality_colab_manifest"}\n',
        encoding="utf-8",
    )

    drive_dir = tmp_path / "drive_mirror"
    record = package_dataset_level_quality_outputs(root=tmp_path, drive_output_dir=str(drive_dir))
    archive_path = tmp_path / record.archive_path

    assert archive_path.exists()
    assert (drive_dir / "dataset_level_quality_package.zip").exists()
    assert record.archive_digest == record.drive_archive_digest
    assert record.archive_entry_count >= 10
    assert (quality_dir / "dataset_level_quality_archive_summary.json").exists()
    assert (quality_dir / "dataset_level_quality_archive_manifest.local.json").exists()
    sidecar_summary = json.loads((quality_dir / "dataset_level_quality_archive_summary.json").read_text(encoding="utf-8"))

    assert sidecar_summary["archive_digest"]
    assert sidecar_summary["drive_archive_digest"]
    assert sidecar_summary["metadata"]["archive_payload_digest"]
    assert sidecar_summary["metadata"]["archive_digest_scope"] == "final_archive_file"
    assert sidecar_summary["metadata"]["final_archive_digest_available_in_sidecar"] is True

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "outputs/dataset_level_quality/dataset_level_quality_result.json" in names
        assert "outputs/dataset_level_quality/dataset_quality_image_records.jsonl" in names
        assert "outputs/dataset_level_quality/dataset_quality_image_resolution_records.jsonl" in names
        assert "outputs/dataset_level_quality/dataset_quality_formal_feature_records.jsonl" in names
        assert "outputs/dataset_level_quality/dataset_quality_formal_feature_import_report.json" in names
        assert "outputs/dataset_level_quality/dataset_quality_metrics.csv" in names
        assert "outputs/dataset_level_quality/dataset_quality_summary.json" in names
        assert "outputs/dataset_level_quality/dataset_level_quality_environment_report.json" in names
        assert "outputs/dataset_level_quality/manifest.local.json" in names
        assert "outputs/dataset_level_quality/dataset_level_quality_colab_manifest.local.json" in names
        assert "outputs/dataset_level_quality/dataset_level_quality_package_input_manifest.json" in names
        assert "outputs/dataset_level_quality/dataset_level_quality_archive_summary.json" in names
        assert "outputs/dataset_level_quality/dataset_level_quality_archive_manifest.local.json" in names
        embedded_summary = json.loads(
            archive.read("outputs/dataset_level_quality/dataset_level_quality_archive_summary.json").decode("utf-8")
        )
        input_manifest = json.loads(
            archive.read("outputs/dataset_level_quality/dataset_level_quality_package_input_manifest.json").decode("utf-8")
        )

        assert "archive_digest" not in embedded_summary
        assert "drive_archive_digest" not in embedded_summary
        assert embedded_summary["metadata"]["archive_payload_digest"]
        assert embedded_summary["metadata"]["archive_payload_digest"] == input_manifest["entry_payload_digest"]
        assert embedded_summary["metadata"]["archive_digest_scope"] == "external_sidecar_after_archive_write"
        assert embedded_summary["metadata"]["final_archive_digest_available_in_sidecar"] is True
