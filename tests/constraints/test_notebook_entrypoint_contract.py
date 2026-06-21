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
from paper_workflow.colab_utils.real_attack_evaluation import package_real_attack_evaluation_outputs
from paper_workflow.colab_utils.threshold_calibration import package_threshold_calibration_outputs
from paper_workflow.colab_utils.sd_runtime_cold_start import package_probe_outputs
from tools.harness.lib.naming_rules import is_allowed_file_name


RUNTIME_NOTEBOOK_PATH = Path("paper_workflow/sd_runtime_cold_start_probe.ipynb")
INJECTION_NOTEBOOK_PATH = Path("paper_workflow/minimal_latent_injection_run.ipynb")
DRIVE_COLD_START_NOTEBOOK_PATH = Path("paper_workflow/colab_drive_cold_start_smoke.ipynb")
DRIVE_RELOAD_NOTEBOOK_PATH = Path("paper_workflow/drive_manifest_reload_smoke.ipynb")
ATTENTION_GEOMETRY_NOTEBOOK_PATH = Path("paper_workflow/attention_geometry_capture_run.ipynb")
ATTENTION_LATENT_INJECTION_NOTEBOOK_PATH = Path("paper_workflow/attention_latent_injection_run.ipynb")
ALIGNED_RESCORING_NOTEBOOK_PATH = Path("paper_workflow/aligned_rescoring_run.ipynb")
THRESHOLD_CALIBRATION_NOTEBOOK_PATH = Path("paper_workflow/threshold_calibration_run.ipynb")
REAL_ATTACK_EVALUATION_NOTEBOOK_PATH = Path("paper_workflow/real_attack_evaluation_run.ipynb")
NOTEBOOK_PATHS = (
    RUNTIME_NOTEBOOK_PATH,
    INJECTION_NOTEBOOK_PATH,
    DRIVE_COLD_START_NOTEBOOK_PATH,
    DRIVE_RELOAD_NOTEBOOK_PATH,
    ATTENTION_GEOMETRY_NOTEBOOK_PATH,
    ATTENTION_LATENT_INJECTION_NOTEBOOK_PATH,
    ALIGNED_RESCORING_NOTEBOOK_PATH,
    THRESHOLD_CALIBRATION_NOTEBOOK_PATH,
    REAL_ATTACK_EVALUATION_NOTEBOOK_PATH,
)
COLAB_RUNTIME_CONSTRAINTS_PATH = Path("configs/colab_sd35_runtime_constraints.txt")
COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND = (
    "%pip install -q --upgrade diffusers transformers accelerate safetensors sentencepiece protobuf huggingface_hub"
)
PAIR_PERCEPTUAL_DEPENDENCY_INSTALL_COMMAND = "%pip install -q --upgrade lpips"


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
def test_colab_notebook_delegates_runtime_logic_to_helper() -> None:
    """Notebook 必须调用 repository helper, 不能成为唯一实现."""
    payload = json.loads(RUNTIME_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    joined_source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])

    assert "paper_workflow.colab_utils.sd_runtime_cold_start" in joined_source
    assert "run_default_model_plan" in joined_source
    assert "package_probe_outputs" in joined_source
    assert "SLM_WM_MODEL_SELECTION', 'both'" in joined_source
    assert "/content/drive/MyDrive/SLM/real_sd_runtime_probe" in joined_source
    assert "datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%sz')" in joined_source
    assert "['git', 'rev-parse', '--short', 'HEAD']" in joined_source
    assert "archive_name=archive_name" in joined_source
    assert COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND in joined_source
    assert "--force-reinstall" not in joined_source
    assert "numpy pillow" not in joined_source
    assert "del sys.modules" not in joined_source
    assert "module_name == 'numpy'" not in joined_source
    assert '"diffusers==' not in joined_source
    assert '"transformers==' not in joined_source
    assert '"accelerate==' not in joined_source


@pytest.mark.constraint
def test_colab_notebook_delegates_injection_logic_to_helper() -> None:
    """Notebook 必须复用 repository helper 执行最小 latent injection."""
    payload = json.loads(INJECTION_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    joined_source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    first_code_cell = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
    first_code_source = "".join(first_code_cell.get("source", []))

    assert "paper_workflow.colab_utils.minimal_latent_injection" in joined_source
    assert "run_default_injection_plan" in joined_source
    assert "package_injection_outputs" in joined_source
    assert "SLM_WM_MODEL_SELECTION', 'auto'" in joined_source
    assert "/content/drive/MyDrive/SLM/minimal_diffusion_latent_injection" in joined_source
    assert "drive.mount('/content/drive')" in first_code_source
    assert "datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%sz')" in joined_source
    assert "['git', 'rev-parse', '--short', 'HEAD']" in joined_source
    assert "archive_name=archive_name" in joined_source
    assert COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND in joined_source
    assert "--force-reinstall" not in joined_source
    assert "numpy pillow" not in joined_source
    assert "del sys.modules" not in joined_source
    assert "module_name == 'numpy'" not in joined_source
    assert '"diffusers==' not in joined_source
    assert '"transformers==' not in joined_source
    assert '"accelerate==' not in joined_source


@pytest.mark.constraint
def test_colab_drive_notebooks_delegate_workflow_logic_to_helper() -> None:
    """Drive workflow Notebook 必须调用 repository helper, 不直接拼接 manifest."""
    cold_start_payload = json.loads(DRIVE_COLD_START_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    reload_payload = json.loads(DRIVE_RELOAD_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cold_start_source = "\n".join("".join(cell.get("source", [])) for cell in cold_start_payload["cells"])
    reload_source = "\n".join("".join(cell.get("source", [])) for cell in reload_payload["cells"])
    first_cold_start_code = next(cell for cell in cold_start_payload["cells"] if cell["cell_type"] == "code")
    first_reload_code = next(cell for cell in reload_payload["cells"] if cell["cell_type"] == "code")

    assert "paper_workflow.colab_utils.drive_workflow" in cold_start_source
    assert "run_colab_drive_workflow" in cold_start_source
    assert "write_reload_smoke_record" in reload_source
    assert "drive.mount('/content/drive')" in "".join(first_cold_start_code.get("source", []))
    assert "drive.mount('/content/drive')" in "".join(first_reload_code.get("source", []))
    assert "/content/drive/MyDrive/SLM" in cold_start_source
    assert "/content/drive/MyDrive/SLM" in reload_source
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
    assert "drive.mount('/content/drive')" in first_code_source
    assert "/content/drive/MyDrive/SLM/attention_geometry" in joined_source
    assert "attention_geometry_ready" in joined_source
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
    assert "drive.mount('/content/drive')" in first_code_source
    assert "/content/drive/MyDrive/SLM/aligned_rescoring" in joined_source
    assert "/content/drive/MyDrive/SLM/attention_geometry" in joined_source
    assert "real_aligned_rescore_count" in joined_source
    assert "SLM_WM_ENABLE_PAIR_PERCEPTUAL_METRICS', '1'" in joined_source
    assert "SLM_WM_REQUIRE_PAIR_PERCEPTUAL_METRICS', '1'" in joined_source
    assert "openai/clip-vit-base-patch32" in joined_source
    assert "SLM_WM_LPIPS_NETWORK', 'alex'" in joined_source
    assert "SLM_WM_PERCEPTUAL_METRIC_DEVICE', 'cpu'" in joined_source
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
    assert "drive.mount('/content/drive')" in first_code_source
    assert "/content/drive/MyDrive/SLM/threshold_calibration" in joined_source
    assert "/content/drive/MyDrive/SLM/attention_latent_injection" in joined_source
    assert "/content/drive/MyDrive/SLM/aligned_rescoring" in joined_source
    assert "attention_latent_injection_package_*.zip" in joined_source
    assert "aligned_rescoring_package_*.zip" in joined_source
    assert "threshold_calibration_ready" in joined_source
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
    assert "drive.mount('/content/drive')" in first_code_source
    assert "/content/drive/MyDrive/SLM/real_attack_evaluation" in joined_source
    assert "/content/drive/MyDrive/SLM/aligned_rescoring" in joined_source
    assert "/content/drive/MyDrive/SLM/threshold_calibration" in joined_source
    assert "aligned_rescoring_package_*.zip" in joined_source
    assert "real_attacked_image_closed_loop_ready" in joined_source
    assert "regeneration_attack_gpu_validation_ready" in joined_source
    assert "attack_detection_rerun_ready" in joined_source
    assert "formal_attack_detection_ready" in joined_source
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
