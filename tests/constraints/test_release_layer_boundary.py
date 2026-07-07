"""验证三层结构发布边界。"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.harness.audits.audit_dependency_boundaries import run_audit as run_dependency_boundary_audit


@pytest.mark.constraint
def test_release_layer_directories_exist() -> None:
    """核心方法、完整论文实验和 Colab 运行层必须都有明确目录。"""
    assert Path("main").is_dir()
    assert Path("experiments").is_dir()
    assert Path("paper_experiments").is_dir()
    assert Path("paper_workflow").is_dir()


@pytest.mark.constraint
def test_external_baseline_not_under_core_experiments() -> None:
    """外部 baseline 适配工程不得继续放在核心 experiments 目录下。"""
    assert not Path("experiments/baselines").exists()
    assert Path("paper_experiments/baselines").is_dir()


@pytest.mark.constraint
def test_paper_result_closure_runner_lives_in_full_experiment_layer() -> None:
    """论文结果闭合正式 runner 必须位于完整论文实验层。"""
    runner_path = Path("paper_experiments/runners/paper_result_closure.py")
    colab_wrapper_path = Path("paper_workflow/colab_utils/paper_result_closure.py")
    wrapper_text = colab_wrapper_path.read_text(encoding="utf-8")

    assert runner_path.is_file()
    assert "paper_experiments.runners.paper_result_closure" in wrapper_text
    assert "write_notebook_runtime_report" in wrapper_text


@pytest.mark.constraint
def test_external_baseline_method_faithful_runner_lives_in_full_experiment_layer() -> None:
    """外部 baseline method-faithful 正式 runner 必须位于完整论文实验层。"""
    runner_path = Path("paper_experiments/runners/external_baseline_method_faithful.py")
    colab_wrapper_path = Path("paper_workflow/colab_utils/external_baseline_method_faithful.py")
    runner_text = runner_path.read_text(encoding="utf-8")
    wrapper_text = colab_wrapper_path.read_text(encoding="utf-8")

    assert runner_path.is_file()
    assert "paper_workflow" not in runner_text
    assert "experiments.runtime.progress" in runner_text
    assert "experiments.runtime.repository_environment" in runner_text
    assert "paper_experiments.runners.external_baseline_method_faithful" in wrapper_text


@pytest.mark.constraint
def test_tree_ring_official_reference_runner_lives_in_full_experiment_layer() -> None:
    """Tree-Ring 官方参考复现正式 runner 必须位于完整论文实验层。"""
    runner_path = Path("paper_experiments/runners/tree_ring_official_reference.py")
    colab_wrapper_path = Path("paper_workflow/colab_utils/tree_ring_official_reference.py")
    runner_text = runner_path.read_text(encoding="utf-8")
    wrapper_text = colab_wrapper_path.read_text(encoding="utf-8")

    assert runner_path.is_file()
    assert "paper_workflow" not in runner_text
    assert "paper_experiments.runners.external_baseline_method_faithful" in runner_text
    assert "paper_experiments.runners.tree_ring_official_reference" in wrapper_text


@pytest.mark.constraint
def test_gaussian_shading_official_reference_runner_lives_in_full_experiment_layer() -> None:
    """Gaussian Shading 官方参考复现正式 runner 必须位于完整论文实验层。"""
    runner_path = Path("paper_experiments/runners/gaussian_shading_official_reference.py")
    colab_wrapper_path = Path("paper_workflow/colab_utils/gaussian_shading_official_reference.py")
    runner_text = runner_path.read_text(encoding="utf-8")
    wrapper_text = colab_wrapper_path.read_text(encoding="utf-8")

    assert runner_path.is_file()
    assert "paper_workflow" not in runner_text
    assert "paper_experiments.runners.external_baseline_method_faithful" in runner_text
    assert "paper_experiments.runners.gaussian_shading_official_reference" in wrapper_text


@pytest.mark.constraint
def test_shallow_diffuse_official_reference_runner_lives_in_full_experiment_layer() -> None:
    """Shallow Diffuse 官方参考复现正式 runner 必须位于完整论文实验层。"""
    runner_path = Path("paper_experiments/runners/shallow_diffuse_official_reference.py")
    colab_wrapper_path = Path("paper_workflow/colab_utils/shallow_diffuse_official_reference.py")
    runner_text = runner_path.read_text(encoding="utf-8")
    wrapper_text = colab_wrapper_path.read_text(encoding="utf-8")

    assert runner_path.is_file()
    assert "paper_workflow" not in runner_text
    assert "paper_experiments.runners.external_baseline_method_faithful" in runner_text
    assert "paper_experiments.runners.shallow_diffuse_official_reference" in wrapper_text


@pytest.mark.constraint
def test_t2smark_full_main_reproduction_runner_lives_in_full_experiment_layer() -> None:
    """T2SMark 官方路径复现正式 runner 必须位于完整论文实验层。"""
    runner_path = Path("paper_experiments/runners/t2smark_full_main_reproduction.py")
    colab_wrapper_path = Path("paper_workflow/colab_utils/t2smark_full_main_reproduction.py")
    runner_text = runner_path.read_text(encoding="utf-8")
    wrapper_text = colab_wrapper_path.read_text(encoding="utf-8")

    assert runner_path.is_file()
    assert "paper_workflow" not in runner_text
    assert "paper_experiments.runners.external_baseline_method_faithful" in runner_text
    assert "paper_experiments.runners.t2smark_full_main_reproduction" in wrapper_text


@pytest.mark.constraint
def test_threshold_calibration_runner_lives_in_core_experiment_layer() -> None:
    """阈值校准正式 runner 必须位于核心方法复现层。"""
    runner_path = Path("experiments/runners/threshold_calibration.py")
    colab_wrapper_path = Path("paper_workflow/colab_utils/threshold_calibration.py")
    runner_text = runner_path.read_text(encoding="utf-8")
    wrapper_text = colab_wrapper_path.read_text(encoding="utf-8")

    assert runner_path.is_file()
    assert "paper_workflow" not in runner_text
    assert "paper_experiments" not in runner_text
    assert "experiments.runtime.repository_environment" in runner_text
    assert "experiments.runners.threshold_calibration" in wrapper_text


@pytest.mark.constraint
def test_threshold_calibration_artifact_builders_live_in_core_experiment_layer() -> None:
    """阈值校准相关产物构建逻辑不得由核心 runner 反向依赖 scripts。"""
    threshold_builder_path = Path("experiments/artifacts/threshold_calibration_outputs.py")
    rescue_builder_path = Path("experiments/artifacts/geometric_rescue_outputs.py")
    threshold_script_path = Path("scripts/write_threshold_calibration_outputs.py")
    rescue_script_path = Path("scripts/write_geometric_rescue_outputs.py")
    threshold_builder_text = threshold_builder_path.read_text(encoding="utf-8")
    rescue_builder_text = rescue_builder_path.read_text(encoding="utf-8")
    threshold_script_text = threshold_script_path.read_text(encoding="utf-8")
    rescue_script_text = rescue_script_path.read_text(encoding="utf-8")

    assert threshold_builder_path.is_file()
    assert rescue_builder_path.is_file()
    assert "scripts." not in threshold_builder_text
    assert "scripts." not in rescue_builder_text
    assert "experiments.artifacts.threshold_calibration_outputs" in threshold_script_text
    assert "experiments.artifacts.geometric_rescue_outputs" in rescue_script_text


@pytest.mark.constraint
def test_dataset_level_quality_runner_lives_in_core_experiment_layer() -> None:
    """数据集级质量正式 runner 必须位于核心方法复现层。"""
    runner_path = Path("experiments/runners/dataset_level_quality.py")
    colab_wrapper_path = Path("paper_workflow/colab_utils/dataset_level_quality.py")
    builder_path = Path("experiments/artifacts/dataset_level_quality_outputs.py")
    script_path = Path("scripts/write_dataset_level_quality_outputs.py")
    runner_text = runner_path.read_text(encoding="utf-8")
    wrapper_text = colab_wrapper_path.read_text(encoding="utf-8")
    builder_text = builder_path.read_text(encoding="utf-8")
    script_text = script_path.read_text(encoding="utf-8")

    assert runner_path.is_file()
    assert builder_path.is_file()
    assert "paper_workflow" not in runner_text
    assert "paper_experiments" not in runner_text
    assert "from scripts" not in runner_text
    assert "scripts." not in builder_text
    assert "experiments.runtime.repository_environment" in runner_text
    assert "experiments.runtime.progress" in runner_text
    assert "experiments.artifacts.dataset_level_quality_outputs" in runner_text
    assert "experiments.runners.dataset_level_quality" in wrapper_text
    assert "experiments.artifacts.dataset_level_quality_outputs" in script_text


@pytest.mark.constraint
def test_real_attack_evaluation_runner_lives_in_core_experiment_layer() -> None:
    """真实攻击闭环正式 runner 必须位于核心方法复现层。"""
    runner_path = Path("experiments/runners/real_attack_evaluation.py")
    colab_wrapper_path = Path("paper_workflow/colab_utils/real_attack_evaluation.py")
    runner_text = runner_path.read_text(encoding="utf-8")
    wrapper_text = colab_wrapper_path.read_text(encoding="utf-8")

    assert runner_path.is_file()
    assert "paper_workflow" not in runner_text
    assert "paper_experiments" not in runner_text
    assert "from scripts" not in runner_text
    assert "experiments.runtime.repository_environment" in runner_text
    assert "experiments.runtime.progress" in runner_text
    assert "experiments.runtime.diffusion.sd3_pipeline_runtime" in runner_text
    assert "experiments.runners.real_attack_evaluation" in wrapper_text


@pytest.mark.constraint
def test_conventional_geometric_attack_runner_lives_in_core_experiment_layer() -> None:
    """常规失真与几何攻击正式 runner 必须位于核心方法复现层。"""
    runner_path = Path("experiments/runners/conventional_geometric_attack_evaluation.py")
    colab_wrapper_path = Path("paper_workflow/colab_utils/conventional_geometric_attack_evaluation.py")
    runner_text = runner_path.read_text(encoding="utf-8")
    wrapper_text = colab_wrapper_path.read_text(encoding="utf-8")

    assert runner_path.is_file()
    assert "paper_workflow" not in runner_text
    assert "paper_experiments" not in runner_text
    assert "from scripts" not in runner_text
    assert "experiments.runtime.repository_environment" in runner_text
    assert "experiments.runtime.progress" in runner_text
    assert "experiments.runners.real_attack_evaluation" in runner_text
    assert "experiments.runners.conventional_geometric_attack_evaluation" in wrapper_text


@pytest.mark.constraint
def test_attention_geometry_runner_lives_in_core_experiment_layer() -> None:
    """attention 几何捕获正式 runner 必须位于核心方法复现层。"""
    runner_path = Path("experiments/runners/attention_geometry_capture.py")
    colab_wrapper_path = Path("paper_workflow/colab_utils/attention_geometry_capture.py")
    builder_path = Path("experiments/artifacts/attention_geometry_outputs.py")
    script_path = Path("scripts/write_attention_geometry_outputs.py")
    runner_text = runner_path.read_text(encoding="utf-8")
    wrapper_text = colab_wrapper_path.read_text(encoding="utf-8")
    builder_text = builder_path.read_text(encoding="utf-8")
    script_text = script_path.read_text(encoding="utf-8")

    assert runner_path.is_file()
    assert builder_path.is_file()
    assert "paper_workflow" not in runner_text
    assert "paper_experiments" not in runner_text
    assert "from scripts" not in runner_text
    assert "scripts." not in builder_text
    assert "experiments.runners.attention_geometry_capture" in wrapper_text
    assert "experiments.artifacts.attention_geometry_outputs" in script_text


@pytest.mark.constraint
def test_attention_latent_injection_runner_lives_in_core_experiment_layer() -> None:
    """attention latent injection 正式 runner 必须位于核心方法复现层。"""
    runner_path = Path("experiments/runners/attention_latent_injection.py")
    colab_wrapper_path = Path("paper_workflow/colab_utils/attention_latent_injection.py")
    runtime_path = Path("experiments/runtime/diffusion/sd3_pipeline_runtime.py")
    artifact_paths = (
        Path("experiments/artifacts/attention_latent_update_outputs.py"),
        Path("experiments/artifacts/content_carrier_outputs.py"),
        Path("experiments/artifacts/prompt_event_protocol_outputs.py"),
        Path("experiments/artifacts/semantic_subspace_outputs.py"),
    )
    script_paths = (
        Path("scripts/write_attention_latent_update_outputs.py"),
        Path("scripts/write_content_carrier_outputs.py"),
        Path("scripts/write_prompt_event_protocol.py"),
        Path("scripts/write_semantic_subspace_outputs.py"),
    )
    runner_text = runner_path.read_text(encoding="utf-8")
    wrapper_text = colab_wrapper_path.read_text(encoding="utf-8")

    assert runner_path.is_file()
    assert runtime_path.is_file()
    assert "paper_workflow" not in runner_text
    assert "paper_experiments" not in runner_text
    assert "from scripts" not in runner_text
    assert "experiments.runtime.diffusion.sd3_pipeline_runtime" in runner_text
    assert "experiments.runners.attention_latent_injection" in wrapper_text
    for artifact_path in artifact_paths:
        artifact_text = artifact_path.read_text(encoding="utf-8")
        assert artifact_path.is_file()
        assert "scripts." not in artifact_text
    for script_path in script_paths:
        script_text = script_path.read_text(encoding="utf-8")
        expected_module = script_path.stem.removeprefix("write_")
        if expected_module == "prompt_event_protocol":
            expected_module = "prompt_event_protocol_outputs"
        elif expected_module == "semantic_subspace_outputs":
            expected_module = "semantic_subspace_outputs"
        assert f"experiments.artifacts.{expected_module}" in script_text


@pytest.mark.constraint
def test_aligned_rescoring_runner_lives_in_core_experiment_layer() -> None:
    """aligned rescoring 正式 runner 必须位于核心方法复现层。"""
    runner_path = Path("experiments/runners/aligned_rescoring.py")
    colab_wrapper_path = Path("paper_workflow/colab_utils/aligned_rescoring.py")
    runner_text = runner_path.read_text(encoding="utf-8")
    wrapper_text = colab_wrapper_path.read_text(encoding="utf-8")

    assert runner_path.is_file()
    assert "paper_workflow" not in runner_text
    assert "paper_experiments" not in runner_text
    assert "from scripts" not in runner_text
    assert "experiments.runners.attention_latent_injection" in runner_text
    assert "experiments.runtime.diffusion.sd3_pipeline_runtime" in runner_text
    assert "experiments.runners.aligned_rescoring" in wrapper_text


@pytest.mark.constraint
def test_sd_runtime_cold_start_runner_lives_in_core_experiment_layer() -> None:
    """真实 SD runtime 诊断 probe 正式 runner 必须位于核心方法复现层。"""
    runner_path = Path("experiments/runners/sd_runtime_cold_start.py")
    colab_wrapper_path = Path("paper_workflow/colab_utils/sd_runtime_cold_start.py")
    runner_text = runner_path.read_text(encoding="utf-8")
    wrapper_text = colab_wrapper_path.read_text(encoding="utf-8")

    assert runner_path.is_file()
    assert "paper_workflow" not in runner_text
    assert "paper_experiments" not in runner_text
    assert "from scripts" not in runner_text
    assert "experiments.runtime.repository_environment" in runner_text
    assert "experiments.runners.sd_runtime_cold_start" in wrapper_text


@pytest.mark.constraint
def test_minimal_latent_injection_runner_lives_in_core_experiment_layer() -> None:
    """最小 latent injection 机制预检正式 runner 必须位于核心方法复现层。"""
    runner_path = Path("experiments/runners/minimal_latent_injection.py")
    colab_wrapper_path = Path("paper_workflow/colab_utils/minimal_latent_injection.py")
    runner_text = runner_path.read_text(encoding="utf-8")
    wrapper_text = colab_wrapper_path.read_text(encoding="utf-8")

    assert runner_path.is_file()
    assert "paper_workflow" not in runner_text
    assert "paper_experiments" not in runner_text
    assert "from scripts" not in runner_text
    assert "experiments.runtime.repository_environment" in runner_text
    assert "experiments.runners.minimal_latent_injection" in wrapper_text


@pytest.mark.constraint
def test_repository_environment_tools_live_outside_colab_layer() -> None:
    """运行环境摘要工具必须位于核心方法复现层, 供服务器和 Colab 共同复用。"""
    environment_tool_path = Path("experiments/runtime/repository_environment.py")
    progress_tool_path = Path("experiments/runtime/progress.py")
    colab_runtime_path = Path("paper_workflow/colab_utils/sd_runtime_cold_start.py")
    notebook_progress_path = Path("paper_workflow/notebook_utils/progress.py")
    colab_runtime_text = colab_runtime_path.read_text(encoding="utf-8")
    notebook_progress_text = notebook_progress_path.read_text(encoding="utf-8")

    assert environment_tool_path.is_file()
    assert progress_tool_path.is_file()
    assert "experiments.runners.sd_runtime_cold_start" in colab_runtime_text
    assert "experiments.runtime.progress" in notebook_progress_text


@pytest.mark.constraint
def test_release_layer_dependency_audit_passes() -> None:
    """三层结构的导入方向必须满足 3 -> 2 -> 1。"""
    report = run_dependency_boundary_audit(Path.cwd())
    assert report["decision"] == "pass", report["violations"]

@pytest.mark.constraint
def test_gpu_server_workflow_does_not_depend_on_colab_wrappers() -> None:
    """服务器 workflow 入口不得通过 Colab wrapper 调用正式 runner。"""
    script_path = Path("scripts/run_gpu_server_workflow.py")
    script_text = script_path.read_text(encoding="utf-8")

    assert "paper_workflow.colab_utils" not in script_text
    assert "paper_workflow.notebook_utils" not in script_text
    assert "experiments.runtime.archive_naming" in script_text
    assert "paper_experiments.runners.external_baseline_method_faithful" in script_text

@pytest.mark.constraint
def test_notebook_archive_naming_delegates_to_core_runtime_tool() -> None:
    """Notebook 入口层不得保留独立归档命名实现。"""
    entrypoint_path = Path("paper_workflow/notebook_utils/notebook_entrypoint.py")
    entrypoint_text = entrypoint_path.read_text(encoding="utf-8")

    assert "experiments.runtime.archive_naming" in entrypoint_text
    assert "def build_workflow_archive_name" not in entrypoint_text
    assert "def resolve_short_commit" not in entrypoint_text
    assert "def utc_archive_token" not in entrypoint_text

