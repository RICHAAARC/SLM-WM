"""验证最小 latent injection helper 的轻量纯函数行为."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from experiments.runners import minimal_latent_injection as injection_helper
from experiments.runners.minimal_latent_injection import (
    InjectionRunConfig,
    build_default_configs,
    build_injection_id,
    compute_image_quality_metrics,
    derive_core_carrier_values,
)
from experiments.runners.sd_runtime_cold_start import (
    COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND,
    build_runtime_environment_report,
    flatten_environment_versions,
)


def make_config(**overrides: object) -> InjectionRunConfig:
    """构造不加载真实模型的最小配置, 用于复用配置与摘要测试."""
    payload = {
        "model_family": "sd35",
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "model_priority": "primary",
        "prompt": "a glass sphere on a wooden table",
        "negative_prompt": "low quality",
        "seed": 1703,
        "width": 512,
        "height": 512,
        "inference_steps": 28,
        "guidance_scale": 4.5,
        "injection_strength": 0.035,
        "injection_step_indices": (8, 14, 20),
        "watermark_key_digest": "0" * 64,
    }
    payload.update(overrides)
    return InjectionRunConfig(**payload)


@pytest.mark.quick
def test_injection_id_is_stable_for_same_config() -> None:
    """相同配置应生成相同 injection_id, 便于复现实验记录."""
    first = build_injection_id(make_config())
    second = build_injection_id(make_config())

    assert first == second
    assert len(first) == 64


@pytest.mark.quick
def test_config_rejects_injection_indices_outside_sampling_boundary() -> None:
    """配置构造层负责拦截越界注入位置, 业务路径无需重复校验."""
    with pytest.raises(ValueError, match="injection_step_indices"):
        make_config(injection_step_indices=(28,))


@pytest.mark.quick
def test_quality_metrics_identical_images_are_lossless() -> None:
    """相同 paired image 应得到零误差和无穷 PSNR."""
    image = Image.new("RGB", (4, 4), color=(120, 80, 40))

    metrics = compute_image_quality_metrics(image, image.copy())

    assert metrics["psnr"] == "inf"
    assert metrics["mse"] == 0.0
    assert metrics["mean_abs_error"] == 0.0
    assert metrics["ssim"] == pytest.approx(1.0)


@pytest.mark.quick
def test_core_carrier_values_reuse_algorithm_primitives() -> None:
    """最小注入载体应由核心算法原语导出, 而不是由 Notebook 临时生成."""
    carrier_values, metadata = derive_core_carrier_values(make_config(), trajectory_index=3, carrier_width=16)

    assert len(carrier_values) == 16
    assert metadata["carrier_source"] == "core_algorithm_primitives"
    assert len(metadata["core_update_digest"]) == 64


@pytest.mark.quick
def test_default_model_selection_keeps_primary_and_fallback() -> None:
    """默认模型选择应同时保留主线模型和兼容对照."""
    configs = build_default_configs(model_selection="both")

    assert [config.model_family for config in configs] == ["sd35", "sd3"]
    assert [config.model_priority for config in configs] == ["primary", "compatibility_fallback"]


@pytest.mark.quick
def test_runtime_environment_report_records_dependency_provenance() -> None:
    """环境快照应记录 Colab 动态升级命令和关键依赖版本, 便于复现实验。"""
    report = build_runtime_environment_report()
    versions = flatten_environment_versions(report)

    assert report["dependency_mode"] == "colab_dynamic_upgrade"
    assert report["manual_version_pins"] is False
    assert report["pip_install_command"] == COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND
    assert report["python_version"]
    assert set(versions) >= {
        "torch_version",
        "diffusers_version",
        "transformers_version",
        "accelerate_version",
        "huggingface_hub_version",
        "numpy_version",
        "pillow_version",
    }


@pytest.mark.quick
def test_injection_writer_persists_environment_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """写出受治理产物时应同步保存环境报告, 即使真实后端不可用也能审计依赖。"""

    def raise_backend_error(config: InjectionRunConfig) -> None:
        raise RuntimeError("simulated_backend_unavailable")

    monkeypatch.setattr(injection_helper, "run_single_injection", raise_backend_error)
    config = make_config(output_dir="outputs/minimal_diffusion_latent_injection")

    result = injection_helper.write_single_injection_outputs(config=config, root=tmp_path)

    output_dir = tmp_path / "outputs" / "minimal_diffusion_latent_injection"
    environment_path = output_dir / "sd35_environment_report.json"
    result_path = output_dir / "sd35_injection_result.json"
    manifest_path = output_dir / "sd35_manifest.local.json"
    environment_report = json.loads(environment_path.read_text(encoding="utf-8"))
    persisted_result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result["run_decision"] == "fail"
    assert environment_report["pip_install_command"] == COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND
    assert persisted_result["metadata"]["environment_report_path"] == (
        "outputs/minimal_diffusion_latent_injection/sd35_environment_report.json"
    )
    assert "outputs/minimal_diffusion_latent_injection/sd35_environment_report.json" in manifest["output_paths"]
