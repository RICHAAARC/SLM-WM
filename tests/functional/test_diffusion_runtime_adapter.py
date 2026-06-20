"""验证 SD runtime adapter 的轻量工程能力。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runtime.diffusion.model_adapter import RuntimeModelConfig
from experiments.runtime.diffusion.sd3_adapter import Sd3RuntimeAdapter
from scripts.run_diffusion_runtime_probe import write_runtime_probe_outputs


@pytest.mark.quick
def test_synthetic_fallback_records_unsupported_reason() -> None:
    """无真实模型环境时, adapter 必须记录 unsupported_reason。"""
    config = RuntimeModelConfig(
        model_family="sd3",
        model_id="stabilityai/stable-diffusion-3-medium-diffusers",
        backend_mode="synthetic_fallback",
        prompt="a glass sphere on a wooden table",
        negative_prompt="low quality",
        seed=1703,
        width=512,
        height=512,
        inference_steps=4,
        guidance_scale=4.5,
        latent_width=6,
    )
    bundle = Sd3RuntimeAdapter().generate(config)

    assert bundle.generation_record["unsupported_reason"] == "real_sd3_backend_unavailable"
    assert bundle.generation_record["metadata"]["records_are_synthetic"] is True
    assert bundle.generation_record["metadata"]["supports_paper_claim"] is False
    assert len(bundle.latent_trace_records) == config.inference_steps
    assert len(bundle.attention_capture_records) == 2


@pytest.mark.quick
def test_runtime_adapter_is_reproducible_for_same_config() -> None:
    """相同 prompt、seed 与模型配置应产生稳定摘要。"""
    config = RuntimeModelConfig(
        model_family="sd3",
        model_id="stabilityai/stable-diffusion-3-medium-diffusers",
        backend_mode="synthetic_fallback",
        prompt="a glass sphere on a wooden table",
        negative_prompt="low quality",
        seed=1703,
        width=512,
        height=512,
        inference_steps=4,
        guidance_scale=4.5,
        latent_width=6,
    )
    first = Sd3RuntimeAdapter().generate(config)
    second = Sd3RuntimeAdapter().generate(config)

    assert first.generation_record["latent_digest"] == second.generation_record["latent_digest"]
    assert first.generation_record["image_digest"] == second.generation_record["image_digest"]
    assert [record.latent_digest for record in first.latent_trace_records] == [
        record.latent_digest for record in second.latent_trace_records
    ]


@pytest.mark.quick
def test_runtime_probe_writer_uses_outputs_directory(tmp_path: Path) -> None:
    """runtime probe writer 的所有持久化产物必须位于 outputs/ 下。"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "model_sd3.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_family: sd3",
                "model_id: stabilityai/stable-diffusion-3-medium-diffusers",
                "backend_mode: synthetic_fallback",
                "prompt: a glass sphere on a wooden table",
                "negative_prompt: low quality",
                "seed: 1703",
                "width: 512",
                "height: 512",
                "inference_steps: 4",
                "guidance_scale: 4.5",
                "latent_width: 6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = write_runtime_probe_outputs(
        root=tmp_path,
        config_paths=(Path("configs/model_sd3.yaml"),),
        output_dir=Path("outputs/sd_runtime_adapter"),
    )

    output_paths = manifest["output_paths"]
    assert output_paths
    assert all(path.startswith("outputs/sd_runtime_adapter/") for path in output_paths)
    summary_path = tmp_path / "outputs" / "sd_runtime_adapter" / "generation_quality_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["decision"] == "pass"
    assert summary["generation_record_count"] == 1
    assert summary["unsupported_reason_count"] == 1
