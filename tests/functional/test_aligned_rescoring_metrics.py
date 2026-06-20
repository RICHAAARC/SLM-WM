"""验证 aligned rescoring 的成对感知指标治理逻辑。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from paper_workflow.colab_utils import aligned_rescoring as helper


def make_config(**overrides: Any) -> helper.AlignedRescoringConfig:
    """构造轻量配置, 避免测试依赖真实 GPU 或真实模型。"""
    payload: dict[str, Any] = {
        "model_family": "sd35",
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "seed": 1703,
        "width": 512,
        "height": 512,
        "inference_steps": 20,
        "guidance_scale": 4.5,
        "attention_runtime_strength": 0.025,
        "injection_step_indices": (6, 10, 14),
    }
    payload.update(overrides)
    return helper.AlignedRescoringConfig(**payload)


@pytest.mark.quick
def test_default_config_requires_pair_perceptual_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认 Colab 配置应要求 LPIPS 与 CLIP pair-level 指标真实可用。"""
    monkeypatch.delenv("SLM_WM_ENABLE_PAIR_PERCEPTUAL_METRICS", raising=False)
    monkeypatch.delenv("SLM_WM_REQUIRE_PAIR_PERCEPTUAL_METRICS", raising=False)
    monkeypatch.delenv("SLM_WM_CLIP_MODEL_ID", raising=False)
    monkeypatch.delenv("SLM_WM_LPIPS_NETWORK", raising=False)

    config = helper.build_default_config()

    assert config.enable_pair_perceptual_metrics is True
    assert config.require_pair_perceptual_metrics is True
    assert config.clip_model_id == helper.DEFAULT_CLIP_MODEL_ID
    assert config.lpips_network == helper.DEFAULT_LPIPS_NETWORK
    assert config.perceptual_metric_device_name == "cpu"


@pytest.mark.quick
def test_pair_perceptual_metrics_ready_requires_lpips_and_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """只有 LPIPS 与 CLIP 都实测时, pair perceptual metrics 才能视为 ready。"""

    def measured_lpips(clean_image: object, aligned_image: object, config: helper.AlignedRescoringConfig) -> dict[str, Any]:
        return {"lpips": 0.12, "lpips_status": "measured", "lpips_error_type": ""}

    def measured_clip(
        clean_image: object,
        aligned_image: object,
        prompt_text: str,
        config: helper.AlignedRescoringConfig,
    ) -> dict[str, Any]:
        return {
            "clip_score": 0.31,
            "clip_score_clean": 0.30,
            "clip_score_aligned": 0.31,
            "clip_score_delta": 0.01,
            "clip_score_status": "measured",
            "clip_score_error_type": "",
        }

    monkeypatch.setattr(helper, "compute_lpips_metric", measured_lpips)
    monkeypatch.setattr(helper, "compute_clip_pair_metrics", measured_clip)

    metrics = helper.compute_pair_perceptual_metrics(object(), object(), "a semantic prompt", make_config())

    assert metrics["lpips"] == 0.12
    assert metrics["clip_score_clean"] == 0.30
    assert metrics["clip_score_aligned"] == 0.31
    assert metrics["clip_score_delta"] == 0.01
    assert metrics["perceptual_metrics_ready"] is True
    assert metrics["fid_status"] == "dataset_level_metric_not_computed_in_pair_run"
    assert metrics["kid_status"] == "dataset_level_metric_not_computed_in_pair_run"


@pytest.mark.quick
def test_pair_metric_status_summary_exposes_missing_metric() -> None:
    """失败摘要应直接显示 LPIPS 与 CLIP 的具体状态。"""
    summary = helper.pair_metric_status_summary(
        [
            {
                "lpips_status": "measured",
                "clip_score_status": "metric_runtime_error",
            }
        ]
    )

    assert summary == "lpips=measured;clip_score=metric_runtime_error"


@pytest.mark.quick
def test_quality_rows_include_pair_clip_columns(tmp_path: Path) -> None:
    """质量指标表应显式区分 clean / aligned CLIP score 和 delta。"""
    path = tmp_path / "quality.csv"
    helper.write_quality_rows(
        path,
        [
            {
                "carrier_id": "carrier",
                "prompt_id": "prompt",
                "attention_graph_id": "graph",
                "capture_id": "capture",
                "trajectory_index": 1,
                "timestep": 2.0,
                "update_norm": 0.1,
                "latent_norm_before": 1.0,
                "latent_norm_after": 1.1,
                "clean_image_path": "outputs/aligned_rescoring/clean_images/sample.png",
                "aligned_image_path": "outputs/aligned_rescoring/aligned_images/sample.png",
                "clean_image_digest": "clean",
                "aligned_image_digest": "aligned",
                "psnr": 30.0,
                "ssim": 0.99,
                "mse": 0.001,
                "mean_abs_error": 0.02,
                "lpips": 0.12,
                "lpips_status": "measured",
                "lpips_error_type": "",
                "clip_score": 0.31,
                "clip_score_clean": 0.30,
                "clip_score_aligned": 0.31,
                "clip_score_delta": 0.01,
                "clip_score_status": "measured",
                "clip_score_error_type": "",
                "fid": "unsupported",
                "fid_status": "dataset_level_metric_not_computed_in_pair_run",
                "kid": "unsupported",
                "kid_status": "dataset_level_metric_not_computed_in_pair_run",
                "image_quality_metrics_ready": True,
                "perceptual_metrics_ready": True,
            }
        ],
    )

    row = next(csv.DictReader(path.open(encoding="utf-8")))

    assert row["clip_score_clean"] == "0.3"
    assert row["clip_score_aligned"] == "0.31"
    assert row["clip_score_delta"] == "0.01"
    assert row["perceptual_metrics_ready"] == "True"
