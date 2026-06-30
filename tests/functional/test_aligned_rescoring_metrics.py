"""验证 aligned rescoring 的成对感知指标治理逻辑。"""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import types
from typing import Any

import pytest

from paper_workflow.colab_utils import aligned_rescoring as helper
from paper_workflow.colab_utils import attention_latent_injection as attention_helper


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
    monkeypatch.delenv("SLM_WM_ENABLE_PIPELINE_PROGRESS_BAR", raising=False)
    monkeypatch.delenv("SLM_WM_ENABLE_CARRIER_PROGRESS_BAR", raising=False)

    config = helper.build_default_config()

    assert config.enable_pair_perceptual_metrics is True
    assert config.require_pair_perceptual_metrics is True
    assert config.clip_model_id == helper.DEFAULT_CLIP_MODEL_ID
    assert config.lpips_network == helper.DEFAULT_LPIPS_NETWORK
    assert config.perceptual_metric_device_name == "cpu"
    assert config.enable_pipeline_progress_bar is False
    assert config.enable_carrier_progress_bar is True


@pytest.mark.quick
def test_pipeline_progress_bar_defaults_to_disabled() -> None:
    """Diffusers 单次推理进度条默认关闭, 由 carrier 总进度承担长流程反馈。"""

    class FakePipeline:
        def __init__(self) -> None:
            self.progress_bar_config: dict[str, bool] = {}

        def set_progress_bar_config(self, **kwargs: bool) -> None:
            self.progress_bar_config.update(kwargs)

    pipeline = FakePipeline()
    status = helper.configure_pipeline_progress_bar(pipeline, make_config())

    assert status == "pipeline_progress_bar_disabled"
    assert pipeline.progress_bar_config == {"disable": True}


@pytest.mark.quick
def test_pipeline_progress_bar_can_be_reenabled() -> None:
    """排查单次 diffusion 调用时, 仍可通过配置恢复 pipeline 内部进度条。"""

    class FakePipeline:
        def __init__(self) -> None:
            self.progress_bar_config: dict[str, bool] = {}

        def set_progress_bar_config(self, **kwargs: bool) -> None:
            self.progress_bar_config.update(kwargs)

    pipeline = FakePipeline()
    status = helper.configure_pipeline_progress_bar(pipeline, make_config(enable_pipeline_progress_bar=True))

    assert status == "pipeline_progress_bar_enabled"
    assert pipeline.progress_bar_config == {"disable": False}


@pytest.mark.quick
def test_carrier_progress_iterator_preserves_record_order_without_tqdm() -> None:
    """关闭总进度条时, carrier 迭代顺序和索引应保持不变。"""
    records = ({"carrier_id": "a"}, {"carrier_id": "b"})

    observed = list(helper.iterate_carriers_with_progress(records, make_config(enable_carrier_progress_bar=False)))

    assert observed == [(0, {"carrier_id": "a"}), (1, {"carrier_id": "b"})]


@pytest.mark.quick
def test_slot_pooled_projection_recovers_repeated_content_carrier() -> None:
    """真实 latent 投影应读取重复写入的 content carrier 槽位信号。"""
    update_values = (0.40, -0.20, 0.35, -0.15, 0.10, -0.30, 0.25, -0.45)
    repeated_values = update_values * 16

    projected = helper.normalized_slot_projection_from_values(repeated_values, len(update_values))

    mean_value = sum(update_values) / len(update_values)
    centered = tuple(value - mean_value for value in update_values)
    norm = sum(value * value for value in centered) ** 0.5
    expected = tuple(value / norm for value in centered)
    similarity = sum(left * right for left, right in zip(projected, expected))
    assert similarity > 0.99


@pytest.mark.quick
def test_runtime_watermark_tensor_is_content_aligned_when_attention_direction_conflicts() -> None:
    """attention 几何方向与内容方向相反时, 运行时 carrier 应翻转几何方向并保持内容可检测。"""
    assert attention_helper.runtime_attention_sign(-0.2) == -1.0
    assert attention_helper.runtime_attention_sign(0.0) == 1.0
    assert attention_helper.runtime_attention_sign(0.2) == 1.0


@pytest.mark.quick
def test_lpips_metric_model_loader_reuses_cached_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """LPIPS 模型加载应在相同配置下复用缓存, 避免每个 carrier 重复初始化。"""
    load_count = {"lpips": 0}

    class FakeLpipsModel:
        def to(self, device: object) -> "FakeLpipsModel":
            self.device = device
            return self

        def eval(self) -> None:
            self.evaluated = True

    class FakeTorch:
        @staticmethod
        def device(device_name: str) -> str:
            return device_name

    fake_lpips = types.ModuleType("lpips")

    def fake_lpips_factory(net: str) -> FakeLpipsModel:
        load_count["lpips"] += 1
        assert net == "alex"
        return FakeLpipsModel()

    fake_lpips.LPIPS = fake_lpips_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lpips", fake_lpips)
    monkeypatch.setattr(helper, "import_runtime_dependencies", lambda: (None, FakeTorch, None, None))
    helper.load_lpips_metric_model.cache_clear()

    try:
        first_model = helper.load_lpips_metric_model("alex", "cpu")
        second_model = helper.load_lpips_metric_model("alex", "cpu")
    finally:
        helper.load_lpips_metric_model.cache_clear()

    assert first_model is second_model
    assert load_count["lpips"] == 1


@pytest.mark.quick
def test_clip_metric_loader_reuses_cached_model_and_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLIP processor 和 model 应在相同配置下复用缓存, 避免成对质量计算重复下载或加载。"""
    load_count = {"processor": 0, "model": 0}

    class FakeTorch:
        @staticmethod
        def device(device_name: str) -> str:
            return device_name

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "FakeProcessor":
            load_count["processor"] += 1
            assert model_id == helper.DEFAULT_CLIP_MODEL_ID
            assert kwargs == {}
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "FakeModel":
            load_count["model"] += 1
            assert model_id == helper.DEFAULT_CLIP_MODEL_ID
            assert kwargs == {}
            return cls()

        def to(self, device: object) -> "FakeModel":
            self.device = device
            return self

        def eval(self) -> None:
            self.evaluated = True

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.CLIPProcessor = FakeProcessor  # type: ignore[attr-defined]
    fake_transformers.CLIPModel = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(helper, "import_runtime_dependencies", lambda: (None, FakeTorch, None, None))
    helper.load_clip_metric_objects.cache_clear()

    try:
        first_processor, first_model = helper.load_clip_metric_objects(helper.DEFAULT_CLIP_MODEL_ID, "cpu", "")
        second_processor, second_model = helper.load_clip_metric_objects(helper.DEFAULT_CLIP_MODEL_ID, "cpu", "")
    finally:
        helper.load_clip_metric_objects.cache_clear()

    assert first_processor is second_processor
    assert first_model is second_model
    assert load_count == {"processor": 1, "model": 1}


@pytest.mark.quick
def test_pair_perceptual_metrics_ready_requires_lpips_and_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """只有 LPIPS 与 CLIP 都实测时, pair perceptual metrics 才能视为 ready。"""

    def measured_lpips(clean_image: object, aligned_image: object, config: helper.AlignedRescoringConfig) -> dict[str, Any]:
        return {"lpips": 0.12, "lpips_status": "measured", "lpips_error_type": "", "lpips_error_message": ""}

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
            "clip_score_error_message": "",
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
def test_clip_forward_api_fallback_reads_logits() -> None:
    """CLIP embedding API 不可用时, 应能从 forward logits 中读取成对分数。"""

    class FakeScalar:
        def __init__(self, value: float) -> None:
            self.value = value

        def detach(self) -> "FakeScalar":
            return self

        def cpu(self) -> "FakeScalar":
            return self

        def item(self) -> float:
            return self.value

    class FakeScores:
        def __init__(self, values: tuple[float, float]) -> None:
            self.values = values

        def float(self) -> "FakeScores":
            return self

        def reshape(self, *shape: int) -> "FakeScores":
            return self

        def __getitem__(self, index: int) -> FakeScalar:
            return FakeScalar(self.values[index])

    class FakeProcessor:
        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            return {"pixel_values": object(), "input_ids": object()}

    class FakeModel:
        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            return {"logits_per_image": FakeScores((0.25, 0.375))}

    class FakeImage:
        def convert(self, mode: str) -> "FakeImage":
            return self

    clean_score, aligned_score = helper.compute_clip_scores_with_forward_api(
        FakeModel(),
        FakeProcessor(),
        FakeImage(),
        FakeImage(),
        "semantic prompt",
        object(),
    )

    assert clean_score == 0.25
    assert aligned_score == 0.375


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
                "lpips_error_message": "",
                "clip_score": 0.31,
                "clip_score_clean": 0.30,
                "clip_score_aligned": 0.31,
                "clip_score_delta": 0.01,
                "clip_score_status": "measured",
                "clip_score_error_type": "",
                "clip_score_error_message": "",
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
