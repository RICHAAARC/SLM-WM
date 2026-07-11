"""验证模型资源登记与主方法精确 revision 传递."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.protocol.method_runtime_config import load_formal_method_runtime_config
from experiments.runners.semantic_watermark_runtime import SemanticWatermarkRuntimeConfig
from experiments.runtime.diffusion import sd3_pipeline_runtime, semantic_features
from experiments.runtime.model_sources import (
    MODEL_SOURCE_REGISTRY_PATH,
    get_model_source,
    load_model_source_registry,
    require_registered_model_reference,
)
from scripts import run_image_only_dataset_runtime


@pytest.mark.quick
def test_primary_model_config_matches_immutable_source_registry() -> None:
    """主方法配置必须与登记表中的 SD3.5 和 CLIP 精确提交一致."""

    config = SemanticWatermarkRuntimeConfig()
    diffusion_source = get_model_source("stabilityai_stable_diffusion_3_5_medium")
    vision_source = get_model_source("openai_clip_vit_base_patch32")
    method_config = load_formal_method_runtime_config(".")

    assert (config.model_id, config.model_revision) == (
        diffusion_source.repository_id,
        diffusion_source.revision,
    )
    assert (config.vision_model_id, config.vision_model_revision) == (
        vision_source.repository_id,
        vision_source.revision,
    )
    assert method_config.model_revision == diffusion_source.revision
    assert method_config.vision_model_revision == vision_source.revision
    assert config.inference_steps == method_config.inference_steps
    assert config.injection_step_indices == method_config.injection_step_indices
    assert config.candidate_count == method_config.jacobian_candidate_count
    assert config.null_rank == method_config.null_space_rank
    assert diffusion_source.revision_url.endswith(f"/tree/{diffusion_source.revision}")
    assert "primary_diffusion_model" in diffusion_source.usage_roles
    assert "semantic_condition_encoder" in vision_source.usage_roles
    assert config.carrier_model_reference == (
        "stabilityai/stable-diffusion-3.5-medium@"
        "b940f670f0eda2d07fbb75229e779da1ad11eb80"
    )


@pytest.mark.quick
def test_dataset_runtime_entrypoint_uses_registered_revision_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """独立脚本未显式覆盖模型时必须沿用登记表中的精确提交."""

    for variable_name in (
        "SLM_WM_MODEL_ID",
        "SLM_WM_MODEL_REVISION",
        "SLM_WM_VISION_MODEL_ID",
        "SLM_WM_VISION_MODEL_REVISION",
    ):
        monkeypatch.delenv(variable_name, raising=False)

    config = run_image_only_dataset_runtime.build_method_config(".")

    assert config.model_revision == get_model_source(
        "stabilityai_stable_diffusion_3_5_medium"
    ).revision
    assert config.vision_model_revision == get_model_source(
        "openai_clip_vit_base_patch32"
    ).revision
    assert config.inference_steps == load_formal_method_runtime_config(".").inference_steps


@pytest.mark.quick
def test_dataset_runtime_rejects_method_environment_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式脚本不得通过环境变量形成未登记的模型或方法参数分支。"""

    monkeypatch.setenv("SLM_WM_MODEL_REVISION", "0" * 40)

    with pytest.raises(ValueError, match="model_sd35.yaml 不一致"):
        run_image_only_dataset_runtime.build_method_config(".")


@pytest.mark.quick
def test_model_source_registry_rejects_mutable_or_unregistered_revisions(tmp_path: Path) -> None:
    """登记表和运行引用都不得接受分支名、短提交或未登记提交."""

    payload = json.loads(MODEL_SOURCE_REGISTRY_PATH.read_text(encoding="utf-8"))
    payload["sources"]["openai_clip_vit_base_patch32"]["revision"] = "main"
    invalid_path = tmp_path / "model_source_registry.json"
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="40位小写十六进制"):
        load_model_source_registry(invalid_path)
    with pytest.raises(ValueError, match="组合未登记"):
        require_registered_model_reference(
            "openai/clip-vit-base-patch32",
            "0" * 40,
        )
    legacy_source = get_model_source("manojb_stable_diffusion_2_1_base")
    with pytest.raises(ValueError, match="用途组合未登记"):
        require_registered_model_reference(
            legacy_source.repository_id,
            legacy_source.revision,
            required_usage_role="primary_diffusion_model",
        )


@pytest.mark.quick
def test_legacy_mirror_records_unavailable_upstream_separately() -> None:
    """公开镜像不得被描述为原始上游仓库。"""

    source = get_model_source("manojb_stable_diffusion_2_1_base")

    assert source.repository_id == "Manojb/stable-diffusion-2-1-base"
    assert source.upstream_repository_id == "stabilityai/stable-diffusion-2-1-base"
    assert source.upstream_access_status == "unavailable"


@pytest.mark.quick
def test_openclip_source_registers_exact_checkpoint_file() -> None:
    """官方参考 OpenCLIP 来源必须同时固定 checkpoint 文件摘要与大小."""

    source = get_model_source("laion_clip_vit_g14")

    assert source.repository_id == "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
    assert source.revision == "4b0305adc6802b2632e11cbe6606a9bdd43d35c9"
    assert "official_reference_openclip_encoder" in source.usage_roles
    assert [item.to_dict() for item in source.required_files] == [
        {
            "path": "open_clip_pytorch_model.bin",
            "sha256": "6aac683f899159946bc4ca15228bb7016f3cbb1a2c51f365cba0b23923f344da",
            "size_bytes": 5467006745,
        }
    ]


@pytest.mark.quick
def test_model_source_registry_rejects_invalid_required_file_digest(tmp_path: Path) -> None:
    """文件级登记不得接受可疑路径或非精确 SHA-256."""

    payload = json.loads(MODEL_SOURCE_REGISTRY_PATH.read_text(encoding="utf-8"))
    payload["sources"]["laion_clip_vit_g14"]["required_files"][0]["sha256"] = "invalid"
    invalid_path = tmp_path / "model_source_registry.json"
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="64位小写 SHA-256"):
        load_model_source_registry(invalid_path)


@pytest.mark.quick
def test_sd35_pipeline_forwards_registered_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """SD3.5 加载边界必须把精确 revision 传给 from_pretrained."""

    captured: dict[str, object] = {}

    class FakeCuda:
        """提供 CUDA 可用性接口."""

        @staticmethod
        def is_available() -> bool:
            """返回测试所需的可用状态."""

            return True

    class FakeTorch:
        """提供加载器使用的最小 torch 接口."""

        cuda = FakeCuda()
        float16 = "float16"

    class FakePipeline:
        """记录 from_pretrained 参数而不下载模型."""

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> "FakePipeline":
            """保存模型标识和关键字参数."""

            captured["model_id"] = model_id
            captured.update(kwargs)
            return cls()

        def to(self, device_name: str) -> "FakePipeline":
            """记录目标设备并返回自身."""

            captured["device_name"] = device_name
            return self

        def set_progress_bar_config(self, disable: bool) -> None:
            """记录进度条配置."""

            captured["progress_disabled"] = disable

    monkeypatch.setattr(
        sd3_pipeline_runtime.repository_environment,
        "require_published_formal_execution_lock",
        lambda root: {"lock_digest": "fixture"},
    )
    monkeypatch.setattr(
        sd3_pipeline_runtime,
        "import_runtime_dependencies",
        lambda: (None, FakeTorch, None, FakePipeline),
    )
    monkeypatch.setattr(
        sd3_pipeline_runtime,
        "build_runtime_environment_report",
        lambda **kwargs: {"environment_decision": "pass"},
    )
    monkeypatch.setattr(sd3_pipeline_runtime, "flatten_environment_versions", lambda report: {})

    config = SemanticWatermarkRuntimeConfig()
    _, runtime_versions = sd3_pipeline_runtime.load_pipeline(config)

    assert captured["model_id"] == config.model_id
    assert captured["revision"] == config.model_revision
    assert runtime_versions["diffusion_model_source"]["revision"] == config.model_revision


@pytest.mark.quick
def test_clip_loader_forwards_registered_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLIP 加载边界必须把精确 revision 传给 from_pretrained."""

    captured: dict[str, object] = {}

    class FakeParameter:
        """提供冻结参数所需接口."""

        def requires_grad_(self, enabled: bool) -> None:
            """记录梯度开关."""

            captured["requires_grad"] = enabled

    class FakeVisionModel:
        """记录 CLIP 加载参数而不下载权重."""

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> "FakeVisionModel":
            """保存模型标识和关键字参数."""

            captured["model_id"] = model_id
            captured.update(kwargs)
            return cls()

        def to(self, device_name: str) -> "FakeVisionModel":
            """记录目标设备并返回自身."""

            captured["device_name"] = device_name
            return self

        def eval(self) -> None:
            """记录推理模式."""

            captured["eval"] = True

        def parameters(self) -> tuple[FakeParameter, ...]:
            """返回一个可冻结的测试参数."""

            return (FakeParameter(),)

    fake_transformers = SimpleNamespace(CLIPVisionModelWithProjection=FakeVisionModel)
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    source = get_model_source("openai_clip_vit_base_patch32")

    semantic_features.load_clip_vision_model(
        source.repository_id,
        source.revision,
        "cpu",
    )

    assert captured["model_id"] == source.repository_id
    assert captured["revision"] == source.revision
    assert captured["attn_implementation"] == "eager"
    assert captured["requires_grad"] is False
