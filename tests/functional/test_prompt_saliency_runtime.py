"""验证冻结 CLIP 图文 runtime 与离线 loader 的轻量协议。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from experiments.runtime.diffusion import prompt_saliency_model_loader as loader
from main.core.digest import build_stable_digest
from main.methods.content.prompt_saliency_runtime import _PromptSaliencyClipRuntime


pytestmark = pytest.mark.unit

MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
FORMAL_LOCK_DIGEST = "a" * 64
PROFILE_DIGEST = "b" * 64
PROFILE_SUMMARY_DIGEST = "c" * 64
COMPLETE_LOCK_DIGEST = "d" * 64


class _FakeSource:
    repository_id = MODEL_ID
    revision = MODEL_REVISION
    usage_roles = ("semantic_condition_encoder",)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": "openai_clip_vit_base_patch32",
            "provider": "hugging_face_hub",
            "repository_id": MODEL_ID,
            "repository_type": "model",
            "revision": MODEL_REVISION,
            "source_url": f"https://huggingface.co/{MODEL_ID}",
            "revision_url": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
            "access_policy": "public",
            "usage_roles": ["semantic_condition_encoder"],
        }


class _FakeVisionTower:
    def __init__(self, attention: str | None = "eager") -> None:
        self.config = SimpleNamespace(
            _attn_implementation=attention,
            image_size=224,
            patch_size=32,
        )
        self.last_post_layernorm_input: torch.Tensor | None = None

    def __call__(self, *, pixel_values: torch.Tensor, return_dict: bool) -> Any:
        assert return_dict is True
        assert tuple(pixel_values.shape) == (1, 3, 224, 224)
        tokens = torch.arange(50 * 4, dtype=torch.float32).reshape(1, 50, 4) + 1.0
        return SimpleNamespace(last_hidden_state=tokens.to(pixel_values.device))

    def post_layernorm(self, value: torch.Tensor) -> torch.Tensor:
        self.last_post_layernorm_input = value.detach().clone()
        return value + 0.25


class _FakeTextTower:
    def __init__(self, attention: str | None = "eager") -> None:
        self.config = SimpleNamespace(
            _attn_implementation=attention,
            bos_token_id=0,
            eos_token_id=2,
            pad_token_id=1,
            max_position_embeddings=77,
        )
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        assert kwargs["return_dict"] is True
        prompt_code = kwargs["input_ids"][0, 1].float()
        pooled = torch.stack(
            (prompt_code + 1.0, prompt_code + 2.0, prompt_code + 3.0, prompt_code + 4.0)
        ).reshape(1, 4)
        misleading = torch.full((1, 77, 4), -1000.0, device=pooled.device)
        return SimpleNamespace(
            pooler_output=pooled.to(kwargs["input_ids"].device),
            last_hidden_state=misleading.to(kwargs["input_ids"].device),
        )


class _FakeCLIPModel:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    trace: list[str] = []
    text_attention: str | None = "eager"
    vision_attention: str | None = "eager"
    last_instance: "_FakeCLIPModel | None" = None

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            projection_dim=512,
            _attn_implementation="eager",
        )
        self.text_model = _FakeTextTower(self.text_attention)
        self.vision_model = _FakeVisionTower(self.vision_attention)
        self._parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.training = True
        self.placed_device: str | None = None

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "_FakeCLIPModel":
        cls.trace.append("model_load")
        cls.calls.append((args, kwargs))
        instance = cls()
        cls.last_instance = instance
        return instance

    def to(self, device_name: str) -> "_FakeCLIPModel":
        self.placed_device = device_name
        return self

    def eval(self) -> "_FakeCLIPModel":
        self.training = False
        return self

    def parameters(self) -> Any:
        return iter((self._parameter,))

    def visual_projection(self, value: torch.Tensor) -> torch.Tensor:
        return value.repeat_interleave(128, dim=-1)

    def text_projection(self, value: torch.Tensor) -> torch.Tensor:
        return value.repeat_interleave(128, dim=-1)


class _FakeCLIPImageProcessor:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    trace: list[str] = []
    backend_value = "torchvision"
    last_instance: "_FakeCLIPImageProcessor | None" = None

    def __init__(self) -> None:
        self.do_resize = True
        self.size = {"shortest_edge": 224}
        self.resample = 3
        self.do_center_crop = True
        self.crop_size = {"height": 224, "width": 224}
        self.do_rescale = True
        self.rescale_factor = 1.0 / 255.0
        self.do_normalize = True
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        self.image_std = [0.26862954, 0.26130258, 0.27577711]
        self.do_convert_rgb = True
        self.preprocess_calls: list[dict[str, Any]] = []
        self.output_override: Any = None

    @property
    def backend(self) -> str:
        return self.backend_value

    @classmethod
    def from_pretrained(
        cls, *args: Any, **kwargs: Any
    ) -> "_FakeCLIPImageProcessor":
        cls.trace.append("processor_load")
        cls.calls.append((args, kwargs))
        instance = cls()
        cls.last_instance = instance
        return instance

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.preprocess_calls.append(kwargs)
        if self.output_override is not None:
            return self.output_override
        image = kwargs["images"]
        base = image.mean().to(dtype=torch.float32)
        return {"pixel_values": torch.ones((1, 3, 224, 224)) * (base + 0.5)}


class _FakeCLIPTokenizerFast:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    trace: list[str] = []
    last_instance: "_FakeCLIPTokenizerFast | None" = None

    def __init__(self) -> None:
        self.model_max_length = 77
        self.bos_token = "<|startoftext|>"
        self.bos_token_id = 49406
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = 49407
        self.pad_token = "<|endoftext|>"
        self.pad_token_id = 49407
        self.tokenize_calls: list[tuple[Any, dict[str, Any]]] = []

    @classmethod
    def from_pretrained(
        cls, *args: Any, **kwargs: Any
    ) -> "_FakeCLIPTokenizerFast":
        cls.trace.append("tokenizer_load")
        cls.calls.append((args, kwargs))
        instance = cls()
        cls.last_instance = instance
        return instance

    def __call__(self, prompt: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.tokenize_calls.append((prompt, kwargs))
        code = sum(ord(character) for character in prompt) % 101 + 3
        input_ids = torch.full((1, 77), self.pad_token_id, dtype=torch.int64)
        input_ids[0, 0] = self.bos_token_id
        input_ids[0, 1] = code
        input_ids[0, 2] = self.eos_token_id
        attention_mask = torch.zeros((1, 77), dtype=torch.int64)
        attention_mask[0, :3] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _ready_environment() -> dict[str, Any]:
    return {
        "dependency_environment_ready": True,
        "dependency_readiness_blockers": [],
        "formal_execution_lock_ready": True,
        "dependency_profile_id": "sd35_method_runtime_gpu",
        "dependency_profile_digest": PROFILE_DIGEST,
        "dependency_profile_summary_digest": PROFILE_SUMMARY_DIGEST,
        "complete_hash_lock_digest": COMPLETE_LOCK_DIGEST,
        "formal_execution_lock_digest": FORMAL_LOCK_DIGEST,
        "package_versions": {
            "transformers": "5.12.1",
            "tokenizers": "0.22.2",
            "pillow": "11.3.0",
        },
    }


def _reset_fakes() -> None:
    _FakeCLIPModel.calls = []
    _FakeCLIPModel.trace = []
    _FakeCLIPModel.text_attention = "eager"
    _FakeCLIPModel.vision_attention = "eager"
    _FakeCLIPModel.last_instance = None
    _FakeCLIPImageProcessor.calls = []
    _FakeCLIPImageProcessor.trace = []
    _FakeCLIPImageProcessor.backend_value = "torchvision"
    _FakeCLIPImageProcessor.last_instance = None
    _FakeCLIPTokenizerFast.calls = []
    _FakeCLIPTokenizerFast.trace = []
    _FakeCLIPTokenizerFast.last_instance = None


def _load_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    environment: dict[str, Any] | None = None,
    trace: list[str] | None = None,
) -> _PromptSaliencyClipRuntime:
    _reset_fakes()
    shared_trace = [] if trace is None else trace
    _FakeCLIPModel.trace = shared_trace
    _FakeCLIPImageProcessor.trace = shared_trace
    _FakeCLIPTokenizerFast.trace = shared_trace
    module = ModuleType("transformers")
    module.CLIPModel = _FakeCLIPModel
    module.CLIPImageProcessor = _FakeCLIPImageProcessor
    module.CLIPTokenizerFast = _FakeCLIPTokenizerFast
    monkeypatch.setitem(sys.modules, "transformers", module)

    def registered(*args: Any, **kwargs: Any) -> _FakeSource:
        shared_trace.append("registry")
        assert args == (MODEL_ID, MODEL_REVISION)
        assert kwargs == {"required_usage_role": "semantic_condition_encoder"}
        return _FakeSource()

    def report(*args: Any, **kwargs: Any) -> dict[str, Any]:
        shared_trace.append("environment")
        assert args == ("sd35_method_runtime_gpu",)
        assert kwargs["torch_module"] is torch
        assert kwargs["verified_formal_execution_lock"] == {
            "formal_execution_lock_digest": FORMAL_LOCK_DIGEST
        }
        assert kwargs["repository_root"] == Path("/repository")
        return deepcopy(_ready_environment() if environment is None else environment)

    monkeypatch.setattr(loader, "require_registered_model_reference", registered)
    monkeypatch.setattr(
        loader.repository_environment,
        "build_runtime_environment_report",
        report,
    )
    return loader.load_prompt_saliency_clip_runtime(
        MODEL_ID,
        MODEL_REVISION,
        "cpu",
        local_files_only=True,
        verified_formal_execution_lock={
            "formal_execution_lock_digest": FORMAL_LOCK_DIGEST
        },
        repository_root=Path("/repository"),
    )


def _expected_identity_payload() -> dict[str, Any]:
    return {
        "schema_version": "prompt_saliency_clip_runtime_identity_v1",
        "model_source": {
            "repository_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "registered_source_digest": build_stable_digest(_FakeSource().to_dict()),
        },
        "dependency_environment": {
            "dependency_profile_id": "sd35_method_runtime_gpu",
            "dependency_profile_digest": PROFILE_DIGEST,
            "dependency_profile_summary_digest": PROFILE_SUMMARY_DIGEST,
            "complete_hash_lock_digest": COMPLETE_LOCK_DIGEST,
            "formal_execution_lock_digest": FORMAL_LOCK_DIGEST,
            "package_versions": {
                "transformers": "5.12.1",
                "tokenizers": "0.22.2",
                "pillow": "11.3.0",
            },
        },
        "model": {
            "class": f"{_FakeCLIPModel.__module__}.{_FakeCLIPModel.__qualname__}",
            "dtype": "float32",
            "attention_implementation": {
                "requested": "eager",
                "observed": {"text_model": "eager", "vision_model": "eager"},
            },
        },
        "image_preprocessing": {
            "processor_class": (
                f"{_FakeCLIPImageProcessor.__module__}."
                f"{_FakeCLIPImageProcessor.__qualname__}"
            ),
            "processor_backend": "torchvision",
            "loaded_config": {
                "do_resize": True,
                "size": {"shortest_edge": 224},
                "resample": "bicubic",
                "do_center_crop": True,
                "crop_size": {"height": 224, "width": 224},
                "do_rescale": True,
                "rescale_factor": 1.0 / 255.0,
                "do_normalize": True,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "do_convert_rgb": True,
            },
            "call_protocol": {
                "input_layout": "single_chw_cpu_float32",
                "input_data_format": "channels_first",
                "return_tensors": "pt",
                "do_rescale": False,
                "range_bridge": (
                    "decoded_rgb_0_1_already_rescaled_disable_second_rescale_v1"
                ),
            },
        },
        "text_preprocessing": {
            "tokenizer_class": (
                f"{_FakeCLIPTokenizerFast.__module__}."
                f"{_FakeCLIPTokenizerFast.__qualname__}"
            ),
            "max_length": 77,
            "padding": "max_length",
            "truncation": True,
            "model_text_config": {
                "bos_token_id": 0,
                "eos_token_id": 2,
                "pad_token_id": 1,
                "max_position_embeddings": 77,
            },
            "tokenizer_special_tokens": {
                "bos_token": "<|startoftext|>",
                "bos_token_id": 49406,
                "eos_token": "<|endoftext|>",
                "eos_token_id": 49407,
                "pad_token": "<|endoftext|>",
                "pad_token_id": 49407,
                "model_max_length": 77,
            },
            "pooling_rule": "transformers_clip_legacy_eos2_argmax_eot_pooling_v1",
        },
        "feature_protocol": {
            "vision_source": "vision_model_final_last_hidden_state",
            "class_token_index": 0,
            "patch_grid": [7, 7],
            "patch_count": 49,
            "patch_post_layernorm": "vision_model.post_layernorm_per_token",
            "vision_projection": "visual_projection",
            "text_source": "text_model.return_dict.pooler_output",
            "text_projection": "text_projection",
            "projection_dimension": 512,
            "normalization": "float32_finite_nonzero_l2_last_dimension",
        },
    }


def test_loader_is_lazy_and_uses_exact_v5121_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "transformers" not in loader.__dict__
    trace: list[str] = []
    runtime = _load_fake_runtime(monkeypatch, trace=trace)

    assert trace == [
        "registry",
        "environment",
        "model_load",
        "processor_load",
        "tokenizer_load",
    ]
    model_args, model_kwargs = _FakeCLIPModel.calls[0]
    assert model_args == (MODEL_ID,)
    assert model_kwargs == {
        "revision": MODEL_REVISION,
        "local_files_only": True,
        "dtype": torch.float32,
        "attn_implementation": "eager",
    }
    assert "torch_dtype" not in model_kwargs
    processor_args, processor_kwargs = _FakeCLIPImageProcessor.calls[0]
    assert processor_args == (MODEL_ID,)
    assert processor_kwargs == {
        "revision": MODEL_REVISION,
        "local_files_only": True,
    }
    assert "backend" not in processor_kwargs
    assert _FakeCLIPTokenizerFast.calls == [
        (
            (MODEL_ID,),
            {"revision": MODEL_REVISION, "local_files_only": True},
        )
    ]
    model = _FakeCLIPModel.last_instance
    assert model is not None
    assert model.placed_device == "cpu"
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert runtime.model_identity_digest == build_stable_digest(
        _expected_identity_payload()
    )


def test_range_bridge_passes_single_chw_cpu_float32_with_exact_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_fake_runtime(monkeypatch)
    processor = _FakeCLIPImageProcessor.last_instance
    assert processor is not None

    for shape in ((1, 3, 5, 7), (1, 3, 3, 3)):
        image = torch.linspace(0.0, 1.0, steps=torch.tensor(shape).prod().item())
        image = image.reshape(shape).to(dtype=torch.float64)
        pixels = runtime.prepare_image_pixels(image)
        call = processor.preprocess_calls[-1]
        assert set(call) == {
            "images",
            "input_data_format",
            "return_tensors",
            "do_rescale",
        }
        assert call["images"].shape == shape[1:]
        assert call["images"].device.type == "cpu"
        assert call["images"].dtype == torch.float32
        assert call["images"].is_contiguous()
        assert call["input_data_format"] == "channels_first"
        assert call["return_tensors"] == "pt"
        assert call["do_rescale"] is False
        assert pixels.shape == (1, 3, 224, 224)
        assert pixels.dtype == torch.float32


def test_runtime_builds_projected_patch_and_standard_pooler_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_fake_runtime(monkeypatch)
    image_features = runtime.encode_image_patch_features(
        torch.full((1, 3, 8, 11), 0.25)
    )
    first_prompt = runtime.encode_prompt_feature("a red square")
    second_prompt = runtime.encode_prompt_feature("a blue circle")

    model = _FakeCLIPModel.last_instance
    tokenizer = _FakeCLIPTokenizerFast.last_instance
    assert model is not None and tokenizer is not None
    post_input = model.vision_model.last_post_layernorm_input
    assert post_input is not None
    expected_tokens = torch.arange(50 * 4, dtype=torch.float32).reshape(1, 50, 4) + 1.0
    assert torch.equal(post_input.cpu(), expected_tokens[:, 1:, :])
    assert image_features.shape == (1, 49, 512)
    assert torch.allclose(
        torch.linalg.vector_norm(image_features, dim=-1),
        torch.ones((1, 49)),
    )
    assert first_prompt.shape == (1, 512)
    assert torch.linalg.vector_norm(first_prompt).item() == pytest.approx(1.0)
    assert not torch.equal(first_prompt, second_prompt)
    assert tokenizer.tokenize_calls[0][1] == {
        "max_length": 77,
        "padding": "max_length",
        "truncation": True,
        "return_tensors": "pt",
    }
    assert model.text_model.config.eos_token_id == 2
    assert tokenizer.eos_token_id == 49407
    assert model.text_model.calls[0]["return_dict"] is True


@pytest.mark.parametrize(
    ("text_attention", "vision_attention"),
    (("sdpa", "eager"), ("eager", "sdpa"), (None, "eager"), ("eager", None)),
)
def test_loader_rejects_non_eager_tower_even_when_top_level_is_eager(
    monkeypatch: pytest.MonkeyPatch,
    text_attention: str | None,
    vision_attention: str | None,
) -> None:
    _reset_fakes()
    _FakeCLIPModel.text_attention = text_attention
    _FakeCLIPModel.vision_attention = vision_attention
    module = ModuleType("transformers")
    module.CLIPModel = _FakeCLIPModel
    module.CLIPImageProcessor = _FakeCLIPImageProcessor
    module.CLIPTokenizerFast = _FakeCLIPTokenizerFast
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setattr(loader, "require_registered_model_reference", lambda *a, **k: _FakeSource())
    monkeypatch.setattr(
        loader.repository_environment,
        "build_runtime_environment_report",
        lambda *a, **k: _ready_environment(),
    )

    with pytest.raises(RuntimeError, match="tower must execute eager"):
        loader.load_prompt_saliency_clip_runtime(
            MODEL_ID,
            MODEL_REVISION,
            "cpu",
            local_files_only=True,
            verified_formal_execution_lock={
                "formal_execution_lock_digest": FORMAL_LOCK_DIGEST
            },
            repository_root="/repository",
        )
    assert _FakeCLIPModel.last_instance is not None
    assert _FakeCLIPModel.last_instance.config._attn_implementation == "eager"


def test_dependency_and_formal_lock_fail_before_transformers_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _ready_environment()
    environment["dependency_environment_ready"] = False
    environment["dependency_readiness_blockers"] = ["lock_mismatch"]
    trace: list[str] = []
    with pytest.raises(RuntimeError, match="dependency environment not ready"):
        _load_fake_runtime(monkeypatch, environment=environment, trace=trace)
    assert trace == ["registry", "environment"]

    environment = _ready_environment()
    environment["formal_execution_lock_digest"] = "e" * 64
    trace = []
    with pytest.raises(RuntimeError, match="formal execution lock digest mismatch"):
        _load_fake_runtime(monkeypatch, environment=environment, trace=trace)
    assert trace == ["registry", "environment"]


def test_loader_rejects_backend_and_concrete_processor_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fakes()
    _FakeCLIPImageProcessor.backend_value = "pil"
    module = ModuleType("transformers")
    module.CLIPModel = _FakeCLIPModel
    module.CLIPImageProcessor = _FakeCLIPImageProcessor
    module.CLIPTokenizerFast = _FakeCLIPTokenizerFast
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setattr(loader, "require_registered_model_reference", lambda *a, **k: _FakeSource())
    monkeypatch.setattr(
        loader.repository_environment,
        "build_runtime_environment_report",
        lambda *a, **k: _ready_environment(),
    )
    with pytest.raises(RuntimeError, match="backend must be torchvision"):
        loader.load_prompt_saliency_clip_runtime(
            MODEL_ID,
            MODEL_REVISION,
            "cpu",
            local_files_only=True,
            verified_formal_execution_lock={
                "formal_execution_lock_digest": FORMAL_LOCK_DIGEST
            },
            repository_root="/repository",
        )


def test_loader_rejects_processor_subclass_and_tokenizer_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ProcessorSubclass(_FakeCLIPImageProcessor):
        pass

    _reset_fakes()
    module = ModuleType("transformers")
    module.CLIPModel = _FakeCLIPModel
    module.CLIPImageProcessor = _FakeCLIPImageProcessor
    module.CLIPTokenizerFast = _FakeCLIPTokenizerFast
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setattr(loader, "require_registered_model_reference", lambda *a, **k: _FakeSource())
    monkeypatch.setattr(
        loader.repository_environment,
        "build_runtime_environment_report",
        lambda *a, **k: _ready_environment(),
    )
    monkeypatch.setattr(
        _FakeCLIPImageProcessor,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: ProcessorSubclass()),
    )
    with pytest.raises(RuntimeError, match="concrete class mismatch"):
        loader.load_prompt_saliency_clip_runtime(
            MODEL_ID,
            MODEL_REVISION,
            "cpu",
            local_files_only=True,
            verified_formal_execution_lock={
                "formal_execution_lock_digest": FORMAL_LOCK_DIGEST
            },
            repository_root="/repository",
        )

    monkeypatch.setattr(
        _FakeCLIPImageProcessor,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: cls()),
    )

    def drifted_tokenizer(cls: type[Any], *args: Any, **kwargs: Any) -> Any:
        instance = cls()
        instance.eos_token_id = 17
        return instance

    monkeypatch.setattr(
        _FakeCLIPTokenizerFast,
        "from_pretrained",
        classmethod(drifted_tokenizer),
    )
    with pytest.raises(RuntimeError, match="tokenizer eos identity mismatch"):
        loader.load_prompt_saliency_clip_runtime(
            MODEL_ID,
            MODEL_REVISION,
            "cpu",
            local_files_only=True,
            verified_formal_execution_lock={
                "formal_execution_lock_digest": FORMAL_LOCK_DIGEST
            },
            repository_root="/repository",
        )


def test_loader_missing_component_fails_before_any_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fakes()
    module = ModuleType("transformers")
    module.CLIPModel = _FakeCLIPModel
    module.CLIPImageProcessor = _FakeCLIPImageProcessor
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setattr(loader, "require_registered_model_reference", lambda *a, **k: _FakeSource())
    monkeypatch.setattr(
        loader.repository_environment,
        "build_runtime_environment_report",
        lambda *a, **k: _ready_environment(),
    )
    with pytest.raises(ImportError):
        loader.load_prompt_saliency_clip_runtime(
            MODEL_ID,
            MODEL_REVISION,
            "cpu",
            local_files_only=True,
            verified_formal_execution_lock={
                "formal_execution_lock_digest": FORMAL_LOCK_DIGEST
            },
            repository_root="/repository",
        )
    assert _FakeCLIPModel.calls == []
    assert _FakeCLIPImageProcessor.calls == []


def test_loader_preserves_local_snapshot_failure_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fakes()
    module = ModuleType("transformers")
    module.CLIPModel = _FakeCLIPModel
    module.CLIPImageProcessor = _FakeCLIPImageProcessor
    module.CLIPTokenizerFast = _FakeCLIPTokenizerFast
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setattr(loader, "require_registered_model_reference", lambda *a, **k: _FakeSource())
    monkeypatch.setattr(
        loader.repository_environment,
        "build_runtime_environment_report",
        lambda *a, **k: _ready_environment(),
    )

    def missing_snapshot(cls: type[Any], *args: Any, **kwargs: Any) -> Any:
        assert kwargs["local_files_only"] is True
        raise OSError("exact local snapshot missing")

    monkeypatch.setattr(
        _FakeCLIPModel,
        "from_pretrained",
        classmethod(missing_snapshot),
    )
    with pytest.raises(OSError, match="exact local snapshot missing"):
        loader.load_prompt_saliency_clip_runtime(
            MODEL_ID,
            MODEL_REVISION,
            "cpu",
            local_files_only=True,
            verified_formal_execution_lock={
                "formal_execution_lock_digest": FORMAL_LOCK_DIGEST
            },
            repository_root="/repository",
        )
    assert _FakeCLIPImageProcessor.calls == []
    assert _FakeCLIPTokenizerFast.calls == []


@pytest.mark.parametrize(
    "invalid",
    [
        "image",
        torch.zeros((1, 3, 2, 2), dtype=torch.int64),
        torch.zeros((2, 3, 2, 2)),
        torch.zeros((1, 1, 2, 2)),
        torch.full((1, 3, 2, 2), float("nan")),
        torch.full((1, 3, 2, 2), -0.1),
        torch.full((1, 3, 2, 2), 1.1),
    ],
)
def test_runtime_rejects_invalid_decoded_images(
    monkeypatch: pytest.MonkeyPatch,
    invalid: Any,
) -> None:
    runtime = _load_fake_runtime(monkeypatch)
    with pytest.raises((TypeError, ValueError)):
        runtime.prepare_image_pixels(invalid)


def test_runtime_rejects_invalid_processor_and_feature_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_fake_runtime(monkeypatch)
    processor = _FakeCLIPImageProcessor.last_instance
    model = _FakeCLIPModel.last_instance
    assert processor is not None and model is not None

    processor.output_override = {
        "pixel_values": torch.zeros((1, 3, 223, 224), dtype=torch.float32)
    }
    with pytest.raises(ValueError, match="shape"):
        runtime.prepare_image_pixels(torch.zeros((1, 3, 4, 5)))
    processor.output_override = {
        "pixel_values": torch.full((1, 3, 224, 224), float("nan"))
    }
    with pytest.raises(ValueError, match="finite"):
        runtime.prepare_image_pixels(torch.zeros((1, 3, 4, 5)))

    processor.output_override = None
    original_projection = model.visual_projection
    model.visual_projection = lambda value: torch.zeros((1, 49, 512))
    with pytest.raises(ValueError, match="non-zero L2"):
        runtime.encode_image_patch_features(torch.zeros((1, 3, 4, 5)))
    model.visual_projection = original_projection


def test_loader_rejects_network_enabled_mode_before_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def registered(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        return _FakeSource()

    monkeypatch.setattr(loader, "require_registered_model_reference", registered)
    with pytest.raises(ValueError, match="local-files-only"):
        loader.load_prompt_saliency_clip_runtime(
            MODEL_ID,
            MODEL_REVISION,
            "cpu",
            local_files_only=False,
            verified_formal_execution_lock={
                "formal_execution_lock_digest": FORMAL_LOCK_DIGEST
            },
            repository_root="/repository",
        )
    assert called is False
