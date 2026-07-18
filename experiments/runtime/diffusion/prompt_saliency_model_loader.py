"""加载冻结 CLIP 图文显著性内部运行时。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.runtime import repository_environment
from experiments.runtime.model_sources import require_registered_model_reference
from main.core.digest import build_stable_digest
from main.methods.content.prompt_saliency_runtime import _PromptSaliencyClipRuntime


_MODEL_ID = "openai/clip-vit-base-patch32"
_MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
_DEPENDENCY_PROFILE_ID = "sd35_method_runtime_gpu"
_MODEL_IDENTITY_SCHEMA_VERSION = "prompt_saliency_clip_runtime_identity_v1"
_RANGE_BRIDGE = "decoded_rgb_0_1_already_rescaled_disable_second_rescale_v1"
_POOLING_RULE = "transformers_clip_legacy_eos2_argmax_eot_pooling_v1"
_EXPECTED_PACKAGE_VERSIONS = {
    "transformers": "5.12.1",
    "tokenizers": "0.22.2",
    "pillow": "11.3.0",
}
_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]


def _qualified_name(value: type[Any] | Any) -> str:
    resolved = value if isinstance(value, type) else type(value)
    return f"{resolved.__module__}.{resolved.__qualname__}"


def _required_digest(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_dependency_environment(
    report: Any,
    verified_formal_execution_lock: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise TypeError("runtime environment report must be a mapping")
    if report.get("dependency_environment_ready") is not True:
        blockers = report.get("dependency_readiness_blockers", [])
        raise RuntimeError(f"dependency environment not ready: {blockers}")
    if report.get("formal_execution_lock_ready") is not True:
        raise RuntimeError("formal execution lock is not ready")
    if report.get("dependency_profile_id") != _DEPENDENCY_PROFILE_ID:
        raise RuntimeError("dependency profile identity mismatch")
    try:
        injected_digest = verified_formal_execution_lock[
            "formal_execution_lock_digest"
        ]
    except KeyError as exc:
        raise ValueError("verified formal execution lock digest is missing") from exc
    report_digest = report.get("formal_execution_lock_digest")
    if report_digest != injected_digest:
        raise RuntimeError("formal execution lock digest mismatch")

    package_versions = report.get("package_versions")
    if not isinstance(package_versions, Mapping):
        raise RuntimeError("dependency package versions are missing")
    for package_name, expected_version in _EXPECTED_PACKAGE_VERSIONS.items():
        if package_versions.get(package_name) != expected_version:
            raise RuntimeError(f"dependency version mismatch: {package_name}")
    for digest_name in (
        "dependency_profile_digest",
        "dependency_profile_summary_digest",
        "complete_hash_lock_digest",
        "formal_execution_lock_digest",
    ):
        _required_digest(report.get(digest_name), digest_name)
    return dict(report)


def _validate_processor(
    processor: Any,
    processor_class: type[Any],
) -> dict[str, Any]:
    if type(processor) is not processor_class:
        raise RuntimeError("loaded image processor concrete class mismatch")
    if processor.backend != "torchvision":
        raise RuntimeError("loaded image processor backend must be torchvision")
    expected_attributes = {
        "do_resize": True,
        "size": {"shortest_edge": 224},
        "do_center_crop": True,
        "crop_size": {"height": 224, "width": 224},
        "do_rescale": True,
        "do_normalize": True,
        "do_convert_rgb": True,
    }
    for name, expected in expected_attributes.items():
        if getattr(processor, name, None) != expected:
            raise RuntimeError(f"loaded image processor config mismatch: {name}")
    if getattr(processor, "rescale_factor", None) != 1.0 / 255.0:
        raise RuntimeError("loaded image processor rescale factor mismatch")
    if list(getattr(processor, "image_mean", [])) != _IMAGE_MEAN:
        raise RuntimeError("loaded image processor mean mismatch")
    if list(getattr(processor, "image_std", [])) != _IMAGE_STD:
        raise RuntimeError("loaded image processor standard deviation mismatch")
    resample = getattr(processor, "resample", None)
    try:
        resample_value = int(resample)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("loaded image processor resample identity is invalid") from exc
    if resample_value != 3:
        raise RuntimeError("loaded image processor must use bicubic resampling")
    return {
        "processor_class": _qualified_name(processor),
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
            "image_mean": list(_IMAGE_MEAN),
            "image_std": list(_IMAGE_STD),
            "do_convert_rgb": True,
        },
        "call_protocol": {
            "input_layout": "single_chw_cpu_float32",
            "input_data_format": "channels_first",
            "return_tensors": "pt",
            "do_rescale": False,
            "range_bridge": _RANGE_BRIDGE,
        },
    }


def _validate_special_token(
    tokenizer: Any,
    token_name: str,
    expected_token: str,
    expected_id: int,
) -> dict[str, Any]:
    token = getattr(tokenizer, f"{token_name}_token", None)
    token_id = getattr(tokenizer, f"{token_name}_token_id", None)
    if token != expected_token or type(token_id) is not int or token_id != expected_id:
        raise RuntimeError(f"loaded tokenizer {token_name} identity mismatch")
    return {f"{token_name}_token": token, f"{token_name}_token_id": token_id}


def _validate_model_and_tokenizer(
    model: Any,
    model_class: type[Any],
    tokenizer: Any,
    tokenizer_class: type[Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(model) is not model_class:
        raise RuntimeError("loaded CLIP model concrete class mismatch")
    if type(tokenizer) is not tokenizer_class:
        raise RuntimeError("loaded CLIP tokenizer concrete class mismatch")
    text_model = getattr(model, "text_model", None)
    vision_model = getattr(model, "vision_model", None)
    text_config = getattr(text_model, "config", None)
    vision_config = getattr(vision_model, "config", None)
    if getattr(text_config, "_attn_implementation", None) != "eager":
        raise RuntimeError("CLIP text tower must execute eager attention")
    if getattr(vision_config, "_attn_implementation", None) != "eager":
        raise RuntimeError("CLIP vision tower must execute eager attention")
    if (
        getattr(text_config, "bos_token_id", None),
        getattr(text_config, "eos_token_id", None),
        getattr(text_config, "pad_token_id", None),
        getattr(text_config, "max_position_embeddings", None),
    ) != (0, 2, 1, 77):
        raise RuntimeError("CLIP text model token config mismatch")
    if (
        getattr(vision_config, "image_size", None),
        getattr(vision_config, "patch_size", None),
        getattr(getattr(model, "config", None), "projection_dim", None),
    ) != (224, 32, 512):
        raise RuntimeError("CLIP vision or projection config mismatch")
    if getattr(tokenizer, "model_max_length", None) != 77:
        raise RuntimeError("loaded tokenizer max length mismatch")

    special_tokens = {}
    special_tokens.update(
        _validate_special_token(
            tokenizer, "bos", "<|startoftext|>", 49406
        )
    )
    special_tokens.update(
        _validate_special_token(
            tokenizer, "eos", "<|endoftext|>", 49407
        )
    )
    special_tokens.update(
        _validate_special_token(
            tokenizer, "pad", "<|endoftext|>", 49407
        )
    )
    special_tokens["model_max_length"] = 77
    return (
        {
            "model_class": _qualified_name(model),
            "attention_implementation": {
                "requested": "eager",
                "observed": {
                    "text_model": "eager",
                    "vision_model": "eager",
                },
            },
        },
        {
            "tokenizer_class": _qualified_name(tokenizer),
            "max_length": 77,
            "padding": "max_length",
            "truncation": True,
            "model_text_config": {
                "bos_token_id": 0,
                "eos_token_id": 2,
                "pad_token_id": 1,
                "max_position_embeddings": 77,
            },
            "tokenizer_special_tokens": special_tokens,
            "pooling_rule": _POOLING_RULE,
        },
    )


def _freeze_model(model: Any, device_name: str, torch_module: Any) -> Any:
    placed = model.to(device_name)
    placed.eval()
    parameters = list(placed.parameters())
    if not parameters:
        raise RuntimeError("loaded CLIP model must expose parameters")
    for parameter in parameters:
        parameter.requires_grad_(False)
    if getattr(placed, "training", True) is not False:
        raise RuntimeError("loaded CLIP model must be in evaluation mode")
    if any(parameter.requires_grad for parameter in parameters):
        raise RuntimeError("loaded CLIP model parameters must be frozen")
    floating_parameters = [
        parameter for parameter in parameters if parameter.is_floating_point()
    ]
    if not floating_parameters or any(
        parameter.dtype != torch_module.float32
        for parameter in floating_parameters
    ):
        raise RuntimeError("loaded CLIP model parameters must use float32")
    return placed


def load_prompt_saliency_clip_runtime(
    model_id: str,
    model_revision: str,
    device_name: str,
    *,
    local_files_only: bool,
    verified_formal_execution_lock: Mapping[str, Any],
    repository_root: str | Path,
) -> _PromptSaliencyClipRuntime:
    """在 registry、依赖锁和具体类身份闭合后加载冻结 CLIP 双塔。"""

    if (model_id, model_revision) != (_MODEL_ID, _MODEL_REVISION):
        raise ValueError("prompt saliency CLIP model identity mismatch")
    if type(device_name) is not str or not device_name:
        raise ValueError("device_name must be a non-empty exact string")
    if local_files_only is not True:
        raise ValueError("prompt saliency model loading must be local-files-only")
    if not isinstance(verified_formal_execution_lock, Mapping):
        raise TypeError("verified_formal_execution_lock must be a mapping")

    model_source = require_registered_model_reference(
        model_id,
        model_revision,
        required_usage_role="semantic_condition_encoder",
    )
    if (
        model_source.repository_id != model_id
        or model_source.revision != model_revision
        or "semantic_condition_encoder" not in model_source.usage_roles
    ):
        raise RuntimeError("registered model source identity mismatch")

    import torch

    report = repository_environment.build_runtime_environment_report(
        _DEPENDENCY_PROFILE_ID,
        torch_module=torch,
        verified_formal_execution_lock=verified_formal_execution_lock,
        repository_root=repository_root,
    )
    environment = _validate_dependency_environment(
        report,
        verified_formal_execution_lock,
    )

    from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast

    model = CLIPModel.from_pretrained(
        model_id,
        revision=model_revision,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    processor = CLIPImageProcessor.from_pretrained(
        model_id,
        revision=model_revision,
        local_files_only=True,
    )
    tokenizer = CLIPTokenizerFast.from_pretrained(
        model_id,
        revision=model_revision,
        local_files_only=True,
    )

    image_preprocessing = _validate_processor(processor, CLIPImageProcessor)
    model_identity, text_preprocessing = _validate_model_and_tokenizer(
        model,
        CLIPModel,
        tokenizer,
        CLIPTokenizerFast,
    )
    model = _freeze_model(model, device_name, torch)

    package_versions = environment["package_versions"]
    payload = {
        "schema_version": _MODEL_IDENTITY_SCHEMA_VERSION,
        "model_source": {
            "repository_id": model_id,
            "revision": model_revision,
            "registered_source_digest": build_stable_digest(
                model_source.to_dict()
            ),
        },
        "dependency_environment": {
            "dependency_profile_id": _DEPENDENCY_PROFILE_ID,
            "dependency_profile_digest": environment[
                "dependency_profile_digest"
            ],
            "dependency_profile_summary_digest": environment[
                "dependency_profile_summary_digest"
            ],
            "complete_hash_lock_digest": environment[
                "complete_hash_lock_digest"
            ],
            "formal_execution_lock_digest": environment[
                "formal_execution_lock_digest"
            ],
            "package_versions": {
                name: package_versions[name]
                for name in ("transformers", "tokenizers", "pillow")
            },
        },
        "model": {
            "class": model_identity["model_class"],
            "dtype": "float32",
            "attention_implementation": model_identity[
                "attention_implementation"
            ],
        },
        "image_preprocessing": image_preprocessing,
        "text_preprocessing": text_preprocessing,
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
    model_identity_digest = build_stable_digest(payload)
    return _PromptSaliencyClipRuntime(
        model=model,
        image_processor=processor,
        tokenizer=tokenizer,
        device_name=device_name,
        model_identity_digest=model_identity_digest,
    )
