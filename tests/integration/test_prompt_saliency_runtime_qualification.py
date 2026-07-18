"""资格化冻结 CLIP 图文 runtime 的真实模型、预处理与池化语义。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.runtime import repository_environment
from experiments.runtime.diffusion.prompt_saliency_model_loader import (
    load_prompt_saliency_clip_runtime,
)


pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


@pytest.fixture(scope="module")
def qualified_runtime() -> object:
    """只在真实锁、CUDA和本地精确快照齐备时构造资格化runtime。"""

    if not torch.cuda.is_available():
        raise RuntimeError("prompt saliency CLIP qualification requires CUDA")
    formal_lock = repository_environment.require_published_formal_execution_lock(
        ROOT
    )
    return load_prompt_saliency_clip_runtime(
        MODEL_ID,
        MODEL_REVISION,
        "cuda",
        local_files_only=True,
        verified_formal_execution_lock=formal_lock,
        repository_root=ROOT,
    )


def _nontrivial_rgb_image() -> torch.Tensor:
    height, width = 193, 317
    y = torch.linspace(0.0, 1.0, steps=height).view(1, 1, height, 1)
    x = torch.linspace(0.0, 1.0, steps=width).view(1, 1, 1, width)
    red = (0.15 + 0.65 * x + 0.20 * y).clamp(0.0, 1.0)
    green = (0.05 + 0.25 * x + 0.70 * y).clamp(0.0, 1.0)
    blue = (0.90 - 0.60 * x + 0.05 * y).clamp(0.0, 1.0)
    return torch.cat((red, green, blue), dim=1).to(dtype=torch.float32)


def _independently_normalize_features(value: torch.Tensor) -> torch.Tensor:
    resolved = value.float()
    assert torch.isfinite(resolved).all()
    norms = torch.linalg.vector_norm(resolved, dim=-1, keepdim=True)
    assert torch.isfinite(norms).all()
    assert torch.all(norms > 0.0)
    normalized = resolved / norms
    assert torch.isfinite(normalized).all()
    return normalized


def test_real_processor_range_bridge_matches_frozen_default_rescale(
    qualified_runtime: object,
) -> None:
    """已rescale的[0,1]路径必须匹配冻结默认的[0,255]路径。"""

    from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast

    runtime = qualified_runtime
    image = _nontrivial_rgb_image()
    bridged = runtime.prepare_image_pixels(image).cpu()
    processor = runtime._image_processor
    default_rescaled = processor(
        images=(image[0] * 255.0).contiguous(),
        input_data_format="channels_first",
        return_tensors="pt",
    )["pixel_values"]

    assert type(runtime._model) is CLIPModel
    assert type(processor) is CLIPImageProcessor
    assert type(runtime._tokenizer) is CLIPTokenizerFast
    assert processor.backend == "torchvision"
    assert processor.do_rescale is True
    assert processor.rescale_factor == 1.0 / 255.0
    torch.testing.assert_close(
        bridged,
        default_rescaled,
        rtol=0.0,
        atol=1.0e-6,
    )


def test_real_clip_patch_text_projection_and_prompt_counterfactual(
    qualified_runtime: object,
) -> None:
    """真实双塔必须产生49个patch投影和Prompt条件文本投影。"""

    runtime = qualified_runtime
    model = runtime._model
    image = _nontrivial_rgb_image()
    image_features = runtime.encode_image_patch_features(image)
    with torch.inference_mode():
        pixel_values = runtime.prepare_image_pixels(image)
        vision_outputs = model.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )
        expected_patch = model.visual_projection(
            model.vision_model.post_layernorm(
                vision_outputs.last_hidden_state[:, 1:]
            )
        )
    expected_patch = _independently_normalize_features(expected_patch)

    first_prompt = "a red geometric object on a plain field"
    first = runtime.encode_prompt_feature(first_prompt)
    tokenizer_inputs = runtime._tokenizer(
        first_prompt,
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    model_device = next(model.parameters()).device
    input_ids = tokenizer_inputs["input_ids"].to(model_device)
    attention_mask = tokenizer_inputs["attention_mask"].to(model_device)
    with torch.inference_mode():
        text_outputs = model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        expected_prompt = model.text_projection(text_outputs.pooler_output)
    expected_prompt = _independently_normalize_features(expected_prompt)

    second = runtime.encode_prompt_feature("a blue animal beside a green tree")

    assert image_features.shape == (1, 49, 512)
    assert image_features.dtype == torch.float32
    assert torch.isfinite(image_features).all()
    torch.testing.assert_close(
        torch.linalg.vector_norm(image_features, dim=-1),
        torch.ones((1, 49), device=image_features.device),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        image_features,
        expected_patch,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert first.shape == second.shape == (1, 512)
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    torch.testing.assert_close(
        first,
        expected_prompt,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert not torch.equal(first, second)
    assert runtime.model_identity_digest != ""


def test_real_legacy_eos_pooler_matches_standard_argmax_rule(
    qualified_runtime: object,
) -> None:
    """legacy eos=2必须由标准text tower按最高token位置完成pooling。"""

    runtime = qualified_runtime
    model = runtime._model
    tokenizer = runtime._tokenizer
    assert model.text_model.config._attn_implementation == "eager"
    assert model.vision_model.config._attn_implementation == "eager"
    assert model.text_model.config.eos_token_id == 2
    assert tokenizer.eos_token_id != model.text_model.config.eos_token_id

    encoded = runtime.tokenize_prompt(
        "an intricate copper sculpture under directional studio lighting"
    )
    with torch.inference_mode():
        output = model.text_model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            return_dict=True,
        )
    last_hidden = output.last_hidden_state
    positions = encoded["input_ids"].to(dtype=torch.int).argmax(dim=-1)
    expected = last_hidden[
        torch.arange(last_hidden.shape[0], device=last_hidden.device),
        positions,
    ]
    torch.testing.assert_close(output.pooler_output, expected, rtol=0.0, atol=0.0)
    assert not torch.equal(output.pooler_output, last_hidden[:, 0, :])
    assert not torch.equal(output.pooler_output, last_hidden.mean(dim=1))
