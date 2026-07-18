"""验证 Prompt 条件语义显著图的纯 Tensor 公式与协议边界。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from main.core.digest import tensor_content_sha256
from main.methods.content import (
    SemanticSaliencyResult,
    build_prompt_conditioned_semantic_saliency,
)


pytestmark = pytest.mark.unit
MODEL_DIGEST = "a" * 64


def _basis(index: int) -> torch.Tensor:
    value = torch.zeros(512, dtype=torch.float32)
    value[index] = 1.0
    return value


def _patches(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(1, 1, 512).repeat(1, 49, 1)


class _StaticRuntime:
    def __init__(
        self,
        image_features: Any,
        prompt_feature: Any,
        *,
        model_identity_digest: str = MODEL_DIGEST,
    ) -> None:
        self.image_features = image_features
        self.prompt_feature = prompt_feature
        self.model_identity_digest = model_identity_digest
        self.image_calls = 0
        self.prompt_calls = 0

    def encode_image_patch_features(self, image: Any) -> Any:
        self.image_calls += 1
        return self.image_features

    def encode_prompt_feature(self, prompt: Any) -> Any:
        self.prompt_calls += 1
        return self.prompt_feature


class _CounterfactualRuntime:
    model_identity_digest = MODEL_DIGEST

    def encode_image_patch_features(self, image: torch.Tensor) -> torch.Tensor:
        selected = _basis(0) if image.mean().item() < 0.5 else _basis(1)
        return _patches(selected)

    def encode_prompt_feature(self, prompt: str) -> torch.Tensor:
        selected = _basis(0) if prompt == "red object" else _basis(1)
        return selected.reshape(1, 512)


def _valid_image(value: float = 0.25) -> torch.Tensor:
    return torch.full((1, 3, 5, 7), value, dtype=torch.float32)


def test_semantic_saliency_result_is_frozen_with_exact_fields() -> None:
    assert tuple(field.name for field in fields(SemanticSaliencyResult)) == (
        "saliency_map",
        "patch_relevance",
        "image_feature_digest",
        "prompt_feature_digest",
        "saliency_map_digest",
        "model_identity_digest",
    )
    runtime = _StaticRuntime(_patches(_basis(0)), _basis(0).reshape(1, 512))
    result = build_prompt_conditioned_semantic_saliency(
        _valid_image(), "red object", runtime
    )
    with pytest.raises(FrozenInstanceError):
        result.model_identity_digest = "b" * 64  # type: ignore[misc]


def test_saliency_matches_raw_patch_cosine_and_row_major_grid() -> None:
    image_features = _patches(_basis(1))
    image_features[:, 0, :] = _basis(0)
    image_features[:, 1, :] = _basis(1)
    image_features[:, 2, :] = -_basis(0)
    runtime = _StaticRuntime(image_features, _basis(0).reshape(1, 512))

    result = build_prompt_conditioned_semantic_saliency(
        _valid_image(), "red object", runtime
    )

    expected_relevance = torch.zeros((1, 49), dtype=torch.float32)
    expected_relevance[0, 0] = 1.0
    expected_relevance[0, 2] = -1.0
    expected_map = ((expected_relevance.reshape(1, 1, 7, 7) + 1.0) / 2.0).clamp(
        0.0, 1.0
    )
    assert torch.equal(result.patch_relevance, expected_relevance)
    assert torch.equal(result.saliency_map, expected_map)
    assert result.saliency_map[0, 0, 0, 0].item() == 1.0
    assert result.saliency_map[0, 0, 0, 1].item() == 0.5
    assert result.saliency_map[0, 0, 0, 2].item() == 0.0
    assert result.patch_relevance.shape == (1, 49)
    assert result.saliency_map.shape == (1, 1, 7, 7)
    assert result.patch_relevance.dtype == torch.float32
    assert result.saliency_map.dtype == torch.float32
    assert result.patch_relevance.device == image_features.device
    assert result.saliency_map.device == image_features.device
    assert torch.isfinite(result.patch_relevance).all()
    assert torch.isfinite(result.saliency_map).all()
    assert torch.all((result.saliency_map >= 0.0) & (result.saliency_map <= 1.0))
    assert runtime.image_calls == runtime.prompt_calls == 1


def test_near_unit_features_preserve_raw_relevance_above_one_and_only_clip_map(
) -> None:
    scaled = _basis(0) * torch.tensor(1.00001, dtype=torch.float32)
    runtime = _StaticRuntime(_patches(scaled), scaled.reshape(1, 512))

    result = build_prompt_conditioned_semantic_saliency(
        _valid_image(), "red object", runtime
    )

    expected = torch.sum(scaled * scaled)
    assert torch.linalg.vector_norm(scaled).item() == pytest.approx(
        1.0000095367, rel=1.0e-6
    )
    assert expected.item() > 1.0
    assert torch.equal(
        result.patch_relevance,
        expected.reshape(1, 1).repeat(1, 49),
    )
    assert torch.all(result.patch_relevance > 1.0)
    assert torch.equal(result.saliency_map, torch.ones((1, 1, 7, 7)))


def test_saliency_digests_keep_feature_map_and_model_responsibilities_separate(
) -> None:
    first = _StaticRuntime(
        _patches(_basis(0)),
        _basis(0).reshape(1, 512),
        model_identity_digest="a" * 64,
    )
    second = _StaticRuntime(
        _patches(_basis(1)),
        _basis(1).reshape(1, 512),
        model_identity_digest="b" * 64,
    )
    model_only = _StaticRuntime(
        first.image_features,
        first.prompt_feature,
        model_identity_digest="c" * 64,
    )

    first_result = build_prompt_conditioned_semantic_saliency(
        _valid_image(), "red object", first
    )
    second_result = build_prompt_conditioned_semantic_saliency(
        _valid_image(), "red object", second
    )
    model_only_result = build_prompt_conditioned_semantic_saliency(
        _valid_image(), "red object", model_only
    )

    assert first_result.image_feature_digest == tensor_content_sha256(
        first.image_features
    )
    assert first_result.prompt_feature_digest == tensor_content_sha256(
        first.prompt_feature
    )
    assert first_result.saliency_map_digest == tensor_content_sha256(
        first_result.saliency_map
    )
    assert first_result.model_identity_digest == "a" * 64
    assert torch.equal(first_result.saliency_map, second_result.saliency_map)
    assert first_result.saliency_map_digest == second_result.saliency_map_digest
    assert first_result.image_feature_digest != second_result.image_feature_digest
    assert first_result.prompt_feature_digest != second_result.prompt_feature_digest
    assert first_result.model_identity_digest != second_result.model_identity_digest
    assert first_result.image_feature_digest == model_only_result.image_feature_digest
    assert first_result.prompt_feature_digest == model_only_result.prompt_feature_digest
    assert first_result.saliency_map_digest == model_only_result.saliency_map_digest
    assert first_result.model_identity_digest != model_only_result.model_identity_digest


def test_image_and_prompt_counterfactuals_change_spatial_relevance() -> None:
    runtime = _CounterfactualRuntime()
    baseline = build_prompt_conditioned_semantic_saliency(
        _valid_image(0.25), "red object", runtime
    )
    prompt_changed = build_prompt_conditioned_semantic_saliency(
        _valid_image(0.25), "blue object", runtime
    )
    image_changed = build_prompt_conditioned_semantic_saliency(
        _valid_image(0.75), "red object", runtime
    )

    assert not torch.equal(baseline.patch_relevance, prompt_changed.patch_relevance)
    assert not torch.equal(baseline.saliency_map, prompt_changed.saliency_map)
    assert not torch.equal(baseline.patch_relevance, image_changed.patch_relevance)
    assert not torch.equal(baseline.saliency_map, image_changed.saliency_map)


@pytest.mark.parametrize(
    "invalid_image",
    [
        "image",
        torch.zeros((1, 3, 2, 2), dtype=torch.int64),
        torch.zeros((1, 3, 2, 2), dtype=torch.complex64),
        torch.zeros((2, 3, 2, 2), dtype=torch.float32),
        torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        torch.full((1, 3, 2, 2), float("nan")),
        torch.full((1, 3, 2, 2), -0.1),
        torch.full((1, 3, 2, 2), 1.1),
    ],
)
def test_static_input_gates_fail_before_runtime_forward(invalid_image: Any) -> None:
    runtime = _StaticRuntime(_patches(_basis(0)), _basis(0).reshape(1, 512))
    with pytest.raises((TypeError, ValueError)):
        build_prompt_conditioned_semantic_saliency(
            invalid_image, "red object", runtime
        )
    assert runtime.image_calls == runtime.prompt_calls == 0

    with pytest.raises(TypeError, match="exact string"):
        build_prompt_conditioned_semantic_saliency(
            _valid_image(), 7, runtime  # type: ignore[arg-type]
        )
    assert runtime.image_calls == runtime.prompt_calls == 0


@pytest.mark.parametrize(
    "runtime",
    [
        SimpleNamespace(model_identity_digest=MODEL_DIGEST),
        SimpleNamespace(
            encode_image_patch_features=None,
            encode_prompt_feature=lambda prompt: None,
            model_identity_digest=MODEL_DIGEST,
        ),
        SimpleNamespace(
            encode_image_patch_features=lambda image: None,
            encode_prompt_feature=lambda prompt: None,
            model_identity_digest="not-a-digest",
        ),
    ],
)
def test_runtime_capability_gates_fail_before_forward(runtime: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_prompt_conditioned_semantic_saliency(
            _valid_image(), "red object", runtime
        )


def test_invalid_model_identity_fails_before_runtime_forward() -> None:
    runtime = _StaticRuntime(
        _patches(_basis(0)),
        _basis(0).reshape(1, 512),
        model_identity_digest="not-a-digest",
    )
    with pytest.raises(ValueError, match="model_identity_digest"):
        build_prompt_conditioned_semantic_saliency(
            _valid_image(), "red object", runtime
        )
    assert runtime.image_calls == runtime.prompt_calls == 0


@pytest.mark.parametrize(
    "invalid_image_features",
    [
        None,
        torch.zeros((1, 49, 512), dtype=torch.float64),
        torch.zeros((1, 49, 512), dtype=torch.bool),
        torch.zeros((1, 49, 512), dtype=torch.complex64),
        torch.zeros((1, 48, 512), dtype=torch.float32),
        torch.full((1, 49, 512), float("nan")),
        torch.zeros((1, 49, 512), dtype=torch.float32),
        _patches(_basis(0) * 1.01),
        torch.full(
            (1, 49, 512),
            torch.finfo(torch.float32).max,
            dtype=torch.float32,
        ),
    ],
)
def test_image_feature_outputs_fail_closed_before_text_forward(
    invalid_image_features: Any,
) -> None:
    runtime = _StaticRuntime(
        invalid_image_features,
        _basis(0).reshape(1, 512),
    )
    with pytest.raises((TypeError, ValueError)):
        build_prompt_conditioned_semantic_saliency(
            _valid_image(), "red object", runtime
        )
    assert runtime.image_calls == 1
    assert runtime.prompt_calls == 0


@pytest.mark.parametrize(
    "invalid_prompt_feature",
    [
        None,
        torch.zeros((1, 512), dtype=torch.float64),
        torch.zeros((1, 512), dtype=torch.bool),
        torch.zeros((1, 512), dtype=torch.complex64),
        torch.zeros((512,), dtype=torch.float32),
        torch.full((1, 512), float("inf")),
        torch.zeros((1, 512), dtype=torch.float32),
        (_basis(0) * 1.01).reshape(1, 512),
    ],
)
def test_prompt_feature_outputs_fail_closed(invalid_prompt_feature: Any) -> None:
    runtime = _StaticRuntime(_patches(_basis(0)), invalid_prompt_feature)
    with pytest.raises((TypeError, ValueError)):
        build_prompt_conditioned_semantic_saliency(
            _valid_image(), "red object", runtime
        )
    assert runtime.image_calls == runtime.prompt_calls == 1


def test_feature_device_mismatch_fails_before_reading_prompt_contents() -> None:
    prompt_feature = torch.empty((1, 512), dtype=torch.float32, device="meta")
    runtime = _StaticRuntime(_patches(_basis(0)), prompt_feature)
    with pytest.raises(ValueError, match="same device"):
        build_prompt_conditioned_semantic_saliency(
            _valid_image(), "red object", runtime
        )
