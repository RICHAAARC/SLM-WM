"""验证四项内容观测进入 latent 路由的单次运行编排。"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_type_hints

import pytest
import torch
import torch.nn.functional as functional

import main.methods as methods
import main.methods.content as content
import main.methods.content.runtime_adapter as adapter_module
from main.core.digest import tensor_content_sha256
from main.core.keyed_prg import KEYED_PRG_VERSION
from main.methods.content.latent_response import (
    LatentResponseResult,
    build_adjacent_latent_response_map,
)
from main.methods.content.local_sensitivity import (
    LocalSensitivityResult,
    build_public_probe_local_sensitivity_map,
)
from main.methods.content.routing import (
    ContentRoutingResult,
    _route_content_observations_to_latent,
)
from main.methods.content.runtime_adapter import (
    ContentObservationRuntimeResult,
    build_content_observation_routing,
)
from main.methods.content.saliency import (
    SemanticSaliencyResult,
    build_prompt_conditioned_semantic_saliency,
)
from main.methods.content.texture import TextureResult, build_texture_complexity_map


pytestmark = pytest.mark.unit

_MODEL_DIGEST = "a" * 64
_PUBLIC_PROBE_IDENTITY = {
    "prg_version": KEYED_PRG_VERSION,
    "key_material": "semantic_saliency_dual_chain_public_probe_v1",
    "domain_fields": {
        "purpose": "local_sensitivity_public_probe",
        "model_revision": "runtime-adapter-test-revision",
        "probe_version": "v1",
    },
}


def _basis(index: int) -> torch.Tensor:
    value = torch.zeros(512, dtype=torch.float32)
    value[index] = 1.0
    return value


class _RecordingSaliencyRuntime:
    model_identity_digest = _MODEL_DIGEST

    def __init__(self) -> None:
        self.image_inputs: list[torch.Tensor] = []
        self.prompt_inputs: list[str] = []

    def encode_image_patch_features(self, image: torch.Tensor) -> torch.Tensor:
        self.image_inputs.append(image)
        first = _basis(0) if float(image.mean().item()) < 0.6 else _basis(1)
        second = _basis(1) if torch.equal(first, _basis(0)) else _basis(0)
        features = second.reshape(1, 1, 512).repeat(1, 49, 1)
        features[:, ::2, :] = first
        return features

    def encode_prompt_feature(self, prompt: str) -> torch.Tensor:
        self.prompt_inputs.append(prompt)
        feature = _basis(0) if prompt == "red object" else _basis(1)
        return feature.reshape(1, 512)


def _decoder_formula(
    latent: torch.Tensor,
    output_shape: tuple[int, int] = (5, 9),
) -> torch.Tensor:
    signal = functional.interpolate(
        latent[:, :1].float(),
        size=output_shape,
        mode="bilinear",
        align_corners=False,
    )
    channels = torch.cat((signal, 0.5 * signal, -0.25 * signal), dim=1)
    return (0.5 + 0.08 * channels).clamp(0.0, 1.0)


class _RecordingDecoder:
    def __init__(self, output_shape: tuple[int, int] = (5, 9)) -> None:
        self.output_shape = output_shape
        self.inputs: list[torch.Tensor] = []

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        self.inputs.append(latent)
        return _decoder_formula(latent, self.output_shape)


def _latents() -> tuple[torch.Tensor, torch.Tensor]:
    previous = torch.tensor(
        [
            [
                [[-0.5, -0.2, 0.1, 0.3], [0.4, 0.2, -0.1, -0.3]],
                [[0.3, -0.4, 0.2, -0.2], [0.1, 0.5, -0.3, 0.4]],
            ]
        ],
        dtype=torch.float64,
    )
    current = previous + torch.tensor(
        [
            [
                [[0.04, -0.01, 0.03, -0.02], [0.01, -0.04, 0.02, 0.03]],
                [[-0.02, 0.03, -0.01, 0.04], [0.02, -0.03, 0.01, -0.04]],
            ]
        ],
        dtype=torch.float64,
    )
    return previous, current


def _inputs() -> dict[str, Any]:
    previous, current = _latents()
    return {
        "previous_scheduler_latent": previous,
        "current_scheduler_latent": current,
        "decoded_current_image": _decoder_formula(current),
        "prompt": "red object",
        "saliency_runtime": _RecordingSaliencyRuntime(),
        "vae_decoder": _RecordingDecoder(),
        "public_probe_identity": {
            "prg_version": _PUBLIC_PROBE_IDENTITY["prg_version"],
            "key_material": _PUBLIC_PROBE_IDENTITY["key_material"],
            "domain_fields": dict(_PUBLIC_PROBE_IDENTITY["domain_fields"]),
        },
        "reference_gradient": 2.0,
        "reference_response": 0.25,
        "reference_sensitivity": 1.0,
    }


def _assert_routing_equal(
    actual: ContentRoutingResult,
    expected: ContentRoutingResult,
) -> None:
    assert torch.equal(actual.writable_capacity_map, expected.writable_capacity_map)
    assert torch.equal(actual.lf_mask, expected.lf_mask)
    assert torch.equal(actual.hf_tail_mask, expected.hf_tail_mask)
    assert actual.routing_identity_digest == expected.routing_identity_digest


def _assert_result_fields_equal(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected)
    for field in fields(actual):
        actual_value = getattr(actual, field.name)
        expected_value = getattr(expected, field.name)
        if isinstance(actual_value, torch.Tensor):
            assert torch.equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def test_public_contract_is_frozen_exact_keyword_only_and_formally_exported(
) -> None:
    assert adapter_module.__all__ == [
        "ContentObservationRuntimeResult",
        "build_content_observation_routing",
    ]
    assert tuple(field.name for field in fields(ContentObservationRuntimeResult)) == (
        "semantic_saliency",
        "texture",
        "latent_response",
        "local_sensitivity",
        "routing",
    )
    hints = get_type_hints(ContentObservationRuntimeResult)
    assert hints == {
        "semantic_saliency": SemanticSaliencyResult,
        "texture": TextureResult,
        "latent_response": LatentResponseResult,
        "local_sensitivity": LocalSensitivityResult,
        "routing": ContentRoutingResult,
    }
    signature = inspect.signature(build_content_observation_routing)
    assert tuple(signature.parameters) == (
        "previous_scheduler_latent",
        "current_scheduler_latent",
        "decoded_current_image",
        "prompt",
        "saliency_runtime",
        "vae_decoder",
        "public_probe_identity",
        "reference_gradient",
        "reference_response",
        "reference_sensitivity",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert not hasattr(content, "ContentObservationRuntimeResult")
    assert not hasattr(content, "build_content_observation_routing")
    assert methods.ContentObservationRuntimeResult is ContentObservationRuntimeResult
    assert methods.build_content_observation_routing is build_content_observation_routing

    result = build_content_observation_routing(**_inputs())
    with pytest.raises(FrozenInstanceError):
        result.routing = result.routing  # type: ignore[misc]


def test_calls_each_observation_once_in_fixed_order_and_reuses_x10(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    previous = inputs["previous_scheduler_latent"]
    current = inputs["current_scheduler_latent"]
    image = inputs["decoded_current_image"]
    decoder = inputs["vae_decoder"]
    calls: list[str] = []

    original_response = adapter_module.build_adjacent_latent_response_map
    original_texture = adapter_module.build_texture_complexity_map
    original_saliency = adapter_module.build_prompt_conditioned_semantic_saliency
    original_sensitivity = adapter_module.build_public_probe_local_sensitivity_map
    original_route = adapter_module._route_content_observations_to_latent

    def response_spy(*args: Any) -> LatentResponseResult:
        calls.append("R")
        assert args[0] is previous and args[1] is current
        return original_response(*args)

    def texture_spy(*args: Any) -> TextureResult:
        calls.append("T")
        assert args[0] is image
        return original_texture(*args)

    def saliency_spy(*args: Any) -> SemanticSaliencyResult:
        calls.append("S")
        assert args[0] is image
        assert args[1] == "red object"
        return original_saliency(*args)

    def sensitivity_spy(*args: Any) -> LocalSensitivityResult:
        calls.append("Q")
        assert args[0] is current and args[1] is image and args[2] is decoder
        return original_sensitivity(*args)

    def route_spy(*args: Any) -> ContentRoutingResult:
        calls.append("route")
        return original_route(*args)

    monkeypatch.setattr(
        adapter_module, "build_adjacent_latent_response_map", response_spy
    )
    monkeypatch.setattr(adapter_module, "build_texture_complexity_map", texture_spy)
    monkeypatch.setattr(
        adapter_module,
        "build_prompt_conditioned_semantic_saliency",
        saliency_spy,
    )
    monkeypatch.setattr(
        adapter_module,
        "build_public_probe_local_sensitivity_map",
        sensitivity_spy,
    )
    monkeypatch.setattr(
        adapter_module, "_route_content_observations_to_latent", route_spy
    )

    result = build_content_observation_routing(**inputs)

    assert calls == ["R", "T", "S", "Q", "route"]
    runtime = inputs["saliency_runtime"]
    assert runtime.image_inputs == [image]
    assert runtime.prompt_inputs == ["red object"]
    assert len(decoder.inputs) == 1
    assert result.local_sensitivity.perturbed_image_digest == tensor_content_sha256(
        _decoder_formula(decoder.inputs[0])
    )
    assert len(decoder.inputs) == 1


def test_matches_independent_existing_primitive_composition_and_preserves_inputs(
) -> None:
    inputs = _inputs()
    sources = (
        inputs["previous_scheduler_latent"],
        inputs["current_scheduler_latent"],
        inputs["decoded_current_image"],
    )
    snapshots = tuple(value.detach().clone() for value in sources)
    strides = tuple(value.stride() for value in sources)
    source_digests = tuple(tensor_content_sha256(value) for value in sources)

    actual = build_content_observation_routing(**inputs)

    expected_runtime = _RecordingSaliencyRuntime()
    expected_decoder = _RecordingDecoder()
    expected_response = build_adjacent_latent_response_map(
        sources[0], sources[1], inputs["reference_response"]
    )
    expected_texture = build_texture_complexity_map(
        sources[2], inputs["reference_gradient"]
    )
    expected_saliency = build_prompt_conditioned_semantic_saliency(
        sources[2], inputs["prompt"], expected_runtime
    )
    expected_sensitivity = build_public_probe_local_sensitivity_map(
        sources[1],
        sources[2],
        expected_decoder,
        inputs["public_probe_identity"],
        inputs["reference_sensitivity"],
    )
    expected_routing = _route_content_observations_to_latent(
        expected_saliency.saliency_map,
        expected_texture.texture_map,
        expected_response.response_map,
        expected_sensitivity.local_sensitivity_map,
    )

    assert torch.equal(actual.semantic_saliency.saliency_map, expected_saliency.saliency_map)
    assert torch.equal(actual.semantic_saliency.patch_relevance, expected_saliency.patch_relevance)
    _assert_result_fields_equal(actual.semantic_saliency, expected_saliency)
    assert torch.equal(actual.texture.texture_map, expected_texture.texture_map)
    assert actual.texture.reference_gradient == expected_texture.reference_gradient
    assert actual.texture.texture_map_digest == expected_texture.texture_map_digest
    assert torch.equal(actual.latent_response.response_map, expected_response.response_map)
    _assert_result_fields_equal(actual.latent_response, expected_response)
    assert torch.equal(
        actual.local_sensitivity.local_sensitivity_map,
        expected_sensitivity.local_sensitivity_map,
    )
    _assert_result_fields_equal(actual.local_sensitivity, expected_sensitivity)
    _assert_routing_equal(actual.routing, expected_routing)
    assert actual.routing.writable_capacity_map.shape == (1, 1, 2, 4)
    assert actual.routing.writable_capacity_map.dtype == torch.float32
    assert tuple(tensor_content_sha256(value) for value in sources) == source_digests
    assert tuple(value.stride() for value in sources) == strides
    assert all(torch.equal(value, snapshot) for value, snapshot in zip(sources, snapshots, strict=True))


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("prompt", 1),
        ("reference_gradient", 1),
        ("reference_response", True),
        ("reference_sensitivity", float("inf")),
        ("reference_sensitivity", 0.0),
        ("vae_decoder", None),
        (
            "saliency_runtime",
            SimpleNamespace(
                encode_image_patch_features=None,
                encode_prompt_feature=lambda prompt: prompt,
            ),
        ),
        (
            "saliency_runtime",
            SimpleNamespace(
                encode_image_patch_features=lambda image: image,
                encode_prompt_feature=None,
            ),
        ),
        (
            "previous_scheduler_latent",
            torch.zeros((2, 2, 2, 4), dtype=torch.float32),
        ),
        (
            "current_scheduler_latent",
            torch.zeros((1, 2, 3, 4), dtype=torch.float32),
        ),
        (
            "decoded_current_image",
            torch.zeros((1, 1, 5, 9), dtype=torch.float32),
        ),
        (
            "decoded_current_image",
            torch.empty((1, 3, 5, 9), device="meta"),
        ),
    ],
)
def test_static_preflight_fails_before_saliency_or_decoder(
    field_name: str,
    invalid_value: Any,
) -> None:
    inputs = _inputs()
    runtime = inputs["saliency_runtime"]
    decoder = inputs["vae_decoder"]
    inputs[field_name] = invalid_value

    with pytest.raises((TypeError, ValueError)):
        build_content_observation_routing(**inputs)

    assert runtime.image_inputs == []
    assert runtime.prompt_inputs == []
    assert decoder.inputs == []


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        (field_name, invalid_value)
        for field_name in (
            "reference_gradient",
            "reference_response",
            "reference_sensitivity",
        )
        for invalid_value in (1.1, 1e308)
    ],
)
def test_reference_binary32_preflight_fails_before_any_observation(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    invalid_value: float,
) -> None:
    inputs = _inputs()
    runtime = inputs["saliency_runtime"]
    decoder = inputs["vae_decoder"]
    inputs[field_name] = invalid_value
    calls = {"R": 0, "T": 0, "S": 0, "Q": 0, "route": 0}

    def forbidden(name: str) -> Any:
        def invoke(*args: Any, **kwargs: Any) -> Any:
            calls[name] += 1
            raise AssertionError(f"{name} must not run before reference preflight")

        return invoke

    monkeypatch.setattr(
        adapter_module,
        "build_adjacent_latent_response_map",
        forbidden("R"),
    )
    monkeypatch.setattr(
        adapter_module,
        "build_texture_complexity_map",
        forbidden("T"),
    )
    monkeypatch.setattr(
        adapter_module,
        "build_prompt_conditioned_semantic_saliency",
        forbidden("S"),
    )
    monkeypatch.setattr(
        adapter_module,
        "build_public_probe_local_sensitivity_map",
        forbidden("Q"),
    )
    monkeypatch.setattr(
        adapter_module,
        "_route_content_observations_to_latent",
        forbidden("route"),
    )

    with pytest.raises(ValueError, match="exactly representable as binary32"):
        build_content_observation_routing(**inputs)

    assert calls == {"R": 0, "T": 0, "S": 0, "Q": 0, "route": 0}
    assert runtime.image_inputs == []
    assert runtime.prompt_inputs == []
    assert decoder.inputs == []


def test_invalid_response_contents_fail_before_saliency_and_decoder() -> None:
    inputs = _inputs()
    runtime = inputs["saliency_runtime"]
    decoder = inputs["vae_decoder"]
    inputs["previous_scheduler_latent"][0, 0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="previous_scheduler_latent"):
        build_content_observation_routing(**inputs)

    assert runtime.image_inputs == []
    assert runtime.prompt_inputs == []
    assert decoder.inputs == []


def test_prompt_latent_and_image_counterfactuals_change_expected_observations(
) -> None:
    baseline_inputs = _inputs()
    baseline = build_content_observation_routing(**baseline_inputs)

    prompt_inputs = _inputs()
    prompt_inputs["prompt"] = "blue object"
    prompt_changed = build_content_observation_routing(**prompt_inputs)

    latent_inputs = _inputs()
    latent_inputs["previous_scheduler_latent"] = (
        latent_inputs["previous_scheduler_latent"] * 0.5
    )
    latent_changed = build_content_observation_routing(**latent_inputs)

    image_inputs = _inputs()
    image_inputs["decoded_current_image"] = torch.full(
        (1, 3, 5, 9), 0.75, dtype=torch.float32
    )
    image_changed = build_content_observation_routing(**image_inputs)

    assert not torch.equal(
        baseline.semantic_saliency.saliency_map,
        prompt_changed.semantic_saliency.saliency_map,
    )
    assert not torch.equal(baseline.routing.lf_mask, prompt_changed.routing.lf_mask)
    assert not torch.equal(
        baseline.latent_response.response_map,
        latent_changed.latent_response.response_map,
    )
    assert not torch.equal(baseline.routing.lf_mask, latent_changed.routing.lf_mask)
    assert not torch.equal(baseline.texture.texture_map, image_changed.texture.texture_map)
    assert not torch.equal(
        baseline.semantic_saliency.saliency_map,
        image_changed.semantic_saliency.saliency_map,
    )
    assert not torch.equal(
        baseline.local_sensitivity.local_sensitivity_map,
        image_changed.local_sensitivity.local_sensitivity_map,
    )
    assert not torch.equal(baseline.routing.lf_mask, image_changed.routing.lf_mask)


def test_repeated_calls_are_deterministic() -> None:
    first = build_content_observation_routing(**_inputs())
    second = build_content_observation_routing(**_inputs())

    _assert_result_fields_equal(first.semantic_saliency, second.semantic_saliency)
    _assert_result_fields_equal(first.texture, second.texture)
    _assert_result_fields_equal(first.latent_response, second.latent_response)
    _assert_result_fields_equal(first.local_sensitivity, second.local_sensitivity)
    _assert_routing_equal(first.routing, second.routing)


def test_source_has_no_experiment_model_network_or_accelerator_dependency() -> None:
    source_path = Path(adapter_module.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert all(not name.startswith("experiments") for name in imported_modules)
    assert "transformers" not in imported_modules
    assert "diffusers" not in imported_modules
    assert "cuda" not in source.lower()
    assert "network" not in source.lower()
