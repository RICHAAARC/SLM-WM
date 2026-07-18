from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import inspect
import math
from typing import Any

import pytest
import torch
import torch.nn.functional as functional

import main.methods.content as content
import main.methods.content.latent_response as latent_response_module
import main.methods.content.local_sensitivity as local_sensitivity_module
import main.methods.content.texture as texture_module
from main.core.digest import tensor_content_sha256
from main.core.keyed_prg import KEYED_PRG_VERSION


pytestmark = pytest.mark.unit


_PUBLIC_PROBE_IDENTITY = {
    "prg_version": KEYED_PRG_VERSION,
    "key_material": "semantic_saliency_dual_chain_public_probe_v1",
    "domain_fields": {
        "purpose": "local_sensitivity_public_probe",
        "model_revision": "exact-test-model-revision",
        "probe_version": "v1",
    },
}


def _independent_gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    image_float = image.float()
    weights = image_float.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    luminance = torch.sum(image_float * weights, dim=1, keepdim=True)
    kernel_x = image_float.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    kernel_y = image_float.new_tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
    ).unsqueeze(0)
    padded = functional.pad(luminance, (1, 1, 1, 1), mode="replicate")
    gradient_x = functional.conv2d(padded, kernel_x, stride=1)
    gradient_y = functional.conv2d(padded, kernel_y, stride=1)
    return torch.sqrt(torch.square(gradient_x) + torch.square(gradient_y))


def _independent_relative_response(
    previous: torch.Tensor,
    current: torch.Tensor,
) -> torch.Tensor:
    previous_float = previous.float()
    current_float = current.float()
    difference_rms = torch.sqrt(
        torch.mean(torch.square(current_float - previous_float), dim=1, keepdim=True)
    )
    current_rms = torch.sqrt(
        torch.mean(torch.square(current_float), dim=1, keepdim=True)
    )
    previous_rms = torch.sqrt(
        torch.mean(torch.square(previous_float), dim=1, keepdim=True)
    )
    return difference_rms / (current_rms + previous_rms + 1.0e-12)


def _fixed_pairwise_mean(value: torch.Tensor) -> torch.Tensor:
    level = value.float().contiguous().reshape(-1)
    while level.numel() > 1:
        paired_count = level.numel() // 2
        paired_end = paired_count * 2
        next_level = level[:paired_end].reshape(paired_count, 2).sum(dim=1)
        if paired_end != level.numel():
            next_level = torch.cat((next_level, level[-1:]))
        level = next_level
    return level[0] / level.new_tensor(value.numel())


def _fixed_pairwise_global_rms(value: torch.Tensor) -> torch.Tensor:
    scale = torch.amax(torch.abs(value))
    safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    normalized = value / safe_scale
    return scale * torch.sqrt(_fixed_pairwise_mean(torch.square(normalized)))


def _decoder_formula(latent: torch.Tensor) -> torch.Tensor:
    spatial = torch.mean(latent.float(), dim=1, keepdim=True)
    return torch.sigmoid(spatial).repeat(1, 3, 1, 1)


def test_raw_observation_kernels_are_private_and_exclude_references() -> None:
    assert tuple(
        inspect.signature(texture_module._measure_gradient_magnitude).parameters
    ) == ("image_float",)
    assert tuple(
        inspect.signature(
            latent_response_module._measure_adjacent_latent_relative_response
        ).parameters
    ) == ("previous_float", "current_float")
    assert tuple(
        inspect.signature(
            local_sensitivity_module._measure_public_probe_local_sensitivity
        ).parameters
    ) == ("latent", "reference_image", "vae_decoder", "identity")
    assert tuple(
        field.name
        for field in fields(local_sensitivity_module._LocalSensitivityObservation)
    ) == (
        "local_difference_sensitivity",
        "public_probe_digest",
        "probe_step",
        "reference_image_digest",
        "perturbed_image_digest",
    )
    observation = local_sensitivity_module._LocalSensitivityObservation(
        local_difference_sensitivity=torch.zeros((1, 1, 1, 1)),
        public_probe_digest="0" * 64,
        probe_step=1.0,
        reference_image_digest="1" * 64,
        perturbed_image_digest="2" * 64,
    )
    with pytest.raises(FrozenInstanceError):
        observation.probe_step = 2.0  # type: ignore[misc]

    private_names = {
        "_measure_gradient_magnitude",
        "_measure_adjacent_latent_relative_response",
        "_measure_public_probe_local_sensitivity",
        "_LocalSensitivityObservation",
    }
    assert private_names.isdisjoint(content.__all__)
    assert all(not hasattr(content, name) for name in private_names)


def test_raw_gradient_matches_formula_and_public_wrapper_measures_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = torch.tensor(
        [
            [
                [[0.0, 0.2, 0.8], [0.1, 0.7, 1.0]],
                [[0.9, 0.1, 0.3], [0.2, 0.6, 0.4]],
                [[0.3, 1.0, 0.0], [0.8, 0.5, 0.2]],
            ]
        ],
        dtype=torch.float64,
    )
    expected_raw = _independent_gradient_magnitude(image)
    assert torch.allclose(
        texture_module._measure_gradient_magnitude(image.float()),
        expected_raw,
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    original = texture_module._measure_gradient_magnitude
    captured: list[torch.Tensor] = []

    def recording_helper(value: torch.Tensor) -> torch.Tensor:
        measured = original(value)
        captured.append(measured.detach().clone())
        return measured

    monkeypatch.setattr(texture_module, "_measure_gradient_magnitude", recording_helper)
    result = texture_module.build_texture_complexity_map(image, 2.0)

    assert len(captured) == 1
    assert torch.equal(captured[0], expected_raw)
    assert torch.equal(result.texture_map, torch.clamp(expected_raw / 2.0, 0.0, 1.0))


def test_raw_gradient_is_reference_independent_while_public_digest_keeps_its_role() -> None:
    image = torch.zeros((1, 3, 3, 4), dtype=torch.float32)
    image[:, :, :, 2:] = 1.0
    raw = texture_module._measure_gradient_magnitude(image)
    smaller = texture_module.build_texture_complexity_map(image, 2.0)
    larger = texture_module.build_texture_complexity_map(image, 8.0)

    assert torch.equal(smaller.texture_map, torch.clamp(raw / 2.0, 0.0, 1.0))
    assert torch.equal(larger.texture_map, torch.clamp(raw / 8.0, 0.0, 1.0))
    assert smaller.texture_map_digest != larger.texture_map_digest


def test_raw_response_matches_formula_and_public_wrapper_measures_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = torch.tensor(
        [[[[0.0, 0.5], [1.0, 1.5]], [[0.25, 0.75], [1.25, 1.75]]]],
        dtype=torch.float64,
    )
    current = torch.tensor(
        [[[[0.5, 0.25], [1.5, 1.0]], [[0.75, 1.0], [1.0, 2.0]]]],
        dtype=torch.float64,
    )
    expected_raw = _independent_relative_response(previous, current)
    assert torch.allclose(
        latent_response_module._measure_adjacent_latent_relative_response(
            previous.float(), current.float()
        ),
        expected_raw,
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    original = latent_response_module._measure_adjacent_latent_relative_response
    captured: list[torch.Tensor] = []

    def recording_helper(
        previous_float: torch.Tensor,
        current_float: torch.Tensor,
    ) -> torch.Tensor:
        measured = original(previous_float, current_float)
        captured.append(measured.detach().clone())
        return measured

    monkeypatch.setattr(
        latent_response_module,
        "_measure_adjacent_latent_relative_response",
        recording_helper,
    )
    result = latent_response_module.build_adjacent_latent_response_map(
        previous,
        current,
        0.75,
    )

    assert len(captured) == 1
    assert torch.equal(captured[0], expected_raw)
    assert torch.equal(
        result.response_map,
        torch.clamp(expected_raw / 0.75, 0.0, 1.0),
    )


def test_saturated_response_maps_allow_equal_bare_map_digests_across_references() -> None:
    previous = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    current = torch.ones_like(previous)
    raw = latent_response_module._measure_adjacent_latent_relative_response(
        previous,
        current,
    )
    first = latent_response_module.build_adjacent_latent_response_map(
        previous, current, 0.5
    )
    second = latent_response_module.build_adjacent_latent_response_map(
        previous, current, 0.25
    )

    assert raw.item() == pytest.approx(1.0)
    assert torch.equal(first.response_map, torch.ones_like(first.response_map))
    assert torch.equal(first.response_map, second.response_map)
    assert first.reference_response != second.reference_response
    assert first.response_map_digest == second.response_map_digest
    assert first.response_map_digest == tensor_content_sha256(first.response_map)


def test_raw_local_sensitivity_matches_independent_formula_and_decodes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_probe = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    monkeypatch.setattr(
        local_sensitivity_module,
        "build_keyed_gaussian_tensor",
        lambda shape, **_: raw_probe.clone(),
    )
    latent = torch.linspace(-0.2, 0.2, 8, dtype=torch.float32).reshape(1, 2, 2, 2)
    reference_image = _decoder_formula(latent)
    decoder_inputs: list[torch.Tensor] = []

    def decoder(value: torch.Tensor) -> torch.Tensor:
        decoder_inputs.append(value.detach().clone())
        return _decoder_formula(value)

    identity = local_sensitivity_module._validate_public_probe_identity(
        _PUBLIC_PROBE_IDENTITY
    )
    observation = local_sensitivity_module._measure_public_probe_local_sensitivity(
        latent,
        reference_image,
        decoder,
        identity,
    )

    centered = raw_probe - _fixed_pairwise_mean(raw_probe)
    probe = centered / _fixed_pairwise_global_rms(centered)
    latent_scale = torch.amax(torch.abs(latent))
    latent_rms = latent_scale * torch.sqrt(
        torch.mean(torch.square(latent / latent_scale))
    )
    step = latent.new_tensor(1.0e-3) * torch.maximum(
        latent_rms,
        latent.new_tensor(1.0e-12),
    )
    perturbed_latent = latent + step * probe
    perturbed_image = _decoder_formula(perturbed_latent)
    difference = perturbed_image - reference_image
    expected_raw = torch.sqrt(
        torch.mean(torch.square(difference), dim=1, keepdim=True)
    ) / step

    assert len(decoder_inputs) == 1
    assert torch.equal(decoder_inputs[0], perturbed_latent)
    assert torch.allclose(
        observation.local_difference_sensitivity,
        expected_raw,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    assert observation.probe_step == float(step.item())
    assert observation.reference_image_digest == tensor_content_sha256(reference_image)
    assert observation.perturbed_image_digest == tensor_content_sha256(perturbed_image)


def test_public_local_sensitivity_reuses_one_raw_measurement_per_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_probe = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    monkeypatch.setattr(
        local_sensitivity_module,
        "build_keyed_gaussian_tensor",
        lambda shape, **_: raw_probe.clone(),
    )
    latent = torch.linspace(-0.2, 0.2, 8, dtype=torch.float32).reshape(1, 2, 2, 2)
    reference_image = _decoder_formula(latent)
    original = local_sensitivity_module._measure_public_probe_local_sensitivity
    observations: list[Any] = []
    decoder_calls = 0

    def decoder(value: torch.Tensor) -> torch.Tensor:
        nonlocal decoder_calls
        decoder_calls += 1
        return _decoder_formula(value)

    def recording_helper(*args: Any, **kwargs: Any) -> Any:
        measured = original(*args, **kwargs)
        observations.append(measured)
        return measured

    monkeypatch.setattr(
        local_sensitivity_module,
        "_measure_public_probe_local_sensitivity",
        recording_helper,
    )
    smaller = local_sensitivity_module.build_public_probe_local_sensitivity_map(
        latent,
        reference_image,
        decoder,
        _PUBLIC_PROBE_IDENTITY,
        0.5,
    )
    larger = local_sensitivity_module.build_public_probe_local_sensitivity_map(
        latent,
        reference_image,
        decoder,
        _PUBLIC_PROBE_IDENTITY,
        2.0,
    )

    assert len(observations) == 2
    assert decoder_calls == 2
    assert torch.equal(
        observations[0].local_difference_sensitivity,
        observations[1].local_difference_sensitivity,
    )
    assert observations[0].public_probe_digest == observations[1].public_probe_digest
    assert torch.equal(
        smaller.local_sensitivity_map,
        torch.clamp(observations[0].local_difference_sensitivity / 0.5, 0.0, 1.0),
    )
    assert torch.equal(
        larger.local_sensitivity_map,
        torch.clamp(observations[1].local_difference_sensitivity / 2.0, 0.0, 1.0),
    )
    assert smaller.local_sensitivity_map_digest != larger.local_sensitivity_map_digest


@pytest.mark.parametrize(
    ("module", "builder_name", "helper_name", "args"),
    [
        (
            texture_module,
            "build_texture_complexity_map",
            "_measure_gradient_magnitude",
            (torch.full((1, 3, 2, 2), math.nan), 0.0),
        ),
        (
            latent_response_module,
            "build_adjacent_latent_response_map",
            "_measure_adjacent_latent_relative_response",
            (
                torch.full((1, 1, 2, 2), math.nan),
                torch.zeros((1, 1, 1, 2)),
                0.0,
            ),
        ),
    ],
)
def test_invalid_reference_prevents_texture_or_response_raw_measurement(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    builder_name: str,
    helper_name: str,
    args: tuple[Any, ...],
) -> None:
    helper_calls = 0

    def forbidden_helper(*_: Any, **__: Any) -> torch.Tensor:
        nonlocal helper_calls
        helper_calls += 1
        raise AssertionError("raw helper must not run")

    monkeypatch.setattr(module, helper_name, forbidden_helper)
    with pytest.raises(ValueError, match="reference_"):
        getattr(module, builder_name)(*args)
    assert helper_calls == 0


def test_texture_float32_reference_gate_remains_after_image_content_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper_calls = 0

    def forbidden_helper(_: torch.Tensor) -> torch.Tensor:
        nonlocal helper_calls
        helper_calls += 1
        raise AssertionError("raw helper must not run")

    monkeypatch.setattr(texture_module, "_measure_gradient_magnitude", forbidden_helper)
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    image[0, 0, 0, 0] = math.nan
    with pytest.raises(ValueError, match="image must contain only finite"):
        texture_module.build_texture_complexity_map(image, 1.0e-50)
    assert helper_calls == 0


@pytest.mark.parametrize("reference", [1.0e-50, 1.0e50])
def test_texture_float32_reference_failure_prevents_raw_measurement(
    monkeypatch: pytest.MonkeyPatch,
    reference: float,
) -> None:
    helper_calls = 0

    def forbidden_helper(_: torch.Tensor) -> torch.Tensor:
        nonlocal helper_calls
        helper_calls += 1
        raise AssertionError("raw helper must not run")

    monkeypatch.setattr(texture_module, "_measure_gradient_magnitude", forbidden_helper)
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="float32"):
        texture_module.build_texture_complexity_map(image, reference)
    assert helper_calls == 0


@pytest.mark.parametrize("reference", [0.0, 1.0e-50, 1.0e50])
def test_invalid_local_reference_prevents_raw_measurement_and_decoder(
    monkeypatch: pytest.MonkeyPatch,
    reference: float,
) -> None:
    helper_calls = 0
    decoder_calls = 0

    def forbidden_helper(*_: Any, **__: Any) -> Any:
        nonlocal helper_calls
        helper_calls += 1
        raise AssertionError("raw helper must not run")

    def decoder(_: torch.Tensor) -> torch.Tensor:
        nonlocal decoder_calls
        decoder_calls += 1
        raise AssertionError("decoder must not run")

    monkeypatch.setattr(
        local_sensitivity_module,
        "_measure_public_probe_local_sensitivity",
        forbidden_helper,
    )
    latent = torch.full((1, 2, 2, 2), math.nan, dtype=torch.float32)
    image = torch.full((1, 3, 2, 2), math.nan, dtype=torch.float32)
    with pytest.raises(ValueError, match="reference_sensitivity"):
        local_sensitivity_module.build_public_probe_local_sensitivity_map(
            latent,
            image,
            decoder,
            _PUBLIC_PROBE_IDENTITY,
            reference,
        )
    assert helper_calls == 0
    assert decoder_calls == 0
