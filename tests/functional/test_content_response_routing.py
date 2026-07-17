from __future__ import annotations

from dataclasses import fields

import pytest
import torch

import main.methods.content as content
from main.core.digest import build_stable_digest, tensor_content_sha256
from main.methods.content import (
    ContentRoutingResult,
    LatentResponseResult,
    build_adjacent_latent_response_map,
    route_content_carriers,
)


pytestmark = pytest.mark.unit


def test_content_subpackage_exports_exact_public_interface() -> None:
    assert content.__all__ == (
        "LatentResponseResult",
        "ContentRoutingResult",
        "build_adjacent_latent_response_map",
        "route_content_carriers",
    )
    assert content.LatentResponseResult is LatentResponseResult
    assert content.ContentRoutingResult is ContentRoutingResult
    assert content.build_adjacent_latent_response_map is build_adjacent_latent_response_map
    assert content.route_content_carriers is route_content_carriers
    assert tuple(field.name for field in fields(LatentResponseResult)) == (
        "response_map",
        "reference_response",
        "previous_latent_digest",
        "current_latent_digest",
        "response_map_digest",
    )
    assert tuple(field.name for field in fields(ContentRoutingResult)) == (
        "writable_capacity_map",
        "lf_mask",
        "hf_tail_mask",
        "routing_identity_digest",
    )


def test_adjacent_latent_response_matches_frozen_formula_and_is_deterministic() -> None:
    previous = torch.tensor(
        [[[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]]],
        dtype=torch.float64,
    )
    current = torch.tensor(
        [[[[1.0, 1.5], [1.0, 5.0]], [[2.0, 1.0], [5.0, 2.0]]]],
        dtype=torch.float64,
    )
    result = build_adjacent_latent_response_map(previous, current, 0.75)
    repeated = build_adjacent_latent_response_map(previous, current, 0.75)

    previous_float = previous.float()
    current_float = current.float()
    expected_difference = torch.sqrt(
        torch.mean((current_float - previous_float) ** 2, dim=1, keepdim=True)
    )
    expected_current = torch.sqrt(torch.mean(current_float**2, dim=1, keepdim=True))
    expected_previous = torch.sqrt(
        torch.mean(previous_float**2, dim=1, keepdim=True)
    )
    expected = torch.clamp(
        expected_difference / (expected_current + expected_previous + 1.0e-12) / 0.75,
        0.0,
        1.0,
    )
    assert torch.equal(result.response_map, expected)
    assert result.response_map.shape == (1, 1, 2, 2)
    assert result.response_map.dtype == torch.float32
    assert result.response_map.device == previous.device
    assert torch.equal(repeated.response_map, result.response_map)
    assert repeated.reference_response == result.reference_response
    assert repeated.previous_latent_digest == result.previous_latent_digest
    assert repeated.current_latent_digest == result.current_latent_digest
    assert repeated.response_map_digest == result.response_map_digest
    assert result.reference_response == 0.75
    assert result.previous_latent_digest == tensor_content_sha256(previous)
    assert result.current_latent_digest == tensor_content_sha256(current)
    assert result.response_map_digest == tensor_content_sha256(result.response_map)


def test_adjacent_latent_response_symmetry_locality_and_reference_monotonicity() -> None:
    previous = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    current = previous.clone()
    current[:, :, 0, 1] = 0.25

    forward = build_adjacent_latent_response_map(previous, current, 1.0)
    reverse = build_adjacent_latent_response_map(current, previous, 1.0)
    larger_reference = build_adjacent_latent_response_map(previous, current, 2.0)

    assert torch.equal(forward.response_map, reverse.response_map)
    assert torch.count_nonzero(forward.response_map).item() == 1
    assert forward.response_map[0, 0, 0, 1] > 0.0
    assert torch.all(larger_reference.response_map <= forward.response_map)
    unchanged = build_adjacent_latent_response_map(previous, previous, 1.0)
    assert torch.count_nonzero(unchanged.response_map).item() == 0


def test_response_digest_only_binds_the_clipped_output() -> None:
    previous = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    first_current = torch.ones_like(previous)
    second_current = torch.full_like(previous, 2.0)
    first = build_adjacent_latent_response_map(previous, first_current, 0.5)
    second = build_adjacent_latent_response_map(previous, second_current, 0.5)

    assert torch.equal(first.response_map, second.response_map)
    assert first.response_map_digest == second.response_map_digest
    assert first.current_latent_digest != second.current_latent_digest
    assert first.reference_response == second.reference_response == 0.5


@pytest.mark.parametrize(
    "invalid",
    [
        "not-a-tensor",
        torch.ones((1, 1, 1, 1), dtype=torch.bool),
        torch.ones((1, 1, 1, 1), dtype=torch.int64),
        torch.ones((1, 1, 1, 1), dtype=torch.complex64),
    ],
)
def test_adjacent_latent_response_rejects_non_real_floating_tensors(invalid: object) -> None:
    valid = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    with pytest.raises(TypeError):
        build_adjacent_latent_response_map(invalid, valid, 1.0)


@pytest.mark.parametrize("reference", [True, 0.0, -1.0, float("inf"), float("nan"), 1 + 2j])
def test_adjacent_latent_response_rejects_invalid_reference(reference: object) -> None:
    latent = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    with pytest.raises((TypeError, ValueError)):
        build_adjacent_latent_response_map(latent, latent, reference)  # type: ignore[arg-type]


def test_adjacent_latent_response_fails_after_float32_overflow_cast() -> None:
    previous = torch.zeros((1, 1, 1, 1), dtype=torch.float64)
    current = torch.full_like(previous, torch.finfo(torch.float64).max)
    with pytest.raises(ValueError, match="after float32 cast"):
        build_adjacent_latent_response_map(previous, current, 1.0)


def test_adjacent_latent_response_avoids_finite_float32_square_overflow() -> None:
    previous = torch.full((1, 1, 1, 1), -1.0e19, dtype=torch.float32)
    current = torch.full_like(previous, 1.0e19)
    result = build_adjacent_latent_response_map(previous, current, 2.0)

    assert result.response_map.item() == pytest.approx(0.5, rel=1.0e-6, abs=1.0e-7)


@pytest.mark.parametrize(
    ("previous_value", "current_value", "error_name"),
    [
        (-torch.finfo(torch.float32).max, torch.finfo(torch.float32).max, "latent_difference"),
        (torch.finfo(torch.float32).max, torch.finfo(torch.float32).max, "response_denominator"),
    ],
)
def test_adjacent_latent_response_rejects_non_finite_float32_intermediates(
    previous_value: float,
    current_value: float,
    error_name: str,
) -> None:
    previous = torch.full((1, 1, 1, 1), previous_value, dtype=torch.float32)
    current = torch.full_like(previous, current_value)

    with pytest.raises(ValueError, match=error_name):
        build_adjacent_latent_response_map(previous, current, 2.0)


def test_adjacent_latent_response_validates_shape_finiteness_and_device_before_content() -> None:
    valid = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="shapes must match"):
        build_adjacent_latent_response_map(valid, torch.zeros((1, 1, 1, 2)), 1.0)
    with pytest.raises(ValueError, match="B=1 NCHW"):
        build_adjacent_latent_response_map(torch.zeros((2, 1, 2, 2)), valid, 1.0)
    non_finite = valid.clone()
    non_finite[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_adjacent_latent_response_map(non_finite, valid, 1.0)
    meta = torch.empty(valid.shape, dtype=torch.float32, device="meta")
    with pytest.raises(ValueError, match="same device"):
        build_adjacent_latent_response_map(valid, meta, 1.0)


def test_content_routing_matches_frozen_formula_without_post_normalization() -> None:
    saliency = torch.tensor([[[[0.125, 0.25], [0.5, 0.75]]]], dtype=torch.float64)
    texture = torch.tensor([[[[0.0, 0.25], [0.75, 1.0]]]], dtype=torch.float64)
    response = torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]], dtype=torch.float64)
    sensitivity = torch.tensor([[[[0.3, 0.5], [0.7, 0.9]]]], dtype=torch.float64)
    result = route_content_carriers(saliency, texture, response, sensitivity)
    repeated = route_content_carriers(saliency, texture, response, sensitivity)

    expected_capacity = torch.pow(
        (1.0 - saliency.float())
        * (1.0 - response.float())
        * (1.0 - sensitivity.float()),
        1.0 / 3.0,
    )
    assert torch.equal(result.writable_capacity_map, expected_capacity)
    assert torch.equal(result.lf_mask, expected_capacity * (1.0 - texture.float()))
    assert torch.equal(result.hf_tail_mask, expected_capacity * texture.float())
    assert torch.equal(result.lf_mask + result.hf_tail_mask, expected_capacity)
    assert result.writable_capacity_map.max().item() < 1.0
    assert result.writable_capacity_map.dtype == torch.float32
    assert result.writable_capacity_map.device == saliency.device
    assert torch.equal(repeated.writable_capacity_map, result.writable_capacity_map)
    assert torch.equal(repeated.lf_mask, result.lf_mask)
    assert torch.equal(repeated.hf_tail_mask, result.hf_tail_mask)
    assert repeated.routing_identity_digest == result.routing_identity_digest


def test_content_routing_digest_binds_structured_input_and_output_summaries() -> None:
    saliency = torch.tensor([[[[0.1, 0.2]]]], dtype=torch.float32)
    texture = torch.tensor([[[[0.3, 0.4]]]], dtype=torch.float32)
    response = torch.tensor([[[[0.5, 0.6]]]], dtype=torch.float32)
    sensitivity = torch.tensor([[[[0.7, 0.8]]]], dtype=torch.float32)
    result = route_content_carriers(saliency, texture, response, sensitivity)
    expected_digest = build_stable_digest(
        {
            "schema_version": "content_routing_result_v1",
            "formula_version": "semantic_saliency_adaptive_content_routing_v1",
            "input_tensor_digests": {
                "saliency_map": tensor_content_sha256(saliency),
                "texture_map": tensor_content_sha256(texture),
                "response_map": tensor_content_sha256(response),
                "local_sensitivity_map": tensor_content_sha256(sensitivity),
            },
            "output_tensor_digests": {
                "writable_capacity_map": tensor_content_sha256(
                    result.writable_capacity_map
                ),
                "lf_mask": tensor_content_sha256(result.lf_mask),
                "hf_tail_mask": tensor_content_sha256(result.hf_tail_mask),
            },
        }
    )
    assert result.routing_identity_digest == expected_digest

    changed_texture = texture.clone()
    changed_texture[0, 0, 0, 0] = 0.35
    changed = route_content_carriers(
        saliency, changed_texture, response, sensitivity
    )
    assert changed.routing_identity_digest != result.routing_identity_digest


def test_content_routing_protected_regions_extremes_and_monotonicity() -> None:
    zeros = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
    texture = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float32)
    unprotected = route_content_carriers(zeros, texture, zeros, zeros)
    assert torch.equal(unprotected.writable_capacity_map, torch.ones_like(zeros))
    assert torch.equal(unprotected.lf_mask, torch.tensor([[[[1.0, 0.0]]]]))
    assert torch.equal(unprotected.hf_tail_mask, torch.tensor([[[[0.0, 1.0]]]]))

    protected = route_content_carriers(torch.ones_like(zeros), texture, zeros, zeros)
    assert torch.count_nonzero(protected.writable_capacity_map).item() == 0
    assert torch.count_nonzero(protected.lf_mask).item() == 0
    assert torch.count_nonzero(protected.hf_tail_mask).item() == 0

    partly_protected = route_content_carriers(
        torch.full_like(zeros, 0.5), texture, zeros, zeros
    )
    assert torch.all(
        partly_protected.writable_capacity_map
        <= unprotected.writable_capacity_map
    )


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        torch.ones((1, 1, 1, 1), dtype=torch.bool),
        torch.ones((1, 1, 1, 1), dtype=torch.int32),
        torch.ones((1, 1, 1, 1), dtype=torch.complex64),
    ],
)
def test_content_routing_rejects_non_real_floating_tensors(invalid: object) -> None:
    valid = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    with pytest.raises(TypeError):
        route_content_carriers(invalid, valid, valid, valid)


def test_content_routing_validates_shape_range_finiteness_and_device_before_content() -> None:
    valid = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="shapes must match"):
        route_content_carriers(valid, torch.zeros((1, 1, 1, 2)), valid, valid)
    with pytest.raises(ValueError, match=r"\[1, 1, H, W\]"):
        route_content_carriers(torch.zeros((1, 2, 2, 2)), valid, valid, valid)
    out_of_range = valid.clone()
    out_of_range[0, 0, 0, 0] = 1.01
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        route_content_carriers(out_of_range, valid, valid, valid)
    non_finite = valid.clone()
    non_finite[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        route_content_carriers(non_finite, valid, valid, valid)
    meta = torch.empty(valid.shape, dtype=torch.float32, device="meta")
    with pytest.raises(ValueError, match="same device"):
        route_content_carriers(valid, meta, valid, valid)
