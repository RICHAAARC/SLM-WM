"""验证原始内容观测到 latent 内容路由的唯一编排边界。"""

from __future__ import annotations

import inspect
from typing import Any, get_type_hints

import pytest
import torch
import torch.nn.functional as functional

import main.methods.content as content
import main.methods.content.routing as routing_module
from main.core.digest import tensor_content_sha256
from main.methods.content.routing import (
    ContentRoutingResult,
    _route_content_observations_to_latent,
    route_content_carriers,
)


pytestmark = pytest.mark.unit


def _maps(dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, ...]:
    saliency = torch.tensor(
        [[[[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]]]],
        dtype=dtype,
    )
    texture = torch.tensor(
        [[[[0.1, 0.3], [0.5, 0.7], [0.9, 1.0]]]],
        dtype=dtype,
    )
    response = torch.tensor(
        [[[[0.05, 0.15, 0.25, 0.35], [0.45, 0.55, 0.65, 0.75], [0.85, 0.9, 0.95, 1.0]]]],
        dtype=dtype,
    )
    sensitivity = torch.tensor(
        [
            [
                [
                    [0.0, 0.1, 0.2, 0.3, 0.4],
                    [0.2, 0.3, 0.4, 0.5, 0.6],
                    [0.4, 0.5, 0.6, 0.7, 0.8],
                    [0.6, 0.7, 0.8, 0.9, 1.0],
                ]
            ]
        ],
        dtype=dtype,
    )
    return saliency, texture, response, sensitivity


def _assert_result_equal(
    actual: ContentRoutingResult,
    expected: ContentRoutingResult,
) -> None:
    assert torch.equal(actual.writable_capacity_map, expected.writable_capacity_map)
    assert torch.equal(actual.lf_mask, expected.lf_mask)
    assert torch.equal(actual.hf_tail_mask, expected.hf_tail_mask)
    assert actual.routing_identity_digest == expected.routing_identity_digest


def test_observation_routing_is_private_with_exact_signature() -> None:
    signature = inspect.signature(_route_content_observations_to_latent)
    hints = get_type_hints(_route_content_observations_to_latent)

    assert tuple(signature.parameters) == (
        "saliency_map",
        "texture_map",
        "response_map",
        "local_sensitivity_map",
    )
    assert all(hints[name] is Any for name in signature.parameters)
    assert hints["return"] is ContentRoutingResult
    assert "_route_content_observations_to_latent" not in content.__all__
    assert not hasattr(content, "_route_content_observations_to_latent")


def test_observations_map_once_in_fixed_order_and_match_independent_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saliency, texture, response, sensitivity = _maps(torch.float64)
    sources = (saliency, texture, response, sensitivity)
    source_digests = tuple(tensor_content_sha256(value) for value in sources)
    target_shape = tuple(response.shape[-2:])
    expected_saliency = functional.interpolate(
        saliency.float(),
        size=target_shape,
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    expected_texture = functional.interpolate(
        texture.float(),
        size=target_shape,
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    expected_sensitivity = functional.interpolate(
        sensitivity.float(),
        size=target_shape,
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    expected = route_content_carriers(
        expected_saliency,
        expected_texture,
        response,
        expected_sensitivity,
    )

    labels = {
        id(saliency): "saliency_map",
        id(texture): "texture_map",
        id(sensitivity): "local_sensitivity_map",
    }
    helper_calls: list[tuple[str, tuple[int, int]]] = []
    interpolate_calls: list[dict[str, object]] = []
    route_calls: list[tuple[torch.Tensor, ...]] = []
    original_resize = routing_module._resize_content_map_to_latent
    original_interpolate = routing_module.functional.interpolate
    original_route = routing_module.route_content_carriers

    def recording_resize(value: Any, shape: Any) -> torch.Tensor:
        helper_calls.append((labels[id(value)], tuple(shape)))
        return original_resize(value, shape)

    def recording_interpolate(
        input_tensor: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        interpolate_calls.append(dict(kwargs))
        return original_interpolate(input_tensor, **kwargs)

    def recording_route(*values: torch.Tensor) -> ContentRoutingResult:
        route_calls.append(values)
        return original_route(*values)

    monkeypatch.setattr(
        routing_module, "_resize_content_map_to_latent", recording_resize
    )
    monkeypatch.setattr(
        routing_module.functional, "interpolate", recording_interpolate
    )
    monkeypatch.setattr(routing_module, "route_content_carriers", recording_route)

    result = _route_content_observations_to_latent(*sources)

    assert helper_calls == [
        ("saliency_map", target_shape),
        ("texture_map", target_shape),
        ("local_sensitivity_map", target_shape),
    ]
    assert interpolate_calls == [
        {
            "size": target_shape,
            "mode": "bilinear",
            "align_corners": False,
            "antialias": False,
        }
    ] * 3
    assert len(route_calls) == 1
    assert route_calls[0][2] is response
    assert all(value is not response for value in route_calls[0][:2] + route_calls[0][3:])
    _assert_result_equal(result, expected)
    assert tuple(tensor_content_sha256(value) for value in sources) == source_digests
    assert all(value.dtype == torch.float32 for value in route_calls[0][:2] + route_calls[0][3:])
    assert result.writable_capacity_map.shape == response.shape
    assert result.writable_capacity_map.dtype == torch.float32
    assert result.writable_capacity_map.device == response.device
    for value in (
        result.writable_capacity_map,
        result.lf_mask,
        result.hf_tail_mask,
    ):
        assert torch.isfinite(value).all()
        assert torch.all((value >= 0.0) & (value <= 1.0))


def test_identity_shape_still_interpolates_three_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    maps = tuple(
        torch.full((1, 1, 3, 5), value, dtype=torch.float32)
        for value in (0.1, 0.2, 0.3, 0.4)
    )
    calls: list[dict[str, object]] = []
    original_interpolate = routing_module.functional.interpolate

    def recording_interpolate(
        input_tensor: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        calls.append(dict(kwargs))
        return original_interpolate(input_tensor, **kwargs)

    monkeypatch.setattr(
        routing_module.functional, "interpolate", recording_interpolate
    )
    _route_content_observations_to_latent(*maps)

    assert calls == [
        {
            "size": (3, 5),
            "mode": "bilinear",
            "align_corners": False,
            "antialias": False,
        }
    ] * 3


def test_observation_routing_is_deterministic_for_float64_inputs() -> None:
    maps = _maps(torch.float64)
    first = _route_content_observations_to_latent(*maps)
    second = _route_content_observations_to_latent(*maps)

    _assert_result_equal(first, second)
    assert first.writable_capacity_map.dtype == torch.float32


@pytest.mark.parametrize(
    ("position", "invalid"),
    [
        (0, "not-a-tensor"),
        (1, torch.ones((1, 1, 2, 2), dtype=torch.bool)),
        (2, torch.ones((1, 2, 2, 2), dtype=torch.float32)),
        (3, torch.empty((1, 1, 0, 2), dtype=torch.float32)),
    ],
)
def test_metadata_failures_precede_interpolation_and_route(
    monkeypatch: pytest.MonkeyPatch,
    position: int,
    invalid: Any,
) -> None:
    maps = list(_maps())
    maps[position] = invalid
    interpolate_calls = 0
    route_calls = 0

    def forbidden_interpolate(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal interpolate_calls
        interpolate_calls += 1
        raise AssertionError("interpolate must not run")

    def forbidden_route(*args: Any, **kwargs: Any) -> ContentRoutingResult:
        nonlocal route_calls
        route_calls += 1
        raise AssertionError("route must not run")

    monkeypatch.setattr(routing_module.functional, "interpolate", forbidden_interpolate)
    monkeypatch.setattr(routing_module, "route_content_carriers", forbidden_route)

    with pytest.raises((TypeError, ValueError)):
        _route_content_observations_to_latent(*maps)
    assert interpolate_calls == 0
    assert route_calls == 0


def test_device_mismatch_fails_before_any_content_read_or_interpolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saliency, _, response, sensitivity = _maps()
    saliency[0, 0, 0, 0] = float("nan")
    texture = torch.empty((1, 1, 3, 2), dtype=torch.float32, device="meta")
    content_reads = 0
    interpolate_calls = 0
    route_calls = 0

    def forbidden_contents(name: str, value: torch.Tensor) -> None:
        nonlocal content_reads
        content_reads += 1
        raise AssertionError("contents must not be read")

    def forbidden_interpolate(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal interpolate_calls
        interpolate_calls += 1
        raise AssertionError("interpolate must not run")

    def forbidden_route(*args: Any, **kwargs: Any) -> ContentRoutingResult:
        nonlocal route_calls
        route_calls += 1
        raise AssertionError("route must not run")

    monkeypatch.setattr(routing_module, "_validate_routing_contents", forbidden_contents)
    monkeypatch.setattr(routing_module.functional, "interpolate", forbidden_interpolate)
    monkeypatch.setattr(routing_module, "route_content_carriers", forbidden_route)

    with pytest.raises(ValueError, match="same device"):
        _route_content_observations_to_latent(
            saliency,
            texture,
            response,
            sensitivity,
        )
    assert content_reads == 0
    assert interpolate_calls == 0
    assert route_calls == 0


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_response_fails_before_interpolation_and_route(
    monkeypatch: pytest.MonkeyPatch,
    invalid_value: float,
) -> None:
    saliency, texture, response, sensitivity = _maps()
    response[0, 0, 0, 0] = invalid_value
    interpolate_calls = 0
    route_calls = 0

    def forbidden_interpolate(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal interpolate_calls
        interpolate_calls += 1
        raise AssertionError("interpolate must not run")

    def forbidden_route(*args: Any, **kwargs: Any) -> ContentRoutingResult:
        nonlocal route_calls
        route_calls += 1
        raise AssertionError("route must not run")

    monkeypatch.setattr(routing_module.functional, "interpolate", forbidden_interpolate)
    monkeypatch.setattr(routing_module, "route_content_carriers", forbidden_route)

    with pytest.raises(ValueError, match=r"finite|\[0, 1\]"):
        _route_content_observations_to_latent(
            saliency,
            texture,
            response,
            sensitivity,
        )
    assert interpolate_calls == 0
    assert route_calls == 0


@pytest.mark.parametrize(
    ("invalid_position", "expected_attempts", "expected_interpolations"),
    [
        (0, ["saliency_map"], 0),
        (1, ["saliency_map", "texture_map"], 1),
        (
            3,
            ["saliency_map", "texture_map", "local_sensitivity_map"],
            2,
        ),
    ],
)
def test_invalid_resized_observation_fails_at_its_helper_without_route(
    monkeypatch: pytest.MonkeyPatch,
    invalid_position: int,
    expected_attempts: list[str],
    expected_interpolations: int,
) -> None:
    maps = list(_maps())
    maps[invalid_position][0, 0, 0, 0] = float("nan")
    labels = {
        id(maps[0]): "saliency_map",
        id(maps[1]): "texture_map",
        id(maps[3]): "local_sensitivity_map",
    }
    helper_attempts: list[str] = []
    interpolate_calls = 0
    route_calls = 0
    original_resize = routing_module._resize_content_map_to_latent
    original_interpolate = routing_module.functional.interpolate

    def recording_resize(value: Any, shape: Any) -> torch.Tensor:
        helper_attempts.append(labels[id(value)])
        return original_resize(value, shape)

    def recording_interpolate(
        input_tensor: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        nonlocal interpolate_calls
        interpolate_calls += 1
        return original_interpolate(input_tensor, **kwargs)

    def forbidden_route(*args: Any, **kwargs: Any) -> ContentRoutingResult:
        nonlocal route_calls
        route_calls += 1
        raise AssertionError("route must not run")

    monkeypatch.setattr(
        routing_module, "_resize_content_map_to_latent", recording_resize
    )
    monkeypatch.setattr(
        routing_module.functional, "interpolate", recording_interpolate
    )
    monkeypatch.setattr(routing_module, "route_content_carriers", forbidden_route)

    with pytest.raises(ValueError, match="finite"):
        _route_content_observations_to_latent(*maps)
    assert helper_attempts == expected_attempts
    assert interpolate_calls == expected_interpolations
    assert route_calls == 0
