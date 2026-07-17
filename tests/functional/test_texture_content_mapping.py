from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch
import torch.nn.functional as functional

from main.core.digest import build_stable_digest, tensor_content_sha256
from main.methods.content import TextureResult, build_texture_complexity_map
from main.methods.content.routing import _resize_content_map_to_latent


pytestmark = pytest.mark.unit


_EXPECTED_FORMULA_PROTOCOL = {
    "input_color_space": "rgb_0_1",
    "luminance_weights": [0.299, 0.587, 0.114],
    "sobel_kernel_x": [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ],
    "sobel_kernel_y": [
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ],
    "padding_mode": "replicate",
    "padding_pixels": 1,
    "stride": 1,
    "gradient_magnitude": "sqrt_gx_squared_plus_gy_squared",
    "normalization": "clip_gradient_magnitude_div_reference_gradient_0_1",
    "computation_dtype": "float32",
    "output_resolution": "input_rgb_spatial_resolution",
}


def _manual_texture_map(image: torch.Tensor, reference: float) -> torch.Tensor:
    image_float = image.float()
    luminance = (
        0.299 * image_float[0, 0]
        + 0.587 * image_float[0, 1]
        + 0.114 * image_float[0, 2]
    )
    kernel_x = (
        (-1.0, 0.0, 1.0),
        (-2.0, 0.0, 2.0),
        (-1.0, 0.0, 1.0),
    )
    kernel_y = (
        (-1.0, -2.0, -1.0),
        (0.0, 0.0, 0.0),
        (1.0, 2.0, 1.0),
    )
    height, width = luminance.shape
    expected = torch.empty((1, 1, height, width), dtype=torch.float32)
    for row in range(height):
        for column in range(width):
            gradient_x = 0.0
            gradient_y = 0.0
            for kernel_row in range(3):
                for kernel_column in range(3):
                    source_row = min(max(row + kernel_row - 1, 0), height - 1)
                    source_column = min(
                        max(column + kernel_column - 1, 0), width - 1
                    )
                    value = float(luminance[source_row, source_column])
                    gradient_x += value * kernel_x[kernel_row][kernel_column]
                    gradient_y += value * kernel_y[kernel_row][kernel_column]
            magnitude = (gradient_x**2 + gradient_y**2) ** 0.5
            expected[0, 0, row, column] = min(max(magnitude / reference, 0.0), 1.0)
    return expected


def _expected_texture_digest(result: TextureResult) -> str:
    return build_stable_digest(
        {
            "schema_version": "texture_result_v1",
            "formula_protocol_version": "frozen_rgb_sobel_texture_complexity_v1",
            "formula_protocol": _EXPECTED_FORMULA_PROTOCOL,
            "reference_gradient": result.reference_gradient,
            "texture_map_content_sha256": tensor_content_sha256(result.texture_map),
        }
    )


def test_texture_result_is_frozen_and_has_exact_fields() -> None:
    assert tuple(field.name for field in fields(TextureResult)) == (
        "texture_map",
        "reference_gradient",
        "texture_map_digest",
    )
    result = build_texture_complexity_map(
        torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        1.0,
    )
    with pytest.raises(FrozenInstanceError):
        result.reference_gradient = 2.0  # type: ignore[misc]


def test_constant_image_has_zero_texture_at_every_boundary() -> None:
    image = torch.full((1, 3, 3, 5), 0.375, dtype=torch.float64)
    result = build_texture_complexity_map(image, 2.0)
    repeated = build_texture_complexity_map(image, 2.0)

    assert torch.count_nonzero(result.texture_map).item() == 0
    assert result.texture_map.shape == (1, 1, 3, 5)
    assert result.texture_map.dtype == torch.float32
    assert result.texture_map.device == image.device
    assert torch.equal(repeated.texture_map, result.texture_map)
    assert repeated.texture_map_digest == result.texture_map_digest


def test_horizontal_and_vertical_edges_match_frozen_sobel_formula() -> None:
    vertical = torch.zeros((1, 3, 3, 4), dtype=torch.float32)
    vertical[:, :, :, 2:] = 1.0
    horizontal = vertical.transpose(-2, -1).contiguous()

    vertical_result = build_texture_complexity_map(vertical, 8.0)
    horizontal_result = build_texture_complexity_map(horizontal, 8.0)

    assert torch.allclose(
        vertical_result.texture_map,
        _manual_texture_map(vertical, 8.0),
        rtol=0.0,
        atol=1.0e-7,
    )
    assert torch.allclose(
        horizontal_result.texture_map,
        _manual_texture_map(horizontal, 8.0),
        rtol=0.0,
        atol=1.0e-7,
    )
    assert torch.allclose(
        horizontal_result.texture_map,
        vertical_result.texture_map.transpose(-2, -1),
        rtol=0.0,
        atol=1.0e-7,
    )


def test_rgb_edges_follow_frozen_luminance_weights() -> None:
    maxima = []
    for channel in range(3):
        image = torch.zeros((1, 3, 3, 4), dtype=torch.float32)
        image[:, channel, :, 2:] = 1.0
        result = build_texture_complexity_map(image, 16.0)
        maxima.append(result.texture_map.max().item())

    assert maxima[0] == pytest.approx(0.299 / 4.0, rel=1.0e-6)
    assert maxima[1] == pytest.approx(0.587 / 4.0, rel=1.0e-6)
    assert maxima[2] == pytest.approx(0.114 / 4.0, rel=1.0e-6)


def test_replicate_padding_matches_manual_asymmetric_boundary_convolution() -> None:
    image = torch.tensor(
        [
            [
                [[0.0, 0.2, 0.9], [0.4, 0.7, 1.0]],
                [[0.8, 0.1, 0.3], [0.2, 0.6, 0.5]],
                [[0.3, 0.9, 0.0], [1.0, 0.4, 0.2]],
            ]
        ],
        dtype=torch.float32,
    )
    result = build_texture_complexity_map(image, 10.0)
    assert torch.allclose(
        result.texture_map,
        _manual_texture_map(image, 10.0),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_reference_gradient_is_monotone_without_single_image_recomputation() -> None:
    image = torch.zeros((1, 3, 3, 4), dtype=torch.float32)
    image[:, :, :, 2:] = 1.0
    smaller = build_texture_complexity_map(image, 2.0)
    larger = build_texture_complexity_map(image, 8.0)

    assert torch.all(larger.texture_map <= smaller.texture_map)
    assert smaller.reference_gradient == 2.0
    assert larger.reference_gradient == 8.0


def test_texture_digest_binds_exact_protocol_reference_and_map_content() -> None:
    zeros = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
    first = build_texture_complexity_map(zeros, 1.0)
    second = build_texture_complexity_map(zeros, 2.0)
    edge = zeros.clone()
    edge[:, :, :, 2] = 1.0
    changed_output = build_texture_complexity_map(edge, 1.0)

    assert first.texture_map_digest == _expected_texture_digest(first)
    assert second.texture_map_digest == _expected_texture_digest(second)
    assert changed_output.texture_map_digest == _expected_texture_digest(changed_output)
    assert torch.equal(first.texture_map, second.texture_map)
    assert first.texture_map_digest != second.texture_map_digest
    assert first.texture_map_digest != changed_output.texture_map_digest
    assert first.texture_map_digest != tensor_content_sha256(first.texture_map)


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        torch.zeros((1, 3, 2, 2), dtype=torch.bool),
        torch.zeros((1, 3, 2, 2), dtype=torch.int64),
        torch.zeros((1, 3, 2, 2), dtype=torch.complex64),
    ],
)
def test_texture_rejects_non_real_floating_tensor_inputs(invalid: object) -> None:
    with pytest.raises(TypeError):
        build_texture_complexity_map(invalid, 1.0)


@pytest.mark.parametrize(
    "invalid",
    [
        torch.zeros((2, 3, 2, 2), dtype=torch.float32),
        torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        torch.zeros((1, 3, 2), dtype=torch.float32),
        torch.empty((1, 3, 0, 2), dtype=torch.float32),
    ],
)
def test_texture_rejects_invalid_rgb_shapes(invalid: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="B=1 RGB NCHW|dimensions"):
        build_texture_complexity_map(invalid, 1.0)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), -0.01, 1.01])
def test_texture_rejects_nonfinite_or_out_of_range_rgb(invalid_value: float) -> None:
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    image[0, 0, 0, 0] = invalid_value
    with pytest.raises(ValueError, match=r"finite|\[0, 1\]"):
        build_texture_complexity_map(image, 1.0)


@pytest.mark.parametrize(
    "invalid_reference",
    [True, 0.0, -1.0, float("nan"), float("inf"), 1 + 2j],
)
def test_texture_rejects_invalid_reference_gradient(invalid_reference: object) -> None:
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    with pytest.raises((TypeError, ValueError)):
        build_texture_complexity_map(image, invalid_reference)  # type: ignore[arg-type]


def test_texture_rejects_float32_reference_underflow_and_normalization_overflow() -> None:
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    image[:, :, :, 1] = 1.0
    with pytest.raises(ValueError, match="float32"):
        build_texture_complexity_map(image, 1.0e-50)

    smallest_positive = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    ).item()
    with pytest.raises(ValueError, match="reference_normalized_gradient"):
        build_texture_complexity_map(image, smallest_positive)


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        torch.zeros((1, 1, 2, 2), dtype=torch.bool),
        torch.zeros((1, 1, 2, 2), dtype=torch.int64),
        torch.zeros((1, 1, 2, 2), dtype=torch.complex64),
        torch.zeros((1, 2, 2, 2), dtype=torch.float32),
    ],
)
def test_content_resize_rejects_invalid_map_type_or_shape(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _resize_content_map_to_latent(invalid, (2, 2))


@pytest.mark.parametrize(
    "invalid_shape",
    [None, (2,), (2, 3, 4), (0, 2), (-1, 2), (True, 2), (2.0, 3)],
)
def test_content_resize_rejects_invalid_latent_spatial_shape(
    invalid_shape: object,
) -> None:
    content_map = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="two positive integers"):
        _resize_content_map_to_latent(content_map, invalid_shape)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), -0.01, 1.01])
def test_content_resize_rejects_nonfinite_or_out_of_range_map(
    invalid_value: float,
) -> None:
    content_map = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    content_map[0, 0, 0, 0] = invalid_value
    with pytest.raises(ValueError, match=r"finite|\[0, 1\]"):
        _resize_content_map_to_latent(content_map, (3, 4))


def test_content_resize_uses_exact_bilinear_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    content_map = torch.tensor(
        [[[[0.2, 0.4, 0.8], [0.3, 0.6, 0.7]]]],
        dtype=torch.float64,
    )
    expected = functional.interpolate(
        content_map.float(),
        size=(5, 4),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    original_interpolate = functional.interpolate
    calls: list[dict[str, object]] = []

    def recording_interpolate(input_tensor: torch.Tensor, **kwargs: object) -> torch.Tensor:
        calls.append(dict(kwargs))
        return original_interpolate(input_tensor, **kwargs)

    monkeypatch.setattr(functional, "interpolate", recording_interpolate)
    resized = _resize_content_map_to_latent(content_map, (5, 4))

    assert calls == [
        {
            "size": (5, 4),
            "mode": "bilinear",
            "align_corners": False,
            "antialias": False,
        }
    ]
    assert torch.equal(resized, expected)
    assert resized.shape == (1, 1, 5, 4)
    assert resized.dtype == torch.float32
    assert resized.device == content_map.device
    assert resized.min().item() > 0.0
    assert resized.max().item() < 1.0


def test_content_resize_handles_non_square_up_down_and_identity_shapes() -> None:
    content_map = torch.tensor(
        [[[[0.0, 0.25, 0.5], [0.75, 1.0, 0.5]]]],
        dtype=torch.float32,
    )
    upsampled = _resize_content_map_to_latent(content_map, (5, 7))
    downsampled = _resize_content_map_to_latent(upsampled, (2, 3))
    identity = _resize_content_map_to_latent(content_map, (2, 3))

    assert upsampled.shape == (1, 1, 5, 7)
    assert downsampled.shape == content_map.shape
    assert torch.equal(identity, content_map)
    assert torch.all((upsampled >= 0.0) & (upsampled <= 1.0))
    assert torch.all((downsampled >= 0.0) & (downsampled <= 1.0))


def test_content_resize_does_not_mutate_or_impersonate_raw_texture_digest() -> None:
    image = torch.zeros((1, 3, 3, 4), dtype=torch.float32)
    image[:, :, :, 2:] = 1.0
    result = build_texture_complexity_map(image, 8.0)
    original_digest = result.texture_map_digest

    small = _resize_content_map_to_latent(result.texture_map, (2, 2))
    large = _resize_content_map_to_latent(result.texture_map, (5, 7))

    assert result.texture_map_digest == original_digest
    assert small.shape == (1, 1, 2, 2)
    assert large.shape == (1, 1, 5, 7)
    assert result.texture_map_digest != tensor_content_sha256(small)
    assert result.texture_map_digest != tensor_content_sha256(large)
