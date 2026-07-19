from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import math
from typing import Any

import pytest
import torch
import torch.nn.functional as functional

import main.methods.content.local_sensitivity as local_sensitivity_module
from main.core.digest import build_stable_digest, tensor_content_sha256
from main.core.keyed_prg import (
    KEYED_PRG_VERSION,
    build_keyed_gaussian_tensor,
    keyed_prg_protocol_record,
)
from main.methods.content import (
    LocalSensitivityResult,
    build_public_probe_local_sensitivity_map,
)
from main.methods.content.routing import _resize_content_map_to_latent


pytestmark = pytest.mark.unit


_KEY_MATERIAL = "semantic_saliency_dual_chain_public_probe_v1"
_DOMAIN_FIELDS = {
    "purpose": "local_sensitivity_public_probe",
    "model_revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
    "probe_version": "v1",
}
_FORMULA_PROTOCOL = {
    "z10_rms_axes": "single_sample_all_chw_global",
    "probe_relative_step": 1.0e-3,
    "probe_rms_floor": 1.0e-12,
    "image_difference_rms_axes": "rgb_channel_per_spatial_position",
    "output_rule": "clip_dq_over_reference_zero_one",
    "output_resolution": "original_rgb",
    "computation_dtype": "float32",
}


def _identity() -> dict[str, Any]:
    return {
        "prg_version": KEYED_PRG_VERSION,
        "key_material": _KEY_MATERIAL,
        "domain_fields": dict(_DOMAIN_FIELDS),
    }


def test_public_probe_identity_builder_matches_the_frozen_q_input() -> None:
    """Runtime integration must not rebuild the public probe identity ad hoc."""

    identity = local_sensitivity_module.build_public_probe_identity(
        _DOMAIN_FIELDS["model_revision"]
    )
    assert identity == _identity()
    with pytest.raises(ValueError):
        local_sensitivity_module.build_public_probe_identity("")


def _stable_global_rms(value: torch.Tensor) -> torch.Tensor:
    scale = torch.amax(torch.abs(value))
    safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    return scale * torch.sqrt(torch.mean(torch.square(value / safe_scale)))


def _independent_pairwise_float32_mean(value: torch.Tensor) -> torch.Tensor:
    level = value.contiguous().reshape(-1)
    while level.numel() > 1:
        pair_count = level.numel() // 2
        pair_end = pair_count * 2
        pair_sums = level[:pair_end:2] + level[1:pair_end:2]
        level = (
            torch.cat((pair_sums, level[-1:]))
            if pair_end != level.numel()
            else pair_sums
        )
    return level[0] / torch.tensor(value.numel(), dtype=torch.float32)


def _independent_pairwise_cpu_global_rms(value: torch.Tensor) -> torch.Tensor:
    scale = torch.amax(torch.abs(value))
    safe_scale = scale if bool((scale > 0.0).item()) else torch.ones_like(scale)
    normalized = value / safe_scale
    return scale * torch.sqrt(
        _independent_pairwise_float32_mean(normalized * normalized)
    )


def _stable_rgb_rms(value: torch.Tensor) -> torch.Tensor:
    scale = torch.amax(torch.abs(value), dim=1, keepdim=True)
    safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    return scale * torch.sqrt(
        torch.mean(torch.square(value / safe_scale), dim=1, keepdim=True)
    )


def _canonical_probe(
    shape: tuple[int, ...],
    identity: dict[str, Any] | None = None,
) -> torch.Tensor:
    resolved = _identity() if identity is None else identity
    raw = build_keyed_gaussian_tensor(
        shape,
        key_material=resolved["key_material"],
        domain_fields=resolved["domain_fields"],
        prg_version=resolved["prg_version"],
    )
    centered = raw - _independent_pairwise_float32_mean(raw)
    return centered / _independent_pairwise_cpu_global_rms(centered)


def _probe_digest(probe: torch.Tensor, shape: tuple[int, ...], identity: dict[str, Any]) -> str:
    return build_stable_digest(
        {
            "schema_version": "public_probe_identity_v1",
            "formula_protocol_version": "frozen_public_probe_direction_v1",
            "prg_version": identity["prg_version"],
            "keyed_prg_protocol_digest": keyed_prg_protocol_record(
                identity["prg_version"]
            )["keyed_prg_protocol_digest"],
            "key_material": identity["key_material"],
            "domain_fields": identity["domain_fields"],
            "latent_shape": list(shape),
            "probe_normalization_formula_version": (
                "fixed_pairwise_flat_chw_centered_unit_rms_cpu_float32_v1"
            ),
            "normalized_probe_content_sha256": tensor_content_sha256(probe),
        }
    )


def _decoder_formula(latent: torch.Tensor, output_shape: tuple[int, int]) -> torch.Tensor:
    signal = functional.interpolate(
        latent[:, :1].float(),
        size=output_shape,
        mode="bilinear",
        align_corners=False,
    )
    channels = torch.cat((signal, 0.5 * signal, -0.25 * signal), dim=1)
    return 0.5 + 0.1 * channels


class _RecordingDecoder:
    def __init__(self, output_shape: tuple[int, int]) -> None:
        self.output_shape = output_shape
        self.inputs: list[torch.Tensor] = []

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        self.inputs.append(latent.detach().clone())
        return _decoder_formula(latent, self.output_shape)


def _example_latent() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[-0.8, -0.2, 0.3], [0.7, 0.1, -0.4]],
                [[0.6, -0.5, 0.2], [-0.1, 0.9, -0.7]],
            ]
        ],
        dtype=torch.float64,
    )


def test_local_sensitivity_result_is_frozen_and_has_exact_fields() -> None:
    assert tuple(field.name for field in fields(LocalSensitivityResult)) == (
        "local_sensitivity_map",
        "reference_sensitivity",
        "public_probe_digest",
        "probe_step",
        "reference_image_digest",
        "perturbed_image_digest",
        "local_sensitivity_map_digest",
    )
    latent = _example_latent()
    reference = _decoder_formula(latent.float(), (3, 5))
    result = build_public_probe_local_sensitivity_map(
        latent,
        reference,
        _RecordingDecoder((3, 5)),
        _identity(),
        1.0,
    )
    with pytest.raises(FrozenInstanceError):
        result.probe_step = 0.0  # type: ignore[misc]


def test_local_sensitivity_matches_frozen_formula_digests_and_single_decode() -> None:
    latent = _example_latent()
    latent_float = latent.float()
    output_shape = (3, 5)
    reference_image = _decoder_formula(latent_float, output_shape).to(torch.float64)
    decoder = _RecordingDecoder(output_shape)
    identity = _identity()
    reference_sensitivity = 0.75

    result = build_public_probe_local_sensitivity_map(
        latent,
        reference_image,
        decoder,
        identity,
        reference_sensitivity,
    )
    repeated = build_public_probe_local_sensitivity_map(
        latent,
        reference_image,
        _RecordingDecoder(output_shape),
        identity,
        reference_sensitivity,
    )

    probe = _canonical_probe(tuple(latent.shape), identity)
    latent_rms = _stable_global_rms(latent_float)
    step_tensor = latent_float.new_tensor(1.0e-3) * torch.maximum(
        latent_rms,
        latent_float.new_tensor(1.0e-12),
    )
    expected_step = float(step_tensor.item())
    expected_perturbed_latent = latent_float + step_tensor * probe
    expected_perturbed_image = _decoder_formula(
        expected_perturbed_latent,
        output_shape,
    )
    difference = expected_perturbed_image - reference_image.float()
    expected_map = torch.clamp(
        _stable_rgb_rms(difference) / step_tensor / reference_sensitivity,
        0.0,
        1.0,
    )
    expected_probe_digest = _probe_digest(probe, tuple(latent.shape), identity)
    expected_reference_digest = tensor_content_sha256(reference_image)
    expected_perturbed_digest = tensor_content_sha256(expected_perturbed_image)
    expected_map_digest = build_stable_digest(
        {
            "schema_version": "local_sensitivity_result_v1",
            "formula_protocol_version": (
                "frozen_public_probe_local_sensitivity_v1"
            ),
            "formula_protocol": _FORMULA_PROTOCOL,
            "public_probe_digest": expected_probe_digest,
            "probe_step": expected_step,
            "reference_sensitivity": reference_sensitivity,
            "reference_image_digest": expected_reference_digest,
            "perturbed_image_digest": expected_perturbed_digest,
            "local_sensitivity_map_content_sha256": tensor_content_sha256(
                expected_map
            ),
        }
    )

    assert len(decoder.inputs) == 1
    assert torch.equal(decoder.inputs[0], expected_perturbed_latent)
    assert result.local_sensitivity_map.shape == (1, 1, *output_shape)
    assert result.local_sensitivity_map.dtype == torch.float32
    assert result.local_sensitivity_map.device == latent.device
    assert torch.equal(result.local_sensitivity_map, expected_map)
    assert result.probe_step == expected_step
    assert result.public_probe_digest == expected_probe_digest
    assert result.reference_image_digest == expected_reference_digest
    assert result.perturbed_image_digest == expected_perturbed_digest
    assert result.local_sensitivity_map_digest == expected_map_digest
    assert result.local_sensitivity_map_digest != tensor_content_sha256(expected_map)
    assert torch.equal(repeated.local_sensitivity_map, result.local_sensitivity_map)
    assert repeated.public_probe_digest == result.public_probe_digest
    assert repeated.local_sensitivity_map_digest == result.local_sensitivity_map_digest


def test_probe_uses_global_chw_centering_and_rms_on_canonical_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = torch.tensor(
        [[[[1.0, 2.0], [4.0, 8.0]], [[16.0, 32.0], [64.0, 128.0]]]],
        dtype=torch.float32,
    )
    captured: dict[str, Any] = {}

    def capture_prg(
        shape: tuple[int, ...],
        key_material: str,
        domain_fields: dict[str, Any],
        prg_version: str,
    ) -> torch.Tensor:
        captured.update(
            shape=shape,
            key_material=key_material,
            domain_fields=dict(domain_fields),
            prg_version=prg_version,
        )
        return raw.clone()

    decoder_inputs: list[torch.Tensor] = []

    def decoder(value: torch.Tensor) -> torch.Tensor:
        decoder_inputs.append(value.detach().clone())
        return torch.zeros((1, 3, 2, 2), dtype=torch.float32)

    monkeypatch.setattr(
        local_sensitivity_module,
        "build_keyed_gaussian_tensor",
        capture_prg,
    )
    latent = torch.zeros_like(raw)
    result = build_public_probe_local_sensitivity_map(
        latent,
        torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        decoder,
        _identity(),
        1.0,
    )

    expected_centered = raw - _independent_pairwise_float32_mean(raw)
    expected_probe = expected_centered / _independent_pairwise_cpu_global_rms(
        expected_centered
    )
    expected_perturbed = (
        torch.tensor(result.probe_step, dtype=torch.float32) * expected_probe
    )
    per_channel = raw - raw.mean(dim=(2, 3), keepdim=True)

    assert captured == {
        "shape": tuple(latent.shape),
        "key_material": _KEY_MATERIAL,
        "domain_fields": _DOMAIN_FIELDS,
        "prg_version": KEYED_PRG_VERSION,
    }
    assert torch.equal(decoder_inputs[0], expected_perturbed)
    assert _independent_pairwise_float32_mean(expected_probe).item() == pytest.approx(
        0.0, abs=1.0e-6
    )
    assert _independent_pairwise_cpu_global_rms(expected_probe).item() == pytest.approx(
        1.0, rel=1.0e-6
    )
    assert not torch.equal(expected_centered, per_channel)


def test_public_probe_normalization_is_exact_across_torch_thread_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    element_count = 16 * 128 * 128
    positions = torch.arange(element_count, dtype=torch.float32)
    dynamic_scale = torch.where(
        positions.to(torch.int64) % 4 == 0,
        torch.tensor(1.0e6, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    raw = (torch.sin(positions * 1.0e-3) * dynamic_scale).reshape(
        1, 16, 128, 128
    )
    expected_centered = raw - _independent_pairwise_float32_mean(raw)
    expected_probe = expected_centered / _independent_pairwise_cpu_global_rms(
        expected_centered
    )
    expected_digest = _probe_digest(expected_probe, tuple(raw.shape), _identity())

    monkeypatch.setattr(
        local_sensitivity_module,
        "build_keyed_gaussian_tensor",
        lambda shape, **_: raw.clone(),
    )
    latent = torch.zeros_like(raw)
    reference = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    original_thread_count = torch.get_num_threads()
    multi_thread_count = min(8, max(2, original_thread_count))
    runs: list[tuple[torch.Tensor, LocalSensitivityResult]] = []
    try:
        for thread_count in (1, multi_thread_count):
            torch.set_num_threads(thread_count)
            decoder_inputs: list[torch.Tensor] = []

            def decoder(value: torch.Tensor) -> torch.Tensor:
                decoder_inputs.append(value.detach().clone())
                return reference.clone()

            result = build_public_probe_local_sensitivity_map(
                latent,
                reference,
                decoder,
                _identity(),
                1.0,
            )
            assert len(decoder_inputs) == 1
            expected_perturbed = (
                torch.tensor(result.probe_step, dtype=torch.float32)
                * expected_probe
            )
            assert torch.equal(decoder_inputs[0], expected_perturbed)
            assert result.public_probe_digest == expected_digest
            runs.append((decoder_inputs[0], result))
    finally:
        torch.set_num_threads(original_thread_count)

    assert torch.equal(runs[0][0], runs[1][0])
    assert runs[0][1].public_probe_digest == runs[1][1].public_probe_digest


def test_global_latent_rms_and_per_pixel_rgb_rms_use_distinct_frozen_axes() -> None:
    latent = _example_latent().float()
    reference = _decoder_formula(latent, (4, 5))
    decoder = _RecordingDecoder((4, 5))
    result = build_public_probe_local_sensitivity_map(
        latent,
        reference,
        decoder,
        _identity(),
        0.5,
    )

    expected_step = float(
        (
            latent.new_tensor(1.0e-3)
            * torch.maximum(
                _stable_global_rms(latent), latent.new_tensor(1.0e-12)
            )
        ).item()
    )
    difference = _decoder_formula(decoder.inputs[0], (4, 5)) - reference
    expected = torch.clamp(
        _stable_rgb_rms(difference) / expected_step / 0.5,
        0.0,
        1.0,
    )

    assert result.probe_step == expected_step
    assert result.local_sensitivity_map.shape == (1, 1, 4, 5)
    assert torch.equal(result.local_sensitivity_map, expected)
    assert torch.unique(result.local_sensitivity_map).numel() > 1


def test_zero_latent_uses_actual_float32_probe_step_floor() -> None:
    latent = torch.zeros((1, 2, 2, 3), dtype=torch.float64)
    reference = torch.zeros((1, 3, 4, 5), dtype=torch.float32)
    captured: list[torch.Tensor] = []

    def decoder(value: torch.Tensor) -> torch.Tensor:
        captured.append(value.detach().clone())
        return reference.clone()

    result = build_public_probe_local_sensitivity_map(
        latent,
        reference,
        decoder,
        _identity(),
        1.0,
    )
    expected = float(
        (torch.tensor(1.0e-3, dtype=torch.float32) * torch.tensor(1.0e-12)).item()
    )
    assert result.probe_step == expected
    assert result.probe_step == pytest.approx(1.0e-15, rel=1.0e-6)
    assert torch.count_nonzero(captured[0]).item() > 0
    assert torch.count_nonzero(result.local_sensitivity_map).item() == 0


def test_reference_sensitivity_is_monotone_and_digest_binds_reference() -> None:
    latent = _example_latent().float()
    reference = _decoder_formula(latent, (3, 5))
    smaller = build_public_probe_local_sensitivity_map(
        latent,
        reference,
        _RecordingDecoder((3, 5)),
        _identity(),
        0.5,
    )
    larger = build_public_probe_local_sensitivity_map(
        latent,
        reference,
        _RecordingDecoder((3, 5)),
        _identity(),
        2.0,
    )

    assert torch.all(larger.local_sensitivity_map <= smaller.local_sensitivity_map)
    assert smaller.public_probe_digest == larger.public_probe_digest
    assert smaller.local_sensitivity_map_digest != larger.local_sensitivity_map_digest


def test_image_digests_bind_validated_original_tensors_not_float32_copies() -> None:
    latent = _example_latent()
    reference = _decoder_formula(latent.float(), (3, 5)).double()

    def decoder(value: torch.Tensor) -> torch.Tensor:
        return _decoder_formula(value, (3, 5)).double()

    result = build_public_probe_local_sensitivity_map(
        latent,
        reference,
        decoder,
        _identity(),
        1.0,
    )
    expected_perturbed = decoder(
        latent.float() + result.probe_step * _canonical_probe(tuple(latent.shape))
    )

    assert result.reference_image_digest == tensor_content_sha256(reference)
    assert result.reference_image_digest != tensor_content_sha256(reference.float())
    assert result.perturbed_image_digest == tensor_content_sha256(expected_perturbed)
    assert result.perturbed_image_digest != tensor_content_sha256(
        expected_perturbed.float()
    )


def test_model_revision_and_shape_change_public_probe_identity() -> None:
    latent = _example_latent().float()
    reference = _decoder_formula(latent, (3, 5))
    first = build_public_probe_local_sensitivity_map(
        latent,
        reference,
        _RecordingDecoder((3, 5)),
        _identity(),
        1.0,
    )
    changed_identity = _identity()
    changed_identity["domain_fields"]["model_revision"] = "different-exact-revision"
    changed_revision = build_public_probe_local_sensitivity_map(
        latent,
        reference,
        _RecordingDecoder((3, 5)),
        changed_identity,
        1.0,
    )
    wider = torch.cat((latent, latent[:, :, :, :1]), dim=-1)
    wider_reference = _decoder_formula(wider, (3, 5))
    changed_shape = build_public_probe_local_sensitivity_map(
        wider,
        wider_reference,
        _RecordingDecoder((3, 5)),
        _identity(),
        1.0,
    )

    assert first.public_probe_digest != changed_revision.public_probe_digest
    assert first.public_probe_digest != changed_shape.public_probe_digest


def test_raw_q_can_resize_without_changing_or_impersonating_its_digest() -> None:
    latent = _example_latent().float()
    reference = _decoder_formula(latent, (4, 5))
    result = build_public_probe_local_sensitivity_map(
        latent,
        reference,
        _RecordingDecoder((4, 5)),
        _identity(),
        1.0,
    )
    original_digest = result.local_sensitivity_map_digest
    resized = _resize_content_map_to_latent(result.local_sensitivity_map, (2, 3))

    assert resized.shape == (1, 1, 2, 3)
    assert result.local_sensitivity_map.shape == (1, 1, 4, 5)
    assert result.local_sensitivity_map_digest == original_digest
    assert original_digest != tensor_content_sha256(resized)


@pytest.mark.parametrize(
    ("scope", "extra_key"),
    [
        ("top", "watermark_key"),
        ("top", "prompt"),
        ("domain", "sample_id"),
        ("domain", "generation_seed"),
        ("domain", "attack_label"),
        ("domain", "operator"),
        ("domain", "model_id"),
    ],
)
def test_public_probe_identity_rejects_every_unauthorized_field(
    scope: str,
    extra_key: str,
) -> None:
    identity = _identity()
    target = identity if scope == "top" else identity["domain_fields"]
    target[extra_key] = "forbidden"
    latent = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="exact governed keys"):
        build_public_probe_local_sensitivity_map(
            latent, image, lambda _: image, identity, 1.0
        )


@pytest.mark.parametrize(
    "mutation",
    [
        ("remove_top", "prg_version"),
        ("remove_domain", "purpose"),
        ("replace", "prg_version"),
        ("replace", "key_material"),
        ("replace_domain", "purpose"),
        ("replace_domain", "probe_version"),
        ("replace_domain", "model_revision"),
    ],
)
def test_public_probe_identity_rejects_missing_or_changed_frozen_values(
    mutation: tuple[str, str],
) -> None:
    identity = _identity()
    operation, key = mutation
    if operation == "remove_top":
        del identity[key]
    elif operation == "remove_domain":
        del identity["domain_fields"][key]
    elif operation == "replace":
        identity[key] = "changed"
    elif operation == "replace_domain":
        identity["domain_fields"][key] = "" if key == "model_revision" else "changed"
    latent = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    with pytest.raises((TypeError, ValueError)):
        build_public_probe_local_sensitivity_map(
            latent, image, lambda _: image, identity, 1.0
        )


@pytest.mark.parametrize("invalid_identity", [None, "probe", [], 1])
def test_public_probe_identity_requires_a_mapping(invalid_identity: object) -> None:
    latent = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    with pytest.raises(TypeError, match="mapping"):
        build_public_probe_local_sensitivity_map(
            latent, image, lambda _: image, invalid_identity, 1.0
        )


def test_public_probe_rejects_single_element_and_constant_prg_zero_energy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    single_latent = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    single_image = torch.zeros((1, 3, 1, 1), dtype=torch.float32)
    with pytest.raises(ValueError, match="non-zero global RMS"):
        build_public_probe_local_sensitivity_map(
            single_latent,
            single_image,
            lambda _: single_image,
            _identity(),
            1.0,
        )

    monkeypatch.setattr(
        local_sensitivity_module,
        "build_keyed_gaussian_tensor",
        lambda shape, **_: torch.ones(shape, dtype=torch.float32),
    )
    latent = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="non-zero global RMS"):
        build_public_probe_local_sensitivity_map(
            latent, image, lambda _: image, _identity(), 1.0
        )


def test_probe_device_move_digest_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_digest = tensor_content_sha256
    probe_digest_calls = 0

    def mismatching_digest(value: torch.Tensor) -> str:
        nonlocal probe_digest_calls
        digest = original_digest(value)
        if tuple(value.shape) == (1, 2, 2, 2):
            probe_digest_calls += 1
            if probe_digest_calls == 2:
                return "0" * 64
        return digest

    monkeypatch.setattr(
        local_sensitivity_module,
        "tensor_content_sha256",
        mismatching_digest,
    )
    latent = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="changed while moving"):
        build_public_probe_local_sensitivity_map(
            latent, image, lambda _: image, _identity(), 1.0
        )


@pytest.mark.parametrize(
    "invalid",
    [
        "not-a-tensor",
        torch.ones((1, 2, 2, 2), dtype=torch.bool),
        torch.ones((1, 2, 2, 2), dtype=torch.int64),
        torch.ones((1, 2, 2, 2), dtype=torch.complex64),
    ],
)
def test_local_sensitivity_rejects_non_real_latent(invalid: object) -> None:
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    with pytest.raises(TypeError):
        build_public_probe_local_sensitivity_map(
            invalid, image, lambda _: image, _identity(), 1.0
        )


@pytest.mark.parametrize(
    "invalid",
    [
        torch.zeros((2, 2, 2, 2), dtype=torch.float32),
        torch.zeros((1, 2, 2), dtype=torch.float32),
        torch.empty((1, 2, 0, 2), dtype=torch.float32),
    ],
)
def test_local_sensitivity_rejects_invalid_latent_shape(invalid: torch.Tensor) -> None:
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="B=1 NCHW|dimensions"):
        build_public_probe_local_sensitivity_map(
            invalid, image, lambda _: image, _identity(), 1.0
        )


@pytest.mark.parametrize(
    "invalid",
    [
        torch.zeros((2, 3, 2, 2), dtype=torch.float32),
        torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        torch.zeros((1, 3, 2), dtype=torch.float32),
        torch.zeros((1, 3, 2, 2), dtype=torch.bool),
    ],
)
def test_local_sensitivity_rejects_invalid_reference_image(invalid: torch.Tensor) -> None:
    latent = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    with pytest.raises((TypeError, ValueError)):
        build_public_probe_local_sensitivity_map(
            latent, invalid, lambda _: invalid, _identity(), 1.0
        )


@pytest.mark.parametrize("invalid", [True, 0.0, -1.0, math.inf, math.nan, 1 + 2j])
def test_local_sensitivity_rejects_invalid_reference_sensitivity(invalid: object) -> None:
    latent = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    with pytest.raises((TypeError, ValueError)):
        build_public_probe_local_sensitivity_map(
            latent, image, lambda _: image, _identity(), invalid  # type: ignore[arg-type]
        )


def test_reference_sensitivity_must_remain_positive_finite_in_float32() -> None:
    latent = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    for invalid in (1.0e-50, 1.0e50):
        with pytest.raises(ValueError, match="float32"):
            build_public_probe_local_sensitivity_map(
                latent, image, lambda _: image, _identity(), invalid
            )


def test_local_sensitivity_rejects_nonfinite_latent_and_out_of_range_reference() -> None:
    latent = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    nonfinite_latent = latent.clone()
    nonfinite_latent[0, 0, 0, 0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        build_public_probe_local_sensitivity_map(
            nonfinite_latent, image, lambda _: image, _identity(), 1.0
        )
    for value in (math.nan, math.inf, -0.1, 1.1):
        invalid_image = image.clone()
        invalid_image[0, 0, 0, 0] = value
        with pytest.raises(ValueError, match=r"finite|\[0, 1\]"):
            build_public_probe_local_sensitivity_map(
                latent, invalid_image, lambda _: image, _identity(), 1.0
            )


@pytest.mark.parametrize("invalid_value", [math.nan, math.inf, -0.1, 1.1])
def test_decoder_nonfinite_or_out_of_range_output_fails_closed(
    invalid_value: float,
) -> None:
    latent = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)

    def decoder(_: torch.Tensor) -> torch.Tensor:
        output = image.clone()
        output[0, 0, 0, 0] = invalid_value
        return output

    with pytest.raises(ValueError, match=r"finite|\[0, 1\]"):
        build_public_probe_local_sensitivity_map(
            latent, image, decoder, _identity(), 1.0
        )


def test_decoder_type_shape_and_callable_boundaries_fail_closed() -> None:
    latent = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    with pytest.raises(TypeError, match="callable"):
        build_public_probe_local_sensitivity_map(
            latent, image, None, _identity(), 1.0  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="Tensor"):
        build_public_probe_local_sensitivity_map(
            latent, image, lambda _: "image", _identity(), 1.0
        )
    with pytest.raises(ValueError, match="shapes must match"):
        build_public_probe_local_sensitivity_map(
            latent,
            image,
            lambda _: torch.zeros((1, 3, 3, 2), dtype=torch.float32),
            _identity(),
            1.0,
        )


def test_float64_cast_overflow_and_perturbation_overflow_fail_closed() -> None:
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    cast_overflow = torch.full(
        (1, 2, 2, 2),
        torch.finfo(torch.float64).max,
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="after float32 cast"):
        build_public_probe_local_sensitivity_map(
            cast_overflow, image, lambda _: image, _identity(), 1.0
        )

    perturbation_overflow = torch.full(
        (1, 2, 2, 2),
        torch.finfo(torch.float32).max,
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="perturbed_latent"):
        build_public_probe_local_sensitivity_map(
            perturbation_overflow, image, lambda _: image, _identity(), 1.0
        )


def test_device_mismatch_is_rejected_before_reading_reference_contents() -> None:
    latent = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    meta_image = torch.empty((1, 3, 2, 2), dtype=torch.float32, device="meta")
    with pytest.raises(ValueError, match="same device"):
        build_public_probe_local_sensitivity_map(
            latent, meta_image, lambda _: meta_image, _identity(), 1.0
        )
