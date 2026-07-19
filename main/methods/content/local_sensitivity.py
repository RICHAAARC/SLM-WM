"""使用冻结公开探针构造单方向局部敏感性图。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import torch

from main.core.digest import build_stable_digest, tensor_content_sha256
from main.core.keyed_prg import (
    KEYED_PRG_VERSION,
    build_keyed_gaussian_tensor,
    keyed_prg_protocol_record,
)


_PUBLIC_PROBE_KEY_MATERIAL = "semantic_saliency_dual_chain_public_probe_v1"
_PUBLIC_PROBE_PURPOSE = "local_sensitivity_public_probe"
_PUBLIC_PROBE_VERSION = "v1"
_PUBLIC_PROBE_IDENTITY_SCHEMA_VERSION = "public_probe_identity_v1"
_PUBLIC_PROBE_FORMULA_PROTOCOL_VERSION = "frozen_public_probe_direction_v1"
_PROBE_NORMALIZATION_FORMULA_VERSION = (
    "fixed_pairwise_flat_chw_centered_unit_rms_cpu_float32_v1"
)
_LOCAL_SENSITIVITY_RESULT_SCHEMA_VERSION = "local_sensitivity_result_v1"
_LOCAL_SENSITIVITY_FORMULA_PROTOCOL_VERSION = (
    "frozen_public_probe_local_sensitivity_v1"
)
_LOCAL_SENSITIVITY_FORMULA_PROTOCOL = {
    "z10_rms_axes": "single_sample_all_chw_global",
    "probe_relative_step": 1.0e-3,
    "probe_rms_floor": 1.0e-12,
    "image_difference_rms_axes": "rgb_channel_per_spatial_position",
    "output_rule": "clip_dq_over_reference_zero_one",
    "output_resolution": "original_rgb",
    "computation_dtype": "float32",
}


def build_public_probe_identity(model_revision: str) -> dict[str, Any]:
    """Build the single public-probe identity consumed by the Q observation."""

    if type(model_revision) is not str or not model_revision:
        raise ValueError("model_revision must be a non-empty exact string")
    return {
        "prg_version": KEYED_PRG_VERSION,
        "key_material": _PUBLIC_PROBE_KEY_MATERIAL,
        "domain_fields": {
            "purpose": _PUBLIC_PROBE_PURPOSE,
            "model_revision": model_revision,
            "probe_version": _PUBLIC_PROBE_VERSION,
        },
    }


@dataclass(frozen=True)
class LocalSensitivityResult:
    """保存公开探针局部敏感性及其输入、探针和输出身份。"""

    local_sensitivity_map: Any
    reference_sensitivity: float
    public_probe_digest: str
    probe_step: float
    reference_image_digest: str
    perturbed_image_digest: str
    local_sensitivity_map_digest: str


@dataclass(frozen=True)
class _LocalSensitivityObservation:
    """保存一次公开探针测得的未归一化局部敏感性与测量身份。"""

    local_difference_sensitivity: Any
    public_probe_digest: str
    probe_step: float
    reference_image_digest: str
    perturbed_image_digest: str


def _validate_real_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a real floating Tensor")
    return value


def _validate_latent_metadata(value: Any) -> torch.Tensor:
    latent = _validate_real_tensor("current_scheduler_latent", value)
    if latent.ndim != 4 or latent.shape[0] != 1:
        raise ValueError("current_scheduler_latent must have B=1 NCHW shape")
    if any(int(size) <= 0 for size in latent.shape):
        raise ValueError("current_scheduler_latent dimensions must be positive")
    return latent


def _validate_image_metadata(name: str, value: Any) -> torch.Tensor:
    image = _validate_real_tensor(name, value)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(f"{name} must have B=1 RGB NCHW shape")
    if any(int(size) <= 0 for size in image.shape):
        raise ValueError(f"{name} dimensions must be positive")
    return image


def _validate_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_unit_interval(name: str, value: torch.Tensor) -> None:
    _validate_finite_tensor(name, value)
    if bool(((value < 0.0) | (value > 1.0)).any().item()):
        raise ValueError(f"{name} values must lie in [0, 1]")


def _stable_global_rms(name: str, value: torch.Tensor) -> torch.Tensor:
    scale = torch.amax(torch.abs(value))
    safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    normalized = value / safe_scale
    _validate_finite_tensor(f"{name} normalized values", normalized)
    rms = scale * torch.sqrt(torch.mean(torch.square(normalized)))
    _validate_finite_tensor(name, rms)
    return rms


def _fixed_pairwise_float32_mean(
    name: str,
    value: torch.Tensor,
) -> torch.Tensor:
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise ValueError(f"{name} requires a CPU float32 tensor")
    level = value.contiguous().reshape(-1)
    if level.numel() == 0:
        raise ValueError(f"{name} requires at least one value")
    while level.numel() > 1:
        paired_count = level.numel() // 2
        paired_end = paired_count * 2
        paired = level[:paired_end].reshape(paired_count, 2)
        next_level = paired[:, 0] + paired[:, 1]
        if paired_end != level.numel():
            next_level = torch.cat((next_level, level[-1:]))
        _validate_finite_tensor(f"{name} pairwise sum", next_level)
        level = next_level
    count = level.new_tensor(value.numel())
    mean = level[0] / count
    _validate_finite_tensor(name, mean)
    return mean


def _fixed_pairwise_cpu_global_rms(
    name: str,
    value: torch.Tensor,
) -> torch.Tensor:
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise ValueError(f"{name} requires a CPU float32 tensor")
    scale = torch.amax(torch.abs(value))
    safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    normalized = value / safe_scale
    _validate_finite_tensor(f"{name} normalized values", normalized)
    normalized_square = torch.square(normalized)
    _validate_finite_tensor(f"{name} normalized squares", normalized_square)
    mean_square = _fixed_pairwise_float32_mean(
        f"{name} normalized square mean",
        normalized_square,
    )
    rms = scale * torch.sqrt(mean_square)
    _validate_finite_tensor(name, rms)
    return rms


def _stable_rgb_rms(name: str, value: torch.Tensor) -> torch.Tensor:
    scale = torch.amax(torch.abs(value), dim=1, keepdim=True)
    safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    normalized = value / safe_scale
    _validate_finite_tensor(f"{name} normalized values", normalized)
    rms = scale * torch.sqrt(
        torch.mean(torch.square(normalized), dim=1, keepdim=True)
    )
    _validate_finite_tensor(name, rms)
    return rms


def _validate_reference_sensitivity(value: Any) -> tuple[float, torch.Tensor]:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("reference_sensitivity must be a real scalar")
    reference = float(value)
    if not math.isfinite(reference) or reference <= 0.0:
        raise ValueError(
            "reference_sensitivity must be finite and strictly positive"
        )
    reference_float = torch.tensor(reference, dtype=torch.float32)
    if (
        not bool(torch.isfinite(reference_float).item())
        or not bool((reference_float > 0.0).item())
    ):
        raise ValueError(
            "reference_sensitivity must remain finite and strictly positive in float32"
        )
    return reference, reference_float


def _exact_mapping(
    name: str,
    value: Any,
    expected_keys: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(value) != expected_keys:
        raise ValueError(f"{name} must contain the exact governed keys")
    return value


def _validate_public_probe_identity(value: Any) -> dict[str, Any]:
    identity = _exact_mapping(
        "public_probe_identity",
        value,
        frozenset({"prg_version", "key_material", "domain_fields"}),
    )
    domain_fields = _exact_mapping(
        "public_probe_identity.domain_fields",
        identity["domain_fields"],
        frozenset({"purpose", "model_revision", "probe_version"}),
    )
    if identity["prg_version"] != KEYED_PRG_VERSION:
        raise ValueError("public probe prg_version does not match the frozen protocol")
    if identity["key_material"] != _PUBLIC_PROBE_KEY_MATERIAL:
        raise ValueError("public probe key_material does not match the frozen protocol")
    if domain_fields["purpose"] != _PUBLIC_PROBE_PURPOSE:
        raise ValueError("public probe purpose does not match the frozen protocol")
    if domain_fields["probe_version"] != _PUBLIC_PROBE_VERSION:
        raise ValueError("public probe probe_version does not match the frozen protocol")
    if type(domain_fields["model_revision"]) is not str or not domain_fields[
        "model_revision"
    ]:
        raise ValueError("public probe model_revision must be a non-empty exact string")
    return {
        "prg_version": identity["prg_version"],
        "key_material": identity["key_material"],
        "domain_fields": dict(domain_fields),
    }


def _canonical_public_probe(
    latent_shape: tuple[int, ...],
    identity: Mapping[str, Any],
    target_device: torch.device,
) -> tuple[torch.Tensor, str]:
    raw_probe_cpu = build_keyed_gaussian_tensor(
        latent_shape,
        key_material=identity["key_material"],
        domain_fields=identity["domain_fields"],
        prg_version=identity["prg_version"],
    )
    if (
        not isinstance(raw_probe_cpu, torch.Tensor)
        or raw_probe_cpu.device.type != "cpu"
        or raw_probe_cpu.dtype != torch.float32
        or tuple(raw_probe_cpu.shape) != latent_shape
    ):
        raise ValueError("canonical public probe must be a CPU float32 tensor of latent shape")
    _validate_finite_tensor("raw_probe_cpu", raw_probe_cpu)

    global_mean = _fixed_pairwise_float32_mean(
        "raw_probe_global_mean",
        raw_probe_cpu,
    )
    centered_probe_cpu = raw_probe_cpu - global_mean
    _validate_finite_tensor("centered_probe_cpu", centered_probe_cpu)
    centered_rms = _fixed_pairwise_cpu_global_rms(
        "centered_probe_global_rms",
        centered_probe_cpu,
    )
    if not bool((centered_rms > 0.0).item()):
        raise ValueError("centered public probe must have non-zero global RMS")
    normalized_probe_cpu = centered_probe_cpu / centered_rms
    _validate_finite_tensor("normalized_probe_cpu", normalized_probe_cpu)

    normalized_mean = _fixed_pairwise_float32_mean(
        "normalized_probe_global_mean",
        normalized_probe_cpu,
    )
    normalized_rms = _fixed_pairwise_cpu_global_rms(
        "normalized_probe_global_rms", normalized_probe_cpu
    )
    if not bool(torch.isclose(normalized_mean, torch.zeros_like(normalized_mean), atol=1.0e-6, rtol=0.0).item()):
        raise ValueError("normalized public probe must have zero global mean")
    if not bool(torch.isclose(normalized_rms, torch.ones_like(normalized_rms), atol=1.0e-6, rtol=1.0e-5).item()):
        raise ValueError("normalized public probe must have unit global RMS")

    canonical_probe_digest = tensor_content_sha256(normalized_probe_cpu)
    probe_on_device = normalized_probe_cpu.to(
        device=target_device,
        dtype=torch.float32,
    )
    _validate_finite_tensor("probe_on_device", probe_on_device)
    if tensor_content_sha256(probe_on_device) != canonical_probe_digest:
        raise ValueError("public probe content changed while moving to latent device")

    public_probe_digest = build_stable_digest(
        {
            "schema_version": _PUBLIC_PROBE_IDENTITY_SCHEMA_VERSION,
            "formula_protocol_version": _PUBLIC_PROBE_FORMULA_PROTOCOL_VERSION,
            "prg_version": identity["prg_version"],
            "keyed_prg_protocol_digest": keyed_prg_protocol_record(
                identity["prg_version"]
            )["keyed_prg_protocol_digest"],
            "key_material": identity["key_material"],
            "domain_fields": dict(identity["domain_fields"]),
            "latent_shape": list(latent_shape),
            "probe_normalization_formula_version": (
                _PROBE_NORMALIZATION_FORMULA_VERSION
            ),
            "normalized_probe_content_sha256": canonical_probe_digest,
        }
    )
    return probe_on_device, public_probe_digest


def _measure_public_probe_local_sensitivity(
    latent: torch.Tensor,
    reference_image: torch.Tensor,
    vae_decoder: Callable[[Any], Any],
    identity: Mapping[str, Any],
) -> _LocalSensitivityObservation:
    """使用冻结公开方向的一次 VAE 有限差分测量未归一化 ``d_Q``。"""

    _validate_finite_tensor("current_scheduler_latent", latent)
    _validate_unit_interval("decoded_current_image", reference_image)
    reference_image_digest = tensor_content_sha256(reference_image)

    latent_float = latent.to(dtype=torch.float32)
    reference_image_float = reference_image.to(dtype=torch.float32)
    _validate_finite_tensor("current_scheduler_latent after float32 cast", latent_float)
    _validate_unit_interval(
        "decoded_current_image after float32 cast", reference_image_float
    )
    probe, public_probe_digest = _canonical_public_probe(
        tuple(int(size) for size in latent.shape),
        identity,
        latent.device,
    )

    latent_rms = _stable_global_rms("current_scheduler_latent_global_rms", latent_float)
    rms_floor = latent_float.new_tensor(
        _LOCAL_SENSITIVITY_FORMULA_PROTOCOL["probe_rms_floor"]
    )
    relative_step = latent_float.new_tensor(
        _LOCAL_SENSITIVITY_FORMULA_PROTOCOL["probe_relative_step"]
    )
    probe_step_tensor = relative_step * torch.maximum(latent_rms, rms_floor)
    _validate_finite_tensor("probe_step", probe_step_tensor)
    if not bool((probe_step_tensor > 0.0).item()):
        raise ValueError("probe_step must be strictly positive")
    probe_step = float(probe_step_tensor.item())

    perturbed_latent = latent_float + probe_step_tensor * probe
    _validate_finite_tensor("perturbed_latent", perturbed_latent)
    perturbed_image = _validate_image_metadata(
        "perturbed_image", vae_decoder(perturbed_latent)
    )
    if perturbed_image.device != reference_image.device:
        raise ValueError("reference and perturbed images must use the same device")
    if perturbed_image.shape != reference_image.shape:
        raise ValueError("reference and perturbed image shapes must match")
    _validate_unit_interval("perturbed_image", perturbed_image)
    perturbed_image_digest = tensor_content_sha256(perturbed_image)

    perturbed_image_float = perturbed_image.to(dtype=torch.float32)
    _validate_unit_interval("perturbed_image after float32 cast", perturbed_image_float)
    image_difference = perturbed_image_float - reference_image_float
    _validate_finite_tensor("image_difference", image_difference)
    rgb_rms = _stable_rgb_rms("image_difference_rgb_rms", image_difference)
    local_difference_sensitivity = rgb_rms / probe_step_tensor
    _validate_finite_tensor(
        "local_difference_sensitivity", local_difference_sensitivity
    )
    return _LocalSensitivityObservation(
        local_difference_sensitivity=local_difference_sensitivity,
        public_probe_digest=public_probe_digest,
        probe_step=probe_step,
        reference_image_digest=reference_image_digest,
        perturbed_image_digest=perturbed_image_digest,
    )


def build_public_probe_local_sensitivity_map(
    current_scheduler_latent: Any,
    decoded_current_image: Any,
    vae_decoder: Callable[[Any], Any],
    public_probe_identity: Any,
    reference_sensitivity: float,
) -> LocalSensitivityResult:
    """按冻结公开方向的一次 VAE 有限差分构造原始 RGB 分辨率 ``Q``。"""

    latent = _validate_latent_metadata(current_scheduler_latent)
    reference_image = _validate_image_metadata(
        "decoded_current_image", decoded_current_image
    )
    if latent.device != reference_image.device:
        raise ValueError("latent and decoded_current_image must use the same device")
    if not callable(vae_decoder):
        raise TypeError("vae_decoder must be callable")
    identity = _validate_public_probe_identity(public_probe_identity)
    reference, reference_float_cpu = _validate_reference_sensitivity(
        reference_sensitivity
    )
    observation = _measure_public_probe_local_sensitivity(
        latent,
        reference_image,
        vae_decoder,
        identity,
    )
    reference_float = reference_float_cpu.to(device=latent.device)
    reference_normalized = (
        observation.local_difference_sensitivity / reference_float
    )
    _validate_finite_tensor("reference_normalized_sensitivity", reference_normalized)
    local_sensitivity_map = torch.clamp(reference_normalized, min=0.0, max=1.0)
    _validate_unit_interval("local_sensitivity_map", local_sensitivity_map)

    local_sensitivity_map_digest = build_stable_digest(
        {
            "schema_version": _LOCAL_SENSITIVITY_RESULT_SCHEMA_VERSION,
            "formula_protocol_version": (
                _LOCAL_SENSITIVITY_FORMULA_PROTOCOL_VERSION
            ),
            "formula_protocol": _LOCAL_SENSITIVITY_FORMULA_PROTOCOL,
            "public_probe_digest": observation.public_probe_digest,
            "probe_step": observation.probe_step,
            "reference_sensitivity": reference,
            "reference_image_digest": observation.reference_image_digest,
            "perturbed_image_digest": observation.perturbed_image_digest,
            "local_sensitivity_map_content_sha256": tensor_content_sha256(
                local_sensitivity_map
            ),
        }
    )
    return LocalSensitivityResult(
        local_sensitivity_map=local_sensitivity_map,
        reference_sensitivity=reference,
        public_probe_digest=observation.public_probe_digest,
        probe_step=observation.probe_step,
        reference_image_digest=observation.reference_image_digest,
        perturbed_image_digest=observation.perturbed_image_digest,
        local_sensitivity_map_digest=local_sensitivity_map_digest,
    )
