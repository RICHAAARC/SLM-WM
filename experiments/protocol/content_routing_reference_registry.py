"""Fail-closed loading for the unique content-routing reference registry."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any

from experiments.protocol.content_routing_reference_quantile import (
    ContentRoutingReferenceScalars,
)
from main.core.digest import build_stable_digest, stable_json_dumps


__all__ = ["load_content_routing_reference_registry"]


_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_REGISTRY_PATH = _ROOT / "configs/content_routing_reference_registry.json"
_EXPECTED_OPEN_FLAGS = (
    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

_REFERENCE_REGISTRY_SCHEMA = {
    "registry_schema": "content_routing_reference_registry_v1",
    "model_id": "stabilityai/stable-diffusion-3.5-medium",
    "model_revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
    "quantile_algorithm": (
        "nearest_rank_full_sort_exact_rational_19_over_20_v1"
    ),
    "top_fields": (
        "registry_schema",
        "method_parameter_partition_id",
        "method_parameter_prompt_list_digest",
        "method_parameter_seed_list_digest_random",
        "method_parameter_sample_count",
        "formal_execution_lock_digest",
        "dependency_profile_digest",
        "model_id",
        "model_revision",
        "runtime_component_identity_digest",
        "content_routing_reference_quantile_algorithm",
        "content_routing_reference_populations",
        "reference_gradient",
        "reference_response",
        "reference_sensitivity",
        "content_routing_reference_registry_digest",
    ),
    "population_order": (
        "gradient_magnitude_rgb_pre_interpolation",
        "latent_response",
        "local_sensitivity_rgb_pre_interpolation",
    ),
    "population_fields": (
        "reference_observation_kind",
        "reference_observation_positive_value_count",
        "reference_observation_member_records_digest",
        "tensor_content_sha256",
    ),
    "member_projection_fields": (
        "reference_observation_kind",
        "reference_observation_member_sequence_index",
        "generation_input_identity_digest",
        "tensor_content_sha256",
    ),
    "runtime_identity_payload_fields": (
        "model_id",
        "model_revision",
        "dependency_profile_digest",
        "formal_execution_lock_digest",
        "vae_preprocess_identity_digest",
        "scheduler_identity_digest",
        "content_observation_formula_identity_digest",
    ),
    "scalar_fields": (
        "reference_gradient",
        "reference_response",
        "reference_sensitivity",
    ),
}


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _require_nonempty_string(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(character in value for character in "\0\r\n\t")
    ):
        raise ValueError(f"{label} must be a nonempty exact string")
    return value


def _require_positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a strict positive integer")
    return value


def _require_exact_binary32(value: Any, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be a finite positive JSON float")
    try:
        converted = struct.unpack(">f", struct.pack(">f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"{label} is not representable as binary32") from exc
    if not math.isfinite(converted) or converted <= 0.0 or converted != value:
        raise ValueError(f"{label} must reload exactly as positive binary32")
    return value


def _read_fixed_registry_bytes() -> bytes:
    try:
        descriptor = os.open(_REFERENCE_REGISTRY_PATH, _EXPECTED_OPEN_FLAGS)
    except OSError as exc:
        raise ValueError("content-routing reference registry cannot be opened") from exc

    try:
        try:
            metadata = os.fstat(descriptor)
        except OSError as exc:
            raise ValueError(
                "content-routing reference registry cannot be inspected"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("content-routing reference registry must be a regular file")
        try:
            payload = os.read(descriptor, metadata.st_size + 1)
        except OSError as exc:
            raise ValueError("content-routing reference registry cannot be read") from exc
        if len(payload) != metadata.st_size:
            raise ValueError("content-routing reference registry changed while reading")
        return payload
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            raise ValueError("content-routing reference registry cannot be closed") from exc


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_object(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("content-routing reference registry JSON is invalid") from exc
    if type(value) is not dict:
        raise ValueError("content-routing reference registry must be an exact object")
    return value


def _validate_registry_structure(
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    schema = _REFERENCE_REGISTRY_SCHEMA
    if type(registry) is not dict or set(registry) != set(schema["top_fields"]):
        raise ValueError("content-routing reference registry has invalid fields")
    if registry["registry_schema"] != schema["registry_schema"]:
        raise ValueError("content-routing reference registry schema is invalid")
    _require_nonempty_string(
        registry["method_parameter_partition_id"],
        label="method_parameter_partition_id",
    )
    for field_name in (
        "method_parameter_prompt_list_digest",
        "method_parameter_seed_list_digest_random",
        "formal_execution_lock_digest",
        "dependency_profile_digest",
        "runtime_component_identity_digest",
        "content_routing_reference_registry_digest",
    ):
        _require_sha256(registry[field_name], label=field_name)
    _require_positive_int(
        registry["method_parameter_sample_count"],
        label="method_parameter_sample_count",
    )
    if registry["model_id"] != schema["model_id"]:
        raise ValueError("content-routing reference model id is invalid")
    if registry["model_revision"] != schema["model_revision"]:
        raise ValueError("content-routing reference model revision is invalid")
    if (
        registry["content_routing_reference_quantile_algorithm"]
        != schema["quantile_algorithm"]
    ):
        raise ValueError("content-routing reference quantile algorithm is invalid")

    populations = registry["content_routing_reference_populations"]
    if type(populations) is not list or len(populations) != 3:
        raise ValueError("content-routing reference populations are invalid")
    resolved: list[dict[str, Any]] = []
    for index, expected_kind in enumerate(schema["population_order"]):
        population = populations[index]
        if type(population) is not dict or set(population) != set(
            schema["population_fields"]
        ):
            raise ValueError("content-routing reference population has invalid fields")
        if population["reference_observation_kind"] != expected_kind:
            raise ValueError("content-routing reference population order is invalid")
        _require_positive_int(
            population["reference_observation_positive_value_count"],
            label=f"population[{index}].positive_value_count",
        )
        _require_sha256(
            population["reference_observation_member_records_digest"],
            label=f"population[{index}].member_records_digest",
        )
        _require_sha256(
            population["tensor_content_sha256"],
            label=f"population[{index}].tensor_content_sha256",
        )
        resolved.append(population)
    return resolved


def _validate_content_routing_reference_registry(
    registry: dict[str, Any],
    *,
    raw_payload: bytes,
    expected_registry_digest: str,
) -> ContentRoutingReferenceScalars:
    _validate_registry_structure(registry)
    canonical_payload = stable_json_dumps(registry).encode("utf-8") + b"\n"
    if raw_payload != canonical_payload:
        raise ValueError("content-routing reference registry is not canonical")

    semantic_payload = dict(registry)
    embedded_digest = semantic_payload.pop(
        "content_routing_reference_registry_digest"
    )
    if build_stable_digest(semantic_payload) != embedded_digest:
        raise ValueError("embedded content-routing reference digest is invalid")
    if embedded_digest != expected_registry_digest:
        raise ValueError("expected content-routing reference digest does not match")

    scalars = {
        field_name: _require_exact_binary32(
            registry[field_name],
            label=field_name,
        )
        for field_name in _REFERENCE_REGISTRY_SCHEMA["scalar_fields"]
    }
    return ContentRoutingReferenceScalars(
        reference_gradient=scalars["reference_gradient"],
        reference_response=scalars["reference_response"],
        reference_sensitivity=scalars["reference_sensitivity"],
    )


def load_content_routing_reference_registry(
    *,
    expected_registry_digest: str,
    expected_file_sha256: str,
) -> ContentRoutingReferenceScalars:
    """从唯一固定配置路径加载已晋升 reference 标量。"""

    expected_registry_digest = _require_sha256(
        expected_registry_digest,
        label="expected_registry_digest",
    )
    expected_file_sha256 = _require_sha256(
        expected_file_sha256,
        label="expected_file_sha256",
    )
    raw_payload = _read_fixed_registry_bytes()
    if hashlib.sha256(raw_payload).hexdigest() != expected_file_sha256:
        raise ValueError("content-routing reference registry file digest does not match")
    registry = _strict_json_object(raw_payload)
    return _validate_content_routing_reference_registry(
        registry,
        raw_payload=raw_payload,
        expected_registry_digest=expected_registry_digest,
    )
