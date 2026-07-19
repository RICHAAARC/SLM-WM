"""Assemble the unique content-routing reference registry in memory."""

from __future__ import annotations

from typing import Any

import torch

from experiments.protocol.content_routing_reference_quantile import (
    _nearest_rank_p95,
    _positive_population,
    _snapshot_observations,
)
from experiments.protocol.content_routing_reference_registry import (
    _REFERENCE_REGISTRY_SCHEMA,
    _require_exact_binary32,
    _require_nonempty_string,
    _require_sha256,
    _validate_content_routing_reference_registry,
)
from main.core.digest import (
    build_stable_digest,
    stable_json_dumps,
    tensor_content_sha256,
)


__all__ = ["assemble_content_routing_reference_registry_payload"]


def _snapshot_exact_list(value: Any, *, label: str) -> tuple[Any, ...]:
    if type(value) is not list:
        raise TypeError(f"{label} must be an exact list")
    snapshot = tuple(value)
    if not snapshot:
        raise ValueError(f"{label} must not be empty")
    return snapshot


def _validate_prompt_projection(value: Any) -> tuple[dict[str, str], ...]:
    snapshot = _snapshot_exact_list(value, label="prompt_projection")
    resolved: list[dict[str, str]] = []
    for index, entry in enumerate(snapshot):
        if type(entry) is not dict or set(entry) != {
            "prompt_id",
            "prompt_text_digest",
        }:
            raise ValueError(f"prompt_projection[{index}] has invalid fields")
        prompt_id = _require_nonempty_string(
            entry["prompt_id"],
            label=f"prompt_projection[{index}].prompt_id",
        )
        prompt_digest = _require_sha256(
            entry["prompt_text_digest"],
            label=f"prompt_projection[{index}].prompt_text_digest",
        )
        resolved.append(
            {
                "prompt_id": prompt_id,
                "prompt_text_digest": prompt_digest,
            }
        )
    return tuple(resolved)


def _validate_seed_projection(value: Any) -> tuple[int, ...]:
    snapshot = _snapshot_exact_list(value, label="seed_projection_random")
    for index, seed in enumerate(snapshot):
        if type(seed) is not int or seed < 0:
            raise ValueError(
                f"seed_projection_random[{index}] must be a nonnegative integer"
            )
    return tuple(snapshot)


def _validate_generation_identity_digests(value: Any) -> tuple[str, ...]:
    snapshot = _snapshot_exact_list(
        value,
        label="generation_input_identity_digests",
    )
    return tuple(
        _require_sha256(
            digest,
            label=f"fixture_only_unqualified_generation_digest[{index}]",
        )
        for index, digest in enumerate(snapshot)
    )


def _validate_runtime_identity_payload(
    value: Any,
    *,
    dependency_profile_digest: str,
    formal_execution_lock_digest: str,
) -> dict[str, Any]:
    schema = _REFERENCE_REGISTRY_SCHEMA
    if type(value) is not dict or set(value) != set(
        schema["runtime_identity_payload_fields"]
    ):
        raise ValueError("runtime_component_identity_payload must be an exact object")
    if value["model_id"] != schema["model_id"]:
        raise ValueError("runtime component identity model id is invalid")
    if value["model_revision"] != schema["model_revision"]:
        raise ValueError("runtime component identity model revision is invalid")
    if value["dependency_profile_digest"] != dependency_profile_digest:
        raise ValueError("runtime dependency identity does not match registry")
    if value["formal_execution_lock_digest"] != formal_execution_lock_digest:
        raise ValueError("runtime formal lock identity does not match registry")
    for field_name in schema["runtime_identity_payload_fields"][4:]:
        _require_sha256(value[field_name], label=field_name)
    return dict(value)


def _validate_observation_metadata(
    observations: tuple[Any, ...],
    *,
    label: str,
) -> tuple[torch.Tensor, ...]:
    resolved: list[torch.Tensor] = []
    for index, member in enumerate(observations):
        member_label = f"{label}[{index}]"
        if not isinstance(member, torch.Tensor):
            raise TypeError(f"{member_label} must be a torch.Tensor")
        if member.dtype != torch.float32:
            raise TypeError(f"{member_label} must have dtype torch.float32")
        if member.device.type != "cpu":
            raise ValueError(f"{member_label} must be materialized on CPU")
        if (
            member.ndim != 4
            or member.shape[0] != 1
            or member.shape[1] != 1
            or member.shape[2] <= 0
            or member.shape[3] <= 0
        ):
            raise ValueError(f"{member_label} must have shape [1, 1, H, W]")
        resolved.append(member)
    return tuple(resolved)


def _population_record(
    *,
    kind: str,
    observations: tuple[torch.Tensor, ...],
    generation_identity_digests: tuple[str, ...],
    label: str,
) -> tuple[dict[str, Any], float]:
    positive_population = _positive_population(observations, label=label)
    member_records: list[dict[str, Any]] = []
    projection_fields = _REFERENCE_REGISTRY_SCHEMA["member_projection_fields"]
    for sequence_index, (member, generation_digest) in enumerate(
        zip(observations, generation_identity_digests, strict=True)
    ):
        record = {
            "reference_observation_kind": kind,
            "reference_observation_member_sequence_index": sequence_index,
            "generation_input_identity_digest": generation_digest,
            "tensor_content_sha256": tensor_content_sha256(member),
        }
        if tuple(record) != projection_fields:
            raise AssertionError("member projection does not match registry schema")
        member_records.append(record)

    selected_scalar = _require_exact_binary32(
        _nearest_rank_p95(positive_population),
        label=f"{label}.selected_scalar",
    )
    population = {
        "reference_observation_kind": kind,
        "reference_observation_positive_value_count": int(
            positive_population.numel()
        ),
        "reference_observation_member_records_digest": build_stable_digest(
            member_records
        ),
        "tensor_content_sha256": tensor_content_sha256(positive_population),
    }
    if tuple(population) != _REFERENCE_REGISTRY_SCHEMA["population_fields"]:
        raise AssertionError("population record does not match registry schema")
    return population, selected_scalar


def assemble_content_routing_reference_registry_payload(
    *,
    method_parameter_partition_id: Any,
    prompt_projection: Any,
    seed_projection_random: Any,
    generation_input_identity_digests: Any,
    gradient_observations: Any,
    response_observations: Any,
    sensitivity_observations: Any,
    formal_execution_lock_digest: Any,
    dependency_profile_digest: Any,
    runtime_component_identity_payload: Any,
) -> bytes:
    """Return canonical registry bytes without persistence or qualification."""

    partition_id = _require_nonempty_string(
        method_parameter_partition_id,
        label="method_parameter_partition_id",
    )
    formal_digest = _require_sha256(
        formal_execution_lock_digest,
        label="formal_execution_lock_digest",
    )
    dependency_digest = _require_sha256(
        dependency_profile_digest,
        label="dependency_profile_digest",
    )
    prompts = _validate_prompt_projection(prompt_projection)
    seeds = _validate_seed_projection(seed_projection_random)
    generation_digests = _validate_generation_identity_digests(
        generation_input_identity_digests
    )
    runtime_payload = _validate_runtime_identity_payload(
        runtime_component_identity_payload,
        dependency_profile_digest=dependency_digest,
        formal_execution_lock_digest=formal_digest,
    )

    gradient_snapshot = _validate_observation_metadata(
        _snapshot_observations(
            gradient_observations,
            label="gradient_observations",
        ),
        label="gradient_observations",
    )
    response_snapshot = _validate_observation_metadata(
        _snapshot_observations(
            response_observations,
            label="response_observations",
        ),
        label="response_observations",
    )
    sensitivity_snapshot = _validate_observation_metadata(
        _snapshot_observations(
            sensitivity_observations,
            label="sensitivity_observations",
        ),
        label="sensitivity_observations",
    )

    sample_count = len(prompts)
    if not (
        len(seeds)
        == len(generation_digests)
        == len(gradient_snapshot)
        == len(response_snapshot)
        == len(sensitivity_snapshot)
        == sample_count
    ):
        raise ValueError("content-routing reference member lengths do not match")

    population_inputs = (
        ("gradient_observations", gradient_snapshot),
        ("response_observations", response_snapshot),
        ("sensitivity_observations", sensitivity_snapshot),
    )
    populations: list[dict[str, Any]] = []
    selected_scalars: list[float] = []
    for kind, (label, observations) in zip(
        _REFERENCE_REGISTRY_SCHEMA["population_order"],
        population_inputs,
        strict=True,
    ):
        population, scalar = _population_record(
            kind=kind,
            observations=observations,
            generation_identity_digests=generation_digests,
            label=label,
        )
        populations.append(population)
        selected_scalars.append(scalar)

    schema = _REFERENCE_REGISTRY_SCHEMA
    registry = {
        "registry_schema": schema["registry_schema"],
        "method_parameter_partition_id": partition_id,
        "method_parameter_prompt_list_digest": build_stable_digest(list(prompts)),
        "method_parameter_seed_list_digest_random": build_stable_digest(list(seeds)),
        "method_parameter_sample_count": sample_count,
        "formal_execution_lock_digest": formal_digest,
        "dependency_profile_digest": dependency_digest,
        "model_id": schema["model_id"],
        "model_revision": schema["model_revision"],
        "runtime_component_identity_digest": build_stable_digest(runtime_payload),
        "content_routing_reference_quantile_algorithm": schema[
            "quantile_algorithm"
        ],
        "content_routing_reference_populations": populations,
        "reference_gradient": selected_scalars[0],
        "reference_response": selected_scalars[1],
        "reference_sensitivity": selected_scalars[2],
        "content_routing_reference_registry_digest": "",
    }
    if tuple(registry) != schema["top_fields"]:
        raise AssertionError("registry payload does not match the unique schema")
    semantic_payload = dict(registry)
    semantic_payload.pop("content_routing_reference_registry_digest")
    registry["content_routing_reference_registry_digest"] = build_stable_digest(
        semantic_payload
    )
    raw_payload = stable_json_dumps(registry).encode("utf-8") + b"\n"
    _validate_content_routing_reference_registry(
        registry,
        raw_payload=raw_payload,
        expected_registry_digest=registry[
            "content_routing_reference_registry_digest"
        ],
    )
    return raw_payload
