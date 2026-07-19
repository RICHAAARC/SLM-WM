"""Test the minimal in-memory content-routing reference registry producer."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import struct
from typing import Any

import pytest
import torch

import experiments.protocol.content_routing_reference_registry as registry_module
import experiments.protocol.content_routing_reference_registry_payload as payload_module
from main.core.digest import tensor_content_sha256


pytestmark = pytest.mark.quick


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _member(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, -1)


def _runtime_payload(
    *,
    dependency_digest: str,
    formal_digest: str,
) -> dict[str, Any]:
    schema = registry_module._REFERENCE_REGISTRY_SCHEMA
    return {
        "model_id": schema["model_id"],
        "model_revision": schema["model_revision"],
        "dependency_profile_digest": dependency_digest,
        "formal_execution_lock_digest": formal_digest,
        "vae_preprocess_identity_digest": _sha("vae-preprocess"),
        "scheduler_identity_digest": _sha("scheduler"),
        "content_observation_formula_identity_digest": _sha("content-formulas"),
    }


def _arguments() -> dict[str, Any]:
    dependency_digest = _sha("dependency")
    formal_digest = _sha("formal")
    return {
        "method_parameter_partition_id": "isolated-method-parameter-partition",
        "prompt_projection": [
            {"prompt_id": "prompt-0", "prompt_text_digest": _sha("prompt-0")},
            {"prompt_id": "prompt-1", "prompt_text_digest": _sha("prompt-1")},
        ],
        "seed_projection_random": [17, 29],
        "generation_input_identity_digests": [
            _sha("fixture_only_unqualified_generation_digest_0"),
            _sha("fixture_only_unqualified_generation_digest_1"),
        ],
        "gradient_observations": [
            _member([0.0, 1.0, 2.0, 3.0]),
            _member([4.0, 5.0, 6.0]),
        ],
        "response_observations": [
            _member([0.0, 0.25, 0.5]),
            _member([0.75, 1.0, 1.25]),
        ],
        "sensitivity_observations": [
            _member([0.0, 10.0, 20.0]),
            _member([30.0, 40.0]),
        ],
        "formal_execution_lock_digest": formal_digest,
        "dependency_profile_digest": dependency_digest,
        "runtime_component_identity_payload": _runtime_payload(
            dependency_digest=dependency_digest,
            formal_digest=formal_digest,
        ),
    }


def _assemble(**overrides: Any) -> bytes:
    arguments = _arguments()
    arguments.update(overrides)
    return payload_module.assemble_content_routing_reference_registry_payload(
        **arguments
    )


def _decoded(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"))
    assert type(value) is dict
    return value


def _positive_population(observations: list[torch.Tensor]) -> torch.Tensor:
    members = []
    for observation in observations:
        flat = observation.detach().contiguous().reshape(-1)
        positive = flat[flat > 0.0]
        if positive.numel() > 0:
            members.append(positive)
    return torch.cat(members)


def test_payload_has_one_keyword_only_public_interface() -> None:
    assert payload_module.__all__ == [
        "assemble_content_routing_reference_registry_payload"
    ]
    signature = inspect.signature(
        payload_module.assemble_content_routing_reference_registry_payload
    )
    assert tuple(signature.parameters) == (
        "method_parameter_partition_id",
        "prompt_projection",
        "seed_projection_random",
        "generation_input_identity_digests",
        "gradient_observations",
        "response_observations",
        "sensitivity_observations",
        "formal_execution_lock_digest",
        "dependency_profile_digest",
        "runtime_component_identity_payload",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    assert signature.return_annotation == "bytes"


def test_payload_and_loader_share_the_same_code_owned_schema() -> None:
    assert (
        payload_module._REFERENCE_REGISTRY_SCHEMA
        is registry_module._REFERENCE_REGISTRY_SCHEMA
    )
    assert len(registry_module._REFERENCE_REGISTRY_SCHEMA["top_fields"]) == 16
    assert len(registry_module._REFERENCE_REGISTRY_SCHEMA["population_fields"]) == 4
    assert (
        registry_module._REFERENCE_REGISTRY_SCHEMA["quantile_algorithm"]
        == "nearest_rank_full_sort_exact_rational_19_over_20_v1"
    )
    assert registry_module._REFERENCE_REGISTRY_SCHEMA[
        "runtime_identity_payload_fields"
    ] == tuple(
        _runtime_payload(
            dependency_digest=_sha("dependency"),
            formal_digest=_sha("formal"),
        )
    )


def test_payload_bytes_identities_and_minimal_fields_are_recomputed() -> None:
    arguments = _arguments()
    raw = payload_module.assemble_content_routing_reference_registry_payload(
        **arguments
    )
    registry = _decoded(raw)
    schema = registry_module._REFERENCE_REGISTRY_SCHEMA

    assert raw == (
        json.dumps(
            registry,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    assert set(registry) == set(schema["top_fields"])
    assert registry["content_routing_reference_quantile_algorithm"] == schema[
        "quantile_algorithm"
    ]
    assert registry["method_parameter_prompt_list_digest"] == _stable_digest(
        arguments["prompt_projection"]
    )
    assert registry["method_parameter_seed_list_digest_random"] == _stable_digest(
        arguments["seed_projection_random"]
    )
    assert registry["runtime_component_identity_digest"] == _stable_digest(
        arguments["runtime_component_identity_payload"]
    )
    semantic = dict(registry)
    embedded = semantic.pop("content_routing_reference_registry_digest")
    assert embedded == _stable_digest(semantic)

    removed_fields = {
        "content_routing_reference_quantile_numerator",
        "content_routing_reference_quantile_denominator",
        "content_routing_reference_quantile_rank_rule",
        "content_routing_reference_quantile_index_rule",
        "reference_gradient_binary32_hex",
        "reference_response_binary32_hex",
        "reference_sensitivity_binary32_hex",
    }
    assert removed_fields.isdisjoint(registry)
    for population in registry["content_routing_reference_populations"]:
        assert set(population) == set(schema["population_fields"])
        assert "reference_observation_selected_rank" not in population
        assert "reference_observation_selected_index" not in population


def test_populations_use_full_sort_exact_19_over_20_and_binary32_scalars() -> None:
    arguments = _arguments()
    registry = _decoded(
        payload_module.assemble_content_routing_reference_registry_payload(
            **arguments
        )
    )
    observation_names = (
        "gradient_observations",
        "response_observations",
        "sensitivity_observations",
    )
    scalar_names = (
        "reference_gradient",
        "reference_response",
        "reference_sensitivity",
    )
    for observations_name, scalar_name, population in zip(
        observation_names,
        scalar_names,
        registry["content_routing_reference_populations"],
        strict=True,
    ):
        positive = _positive_population(arguments[observations_name])
        sorted_values = torch.sort(positive).values
        n = int(sorted_values.numel())
        expected_index = (19 * n + 19) // 20 - 1
        expected = sorted_values[expected_index].item()
        assert registry[scalar_name] == expected
        assert type(registry[scalar_name]) is float
        assert math.isfinite(registry[scalar_name]) and registry[scalar_name] > 0.0
        assert struct.unpack(">f", struct.pack(">f", registry[scalar_name]))[0] == (
            registry[scalar_name]
        )
        assert population["reference_observation_positive_value_count"] == n
        assert population["tensor_content_sha256"] == tensor_content_sha256(
            positive
        )


def test_member_digest_binds_order_kind_generation_and_raw_tensor() -> None:
    arguments = _arguments()
    registry = _decoded(
        payload_module.assemble_content_routing_reference_registry_payload(
            **arguments
        )
    )
    observation_groups = (
        arguments["gradient_observations"],
        arguments["response_observations"],
        arguments["sensitivity_observations"],
    )
    for kind, observations, population in zip(
        registry_module._REFERENCE_REGISTRY_SCHEMA["population_order"],
        observation_groups,
        registry["content_routing_reference_populations"],
        strict=True,
    ):
        records = [
            {
                "reference_observation_kind": kind,
                "reference_observation_member_sequence_index": index,
                "generation_input_identity_digest": arguments[
                    "generation_input_identity_digests"
                ][index],
                "tensor_content_sha256": tensor_content_sha256(observation),
            }
            for index, observation in enumerate(observations)
        ]
        assert tuple(records[0]) == registry_module._REFERENCE_REGISTRY_SCHEMA[
            "member_projection_fields"
        ]
        assert population["reference_observation_member_records_digest"] == (
            _stable_digest(records)
        )


def test_each_population_uses_each_governed_kernel_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {"snapshot": 0, "positive": 0, "nearest": 0}
    real_snapshot = payload_module._snapshot_observations
    real_positive = payload_module._positive_population
    real_nearest = payload_module._nearest_rank_p95

    def snapshot(*args: Any, **kwargs: Any) -> Any:
        counts["snapshot"] += 1
        return real_snapshot(*args, **kwargs)

    def positive(*args: Any, **kwargs: Any) -> Any:
        counts["positive"] += 1
        return real_positive(*args, **kwargs)

    def nearest(*args: Any, **kwargs: Any) -> Any:
        counts["nearest"] += 1
        return real_nearest(*args, **kwargs)

    monkeypatch.setattr(payload_module, "_snapshot_observations", snapshot)
    monkeypatch.setattr(payload_module, "_positive_population", positive)
    monkeypatch.setattr(payload_module, "_nearest_rank_p95", nearest)
    _assemble()
    assert counts == {"snapshot": 3, "positive": 3, "nearest": 3}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method_parameter_partition_id", ""),
        ("prompt_projection", ()),
        ("seed_projection_random", [True, 2]),
        ("generation_input_identity_digests", ["not-a-digest"]),
        ("formal_execution_lock_digest", "not-a-digest"),
        ("dependency_profile_digest", "not-a-digest"),
    ],
)
def test_invalid_static_identity_inputs_fail_closed(field: str, value: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        _assemble(**{field: value})


def test_runtime_identity_must_match_model_dependency_and_formal_lock() -> None:
    arguments = _arguments()
    for field, value in (
        ("model_revision", "wrong"),
        ("dependency_profile_digest", _sha("wrong-dependency")),
        ("formal_execution_lock_digest", _sha("wrong-formal")),
    ):
        runtime = dict(arguments["runtime_component_identity_payload"])
        runtime[field] = value
        with pytest.raises(ValueError):
            _assemble(runtime_component_identity_payload=runtime)


@pytest.mark.parametrize(
    "component_field",
    registry_module._REFERENCE_REGISTRY_SCHEMA[
        "runtime_identity_payload_fields"
    ][4:],
)
@pytest.mark.parametrize("mutation", ["missing", "invalid_sha"])
def test_runtime_component_digests_fail_before_observation_content(
    component_field: str,
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments()
    runtime = dict(arguments["runtime_component_identity_payload"])
    if mutation == "missing":
        runtime.pop(component_field)
    else:
        runtime[component_field] = "not-a-sha256"

    content_reads = 0

    def forbidden_content_read(*args: Any, **kwargs: Any) -> Any:
        nonlocal content_reads
        content_reads += 1
        raise AssertionError("observation content was read before identity validation")

    monkeypatch.setattr(payload_module, "_positive_population", forbidden_content_read)
    with pytest.raises(ValueError):
        _assemble(runtime_component_identity_payload=runtime)
    assert content_reads == 0


def test_runtime_identity_extra_field_fails_before_observation_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments()
    runtime = dict(arguments["runtime_component_identity_payload"])
    runtime["extra_identity_digest"] = _sha("extra")
    content_reads = 0

    def forbidden_content_read(*args: Any, **kwargs: Any) -> Any:
        nonlocal content_reads
        content_reads += 1
        raise AssertionError("observation content was read before identity validation")

    monkeypatch.setattr(payload_module, "_positive_population", forbidden_content_read)
    with pytest.raises(ValueError):
        _assemble(runtime_component_identity_payload=runtime)
    assert content_reads == 0


def test_member_lengths_and_observation_metadata_fail_before_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Tensor content was read before metadata preflight")

    monkeypatch.setattr(payload_module, "_positive_population", forbidden)
    with pytest.raises(ValueError):
        _assemble(response_observations=[torch.empty((1, 1, 0, 2))])


def test_invalid_observation_contents_fail_closed() -> None:
    for observations in (
        [_member([0.0, -1.0])],
        [_member([0.0, float("nan")])],
        [_member([0.0, 0.0])],
    ):
        with pytest.raises(ValueError):
            _assemble(
                prompt_projection=[
                    {"prompt_id": "prompt", "prompt_text_digest": _sha("prompt")}
                ],
                seed_projection_random=[1],
                generation_input_identity_digests=[
                    _sha("fixture_only_unqualified_generation_digest")
                ],
                gradient_observations=observations,
                response_observations=[_member([1.0])],
                sensitivity_observations=[_member([1.0])],
            )


def test_list_and_tuple_observations_are_equivalent_and_inputs_unchanged() -> None:
    arguments = _arguments()
    tensors = tuple(
        tensor
        for name in (
            "gradient_observations",
            "response_observations",
            "sensitivity_observations",
        )
        for tensor in arguments[name]
    )
    before = tuple(tensor.clone() for tensor in tensors)
    first = payload_module.assemble_content_routing_reference_registry_payload(
        **arguments
    )
    arguments["gradient_observations"] = tuple(arguments["gradient_observations"])
    arguments["response_observations"] = tuple(arguments["response_observations"])
    arguments["sensitivity_observations"] = tuple(
        arguments["sensitivity_observations"]
    )
    second = payload_module.assemble_content_routing_reference_registry_payload(
        **arguments
    )
    assert first == second
    assert all(torch.equal(current, original) for current, original in zip(tensors, before))
