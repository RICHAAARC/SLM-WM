"""Validate deterministic in-memory assembly of the reference registry."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import re
import struct
from typing import Any, Callable

import pytest
import torch

import experiments.protocol.content_routing_reference_registry as registry_module
import experiments.protocol.content_routing_reference_registry_payload as payload_module
from main.core.digest import tensor_content_sha256


pytestmark = pytest.mark.quick

ROOT = Path(__file__).resolve().parents[2]
FIXED_REGISTRY = ROOT / "configs/content_routing_reference_registry.json"
CONTRACT_NAME = "CONTENT_ROUTING_REFERENCE_REGISTRY_MACHINE_CONTRACT"


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _member(values: list[float] | tuple[float, ...]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, -1)


def _value_for_rule(rule: dict[str, Any], *, label: str) -> Any:
    if "exact_value" in rule:
        return rule["exact_value"]
    predicate = rule["predicate"]
    if predicate == "sha256_lower_hex_str":
        return _sha(label)
    if predicate == "nonempty_exact_str":
        return f"governed-{label}"
    if predicate == "strict_positive_int":
        return 1
    if predicate == "nonnegative_int":
        return 0
    if predicate == "strict_positive_finite_json_float":
        return 1.0
    raise AssertionError(f"fixture lacks a value for {label}: {predicate}")


def _runtime_payload(
    contract: dict[str, Any],
    *,
    dependency_profile_digest: str,
    formal_execution_lock_digest: str,
) -> dict[str, Any]:
    runtime_contract = contract["runtime_component_identity_payload_contract"]
    probe_contract = runtime_contract["public_probe_identity_contract"]
    domain_fields = {
        field_name: _value_for_rule(rule, label=f"domain-{field_name}")
        for field_name, rule in probe_contract["domain_field_rules"].items()
    }
    public_probe = {
        field_name: (
            domain_fields
            if field_name == "domain_fields"
            else _value_for_rule(rule, label=f"probe-{field_name}")
        )
        for field_name, rule in probe_contract["field_rules"].items()
    }
    payload = {
        field_name: (
            public_probe
            if field_name == "public_probe_identity"
            else _value_for_rule(rule, label=f"runtime-{field_name}")
        )
        for field_name, rule in runtime_contract["field_rules"].items()
    }
    payload["dependency_profile_digest"] = dependency_profile_digest
    payload["formal_execution_lock_digest"] = formal_execution_lock_digest
    return payload


def _valid_arguments() -> dict[str, Any]:
    contract = payload_module._load_machine_contract()
    dependency_digest = _sha("dependency-profile")
    formal_lock_digest = _sha("formal-lock")
    return {
        "method_parameter_partition_id": "method-parameter-partition-v1",
        "prompt_projection": [
            {"prompt_id": "prompt-0", "prompt_text_digest": _sha("prompt-0")},
            {"prompt_id": "prompt-1", "prompt_text_digest": _sha("prompt-1")},
        ],
        "seed_projection_random": [17, 29],
        "generation_input_identity_digests": [
            _sha("generation-0"),
            _sha("generation-1"),
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
        "formal_execution_lock_digest": formal_lock_digest,
        "dependency_profile_digest": dependency_digest,
        "runtime_component_identity_payload": _runtime_payload(
            contract,
            dependency_profile_digest=dependency_digest,
            formal_execution_lock_digest=formal_lock_digest,
        ),
    }


def _assemble(**overrides: Any) -> bytes:
    arguments = _valid_arguments()
    arguments.update(overrides)
    return payload_module.assemble_content_routing_reference_registry_payload(
        **arguments
    )


def _decoded(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"))
    assert type(value) is dict
    return value


def _positive_population(observations: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        tuple(
            member.detach().contiguous().reshape(-1)[
                member.detach().contiguous().reshape(-1) > 0
            ]
            for member in observations
        ),
        dim=0,
    )


def _assert_content_not_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Tensor content was read before static preflight")

    monkeypatch.setattr(payload_module, "_positive_population", forbidden)
    monkeypatch.setattr(payload_module, "_nearest_rank_p95", forbidden)
    monkeypatch.setattr(payload_module, "tensor_content_sha256", forbidden)


def _assert_observations_not_snapshotted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_content_not_read(monkeypatch)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("observations were snapshotted before contract preflight")

    monkeypatch.setattr(payload_module, "_snapshot_observations", forbidden)


def test_payload_assembler_has_one_keyword_only_public_interface() -> None:
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


def test_payload_bytes_and_all_identity_digests_are_independently_recomputed() -> None:
    arguments = _valid_arguments()
    contract = payload_module._load_machine_contract()

    raw = payload_module.assemble_content_routing_reference_registry_payload(
        **arguments
    )
    registry = _decoded(raw)

    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert raw == (
        json.dumps(
            registry,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    semantic_payload = dict(registry)
    embedded_digest = semantic_payload.pop(
        "content_routing_reference_registry_digest"
    )
    assert embedded_digest == _stable_digest(semantic_payload)
    assert registry["method_parameter_prompt_list_digest"] == _stable_digest(
        arguments["prompt_projection"]
    )
    assert registry["method_parameter_seed_list_digest_random"] == _stable_digest(
        arguments["seed_projection_random"]
    )
    assert registry["runtime_component_identity_digest"] == _stable_digest(
        arguments["runtime_component_identity_payload"]
    )

    numerator = contract["registry_top_level_field_rules"][
        "content_routing_reference_quantile_numerator"
    ]["exact_value"]
    denominator = contract["registry_top_level_field_rules"][
        "content_routing_reference_quantile_denominator"
    ]["exact_value"]
    observation_groups = (
        arguments["gradient_observations"],
        arguments["response_observations"],
        arguments["sensitivity_observations"],
    )
    for population, kind, observations in zip(
        registry["content_routing_reference_populations"],
        contract["population_order"],
        observation_groups,
        strict=True,
    ):
        positive = _positive_population(observations)
        sorted_values = torch.sort(positive).values
        rank = (numerator * positive.numel() + denominator - 1) // denominator
        index = rank - 1
        scalar_binding = contract["population_scalar_binding"][kind]
        scalar = sorted_values[index].item()
        member_records = [
            {
                "reference_observation_kind": kind,
                "reference_observation_member_sequence_index": sequence_index,
                "generation_input_identity_digest": generation_digest,
                "tensor_content_sha256": tensor_content_sha256(member),
            }
            for sequence_index, (member, generation_digest) in enumerate(
                zip(
                    observations,
                    arguments["generation_input_identity_digests"],
                    strict=True,
                )
            )
        ]
        assert population == {
            "reference_observation_kind": kind,
            "reference_observation_member_count": len(observations),
            "reference_observation_positive_value_count": positive.numel(),
            "reference_observation_member_records_digest": _stable_digest(
                member_records
            ),
            "tensor_content_sha256": tensor_content_sha256(positive),
            "reference_observation_selected_rank": rank,
            "reference_observation_selected_index": index,
        }
        assert registry[scalar_binding["scalar_field"]] == scalar
        assert registry[scalar_binding["binary32_hex_field"]] == struct.pack(
            ">f", scalar
        ).hex()


def test_each_population_uses_each_governed_kernel_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_calls: list[str] = []
    positive_calls: list[str] = []
    selection_sizes: list[int] = []
    validator_calls: list[bytes] = []
    original_snapshot = payload_module._snapshot_observations
    original_positive = payload_module._positive_population
    original_selection = payload_module._nearest_rank_p95
    original_validator = payload_module._validate_content_routing_reference_registry

    def snapshot(value: Any, *, label: str) -> tuple[torch.Tensor, ...]:
        snapshot_calls.append(label)
        return original_snapshot(value, label=label)

    def positive(
        observations: tuple[torch.Tensor, ...], *, label: str
    ) -> torch.Tensor:
        positive_calls.append(label)
        return original_positive(observations, label=label)

    def select(population: torch.Tensor) -> float:
        selection_sizes.append(population.numel())
        return original_selection(population)

    def validate(payload: Any, **kwargs: Any) -> None:
        validator_calls.append(kwargs["raw_payload"])
        original_validator(payload, **kwargs)

    monkeypatch.setattr(payload_module, "_snapshot_observations", snapshot)
    monkeypatch.setattr(payload_module, "_positive_population", positive)
    monkeypatch.setattr(payload_module, "_nearest_rank_p95", select)
    monkeypatch.setattr(
        payload_module,
        "_validate_content_routing_reference_registry",
        validate,
    )

    raw = _assemble()

    expected_labels = [
        "gradient_observations",
        "response_observations",
        "sensitivity_observations",
    ]
    assert snapshot_calls == expected_labels
    assert positive_calls == expected_labels
    assert selection_sizes == [6, 5, 4]
    assert validator_calls == [raw]


def test_assembler_does_not_implement_a_second_selection_algorithm() -> None:
    source = inspect.getsource(payload_module)
    for forbidden in (
        "torch.sort",
        "torch.quantile",
        "numpy.quantile",
        ".kthvalue",
        ".topk",
        "math.ceil",
        "0.95",
        "positive_population[selected_index]",
    ):
        assert forbidden not in source
    assert source.count("_nearest_rank_p95(positive_population)") == 1
    assert "(numerator * positive_count + denominator - 1) // denominator" in source


@pytest.mark.parametrize(
    ("argument_name", "invalid_value"),
    [
        ("prompt_projection", ()),
        ("prompt_projection", iter(())),
        ("prompt_projection", {}),
        ("prompt_projection", "prompt"),
        ("prompt_projection", b"prompt"),
        ("seed_projection_random", (17, 29)),
        ("seed_projection_random", iter([17, 29])),
        ("generation_input_identity_digests", (_sha("a"), _sha("b"))),
        ("generation_input_identity_digests", iter([_sha("a"), _sha("b")])),
    ],
)
def test_identity_projections_require_exact_materialized_lists(
    argument_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _assemble(**{argument_name: invalid_value})


def test_rejected_generation_identity_generator_is_not_consumed() -> None:
    consumed = False

    def values() -> Any:
        nonlocal consumed
        consumed = True
        yield _sha("generation")

    with pytest.raises(TypeError, match="exact list"):
        _assemble(generation_input_identity_digests=values())
    assert not consumed


def test_list_and_tuple_observations_are_equivalent_and_inputs_are_unchanged() -> None:
    arguments = _valid_arguments()
    base = torch.arange(1, 13, dtype=torch.float32).reshape(1, 1, 3, 4)
    member = base.transpose(-2, -1).detach().requires_grad_()
    arguments["gradient_observations"][0] = member
    before = member.detach().clone()
    before_shape = member.shape
    before_stride = member.stride()
    before_requires_grad = member.requires_grad
    before_grad = member.grad

    first = payload_module.assemble_content_routing_reference_registry_payload(
        **arguments
    )
    arguments["gradient_observations"] = tuple(
        arguments["gradient_observations"]
    )
    arguments["response_observations"] = tuple(
        arguments["response_observations"]
    )
    arguments["sensitivity_observations"] = tuple(
        arguments["sensitivity_observations"]
    )
    second = payload_module.assemble_content_routing_reference_registry_payload(
        **arguments
    )

    assert first == second
    assert torch.equal(member.detach(), before)
    assert member.shape == before_shape
    assert member.stride() == before_stride
    assert member.requires_grad is before_requires_grad
    assert member.grad is before_grad


@pytest.mark.parametrize(
    "mutation",
    [
        lambda contract: contract["registry_top_level_field_rules"].pop(
            "registry_schema"
        ),
        lambda contract: contract["registry_top_level_field_rules"].update(
            {"unexpected": {"predicate": "sha256_lower_hex_str"}}
        ),
        lambda contract: contract["population_field_rules"].pop(
            "reference_observation_selected_index"
        ),
        lambda contract: contract["population_field_rules"].update(
            {"unexpected": {"predicate": "strict_positive_int"}}
        ),
        lambda contract: contract["member_record_field_rules"].pop(
            "generation_input_identity_digest"
        ),
        lambda contract: contract["member_record_field_rules"].update(
            {"unexpected": {"predicate": "sha256_lower_hex_str"}}
        ),
        lambda contract: contract["member_record_field_rules"][
            "generation_input_identity_digest"
        ].update({"predicate": "unknown_predicate"}),
        lambda contract: contract["prompt_projection_contract"][
            "entry_field_rules"
        ].pop("prompt_id"),
        lambda contract: contract["prompt_projection_contract"][
            "entry_field_rules"
        ].update({"unexpected": {"predicate": "nonempty_exact_str"}}),
        lambda contract: contract["prompt_projection_contract"].update(
            {"digest_rule": "sha256(prompt_projection)"}
        ),
        lambda contract: contract["seed_projection_contract"].update(
            {"element_predicate": "unknown_predicate"}
        ),
        lambda contract: contract[
            "runtime_component_identity_payload_contract"
        ]["field_rules"].pop("pipeline_class_name"),
        lambda contract: contract[
            "runtime_component_identity_payload_contract"
        ]["public_probe_identity_contract"]["domain_field_rules"].update(
            {"unexpected": {"predicate": "exact_token_str", "exact_value": "x"}}
        ),
        lambda contract: contract[
            "runtime_component_identity_payload_contract"
        ].update({"digest_rule": "sha256(runtime_payload)"}),
        lambda contract: contract["population_scalar_binding"].pop(
            "latent_response"
        ),
    ],
)
def test_consumed_contract_mutations_fail_before_tensor_content(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], Any],
) -> None:
    arguments = _valid_arguments()
    contract = copy.deepcopy(payload_module._load_machine_contract())
    mutation(contract)
    monkeypatch.setattr(payload_module, "_load_machine_contract", lambda: contract)
    _assert_observations_not_snapshotted(monkeypatch)

    with pytest.raises(ValueError):
        payload_module.assemble_content_routing_reference_registry_payload(
            **arguments
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda contract: contract["prompt_projection_contract"].update(
            {"digest_rule": "build_stable_digest(wrong_payload)"}
        ),
        lambda contract: contract["seed_projection_contract"].update(
            {"digest_rule": "build_stable_digest(wrong_payload)"}
        ),
        lambda contract: contract["population_field_rules"][
            "reference_observation_member_records_digest"
        ].update({"digest_rule": "build_stable_digest(wrong_payload)"}),
        lambda contract: contract[
            "runtime_component_identity_payload_contract"
        ].update({"digest_rule": "build_stable_digest(wrong_payload)"}),
        lambda contract: contract[
            "runtime_component_identity_payload_contract"
        ].update(
            {"component_config_digest_rule": "build_stable_digest(wrong_payload)"}
        ),
        lambda contract: contract.update(
            {"semantic_digest_rule": "build_stable_digest(wrong_payload)"}
        ),
        lambda contract: contract.update(
            {"file_sha256_rule": "sha256(wrong_payload)"}
        ),
        lambda contract: contract.update(
            {"binary32_hex_rule": "struct.pack('<f', scalar).hex()"}
        ),
    ],
)
def test_digest_and_binary32_rule_drift_fails_before_observation_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], Any],
) -> None:
    arguments = _valid_arguments()
    contract = copy.deepcopy(payload_module._load_machine_contract())
    mutation(contract)
    monkeypatch.setattr(payload_module, "_load_machine_contract", lambda: contract)
    _assert_observations_not_snapshotted(monkeypatch)

    with pytest.raises(ValueError):
        payload_module.assemble_content_routing_reference_registry_payload(
            **arguments
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda contract: contract["registry_top_level_field_rules"][
            "registry_schema"
        ].pop("exact_value"),
        lambda contract: contract["registry_top_level_field_rules"][
            "content_routing_reference_quantile_algorithm"
        ].pop("exact_value"),
        lambda contract: contract["registry_top_level_field_rules"][
            "content_routing_reference_quantile_rank_rule"
        ].pop("exact_value"),
        lambda contract: contract["registry_top_level_field_rules"][
            "model_revision"
        ].pop("exact_value"),
        lambda contract: contract["registry_top_level_field_rules"][
            "content_routing_reference_quantile_algorithm"
        ].update({"predicate": "nonempty_exact_str"}),
        lambda contract: contract[
            "runtime_component_identity_payload_contract"
        ]["field_rules"]["pipeline_class_name"].pop("exact_value"),
        lambda contract: contract[
            "runtime_component_identity_payload_contract"
        ]["field_rules"]["vae_decode_protocol"].pop("exact_value"),
        lambda contract: contract[
            "runtime_component_identity_payload_contract"
        ]["field_rules"]["pipeline_class_name"].update(
            {"predicate": "nonempty_exact_str"}
        ),
        lambda contract: contract[
            "runtime_component_identity_payload_contract"
        ]["field_rules"]["previous_latent_callback_index"].update(
            {"predicate": "strict_positive_int"}
        ),
        lambda contract: contract[
            "runtime_component_identity_payload_contract"
        ]["public_probe_identity_contract"]["field_rules"][
            "prg_version"
        ].pop("exact_value"),
        lambda contract: contract[
            "runtime_component_identity_payload_contract"
        ]["public_probe_identity_contract"]["field_rules"][
            "prg_version"
        ].update({"predicate": "nonempty_exact_str"}),
        lambda contract: contract[
            "runtime_component_identity_payload_contract"
        ]["public_probe_identity_contract"]["domain_field_rules"][
            "purpose"
        ].pop("exact_value"),
        lambda contract: contract["population_field_rules"][
            "reference_observation_kind"
        ].update({"exact_value_source": "wrong_source"}),
        lambda contract: contract["member_record_field_rules"][
            "reference_observation_kind"
        ].pop("exact_value_source"),
        lambda contract: contract["registry_top_level_field_rules"][
            "content_routing_reference_populations"
        ].pop("exact_length"),
        lambda contract: contract["registry_top_level_field_rules"][
            "method_parameter_prompt_list_digest"
        ].pop("digest_contract"),
        lambda contract: contract["registry_top_level_field_rules"][
            "method_parameter_seed_list_digest_random"
        ].update({"digest_contract": "wrong_contract"}),
        lambda contract: contract["registry_top_level_field_rules"][
            "runtime_component_identity_digest"
        ].pop("digest_contract"),
        lambda contract: contract["prompt_projection_contract"].update(
            {"container_predicate": "exact_object"}
        ),
        lambda contract: contract[
            "runtime_component_identity_payload_contract"
        ]["field_rules"]["public_probe_identity"].update(
            {"predicate": "nonempty_exact_str"}
        ),
    ],
)
def test_required_contract_attributes_fail_before_observation_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], Any],
) -> None:
    arguments = _valid_arguments()
    contract = copy.deepcopy(payload_module._load_machine_contract())
    mutation(contract)
    monkeypatch.setattr(payload_module, "_load_machine_contract", lambda: contract)
    _assert_observations_not_snapshotted(monkeypatch)

    with pytest.raises(ValueError):
        payload_module.assemble_content_routing_reference_registry_payload(
            **arguments
        )


def _swap_population_scalar_bindings(contract: dict[str, Any]) -> None:
    bindings = contract["population_scalar_binding"]
    gradient = dict(bindings["gradient_magnitude_rgb_pre_interpolation"])
    response = dict(bindings["latent_response"])
    bindings["gradient_magnitude_rgb_pre_interpolation"] = response
    bindings["latent_response"] = gradient


def _cross_population_scalar_hex_responsibilities(
    contract: dict[str, Any],
) -> None:
    bindings = contract["population_scalar_binding"]
    gradient_hex = bindings["gradient_magnitude_rgb_pre_interpolation"][
        "binary32_hex_field"
    ]
    response_hex = bindings["latent_response"]["binary32_hex_field"]
    bindings["gradient_magnitude_rgb_pre_interpolation"][
        "binary32_hex_field"
    ] = response_hex
    bindings["latent_response"]["binary32_hex_field"] = gradient_hex


@pytest.mark.parametrize(
    "mutation",
    [
        lambda contract: contract["population_scalar_binding"][
            "latent_response"
        ].update({"scalar_field": "reference_gradient"}),
        lambda contract: contract["population_scalar_binding"][
            "latent_response"
        ].update({"binary32_hex_field": "reference_gradient_binary32_hex"}),
        _swap_population_scalar_bindings,
        _cross_population_scalar_hex_responsibilities,
    ],
)
def test_population_scalar_binding_must_be_an_exact_bijection_before_content(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], Any],
) -> None:
    arguments = _valid_arguments()
    contract = copy.deepcopy(payload_module._load_machine_contract())
    mutation(contract)
    monkeypatch.setattr(payload_module, "_load_machine_contract", lambda: contract)
    _assert_observations_not_snapshotted(monkeypatch)

    with pytest.raises(ValueError):
        payload_module.assemble_content_routing_reference_registry_payload(
            **arguments
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda arguments: arguments.update(
            {"method_parameter_partition_id": ""}
        ),
        lambda arguments: arguments.update(
            {"formal_execution_lock_digest": "not-a-digest"}
        ),
        lambda arguments: arguments.update(
            {"dependency_profile_digest": "not-a-digest"}
        ),
        lambda arguments: arguments["prompt_projection"][0].pop("prompt_id"),
        lambda arguments: arguments["prompt_projection"][0].update(
            {"unexpected": "field"}
        ),
        lambda arguments: arguments.update(
            {"prompt_projection": arguments["prompt_projection"][:1]}
        ),
        lambda arguments: arguments["seed_projection_random"].__setitem__(0, True),
        lambda arguments: arguments.update(
            {"seed_projection_random": arguments["seed_projection_random"][:1]}
        ),
        lambda arguments: arguments[
            "generation_input_identity_digests"
        ].__setitem__(0, "not-a-digest"),
        lambda arguments: arguments.update(
            {
                "generation_input_identity_digests": arguments[
                    "generation_input_identity_digests"
                ][:1]
            }
        ),
    ],
)
def test_static_identity_failures_precede_observation_content(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], Any],
) -> None:
    arguments = _valid_arguments()
    mutation(arguments)
    _assert_observations_not_snapshotted(monkeypatch)

    with pytest.raises((TypeError, ValueError)):
        payload_module.assemble_content_routing_reference_registry_payload(
            **arguments
        )


@pytest.mark.parametrize(
    "invalid_observations",
    [
        _member([1.0]),
        iter([_member([1.0]), _member([2.0])]),
        {"member": _member([1.0])},
        "observations",
        b"observations",
    ],
)
def test_observation_containers_reuse_the_exact_quantile_boundary(
    invalid_observations: object,
) -> None:
    with pytest.raises(TypeError, match="exact list or tuple"):
        _assemble(gradient_observations=invalid_observations)


@pytest.mark.parametrize("document_case", ["missing", "duplicate", "nonliteral"])
def test_ast_contract_failures_precede_observation_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    document_case: str,
) -> None:
    arguments = _valid_arguments()
    contract = payload_module._load_machine_contract()
    assignment = f"{CONTRACT_NAME} = {contract!r}"
    if document_case == "missing":
        document = "```python\nUNRELATED = {}\n```\n"
    elif document_case == "duplicate":
        document = f"```python\n{assignment}\n{assignment}\n```\n"
    else:
        document = f"```python\n{CONTRACT_NAME} = dict()\n```\n"
    method_document = tmp_path / "method.md"
    method_document.write_text(document, encoding="utf-8")
    monkeypatch.setattr(registry_module, "_METHOD_DOCUMENT_PATH", method_document)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("observations were touched before contract parsing")

    monkeypatch.setattr(payload_module, "_snapshot_observations", forbidden)
    with pytest.raises(ValueError):
        payload_module.assemble_content_routing_reference_registry_payload(
            **arguments
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda runtime: runtime.update({"device_name": "cuda:0"}),
        lambda runtime: runtime.pop("pipeline_class_name"),
        lambda runtime: runtime["public_probe_identity"].update(
            {"watermark_key": "secret"}
        ),
        lambda runtime: runtime["public_probe_identity"]["domain_fields"].update(
            {"prompt": "forbidden"}
        ),
        lambda runtime: runtime.update({"model_id": "wrong/model"}),
        lambda runtime: runtime.update(
            {"dependency_profile_digest": _sha("wrong-dependency")}
        ),
        lambda runtime: runtime.update(
            {"formal_execution_lock_digest": _sha("wrong-lock")}
        ),
    ],
)
def test_runtime_identity_exactness_and_cross_fields_fail_before_tensor_content(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], Any],
) -> None:
    arguments = _valid_arguments()
    runtime_payload = copy.deepcopy(arguments["runtime_component_identity_payload"])
    mutation(runtime_payload)
    _assert_observations_not_snapshotted(monkeypatch)

    with pytest.raises(ValueError):
        payload_module.assemble_content_routing_reference_registry_payload(
            **{**arguments, "runtime_component_identity_payload": runtime_payload}
        )


def test_all_observation_metadata_is_checked_before_population_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _valid_arguments()
    arguments["sensitivity_observations"][1] = torch.ones(
        (1, 1, 1, 1), dtype=torch.float64
    )
    _assert_content_not_read(monkeypatch)

    with pytest.raises(TypeError, match="torch.float32"):
        payload_module.assemble_content_routing_reference_registry_payload(
            **arguments
        )


def test_non_cpu_metadata_fails_before_content_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _valid_arguments()
    arguments["response_observations"][0] = torch.empty(
        (1, 1, 2, 2), dtype=torch.float32, device="meta"
    )
    _assert_content_not_read(monkeypatch)

    with pytest.raises(ValueError, match="materialized on CPU"):
        payload_module.assemble_content_routing_reference_registry_payload(
            **arguments
        )


@pytest.mark.parametrize("invalid_value", [-1.0, float("nan"), float("inf")])
def test_content_failures_are_delegated_to_the_shared_population_kernel(
    invalid_value: float,
) -> None:
    arguments = _valid_arguments()
    arguments["gradient_observations"][0] = _member([1.0, invalid_value])
    with pytest.raises(ValueError):
        payload_module.assemble_content_routing_reference_registry_payload(
            **arguments
        )


def test_source_has_no_artifact_io_model_or_runtime_side_effects() -> None:
    source = inspect.getsource(payload_module)
    for forbidden in (
        "from_pretrained",
        "decode_latent",
        "build_keyed_gaussian_tensor",
        "torch.cuda",
        "os.open",
        "Path(",
        ".write_bytes(",
        ".write_text(",
        "outputs/",
    ):
        assert forbidden not in source
    assert not FIXED_REGISTRY.exists()


def test_contract_source_contains_one_literal_assignment() -> None:
    text = registry_module._METHOD_DOCUMENT_PATH.read_text(encoding="utf-8-sig")
    assignments = []
    for code_block in re.findall(r"```python\n(.*?)\n```", text, flags=re.DOTALL):
        tree = ast.parse(code_block)
        for statement in tree.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id == CONTRACT_NAME:
                assignments.append(ast.literal_eval(statement.value))
    assert len(assignments) == 1
    assert assignments[0] == payload_module._load_machine_contract()
