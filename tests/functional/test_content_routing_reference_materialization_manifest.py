"""Validate deterministic in-memory materialization-manifest assembly."""

from __future__ import annotations

import builtins
import copy
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
import torch

import experiments.protocol.content_routing_reference_raw_member as raw_module
import experiments.protocol.content_routing_reference_registry as registry_module
import experiments.protocol.content_routing_reference_registry_payload as payload_module
from experiments.protocol import (
    content_routing_reference_generation_input as generation_module,
    content_routing_reference_materialization_manifest as manifest_module,
)
from main.core.digest import build_stable_digest


pytestmark = pytest.mark.quick

ROOT = Path(__file__).resolve().parents[2]
FIXED_REGISTRY = ROOT / "configs/content_routing_reference_registry.json"
CODE_COMMIT = "4" * 40


def _stable_digest(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


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


@pytest.fixture(scope="module")
def valid_bundle() -> dict[str, Any]:
    contract = registry_module._load_machine_contract()
    dependency_digest = _sha("manifest-dependency")
    formal_lock_digest = _sha("manifest-formal-lock")
    formal_method_digest = _sha("manifest-method-config")
    runtime_payload = _runtime_payload(
        contract,
        dependency_profile_digest=dependency_digest,
        formal_execution_lock_digest=formal_lock_digest,
    )
    runtime_digest = build_stable_digest(runtime_payload)

    generation_records = [
        generation_module.build_content_routing_reference_generation_input_record(
            reference_observation_member_sequence_index=sequence_index,
            prompt_id=f"prompt-{sequence_index}",
            prompt_text=f"A governed prompt number {sequence_index}",
            generation_seed_random=seed,
            formal_method_config_digest=formal_method_digest,
            dependency_profile_digest=dependency_digest,
            formal_execution_lock_digest=formal_lock_digest,
            runtime_component_identity_digest=runtime_digest,
        )
        for sequence_index, seed in enumerate((17, 29))
    ]
    generation_digests = [
        record["generation_input_identity_digest"] for record in generation_records
    ]
    prompt_projection = [
        {
            "prompt_id": record["generation_input_identity_payload"]["prompt_id"],
            "prompt_text_digest": record["generation_input_identity_payload"][
                "prompt_text_digest"
            ],
        }
        for record in generation_records
    ]
    seed_projection = [
        record["generation_input_identity_payload"]["generation_seed_random"]
        for record in generation_records
    ]
    population_order = contract["population_order"]
    observation_groups = (
        [
            torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]], dtype=torch.float32),
            torch.tensor([[[[4.0, 5.0, 6.0]]]], dtype=torch.float32),
        ],
        [
            torch.tensor([[[[0.0, 0.25, 0.5]]]], dtype=torch.float32),
            torch.tensor([[[[0.75, 1.0, 1.25]]]], dtype=torch.float32),
        ],
        [
            torch.tensor([[[[0.0, 10.0], [20.0, 30.0]]]], dtype=torch.float32),
            torch.tensor([[[[40.0, 50.0]]]], dtype=torch.float32),
        ],
    )
    raw_populations: list[dict[str, Any]] = []
    for kind, observations in zip(
        population_order,
        observation_groups,
        strict=True,
    ):
        records: list[dict[str, Any]] = []
        for sequence_index, (observation, generation_digest) in enumerate(
            zip(observations, generation_digests, strict=True)
        ):
            _, record = raw_module.encode_content_routing_reference_raw_member(
                reference_observation_kind=kind,
                reference_observation_member_sequence_index=sequence_index,
                generation_input_identity_digest=generation_digest,
                observation=observation,
            )
            records.append(record)
        raw_populations.append(
            {
                "reference_observation_kind": kind,
                "raw_member_file_records": records,
            }
        )

    registry_arguments = {
        "method_parameter_partition_id": "method-parameter-partition-v1",
        "prompt_projection": prompt_projection,
        "seed_projection_random": seed_projection,
        "generation_input_identity_digests": generation_digests,
        "gradient_observations": observation_groups[0],
        "response_observations": observation_groups[1],
        "sensitivity_observations": observation_groups[2],
        "formal_execution_lock_digest": formal_lock_digest,
        "dependency_profile_digest": dependency_digest,
        "runtime_component_identity_payload": runtime_payload,
    }
    candidate_registry_bytes = (
        payload_module.assemble_content_routing_reference_registry_payload(
            **registry_arguments
        )
    )
    return {
        "candidate_registry_bytes": candidate_registry_bytes,
        "contract": contract,
        "generation_input_records": generation_records,
        "observation_groups": observation_groups,
        "raw_member_populations": raw_populations,
        "registry_arguments": registry_arguments,
    }


def _arguments(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "code_commit": CODE_COMMIT,
        "generation_input_records": copy.deepcopy(
            bundle["generation_input_records"]
        ),
        "raw_member_populations": copy.deepcopy(bundle["raw_member_populations"]),
        "candidate_registry_bytes": bundle["candidate_registry_bytes"],
    }


def _assemble(bundle: dict[str, Any], **overrides: Any) -> bytes:
    arguments = _arguments(bundle)
    arguments.update(overrides)
    return manifest_module.assemble_content_routing_reference_materialization_manifest(
        **arguments
    )


def _decoded(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"))
    assert type(value) is dict
    return value


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def test_manifest_assembler_has_one_keyword_only_public_interface() -> None:
    assert manifest_module.__all__ == [
        "assemble_content_routing_reference_materialization_manifest"
    ]
    signature = inspect.signature(
        manifest_module.assemble_content_routing_reference_materialization_manifest
    )
    assert tuple(signature.parameters) == (
        "code_commit",
        "generation_input_records",
        "raw_member_populations",
        "candidate_registry_bytes",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    assert signature.return_annotation == "bytes"


def test_manifest_bytes_and_identity_dag_are_independently_recomputed(
    valid_bundle: dict[str, Any],
) -> None:
    raw = _assemble(valid_bundle)
    manifest = _decoded(raw)
    contract = valid_bundle["contract"]
    materialization = contract["materialization_contract"]
    manifest_contract = materialization["materialization_manifest_contract"]
    registry = _decoded(valid_bundle["candidate_registry_bytes"])
    generation_records = valid_bundle["generation_input_records"]

    assert raw == _canonical_bytes(manifest)
    semantic_payload = dict(manifest)
    embedded = semantic_payload.pop(
        "content_routing_reference_materialization_manifest_digest"
    )
    assert embedded == _stable_digest(semantic_payload)
    assert manifest["schema_version"] == manifest_contract["schema_token"]
    assert manifest["supports_paper_claim"] is False
    assert manifest["qualification_report_path"] == "qualification_report.json"
    assert manifest["generation_input_records"] == generation_records
    assert manifest["raw_member_populations"] == valid_bundle[
        "raw_member_populations"
    ]
    assert manifest["method_parameter_partition_id"] == registry[
        "method_parameter_partition_id"
    ]
    assert manifest["method_parameter_sample_count"] == len(generation_records)
    assert manifest["formal_method_config_digest"] == generation_records[0][
        "generation_input_identity_payload"
    ]["formal_method_config_digest"]
    assert manifest["candidate_registry"] == {
        "path": materialization["candidate_registry_file_record_contract"][
            "field_rules"
        ]["path"]["exact_value"],
        "content_routing_reference_registry_digest": registry[
            "content_routing_reference_registry_digest"
        ],
        "content_routing_reference_registry_file_sha256": hashlib.sha256(
            valid_bundle["candidate_registry_bytes"]
        ).hexdigest(),
    }
    for forbidden in manifest_contract["forbidden_fields"]:
        assert forbidden not in manifest


def test_raw_member_projection_digests_bind_each_registry_population(
    valid_bundle: dict[str, Any],
) -> None:
    manifest = _decoded(_assemble(valid_bundle))
    registry = _decoded(valid_bundle["candidate_registry_bytes"])
    contract = valid_bundle["contract"]
    projection_fields = contract["materialization_contract"][
        "raw_member_population_contract"
    ]["registry_member_projection_fields"]
    registry_by_kind = {
        population["reference_observation_kind"]: population
        for population in registry["content_routing_reference_populations"]
    }

    for population in manifest["raw_member_populations"]:
        projected = [
            {field_name: record[field_name] for field_name in projection_fields}
            for record in population["raw_member_file_records"]
        ]
        expected_digest = _stable_digest(projected)
        kind = population["reference_observation_kind"]
        assert expected_digest == registry_by_kind[kind][
            "reference_observation_member_records_digest"
        ]
        assert all(
            set(record) == set(contract["member_record_field_rules"])
            for record in projected
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "manifest_nested_mapping",
        "projection_order",
        "projection_duplicate",
        "raw_projection_rule",
        "member_digest_predicate",
        "member_digest_rule",
        "candidate_file_digest_rule",
        "manifest_semantic_digest_rule",
    ),
)
def test_contract_drift_fails_before_candidate_bytes_are_parsed(
    monkeypatch: pytest.MonkeyPatch,
    valid_bundle: dict[str, Any],
    mutation: str,
) -> None:
    contract = copy.deepcopy(valid_bundle["contract"])
    materialization = contract["materialization_contract"]
    if mutation == "manifest_nested_mapping":
        del materialization["materialization_manifest_contract"][
            "nested_contracts"
        ]["candidate_registry"]
    elif mutation == "projection_order":
        projection = list(
            materialization["raw_member_population_contract"][
                "registry_member_projection_fields"
            ]
        )
        projection[0], projection[1] = projection[1], projection[0]
        materialization["raw_member_population_contract"][
            "registry_member_projection_fields"
        ] = tuple(projection)
    elif mutation == "projection_duplicate":
        projection = list(
            materialization["raw_member_population_contract"][
                "registry_member_projection_fields"
            ]
        )
        projection[-1] = projection[0]
        materialization["raw_member_population_contract"][
            "registry_member_projection_fields"
        ] = tuple(projection)
    elif mutation == "raw_projection_rule":
        materialization["raw_member_file_record_contract"]["field_rules"][
            "tensor_content_sha256"
        ]["predicate"] = "nonempty_exact_str"
    elif mutation == "member_digest_predicate":
        contract["population_field_rules"][
            "reference_observation_member_records_digest"
        ]["predicate"] = "nonempty_exact_str"
    elif mutation == "member_digest_rule":
        contract["population_field_rules"][
            "reference_observation_member_records_digest"
        ]["digest_rule"] = "build_stable_digest(wrong_ordered_member_list)"
    elif mutation == "candidate_file_digest_rule":
        materialization["candidate_registry_file_record_contract"][
            "file_sha256_rule"
        ] = "sha256(wrong_candidate_bytes)"
    elif mutation == "manifest_semantic_digest_rule":
        materialization["materialization_manifest_contract"][
            "semantic_digest_rule"
        ] = "build_stable_digest(wrong_manifest_payload)"
    else:  # pragma: no cover
        raise AssertionError(mutation)

    parse_calls = 0

    def forbidden_parser(_payload: bytes) -> dict[str, Any]:
        nonlocal parse_calls
        parse_calls += 1
        raise AssertionError("candidate bytes parsed before contract preflight")

    monkeypatch.setattr(manifest_module, "_load_machine_contract", lambda: contract)
    monkeypatch.setattr(manifest_module, "_strict_json_object", forbidden_parser)
    with pytest.raises(ValueError):
        _assemble(valid_bundle)
    assert parse_calls == 0


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("code_commit", "not-a-commit"),
        ("generation_input_records", ()),
        ("generation_input_records", []),
        ("raw_member_populations", ()),
        ("raw_member_populations", []),
        ("candidate_registry_bytes", bytearray(b"{}")),
    ),
)
def test_exact_input_containers_and_static_values_fail_closed(
    valid_bundle: dict[str, Any],
    field: str,
    invalid: Any,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _assemble(valid_bundle, **{field: invalid})


def test_generation_sequence_digest_and_shared_identity_fail_closed(
    valid_bundle: dict[str, Any],
) -> None:
    mutations: list[list[dict[str, Any]]] = []

    wrong_index = copy.deepcopy(valid_bundle["generation_input_records"])
    wrong_index[1]["reference_observation_member_sequence_index"] = 7
    mutations.append(wrong_index)

    wrong_digest = copy.deepcopy(valid_bundle["generation_input_records"])
    wrong_digest[0]["generation_input_identity_digest"] = _sha("wrong-generation")
    mutations.append(wrong_digest)

    wrong_prompt_digest = copy.deepcopy(valid_bundle["generation_input_records"])
    payload = wrong_prompt_digest[0]["generation_input_identity_payload"]
    payload["prompt_text_digest"] = _sha("wrong-prompt")
    wrong_prompt_digest[0]["generation_input_identity_digest"] = _stable_digest(
        payload
    )
    mutations.append(wrong_prompt_digest)

    divergent_method = copy.deepcopy(valid_bundle["generation_input_records"])
    payload = divergent_method[1]["generation_input_identity_payload"]
    payload["formal_method_config_digest"] = _sha("other-method")
    divergent_method[1]["generation_input_identity_digest"] = _stable_digest(payload)
    mutations.append(divergent_method)

    for records in mutations:
        with pytest.raises(ValueError):
            _assemble(valid_bundle, generation_input_records=records)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("path", "raw/latent_response/not-the-fixed-path.f32be"),
        ("tensor_dtype", "torch.float64"),
    ),
)
def test_fixed_path_and_dtype_drift_fail_before_projection_digest(
    monkeypatch: pytest.MonkeyPatch,
    valid_bundle: dict[str, Any],
    field: str,
    replacement: str,
) -> None:
    populations = copy.deepcopy(valid_bundle["raw_member_populations"])
    populations[0]["raw_member_file_records"][0][field] = replacement
    projection_calls = 0

    def forbidden_projection(*_args: Any, **_kwargs: Any) -> str:
        nonlocal projection_calls
        projection_calls += 1
        raise AssertionError("projection digest ran before fixed-field validation")

    monkeypatch.setattr(
        manifest_module,
        "_project_raw_population_registry_member_digest",
        forbidden_projection,
    )
    with pytest.raises(ValueError):
        _assemble(valid_bundle, raw_member_populations=populations)
    assert projection_calls == 0


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("sha256", "a" * 64),
        ("tensor_shape", [1, 1, 3, 2]),
    ),
)
def test_legal_manifest_only_identity_changes_do_not_change_registry_projection(
    valid_bundle: dict[str, Any],
    field: str,
    replacement: Any,
) -> None:
    baseline = _decoded(_assemble(valid_bundle))
    populations = copy.deepcopy(valid_bundle["raw_member_populations"])
    populations[0]["raw_member_file_records"][0][field] = replacement
    changed = _decoded(
        _assemble(valid_bundle, raw_member_populations=populations)
    )

    assert changed["candidate_registry"] == baseline["candidate_registry"]
    assert changed["raw_member_populations"] != baseline["raw_member_populations"]
    assert changed[
        "content_routing_reference_materialization_manifest_digest"
    ] != baseline["content_routing_reference_materialization_manifest_digest"]
    assert _canonical_bytes(changed) != _canonical_bytes(baseline)


@pytest.mark.parametrize("population_index", (0, 1, 2))
def test_projection_identity_change_is_rejected_against_candidate_registry(
    valid_bundle: dict[str, Any],
    population_index: int,
) -> None:
    populations = copy.deepcopy(valid_bundle["raw_member_populations"])
    populations[population_index]["raw_member_file_records"][0][
        "tensor_content_sha256"
    ] = _sha(f"different-declared-tensor-{population_index}")
    with pytest.raises(ValueError, match="member-record digest"):
        _assemble(valid_bundle, raw_member_populations=populations)


def test_internally_valid_but_unrelated_registry_is_rejected(
    valid_bundle: dict[str, Any],
) -> None:
    arguments = copy.deepcopy(valid_bundle["registry_arguments"])
    arguments["gradient_observations"] = [
        observation + 100.0
        for observation in arguments["gradient_observations"]
    ]
    unrelated = payload_module.assemble_content_routing_reference_registry_payload(
        **arguments
    )
    registry = _decoded(unrelated)
    semantic = dict(registry)
    embedded = semantic.pop("content_routing_reference_registry_digest")
    assert embedded == _stable_digest(semantic)

    with pytest.raises(ValueError, match="member-record digest"):
        _assemble(valid_bundle, candidate_registry_bytes=unrelated)


def test_registry_identity_drift_is_rejected_after_internal_validation(
    valid_bundle: dict[str, Any],
) -> None:
    registry = _decoded(valid_bundle["candidate_registry_bytes"])
    registry["formal_execution_lock_digest"] = _sha("different-formal-lock")
    semantic = dict(registry)
    semantic.pop("content_routing_reference_registry_digest")
    registry["content_routing_reference_registry_digest"] = _stable_digest(semantic)
    changed_bytes = _canonical_bytes(registry)

    with pytest.raises(ValueError, match="formal_execution_lock_digest"):
        _assemble(valid_bundle, candidate_registry_bytes=changed_bytes)


def test_repeated_calls_are_deterministic_and_do_not_mutate_inputs(
    valid_bundle: dict[str, Any],
) -> None:
    arguments = _arguments(valid_bundle)
    before = copy.deepcopy(arguments)
    first = (
        manifest_module.assemble_content_routing_reference_materialization_manifest(
            **arguments
        )
    )
    second = (
        manifest_module.assemble_content_routing_reference_materialization_manifest(
            **arguments
        )
    )
    assert first == second
    assert arguments == before


def test_assembler_performs_no_artifact_io_or_resource_execution(
    monkeypatch: pytest.MonkeyPatch,
    valid_bundle: dict[str, Any],
) -> None:
    contract = copy.deepcopy(valid_bundle["contract"])

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("artifact I/O or fixed-path loading is forbidden")

    monkeypatch.setattr(manifest_module, "_load_machine_contract", lambda: contract)
    monkeypatch.setattr(
        registry_module,
        "load_content_routing_reference_registry",
        forbidden,
    )
    monkeypatch.setattr(builtins, "open", forbidden)

    assert _assemble(valid_bundle).endswith(b"\n")
    assert not FIXED_REGISTRY.exists()
    source = inspect.getsource(manifest_module)
    assert "transformers" not in source
    assert "diffusers" not in source
    assert "torch.cuda" not in source
    assert "os.replace" not in source
    assert "fsync" not in source
