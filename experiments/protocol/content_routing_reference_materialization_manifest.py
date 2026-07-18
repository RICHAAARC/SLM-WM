"""Assemble a governed content-routing materialization manifest in memory."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any

from experiments.protocol.content_routing_reference_generation_input import (
    _preflight_generation_input_contract,
)
from experiments.protocol.content_routing_reference_raw_member import (
    _preflight_raw_member_contract,
)
from experiments.protocol.content_routing_reference_registry import (
    _load_machine_contract,
    _strict_json_object,
    _validate_content_routing_reference_registry,
    _validate_exact_object,
    _validate_rule,
)
from experiments.protocol.content_routing_reference_registry_payload import (
    _preflight_payload_assembler_contract,
)
from main.core.digest import build_stable_digest, stable_json_dumps


__all__ = ["assemble_content_routing_reference_materialization_manifest"]


_MATERIALIZATION_TOKEN = "content_routing_reference_materialization_contract_v1"
_MANIFEST_SEMANTIC_DIGEST_RULE = (
    "build_stable_digest(top_level_object_with_only_"
    "content_routing_reference_materialization_manifest_digest_removed)"
)
_CANONICAL_FILE_BYTES_RULE = (
    "stable_json_dumps(payload).encode('utf-8') plus one LF byte"
)
_MEMBER_RECORD_DIGEST_RULE = (
    "build_stable_digest(exact_ordered_member_record_list)"
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "content_routing_reference_materialization_manifest_digest",
        "supports_paper_claim",
        "code_commit",
        "method_parameter_partition_id",
        "method_parameter_prompt_list_digest",
        "method_parameter_seed_list_digest_random",
        "method_parameter_sample_count",
        "formal_method_config_digest",
        "formal_execution_lock_digest",
        "dependency_profile_digest",
        "model_id",
        "model_revision",
        "runtime_component_identity_digest",
        "generation_input_records",
        "raw_member_populations",
        "candidate_registry",
        "qualification_report_path",
    }
)
_EXPECTED_MANIFEST_RULES = {
    "schema_version": {
        "predicate": "exact_token_str",
        "exact_value": "content_routing_reference_materialization_manifest_v1",
    },
    "content_routing_reference_materialization_manifest_digest": {
        "predicate": "sha256_lower_hex_str",
    },
    "supports_paper_claim": {"predicate": "exact_bool", "exact_value": False},
    "code_commit": {"predicate": "git_commit_lower_hex_str"},
    "method_parameter_partition_id": {"predicate": "nonempty_exact_str"},
    "method_parameter_prompt_list_digest": {
        "predicate": "sha256_lower_hex_str",
    },
    "method_parameter_seed_list_digest_random": {
        "predicate": "sha256_lower_hex_str",
    },
    "method_parameter_sample_count": {"predicate": "strict_positive_int"},
    "formal_method_config_digest": {"predicate": "sha256_lower_hex_str"},
    "formal_execution_lock_digest": {"predicate": "sha256_lower_hex_str"},
    "dependency_profile_digest": {"predicate": "sha256_lower_hex_str"},
    "model_id": {
        "predicate": "exact_token_str",
        "exact_value": "stabilityai/stable-diffusion-3.5-medium",
    },
    "model_revision": {
        "predicate": "exact_token_str",
        "exact_value": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
    },
    "runtime_component_identity_digest": {"predicate": "sha256_lower_hex_str"},
    "generation_input_records": {"predicate": "exact_list"},
    "raw_member_populations": {
        "predicate": "exact_list",
        "exact_length": 3,
    },
    "candidate_registry": {"predicate": "exact_object"},
    "qualification_report_path": {
        "predicate": "exact_token_str",
        "exact_value": "qualification_report.json",
    },
}
_EXPECTED_MANIFEST_NESTED_CONTRACTS = {
    "generation_input_records": "ordered_generation_input_record_contract",
    "raw_member_populations": "raw_member_population_contract",
    "candidate_registry": "candidate_registry_file_record_contract",
}
_FORBIDDEN_MANIFEST_FIELDS = (
    "content_routing_reference_qualification_report_digest",
    "content_routing_reference_qualification_report_file_sha256",
    "content_routing_reference_qualification_ready",
    "qualification_checks",
    "qualification_errors",
)
_EXPECTED_RAW_POPULATION_RULES = {
    "reference_observation_kind": {
        "predicate": "exact_token_str",
        "exact_value_source": "population_order_entry",
    },
    "raw_member_file_records": {"predicate": "exact_list"},
}
_EXPECTED_RAW_POPULATION_VALUES = {
    "container_predicate": "exact_list",
    "exact_length": 3,
    "nested_contracts": {
        "raw_member_file_records": "raw_member_file_record_contract"
    },
    "order_source": "population_order",
    "member_length_rule": (
        "raw_member_file_record_count_equals_method_parameter_sample_count"
    ),
    "member_order_rule": "raw_member_file_records_follow_sequence_index_order",
    "cross_field_invariants": (
        "same_sequence_index_has_same_generation_input_identity_digest_"
        "across_populations",
        "raw_path_file_sha_shape_and_dtype_are_manifest_only_not_registry_"
        "member_fields",
    ),
}
_EXPECTED_CANDIDATE_REGISTRY_RULES = {
    "path": {
        "predicate": "exact_token_str",
        "exact_value": "content_routing_reference_registry.json",
    },
    "content_routing_reference_registry_digest": {
        "predicate": "sha256_lower_hex_str"
    },
    "content_routing_reference_registry_file_sha256": {
        "predicate": "sha256_lower_hex_str"
    },
}
_EXPECTED_CANDIDATE_REGISTRY_VALUES = {
    "semantic_digest_rule": "registry_semantic_digest_rule",
    "file_sha256_rule": "sha256(exact_candidate_registry_file_bytes)",
    "cross_field_invariants": (
        "registry_partition_sample_model_dependency_formal_and_runtime_"
        "identities_equal_manifest",
        "candidate_registry_bytes_equal_payload_assembler_canonical_bytes",
    ),
}
_EXPECTED_MANIFEST_VALUES = {
    "schema_token": "content_routing_reference_materialization_manifest_v1",
    "filename": "materialization_manifest.json",
    "semantic_digest_rule": _MANIFEST_SEMANTIC_DIGEST_RULE,
    "canonical_file_bytes_rule": _CANONICAL_FILE_BYTES_RULE,
    "file_sha256_location": "qualification_report_and_external_promotion_input_only",
    "candidate_root_rule": (
        "outputs/content_routing_reference_materialization/"
        "<content_routing_reference_materialization_manifest_digest>/"
    ),
    "candidate_directory_identity_rule": (
        "candidate_root_final_component_equals_manifest_semantic_digest"
    ),
    "forbidden_fields": _FORBIDDEN_MANIFEST_FIELDS,
}
_SHARED_GENERATION_IDENTITY_FIELDS = (
    "formal_method_config_digest",
    "formal_execution_lock_digest",
    "dependency_profile_digest",
    "model_id",
    "model_revision",
    "runtime_component_identity_digest",
)


def _require_exact_contract_object(
    value: Any,
    *,
    expected_fields: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected_fields:
        raise ValueError(f"{label} has missing or extra fields")
    return value


def _preflight_materialization_manifest_assembler_contract(
    contract: Any,
) -> dict[str, Any]:
    if type(contract) is not dict:
        raise ValueError("content-routing reference machine contract is invalid")
    materialization = contract.get("materialization_contract")
    if type(materialization) is not dict or materialization.get(
        "contract_schema_token"
    ) != _MATERIALIZATION_TOKEN:
        raise ValueError("materialization manifest contract is incomplete")

    payload_parts = _preflight_payload_assembler_contract(contract)
    payload_rules, generation_record_rules, _ = (
        _preflight_generation_input_contract(contract)
    )
    raw_record_rules, raw_population_order, path_templates = (
        _preflight_raw_member_contract(contract)
    )
    population_order = payload_parts["population_order"]
    if raw_population_order != population_order:
        raise ValueError("raw and registry population order is inconsistent")

    raw_population_contract = _require_exact_contract_object(
        materialization.get("raw_member_population_contract"),
        expected_fields=frozenset(
            {
                "container_predicate",
                "exact_length",
                "field_rules",
                "nested_contracts",
                "order_source",
                "member_length_rule",
                "member_order_rule",
                "registry_member_projection_fields",
                "cross_field_invariants",
            }
        ),
        label="raw_member_population_contract",
    )
    if raw_population_contract.get("field_rules") != _EXPECTED_RAW_POPULATION_RULES:
        raise ValueError("raw population field rules are invalid")
    for field_name, expected in _EXPECTED_RAW_POPULATION_VALUES.items():
        if raw_population_contract.get(field_name) != expected:
            raise ValueError(f"raw population {field_name} is invalid")

    member_rules = payload_parts["member_rules"]
    projection_fields = raw_population_contract.get(
        "registry_member_projection_fields"
    )
    if (
        type(projection_fields) is not tuple
        or not projection_fields
        or len(set(projection_fields)) != len(projection_fields)
        or projection_fields != tuple(member_rules)
    ):
        raise ValueError("registry member projection fields are invalid")
    for field_name in projection_fields:
        if (
            field_name not in raw_record_rules
            or raw_record_rules[field_name] != member_rules[field_name]
        ):
            raise ValueError("raw and registry member rules are inconsistent")

    population_digest_rule = payload_parts["population_rules"][
        "reference_observation_member_records_digest"
    ]
    if population_digest_rule != {
        "predicate": "sha256_lower_hex_str",
        "digest_rule": _MEMBER_RECORD_DIGEST_RULE,
    }:
        raise ValueError("registry member-record digest rule is invalid")

    candidate_contract = _require_exact_contract_object(
        materialization.get("candidate_registry_file_record_contract"),
        expected_fields=frozenset(
            {"field_rules", *_EXPECTED_CANDIDATE_REGISTRY_VALUES}
        ),
        label="candidate_registry_file_record_contract",
    )
    if candidate_contract.get("field_rules") != _EXPECTED_CANDIDATE_REGISTRY_RULES:
        raise ValueError("candidate registry field rules are invalid")
    for field_name, expected in _EXPECTED_CANDIDATE_REGISTRY_VALUES.items():
        if candidate_contract.get(field_name) != expected:
            raise ValueError(f"candidate registry {field_name} is invalid")

    manifest_contract = _require_exact_contract_object(
        materialization.get("materialization_manifest_contract"),
        expected_fields=frozenset(
            {
                "schema_token",
                "filename",
                "field_rules",
                "nested_contracts",
                "semantic_digest_rule",
                "canonical_file_bytes_rule",
                "file_sha256_location",
                "candidate_root_rule",
                "candidate_directory_identity_rule",
                "forbidden_fields",
            }
        ),
        label="materialization_manifest_contract",
    )
    manifest_rules = manifest_contract.get("field_rules")
    if (
        type(manifest_rules) is not dict
        or set(manifest_rules) != _MANIFEST_FIELDS
        or manifest_rules != _EXPECTED_MANIFEST_RULES
    ):
        raise ValueError("materialization manifest field rules are invalid")
    if manifest_contract.get("nested_contracts") != (
        _EXPECTED_MANIFEST_NESTED_CONTRACTS
    ):
        raise ValueError("materialization manifest nested contracts are invalid")
    for field_name, expected in _EXPECTED_MANIFEST_VALUES.items():
        if manifest_contract.get(field_name) != expected:
            raise ValueError(f"materialization manifest {field_name} is invalid")
    for field_name, rule in manifest_rules.items():
        if type(rule) is not dict or type(rule.get("predicate")) is not str:
            raise ValueError(f"materialization manifest rule is invalid: {field_name}")
        if "exact_value" in rule:
            _validate_rule(
                rule["exact_value"],
                rule,
                label=f"materialization_manifest_contract.{field_name}",
            )

    return {
        "candidate_rules": candidate_contract["field_rules"],
        "generation_payload_rules": payload_rules,
        "generation_record_rules": generation_record_rules,
        "manifest_rules": manifest_rules,
        "member_rules": member_rules,
        "path_templates": path_templates,
        "population_order": population_order,
        "projection_fields": projection_fields,
        "raw_population_rules": raw_population_contract["field_rules"],
        "raw_record_rules": raw_record_rules,
        "registry_population_rules": payload_parts["population_rules"],
        "registry_top_rules": payload_parts["top_rules"],
    }


def _snapshot_exact_list(value: Any, *, label: str) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{label} must be an exact list")
    return deepcopy(value)


def _validated_generation_input_manifest_records(
    generation_input_records: Any,
    *,
    contract_parts: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = _snapshot_exact_list(
        generation_input_records,
        label="generation_input_records",
    )
    if not records:
        raise ValueError("generation_input_records must not be empty")

    record_rules = contract_parts["generation_record_rules"]
    payload_rules = contract_parts["generation_payload_rules"]
    prompt_projection: list[dict[str, Any]] = []
    seed_projection: list[int] = []
    generation_digests: list[str] = []
    shared_identity: dict[str, Any] | None = None
    for sequence_index, record in enumerate(records):
        _validate_exact_object(
            record,
            record_rules,
            label=f"generation_input_records[{sequence_index}]",
        )
        if record["reference_observation_member_sequence_index"] != sequence_index:
            raise ValueError("generation input sequence indices are not contiguous")
        payload = _validate_exact_object(
            record["generation_input_identity_payload"],
            payload_rules,
            label=(
                f"generation_input_records[{sequence_index}]."
                "generation_input_identity_payload"
            ),
        )
        if payload["prompt_text_digest"] != build_stable_digest(
            {"prompt_text": payload["prompt_text"]}
        ):
            raise ValueError("generation prompt text digest is invalid")
        if record["generation_input_identity_digest"] != build_stable_digest(payload):
            raise ValueError("generation input identity digest is invalid")

        current_identity = {
            field_name: payload[field_name]
            for field_name in _SHARED_GENERATION_IDENTITY_FIELDS
        }
        if shared_identity is None:
            shared_identity = current_identity
        elif current_identity != shared_identity:
            raise ValueError("generation input shared identities are inconsistent")
        prompt_projection.append(
            {
                "prompt_id": payload["prompt_id"],
                "prompt_text_digest": payload["prompt_text_digest"],
            }
        )
        seed_projection.append(payload["generation_seed_random"])
        generation_digests.append(record["generation_input_identity_digest"])

    assert shared_identity is not None
    return records, {
        **shared_identity,
        "generation_digests": generation_digests,
        "method_parameter_prompt_list_digest": build_stable_digest(
            prompt_projection
        ),
        "method_parameter_sample_count": len(records),
        "method_parameter_seed_list_digest_random": build_stable_digest(
            seed_projection
        ),
    }


def _project_raw_population_registry_member_digest(
    records: list[dict[str, Any]],
    *,
    projection_fields: tuple[str, ...],
    member_rules: dict[str, dict[str, Any]],
    label: str,
) -> str:
    projected_records: list[dict[str, Any]] = []
    for member_index, record in enumerate(records):
        projected = {field_name: record[field_name] for field_name in projection_fields}
        _validate_exact_object(
            projected,
            member_rules,
            label=f"{label}.projected_member[{member_index}]",
        )
        projected_records.append(projected)
    return build_stable_digest(projected_records)


def _validated_raw_member_manifest_populations(
    raw_member_populations: Any,
    *,
    contract_parts: dict[str, Any],
    generation_identity: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    populations = _snapshot_exact_list(
        raw_member_populations,
        label="raw_member_populations",
    )
    population_order = contract_parts["population_order"]
    if len(populations) != len(population_order):
        raise ValueError("raw member population count is invalid")

    sample_count = generation_identity["method_parameter_sample_count"]
    generation_digests = generation_identity["generation_digests"]
    projected_digests: dict[str, str] = {}
    for population_index, expected_kind in enumerate(population_order):
        population = _validate_exact_object(
            populations[population_index],
            contract_parts["raw_population_rules"],
            label=f"raw_member_populations[{population_index}]",
        )
        if population["reference_observation_kind"] != expected_kind:
            raise ValueError("raw member population order is invalid")
        records = population["raw_member_file_records"]
        if type(records) is not list or len(records) != sample_count:
            raise ValueError("raw member record count is invalid")
        for sequence_index, record in enumerate(records):
            _validate_exact_object(
                record,
                contract_parts["raw_record_rules"],
                label=(
                    f"raw_member_populations[{population_index}]."
                    f"raw_member_file_records[{sequence_index}]"
                ),
            )
            if record["reference_observation_kind"] != expected_kind:
                raise ValueError("raw member kind does not match its population")
            if (
                record["reference_observation_member_sequence_index"]
                != sequence_index
            ):
                raise ValueError("raw member sequence indices are not contiguous")
            if record["generation_input_identity_digest"] != generation_digests[
                sequence_index
            ]:
                raise ValueError("raw member generation identity is inconsistent")
            expected_path = contract_parts["path_templates"][expected_kind].format(
                sequence_index=sequence_index
            )
            if record["path"] != expected_path:
                raise ValueError("raw member path does not match its fixed template")

        projected_digests[expected_kind] = (
            _project_raw_population_registry_member_digest(
                records,
                projection_fields=contract_parts["projection_fields"],
                member_rules=contract_parts["member_rules"],
                label=f"raw_member_populations[{population_index}]",
            )
        )
    return populations, projected_digests


def _validated_candidate_registry_manifest_record(
    candidate_registry_bytes: Any,
    *,
    contract: dict[str, Any],
    contract_parts: dict[str, Any],
    generation_identity: dict[str, Any],
    projected_member_digests: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(candidate_registry_bytes) is not bytes:
        raise TypeError("candidate_registry_bytes must be exact bytes")
    registry = _strict_json_object(candidate_registry_bytes)
    registry_digest = registry.get("content_routing_reference_registry_digest")
    _validate_rule(
        registry_digest,
        contract_parts["registry_top_rules"][
            "content_routing_reference_registry_digest"
        ],
        label="candidate_registry.content_routing_reference_registry_digest",
    )
    _validate_content_routing_reference_registry(
        registry,
        raw_payload=candidate_registry_bytes,
        expected_registry_digest=registry_digest,
        contract=contract,
    )

    expected_registry_identity = {
        "method_parameter_prompt_list_digest": generation_identity[
            "method_parameter_prompt_list_digest"
        ],
        "method_parameter_seed_list_digest_random": generation_identity[
            "method_parameter_seed_list_digest_random"
        ],
        "method_parameter_sample_count": generation_identity[
            "method_parameter_sample_count"
        ],
        "formal_execution_lock_digest": generation_identity[
            "formal_execution_lock_digest"
        ],
        "dependency_profile_digest": generation_identity[
            "dependency_profile_digest"
        ],
        "model_id": generation_identity["model_id"],
        "model_revision": generation_identity["model_revision"],
        "runtime_component_identity_digest": generation_identity[
            "runtime_component_identity_digest"
        ],
    }
    for field_name, expected_value in expected_registry_identity.items():
        if registry[field_name] != expected_value:
            raise ValueError(f"candidate registry identity mismatch: {field_name}")

    registry_populations = registry["content_routing_reference_populations"]
    for population, expected_kind in zip(
        registry_populations,
        contract_parts["population_order"],
        strict=True,
    ):
        if population["reference_observation_member_records_digest"] != (
            projected_member_digests[expected_kind]
        ):
            raise ValueError("candidate registry member-record digest is inconsistent")

    record = {
        "path": contract_parts["candidate_rules"]["path"]["exact_value"],
        "content_routing_reference_registry_digest": registry_digest,
        "content_routing_reference_registry_file_sha256": hashlib.sha256(
            candidate_registry_bytes
        ).hexdigest(),
    }
    _validate_exact_object(
        record,
        contract_parts["candidate_rules"],
        label="candidate_registry",
    )
    return registry, record


def assemble_content_routing_reference_materialization_manifest(
    *,
    code_commit: Any,
    generation_input_records: Any,
    raw_member_populations: Any,
    candidate_registry_bytes: Any,
) -> bytes:
    """Return canonical materialization-manifest bytes without persistence."""

    contract = _load_machine_contract()
    contract_parts = _preflight_materialization_manifest_assembler_contract(contract)
    _validate_rule(
        code_commit,
        contract_parts["manifest_rules"]["code_commit"],
        label="code_commit",
    )
    generation_records, generation_identity = (
        _validated_generation_input_manifest_records(
            generation_input_records,
            contract_parts=contract_parts,
        )
    )
    raw_populations, projected_member_digests = (
        _validated_raw_member_manifest_populations(
            raw_member_populations,
            contract_parts=contract_parts,
            generation_identity=generation_identity,
        )
    )
    registry, candidate_record = _validated_candidate_registry_manifest_record(
        candidate_registry_bytes,
        contract=contract,
        contract_parts=contract_parts,
        generation_identity=generation_identity,
        projected_member_digests=projected_member_digests,
    )

    manifest_rules = contract_parts["manifest_rules"]
    manifest = {
        "schema_version": manifest_rules["schema_version"]["exact_value"],
        "supports_paper_claim": manifest_rules["supports_paper_claim"][
            "exact_value"
        ],
        "code_commit": code_commit,
        "method_parameter_partition_id": registry[
            "method_parameter_partition_id"
        ],
        "method_parameter_prompt_list_digest": generation_identity[
            "method_parameter_prompt_list_digest"
        ],
        "method_parameter_seed_list_digest_random": generation_identity[
            "method_parameter_seed_list_digest_random"
        ],
        "method_parameter_sample_count": generation_identity[
            "method_parameter_sample_count"
        ],
        "formal_method_config_digest": generation_identity[
            "formal_method_config_digest"
        ],
        "formal_execution_lock_digest": generation_identity[
            "formal_execution_lock_digest"
        ],
        "dependency_profile_digest": generation_identity[
            "dependency_profile_digest"
        ],
        "model_id": generation_identity["model_id"],
        "model_revision": generation_identity["model_revision"],
        "runtime_component_identity_digest": generation_identity[
            "runtime_component_identity_digest"
        ],
        "generation_input_records": generation_records,
        "raw_member_populations": raw_populations,
        "candidate_registry": candidate_record,
        "qualification_report_path": manifest_rules["qualification_report_path"][
            "exact_value"
        ],
    }
    digest = build_stable_digest(manifest)
    manifest["content_routing_reference_materialization_manifest_digest"] = digest
    _validate_exact_object(manifest, manifest_rules, label="materialization_manifest")
    raw_manifest = stable_json_dumps(manifest).encode("utf-8") + b"\n"
    if _strict_json_object(raw_manifest) != manifest:
        raise ValueError("materialization manifest bytes are not canonical")
    return raw_manifest
