"""Assemble governed content-routing reference registry bytes in memory.

The assembler consumes already materialized CPU observations and caller-supplied
identities.  It does not run a model, qualify provenance, or persist an artifact.
"""

from __future__ import annotations

import struct
from typing import Any

import torch

from experiments.protocol.content_routing_reference_quantile import (
    _nearest_rank_p95,
    _positive_population,
    _snapshot_observations,
)
from experiments.protocol.content_routing_reference_registry import (
    _SUPPORTED_PREDICATES,
    _load_machine_contract,
    _validate_content_routing_reference_registry,
    _validate_exact_object,
    _validate_rule,
)
from main.core.digest import (
    build_stable_digest,
    stable_json_dumps,
    tensor_content_sha256,
)


__all__ = ["assemble_content_routing_reference_registry_payload"]


_TOP_LEVEL_FIELDS = frozenset(
    {
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
        "content_routing_reference_quantile_numerator",
        "content_routing_reference_quantile_denominator",
        "content_routing_reference_quantile_rank_rule",
        "content_routing_reference_quantile_index_rule",
        "content_routing_reference_populations",
        "reference_gradient",
        "reference_gradient_binary32_hex",
        "reference_response",
        "reference_response_binary32_hex",
        "reference_sensitivity",
        "reference_sensitivity_binary32_hex",
        "content_routing_reference_registry_digest",
    }
)
_POPULATION_FIELDS = frozenset(
    {
        "reference_observation_kind",
        "reference_observation_member_count",
        "reference_observation_positive_value_count",
        "reference_observation_member_records_digest",
        "tensor_content_sha256",
        "reference_observation_selected_rank",
        "reference_observation_selected_index",
    }
)
_MEMBER_RECORD_FIELDS = frozenset(
    {
        "reference_observation_kind",
        "reference_observation_member_sequence_index",
        "generation_input_identity_digest",
        "tensor_content_sha256",
    }
)
_PROMPT_CONTRACT_FIELDS = frozenset(
    {
        "container_predicate",
        "entry_predicate",
        "entry_field_rules",
        "order_source",
        "order_rule",
        "length_rule",
        "digest_rule",
    }
)
_PROMPT_ENTRY_FIELDS = frozenset({"prompt_id", "prompt_text_digest"})
_SEED_CONTRACT_FIELDS = frozenset(
    {
        "container_predicate",
        "element_predicate",
        "order_source",
        "order_rule",
        "length_rule",
        "digest_rule",
    }
)
_RUNTIME_CONTRACT_FIELDS = frozenset(
    {
        "schema_token",
        "component_config_digest_rule",
        "digest_rule",
        "field_rules",
        "public_probe_identity_contract",
        "cross_field_invariants",
        "forbidden_fields",
    }
)
_RUNTIME_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "model_id",
        "model_revision",
        "pipeline_class_name",
        "vae_class_name",
        "transformer_class_name",
        "scheduler_class_name",
        "pipeline_config_digest",
        "vae_config_digest",
        "transformer_config_digest",
        "scheduler_config_digest",
        "vae_scaling_factor",
        "vae_shift_factor",
        "vae_decode_protocol",
        "scheduler_inference_step_count",
        "callback_api",
        "previous_latent_callback_index",
        "current_latent_callback_index",
        "callback_latent_semantics",
        "decoded_rgb_protocol",
        "latent_torch_dtype",
        "reference_observation_dtype",
        "texture_formula_protocol_version",
        "latent_response_formula_protocol_version",
        "local_sensitivity_formula_protocol_version",
        "public_probe_identity",
        "dependency_profile_digest",
        "formal_execution_lock_digest",
    }
)
_PROBE_CONTRACT_FIELDS = frozenset({"field_rules", "domain_field_rules"})
_PROBE_FIELDS = frozenset({"prg_version", "key_material", "domain_fields"})
_PROBE_DOMAIN_FIELDS = frozenset({"purpose", "model_revision", "probe_version"})
_SCALAR_BINDING_FIELDS = frozenset({"scalar_field", "binary32_hex_field"})
_RULE_KEYS = frozenset(
    {
        "predicate",
        "exact_value",
        "exact_length",
        "exact_value_source",
        "digest_contract",
        "digest_rule",
    }
)
_TOP_EXACT_VALUE_FIELDS = frozenset(
    {
        "registry_schema",
        "model_id",
        "model_revision",
        "content_routing_reference_quantile_algorithm",
        "content_routing_reference_quantile_numerator",
        "content_routing_reference_quantile_denominator",
        "content_routing_reference_quantile_rank_rule",
        "content_routing_reference_quantile_index_rule",
    }
)
_TOP_EXACT_TOKEN_FIELDS = _TOP_EXACT_VALUE_FIELDS - {
    "content_routing_reference_quantile_numerator",
    "content_routing_reference_quantile_denominator",
}
_RUNTIME_UNFROZEN_VALUE_FIELDS = frozenset(
    {
        "pipeline_config_digest",
        "vae_config_digest",
        "transformer_config_digest",
        "scheduler_config_digest",
        "public_probe_identity",
        "dependency_profile_digest",
        "formal_execution_lock_digest",
    }
)
_RUNTIME_EXACT_VALUE_FIELDS = (
    _RUNTIME_PAYLOAD_FIELDS - _RUNTIME_UNFROZEN_VALUE_FIELDS
)
_RUNTIME_EXACT_FLOAT_FIELDS = frozenset(
    {"vae_scaling_factor", "vae_shift_factor"}
)
_RUNTIME_EXACT_POSITIVE_INT_FIELDS = frozenset(
    {"scheduler_inference_step_count"}
)
_RUNTIME_EXACT_NONNEGATIVE_INT_FIELDS = frozenset(
    {"previous_latent_callback_index", "current_latent_callback_index"}
)
_RUNTIME_EXACT_TOKEN_FIELDS = _RUNTIME_EXACT_VALUE_FIELDS - (
    _RUNTIME_EXACT_FLOAT_FIELDS
    | _RUNTIME_EXACT_POSITIVE_INT_FIELDS
    | _RUNTIME_EXACT_NONNEGATIVE_INT_FIELDS
)
_EXPECTED_TOP_DIGEST_CONTRACTS = {
    "method_parameter_prompt_list_digest": "prompt_projection_contract",
    "method_parameter_seed_list_digest_random": "seed_projection_contract",
    "runtime_component_identity_digest": (
        "runtime_component_identity_payload_contract"
    ),
}
_EXPECTED_POPULATION_SCALAR_BINDING = {
    "gradient_magnitude_rgb_pre_interpolation": {
        "scalar_field": "reference_gradient",
        "binary32_hex_field": "reference_gradient_binary32_hex",
    },
    "latent_response": {
        "scalar_field": "reference_response",
        "binary32_hex_field": "reference_response_binary32_hex",
    },
    "local_sensitivity_rgb_pre_interpolation": {
        "scalar_field": "reference_sensitivity",
        "binary32_hex_field": "reference_sensitivity_binary32_hex",
    },
}
_EXPECTED_DIGEST_RULES = {
    "prompt_projection": (
        "build_stable_digest(exact_ordered_prompt_projection_list)"
    ),
    "seed_projection": "build_stable_digest(exact_ordered_seed_projection_list)",
    "member_records": "build_stable_digest(exact_ordered_member_record_list)",
    "runtime_component_config": (
        "build_stable_digest(dict(actual_component.config))"
    ),
    "runtime_component_identity": (
        "build_stable_digest(exact_runtime_component_identity_payload)"
    ),
    "registry_semantic": (
        "build_stable_digest("
        "top_level_object_with_only_"
        "content_routing_reference_registry_digest_removed)"
    ),
    "registry_file": (
        "sha256(exact_utf8_stable_json_dumps_payload_plus_one_lf_byte)"
    ),
}
_EXPECTED_BINARY32_HEX_RULE = "struct.pack('>f', scalar).hex()"


def _require_exact_keys(value: Any, expected: frozenset[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} has missing or extra contract fields")
    return value


def _require_supported_rule_map(
    value: Any,
    expected: frozenset[str],
    predicates: dict[str, Any],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    rules = _require_exact_keys(value, expected, label=label)
    for field_name, rule in rules.items():
        if (
            type(rule) is not dict
            or not set(rule).issubset(_RULE_KEYS)
            or type(rule.get("predicate")) is not str
        ):
            raise ValueError(f"{label}.{field_name} is not a governed rule")
        predicate = rule["predicate"]
        if predicate not in predicates or predicate not in _SUPPORTED_PREDICATES:
            raise ValueError(f"{label}.{field_name} uses an unknown predicate")
    return rules


def _require_digest_rule(value: Any, *, expected: str, label: str) -> str:
    if type(value) is not str or value != expected:
        raise ValueError(f"{label} is not a governed digest rule")
    return value


def _require_exact_rule_value(rule: Any, *, label: str) -> Any:
    if type(rule) is not dict or "exact_value" not in rule:
        raise ValueError(f"{label} must retain its governed exact value")
    value = rule["exact_value"]
    _validate_rule(value, rule, label=label)
    return value


def _require_rule_attribute(
    rule: Any,
    *,
    attribute: str,
    expected: Any,
    label: str,
) -> None:
    if type(rule) is not dict or rule.get(attribute) != expected:
        raise ValueError(f"{label} does not retain governed {attribute}")


def _preflight_payload_assembler_contract(contract: Any) -> dict[str, Any]:
    if type(contract) is not dict:
        raise ValueError("content-routing reference machine contract must be an object")
    predicates = contract.get("type_predicates")
    if type(predicates) is not dict:
        raise ValueError("content-routing reference predicates must be an object")

    top_rules = _require_supported_rule_map(
        contract.get("registry_top_level_field_rules"),
        _TOP_LEVEL_FIELDS,
        predicates,
        label="registry_top_level_field_rules",
    )
    population_rules = _require_supported_rule_map(
        contract.get("population_field_rules"),
        _POPULATION_FIELDS,
        predicates,
        label="population_field_rules",
    )
    member_rules = _require_supported_rule_map(
        contract.get("member_record_field_rules"),
        _MEMBER_RECORD_FIELDS,
        predicates,
        label="member_record_field_rules",
    )
    for field_name in _TOP_EXACT_VALUE_FIELDS:
        _require_exact_rule_value(
            top_rules[field_name],
            label=f"registry_top_level_field_rules.{field_name}",
        )
    for field_name in _TOP_EXACT_TOKEN_FIELDS:
        _require_rule_attribute(
            top_rules[field_name],
            attribute="predicate",
            expected="exact_token_str",
            label=f"registry_top_level_field_rules.{field_name}",
        )
    for field_name in (
        "content_routing_reference_quantile_numerator",
        "content_routing_reference_quantile_denominator",
    ):
        _require_rule_attribute(
            top_rules[field_name],
            attribute="predicate",
            expected="strict_positive_int",
            label=f"registry_top_level_field_rules.{field_name}",
        )
    for field_name, digest_contract in _EXPECTED_TOP_DIGEST_CONTRACTS.items():
        _require_rule_attribute(
            top_rules[field_name],
            attribute="digest_contract",
            expected=digest_contract,
            label=f"registry_top_level_field_rules.{field_name}",
        )
        _require_rule_attribute(
            top_rules[field_name],
            attribute="predicate",
            expected="sha256_lower_hex_str",
            label=f"registry_top_level_field_rules.{field_name}",
        )
    _require_rule_attribute(
        top_rules["content_routing_reference_populations"],
        attribute="predicate",
        expected="exact_list",
        label="registry_top_level_field_rules.content_routing_reference_populations",
    )
    _require_rule_attribute(
        member_rules["reference_observation_kind"],
        attribute="predicate",
        expected="exact_token_str",
        label="member_record_field_rules.reference_observation_kind",
    )
    _require_rule_attribute(
        member_rules["reference_observation_kind"],
        attribute="exact_value_source",
        expected="parent_population_kind",
        label="member_record_field_rules.reference_observation_kind",
    )
    _require_rule_attribute(
        member_rules["generation_input_identity_digest"],
        attribute="predicate",
        expected="sha256_lower_hex_str",
        label="member_record_field_rules.generation_input_identity_digest",
    )
    _require_rule_attribute(
        population_rules["reference_observation_kind"],
        attribute="predicate",
        expected="exact_token_str",
        label="population_field_rules.reference_observation_kind",
    )
    _require_rule_attribute(
        population_rules["reference_observation_kind"],
        attribute="exact_value_source",
        expected="population_order_entry",
        label="population_field_rules.reference_observation_kind",
    )

    prompt_contract = _require_exact_keys(
        contract.get("prompt_projection_contract"),
        _PROMPT_CONTRACT_FIELDS,
        label="prompt_projection_contract",
    )
    prompt_rules = _require_supported_rule_map(
        prompt_contract.get("entry_field_rules"),
        _PROMPT_ENTRY_FIELDS,
        predicates,
        label="prompt_projection_contract.entry_field_rules",
    )
    seed_contract = _require_exact_keys(
        contract.get("seed_projection_contract"),
        _SEED_CONTRACT_FIELDS,
        label="seed_projection_contract",
    )
    if (
        prompt_contract["container_predicate"] != "exact_list"
        or prompt_contract["entry_predicate"] != "exact_object"
        or seed_contract["container_predicate"] != "exact_list"
        or seed_contract["element_predicate"] != "nonnegative_int"
    ):
        raise ValueError("projection predicates do not match assembler semantics")
    _require_rule_attribute(
        prompt_rules["prompt_id"],
        attribute="predicate",
        expected="nonempty_exact_str",
        label="prompt_projection_contract.entry_field_rules.prompt_id",
    )
    _require_rule_attribute(
        prompt_rules["prompt_text_digest"],
        attribute="predicate",
        expected="sha256_lower_hex_str",
        label="prompt_projection_contract.entry_field_rules.prompt_text_digest",
    )
    for label, predicate in (
        ("prompt container", prompt_contract.get("container_predicate")),
        ("prompt entry", prompt_contract.get("entry_predicate")),
        ("seed container", seed_contract.get("container_predicate")),
        ("seed element", seed_contract.get("element_predicate")),
    ):
        if (
            type(predicate) is not str
            or predicate not in predicates
            or predicate not in _SUPPORTED_PREDICATES
        ):
            raise ValueError(f"{label} predicate is not governed")
    _require_digest_rule(
        prompt_contract.get("digest_rule"),
        expected=_EXPECTED_DIGEST_RULES["prompt_projection"],
        label="prompt projection digest rule",
    )
    _require_digest_rule(
        seed_contract.get("digest_rule"),
        expected=_EXPECTED_DIGEST_RULES["seed_projection"],
        label="seed projection digest rule",
    )

    population_order = contract.get("population_order")
    if (
        type(population_order) not in {list, tuple}
        or len(population_order) != 3
        or any(type(kind) is not str or not kind for kind in population_order)
        or len(set(population_order)) != 3
    ):
        raise ValueError("population_order must contain three unique kinds")
    scalar_binding = contract.get("population_scalar_binding")
    if type(scalar_binding) is not dict or tuple(scalar_binding) != tuple(population_order):
        raise ValueError("population scalar binding does not match population order")
    for kind, binding in scalar_binding.items():
        _require_exact_keys(
            binding,
            _SCALAR_BINDING_FIELDS,
            label=f"population_scalar_binding.{kind}",
        )
        if (
            type(binding["scalar_field"]) is not str
            or type(binding["binary32_hex_field"]) is not str
            or binding["scalar_field"] not in top_rules
            or binding["binary32_hex_field"] not in top_rules
        ):
            raise ValueError("population scalar binding references an unknown field")
    scalar_fields = [
        scalar_binding[kind]["scalar_field"] for kind in population_order
    ]
    binary32_fields = [
        scalar_binding[kind]["binary32_hex_field"] for kind in population_order
    ]
    expected_scalar_fields = {
        field_name
        for field_name, rule in top_rules.items()
        if rule["predicate"] == "strict_positive_finite_json_float"
    }
    expected_binary32_fields = {
        field_name
        for field_name, rule in top_rules.items()
        if rule["predicate"] == "binary32_lower_hex_str"
    }
    if (
        len(set(scalar_fields)) != len(population_order)
        or len(set(binary32_fields)) != len(population_order)
        or set(scalar_fields) != expected_scalar_fields
        or set(binary32_fields) != expected_binary32_fields
        or scalar_binding != _EXPECTED_POPULATION_SCALAR_BINDING
    ):
        raise ValueError("population scalar binding is not an exact bijection")

    runtime_contract = _require_exact_keys(
        contract.get("runtime_component_identity_payload_contract"),
        _RUNTIME_CONTRACT_FIELDS,
        label="runtime_component_identity_payload_contract",
    )
    runtime_rules = _require_supported_rule_map(
        runtime_contract.get("field_rules"),
        _RUNTIME_PAYLOAD_FIELDS,
        predicates,
        label="runtime_component_identity_payload_contract.field_rules",
    )
    probe_contract = _require_exact_keys(
        runtime_contract.get("public_probe_identity_contract"),
        _PROBE_CONTRACT_FIELDS,
        label="public_probe_identity_contract",
    )
    probe_rules = _require_supported_rule_map(
        probe_contract.get("field_rules"),
        _PROBE_FIELDS,
        predicates,
        label="public_probe_identity_contract.field_rules",
    )
    probe_domain_rules = _require_supported_rule_map(
        probe_contract.get("domain_field_rules"),
        _PROBE_DOMAIN_FIELDS,
        predicates,
        label="public_probe_identity_contract.domain_field_rules",
    )
    for field_name in _RUNTIME_EXACT_VALUE_FIELDS:
        _require_exact_rule_value(
            runtime_rules[field_name],
            label=(
                "runtime_component_identity_payload_contract."
                f"field_rules.{field_name}"
            ),
        )
    for field_name in _RUNTIME_EXACT_TOKEN_FIELDS:
        _require_rule_attribute(
            runtime_rules[field_name],
            attribute="predicate",
            expected="exact_token_str",
            label=(
                "runtime_component_identity_payload_contract."
                f"field_rules.{field_name}"
            ),
        )
    for field_name in _RUNTIME_EXACT_FLOAT_FIELDS:
        _require_rule_attribute(
            runtime_rules[field_name],
            attribute="predicate",
            expected="strict_positive_finite_json_float",
            label=(
                "runtime_component_identity_payload_contract."
                f"field_rules.{field_name}"
            ),
        )
    for field_name in _RUNTIME_EXACT_POSITIVE_INT_FIELDS:
        _require_rule_attribute(
            runtime_rules[field_name],
            attribute="predicate",
            expected="strict_positive_int",
            label=(
                "runtime_component_identity_payload_contract."
                f"field_rules.{field_name}"
            ),
        )
    for field_name in _RUNTIME_EXACT_NONNEGATIVE_INT_FIELDS:
        _require_rule_attribute(
            runtime_rules[field_name],
            attribute="predicate",
            expected="nonnegative_int",
            label=(
                "runtime_component_identity_payload_contract."
                f"field_rules.{field_name}"
            ),
        )
    for field_name in ("prg_version", "key_material"):
        _require_exact_rule_value(
            probe_rules[field_name],
            label=f"public_probe_identity_contract.field_rules.{field_name}",
        )
        _require_rule_attribute(
            probe_rules[field_name],
            attribute="predicate",
            expected="exact_token_str",
            label=f"public_probe_identity_contract.field_rules.{field_name}",
        )
    for field_name in _PROBE_DOMAIN_FIELDS:
        _require_exact_rule_value(
            probe_domain_rules[field_name],
            label=f"public_probe_identity_contract.domain_field_rules.{field_name}",
        )
        _require_rule_attribute(
            probe_domain_rules[field_name],
            attribute="predicate",
            expected="exact_token_str",
            label=f"public_probe_identity_contract.domain_field_rules.{field_name}",
        )
    _require_rule_attribute(
        runtime_rules["public_probe_identity"],
        attribute="predicate",
        expected="exact_object",
        label="runtime_component_identity_payload_contract.public_probe_identity",
    )
    _require_rule_attribute(
        probe_rules["domain_fields"],
        attribute="predicate",
        expected="exact_object",
        label="public_probe_identity_contract.field_rules.domain_fields",
    )
    if runtime_contract["schema_token"] != runtime_rules["schema_version"]["exact_value"]:
        raise ValueError("runtime schema token does not bind schema_version")
    _require_digest_rule(
        runtime_contract.get("component_config_digest_rule"),
        expected=_EXPECTED_DIGEST_RULES["runtime_component_config"],
        label="runtime component config digest rule",
    )
    _require_digest_rule(
        runtime_contract.get("digest_rule"),
        expected=_EXPECTED_DIGEST_RULES["runtime_component_identity"],
        label="runtime component identity digest rule",
    )
    _require_digest_rule(
        population_rules["reference_observation_member_records_digest"].get(
            "digest_rule"
        ),
        expected=_EXPECTED_DIGEST_RULES["member_records"],
        label="member records digest rule",
    )
    _require_rule_attribute(
        population_rules["reference_observation_member_records_digest"],
        attribute="predicate",
        expected="sha256_lower_hex_str",
        label="population_field_rules.reference_observation_member_records_digest",
    )
    _require_digest_rule(
        contract.get("semantic_digest_rule"),
        expected=_EXPECTED_DIGEST_RULES["registry_semantic"],
        label="registry semantic digest rule",
    )
    _require_digest_rule(
        contract.get("file_sha256_rule"),
        expected=_EXPECTED_DIGEST_RULES["registry_file"],
        label="registry file digest rule",
    )
    binary32_rule = contract.get("binary32_hex_rule")
    if binary32_rule != _EXPECTED_BINARY32_HEX_RULE:
        raise ValueError("binary32 hex rule is invalid")

    invariants = contract.get("cross_field_invariants")
    if type(invariants) not in {list, tuple} or any(type(value) is not str for value in invariants):
        raise ValueError("content-routing reference invariants are invalid")
    for projection_contract in (prompt_contract, seed_contract):
        if (
            projection_contract.get("order_rule") not in invariants
            or projection_contract.get("length_rule") not in invariants
        ):
            raise ValueError("projection invariants are not governed")
    if prompt_contract.get("order_source") != seed_contract.get("order_source"):
        raise ValueError("prompt and seed projections do not share one order source")
    _require_rule_attribute(
        top_rules["content_routing_reference_populations"],
        attribute="exact_length",
        expected=len(population_order),
        label="registry_top_level_field_rules.content_routing_reference_populations",
    )

    runtime_invariants = runtime_contract.get("cross_field_invariants")
    forbidden_fields = runtime_contract.get("forbidden_fields")
    if (
        type(runtime_invariants) not in {list, tuple}
        or any(type(value) is not str for value in runtime_invariants)
        or type(forbidden_fields) not in {list, tuple}
        or any(type(value) is not str for value in forbidden_fields)
        or len(set(forbidden_fields)) != len(forbidden_fields)
        or not set(forbidden_fields).isdisjoint(runtime_rules)
    ):
        raise ValueError("runtime identity invariants are invalid")

    return {
        "top_rules": top_rules,
        "population_rules": population_rules,
        "member_rules": member_rules,
        "prompt_contract": prompt_contract,
        "prompt_rules": prompt_rules,
        "seed_contract": seed_contract,
        "population_order": tuple(population_order),
        "scalar_binding": scalar_binding,
        "runtime_contract": runtime_contract,
        "runtime_rules": runtime_rules,
        "probe_rules": probe_rules,
        "probe_domain_rules": probe_domain_rules,
    }


def _snapshot_exact_list(value: Any, *, label: str) -> tuple[Any, ...]:
    if type(value) is not list:
        raise TypeError(f"{label} must be an exact list")
    snapshot = tuple(value)
    if not snapshot:
        raise ValueError(f"{label} must not be empty")
    return snapshot


def _validate_observation_metadata(
    observations: tuple[torch.Tensor, ...],
    *,
    label: str,
) -> None:
    for member_index, member in enumerate(observations):
        member_label = f"{label}[{member_index}]"
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


def _validated_runtime_payload(
    value: Any,
    *,
    contract_parts: dict[str, Any],
    dependency_profile_digest: str,
    formal_execution_lock_digest: str,
) -> dict[str, Any]:
    runtime_payload = _validate_exact_object(
        value,
        contract_parts["runtime_rules"],
        label="runtime_component_identity_payload",
    )
    public_probe = _validate_exact_object(
        runtime_payload["public_probe_identity"],
        contract_parts["probe_rules"],
        label="runtime_component_identity_payload.public_probe_identity",
    )
    _validate_exact_object(
        public_probe["domain_fields"],
        contract_parts["probe_domain_rules"],
        label="runtime_component_identity_payload.public_probe_identity.domain_fields",
    )
    top_rules = contract_parts["top_rules"]
    if (
        runtime_payload["model_id"] != top_rules["model_id"]["exact_value"]
        or runtime_payload["model_revision"]
        != top_rules["model_revision"]["exact_value"]
        or runtime_payload["dependency_profile_digest"] != dependency_profile_digest
        or runtime_payload["formal_execution_lock_digest"]
        != formal_execution_lock_digest
    ):
        raise ValueError("runtime component identity does not match registry identity")
    forbidden_fields = set(
        contract_parts["runtime_contract"]["forbidden_fields"]
    )
    if forbidden_fields.intersection(runtime_payload):
        raise ValueError("runtime component identity contains forbidden fields")
    return dict(runtime_payload)


def _population_record(
    *,
    kind: str,
    observations: tuple[torch.Tensor, ...],
    generation_identity_digests: tuple[str, ...],
    contract_parts: dict[str, Any],
    numerator: int,
    denominator: int,
    label: str,
) -> tuple[dict[str, Any], str, str, float]:
    positive_population = _positive_population(observations, label=label)
    member_rules = contract_parts["member_rules"]
    member_records: list[dict[str, Any]] = []
    for sequence_index, (member, generation_digest) in enumerate(
        zip(observations, generation_identity_digests, strict=True)
    ):
        record = {
            "reference_observation_kind": kind,
            "reference_observation_member_sequence_index": sequence_index,
            "generation_input_identity_digest": generation_digest,
            "tensor_content_sha256": tensor_content_sha256(member),
        }
        _validate_exact_object(
            record,
            member_rules,
            label=f"{label}.member_record[{sequence_index}]",
        )
        member_records.append(record)

    positive_count = int(positive_population.numel())
    selected_rank = (numerator * positive_count + denominator - 1) // denominator
    selected_index = selected_rank - 1
    selected_scalar = _nearest_rank_p95(positive_population)
    scalar_binding = contract_parts["scalar_binding"][kind]
    scalar_field = scalar_binding["scalar_field"]
    binary32_field = scalar_binding["binary32_hex_field"]
    population = {
        "reference_observation_kind": kind,
        "reference_observation_member_count": len(observations),
        "reference_observation_positive_value_count": positive_count,
        "reference_observation_member_records_digest": build_stable_digest(
            member_records
        ),
        "tensor_content_sha256": tensor_content_sha256(positive_population),
        "reference_observation_selected_rank": selected_rank,
        "reference_observation_selected_index": selected_index,
    }
    _validate_exact_object(
        population,
        contract_parts["population_rules"],
        label=f"{label}.population",
    )
    return (
        population,
        scalar_field,
        binary32_field,
        selected_scalar,
    )


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

    contract = _load_machine_contract()
    contract_parts = _preflight_payload_assembler_contract(contract)
    top_rules = contract_parts["top_rules"]

    _validate_rule(
        method_parameter_partition_id,
        top_rules["method_parameter_partition_id"],
        label="method_parameter_partition_id",
    )
    _validate_rule(
        formal_execution_lock_digest,
        top_rules["formal_execution_lock_digest"],
        label="formal_execution_lock_digest",
    )
    _validate_rule(
        dependency_profile_digest,
        top_rules["dependency_profile_digest"],
        label="dependency_profile_digest",
    )

    prompt_contract = contract_parts["prompt_contract"]
    _validate_rule(
        prompt_projection,
        {"predicate": prompt_contract["container_predicate"]},
        label="prompt_projection",
    )
    prompt_snapshot = _snapshot_exact_list(
        prompt_projection,
        label="prompt_projection",
    )
    prompt_list: list[dict[str, Any]] = []
    for entry_index, entry in enumerate(prompt_snapshot):
        _validate_rule(
            entry,
            {"predicate": prompt_contract["entry_predicate"]},
            label=f"prompt_projection[{entry_index}]",
        )
        validated_entry = _validate_exact_object(
            entry,
            contract_parts["prompt_rules"],
            label=f"prompt_projection[{entry_index}]",
        )
        prompt_list.append(dict(validated_entry))

    seed_contract = contract_parts["seed_contract"]
    _validate_rule(
        seed_projection_random,
        {"predicate": seed_contract["container_predicate"]},
        label="seed_projection_random",
    )
    seed_snapshot = _snapshot_exact_list(
        seed_projection_random,
        label="seed_projection_random",
    )
    seed_rule = {"predicate": seed_contract["element_predicate"]}
    for seed_index, seed in enumerate(seed_snapshot):
        _validate_rule(
            seed,
            seed_rule,
            label=f"seed_projection_random[{seed_index}]",
        )

    generation_snapshot = _snapshot_exact_list(
        generation_input_identity_digests,
        label="generation_input_identity_digests",
    )
    generation_rule = contract_parts["member_rules"][
        "generation_input_identity_digest"
    ]
    for identity_index, identity_digest in enumerate(generation_snapshot):
        _validate_rule(
            identity_digest,
            generation_rule,
            label=f"generation_input_identity_digests[{identity_index}]",
        )

    runtime_payload = _validated_runtime_payload(
        runtime_component_identity_payload,
        contract_parts=contract_parts,
        dependency_profile_digest=dependency_profile_digest,
        formal_execution_lock_digest=formal_execution_lock_digest,
    )

    sample_count = len(prompt_snapshot)
    if len(seed_snapshot) != sample_count or len(generation_snapshot) != sample_count:
        raise ValueError("ordered identity projections must share one sample count")

    observation_snapshots = (
        _snapshot_observations(
            gradient_observations,
            label="gradient_observations",
        ),
        _snapshot_observations(
            response_observations,
            label="response_observations",
        ),
        _snapshot_observations(
            sensitivity_observations,
            label="sensitivity_observations",
        ),
    )
    if any(len(snapshot) != sample_count for snapshot in observation_snapshots):
        raise ValueError("observation populations must share the method sample count")
    for label, observations in zip(
        ("gradient_observations", "response_observations", "sensitivity_observations"),
        observation_snapshots,
        strict=True,
    ):
        _validate_observation_metadata(observations, label=label)

    numerator = top_rules["content_routing_reference_quantile_numerator"].get(
        "exact_value"
    )
    denominator = top_rules["content_routing_reference_quantile_denominator"].get(
        "exact_value"
    )
    _validate_rule(
        numerator,
        top_rules["content_routing_reference_quantile_numerator"],
        label="content_routing_reference_quantile_numerator",
    )
    _validate_rule(
        denominator,
        top_rules["content_routing_reference_quantile_denominator"],
        label="content_routing_reference_quantile_denominator",
    )

    populations: list[dict[str, Any]] = []
    scalar_values: dict[str, Any] = {}
    for kind, observations, label in zip(
        contract_parts["population_order"],
        observation_snapshots,
        ("gradient_observations", "response_observations", "sensitivity_observations"),
        strict=True,
    ):
        population, scalar_field, binary32_field, scalar = _population_record(
            kind=kind,
            observations=observations,
            generation_identity_digests=generation_snapshot,
            contract_parts=contract_parts,
            numerator=numerator,
            denominator=denominator,
            label=label,
        )
        populations.append(population)
        scalar_values[scalar_field] = scalar
        scalar_values[binary32_field] = struct.pack(">f", scalar).hex()

    registry = {
        "registry_schema": top_rules["registry_schema"]["exact_value"],
        "method_parameter_partition_id": method_parameter_partition_id,
        "method_parameter_prompt_list_digest": build_stable_digest(prompt_list),
        "method_parameter_seed_list_digest_random": build_stable_digest(
            list(seed_snapshot)
        ),
        "method_parameter_sample_count": sample_count,
        "formal_execution_lock_digest": formal_execution_lock_digest,
        "dependency_profile_digest": dependency_profile_digest,
        "model_id": top_rules["model_id"]["exact_value"],
        "model_revision": top_rules["model_revision"]["exact_value"],
        "runtime_component_identity_digest": build_stable_digest(runtime_payload),
        "content_routing_reference_quantile_algorithm": top_rules[
            "content_routing_reference_quantile_algorithm"
        ]["exact_value"],
        "content_routing_reference_quantile_numerator": numerator,
        "content_routing_reference_quantile_denominator": denominator,
        "content_routing_reference_quantile_rank_rule": top_rules[
            "content_routing_reference_quantile_rank_rule"
        ]["exact_value"],
        "content_routing_reference_quantile_index_rule": top_rules[
            "content_routing_reference_quantile_index_rule"
        ]["exact_value"],
        "content_routing_reference_populations": populations,
        **scalar_values,
    }
    if set(registry) != _TOP_LEVEL_FIELDS - {
        "content_routing_reference_registry_digest"
    }:
        raise ValueError("assembled registry fields do not match the machine contract")
    registry_digest = build_stable_digest(registry)
    registry["content_routing_reference_registry_digest"] = registry_digest
    raw_payload = stable_json_dumps(registry).encode("utf-8") + b"\n"
    _validate_content_routing_reference_registry(
        registry,
        raw_payload=raw_payload,
        expected_registry_digest=registry_digest,
        contract=contract,
    )
    return raw_payload
