"""Build one governed content-routing reference generation identity in memory."""

from __future__ import annotations

from typing import Any

import torch

from experiments.protocol.content_routing_reference_registry import (
    _SUPPORTED_PREDICATES,
    _load_machine_contract,
    _validate_exact_object,
    _validate_rule,
)
from experiments.protocol.formal_randomization import (
    FORMAL_BASE_LATENT_GENERATION_PROTOCOL,
    build_canonical_sd35_base_latent,
    formal_randomization_protocol_record,
)
from main.core.digest import build_stable_digest, tensor_content_sha256


__all__ = ["build_content_routing_reference_generation_input_record"]


_PAYLOAD_RULE_PREDICATES = {
    "prompt_id": "nonempty_exact_str",
    "prompt_text": "nonempty_exact_str",
    "prompt_text_digest": "sha256_lower_hex_str",
    "generation_seed_random": "nonnegative_int",
    "base_latent_identity_digest_random": "sha256_lower_hex_str",
    "formal_randomization_protocol_digest": "sha256_lower_hex_str",
    "model_id": "exact_token_str",
    "model_revision": "exact_token_str",
    "negative_prompt": "exact_token_str",
    "width": "strict_positive_int",
    "height": "strict_positive_int",
    "inference_steps": "strict_positive_int",
    "guidance_scale": "strict_positive_finite_json_float",
    "formal_method_config_digest": "sha256_lower_hex_str",
    "dependency_profile_digest": "sha256_lower_hex_str",
    "formal_execution_lock_digest": "sha256_lower_hex_str",
    "runtime_component_identity_digest": "sha256_lower_hex_str",
}
_PAYLOAD_RULE_EXACT_VALUES = {
    "model_id": "stabilityai/stable-diffusion-3.5-medium",
    "model_revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
    "negative_prompt": "low quality, blurry",
    "width": 512,
    "height": 512,
    "inference_steps": 20,
    "guidance_scale": 4.5,
}
_RECORD_RULE_PREDICATES = {
    "reference_observation_member_sequence_index": "nonnegative_int",
    "generation_input_identity_payload": "exact_object",
    "generation_input_identity_digest": "sha256_lower_hex_str",
}
_BASE_IDENTITY_FIELDS = (
    "generation_seed_random",
    "base_latent_generation_protocol",
    "base_latent_keyed_prg_version",
    "base_latent_keyed_prg_protocol_digest",
    "formal_randomization_protocol_digest",
    "base_latent_dtype",
    "base_latent_shape",
    "base_latent_content_digest_random",
    "base_latent_identity_digest_random",
)
_PROMPT_DIGEST_RULE = "build_stable_digest({'prompt_text': exact_prompt_text})"
_GENERATION_DIGEST_RULE = (
    "build_stable_digest(exact_generation_input_identity_payload)"
)
_BASE_IDENTITY_DIGEST_RULE = (
    "build_stable_digest(exact_helper_identity_without_"
    "base_latent_identity_digest_random)"
)
_BASE_ARGUMENT_SOURCES = {
    "shape": "base_latent_identity_reconstruction_contract.shape",
    "generation_seed_random": (
        "generation_input_identity_payload.generation_seed_random"
    ),
    "model_id": "generation_input_identity_payload.model_id",
    "model_revision": "generation_input_identity_payload.model_revision",
    "qualification_device": (
        "base_latent_identity_reconstruction_contract."
        "identity_reconstruction_device"
    ),
    "producer_device": "actual_qualified_pipeline_execution_device",
    "dtype": "base_latent_identity_reconstruction_contract.dtype",
}
_PAYLOAD_CROSS_FIELD_INVARIANTS = (
    "model_id_and_revision_equal_registry_and_manifest",
    "dependency_profile_digest_equals_registry_and_manifest",
    "formal_execution_lock_digest_equals_registry_and_manifest",
    "runtime_component_identity_digest_equals_registry_and_manifest",
    "prompt_and_seed_projection_order_equals_method_parameter_member_order",
)
_BASE_CROSS_FIELD_INVARIANTS = (
    "returned_generation_seed_random_equals_generation_input_payload",
    "returned_formal_randomization_protocol_digest_equals_generation_input_payload",
    "returned_base_latent_generation_protocol_equals_formal_randomization_protocol_record",
    "returned_keyed_prg_version_and_digest_equal_formal_randomization_protocol_record",
    "returned_base_latent_dtype_equals_torch_float16",
    "returned_base_latent_shape_equals_1_16_64_64",
    "returned_base_latent_identity_digest_random_equals_generation_input_payload",
    "returned_base_latent_content_digest_random_is_rebuilt_not_caller_supplied",
    "producer_cuda_tensor_roundtrip_content_digest_equals_returned_base_latent_content_digest_random",
    "cuda_rng_and_alternate_latent_construction_are_forbidden",
)
_ORDERED_RECORD_RULES = {
    "length_rule": (
        "generation_input_record_count_equals_method_parameter_sample_count"
    ),
    "order_rule": (
        "sequence_indices_are_exactly_zero_through_sample_count_minus_one"
    ),
    "prompt_projection_rule": (
        "ordered_prompt_id_and_prompt_text_digest_projection_matches_registry_digest"
    ),
    "seed_projection_rule": (
        "ordered_generation_seed_random_projection_matches_registry_digest"
    ),
}


def _require_rule_map(
    value: Any,
    *,
    expected_predicates: dict[str, str],
    expected_exact_values: dict[str, Any],
    predicates: dict[str, Any],
    label: str,
) -> dict[str, dict[str, Any]]:
    if type(value) is not dict or set(value) != set(expected_predicates):
        raise ValueError(f"{label} has missing or extra fields")
    for field_name, rule in value.items():
        expected_keys = {"predicate"}
        if field_name in expected_exact_values:
            expected_keys.add("exact_value")
        if type(rule) is not dict or set(rule) != expected_keys:
            raise ValueError(f"{label}.{field_name} is invalid")
        predicate = rule["predicate"]
        if (
            type(predicate) is not str
            or predicate != expected_predicates[field_name]
            or predicate not in predicates
            or predicate not in _SUPPORTED_PREDICATES
        ):
            raise ValueError(f"{label}.{field_name} has an invalid predicate")
        if (
            field_name in expected_exact_values
            and rule["exact_value"] != expected_exact_values[field_name]
        ):
            raise ValueError(f"{label}.{field_name} has an invalid exact value")
    return value


def _preflight_generation_input_contract(
    contract: Any,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    if type(contract) is not dict:
        raise ValueError("generation input machine contract is invalid")
    predicates = contract.get("type_predicates")
    materialization = contract.get("materialization_contract")
    if type(predicates) is not dict or type(materialization) is not dict:
        raise ValueError("generation input machine contract is incomplete")
    if materialization.get("contract_schema_token") != (
        "content_routing_reference_materialization_contract_v1"
    ):
        raise ValueError("generation input materialization token is invalid")

    payload_contract = materialization.get(
        "generation_input_identity_payload_contract"
    )
    if type(payload_contract) is not dict or set(payload_contract) != {
        "field_rules",
        "prompt_text_digest_rule",
        "digest_rule",
        "order_source",
        "cross_field_invariants",
    }:
        raise ValueError("generation input payload contract is invalid")
    payload_rules = _require_rule_map(
        payload_contract.get("field_rules"),
        expected_predicates=_PAYLOAD_RULE_PREDICATES,
        expected_exact_values=_PAYLOAD_RULE_EXACT_VALUES,
        predicates=predicates,
        label="generation_input_identity_payload_contract.field_rules",
    )
    if payload_contract.get("prompt_text_digest_rule") != _PROMPT_DIGEST_RULE:
        raise ValueError("generation input prompt digest rule is invalid")
    if payload_contract.get("digest_rule") != _GENERATION_DIGEST_RULE:
        raise ValueError("generation input identity digest rule is invalid")
    if payload_contract.get("order_source") != (
        "method_parameter_partition_generation_member_order"
    ):
        raise ValueError("generation input order source is invalid")
    if payload_contract.get("cross_field_invariants") != (
        _PAYLOAD_CROSS_FIELD_INVARIANTS
    ):
        raise ValueError("generation input payload invariants are invalid")

    record_contract = materialization.get("ordered_generation_input_record_contract")
    if type(record_contract) is not dict or set(record_contract) != {
        "container_predicate",
        "field_rules",
        "nested_contracts",
        "length_rule",
        "order_rule",
        "prompt_projection_rule",
        "seed_projection_rule",
    }:
        raise ValueError("ordered generation input record contract is invalid")
    if record_contract.get("container_predicate") != "exact_list":
        raise ValueError("ordered generation input record container is invalid")
    if record_contract.get("nested_contracts") != {
        "generation_input_identity_payload": (
            "generation_input_identity_payload_contract"
        )
    }:
        raise ValueError("ordered generation input nested contract is invalid")
    for rule_name, expected_value in _ORDERED_RECORD_RULES.items():
        if record_contract.get(rule_name) != expected_value:
            raise ValueError(f"ordered generation input {rule_name} is invalid")
    record_rules = _require_rule_map(
        record_contract.get("field_rules"),
        expected_predicates=_RECORD_RULE_PREDICATES,
        expected_exact_values={},
        predicates=predicates,
        label="ordered_generation_input_record_contract.field_rules",
    )

    base_contract = materialization.get("base_latent_identity_reconstruction_contract")
    if type(base_contract) is not dict or set(base_contract) != {
        "builder_module",
        "builder_symbol",
        "identity_reconstruction_device",
        "producer_execution_device_source",
        "shape",
        "shape_derivation_rule",
        "dtype",
        "argument_sources",
        "returned_identity_fields",
        "digest_rule",
        "cross_field_invariants",
    }:
        raise ValueError("base latent reconstruction contract is invalid")
    expected_values = {
        "builder_module": "experiments.protocol.formal_randomization",
        "builder_symbol": "build_canonical_sd35_base_latent",
        "identity_reconstruction_device": "cpu",
        "producer_execution_device_source": (
            "actual_qualified_pipeline_execution_device"
        ),
        "shape": (1, 16, 64, 64),
        "shape_derivation_rule": (
            "exact_512_by_512_rgb_generation_with_vae_scale_8_and_"
            "16_latent_channels"
        ),
        "dtype": "torch.float16",
        "argument_sources": _BASE_ARGUMENT_SOURCES,
        "returned_identity_fields": _BASE_IDENTITY_FIELDS,
        "digest_rule": _BASE_IDENTITY_DIGEST_RULE,
    }
    for field_name, expected_value in expected_values.items():
        if base_contract.get(field_name) != expected_value:
            raise ValueError(f"base latent {field_name} is invalid")
    if base_contract.get("cross_field_invariants") != _BASE_CROSS_FIELD_INVARIANTS:
        raise ValueError("base latent reconstruction invariants are invalid")

    for field_name, rule in payload_rules.items():
        if "exact_value" in rule:
            _validate_rule(
                rule["exact_value"],
                rule,
                label=f"generation_input_identity_payload_contract.{field_name}",
            )
    return payload_rules, record_rules, base_contract


def _validate_base_latent_identity(
    latent: Any,
    identity: Any,
    *,
    expected_shape: tuple[int, ...],
    generation_seed_random: int,
) -> dict[str, Any]:
    if not isinstance(latent, torch.Tensor):
        raise ValueError("canonical base latent must be a Tensor")
    if latent.device.type != "cpu":
        raise ValueError("canonical base latent must be rebuilt on CPU")
    if latent.dtype != torch.float16 or tuple(latent.shape) != expected_shape:
        raise ValueError("canonical base latent dtype or shape is invalid")
    if not bool(torch.isfinite(latent).all().item()):
        raise ValueError("canonical base latent must be finite")
    if type(identity) is not dict or set(identity) != set(_BASE_IDENTITY_FIELDS):
        raise ValueError("canonical base latent identity fields are invalid")

    protocol = formal_randomization_protocol_record()
    expected_pairs = {
        "generation_seed_random": generation_seed_random,
        "base_latent_generation_protocol": FORMAL_BASE_LATENT_GENERATION_PROTOCOL,
        "base_latent_keyed_prg_version": protocol["base_latent_keyed_prg_version"],
        "base_latent_keyed_prg_protocol_digest": protocol[
            "base_latent_keyed_prg_protocol_digest"
        ],
        "formal_randomization_protocol_digest": protocol[
            "formal_randomization_protocol_digest"
        ],
        "base_latent_dtype": "torch.float16",
        "base_latent_shape": list(expected_shape),
        "base_latent_content_digest_random": tensor_content_sha256(latent),
    }
    for field_name, expected_value in expected_pairs.items():
        if identity.get(field_name) != expected_value:
            raise ValueError(f"canonical base latent identity drift: {field_name}")
    identity_without_digest = dict(identity)
    embedded_digest = identity_without_digest.pop(
        "base_latent_identity_digest_random",
        None,
    )
    if type(embedded_digest) is not str or embedded_digest != build_stable_digest(
        identity_without_digest
    ):
        raise ValueError("canonical base latent identity digest is invalid")
    return dict(identity)


def build_content_routing_reference_generation_input_record(
    *,
    reference_observation_member_sequence_index: Any,
    prompt_id: Any,
    prompt_text: Any,
    generation_seed_random: Any,
    formal_method_config_digest: Any,
    dependency_profile_digest: Any,
    formal_execution_lock_digest: Any,
    runtime_component_identity_digest: Any,
) -> dict[str, Any]:
    """Return one exact ordered generation input record without persistence."""

    contract = _load_machine_contract()
    payload_rules, record_rules, base_contract = (
        _preflight_generation_input_contract(contract)
    )
    input_fields = {
        "reference_observation_member_sequence_index": (
            reference_observation_member_sequence_index,
            record_rules["reference_observation_member_sequence_index"],
        ),
        "prompt_id": (prompt_id, payload_rules["prompt_id"]),
        "prompt_text": (prompt_text, payload_rules["prompt_text"]),
        "generation_seed_random": (
            generation_seed_random,
            payload_rules["generation_seed_random"],
        ),
        "formal_method_config_digest": (
            formal_method_config_digest,
            payload_rules["formal_method_config_digest"],
        ),
        "dependency_profile_digest": (
            dependency_profile_digest,
            payload_rules["dependency_profile_digest"],
        ),
        "formal_execution_lock_digest": (
            formal_execution_lock_digest,
            payload_rules["formal_execution_lock_digest"],
        ),
        "runtime_component_identity_digest": (
            runtime_component_identity_digest,
            payload_rules["runtime_component_identity_digest"],
        ),
    }
    for field_name, (value, rule) in input_fields.items():
        _validate_rule(value, rule, label=field_name)

    expected_shape = tuple(base_contract["shape"])
    model_id = payload_rules["model_id"]["exact_value"]
    model_revision = payload_rules["model_revision"]["exact_value"]
    latent, identity = build_canonical_sd35_base_latent(
        shape=expected_shape,
        generation_seed_random=generation_seed_random,
        model_id=model_id,
        model_revision=model_revision,
        device=base_contract["identity_reconstruction_device"],
        dtype=torch.float16,
    )
    base_identity = _validate_base_latent_identity(
        latent,
        identity,
        expected_shape=expected_shape,
        generation_seed_random=generation_seed_random,
    )

    prompt_text_digest = build_stable_digest({"prompt_text": prompt_text})
    payload = {
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "prompt_text_digest": prompt_text_digest,
        "generation_seed_random": generation_seed_random,
        "base_latent_identity_digest_random": base_identity[
            "base_latent_identity_digest_random"
        ],
        "formal_randomization_protocol_digest": base_identity[
            "formal_randomization_protocol_digest"
        ],
        "model_id": model_id,
        "model_revision": model_revision,
        "negative_prompt": payload_rules["negative_prompt"]["exact_value"],
        "width": payload_rules["width"]["exact_value"],
        "height": payload_rules["height"]["exact_value"],
        "inference_steps": payload_rules["inference_steps"]["exact_value"],
        "guidance_scale": payload_rules["guidance_scale"]["exact_value"],
        "formal_method_config_digest": formal_method_config_digest,
        "dependency_profile_digest": dependency_profile_digest,
        "formal_execution_lock_digest": formal_execution_lock_digest,
        "runtime_component_identity_digest": runtime_component_identity_digest,
    }
    _validate_exact_object(
        payload,
        payload_rules,
        label="generation_input_identity_payload",
    )
    generation_digest = build_stable_digest(payload)
    _validate_rule(
        generation_digest,
        record_rules["generation_input_identity_digest"],
        label="generation_input_identity_digest",
    )
    record = {
        "reference_observation_member_sequence_index": (
            reference_observation_member_sequence_index
        ),
        "generation_input_identity_payload": payload,
        "generation_input_identity_digest": generation_digest,
    }
    _validate_exact_object(record, record_rules, label="generation_input_record")
    return record
