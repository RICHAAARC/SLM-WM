"""Decode and verify one governed content-routing reference observation."""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

import torch

from experiments.protocol.content_routing_reference_registry import (
    _SUPPORTED_PREDICATES,
    _load_machine_contract,
    _validate_exact_object,
    _validate_rule,
)
from main.core.digest import tensor_content_sha256


__all__ = ["decode_content_routing_reference_raw_member"]


_MATERIALIZATION_PREDICATE_EXTENSIONS = (
    "exact_bool",
    "exact_null",
    "exact_null_or_nonempty_exact_str",
    "exact_status_token",
    "git_commit_lower_hex_str",
    "positive_int_list",
    "safe_relative_posix_path_str",
)
_RAW_MEMBER_FIELDS = frozenset(
    {
        "reference_observation_kind",
        "reference_observation_member_sequence_index",
        "generation_input_identity_digest",
        "path",
        "sha256",
        "tensor_shape",
        "tensor_dtype",
        "tensor_content_sha256",
    }
)
_EXPECTED_FIELD_RULES = {
    "reference_observation_kind": {
        "predicate": "exact_token_str",
        "exact_value_source": "parent_population_kind",
    },
    "reference_observation_member_sequence_index": {
        "predicate": "nonnegative_int",
    },
    "generation_input_identity_digest": {
        "predicate": "sha256_lower_hex_str",
    },
    "path": {
        "predicate": "safe_relative_posix_path_str",
    },
    "sha256": {
        "predicate": "sha256_lower_hex_str",
    },
    "tensor_shape": {
        "predicate": "positive_int_list",
        "exact_length": 4,
        "exact_prefix": (1, 1),
    },
    "tensor_dtype": {
        "predicate": "exact_token_str",
        "exact_value": "torch.float32",
    },
    "tensor_content_sha256": {
        "predicate": "sha256_lower_hex_str",
    },
}
_EXPECTED_RAW_CONTRACT_VALUES = {
    "encoding_token": "flat_nchw_ieee754_binary32_big_endian_v1",
    "encoding_rule": (
        "concatenate_struct_pack_big_endian_float_for_flat_nchw_cpu_float32_values"
    ),
    "decoding_rule": "decode_big_endian_binary32_then_rebuild_cpu_float32_tensor",
    "file_length_rule": "exact_file_length_equals_product_tensor_shape_times_four",
    "file_sha256_rule": "sha256(exact_raw_member_file_bytes)",
    "tensor_digest_rule": "tensor_content_sha256(rebuilt_cpu_float32_tensor)",
    "path_rule": "safe_unique_candidate_root_relative_posix_path_without_symlink_follow",
}
_EXPECTED_CROSS_FIELD_INVARIANTS = (
    "kind_equals_parent_population_kind",
    "sequence_index_and_generation_digest_equal_ordered_generation_input_record",
    "tensor_is_materialized_finite_nonnegative_cpu_float32_b1c1nchw",
    "negative_values_fail_before_raw_file_write",
    "zero_values_remain_in_raw_file_but_are_excluded_from_positive_population",
    "sequence_index_is_zero_padded_to_at_least_eight_decimal_digits_without_truncation",
)
_UNPACK_CHUNK_ELEMENT_COUNT = 65_536


def _preflight_raw_member_decoder_contract(
    contract: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...], dict[str, str]]:
    predicates = contract.get("type_predicates")
    materialization = contract.get("materialization_contract")
    population_order = contract.get("population_order")
    if type(predicates) is not dict or type(materialization) is not dict:
        raise ValueError("raw member decoder machine contract is incomplete")
    if (
        type(population_order) is not tuple
        or not population_order
        or not all(type(kind) is str and kind for kind in population_order)
        or len(set(population_order)) != len(population_order)
    ):
        raise ValueError("raw member decoder population order is invalid")
    if materialization.get("contract_schema_token") != (
        "content_routing_reference_materialization_contract_v1"
    ):
        raise ValueError("raw member decoder materialization contract token is invalid")
    if materialization.get("type_predicate_extensions") != (
        _MATERIALIZATION_PREDICATE_EXTENSIONS
    ):
        raise ValueError("raw member decoder predicate extensions are invalid")
    if any(
        predicate not in predicates or predicate not in _SUPPORTED_PREDICATES
        for predicate in _MATERIALIZATION_PREDICATE_EXTENSIONS
    ):
        raise ValueError("raw member decoder predicate extension is unsupported")

    raw_contract = materialization.get("raw_member_file_record_contract")
    if type(raw_contract) is not dict:
        raise ValueError("raw member decoder file contract is invalid")
    expected_contract_fields = {
        "field_rules",
        "path_templates",
        "cross_field_invariants",
        *_EXPECTED_RAW_CONTRACT_VALUES,
    }
    if set(raw_contract) != expected_contract_fields:
        raise ValueError("raw member decoder contract has missing or extra fields")
    field_rules = raw_contract.get("field_rules")
    if type(field_rules) is not dict or set(field_rules) != _RAW_MEMBER_FIELDS:
        raise ValueError("raw member decoder field rules are invalid")
    if field_rules != _EXPECTED_FIELD_RULES:
        raise ValueError("raw member decoder field rule semantics are invalid")
    for field_name, rule in field_rules.items():
        predicate = rule.get("predicate") if type(rule) is dict else None
        if predicate not in predicates or predicate not in _SUPPORTED_PREDICATES:
            raise ValueError(f"raw member decoder predicate is unsupported: {field_name}")
    for rule_name, expected_value in _EXPECTED_RAW_CONTRACT_VALUES.items():
        if raw_contract.get(rule_name) != expected_value:
            raise ValueError(f"raw member decoder {rule_name} is invalid")
    if raw_contract.get("cross_field_invariants") != _EXPECTED_CROSS_FIELD_INVARIANTS:
        raise ValueError("raw member decoder cross-field invariants are invalid")

    path_templates = raw_contract.get("path_templates")
    if type(path_templates) is not dict or tuple(path_templates) != population_order:
        raise ValueError("raw member decoder path template order is invalid")
    expected_templates = {
        kind: f"raw/{kind}/{{sequence_index:08d}}.f32be"
        for kind in population_order
    }
    if path_templates != expected_templates:
        raise ValueError("raw member decoder path templates are invalid")
    return field_rules, population_order, path_templates


def _raw_member_bytes_sha256(raw_member_bytes: bytes) -> str:
    return hashlib.sha256(raw_member_bytes).hexdigest()


def _decode_flat_binary32_big_endian(
    raw_member_bytes: bytes,
    *,
    element_count: int,
) -> torch.Tensor:
    decoded = torch.empty(element_count, dtype=torch.float32, device="cpu")
    for start in range(0, element_count, _UNPACK_CHUNK_ELEMENT_COUNT):
        stop = min(start + _UNPACK_CHUNK_ELEMENT_COUNT, element_count)
        raw_chunk = raw_member_bytes[start * 4 : stop * 4]
        values = struct.unpack(f">{stop - start}f", raw_chunk)
        decoded[start:stop].copy_(torch.tensor(values, dtype=torch.float32))
    return decoded


def decode_content_routing_reference_raw_member(
    *,
    reference_observation_kind: Any,
    reference_observation_member_sequence_index: Any,
    generation_input_identity_digest: Any,
    raw_member_bytes: Any,
    raw_member_file_record: Any,
) -> torch.Tensor:
    """Return a verified CPU float32 observation reconstructed from exact bytes."""

    contract = _load_machine_contract()
    field_rules, population_order, path_templates = (
        _preflight_raw_member_decoder_contract(contract)
    )
    _validate_rule(
        reference_observation_kind,
        field_rules["reference_observation_kind"],
        label="reference_observation_kind",
    )
    if reference_observation_kind not in population_order:
        raise ValueError("reference_observation_kind is not governed")
    _validate_rule(
        reference_observation_member_sequence_index,
        field_rules["reference_observation_member_sequence_index"],
        label="reference_observation_member_sequence_index",
    )
    _validate_rule(
        generation_input_identity_digest,
        field_rules["generation_input_identity_digest"],
        label="generation_input_identity_digest",
    )
    record = _validate_exact_object(
        raw_member_file_record,
        field_rules,
        label="raw_member_file_record",
    )
    if record["reference_observation_kind"] != reference_observation_kind:
        raise ValueError("raw member record kind does not match the expected identity")
    if (
        record["reference_observation_member_sequence_index"]
        != reference_observation_member_sequence_index
    ):
        raise ValueError("raw member record sequence does not match the expected identity")
    if record["generation_input_identity_digest"] != generation_input_identity_digest:
        raise ValueError("raw member record generation identity does not match")
    expected_path = path_templates[reference_observation_kind].format(
        sequence_index=reference_observation_member_sequence_index
    )
    if record["path"] != expected_path:
        raise ValueError("raw member record path does not match the governed template")

    if type(raw_member_bytes) is not bytes:
        raise TypeError("raw_member_bytes must be exact bytes")
    tensor_shape = tuple(record["tensor_shape"])
    element_count = math.prod(tensor_shape)
    expected_byte_count = element_count * 4
    if len(raw_member_bytes) != expected_byte_count:
        raise ValueError("raw member byte length is invalid")
    if _raw_member_bytes_sha256(raw_member_bytes) != record["sha256"]:
        raise ValueError("raw member file SHA-256 does not match the record")

    flat_observation = _decode_flat_binary32_big_endian(
        raw_member_bytes,
        element_count=element_count,
    )
    observation = flat_observation.reshape(tensor_shape).contiguous()
    if not bool(torch.isfinite(observation).all().item()):
        raise ValueError("decoded observation must contain only finite values")
    if bool((observation < 0.0).any().item()):
        raise ValueError("decoded observation must contain only nonnegative values")
    if tensor_content_sha256(observation) != record["tensor_content_sha256"]:
        raise ValueError("decoded observation content digest does not match the record")
    return observation
