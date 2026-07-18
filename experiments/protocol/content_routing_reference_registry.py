"""Fail-closed loading for the governed content-routing references."""

from __future__ import annotations

import ast
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
_METHOD_DOCUMENT_PATH = (
    _ROOT
    / "docs/builds/"
    "method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md"
)
_REFERENCE_REGISTRY_PATH = _ROOT / "configs/content_routing_reference_registry.json"
_CONTRACT_NAME = "CONTENT_ROUTING_REFERENCE_REGISTRY_MACHINE_CONTRACT"
_EXPECTED_OPEN_FLAGS = (
    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_BINARY32_PATTERN = re.compile(r"[0-9a-f]{8}")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SUPPORTED_PREDICATES = frozenset(
    {
        "exact_bool",
        "exact_object",
        "exact_list",
        "exact_null",
        "exact_null_or_nonempty_exact_str",
        "exact_status_token",
        "exact_token_str",
        "git_commit_lower_hex_str",
        "nonempty_exact_str",
        "sha256_lower_hex_str",
        "binary32_lower_hex_str",
        "positive_int_list",
        "safe_relative_posix_path_str",
        "strict_positive_int",
        "nonnegative_int",
        "strict_positive_finite_json_float",
    }
)


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _load_machine_contract() -> dict[str, Any]:
    try:
        document = _METHOD_DOCUMENT_PATH.read_text(encoding="utf-8-sig")
        values: list[dict[str, Any]] = []
        for code_block in re.findall(
            r"```python\n(.*?)\n```",
            document,
            flags=re.DOTALL,
        ):
            syntax_tree = ast.parse(code_block)
            for statement in syntax_tree.body:
                if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                    continue
                target = statement.targets[0]
                if isinstance(target, ast.Name) and target.id == _CONTRACT_NAME:
                    value = ast.literal_eval(statement.value)
                    if type(value) is not dict:
                        raise ValueError("machine contract must be an exact object")
                    values.append(value)
    except (OSError, SyntaxError, ValueError) as exc:
        raise ValueError("content-routing reference machine contract is invalid") from exc
    if len(values) != 1:
        raise ValueError("content-routing reference machine contract must be unique")
    contract = values[0]
    _validate_consumed_contract(contract)
    return contract


def _validate_consumed_contract(contract: dict[str, Any]) -> None:
    predicates = contract.get("type_predicates")
    if type(predicates) is not dict:
        raise ValueError("machine contract predicates must be an exact object")
    for rules_name in (
        "registry_top_level_field_rules",
        "population_field_rules",
    ):
        rules = contract.get(rules_name)
        if type(rules) is not dict or not rules:
            raise ValueError(f"machine contract {rules_name} is invalid")
        for field_name, rule in rules.items():
            if type(rule) is not dict or type(rule.get("predicate")) is not str:
                raise ValueError(f"machine contract rule is invalid for {field_name}")
            predicate = rule["predicate"]
            if predicate not in predicates or predicate not in _SUPPORTED_PREDICATES:
                raise ValueError(f"machine contract predicate is unknown: {predicate}")


def _read_fixed_registry_bytes() -> bytes:
    try:
        descriptor = os.open(_REFERENCE_REGISTRY_PATH, _EXPECTED_OPEN_FLAGS)
    except OSError as exc:
        raise ValueError("content-routing reference registry cannot be opened") from exc

    try:
        try:
            metadata = os.fstat(descriptor)
        except OSError as exc:
            raise ValueError("content-routing reference registry cannot be inspected") from exc
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


def _float32_value(value: float, *, label: str) -> float:
    try:
        converted = struct.unpack(">f", struct.pack(">f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"{label} is not representable as binary32") from exc
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{label} must remain finite and positive as binary32")
    return converted


def _validate_predicate(
    value: Any,
    predicate: str,
    *,
    label: str,
) -> None:
    if predicate == "exact_bool":
        valid = type(value) is bool
    elif predicate == "exact_object":
        valid = type(value) is dict
    elif predicate == "exact_list":
        valid = type(value) is list
    elif predicate == "exact_null":
        valid = value is None
    elif predicate == "exact_null_or_nonempty_exact_str":
        valid = value is None or (
            type(value) is str
            and bool(value)
            and value == value.strip()
            and not any(character in value for character in "\0\r\n\t")
        )
    elif predicate == "exact_status_token":
        valid = type(value) is str
    elif predicate == "exact_token_str":
        valid = type(value) is str
    elif predicate == "git_commit_lower_hex_str":
        valid = (
            type(value) is str
            and _GIT_COMMIT_PATTERN.fullmatch(value) is not None
        )
    elif predicate == "nonempty_exact_str":
        valid = (
            type(value) is str
            and bool(value)
            and value == value.strip()
            and not any(character in value for character in "\0\r\n\t")
        )
    elif predicate == "sha256_lower_hex_str":
        valid = type(value) is str and _SHA256_PATTERN.fullmatch(value) is not None
    elif predicate == "binary32_lower_hex_str":
        valid = type(value) is str and _BINARY32_PATTERN.fullmatch(value) is not None
    elif predicate == "positive_int_list":
        valid = (
            type(value) is list
            and all(type(member) is int and member > 0 for member in value)
        )
    elif predicate == "safe_relative_posix_path_str":
        valid = False
        if (
            type(value) is str
            and bool(value)
            and not value.startswith("/")
            and "\\" not in value
            and not any(character in value for character in "\0\r\n\t")
        ):
            segments = value.split("/")
            valid = all(segment not in {"", ".", ".."} for segment in segments)
    elif predicate == "strict_positive_int":
        valid = type(value) is int and value > 0
    elif predicate == "nonnegative_int":
        valid = type(value) is int and value >= 0
    elif predicate == "strict_positive_finite_json_float":
        valid = type(value) is float and math.isfinite(value) and value > 0.0
        if valid:
            _float32_value(value, label=label)
    else:
        raise ValueError(f"unknown governed predicate for {label}: {predicate}")
    if not valid:
        raise ValueError(f"{label} violates governed predicate {predicate}")


def _validate_rule(value: Any, rule: dict[str, Any], *, label: str) -> None:
    if type(rule) is not dict or type(rule.get("predicate")) is not str:
        raise ValueError(f"machine contract rule is invalid for {label}")
    predicate = rule["predicate"]
    _validate_predicate(value, predicate, label=label)
    if "exact_value" in rule and value != rule["exact_value"]:
        raise ValueError(f"{label} does not match the governed exact value")
    if "exact_length" in rule and len(value) != rule["exact_length"]:
        raise ValueError(f"{label} does not match the governed exact length")
    has_allowed_tokens = "allowed_tokens" in rule
    if predicate == "exact_status_token":
        allowed_tokens = rule.get("allowed_tokens")
        if (
            type(allowed_tokens) is not tuple
            or not allowed_tokens
            or not all(type(token) is str for token in allowed_tokens)
            or len(set(allowed_tokens)) != len(allowed_tokens)
            or value not in allowed_tokens
        ):
            raise ValueError(f"{label} is not a governed allowed token")
    elif has_allowed_tokens:
        raise ValueError(f"{label} attaches allowed_tokens to an invalid predicate")
    if "exact_prefix" in rule:
        exact_prefix = rule["exact_prefix"]
        if (
            predicate != "positive_int_list"
            or type(exact_prefix) is not tuple
            or not exact_prefix
            or not all(type(member) is int and member > 0 for member in exact_prefix)
            or len(value) < len(exact_prefix)
            or tuple(value[: len(exact_prefix)]) != exact_prefix
        ):
            raise ValueError(f"{label} does not match the governed exact prefix")


def _validate_exact_object(
    value: Any,
    rules: dict[str, dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an exact object")
    if set(value) != set(rules):
        raise ValueError(f"{label} has missing or extra fields")
    for field_name, rule in rules.items():
        _validate_rule(value[field_name], rule, label=f"{label}.{field_name}")
    return value


def _validate_registry_structure(
    registry: dict[str, Any],
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    top_rules = contract.get("registry_top_level_field_rules")
    population_rules = contract.get("population_field_rules")
    population_order = contract.get("population_order")
    if (
        type(top_rules) is not dict
        or type(population_rules) is not dict
        or type(population_order) not in {list, tuple}
    ):
        raise ValueError("content-routing reference machine contract is incomplete")
    _validate_exact_object(registry, top_rules, label="registry")

    populations = registry["content_routing_reference_populations"]
    if len(populations) != len(population_order):
        raise ValueError("registry population count is invalid")
    resolved: list[dict[str, Any]] = []
    for population_index, expected_kind in enumerate(population_order):
        population = _validate_exact_object(
            populations[population_index],
            population_rules,
            label=f"registry.population[{population_index}]",
        )
        if population["reference_observation_kind"] != expected_kind:
            raise ValueError("registry population order is invalid")
        resolved.append(population)
    return resolved


def _validate_registry_consistency(
    registry: dict[str, Any],
    populations: list[dict[str, Any]],
    contract: dict[str, Any],
) -> ContentRoutingReferenceScalars:
    sample_count = registry["method_parameter_sample_count"]
    population_order = tuple(contract["population_order"])
    scalar_binding = contract["population_scalar_binding"]
    if tuple(scalar_binding) != population_order:
        raise ValueError("machine contract population-scalar binding is invalid")

    resolved_scalars: dict[str, float] = {}
    for expected_kind, population in zip(population_order, populations, strict=True):
        if population["reference_observation_member_count"] != sample_count:
            raise ValueError("registry population member count is invalid")
        positive_count = population["reference_observation_positive_value_count"]
        expected_rank = (19 * positive_count + 19) // 20
        if population["reference_observation_selected_rank"] != expected_rank:
            raise ValueError("registry population selected rank is invalid")
        if population["reference_observation_selected_index"] != expected_rank - 1:
            raise ValueError("registry population selected index is invalid")

        binding = scalar_binding[expected_kind]
        scalar_field = binding["scalar_field"]
        binary32_field = binding["binary32_hex_field"]
        scalar = registry[scalar_field]
        binary32 = _float32_value(scalar, label=scalar_field)
        if binary32 != scalar:
            raise ValueError(f"{scalar_field} must reload exactly as binary32")
        if struct.pack(">f", scalar).hex() != registry[binary32_field]:
            raise ValueError(f"{binary32_field} does not bind {scalar_field}")
        resolved_scalars[scalar_field] = scalar

    return ContentRoutingReferenceScalars(
        reference_gradient=resolved_scalars["reference_gradient"],
        reference_response=resolved_scalars["reference_response"],
        reference_sensitivity=resolved_scalars["reference_sensitivity"],
    )


def _validate_content_routing_reference_registry(
    registry: dict[str, Any],
    *,
    raw_payload: bytes,
    expected_registry_digest: str,
    contract: dict[str, Any],
) -> ContentRoutingReferenceScalars:
    populations = _validate_registry_structure(registry, contract)

    canonical_payload = stable_json_dumps(registry).encode("utf-8") + b"\n"
    if raw_payload != canonical_payload:
        raise ValueError("content-routing reference registry is not canonical")

    digest_field = "content_routing_reference_registry_digest"
    semantic_payload = dict(registry)
    embedded_digest = semantic_payload.pop(digest_field)
    computed_digest = build_stable_digest(semantic_payload)
    if computed_digest != embedded_digest:
        raise ValueError("embedded content-routing reference digest is invalid")
    if embedded_digest != expected_registry_digest:
        raise ValueError("expected content-routing reference digest does not match")

    return _validate_registry_consistency(registry, populations, contract)


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
    contract = _load_machine_contract()
    raw_payload = _read_fixed_registry_bytes()
    if hashlib.sha256(raw_payload).hexdigest() != expected_file_sha256:
        raise ValueError("content-routing reference registry file digest does not match")
    registry = _strict_json_object(raw_payload)
    return _validate_content_routing_reference_registry(
        registry,
        raw_payload=raw_payload,
        expected_registry_digest=expected_registry_digest,
        contract=contract,
    )
