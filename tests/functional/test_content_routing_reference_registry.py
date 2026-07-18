from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any
import stat
import struct

import pytest

from experiments.protocol import content_routing_reference_registry as registry
from experiments.protocol.content_routing_reference_quantile import (
    ContentRoutingReferenceScalars,
)


pytestmark = pytest.mark.quick

ROOT = Path(__file__).resolve().parents[2]
METHOD_DOCUMENT = (
    ROOT
    / "docs/builds/"
    "method_mechanism_design_content_adaptive_dual_carrier_latent_watermark.md"
)
CONTRACT_NAME = "CONTENT_ROUTING_REFERENCE_REGISTRY_MACHINE_CONTRACT"
FIXED_REGISTRY = ROOT / "configs/content_routing_reference_registry.json"
EXPECTED_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_REAL_OS_OPEN = os.open


def _machine_contract() -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    text = METHOD_DOCUMENT.read_text(encoding="utf-8-sig")
    for code_block in re.findall(r"```python\n(.*?)\n```", text, flags=re.DOTALL):
        tree = ast.parse(code_block)
        for statement in tree.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id == CONTRACT_NAME:
                value = ast.literal_eval(statement.value)
                assert type(value) is dict
                values.append(value)
    assert len(values) == 1
    return values[0]


def _contract_document(*assignments: str) -> str:
    return "\n\n".join(
        f"```python\n{assignment}\n```" for assignment in assignments
    )


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


def _default_rule_value(rule: dict[str, Any], *, label: str) -> Any:
    if "exact_value" in rule:
        return rule["exact_value"]
    predicate = rule["predicate"]
    if predicate == "nonempty_exact_str":
        return f"governed-{label}"
    if predicate == "sha256_lower_hex_str":
        return _sha(label)
    if predicate == "strict_positive_int":
        return 1
    if predicate == "nonnegative_int":
        return 0
    raise AssertionError(f"fixture has no value for {label}: {predicate}")


def _valid_registry() -> dict[str, Any]:
    contract = _machine_contract()
    top_rules = contract["registry_top_level_field_rules"]
    sample_count = 3
    scalars = {
        "reference_gradient": 1.25,
        "reference_response": 2.5,
        "reference_sensitivity": 4.0,
    }
    payload: dict[str, Any] = {}
    for field_name, rule in top_rules.items():
        if field_name == "content_routing_reference_populations":
            continue
        if field_name == "method_parameter_sample_count":
            payload[field_name] = sample_count
        elif field_name in scalars:
            payload[field_name] = scalars[field_name]
        elif field_name.endswith("_binary32_hex"):
            scalar_field = field_name.removesuffix("_binary32_hex")
            payload[field_name] = struct.pack(">f", scalars[scalar_field]).hex()
        elif field_name == "content_routing_reference_registry_digest":
            continue
        else:
            payload[field_name] = _default_rule_value(rule, label=field_name)

    positive_counts = (20, 21, 39)
    populations: list[dict[str, Any]] = []
    for population_index, population_kind in enumerate(contract["population_order"]):
        positive_count = positive_counts[population_index]
        rank = (19 * positive_count + 19) // 20
        population: dict[str, Any] = {}
        for field_name, rule in contract["population_field_rules"].items():
            if field_name == "reference_observation_kind":
                value: Any = population_kind
            elif field_name == "reference_observation_member_count":
                value = sample_count
            elif field_name == "reference_observation_positive_value_count":
                value = positive_count
            elif field_name == "reference_observation_selected_rank":
                value = rank
            elif field_name == "reference_observation_selected_index":
                value = rank - 1
            else:
                value = _default_rule_value(
                    rule,
                    label=f"{population_kind}-{field_name}",
                )
            population[field_name] = value
        populations.append(population)
    payload["content_routing_reference_populations"] = populations
    payload["content_routing_reference_registry_digest"] = _stable_digest(payload)
    return payload


def _registry_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _refresh_semantic_digest(payload: dict[str, Any]) -> None:
    semantic_payload = dict(payload)
    semantic_payload.pop("content_routing_reference_registry_digest", None)
    payload["content_routing_reference_registry_digest"] = _stable_digest(
        semantic_payload
    )


def _redirect_open(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    *,
    events: list[str] | None = None,
) -> None:
    def guarded_open(target: Any, flags: int, *args: Any) -> int:
        if events is not None:
            events.append("open")
        assert Path(target) == FIXED_REGISTRY
        assert flags == EXPECTED_FLAGS
        assert flags & os.O_ACCMODE == os.O_RDONLY
        return _REAL_OS_OPEN(path, flags, *args)

    monkeypatch.setattr(registry.os, "open", guarded_open)


def _load_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: dict[str, Any],
    *,
    raw_payload: bytes | None = None,
    expected_registry_digest: str | None = None,
) -> ContentRoutingReferenceScalars:
    raw = _registry_bytes(payload) if raw_payload is None else raw_payload
    source = tmp_path / "registry.json"
    source.write_bytes(raw)
    _redirect_open(monkeypatch, source)
    return registry.load_content_routing_reference_registry(
        expected_registry_digest=(
            payload["content_routing_reference_registry_digest"]
            if expected_registry_digest is None
            else expected_registry_digest
        ),
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
    )


def test_loader_has_exact_public_interface_and_fixed_path() -> None:
    assert registry.__all__ == ["load_content_routing_reference_registry"]
    signature = inspect.signature(registry.load_content_routing_reference_registry)
    assert tuple(signature.parameters) == (
        "expected_registry_digest",
        "expected_file_sha256",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert signature.return_annotation == "ContentRoutingReferenceScalars"
    assert registry._REFERENCE_REGISTRY_PATH == FIXED_REGISTRY
    assert registry._EXPECTED_OPEN_FLAGS == EXPECTED_FLAGS


@pytest.mark.parametrize(
    "contract_case",
    [
        "missing",
        "duplicate",
        "nonliteral",
        "unknown_top_predicate",
        "unknown_population_predicate",
    ],
)
def test_machine_contract_failures_precede_registry_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contract_case: str,
) -> None:
    contract = copy.deepcopy(_machine_contract())
    assignment = f"{CONTRACT_NAME} = {contract!r}"
    if contract_case == "missing":
        document = _contract_document("UNRELATED_CONTRACT = {}")
    elif contract_case == "duplicate":
        document = _contract_document(assignment, assignment)
    elif contract_case == "nonliteral":
        document = _contract_document(f"{CONTRACT_NAME} = dict()")
    else:
        if contract_case == "unknown_top_predicate":
            rules = contract["registry_top_level_field_rules"]
            rules["registry_schema"]["predicate"] = "unknown_predicate"
        else:
            rules = contract["population_field_rules"]
            rules["reference_observation_kind"]["predicate"] = (
                "unknown_predicate"
            )
        document = _contract_document(f"{CONTRACT_NAME} = {contract!r}")

    method_document = tmp_path / "method_contract.md"
    method_document.write_text(document, encoding="utf-8")
    monkeypatch.setattr(registry, "_METHOD_DOCUMENT_PATH", method_document)
    open_calls = 0

    def forbidden_open(*args: Any, **kwargs: Any) -> int:
        nonlocal open_calls
        open_calls += 1
        raise AssertionError("registry open must follow contract validation")

    monkeypatch.setattr(registry.os, "open", forbidden_open)
    with pytest.raises(ValueError):
        registry.load_content_routing_reference_registry(
            expected_registry_digest="a" * 64,
            expected_file_sha256="b" * 64,
        )
    assert open_calls == 0


def test_valid_registry_uses_one_read_only_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _valid_registry()
    raw = _registry_bytes(payload)
    source = tmp_path / "registry.json"
    source.write_bytes(raw)
    events: list[str] = []
    _redirect_open(monkeypatch, source, events=events)

    real_fstat = os.fstat
    real_read = os.read
    real_close = os.close

    def tracked_fstat(descriptor: int) -> os.stat_result:
        events.append("fstat")
        return real_fstat(descriptor)

    def tracked_read(descriptor: int, count: int) -> bytes:
        events.append("read")
        return real_read(descriptor, count)

    def tracked_close(descriptor: int) -> None:
        events.append("close")
        real_close(descriptor)

    monkeypatch.setattr(registry.os, "fstat", tracked_fstat)
    monkeypatch.setattr(registry.os, "read", tracked_read)
    monkeypatch.setattr(registry.os, "close", tracked_close)
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: (_ for _ in ()).throw(AssertionError("path precheck")),
    )
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: (_ for _ in ()).throw(AssertionError("path precheck")),
    )

    result = registry.load_content_routing_reference_registry(
        expected_registry_digest=payload[
            "content_routing_reference_registry_digest"
        ],
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
    )

    assert result == ContentRoutingReferenceScalars(1.25, 2.5, 4.0)
    assert events == ["open", "fstat", "read", "close"]


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("expected_registry_digest", "A" * 64),
        ("expected_file_sha256", "not-a-digest"),
    ],
)
def test_expected_digest_format_fails_before_open(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    invalid_value: str,
) -> None:
    monkeypatch.setattr(
        registry.os,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("opened")),
    )
    arguments = {
        "expected_registry_digest": "a" * 64,
        "expected_file_sha256": "b" * 64,
    }
    arguments[field_name] = invalid_value
    with pytest.raises(ValueError):
        registry.load_content_routing_reference_registry(**arguments)


def test_fifo_is_nonblocking_and_rejected_before_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fifo = tmp_path / "registry.fifo"
    os.mkfifo(fifo)
    real_open = os.open
    real_close = os.close
    calls = {"open": 0, "read": 0, "close": 0}

    def guarded_fifo_open(target: Any, flags: int, *args: Any) -> int:
        calls["open"] += 1
        assert Path(target) == FIXED_REGISTRY
        assert flags == EXPECTED_FLAGS
        return real_open(fifo, flags, *args)

    def forbidden_read(descriptor: int, count: int) -> bytes:
        calls["read"] += 1
        raise AssertionError("nonregular file was read")

    def tracked_close(descriptor: int) -> None:
        calls["close"] += 1
        real_close(descriptor)

    monkeypatch.setattr(registry.os, "open", guarded_fifo_open)
    monkeypatch.setattr(registry.os, "read", forbidden_read)
    monkeypatch.setattr(registry.os, "close", tracked_close)
    with pytest.raises(ValueError, match="regular file"):
        registry.load_content_routing_reference_registry(
            expected_registry_digest="a" * 64,
            expected_file_sha256="b" * 64,
        )
    assert calls == {"open": 1, "read": 0, "close": 1}


@pytest.mark.parametrize(
    "file_type",
    [stat.S_IFIFO, stat.S_IFSOCK, stat.S_IFCHR, stat.S_IFDIR],
)
def test_nonregular_modes_fail_before_read_and_close_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    file_type: int,
) -> None:
    calls = {"read": 0, "close": 0}

    def controlled_open(target: Any, flags: int) -> int:
        assert Path(target) == FIXED_REGISTRY
        assert flags == EXPECTED_FLAGS
        return 73

    monkeypatch.setattr(registry.os, "open", controlled_open)
    monkeypatch.setattr(
        registry.os,
        "fstat",
        lambda descriptor: SimpleNamespace(st_mode=file_type | 0o600, st_size=0),
    )
    monkeypatch.setattr(
        registry.os,
        "read",
        lambda *args: calls.__setitem__("read", calls["read"] + 1),
    )
    monkeypatch.setattr(
        registry.os,
        "close",
        lambda descriptor: calls.__setitem__("close", calls["close"] + 1),
    )
    with pytest.raises(ValueError, match="regular file"):
        registry.load_content_routing_reference_registry(
            expected_registry_digest="a" * 64,
            expected_file_sha256="b" * 64,
        )
    assert calls == {"read": 0, "close": 1}


def test_terminal_symlink_is_rejected_without_descriptor_follow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}\n")
    link = tmp_path / "registry-link.json"
    link.symlink_to(target)
    real_open = os.open
    close_calls = 0

    def guarded_open(path: Any, flags: int) -> int:
        assert Path(path) == FIXED_REGISTRY
        assert flags == EXPECTED_FLAGS
        return real_open(link, flags)

    def forbidden_close(descriptor: int) -> None:
        nonlocal close_calls
        close_calls += 1

    monkeypatch.setattr(registry.os, "open", guarded_open)
    monkeypatch.setattr(registry.os, "close", forbidden_close)
    with pytest.raises(ValueError, match="cannot be opened"):
        registry.load_content_routing_reference_registry(
            expected_registry_digest="a" * 64,
            expected_file_sha256="b" * 64,
        )
    assert close_calls == 0


@pytest.mark.parametrize("failure_operation", ["open", "fstat", "read", "close"])
def test_descriptor_errors_fail_closed_and_close_when_opened(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_operation: str,
) -> None:
    payload = _valid_registry()
    raw = _registry_bytes(payload)
    source = tmp_path / "registry.json"
    source.write_bytes(raw)
    real_open = os.open
    real_fstat = os.fstat
    real_read = os.read
    real_close = os.close
    close_calls = 0

    def controlled_open(target: Any, flags: int) -> int:
        assert flags == EXPECTED_FLAGS
        if failure_operation == "open":
            raise OSError("open failure")
        return real_open(source, flags)

    def controlled_fstat(descriptor: int) -> os.stat_result:
        if failure_operation == "fstat":
            raise OSError("fstat failure")
        return real_fstat(descriptor)

    def controlled_read(descriptor: int, count: int) -> bytes:
        if failure_operation == "read":
            raise OSError("read failure")
        return real_read(descriptor, count)

    def controlled_close(descriptor: int) -> None:
        nonlocal close_calls
        close_calls += 1
        real_close(descriptor)
        if failure_operation == "close":
            raise OSError("close failure")

    monkeypatch.setattr(registry.os, "open", controlled_open)
    monkeypatch.setattr(registry.os, "fstat", controlled_fstat)
    monkeypatch.setattr(registry.os, "read", controlled_read)
    monkeypatch.setattr(registry.os, "close", controlled_close)
    with pytest.raises(ValueError):
        registry.load_content_routing_reference_registry(
            expected_registry_digest=payload[
                "content_routing_reference_registry_digest"
            ],
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )
    assert close_calls == (0 if failure_operation == "open" else 1)


def test_file_sha_is_checked_before_json_parsing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _valid_registry()
    raw = _registry_bytes(payload)
    source = tmp_path / "registry.json"
    source.write_bytes(raw)
    _redirect_open(monkeypatch, source)
    monkeypatch.setattr(
        registry,
        "_strict_json_object",
        lambda value: (_ for _ in ()).throw(AssertionError("JSON parsed")),
    )
    with pytest.raises(ValueError, match="file digest"):
        registry.load_content_routing_reference_registry(
            expected_registry_digest=payload[
                "content_routing_reference_registry_digest"
            ],
            expected_file_sha256="0" * 64,
        )


@pytest.mark.parametrize("raw_kind", ["invalid_utf8", "duplicate", "nan", "infinity"])
def test_strict_utf8_and_json_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_kind: str,
) -> None:
    payload = _valid_registry()
    raw = _registry_bytes(payload)
    if raw_kind == "invalid_utf8":
        raw = b"\xff\n"
    elif raw_kind == "duplicate":
        raw = raw.replace(
            b"{",
            b'{"registry_schema":"duplicate",',
            1,
        )
    elif raw_kind == "nan":
        raw = raw.replace(b'"reference_gradient":1.25', b'"reference_gradient":NaN')
    else:
        raw = raw.replace(
            b'"reference_gradient":1.25',
            b'"reference_gradient":Infinity',
        )
    source = tmp_path / "registry.json"
    source.write_bytes(raw)
    _redirect_open(monkeypatch, source)
    with pytest.raises(ValueError, match="JSON"):
        registry.load_content_routing_reference_registry(
            expected_registry_digest=payload[
                "content_routing_reference_registry_digest"
            ],
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


@pytest.mark.parametrize("canonical_variant", ["missing_lf", "pretty", "extra_lf"])
def test_noncanonical_registry_bytes_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    canonical_variant: str,
) -> None:
    payload = _valid_registry()
    if canonical_variant == "missing_lf":
        raw = _registry_bytes(payload)[:-1]
    elif canonical_variant == "pretty":
        raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    else:
        raw = _registry_bytes(payload) + b"\n"
    with pytest.raises(ValueError, match="canonical"):
        _load_payload(
            monkeypatch,
            tmp_path,
            payload,
            raw_payload=raw,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_top_field",
        "missing_top_field",
        "extra_population_field",
        "missing_population_field",
        "population_list_length",
        "model_token",
        "integer_scalar",
        "bool_positive_int",
        "bool_nonnegative_int",
        "population_order",
        "member_count",
        "rank",
        "index",
        "binary32",
        "non_binary32_scalar",
    ],
)
def test_exact_schema_and_internal_consistency_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = _valid_registry()
    populations = payload["content_routing_reference_populations"]
    if mutation == "extra_top_field":
        payload["extra"] = "forbidden"
    elif mutation == "missing_top_field":
        del payload["dependency_profile_digest"]
    elif mutation == "extra_population_field":
        populations[0]["extra"] = "forbidden"
    elif mutation == "missing_population_field":
        del populations[0]["tensor_content_sha256"]
    elif mutation == "population_list_length":
        populations.pop()
    elif mutation == "model_token":
        payload["model_id"] = "wrong/model"
    elif mutation == "integer_scalar":
        payload["reference_gradient"] = 1
    elif mutation == "bool_positive_int":
        payload["method_parameter_sample_count"] = True
    elif mutation == "bool_nonnegative_int":
        populations[0]["reference_observation_selected_index"] = False
    elif mutation == "population_order":
        populations[0], populations[1] = populations[1], populations[0]
    elif mutation == "member_count":
        populations[0]["reference_observation_member_count"] += 1
    elif mutation == "rank":
        populations[0]["reference_observation_selected_rank"] -= 1
    elif mutation == "index":
        populations[0]["reference_observation_selected_index"] -= 1
    elif mutation == "binary32":
        payload["reference_gradient_binary32_hex"] = "00000000"
    else:
        payload["reference_gradient"] = 1.1
        payload["reference_gradient_binary32_hex"] = struct.pack(">f", 1.1).hex()
    _refresh_semantic_digest(payload)
    with pytest.raises(ValueError):
        _load_payload(monkeypatch, tmp_path, payload)


@pytest.mark.parametrize("top_level_value", [[], 7])
def test_nonobject_top_level_json_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    top_level_value: Any,
) -> None:
    raw = json.dumps(top_level_value, separators=(",", ":")).encode("utf-8") + b"\n"
    source = tmp_path / "registry.json"
    source.write_bytes(raw)
    _redirect_open(monkeypatch, source)
    with pytest.raises(ValueError, match="exact object"):
        registry.load_content_routing_reference_registry(
            expected_registry_digest="a" * 64,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_semantic_digest_is_checked_embedded_then_expected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _valid_registry()
    payload["content_routing_reference_registry_digest"] = "0" * 64
    with pytest.raises(ValueError, match="embedded"):
        _load_payload(
            monkeypatch,
            tmp_path,
            payload,
            expected_registry_digest="0" * 64,
        )

    payload = _valid_registry()
    with pytest.raises(ValueError, match="expected"):
        _load_payload(
            monkeypatch,
            tmp_path,
            payload,
            expected_registry_digest="f" * 64,
        )


def test_loader_does_not_require_absent_projection_or_runtime_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _valid_registry()
    payload["method_parameter_prompt_list_digest"] = "1" * 64
    payload["method_parameter_seed_list_digest_random"] = "2" * 64
    payload["runtime_component_identity_digest"] = "3" * 64
    for index, population in enumerate(
        payload["content_routing_reference_populations"],
        start=4,
    ):
        population["reference_observation_member_records_digest"] = str(index) * 64
        population["tensor_content_sha256"] = str(index + 3) * 64
    _refresh_semantic_digest(payload)
    assert _load_payload(monkeypatch, tmp_path, payload) == (
        ContentRoutingReferenceScalars(1.25, 2.5, 4.0)
    )
