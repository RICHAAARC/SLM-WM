"""Validate governed in-memory encoding of content reference observations."""

from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path
import struct
from typing import Any, Callable

import pytest
import torch

import experiments.protocol.content_routing_reference_raw_member as raw_module
import experiments.protocol.content_routing_reference_registry as registry_module
from main.core.digest import tensor_content_sha256


pytestmark = pytest.mark.quick

ROOT = Path(__file__).resolve().parents[2]
FIXED_REGISTRY = ROOT / "configs/content_routing_reference_registry.json"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _member(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, -1)


def _encode(
    observation: Any,
    *,
    kind: Any = "gradient_magnitude_rgb_pre_interpolation",
    sequence_index: Any = 3,
    generation_digest: Any = None,
) -> tuple[bytes, dict[str, Any]]:
    return raw_module.encode_content_routing_reference_raw_member(
        reference_observation_kind=kind,
        reference_observation_member_sequence_index=sequence_index,
        generation_input_identity_digest=(
            _sha("generation") if generation_digest is None else generation_digest
        ),
        observation=observation,
    )


def _independent_big_endian_bytes(observation: torch.Tensor) -> bytes:
    values = observation.detach().contiguous().reshape(-1).tolist()
    return b"".join(struct.pack(">f", value) for value in values)


def _decoded_tensor(raw_bytes: bytes, shape: list[int]) -> torch.Tensor:
    values = [value[0] for value in struct.iter_unpack(">f", raw_bytes)]
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def test_encoder_has_one_keyword_only_public_interface() -> None:
    assert raw_module.__all__ == ["encode_content_routing_reference_raw_member"]
    signature = inspect.signature(
        raw_module.encode_content_routing_reference_raw_member
    )
    assert tuple(signature.parameters) == (
        "reference_observation_kind",
        "reference_observation_member_sequence_index",
        "generation_input_identity_digest",
        "observation",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    assert signature.return_annotation == "tuple[bytes, dict[str, Any]]"


@pytest.mark.parametrize(
    ("kind", "sequence_index"),
    [
        ("gradient_magnitude_rgb_pre_interpolation", 0),
        ("latent_response", 17),
        ("local_sensitivity_rgb_pre_interpolation", 123_456_789),
    ],
)
def test_bytes_record_and_paths_match_the_machine_contract(
    kind: str,
    sequence_index: int,
) -> None:
    generation_digest = _sha(f"generation-{kind}")
    observation = torch.tensor(
        [[[[0.0, -0.0, 1.25], [2.5, 3.75, 4.0]]]],
        dtype=torch.float32,
    )
    raw_bytes, record = _encode(
        observation,
        kind=kind,
        sequence_index=sequence_index,
        generation_digest=generation_digest,
    )
    contract = raw_module._load_machine_contract()
    raw_contract = contract["materialization_contract"][
        "raw_member_file_record_contract"
    ]
    expected_bytes = _independent_big_endian_bytes(observation)
    expected_path = raw_contract["path_templates"][kind].format(
        sequence_index=sequence_index
    )

    assert raw_bytes == expected_bytes
    assert set(record) == set(raw_contract["field_rules"])
    assert record == {
        "reference_observation_kind": kind,
        "reference_observation_member_sequence_index": sequence_index,
        "generation_input_identity_digest": generation_digest,
        "path": expected_path,
        "sha256": hashlib.sha256(expected_bytes).hexdigest(),
        "tensor_shape": list(observation.shape),
        "tensor_dtype": "torch.float32",
        "tensor_content_sha256": tensor_content_sha256(observation),
    }
    rebuilt = _decoded_tensor(raw_bytes, record["tensor_shape"])
    assert torch.equal(rebuilt, observation)
    assert tensor_content_sha256(rebuilt) == record["tensor_content_sha256"]
    assert len(raw_bytes) == observation.numel() * 4
    assert record["path"].endswith(f"/{sequence_index:08d}.f32be")


def test_signed_zero_bits_and_all_zero_member_are_preserved() -> None:
    observation = _member([0.0, -0.0, 0.0])
    raw_bytes, record = _encode(observation)

    assert raw_bytes.hex() == "000000008000000000000000"
    assert record["tensor_content_sha256"] == tensor_content_sha256(observation)
    decoded_signbits = torch.signbit(
        _decoded_tensor(raw_bytes, [1, 1, 1, 3])
    )[0, 0, 0].tolist()
    assert decoded_signbits == [False, True, False]


def test_noncontiguous_requires_grad_input_is_read_without_mutation() -> None:
    observation = (
        torch.tensor(
            [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]],
            dtype=torch.float32,
        )
        .transpose(-1, -2)
        .requires_grad_(True)
    )
    before = observation.detach().clone()
    before_shape = observation.shape
    before_stride = observation.stride()
    before_grad = observation.grad

    first = _encode(observation, kind="latent_response")
    second = _encode(observation, kind="latent_response")

    assert first == second
    assert first[0] == _independent_big_endian_bytes(observation)
    assert torch.equal(observation.detach(), before)
    assert observation.shape == before_shape
    assert observation.stride() == before_stride
    assert observation.requires_grad is True
    assert observation.grad is before_grad


@pytest.mark.parametrize(
    "observation",
    [
        "not-a-tensor",
        torch.ones((1, 1, 1, 1), dtype=torch.float64),
        torch.ones((1, 1, 1, 1), dtype=torch.int64),
        torch.ones((1, 1, 1, 1), dtype=torch.bool),
        torch.ones((1, 1, 1, 1), dtype=torch.complex64),
        torch.ones((2, 1, 1, 1), dtype=torch.float32),
        torch.ones((1, 2, 1, 1), dtype=torch.float32),
        torch.ones((1, 1, 2), dtype=torch.float32),
        torch.empty((1, 1, 0, 2), dtype=torch.float32),
    ],
)
def test_invalid_type_dtype_or_shape_fails_closed(observation: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        _encode(observation)


def test_meta_tensor_fails_before_any_content_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = torch.empty((1, 1, 2, 3), dtype=torch.float32, device="meta")

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("meta tensor content was accessed")

    monkeypatch.setattr(raw_module, "_encode_flat_binary32_big_endian", forbidden)
    monkeypatch.setattr(raw_module, "tensor_content_sha256", forbidden)
    with pytest.raises(ValueError, match="materialized on CPU"):
        _encode(observation)


@pytest.mark.parametrize(
    "observation",
    [
        _member([-1.0, 0.0]),
        _member([float("nan")]),
        _member([float("inf")]),
        _member([float("-inf")]),
    ],
)
def test_negative_or_nonfinite_content_fails_before_encoding_and_digest(
    monkeypatch: pytest.MonkeyPatch,
    observation: torch.Tensor,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("invalid content reached encoding or digest")

    monkeypatch.setattr(raw_module, "_encode_flat_binary32_big_endian", forbidden)
    monkeypatch.setattr(raw_module, "tensor_content_sha256", forbidden)
    with pytest.raises(ValueError):
        _encode(observation)


@pytest.mark.parametrize(
    ("overrides", "error_type"),
    [
        ({"kind": "unknown"}, ValueError),
        ({"kind": 1}, ValueError),
        ({"sequence_index": -1}, ValueError),
        ({"sequence_index": True}, ValueError),
        ({"generation_digest": "0" * 63}, ValueError),
        ({"generation_digest": 1}, ValueError),
    ],
)
def test_static_identity_inputs_fail_before_observation_content(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    error_type: type[Exception],
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("observation metadata was reached")

    monkeypatch.setattr(raw_module, "_validate_observation_metadata", forbidden)
    with pytest.raises(error_type):
        _encode(_member([1.0]), **overrides)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda contract: contract["materialization_contract"].update(
            {"contract_schema_token": "wrong"}
        ),
        lambda contract: contract["materialization_contract"].update(
            {
                "type_predicate_extensions": contract["materialization_contract"][
                    "type_predicate_extensions"
                ][:-1]
            }
        ),
        lambda contract: contract["type_predicates"].pop(
            "safe_relative_posix_path_str"
        ),
        lambda contract: contract["materialization_contract"][
            "raw_member_file_record_contract"
        ].update({"encoding_token": "wrong"}),
        lambda contract: contract["materialization_contract"][
            "raw_member_file_record_contract"
        ].update({"decoding_rule": "wrong"}),
        lambda contract: contract["materialization_contract"][
            "raw_member_file_record_contract"
        ].update({"file_length_rule": "wrong"}),
        lambda contract: contract["materialization_contract"][
            "raw_member_file_record_contract"
        ].update({"file_sha256_rule": "wrong"}),
        lambda contract: contract["materialization_contract"][
            "raw_member_file_record_contract"
        ].update({"tensor_digest_rule": "wrong"}),
        lambda contract: contract["materialization_contract"][
            "raw_member_file_record_contract"
        ]["field_rules"]["tensor_shape"].pop("exact_prefix"),
        lambda contract: contract["materialization_contract"][
            "raw_member_file_record_contract"
        ]["path_templates"].pop("latent_response"),
        lambda contract: contract["materialization_contract"][
            "raw_member_file_record_contract"
        ]["path_templates"].update(
            {"latent_response": "../latent_response/{sequence_index:08d}.f32be"}
        ),
        lambda contract: contract["materialization_contract"][
            "raw_member_file_record_contract"
        ].update({"cross_field_invariants": ()}),
    ],
)
def test_contract_drift_fails_before_observation_metadata(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], Any],
) -> None:
    contract = copy.deepcopy(raw_module._load_machine_contract())
    mutation(contract)
    monkeypatch.setattr(raw_module, "_load_machine_contract", lambda: contract)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("observation metadata was reached before contract preflight")

    monkeypatch.setattr(raw_module, "_validate_observation_metadata", forbidden)
    with pytest.raises(ValueError):
        _encode(_member([1.0]))


@pytest.mark.parametrize(
    ("value", "rule"),
    [
        (True, {"predicate": "exact_bool", "exact_value": True}),
        (None, {"predicate": "exact_null"}),
        (
            "blocked",
            {
                "predicate": "exact_status_token",
                "allowed_tokens": ("pass", "blocked"),
            },
        ),
        ("a" * 40, {"predicate": "git_commit_lower_hex_str"}),
        (
            [1, 1, 2, 3],
            {"predicate": "positive_int_list", "exact_prefix": (1, 1)},
        ),
        ("raw/kind/00000000.f32be", {"predicate": "safe_relative_posix_path_str"}),
    ],
)
def test_shared_materialization_predicates_accept_governed_values(
    value: Any,
    rule: dict[str, Any],
) -> None:
    registry_module._validate_rule(value, rule, label="value")


@pytest.mark.parametrize(
    ("value", "rule"),
    [
        (1, {"predicate": "exact_bool"}),
        (False, {"predicate": "positive_int_list"}),
        ([1, True], {"predicate": "positive_int_list"}),
        (
            "failed",
            {
                "predicate": "exact_status_token",
                "allowed_tokens": ("pass", "blocked"),
            },
        ),
        ("A" * 40, {"predicate": "git_commit_lower_hex_str"}),
        ("/absolute/file", {"predicate": "safe_relative_posix_path_str"}),
        ("raw//file", {"predicate": "safe_relative_posix_path_str"}),
        ("raw/./file", {"predicate": "safe_relative_posix_path_str"}),
        ("raw/../file", {"predicate": "safe_relative_posix_path_str"}),
        ("raw\\file", {"predicate": "safe_relative_posix_path_str"}),
        (
            [1, 2, 3, 4],
            {"predicate": "positive_int_list", "exact_prefix": (1, 1)},
        ),
    ],
)
def test_shared_materialization_predicates_reject_drift(
    value: Any,
    rule: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        registry_module._validate_rule(value, rule, label="value")


@pytest.mark.parametrize(
    ("value", "rule"),
    [
        ("anything", {"predicate": "exact_status_token"}),
        (
            "pass",
            {"predicate": "exact_status_token", "allowed_tokens": ()},
        ),
        (
            "pass",
            {
                "predicate": "exact_status_token",
                "allowed_tokens": ["pass", "blocked"],
            },
        ),
        (
            "pass",
            {
                "predicate": "exact_status_token",
                "allowed_tokens": ("pass", "pass"),
            },
        ),
        (
            "pass",
            {
                "predicate": "exact_status_token",
                "allowed_tokens": ("pass", 1),
            },
        ),
        (
            "pass",
            {
                "predicate": "exact_token_str",
                "allowed_tokens": ("pass", "blocked"),
            },
        ),
        (
            "value",
            {"predicate": "exact_token_str", "exact_prefix": (1,)},
        ),
        (
            [1, 2],
            {"predicate": "positive_int_list", "exact_prefix": [1]},
        ),
        (
            [1, 2],
            {"predicate": "positive_int_list", "exact_prefix": ()},
        ),
        (
            [1, 2],
            {"predicate": "positive_int_list", "exact_prefix": (True,)},
        ),
    ],
)
def test_rule_attributes_are_bound_to_their_governed_predicates(
    value: Any,
    rule: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        registry_module._validate_rule(value, rule, label="value")


def test_encoder_performs_no_artifact_io_and_fixed_registry_remains_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = raw_module._load_machine_contract()
    monkeypatch.setattr(raw_module, "_load_machine_contract", lambda: contract)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("artifact I/O is forbidden in the raw member encoder")

    monkeypatch.setattr("os.open", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    raw_bytes, record = _encode(_member([0.0, 1.0]))

    assert raw_bytes
    assert record["path"].startswith("raw/")
    assert not FIXED_REGISTRY.exists()
