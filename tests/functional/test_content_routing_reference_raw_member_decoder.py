"""Validate governed in-memory decoding of raw reference observations."""

from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path
import struct
from typing import Any, Callable

import pytest
import torch

import experiments.protocol.content_routing_reference_raw_member_decoder as decoder_module
from experiments.protocol.content_routing_reference_raw_member import (
    encode_content_routing_reference_raw_member,
)
from main.core.digest import tensor_content_sha256


pytestmark = pytest.mark.quick

ROOT = Path(__file__).resolve().parents[2]
FIXED_REGISTRY = ROOT / "configs/content_routing_reference_registry.json"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _big_endian_bytes(observation: torch.Tensor) -> bytes:
    return b"".join(
        struct.pack(">f", value)
        for value in observation.detach().contiguous().reshape(-1).tolist()
    )


def _fixture(
    observation: torch.Tensor | None = None,
    *,
    kind: str = "gradient_magnitude_rgb_pre_interpolation",
    sequence_index: int = 7,
    generation_digest: str | None = None,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    if observation is None:
        observation = torch.tensor(
            [[[[0.0, -0.0, 1.25], [2.5, 3.75, 4.0]]]],
            dtype=torch.float32,
        )
    generation_digest = generation_digest or _sha("generation")
    raw_bytes = _big_endian_bytes(observation)
    record = {
        "reference_observation_kind": kind,
        "reference_observation_member_sequence_index": sequence_index,
        "generation_input_identity_digest": generation_digest,
        "path": f"raw/{kind}/{sequence_index:08d}.f32be",
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "tensor_shape": list(observation.shape),
        "tensor_dtype": "torch.float32",
        "tensor_content_sha256": tensor_content_sha256(observation),
    }
    expected = {
        "reference_observation_kind": kind,
        "reference_observation_member_sequence_index": sequence_index,
        "generation_input_identity_digest": generation_digest,
    }
    return raw_bytes, record, expected


def _decode(
    raw_bytes: Any,
    record: Any,
    expected: dict[str, Any],
) -> torch.Tensor:
    return decoder_module.decode_content_routing_reference_raw_member(
        **expected,
        raw_member_bytes=raw_bytes,
        raw_member_file_record=record,
    )


def test_decoder_has_one_keyword_only_public_interface() -> None:
    assert decoder_module.__all__ == [
        "decode_content_routing_reference_raw_member"
    ]
    signature = inspect.signature(
        decoder_module.decode_content_routing_reference_raw_member
    )
    assert tuple(signature.parameters) == (
        "reference_observation_kind",
        "reference_observation_member_sequence_index",
        "generation_input_identity_digest",
        "raw_member_bytes",
        "raw_member_file_record",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    assert signature.return_annotation == "torch.Tensor"


@pytest.mark.parametrize(
    "kind",
    [
        "gradient_magnitude_rgb_pre_interpolation",
        "latent_response",
        "local_sensitivity_rgb_pre_interpolation",
    ],
)
def test_independent_decode_rebuilds_exact_cpu_float32_tensor(kind: str) -> None:
    observation = torch.tensor(
        [[[[0.0, -0.0, 1.25], [2.5, 3.75, 4.0]]]],
        dtype=torch.float32,
    )
    raw_bytes, record, expected = _fixture(observation, kind=kind)

    rebuilt = _decode(raw_bytes, record, expected)

    assert rebuilt.device.type == "cpu"
    assert rebuilt.dtype == torch.float32
    assert rebuilt.shape == observation.shape
    assert rebuilt.is_contiguous()
    assert torch.equal(rebuilt, observation)
    assert torch.equal(torch.signbit(rebuilt), torch.signbit(observation))
    assert tensor_content_sha256(rebuilt) == record["tensor_content_sha256"]


def test_encoder_decoder_roundtrip_preserves_signed_zero() -> None:
    observation = torch.tensor([[[[0.0, -0.0, 0.5, 9.0]]]], dtype=torch.float32)
    raw_bytes, record = encode_content_routing_reference_raw_member(
        reference_observation_kind="latent_response",
        reference_observation_member_sequence_index=11,
        generation_input_identity_digest=_sha("roundtrip"),
        observation=observation,
    )
    expected = {
        "reference_observation_kind": "latent_response",
        "reference_observation_member_sequence_index": 11,
        "generation_input_identity_digest": _sha("roundtrip"),
    }

    rebuilt = _decode(raw_bytes, record, expected)

    assert torch.equal(rebuilt, observation)
    assert raw_bytes.hex().startswith("0000000080000000")
    assert torch.signbit(rebuilt)[0, 0, 0].tolist() == [False, True, False, False]


def test_chunked_decode_is_deterministic_and_does_not_mutate_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(decoder_module, "_UNPACK_CHUNK_ELEMENT_COUNT", 3)
    observation = torch.arange(15, dtype=torch.float32).reshape(1, 1, 3, 5)
    raw_bytes, record, expected = _fixture(observation)
    record_before = copy.deepcopy(record)

    first = _decode(raw_bytes, record, expected)
    second = _decode(raw_bytes, record, expected)

    assert torch.equal(first, second)
    assert tensor_content_sha256(first) == tensor_content_sha256(second)
    assert raw_bytes == _big_endian_bytes(observation)
    assert record == record_before


@pytest.mark.parametrize(
    "raw_bytes",
    [bytearray(b"\0" * 4), memoryview(b"\0" * 4), [0, 0, 0, 0], "bytes"],
)
def test_raw_member_bytes_requires_exact_bytes(raw_bytes: Any) -> None:
    _, record, expected = _fixture(torch.zeros((1, 1, 1, 1)))
    with pytest.raises(TypeError, match="exact bytes"):
        _decode(raw_bytes, record, expected)


@pytest.mark.parametrize("delta", [-4, -1, 1, 4])
def test_length_mismatch_fails_before_sha_and_decode(
    monkeypatch: pytest.MonkeyPatch,
    delta: int,
) -> None:
    raw_bytes, record, expected = _fixture()
    changed = raw_bytes[: len(raw_bytes) + delta] if delta < 0 else raw_bytes + b"x" * delta

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("length mismatch reached hashing or decoding")

    monkeypatch.setattr(decoder_module, "_raw_member_bytes_sha256", forbidden)
    monkeypatch.setattr(decoder_module, "_decode_flat_binary32_big_endian", forbidden)
    with pytest.raises(ValueError, match="byte length"):
        _decode(changed, record, expected)


def test_file_sha_mismatch_fails_before_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_bytes, record, expected = _fixture()
    changed = bytes([raw_bytes[0] ^ 1]) + raw_bytes[1:]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("SHA mismatch reached decoding")

    monkeypatch.setattr(decoder_module, "_decode_flat_binary32_big_endian", forbidden)
    with pytest.raises(ValueError, match="file SHA-256"):
        _decode(changed, record, expected)


@pytest.mark.parametrize(
    "change",
    [
        lambda record: record.update({"reference_observation_kind": "latent_response"}),
        lambda record: record.update(
            {"reference_observation_member_sequence_index": 8}
        ),
        lambda record: record.update({"generation_input_identity_digest": _sha("other")}),
        lambda record: record.update({"path": "raw/wrong/00000007.f32be"}),
        lambda record: record.update({"tensor_dtype": "torch.float64"}),
        lambda record: record.update({"tensor_shape": [1, 2, 2, 3]}),
        lambda record: record.update({"extra": "field"}),
        lambda record: record.pop("sha256"),
    ],
)
def test_record_drift_fails_before_byte_hashing(
    monkeypatch: pytest.MonkeyPatch,
    change: Callable[[dict[str, Any]], Any],
) -> None:
    raw_bytes, record, expected = _fixture()
    change(record)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("invalid record reached byte hashing")

    monkeypatch.setattr(decoder_module, "_raw_member_bytes_sha256", forbidden)
    with pytest.raises(ValueError):
        _decode(raw_bytes, record, expected)


@pytest.mark.parametrize(
    "expected_change",
    [
        {"reference_observation_kind": "unknown"},
        {"reference_observation_kind": True},
        {"reference_observation_member_sequence_index": -1},
        {"reference_observation_member_sequence_index": True},
        {"generation_input_identity_digest": "0" * 63},
    ],
)
def test_invalid_expected_identity_fails_before_record_validation(
    monkeypatch: pytest.MonkeyPatch,
    expected_change: dict[str, Any],
) -> None:
    raw_bytes, record, expected = _fixture()
    expected.update(expected_change)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("invalid expected identity reached record validation")

    monkeypatch.setattr(decoder_module, "_validate_exact_object", forbidden)
    with pytest.raises(ValueError):
        _decode(raw_bytes, record, expected)


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
        lambda contract: contract["type_predicates"].pop("positive_int_list"),
        lambda contract: contract["materialization_contract"][
            "raw_member_file_record_contract"
        ].update({"decoding_rule": "guess_endian_then_decode"}),
        lambda contract: contract["materialization_contract"][
            "raw_member_file_record_contract"
        ].update({"file_length_rule": "at_least_shape_bytes"}),
        lambda contract: contract["materialization_contract"][
            "raw_member_file_record_contract"
        ].update({"file_sha256_rule": "sha256(decoded_tensor)"}),
        lambda contract: contract["materialization_contract"][
            "raw_member_file_record_contract"
        ].update({"tensor_digest_rule": "sha256(raw_bytes)"}),
        lambda contract: contract["materialization_contract"][
            "raw_member_file_record_contract"
        ]["field_rules"]["tensor_shape"].pop("exact_prefix"),
        lambda contract: contract["materialization_contract"][
            "raw_member_file_record_contract"
        ]["field_rules"]["tensor_dtype"].update({"exact_value": "torch.float64"}),
        lambda contract: contract["materialization_contract"][
            "raw_member_file_record_contract"
        ]["path_templates"].pop("latent_response"),
        lambda contract: contract["materialization_contract"][
            "raw_member_file_record_contract"
        ].update({"cross_field_invariants": ()}),
    ],
)
def test_contract_drift_fails_before_record_or_bytes_are_consumed(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], Any],
) -> None:
    contract = copy.deepcopy(decoder_module._load_machine_contract())
    mutation(contract)
    monkeypatch.setattr(decoder_module, "_load_machine_contract", lambda: contract)
    raw_bytes, record, expected = _fixture()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("contract drift reached record or byte consumption")

    monkeypatch.setattr(decoder_module, "_validate_exact_object", forbidden)
    monkeypatch.setattr(decoder_module, "_raw_member_bytes_sha256", forbidden)
    with pytest.raises(ValueError):
        _decode(raw_bytes, record, expected)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_nonfinite_or_negative_decoded_values_fail_before_tensor_digest(
    monkeypatch: pytest.MonkeyPatch,
    value: float,
) -> None:
    observation = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    raw_bytes, record, expected = _fixture(observation)
    changed = struct.pack(">f", value)
    record["sha256"] = hashlib.sha256(changed).hexdigest()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("invalid decoded values reached tensor digest")

    monkeypatch.setattr(decoder_module, "tensor_content_sha256", forbidden)
    with pytest.raises(ValueError):
        _decode(changed, record, expected)


def test_wrong_endianness_with_matching_file_sha_fails_tensor_identity() -> None:
    observation = torch.tensor([[[[1.0, 2.0, 4.0]]]], dtype=torch.float32)
    _, record, expected = _fixture(observation)
    little_endian = b"".join(struct.pack("<f", value) for value in [1.0, 2.0, 4.0])
    record["sha256"] = hashlib.sha256(little_endian).hexdigest()

    with pytest.raises(ValueError, match="content digest"):
        _decode(little_endian, record, expected)


def test_tensor_digest_mismatch_fails_closed() -> None:
    raw_bytes, record, expected = _fixture()
    record["tensor_content_sha256"] = _sha("wrong-tensor")
    with pytest.raises(ValueError, match="content digest"):
        _decode(raw_bytes, record, expected)


def test_source_is_independent_and_has_no_io_or_scientific_postprocessing() -> None:
    source = inspect.getsource(decoder_module)
    forbidden = (
        "content_routing_reference_raw_member import",
        "encode_content_routing_reference_raw_member",
        "_encode_flat_binary32_big_endian",
        "open(",
        "Path(",
        "torch.quantile",
        "torch.clamp",
        ".to(",
        ".cuda(",
        "decoder",
        "transformers",
    )
    for token in forbidden:
        if token == "decoder":
            continue
        assert token not in source
    assert "struct.unpack" in source
    assert ">" in source
    assert not FIXED_REGISTRY.exists()
