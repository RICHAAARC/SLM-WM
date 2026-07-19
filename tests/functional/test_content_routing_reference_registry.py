"""Test fail-closed loading of the unique content-routing registry."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
from typing import Any

import pytest
import torch

import experiments.protocol.content_routing_reference_registry as registry_module
import experiments.protocol.content_routing_reference_registry_payload as payload_module
from experiments.protocol.content_routing_reference_quantile import (
    ContentRoutingReferenceScalars,
)


pytestmark = pytest.mark.quick


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _member(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, -1)


def _registry_bytes() -> bytes:
    schema = registry_module._REFERENCE_REGISTRY_SCHEMA
    dependency = _sha("dependency")
    formal = _sha("formal")
    return payload_module.assemble_content_routing_reference_registry_payload(
        method_parameter_partition_id="isolated-method-parameter-partition",
        prompt_projection=[
            {"prompt_id": "prompt", "prompt_text_digest": _sha("prompt")}
        ],
        seed_projection_random=[17],
        generation_input_identity_digests=[
            _sha("fixture_only_unqualified_generation_digest")
        ],
        gradient_observations=[_member([0.0, 1.0, 2.0])],
        response_observations=[_member([0.0, 0.5, 1.0])],
        sensitivity_observations=[_member([0.0, 10.0, 20.0])],
        formal_execution_lock_digest=formal,
        dependency_profile_digest=dependency,
        runtime_component_identity_payload={
            "model_id": schema["model_id"],
            "model_revision": schema["model_revision"],
            "dependency_profile_digest": dependency,
            "formal_execution_lock_digest": formal,
            "vae_preprocess_identity_digest": _sha("vae-preprocess"),
            "scheduler_identity_digest": _sha("scheduler"),
            "content_observation_formula_identity_digest": _sha(
                "content-formulas"
            ),
        },
    )


def _identity(payload: bytes) -> tuple[str, str]:
    decoded = json.loads(payload.decode("utf-8"))
    return (
        decoded["content_routing_reference_registry_digest"],
        hashlib.sha256(payload).hexdigest(),
    )


def _load_from(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    *,
    expected_registry_digest: str,
    expected_file_sha256: str,
) -> ContentRoutingReferenceScalars:
    monkeypatch.setattr(registry_module, "_REFERENCE_REGISTRY_PATH", path)
    return registry_module.load_content_routing_reference_registry(
        expected_registry_digest=expected_registry_digest,
        expected_file_sha256=expected_file_sha256,
    )


def test_loader_has_exact_public_interface_and_fixed_path() -> None:
    assert registry_module.__all__ == ["load_content_routing_reference_registry"]
    signature = inspect.signature(
        registry_module.load_content_routing_reference_registry
    )
    assert tuple(signature.parameters) == (
        "expected_registry_digest",
        "expected_file_sha256",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    assert signature.return_annotation == "ContentRoutingReferenceScalars"
    assert registry_module._REFERENCE_REGISTRY_PATH.name == (
        "content_routing_reference_registry.json"
    )


def test_loader_returns_exact_binary32_reference_scalars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _registry_bytes()
    semantic, file_sha = _identity(payload)
    path = tmp_path / "content_routing_reference_registry.json"
    path.write_bytes(payload)
    result = _load_from(
        monkeypatch,
        path,
        expected_registry_digest=semantic,
        expected_file_sha256=file_sha,
    )
    assert result == ContentRoutingReferenceScalars(
        reference_gradient=2.0,
        reference_response=1.0,
        reference_sensitivity=20.0,
    )


def test_loader_uses_exact_read_flags_and_fstat_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _registry_bytes()
    semantic, file_sha = _identity(payload)
    path = tmp_path / "content_routing_reference_registry.json"
    path.write_bytes(payload)
    expected_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    calls: list[str] = []
    real_open = os.open
    real_fstat = os.fstat
    real_read = os.read

    def guarded_open(target: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        assert flags == expected_flags
        calls.append("open")
        return real_open(target, flags, *args, **kwargs)

    def guarded_fstat(descriptor: int) -> os.stat_result:
        calls.append("fstat")
        return real_fstat(descriptor)

    def guarded_read(descriptor: int, size: int) -> bytes:
        assert calls[-1] == "fstat"
        calls.append("read")
        return real_read(descriptor, size)

    monkeypatch.setattr(registry_module.os, "open", guarded_open)
    monkeypatch.setattr(registry_module.os, "fstat", guarded_fstat)
    monkeypatch.setattr(registry_module.os, "read", guarded_read)
    _load_from(
        monkeypatch,
        path,
        expected_registry_digest=semantic,
        expected_file_sha256=file_sha,
    )
    assert calls == ["open", "fstat", "read"]


def test_nonregular_file_is_rejected_before_read_and_fd_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = 0
    closes = 0

    monkeypatch.setattr(registry_module.os, "open", lambda *_args: 71)
    monkeypatch.setattr(
        registry_module.os,
        "fstat",
        lambda _fd: type("Metadata", (), {"st_mode": stat.S_IFIFO, "st_size": 0})(),
    )

    def forbidden_read(*_args: Any) -> bytes:
        nonlocal reads
        reads += 1
        return b""

    def close(_fd: int) -> None:
        nonlocal closes
        closes += 1

    monkeypatch.setattr(registry_module.os, "read", forbidden_read)
    monkeypatch.setattr(registry_module.os, "close", close)
    with pytest.raises(ValueError, match="regular file"):
        registry_module._read_fixed_registry_bytes()
    assert reads == 0
    assert closes == 1


@pytest.mark.parametrize("expected_name", ["semantic", "file"])
def test_expected_identity_mismatch_fails_closed(
    expected_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _registry_bytes()
    semantic, file_sha = _identity(payload)
    path = tmp_path / "content_routing_reference_registry.json"
    path.write_bytes(payload)
    if expected_name == "semantic":
        semantic = _sha("wrong-semantic")
    else:
        file_sha = _sha("wrong-file")
    with pytest.raises(ValueError):
        _load_from(
            monkeypatch,
            path,
            expected_registry_digest=semantic,
            expected_file_sha256=file_sha,
        )


def test_same_semantics_reserialized_bytes_fail_exact_file_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _registry_bytes()
    semantic, original_file_sha = _identity(payload)
    decoded = json.loads(payload.decode("utf-8"))
    reserialized = json.dumps(decoded, ensure_ascii=False, indent=2).encode("utf-8")
    path = tmp_path / "content_routing_reference_registry.json"
    path.write_bytes(reserialized)
    with pytest.raises(ValueError, match="file digest"):
        _load_from(
            monkeypatch,
            path,
            expected_registry_digest=semantic,
            expected_file_sha256=original_file_sha,
        )


def test_noncanonical_bytes_fail_even_with_matching_file_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _registry_bytes()
    semantic, _ = _identity(payload)
    decoded = json.loads(payload.decode("utf-8"))
    noncanonical = json.dumps(decoded, indent=1).encode("utf-8")
    path = tmp_path / "content_routing_reference_registry.json"
    path.write_bytes(noncanonical)
    with pytest.raises(ValueError, match="not canonical"):
        _load_from(
            monkeypatch,
            path,
            expected_registry_digest=semantic,
            expected_file_sha256=hashlib.sha256(noncanonical).hexdigest(),
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.__setitem__("extra", 1),
        lambda value: value.pop("method_parameter_partition_id"),
        lambda value: value.__setitem__("registry_schema", "wrong"),
        lambda value: value["content_routing_reference_populations"].reverse(),
        lambda value: value["content_routing_reference_populations"][0].__setitem__(
            "extra", 1
        ),
    ],
)
def test_exact_schema_mutations_fail_closed(
    mutator: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = json.loads(_registry_bytes().decode("utf-8"))
    mutator(registry)
    semantic = registry.get("content_routing_reference_registry_digest", _sha("x"))
    raw = json.dumps(
        registry,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    path = tmp_path / "content_routing_reference_registry.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        _load_from(
            monkeypatch,
            path,
            expected_registry_digest=semantic,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_finite_positive_nonbinary32_json_double_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = json.loads(_registry_bytes().decode("utf-8"))
    registry["reference_gradient"] = 1.0000000001
    semantic_payload = dict(registry)
    semantic_payload.pop("content_routing_reference_registry_digest")
    registry["content_routing_reference_registry_digest"] = hashlib.sha256(
        json.dumps(
            semantic_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    raw = json.dumps(
        registry,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    path = tmp_path / "content_routing_reference_registry.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="binary32"):
        _load_from(
            monkeypatch,
            path,
            expected_registry_digest=registry[
                "content_routing_reference_registry_digest"
            ],
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


@pytest.mark.parametrize(
    "payload",
    [
        b"[]",
        b'{"x": NaN}',
        b'{"x": 1, "x": 2}',
        b"\xff",
    ],
)
def test_strict_utf8_and_json_fail_closed(
    payload: bytes,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "content_routing_reference_registry.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        _load_from(
            monkeypatch,
            path,
            expected_registry_digest=_sha("semantic"),
            expected_file_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_missing_fixed_registry_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "content_routing_reference_registry.json"
    with pytest.raises(ValueError, match="cannot be opened"):
        _load_from(
            monkeypatch,
            missing,
            expected_registry_digest=_sha("semantic"),
            expected_file_sha256=_sha("file"),
        )
