from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
from typing import Any

import pytest
import torch

from experiments.protocol import content_routing_reference_generation_input as generation
from experiments.protocol import content_routing_reference_registry as registry
from experiments.protocol.formal_randomization import (
    FORMAL_BASE_LATENT_GENERATION_PROTOCOL,
    build_canonical_sd35_base_latent,
    formal_randomization_protocol_record,
)
from main.core.digest import build_stable_digest, tensor_content_sha256


pytestmark = pytest.mark.quick


ROOT = Path(__file__).resolve().parents[2]
FIXED_REGISTRY = ROOT / "configs" / "content_routing_reference_registry.json"


def _sha(character: str) -> str:
    return character * 64


def _arguments(**overrides: Any) -> dict[str, Any]:
    arguments = {
        "reference_observation_member_sequence_index": 0,
        "prompt_id": "prompt-000",
        "prompt_text": "A lighthouse above a quiet sea",
        "generation_seed_random": 17,
        "formal_method_config_digest": _sha("a"),
        "dependency_profile_digest": _sha("b"),
        "formal_execution_lock_digest": _sha("c"),
        "runtime_component_identity_digest": _sha("d"),
    }
    arguments.update(overrides)
    return arguments


def _fake_canonical_builder(**arguments: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    latent = torch.linspace(
        -1.0,
        1.0,
        steps=1 * 16 * 64 * 64,
        dtype=arguments["dtype"],
        device=arguments["device"],
    ).reshape(arguments["shape"])
    protocol = formal_randomization_protocol_record()
    identity = {
        "generation_seed_random": arguments["generation_seed_random"],
        "base_latent_generation_protocol": FORMAL_BASE_LATENT_GENERATION_PROTOCOL,
        "base_latent_keyed_prg_version": protocol["base_latent_keyed_prg_version"],
        "base_latent_keyed_prg_protocol_digest": protocol[
            "base_latent_keyed_prg_protocol_digest"
        ],
        "formal_randomization_protocol_digest": protocol[
            "formal_randomization_protocol_digest"
        ],
        "base_latent_dtype": str(latent.dtype),
        "base_latent_shape": [int(value) for value in latent.shape],
        "base_latent_content_digest_random": tensor_content_sha256(latent),
    }
    identity["base_latent_identity_digest_random"] = build_stable_digest(identity)
    return latent, identity


def _build(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> dict[str, Any]:
    monkeypatch.setattr(
        generation,
        "build_canonical_sd35_base_latent",
        _fake_canonical_builder,
    )
    return generation.build_content_routing_reference_generation_input_record(
        **_arguments(**overrides)
    )


def _consumed_rule_locations() -> tuple[tuple[str, str], ...]:
    materialization = registry._load_machine_contract()["materialization_contract"]
    locations: list[tuple[str, str]] = []
    for contract_name in (
        "generation_input_identity_payload_contract",
        "ordered_generation_input_record_contract",
    ):
        locations.extend(
            (contract_name, field_name)
            for field_name in materialization[contract_name]["field_rules"]
        )
    return tuple(locations)


_CONSUMED_RULE_LOCATIONS = _consumed_rule_locations()
_EXACT_VALUE_RULE_LOCATIONS = tuple(
    (contract_name, field_name)
    for contract_name, field_name in _CONSUMED_RULE_LOCATIONS
    if "exact_value"
    in registry._load_machine_contract()["materialization_contract"][contract_name][
        "field_rules"
    ][field_name]
)


def _assert_contract_mutation_fails_before_builder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    contract_name: str,
    field_name: str,
    mutation: Any,
) -> None:
    contract = deepcopy(registry._load_machine_contract())
    rule = contract["materialization_contract"][contract_name]["field_rules"][
        field_name
    ]
    mutation(rule)
    call_count = 0

    def forbidden(**_arguments: Any) -> tuple[Any, Any]:
        nonlocal call_count
        call_count += 1
        raise AssertionError("canonical builder must not run")

    monkeypatch.setattr(generation, "_load_machine_contract", lambda: contract)
    monkeypatch.setattr(generation, "build_canonical_sd35_base_latent", forbidden)
    with pytest.raises(ValueError):
        generation.build_content_routing_reference_generation_input_record(
            **_arguments()
        )
    assert call_count == 0


def test_generation_input_builder_has_one_keyword_only_public_interface() -> None:
    assert generation.__all__ == [
        "build_content_routing_reference_generation_input_record"
    ]
    signature = inspect.signature(
        generation.build_content_routing_reference_generation_input_record
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert set(signature.parameters) == set(_arguments())
    assert all(
        forbidden not in signature.parameters
        for forbidden in ("path", "base_latent", "base_latent_identity", "device")
    )


def test_record_and_digests_match_the_machine_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def capture(**arguments: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        calls.append(dict(arguments))
        return _fake_canonical_builder(**arguments)

    monkeypatch.setattr(generation, "build_canonical_sd35_base_latent", capture)
    arguments = _arguments()
    record = generation.build_content_routing_reference_generation_input_record(
        **arguments
    )
    assert calls == [
        {
            "shape": (1, 16, 64, 64),
            "generation_seed_random": 17,
            "model_id": "stabilityai/stable-diffusion-3.5-medium",
            "model_revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
            "device": "cpu",
            "dtype": torch.float16,
        }
    ]
    payload = record["generation_input_identity_payload"]
    assert payload["prompt_text_digest"] == build_stable_digest(
        {"prompt_text": arguments["prompt_text"]}
    )
    assert payload["negative_prompt"] == "low quality, blurry"
    assert (payload["width"], payload["height"]) == (512, 512)
    assert payload["inference_steps"] == 20
    assert payload["guidance_scale"] == 4.5
    assert record == {
        "reference_observation_member_sequence_index": 0,
        "generation_input_identity_payload": payload,
        "generation_input_identity_digest": build_stable_digest(payload),
    }


def test_sequence_index_is_not_part_of_generation_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _build(monkeypatch, reference_observation_member_sequence_index=0)
    second = _build(monkeypatch, reference_observation_member_sequence_index=9)
    assert first["generation_input_identity_payload"] == second[
        "generation_input_identity_payload"
    ]
    assert first["generation_input_identity_digest"] == second[
        "generation_input_identity_digest"
    ]
    assert first != second


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("reference_observation_member_sequence_index", True),
        ("reference_observation_member_sequence_index", -1),
        ("prompt_id", ""),
        ("prompt_text", " padded "),
        ("generation_seed_random", False),
        ("generation_seed_random", -1),
        ("formal_method_config_digest", "not-a-digest"),
        ("dependency_profile_digest", "A" * 64),
        ("formal_execution_lock_digest", 7),
        ("runtime_component_identity_digest", None),
    ),
)
def test_invalid_static_inputs_fail_before_canonical_builder(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid: Any,
) -> None:
    call_count = 0

    def forbidden(**_arguments: Any) -> tuple[Any, Any]:
        nonlocal call_count
        call_count += 1
        raise AssertionError("canonical builder must not run")

    monkeypatch.setattr(generation, "build_canonical_sd35_base_latent", forbidden)
    with pytest.raises(ValueError):
        generation.build_content_routing_reference_generation_input_record(
            **_arguments(**{field: invalid})
        )
    assert call_count == 0


@pytest.mark.parametrize(
    ("contract_name", "field_name"),
    _CONSUMED_RULE_LOCATIONS,
)
def test_supported_but_wrong_field_predicate_fails_before_canonical_builder(
    monkeypatch: pytest.MonkeyPatch,
    contract_name: str,
    field_name: str,
) -> None:
    def use_wrong_supported_predicate(rule: dict[str, Any]) -> None:
        rule.clear()
        rule["predicate"] = "exact_bool"

    _assert_contract_mutation_fails_before_builder(
        monkeypatch,
        contract_name=contract_name,
        field_name=field_name,
        mutation=use_wrong_supported_predicate,
    )


@pytest.mark.parametrize(
    ("contract_name", "field_name"),
    _CONSUMED_RULE_LOCATIONS,
)
def test_missing_predicate_fails_before_canonical_builder(
    monkeypatch: pytest.MonkeyPatch,
    contract_name: str,
    field_name: str,
) -> None:
    _assert_contract_mutation_fails_before_builder(
        monkeypatch,
        contract_name=contract_name,
        field_name=field_name,
        mutation=lambda rule: rule.pop("predicate"),
    )


@pytest.mark.parametrize(
    ("contract_name", "field_name"),
    _CONSUMED_RULE_LOCATIONS,
)
def test_extra_rule_attribute_fails_before_canonical_builder(
    monkeypatch: pytest.MonkeyPatch,
    contract_name: str,
    field_name: str,
) -> None:
    _assert_contract_mutation_fails_before_builder(
        monkeypatch,
        contract_name=contract_name,
        field_name=field_name,
        mutation=lambda rule: rule.__setitem__("unexpected", "drift"),
    )


@pytest.mark.parametrize(
    ("contract_name", "field_name"),
    _EXACT_VALUE_RULE_LOCATIONS,
)
def test_missing_exact_value_fails_before_canonical_builder(
    monkeypatch: pytest.MonkeyPatch,
    contract_name: str,
    field_name: str,
) -> None:
    _assert_contract_mutation_fails_before_builder(
        monkeypatch,
        contract_name=contract_name,
        field_name=field_name,
        mutation=lambda rule: rule.pop("exact_value"),
    )


@pytest.mark.parametrize(
    ("contract_name", "field_name"),
    _EXACT_VALUE_RULE_LOCATIONS,
)
def test_drifted_exact_value_fails_before_canonical_builder(
    monkeypatch: pytest.MonkeyPatch,
    contract_name: str,
    field_name: str,
) -> None:
    def drift_exact_value(rule: dict[str, Any]) -> None:
        value = rule["exact_value"]
        rule["exact_value"] = (
            "supported-but-wrong" if type(value) is str else value + 1
        )

    _assert_contract_mutation_fails_before_builder(
        monkeypatch,
        contract_name=contract_name,
        field_name=field_name,
        mutation=drift_exact_value,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda contract: contract["materialization_contract"].__setitem__(
            "contract_schema_token", "wrong"
        ),
        lambda contract: contract["materialization_contract"][
            "generation_input_identity_payload_contract"
        ].__setitem__("digest_rule", "build_stable_digest(wrong)"),
        lambda contract: contract["materialization_contract"][
            "ordered_generation_input_record_contract"
        ]["nested_contracts"].clear(),
        lambda contract: contract["materialization_contract"][
            "ordered_generation_input_record_contract"
        ].__setitem__("order_rule", "wrong_order"),
        lambda contract: contract["materialization_contract"][
            "generation_input_identity_payload_contract"
        ].__setitem__("cross_field_invariants", ("wrong",)),
        lambda contract: contract["materialization_contract"][
            "base_latent_identity_reconstruction_contract"
        ].__setitem__("builder_symbol", "wrong_builder"),
        lambda contract: contract["materialization_contract"][
            "base_latent_identity_reconstruction_contract"
        ].__setitem__("shape", (1, 16, 32, 32)),
        lambda contract: contract["materialization_contract"][
            "base_latent_identity_reconstruction_contract"
        ].__setitem__("dtype", "torch.float32"),
        lambda contract: contract["materialization_contract"][
            "base_latent_identity_reconstruction_contract"
        ]["argument_sources"].__setitem__("shape", "caller"),
        lambda contract: contract["materialization_contract"][
            "base_latent_identity_reconstruction_contract"
        ].__setitem__("returned_identity_fields", ("generation_seed_random",)),
        lambda contract: contract["materialization_contract"][
            "base_latent_identity_reconstruction_contract"
        ].__setitem__("cross_field_invariants", ("wrong",)),
    ),
)
def test_contract_drift_fails_before_canonical_builder(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
) -> None:
    contract = deepcopy(registry._load_machine_contract())
    mutation(contract)
    call_count = 0

    def forbidden(**_arguments: Any) -> tuple[Any, Any]:
        nonlocal call_count
        call_count += 1
        raise AssertionError("canonical builder must not run")

    monkeypatch.setattr(generation, "_load_machine_contract", lambda: contract)
    monkeypatch.setattr(generation, "build_canonical_sd35_base_latent", forbidden)
    with pytest.raises(ValueError):
        generation.build_content_routing_reference_generation_input_record(
            **_arguments()
        )
    assert call_count == 0


@pytest.mark.parametrize(
    "mutation",
    (
        lambda latent, identity: (latent.to(torch.float32), identity),
        lambda latent, identity: (latent.reshape(1, 16, 32, 128), identity),
        lambda latent, identity: (
            latent.clone().index_fill_(0, torch.tensor([0]), float("nan")),
            identity,
        ),
        lambda latent, identity: (
            latent,
            {key: value for key, value in identity.items() if key != "base_latent_shape"},
        ),
        lambda latent, identity: (
            latent,
            {**identity, "generation_seed_random": 18},
        ),
        lambda latent, identity: (
            latent,
            {**identity, "base_latent_generation_protocol": "wrong"},
        ),
        lambda latent, identity: (
            latent,
            {**identity, "base_latent_keyed_prg_version": "wrong"},
        ),
        lambda latent, identity: (
            latent,
            {**identity, "formal_randomization_protocol_digest": _sha("0")},
        ),
        lambda latent, identity: (
            latent,
            {**identity, "base_latent_dtype": "torch.float32"},
        ),
        lambda latent, identity: (
            latent,
            {**identity, "base_latent_shape": [1, 16, 32, 128]},
        ),
        lambda latent, identity: (
            latent,
            {**identity, "base_latent_content_digest_random": _sha("e")},
        ),
        lambda latent, identity: (
            latent,
            {**identity, "base_latent_identity_digest_random": _sha("f")},
        ),
    ),
)
def test_helper_output_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
) -> None:
    def drifted(**arguments: Any) -> tuple[Any, Any]:
        latent, identity = _fake_canonical_builder(**arguments)
        return mutation(latent, identity)

    monkeypatch.setattr(
        generation,
        "build_canonical_sd35_base_latent",
        drifted,
    )
    with pytest.raises(ValueError):
        generation.build_content_routing_reference_generation_input_record(
            **_arguments()
        )


def test_prompt_seed_and_repeated_call_identity_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _build(monkeypatch)
    repeated = _build(monkeypatch)
    prompt_changed = _build(monkeypatch, prompt_text="A city beneath the rain")
    seed_changed = _build(monkeypatch, generation_seed_random=18)
    config_changed = _build(monkeypatch, formal_method_config_digest=_sha("e"))
    assert first == repeated
    assert first["generation_input_identity_digest"] != prompt_changed[
        "generation_input_identity_digest"
    ]
    assert first["generation_input_identity_payload"][
        "base_latent_identity_digest_random"
    ] == prompt_changed["generation_input_identity_payload"][
        "base_latent_identity_digest_random"
    ]
    assert first["generation_input_identity_payload"][
        "base_latent_identity_digest_random"
    ] != seed_changed["generation_input_identity_payload"][
        "base_latent_identity_digest_random"
    ]
    assert first["generation_input_identity_digest"] != config_changed[
        "generation_input_identity_digest"
    ]


def test_real_cpu_helper_smoke_is_deterministic_and_exact() -> None:
    arguments = _arguments()
    first = generation.build_content_routing_reference_generation_input_record(
        **arguments
    )
    second = generation.build_content_routing_reference_generation_input_record(
        **arguments
    )
    direct_latent, direct_identity = build_canonical_sd35_base_latent(
        shape=(1, 16, 64, 64),
        generation_seed_random=arguments["generation_seed_random"],
        model_id="stabilityai/stable-diffusion-3.5-medium",
        model_revision="b940f670f0eda2d07fbb75229e779da1ad11eb80",
        device="cpu",
        dtype=torch.float16,
    )
    assert first == second
    assert direct_latent.device.type == "cpu"
    assert first["generation_input_identity_payload"][
        "base_latent_identity_digest_random"
    ] == direct_identity["base_latent_identity_digest_random"]


def test_builder_has_no_artifact_model_network_or_cuda_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = registry._load_machine_contract()
    monkeypatch.setattr(generation, "_load_machine_contract", lambda: contract)

    def forbidden(*_arguments: Any, **_keywords: Any) -> Any:
        raise AssertionError("artifact or CUDA path must not run")

    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    _build(monkeypatch)
    source = inspect.getsource(generation)
    for forbidden_text in (
        "transformers",
        "diffusers",
        "requests",
        "urllib",
        "vae_decoder",
        "os.replace",
        "open(",
    ):
        assert forbidden_text not in source
    assert not FIXED_REGISTRY.exists()
