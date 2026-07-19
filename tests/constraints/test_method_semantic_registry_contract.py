"""验证方法语义追踪只覆盖当前正式内容双链。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from main.methods.method_definition import (
    METHOD_DEFINITION_SCHEMA,
    semantic_conditioned_latent_method_definition_digest,
)
from tools.harness.lib.method_semantic_registry import (
    EXPECTED_INVARIANT_IDS,
    EXPECTED_NORMATIVE_TRACE_DIGEST,
    REGISTRY_SCHEMA,
    REGISTRY_SCOPE,
    load_method_semantic_registry,
    method_semantic_normative_trace_digest,
    validate_method_semantic_registry,
)


ROOT = Path(__file__).resolve().parents[2]


def _rules(payload: dict[str, object]) -> set[str]:
    return {
        item["rule"]
        for item in validate_method_semantic_registry(
            ROOT,
            payload,
            expected_method_definition_schema=METHOD_DEFINITION_SCHEMA,
            expected_method_definition_digest=(
                semantic_conditioned_latent_method_definition_digest()
            ),
        )
    }


@pytest.mark.constraint
def test_registry_tracks_only_current_formal_dual_chain() -> None:
    payload = load_method_semantic_registry(ROOT)
    assert _rules(payload) == set()
    assert payload["registry_schema"] == REGISTRY_SCHEMA
    assert payload["registry_scope"] == REGISTRY_SCOPE
    assert tuple(
        item["invariant_id"] for item in payload["invariants"]
    ) == EXPECTED_INVARIANT_IDS
    assert method_semantic_normative_trace_digest(payload) == (
        EXPECTED_NORMATIVE_TRACE_DIGEST
    )
    encoded = str(payload)
    for legacy_token in (
        "complete_716_feature_jacobian",
        "JacobianNullSpaceResult",
        "build_tail_robust_template",
        "torch.func.linearize",
        "torch.func.vjp",
        "psd_cg",
    ):
        assert legacy_token not in encoded


@pytest.mark.constraint
def test_registry_bindings_are_real_and_cross_layer() -> None:
    payload = load_method_semantic_registry(ROOT)
    for invariant in payload["invariants"]:
        assert all(
            binding["path"].startswith("main/")
            for binding in invariant["method_implementation_symbols"]
        )
        assert invariant["runtime_binding_symbols"] == [
            {
                "path": "experiments/runners/semantic_watermark_runtime.py",
                "symbol": "run_semantic_watermark_runtime",
            }
        ]
        assert invariant["specification_test_nodes"]
        assert invariant["cpu_property_test_nodes"]


@pytest.mark.constraint
def test_registry_rejects_identity_scope_and_self_assertion_drift() -> None:
    base = load_method_semantic_registry(ROOT)
    mutations = []
    reordered = deepcopy(base)
    reordered["invariants"].reverse()
    mutations.append((reordered, "invariant_exact_set"))
    self_asserted = deepcopy(base)
    self_asserted["invariants"][0]["supports_paper_claim"] = True
    mutations.append((self_asserted, "self_asserted_conformance"))
    schema = deepcopy(base)
    schema["method_definition_schema"] = "legacy_method"
    mutations.append((schema, "method_definition_schema"))
    for payload, expected_rule in mutations:
        assert expected_rule in _rules(payload)


@pytest.mark.constraint
def test_registry_rejects_broken_symbol_test_config_and_field_links() -> None:
    base = load_method_semantic_registry(ROOT)
    mutations = []
    symbol = deepcopy(base)
    symbol["invariants"][0]["method_implementation_symbols"][0][
        "symbol"
    ] = "missing_symbol"
    mutations.append((symbol, "method_implementation_symbols"))
    test_node = deepcopy(base)
    test_node["invariants"][0]["cpu_property_test_nodes"] = [
        "tests/functional/test_content_runtime_adapter.py::test_missing"
    ]
    mutations.append((test_node, "cpu_property_test_nodes"))
    config = deepcopy(base)
    config["invariants"][0]["configuration_fields"] = ["missing_field"]
    mutations.append((config, "configuration_fields"))
    field = deepcopy(base)
    field["invariants"][0]["runtime_evidence_fields"] = [
        "unregistered_runtime_fact"
    ]
    mutations.append((field, "field_registry"))
    for payload, expected_rule in mutations:
        assert expected_rule in _rules(payload)


@pytest.mark.constraint
def test_registry_digest_binds_human_readable_method_responsibilities() -> None:
    payload = deepcopy(load_method_semantic_registry(ROOT))
    payload["invariants"][0]["claim_boundary"] += " drift"
    assert "normative_trace_digest" in _rules(payload)
