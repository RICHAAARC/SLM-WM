"""论文分主张三态决策和兼容字段派生测试。"""

from __future__ import annotations

import pytest

from paper_experiments.analysis.paper_claim_decisions import (
    ClaimDecisionGovernanceError,
    build_claim_decision,
    build_claim_decision_bundle,
    load_paper_claim_registry,
    validate_claim_decision_bundle,
)


pytestmark = pytest.mark.quick


def _decision(
    claim_id: str,
    *,
    evidence_complete: bool = True,
    scientific_support: bool | None = True,
) -> dict[str, object]:
    """构造只改变指定主张状态的最小受治理决策。"""

    return build_claim_decision(
        claim_id,
        evidence_complete=evidence_complete,
        scientific_support=scientific_support,
        evidence_artifact_ids=(f"{claim_id}_artifact",),
        evidence_blockers=(
            (f"{claim_id}_evidence_missing",) if not evidence_complete else ()
        ),
    )


def test_parameter_sensitivity_cannot_enter_registered_claim_bundle() -> None:
    """参数敏感性是诊断证据，不能伪装成可选论文主张。"""

    registry = load_paper_claim_registry()
    decisions = {
        claim_id: _decision(claim_id)
        for claim_id in registry["registered_claim_ids"]
    }
    decisions["parameter_robustness"] = _decision(
        "parameter_robustness",
        evidence_complete=False,
        scientific_support=None,
    )

    assert registry["optional_claims"] == []
    with pytest.raises(
        ClaimDecisionGovernanceError,
        match="未精确覆盖登记主张集合",
    ):
        build_claim_decision_bundle(decisions, registry=registry)


def test_required_claim_distinguishes_negative_measurement_from_missing_evidence() -> None:
    """完整但未支持与证据缺失必须形成不同三态结论。"""

    registry = load_paper_claim_registry()
    decisions = {
        claim_id: _decision(claim_id)
        for claim_id in registry["registered_claim_ids"]
    }
    decisions["baseline_superiority"] = _decision(
        "baseline_superiority",
        scientific_support=False,
    )
    measured = build_claim_decision_bundle(decisions, registry=registry)
    assert measured["registered_claim_set_decision"] == "measured_not_supported"
    assert measured["registered_claim_set_evidence_complete"] is True
    assert measured["registered_claim_set_scientific_support"] is False

    decisions["quality_preservation"] = _decision(
        "quality_preservation",
        evidence_complete=False,
        scientific_support=None,
    )
    incomplete = build_claim_decision_bundle(decisions, registry=registry)
    assert incomplete["registered_claim_set_decision"] == "evidence_incomplete"
    assert incomplete["registered_claim_set_evidence_complete"] is False
    assert incomplete["registered_claim_set_scientific_support"] is None


def test_compatibility_boolean_cannot_override_registered_claim_decisions() -> None:
    """旧布尔字段被人工改写时必须由集中 validator 拒绝。"""

    registry = load_paper_claim_registry()
    bundle = build_claim_decision_bundle(
        {
            claim_id: _decision(claim_id)
            for claim_id in registry["registered_claim_ids"]
        },
        registry=registry,
    )
    forged = {**bundle, "supports_paper_claim": False}

    with pytest.raises(
        ClaimDecisionGovernanceError,
        match="supports_paper_claim",
    ):
        validate_claim_decision_bundle(forged, registry=registry)
