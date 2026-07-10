"""验证纯算法原语闭环。"""

from __future__ import annotations

import pytest

from main.methods.algorithm_primitives import (
    build_semantic_risk_field,
    compose_latent_update,
    compute_content_score,
    decide_evidence_and_final,
    derive_attention_carrier_stub,
    derive_tail_carrier,
    derive_lf_carrier,
    estimate_safe_basis,
    evaluate_geometry_reliability,
    project_latent_mask,
)


def _build_synthetic_basis():
    """构造轻量 synthetic basis, 供多个功能测试复用。"""
    latent_values = (0.2, -0.1, 0.4, -0.3, 0.6, -0.2, 0.1, -0.5)
    risk_field = build_semantic_risk_field(
        semantic_values=(0.2, 0.3, 0.4, 0.5, 0.7, 0.6, 0.2, 0.3),
        texture_values=(0.2, 0.3, 0.7, 0.8, 0.9, 0.6, 0.2, 0.7),
        stability_values=(0.8, 0.7, 0.9, 0.6, 0.8, 0.5, 0.7, 0.9),
        saliency_values=(0.2, 0.2, 0.3, 0.3, 0.6, 0.5, 0.2, 0.4),
        attention_stability_values=(0.7, 0.6, 0.8, 0.5, 0.9, 0.5, 0.7, 0.8),
    )
    projection = project_latent_mask(latent_values, mask_values=(1.0, 0.8, 0.9, 0.7))
    basis = estimate_safe_basis(latent_values, projection, risk_field, basis_rank=4)
    return latent_values, risk_field, basis


@pytest.mark.quick
def test_correct_key_content_score_is_higher_than_wrong_key() -> None:
    """正确 key 的 LF/尾部截断融合分数必须高于错误 key。"""
    _, risk_field, basis = _build_synthetic_basis()
    lf_carrier = derive_lf_carrier(basis, key="correct_key", event_digest="event_unit")
    tail_carrier = derive_tail_carrier(basis, risk_field, key="correct_key", event_digest="event_unit")
    attention_carrier = derive_attention_carrier_stub(basis, key="correct_key", event_digest="event_unit")
    update = compose_latent_update(lf_carrier, tail_carrier, attention_carrier)

    correct_score = compute_content_score(update.combined_update_values, lf_carrier, tail_carrier)
    wrong_lf_carrier = derive_lf_carrier(basis, key="wrong_key", event_digest="event_unit")
    wrong_tail_carrier = derive_tail_carrier(basis, risk_field, key="wrong_key", event_digest="event_unit")
    wrong_score = compute_content_score(update.combined_update_values, wrong_lf_carrier, wrong_tail_carrier)

    assert correct_score.content_score > wrong_score.content_score
    assert correct_score.used_independent_branch_vote is False


@pytest.mark.quick
def test_tail_truncation_changes_tail_score_distribution() -> None:
    """幅值尾部截断必须改变尾部分支的分数行为。"""
    _, risk_field, basis = _build_synthetic_basis()
    truncated_tail = derive_tail_carrier(
        basis,
        risk_field,
        key="correct_key",
        event_digest="event_unit",
        tail_fraction=0.5,
    )
    full_tail = derive_tail_carrier(
        basis,
        risk_field,
        key="correct_key",
        event_digest="event_unit",
        tail_fraction=1.0,
    )
    lf_carrier = derive_lf_carrier(basis, key="correct_key", event_digest="event_unit")
    truncated_score = compute_content_score(truncated_tail.update_values, lf_carrier, truncated_tail)
    full_score = compute_content_score(truncated_tail.update_values, lf_carrier, full_tail)

    assert truncated_tail.retained_fraction < full_tail.retained_fraction
    assert abs(truncated_score.tail_score - full_score.tail_score) > 1e-6


@pytest.mark.quick
def test_geometry_rescue_is_boundary_limited_and_not_direct_positive() -> None:
    """几何链只能在边界失败窗口 rescue, 不能直接判 positive。"""
    geometry = evaluate_geometry_reliability(
        registration_confidence=0.9,
        anchor_inlier_ratio=0.8,
        recovered_sync_consistency=0.85,
        alignment_residual=0.1,
    )
    rescued = decide_evidence_and_final(
        raw_content_score=0.48,
        aligned_content_score=0.53,
        content_threshold=0.50,
        geometry=geometry,
        fail_reason="geometry_suspected",
        attestation_pass=True,
        rescue_margin_low=-0.05,
    )
    blocked = decide_evidence_and_final(
        raw_content_score=0.30,
        aligned_content_score=0.70,
        content_threshold=0.50,
        geometry=geometry,
        fail_reason="geometry_suspected",
        attestation_pass=True,
        rescue_margin_low=-0.05,
    )

    assert geometry.geometry_reliable is True
    assert geometry.direct_positive_decision is False
    assert rescued.rescue_applied is True
    assert rescued.evidence_level is True
    assert blocked.rescue_eligible is False
    assert blocked.evidence_level is False


@pytest.mark.quick
def test_attestation_changes_final_level_not_evidence_level() -> None:
    """Attestation 只能影响 final-level, 不得改变 evidence-level。"""
    geometry = evaluate_geometry_reliability(
        registration_confidence=0.9,
        anchor_inlier_ratio=0.8,
        recovered_sync_consistency=0.85,
        alignment_residual=0.1,
    )
    unattested = decide_evidence_and_final(
        raw_content_score=0.55,
        aligned_content_score=0.55,
        content_threshold=0.50,
        geometry=geometry,
        fail_reason="none",
        attestation_pass=False,
    )
    attested = decide_evidence_and_final(
        raw_content_score=0.55,
        aligned_content_score=0.55,
        content_threshold=0.50,
        geometry=geometry,
        fail_reason="none",
        attestation_pass=True,
    )

    assert unattested.evidence_level is True
    assert attested.evidence_level is True
    assert unattested.final_level is False
    assert attested.final_level is True
