"""构造 SLM-WM 核心方法的 synthetic smoke 场景。

本模块只组织 `algorithm_primitives` 中已经冻结的纯算法原语, 不写文件、不接入
外部模型、不访问 Notebook 或远程运行环境。它的作用是提供一个可由脚本和测试
共同调用的最小闭环, 使业务路径集中呈现核心方法链路。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from main.core.digest import build_stable_digest
from main.methods.algorithm_primitives import (
    CarrierPrimitive,
    ContentScoreResult,
    GeometryReliabilityResult,
    build_semantic_risk_field,
    compose_latent_update,
    compute_content_score,
    decide_evidence_and_final,
    derive_attention_carrier_stub,
    derive_hf_carrier,
    derive_lf_carrier,
    estimate_safe_basis,
    evaluate_geometry_reliability,
    project_latent_mask,
)


SMOKE_STAGE_NAME = "stage_02_core_method_smoke_test"
CONTENT_THRESHOLD = 0.61
RESCUE_MARGIN_LOW = -0.05


@dataclass(frozen=True)
class CoreSmokeScenario:
    """单个 synthetic latent 场景的判定结果。"""

    scenario_id: str
    scenario_role: str
    observed_digest: str
    content_score: float
    aligned_content_score: float
    score_margin: float
    positive_by_content: bool
    rescue_eligible: bool
    rescue_applied: bool
    evidence_decision: bool
    final_decision: bool
    final_label: str
    geometry_reliable: bool
    attestation_pass: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化字典, 供脚本写出 JSONL records。"""
        return asdict(self)


@dataclass(frozen=True)
class CoreSmokeBundle:
    """stage02 synthetic smoke 的完整内存结果。"""

    scenarios: tuple[CoreSmokeScenario, ...]
    metrics: dict[str, Any]
    carrier_digests: dict[str, str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化字典, 供测试或脚本复用。"""
        return asdict(self)


def _vector_digest(values: tuple[float, ...]) -> str:
    """为 synthetic latent 向量生成稳定摘要。"""
    return build_stable_digest([round(value, 12) for value in values])


def _interpolate_vectors(left: tuple[float, ...], right: tuple[float, ...], ratio: float) -> tuple[float, ...]:
    """按比例插值两个等长向量。"""
    return tuple((1.0 - ratio) * left_value + ratio * right_value for left_value, right_value in zip(left, right))


def _build_base_carriers() -> dict[str, Any]:
    """构造 stage02 smoke 复用的 synthetic latent、基底和载体。"""
    latent_values = (0.2, -0.1, 0.4, -0.3, 0.6, -0.2, 0.1, -0.5)
    risk_field = build_semantic_risk_field(
        semantic_values=(0.2, 0.3, 0.4, 0.5, 0.7, 0.6, 0.2, 0.3),
        texture_values=(0.2, 0.3, 0.7, 0.8, 0.9, 0.6, 0.2, 0.7),
        stability_values=(0.8, 0.7, 0.9, 0.6, 0.8, 0.5, 0.7, 0.9),
        saliency_values=(0.2, 0.2, 0.3, 0.3, 0.6, 0.5, 0.2, 0.4),
        attention_stability_values=(0.7, 0.6, 0.8, 0.5, 0.9, 0.5, 0.7, 0.8),
    )
    projection = project_latent_mask(latent_values, mask_values=(1.0, 0.8, 0.9, 0.7))
    safe_basis = estimate_safe_basis(latent_values, projection, risk_field, basis_rank=4)
    event_digest = build_stable_digest({"stage_name": SMOKE_STAGE_NAME, "event_name": "synthetic_smoke_event"})
    lf_carrier = derive_lf_carrier(safe_basis, key="correct_key", event_digest=event_digest)
    hf_carrier = derive_hf_carrier(safe_basis, risk_field, key="correct_key", event_digest=event_digest)
    attention_carrier = derive_attention_carrier_stub(safe_basis, key="correct_key", event_digest=event_digest)
    wrong_lf_carrier = derive_lf_carrier(safe_basis, key="wrong_key", event_digest=event_digest)
    wrong_hf_carrier = derive_hf_carrier(safe_basis, risk_field, key="wrong_key", event_digest=event_digest)
    update = compose_latent_update(lf_carrier, hf_carrier, attention_carrier)
    wrong_update = compose_latent_update(wrong_lf_carrier, wrong_hf_carrier, attention_carrier)
    return {
        "latent_values": latent_values,
        "lf_carrier": lf_carrier,
        "hf_carrier": hf_carrier,
        "attention_carrier": attention_carrier,
        "wrong_lf_carrier": wrong_lf_carrier,
        "wrong_hf_carrier": wrong_hf_carrier,
        "watermarked_values": update.combined_update_values,
        "wrong_key_values": wrong_update.combined_update_values,
        "event_digest": event_digest,
    }


def _score_observed(
    observed_values: tuple[float, ...],
    lf_carrier: CarrierPrimitive,
    hf_carrier: CarrierPrimitive,
) -> ContentScoreResult:
    """计算 smoke 使用的内容分数。"""
    return compute_content_score(observed_values, lf_carrier, hf_carrier)


def _build_boundary_shifted_values(
    watermarked_values: tuple[float, ...],
    low_reference_values: tuple[float, ...],
    lf_carrier: CarrierPrimitive,
    hf_carrier: CarrierPrimitive,
    target_score: float | None = None,
) -> tuple[float, ...]:
    """构造落在 rescue 边界窗口内的 synthetic 几何失配向量。"""
    target = CONTENT_THRESHOLD - 0.02 if target_score is None else target_score
    low_score = _score_observed(low_reference_values, lf_carrier, hf_carrier).content_score
    high_score = _score_observed(watermarked_values, lf_carrier, hf_carrier).content_score
    lower_bound = min(low_score, high_score) + 1e-6
    upper_bound = max(low_score, high_score) - 1e-6
    bounded_target = min(max(target, lower_bound), upper_bound)
    low = 0.0
    high = 1.0
    for _ in range(40):
        middle = (low + high) / 2.0
        candidate = _interpolate_vectors(low_reference_values, watermarked_values, middle)
        score = _score_observed(candidate, lf_carrier, hf_carrier).content_score
        if score < bounded_target:
            low = middle
        else:
            high = middle
    return _interpolate_vectors(low_reference_values, watermarked_values, high)


def _make_scenario(
    scenario_id: str,
    scenario_role: str,
    observed_values: tuple[float, ...],
    aligned_values: tuple[float, ...],
    lf_carrier: CarrierPrimitive,
    hf_carrier: CarrierPrimitive,
    geometry: GeometryReliabilityResult,
    fail_reason: str,
    attestation_pass: bool,
    metadata: dict[str, Any] | None = None,
) -> CoreSmokeScenario:
    """根据 synthetic latent 和统一决策规则构造场景结果。"""
    raw_score = _score_observed(observed_values, lf_carrier, hf_carrier)
    aligned_score = _score_observed(aligned_values, lf_carrier, hf_carrier)
    decision = decide_evidence_and_final(
        raw_content_score=raw_score.content_score,
        aligned_content_score=aligned_score.content_score,
        content_threshold=CONTENT_THRESHOLD,
        geometry=geometry,
        fail_reason=fail_reason,
        attestation_pass=attestation_pass,
        rescue_margin_low=RESCUE_MARGIN_LOW,
    )
    return CoreSmokeScenario(
        scenario_id=scenario_id,
        scenario_role=scenario_role,
        observed_digest=_vector_digest(observed_values),
        content_score=raw_score.content_score,
        aligned_content_score=aligned_score.content_score,
        score_margin=decision.raw_content_margin,
        positive_by_content=decision.positive_by_content,
        rescue_eligible=decision.rescue_eligible,
        rescue_applied=decision.rescue_applied,
        evidence_decision=decision.evidence_level,
        final_decision=decision.final_level,
        final_label=decision.final_label,
        geometry_reliable=geometry.geometry_reliable,
        attestation_pass=attestation_pass,
        metadata=metadata or {},
    )


def build_core_method_smoke_bundle() -> CoreSmokeBundle:
    """构造 stage02 核心方法最小闭环 smoke 结果。"""
    carrier_bundle = _build_base_carriers()
    latent_values = carrier_bundle["latent_values"]
    watermarked_values = carrier_bundle["watermarked_values"]
    wrong_key_values = carrier_bundle["wrong_key_values"]
    lf_carrier = carrier_bundle["lf_carrier"]
    hf_carrier = carrier_bundle["hf_carrier"]
    wrong_lf_carrier = carrier_bundle["wrong_lf_carrier"]
    wrong_hf_carrier = carrier_bundle["wrong_hf_carrier"]
    shifted_values = _build_boundary_shifted_values(watermarked_values, latent_values, lf_carrier, hf_carrier)

    reliable_geometry = evaluate_geometry_reliability(
        registration_confidence=0.9,
        anchor_inlier_ratio=0.8,
        recovered_sync_consistency=0.85,
        alignment_residual=0.1,
    )
    unreliable_geometry = evaluate_geometry_reliability(
        registration_confidence=0.4,
        anchor_inlier_ratio=0.3,
        recovered_sync_consistency=0.4,
        alignment_residual=0.8,
    )

    scenarios = (
        _make_scenario(
            "clean_synthetic_latent",
            "clean_negative",
            latent_values,
            latent_values,
            lf_carrier,
            hf_carrier,
            unreliable_geometry,
            "none",
            False,
        ),
        _make_scenario(
            "watermarked_synthetic_latent",
            "watermarked_positive",
            watermarked_values,
            watermarked_values,
            lf_carrier,
            hf_carrier,
            reliable_geometry,
            "none",
            True,
        ),
        _make_scenario(
            "wrong_key_negative",
            "wrong_key_negative",
            watermarked_values,
            watermarked_values,
            wrong_lf_carrier,
            wrong_hf_carrier,
            reliable_geometry,
            "none",
            True,
        ),
        _make_scenario(
            "geometric_shifted_latent",
            "geometric_boundary_failure",
            shifted_values,
            shifted_values,
            lf_carrier,
            hf_carrier,
            reliable_geometry,
            "geometry_suspected",
            True,
        ),
        _make_scenario(
            "aligned_recovered_latent",
            "same_threshold_rescue",
            shifted_values,
            watermarked_values,
            lf_carrier,
            hf_carrier,
            reliable_geometry,
            "geometry_suspected",
            True,
        ),
        _make_scenario(
            "unreliable_geometry_shifted_latent",
            "rescue_blocked_by_geometry",
            shifted_values,
            watermarked_values,
            lf_carrier,
            hf_carrier,
            unreliable_geometry,
            "geometry_suspected",
            True,
        ),
        _make_scenario(
            "unattested_positive",
            "evidence_positive_but_unattested",
            watermarked_values,
            watermarked_values,
            lf_carrier,
            hf_carrier,
            reliable_geometry,
            "none",
            False,
        ),
        _make_scenario(
            "final_positive",
            "final_positive",
            watermarked_values,
            watermarked_values,
            lf_carrier,
            hf_carrier,
            reliable_geometry,
            "none",
            True,
        ),
    )

    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    rescue_trigger_rate = sum(1 for scenario in scenarios if scenario.rescue_applied) / len(scenarios)
    final_positive_count = sum(1 for scenario in scenarios if scenario.final_decision)
    evidence_positive_count = sum(1 for scenario in scenarios if scenario.evidence_decision)
    metrics = {
        "content_threshold": CONTENT_THRESHOLD,
        "rescue_margin_low": RESCUE_MARGIN_LOW,
        "key_separation_margin": (
            scenario_by_id["watermarked_synthetic_latent"].content_score
            - scenario_by_id["wrong_key_negative"].content_score
        ),
        "score_margin_min": min(scenario.score_margin for scenario in scenarios),
        "rescue_trigger_rate": rescue_trigger_rate,
        "wrong_key_over_threshold": scenario_by_id["wrong_key_negative"].positive_by_content,
        "geometry_unreliable_rescue_blocked": (
            scenario_by_id["unreliable_geometry_shifted_latent"].rescue_applied is False
        ),
        "attestation_layering_pass": (
            scenario_by_id["unattested_positive"].evidence_decision
            and not scenario_by_id["unattested_positive"].final_decision
        ),
        "final_positive_count": final_positive_count,
        "evidence_positive_count": evidence_positive_count,
    }
    return CoreSmokeBundle(
        scenarios=scenarios,
        metrics=metrics,
        carrier_digests={
            "lf_carrier_digest": lf_carrier.carrier_digest,
            "hf_carrier_digest": hf_carrier.carrier_digest,
            "attention_carrier_digest": carrier_bundle["attention_carrier"].carrier_digest,
            "event_digest": carrier_bundle["event_digest"],
        },
        metadata={
            "stage_name": SMOKE_STAGE_NAME,
            "attention_runtime": "not_connected",
            "records_are_synthetic": True,
        },
    )
