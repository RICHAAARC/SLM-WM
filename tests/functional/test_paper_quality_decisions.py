"""论文质量语义、Prompt 聚类推断和三态组合测试。"""

from __future__ import annotations

import pytest

from paper_experiments.analysis.paper_quality_decisions import (
    build_prompt_cluster_mean_inference,
    build_quality_preservation_decisions,
    load_paper_quality_claim_protocol,
)


pytestmark = pytest.mark.quick


def test_quality_protocol_separates_distribution_shift_from_reference_quality() -> None:
    """clean-watermarked FID/KID 不得冒充相对真实参考分布的生成质量。"""

    protocol = load_paper_quality_claim_protocol()
    distribution = protocol["distributional_preservation_noninferiority"]

    assert protocol["primary_sampling_unit"] == "prompt"
    assert protocol["nested_sampling_unit"] == (
        "registered_randomization_repeat_within_prompt"
    )
    assert distribution["clean_watermarked_comparison_interpretation"] == (
        "distributional_preservation_only"
    )
    assert distribution["fid_evidence_role"] == "descriptive"
    assert (
        distribution[
            "generation_quality_against_real_reference_allowed_without_reference"
        ]
        is False
    )


def test_prompt_cluster_inference_is_order_invariant_and_reproducible() -> None:
    """重采样单位必须是唯一 Prompt, 输入映射顺序不得改变区间。"""

    values = {f"prompt_{index:03d}": index / 1000.0 for index in range(20)}
    forward = build_prompt_cluster_mean_inference(
        values,
        analysis_id="paired_perceptual_quality_noninferiority",
    )
    reverse = build_prompt_cluster_mean_inference(
        dict(reversed(tuple(values.items()))),
        analysis_id="paired_perceptual_quality_noninferiority",
    )

    assert forward == reverse
    assert forward["prompt_count"] == 20
    assert forward["primary_sampling_unit"] == "prompt"
    assert forward["confidence_interval_low"] <= forward["estimate"]
    assert forward["estimate"] <= forward["confidence_interval_high"]


def test_quality_decisions_keep_missing_and_negative_evidence_distinct() -> None:
    """分布指标超界是测得不支持, 缺少感知和语义证据仍是证据不完整。"""

    inference = build_prompt_cluster_mean_inference(
        {f"prompt_{index:03d}": 0.01 for index in range(20)},
        analysis_id="distributional_preservation_prompt_conditional_kid",
    )
    decisions = build_quality_preservation_decisions(
        distributional_inference=inference,
        evidence_artifact_id="quality_manifest",
    )

    assert decisions["quality_subclaim_decisions"][
        "distributional_preservation_noninferiority"
    ]["decision"] == "measured_not_supported"
    assert decisions["quality_subclaim_decisions"][
        "paired_perceptual_quality_noninferiority"
    ]["decision"] == "evidence_incomplete"
    assert decisions["cross_attack_quality_decision"]["decision"] == (
        "evidence_incomplete"
    )
    assert set(decisions["per_attack_quality_decisions"]) == set(
        decisions["paper_quality_claim_protocol"]["registered_attack_ids"]
    )
    assert all(
        record["decision"] == "evidence_incomplete"
        for record in decisions["per_attack_quality_decisions"].values()
    )
    assert decisions["quality_preservation_claim_decision"]["decision"] == (
        "evidence_incomplete"
    )


def test_complete_quality_components_can_support_without_workflow_shortcut() -> None:
    """只有三类子主张和每项攻击都完整支持时总体质量主张才成立。"""

    protocol = load_paper_quality_claim_protocol()
    distribution = build_prompt_cluster_mean_inference(
        {f"prompt_{index:03d}": 0.0 for index in range(20)},
        analysis_id="distributional_preservation_prompt_conditional_kid",
        protocol=protocol,
    )
    perceptual = build_prompt_cluster_mean_inference(
        {f"prompt_{index:03d}": 1.0 for index in range(20)},
        analysis_id="paired_perceptual_quality_noninferiority",
        protocol=protocol,
    )
    semantic = build_prompt_cluster_mean_inference(
        {f"prompt_{index:03d}": 1.0 for index in range(20)},
        analysis_id="independent_visual_content_preservation_noninferiority",
        protocol=protocol,
    )
    decisions = build_quality_preservation_decisions(
        distributional_inference=distribution,
        paired_perceptual_inference=perceptual,
        independent_visual_content_inference=semantic,
        per_attack_inference={
            attack_id: {
                "paired_perceptual_quality_noninferiority": perceptual,
                "independent_visual_content_preservation_noninferiority": semantic,
                "distributional_preservation_noninferiority": distribution,
            }
            for attack_id in protocol["registered_attack_ids"]
        },
        evidence_artifact_id="quality_manifest",
        protocol=protocol,
    )

    assert decisions["cross_attack_quality_decision"]["decision"] == "supported"
    assert decisions["quality_preservation_claim_decision"]["decision"] == (
        "supported"
    )
