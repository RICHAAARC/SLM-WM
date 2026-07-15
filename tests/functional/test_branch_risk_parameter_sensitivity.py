"""验证单模型内部风险参数敏感性协议。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.ablations.branch_risk_sensitivity import (
    FORMAL_BRANCH_RISK_SENSITIVITY_IDS,
    FORMAL_BRANCH_RISK_SENSITIVITY_SPECS,
    branch_risk_sensitivity_contract,
)
from experiments.ablations.branch_risk_sensitivity_runtime import (
    _sensitivity_metric_row,
)
from experiments.runners.semantic_watermark_runtime import SemanticWatermarkRuntimeConfig


@pytest.mark.quick
def test_branch_risk_sensitivity_uses_exact_one_parameter_protocol() -> None:
    """正式集合必须覆盖参考设置与全部登记的数值参数变化。"""

    contract = branch_risk_sensitivity_contract(
        FORMAL_BRANCH_RISK_SENSITIVITY_SPECS
    )

    assert contract["sensitivity_exact_set_ready"] is True
    assert contract["sensitivity_setting_count"] == 18
    assert FORMAL_BRANCH_RISK_SENSITIVITY_IDS[0] == "formal_reference"
    assert contract["cross_model_evidence_provided"] is False
    assert contract["supports_paper_claim"] is False


@pytest.mark.quick
def test_branch_risk_sensitivity_changes_only_registered_risk_field() -> None:
    """单个设置只能改变三个分支中的同名风险参数。"""

    base = SemanticWatermarkRuntimeConfig()
    spec = next(
        item
        for item in FORMAL_BRANCH_RISK_SENSITIVITY_SPECS
        if item.sensitivity_id == "semantic_weight_high"
    )
    changed = spec.apply(base, "outputs/formal_branch_risk_sensitivity")

    assert changed.risk_parameter_protocol == "single_model_internal_sensitivity"
    assert changed.semantic_routing_enabled == base.semantic_routing_enabled
    assert changed.model_id == base.model_id
    assert changed.seed == base.seed
    assert changed.lf_content_risk_config.semantic_weight == pytest.approx(
        base.lf_content_risk_config.semantic_weight * 1.5
    )
    assert (
        changed.lf_content_risk_config.local_contrast_risk_weight
        == base.lf_content_risk_config.local_contrast_risk_weight
    )


@pytest.mark.quick
def test_runtime_config_rejects_unscoped_risk_parameter_change() -> None:
    """普通方法运行不得绕过正式 YAML 参数锁。"""

    base = SemanticWatermarkRuntimeConfig()
    changed_lf = replace(
        base.lf_content_risk_config,
        semantic_weight=base.lf_content_risk_config.semantic_weight * 1.5,
    )
    with pytest.raises(ValueError, match="必须精确继承"):
        replace(base, lf_content_risk_config=changed_lf)


@pytest.mark.quick
def test_reference_sensitivity_keeps_formal_risk_parameters() -> None:
    """参考设置只改变实验身份, 不改变任何风险函数数值。"""

    base = SemanticWatermarkRuntimeConfig()
    reference = FORMAL_BRANCH_RISK_SENSITIVITY_SPECS[0].apply(
        base,
        "outputs/formal_branch_risk_sensitivity",
    )

    assert reference.lf_content_risk_config == base.lf_content_risk_config
    assert reference.tail_robust_risk_config == base.tail_robust_risk_config
    assert (
        reference.attention_geometry_risk_config
        == base.attention_geometry_risk_config
    )


@pytest.mark.quick
def test_sensitivity_metric_row_discloses_prompt_level_confidence_intervals() -> None:
    """每个设置必须披露样本数与有界置信区间, 不能只报告点估计。"""

    records = [
        {
            "clean_negative_positive": False,
            "wrong_key_negative_positive": False,
            "positive_source_positive": True,
            "attacked_positive_rate": 0.75,
            "attacked_negative_rate": 0.0,
            "paired_ssim": 0.90,
        },
        {
            "clean_negative_positive": False,
            "wrong_key_negative_positive": True,
            "positive_source_positive": True,
            "attacked_positive_rate": 0.25,
            "attacked_negative_rate": 0.0,
            "paired_ssim": 0.80,
        },
    ]

    row = _sensitivity_metric_row("semantic_weight_low", records, "d" * 64)

    assert row["test_prompt_count"] == 2
    assert row["attacked_true_positive_rate"] == pytest.approx(0.5)
    assert row["paired_ssim_mean"] == pytest.approx(0.85)
    assert row["attacked_true_positive_rate_sample_count"] == 2
    assert 0.0 <= row["attacked_true_positive_rate_ci_low"] <= 0.5
    assert 0.5 <= row["attacked_true_positive_rate_ci_high"] <= 1.0
    assert row["confidence_interval_method"] == "bounded_hoeffding"
