"""验证精确9重复的单模型参数敏感性统计。"""

from __future__ import annotations

import pytest

from experiments.ablations.branch_risk_sensitivity import (
    FORMAL_BRANCH_RISK_SENSITIVITY_IDS,
)
from experiments.protocol.formal_randomization import formal_randomization_repeat_ids
from paper_experiments.runners.randomization_parameter_sensitivity import (
    _aggregate_metric_rows,
)


@pytest.mark.quick
def test_parameter_sensitivity_aggregate_uses_registered_repeat_means() -> None:
    """聚合区间必须以9个注册 repeat 为单位, 不能把 prompt 伪装成独立重复。"""

    repeat_rows = tuple(
        {
            "randomization_repeat_id": repeat_id,
            "sensitivity_id": sensitivity_id,
            "test_prompt_count": 28,
            "clean_false_positive_rate": 0.0,
            "wrong_key_false_positive_rate": 0.0,
            "clean_true_positive_rate": 0.9,
            "attacked_true_positive_rate": (
                0.8
                if sensitivity_id == "formal_reference"
                else 0.7
            ),
            "attacked_false_positive_rate": 0.0,
            "paired_ssim_mean": 0.95,
        }
        for repeat_id in formal_randomization_repeat_ids()
        for sensitivity_id in FORMAL_BRANCH_RISK_SENSITIVITY_IDS
    )

    rows = _aggregate_metric_rows(repeat_rows)

    assert len(rows) == 18
    changed = next(
        row for row in rows if row["sensitivity_id"] == "semantic_weight_low"
    )
    assert changed["randomization_repeat_count"] == 9
    assert changed["attacked_true_positive_rate"] == pytest.approx(0.7)
    assert changed["attacked_true_positive_rate_delta"] == pytest.approx(-0.1)
    assert (
        changed["attacked_true_positive_rate_delta_ci_low"]
        <= changed["attacked_true_positive_rate_delta"]
        <= changed["attacked_true_positive_rate_delta_ci_high"]
    )
    assert changed["confidence_interval_method"] == (
        "bounded_hoeffding_repeat_mean"
    )


@pytest.mark.quick
def test_parameter_sensitivity_reference_has_no_self_delta() -> None:
    """参考设置只报告绝对指标, 不生成没有信息量的自比较差值。"""

    repeat_rows = tuple(
        {
            "randomization_repeat_id": repeat_id,
            "sensitivity_id": sensitivity_id,
            "clean_false_positive_rate": 0.0,
            "wrong_key_false_positive_rate": 0.0,
            "clean_true_positive_rate": 1.0,
            "attacked_true_positive_rate": 0.75,
            "attacked_false_positive_rate": 0.0,
            "paired_ssim_mean": 0.9,
        }
        for repeat_id in formal_randomization_repeat_ids()
        for sensitivity_id in FORMAL_BRANCH_RISK_SENSITIVITY_IDS
    )

    reference = _aggregate_metric_rows(repeat_rows)[0]

    assert reference["sensitivity_id"] == "formal_reference"
    assert "attacked_true_positive_rate_delta" not in reference
