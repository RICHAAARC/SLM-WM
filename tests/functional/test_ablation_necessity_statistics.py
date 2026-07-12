"""验证正式消融逐 Prompt 配对必要性统计。"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.ablations.necessity_statistics import (
    AblationNecessityStatisticsError,
    _cluster_bootstrap_interval,
    _shared_cluster_bootstrap_intervals,
    build_ablation_necessity_statistics,
)
from experiments.ablations.runtime_rerun import (
    FORMAL_RUNTIME_RERUN_ABLATION_IDS,
)


VARIANT_IDS = tuple(
    ablation_id
    for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
    if ablation_id != "complete_method"
)


def test_shared_bootstrap_reuses_indices_without_changing_intervals() -> None:
    """共享 Prompt 索引优化必须与三次独立同 seed 计算逐项完全一致。"""

    effect_groups = {
        "attack": np.asarray((0.0, 0.2, 0.4, 0.1, 0.3), dtype=np.float64),
        "clean": np.asarray((0.0, 1.0, 0.0, 1.0, 0.0), dtype=np.float64),
        "ssim": np.asarray((0.01, -0.01, 0.02, 0.0, 0.03), dtype=np.float64),
    }
    shared = _shared_cluster_bootstrap_intervals(
        effect_groups,
        confidence_level=0.95,
        resample_count=500,
        seed=1703,
    )

    for field_name, values in effect_groups.items():
        expected = _cluster_bootstrap_interval(
            values,
            confidence_level=0.95,
            resample_count=500,
            seed=1703,
        )
        assert shared[field_name] == expected


def _records() -> list[dict]:
    """构造含一个真实负结论和一个质量代价诊断的配对记录。"""

    rows = []
    for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS:
        for prompt_index in range(40):
            measured_not_supported = (
                ablation_id == "without_attention_geometry"
            )
            rows.append(
                {
                    "ablation_id": ablation_id,
                    "prompt_id": f"prompt_{prompt_index:03d}",
                    "split": "test",
                    "formal_attack_coverage_ready": True,
                    "attacked_positive_rate": (
                        1.0
                        if ablation_id == "complete_method"
                        or measured_not_supported
                        else 0.0
                    ),
                    "positive_source_positive": (
                        ablation_id == "complete_method"
                    ),
                    "paired_ssim": (
                        0.95
                        if ablation_id == "without_branch_risk_routing"
                        else 0.90
                    ),
                }
            )
    return rows


@pytest.mark.quick
def test_paired_statistics_are_deterministic_and_honest() -> None:
    """主指标统计与质量非劣门禁共同决定必要性, 诊断量必须完整披露。"""

    first_rows, first_summary = build_ablation_necessity_statistics(
        _records(),
        expected_ablation_ids=VARIANT_IDS,
        expected_paired_prompt_count=40,
        bootstrap_resample_count=2000,
    )
    second_rows, second_summary = build_ablation_necessity_statistics(
        reversed(_records()),
        expected_ablation_ids=VARIANT_IDS,
        expected_paired_prompt_count=40,
        bootstrap_resample_count=2000,
    )

    assert first_rows == second_rows
    assert first_summary == second_summary
    rows_by_id = {row["ablation_id"]: row for row in first_rows}
    unsupported = rows_by_id["without_attention_geometry"]
    assert unsupported["necessity_claim_decision"] == "measured_not_supported"
    assert unsupported["necessity_claim_supported"] is False
    assert first_summary["all_mechanism_necessity_claims_supported"] is False
    assert first_summary["necessity_not_supported_ablation_ids"] == [
        "without_branch_risk_routing",
        "without_attention_geometry",
    ]
    quality_cost = rows_by_id["without_branch_risk_routing"]
    assert quality_cost["necessity_claim_supported"] is False
    assert quality_cost["paired_ssim_noninferiority_ready"] is False
    assert quality_cost["clean_true_positive_mean_paired_effect"] == 1.0


@pytest.mark.quick
def test_paired_statistics_reject_missing_or_duplicate_prompt_pair() -> None:
    """缺失或重复任一 Prompt 配对应 fail-closed。"""

    records = _records()
    missing = records[:-1]
    with pytest.raises(AblationNecessityStatisticsError, match="同一非空 test Prompt"):
        build_ablation_necessity_statistics(
            missing,
            expected_ablation_ids=VARIANT_IDS,
            expected_paired_prompt_count=40,
            bootstrap_resample_count=100,
        )

    with pytest.raises(AblationNecessityStatisticsError, match="重复 Prompt"):
        build_ablation_necessity_statistics(
            [*records, dict(records[0])],
            expected_ablation_ids=VARIANT_IDS,
            expected_paired_prompt_count=40,
            bootstrap_resample_count=100,
        )


@pytest.mark.quick
def test_paired_statistics_enforce_current_test_scale() -> None:
    """配对数量不得小于或大于当前 paper_run 的 test 规模。"""

    with pytest.raises(AblationNecessityStatisticsError, match="paper_run test"):
        build_ablation_necessity_statistics(
            _records(),
            expected_ablation_ids=VARIANT_IDS,
            expected_paired_prompt_count=34,
            bootstrap_resample_count=100,
        )


@pytest.mark.quick
def test_paired_statistics_reject_string_boolean_outcome() -> None:
    """字符串形式的真假值不得冒充逐 Prompt 真实检测结果。"""

    records = _records()
    records[0]["positive_source_positive"] = "False"

    with pytest.raises(AblationNecessityStatisticsError, match="布尔检测结果"):
        build_ablation_necessity_statistics(
            records,
            expected_ablation_ids=VARIANT_IDS,
            expected_paired_prompt_count=40,
            bootstrap_resample_count=100,
        )
