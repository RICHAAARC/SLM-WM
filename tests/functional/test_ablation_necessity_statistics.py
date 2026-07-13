"""验证正式消融逐 Prompt 配对必要性统计。"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.ablations.necessity_statistics import (
    AblationNecessityStatisticsError,
    _cluster_bootstrap_interval,
    _shared_cluster_bootstrap_intervals,
    build_ablation_necessity_statistics,
    build_randomization_aggregate_ablation_necessity_statistics,
)
from experiments.ablations.runtime_rerun import (
    FORMAL_RUNTIME_RERUN_ABLATION_IDS,
)
from experiments.protocol.formal_randomization import (
    formal_randomization_repeat_ids,
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


def _aggregate_records(
    effect_by_repeat_and_prompt: tuple[tuple[float, ...], ...],
) -> list[dict]:
    """把给定的逐 repeat、逐 Prompt 配对效应展开为完整正式记录。"""

    repeat_ids = formal_randomization_repeat_ids()
    assert len(effect_by_repeat_and_prompt) == len(repeat_ids)
    prompt_count = len(effect_by_repeat_and_prompt[0])
    assert prompt_count > 0
    assert all(
        len(prompt_effects) == prompt_count
        for prompt_effects in effect_by_repeat_and_prompt
    )
    rows: list[dict] = []
    for repeat_index, repeat_id in enumerate(repeat_ids):
        for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS:
            for prompt_index in range(prompt_count):
                effect = effect_by_repeat_and_prompt[repeat_index][prompt_index]
                if effect not in (-1.0, 0.0, 1.0):
                    raise AssertionError("测试夹具只接受 -1、0 或1的单元效应")
                complete = ablation_id == "complete_method"
                if effect > 0.0:
                    attacked_positive_rate = 1.0 if complete else 0.0
                    positive_source_positive = complete
                elif effect < 0.0:
                    attacked_positive_rate = 0.0 if complete else 1.0
                    positive_source_positive = not complete
                else:
                    attacked_positive_rate = 0.5
                    positive_source_positive = True
                rows.append(
                    {
                        "randomization_repeat_id": repeat_id,
                        "ablation_id": ablation_id,
                        "prompt_index": prompt_index,
                        "prompt_id": f"prompt_{prompt_index:03d}",
                        "prompt_digest": f"{prompt_index + 1:064x}",
                        "split": "test",
                        "formal_attack_coverage_ready": True,
                        "attacked_positive_rate": attacked_positive_rate,
                        "positive_source_positive": positive_source_positive,
                        "paired_ssim": 0.95,
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
    assert unsupported["necessity_component_decision"] == "measured_not_supported"
    assert unsupported["necessity_component_supported"] is False
    assert first_summary["all_mechanism_necessity_components_supported"] is False
    assert first_summary["necessity_component_not_supported_ablation_ids"] == [
        "without_branch_risk_routing",
        "without_attention_geometry",
    ]
    quality_cost = rows_by_id["without_branch_risk_routing"]
    assert quality_cost["necessity_component_supported"] is False
    assert quality_cost["paired_ssim_noninferiority_ready"] is False
    assert quality_cost["clean_true_positive_mean_paired_effect"] == 1.0
    assert all(row["supports_paper_claim"] is False for row in first_rows)
    assert first_summary["supports_paper_claim"] is False


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


@pytest.mark.quick
def test_aggregate_statistics_average_repeats_before_prompt_inference() -> None:
    """全部9重复必须先在各 Prompt 内求均值, 不能挑选表现最好重复。"""

    effects = (
        *((1.0, 1.0),),
        *(((-1.0, -1.0),) * 8),
    )
    rows, summary = (
        build_randomization_aggregate_ablation_necessity_statistics(
            _aggregate_records(effects),
            expected_ablation_ids=VARIANT_IDS,
            expected_paired_prompt_count=2,
            bootstrap_resample_count=100,
        )
    )

    assert len(rows) == len(VARIANT_IDS)
    assert rows[0]["mean_paired_effect"] == pytest.approx(-7.0 / 9.0)
    assert rows[0]["paired_prompt_count"] == 2
    assert rows[0]["necessity_component_decision"] == "measured_not_supported"
    assert rows[0]["supports_paper_claim"] is False
    assert summary["randomization_repeat_count"] == 9
    assert summary["paired_prompt_count"] == 2
    assert summary["paired_observation_count"] == 18
    assert summary["randomization_aggregate_statistics_ready"] is True
    assert summary["supports_paper_claim"] is False


@pytest.mark.quick
def test_aggregate_statistics_do_not_treat_repeat_cells_as_independent() -> None:
    """Prompt 数而非9倍单元数必须控制区间、检验和顺序不变性。"""

    effects = tuple((1.0, -1.0) for _ in formal_randomization_repeat_ids())
    records = _aggregate_records(effects)
    first_rows, first_summary = (
        build_randomization_aggregate_ablation_necessity_statistics(
            records,
            expected_ablation_ids=VARIANT_IDS,
            expected_paired_prompt_count=2,
            bootstrap_resample_count=500,
        )
    )
    second_rows, second_summary = (
        build_randomization_aggregate_ablation_necessity_statistics(
            reversed(records),
            expected_ablation_ids=VARIANT_IDS,
            expected_paired_prompt_count=2,
            bootstrap_resample_count=500,
        )
    )

    assert first_rows == second_rows
    assert first_summary == second_summary
    assert first_rows[0]["mean_paired_effect"] == 0.0
    assert first_rows[0]["mean_paired_effect_ci_low"] == -1.0
    assert first_rows[0]["mean_paired_effect_ci_high"] == 1.0
    assert first_rows[0]["one_sided_paired_p_value"] == 1.0
    assert first_rows[0]["paired_prompt_count"] == 2


@pytest.mark.quick
def test_aggregate_statistics_require_every_registered_repeat_and_cell() -> None:
    """只给最佳重复或用另一重复的副本补足总行数都必须拒绝。"""

    effects = tuple((1.0, 1.0) for _ in formal_randomization_repeat_ids())
    records = _aggregate_records(effects)
    first_repeat = formal_randomization_repeat_ids()[0]
    selected = [
        record
        for record in records
        if record["randomization_repeat_id"] == first_repeat
    ]
    with pytest.raises(AblationNecessityStatisticsError, match="9个随机重复"):
        build_randomization_aggregate_ablation_necessity_statistics(
            selected,
            expected_ablation_ids=VARIANT_IDS,
            expected_paired_prompt_count=2,
            bootstrap_resample_count=100,
        )

    missing_index = next(
        index
        for index, record in enumerate(records)
        if record["randomization_repeat_id"] == first_repeat
        and record["ablation_id"] == VARIANT_IDS[0]
        and record["prompt_id"] == "prompt_000"
    )
    duplicated_index = next(
        index
        for index, record in enumerate(records)
        if record["randomization_repeat_id"]
        == formal_randomization_repeat_ids()[1]
        and record["ablation_id"] == VARIANT_IDS[0]
        and record["prompt_id"] == "prompt_000"
    )
    same_total_count = list(records)
    same_total_count[missing_index] = dict(records[duplicated_index])
    with pytest.raises(
        AblationNecessityStatisticsError,
        match="重复 Prompt|同一非空 test Prompt",
    ):
        build_randomization_aggregate_ablation_necessity_statistics(
            same_total_count,
            expected_ablation_ids=VARIANT_IDS,
            expected_paired_prompt_count=2,
            bootstrap_resample_count=100,
        )


@pytest.mark.quick
def test_aggregate_statistics_reject_prompt_identity_drift() -> None:
    """同一 Prompt 在任一重复中的索引或摘要漂移都不得进入聚合。"""

    effects = tuple((1.0, 1.0) for _ in formal_randomization_repeat_ids())
    records = _aggregate_records(effects)
    target = next(
        record
        for record in records
        if record["randomization_repeat_id"]
        == formal_randomization_repeat_ids()[-1]
        and record["prompt_id"] == "prompt_000"
    )
    target["prompt_digest"] = "f" * 64

    with pytest.raises(AblationNecessityStatisticsError, match="Prompt 身份"):
        build_randomization_aggregate_ablation_necessity_statistics(
            records,
            expected_ablation_ids=VARIANT_IDS,
            expected_paired_prompt_count=2,
            bootstrap_resample_count=100,
        )
