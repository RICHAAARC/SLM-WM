"""验证精确9重复绝对检测统计公式、集合边界与负结果语义."""

from __future__ import annotations

import copy
import random

import pytest

from experiments.protocol.attacks import attack_config_digest, default_attack_configs
from experiments.protocol.formal_randomization import formal_randomization_repeat_ids
from experiments.protocol.pilot_paper_fixed_fpr import (
    bounded_hoeffding_confidence_interval,
)
from main.core.digest import build_stable_digest
from paper_experiments.analysis import randomization_detection_statistics as analysis
from paper_experiments.analysis.randomization_detection_statistics import (
    ATTACKED_FALSE_POSITIVE_SCOPE,
    ATTACKED_TRUE_POSITIVE_SCOPE,
    CLEAN_FALSE_POSITIVE_SCOPE,
    CLEAN_TRUE_POSITIVE_SCOPE,
    DETECTION_METHOD_IDS,
    PROPOSED_METHOD_ID,
    WRONG_KEY_FALSE_POSITIVE_SCOPE,
    RandomizationDetectionStatisticsError,
    build_randomization_detection_statistics,
)


pytestmark = pytest.mark.quick

PAPER_RUN_NAME = "probe_paper"
TARGET_FPR = 0.1
REPEAT_COUNT = len(formal_randomization_repeat_ids())
THRESHOLD_MAP_DIGEST = build_stable_digest({"threshold_map": "exact9"})


def _attack_registry() -> tuple[dict[str, str], ...]:
    """从唯一正式配置构造17项测试攻击 registry."""

    return tuple(
        {
            "attack_id": config.attack_id,
            "attack_family": config.attack_family,
            "attack_name": config.attack_name,
            "resource_profile": config.resource_profile,
            "attack_config_digest": attack_config_digest(config),
        }
        for config in default_attack_configs()
        if config.enabled and config.resource_profile in {"full_main", "full_extra"}
    )


ATTACK_REGISTRY = _attack_registry()


def _cluster_record(
    *,
    method_id: str,
    prompt_id: str,
    scope: str,
    positive_repeat_count: int,
    attack: dict[str, str] | None = None,
    repeat_decision_map: dict[str, bool] | None = None,
    raw_repeat_decision_map: dict[str, bool] | None = None,
) -> dict[str, object]:
    """构造一个可由9个二元判定重建的 Prompt 聚类记录."""

    repeat_ids = formal_randomization_repeat_ids()
    decisions = repeat_decision_map or {
        repeat_id: index < positive_repeat_count
        for index, repeat_id in enumerate(repeat_ids)
    }
    raw_decisions = raw_repeat_decision_map or dict(decisions)
    resolved_positive_count = sum(decisions.values())
    raw_positive_count = sum(raw_decisions.values())
    payload: dict[str, object] = {
        "method_id": method_id,
        "prompt_id": prompt_id,
        "metric_scope": scope,
        "normalized_sample_role": (
            "positive"
            if scope in {CLEAN_TRUE_POSITIVE_SCOPE, ATTACKED_TRUE_POSITIVE_SCOPE}
            else "negative"
        ),
        "attack_id": "" if attack is None else attack["attack_id"],
        "attack_family": "" if attack is None else attack["attack_family"],
        "attack_name": "" if attack is None else attack["attack_name"],
        "resource_profile": "" if attack is None else attack["resource_profile"],
        "attack_config_digest": (
            "" if attack is None else attack["attack_config_digest"]
        ),
        "randomization_repeat_count": REPEAT_COUNT,
        "registered_repeat_decision_map": decisions,
        "registered_repeat_raw_decision_map": raw_decisions,
        "positive_repeat_count": resolved_positive_count,
        "prompt_cluster_positive_rate": resolved_positive_count / REPEAT_COUNT,
        "raw_positive_repeat_count": raw_positive_count,
        "prompt_cluster_raw_positive_rate": raw_positive_count / REPEAT_COUNT,
        "method_repeat_threshold_map_digest": THRESHOLD_MAP_DIGEST,
        "source_outcome_set_digest": build_stable_digest(
            {
                "method_id": method_id,
                "prompt_id": prompt_id,
                "scope": scope,
                "attack_id": "" if attack is None else attack["attack_id"],
                "positive_repeat_count": resolved_positive_count,
                "raw_positive_repeat_count": raw_positive_count,
            }
        ),
    }
    payload["cluster_record_digest"] = build_stable_digest(payload)
    return payload


def _cluster_records(
    *,
    clean_false_positive_counts: dict[str, int] | None = None,
    wrong_key_false_positive_counts: dict[str, int] | None = None,
    attacked_false_positive_counts: dict[str, int] | None = None,
    main_attacked_positive_count: int = REPEAT_COUNT,
) -> list[dict[str, object]]:
    """构造34 Prompt、5方法、17攻击的完整精确集合."""

    clean_fp = clean_false_positive_counts or {}
    wrong_key_fp = wrong_key_false_positive_counts or {}
    attacked_fp = attacked_false_positive_counts or {}
    rows: list[dict[str, object]] = []
    for prompt_index in range(34):
        prompt_id = f"prompt_{prompt_index:03d}"
        for method_id in DETECTION_METHOD_IDS:
            rows.extend(
                (
                    _cluster_record(
                        method_id=method_id,
                        prompt_id=prompt_id,
                        scope=CLEAN_TRUE_POSITIVE_SCOPE,
                        positive_repeat_count=(
                            REPEAT_COUNT if method_id == PROPOSED_METHOD_ID else 0
                        ),
                    ),
                    _cluster_record(
                        method_id=method_id,
                        prompt_id=prompt_id,
                        scope=CLEAN_FALSE_POSITIVE_SCOPE,
                        positive_repeat_count=clean_fp.get(prompt_id, 0),
                    ),
                )
            )
            for attack in ATTACK_REGISTRY:
                rows.extend(
                    (
                        _cluster_record(
                            method_id=method_id,
                            prompt_id=prompt_id,
                            scope=ATTACKED_TRUE_POSITIVE_SCOPE,
                            positive_repeat_count=(
                                main_attacked_positive_count
                                if method_id == PROPOSED_METHOD_ID
                                else 0
                            ),
                            attack=attack,
                        ),
                        _cluster_record(
                            method_id=method_id,
                            prompt_id=prompt_id,
                            scope=ATTACKED_FALSE_POSITIVE_SCOPE,
                            positive_repeat_count=attacked_fp.get(prompt_id, 0),
                            attack=attack,
                        ),
                    )
                )
        rows.append(
            _cluster_record(
                method_id=PROPOSED_METHOD_ID,
                prompt_id=prompt_id,
                scope=WRONG_KEY_FALSE_POSITIVE_SCOPE,
                positive_repeat_count=wrong_key_fp.get(prompt_id, 0),
            )
        )
    return rows


def _statistics(rows: list[dict[str, object]]):
    """调用统一 probe/pilot/full 统计入口."""

    return build_randomization_detection_statistics(
        rows,
        paper_run_name=PAPER_RUN_NAME,
        target_fpr=TARGET_FPR,
        attack_registry_rows=ATTACK_REGISTRY,
    )


def test_exact9_detection_statistics_builds_complete_paper_tables() -> None:
    """完整集合必须形成5/85/1/17行并保留 Prompt 独立单位."""

    operating, per_attack, wrong_key, comparisons, summary = _statistics(
        _cluster_records()
    )

    assert len(operating) == 5
    assert len(per_attack) == 5 * 17
    assert len(wrong_key) == 1
    assert len(comparisons) == 17
    assert summary["prompt_cluster_count"] == 34
    assert summary["randomization_repeat_count"] == 9
    assert summary["attack_count"] == 17
    assert summary["statistical_unit"] == "prompt_cluster"
    assert summary["randomization_detection_statistics_ready"] is True
    assert summary["supports_paper_claim"] is False
    assert summary["per_attack_comparison_family_size"] == 17 * 4
    assert summary["universal_per_attack_superiority_claim_ready"] is True

    slm = next(row for row in operating if row["method_id"] == PROPOSED_METHOD_ID)
    assert slm["clean_positive_observation_count"] == 34 * 9
    assert slm["attacked_positive_observation_count"] == 34 * 9 * 17
    assert slm["clean_true_positive_rate"] == 1.0
    assert slm["attacked_true_positive_rate"] == 1.0
    assert slm["clean_fixed_fpr_ready"] is True
    assert wrong_key[0]["wrong_key_fixed_fpr_ready"] is True
    assert all(row["holm_adjusted_p_value"] < 0.05 for row in comparisons)
    assert all(
        row["mean_paired_difference_simultaneous_ci_low"] > 0.0
        for row in comparisons
    )


def test_raw_content_and_rescue_false_positive_rates_are_both_rebuilt() -> None:
    """最终 FPR 必须同时披露 raw-content FPR 与 rescue 新增误报."""

    rows = _cluster_records()
    target = next(
        row
        for row in rows
        if row["method_id"] == PROPOSED_METHOD_ID
        and row["prompt_id"] == "prompt_000"
        and row["metric_scope"] == CLEAN_FALSE_POSITIVE_SCOPE
    )
    first_repeat = formal_randomization_repeat_ids()[0]
    target["registered_repeat_decision_map"][first_repeat] = True
    target["positive_repeat_count"] = 1
    target["prompt_cluster_positive_rate"] = 1 / 9
    target["cluster_record_digest"] = build_stable_digest(
        {
            key: value
            for key, value in target.items()
            if key != "cluster_record_digest"
        }
    )

    operating, *_ = _statistics(rows)
    slm = next(row for row in operating if row["method_id"] == PROPOSED_METHOD_ID)
    assert slm["clean_false_positive_count"] == 1
    assert slm["clean_raw_false_positive_count"] == 0
    assert slm["clean_rescue_added_false_positive_count"] == 1
    assert slm["clean_rescue_added_false_positive_rate"] == pytest.approx(1 / (34 * 9))


def test_nonselected_baseline_fpr_failure_blocks_universal_claim() -> None:
    """未被展示为最保守者的 baseline FPR 失败也必须阻断普遍优势主张."""

    rows = _cluster_records()
    target_attack_id = ATTACK_REGISTRY[0]["attack_id"]
    targets = [
        row
        for row in rows
        if row["method_id"] == "t2smark"
        and row["prompt_id"] == "prompt_000"
        and (
            row["metric_scope"] == CLEAN_FALSE_POSITIVE_SCOPE
            or (
                row["metric_scope"] == ATTACKED_FALSE_POSITIVE_SCOPE
                and row["attack_id"] == target_attack_id
            )
        )
    ]
    assert len(targets) == 2
    first_repeat = formal_randomization_repeat_ids()[0]
    for target in targets:
        target["registered_repeat_decision_map"][first_repeat] = True
        target["registered_repeat_raw_decision_map"][first_repeat] = True
        target["positive_repeat_count"] = 1
        target["raw_positive_repeat_count"] = 1
        target["prompt_cluster_positive_rate"] = 1 / 9
        target["prompt_cluster_raw_positive_rate"] = 1 / 9
        target["cluster_record_digest"] = build_stable_digest(
            {
                key: value
                for key, value in target.items()
                if key != "cluster_record_digest"
            }
        )

    _operating, _per_attack, _wrong_key, comparisons, summary = _statistics(rows)
    target_comparison = next(
        row for row in comparisons if row["attack_id"] == target_attack_id
    )
    assert target_comparison["most_conservative_baseline_id"] != "t2smark"
    assert target_comparison["all_methods_clean_fixed_fpr_ready"] is False
    assert target_comparison["all_methods_attacked_fixed_fpr_ready"] is False
    assert target_comparison["superiority_claim_ready"] is False
    assert summary["all_methods_clean_fixed_fpr_ready"] is False
    assert summary["all_per_attack_fixed_fpr_ready"] is False
    assert summary["universal_per_attack_superiority_claim_ready"] is False


def test_confidence_interval_uses_prompt_count_not_repeat_expanded_count() -> None:
    """Hoeffding 区间样本量必须是34, 禁止使用306形成伪重复."""

    operating, *_ = _statistics(_cluster_records(main_attacked_positive_count=5))
    slm = next(row for row in operating if row["method_id"] == PROPOSED_METHOD_ID)
    expected = bounded_hoeffding_confidence_interval(5 / 9, 34, 0.95)
    pseudo_replicated = bounded_hoeffding_confidence_interval(5 / 9, 34 * 9, 0.95)

    assert slm["clean_true_positive_rate_ci_low"] == pytest.approx(
        bounded_hoeffding_confidence_interval(1.0, 34, 0.95)[0]
    )
    assert slm["attacked_true_positive_rate_ci_low"] == pytest.approx(expected[0])
    assert slm["attacked_true_positive_rate_ci_high"] == pytest.approx(expected[1])
    assert slm["attacked_true_positive_rate_ci_low"] != pytest.approx(
        pseudo_replicated[0]
    )


def test_repeat_wilson_bound_is_invariant_to_cross_prompt_overlap() -> None:
    """相同逐 repeat 误报计数不得因误报 Prompt 的重叠方式改变门禁."""

    repeat_ids = formal_randomization_repeat_ids()
    overlapping = [
        _cluster_record(
            method_id=PROPOSED_METHOD_ID,
            prompt_id=f"prompt_{index:03d}",
            scope=CLEAN_FALSE_POSITIVE_SCOPE,
            positive_repeat_count=0,
            repeat_decision_map={
                repeat_id: index == 0 for repeat_id in repeat_ids
            },
        )
        for index in range(34)
    ]
    separated = [
        _cluster_record(
            method_id=PROPOSED_METHOD_ID,
            prompt_id=f"prompt_{index:03d}",
            scope=CLEAN_FALSE_POSITIVE_SCOPE,
            positive_repeat_count=0,
            repeat_decision_map={
                repeat_id: index == repeat_index
                for repeat_index, repeat_id in enumerate(repeat_ids)
            },
        )
        for index in range(34)
    ]
    overlapping_bound = analysis._repeat_false_positive_bound_statistics(
        overlapping,
        confidence_level=0.95,
    )
    separated_bound = analysis._repeat_false_positive_bound_statistics(
        separated,
        confidence_level=0.95,
    )

    assert overlapping_bound["false_positive_count_by_repeat"] == {
        repeat_id: 1 for repeat_id in repeat_ids
    }
    assert separated_bound["false_positive_count_by_repeat"] == {
        repeat_id: 1 for repeat_id in repeat_ids
    }
    assert overlapping_bound == separated_bound


def test_probe_zero_cluster_failures_pass_but_one_failure_is_measured_negative() -> None:
    """probe 的0失败通过而1失败不通过, 负结果仍保持统计就绪."""

    zero_operating, *_rest, zero_summary = _statistics(_cluster_records())
    one_operating, one_per_attack, one_wrong_key, _comparisons, one_summary = (
        _statistics(
            _cluster_records(
                clean_false_positive_counts={"prompt_000": 1},
                wrong_key_false_positive_counts={"prompt_000": 1},
                attacked_false_positive_counts={"prompt_000": 1},
            )
        )
    )
    zero_slm = next(
        row for row in zero_operating if row["method_id"] == PROPOSED_METHOD_ID
    )
    one_slm = next(
        row for row in one_operating if row["method_id"] == PROPOSED_METHOD_ID
    )

    assert zero_slm[
        "clean_maximum_repeat_false_positive_rate_upper_bound"
    ] == pytest.approx(
        0.073709, abs=1e-6
    )
    assert zero_slm["clean_fixed_fpr_ready"] is True
    assert one_slm[
        "clean_maximum_repeat_false_positive_rate_upper_bound"
    ] == pytest.approx(
        0.121608, abs=1e-6
    )
    assert one_slm["clean_fixed_fpr_ready"] is False
    assert one_wrong_key[0]["wrong_key_fixed_fpr_ready"] is False
    assert all(row["attacked_fixed_fpr_ready"] is False for row in one_per_attack)
    assert one_summary["randomization_detection_statistics_ready"] is True
    assert one_summary["supports_paper_claim"] is False
    assert zero_summary["randomization_detection_statistics_ready"] is True


def test_exact_set_rejects_baseline_wrong_key_and_missing_attack_cell() -> None:
    """baseline wrong-key 和缺失攻击单元都不能通过 exact-set."""

    rows = _cluster_records()
    baseline_wrong_key = _cluster_record(
        method_id="tree_ring",
        prompt_id="prompt_000",
        scope=WRONG_KEY_FALSE_POSITIVE_SCOPE,
        positive_repeat_count=0,
    )
    with pytest.raises(RandomizationDetectionStatisticsError, match="wrong-key"):
        _statistics([*rows, baseline_wrong_key])

    missing = [
        row
        for row in rows
        if not (
            row["method_id"] == "tree_ring"
            and row["prompt_id"] == "prompt_000"
            and row["metric_scope"] == ATTACKED_TRUE_POSITIVE_SCOPE
            and row["attack_id"] == ATTACK_REGISTRY[0]["attack_id"]
        )
    ]
    with pytest.raises(
        RandomizationDetectionStatisticsError,
        match="未共享同一 test Prompt 集合|精确覆盖",
    ):
        _statistics(missing)


def test_cluster_formula_rejects_repeat_count_and_role_tampering() -> None:
    """伪造9重复计数或交换正负角色必须在统计前失败."""

    rows = _cluster_records()
    repeat_tampered = copy.deepcopy(rows)
    repeat_tampered[0]["randomization_repeat_count"] = 8
    repeat_tampered[0]["cluster_record_digest"] = build_stable_digest(
        {
            key: value
            for key, value in repeat_tampered[0].items()
            if key != "cluster_record_digest"
        }
    )
    with pytest.raises(RandomizationDetectionStatisticsError, match="9个注册重复"):
        _statistics(repeat_tampered)

    role_tampered = copy.deepcopy(rows)
    role_tampered[0]["normalized_sample_role"] = "negative"
    role_tampered[0]["cluster_record_digest"] = build_stable_digest(
        {
            key: value
            for key, value in role_tampered[0].items()
            if key != "cluster_record_digest"
        }
    )
    with pytest.raises(RandomizationDetectionStatisticsError, match="样本极性"):
        _statistics(role_tampered)


def test_input_order_does_not_change_any_detection_statistics() -> None:
    """输入行序不得改变表格、摘要或来源集合摘要."""

    rows = _cluster_records()
    expected = _statistics(rows)
    shuffled = list(rows)
    random.Random(20260713).shuffle(shuffled)

    assert _statistics(shuffled) == expected
