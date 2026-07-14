"""重建注册9重复上的绝对检测率与逐攻击检测率.

该模块只消费已经按 Prompt 聚合9个注册重复的检测判定记录. 每个 Prompt
始终是唯一独立统计单位; 重复与攻击只在 Prompt 内平均, 不会把样本量错误
扩大为 ``9T`` 或 ``9AT``. 数值是否达到目标与统计输入是否完整严格分离,
因此真实负结果仍能形成完整论文证据.
"""

from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Any, Iterable, Mapping

import numpy as np

from experiments.protocol.calibration import binomial_rate_upper_confidence_bound
from experiments.protocol.formal_randomization import formal_randomization_repeat_ids
from experiments.protocol.paper_run_config import (
    RUN_EXPECTED_PROMPT_COUNTS,
    normalize_paper_run_name,
    validate_frozen_paper_run_target_fpr,
)
from experiments.protocol.pilot_paper_fixed_fpr import (
    PILOT_PAPER_CONFIDENCE_LEVEL,
    bounded_hoeffding_confidence_interval,
)
from experiments.protocol.splits import build_group_split_counts
from main.core.digest import build_stable_digest
from paper_experiments.analysis.paired_superiority import (
    PRIMARY_BASELINE_IDS,
    canonical_attack_registry_rows,
)


PROPOSED_METHOD_ID = "slm_wm"
DETECTION_METHOD_IDS = (PROPOSED_METHOD_ID, *PRIMARY_BASELINE_IDS)
RANDOMIZATION_DETECTION_STATISTICAL_UNIT = "prompt_cluster"
RANDOMIZATION_DETECTION_CONFIDENCE_INTERVAL_METHOD = "bounded_hoeffding"
RANDOMIZATION_DETECTION_FALSE_POSITIVE_BOUND_METHOD = (
    "one_sided_wilson_per_registered_repeat_maximum"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

CLEAN_TRUE_POSITIVE_SCOPE = "clean_true_positive"
CLEAN_FALSE_POSITIVE_SCOPE = "clean_false_positive"
WRONG_KEY_FALSE_POSITIVE_SCOPE = "wrong_key_false_positive"
ATTACKED_TRUE_POSITIVE_SCOPE = "attacked_true_positive"
ATTACKED_FALSE_POSITIVE_SCOPE = "attacked_false_positive"
DETECTION_METRIC_SCOPES = (
    CLEAN_TRUE_POSITIVE_SCOPE,
    CLEAN_FALSE_POSITIVE_SCOPE,
    WRONG_KEY_FALSE_POSITIVE_SCOPE,
    ATTACKED_TRUE_POSITIVE_SCOPE,
    ATTACKED_FALSE_POSITIVE_SCOPE,
)

RANDOMIZATION_DETECTION_CLUSTER_FIELDNAMES = (
    "method_id",
    "prompt_id",
    "metric_scope",
    "normalized_sample_role",
    "attack_id",
    "attack_family",
    "attack_name",
    "resource_profile",
    "attack_config_digest",
    "randomization_repeat_count",
    "registered_repeat_decision_map",
    "registered_repeat_raw_decision_map",
    "positive_repeat_count",
    "prompt_cluster_positive_rate",
    "raw_positive_repeat_count",
    "prompt_cluster_raw_positive_rate",
    "method_repeat_threshold_map_digest",
    "source_outcome_set_digest",
    "cluster_record_digest",
)

RANDOMIZATION_DETECTION_OPERATING_POINT_FIELDNAMES = (
    "paper_claim_scale",
    "method_id",
    "prompt_cluster_count",
    "randomization_repeat_count",
    "attack_count",
    "statistical_unit",
    "target_fpr",
    "confidence_level",
    "confidence_interval_method",
    "false_positive_bound_method",
    "clean_positive_observation_count",
    "clean_true_positive_count",
    "clean_true_positive_rate",
    "clean_true_positive_rate_ci_low",
    "clean_true_positive_rate_ci_high",
    "clean_negative_observation_count",
    "clean_false_positive_count",
    "clean_false_positive_rate",
    "clean_false_positive_rate_ci_low",
    "clean_false_positive_rate_ci_high",
    "clean_raw_false_positive_count",
    "clean_raw_false_positive_rate",
    "clean_raw_false_positive_rate_ci_low",
    "clean_raw_false_positive_rate_ci_high",
    "clean_rescue_added_false_positive_count",
    "clean_rescue_added_false_positive_rate",
    "clean_maximum_repeat_false_positive_count",
    "clean_maximum_repeat_false_positive_rate",
    "clean_maximum_repeat_false_positive_rate_upper_bound",
    "clean_maximum_repeat_false_positive_bound_repeat_id",
    "clean_fixed_fpr_ready",
    "attacked_positive_observation_count",
    "attacked_true_positive_count",
    "attacked_true_positive_rate",
    "attacked_true_positive_rate_ci_low",
    "attacked_true_positive_rate_ci_high",
    "attacked_negative_observation_count",
    "attacked_false_positive_count",
    "attacked_false_positive_rate",
    "attacked_false_positive_rate_ci_low",
    "attacked_false_positive_rate_ci_high",
    "attacked_raw_false_positive_count",
    "attacked_raw_false_positive_rate",
    "attacked_raw_false_positive_rate_ci_low",
    "attacked_raw_false_positive_rate_ci_high",
    "attacked_rescue_added_false_positive_count",
    "attacked_rescue_added_false_positive_rate",
    "method_repeat_threshold_map_digest",
    "cluster_record_set_digest",
    "randomization_detection_statistics_ready",
    "supports_paper_claim",
)

RANDOMIZATION_PER_ATTACK_DETECTION_FIELDNAMES = (
    "paper_claim_scale",
    "method_id",
    "attack_id",
    "attack_family",
    "attack_name",
    "resource_profile",
    "attack_config_digest",
    "prompt_cluster_count",
    "randomization_repeat_count",
    "statistical_unit",
    "target_fpr",
    "confidence_level",
    "confidence_interval_method",
    "false_positive_bound_method",
    "attacked_positive_observation_count",
    "attacked_true_positive_count",
    "attacked_true_positive_rate",
    "attacked_true_positive_rate_ci_low",
    "attacked_true_positive_rate_ci_high",
    "attacked_negative_observation_count",
    "attacked_false_positive_count",
    "attacked_false_positive_rate",
    "attacked_false_positive_rate_ci_low",
    "attacked_false_positive_rate_ci_high",
    "attacked_raw_false_positive_count",
    "attacked_raw_false_positive_rate",
    "attacked_raw_false_positive_rate_ci_low",
    "attacked_raw_false_positive_rate_ci_high",
    "attacked_rescue_added_false_positive_count",
    "attacked_rescue_added_false_positive_rate",
    "attacked_maximum_repeat_false_positive_count",
    "attacked_maximum_repeat_false_positive_rate",
    "attacked_maximum_repeat_false_positive_rate_upper_bound",
    "attacked_maximum_repeat_false_positive_bound_repeat_id",
    "attacked_fixed_fpr_ready",
    "method_repeat_threshold_map_digest",
    "cluster_record_set_digest",
    "randomization_detection_statistics_ready",
    "supports_paper_claim",
)

RANDOMIZATION_WRONG_KEY_FIELDNAMES = (
    "paper_claim_scale",
    "method_id",
    "prompt_cluster_count",
    "randomization_repeat_count",
    "statistical_unit",
    "target_fpr",
    "confidence_level",
    "confidence_interval_method",
    "false_positive_bound_method",
    "wrong_key_observation_count",
    "wrong_key_false_positive_count",
    "wrong_key_false_positive_rate",
    "wrong_key_false_positive_rate_ci_low",
    "wrong_key_false_positive_rate_ci_high",
    "wrong_key_raw_false_positive_count",
    "wrong_key_raw_false_positive_rate",
    "wrong_key_raw_false_positive_rate_ci_low",
    "wrong_key_raw_false_positive_rate_ci_high",
    "wrong_key_rescue_added_false_positive_count",
    "wrong_key_rescue_added_false_positive_rate",
    "wrong_key_maximum_repeat_false_positive_count",
    "wrong_key_maximum_repeat_false_positive_rate",
    "wrong_key_maximum_repeat_false_positive_rate_upper_bound",
    "wrong_key_maximum_repeat_false_positive_bound_repeat_id",
    "wrong_key_fixed_fpr_ready",
    "method_repeat_threshold_map_digest",
    "cluster_record_set_digest",
    "randomization_detection_statistics_ready",
    "supports_paper_claim",
)

RANDOMIZATION_PER_ATTACK_COMPARISON_FIELDNAMES = (
    "paper_claim_scale",
    "attack_id",
    "attack_family",
    "attack_name",
    "resource_profile",
    "attack_config_digest",
    "most_conservative_baseline_id",
    "paired_prompt_count",
    "randomization_repeat_count",
    "statistical_unit",
    "target_fpr",
    "mean_paired_true_positive_rate_difference",
    "mean_paired_difference_simultaneous_ci_low",
    "mean_paired_difference_simultaneous_ci_high",
    "one_sided_bounded_hoeffding_mean_p_value",
    "holm_adjusted_p_value",
    "comparison_family_size",
    "simultaneous_confidence_level",
    "confidence_interval_method",
    "claim_p_value_method",
    "slm_clean_fixed_fpr_ready",
    "baseline_clean_fixed_fpr_ready",
    "slm_attacked_fixed_fpr_ready",
    "baseline_attacked_fixed_fpr_ready",
    "all_methods_clean_fixed_fpr_ready",
    "all_methods_attacked_fixed_fpr_ready",
    "method_repeat_threshold_map_digest",
    "cluster_record_set_digest",
    "superiority_claim_ready",
    "randomization_detection_statistics_ready",
    "supports_paper_claim",
)


class RandomizationDetectionStatisticsError(ValueError):
    """表示检测聚类记录不能形成公平的精确9重复统计."""


def _require_sha256(value: Any, field_name: str) -> str:
    """读取规范小写 SHA-256, 用于运行混选边界."""

    resolved = str(value)
    if SHA256_PATTERN.fullmatch(resolved) is None:
        raise RandomizationDetectionStatisticsError(
            f"{field_name} 必须是小写 SHA-256"
        )
    return resolved


def _strict_integer(value: Any, field_name: str) -> int:
    """读取精确整数, 禁止 bool 和小数静默转换."""

    if isinstance(value, bool):
        raise RandomizationDetectionStatisticsError(f"{field_name} 必须是整数")
    try:
        resolved = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RandomizationDetectionStatisticsError(
            f"{field_name} 必须是整数"
        ) from exc
    if not math.isfinite(numeric) or numeric != resolved:
        raise RandomizationDetectionStatisticsError(f"{field_name} 必须是整数")
    return resolved


def _canonical_cluster_records(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], str]:
    """校验聚类记录公式、自摘要和字段集合并规范排序."""

    repeat_count = len(formal_randomization_repeat_ids())
    expected_fields = set(RANDOMIZATION_DETECTION_CLUSTER_FIELDNAMES)
    materialized: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(rows):
        row = dict(raw_row)
        if set(row) != expected_fields:
            raise RandomizationDetectionStatisticsError(
                f"检测聚类记录字段集合不一致: row={row_index}"
            )
        declared_digest = _require_sha256(
            row["cluster_record_digest"], "cluster_record_digest"
        )
        payload = {
            field_name: value
            for field_name, value in row.items()
            if field_name != "cluster_record_digest"
        }
        if build_stable_digest(payload) != declared_digest:
            raise RandomizationDetectionStatisticsError("检测聚类记录摘要无法重建")
        method_id = str(row["method_id"])
        metric_scope = str(row["metric_scope"])
        normalized_role = str(row["normalized_sample_role"])
        if method_id not in DETECTION_METHOD_IDS or metric_scope not in DETECTION_METRIC_SCOPES:
            raise RandomizationDetectionStatisticsError("检测方法或指标范围未注册")
        if metric_scope == WRONG_KEY_FALSE_POSITIVE_SCOPE and method_id != PROPOSED_METHOD_ID:
            raise RandomizationDetectionStatisticsError("baseline 不具有 wrong-key 语义")
        expected_role = (
            "positive"
            if metric_scope in {CLEAN_TRUE_POSITIVE_SCOPE, ATTACKED_TRUE_POSITIVE_SCOPE}
            else "negative"
        )
        if normalized_role != expected_role:
            raise RandomizationDetectionStatisticsError("指标范围与样本极性不一致")
        actual_repeat_count = _strict_integer(
            row["randomization_repeat_count"], "randomization_repeat_count"
        )
        decision_map = row["registered_repeat_decision_map"]
        raw_decision_map = row["registered_repeat_raw_decision_map"]
        expected_repeat_ids = set(formal_randomization_repeat_ids())
        if (
            not isinstance(decision_map, Mapping)
            or not isinstance(raw_decision_map, Mapping)
            or set(decision_map) != expected_repeat_ids
            or set(raw_decision_map) != expected_repeat_ids
            or any(not isinstance(value, bool) for value in decision_map.values())
            or any(
                not isinstance(value, bool) for value in raw_decision_map.values()
            )
            or any(
                bool(raw_decision_map[repeat_id])
                and not bool(decision_map[repeat_id])
                for repeat_id in expected_repeat_ids
            )
        ):
            raise RandomizationDetectionStatisticsError(
                "检测聚类的注册重复判定映射无效"
            )
        positive_count = _strict_integer(
            row["positive_repeat_count"], "positive_repeat_count"
        )
        raw_positive_count = _strict_integer(
            row["raw_positive_repeat_count"], "raw_positive_repeat_count"
        )
        if actual_repeat_count != repeat_count or not 0 <= positive_count <= repeat_count:
            raise RandomizationDetectionStatisticsError("检测聚类未精确包含9个注册重复")
        if (
            positive_count != sum(bool(value) for value in decision_map.values())
            or raw_positive_count
            != sum(bool(value) for value in raw_decision_map.values())
            or not 0 <= raw_positive_count <= positive_count
        ):
            raise RandomizationDetectionStatisticsError(
                "检测聚类计数无法由注册重复判定映射重建"
            )
        rate = float(row["prompt_cluster_positive_rate"])
        raw_rate = float(row["prompt_cluster_raw_positive_rate"])
        if not math.isfinite(rate) or not math.isclose(
            rate,
            positive_count / repeat_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RandomizationDetectionStatisticsError("Prompt 聚类率无法由重复计数重建")
        if not math.isfinite(raw_rate) or not math.isclose(
            raw_rate,
            raw_positive_count / repeat_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RandomizationDetectionStatisticsError(
                "Prompt 聚类 raw-content 率无法由重复计数重建"
            )
        _require_sha256(
            row["method_repeat_threshold_map_digest"],
            "method_repeat_threshold_map_digest",
        )
        _require_sha256(row["source_outcome_set_digest"], "source_outcome_set_digest")
        attacked = metric_scope in {
            ATTACKED_TRUE_POSITIVE_SCOPE,
            ATTACKED_FALSE_POSITIVE_SCOPE,
        }
        attack_identity = tuple(
            str(row[field_name])
            for field_name in (
                "attack_id",
                "attack_family",
                "attack_name",
                "resource_profile",
                "attack_config_digest",
            )
        )
        if attacked:
            if not all(attack_identity[:-1]):
                raise RandomizationDetectionStatisticsError("攻击聚类缺少完整攻击身份")
            _require_sha256(attack_identity[-1], "attack_config_digest")
        elif any(attack_identity):
            raise RandomizationDetectionStatisticsError("未攻击聚类不得携带攻击身份")
        if not str(row["prompt_id"]):
            raise RandomizationDetectionStatisticsError("检测聚类缺少 prompt_id")
        materialized.append(row)
    canonical = tuple(
        sorted(
            materialized,
            key=lambda row: (
                str(row["method_id"]),
                str(row["metric_scope"]),
                str(row["attack_id"]),
                str(row["prompt_id"]),
            ),
        )
    )
    return canonical, build_stable_digest(canonical)


def _bounded_rate_statistics(
    cluster_rows: Iterable[Mapping[str, Any]],
    *,
    confidence_level: float,
    raw_content: bool = False,
) -> dict[str, Any]:
    """从 Prompt 聚类率计算均值、Hoeffding 区间和原始判定计数."""

    materialized = tuple(cluster_rows)
    if not materialized:
        raise RandomizationDetectionStatisticsError("检测率统计缺少 Prompt 聚类")
    repeat_count = len(formal_randomization_repeat_ids())
    rate_field = (
        "prompt_cluster_raw_positive_rate"
        if raw_content
        else "prompt_cluster_positive_rate"
    )
    count_field = (
        "raw_positive_repeat_count"
        if raw_content
        else "positive_repeat_count"
    )
    values = np.asarray(
        [float(row[rate_field]) for row in materialized],
        dtype=np.float64,
    )
    rate = float(values.mean())
    ci_low, ci_high = bounded_hoeffding_confidence_interval(
        rate,
        len(materialized),
        confidence_level,
    )
    positive_count = sum(int(row[count_field]) for row in materialized)
    observation_count = len(materialized) * repeat_count
    if not math.isclose(
        rate,
        positive_count / observation_count,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RandomizationDetectionStatisticsError("检测率未匹配逐重复二元判定计数")
    return {
        "prompt_cluster_count": len(materialized),
        "observation_count": observation_count,
        "positive_count": positive_count,
        "rate": rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _repeat_false_positive_bound_statistics(
    cluster_rows: Iterable[Mapping[str, Any]],
    *,
    confidence_level: float,
) -> dict[str, Any]:
    """逐 repeat 计算 Wilson 上界并返回最不利的正式 operating point."""

    materialized = tuple(cluster_rows)
    if not materialized:
        raise RandomizationDetectionStatisticsError("逐重复 FPR 统计缺少 Prompt 聚类")
    repeat_ids = formal_randomization_repeat_ids()
    prompt_count = len(materialized)
    counts = {
        repeat_id: sum(
            bool(row["registered_repeat_decision_map"][repeat_id])
            for row in materialized
        )
        for repeat_id in repeat_ids
    }
    upper_bounds = {
        repeat_id: binomial_rate_upper_confidence_bound(
            counts[repeat_id],
            prompt_count,
            confidence_level,
        )
        for repeat_id in repeat_ids
    }
    maximum_repeat_id = max(
        repeat_ids,
        key=lambda repeat_id: (
            upper_bounds[repeat_id],
            counts[repeat_id],
            repeat_id,
        ),
    )
    return {
        "false_positive_count_by_repeat": counts,
        "false_positive_rate_upper_bound_by_repeat": upper_bounds,
        "maximum_repeat_false_positive_count": counts[maximum_repeat_id],
        "maximum_repeat_false_positive_rate": (
            counts[maximum_repeat_id] / prompt_count
        ),
        "maximum_repeat_false_positive_rate_upper_bound": upper_bounds[
            maximum_repeat_id
        ],
        "maximum_repeat_false_positive_bound_repeat_id": maximum_repeat_id,
    }


def _bounded_paired_difference_p_value(prompt_values: np.ndarray) -> float:
    """计算范围为 [-1, 1] 的 Prompt 配对均值单侧 Hoeffding 上界."""

    if prompt_values.ndim != 1 or prompt_values.size <= 0:
        raise RandomizationDetectionStatisticsError("配对检验需要非空 Prompt 差值")
    observed_mean = float(prompt_values.mean())
    if observed_mean <= 0.0:
        return 1.0
    return max(
        math.exp(-float(prompt_values.size) * observed_mean * observed_mean / 2.0),
        float(np.finfo(np.float64).tiny),
    )


def _holm_adjusted_p_values(
    keyed_p_values: Mapping[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    """对预注册 attack × baseline 家族执行 Holm 步降校正."""

    ordered = sorted(
        keyed_p_values.items(),
        key=lambda item: (float(item[1]), item[0]),
    )
    family_size = len(ordered)
    running_adjusted = 0.0
    adjusted: dict[tuple[str, str], float] = {}
    for rank, (key, p_value) in enumerate(ordered, start=1):
        value = min(
            1.0,
            max(
                running_adjusted,
                float(p_value) * (family_size - rank + 1),
            ),
        )
        running_adjusted = value
        adjusted[key] = value
    return adjusted


def _attacked_overall_statistics(
    rows: Iterable[Mapping[str, Any]],
    *,
    prompt_ids: tuple[str, ...],
    attack_ids: tuple[str, ...],
    confidence_level: float,
    raw_content: bool = False,
) -> dict[str, Any]:
    """先在 Prompt 内平均攻击与重复, 再计算总体攻击检测率."""

    materialized = tuple(rows)
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        by_prompt[str(row["prompt_id"])].append(dict(row))
    if set(by_prompt) != set(prompt_ids) or any(
        {str(row["attack_id"]) for row in prompt_rows} != set(attack_ids)
        or len(prompt_rows) != len(attack_ids)
        for prompt_rows in by_prompt.values()
    ):
        raise RandomizationDetectionStatisticsError("总体攻击率未覆盖完整 Prompt 与攻击笛卡尔积")
    repeat_count = len(formal_randomization_repeat_ids())
    rate_field = (
        "prompt_cluster_raw_positive_rate"
        if raw_content
        else "prompt_cluster_positive_rate"
    )
    count_field = (
        "raw_positive_repeat_count"
        if raw_content
        else "positive_repeat_count"
    )
    prompt_values = np.asarray(
        [
            float(
                np.mean(
                    [
                        float(row[rate_field])
                        for row in by_prompt[prompt_id]
                    ]
                )
            )
            for prompt_id in prompt_ids
        ],
        dtype=np.float64,
    )
    rate = float(prompt_values.mean())
    ci_low, ci_high = bounded_hoeffding_confidence_interval(
        rate,
        len(prompt_ids),
        confidence_level,
    )
    positive_count = sum(int(row[count_field]) for row in materialized)
    observation_count = len(prompt_ids) * len(attack_ids) * repeat_count
    if not math.isclose(
        rate,
        positive_count / observation_count,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RandomizationDetectionStatisticsError("总体攻击率未匹配原始二元计数")
    return {
        "observation_count": observation_count,
        "positive_count": positive_count,
        "rate": rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def build_randomization_detection_statistics(
    cluster_records: Iterable[Mapping[str, Any]],
    *,
    paper_run_name: str,
    target_fpr: float,
    attack_registry_rows: Iterable[Mapping[str, Any]],
    confidence_level: float = PILOT_PAPER_CONFIDENCE_LEVEL,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    """构造5方法绝对率、85个逐攻击率、wrong-key 与保守比较表."""

    run_name = normalize_paper_run_name(paper_run_name)
    resolved_target_fpr = validate_frozen_paper_run_target_fpr(
        run_name,
        target_fpr,
    )
    if not 0.0 < float(confidence_level) < 1.0:
        raise RandomizationDetectionStatisticsError("confidence_level 必须位于 (0, 1)")
    expected_prompt_count = build_group_split_counts(
        RUN_EXPECTED_PROMPT_COUNTS[run_name]
    )["test"]
    attack_registry = canonical_attack_registry_rows(attack_registry_rows)
    attack_by_id = {str(row["attack_id"]): dict(row) for row in attack_registry}
    attack_ids = tuple(sorted(attack_by_id))
    if not attack_ids:
        raise RandomizationDetectionStatisticsError("正式攻击 registry 不得为空")
    canonical, cluster_record_set_digest = _canonical_cluster_records(
        cluster_records
    )
    threshold_map_digests = {
        str(row["method_repeat_threshold_map_digest"]) for row in canonical
    }
    if len(threshold_map_digests) != 1:
        raise RandomizationDetectionStatisticsError("检测聚类记录混用了多套阈值映射")
    threshold_map_digest = next(iter(threshold_map_digests))

    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    prompt_sets: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in canonical:
        method_id = str(row["method_id"])
        scope = str(row["metric_scope"])
        attack_id = str(row["attack_id"])
        prompt_id = str(row["prompt_id"])
        key = (method_id, scope, attack_id, prompt_id)
        if key in by_key:
            raise RandomizationDetectionStatisticsError("检测聚类键重复")
        by_key[key] = row
        prompt_sets[(method_id, scope, attack_id)].add(prompt_id)
        if attack_id:
            expected_attack = attack_by_id.get(attack_id)
            if expected_attack is None or any(
                str(row[field_name]) != str(expected_attack[field_name])
                for field_name in (
                    "attack_family",
                    "attack_name",
                    "resource_profile",
                    "attack_config_digest",
                )
            ):
                raise RandomizationDetectionStatisticsError("检测聚类攻击身份未匹配冻结 registry")

    common_prompt_sets = {frozenset(values) for values in prompt_sets.values()}
    if len(common_prompt_sets) != 1:
        raise RandomizationDetectionStatisticsError("方法、范围或攻击未共享同一 test Prompt 集合")
    prompt_ids = tuple(sorted(next(iter(common_prompt_sets))))
    if len(prompt_ids) != expected_prompt_count:
        raise RandomizationDetectionStatisticsError("test Prompt 数量未匹配论文运行层级")

    expected_keys: set[tuple[str, str, str, str]] = set()
    for method_id in DETECTION_METHOD_IDS:
        for scope in (CLEAN_TRUE_POSITIVE_SCOPE, CLEAN_FALSE_POSITIVE_SCOPE):
            expected_keys.update(
                (method_id, scope, "", prompt_id) for prompt_id in prompt_ids
            )
        for scope in (ATTACKED_TRUE_POSITIVE_SCOPE, ATTACKED_FALSE_POSITIVE_SCOPE):
            expected_keys.update(
                (method_id, scope, attack_id, prompt_id)
                for attack_id in attack_ids
                for prompt_id in prompt_ids
            )
    expected_keys.update(
        (PROPOSED_METHOD_ID, WRONG_KEY_FALSE_POSITIVE_SCOPE, "", prompt_id)
        for prompt_id in prompt_ids
    )
    if set(by_key) != expected_keys:
        raise RandomizationDetectionStatisticsError(
            "检测聚类未精确覆盖9重复、5方法、Prompt、角色与攻击协议"
        )

    repeat_count = len(formal_randomization_repeat_ids())
    common_fields = {
        "paper_claim_scale": run_name,
        "prompt_cluster_count": len(prompt_ids),
        "randomization_repeat_count": repeat_count,
        "statistical_unit": RANDOMIZATION_DETECTION_STATISTICAL_UNIT,
        "target_fpr": resolved_target_fpr,
        "confidence_level": float(confidence_level),
        "confidence_interval_method": (
            RANDOMIZATION_DETECTION_CONFIDENCE_INTERVAL_METHOD
        ),
        "false_positive_bound_method": (
            RANDOMIZATION_DETECTION_FALSE_POSITIVE_BOUND_METHOD
        ),
        "method_repeat_threshold_map_digest": threshold_map_digest,
        "cluster_record_set_digest": cluster_record_set_digest,
        "randomization_detection_statistics_ready": True,
        "supports_paper_claim": False,
    }
    operating_rows: list[dict[str, Any]] = []
    per_attack_rows: list[dict[str, Any]] = []
    for method_id in DETECTION_METHOD_IDS:
        clean_positive_rows = tuple(
            by_key[(method_id, CLEAN_TRUE_POSITIVE_SCOPE, "", prompt_id)]
            for prompt_id in prompt_ids
        )
        clean_negative_rows = tuple(
            by_key[(method_id, CLEAN_FALSE_POSITIVE_SCOPE, "", prompt_id)]
            for prompt_id in prompt_ids
        )
        clean_positive = _bounded_rate_statistics(
            clean_positive_rows,
            confidence_level=confidence_level,
        )
        clean_negative = _bounded_rate_statistics(
            clean_negative_rows,
            confidence_level=confidence_level,
        )
        clean_raw_negative = _bounded_rate_statistics(
            clean_negative_rows,
            confidence_level=confidence_level,
            raw_content=True,
        )
        clean_repeat_bound = _repeat_false_positive_bound_statistics(
            clean_negative_rows,
            confidence_level=confidence_level,
        )
        attacked_positive_rows = tuple(
            by_key[(method_id, ATTACKED_TRUE_POSITIVE_SCOPE, attack_id, prompt_id)]
            for prompt_id in prompt_ids
            for attack_id in attack_ids
        )
        attacked_negative_rows = tuple(
            by_key[(method_id, ATTACKED_FALSE_POSITIVE_SCOPE, attack_id, prompt_id)]
            for prompt_id in prompt_ids
            for attack_id in attack_ids
        )
        attacked_positive = _attacked_overall_statistics(
            attacked_positive_rows,
            prompt_ids=prompt_ids,
            attack_ids=attack_ids,
            confidence_level=confidence_level,
        )
        attacked_negative = _attacked_overall_statistics(
            attacked_negative_rows,
            prompt_ids=prompt_ids,
            attack_ids=attack_ids,
            confidence_level=confidence_level,
        )
        attacked_raw_negative = _attacked_overall_statistics(
            attacked_negative_rows,
            prompt_ids=prompt_ids,
            attack_ids=attack_ids,
            confidence_level=confidence_level,
            raw_content=True,
        )
        operating_rows.append(
            {
                **common_fields,
                "method_id": method_id,
                "attack_count": len(attack_ids),
                "clean_positive_observation_count": clean_positive["observation_count"],
                "clean_true_positive_count": clean_positive["positive_count"],
                "clean_true_positive_rate": clean_positive["rate"],
                "clean_true_positive_rate_ci_low": clean_positive["ci_low"],
                "clean_true_positive_rate_ci_high": clean_positive["ci_high"],
                "clean_negative_observation_count": clean_negative["observation_count"],
                "clean_false_positive_count": clean_negative["positive_count"],
                "clean_false_positive_rate": clean_negative["rate"],
                "clean_false_positive_rate_ci_low": clean_negative["ci_low"],
                "clean_false_positive_rate_ci_high": clean_negative["ci_high"],
                "clean_raw_false_positive_count": clean_raw_negative[
                    "positive_count"
                ],
                "clean_raw_false_positive_rate": clean_raw_negative["rate"],
                "clean_raw_false_positive_rate_ci_low": clean_raw_negative[
                    "ci_low"
                ],
                "clean_raw_false_positive_rate_ci_high": clean_raw_negative[
                    "ci_high"
                ],
                "clean_rescue_added_false_positive_count": (
                    clean_negative["positive_count"]
                    - clean_raw_negative["positive_count"]
                ),
                "clean_rescue_added_false_positive_rate": (
                    clean_negative["rate"] - clean_raw_negative["rate"]
                ),
                "clean_maximum_repeat_false_positive_count": clean_repeat_bound[
                    "maximum_repeat_false_positive_count"
                ],
                "clean_maximum_repeat_false_positive_rate": clean_repeat_bound[
                    "maximum_repeat_false_positive_rate"
                ],
                "clean_maximum_repeat_false_positive_rate_upper_bound": (
                    clean_repeat_bound[
                        "maximum_repeat_false_positive_rate_upper_bound"
                    ]
                ),
                "clean_maximum_repeat_false_positive_bound_repeat_id": (
                    clean_repeat_bound[
                        "maximum_repeat_false_positive_bound_repeat_id"
                    ]
                ),
                "clean_fixed_fpr_ready": bool(
                    clean_repeat_bound[
                        "maximum_repeat_false_positive_rate_upper_bound"
                    ]
                    <= resolved_target_fpr
                ),
                "attacked_positive_observation_count": attacked_positive[
                    "observation_count"
                ],
                "attacked_true_positive_count": attacked_positive["positive_count"],
                "attacked_true_positive_rate": attacked_positive["rate"],
                "attacked_true_positive_rate_ci_low": attacked_positive["ci_low"],
                "attacked_true_positive_rate_ci_high": attacked_positive["ci_high"],
                "attacked_negative_observation_count": attacked_negative[
                    "observation_count"
                ],
                "attacked_false_positive_count": attacked_negative["positive_count"],
                "attacked_false_positive_rate": attacked_negative["rate"],
                "attacked_false_positive_rate_ci_low": attacked_negative["ci_low"],
                "attacked_false_positive_rate_ci_high": attacked_negative["ci_high"],
                "attacked_raw_false_positive_count": attacked_raw_negative[
                    "positive_count"
                ],
                "attacked_raw_false_positive_rate": attacked_raw_negative["rate"],
                "attacked_raw_false_positive_rate_ci_low": attacked_raw_negative[
                    "ci_low"
                ],
                "attacked_raw_false_positive_rate_ci_high": attacked_raw_negative[
                    "ci_high"
                ],
                "attacked_rescue_added_false_positive_count": (
                    attacked_negative["positive_count"]
                    - attacked_raw_negative["positive_count"]
                ),
                "attacked_rescue_added_false_positive_rate": (
                    attacked_negative["rate"] - attacked_raw_negative["rate"]
                ),
            }
        )

        for attack_id in attack_ids:
            attack = attack_by_id[attack_id]
            positive_rows = tuple(
                by_key[(method_id, ATTACKED_TRUE_POSITIVE_SCOPE, attack_id, prompt_id)]
                for prompt_id in prompt_ids
            )
            negative_rows = tuple(
                by_key[(method_id, ATTACKED_FALSE_POSITIVE_SCOPE, attack_id, prompt_id)]
                for prompt_id in prompt_ids
            )
            positive = _bounded_rate_statistics(
                positive_rows,
                confidence_level=confidence_level,
            )
            negative = _bounded_rate_statistics(
                negative_rows,
                confidence_level=confidence_level,
            )
            raw_negative = _bounded_rate_statistics(
                negative_rows,
                confidence_level=confidence_level,
                raw_content=True,
            )
            repeat_bound = _repeat_false_positive_bound_statistics(
                negative_rows,
                confidence_level=confidence_level,
            )
            per_attack_rows.append(
                {
                    **common_fields,
                    "method_id": method_id,
                    "attack_id": attack_id,
                    "attack_family": attack["attack_family"],
                    "attack_name": attack["attack_name"],
                    "resource_profile": attack["resource_profile"],
                    "attack_config_digest": attack["attack_config_digest"],
                    "attacked_positive_observation_count": positive[
                        "observation_count"
                    ],
                    "attacked_true_positive_count": positive["positive_count"],
                    "attacked_true_positive_rate": positive["rate"],
                    "attacked_true_positive_rate_ci_low": positive["ci_low"],
                    "attacked_true_positive_rate_ci_high": positive["ci_high"],
                    "attacked_negative_observation_count": negative[
                        "observation_count"
                    ],
                    "attacked_false_positive_count": negative["positive_count"],
                    "attacked_false_positive_rate": negative["rate"],
                    "attacked_false_positive_rate_ci_low": negative["ci_low"],
                    "attacked_false_positive_rate_ci_high": negative["ci_high"],
                    "attacked_raw_false_positive_count": raw_negative[
                        "positive_count"
                    ],
                    "attacked_raw_false_positive_rate": raw_negative["rate"],
                    "attacked_raw_false_positive_rate_ci_low": raw_negative[
                        "ci_low"
                    ],
                    "attacked_raw_false_positive_rate_ci_high": raw_negative[
                        "ci_high"
                    ],
                    "attacked_rescue_added_false_positive_count": (
                        negative["positive_count"] - raw_negative["positive_count"]
                    ),
                    "attacked_rescue_added_false_positive_rate": (
                        negative["rate"] - raw_negative["rate"]
                    ),
                    "attacked_maximum_repeat_false_positive_count": repeat_bound[
                        "maximum_repeat_false_positive_count"
                    ],
                    "attacked_maximum_repeat_false_positive_rate": repeat_bound[
                        "maximum_repeat_false_positive_rate"
                    ],
                    "attacked_maximum_repeat_false_positive_rate_upper_bound": (
                        repeat_bound[
                            "maximum_repeat_false_positive_rate_upper_bound"
                        ]
                    ),
                    "attacked_maximum_repeat_false_positive_bound_repeat_id": (
                        repeat_bound[
                            "maximum_repeat_false_positive_bound_repeat_id"
                        ]
                    ),
                    "attacked_fixed_fpr_ready": bool(
                        repeat_bound[
                            "maximum_repeat_false_positive_rate_upper_bound"
                        ]
                        <= resolved_target_fpr
                    ),
                }
            )

    wrong_key_cluster_rows = tuple(
        by_key[(PROPOSED_METHOD_ID, WRONG_KEY_FALSE_POSITIVE_SCOPE, "", prompt_id)]
        for prompt_id in prompt_ids
    )
    wrong_key = _bounded_rate_statistics(
        wrong_key_cluster_rows,
        confidence_level=confidence_level,
    )
    wrong_key_raw = _bounded_rate_statistics(
        wrong_key_cluster_rows,
        confidence_level=confidence_level,
        raw_content=True,
    )
    wrong_key_repeat_bound = _repeat_false_positive_bound_statistics(
        wrong_key_cluster_rows,
        confidence_level=confidence_level,
    )
    wrong_key_rows = (
        {
            **common_fields,
            "method_id": PROPOSED_METHOD_ID,
            "wrong_key_observation_count": wrong_key["observation_count"],
            "wrong_key_false_positive_count": wrong_key["positive_count"],
            "wrong_key_false_positive_rate": wrong_key["rate"],
            "wrong_key_false_positive_rate_ci_low": wrong_key["ci_low"],
            "wrong_key_false_positive_rate_ci_high": wrong_key["ci_high"],
            "wrong_key_raw_false_positive_count": wrong_key_raw[
                "positive_count"
            ],
            "wrong_key_raw_false_positive_rate": wrong_key_raw["rate"],
            "wrong_key_raw_false_positive_rate_ci_low": wrong_key_raw["ci_low"],
            "wrong_key_raw_false_positive_rate_ci_high": wrong_key_raw["ci_high"],
            "wrong_key_rescue_added_false_positive_count": (
                wrong_key["positive_count"] - wrong_key_raw["positive_count"]
            ),
            "wrong_key_rescue_added_false_positive_rate": (
                wrong_key["rate"] - wrong_key_raw["rate"]
            ),
            "wrong_key_maximum_repeat_false_positive_count": (
                wrong_key_repeat_bound["maximum_repeat_false_positive_count"]
            ),
            "wrong_key_maximum_repeat_false_positive_rate": (
                wrong_key_repeat_bound["maximum_repeat_false_positive_rate"]
            ),
            "wrong_key_maximum_repeat_false_positive_rate_upper_bound": (
                wrong_key_repeat_bound[
                    "maximum_repeat_false_positive_rate_upper_bound"
                ]
            ),
            "wrong_key_maximum_repeat_false_positive_bound_repeat_id": (
                wrong_key_repeat_bound[
                    "maximum_repeat_false_positive_bound_repeat_id"
                ]
            ),
            "wrong_key_fixed_fpr_ready": bool(
                wrong_key_repeat_bound[
                    "maximum_repeat_false_positive_rate_upper_bound"
                ]
                <= resolved_target_fpr
            ),
        },
    )

    per_attack_by_key = {
        (str(row["method_id"]), str(row["attack_id"])): row
        for row in per_attack_rows
    }
    operating_by_method = {
        str(row["method_id"]): row for row in operating_rows
    }
    comparison_family_size = len(attack_ids) * len(PRIMARY_BASELINE_IDS)
    family_confidence_level = 1.0 - (
        (1.0 - float(confidence_level)) / comparison_family_size
    )
    paired_statistics: dict[tuple[str, str], dict[str, Any]] = {}
    for attack_id in attack_ids:
        for baseline_id in PRIMARY_BASELINE_IDS:
            prompt_values = np.asarray(
                [
                    float(
                        np.mean(
                            [
                                int(
                                    by_key[
                                        (
                                            PROPOSED_METHOD_ID,
                                            ATTACKED_TRUE_POSITIVE_SCOPE,
                                            attack_id,
                                            prompt_id,
                                        )
                                    ]["registered_repeat_decision_map"][repeat_id]
                                )
                                - int(
                                    by_key[
                                        (
                                            baseline_id,
                                            ATTACKED_TRUE_POSITIVE_SCOPE,
                                            attack_id,
                                            prompt_id,
                                        )
                                    ]["registered_repeat_decision_map"][repeat_id]
                                )
                                for repeat_id in formal_randomization_repeat_ids()
                            ]
                        )
                    )
                    for prompt_id in prompt_ids
                ],
                dtype=np.float64,
            )
            mean_difference = float(prompt_values.mean())
            ci_low, ci_high = bounded_hoeffding_confidence_interval(
                mean_difference,
                len(prompt_ids),
                family_confidence_level,
                lower_bound=-1.0,
                upper_bound=1.0,
            )
            paired_statistics[(attack_id, baseline_id)] = {
                "mean_difference": mean_difference,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": _bounded_paired_difference_p_value(prompt_values),
            }
    adjusted_p_values = _holm_adjusted_p_values(
        {
            key: float(statistics["p_value"])
            for key, statistics in paired_statistics.items()
        }
    )
    comparison_rows: list[dict[str, Any]] = []
    for attack_id in attack_ids:
        attack = attack_by_id[attack_id]
        baseline_id = min(
            PRIMARY_BASELINE_IDS,
            key=lambda candidate: (
                float(paired_statistics[(attack_id, candidate)]["ci_low"]),
                -float(adjusted_p_values[(attack_id, candidate)]),
                candidate,
            ),
        )
        paired = paired_statistics[(attack_id, baseline_id)]
        slm_attack = per_attack_by_key[(PROPOSED_METHOD_ID, attack_id)]
        baseline_attack = per_attack_by_key[(baseline_id, attack_id)]
        slm_clean_ready = bool(
            operating_by_method[PROPOSED_METHOD_ID]["clean_fixed_fpr_ready"]
        )
        baseline_clean_ready = bool(
            operating_by_method[baseline_id]["clean_fixed_fpr_ready"]
        )
        slm_attacked_ready = bool(slm_attack["attacked_fixed_fpr_ready"])
        baseline_attacked_ready = bool(
            baseline_attack["attacked_fixed_fpr_ready"]
        )
        all_methods_clean_ready = all(
            bool(operating_by_method[method_id]["clean_fixed_fpr_ready"])
            for method_id in DETECTION_METHOD_IDS
        )
        all_methods_attacked_ready = all(
            bool(
                per_attack_by_key[(method_id, attack_id)][
                    "attacked_fixed_fpr_ready"
                ]
            )
            for method_id in DETECTION_METHOD_IDS
        )
        adjusted_p_value = float(adjusted_p_values[(attack_id, baseline_id)])
        comparison_rows.append(
            {
                "paper_claim_scale": run_name,
                "attack_id": attack_id,
                "attack_family": attack["attack_family"],
                "attack_name": attack["attack_name"],
                "resource_profile": attack["resource_profile"],
                "attack_config_digest": attack["attack_config_digest"],
                "most_conservative_baseline_id": baseline_id,
                "paired_prompt_count": len(prompt_ids),
                "randomization_repeat_count": repeat_count,
                "statistical_unit": RANDOMIZATION_DETECTION_STATISTICAL_UNIT,
                "target_fpr": resolved_target_fpr,
                "mean_paired_true_positive_rate_difference": paired[
                    "mean_difference"
                ],
                "mean_paired_difference_simultaneous_ci_low": paired["ci_low"],
                "mean_paired_difference_simultaneous_ci_high": paired["ci_high"],
                "one_sided_bounded_hoeffding_mean_p_value": paired["p_value"],
                "holm_adjusted_p_value": adjusted_p_value,
                "comparison_family_size": comparison_family_size,
                "simultaneous_confidence_level": family_confidence_level,
                "confidence_interval_method": "bonferroni_bounded_hoeffding",
                "claim_p_value_method": "bounded_hoeffding_prompt_cluster_mean",
                "slm_clean_fixed_fpr_ready": slm_clean_ready,
                "baseline_clean_fixed_fpr_ready": baseline_clean_ready,
                "slm_attacked_fixed_fpr_ready": slm_attacked_ready,
                "baseline_attacked_fixed_fpr_ready": baseline_attacked_ready,
                "all_methods_clean_fixed_fpr_ready": all_methods_clean_ready,
                "all_methods_attacked_fixed_fpr_ready": (
                    all_methods_attacked_ready
                ),
                "method_repeat_threshold_map_digest": threshold_map_digest,
                "cluster_record_set_digest": cluster_record_set_digest,
                "superiority_claim_ready": bool(
                    float(paired["ci_low"]) > 0.0
                    and adjusted_p_value < 0.05
                    and slm_clean_ready
                    and baseline_clean_ready
                    and slm_attacked_ready
                    and baseline_attacked_ready
                    and all_methods_clean_ready
                    and all_methods_attacked_ready
                ),
                "randomization_detection_statistics_ready": True,
                "supports_paper_claim": False,
            }
        )

    canonical_operating = tuple(
        sorted(operating_rows, key=lambda row: str(row["method_id"]))
    )
    canonical_per_attack = tuple(
        sorted(
            per_attack_rows,
            key=lambda row: (str(row["attack_id"]), str(row["method_id"])),
        )
    )
    canonical_comparisons = tuple(
        sorted(comparison_rows, key=lambda row: str(row["attack_id"]))
    )
    if any(
        set(row) != set(RANDOMIZATION_DETECTION_OPERATING_POINT_FIELDNAMES)
        for row in canonical_operating
    ) or any(
        set(row) != set(RANDOMIZATION_PER_ATTACK_DETECTION_FIELDNAMES)
        for row in canonical_per_attack
    ) or any(
        set(row) != set(RANDOMIZATION_WRONG_KEY_FIELDNAMES)
        for row in wrong_key_rows
    ) or any(
        set(row) != set(RANDOMIZATION_PER_ATTACK_COMPARISON_FIELDNAMES)
        for row in canonical_comparisons
    ):
        raise RandomizationDetectionStatisticsError("检测统计输出字段集合发生漂移")

    all_methods_clean_fixed_fpr_ready = all(
        row["clean_fixed_fpr_ready"] is True for row in canonical_operating
    )
    all_per_attack_fixed_fpr_ready = all(
        row["attacked_fixed_fpr_ready"] is True
        for row in canonical_per_attack
    )
    universal_per_attack_superiority_claim_ready = bool(
        all_methods_clean_fixed_fpr_ready
        and all_per_attack_fixed_fpr_ready
        and all(
            row["superiority_claim_ready"] is True
            for row in canonical_comparisons
        )
    )
    summary_payload = {
        "summary_schema": "randomization_detection_statistics_summary",
        "paper_claim_scale": run_name,
        "target_fpr": resolved_target_fpr,
        "method_ids": list(DETECTION_METHOD_IDS),
        "prompt_cluster_count": len(prompt_ids),
        "test_prompt_id_digest": build_stable_digest(list(prompt_ids)),
        "randomization_repeat_ids": list(formal_randomization_repeat_ids()),
        "randomization_repeat_count": repeat_count,
        "statistical_unit": RANDOMIZATION_DETECTION_STATISTICAL_UNIT,
        "attack_count": len(attack_registry),
        "attack_registry_digest": build_stable_digest(list(attack_registry)),
        "method_repeat_threshold_map_digest": threshold_map_digest,
        "cluster_record_count": len(canonical),
        "cluster_record_set_digest": cluster_record_set_digest,
        "operating_point_rows_digest": build_stable_digest(canonical_operating),
        "per_attack_detection_rows_digest": build_stable_digest(
            canonical_per_attack
        ),
        "wrong_key_rows_digest": build_stable_digest(wrong_key_rows),
        "per_attack_comparison_rows_digest": build_stable_digest(
            canonical_comparisons
        ),
        "confidence_level": float(confidence_level),
        "confidence_interval_method": (
            RANDOMIZATION_DETECTION_CONFIDENCE_INTERVAL_METHOD
        ),
        "false_positive_bound_method": (
            RANDOMIZATION_DETECTION_FALSE_POSITIVE_BOUND_METHOD
        ),
        "per_attack_comparison_family_size": comparison_family_size,
        "per_attack_simultaneous_confidence_level": family_confidence_level,
        "per_attack_confidence_interval_method": (
            "bonferroni_bounded_hoeffding"
        ),
        "per_attack_claim_p_value_method": (
            "bounded_hoeffding_prompt_cluster_mean_with_holm"
        ),
        "all_methods_clean_fixed_fpr_ready": (
            all_methods_clean_fixed_fpr_ready
        ),
        "main_method_clean_fixed_fpr_ready": next(
            row["clean_fixed_fpr_ready"]
            for row in canonical_operating
            if row["method_id"] == PROPOSED_METHOD_ID
        ),
        "main_method_wrong_key_fixed_fpr_ready": wrong_key_rows[0][
            "wrong_key_fixed_fpr_ready"
        ],
        "all_per_attack_fixed_fpr_ready": all_per_attack_fixed_fpr_ready,
        "universal_per_attack_superiority_claim_ready": (
            universal_per_attack_superiority_claim_ready
        ),
        "randomization_detection_statistics_ready": True,
        "supports_paper_claim": False,
    }
    summary_payload["randomization_detection_statistics_summary_digest"] = (
        build_stable_digest(summary_payload)
    )
    return (
        canonical_operating,
        canonical_per_attack,
        wrong_key_rows,
        canonical_comparisons,
        summary_payload,
    )


__all__ = [
    "ATTACKED_FALSE_POSITIVE_SCOPE",
    "ATTACKED_TRUE_POSITIVE_SCOPE",
    "CLEAN_FALSE_POSITIVE_SCOPE",
    "CLEAN_TRUE_POSITIVE_SCOPE",
    "DETECTION_METHOD_IDS",
    "PROPOSED_METHOD_ID",
    "RANDOMIZATION_DETECTION_CLUSTER_FIELDNAMES",
    "RANDOMIZATION_DETECTION_OPERATING_POINT_FIELDNAMES",
    "RANDOMIZATION_PER_ATTACK_COMPARISON_FIELDNAMES",
    "RANDOMIZATION_PER_ATTACK_DETECTION_FIELDNAMES",
    "RANDOMIZATION_WRONG_KEY_FIELDNAMES",
    "RandomizationDetectionStatisticsError",
    "WRONG_KEY_FALSE_POSITIVE_SCOPE",
    "build_randomization_detection_statistics",
]
