"""计算注册9重复上的 Prompt 聚类配对总体优势统计.

该模块只消费已经逐重复完成 fixed-FPR 阈值绑定的配对 outcome. 对每个
baseline 和 Prompt, 先在同一重复内平均完整攻击集合, 再平均9个注册重复;
bootstrap 与 Hoeffding 检验始终只把 Prompt 视为独立统计单位. 因此9重复
用于稳定估计同一 Prompt 的效应, 不会被展开为9倍独立样本.
"""

from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Any, Iterable, Mapping

import numpy as np

from experiments.protocol.formal_randomization import formal_randomization_repeat_ids
from experiments.protocol.paper_run_config import (
    RUN_EXPECTED_PROMPT_COUNTS,
    normalize_paper_run_name,
    validate_frozen_paper_run_target_fpr,
)
from experiments.protocol.splits import build_group_split_counts
from main.core.digest import build_stable_digest
from paper_experiments.analysis.paired_superiority import (
    BOOTSTRAP_BIT_GENERATOR,
    BOOTSTRAP_QUANTILE_METHOD,
    DEFAULT_CONFIDENCE_LEVEL,
    PRIMARY_BASELINE_IDS,
    SHARP_NULL_DIAGNOSTIC_METHOD,
    canonical_attack_registry_rows,
)


RANDOMIZATION_PAIRED_BOOTSTRAP_RESAMPLE_COUNT = 100_000
RANDOMIZATION_PAIRED_BOOTSTRAP_SCHEMA = (
    "paired_prompt_cluster_registered_repeat_mean_bootstrap_v1"
)
RANDOMIZATION_PAIRED_CLAIM_P_VALUE_METHOD = (
    "bounded_hoeffding_prompt_cluster_registered_repeat_mean"
)
RANDOMIZATION_PAIRED_STATISTICAL_UNIT = "prompt_cluster"
RANDOMIZATION_PAIRED_HOLM_FAMILY_ID = "overall_primary_baseline_superiority"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

_COMMON_CELL_FIELDS = (
    "randomization_repeat_id",
    "prompt_id",
    "generation_seed_index",
    "generation_seed_offset",
    "generation_seed_random",
    "watermark_key_index",
    "watermark_key_seed_random",
    "watermark_key_material_digest_random",
    "formal_randomization_protocol_digest",
    "formal_randomization_identity_digest_random",
    "base_latent_content_digest_random",
    "base_latent_identity_digest_random",
    "attack_id",
    "attack_family",
    "attack_name",
    "resource_profile",
    "attack_config_digest",
    "attack_seed_random",
    "formal_attack_seed_protocol_digest",
    "proposed_method_threshold_digest",
    "proposed_decision",
    "proposed_detector_digest",
    "proposed_attacked_image_digest",
)

RANDOMIZATION_PAIRED_SUPERIORITY_FIELDNAMES = (
    "analysis_scope",
    "baseline_id",
    "paired_prompt_count",
    "randomization_repeat_count",
    "paired_attack_count",
    "paired_observation_count",
    "statistical_unit",
    "mean_paired_true_positive_rate_difference",
    "mean_paired_difference_ci_low",
    "mean_paired_difference_ci_high",
    "positive_prompt_cluster_count",
    "negative_prompt_cluster_count",
    "tied_prompt_cluster_count",
    "one_sided_bounded_hoeffding_mean_p_value",
    "one_sided_exact_prompt_cluster_sign_flip_p_value",
    "exact_prompt_cluster_sign_flip_p_value_is_diagnostic",
    "claim_p_value_method",
    "sharp_null_diagnostic_method",
    "holm_family_id",
    "holm_family_size",
    "holm_adjusted_p_value",
    "confidence_level",
    "bootstrap_resample_count",
    "bootstrap_seed_digest_random",
    "bootstrap_analysis_schema",
    "bootstrap_bit_generator",
    "bootstrap_quantile_method",
    "proposed_method_repeat_threshold_map_digest",
    "baseline_method_repeat_threshold_map_digest",
    "registered_repeat_ids_digest",
    "paired_test_prompt_id_digest",
    "paired_attack_registry_digest",
    "paired_outcome_set_digest",
    "protocol_digest",
    "randomization_paired_statistics_ready",
    "paired_superiority_ready",
    "supports_paper_claim",
)


class RandomizationPairedSuperiorityError(ValueError):
    """表示跨重复配对输入不能形成公平的 Prompt 聚类统计."""


def _require_sha256(value: Any, field_name: str) -> str:
    """读取统计身份所需的小写 SHA-256."""

    resolved = str(value)
    if SHA256_PATTERN.fullmatch(resolved) is None:
        raise RandomizationPairedSuperiorityError(
            f"{field_name} 必须是小写 SHA-256"
        )
    return resolved


def _strict_bool(value: Any, field_name: str) -> bool:
    """拒绝把整数或文本静默解释为二元检测判定."""

    if not isinstance(value, bool):
        raise RandomizationPairedSuperiorityError(f"{field_name} 必须是布尔值")
    return value


def _canonical_outcome_digest(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], str]:
    """校验逐行摘要并按 repeat、baseline、Prompt、attack 规范排序."""

    repeat_order = {
        repeat_id: index
        for index, repeat_id in enumerate(formal_randomization_repeat_ids())
    }
    baseline_order = {
        baseline_id: index
        for index, baseline_id in enumerate(PRIMARY_BASELINE_IDS)
    }
    materialized: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(rows):
        row = dict(raw_row)
        declared_digest = _require_sha256(
            row.get("paired_outcome_digest", ""),
            "paired_outcome_digest",
        )
        digest_payload = {
            field_name: field_value
            for field_name, field_value in row.items()
            if field_name != "paired_outcome_digest"
        }
        if build_stable_digest(digest_payload) != declared_digest:
            raise RandomizationPairedSuperiorityError(
                f"paired outcome 摘要无法重建: row={row_index}"
            )
        materialized.append(row)
    canonical = tuple(
        sorted(
            materialized,
            key=lambda row: (
                repeat_order.get(str(row.get("randomization_repeat_id", "")), 99),
                baseline_order.get(str(row.get("baseline_id", "")), 99),
                str(row.get("prompt_id", "")),
                str(row.get("attack_id", "")),
            ),
        )
    )
    if not canonical:
        raise RandomizationPairedSuperiorityError("跨重复 paired outcome 不得为空")
    return canonical, build_stable_digest(canonical)


def _cluster_bootstrap_interval(
    prompt_values: np.ndarray,
    *,
    confidence_level: float,
    resample_count: int,
    seed: int,
) -> tuple[float, float]:
    """只重采样 Prompt, 每次整体移动其9重复和完整攻击块."""

    if prompt_values.ndim != 1 or prompt_values.size <= 0:
        raise RandomizationPairedSuperiorityError("bootstrap 需要非空 Prompt 差值")
    if not 0.0 < confidence_level < 1.0:
        raise RandomizationPairedSuperiorityError(
            "confidence_level 必须位于 (0, 1)"
        )
    if type(resample_count) is not int or resample_count <= 0:
        raise RandomizationPairedSuperiorityError(
            "bootstrap_resample_count 必须是正整数"
        )
    generator = np.random.Generator(np.random.PCG64(seed))
    estimates = np.empty(resample_count, dtype=np.float64)
    prompt_count = int(prompt_values.size)
    batch_size = 128
    for start in range(0, resample_count, batch_size):
        stop = min(start + batch_size, resample_count)
        indices = generator.integers(
            0,
            prompt_count,
            size=(stop - start, prompt_count),
            endpoint=False,
        )
        estimates[start:stop] = prompt_values[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    low, high = np.quantile(
        estimates,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method=BOOTSTRAP_QUANTILE_METHOD,
    )
    return float(low), float(high)


def _bounded_hoeffding_mean_p_value(prompt_values: np.ndarray) -> float:
    """计算范围为 [-1, 1] 的 Prompt 聚类均值单侧 Hoeffding 上界."""

    observed_mean = float(prompt_values.mean())
    if observed_mean <= 0.0:
        return 1.0
    return max(
        math.exp(-float(prompt_values.size) * observed_mean * observed_mean / 2.0),
        float(np.finfo(np.float64).tiny),
    )


def _exact_cluster_sign_flip_p_value(
    prompt_integer_sums: np.ndarray,
) -> float:
    """精确计算 Prompt 块 sign-flip 概率, 仅作为 sharp-null 诊断."""

    if prompt_integer_sums.ndim != 1 or prompt_integer_sums.size <= 0:
        raise RandomizationPairedSuperiorityError("sign-flip 需要非空 Prompt 整数和")
    observed_sum = int(prompt_integer_sums.sum())
    magnitudes = sorted(
        abs(int(value))
        for value in prompt_integer_sums.tolist()
        if int(value) != 0
    )
    distribution = np.asarray([1.0], dtype=np.float64)
    maximum_sum = 0
    for magnitude in magnitudes:
        updated = np.zeros(distribution.size + 2 * magnitude, dtype=np.float64)
        updated[: distribution.size] += 0.5 * distribution
        updated[2 * magnitude :] += 0.5 * distribution
        distribution = updated
        maximum_sum += magnitude
    threshold_index = max(0, observed_sum + maximum_sum)
    if threshold_index >= distribution.size:
        return float(np.finfo(np.float64).tiny)
    return min(
        max(
            float(distribution[threshold_index:].sum(dtype=np.float64)),
            float(np.finfo(np.float64).tiny),
        ),
        1.0,
    )


def _build_randomization_aggregate_paired_superiority_statistics(
    paired_outcomes: Iterable[Mapping[str, Any]],
    *,
    paper_run_name: str,
    target_fpr: float,
    protocol_digest: str,
    attack_registry_rows: Iterable[Mapping[str, Any]],
    confidence_level: float,
    bootstrap_resample_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """实现可由小重采样测试调用的精确9重复统计内核."""

    run_name = normalize_paper_run_name(paper_run_name)
    resolved_target_fpr = validate_frozen_paper_run_target_fpr(
        run_name,
        target_fpr,
    )
    resolved_protocol_digest = _require_sha256(protocol_digest, "protocol_digest")
    expected_prompt_count = build_group_split_counts(
        RUN_EXPECTED_PROMPT_COUNTS[run_name]
    )["test"]
    repeat_ids = formal_randomization_repeat_ids()
    repeat_id_set = set(repeat_ids)
    attack_registry = canonical_attack_registry_rows(attack_registry_rows)
    attack_by_id = {str(row["attack_id"]): dict(row) for row in attack_registry}
    attack_ids = tuple(sorted(attack_by_id))
    attack_id_set = set(attack_ids)
    canonical, outcome_set_digest = _canonical_outcome_digest(paired_outcomes)

    expected_keys: set[tuple[str, str, str, str]] = set()
    actual_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    prompt_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    attack_sets: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    common_cell_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    prompt_randomization_identity: dict[tuple[str, str], dict[str, Any]] = {}
    method_threshold_map: dict[str, dict[str, str]] = {
        repeat_id: {} for repeat_id in repeat_ids
    }

    for row in canonical:
        repeat_id = str(row.get("randomization_repeat_id", ""))
        baseline_id = str(row.get("baseline_id", ""))
        prompt_id = str(row.get("prompt_id", ""))
        attack_id = str(row.get("attack_id", ""))
        if repeat_id not in repeat_id_set:
            raise RandomizationPairedSuperiorityError("paired outcome 含未注册 repeat")
        if baseline_id not in PRIMARY_BASELINE_IDS:
            raise RandomizationPairedSuperiorityError("paired outcome 含未注册 baseline")
        if not prompt_id or attack_id not in attack_id_set:
            raise RandomizationPairedSuperiorityError("paired outcome 的 Prompt 或攻击无效")
        key = (repeat_id, baseline_id, prompt_id, attack_id)
        if key in actual_by_key:
            raise RandomizationPairedSuperiorityError("跨重复 paired outcome 键重复")
        actual_by_key[key] = row
        prompt_sets[(repeat_id, baseline_id)].add(prompt_id)
        attack_sets[(repeat_id, baseline_id, prompt_id)].add(attack_id)

        expected_attack = attack_by_id[attack_id]
        if any(
            str(row.get(field_name, "")) != str(expected_attack[field_name])
            for field_name in (
                "attack_family",
                "attack_name",
                "resource_profile",
                "attack_config_digest",
            )
        ):
            raise RandomizationPairedSuperiorityError(
                "paired outcome 攻击身份未匹配冻结 registry"
            )
        proposed = _strict_bool(row.get("proposed_decision"), "proposed_decision")
        baseline = _strict_bool(row.get("baseline_decision"), "baseline_decision")
        difference = row.get("paired_difference")
        if type(difference) is not int or difference != int(proposed) - int(baseline):
            raise RandomizationPairedSuperiorityError(
                "paired_difference 无法由二元判定重建"
            )
        for field_name in (
            "watermark_key_material_digest_random",
            "formal_randomization_protocol_digest",
            "formal_randomization_identity_digest_random",
            "base_latent_content_digest_random",
            "base_latent_identity_digest_random",
            "attack_config_digest",
            "formal_attack_seed_protocol_digest",
            "proposed_method_threshold_digest",
            "baseline_method_threshold_digest",
            "proposed_detector_digest",
            "proposed_attacked_image_digest",
            "baseline_attacked_image_digest",
        ):
            _require_sha256(row.get(field_name, ""), field_name)

        cell_key = (repeat_id, prompt_id, attack_id)
        cell_identity = {
            field_name: row.get(field_name) for field_name in _COMMON_CELL_FIELDS
        }
        previous_cell_identity = common_cell_identity.setdefault(
            cell_key,
            cell_identity,
        )
        if previous_cell_identity != cell_identity:
            raise RandomizationPairedSuperiorityError(
                "4个 baseline 未共享同一主方法判定、随机身份或攻击事实"
            )
        prompt_identity = {
            field_name: row.get(field_name)
            for field_name in _COMMON_CELL_FIELDS
            if field_name
            not in {
                "attack_id",
                "attack_family",
                "attack_name",
                "resource_profile",
                "attack_config_digest",
                "attack_seed_random",
                "formal_attack_seed_protocol_digest",
                "proposed_method_threshold_digest",
                "proposed_decision",
                "proposed_detector_digest",
                "proposed_attacked_image_digest",
            }
        }
        previous_prompt_identity = prompt_randomization_identity.setdefault(
            (repeat_id, prompt_id),
            prompt_identity,
        )
        if previous_prompt_identity != prompt_identity:
            raise RandomizationPairedSuperiorityError(
                "同一 repeat-Prompt 在攻击间随机身份发生漂移"
            )

        proposed_threshold = str(row["proposed_method_threshold_digest"])
        baseline_threshold = str(row["baseline_method_threshold_digest"])
        for method_id, threshold_digest in (
            ("slm_wm", proposed_threshold),
            (baseline_id, baseline_threshold),
        ):
            previous_threshold = method_threshold_map[repeat_id].setdefault(
                method_id,
                threshold_digest,
            )
            if previous_threshold != threshold_digest:
                raise RandomizationPairedSuperiorityError(
                    "同一 method-repeat 使用了多套 fixed-FPR 阈值"
                )

    common_prompt_sets = {frozenset(values) for values in prompt_sets.values()}
    if len(common_prompt_sets) != 1:
        raise RandomizationPairedSuperiorityError(
            "9重复与4个 baseline 未共享同一 test Prompt 集合"
        )
    prompt_ids = tuple(sorted(next(iter(common_prompt_sets))))
    if len(prompt_ids) != expected_prompt_count:
        raise RandomizationPairedSuperiorityError(
            "test Prompt 数量未匹配论文运行层级"
        )
    for repeat_id in repeat_ids:
        if set(method_threshold_map[repeat_id]) != {"slm_wm", *PRIMARY_BASELINE_IDS}:
            raise RandomizationPairedSuperiorityError(
                "方法阈值映射未精确覆盖9重复与5方法"
            )
        for baseline_id in PRIMARY_BASELINE_IDS:
            if prompt_sets.get((repeat_id, baseline_id)) != set(prompt_ids):
                raise RandomizationPairedSuperiorityError(
                    "method-repeat 未精确覆盖完整 test Prompt"
                )
            for prompt_id in prompt_ids:
                if attack_sets.get((repeat_id, baseline_id, prompt_id)) != attack_id_set:
                    raise RandomizationPairedSuperiorityError(
                        "method-repeat-Prompt 未精确覆盖完整攻击 registry"
                    )
                expected_keys.update(
                    (repeat_id, baseline_id, prompt_id, attack_id)
                    for attack_id in attack_ids
                )
    if set(actual_by_key) != expected_keys:
        raise RandomizationPairedSuperiorityError(
            "paired outcome 未精确覆盖9重复、4 baseline、Prompt 与攻击笛卡尔积"
        )

    prompt_id_digest = build_stable_digest(list(prompt_ids))
    attack_registry_digest = build_stable_digest(list(attack_registry))
    repeat_ids_digest = build_stable_digest(list(repeat_ids))
    threshold_map_digest = build_stable_digest(method_threshold_map)
    statistic_rows: list[dict[str, Any]] = []
    for baseline_id in PRIMARY_BASELINE_IDS:
        prompt_values = np.asarray(
            [
                float(
                    np.mean(
                        [
                            int(
                                actual_by_key[
                                    (repeat_id, baseline_id, prompt_id, attack_id)
                                ]["paired_difference"]
                            )
                            for repeat_id in repeat_ids
                            for attack_id in attack_ids
                        ]
                    )
                )
                for prompt_id in prompt_ids
            ],
            dtype=np.float64,
        )
        prompt_integer_sums = np.asarray(
            [
                sum(
                    int(
                        actual_by_key[
                            (repeat_id, baseline_id, prompt_id, attack_id)
                        ]["paired_difference"]
                    )
                    for repeat_id in repeat_ids
                    for attack_id in attack_ids
                )
                for prompt_id in prompt_ids
            ],
            dtype=np.int64,
        )
        bootstrap_seed_payload = {
            "analysis_schema": RANDOMIZATION_PAIRED_BOOTSTRAP_SCHEMA,
            "baseline_id": baseline_id,
            "paired_test_prompt_id_digest": prompt_id_digest,
            "paired_attack_registry_digest": attack_registry_digest,
            "registered_repeat_ids_digest": repeat_ids_digest,
            "paired_outcome_set_digest": outcome_set_digest,
            "confidence_level": float(confidence_level),
            "bootstrap_resample_count": int(bootstrap_resample_count),
            "bootstrap_bit_generator": BOOTSTRAP_BIT_GENERATOR,
            "bootstrap_quantile_method": BOOTSTRAP_QUANTILE_METHOD,
        }
        bootstrap_seed_digest = build_stable_digest(bootstrap_seed_payload)
        ci_low, ci_high = _cluster_bootstrap_interval(
            prompt_values,
            confidence_level=confidence_level,
            resample_count=bootstrap_resample_count,
            seed=int(bootstrap_seed_digest, 16),
        )
        statistic_rows.append(
            {
                "analysis_scope": "overall",
                "baseline_id": baseline_id,
                "paired_prompt_count": len(prompt_ids),
                "randomization_repeat_count": len(repeat_ids),
                "paired_attack_count": len(attack_ids),
                "paired_observation_count": len(prompt_ids)
                * len(repeat_ids)
                * len(attack_ids),
                "statistical_unit": RANDOMIZATION_PAIRED_STATISTICAL_UNIT,
                "mean_paired_true_positive_rate_difference": float(
                    prompt_values.mean()
                ),
                "mean_paired_difference_ci_low": ci_low,
                "mean_paired_difference_ci_high": ci_high,
                "positive_prompt_cluster_count": int(
                    np.count_nonzero(prompt_values > 0.0)
                ),
                "negative_prompt_cluster_count": int(
                    np.count_nonzero(prompt_values < 0.0)
                ),
                "tied_prompt_cluster_count": int(
                    np.count_nonzero(prompt_values == 0.0)
                ),
                "one_sided_bounded_hoeffding_mean_p_value": (
                    _bounded_hoeffding_mean_p_value(prompt_values)
                ),
                "one_sided_exact_prompt_cluster_sign_flip_p_value": (
                    _exact_cluster_sign_flip_p_value(prompt_integer_sums)
                ),
                "exact_prompt_cluster_sign_flip_p_value_is_diagnostic": True,
                "claim_p_value_method": RANDOMIZATION_PAIRED_CLAIM_P_VALUE_METHOD,
                "sharp_null_diagnostic_method": SHARP_NULL_DIAGNOSTIC_METHOD,
                "holm_family_id": RANDOMIZATION_PAIRED_HOLM_FAMILY_ID,
                "holm_family_size": len(PRIMARY_BASELINE_IDS),
                "confidence_level": float(confidence_level),
                "bootstrap_resample_count": int(bootstrap_resample_count),
                "bootstrap_seed_digest_random": bootstrap_seed_digest,
                "bootstrap_analysis_schema": RANDOMIZATION_PAIRED_BOOTSTRAP_SCHEMA,
                "bootstrap_bit_generator": BOOTSTRAP_BIT_GENERATOR,
                "bootstrap_quantile_method": BOOTSTRAP_QUANTILE_METHOD,
                "proposed_method_repeat_threshold_map_digest": build_stable_digest(
                    {
                        repeat_id: method_threshold_map[repeat_id]["slm_wm"]
                        for repeat_id in repeat_ids
                    }
                ),
                "baseline_method_repeat_threshold_map_digest": build_stable_digest(
                    {
                        repeat_id: method_threshold_map[repeat_id][baseline_id]
                        for repeat_id in repeat_ids
                    }
                ),
                "registered_repeat_ids_digest": repeat_ids_digest,
                "paired_test_prompt_id_digest": prompt_id_digest,
                "paired_attack_registry_digest": attack_registry_digest,
                "paired_outcome_set_digest": outcome_set_digest,
                "protocol_digest": resolved_protocol_digest,
                "randomization_paired_statistics_ready": True,
            }
        )

    ordered = sorted(
        enumerate(statistic_rows),
        key=lambda item: (
            float(item[1]["one_sided_bounded_hoeffding_mean_p_value"]),
            str(item[1]["baseline_id"]),
        ),
    )
    running_adjusted = 0.0
    family_size = len(statistic_rows)
    for rank, (original_index, row) in enumerate(ordered, start=1):
        adjusted = min(
            1.0,
            max(
                running_adjusted,
                float(row["one_sided_bounded_hoeffding_mean_p_value"])
                * (family_size - rank + 1),
            ),
        )
        running_adjusted = adjusted
        statistic_rows[original_index]["holm_adjusted_p_value"] = adjusted
    for row in statistic_rows:
        ready = bool(
            float(row["mean_paired_true_positive_rate_difference"]) > 0.0
            and float(row["mean_paired_difference_ci_low"]) > 0.0
            and float(row["holm_adjusted_p_value"]) < 0.05
        )
        row["paired_superiority_ready"] = ready
        # 该模块只完成全样本总体优势分量. 正式论文主张还必须合并
        # 质量匹配比较, 因而这里不得提前发布 claim-ready 结论.
        row["supports_paper_claim"] = False

    canonical_statistics = tuple(
        sorted(statistic_rows, key=lambda row: str(row["baseline_id"]))
    )
    ready_ids = [
        baseline_id
        for baseline_id in PRIMARY_BASELINE_IDS
        if any(
            row["baseline_id"] == baseline_id
            and row["paired_superiority_ready"] is True
            for row in canonical_statistics
        )
    ]
    overall_ready = len(ready_ids) == len(PRIMARY_BASELINE_IDS)
    summary_payload = {
        "summary_schema": "randomization_paired_superiority_summary",
        "paper_claim_scale": run_name,
        "target_fpr": resolved_target_fpr,
        "primary_baseline_ids": list(PRIMARY_BASELINE_IDS),
        "randomization_repeat_ids": list(repeat_ids),
        "randomization_repeat_count": len(repeat_ids),
        "paired_test_prompt_count": len(prompt_ids),
        "paired_test_prompt_id_digest": prompt_id_digest,
        "paired_attack_count": len(attack_ids),
        "paired_attack_registry_digest": attack_registry_digest,
        "paired_observation_count": len(canonical),
        "statistical_unit": RANDOMIZATION_PAIRED_STATISTICAL_UNIT,
        "method_repeat_threshold_digest_map": method_threshold_map,
        "method_repeat_threshold_map_digest": threshold_map_digest,
        "paired_outcome_set_digest": outcome_set_digest,
        "paired_superiority_rows_digest": build_stable_digest(canonical_statistics),
        "protocol_digest": resolved_protocol_digest,
        "confidence_level": float(confidence_level),
        "bootstrap_resample_count": int(bootstrap_resample_count),
        "bootstrap_analysis_schema": RANDOMIZATION_PAIRED_BOOTSTRAP_SCHEMA,
        "bootstrap_bit_generator": BOOTSTRAP_BIT_GENERATOR,
        "bootstrap_quantile_method": BOOTSTRAP_QUANTILE_METHOD,
        "claim_p_value_method": RANDOMIZATION_PAIRED_CLAIM_P_VALUE_METHOD,
        "sharp_null_diagnostic_method": SHARP_NULL_DIAGNOSTIC_METHOD,
        "holm_family_id": RANDOMIZATION_PAIRED_HOLM_FAMILY_ID,
        "holm_family_size": len(PRIMARY_BASELINE_IDS),
        "paired_superiority_ready_ids": ready_ids,
        "paired_superiority_exact_set_ready": True,
        "overall_paired_superiority_ready": overall_ready,
        "randomization_paired_statistics_ready": True,
        "conclusion_decision": (
            "all_sample_superiority_ready"
            if overall_ready
            else "measured_not_supported"
        ),
        "supports_paper_claim": False,
    }
    summary_payload["randomization_paired_superiority_summary_digest"] = (
        build_stable_digest(summary_payload)
    )
    return [dict(row) for row in canonical_statistics], summary_payload


def build_randomization_aggregate_paired_superiority_statistics(
    paired_outcomes: Iterable[Mapping[str, Any]],
    *,
    paper_run_name: str,
    target_fpr: float,
    protocol_digest: str,
    attack_registry_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """使用冻结100000次 bootstrap 构造正式9重复总体优势统计."""

    return _build_randomization_aggregate_paired_superiority_statistics(
        paired_outcomes,
        paper_run_name=paper_run_name,
        target_fpr=target_fpr,
        protocol_digest=protocol_digest,
        attack_registry_rows=attack_registry_rows,
        confidence_level=DEFAULT_CONFIDENCE_LEVEL,
        bootstrap_resample_count=RANDOMIZATION_PAIRED_BOOTSTRAP_RESAMPLE_COUNT,
    )


__all__ = [
    "RANDOMIZATION_PAIRED_BOOTSTRAP_RESAMPLE_COUNT",
    "RANDOMIZATION_PAIRED_BOOTSTRAP_SCHEMA",
    "RANDOMIZATION_PAIRED_CLAIM_P_VALUE_METHOD",
    "RANDOMIZATION_PAIRED_HOLM_FAMILY_ID",
    "RANDOMIZATION_PAIRED_STATISTICAL_UNIT",
    "RANDOMIZATION_PAIRED_SUPERIORITY_FIELDNAMES",
    "RandomizationPairedSuperiorityError",
    "build_randomization_aggregate_paired_superiority_statistics",
]
