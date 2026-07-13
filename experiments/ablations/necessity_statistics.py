"""构建正式机制消融的 Prompt 配对必要性统计。

该模块只消费真实重运行产生的逐 Prompt 记录。完整方法与每个单机制变体
必须共享同一 test Prompt 集合，随后以 Prompt 为聚类单位比较完整方法减去
变体的攻击后检测率。单 repeat 组件只记录预注册效应方向、最小效应、置信
区间、多重校正显著性和图像质量非劣性是否成立, 不直接支持论文结论。质量
门禁用于排除完整方法仅依靠更大图像失真换取检测率提升这一混杂解释。跨
repeat 机制必要性结论必须由外层精确9重复聚合器重新计算。
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping

import numpy as np

from experiments.protocol.formal_randomization import (
    formal_randomization_repeat_ids,
    formal_randomization_repeat_registry_digest,
)
from main.core.digest import build_stable_digest


ABLATION_NECESSITY_ANALYSIS_SCHEMA = "ablation_prompt_cluster_component_v2"
ABLATION_NECESSITY_AGGREGATE_ANALYSIS_SCHEMA = (
    "ablation_prompt_cluster_registered_repeat_mean"
)
ABLATION_NECESSITY_PRIMARY_METRIC = "attacked_true_positive_rate"
ABLATION_NECESSITY_EFFECT_DIRECTION = "complete_method_minus_ablation"
ABLATION_NECESSITY_MINIMUM_EFFECT_SIZE = 0.01
ABLATION_NECESSITY_CONFIDENCE_LEVEL = 0.95
ABLATION_NECESSITY_ALPHA = 0.05
ABLATION_NECESSITY_PAIRED_SSIM_NONINFERIORITY_MARGIN = 0.01
ABLATION_NECESSITY_BOOTSTRAP_RESAMPLE_COUNT = 100_000
ABLATION_NECESSITY_BOOTSTRAP_BIT_GENERATOR = "PCG64"
ABLATION_NECESSITY_BOOTSTRAP_QUANTILE_METHOD = "linear"
ABLATION_NECESSITY_P_VALUE_METHOD = (
    "one_sided_bounded_hoeffding_paired_prompt_minimum_effect"
)

ABLATION_NECESSITY_FIELDNAMES = (
    "ablation_id",
    "primary_metric_name",
    "effect_direction",
    "paired_prompt_count",
    "mean_paired_effect",
    "mean_paired_effect_ci_low",
    "mean_paired_effect_ci_high",
    "clean_true_positive_mean_paired_effect",
    "clean_true_positive_mean_paired_effect_ci_low",
    "clean_true_positive_mean_paired_effect_ci_high",
    "paired_ssim_mean_paired_effect",
    "paired_ssim_mean_paired_effect_ci_low",
    "paired_ssim_mean_paired_effect_ci_high",
    "paired_ssim_noninferiority_margin",
    "paired_ssim_noninferiority_ready",
    "minimum_effect_size",
    "one_sided_paired_p_value",
    "holm_adjusted_p_value",
    "effect_direction_ready",
    "minimum_effect_ready",
    "confidence_interval_ready",
    "adjusted_significance_ready",
    "necessity_component_supported",
    "necessity_component_decision",
    "confidence_level",
    "significance_alpha",
    "bootstrap_resample_count",
    "bootstrap_seed_digest_random",
    "bootstrap_analysis_schema",
    "bootstrap_bit_generator",
    "bootstrap_quantile_method",
    "paired_p_value_method",
    "paired_prompt_id_digest",
    "input_record_digest",
    "supports_paper_claim",
)


class AblationNecessityStatisticsError(ValueError):
    """表示消融记录不能形成唯一的正式 Prompt 配对统计。"""


def canonicalize_ablation_necessity_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """转换为 CSV 读回后的稳定字符串表示, 用于跨文件摘要校验。"""

    return [
        {field_name: str(row[field_name]) for field_name in ABLATION_NECESSITY_FIELDNAMES}
        for row in rows
    ]


def _finite_rate(record: Mapping[str, Any], field_name: str) -> float:
    """读取位于闭区间 [0, 1] 的有限比率。"""

    try:
        value = float(record[field_name])
    except (KeyError, TypeError, ValueError) as exc:
        raise AblationNecessityStatisticsError(
            f"消融记录缺少有限比率字段: {field_name}"
        ) from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise AblationNecessityStatisticsError(
            f"消融记录比率超出 [0, 1]: {field_name}={value}"
        )
    return value


def _boolean_outcome(record: Mapping[str, Any], field_name: str) -> float:
    """读取真实布尔检测结果, 拒绝字符串或数值伪装的布尔值。"""

    value = record.get(field_name)
    if not isinstance(value, bool):
        raise AblationNecessityStatisticsError(
            f"消融记录缺少布尔检测结果: {field_name}"
        )
    return float(value)


def _canonical_test_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_ablation_ids: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    """要求每个正式消融在同一 test Prompt 集合上恰有一条记录。"""

    materialized = tuple(dict(record) for record in records)
    allowed_ids = {"complete_method", *expected_ablation_ids}
    actual_ids = {str(record.get("ablation_id", "")) for record in materialized}
    if actual_ids != allowed_ids:
        raise AblationNecessityStatisticsError("消融记录未精确覆盖完整方法与全部正式变体")

    test_records = tuple(
        record for record in materialized if str(record.get("split", "")) == "test"
    )
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in test_records:
        ablation_id = str(record.get("ablation_id", ""))
        prompt_id = str(record.get("prompt_id", ""))
        if not prompt_id:
            raise AblationNecessityStatisticsError("正式消融 test 记录缺少 prompt_id")
        if prompt_id in grouped[ablation_id]:
            raise AblationNecessityStatisticsError(
                f"正式消融存在重复 Prompt 配对: {ablation_id}/{prompt_id}"
            )
        if record.get("formal_attack_coverage_ready") is not True:
            raise AblationNecessityStatisticsError(
                f"正式消融记录未覆盖完整攻击矩阵: {ablation_id}/{prompt_id}"
            )
        _finite_rate(record, "attacked_positive_rate")
        grouped[ablation_id][prompt_id] = record

    if set(grouped) != allowed_ids:
        raise AblationNecessityStatisticsError("test 记录未精确覆盖全部正式消融身份")
    prompt_sets = {frozenset(rows) for rows in grouped.values()}
    if len(prompt_sets) != 1 or not next(iter(prompt_sets), frozenset()):
        raise AblationNecessityStatisticsError("各消融变体未共享同一非空 test Prompt 集合")
    return tuple(
        sorted(
            test_records,
            key=lambda record: (
                str(record["ablation_id"]),
                str(record["prompt_id"]),
            ),
        )
    )


def _validate_necessity_parameters(
    *,
    expected_ablation_ids: tuple[str, ...],
    expected_paired_prompt_count: int,
    minimum_effect_size: float,
    significance_alpha: float,
) -> None:
    """集中校验单重复与跨重复统计共享的冻结参数。"""

    if not expected_ablation_ids or "complete_method" in expected_ablation_ids:
        raise AblationNecessityStatisticsError(
            "expected_ablation_ids 只能包含非完整方法变体"
        )
    if len(set(expected_ablation_ids)) != len(expected_ablation_ids):
        raise AblationNecessityStatisticsError("expected_ablation_ids 不得重复")
    if expected_paired_prompt_count <= 0:
        raise AblationNecessityStatisticsError(
            "expected_paired_prompt_count 必须为正整数"
        )
    if not 0.0 < minimum_effect_size < 1.0:
        raise AblationNecessityStatisticsError(
            "minimum_effect_size 必须位于 (0, 1)"
        )
    if not 0.0 < significance_alpha < 1.0:
        raise AblationNecessityStatisticsError(
            "significance_alpha 必须位于 (0, 1)"
        )


def _bootstrap_seed_digest_random(
    *,
    ablation_id: str,
    paired_prompt_id_digest: str,
    input_record_digest: str,
    minimum_effect_size: float,
    confidence_level: float,
    resample_count: int,
    analysis_schema: str = ABLATION_NECESSITY_ANALYSIS_SCHEMA,
    bootstrap_context_digest: str | None = None,
) -> tuple[str, int]:
    """由冻结协议和输入摘要派生无自由随机源。"""

    payload = {
        "analysis_schema": analysis_schema,
        "ablation_id": ablation_id,
        "primary_metric_name": ABLATION_NECESSITY_PRIMARY_METRIC,
        "effect_direction": ABLATION_NECESSITY_EFFECT_DIRECTION,
        "paired_prompt_id_digest": paired_prompt_id_digest,
        "input_record_digest": input_record_digest,
        "minimum_effect_size": float(minimum_effect_size),
        "paired_ssim_noninferiority_margin": (
            ABLATION_NECESSITY_PAIRED_SSIM_NONINFERIORITY_MARGIN
        ),
        "confidence_level": float(confidence_level),
        "bootstrap_resample_count": int(resample_count),
        "bootstrap_bit_generator": ABLATION_NECESSITY_BOOTSTRAP_BIT_GENERATOR,
        "bootstrap_quantile_method": ABLATION_NECESSITY_BOOTSTRAP_QUANTILE_METHOD,
    }
    if bootstrap_context_digest is not None:
        payload["bootstrap_context_digest"] = bootstrap_context_digest
    digest = build_stable_digest(payload)
    return digest, int(digest, 16)


def _cluster_bootstrap_interval(
    prompt_effects: np.ndarray,
    *,
    confidence_level: float,
    resample_count: int,
    seed: int,
) -> tuple[float, float]:
    """以 Prompt 为抽样单位计算确定性 percentile bootstrap 区间。"""

    if prompt_effects.ndim != 1 or prompt_effects.size <= 0:
        raise AblationNecessityStatisticsError("bootstrap 需要非空 Prompt 配对效应")
    if not 0.0 < confidence_level < 1.0 or resample_count <= 0:
        raise AblationNecessityStatisticsError("bootstrap 置信度和重采样次数无效")
    if np.all(prompt_effects == prompt_effects[0]):
        # 所有 Prompt 的配对效应完全相同时,任意有放回重采样的均值均相同.
        # 直接返回退化区间不会改变统计语义，同时避免闭合测试无意义地生成
        # 大量相同索引;该优化也可复用于真实数据中的退化诊断指标.
        value = float(prompt_effects[0])
        return value, value
    generator = np.random.Generator(np.random.PCG64(seed))
    estimates = np.empty(resample_count, dtype=np.float64)
    prompt_count = int(prompt_effects.size)
    batch_size = 128
    for start in range(0, resample_count, batch_size):
        stop = min(start + batch_size, resample_count)
        indices = generator.integers(
            0,
            prompt_count,
            size=(stop - start, prompt_count),
            endpoint=False,
        )
        estimates[start:stop] = prompt_effects[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    low, high = np.quantile(
        estimates,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method=ABLATION_NECESSITY_BOOTSTRAP_QUANTILE_METHOD,
    )
    return float(low), float(high)


def _shared_cluster_bootstrap_intervals(
    prompt_effect_groups: Mapping[str, np.ndarray],
    *,
    confidence_level: float,
    resample_count: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """用同一组 Prompt bootstrap 索引同时计算多项配对指标区间。

    三项正式消融指标共享同一 Prompt 聚类抽样单位和同一预注册 seed。一次
    生成索引后同时计算全部非退化指标,与分别重置同一 PCG64 seed 的结果完全
    等价,但避免重复生成两份相同的100000次 Prompt 索引矩阵。该结构可复用于
    任何需要在相同聚类抽样上比较多项配对指标的实验。
    """

    if not prompt_effect_groups:
        raise AblationNecessityStatisticsError("共享 bootstrap 至少需要一项配对指标")
    materialized = {
        field_name: np.asarray(values, dtype=np.float64)
        for field_name, values in prompt_effect_groups.items()
    }
    prompt_counts = {values.size for values in materialized.values()}
    if (
        len(prompt_counts) != 1
        or next(iter(prompt_counts), 0) <= 0
        or any(values.ndim != 1 for values in materialized.values())
    ):
        raise AblationNecessityStatisticsError(
            "共享 bootstrap 的全部指标必须具有相同非空一维 Prompt 宽度"
        )
    if not 0.0 < confidence_level < 1.0 or resample_count <= 0:
        raise AblationNecessityStatisticsError("bootstrap 置信度和重采样次数无效")

    intervals: dict[str, tuple[float, float]] = {}
    active_names: list[str] = []
    active_rows: list[np.ndarray] = []
    for field_name, values in materialized.items():
        if np.all(values == values[0]):
            value = float(values[0])
            intervals[field_name] = (value, value)
        else:
            active_names.append(field_name)
            active_rows.append(values)
    if not active_rows:
        return intervals

    generator = np.random.Generator(np.random.PCG64(seed))
    prompt_count = next(iter(prompt_counts))
    estimates = np.empty(
        (len(active_rows), resample_count),
        dtype=np.float64,
    )
    effect_matrix = np.stack(active_rows, axis=0)
    batch_size = 128
    for start in range(0, resample_count, batch_size):
        stop = min(start + batch_size, resample_count)
        indices = generator.integers(
            0,
            prompt_count,
            size=(stop - start, prompt_count),
            endpoint=False,
        )
        estimates[:, start:stop] = effect_matrix[:, indices].mean(axis=2)

    alpha = 1.0 - confidence_level
    for row_index, field_name in enumerate(active_names):
        low, high = np.quantile(
            estimates[row_index],
            [alpha / 2.0, 1.0 - alpha / 2.0],
            method=ABLATION_NECESSITY_BOOTSTRAP_QUANTILE_METHOD,
        )
        intervals[field_name] = (float(low), float(high))
    return intervals


def _minimum_effect_p_value(
    prompt_effects: np.ndarray,
    minimum_effect_size: float,
) -> float:
    """检验配对均值不超过预注册最小效应的单侧 Hoeffding 上界。"""

    shifted_mean = float(prompt_effects.mean()) - float(minimum_effect_size)
    if shifted_mean <= 0.0:
        return 1.0
    exponent = -float(prompt_effects.size) * shifted_mean * shifted_mean / 2.0
    return max(math.exp(exponent), float(np.finfo(np.float64).tiny))


def _apply_holm_adjustment(rows: list[dict[str, Any]]) -> None:
    """对全部单机制必要性检验执行 Holm 家族错误率校正。"""

    ordered = sorted(
        enumerate(rows),
        key=lambda item: (float(item[1]["one_sided_paired_p_value"]), item[1]["ablation_id"]),
    )
    running_adjusted = 0.0
    comparison_count = len(rows)
    for rank, (original_index, row) in enumerate(ordered, start=1):
        adjusted = min(
            1.0,
            max(
                running_adjusted,
                float(row["one_sided_paired_p_value"])
                * (comparison_count - rank + 1),
            ),
        )
        running_adjusted = adjusted
        rows[original_index]["holm_adjusted_p_value"] = adjusted


def _build_necessity_rows_from_prompt_effects(
    effect_groups_by_ablation: Mapping[str, Mapping[str, np.ndarray]],
    *,
    expected_ablation_ids: tuple[str, ...],
    prompt_ids: tuple[str, ...],
    paired_prompt_id_digest: str,
    input_record_digest: str,
    minimum_effect_size: float,
    confidence_level: float,
    significance_alpha: float,
    bootstrap_resample_count: int,
    analysis_schema: str,
    supports_paper_claim: bool | None,
    bootstrap_context_digest: str | None = None,
    summary_extensions: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """从已按 Prompt 聚合的三项配对效应构造统一统计表."""

    expected_metric_names = {
        "attacked_positive_rate",
        "clean_true_positive",
        "paired_ssim",
    }
    if set(effect_groups_by_ablation) != set(expected_ablation_ids):
        raise AblationNecessityStatisticsError("配对效应未精确覆盖全部正式消融")
    if any(
        set(effect_groups_by_ablation[ablation_id])
        != expected_metric_names
        for ablation_id in expected_ablation_ids
    ):
        raise AblationNecessityStatisticsError("配对效应未精确覆盖冻结指标集合")

    rows: list[dict[str, Any]] = []
    for ablation_id in expected_ablation_ids:
        prompt_effects = np.asarray(
            effect_groups_by_ablation[ablation_id][
                "attacked_positive_rate"
            ],
            dtype=np.float64,
        )
        clean_true_positive_effects = np.asarray(
            effect_groups_by_ablation[ablation_id][
                "clean_true_positive"
            ],
            dtype=np.float64,
        )
        paired_ssim_effects = np.asarray(
            effect_groups_by_ablation[ablation_id]["paired_ssim"],
            dtype=np.float64,
        )
        if any(
            values.ndim != 1 or values.size != len(prompt_ids)
            for values in (
                prompt_effects,
                clean_true_positive_effects,
                paired_ssim_effects,
            )
        ):
            raise AblationNecessityStatisticsError(
                "配对效应宽度未匹配规范 Prompt 集合"
            )
        seed_digest, seed = _bootstrap_seed_digest_random(
            ablation_id=ablation_id,
            paired_prompt_id_digest=paired_prompt_id_digest,
            input_record_digest=input_record_digest,
            minimum_effect_size=minimum_effect_size,
            confidence_level=confidence_level,
            resample_count=bootstrap_resample_count,
            analysis_schema=analysis_schema,
            bootstrap_context_digest=bootstrap_context_digest,
        )
        shared_intervals = _shared_cluster_bootstrap_intervals(
            {
                "attacked_positive_rate": prompt_effects,
                "clean_true_positive": clean_true_positive_effects,
                "paired_ssim": paired_ssim_effects,
            },
            confidence_level=confidence_level,
            resample_count=bootstrap_resample_count,
            seed=seed,
        )
        ci_low, ci_high = shared_intervals["attacked_positive_rate"]
        clean_ci_low, clean_ci_high = shared_intervals[
            "clean_true_positive"
        ]
        ssim_ci_low, ssim_ci_high = shared_intervals["paired_ssim"]
        mean_effect = float(prompt_effects.mean())
        rows.append(
            {
                "ablation_id": ablation_id,
                "primary_metric_name": ABLATION_NECESSITY_PRIMARY_METRIC,
                "effect_direction": ABLATION_NECESSITY_EFFECT_DIRECTION,
                "paired_prompt_count": len(prompt_ids),
                "mean_paired_effect": mean_effect,
                "mean_paired_effect_ci_low": ci_low,
                "mean_paired_effect_ci_high": ci_high,
                "clean_true_positive_mean_paired_effect": float(
                    clean_true_positive_effects.mean()
                ),
                "clean_true_positive_mean_paired_effect_ci_low": clean_ci_low,
                "clean_true_positive_mean_paired_effect_ci_high": clean_ci_high,
                "paired_ssim_mean_paired_effect": float(
                    paired_ssim_effects.mean()
                ),
                "paired_ssim_mean_paired_effect_ci_low": ssim_ci_low,
                "paired_ssim_mean_paired_effect_ci_high": ssim_ci_high,
                "paired_ssim_noninferiority_margin": (
                    ABLATION_NECESSITY_PAIRED_SSIM_NONINFERIORITY_MARGIN
                ),
                "paired_ssim_noninferiority_ready": (
                    ssim_ci_low
                    >= -ABLATION_NECESSITY_PAIRED_SSIM_NONINFERIORITY_MARGIN
                ),
                "minimum_effect_size": float(minimum_effect_size),
                "one_sided_paired_p_value": _minimum_effect_p_value(
                    prompt_effects,
                    minimum_effect_size,
                ),
                "holm_adjusted_p_value": 1.0,
                "effect_direction_ready": mean_effect > 0.0,
                "minimum_effect_ready": mean_effect >= minimum_effect_size,
                "confidence_interval_ready": ci_low > minimum_effect_size,
                "adjusted_significance_ready": False,
                "necessity_component_supported": False,
                "necessity_component_decision": "measured_not_supported",
                "confidence_level": float(confidence_level),
                "significance_alpha": float(significance_alpha),
                "bootstrap_resample_count": int(bootstrap_resample_count),
                "bootstrap_seed_digest_random": seed_digest,
                "bootstrap_analysis_schema": analysis_schema,
                "bootstrap_bit_generator": (
                    ABLATION_NECESSITY_BOOTSTRAP_BIT_GENERATOR
                ),
                "bootstrap_quantile_method": (
                    ABLATION_NECESSITY_BOOTSTRAP_QUANTILE_METHOD
                ),
                "paired_p_value_method": ABLATION_NECESSITY_P_VALUE_METHOD,
                "paired_prompt_id_digest": paired_prompt_id_digest,
                "input_record_digest": input_record_digest,
                "supports_paper_claim": bool(supports_paper_claim),
            }
        )

    _apply_holm_adjustment(rows)
    for row in rows:
        adjusted_ready = (
            float(row["holm_adjusted_p_value"]) < significance_alpha
        )
        supported = bool(
            row["effect_direction_ready"]
            and row["minimum_effect_ready"]
            and row["confidence_interval_ready"]
            and adjusted_ready
            and row["paired_ssim_noninferiority_ready"]
        )
        row["adjusted_significance_ready"] = adjusted_ready
        row["necessity_component_supported"] = supported
        row["necessity_component_decision"] = (
            "measured_supported" if supported else "measured_not_supported"
        )
        row["supports_paper_claim"] = (
            supported
            if supports_paper_claim is None
            else supports_paper_claim
        )

    supported_ids = [
        str(row["ablation_id"])
        for row in rows
        if row["necessity_component_supported"]
    ]
    not_supported_ids = [
        str(row["ablation_id"])
        for row in rows
        if not row["necessity_component_supported"]
    ]
    rows_digest = build_stable_digest(
        canonicalize_ablation_necessity_rows(rows)
    )
    summary = {
        "construction_unit_name": "mechanism_necessity_statistics",
        "analysis_schema": analysis_schema,
        "primary_metric_name": ABLATION_NECESSITY_PRIMARY_METRIC,
        "effect_direction": ABLATION_NECESSITY_EFFECT_DIRECTION,
        "minimum_effect_size": float(minimum_effect_size),
        "confidence_level": float(confidence_level),
        "paired_ssim_noninferiority_margin": (
            ABLATION_NECESSITY_PAIRED_SSIM_NONINFERIORITY_MARGIN
        ),
        "significance_alpha": float(significance_alpha),
        "bootstrap_resample_count": int(bootstrap_resample_count),
        "bootstrap_bit_generator": (
            ABLATION_NECESSITY_BOOTSTRAP_BIT_GENERATOR
        ),
        "bootstrap_quantile_method": (
            ABLATION_NECESSITY_BOOTSTRAP_QUANTILE_METHOD
        ),
        "paired_p_value_method": ABLATION_NECESSITY_P_VALUE_METHOD,
        "expected_variant_ablation_ids": list(expected_ablation_ids),
        "paired_prompt_count": len(prompt_ids),
        "expected_paired_prompt_count": len(prompt_ids),
        "paired_prompt_id_digest": paired_prompt_id_digest,
        "input_record_digest": input_record_digest,
        "necessity_statistic_row_count": len(rows),
        "necessity_statistic_rows_digest": rows_digest,
        "necessity_component_supported_ablation_ids": supported_ids,
        "necessity_component_not_supported_ablation_ids": not_supported_ids,
        "ablation_necessity_statistics_ready": (
            len(rows) == len(expected_ablation_ids)
        ),
        "all_mechanism_necessity_components_supported": (
            not not_supported_ids
        ),
        "necessity_component_decision": (
            "measured_supported"
            if not not_supported_ids
            else "measured_not_supported"
        ),
        "supports_paper_claim": (
            not not_supported_ids
            if supports_paper_claim is None
            else supports_paper_claim
        ),
        **dict(summary_extensions or {}),
    }
    return rows, summary


def build_ablation_necessity_statistics(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_ablation_ids: tuple[str, ...],
    expected_paired_prompt_count: int,
    minimum_effect_size: float = ABLATION_NECESSITY_MINIMUM_EFFECT_SIZE,
    confidence_level: float = ABLATION_NECESSITY_CONFIDENCE_LEVEL,
    significance_alpha: float = ABLATION_NECESSITY_ALPHA,
    bootstrap_resample_count: int = ABLATION_NECESSITY_BOOTSTRAP_RESAMPLE_COUNT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """从逐 Prompt 真实重运行记录构造必要性统计表和摘要。"""

    _validate_necessity_parameters(
        expected_ablation_ids=expected_ablation_ids,
        expected_paired_prompt_count=expected_paired_prompt_count,
        minimum_effect_size=minimum_effect_size,
        significance_alpha=significance_alpha,
    )

    canonical_records = _canonical_test_records(
        records,
        expected_ablation_ids=expected_ablation_ids,
    )
    input_record_digest = build_stable_digest(canonical_records)
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in canonical_records:
        grouped[str(record["ablation_id"])][str(record["prompt_id"])] = record
    prompt_ids = tuple(sorted(grouped["complete_method"]))
    if len(prompt_ids) != expected_paired_prompt_count:
        raise AblationNecessityStatisticsError(
            "配对 Prompt 数量与当前 paper_run test 规模不一致"
        )
    paired_prompt_id_digest = build_stable_digest(list(prompt_ids))

    effect_groups_by_ablation = {
        ablation_id: {
            "attacked_positive_rate": np.asarray(
                [
                    _finite_rate(
                        grouped["complete_method"][prompt_id],
                        "attacked_positive_rate",
                    )
                    - _finite_rate(
                        grouped[ablation_id][prompt_id],
                        "attacked_positive_rate",
                    )
                    for prompt_id in prompt_ids
                ],
                dtype=np.float64,
            ),
            "clean_true_positive": np.asarray(
                [
                    _boolean_outcome(
                        grouped["complete_method"][prompt_id],
                        "positive_source_positive",
                    )
                    - _boolean_outcome(
                        grouped[ablation_id][prompt_id],
                        "positive_source_positive",
                    )
                    for prompt_id in prompt_ids
                ],
                dtype=np.float64,
            ),
            "paired_ssim": np.asarray(
                [
                    _finite_rate(
                        grouped["complete_method"][prompt_id],
                        "paired_ssim",
                    )
                    - _finite_rate(
                        grouped[ablation_id][prompt_id],
                        "paired_ssim",
                    )
                    for prompt_id in prompt_ids
                ],
                dtype=np.float64,
            ),
        }
        for ablation_id in expected_ablation_ids
    }
    rows, summary = _build_necessity_rows_from_prompt_effects(
        effect_groups_by_ablation,
        expected_ablation_ids=expected_ablation_ids,
        prompt_ids=prompt_ids,
        paired_prompt_id_digest=paired_prompt_id_digest,
        input_record_digest=input_record_digest,
        minimum_effect_size=minimum_effect_size,
        confidence_level=confidence_level,
        significance_alpha=significance_alpha,
        bootstrap_resample_count=bootstrap_resample_count,
        analysis_schema=ABLATION_NECESSITY_ANALYSIS_SCHEMA,
        supports_paper_claim=False,
    )
    summary["expected_paired_prompt_count"] = expected_paired_prompt_count
    return rows, summary


def build_randomization_aggregate_ablation_necessity_statistics(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_ablation_ids: tuple[str, ...],
    expected_paired_prompt_count: int,
    minimum_effect_size: float = ABLATION_NECESSITY_MINIMUM_EFFECT_SIZE,
    confidence_level: float = ABLATION_NECESSITY_CONFIDENCE_LEVEL,
    significance_alpha: float = ABLATION_NECESSITY_ALPHA,
    bootstrap_resample_count: int = ABLATION_NECESSITY_BOOTSTRAP_RESAMPLE_COUNT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """从正式9重复原始记录构造以 Prompt 为聚类单位的必要性统计。

    该函数先在每个 ``(repeat, Prompt)`` 内计算完整方法减消融变体的
    配对效应, 再对同一 Prompt 的9个注册 repeat 求均值。bootstrap、
    Hoeffding 检验和 Holm 校正只消费每个 Prompt 的 repeat 均值, 因此
    统计样本量始终是 test Prompt 数, 不会把9个相关重复误当成9倍独立样本。

    这一聚合公式属于正式论文统计的项目特定实现。以 Prompt 为聚类单位、
    先做配对差再做重复均值的结构可复用于其他固定重复面板实验。
    """

    _validate_necessity_parameters(
        expected_ablation_ids=expected_ablation_ids,
        expected_paired_prompt_count=expected_paired_prompt_count,
        minimum_effect_size=minimum_effect_size,
        significance_alpha=significance_alpha,
    )

    repeat_ids = formal_randomization_repeat_ids()
    materialized = tuple(dict(record) for record in records)
    actual_repeat_ids = {
        str(record.get("randomization_repeat_id", ""))
        for record in materialized
    }
    if actual_repeat_ids != set(repeat_ids):
        raise AblationNecessityStatisticsError(
            "正式消融记录未精确覆盖注册的9个随机重复"
        )

    records_by_repeat: dict[str, list[dict[str, Any]]] = {
        repeat_id: [] for repeat_id in repeat_ids
    }
    for record in materialized:
        repeat_id = str(record.get("randomization_repeat_id", ""))
        records_by_repeat[repeat_id].append(record)

    canonical_by_repeat: dict[str, tuple[dict[str, Any], ...]] = {}
    grouped_by_repeat: dict[
        str,
        dict[str, dict[str, dict[str, Any]]],
    ] = {}
    prompt_identity_by_id: dict[str, tuple[int, str]] = {}
    for repeat_id in repeat_ids:
        canonical = _canonical_test_records(
            records_by_repeat[repeat_id],
            expected_ablation_ids=expected_ablation_ids,
        )
        canonical_by_repeat[repeat_id] = canonical
        grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for record in canonical:
            ablation_id = str(record["ablation_id"])
            prompt_id = str(record["prompt_id"])
            prompt_index = record.get("prompt_index")
            prompt_digest = record.get("prompt_digest")
            if type(prompt_index) is not int or not isinstance(
                prompt_digest,
                str,
            ) or not prompt_digest:
                raise AblationNecessityStatisticsError(
                    "正式消融 test 记录缺少规范 Prompt 索引或摘要"
                )
            prompt_identity = (prompt_index, prompt_digest)
            previous_identity = prompt_identity_by_id.setdefault(
                prompt_id,
                prompt_identity,
            )
            if previous_identity != prompt_identity:
                raise AblationNecessityStatisticsError(
                    f"跨重复 Prompt 身份不一致: {prompt_id}"
                )
            grouped[ablation_id][prompt_id] = record
        grouped_by_repeat[repeat_id] = grouped

    prompt_sets = {
        frozenset(grouped_by_repeat[repeat_id]["complete_method"])
        for repeat_id in repeat_ids
    }
    if len(prompt_sets) != 1 or not next(iter(prompt_sets), frozenset()):
        raise AblationNecessityStatisticsError(
            "9个正式重复未共享同一非空 test Prompt 集合"
        )
    prompt_ids = tuple(sorted(next(iter(prompt_sets))))
    if len(prompt_ids) != expected_paired_prompt_count:
        raise AblationNecessityStatisticsError(
            "配对 Prompt 数量与当前 paper_run test 规模不一致"
        )

    canonical_records = tuple(
        {
            **record,
            "randomization_repeat_id": repeat_id,
        }
        for repeat_id in repeat_ids
        for record in canonical_by_repeat[repeat_id]
    )
    input_record_digest = build_stable_digest(canonical_records)
    paired_prompt_id_digest = build_stable_digest(list(prompt_ids))

    effect_groups_by_ablation: dict[str, dict[str, np.ndarray]] = {}
    for ablation_id in expected_ablation_ids:
        attacked_effects: list[float] = []
        clean_effects: list[float] = []
        ssim_effects: list[float] = []
        for prompt_id in prompt_ids:
            attacked_by_repeat: list[float] = []
            clean_by_repeat: list[float] = []
            ssim_by_repeat: list[float] = []
            for repeat_id in repeat_ids:
                complete_record = grouped_by_repeat[repeat_id][
                    "complete_method"
                ][prompt_id]
                ablation_record = grouped_by_repeat[repeat_id][ablation_id][
                    prompt_id
                ]
                attacked_by_repeat.append(
                    _finite_rate(
                        complete_record,
                        "attacked_positive_rate",
                    )
                    - _finite_rate(
                        ablation_record,
                        "attacked_positive_rate",
                    )
                )
                clean_by_repeat.append(
                    _boolean_outcome(
                        complete_record,
                        "positive_source_positive",
                    )
                    - _boolean_outcome(
                        ablation_record,
                        "positive_source_positive",
                    )
                )
                ssim_by_repeat.append(
                    _finite_rate(complete_record, "paired_ssim")
                    - _finite_rate(ablation_record, "paired_ssim")
                )
            attacked_effects.append(float(np.mean(attacked_by_repeat)))
            clean_effects.append(float(np.mean(clean_by_repeat)))
            ssim_effects.append(float(np.mean(ssim_by_repeat)))
        effect_groups_by_ablation[ablation_id] = {
            "attacked_positive_rate": np.asarray(
                attacked_effects,
                dtype=np.float64,
            ),
            "clean_true_positive": np.asarray(
                clean_effects,
                dtype=np.float64,
            ),
            "paired_ssim": np.asarray(ssim_effects, dtype=np.float64),
        }

    repeat_registry_digest = formal_randomization_repeat_registry_digest()
    bootstrap_context_digest = build_stable_digest(
        {
            "randomization_repeat_ids": list(repeat_ids),
            "randomization_repeat_registry_digest": repeat_registry_digest,
        }
    )
    rows, summary = _build_necessity_rows_from_prompt_effects(
        effect_groups_by_ablation,
        expected_ablation_ids=expected_ablation_ids,
        prompt_ids=prompt_ids,
        paired_prompt_id_digest=paired_prompt_id_digest,
        input_record_digest=input_record_digest,
        minimum_effect_size=minimum_effect_size,
        confidence_level=confidence_level,
        significance_alpha=significance_alpha,
        bootstrap_resample_count=bootstrap_resample_count,
        analysis_schema=ABLATION_NECESSITY_AGGREGATE_ANALYSIS_SCHEMA,
        supports_paper_claim=None,
        bootstrap_context_digest=bootstrap_context_digest,
        summary_extensions={
            "randomization_repeat_ids": list(repeat_ids),
            "randomization_repeat_count": len(repeat_ids),
            "randomization_repeat_registry_digest": repeat_registry_digest,
            "bootstrap_context_digest": bootstrap_context_digest,
            "paired_observation_count": (
                len(repeat_ids) * len(prompt_ids)
            ),
            "repeat_aggregation_protocol": (
                "mean_within_prompt_across_registered_repeats_before_"
                "prompt_cluster_inference"
            ),
            "randomization_aggregate_statistics_ready": True,
            "randomization_aggregate_provenance_required": True,
        },
    )
    summary["expected_paired_prompt_count"] = expected_paired_prompt_count
    return rows, summary
