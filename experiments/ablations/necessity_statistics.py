"""构建正式机制消融的 Prompt 配对必要性统计。

该模块只消费真实重运行产生的逐 Prompt 记录。完整方法与每个单机制变体
必须共享同一 test Prompt 集合，随后以 Prompt 为聚类单位比较完整方法减去
变体的攻击后检测率。协议闭合仅说明实验已经执行；只有预注册效应方向、
最小效应、置信区间、多重校正显著性和图像质量非劣性同时成立时，才支持
对应机制必要性主张。质量门禁用于排除完整方法仅依靠更大图像失真换取检测率
提升这一混杂解释。
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping

import numpy as np

from main.core.digest import build_stable_digest


ABLATION_NECESSITY_ANALYSIS_SCHEMA = "ablation_prompt_cluster_necessity_v1"
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
    "necessity_claim_supported",
    "necessity_claim_decision",
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


def _bootstrap_seed_digest_random(
    *,
    ablation_id: str,
    paired_prompt_id_digest: str,
    input_record_digest: str,
    minimum_effect_size: float,
    confidence_level: float,
    resample_count: int,
) -> tuple[str, int]:
    """由冻结协议和输入摘要派生无自由随机源。"""

    payload = {
        "analysis_schema": ABLATION_NECESSITY_ANALYSIS_SCHEMA,
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

    if not expected_ablation_ids or "complete_method" in expected_ablation_ids:
        raise AblationNecessityStatisticsError("expected_ablation_ids 只能包含非完整方法变体")
    if len(set(expected_ablation_ids)) != len(expected_ablation_ids):
        raise AblationNecessityStatisticsError("expected_ablation_ids 不得重复")
    if expected_paired_prompt_count <= 0:
        raise AblationNecessityStatisticsError(
            "expected_paired_prompt_count 必须为正整数"
        )
    if not 0.0 < minimum_effect_size < 1.0:
        raise AblationNecessityStatisticsError("minimum_effect_size 必须位于 (0, 1)")
    if not 0.0 < significance_alpha < 1.0:
        raise AblationNecessityStatisticsError("significance_alpha 必须位于 (0, 1)")

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

    rows: list[dict[str, Any]] = []
    for ablation_id in expected_ablation_ids:
        prompt_effects = np.asarray(
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
        )
        clean_true_positive_effects = np.asarray(
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
        )
        paired_ssim_effects = np.asarray(
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
        )
        seed_digest, seed = _bootstrap_seed_digest_random(
            ablation_id=ablation_id,
            paired_prompt_id_digest=paired_prompt_id_digest,
            input_record_digest=input_record_digest,
            minimum_effect_size=minimum_effect_size,
            confidence_level=confidence_level,
            resample_count=bootstrap_resample_count,
        )
        ci_low, ci_high = _cluster_bootstrap_interval(
            prompt_effects,
            confidence_level=confidence_level,
            resample_count=bootstrap_resample_count,
            seed=seed,
        )
        clean_ci_low, clean_ci_high = _cluster_bootstrap_interval(
            clean_true_positive_effects,
            confidence_level=confidence_level,
            resample_count=bootstrap_resample_count,
            seed=seed,
        )
        ssim_ci_low, ssim_ci_high = _cluster_bootstrap_interval(
            paired_ssim_effects,
            confidence_level=confidence_level,
            resample_count=bootstrap_resample_count,
            seed=seed,
        )
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
                "necessity_claim_supported": False,
                "necessity_claim_decision": "measured_not_supported",
                "confidence_level": float(confidence_level),
                "significance_alpha": float(significance_alpha),
                "bootstrap_resample_count": int(bootstrap_resample_count),
                "bootstrap_seed_digest_random": seed_digest,
                "bootstrap_analysis_schema": ABLATION_NECESSITY_ANALYSIS_SCHEMA,
                "bootstrap_bit_generator": ABLATION_NECESSITY_BOOTSTRAP_BIT_GENERATOR,
                "bootstrap_quantile_method": ABLATION_NECESSITY_BOOTSTRAP_QUANTILE_METHOD,
                "paired_p_value_method": ABLATION_NECESSITY_P_VALUE_METHOD,
                "paired_prompt_id_digest": paired_prompt_id_digest,
                "input_record_digest": input_record_digest,
                "supports_paper_claim": False,
            }
        )

    _apply_holm_adjustment(rows)
    for row in rows:
        adjusted_ready = float(row["holm_adjusted_p_value"]) < significance_alpha
        supported = bool(
            row["effect_direction_ready"]
            and row["minimum_effect_ready"]
            and row["confidence_interval_ready"]
            and adjusted_ready
            and row["paired_ssim_noninferiority_ready"]
        )
        row["adjusted_significance_ready"] = adjusted_ready
        row["necessity_claim_supported"] = supported
        row["necessity_claim_decision"] = (
            "measured_supported" if supported else "measured_not_supported"
        )
        row["supports_paper_claim"] = supported

    supported_ids = [
        str(row["ablation_id"]) for row in rows if row["necessity_claim_supported"]
    ]
    not_supported_ids = [
        str(row["ablation_id"])
        for row in rows
        if not row["necessity_claim_supported"]
    ]
    rows_digest = build_stable_digest(canonicalize_ablation_necessity_rows(rows))
    summary = {
        "construction_unit_name": "mechanism_necessity_statistics",
        "analysis_schema": ABLATION_NECESSITY_ANALYSIS_SCHEMA,
        "primary_metric_name": ABLATION_NECESSITY_PRIMARY_METRIC,
        "effect_direction": ABLATION_NECESSITY_EFFECT_DIRECTION,
        "minimum_effect_size": float(minimum_effect_size),
        "confidence_level": float(confidence_level),
        "paired_ssim_noninferiority_margin": (
            ABLATION_NECESSITY_PAIRED_SSIM_NONINFERIORITY_MARGIN
        ),
        "significance_alpha": float(significance_alpha),
        "bootstrap_resample_count": int(bootstrap_resample_count),
        "bootstrap_bit_generator": ABLATION_NECESSITY_BOOTSTRAP_BIT_GENERATOR,
        "bootstrap_quantile_method": ABLATION_NECESSITY_BOOTSTRAP_QUANTILE_METHOD,
        "paired_p_value_method": ABLATION_NECESSITY_P_VALUE_METHOD,
        "expected_variant_ablation_ids": list(expected_ablation_ids),
        "paired_prompt_count": len(prompt_ids),
        "expected_paired_prompt_count": expected_paired_prompt_count,
        "paired_prompt_id_digest": paired_prompt_id_digest,
        "input_record_digest": input_record_digest,
        "necessity_statistic_row_count": len(rows),
        "necessity_statistic_rows_digest": rows_digest,
        "necessity_supported_ablation_ids": supported_ids,
        "necessity_not_supported_ablation_ids": not_supported_ids,
        "ablation_necessity_statistics_ready": len(rows) == len(expected_ablation_ids),
        "all_mechanism_necessity_claims_supported": not not_supported_ids,
        "necessity_claim_decision": (
            "measured_supported" if not not_supported_ids else "measured_not_supported"
        ),
        "supports_paper_claim": not not_supported_ids,
    }
    return rows, summary
