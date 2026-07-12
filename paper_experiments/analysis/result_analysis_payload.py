"""从正式记录重建论文结果分析 payload 并核验语义身份.

该模块同时约束两类互补证据. 字节绑定负责确认闭合包读取的是规范仓库路径
下的同一文件, 语义重建负责从逐样本记录重新计算每个表格单元和失败案例图.
二者必须同时通过, 从而避免 summary、manifest 与派生文件互相引用形成自证循环.
"""

from __future__ import annotations

import hashlib
import html
import math
from pathlib import Path
import posixpath
import re
from typing import Any, Iterable, Mapping, Sequence

from experiments.artifacts.attack_family_metrics import (
    ATTACK_FAMILY_METRIC_FIELDS,
    build_attack_family_metrics,
)
from main.core.digest import build_stable_digest
from paper_experiments.analysis.formal_record_statistics import (
    rebuild_and_validate_formal_fid_kid_metrics,
)
from paper_experiments.analysis.paired_superiority import PRIMARY_BASELINE_IDS
from paper_experiments.baselines import default_baseline_specs


RESULT_ANALYSIS_PAYLOAD_FILE_NAMES = {
    "main_confidence_interval_table": "confidence_interval_table.csv",
    "per_attack_superiority_table": "per_attack_superiority_table.csv",
    "failure_case_records": "failure_case_records.jsonl",
    "failure_case_figure": "failure_case_figure.svg",
}
FORMAL_FAILURE_CASE_LIMIT = 12
SUPERIORITY_MARGIN_ABSOLUTE_TOLERANCE = 1e-12
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

PROPOSED_METHOD_ID = "slm_wm_current"
FORMAL_METHOD_IDS = (PROPOSED_METHOD_ID, *PRIMARY_BASELINE_IDS)
MAIN_COMPARISON_FIELDNAMES = (
    "method_id",
    "method_role",
    "comparison_scope",
    "common_prompt_protocol_ready",
    "common_attack_protocol_ready",
    "common_threshold_protocol_ready",
    "metric_status",
    "true_positive_rate",
    "false_positive_rate",
    "clean_false_positive_rate",
    "attacked_false_positive_rate",
    "quality_score_mean",
    "supports_paper_claim",
)
CONFIDENCE_INTERVAL_FIELDNAMES = (
    "paper_claim_scale",
    "method_id",
    "attack_family",
    "attack_name",
    "resource_profile",
    "true_positive_rate",
    "true_positive_rate_ci_low",
    "true_positive_rate_ci_high",
    "false_positive_rate",
    "false_positive_rate_ci_low",
    "false_positive_rate_ci_high",
    "clean_false_positive_rate",
    "clean_false_positive_rate_ci_low",
    "clean_false_positive_rate_ci_high",
    "attacked_false_positive_rate",
    "attacked_false_positive_rate_ci_low",
    "attacked_false_positive_rate_ci_high",
    "positive_count",
    "negative_count",
    "confidence_interval_method",
    "confidence_level",
    "supports_paper_claim",
)
PER_ATTACK_SUPERIORITY_FIELDNAMES = (
    "attack_family",
    "attack_name",
    "slm_true_positive_rate",
    "slm_true_positive_rate_ci_low",
    "slm_true_positive_rate_ci_high",
    "best_baseline_id",
    "best_baseline_true_positive_rate",
    "best_baseline_true_positive_rate_ci_low",
    "best_baseline_true_positive_rate_ci_high",
    "slm_minus_best_baseline_tpr",
    "conservative_ci_margin",
    "superiority_claim_ready",
    "supports_paper_claim",
)
FAILURE_CASE_FIELDNAMES = (
    "failure_case_rank",
    "attack_family",
    "attack_name",
    "sample_role",
    "source_record_id",
    "attack_record_id",
    "aligned_content_score_after",
    "aligned_content_score_before",
    "score_retention",
    "evidence_decision",
    "attacked_image_path",
    "attacked_image_digest",
    "source_image_digest",
    "supports_paper_claim",
    "failure_case_record_digest",
)
PAIRED_SUPERIORITY_FIELDNAMES = (
    "baseline_id",
    "paired_prompt_count",
    "paired_attack_count",
    "paired_observation_count",
    "mean_paired_true_positive_rate_difference",
    "mean_paired_difference_ci_low",
    "mean_paired_difference_ci_high",
    "positive_prompt_cluster_count",
    "negative_prompt_cluster_count",
    "tied_prompt_cluster_count",
    "one_sided_bounded_hoeffding_mean_p_value",
    "one_sided_exact_prompt_cluster_sign_flip_p_value",
    "exact_prompt_cluster_sign_flip_p_value_is_diagnostic",
    "sharp_null_diagnostic_method",
    "claim_p_value_method",
    "holm_adjusted_p_value",
    "confidence_level",
    "bootstrap_resample_count",
    "bootstrap_seed_digest_random",
    "bootstrap_analysis_schema",
    "bootstrap_bit_generator",
    "bootstrap_quantile_method",
    "proposed_method_threshold_digest",
    "baseline_method_threshold_digest",
    "paired_test_prompt_id_digest",
    "paired_attack_registry_digest",
    "paired_outcome_set_digest",
    "protocol_digest",
    "paired_superiority_ready",
    "quality_matching_protocol_schema",
    "quality_matching_protocol_digest",
    "quality_metric_name",
    "quality_match_caliper",
    "minimum_matched_prompt_fraction",
    "total_quality_prompt_count",
    "minimum_matched_prompt_count",
    "matched_prompt_count",
    "unmatched_prompt_count",
    "matched_prompt_fraction",
    "proposed_embedding_pair_ssim_mean",
    "baseline_embedding_pair_ssim_mean",
    "mean_embedding_pair_ssim_gap",
    "max_absolute_embedding_pair_ssim_gap",
    "quality_match_coverage_ready",
    "quality_matched_observation_count",
    "quality_matched_mean_paired_true_positive_rate_difference",
    "quality_matched_mean_paired_difference_ci_low",
    "quality_matched_mean_paired_difference_ci_high",
    "quality_matched_holm_adjusted_p_value",
    "quality_matched_superiority_ready",
    "quality_matched_row_digest",
    "supports_paper_claim",
)


class ResultAnalysisSemanticError(ValueError):
    """表示论文结果分析 payload 无法由正式证据逐单元重建."""


def _strict_bool(value: Any, field_name: str) -> bool:
    """把 JSON/CSV 布尔值规整为无歧义的 Python 布尔值."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip() in {"True", "False"}:
        return value.strip() == "True"
    raise ResultAnalysisSemanticError(f"{field_name} 必须是严格布尔值")


def _finite_float(value: Any, field_name: str) -> float:
    """读取有限浮点值, 禁止布尔值借用数值语义."""

    if isinstance(value, bool):
        raise ResultAnalysisSemanticError(f"{field_name} 必须是有限数值")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ResultAnalysisSemanticError(f"{field_name} 必须是有限数值") from exc
    if not math.isfinite(resolved):
        raise ResultAnalysisSemanticError(f"{field_name} 必须是有限数值")
    return resolved


def _nonnegative_int(value: Any, field_name: str) -> int:
    """读取非负整数并拒绝非整型浮点文本."""

    if isinstance(value, bool):
        raise ResultAnalysisSemanticError(f"{field_name} 必须是非负整数")
    try:
        resolved = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ResultAnalysisSemanticError(f"{field_name} 必须是非负整数") from exc
    if not math.isfinite(numeric) or numeric != resolved or resolved < 0:
        raise ResultAnalysisSemanticError(f"{field_name} 必须是非负整数")
    return resolved


def _finite_or_unsupported(value: Any, field_name: str) -> float | str:
    """读取主比较表中的有限指标或显式 unsupported 标记."""

    if isinstance(value, str) and value.strip() == "unsupported":
        return "unsupported"
    return _finite_float(value, field_name)


def _finite_or_empty(value: Any, field_name: str) -> float | str:
    """读取失败记录中的可选数值, 空值保持为空文本."""

    if value in {None, ""}:
        return ""
    return _finite_float(value, field_name)


def _rows_with_exact_fields(
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
    role: str,
) -> tuple[dict[str, Any], ...]:
    """物化表格行并要求每行字段集合严格等于冻结 schema."""

    expected_fields = set(fieldnames)
    materialized = tuple(dict(row) for row in rows)
    for row_index, row in enumerate(materialized):
        if set(row) != expected_fields:
            missing = sorted(expected_fields - set(row))
            unexpected = sorted(set(row) - expected_fields)
            raise ResultAnalysisSemanticError(
                f"{role} 字段集合不一致: row={row_index}, "
                f"missing={missing}, unexpected={unexpected}"
            )
    return materialized


def _assert_float_equal(
    reported: float,
    rebuilt: float,
    *,
    role: str,
    row_index: int,
    field_name: str,
) -> None:
    """以仅覆盖 CSV 十进制往返误差的容差核对单个数值单元."""

    if not math.isclose(reported, rebuilt, rel_tol=1e-12, abs_tol=1e-12):
        raise ResultAnalysisSemanticError(
            f"{role} 与正式记录重建值不一致: row={row_index}, field={field_name}"
        )


def _weighted_mean(
    rows: Sequence[Mapping[str, Any]],
    value_field: str,
    weight_field: str,
) -> float:
    """按正式样本计数计算主表聚合均值."""

    weighted_sum = 0.0
    total_weight = 0
    for row in rows:
        weight = _nonnegative_int(row.get(weight_field), weight_field)
        if weight <= 0:
            raise ResultAnalysisSemanticError(
                f"{weight_field} 必须为正数才能聚合主比较表"
            )
        weighted_sum += _finite_float(row.get(value_field), value_field) * weight
        total_weight += weight
    if total_weight <= 0:
        raise ResultAnalysisSemanticError("主比较表聚合分母不得为0")
    return weighted_sum / total_weight


def build_confidence_interval_rows(
    result_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """从正式 result records 重建逐方法、逐攻击置信区间表."""

    rows: list[dict[str, Any]] = []
    for raw_record in result_records:
        record = dict(raw_record)
        if str(record.get("metric_status", "unsupported")) == "unsupported":
            continue
        rows.append(
            {field_name: record.get(field_name, "") for field_name in CONFIDENCE_INTERVAL_FIELDNAMES}
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["attack_family"]),
            str(row["attack_name"]),
            str(row["resource_profile"]),
            str(row["method_id"]),
        ),
    )


def build_per_attack_superiority_rows(
    result_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """从正式 result records 重建逐攻击最强 baseline 比较表."""

    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    identity_by_key: dict[tuple[str, str], tuple[str, str, str]] = {}
    for raw_record in result_records:
        record = dict(raw_record)
        if str(record.get("metric_status", "unsupported")) == "unsupported":
            continue
        key = (
            str(record.get("attack_family", "")),
            str(record.get("attack_name", "")),
        )
        identity = (
            str(record.get("attack_id", "")),
            str(record.get("resource_profile", "")),
            str(record.get("attack_config_digest", "")),
        )
        previous_identity = identity_by_key.setdefault(key, identity)
        if previous_identity != identity:
            raise ResultAnalysisSemanticError(
                "同一 attack_family/attack_name 绑定了多个正式攻击身份"
            )
        method_id = str(record.get("method_id", ""))
        if method_id in grouped.setdefault(key, {}):
            raise ResultAnalysisSemanticError("逐攻击比较输入包含重复 method 记录")
        grouped[key][method_id] = record

    rows: list[dict[str, Any]] = []
    for (attack_family, attack_name), method_records in sorted(grouped.items()):
        slm_record = method_records.get(PROPOSED_METHOD_ID)
        baseline_records = [
            method_records[method_id]
            for method_id in PRIMARY_BASELINE_IDS
            if method_id in method_records
        ]
        if slm_record is None or not baseline_records:
            continue
        # 逐攻击保守结论必须针对 CI 上界最大的 baseline, 而不是只针对点估计
        # 最大者。否则点估计略低但不确定性更大的方法会被错误排除在比较外。
        best_baseline = max(
            baseline_records,
            key=lambda row: (
                _finite_float(
                    row.get("true_positive_rate_ci_high"),
                    "true_positive_rate_ci_high",
                ),
                _finite_float(
                    row.get("true_positive_rate"), "true_positive_rate"
                ),
            ),
        )
        slm_tpr = _finite_float(
            slm_record.get("true_positive_rate"), "true_positive_rate"
        )
        best_tpr = _finite_float(
            best_baseline.get("true_positive_rate"), "true_positive_rate"
        )
        slm_ci_low = _finite_float(
            slm_record.get("true_positive_rate_ci_low"),
            "true_positive_rate_ci_low",
        )
        best_ci_high = _finite_float(
            best_baseline.get("true_positive_rate_ci_high"),
            "true_positive_rate_ci_high",
        )
        evidence_ready = bool(
            _strict_bool(
                slm_record.get("supports_paper_claim"),
                "supports_paper_claim",
            )
            and _strict_bool(
                best_baseline.get("supports_paper_claim"),
                "supports_paper_claim",
            )
        )
        conservative_margin = slm_ci_low - best_ci_high
        rows.append(
            {
                "attack_family": attack_family,
                "attack_name": attack_name,
                "slm_true_positive_rate": slm_tpr,
                "slm_true_positive_rate_ci_low": slm_record.get(
                    "true_positive_rate_ci_low", ""
                ),
                "slm_true_positive_rate_ci_high": slm_record.get(
                    "true_positive_rate_ci_high", ""
                ),
                "best_baseline_id": best_baseline.get("method_id", ""),
                "best_baseline_true_positive_rate": best_tpr,
                "best_baseline_true_positive_rate_ci_low": best_baseline.get(
                    "true_positive_rate_ci_low", ""
                ),
                "best_baseline_true_positive_rate_ci_high": best_baseline.get(
                    "true_positive_rate_ci_high", ""
                ),
                "slm_minus_best_baseline_tpr": slm_tpr - best_tpr,
                "conservative_ci_margin": conservative_margin,
                "superiority_claim_ready": bool(
                    evidence_ready
                    and conservative_margin > SUPERIORITY_MARGIN_ABSOLUTE_TOLERANCE
                ),
                "supports_paper_claim": evidence_ready,
            }
        )
    return rows


def _attacked_image_path(record: Mapping[str, Any]) -> str:
    """从攻击检测记录的 metadata 中读取受治理攻击图像路径."""

    metadata = record.get("metadata", {})
    return (
        str(metadata.get("attacked_image_path", ""))
        if isinstance(metadata, Mapping)
        else ""
    )


def build_failure_case_records(
    formal_detection_records: Iterable[Mapping[str, Any]],
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """确定性筛选实际 false-negative 记录, 不丢弃零失败这一负结果."""

    resolved_limit = _nonnegative_int(limit, "failure_case_limit")
    if resolved_limit < 1:
        raise ResultAnalysisSemanticError("正式失败案例上限必须为正整数")
    failures = [
        dict(record)
        for record in formal_detection_records
        if record.get("sample_role") == "positive_source"
        and record.get("evidence_decision") is False
    ]
    failures.sort(
        key=lambda row: (
            _finite_float(
                row.get("aligned_content_score_after"),
                "aligned_content_score_after",
            ),
            str(row.get("attack_family", "")),
            str(row.get("attack_name", "")),
            str(row.get("source_record_id", "")),
            str(row.get("attack_record_id", "")),
        )
    )
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(failures[:resolved_limit], start=1):
        payload = {
            "failure_case_rank": index,
            "attack_family": record.get("attack_family", ""),
            "attack_name": record.get("attack_name", ""),
            "sample_role": record.get("sample_role", ""),
            "source_record_id": record.get("source_record_id", ""),
            "attack_record_id": record.get("attack_record_id", ""),
            "aligned_content_score_after": record.get(
                "aligned_content_score_after", ""
            ),
            "aligned_content_score_before": record.get(
                "aligned_content_score_before", ""
            ),
            "score_retention": record.get("score_retention", ""),
            "evidence_decision": record.get("evidence_decision"),
            "attacked_image_path": _attacked_image_path(record),
            "attacked_image_digest": record.get("attacked_image_digest", ""),
            "source_image_digest": record.get("source_image_digest", ""),
            "supports_paper_claim": record.get("supports_paper_claim", False),
        }
        payload["failure_case_record_digest"] = build_stable_digest(payload)
        rows.append(payload)
    return rows


def _canonical_repository_relative_path(value: Any, field_name: str) -> str:
    """要求路径使用无反斜杠、无点段的规范仓库相对 POSIX 形式."""

    path_text = str(value)
    if (
        not path_text
        or "\\" in path_text
        or path_text.startswith("/")
        or re.match(r"^[A-Za-z]:", path_text)
        or posixpath.normpath(path_text) != path_text
        or path_text.startswith("../")
    ):
        raise ResultAnalysisSemanticError(f"{field_name} 不是规范仓库相对路径")
    return path_text


def build_failure_case_svg_text(
    failure_cases: Iterable[Mapping[str, Any]],
    *,
    failure_figure_path: str,
) -> str:
    """仅由受治理失败记录和规范图路径确定性重建 SVG 全文."""

    cases = tuple(dict(item) for item in failure_cases)
    figure_path = _canonical_repository_relative_path(
        failure_figure_path, "failure_figure_path"
    )
    figure_parent = posixpath.dirname(figure_path)
    card_width = 260
    card_height = 250
    columns = 3
    row_count = max(1, (len(cases) + columns - 1) // columns)
    width = columns * card_width + 40
    height = row_count * card_height + 80
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial, sans-serif;} .title{font-size:18px;font-weight:700;} '
        '.label{font-size:11px;} .small{font-size:10px;fill:#333;} .card{fill:#fff;stroke:#bbb;stroke-width:1;}</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="28" class="title">SLM-WM failure cases under fixed-FPR attack protocol</text>',
        '<text x="20" y="48" class="small">Each panel renders the attacked image referenced by its governed detection record.</text>',
    ]
    if not cases:
        svg_parts.append(
            '<text x="20" y="82" class="label">No false-negative case was observed in the governed detection records.</text>'
        )
    for index, item in enumerate(cases):
        image_path = _canonical_repository_relative_path(
            item.get("attacked_image_path", ""), "attacked_image_path"
        )
        href = posixpath.relpath(image_path, start=figure_parent)
        column = index % columns
        row = index // columns
        x = 20 + column * card_width
        y = 65 + row * card_height
        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{card_width - 16}" height="{card_height - 14}" class="card"/>'
        )
        svg_parts.append(
            f'<image href="{html.escape(href)}" x="{x + 12}" y="{y + 12}" width="220" height="150" '
            'preserveAspectRatio="xMidYMid meet"/>'
        )
        svg_parts.append(
            f'<text x="{x + 12}" y="{y + 178}" class="label">#{item["failure_case_rank"]} '
            f'{html.escape(str(item.get("attack_name", "")))}</text>'
        )
        svg_parts.append(
            f'<text x="{x + 12}" y="{y + 195}" class="small">score='
            f'{html.escape(str(item.get("aligned_content_score_after", "")))[:12]} '
            f'retention={html.escape(str(item.get("score_retention", "")))[:12]}</text>'
        )
        svg_parts.append(
            f'<text x="{x + 12}" y="{y + 212}" class="small">digest='
            f'{html.escape(str(item.get("attacked_image_digest", "")))[:18]}</text>'
        )
    svg_parts.append("</svg>")
    return "\n".join(svg_parts) + "\n"


def build_main_comparison_rows_from_result_records(
    result_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """从五种正式方法的逐攻击 records 重建论文主比较表."""

    records = tuple(dict(record) for record in result_records)
    if not records:
        raise ResultAnalysisSemanticError("主比较表重建缺少正式 result records")
    grouped: dict[str, list[dict[str, Any]]] = {
        method_id: [] for method_id in FORMAL_METHOD_IDS
    }
    keys_by_method: dict[str, set[tuple[str, str, str, str]]] = {
        method_id: set() for method_id in FORMAL_METHOD_IDS
    }
    claim_values: list[bool] = []
    scale_values: set[str] = set()
    for record in records:
        method_id = str(record.get("method_id", ""))
        if method_id not in grouped:
            raise ResultAnalysisSemanticError(
                f"正式 result records 包含未登记方法: {method_id}"
            )
        if str(record.get("metric_status", "unsupported")) == "unsupported":
            raise ResultAnalysisSemanticError("主比较表不得由 unsupported 记录重建")
        key = (
            str(record.get("attack_id", "")),
            str(record.get("attack_family", "")),
            str(record.get("attack_name", "")),
            str(record.get("resource_profile", "")),
        )
        if not all(key):
            raise ResultAnalysisSemanticError("正式 result record 缺少完整攻击身份")
        if key in keys_by_method[method_id]:
            raise ResultAnalysisSemanticError("正式 result records 包含重复方法攻击键")
        keys_by_method[method_id].add(key)
        grouped[method_id].append(record)
        claim_values.append(
            _strict_bool(
                record.get("supports_paper_claim"), "supports_paper_claim"
            )
        )
        scale_values.add(str(record.get("paper_claim_scale", "")))
    if len(scale_values) != 1 or "" in scale_values:
        raise ResultAnalysisSemanticError("正式 result records 的论文层级不唯一")
    common_attack_sets = {frozenset(keys) for keys in keys_by_method.values()}
    if len(common_attack_sets) != 1 or not next(iter(common_attack_sets)):
        raise ResultAnalysisSemanticError("五种正式方法未覆盖同一非空攻击集合")
    comparison_claim_ready = all(claim_values)
    if not comparison_claim_ready:
        raise ResultAnalysisSemanticError(
            "主比较表重建只接受已通过正式证据门禁的 result records"
        )

    def aggregate(method_id: str) -> dict[str, float]:
        method_rows = grouped[method_id]
        return {
            "true_positive_rate": _weighted_mean(
                method_rows, "true_positive_rate", "positive_count"
            ),
            "false_positive_rate": _weighted_mean(
                method_rows, "false_positive_rate", "negative_count"
            ),
            "clean_false_positive_rate": _weighted_mean(
                method_rows, "clean_false_positive_rate", "negative_count"
            ),
            "attacked_false_positive_rate": _weighted_mean(
                method_rows,
                "attacked_false_positive_rate",
                "negative_count",
            ),
            "quality_score_mean": _weighted_mean(
                method_rows, "quality_score_mean", "supported_record_count"
            ),
        }

    rows = [
        {
            "method_id": PROPOSED_METHOD_ID,
            "method_role": "proposed_method_governed_result",
            "comparison_scope": "common_protocol_governed_result",
            "common_prompt_protocol_ready": True,
            "common_attack_protocol_ready": True,
            "common_threshold_protocol_ready": True,
            "metric_status": "measured_from_attack_matrix_formal_records",
            **aggregate(PROPOSED_METHOD_ID),
            "supports_paper_claim": True,
        }
    ]
    specs = {spec.baseline_id: spec for spec in default_baseline_specs()}
    if set(PRIMARY_BASELINE_IDS) - set(specs):
        raise ResultAnalysisSemanticError("主表 baseline 登记不完整")
    for baseline_id in sorted(specs):
        spec = specs[baseline_id]
        if baseline_id in PRIMARY_BASELINE_IDS:
            rows.append(
                {
                    "method_id": baseline_id,
                    "method_role": "external_baseline_primary",
                    "comparison_scope": "common_protocol_governed_result",
                    "common_prompt_protocol_ready": True,
                    "common_attack_protocol_ready": True,
                    "common_threshold_protocol_ready": True,
                    "metric_status": "measured",
                    **aggregate(baseline_id),
                    "supports_paper_claim": True,
                }
            )
        else:
            rows.append(
                {
                    "method_id": baseline_id,
                    "method_role": f"external_baseline_{spec.comparison_group}",
                    "comparison_scope": "common_protocol_result_missing",
                    "common_prompt_protocol_ready": True,
                    "common_attack_protocol_ready": True,
                    "common_threshold_protocol_ready": True,
                    "metric_status": "unsupported",
                    "true_positive_rate": "unsupported",
                    "false_positive_rate": "unsupported",
                    "clean_false_positive_rate": "unsupported",
                    "attacked_false_positive_rate": "unsupported",
                    "quality_score_mean": "unsupported",
                    "supports_paper_claim": False,
                }
            )
    return rows


def _normalize_main_comparison_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """规整主比较表用于逐字段语义比较."""

    materialized = _rows_with_exact_fields(
        rows, MAIN_COMPARISON_FIELDNAMES, "主比较表"
    )
    normalized = []
    for row in materialized:
        normalized.append(
            {
                **{
                    field_name: str(row[field_name])
                    for field_name in (
                        "method_id",
                        "method_role",
                        "comparison_scope",
                        "metric_status",
                    )
                },
                **{
                    field_name: _strict_bool(row[field_name], field_name)
                    for field_name in (
                        "common_prompt_protocol_ready",
                        "common_attack_protocol_ready",
                        "common_threshold_protocol_ready",
                        "supports_paper_claim",
                    )
                },
                **{
                    field_name: _finite_or_unsupported(row[field_name], field_name)
                    for field_name in (
                        "true_positive_rate",
                        "false_positive_rate",
                        "clean_false_positive_rate",
                        "attacked_false_positive_rate",
                        "quality_score_mean",
                    )
                },
            }
        )
    return tuple(normalized)


def _validate_main_comparison_rows(
    result_records: Iterable[Mapping[str, Any]],
    reported_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """逐单元核对主比较表与正式 result records 聚合值."""

    rebuilt = _normalize_main_comparison_rows(
        build_main_comparison_rows_from_result_records(result_records)
    )
    reported = _normalize_main_comparison_rows(reported_rows)
    if len(reported) != len(rebuilt):
        raise ResultAnalysisSemanticError("主比较表行数与重建结果不一致")
    for row_index, (actual, expected) in enumerate(
        zip(reported, rebuilt, strict=True)
    ):
        for field_name in MAIN_COMPARISON_FIELDNAMES:
            actual_value = actual[field_name]
            expected_value = expected[field_name]
            if isinstance(expected_value, float):
                if not isinstance(actual_value, float):
                    raise ResultAnalysisSemanticError(
                        f"主比较表数值单元类型不一致: row={row_index}, field={field_name}"
                    )
                _assert_float_equal(
                    actual_value,
                    expected_value,
                    role="主比较表",
                    row_index=row_index,
                    field_name=field_name,
                )
            elif actual_value != expected_value:
                raise ResultAnalysisSemanticError(
                    f"主比较表与正式记录重建值不一致: row={row_index}, field={field_name}"
                )
    return {
        "main_comparison_rebuilt_rows_digest": build_stable_digest(rebuilt),
        "main_comparison_semantic_rebuild_ready": True,
    }


def _normalize_attack_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """按攻击身份规整正式攻击指标表."""

    materialized = _rows_with_exact_fields(
        rows, ATTACK_FAMILY_METRIC_FIELDS, "攻击指标表"
    )
    text_fields = {
        "attack_id",
        "attack_family",
        "attack_name",
        "resource_profile",
        "attack_config_digest",
        "metric_status",
    }
    integer_fields = {
        "attack_record_count",
        "supported_record_count",
        "unsupported_record_count",
        "positive_count",
        "negative_count",
    }
    boolean_fields = {"fixed_fpr_upper_bound_ready", "supports_paper_claim"}
    normalized = []
    for row in materialized:
        payload: dict[str, Any] = {}
        for field_name in ATTACK_FAMILY_METRIC_FIELDS:
            if field_name in text_fields:
                payload[field_name] = str(row[field_name])
            elif field_name in integer_fields:
                payload[field_name] = _nonnegative_int(row[field_name], field_name)
            elif field_name in boolean_fields:
                payload[field_name] = _strict_bool(row[field_name], field_name)
            else:
                payload[field_name] = _finite_float(row[field_name], field_name)
        normalized.append(payload)
    return tuple(
        sorted(
            normalized,
            key=lambda row: (
                row["attack_family"],
                row["attack_name"],
                row["resource_profile"],
                row["attack_id"],
            ),
        )
    )


def _validate_attack_rows(
    attack_detection_records: Iterable[Mapping[str, Any]],
    reported_rows: Iterable[Mapping[str, Any]],
    *,
    target_fpr: float,
) -> dict[str, Any]:
    """从逐样本攻击检测记录重算攻击表并逐单元核对."""

    records = tuple(dict(record) for record in attack_detection_records)
    if not records:
        raise ResultAnalysisSemanticError("攻击指标表重建缺少正式检测记录")
    supports_claim = all(
        _strict_bool(record.get("supports_paper_claim"), "supports_paper_claim")
        for record in records
    )
    try:
        rebuilt_raw = build_attack_family_metrics(
            records,
            _finite_float(target_fpr, "target_fpr"),
            supports_claim,
        )
    except (TypeError, ValueError) as exc:
        raise ResultAnalysisSemanticError(
            "攻击检测 records 无法重建正式攻击指标表"
        ) from exc
    rebuilt = _normalize_attack_rows(rebuilt_raw)
    reported = _normalize_attack_rows(reported_rows)
    if len(reported) != len(rebuilt):
        raise ResultAnalysisSemanticError("攻击指标表行数与重建结果不一致")
    for row_index, (actual, expected) in enumerate(
        zip(reported, rebuilt, strict=True)
    ):
        for field_name in ATTACK_FAMILY_METRIC_FIELDS:
            if isinstance(expected[field_name], float):
                _assert_float_equal(
                    float(actual[field_name]),
                    expected[field_name],
                    role="攻击指标表",
                    row_index=row_index,
                    field_name=field_name,
                )
            elif actual[field_name] != expected[field_name]:
                raise ResultAnalysisSemanticError(
                    f"攻击指标表与检测 records 重建值不一致: row={row_index}, field={field_name}"
                )
    return {
        "attack_table_rebuilt_rows_digest": build_stable_digest(rebuilt),
        "attack_table_semantic_rebuild_ready": True,
    }


def _validate_slm_result_attack_consistency(
    result_records: Iterable[Mapping[str, Any]],
    attack_family_metrics: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """核对主方法 result records 与逐样本重建攻击表的共同指标."""

    slm_rows = {
        (
            str(row.get("attack_id", "")),
            str(row.get("attack_family", "")),
            str(row.get("attack_name", "")),
            str(row.get("resource_profile", "")),
            str(row.get("attack_config_digest", "")),
        ): dict(row)
        for row in result_records
        if str(row.get("method_id", "")) == PROPOSED_METHOD_ID
    }
    attack_rows = {
        (
            str(row.get("attack_id", "")),
            str(row.get("attack_family", "")),
            str(row.get("attack_name", "")),
            str(row.get("resource_profile", "")),
            str(row.get("attack_config_digest", "")),
        ): dict(row)
        for row in attack_family_metrics
    }
    if not slm_rows or set(slm_rows) != set(attack_rows):
        raise ResultAnalysisSemanticError(
            "主方法 result records 与攻击指标表的攻击身份集合不一致"
        )
    compared_fields = (
        "positive_count",
        "negative_count",
        "true_positive_rate",
        "attacked_false_positive_rate",
        "quality_score_mean",
    )
    canonical = []
    for key in sorted(slm_rows):
        result_row = slm_rows[key]
        attack_row = attack_rows[key]
        for field_name in compared_fields:
            result_value = _finite_float(result_row.get(field_name), field_name)
            attack_value = _finite_float(attack_row.get(field_name), field_name)
            _assert_float_equal(
                result_value,
                attack_value,
                role="主方法 result/attack 交叉核对",
                row_index=len(canonical),
                field_name=field_name,
            )
        canonical.append(
            {
                "attack_identity": key,
                **{
                    field_name: _finite_float(
                        result_row.get(field_name), field_name
                    )
                    for field_name in compared_fields
                },
            }
        )
    return {
        "slm_result_attack_consistency_digest": build_stable_digest(canonical),
        "slm_result_attack_consistency_ready": True,
    }


def _normalize_confidence_interval_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """规整 CI 表并核验每个区间包含对应点估计."""

    materialized = _rows_with_exact_fields(
        rows, CONFIDENCE_INTERVAL_FIELDNAMES, "置信区间表"
    )
    numeric_fields = set(CONFIDENCE_INTERVAL_FIELDNAMES[5:17]) | {
        "confidence_level"
    }
    normalized = []
    for row in materialized:
        payload: dict[str, Any] = {}
        for field_name in CONFIDENCE_INTERVAL_FIELDNAMES:
            if field_name in numeric_fields:
                payload[field_name] = _finite_float(row[field_name], field_name)
            elif field_name in {"positive_count", "negative_count"}:
                payload[field_name] = _nonnegative_int(row[field_name], field_name)
            elif field_name == "supports_paper_claim":
                payload[field_name] = _strict_bool(row[field_name], field_name)
            else:
                payload[field_name] = str(row[field_name])
        for low_name, value_name, high_name in (
            (
                "true_positive_rate_ci_low",
                "true_positive_rate",
                "true_positive_rate_ci_high",
            ),
            (
                "false_positive_rate_ci_low",
                "false_positive_rate",
                "false_positive_rate_ci_high",
            ),
            (
                "clean_false_positive_rate_ci_low",
                "clean_false_positive_rate",
                "clean_false_positive_rate_ci_high",
            ),
            (
                "attacked_false_positive_rate_ci_low",
                "attacked_false_positive_rate",
                "attacked_false_positive_rate_ci_high",
            ),
        ):
            if not 0.0 <= payload[low_name] <= payload[value_name] <= payload[high_name] <= 1.0:
                raise ResultAnalysisSemanticError("置信区间表包含无效区间")
        normalized.append(payload)
    return tuple(
        sorted(
            normalized,
            key=lambda row: (
                row["attack_family"],
                row["attack_name"],
                row["resource_profile"],
                row["method_id"],
            ),
        )
    )


def _normalize_per_attack_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """规整逐攻击比较表并保留未胜出的真实负结果."""

    materialized = _rows_with_exact_fields(
        rows, PER_ATTACK_SUPERIORITY_FIELDNAMES, "逐攻击比较表"
    )
    numeric_fields = set(PER_ATTACK_SUPERIORITY_FIELDNAMES[2:5]) | set(
        PER_ATTACK_SUPERIORITY_FIELDNAMES[6:11]
    )
    normalized = []
    for row in materialized:
        payload: dict[str, Any] = {}
        for field_name in PER_ATTACK_SUPERIORITY_FIELDNAMES:
            if field_name in numeric_fields:
                payload[field_name] = _finite_float(row[field_name], field_name)
            elif field_name in {"superiority_claim_ready", "supports_paper_claim"}:
                payload[field_name] = _strict_bool(row[field_name], field_name)
            else:
                payload[field_name] = str(row[field_name])
        expected_ready = bool(
            payload["supports_paper_claim"]
            and payload["conservative_ci_margin"]
            > SUPERIORITY_MARGIN_ABSOLUTE_TOLERANCE
        )
        if payload["superiority_claim_ready"] != expected_ready:
            raise ResultAnalysisSemanticError(
                "逐攻击比较表的胜出标记与保守 CI margin 不一致"
            )
        normalized.append(payload)
    return tuple(
        sorted(
            normalized,
            key=lambda row: (row["attack_family"], row["attack_name"]),
        )
    )


def _normalize_failure_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """规整失败案例 JSONL, 包括记录摘要和负判定身份."""

    materialized = _rows_with_exact_fields(
        rows, FAILURE_CASE_FIELDNAMES, "失败案例记录"
    )
    normalized = []
    for row in materialized:
        payload = {
            "failure_case_rank": _nonnegative_int(
                row["failure_case_rank"], "failure_case_rank"
            ),
            **{
                field_name: str(row[field_name])
                for field_name in (
                    "attack_family",
                    "attack_name",
                    "sample_role",
                    "source_record_id",
                    "attack_record_id",
                    "attacked_image_path",
                    "attacked_image_digest",
                    "source_image_digest",
                    "failure_case_record_digest",
                )
            },
            **{
                field_name: _finite_or_empty(row[field_name], field_name)
                for field_name in (
                    "aligned_content_score_after",
                    "aligned_content_score_before",
                    "score_retention",
                )
            },
            "evidence_decision": _strict_bool(
                row["evidence_decision"], "evidence_decision"
            ),
            "supports_paper_claim": _strict_bool(
                row["supports_paper_claim"], "supports_paper_claim"
            ),
        }
        if payload["evidence_decision"] is not False:
            raise ResultAnalysisSemanticError("失败案例必须对应 false-negative 判定")
        _canonical_repository_relative_path(
            payload["attacked_image_path"], "attacked_image_path"
        )
        digest_payload = {
            field_name: payload[field_name]
            for field_name in FAILURE_CASE_FIELDNAMES
            if field_name != "failure_case_record_digest"
        }
        if payload["failure_case_record_digest"] != build_stable_digest(
            digest_payload
        ):
            raise ResultAnalysisSemanticError("失败案例记录摘要与内容不一致")
        normalized.append(payload)
    ranks = [row["failure_case_rank"] for row in normalized]
    if ranks != list(range(1, len(normalized) + 1)):
        raise ResultAnalysisSemanticError("失败案例 rank 必须连续且从1开始")
    return tuple(normalized)


def _assert_typed_rows_equal(
    reported: Sequence[Mapping[str, Any]],
    rebuilt: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
    role: str,
) -> None:
    """对已规整表执行逐行、逐单元比较."""

    if len(reported) != len(rebuilt):
        raise ResultAnalysisSemanticError(f"{role} 行数与重建结果不一致")
    for row_index, (actual, expected) in enumerate(
        zip(reported, rebuilt, strict=True)
    ):
        for field_name in fieldnames:
            actual_value = actual[field_name]
            expected_value = expected[field_name]
            if isinstance(expected_value, float):
                _assert_float_equal(
                    float(actual_value),
                    expected_value,
                    role=role,
                    row_index=row_index,
                    field_name=field_name,
                )
            elif actual_value != expected_value:
                raise ResultAnalysisSemanticError(
                    f"{role} 与正式记录重建值不一致: row={row_index}, field={field_name}"
                )


def rebuild_and_validate_result_analysis_derived_payload(
    *,
    result_records: Iterable[Mapping[str, Any]],
    attack_detection_records: Iterable[Mapping[str, Any]],
    confidence_interval_rows: Iterable[Mapping[str, Any]],
    per_attack_superiority_rows: Iterable[Mapping[str, Any]],
    failure_case_rows: Iterable[Mapping[str, Any]],
    failure_case_svg_text: str,
    failure_figure_path: str,
    failure_case_limit: int,
) -> dict[str, Any]:
    """独立重建结果分析目录中的两张表、失败记录和 SVG."""

    if _nonnegative_int(failure_case_limit, "failure_case_limit") != FORMAL_FAILURE_CASE_LIMIT:
        raise ResultAnalysisSemanticError(
            f"正式失败案例上限必须冻结为 {FORMAL_FAILURE_CASE_LIMIT}"
        )
    result_materialized = tuple(dict(row) for row in result_records)
    attack_materialized = tuple(dict(row) for row in attack_detection_records)
    rebuilt_ci = _normalize_confidence_interval_rows(
        build_confidence_interval_rows(result_materialized)
    )
    reported_ci = _normalize_confidence_interval_rows(confidence_interval_rows)
    _assert_typed_rows_equal(
        reported_ci,
        rebuilt_ci,
        fieldnames=CONFIDENCE_INTERVAL_FIELDNAMES,
        role="置信区间表",
    )
    rebuilt_per_attack = _normalize_per_attack_rows(
        build_per_attack_superiority_rows(result_materialized)
    )
    reported_per_attack = _normalize_per_attack_rows(
        per_attack_superiority_rows
    )
    _assert_typed_rows_equal(
        reported_per_attack,
        rebuilt_per_attack,
        fieldnames=PER_ATTACK_SUPERIORITY_FIELDNAMES,
        role="逐攻击比较表",
    )
    rebuilt_failures = _normalize_failure_rows(
        build_failure_case_records(
            attack_materialized,
            limit=failure_case_limit,
        )
    )
    observed_false_negative_count = sum(
        record.get("sample_role") == "positive_source"
        and record.get("evidence_decision") is False
        for record in attack_materialized
    )
    expected_failure_record_count = min(
        FORMAL_FAILURE_CASE_LIMIT,
        observed_false_negative_count,
    )
    if len(rebuilt_failures) != expected_failure_record_count:
        raise ResultAnalysisSemanticError("失败案例数量未精确披露受治理 false-negative")
    reported_failures = _normalize_failure_rows(failure_case_rows)
    _assert_typed_rows_equal(
        reported_failures,
        rebuilt_failures,
        fieldnames=FAILURE_CASE_FIELDNAMES,
        role="失败案例记录",
    )
    rebuilt_svg = build_failure_case_svg_text(
        rebuilt_failures,
        failure_figure_path=failure_figure_path,
    )
    if failure_case_svg_text != rebuilt_svg:
        raise ResultAnalysisSemanticError("失败案例 SVG 无法由受治理失败记录精确重建")
    evidence = {
        "confidence_interval_rebuilt_rows_digest": build_stable_digest(
            rebuilt_ci
        ),
        "per_attack_rebuilt_rows_digest": build_stable_digest(
            rebuilt_per_attack
        ),
        "failure_case_rebuilt_rows_digest": build_stable_digest(
            rebuilt_failures
        ),
        "failure_case_rebuilt_svg_sha256": hashlib.sha256(
            rebuilt_svg.encode("utf-8")
        ).hexdigest(),
        "failure_case_observed_false_negative_count": (
            observed_false_negative_count
        ),
        "failure_case_expected_record_count": expected_failure_record_count,
    }
    return {
        **evidence,
        "result_analysis_semantic_rebuild_digest": build_stable_digest(evidence),
        "result_analysis_semantic_rebuild_ready": True,
    }


def _validate_paired_superiority_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """核对配对统计判定公式, 但不要求所有比较都显著胜出."""

    materialized = _rows_with_exact_fields(
        rows, PAIRED_SUPERIORITY_FIELDNAMES, "配对优势表"
    )
    normalized = []
    for row in materialized:
        baseline_id = str(row["baseline_id"])
        mean = _finite_float(
            row["mean_paired_true_positive_rate_difference"],
            "mean_paired_true_positive_rate_difference",
        )
        ci_low = _finite_float(
            row["mean_paired_difference_ci_low"],
            "mean_paired_difference_ci_low",
        )
        ci_high = _finite_float(
            row["mean_paired_difference_ci_high"],
            "mean_paired_difference_ci_high",
        )
        adjusted_p = _finite_float(
            row["holm_adjusted_p_value"], "holm_adjusted_p_value"
        )
        if not -1.0 <= ci_low <= mean <= ci_high <= 1.0:
            raise ResultAnalysisSemanticError("配对优势表置信区间无效")
        if not 0.0 <= adjusted_p <= 1.0:
            raise ResultAnalysisSemanticError("配对优势表 Holm p 值无效")
        ready = _strict_bool(
            row["paired_superiority_ready"], "paired_superiority_ready"
        )
        quality_coverage_ready = _strict_bool(
            row["quality_match_coverage_ready"],
            "quality_match_coverage_ready",
        )
        quality_mean = _finite_float(
            row[
                "quality_matched_mean_paired_true_positive_rate_difference"
            ],
            "quality_matched_mean_paired_true_positive_rate_difference",
        )
        quality_ci_low = _finite_float(
            row["quality_matched_mean_paired_difference_ci_low"],
            "quality_matched_mean_paired_difference_ci_low",
        )
        quality_ci_high = _finite_float(
            row["quality_matched_mean_paired_difference_ci_high"],
            "quality_matched_mean_paired_difference_ci_high",
        )
        quality_adjusted_p = _finite_float(
            row["quality_matched_holm_adjusted_p_value"],
            "quality_matched_holm_adjusted_p_value",
        )
        if not (
            -1.0 <= quality_ci_low <= quality_mean <= quality_ci_high <= 1.0
            and 0.0 <= quality_adjusted_p <= 1.0
        ):
            raise ResultAnalysisSemanticError("质量匹配配对优势统计值无效")
        quality_ready = _strict_bool(
            row["quality_matched_superiority_ready"],
            "quality_matched_superiority_ready",
        )
        supports = _strict_bool(
            row["supports_paper_claim"], "supports_paper_claim"
        )
        expected_ready = bool(mean > 0.0 and ci_low > 0.0 and adjusted_p < 0.05)
        expected_quality_ready = bool(
            quality_coverage_ready
            and quality_mean > 0.0
            and quality_ci_low > 0.0
            and quality_adjusted_p < 0.05
        )
        if (
            ready != expected_ready
            or quality_ready != expected_quality_ready
            or supports != (expected_ready and expected_quality_ready)
        ):
            raise ResultAnalysisSemanticError(
                "配对优势表结论与全样本或质量匹配统计不一致"
            )
        payload = dict(row)
        payload.update(
            {
                "baseline_id": baseline_id,
                "mean_paired_true_positive_rate_difference": mean,
                "mean_paired_difference_ci_low": ci_low,
                "mean_paired_difference_ci_high": ci_high,
                "holm_adjusted_p_value": adjusted_p,
                "paired_superiority_ready": ready,
                "quality_match_coverage_ready": quality_coverage_ready,
                "quality_matched_mean_paired_true_positive_rate_difference": (
                    quality_mean
                ),
                "quality_matched_mean_paired_difference_ci_low": (
                    quality_ci_low
                ),
                "quality_matched_mean_paired_difference_ci_high": (
                    quality_ci_high
                ),
                "quality_matched_holm_adjusted_p_value": quality_adjusted_p,
                "quality_matched_superiority_ready": quality_ready,
                "supports_paper_claim": supports,
            }
        )
        normalized.append(payload)
    if (
        len(normalized) != len(PRIMARY_BASELINE_IDS)
        or {row["baseline_id"] for row in normalized} != set(PRIMARY_BASELINE_IDS)
    ):
        raise ResultAnalysisSemanticError("配对优势表未精确覆盖4个主表 baseline")
    ordered = tuple(sorted(normalized, key=lambda row: row["baseline_id"]))
    return {
        "paired_superiority_semantic_rows_digest": build_stable_digest(ordered),
        "paired_superiority_negative_result_count": sum(
            not row["paired_superiority_ready"] for row in ordered
        ),
        "paired_superiority_semantic_rebuild_ready": True,
    }


def rebuild_and_validate_result_analysis_semantics(
    *,
    paper_claim_scale: str,
    governed_payload_path_map: Mapping[str, Any],
    target_fpr: float,
    result_records: Iterable[Mapping[str, Any]],
    attack_detection_records: Iterable[Mapping[str, Any]],
    attack_family_metrics: Iterable[Mapping[str, Any]],
    baseline_comparison_rows: Iterable[Mapping[str, Any]],
    dataset_quality_feature_records: Iterable[Mapping[str, Any]],
    dataset_quality_metric_rows: Iterable[Mapping[str, Any]],
    expected_quality_pair_count: int,
    paired_superiority_rows: Iterable[Mapping[str, Any]],
    confidence_interval_rows: Iterable[Mapping[str, Any]],
    per_attack_superiority_rows: Iterable[Mapping[str, Any]],
    failure_case_rows: Iterable[Mapping[str, Any]],
    failure_case_svg_text: str,
    failure_figure_path: str,
    failure_case_limit: int,
) -> dict[str, Any]:
    """核对七类论文 payload 与其原子或受治理来源的完整语义链."""

    scale = str(paper_claim_scale).strip()
    if not scale:
        raise ResultAnalysisSemanticError("paper_claim_scale 不得为空")
    canonical_path_map = require_governed_paper_payload_path_map(
        scale, governed_payload_path_map
    )
    result_materialized = tuple(dict(row) for row in result_records)
    if {str(row.get("paper_claim_scale", "")) for row in result_materialized} != {
        scale
    }:
        raise ResultAnalysisSemanticError("result records 与论文层级不一致")
    attack_materialized = tuple(dict(row) for row in attack_detection_records)
    attack_metric_materialized = tuple(dict(row) for row in attack_family_metrics)
    main_evidence = _validate_main_comparison_rows(
        result_materialized, baseline_comparison_rows
    )
    attack_evidence = _validate_attack_rows(
        attack_materialized,
        attack_metric_materialized,
        target_fpr=target_fpr,
    )
    cross_table_evidence = _validate_slm_result_attack_consistency(
        result_materialized,
        attack_metric_materialized,
    )
    try:
        quality_evidence = rebuild_and_validate_formal_fid_kid_metrics(
            dataset_quality_feature_records,
            dataset_quality_metric_rows,
            expected_pair_count=_nonnegative_int(
                expected_quality_pair_count, "expected_quality_pair_count"
            ),
        )
    except ValueError as exc:
        raise ResultAnalysisSemanticError(
            "质量表无法由正式 Inception feature records 重建"
        ) from exc
    paired_evidence = _validate_paired_superiority_rows(
        paired_superiority_rows
    )
    derived_evidence = rebuild_and_validate_result_analysis_derived_payload(
        result_records=result_materialized,
        attack_detection_records=attack_materialized,
        confidence_interval_rows=confidence_interval_rows,
        per_attack_superiority_rows=per_attack_superiority_rows,
        failure_case_rows=failure_case_rows,
        failure_case_svg_text=failure_case_svg_text,
        failure_figure_path=failure_figure_path,
        failure_case_limit=failure_case_limit,
    )
    evidence = {
        "paper_claim_scale": scale,
        "governed_paper_payload_path_map": canonical_path_map,
        **main_evidence,
        **attack_evidence,
        **cross_table_evidence,
        **quality_evidence,
        **paired_evidence,
        **derived_evidence,
    }
    return {
        **evidence,
        "governed_paper_payload_semantic_digest": build_stable_digest(evidence),
        "governed_paper_payload_semantic_rebuild_ready": True,
    }


def build_governed_paper_payload_path_map(paper_claim_scale: str) -> dict[str, str]:
    """返回完整包必须逐字节复验的主表、攻击表、质量表和结果分析表图."""

    scale = str(paper_claim_scale).strip()
    if not scale:
        raise ValueError("paper_claim_scale 不得为空")
    result_analysis_root = f"outputs/pilot_paper_result_analysis/{scale}"
    return {
        "main_comparison_table": (
            f"outputs/external_baseline_comparison/{scale}/baseline_comparison_table.csv"
        ),
        "attack_table": f"outputs/attack_matrix/{scale}/attack_family_metrics.csv",
        "quality_table": (
            f"outputs/dataset_level_quality/{scale}/dataset_quality_metrics.csv"
        ),
        **{
            role: f"{result_analysis_root}/{file_name}"
            for role, file_name in RESULT_ANALYSIS_PAYLOAD_FILE_NAMES.items()
        },
    }


def require_governed_paper_payload_path_map(
    paper_claim_scale: str,
    path_map: Mapping[str, Any],
) -> dict[str, str]:
    """要求七类论文 payload 路径逐字符等于规范仓库相对路径."""

    expected = build_governed_paper_payload_path_map(paper_claim_scale)
    reported = {str(role): str(path) for role, path in path_map.items()}
    if reported != expected:
        raise ResultAnalysisSemanticError(
            "论文 payload 路径角色或规范仓库相对路径不一致"
        )
    return reported


def _recorded_path(path: Path, repository_root: Path) -> str:
    """仓库内文件使用相对路径, 仓库外文件保留绝对路径."""

    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def file_sha256(path: Path) -> str:
    """流式计算普通文件 SHA-256."""

    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"结果分析 payload 必须是普通文件: {path.as_posix()}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_result_analysis_payload_binding(
    *,
    repository_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """从刚写出的四类 payload 构造精确路径与字节摘要绑定."""

    root = Path(repository_root).resolve()
    resolved_output_dir = Path(output_dir).resolve()
    path_map = {
        role: _recorded_path(resolved_output_dir / file_name, root)
        for role, file_name in RESULT_ANALYSIS_PAYLOAD_FILE_NAMES.items()
    }
    sha256_map = {
        role: file_sha256(resolved_output_dir / file_name)
        for role, file_name in RESULT_ANALYSIS_PAYLOAD_FILE_NAMES.items()
    }
    digest_payload = {
        "result_analysis_payload_path_map": path_map,
        "result_analysis_payload_sha256_map": sha256_map,
    }
    return {
        **digest_payload,
        "result_analysis_payload_digest": build_stable_digest(digest_payload),
    }


def build_result_analysis_manifest_config(summary: Mapping[str, Any]) -> dict[str, Any]:
    """从 summary 重建结果分析 manifest 的完整配置身份."""

    return {
        "failure_case_limit": summary.get("failure_case_limit"),
        "primary_baseline_method_ids": list(PRIMARY_BASELINE_IDS),
        "proposed_method_id": "slm_wm_current",
        "result_record_set_digest": summary.get("result_record_set_digest", ""),
        "paired_superiority_rows_digest": summary.get(
            "paired_superiority_rows_digest", ""
        ),
        "paired_superiority_protocol_digest": summary.get(
            "paired_superiority_protocol_digest", ""
        ),
        "quality_matching_protocol_schema": summary.get(
            "quality_matching_protocol_schema", ""
        ),
        "quality_matching_protocol_digest": summary.get(
            "quality_matching_protocol_digest", ""
        ),
        "quality_metric_name": summary.get("quality_metric_name", ""),
        "quality_match_caliper": summary.get("quality_match_caliper", 0.0),
        "minimum_matched_prompt_fraction": summary.get(
            "minimum_matched_prompt_fraction", 0.0
        ),
        "quality_matched_rows_digest": summary.get(
            "quality_matched_rows_digest", ""
        ),
        "paired_test_prompt_count": summary.get("paired_test_prompt_count", 0),
        "paired_test_prompt_id_digest": summary.get(
            "paired_test_prompt_id_digest", ""
        ),
        "paired_attack_registry_digest": summary.get(
            "paired_attack_registry_digest", ""
        ),
        "method_observation_source_sha256_map": summary.get(
            "method_observation_source_sha256_map", {}
        ),
        "threshold_audit_rows_digest": summary.get(
            "threshold_audit_rows_digest", ""
        ),
        "claim_p_value_method": summary.get("claim_p_value_method", ""),
        "sharp_null_diagnostic_method": summary.get(
            "sharp_null_diagnostic_method", ""
        ),
        "bootstrap_analysis_schema": summary.get("bootstrap_analysis_schema", ""),
        "bootstrap_bit_generator": summary.get("bootstrap_bit_generator", ""),
        "bootstrap_quantile_method": summary.get("bootstrap_quantile_method", ""),
        "bootstrap_resample_count": summary.get("bootstrap_resample_count", 0),
        "confidence_level": summary.get("confidence_level", 0.0),
        "result_analysis_payload_digest": summary.get(
            "result_analysis_payload_digest", ""
        ),
        "result_analysis_semantic_rebuild_digest": summary.get(
            "result_analysis_semantic_rebuild_digest", ""
        ),
        "result_analysis_semantic_rebuild_ready": summary.get(
            "result_analysis_semantic_rebuild_ready", False
        ),
    }


def result_analysis_payload_binding_ready(
    *,
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    actual_source_sha256: Mapping[str, str],
) -> bool:
    """核验 summary、manifest、实际源文件摘要和固定角色集合完全一致."""

    path_map = summary.get("result_analysis_payload_path_map")
    sha256_map = summary.get("result_analysis_payload_sha256_map")
    declared_digest = str(summary.get("result_analysis_payload_digest", ""))
    semantic_digest = str(
        summary.get("result_analysis_semantic_rebuild_digest", "")
    )
    metadata = manifest.get("metadata")
    output_paths = manifest.get("output_paths")
    expected_roles = set(RESULT_ANALYSIS_PAYLOAD_FILE_NAMES)
    if (
        not isinstance(path_map, Mapping)
        or set(path_map) != expected_roles
        or not isinstance(sha256_map, Mapping)
        or set(sha256_map) != expected_roles
        or not isinstance(metadata, Mapping)
        or not isinstance(output_paths, list | tuple)
    ):
        return False
    normalized_paths = {role: str(path_map[role]) for role in expected_roles}
    normalized_sha256 = {role: str(sha256_map[role]) for role in expected_roles}
    paper_claim_scale = str(summary.get("paper_claim_scale", ""))
    governed_paths = build_governed_paper_payload_path_map(paper_claim_scale)
    expected_path_map = {
        role: governed_paths[role] for role in RESULT_ANALYSIS_PAYLOAD_FILE_NAMES
    }
    digest_payload = {
        "result_analysis_payload_path_map": normalized_paths,
        "result_analysis_payload_sha256_map": normalized_sha256,
    }
    output_path_set = {str(path) for path in output_paths}
    return bool(
        len(set(normalized_paths.values())) == len(expected_roles)
        and all(_SHA256_PATTERN.fullmatch(value) for value in normalized_sha256.values())
        and normalized_paths == expected_path_map
        and all(
            normalized_paths[role] in output_path_set
            for role in expected_roles
        )
        and all(
            actual_source_sha256.get(normalized_paths[role])
            == normalized_sha256[role]
            for role in expected_roles
        )
        and _SHA256_PATTERN.fullmatch(declared_digest) is not None
        and declared_digest == build_stable_digest(digest_payload)
        and summary.get("result_analysis_semantic_rebuild_ready") is True
        and _SHA256_PATTERN.fullmatch(semantic_digest) is not None
        and metadata.get("result_analysis_payload_path_map") == dict(path_map)
        and metadata.get("result_analysis_payload_sha256_map") == dict(sha256_map)
        and metadata.get("result_analysis_payload_digest") == declared_digest
        and metadata.get("result_analysis_semantic_rebuild_digest")
        == semantic_digest
        and metadata.get("result_analysis_semantic_rebuild_ready") is True
        and manifest.get("config_digest")
        == build_stable_digest(build_result_analysis_manifest_config(summary))
    )
