"""重建 pilot_paper 论文结果分析表与失败案例图。

该脚本只读取已经闭合的 records 与 manifests, 不重新运行 GPU 推理, 也不手工
拼接论文结论。它用于把 fixed-FPR 结果记录转换为可直接进入论文图表的
Hoeffding 置信区间表、per-attack superiority 表和失败案例 SVG。
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.protocol.pilot_paper_fixed_fpr import (
    PILOT_PAPER_CONFIDENCE_LEVEL,
    PILOT_PAPER_PAIRED_BOOTSTRAP_ANALYSIS_SCHEMA,
    PILOT_PAPER_PAIRED_BOOTSTRAP_BIT_GENERATOR,
    PILOT_PAPER_PAIRED_BOOTSTRAP_QUANTILE_METHOD,
    PILOT_PAPER_PAIRED_BOOTSTRAP_RESAMPLE_COUNT,
    PILOT_PAPER_PAIRED_CLAIM_P_VALUE_METHOD,
    PILOT_PAPER_PAIRED_SHARP_NULL_DIAGNOSTIC_METHOD,
    build_pilot_paper_result_record_set_digest,
)
from experiments.protocol.attacks import default_attack_configs
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest

CONSTRUCTION_UNIT_NAME = "pilot_paper_result_analysis"
DEFAULT_OUTPUT_ROOT = Path("outputs/pilot_paper_result_analysis")
DEFAULT_RESULT_RECORDS_ROOT = Path("outputs/pilot_paper_fixed_fpr_results")
DEFAULT_ATTACK_MATRIX_ROOT = Path("outputs/attack_matrix")
DEFAULT_PAIRED_SUPERIORITY_ROOT = Path("outputs/paired_superiority_analysis")
PRIMARY_BASELINE_METHOD_IDS = ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
PROPOSED_METHOD_ID = "slm_wm_current"
FORMAL_METHOD_IDS = (PROPOSED_METHOD_ID, *PRIMARY_BASELINE_METHOD_IDS)


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: dict[str, Any]) -> str:
    """把字典转换为 JSONL 单行。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """读取正式 JSONL 输入; 文件缺失时立即停止结果闭合。"""

    if not path.is_file():
        raise FileNotFoundError(f"论文结果分析缺少正式 JSONL 输入: {path.as_posix()}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    """写出 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json_text(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """写出 JSONL 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json_line(row) for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写出 CSV 表格。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_code_version(root_path: Path) -> str:
    """读取 Git 短提交标识。"""

    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--short"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "git_version_unavailable"
    commit_id = commit_result.stdout.strip()
    if not commit_id:
        return "git_version_unavailable"
    return f"{commit_id}-dirty" if status_result.stdout.strip() else commit_id


def resolve_input_path(root_path: Path, path: str | Path) -> Path:
    """解析输入路径。"""

    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先返回相对仓库根目录路径。"""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def ensure_output_dir_under_outputs(root_path: Path, output_dir: str | Path) -> Path:
    """确保输出目录位于 outputs/ 下。"""

    candidate = Path(output_dir)
    resolved = candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("pilot_paper 结果分析输出目录必须位于 outputs/ 下") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _float_value(row: dict[str, Any], field_name: str, default_value: float = 0.0) -> float:
    """解析浮点字段。"""

    value = row.get(field_name, default_value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default_value)


def build_confidence_interval_rows(result_records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """从结果记录重建 Hoeffding 置信区间表。"""

    rows: list[dict[str, Any]] = []
    for record in result_records:
        if str(record.get("metric_status", "unsupported")) == "unsupported":
            continue
        rows.append(
            {
                "paper_claim_scale": record.get("paper_claim_scale", ""),
                "method_id": record.get("method_id", ""),
                "attack_family": record.get("attack_family", ""),
                "attack_name": record.get("attack_name", ""),
                "resource_profile": record.get("resource_profile", ""),
                "true_positive_rate": record.get("true_positive_rate", ""),
                "true_positive_rate_ci_low": record.get("true_positive_rate_ci_low", ""),
                "true_positive_rate_ci_high": record.get("true_positive_rate_ci_high", ""),
                "false_positive_rate": record.get("false_positive_rate", ""),
                "false_positive_rate_ci_low": record.get("false_positive_rate_ci_low", ""),
                "false_positive_rate_ci_high": record.get("false_positive_rate_ci_high", ""),
                "clean_false_positive_rate": record.get("clean_false_positive_rate", ""),
                "clean_false_positive_rate_ci_low": record.get("clean_false_positive_rate_ci_low", ""),
                "clean_false_positive_rate_ci_high": record.get("clean_false_positive_rate_ci_high", ""),
                "attacked_false_positive_rate": record.get("attacked_false_positive_rate", ""),
                "attacked_false_positive_rate_ci_low": record.get("attacked_false_positive_rate_ci_low", ""),
                "attacked_false_positive_rate_ci_high": record.get("attacked_false_positive_rate_ci_high", ""),
                "positive_count": record.get("positive_count", ""),
                "negative_count": record.get("negative_count", ""),
                "confidence_interval_method": record.get("confidence_interval_method", ""),
                "confidence_level": record.get("confidence_level", ""),
                "supports_paper_claim": record.get("supports_paper_claim", False),
            }
        )
    return sorted(rows, key=lambda row: (row["attack_name"], row["method_id"]))


def _finite_float(row: dict[str, Any], field_name: str) -> float | None:
    """读取有限浮点数; 缺失或非法值用于阻断完整统计披露门禁。"""

    try:
        value = float(row[field_name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _confidence_interval_row_ready(row: dict[str, Any]) -> bool:
    """核验单条正式 CI 表行具有完整且自洽的检测统计。"""

    interval_groups = (
        ("true_positive_rate_ci_low", "true_positive_rate", "true_positive_rate_ci_high"),
        ("false_positive_rate_ci_low", "false_positive_rate", "false_positive_rate_ci_high"),
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
    )
    interval_ready = True
    for low_name, value_name, high_name in interval_groups:
        low = _finite_float(row, low_name)
        value = _finite_float(row, value_name)
        high = _finite_float(row, high_name)
        interval_ready = interval_ready and (
            low is not None
            and value is not None
            and high is not None
            and 0.0 <= low <= value <= high <= 1.0
        )
    positive_count = _finite_float(row, "positive_count")
    negative_count = _finite_float(row, "negative_count")
    confidence_level = _finite_float(row, "confidence_level")
    return bool(
        interval_ready
        and positive_count is not None
        and positive_count > 0
        and negative_count is not None
        and negative_count > 0
        and confidence_level is not None
        and 0.0 < confidence_level < 1.0
        and str(row.get("confidence_interval_method", "")).strip()
        and bool(row.get("supports_paper_claim", False))
    )


def _superiority_evaluation_row_ready(row: dict[str, Any]) -> bool:
    """核验逐攻击比较行可审计, 但不把显著胜出当作完整披露前提。"""

    numeric_fields = (
        "slm_true_positive_rate",
        "slm_true_positive_rate_ci_low",
        "slm_true_positive_rate_ci_high",
        "best_baseline_true_positive_rate",
        "best_baseline_true_positive_rate_ci_low",
        "best_baseline_true_positive_rate_ci_high",
        "slm_minus_best_baseline_tpr",
        "conservative_ci_margin",
    )
    return bool(
        str(row.get("attack_family", "")).strip()
        and str(row.get("attack_name", "")).strip()
        and str(row.get("best_baseline_id", "")) in PRIMARY_BASELINE_METHOD_IDS
        and all(_finite_float(row, field_name) is not None for field_name in numeric_fields)
        and bool(row.get("supports_paper_claim", False))
    )


def build_per_attack_superiority_rows(result_records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """重建每个攻击下 SLM-WM 相对最强主表 baseline 的优势表。"""

    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for record in result_records:
        if str(record.get("metric_status", "unsupported")) == "unsupported":
            continue
        key = (str(record.get("attack_family", "")), str(record.get("attack_name", "")))
        grouped.setdefault(key, {})[str(record.get("method_id", ""))] = record

    rows: list[dict[str, Any]] = []
    for (attack_family, attack_name), method_records in sorted(grouped.items()):
        slm_record = method_records.get(PROPOSED_METHOD_ID)
        if slm_record is None:
            continue
        baseline_records = [
            method_records[method_id] for method_id in PRIMARY_BASELINE_METHOD_IDS if method_id in method_records
        ]
        if not baseline_records:
            continue
        best_baseline = max(baseline_records, key=lambda row: _float_value(row, "true_positive_rate"))
        slm_tpr = _float_value(slm_record, "true_positive_rate")
        best_tpr = _float_value(best_baseline, "true_positive_rate")
        slm_ci_low = _float_value(slm_record, "true_positive_rate_ci_low", slm_tpr)
        best_ci_high = _float_value(best_baseline, "true_positive_rate_ci_high", best_tpr)
        margin = slm_tpr - best_tpr
        conservative_margin = slm_ci_low - best_ci_high
        rows.append(
            {
                "attack_family": attack_family,
                "attack_name": attack_name,
                "slm_true_positive_rate": slm_tpr,
                "slm_true_positive_rate_ci_low": slm_record.get("true_positive_rate_ci_low", ""),
                "slm_true_positive_rate_ci_high": slm_record.get("true_positive_rate_ci_high", ""),
                "best_baseline_id": best_baseline.get("method_id", ""),
                "best_baseline_true_positive_rate": best_tpr,
                "best_baseline_true_positive_rate_ci_low": best_baseline.get("true_positive_rate_ci_low", ""),
                "best_baseline_true_positive_rate_ci_high": best_baseline.get("true_positive_rate_ci_high", ""),
                "slm_minus_best_baseline_tpr": margin,
                "conservative_ci_margin": conservative_margin,
                "superiority_claim_ready": bool(
                    slm_record.get("supports_paper_claim", False)
                    and best_baseline.get("supports_paper_claim", False)
                    and conservative_margin > 0
                ),
                "supports_paper_claim": bool(
                    slm_record.get("supports_paper_claim", False)
                    and best_baseline.get("supports_paper_claim", False)
                ),
            }
        )
    return rows


def read_json_object(path: Path) -> dict[str, Any]:
    """读取必须存在的 JSON 对象."""

    if not path.is_file():
        raise FileNotFoundError(f"论文结果分析缺少 JSON 输入: {path.as_posix()}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"论文结果分析输入必须是 JSON 对象: {path.as_posix()}")
    return dict(payload)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读取必须存在且非空的 CSV 统计表."""

    if not path.is_file():
        raise FileNotFoundError(f"论文结果分析缺少 CSV 输入: {path.as_posix()}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"论文结果分析 CSV 输入不得为空: {path.as_posix()}")
    return rows


def build_result_template_coverage(result_records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """检查五种方法是否完整覆盖正式攻击矩阵。"""

    records = tuple(result_records)
    attack_keys = {
        (config.attack_family, config.attack_name, config.resource_profile)
        for config in default_attack_configs()
        if config.enabled and config.resource_profile in {"full_main", "full_extra"}
    }
    expected_keys = {
        (method_id, attack_family, attack_name, resource_profile)
        for method_id in FORMAL_METHOD_IDS
        for attack_family, attack_name, resource_profile in attack_keys
    }
    actual_key_rows = [
        (
            str(record.get("method_id", "")),
            str(record.get("attack_family", "")),
            str(record.get("attack_name", "")),
            str(record.get("resource_profile", "")),
        )
        for record in records
    ]
    actual_keys = set(actual_key_rows)
    missing_keys = sorted(expected_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)
    duplicate_count = len(actual_key_rows) - len(actual_keys)
    return {
        "expected_superiority_row_count": len(attack_keys),
        "expected_result_record_count": len(expected_keys),
        "actual_result_record_count": len(actual_key_rows),
        "unique_result_record_key_count": len(actual_keys),
        "duplicate_result_record_count": duplicate_count,
        "missing_result_record_count": len(missing_keys),
        "unexpected_result_record_count": len(unexpected_keys),
        "result_template_coverage_ready": actual_keys == expected_keys and duplicate_count == 0,
        "missing_result_record_examples": [list(key) for key in missing_keys[:20]],
        "unexpected_result_record_examples": [list(key) for key in unexpected_keys[:20]],
    }


def _attacked_image_path(record: dict[str, Any]) -> str:
    """读取攻击后图像路径。"""

    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        return str(metadata.get("attacked_image_path", ""))
    return ""


def build_failure_case_records(
    formal_detection_records: Iterable[dict[str, Any]],
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """筛选 positive_source 中未通过检测的代表性失败案例。"""

    failures = [
        record
        for record in formal_detection_records
        if record.get("sample_role") == "positive_source" and bool(record.get("evidence_decision", False)) is False
    ]
    failures.sort(key=lambda row: (_float_value(row, "aligned_content_score_after"), str(row.get("attack_name", ""))))
    selected = failures[: max(0, int(limit))]
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(selected, start=1):
        payload = {
            "failure_case_rank": index,
            "attack_family": record.get("attack_family", ""),
            "attack_name": record.get("attack_name", ""),
            "sample_role": record.get("sample_role", ""),
            "source_record_id": record.get("source_record_id", ""),
            "attack_record_id": record.get("attack_record_id", ""),
            "aligned_content_score_after": record.get("aligned_content_score_after", ""),
            "aligned_content_score_before": record.get("aligned_content_score_before", ""),
            "score_retention": record.get("score_retention", ""),
            "evidence_decision": record.get("evidence_decision", ""),
            "attacked_image_path": _attacked_image_path(record),
            "attacked_image_digest": record.get("attacked_image_digest", ""),
            "source_image_digest": record.get("source_image_digest", ""),
            "supports_paper_claim": record.get("supports_paper_claim", False),
        }
        payload["failure_case_record_digest"] = build_stable_digest(payload)
        rows.append(payload)
    return rows


def build_failure_case_svg(root_path: Path, output_dir: Path, failure_cases: list[dict[str, Any]]) -> str:
    """从实际攻击图像生成失败案例 SVG 图。

    该图只引用受治理记录给出的真实攻击图像。只要记录声明了失败案例, 对应
    图像就必须存在; 缺失图像不能通过空白卡片代替, 因为那会破坏论文图与
    样本证据之间的一一对应关系。
    """

    card_width = 260
    card_height = 250
    columns = 3
    rows = max(1, (len(failure_cases) + columns - 1) // columns)
    width = columns * card_width + 40
    height = rows * card_height + 80
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial, sans-serif;} .title{font-size:18px;font-weight:700;} '
        '.label{font-size:11px;} .small{font-size:10px;fill:#333;} .card{fill:#fff;stroke:#bbb;stroke-width:1;}</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="28" class="title">SLM-WM failure cases under fixed-FPR attack protocol</text>',
        '<text x="20" y="48" class="small">Each panel renders the attacked image referenced by its governed detection record.</text>',
    ]
    if not failure_cases:
        svg_parts.append(
            '<text x="20" y="82" class="label">No false-negative case was observed in the governed detection records.</text>'
        )
    for index, item in enumerate(failure_cases):
        column = index % columns
        row = index // columns
        x = 20 + column * card_width
        y = 65 + row * card_height
        image_path_text = str(item.get("attacked_image_path", ""))
        if not image_path_text:
            raise ValueError(
                f"失败案例 #{item.get('failure_case_rank', index + 1)} 缺少 attacked_image_path"
            )
        image_abs_path = resolve_input_path(root_path, image_path_text)
        if not image_abs_path.is_file():
            raise FileNotFoundError(f"失败案例攻击图像不存在: {image_abs_path.as_posix()}")
        href = Path(os.path.relpath(image_abs_path, output_dir)).as_posix()
        svg_parts.append(f'<rect x="{x}" y="{y}" width="{card_width - 16}" height="{card_height - 14}" class="card"/>')
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


def write_pilot_paper_result_analysis_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
    result_records_path: str | Path | None = None,
    attack_detection_records_path: str | Path | None = None,
    paired_superiority_summary_path: str | Path | None = None,
    paired_superiority_table_path: str | Path | None = None,
    paired_superiority_manifest_path: str | Path | None = None,
    failure_case_limit: int = 12,
) -> dict[str, Any]:
    """写出 pilot_paper 结果分析表和失败案例图。"""

    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(root_path)
    output_path = ensure_output_dir_under_outputs(
        root_path,
        output_dir or DEFAULT_OUTPUT_ROOT / paper_run.run_name,
    )
    resolved_result_records_path = resolve_input_path(
        root_path,
        result_records_path
        or DEFAULT_RESULT_RECORDS_ROOT / paper_run.run_name / "pilot_paper_result_records.jsonl",
    )
    resolved_attack_records_path = resolve_input_path(
        root_path,
        attack_detection_records_path
        or DEFAULT_ATTACK_MATRIX_ROOT / paper_run.run_name / "attack_detection_records.jsonl",
    )
    paired_root = DEFAULT_PAIRED_SUPERIORITY_ROOT / paper_run.run_name
    resolved_paired_summary_path = resolve_input_path(
        root_path,
        paired_superiority_summary_path
        or paired_root / "paired_superiority_summary.json",
    )
    resolved_paired_table_path = resolve_input_path(
        root_path,
        paired_superiority_table_path or paired_root / "paired_superiority_table.csv",
    )
    resolved_paired_manifest_path = resolve_input_path(
        root_path,
        paired_superiority_manifest_path or paired_root / "manifest.local.json",
    )

    result_records = read_jsonl_rows(resolved_result_records_path)
    result_record_set_digest = build_pilot_paper_result_record_set_digest(result_records)
    formal_detection_records = read_jsonl_rows(resolved_attack_records_path)
    paired_summary = read_json_object(resolved_paired_summary_path)
    paired_rows = read_csv_rows(resolved_paired_table_path)
    paired_manifest = read_json_object(resolved_paired_manifest_path)
    paired_metadata = paired_manifest.get("metadata", {})
    paired_observation_sha256_map = paired_summary.get(
        "method_observation_source_sha256_map",
        {},
    )
    paired_digest_fields = (
        "paired_outcome_set_digest",
        "paired_superiority_rows_digest",
        "paired_superiority_protocol_digest",
        "paired_test_prompt_id_digest",
        "paired_attack_registry_digest",
        "threshold_audit_rows_digest",
    )
    paired_superiority_ready = bool(
        paired_summary.get("paper_claim_scale") == paper_run.run_name
        and math.isclose(
            float(paired_summary.get("target_fpr", float("nan"))),
            paper_run.target_fpr,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and paired_summary.get("paired_superiority_exact_set_ready") is True
        and paired_summary.get("paired_superiority_scale_ready") is True
        and paired_summary.get("overall_paired_superiority_ready") is True
        and paired_summary.get("supports_paper_claim") is True
        and int(paired_summary.get("paired_test_prompt_count", 0)) > 0
        and int(paired_summary.get("expected_attack_count", 0)) > 0
        and int(paired_summary.get("bootstrap_resample_count", 0))
        == PILOT_PAPER_PAIRED_BOOTSTRAP_RESAMPLE_COUNT
        and math.isclose(
            float(paired_summary.get("confidence_level", math.nan)),
            PILOT_PAPER_CONFIDENCE_LEVEL,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and paired_summary.get("bootstrap_analysis_schema")
        == PILOT_PAPER_PAIRED_BOOTSTRAP_ANALYSIS_SCHEMA
        and paired_summary.get("bootstrap_bit_generator")
        == PILOT_PAPER_PAIRED_BOOTSTRAP_BIT_GENERATOR
        and paired_summary.get("bootstrap_quantile_method")
        == PILOT_PAPER_PAIRED_BOOTSTRAP_QUANTILE_METHOD
        and paired_summary.get("claim_p_value_method")
        == PILOT_PAPER_PAIRED_CLAIM_P_VALUE_METHOD
        and paired_summary.get("sharp_null_diagnostic_method")
        == PILOT_PAPER_PAIRED_SHARP_NULL_DIAGNOSTIC_METHOD
        and all(
            len(str(paired_summary.get(field_name, ""))) == 64
            and all(
                character in "0123456789abcdef"
                for character in str(paired_summary.get(field_name, ""))
            )
            for field_name in paired_digest_fields
        )
        and isinstance(paired_observation_sha256_map, dict)
        and set(paired_observation_sha256_map)
        == {"slm_wm", *PRIMARY_BASELINE_METHOD_IDS}
        and all(
            len(str(digest)) == 64
            and all(character in "0123456789abcdef" for character in str(digest))
            for digest in paired_observation_sha256_map.values()
        )
        and len(paired_rows) == len(PRIMARY_BASELINE_METHOD_IDS)
        and {row.get("baseline_id") for row in paired_rows}
        == set(PRIMARY_BASELINE_METHOD_IDS)
        and all(
            str(row.get("paired_superiority_ready", "")).lower() == "true"
            and str(row.get("supports_paper_claim", "")).lower() == "true"
            for row in paired_rows
        )
        and paired_manifest.get("artifact_id")
        == "paired_superiority_analysis_manifest"
        and isinstance(paired_metadata, dict)
        and all(
            paired_metadata.get(field_name) == paired_summary.get(field_name)
            for field_name in (
                "paired_outcome_set_digest",
                "paired_superiority_rows_digest",
                "paired_superiority_protocol_digest",
                "paired_test_prompt_count",
                "paired_test_prompt_id_digest",
                "paired_attack_registry_digest",
                "method_observation_source_sha256_map",
                "method_observation_source_path_map",
                "method_threshold_digest_map",
                "threshold_audit_rows_digest",
                "claim_p_value_method",
                "sharp_null_diagnostic_method",
                "bootstrap_analysis_schema",
                "bootstrap_bit_generator",
                "bootstrap_quantile_method",
                "bootstrap_resample_count",
                "confidence_level",
                "overall_paired_superiority_ready",
            )
        )
    )
    confidence_interval_rows = build_confidence_interval_rows(result_records)
    superiority_rows = build_per_attack_superiority_rows(result_records)
    template_coverage = build_result_template_coverage(result_records)
    failure_rows = build_failure_case_records(formal_detection_records, limit=failure_case_limit)

    confidence_interval_path = output_path / "confidence_interval_table.csv"
    superiority_path = output_path / "per_attack_superiority_table.csv"
    failure_records_path = output_path / "failure_case_records.jsonl"
    failure_figure_path = output_path / "failure_case_figure.svg"
    summary_path = output_path / "result_analysis_summary.json"
    manifest_path = output_path / "manifest.local.json"

    write_csv(
        confidence_interval_path,
        confidence_interval_rows,
        [
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
        ],
    )
    write_csv(
        superiority_path,
        superiority_rows,
        [
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
        ],
    )
    write_jsonl(failure_records_path, failure_rows)
    failure_figure_path.write_text(build_failure_case_svg(root_path, output_path, failure_rows), encoding="utf-8")

    per_attack_ci_coverage_ready = bool(confidence_interval_rows) and bool(
        template_coverage["result_template_coverage_ready"]
        and len(confidence_interval_rows) == template_coverage["expected_result_record_count"]
        and all(_confidence_interval_row_ready(row) for row in confidence_interval_rows)
    )
    per_attack_superiority_evaluation_ready = bool(superiority_rows) and bool(
        template_coverage["result_template_coverage_ready"]
        and len(superiority_rows) == template_coverage["expected_superiority_row_count"]
        and all(_superiority_evaluation_row_ready(row) for row in superiority_rows)
    )
    superiority_claim_ready_count = sum(1 for row in superiority_rows if row["superiority_claim_ready"])
    universal_per_attack_superiority_claim_ready = bool(
        per_attack_superiority_evaluation_ready
        and superiority_claim_ready_count == len(superiority_rows)
    )

    summary = {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_claim_scale": paper_run.run_name,
        "target_fpr": paper_run.target_fpr,
        "result_record_count": len(result_records),
        "result_record_set_digest": result_record_set_digest,
        "confidence_interval_row_count": len(confidence_interval_rows),
        "per_attack_superiority_row_count": len(superiority_rows),
        "superiority_claim_ready_count": superiority_claim_ready_count,
        "per_attack_ci_coverage_ready": per_attack_ci_coverage_ready,
        "per_attack_superiority_evaluation_ready": per_attack_superiority_evaluation_ready,
        "universal_per_attack_superiority_claim_ready": universal_per_attack_superiority_claim_ready,
        "paired_superiority_row_count": len(paired_rows),
        "paired_superiority_ready": paired_superiority_ready,
        "overall_paired_superiority_ready": paired_summary.get(
            "overall_paired_superiority_ready", False
        ),
        "paired_outcome_set_digest": paired_summary.get(
            "paired_outcome_set_digest", ""
        ),
        "paired_superiority_rows_digest": paired_summary.get(
            "paired_superiority_rows_digest", ""
        ),
        "paired_superiority_protocol_digest": paired_summary.get(
            "paired_superiority_protocol_digest", ""
        ),
        "paired_test_prompt_count": paired_summary.get(
            "paired_test_prompt_count", 0
        ),
        "paired_test_prompt_id_digest": paired_summary.get(
            "paired_test_prompt_id_digest", ""
        ),
        "paired_attack_registry_digest": paired_summary.get(
            "paired_attack_registry_digest", ""
        ),
        "method_observation_source_sha256_map": paired_summary.get(
            "method_observation_source_sha256_map", {}
        ),
        "threshold_audit_rows_digest": paired_summary.get(
            "threshold_audit_rows_digest", ""
        ),
        "claim_p_value_method": paired_summary.get("claim_p_value_method", ""),
        "sharp_null_diagnostic_method": paired_summary.get(
            "sharp_null_diagnostic_method", ""
        ),
        "bootstrap_analysis_schema": paired_summary.get(
            "bootstrap_analysis_schema", ""
        ),
        "bootstrap_bit_generator": paired_summary.get(
            "bootstrap_bit_generator", ""
        ),
        "bootstrap_quantile_method": paired_summary.get(
            "bootstrap_quantile_method", ""
        ),
        "bootstrap_resample_count": paired_summary.get(
            "bootstrap_resample_count", 0
        ),
        "confidence_level": paired_summary.get("confidence_level", 0.0),
        "failure_case_record_count": len(failure_rows),
        "failure_case_figure_ready": failure_figure_path.is_file(),
        **template_coverage,
        "supports_paper_claim": per_attack_ci_coverage_ready
        and per_attack_superiority_evaluation_ready
        and paired_superiority_ready,
    }
    write_json(summary_path, summary)
    manifest = build_artifact_manifest(
        artifact_id="pilot_paper_result_analysis_manifest",
        artifact_type="local_manifest",
        input_paths=(
            relative_or_absolute(resolved_result_records_path, root_path),
            relative_or_absolute(resolved_attack_records_path, root_path),
            relative_or_absolute(resolved_paired_summary_path, root_path),
            relative_or_absolute(resolved_paired_table_path, root_path),
            relative_or_absolute(resolved_paired_manifest_path, root_path),
        ),
        output_paths=(
            relative_or_absolute(confidence_interval_path, root_path),
            relative_or_absolute(superiority_path, root_path),
            relative_or_absolute(failure_records_path, root_path),
            relative_or_absolute(failure_figure_path, root_path),
            relative_or_absolute(summary_path, root_path),
            relative_or_absolute(manifest_path, root_path),
        ),
        config={
            "failure_case_limit": int(failure_case_limit),
            "primary_baseline_method_ids": list(PRIMARY_BASELINE_METHOD_IDS),
            "proposed_method_id": PROPOSED_METHOD_ID,
            "result_record_set_digest": result_record_set_digest,
            "paired_superiority_rows_digest": paired_summary.get(
                "paired_superiority_rows_digest", ""
            ),
            "paired_superiority_protocol_digest": paired_summary.get(
                "paired_superiority_protocol_digest", ""
            ),
            "paired_test_prompt_count": paired_summary.get(
                "paired_test_prompt_count", 0
            ),
            "paired_test_prompt_id_digest": paired_summary.get(
                "paired_test_prompt_id_digest", ""
            ),
            "paired_attack_registry_digest": paired_summary.get(
                "paired_attack_registry_digest", ""
            ),
            "method_observation_source_sha256_map": paired_summary.get(
                "method_observation_source_sha256_map", {}
            ),
            "threshold_audit_rows_digest": paired_summary.get(
                "threshold_audit_rows_digest", ""
            ),
            "claim_p_value_method": paired_summary.get(
                "claim_p_value_method", ""
            ),
            "sharp_null_diagnostic_method": paired_summary.get(
                "sharp_null_diagnostic_method", ""
            ),
            "bootstrap_analysis_schema": paired_summary.get(
                "bootstrap_analysis_schema", ""
            ),
            "bootstrap_bit_generator": paired_summary.get(
                "bootstrap_bit_generator", ""
            ),
            "bootstrap_quantile_method": paired_summary.get(
                "bootstrap_quantile_method", ""
            ),
            "bootstrap_resample_count": paired_summary.get(
                "bootstrap_resample_count", 0
            ),
            "confidence_level": paired_summary.get("confidence_level", 0.0),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_pilot_paper_result_analysis_outputs.py",
        metadata=summary,
    ).to_dict()
    write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="重建 pilot_paper 论文结果分析表与失败案例图。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录; 默认写入 outputs/pilot_paper_result_analysis/<paper_run_name>。",
    )
    parser.add_argument(
        "--result-records-path",
        default=None,
        help="当前论文运行的受治理结果记录 JSONL。",
    )
    parser.add_argument(
        "--attack-detection-records-path",
        default=None,
        help="当前论文运行攻击矩阵中的统一真实检测记录 JSONL。",
    )
    parser.add_argument("--paired-superiority-summary-path", default=None)
    parser.add_argument("--paired-superiority-table-path", default=None)
    parser.add_argument("--paired-superiority-manifest-path", default=None)
    parser.add_argument("--failure-case-limit", type=int, default=12, help="失败案例图最多展示的样本数。")
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    manifest = write_pilot_paper_result_analysis_outputs(
        root=args.root,
        output_dir=args.output_dir,
        result_records_path=args.result_records_path,
        attack_detection_records_path=args.attack_detection_records_path,
        paired_superiority_summary_path=args.paired_superiority_summary_path,
        paired_superiority_table_path=args.paired_superiority_table_path,
        paired_superiority_manifest_path=args.paired_superiority_manifest_path,
        failure_case_limit=args.failure_case_limit,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
