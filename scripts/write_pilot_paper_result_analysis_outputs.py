"""重建 pilot_paper 论文结果分析表与失败案例图。

该脚本只读取已经闭合的 records 与 manifests, 不重新运行 GPU 推理, 也不手工
拼接论文结论。它用于把 fixed-FPR 结果记录转换为可直接进入论文图表的
bootstrap CI 表、per-attack superiority 表和失败案例 SVG。
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol.paper_run_config import build_paper_run_config
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest

CONSTRUCTION_UNIT_NAME = "pilot_paper_result_analysis"
DEFAULT_OUTPUT_DIR = Path("outputs/pilot_paper_result_analysis")
DEFAULT_RESULT_RECORDS_PATH = Path("outputs/pilot_paper_fixed_fpr_results/pilot_paper_result_records.jsonl")
DEFAULT_REAL_ATTACK_FORMAL_RECORDS_PATH = Path("outputs/real_attack_evaluation/formal_attack_detection_records.jsonl")
DEFAULT_CONVENTIONAL_ATTACK_FORMAL_RECORDS_PATH = Path(
    "outputs/conventional_geometric_attack_evaluation/formal_attack_detection_records.jsonl"
)
PRIMARY_BASELINE_METHOD_IDS = ("tree_ring", "gaussian_shading", "shallow_diffuse", "t2smark")
PROPOSED_METHOD_ID = "slm_wm_current"


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: dict[str, Any]) -> str:
    """把字典转换为 JSONL 单行。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件; 文件缺失时返回空集合。"""

    if not path.exists():
        return []
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


def build_bootstrap_ci_rows(result_records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """从结果记录重建 bootstrap CI 表。"""

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
                "bootstrap_iteration_count": record.get("bootstrap_iteration_count", ""),
                "confidence_level": record.get("confidence_level", ""),
                "supports_paper_claim": record.get("supports_paper_claim", False),
            }
        )
    return sorted(rows, key=lambda row: (row["attack_name"], row["method_id"]))


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
    """生成失败案例 SVG 图。"""

    card_width = 260
    card_height = 250
    columns = 3
    rows = max(1, (len(failure_cases) + columns - 1) // columns)
    width = columns * card_width + 40
    height = rows * card_height + 80
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial, sans-serif;} .title{font-size:18px;font-weight:700;} '
        '.label{font-size:11px;} .small{font-size:10px;fill:#333;} .card{fill:#fff;stroke:#bbb;stroke-width:1;} '
        ".placeholder{fill:#f1f3f5;stroke:#ccd;stroke-width:1;}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="28" class="title">SLM-WM failure cases under fixed-FPR attack protocol</text>',
        '<text x="20" y="48" class="small">Each panel links to the governed attacked image path when extracted with outputs/.</text>',
    ]
    for index, item in enumerate(failure_cases):
        column = index % columns
        row = index // columns
        x = 20 + column * card_width
        y = 65 + row * card_height
        image_path_text = str(item.get("attacked_image_path", ""))
        image_abs_path = root_path / image_path_text if image_path_text else None
        href = (
            Path(
                os.path.relpath(
                    image_abs_path,
                    output_dir,
                )
            ).as_posix()
            if image_abs_path is not None
            else ""
        )
        svg_parts.append(f'<rect x="{x}" y="{y}" width="{card_width - 16}" height="{card_height - 14}" class="card"/>')
        svg_parts.append(f'<rect x="{x + 12}" y="{y + 12}" width="220" height="150" class="placeholder"/>')
        if href:
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
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    result_records_path: str | Path = DEFAULT_RESULT_RECORDS_PATH,
    real_attack_formal_records_path: str | Path = DEFAULT_REAL_ATTACK_FORMAL_RECORDS_PATH,
    conventional_attack_formal_records_path: str | Path = DEFAULT_CONVENTIONAL_ATTACK_FORMAL_RECORDS_PATH,
    failure_case_limit: int = 12,
) -> dict[str, Any]:
    """写出 pilot_paper 结果分析表和失败案例图。"""

    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(root_path)
    output_path = ensure_output_dir_under_outputs(root_path, output_dir)
    resolved_result_records_path = resolve_input_path(root_path, result_records_path)
    resolved_real_attack_path = resolve_input_path(root_path, real_attack_formal_records_path)
    resolved_conventional_attack_path = resolve_input_path(root_path, conventional_attack_formal_records_path)

    result_records = read_jsonl_rows(resolved_result_records_path)
    formal_detection_records = read_jsonl_rows(resolved_real_attack_path) + read_jsonl_rows(resolved_conventional_attack_path)
    bootstrap_rows = build_bootstrap_ci_rows(result_records)
    superiority_rows = build_per_attack_superiority_rows(result_records)
    failure_rows = build_failure_case_records(formal_detection_records, limit=failure_case_limit)

    bootstrap_path = output_path / "bootstrap_ci_table.csv"
    superiority_path = output_path / "per_attack_superiority_table.csv"
    failure_records_path = output_path / "failure_case_records.jsonl"
    failure_figure_path = output_path / "failure_case_figure.svg"
    summary_path = output_path / "result_analysis_summary.json"
    manifest_path = output_path / "manifest.local.json"

    write_csv(
        bootstrap_path,
        bootstrap_rows,
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
            "bootstrap_iteration_count",
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

    summary = {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_claim_scale": paper_run.run_name,
        "target_fpr": paper_run.target_fpr,
        "result_record_count": len(result_records),
        "bootstrap_ci_row_count": len(bootstrap_rows),
        "per_attack_superiority_row_count": len(superiority_rows),
        "superiority_claim_ready_count": sum(1 for row in superiority_rows if row["superiority_claim_ready"]),
        "failure_case_record_count": len(failure_rows),
        "failure_case_figure_ready": failure_figure_path.exists() and bool(failure_rows),
        "supports_paper_claim": bool(superiority_rows) and all(row["supports_paper_claim"] for row in superiority_rows),
    }
    write_json(summary_path, summary)
    manifest = build_artifact_manifest(
        artifact_id="pilot_paper_result_analysis_manifest",
        artifact_type="local_manifest",
        input_paths=(
            relative_or_absolute(resolved_result_records_path, root_path),
            relative_or_absolute(resolved_real_attack_path, root_path),
            relative_or_absolute(resolved_conventional_attack_path, root_path),
        ),
        output_paths=(
            relative_or_absolute(bootstrap_path, root_path),
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
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--result-records-path", default=str(DEFAULT_RESULT_RECORDS_PATH), help="pilot_paper 结果记录 JSONL。")
    parser.add_argument(
        "--real-attack-formal-records-path",
        default=str(DEFAULT_REAL_ATTACK_FORMAL_RECORDS_PATH),
        help="真实再扩散攻击正式检测记录 JSONL。",
    )
    parser.add_argument(
        "--conventional-attack-formal-records-path",
        default=str(DEFAULT_CONVENTIONAL_ATTACK_FORMAL_RECORDS_PATH),
        help="常规失真与几何攻击正式检测记录 JSONL。",
    )
    parser.add_argument("--failure-case-limit", type=int, default=12, help="失败案例图最多展示的样本数。")
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    manifest = write_pilot_paper_result_analysis_outputs(
        root=args.root,
        output_dir=args.output_dir,
        result_records_path=args.result_records_path,
        real_attack_formal_records_path=args.real_attack_formal_records_path,
        conventional_attack_formal_records_path=args.conventional_attack_formal_records_path,
        failure_case_limit=args.failure_case_limit,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
