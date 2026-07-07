"""写出主表 external baseline 正式结果导入协议产物。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.baselines import (
    build_primary_baseline_formal_evidence_collection_rows,
    build_primary_baseline_formal_evidence_collection_summary,
    build_primary_baseline_formal_import_readiness_rows,
    build_primary_baseline_formal_import_readiness_summary,
    build_primary_baseline_formal_import_schema,
    build_primary_baseline_formal_template_coverage_rows,
    build_primary_baseline_formal_template_coverage_summary,
    build_primary_result_templates,
    build_primary_baseline_execution_plans,
    load_baseline_source_registry,
    validate_primary_baseline_formal_import_rows,
)
from experiments.protocol.pilot_paper_fixed_fpr import PILOT_PAPER_FIXED_FPR
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest

DEFAULT_OUTPUT_DIR = Path("outputs/primary_baseline_formal_import")
DEFAULT_SOURCE_REGISTRY_PATH = Path("external_baseline/source_registry.json")
DEFAULT_ATTACK_MANIFEST_PATH = Path("outputs/attack_matrix/attack_manifest.json")
DEFAULT_ATTACK_FAMILY_METRICS_PATH = Path("outputs/attack_matrix/attack_family_metrics.csv")
DEFAULT_CANDIDATE_RECORDS_PATH = Path("outputs/external_baseline_results/baseline_result_records.jsonl")


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: dict[str, Any]) -> str:
    """把字典转换为 JSONL 单行。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """按固定字段顺序写出 CSV 表格。"""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""

    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """读取 CSV 表格。"""

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 记录, 缺失时返回空集合。"""

    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_code_version(root_path: Path) -> str:
    """读取 Git 提交标识, 工作区有变更时追加 dirty 标记。"""

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


def ensure_output_dir_under_outputs(root_path: Path, output_dir: Path) -> Path:
    """确保正式导入协议输出目录位于 outputs 下。"""

    resolved_output_dir = (root_path / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved_output_dir.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("主表 baseline 正式导入协议输出目录必须位于 outputs/ 下。") from exc
    return resolved_output_dir


def resolve_input_path(root_path: Path, path: str | Path) -> Path:
    """解析输入路径。"""

    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """将路径尽量转为相对仓库根目录的字符串。"""

    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_primary_baseline_formal_import_protocol_outputs(
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    source_registry_path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH,
    attack_manifest_path: str | Path = DEFAULT_ATTACK_MANIFEST_PATH,
    attack_family_metrics_path: str | Path = DEFAULT_ATTACK_FAMILY_METRICS_PATH,
    candidate_records_path: str | Path = DEFAULT_CANDIDATE_RECORDS_PATH,
) -> dict[str, Any]:
    """写出正式导入 schema、模板、候选记录校验报告和 manifest。"""

    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, Path(output_dir))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_source_registry_path = resolve_input_path(root_path, source_registry_path)
    resolved_attack_manifest_path = resolve_input_path(root_path, attack_manifest_path)
    resolved_attack_family_metrics_path = resolve_input_path(root_path, attack_family_metrics_path)
    resolved_candidate_records_path = resolve_input_path(root_path, candidate_records_path)

    source_registry = load_baseline_source_registry(resolved_source_registry_path)
    attack_manifest = read_json(resolved_attack_manifest_path)
    attack_rows = read_csv_rows(resolved_attack_family_metrics_path)
    target_fpr = float(attack_manifest.get("evaluation_boundary", {}).get("target_fpr", PILOT_PAPER_FIXED_FPR))
    execution_plans = build_primary_baseline_execution_plans(source_registry, root=root_path)
    template_rows = build_primary_result_templates(execution_plans, attack_rows, attack_manifest.get("evaluation_boundary", {}))
    formal_template_rows = [row for row in template_rows if str(row.get("resource_profile", "")) == "full_main"]
    schema = build_primary_baseline_formal_import_schema(target_fpr=target_fpr, root=root_path)
    candidate_rows = read_jsonl_rows(resolved_candidate_records_path)
    formal_candidate_rows = [row for row in candidate_rows if str(row.get("resource_profile", "")) == "full_main"]
    validation_report = validate_primary_baseline_formal_import_rows(
        formal_candidate_rows,
        evidence_root=root_path,
        target_fpr=target_fpr,
        require_existing_evidence=True,
        prompt_protocol_name=str(schema["prompt_protocol_name"]),
    )
    readiness_rows = build_primary_baseline_formal_import_readiness_rows(formal_candidate_rows, validation_report)
    readiness_summary = build_primary_baseline_formal_import_readiness_summary(readiness_rows)
    coverage_rows = build_primary_baseline_formal_template_coverage_rows(
        formal_template_rows,
        formal_candidate_rows,
        validation_report,
    )
    coverage_summary = build_primary_baseline_formal_template_coverage_summary(coverage_rows)
    collection_rows = build_primary_baseline_formal_evidence_collection_rows(
        formal_template_rows,
        formal_candidate_rows,
        validation_report,
    )
    collection_summary = build_primary_baseline_formal_evidence_collection_summary(collection_rows)
    summary = {
        "construction_unit_name": "primary_baseline_formal_import_protocol",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_fpr": target_fpr,
        "template_record_count": len(formal_template_rows),
        "candidate_record_count": len(candidate_rows),
        "formal_candidate_record_count": len(formal_candidate_rows),
        "accepted_formal_import_count": validation_report["accepted_formal_import_count"],
        "rejected_formal_import_count": validation_report["rejected_formal_import_count"],
        "formal_import_validation_ready": validation_report["formal_import_validation_ready"],
        "formal_result_ready_count": readiness_summary["formal_result_ready_count"],
        "formal_template_coverage_ready_count": coverage_summary["formal_template_coverage_ready_count"],
        "candidate_template_match_count": coverage_summary["candidate_template_match_count"],
        "accepted_template_match_count": coverage_summary["accepted_template_match_count"],
        "missing_candidate_template_count": coverage_summary["missing_candidate_template_count"],
        "missing_formal_template_count": coverage_summary["missing_formal_template_count"],
        "formal_evidence_collection_task_count": collection_summary["formal_evidence_collection_task_count"],
        "missing_formal_evidence_collection_task_count": collection_summary[
            "missing_formal_evidence_collection_task_count"
        ],
        "primary_baseline_formal_ready": readiness_summary["primary_baseline_formal_ready"]
        and coverage_summary["primary_baseline_formal_template_coverage_ready"]
        and collection_summary["primary_baseline_formal_evidence_collection_ready"],
        "supports_paper_claim": False,
    }

    schema_path = resolved_output_dir / "primary_baseline_formal_import_schema.json"
    template_path = resolved_output_dir / "primary_baseline_formal_result_template.jsonl"
    validation_path = resolved_output_dir / "primary_baseline_formal_import_validation_report.json"
    readiness_path = resolved_output_dir / "primary_baseline_formal_import_readiness.csv"
    readiness_summary_path = resolved_output_dir / "primary_baseline_formal_import_readiness_summary.json"
    coverage_path = resolved_output_dir / "primary_baseline_formal_template_coverage.csv"
    coverage_summary_path = resolved_output_dir / "primary_baseline_formal_template_coverage_summary.json"
    collection_path = resolved_output_dir / "primary_baseline_formal_evidence_collection_plan.jsonl"
    collection_summary_path = resolved_output_dir / "primary_baseline_formal_evidence_collection_summary.json"
    summary_path = resolved_output_dir / "primary_baseline_formal_import_summary.json"
    manifest_path = resolved_output_dir / "manifest.local.json"

    schema_path.write_text(stable_json_text(schema), encoding="utf-8")
    template_path.write_text("".join(json_line(row) for row in formal_template_rows), encoding="utf-8")
    validation_path.write_text(stable_json_text(validation_report), encoding="utf-8")
    write_csv(
        readiness_path,
        readiness_rows,
        [
            "baseline_id",
            "candidate_record_count",
            "accepted_formal_import_count",
            "rejected_formal_import_count",
            "formal_import_issue_count",
            "formal_result_ready",
            "blocking_reason_count",
            "blocking_reasons",
            "missing_resource_profile_full_main",
            "missing_full_main_prompt_protocol",
            "missing_fixed_fpr_baseline_calibration",
            "missing_attack_matrix_baseline_detection",
            "formal_evidence_paths_ready",
            "supports_paper_claim",
        ],
    )
    readiness_summary_path.write_text(stable_json_text(readiness_summary), encoding="utf-8")
    write_csv(
        coverage_path,
        coverage_rows,
        [
            "baseline_id",
            "expected_formal_template_count",
            "candidate_template_match_count",
            "accepted_template_match_count",
            "missing_formal_template_count",
            "formal_template_coverage_ready",
            "supports_paper_claim",
        ],
    )
    coverage_summary_path.write_text(stable_json_text(coverage_summary), encoding="utf-8")
    collection_path.write_text("".join(json_line(row) for row in collection_rows), encoding="utf-8")
    collection_summary_path.write_text(stable_json_text(collection_summary), encoding="utf-8")
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")

    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            schema_path,
            template_path,
            validation_path,
            readiness_path,
            readiness_summary_path,
            coverage_path,
            coverage_summary_path,
            collection_path,
            collection_summary_path,
            summary_path,
            manifest_path,
        )
    )
    input_paths = [
        relative_or_absolute(resolved_source_registry_path, root_path),
        relative_or_absolute(resolved_attack_manifest_path, root_path),
        relative_or_absolute(resolved_attack_family_metrics_path, root_path),
    ]
    if resolved_candidate_records_path.exists():
        input_paths.append(relative_or_absolute(resolved_candidate_records_path, root_path))
    manifest = build_artifact_manifest(
        artifact_id="primary_baseline_formal_import_protocol_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(input_paths),
        output_paths=output_paths,
        config={
            "schema_digest": build_stable_digest(schema),
            "template_digest": build_stable_digest(formal_template_rows),
            "validation_report_digest": build_stable_digest(validation_report),
            "formal_import_readiness_digest": build_stable_digest(readiness_rows),
            "formal_import_readiness_summary_digest": build_stable_digest(readiness_summary),
            "formal_template_coverage_digest": build_stable_digest(coverage_rows),
            "formal_template_coverage_summary_digest": build_stable_digest(coverage_summary),
            "formal_evidence_collection_plan_digest": build_stable_digest(collection_rows),
            "formal_evidence_collection_summary_digest": build_stable_digest(collection_summary),
            "summary_digest": build_stable_digest(summary),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_primary_baseline_formal_import_protocol.py",
        metadata=summary,
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="写出主表 external baseline 正式结果导入协议产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--source-registry-path", default=str(DEFAULT_SOURCE_REGISTRY_PATH), help="外部 baseline 源码登记路径。")
    parser.add_argument("--attack-manifest-path", default=str(DEFAULT_ATTACK_MANIFEST_PATH), help="攻击矩阵 manifest 路径。")
    parser.add_argument("--attack-family-metrics-path", default=str(DEFAULT_ATTACK_FAMILY_METRICS_PATH), help="攻击矩阵 family metrics 表路径。")
    parser.add_argument("--candidate-records-path", default=str(DEFAULT_CANDIDATE_RECORDS_PATH), help="待校验 baseline 结果 JSONL 路径。")
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    manifest = write_primary_baseline_formal_import_protocol_outputs(
        root=args.root,
        output_dir=args.output_dir,
        source_registry_path=args.source_registry_path,
        attack_manifest_path=args.attack_manifest_path,
        attack_family_metrics_path=args.attack_family_metrics_path,
        candidate_records_path=args.candidate_records_path,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()

