"""写出主表外部 baseline 官方复现计划和结果导入模板。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.baselines import (
    build_primary_baseline_execution_plans,
    build_primary_baseline_report,
    build_primary_result_templates,
    load_baseline_source_registry,
)
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.runtime.repository_environment import resolve_code_version
from main.core.digest import build_stable_digest

DEFAULT_OUTPUT_DIR = Path("outputs/primary_baseline_reproduction")
DEFAULT_SOURCE_REGISTRY_PATH = Path("external_baseline/source_registry.json")
DEFAULT_ATTACK_MANIFEST_PATH = Path("outputs/attack_matrix/attack_manifest.json")
DEFAULT_ATTACK_FAMILY_METRICS_PATH = Path("outputs/attack_matrix/attack_family_metrics.csv")


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: dict[str, Any]) -> str:
    """把字典转换为 JSONL 单行。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """读取 CSV 表格。"""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_output_dir_under_outputs(root_path: Path, output_dir: Path) -> Path:
    """确保持久化输出目录位于 outputs 下。"""
    resolved_output_dir = (root_path / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved_output_dir.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("主表 baseline 复现计划输出目录必须位于 outputs/ 下") from exc
    return resolved_output_dir


def resolve_input_path(root_path: Path, path: str | Path) -> Path:
    """解析输入路径。"""
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """将路径尽量转换为相对仓库根目录的字符串。"""
    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_primary_baseline_reproduction_plan(
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    source_registry_path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH,
    attack_manifest_path: str | Path = DEFAULT_ATTACK_MANIFEST_PATH,
    attack_family_metrics_path: str | Path = DEFAULT_ATTACK_FAMILY_METRICS_PATH,
) -> dict[str, Any]:
    """写出主表 baseline 官方复现计划和共同协议结果导入模板。"""
    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, Path(output_dir))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_source_registry_path = resolve_input_path(root_path, source_registry_path)
    resolved_attack_manifest_path = resolve_input_path(root_path, attack_manifest_path)
    resolved_attack_family_metrics_path = resolve_input_path(root_path, attack_family_metrics_path)

    source_registry = load_baseline_source_registry(resolved_source_registry_path)
    attack_manifest = read_json(resolved_attack_manifest_path)
    attack_rows = read_csv_rows(resolved_attack_family_metrics_path)
    boundary = attack_manifest.get("evaluation_boundary", {})

    execution_plans = build_primary_baseline_execution_plans(source_registry, root=root_path)
    result_templates = build_primary_result_templates(execution_plans, attack_rows, boundary)
    report = build_primary_baseline_report(execution_plans, result_templates)
    report = {
        **report,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    plan_path = resolved_output_dir / "primary_baseline_execution_plan.jsonl"
    template_path = resolved_output_dir / "primary_baseline_result_record_template.jsonl"
    report_path = resolved_output_dir / "primary_baseline_reproduction_report.json"
    manifest_path = resolved_output_dir / "manifest.local.json"

    plan_path.write_text("".join(json_line(row) for row in execution_plans), encoding="utf-8")
    template_path.write_text("".join(json_line(row) for row in result_templates), encoding="utf-8")
    report_path.write_text(stable_json_text(report), encoding="utf-8")

    output_paths = tuple(
        relative_or_absolute(path, root_path) for path in (plan_path, template_path, report_path, manifest_path)
    )
    input_paths = (
        relative_or_absolute(resolved_source_registry_path, root_path),
        relative_or_absolute(resolved_attack_manifest_path, root_path),
        relative_or_absolute(resolved_attack_family_metrics_path, root_path),
    )
    manifest = build_artifact_manifest(
        artifact_id="primary_baseline_reproduction_manifest",
        artifact_type="local_manifest",
        input_paths=input_paths,
        output_paths=output_paths,
        config={
            "source_registry_digest": build_stable_digest(source_registry),
            "execution_plan_digest": build_stable_digest(execution_plans),
            "result_template_digest": build_stable_digest(result_templates),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_primary_baseline_reproduction_plan.py",
        metadata=report,
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="写出主表外部 baseline 官方复现计划和结果导入模板。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--source-registry-path", default=str(DEFAULT_SOURCE_REGISTRY_PATH), help="外部 baseline 源码登记路径。")
    parser.add_argument("--attack-manifest-path", default=str(DEFAULT_ATTACK_MANIFEST_PATH), help="攻击矩阵 manifest 路径。")
    parser.add_argument(
        "--attack-family-metrics-path",
        default=str(DEFAULT_ATTACK_FAMILY_METRICS_PATH),
        help="攻击矩阵 family metrics 表路径。",
    )
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    manifest = write_primary_baseline_reproduction_plan(
        root=args.root,
        output_dir=args.output_dir,
        source_registry_path=args.source_registry_path,
        attack_manifest_path=args.attack_manifest_path,
        attack_family_metrics_path=args.attack_family_metrics_path,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()

