"""写出数据集级图像质量证据产物。"""

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

from experiments.protocol import (
    build_dataset_quality_image_records,
    build_dataset_quality_metric_rows,
    build_dataset_quality_summary,
)
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest

DEFAULT_OUTPUT_DIR = Path("outputs/dataset_level_quality")
DEFAULT_REAL_ATTACK_REGISTRY_PATH = Path("outputs/real_attack_evaluation/real_attacked_image_registry.jsonl")


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: dict[str, Any]) -> str:
    """把单条记录转换为 JSONL 行。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL registry 行。"""

    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写出 CSV 表格。"""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_path(root_path: Path, path: str | Path) -> Path:
    """将输入路径解析为绝对路径。"""

    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先返回相对仓库根目录路径。"""

    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def ensure_output_dir_under_outputs(root_path: Path, output_dir: str | Path) -> Path:
    """确保持久输出目录位于 outputs/ 下。"""

    resolved = resolve_path(root_path, output_dir)
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("数据集级质量证据输出目录必须位于 outputs/ 下。") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_code_version(root_path: Path) -> str:
    """读取当前 Git 短提交标识, 工作区有变更时附加 dirty 标记。"""

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


def write_dataset_level_quality_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    real_attack_registry_path: str | Path = DEFAULT_REAL_ATTACK_REGISTRY_PATH,
) -> dict[str, Any]:
    """写出数据集级质量 records、metrics、summary 和 manifest。

    当前实现只生成小样本 pixel feature proxy, 明确保持正式 FID / KID 为 unsupported。
    """

    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, output_dir)
    resolved_registry_path = resolve_path(root_path, real_attack_registry_path)
    registry_rows = read_jsonl_rows(resolved_registry_path)

    records = build_dataset_quality_image_records(registry_rows, root_path)
    metric_rows = build_dataset_quality_metric_rows(records, root_path)

    records_path = resolved_output_dir / "dataset_quality_image_records.jsonl"
    metrics_path = resolved_output_dir / "dataset_quality_metrics.csv"
    summary_path = resolved_output_dir / "dataset_quality_summary.json"
    manifest_path = resolved_output_dir / "manifest.local.json"

    summary = {
        **build_dataset_quality_summary(records, metric_rows),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "real_attack_registry_path": relative_or_absolute(resolved_registry_path, root_path),
        "dataset_quality_metrics_path": relative_or_absolute(metrics_path, root_path),
    }

    records_path.write_text("".join(json_line(record.to_dict()) for record in records), encoding="utf-8")
    write_csv(
        metrics_path,
        metric_rows,
        [
            "quality_metric_name",
            "quality_metric_value",
            "metric_status",
            "paper_metric_name",
            "feature_backend",
            "source_image_count",
            "comparison_image_count",
            "sample_pair_count",
            "supports_paper_claim",
        ],
    )
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")

    input_paths = (relative_or_absolute(resolved_registry_path, root_path),) if resolved_registry_path.exists() else ()
    output_paths = tuple(relative_or_absolute(path, root_path) for path in (records_path, metrics_path, summary_path, manifest_path))
    manifest = build_artifact_manifest(
        artifact_id="dataset_level_quality_manifest",
        artifact_type="local_manifest",
        input_paths=input_paths,
        output_paths=output_paths,
        config={
            "records_digest": build_stable_digest([record.to_dict() for record in records]),
            "metric_rows_digest": build_stable_digest(metric_rows),
            "summary_digest": build_stable_digest(summary),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_dataset_level_quality_outputs.py",
        metadata=summary,
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="写出数据集级质量证据产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--real-attack-registry-path", default=str(DEFAULT_REAL_ATTACK_REGISTRY_PATH), help="真实攻击图像 registry 路径。")
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    manifest = write_dataset_level_quality_outputs(
        root=args.root,
        output_dir=args.output_dir,
        real_attack_registry_path=args.real_attack_registry_path,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
