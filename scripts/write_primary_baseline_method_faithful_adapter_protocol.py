"""写出主表 method-faithful SD3.5 adapter 协议产物。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.runtime.repository_environment import resolve_code_version
from main.core.digest import build_stable_digest
from paper_experiments.baselines import (
    build_method_faithful_adapter_status_records,
    build_method_faithful_adapter_summary,
    build_primary_baseline_method_faithful_adapter_schema,
)
from paper_experiments.baselines.method_faithful_observation_collection import (
    DEFAULT_METHOD_FAITHFUL_COLLECTION_ROOT,
    load_method_faithful_observation_collection,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/primary_baseline_method_faithful_adapter_protocol")
DEFAULT_COLLECTION_ROOT = DEFAULT_METHOD_FAITHFUL_COLLECTION_ROOT


def stable_json_text(value: Any) -> str:
    """以稳定顺序序列化 JSON。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: dict[str, Any]) -> str:
    """把单条记录写成稳定 JSONL 行。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def resolve_path(root_path: Path, path: str | Path) -> Path:
    """把相对路径解析到仓库根目录。"""

    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def ensure_output_dir_under_outputs(root_path: Path, output_dir: str | Path) -> Path:
    """确保协议产物输出目录位于 outputs/ 下。"""

    resolved = resolve_path(root_path, output_dir)
    try:
        resolved.relative_to((root_path / "outputs").resolve())
    except ValueError as exc:
        raise ValueError(f"方法忠实适配协议输出目录必须位于 outputs/ 下: {resolved}") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录相对仓库根目录的路径。"""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_primary_baseline_method_faithful_adapter_protocol_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
    collection_root: str | Path | None = None,
) -> dict[str, Any]:
    """从 exact-set collection 写出 adapter schema、状态记录与摘要。"""

    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(root_path)
    resolved_output_dir = ensure_output_dir_under_outputs(
        root_path,
        output_dir or DEFAULT_OUTPUT_ROOT / paper_run.run_name,
    )
    resolved_collection_root = resolve_path(
        root_path,
        collection_root or DEFAULT_COLLECTION_ROOT / paper_run.run_name,
    )
    sources = load_method_faithful_observation_collection(
        resolved_collection_root,
        project_root=root_path,
    )
    observation_rows = [dict(row) for source in sources for row in source.rows]
    schema = build_primary_baseline_method_faithful_adapter_schema()
    records = build_method_faithful_adapter_status_records(observation_rows)
    summary = build_method_faithful_adapter_summary(records)
    summary.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "collection_root": relative_or_absolute(resolved_collection_root, root_path),
            "input_baseline_ids": [source.baseline_id for source in sources],
            "input_observation_count": len(observation_rows),
            "source_manifest_digest": build_stable_digest(
                [source.transfer_manifest for source in sources]
            ),
        }
    )

    schema_path = resolved_output_dir / "primary_baseline_method_faithful_adapter_schema.json"
    records_path = resolved_output_dir / "primary_baseline_method_faithful_adapter_status_records.jsonl"
    summary_path = resolved_output_dir / "primary_baseline_method_faithful_adapter_summary.json"
    manifest_path = resolved_output_dir / "manifest.local.json"
    schema_path.write_text(stable_json_text(schema), encoding="utf-8")
    records_path.write_text("".join(json_line(row) for row in records), encoding="utf-8")
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")

    input_paths = tuple(
        relative_or_absolute(path, root_path)
        for source in sources
        for path in (
            source.observations_path,
            source.transfer_manifest_path,
            source.prompt_plan_path,
            source.adapter_manifest_path,
            source.execution_manifest_path,
        )
    )
    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (schema_path, records_path, summary_path, manifest_path)
    )
    manifest = build_artifact_manifest(
        artifact_id="primary_baseline_method_faithful_adapter_protocol_manifest",
        artifact_type="local_manifest",
        input_paths=input_paths,
        output_paths=output_paths,
        config={
            "schema_digest": build_stable_digest(schema),
            "records_digest": build_stable_digest(records),
            "summary_digest": build_stable_digest(summary),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_primary_baseline_method_faithful_adapter_protocol.py",
        metadata=summary,
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="写出主表 external baseline 方法忠实 SD3.5 适配协议产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录; 默认写入当前论文运行子目录, 且必须位于 outputs/ 下。",
    )
    parser.add_argument(
        "--collection-root",
        default=None,
        help="三个 method-faithful baseline 的 exact-set 物化根目录; 默认读取当前论文运行子目录。",
    )
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    manifest = write_primary_baseline_method_faithful_adapter_protocol_outputs(
        root=args.root,
        output_dir=args.output_dir,
        collection_root=args.collection_root,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
