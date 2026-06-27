"""写出主表 external baseline 方法忠实 SD3.5 适配协议产物。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.baselines import (
    build_method_faithful_adapter_status_records,
    build_method_faithful_adapter_summary,
    build_primary_baseline_method_faithful_adapter_schema,
)
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest

DEFAULT_OUTPUT_DIR = Path("outputs/primary_baseline_method_faithful_adapter_protocol")
DEFAULT_OBSERVATIONS_PATH = Path("outputs/external_baseline_method_faithful/execution/baseline_observations.json")
DEFAULT_SPLIT_OBSERVATIONS_DIR = Path("outputs/external_baseline_method_faithful/split_observations")
PACKAGE_OBSERVATIONS_ENTRY = "outputs/external_baseline_method_faithful/execution/baseline_observations.json"
PACKAGE_SPLIT_OBSERVATIONS_ENTRY_PREFIX = "outputs/external_baseline_method_faithful/split_observations/"


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
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
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


def resolve_code_version(root_path: Path) -> str:
    """读取当前 Git 短提交标识。"""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "git_version_unavailable"
    return result.stdout.strip() or "git_version_unavailable"


def load_optional_json(path: Path) -> Any:
    """读取可选 JSON 文件, 缺失时返回空列表。"""

    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_json_from_package(package_path: Path, entry_name: str) -> Any:
    """从结果 zip 包中读取 JSON 文件, 缺失时返回空列表。"""

    if not package_path.is_file():
        return []
    with ZipFile(package_path) as archive:
        if entry_name not in archive.namelist():
            return []
        with archive.open(entry_name) as handle:
            return json.loads(handle.read().decode("utf-8-sig"))


def load_split_json_from_package(package_path: Path, entry_prefix: str) -> list[dict[str, Any]]:
    """从结果包读取单 baseline 拆分 observation。"""

    if not package_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with ZipFile(package_path) as archive:
        for entry_name in sorted(archive.namelist()):
            normalized_name = entry_name.replace("\\", "/")
            if not normalized_name.startswith(entry_prefix) or not normalized_name.endswith("_baseline_observations.json"):
                continue
            with archive.open(entry_name) as handle:
                payload = json.loads(handle.read().decode("utf-8-sig"))
            if isinstance(payload, list):
                rows.extend(dict(row) for row in payload)
    return rows


def load_observation_rows(
    *,
    observations_path: Path,
    method_faithful_package_path: Path | None = None,
) -> list[dict[str, Any]]:
    """读取 observation 行, 并兼容单 baseline 拆分文件。"""

    rows = []
    if method_faithful_package_path:
        package_rows = load_json_from_package(method_faithful_package_path, PACKAGE_OBSERVATIONS_ENTRY)
        rows.extend(package_rows if isinstance(package_rows, list) else [])
        rows.extend(load_split_json_from_package(method_faithful_package_path, PACKAGE_SPLIT_OBSERVATIONS_ENTRY_PREFIX))
    if not rows:
        local_rows = load_optional_json(observations_path)
        rows.extend(local_rows if isinstance(local_rows, list) else [])
    split_dir = observations_path.parent.parent / DEFAULT_SPLIT_OBSERVATIONS_DIR.name
    if split_dir.is_dir():
        for path in sorted(split_dir.glob("*_baseline_observations.json")):
            payload = load_optional_json(path)
            if isinstance(payload, list):
                rows.extend(dict(row) for row in payload)
    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        digest = build_stable_digest(row)
        if digest in seen:
            continue
        seen.add(digest)
        deduplicated.append(dict(row))
    return deduplicated


def write_primary_baseline_method_faithful_adapter_protocol_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    observations_path: str | Path = DEFAULT_OBSERVATIONS_PATH,
    method_faithful_package_path: str | Path | None = None,
) -> dict[str, Any]:
    """写出方法忠实 adapter schema、状态记录、摘要和 manifest。"""

    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, output_dir)
    resolved_observations_path = resolve_path(root_path, observations_path)
    resolved_method_faithful_package_path = resolve_path(root_path, method_faithful_package_path) if method_faithful_package_path else None

    observation_rows = load_observation_rows(
        observations_path=resolved_observations_path,
        method_faithful_package_path=resolved_method_faithful_package_path,
    )
    schema = build_primary_baseline_method_faithful_adapter_schema()
    records = build_method_faithful_adapter_status_records(observation_rows)
    summary = build_method_faithful_adapter_summary(records)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["observations_path"] = relative_or_absolute(resolved_observations_path, root_path)
    summary["method_faithful_package_path"] = (
        relative_or_absolute(resolved_method_faithful_package_path, root_path) if resolved_method_faithful_package_path else ""
    )
    summary["input_observation_count"] = len(observation_rows)

    schema_path = resolved_output_dir / "primary_baseline_method_faithful_adapter_schema.json"
    records_path = resolved_output_dir / "primary_baseline_method_faithful_adapter_status_records.jsonl"
    summary_path = resolved_output_dir / "primary_baseline_method_faithful_adapter_summary.json"
    manifest_path = resolved_output_dir / "manifest.local.json"

    schema_path.write_text(stable_json_text(schema), encoding="utf-8")
    records_path.write_text("".join(json_line(row) for row in records), encoding="utf-8")
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")

    input_paths = []
    if resolved_observations_path.exists():
        input_paths.append(relative_or_absolute(resolved_observations_path, root_path))
    if resolved_method_faithful_package_path and resolved_method_faithful_package_path.exists():
        input_paths.append(relative_or_absolute(resolved_method_faithful_package_path, root_path))
    output_paths = tuple(
        relative_or_absolute(path, root_path) for path in (schema_path, records_path, summary_path, manifest_path)
    )
    manifest = build_artifact_manifest(
        artifact_id="primary_baseline_method_faithful_adapter_protocol_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(input_paths),
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
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--observations-path", default=str(DEFAULT_OBSERVATIONS_PATH), help="baseline observation JSON 路径。")
    parser.add_argument("--method-faithful-package-path", default=None, help="可选 external baseline method-faithful 结果 zip 包。")
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    manifest = write_primary_baseline_method_faithful_adapter_protocol_outputs(
        root=args.root,
        output_dir=args.output_dir,
        observations_path=args.observations_path,
        method_faithful_package_path=args.method_faithful_package_path,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
