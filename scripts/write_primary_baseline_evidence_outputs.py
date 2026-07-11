"""写出主表 external baseline 证据边界审计产物。"""

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

from paper_experiments.baselines.primary_evidence import (
    build_primary_baseline_evidence_records,
    build_primary_baseline_evidence_summary,
    load_optional_json,
)
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest

DEFAULT_OUTPUT_DIR = Path("outputs/primary_baseline_evidence")
DEFAULT_SOURCE_REGISTRY_PATH = Path("external_baseline/source_registry.json")
DEFAULT_COMMAND_RESULTS_PATH = Path("outputs/external_baseline_method_faithful/execution/baseline_command_results.json")
DEFAULT_OBSERVATIONS_PATH = Path("outputs/external_baseline_method_faithful/execution/baseline_observations.json")
DEFAULT_SPLIT_OBSERVATIONS_DIR = Path("outputs/external_baseline_method_faithful/split_observations")
PACKAGE_SOURCE_REGISTRY_ENTRY = "external_baseline/source_registry.json"
PACKAGE_COMMAND_RESULTS_ENTRY = "outputs/external_baseline_method_faithful/execution/baseline_command_results.json"
PACKAGE_OBSERVATIONS_ENTRY = "outputs/external_baseline_method_faithful/execution/baseline_observations.json"
PACKAGE_SPLIT_ENTRY_PREFIX = "outputs/external_baseline_method_faithful/split_observations/"


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
    """确保输出目录位于 outputs/ 下。"""

    resolved = resolve_path(root_path, output_dir)
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError(f"主表 baseline 证据输出目录必须位于 outputs/ 下: {resolved}") from exc
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


def load_json_from_package(package_path: Path, entry_name: str) -> Any:
    """从结果 zip 包中读取 JSON 文件, 缺失时返回空列表。"""

    if not package_path.is_file():
        return []
    with ZipFile(package_path) as archive:
        if entry_name not in archive.namelist():
            return []
        with archive.open(entry_name) as handle:
            return json.loads(handle.read().decode("utf-8-sig"))


def load_split_json_from_package(package_path: Path, suffix: str) -> list[dict[str, Any]]:
    """从结果包读取单 baseline 拆分 JSON 数组。"""

    if not package_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with ZipFile(package_path) as archive:
        for entry_name in sorted(archive.namelist()):
            normalized_name = entry_name.replace("\\", "/")
            if not normalized_name.startswith(PACKAGE_SPLIT_ENTRY_PREFIX) or not normalized_name.endswith(suffix):
                continue
            with archive.open(entry_name) as handle:
                payload = json.loads(handle.read().decode("utf-8-sig"))
            if isinstance(payload, list):
                rows.extend(dict(row) for row in payload)
    return rows


def load_json_array(path: Path) -> list[dict[str, Any]]:
    """读取 JSON 数组文件, 缺失或内容不匹配时返回空列表。"""

    payload = load_optional_json(path)
    return [dict(row) for row in payload] if isinstance(payload, list) else []


def merge_split_json_rows(base_rows: Any, split_dir: Path, suffix: str) -> list[dict[str, Any]]:
    """合并主 execution JSON 与单 baseline 拆分 JSON。"""

    rows = [dict(row) for row in base_rows] if isinstance(base_rows, list) else []
    if split_dir.is_dir():
        for path in sorted(split_dir.glob(f"*{suffix}")):
            rows.extend(load_json_array(path))
    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        digest = build_stable_digest(row)
        if digest in seen:
            continue
        seen.add(digest)
        deduplicated.append(row)
    return deduplicated


def write_primary_baseline_evidence_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    source_registry_path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH,
    command_results_path: str | Path = DEFAULT_COMMAND_RESULTS_PATH,
    observations_path: str | Path = DEFAULT_OBSERVATIONS_PATH,
    method_faithful_package_path: str | Path | None = None,
    paper_run_prompt_protocol_ready: bool = False,
    fixed_fpr_baseline_calibration_ready: bool = False,
    attack_matrix_baseline_detection_ready: bool = False,
    formal_evidence_paths_ready: bool = False,
) -> dict[str, Any]:
    """写出主表 baseline method-faithful 证据与正式结果缺口记录。"""

    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, output_dir)
    resolved_source_registry_path = resolve_path(root_path, source_registry_path)
    resolved_command_results_path = resolve_path(root_path, command_results_path)
    resolved_observations_path = resolve_path(root_path, observations_path)
    resolved_method_faithful_package_path = resolve_path(root_path, method_faithful_package_path) if method_faithful_package_path else None

    if resolved_method_faithful_package_path:
        source_registry = load_optional_json(resolved_source_registry_path) or load_json_from_package(
            resolved_method_faithful_package_path,
            PACKAGE_SOURCE_REGISTRY_ENTRY,
        ) or {}
        package_command_results = load_json_from_package(resolved_method_faithful_package_path, PACKAGE_COMMAND_RESULTS_ENTRY) or []
        command_results = package_command_results if isinstance(package_command_results, list) else []
        command_results = merge_split_json_rows(
            command_results + load_split_json_from_package(resolved_method_faithful_package_path, "_baseline_command_results.json"),
            resolved_command_results_path.parent.parent / DEFAULT_SPLIT_OBSERVATIONS_DIR.name,
            "_baseline_command_results.json",
        )
        package_observation_rows = load_json_from_package(resolved_method_faithful_package_path, PACKAGE_OBSERVATIONS_ENTRY) or []
        observation_rows = package_observation_rows if isinstance(package_observation_rows, list) else []
        observation_rows = merge_split_json_rows(
            observation_rows + load_split_json_from_package(resolved_method_faithful_package_path, "_baseline_observations.json"),
            resolved_observations_path.parent.parent / DEFAULT_SPLIT_OBSERVATIONS_DIR.name,
            "_baseline_observations.json",
        )
    else:
        source_registry = load_optional_json(resolved_source_registry_path) or {}
        command_results = merge_split_json_rows(
            load_optional_json(resolved_command_results_path) or [],
            resolved_command_results_path.parent.parent / DEFAULT_SPLIT_OBSERVATIONS_DIR.name,
            "_baseline_command_results.json",
        )
        observation_rows = merge_split_json_rows(
            load_optional_json(resolved_observations_path) or [],
            resolved_observations_path.parent.parent / DEFAULT_SPLIT_OBSERVATIONS_DIR.name,
            "_baseline_observations.json",
        )
    records = build_primary_baseline_evidence_records(
        source_registry=source_registry,
        command_results=command_results,
        observation_rows=observation_rows,
        paper_run_prompt_protocol_ready=paper_run_prompt_protocol_ready,
        fixed_fpr_baseline_calibration_ready=fixed_fpr_baseline_calibration_ready,
        attack_matrix_baseline_detection_ready=attack_matrix_baseline_detection_ready,
        formal_evidence_paths_ready=formal_evidence_paths_ready,
    )
    summary = build_primary_baseline_evidence_summary(records)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["source_registry_path"] = relative_or_absolute(resolved_source_registry_path, root_path)
    summary["command_results_path"] = relative_or_absolute(resolved_command_results_path, root_path)
    summary["observations_path"] = relative_or_absolute(resolved_observations_path, root_path)
    summary["method_faithful_package_path"] = (
        relative_or_absolute(resolved_method_faithful_package_path, root_path) if resolved_method_faithful_package_path else ""
    )

    records_path = resolved_output_dir / "primary_baseline_evidence_records.jsonl"
    summary_path = resolved_output_dir / "primary_baseline_evidence_summary.json"
    manifest_path = resolved_output_dir / "manifest.local.json"
    records_path.write_text("".join(json_line(row) for row in records), encoding="utf-8")
    summary_path.write_text(stable_json_text(summary), encoding="utf-8")

    input_paths = []
    for candidate in (resolved_source_registry_path, resolved_command_results_path, resolved_observations_path):
        if candidate.exists():
            input_paths.append(relative_or_absolute(candidate, root_path))
    if resolved_method_faithful_package_path and resolved_method_faithful_package_path.exists():
        input_paths.append(relative_or_absolute(resolved_method_faithful_package_path, root_path))
    output_paths = (
        relative_or_absolute(records_path, root_path),
        relative_or_absolute(summary_path, root_path),
        relative_or_absolute(manifest_path, root_path),
    )
    manifest = build_artifact_manifest(
        artifact_id="primary_baseline_evidence_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(input_paths),
        output_paths=output_paths,
        config={
            "records_digest": build_stable_digest(records),
            "summary_digest": build_stable_digest(summary),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_primary_baseline_evidence_outputs.py",
        metadata=summary,
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="写出主表 external baseline 证据边界审计产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--source-registry-path", default=str(DEFAULT_SOURCE_REGISTRY_PATH))
    parser.add_argument("--command-results-path", default=str(DEFAULT_COMMAND_RESULTS_PATH))
    parser.add_argument("--observations-path", default=str(DEFAULT_OBSERVATIONS_PATH))
    parser.add_argument("--method-faithful-package-path", default=None, help="可选 external baseline method-faithful 结果 zip 包。")
    parser.add_argument("--paper-run-prompt-protocol-ready", action="store_true")
    parser.add_argument("--fixed-fpr-baseline-calibration-ready", action="store_true")
    parser.add_argument("--attack-matrix-baseline-detection-ready", action="store_true")
    parser.add_argument("--formal-evidence-paths-ready", action="store_true")
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    manifest = write_primary_baseline_evidence_outputs(
        root=args.root,
        output_dir=args.output_dir,
        source_registry_path=args.source_registry_path,
        command_results_path=args.command_results_path,
        observations_path=args.observations_path,
        method_faithful_package_path=args.method_faithful_package_path,
        paper_run_prompt_protocol_ready=args.paper_run_prompt_protocol_ready,
        fixed_fpr_baseline_calibration_ready=args.fixed_fpr_baseline_calibration_ready,
        attack_matrix_baseline_detection_ready=args.attack_matrix_baseline_detection_ready,
        formal_evidence_paths_ready=args.formal_evidence_paths_ready,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()

