"""写出 prompt、split 与 event protocol 产物。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from experiments.protocol.calibration import build_prompt_statistics
from experiments.protocol.events import build_event_records
from experiments.protocol.prompts import PROMPT_FILES, load_prompt_records
from experiments.protocol.records import validate_unique_ids, write_event_records, write_prompt_records
from experiments.protocol.splits import SPLIT_NAMES, apply_split_assignments, group_prompt_ids_by_split

PROTOCOL_UNIT_NAME = "prompt_event_protocol"
DEFAULT_OUTPUT_DIR = Path("outputs/prompt_event_protocol")


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定 JSON 文本。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


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
    """确保协议输出目录位于 outputs 下。"""
    resolved_output_dir = (root_path / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved_output_dir.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("协议输出目录必须位于 outputs/ 下。") from exc
    return resolved_output_dir


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """将路径尽量转为相对仓库根目录的可移植字符串。"""
    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def latest_drive_workflow_package(root_path: Path) -> Path | None:
    """查找最近一次 Colab Drive workflow 结果包。"""
    candidates = sorted((root_path / "outputs").glob("colab_drive_workflow-*.zip"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def existing_input_paths(root_path: Path, explicit_input_paths: tuple[Path, ...]) -> tuple[str, ...]:
    """汇总存在的输入路径, 用于 manifest provenance。"""
    paths: list[Path] = []
    for path in explicit_input_paths:
        candidate = path if path.is_absolute() else root_path / path
        if candidate.exists():
            paths.append(candidate)
    latest_package = latest_drive_workflow_package(root_path)
    if latest_package is not None and latest_package not in paths:
        paths.append(latest_package)
    prompt_bank_archive = root_path / "outputs" / "prompts.zip"
    if prompt_bank_archive.exists() and prompt_bank_archive not in paths:
        paths.append(prompt_bank_archive)
    for prompt_path in PROMPT_FILES.values():
        candidate = root_path / prompt_path
        if candidate.exists():
            paths.append(candidate)
    return tuple(relative_or_absolute(path, root_path) for path in paths)


def build_split_manifest(prompt_records: tuple[Any, ...]) -> dict[str, Any]:
    """构造 split manifest。"""
    split_groups = group_prompt_ids_by_split(prompt_records)
    return {
        "construction_unit_name": PROTOCOL_UNIT_NAME,
        "protocol_decision": "pass",
        "split_names": list(SPLIT_NAMES),
        "split_counts": {name: len(ids) for name, ids in split_groups.items()},
        "split_prompt_ids": {name: list(ids) for name, ids in split_groups.items()},
        "supports_paper_claim": False,
    }


def build_prompt_manifest(prompt_records: tuple[Any, ...]) -> dict[str, Any]:
    """构造 prompt manifest。"""
    records = [record.to_dict() for record in prompt_records]
    return {
        "construction_unit_name": PROTOCOL_UNIT_NAME,
        "protocol_decision": "pass" if validate_unique_ids(records, "prompt_id") else "fail",
        "prompt_count": len(records),
        "prompt_records": records,
        "supports_paper_claim": False,
    }


def build_event_manifest(event_records: tuple[Any, ...]) -> dict[str, Any]:
    """构造 event protocol manifest。"""
    records = [record.to_dict() for record in event_records]
    return {
        "construction_unit_name": PROTOCOL_UNIT_NAME,
        "protocol_decision": "pass" if validate_unique_ids(records, "event_id") else "fail",
        "event_count": len(records),
        "event_records": records,
        "supports_paper_claim": False,
    }


def write_prompt_event_protocol_outputs(
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    input_paths: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """写出 prompt、split、event protocol 与 manifest。"""
    root_path = Path(root).resolve()
    resolved_output_dir = ensure_output_dir_under_outputs(root_path, Path(output_dir))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    prompt_files = {name: root_path / path for name, path in PROMPT_FILES.items()}
    prompt_records = apply_split_assignments(load_prompt_records(prompt_files=prompt_files))
    event_records = build_event_records(prompt_records)
    prompt_manifest = build_prompt_manifest(prompt_records)
    split_manifest = build_split_manifest(prompt_records)
    event_manifest = build_event_manifest(event_records)
    statistics = build_prompt_statistics(prompt_records, event_records)

    prompt_records_path = resolved_output_dir / "prompt_records.jsonl"
    event_records_path = resolved_output_dir / "event_records.jsonl"
    prompt_manifest_path = resolved_output_dir / "prompt_manifest.json"
    split_manifest_path = resolved_output_dir / "split_manifest.json"
    event_manifest_path = resolved_output_dir / "event_protocol_manifest.json"
    statistics_path = resolved_output_dir / "prompt_statistics.json"
    manifest_path = resolved_output_dir / "manifest.local.json"

    write_prompt_records(prompt_records_path, prompt_records)
    write_event_records(event_records_path, event_records)
    prompt_manifest_path.write_text(stable_json_text(prompt_manifest), encoding="utf-8")
    split_manifest_path.write_text(stable_json_text(split_manifest), encoding="utf-8")
    event_manifest_path.write_text(stable_json_text(event_manifest), encoding="utf-8")
    statistics_path.write_text(stable_json_text(statistics), encoding="utf-8")

    output_paths = tuple(
        relative_or_absolute(path, root_path)
        for path in (
            prompt_records_path,
            event_records_path,
            prompt_manifest_path,
            split_manifest_path,
            event_manifest_path,
            statistics_path,
            manifest_path,
        )
    )
    resolved_input_paths = existing_input_paths(root_path, tuple(Path(path) for path in input_paths))
    manifest = build_artifact_manifest(
        artifact_id="prompt_event_protocol_manifest",
        artifact_type="local_manifest",
        input_paths=resolved_input_paths,
        output_paths=output_paths,
        config={
            "construction_unit_name": PROTOCOL_UNIT_NAME,
            "prompt_manifest_digest": build_stable_digest(prompt_manifest),
            "split_manifest_digest": build_stable_digest(split_manifest),
            "event_protocol_digest": build_stable_digest(event_manifest),
            "prompt_statistics_digest": build_stable_digest(statistics),
            "prompt_count": statistics["prompt_count"],
            "event_count": statistics["event_count"],
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_prompt_event_protocol.py",
        metadata={
            "construction_unit_name": PROTOCOL_UNIT_NAME,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol_decision": statistics["protocol_decision"],
            "supports_paper_claim": False,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="写出 prompt 与 event protocol 产物。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--input-path", action="append", default=None, help="额外输入路径, 可重复传入。")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    manifest = write_prompt_event_protocol_outputs(
        root=args.root,
        output_dir=args.output_dir,
        input_paths=tuple(args.input_path or ()),
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()

