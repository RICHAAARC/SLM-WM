"""组合 Colab Drive workflow 的挂载、镜像、manifest 和重载校验。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from main.core.digest import build_stable_digest

from paper_workflow.colab_utils.drive_paths import (
    DEFAULT_DRIVE_ROOT,
    DEFAULT_LOCAL_OUTPUT_DIR,
    DEFAULT_WORKFLOW_NAME,
    build_drive_workflow_paths,
)
from paper_workflow.colab_utils.manifest_io import (
    build_sync_report,
    discover_local_manifests,
    mirror_files_to_drive,
    register_drive_source_files,
    stable_json_text,
    verify_drive_manifest,
    write_json,
    write_jsonl,
    write_manifest_bundle,
)
from paper_workflow.colab_utils.mount_drive import build_drive_mount_report
from paper_workflow.colab_utils.notebook_runtime import mark_notebook_runtime_start, write_notebook_runtime_report
from paper_workflow.colab_utils.runtime_setup import build_runtime_setup_report


def run_local_output_sync(
    root: str | Path = ".",
    drive_root: str | Path = DEFAULT_DRIVE_ROOT,
    local_output_dir: str | Path = DEFAULT_LOCAL_OUTPUT_DIR,
    workflow_name: str = DEFAULT_WORKFLOW_NAME,
) -> dict[str, Any]:
    """把本地 outputs 中的受治理产物镜像到 Drive workflow 目录。"""
    paths = build_drive_workflow_paths(root, local_output_dir, drive_root, workflow_name)
    paths.local_output_dir.mkdir(parents=True, exist_ok=True)
    paths.drive_local_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_references = discover_local_manifests(paths.repository_root)
    local_records = mirror_files_to_drive(paths.repository_root, paths)
    drive_records = register_drive_source_files(paths)
    mirror_records = tuple(sorted(local_records + drive_records, key=lambda record: record.destination_path))
    sync_report = build_sync_report(paths.repository_root, paths, mirror_records, manifest_references)
    write_json(paths.local_output_dir / "local_output_sync_report.json", sync_report)
    write_json(paths.drive_workflow_dir / "local_output_sync_report.json", sync_report)
    return sync_report


def write_workflow_manifest_files(
    root: str | Path = ".",
    drive_root: str | Path = DEFAULT_DRIVE_ROOT,
    local_output_dir: str | Path = DEFAULT_LOCAL_OUTPUT_DIR,
    workflow_name: str = DEFAULT_WORKFLOW_NAME,
    sync_report: dict[str, Any] | None = None,
    mount_report: dict[str, Any] | None = None,
    runtime_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """写入本地和 Drive 侧的 manifest 文件集合。"""
    paths = build_drive_workflow_paths(root, local_output_dir, drive_root, workflow_name)
    paths.local_output_dir.mkdir(parents=True, exist_ok=True)
    paths.drive_workflow_dir.mkdir(parents=True, exist_ok=True)
    resolved_sync_report = sync_report or run_local_output_sync(root, drive_root, local_output_dir, workflow_name)
    resolved_mount_report = mount_report or build_drive_mount_report(perform_mount=False).to_dict()
    resolved_runtime_report = runtime_report or build_runtime_setup_report()
    manifest = write_manifest_bundle(
        paths.repository_root,
        paths,
        resolved_sync_report,
        resolved_runtime_report,
        resolved_mount_report,
    )
    return manifest


def write_reload_smoke_record(
    root: str | Path = ".",
    drive_root: str | Path = DEFAULT_DRIVE_ROOT,
    local_output_dir: str | Path = DEFAULT_LOCAL_OUTPUT_DIR,
    workflow_name: str = DEFAULT_WORKFLOW_NAME,
) -> dict[str, Any]:
    """从 Drive manifest 重载镜像文件并写出校验记录。"""
    paths = build_drive_workflow_paths(root, local_output_dir, drive_root, workflow_name)
    record = verify_drive_manifest(paths.drive_workflow_dir / "manifest.json").to_dict()
    write_jsonl(paths.local_output_dir / "reload_smoke_record.jsonl", [record])
    write_jsonl(paths.drive_workflow_dir / "reload_smoke_record.jsonl", [record])
    return record


def run_colab_drive_workflow(
    root: str | Path = ".",
    drive_root: str | Path = DEFAULT_DRIVE_ROOT,
    local_output_dir: str | Path = DEFAULT_LOCAL_OUTPUT_DIR,
    workflow_name: str = DEFAULT_WORKFLOW_NAME,
    perform_mount: bool = False,
) -> dict[str, Any]:
    """执行 Colab Drive workflow 的轻量可审计闭环。"""
    mark_notebook_runtime_start(
        workflow_name=workflow_name,
        source="run_colab_drive_workflow",
    )
    paths = build_drive_workflow_paths(root, local_output_dir, drive_root, workflow_name)
    paths.local_output_dir.mkdir(parents=True, exist_ok=True)
    paths.drive_workflow_dir.mkdir(parents=True, exist_ok=True)

    mount_report = build_drive_mount_report(perform_mount=perform_mount).to_dict()
    runtime_report = build_runtime_setup_report()
    sync_report = run_local_output_sync(root, drive_root, local_output_dir, workflow_name)
    manifest = write_workflow_manifest_files(
        root,
        drive_root,
        local_output_dir,
        workflow_name,
        sync_report=sync_report,
        mount_report=mount_report,
        runtime_report=runtime_report,
    )
    reload_record = write_reload_smoke_record(root, drive_root, local_output_dir, workflow_name)
    cold_start_record = {
        "construction_unit_name": "colab_drive_workflow",
        "workflow_name": workflow_name,
        "workflow_decision": reload_record["reload_decision"],
        "manifest_digest": build_stable_digest(manifest),
        "local_manifest_count": sync_report["local_manifest_count"],
        "mirrored_file_count": sync_report["mirrored_file_count"],
        "unsupported_reason": reload_record["unsupported_reason"],
        "supports_paper_claim": False,
    }

    write_json(paths.local_output_dir / "colab_env_report.json", runtime_report)
    write_json(paths.local_output_dir / "drive_mount_report.json", mount_report)
    write_jsonl(paths.local_output_dir / "cold_start_smoke_record.jsonl", [cold_start_record])
    notebook_runtime_report_path = write_notebook_runtime_report(
        root=root,
        workflow_name=workflow_name,
        output_dir=paths.local_output_dir,
        drive_output_dir=paths.drive_workflow_dir.as_posix(),
    )
    write_json(paths.drive_workflow_dir / "colab_env_report.json", runtime_report)
    write_json(paths.drive_workflow_dir / "drive_mount_report.json", mount_report)
    write_jsonl(paths.drive_workflow_dir / "cold_start_smoke_record.jsonl", [cold_start_record])
    write_json(
        paths.drive_workflow_dir / "notebook_runtime_report.json",
        json.loads(notebook_runtime_report_path.read_text(encoding="utf-8")),
    )

    return {
        "workflow_decision": cold_start_record["workflow_decision"],
        "local_output_dir": paths.local_output_dir.as_posix(),
        "drive_workflow_dir": paths.drive_workflow_dir.as_posix(),
        "local_manifest_count": sync_report["local_manifest_count"],
        "mirrored_file_count": sync_report["mirrored_file_count"],
        "reload_decision": reload_record["reload_decision"],
        "unsupported_reason": cold_start_record["unsupported_reason"],
        "supports_paper_claim": False,
    }


def workflow_summary_text(summary: dict[str, Any]) -> str:
    """返回便于 Notebook 或 CLI 打印的稳定 JSON 摘要。"""
    return stable_json_text(summary)
