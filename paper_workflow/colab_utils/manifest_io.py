"""读写 Drive workflow manifest 与本地 outputs 镜像报告。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from experiments.artifacts.manifest_schema import validate_manifest
from paper_workflow.colab_utils.drive_paths import DriveWorkflowPaths
from experiments.runtime.repository_environment import resolve_code_version

WORKFLOW_UNIT_NAME = "colab_drive_workflow"
LOCAL_ARTIFACT_DIRECTORIES = (
    "core_package_boundary_freeze",
    "algorithm_primitives",
    "core_method_synthetic_smoke",
    "sd_runtime_adapter",
    "real_sd_runtime_probe",
    "minimal_diffusion_latent_injection",
)
LOCAL_ARCHIVE_PATTERNS = (
    "real_sd_runtime_probe_package_*.zip",
    "minimal_latent_injection_package_*.zip",
)
DRIVE_SOURCE_DIRECTORIES = (
    "real_sd_runtime_probe",
    "minimal_diffusion_latent_injection",
)
DRIVE_SOURCE_PATTERNS = (
    "*.json",
    "*.jsonl",
    "*.csv",
    "*.zip",
    "*.png",
)


@dataclass(frozen=True)
class LocalManifestReference:
    """记录一个本地 manifest 的摘要与核心字段。"""

    manifest_path: str
    manifest_digest: str
    artifact_id: str
    artifact_type: str
    output_count: int
    decision: str
    unsupported_reason: str

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的 manifest 引用。"""
        return asdict(self)


@dataclass(frozen=True)
class FileMirrorRecord:
    """记录一个 outputs 文件到 Drive 目录的镜像结果。"""

    source_path: str
    destination_path: str
    file_digest: str
    byte_count: int
    copy_decision: str
    unsupported_reason: str

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的镜像记录。"""
        return asdict(self)


@dataclass(frozen=True)
class ReloadCheckRecord:
    """记录从 Drive manifest 重载并校验文件摘要的结果。"""

    reload_decision: str
    manifest_path: str
    manifest_digest: str
    verified_file_count: int
    missing_input_count: int
    missing_input_paths: tuple[str, ...]
    digest_mismatch_count: int
    unsupported_reason: str

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的重载记录。"""
        return asdict(self)


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定且便于审计的文本。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: Any) -> str:
    """把 JSON 兼容对象转为单行 JSONL 文本。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def file_digest(path: Path) -> str:
    """计算文件 SHA-256 摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_root(root: Path, path: Path) -> str:
    """把路径转为相对仓库根目录的 POSIX 表示。"""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_json(path: Path, payload: Any) -> None:
    """写入稳定 JSON 文本。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json_text(payload), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """写入 JSONL 记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json_line(record) for record in records), encoding="utf-8")


def discover_local_files(root_path: Path) -> tuple[Path, ...]:
    """枚举需要镜像到 Drive 的本地 outputs 文件。"""
    outputs_root = root_path / "outputs"
    discovered: set[Path] = set()
    for directory_name in LOCAL_ARTIFACT_DIRECTORIES:
        directory = outputs_root / directory_name
        if directory.exists():
            discovered.update(path for path in directory.rglob("*") if path.is_file())
    for pattern in LOCAL_ARCHIVE_PATTERNS:
        discovered.update(path for path in outputs_root.glob(pattern) if path.is_file())
    return tuple(sorted(discovered))


def discover_drive_source_files(paths: DriveWorkflowPaths) -> tuple[Path, ...]:
    """枚举 Drive 中已有的前序真实运行产物。"""
    discovered: set[Path] = set()
    for directory_name in DRIVE_SOURCE_DIRECTORIES:
        directory = paths.drive_root / directory_name
        if directory.exists():
            for pattern in DRIVE_SOURCE_PATTERNS:
                discovered.update(path for path in directory.rglob(pattern) if path.is_file())
    return tuple(sorted(discovered))


def discover_local_manifests(root_path: Path) -> tuple[LocalManifestReference, ...]:
    """读取已存在的本地 manifest 摘要。"""
    references: list[LocalManifestReference] = []
    for manifest_path in sorted((root_path / "outputs").rglob("manifest.local.json")):
        relative_parts = manifest_path.relative_to(root_path).parts
        if WORKFLOW_UNIT_NAME in relative_parts:
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        missing_fields = validate_manifest(payload)
        decision = "pass" if not missing_fields else "fail"
        references.append(
            LocalManifestReference(
                manifest_path=relative_to_root(root_path, manifest_path),
                manifest_digest=file_digest(manifest_path),
                artifact_id=str(payload.get("artifact_id", "unknown")),
                artifact_type=str(payload.get("artifact_type", "unknown")),
                output_count=len(payload.get("output_paths", [])),
                decision=decision,
                unsupported_reason="" if decision == "pass" else "manifest_required_fields_missing",
            )
        )
    return tuple(references)


def mirror_files_to_drive(
    root_path: Path,
    paths: DriveWorkflowPaths,
    source_paths: tuple[Path, ...] | None = None,
) -> tuple[FileMirrorRecord, ...]:
    """把本地 outputs 文件复制到 Drive workflow 目录。"""
    files = source_paths if source_paths is not None else discover_local_files(root_path)
    records: list[FileMirrorRecord] = []
    for source_path in files:
        relative_source = source_path.resolve().relative_to(root_path.resolve())
        destination_path = paths.drive_local_output_dir / relative_source
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        records.append(
            FileMirrorRecord(
                source_path=relative_source.as_posix(),
                destination_path=destination_path.as_posix(),
                file_digest=file_digest(source_path),
                byte_count=source_path.stat().st_size,
                copy_decision="copied_to_drive_workflow",
                unsupported_reason="",
            )
        )
    return tuple(records)


def register_drive_source_files(paths: DriveWorkflowPaths) -> tuple[FileMirrorRecord, ...]:
    """登记 Drive 中已存在的前序产物, 避免依赖 Colab clone 后的空 outputs。"""
    records: list[FileMirrorRecord] = []
    for source_path in discover_drive_source_files(paths):
        records.append(
            FileMirrorRecord(
                source_path=source_path.as_posix(),
                destination_path=source_path.as_posix(),
                file_digest=file_digest(source_path),
                byte_count=source_path.stat().st_size,
                copy_decision="registered_existing_drive_file",
                unsupported_reason="",
            )
        )
    return tuple(records)


def build_sync_report(
    root_path: Path,
    paths: DriveWorkflowPaths,
    mirror_records: tuple[FileMirrorRecord, ...],
    manifest_references: tuple[LocalManifestReference, ...],
) -> dict[str, Any]:
    """构造本地 outputs 与 Drive 已有产物的登记报告。"""
    return {
        "construction_unit_name": WORKFLOW_UNIT_NAME,
        "workflow_name": paths.workflow_name,
        "workflow_decision": "pass" if mirror_records else "unsupported",
        "repository_root": root_path.as_posix(),
        "drive_root": paths.drive_root.as_posix(),
        "drive_workflow_dir": paths.drive_workflow_dir.as_posix(),
        "local_manifest_count": len(manifest_references),
        "mirrored_file_count": len(mirror_records),
        "input_file_count": len(mirror_records),
        "unsupported_reason": "" if mirror_records else "no_local_or_drive_artifact_found",
        "supports_paper_claim": False,
        "local_manifests": [reference.to_dict() for reference in manifest_references],
        "mirrored_files": [record.to_dict() for record in mirror_records],
    }


def build_input_manifest_payload(
    manifest_references: tuple[LocalManifestReference, ...],
    mirror_records: tuple[FileMirrorRecord, ...],
) -> dict[str, Any]:
    """构造输入 manifest 摘要。"""
    has_inputs = bool(manifest_references or mirror_records)
    return {
        "construction_unit_name": WORKFLOW_UNIT_NAME,
        "workflow_decision": "pass" if has_inputs else "unsupported",
        "local_manifest_count": len(manifest_references),
        "mirrored_file_count": len(mirror_records),
        "input_file_count": len(mirror_records),
        "unsupported_reason": "" if has_inputs else "no_registered_input_found",
        "supports_paper_claim": False,
        "local_manifests": [reference.to_dict() for reference in manifest_references],
        "mirrored_files": [record.to_dict() for record in mirror_records],
    }


def build_output_manifest_payload(mirror_records: tuple[FileMirrorRecord, ...]) -> dict[str, Any]:
    """构造 Drive 镜像输出摘要。"""
    return {
        "construction_unit_name": WORKFLOW_UNIT_NAME,
        "workflow_decision": "pass" if mirror_records else "unsupported",
        "mirrored_file_count": len(mirror_records),
        "unsupported_reason": "" if mirror_records else "no_mirrored_file_found",
        "supports_paper_claim": False,
        "mirrored_files": [record.to_dict() for record in mirror_records],
    }


def build_drive_manifest_payload(
    root_path: Path,
    paths: DriveWorkflowPaths,
    sync_report: dict[str, Any],
    dependency_report: dict[str, Any],
    mount_report: dict[str, Any],
) -> dict[str, Any]:
    """构造 Drive 入口 manifest。"""
    config = {
        "workflow_name": paths.workflow_name,
        "sync_report_digest": build_stable_digest(sync_report),
        "dependency_report_digest": build_stable_digest(dependency_report),
        "mount_report_digest": build_stable_digest(mount_report),
    }
    output_paths = (
        (paths.local_output_dir / "colab_env_report.json"),
        (paths.local_output_dir / "drive_mount_report.json"),
        (paths.local_output_dir / "cold_start_smoke_record.jsonl"),
        (paths.local_output_dir / "reload_smoke_record.jsonl"),
        (paths.local_output_dir / "local_output_sync_report.json"),
        (paths.local_output_dir / "manifest.local.json"),
    )
    manifest = build_artifact_manifest(
        artifact_id="colab_drive_workflow_manifest",
        artifact_type="drive_manifest",
        input_paths=tuple(reference["manifest_path"] for reference in sync_report.get("local_manifests", []))
        + tuple(record["destination_path"] for record in sync_report.get("mirrored_files", [])),
        output_paths=tuple(relative_to_root(root_path, path) for path in output_paths),
        config=config,
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/colab_drive_entry.py",
        metadata={
            "construction_unit_name": WORKFLOW_UNIT_NAME,
            "workflow_name": paths.workflow_name,
            "workflow_decision": sync_report.get("workflow_decision", "unsupported"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "supports_paper_claim": False,
            "drive_root": paths.drive_root.as_posix(),
            "drive_workflow_dir": paths.drive_workflow_dir.as_posix(),
        },
    ).to_dict()
    manifest["mirrored_files"] = sync_report.get("mirrored_files", [])
    return manifest


def write_manifest_bundle(
    root_path: Path,
    paths: DriveWorkflowPaths,
    sync_report: dict[str, Any],
    dependency_report: dict[str, Any],
    mount_report: dict[str, Any],
) -> dict[str, Any]:
    """同时写入本地与 Drive manifest 文件。"""
    manifest_references = tuple(
        LocalManifestReference(**item) for item in sync_report.get("local_manifests", [])
    )
    mirror_records = tuple(FileMirrorRecord(**item) for item in sync_report.get("mirrored_files", []))
    input_manifest = build_input_manifest_payload(manifest_references, mirror_records)
    output_manifest = build_output_manifest_payload(mirror_records)
    drive_manifest = build_drive_manifest_payload(root_path, paths, sync_report, dependency_report, mount_report)
    artifact_manifest = {
        "construction_unit_name": WORKFLOW_UNIT_NAME,
        "artifact_type": "drive_artifact_manifest",
        "workflow_name": paths.workflow_name,
        "workflow_decision": drive_manifest["metadata"].get("workflow_decision", "unsupported"),
        "manifest_digest": build_stable_digest(drive_manifest),
        "input_manifest_digest": build_stable_digest(input_manifest),
        "output_manifest_digest": build_stable_digest(output_manifest),
        "supports_paper_claim": False,
    }

    local_files = {
        "manifest.local.json": drive_manifest,
        "input_manifest.json": input_manifest,
        "output_manifest.json": output_manifest,
        "artifact_manifest.json": artifact_manifest,
    }
    drive_files = {
        "manifest.json": drive_manifest,
        "input_manifest.json": input_manifest,
        "output_manifest.json": output_manifest,
        "artifact_manifest.json": artifact_manifest,
    }
    for name, payload in local_files.items():
        write_json(paths.local_output_dir / name, payload)
    for name, payload in drive_files.items():
        write_json(paths.drive_workflow_dir / name, payload)
    return drive_manifest


def verify_drive_manifest(manifest_path: str | Path) -> ReloadCheckRecord:
    """根据 Drive manifest 重新校验镜像文件是否存在且摘要一致。"""
    resolved_manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    missing_paths: list[str] = []
    digest_mismatch_count = 0
    verified_file_count = 0
    mirrored_files = manifest.get("mirrored_files", [])
    for item in mirrored_files:
        destination_path = Path(item["destination_path"])
        if not destination_path.exists():
            missing_paths.append(destination_path.as_posix())
            continue
        if file_digest(destination_path) != item["file_digest"]:
            digest_mismatch_count += 1
            continue
        verified_file_count += 1
    if not mirrored_files:
        decision = "unsupported"
        unsupported_reason = "no_manifest_file_registered"
    else:
        decision = "pass" if not missing_paths and digest_mismatch_count == 0 else "fail"
        unsupported_reason = "" if decision == "pass" else "drive_manifest_reload_failed"
    return ReloadCheckRecord(
        reload_decision=decision,
        manifest_path=resolved_manifest_path.as_posix(),
        manifest_digest=file_digest(resolved_manifest_path),
        verified_file_count=verified_file_count,
        missing_input_count=len(missing_paths),
        missing_input_paths=tuple(missing_paths),
        digest_mismatch_count=digest_mismatch_count,
        unsupported_reason=unsupported_reason,
    )
