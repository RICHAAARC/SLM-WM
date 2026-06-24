"""Colab threshold calibration packaging helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.protocol.pilot_paper_fixed_fpr import (
    PILOT_PAPER_FIXED_FPR,
    PILOT_PAPER_MINIMUM_CLEAN_NEGATIVE_COUNT,
)
from paper_workflow.colab_utils.sd_runtime_cold_start import (
    build_runtime_environment_report,
    file_digest,
    resolve_code_version,
)
from scripts.write_geometric_rescue_outputs import write_geometric_rescue_outputs
from scripts.write_threshold_calibration_outputs import write_threshold_calibration_outputs

DEFAULT_OUTPUT_DIR = "outputs/threshold_calibration"
DEFAULT_GEOMETRIC_RESCUE_DIR = "outputs/geometric_rescue"
DEFAULT_CONTENT_CARRIER_DIR = "outputs/content_carriers"
DEFAULT_DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/SLM/pilot_paper_results/threshold_calibration"
DEFAULT_ATTENTION_INJECTION_DRIVE_DIR = "/content/drive/MyDrive/SLM/pilot_paper_results/attention_latent_injection"
DEFAULT_ALIGNED_RESCORING_DRIVE_DIR = "/content/drive/MyDrive/SLM/pilot_paper_results/aligned_rescoring"
DEFAULT_TARGET_FPR = PILOT_PAPER_FIXED_FPR
DEFAULT_MAX_CONTENT_RECORDS: int | None = None
DEFAULT_MINIMUM_CLEAN_NEGATIVE_COUNT = PILOT_PAPER_MINIMUM_CLEAN_NEGATIVE_COUNT
ATTENTION_INJECTION_PACKAGE_PATTERN = "attention_latent_injection_package_*.zip"
ALIGNED_RESCORING_PACKAGE_PATTERN = "aligned_rescoring_package_*.zip"
CONTENT_CARRIER_PREFIXES = ("outputs/content_carriers/",)
PACKAGE_EXTRA_PATHS = (
    "paper_workflow/threshold_calibration_run.ipynb",
    "paper_workflow/colab_utils/threshold_calibration.py",
    "scripts/write_geometric_rescue_outputs.py",
    "scripts/write_threshold_calibration_outputs.py",
)


@dataclass(frozen=True)
class ThresholdCalibrationArchiveRecord:
    """记录 threshold calibration 结果包及其 Drive 镜像信息。"""

    archive_path: str
    archive_digest: str
    archive_entry_count: int
    drive_archive_path: str
    drive_archive_digest: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 JSON 的普通字典。"""
        return asdict(self)


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_line(value: Any) -> str:
    """把 JSON 兼容对象转换为 JSONL 单行文本。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先返回相对仓库根目录的路径, 外部路径保留绝对路径。"""
    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def latest_drive_package(drive_dir: str | Path, pattern: str) -> Path:
    """从 Google Drive 目录中选择名称排序最新的结果包。"""
    candidates = sorted(Path(drive_dir).expanduser().glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"drive_package_missing:{drive_dir}:{pattern}")
    return candidates[-1]


def parse_optional_record_limit(value: int | str | None) -> int | None:
    """解析可选记录上限, all 或空值表示使用全部记录。

    该函数属于配置解析层: Notebook 只传入环境变量文本, 由 helper 统一完成
    语义归一化, 避免入口 cell 中散落重复解析逻辑。
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    normalized = value.strip().lower()
    if normalized in {"", "all", "none", "unlimited"}:
        return None
    return max(0, int(normalized))


def parse_non_negative_count(value: int | str | None, default: int) -> int:
    """解析非负计数配置。"""
    if value is None:
        return default
    if isinstance(value, int):
        return max(0, value)
    normalized = value.strip()
    if not normalized:
        return default
    return max(0, int(normalized))


def copy_package_to_input_dir(package_path: Path, input_dir: Path) -> Path:
    """把 Drive 中的前序结果包复制到本次输出目录, 便于一起打包核对。"""
    input_dir.mkdir(parents=True, exist_ok=True)
    target_path = input_dir / package_path.name
    if package_path.resolve() != target_path.resolve():
        shutil.copy2(package_path, target_path)
    return target_path


def safe_extract_selected_entries(
    package_path: Path,
    root_path: Path,
    allowed_prefixes: tuple[str, ...],
) -> tuple[str, ...]:
    """只解压允许进入工作区的 outputs 输入文件。"""
    extracted: list[str] = []
    with ZipFile(package_path) as archive:
        for member in archive.infolist():
            normalized_name = member.filename.replace("\\", "/")
            if member.is_dir() or not any(normalized_name.startswith(prefix) for prefix in allowed_prefixes):
                continue
            target_path = (root_path / normalized_name).resolve()
            if not target_path.is_relative_to(root_path):
                raise RuntimeError(f"unsafe_zip_entry:{normalized_name}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source_handle, target_path.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)
            extracted.append(normalized_name)
    return tuple(extracted)


def materialize_threshold_calibration_inputs(
    root: str | Path = ".",
    attention_injection_drive_dir: str = DEFAULT_ATTENTION_INJECTION_DRIVE_DIR,
    aligned_rescoring_drive_dir: str = DEFAULT_ALIGNED_RESCORING_DRIVE_DIR,
) -> dict[str, Any]:
    """从 Google Drive 准备 threshold calibration 所需的前序结果包与内容检测记录。"""
    root_path = Path(root).resolve()
    output_dir = root_path / DEFAULT_OUTPUT_DIR
    input_dir = output_dir / "input_packages"
    output_dir.mkdir(parents=True, exist_ok=True)
    attention_package = latest_drive_package(attention_injection_drive_dir, ATTENTION_INJECTION_PACKAGE_PATTERN)
    aligned_package = latest_drive_package(aligned_rescoring_drive_dir, ALIGNED_RESCORING_PACKAGE_PATTERN)
    local_attention_package = copy_package_to_input_dir(attention_package, input_dir)
    local_aligned_package = copy_package_to_input_dir(aligned_package, input_dir)
    extracted_content_entries = safe_extract_selected_entries(local_aligned_package, root_path, CONTENT_CARRIER_PREFIXES)
    content_records_path = root_path / DEFAULT_CONTENT_CARRIER_DIR / "content_detection_records.jsonl"
    if not content_records_path.exists():
        raise FileNotFoundError("content_detection_records_missing_from_aligned_rescoring_package")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "attention_injection_drive_package_path": str(attention_package),
        "attention_injection_drive_package_digest": file_digest(attention_package),
        "attention_injection_input_package_path": relative_or_absolute(local_attention_package, root_path),
        "attention_injection_input_package_digest": file_digest(local_attention_package),
        "aligned_rescoring_drive_package_path": str(aligned_package),
        "aligned_rescoring_drive_package_digest": file_digest(aligned_package),
        "aligned_rescoring_input_package_path": relative_or_absolute(local_aligned_package, root_path),
        "aligned_rescoring_input_package_digest": file_digest(local_aligned_package),
        "content_records_path": relative_or_absolute(content_records_path, root_path),
        "content_extracted_entry_count": len(extracted_content_entries),
        "content_extracted_entries": extracted_content_entries,
    }
    manifest_path = output_dir / "threshold_calibration_input_package_manifest.json"
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return manifest


def write_failure_outputs(root_path: Path, error: Exception) -> dict[str, Any]:
    """在 Drive 输入缺失或前序记录不完整时写出可打包的失败诊断。"""
    output_dir = root_path / DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    environment_report = build_runtime_environment_report()
    result = {
        "run_decision": "fail",
        "threshold_calibration_ready": False,
        "geometric_rescue_ready": False,
        "supports_paper_claim": False,
        "unsupported_reason": f"{type(error).__name__}:{error}",
        "environment_report_path": f"{DEFAULT_OUTPUT_DIR}/threshold_calibration_environment_report.json",
        "manifest_path": f"{DEFAULT_OUTPUT_DIR}/threshold_calibration_manifest.local.json",
        "metadata": {
            "claim_boundary": "not_paper_ready",
            "runtime_environment": environment_report,
        },
    }
    environment_path = output_dir / "threshold_calibration_environment_report.json"
    result_path = output_dir / "threshold_calibration_result.json"
    manifest_path = output_dir / "threshold_calibration_manifest.local.json"
    environment_path.write_text(stable_json_text(environment_report), encoding="utf-8")
    result_path.write_text(stable_json_text(result), encoding="utf-8")
    manifest = {
        "artifact_id": "threshold_calibration_failure_manifest",
        "artifact_type": "local_manifest",
        "code_version": resolve_code_version(root_path),
        "input_paths": [],
        "output_paths": [
            relative_or_absolute(result_path, root_path),
            relative_or_absolute(environment_path, root_path),
        ],
        "metadata": {
            "run_decision": "fail",
            "supports_paper_claim": False,
        },
        "rebuild_command": "运行 paper_workflow/threshold_calibration_run.ipynb",
    }
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return result


def run_default_threshold_calibration_from_drive_plan(
    root: str | Path = ".",
    attention_injection_drive_dir: str = DEFAULT_ATTENTION_INJECTION_DRIVE_DIR,
    aligned_rescoring_drive_dir: str = DEFAULT_ALIGNED_RESCORING_DRIVE_DIR,
    target_fpr: float = DEFAULT_TARGET_FPR,
    max_content_records: int | str | None = DEFAULT_MAX_CONTENT_RECORDS,
    minimum_clean_negative_count: int | str | None = DEFAULT_MINIMUM_CLEAN_NEGATIVE_COUNT,
) -> dict[str, Any]:
    """从 Google Drive 前序结果包重建几何恢复记录并写出 threshold calibration 产物。"""
    root_path = Path(root).resolve()
    resolved_max_content_records = parse_optional_record_limit(max_content_records)
    resolved_minimum_clean_negative_count = parse_non_negative_count(
        minimum_clean_negative_count,
        DEFAULT_MINIMUM_CLEAN_NEGATIVE_COUNT,
    )
    try:
        input_manifest = materialize_threshold_calibration_inputs(
            root=root_path,
            attention_injection_drive_dir=attention_injection_drive_dir,
            aligned_rescoring_drive_dir=aligned_rescoring_drive_dir,
        )
        geometric_manifest = write_geometric_rescue_outputs(
            root=root_path,
            content_records_path=input_manifest["content_records_path"],
            attention_injection_package_path=input_manifest["attention_injection_input_package_path"],
            max_content_records=resolved_max_content_records,
        )
        threshold_manifest = write_threshold_calibration_outputs(
            root=root_path,
            target_fpr=target_fpr,
            aligned_rescoring_package_path=input_manifest["aligned_rescoring_input_package_path"],
            minimum_clean_negative_count=resolved_minimum_clean_negative_count,
        )
    except Exception as error:
        return write_failure_outputs(root_path, error)

    output_dir = root_path / DEFAULT_OUTPUT_DIR
    rescue_audit_path = root_path / DEFAULT_GEOMETRIC_RESCUE_DIR / "geometry_rescue_audit.json"
    rescue_records_path = root_path / DEFAULT_GEOMETRIC_RESCUE_DIR / "aligned_detection_records.jsonl"
    threshold_report_path = output_dir / "threshold_degeneracy_report.json"
    threshold_report = json.loads(threshold_report_path.read_text(encoding="utf-8"))
    rescue_audit = json.loads(rescue_audit_path.read_text(encoding="utf-8"))
    rescue_record_count = sum(1 for line in rescue_records_path.read_text(encoding="utf-8").splitlines() if line.strip())
    geometric_ready = bool(rescue_audit.get("protocol_decision") == "pass" and rescue_record_count > 0)
    threshold_ready = bool(threshold_manifest.get("metadata", {}).get("protocol_decision") == "pass")
    result = {
        "run_decision": "pass" if geometric_ready and threshold_ready else "fail",
        "threshold_calibration_ready": threshold_ready,
        "geometric_rescue_ready": geometric_ready,
        "geometric_rescue_record_count": rescue_record_count,
        "target_fpr": target_fpr,
        "threshold_manifest_path": f"{DEFAULT_OUTPUT_DIR}/manifest.local.json",
        "geometric_rescue_manifest_path": f"{DEFAULT_GEOMETRIC_RESCUE_DIR}/manifest.local.json",
        "threshold_report_path": f"{DEFAULT_OUTPUT_DIR}/threshold_degeneracy_report.json",
        "rescue_audit_path": f"{DEFAULT_GEOMETRIC_RESCUE_DIR}/geometry_rescue_audit.json",
        "input_manifest_path": f"{DEFAULT_OUTPUT_DIR}/threshold_calibration_input_package_manifest.json",
        "supports_paper_claim": bool(threshold_report.get("supports_paper_claim", False)),
        "metadata": {
            "claim_boundary": "paper_ready_only_after_full_external_protocol",
            "threshold_protocol_decision": threshold_manifest.get("metadata", {}).get("protocol_decision", ""),
            "geometric_protocol_decision": rescue_audit.get("protocol_decision", ""),
            "max_content_records": resolved_max_content_records,
            "minimum_clean_negative_count": threshold_report.get(
                "minimum_clean_negative_count",
                resolved_minimum_clean_negative_count,
            ),
            "minimum_clean_negative_count_ready": threshold_report.get("minimum_clean_negative_count_ready", False),
            "calibration_negative_count": threshold_report.get("calibration_negative_count", 0),
            "clean_negative_count": threshold_report.get("clean_negative_count", 0),
            "fixed_fpr_control_scope": threshold_report.get("fixed_fpr_control_scope", "calibration_clean_negative"),
            "fixed_fpr_denominator_role": threshold_report.get("fixed_fpr_denominator_role", "clean_negative_only"),
            "rescue_control_scope": threshold_report.get("rescue_control_scope", "evidence_clean_negative"),
            "rescue_changes_fpr_denominator": threshold_report.get("rescue_changes_fpr_denominator", False),
            "attacked_negative_boundary_role": threshold_report.get(
                "attacked_negative_boundary_role",
                "attack_robustness_diagnostic_not_fpr_denominator",
            ),
            "attacked_negative_governs_fixed_fpr": threshold_report.get("attacked_negative_governs_fixed_fpr", False),
            "evidence_fpr_exceeds_target": threshold_report.get("evidence_fpr_exceeds_target", False),
            "attacked_fpr_diagnostic_exceeds_target": threshold_report.get("attacked_fpr_diagnostic_exceeds_target", False),
            "aligned_rescoring_quality_metrics_ready": threshold_report.get("aligned_rescoring_quality_metrics_ready", False),
            "perceptual_metrics_ready": threshold_report.get("perceptual_metrics_ready", False),
        },
    }
    result_path = output_dir / "threshold_calibration_result.json"
    result_path.write_text(stable_json_text(result), encoding="utf-8")
    return result


def collect_package_entries(root_path: Path, output_dir: Path, archive_path: Path) -> tuple[Path, ...]:
    """收集 threshold calibration 结果包所需的核对文件。"""
    entries: list[Path] = []
    for relative_dir in (DEFAULT_OUTPUT_DIR, DEFAULT_GEOMETRIC_RESCUE_DIR, DEFAULT_CONTENT_CARRIER_DIR):
        directory = root_path / relative_dir
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.resolve() != archive_path.resolve() and path.suffix.lower() != ".zip":
                entries.append(path)
    for relative_path in PACKAGE_EXTRA_PATHS:
        path = root_path / relative_path
        if path.exists():
            entries.append(path)
    unique_entries: list[Path] = []
    for entry in entries:
        if entry not in unique_entries:
            unique_entries.append(entry)
    return tuple(unique_entries)


def write_archive(path: Path, entries: tuple[Path, ...], root_path: Path) -> None:
    """根据给定 entry 列表写出 zip 文件。"""
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(entry, entry.relative_to(root_path).as_posix())


def package_threshold_calibration_outputs(
    root: str | Path = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    drive_output_dir: str = DEFAULT_DRIVE_OUTPUT_DIR,
    archive_name: str = "threshold_calibration_package.zip",
) -> ThresholdCalibrationArchiveRecord:
    """打包 threshold calibration 产物并镜像到 Google Drive。"""
    root_path = Path(root).resolve()
    source_dir = (root_path / output_dir).resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / archive_name
    package_manifest_path = source_dir / "threshold_calibration_package_input_manifest.json"
    summary_path = source_dir / "threshold_calibration_archive_summary.json"
    manifest_path = source_dir / "threshold_calibration_archive_manifest.local.json"
    for stale_path in (package_manifest_path, summary_path, manifest_path):
        if stale_path.exists():
            stale_path.unlink()

    entries = collect_package_entries(root_path, source_dir, archive_path)
    package_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry_paths": [entry.relative_to(root_path).as_posix() for entry in entries],
        "entry_count": len(entries),
    }
    package_manifest_path.write_text(stable_json_text(package_manifest), encoding="utf-8")
    preliminary_record = ThresholdCalibrationArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest="",
        archive_entry_count=len(entries) + 3,
        drive_archive_path=str(Path(drive_output_dir).expanduser() / archive_name),
        drive_archive_digest="",
        metadata={
            "construction_unit_name": "threshold_calibration",
            "drive_output_dir": str(Path(drive_output_dir).expanduser()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "embedded_digest_scope": "external_summary_records_final_archive_digest",
        },
    )
    summary_path.write_text(stable_json_text(preliminary_record.to_dict()), encoding="utf-8")
    archive_manifest = {
        "artifact_id": "threshold_calibration_archive_manifest",
        "artifact_type": "local_manifest",
        "code_version": resolve_code_version(root_path),
        "input_paths": [entry.relative_to(root_path).as_posix() for entry in entries] + [
            package_manifest_path.relative_to(root_path).as_posix(),
        ],
        "output_paths": [
            archive_path.relative_to(root_path).as_posix(),
            summary_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ],
        "metadata": {
            "construction_unit_name": "threshold_calibration",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "embedded_digest_scope": "external_summary_records_final_archive_digest",
        },
        "rebuild_command": "运行 paper_workflow/threshold_calibration_run.ipynb",
    }
    manifest_path.write_text(stable_json_text(archive_manifest), encoding="utf-8")

    entries = collect_package_entries(root_path, source_dir, archive_path)
    write_archive(archive_path, entries, root_path)
    drive_dir = Path(drive_output_dir).expanduser()
    drive_dir.mkdir(parents=True, exist_ok=True)
    mirrored_path = drive_dir / archive_name
    shutil.copy2(archive_path, mirrored_path)
    record = ThresholdCalibrationArchiveRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest=file_digest(archive_path),
        archive_entry_count=len(entries),
        drive_archive_path=str(mirrored_path),
        drive_archive_digest=file_digest(mirrored_path),
        metadata={
            "construction_unit_name": "threshold_calibration",
            "drive_output_dir": str(drive_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    summary_path.write_text(stable_json_text(record.to_dict()), encoding="utf-8")
    archive_manifest["metadata"].update(
        {
            "archive_digest": record.archive_digest,
            "drive_archive_digest": record.drive_archive_digest,
        }
    )
    manifest_path.write_text(stable_json_text(archive_manifest), encoding="utf-8")
    return record
