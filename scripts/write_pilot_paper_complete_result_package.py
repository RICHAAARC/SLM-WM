"""打包 pilot_paper 完整结果包。

该脚本只负责把已经由 Notebook 或本地受治理 builder 产出的 records、tables、
reports、manifests 和官方复现核对文件收敛到一个可审计压缩包中。它不直接生成
GPU 结果, 也不手写正式论文表格。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol.paper_run_config import build_paper_run_config
from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.digest import build_stable_digest
from scripts.write_pilot_paper_result_records import WorkProgress, expand_package_paths, materialize_output_entries

CONSTRUCTION_UNIT_NAME = "pilot_paper_complete_result_package"
DEFAULT_OUTPUT_DIR = Path("outputs/pilot_paper_complete_result_package")
DEFAULT_COMMON_PROTOCOL_SUMMARY_PATH = Path(
    "outputs/pilot_paper_fixed_fpr_common_protocol/pilot_paper_common_protocol_summary.json"
)
DEFAULT_DRIVE_OUTPUT_DIR = ""
DEFAULT_PACKAGE_SEARCH_ROOT = ""
ZIP_COMPRESSION_METHODS = {
    "stored": ZIP_STORED,
    "deflated": ZIP_DEFLATED,
}
REQUIRED_OUTPUT_DIRS = (
    "outputs/real_attention_geometry",
    "outputs/attention_geometry",
    "outputs/attention_latent_injection",
    "outputs/attention_latent_update",
    "outputs/aligned_rescoring",
    "outputs/threshold_calibration",
    "outputs/geometric_rescue",
    "outputs/attack_matrix",
    "outputs/real_attack_evaluation",
    "outputs/conventional_geometric_attack_evaluation",
    "outputs/image_attack_evidence",
    "outputs/dataset_level_quality",
    "outputs/external_baseline_method_faithful",
    "outputs/tree_ring_official_reference",
    "outputs/gaussian_shading_official_reference",
    "outputs/shallow_diffuse_official_reference",
    "outputs/t2smark_full_main_reproduction",
    "outputs/external_baseline_results",
    "outputs/primary_baseline_formal_import",
    "outputs/external_baseline_comparison",
    "outputs/internal_ablation_evidence",
    "outputs/pilot_paper_fixed_fpr_results",
    "outputs/pilot_paper_fixed_fpr_common_protocol",
    "outputs/pilot_paper_result_analysis",
)
PACKAGE_EXTRA_PATHS = (
    "configs/paper_main_probe_paper_prompts.txt",
    "configs/paper_main_pilot_paper_prompts.txt",
    "configs/paper_main_full_paper_prompts.txt",
    "paper_workflow/README.md",
    "scripts/write_pilot_paper_result_records.py",
    "scripts/write_attack_matrix_outputs.py",
    "scripts/write_pilot_paper_fixed_fpr_common_protocol_outputs.py",
    "scripts/write_primary_baseline_result_candidates.py",
    "scripts/write_primary_baseline_formal_import_protocol.py",
    "scripts/write_external_baseline_comparison_outputs.py",
    "scripts/write_internal_ablation_outputs.py",
    "scripts/write_pilot_paper_result_analysis_outputs.py",
)


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件, 文件不存在时返回空字典。"""

    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


@dataclass(frozen=True)
class PilotPaperCompletePackageRecord:
    """记录 pilot_paper 完整结果包与 Drive 镜像信息。"""

    archive_path: str
    archive_digest: str
    archive_entry_count: int
    drive_archive_path: str
    drive_archive_digest: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        return asdict(self)


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定文本。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, payload: Any) -> None:
    """写出 JSON 文件并创建父目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json_text(payload), encoding="utf-8")


def file_digest(path: Path) -> str:
    """计算文件 SHA-256 摘要。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_digest_with_progress(path: Path, label: str) -> str:
    """计算大文件摘要并输出读取进度。"""

    total_bytes = path.stat().st_size if path.is_file() else 0
    progress = WorkProgress(label, 1, total_bytes=total_bytes, emit_every_count=1)
    progress.emit(0, copied_bytes=0, profile=f"file={path.name}", force=True)
    digest = hashlib.sha256()
    copied_bytes = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            copied_bytes += len(chunk)
            progress.emit(0, copied_bytes=copied_bytes, profile=f"file={path.name}")
    progress.emit(1, copied_bytes=copied_bytes, profile=f"file={path.name} done", force=True)
    return digest.hexdigest()


def copy_file_with_progress(source_path: Path, target_path: Path, label: str) -> int:
    """以流式复制方式镜像大文件并输出复制进度。"""

    total_bytes = source_path.stat().st_size
    progress = WorkProgress(label, 1, total_bytes=total_bytes, emit_every_count=1)
    progress.emit(0, copied_bytes=0, profile=f"file={source_path.name}", force=True)
    copied_bytes = 0
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as source_handle, target_path.open("wb") as target_handle:
        for chunk in iter(lambda: source_handle.read(8 * 1024 * 1024), b""):
            target_handle.write(chunk)
            copied_bytes += len(chunk)
            progress.emit(0, copied_bytes=copied_bytes, profile=f"file={source_path.name}")
    shutil.copystat(source_path, target_path)
    progress.emit(1, copied_bytes=copied_bytes, profile=f"file={source_path.name} done", force=True)
    return copied_bytes


def resolve_code_version(root_path: Path) -> str:
    """读取 Git 短提交标识, 工作区有变更时附加 dirty 标记。"""

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


def resolve_path(root_path: Path, path: str | Path | None) -> Path | None:
    """解析可选路径。"""

    if path is None or not str(path).strip():
        return None
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录相对仓库根目录路径。"""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def ensure_output_dir_under_outputs(root_path: Path, output_dir: str | Path) -> Path:
    """确保完整结果包本地输出目录位于 outputs/ 下。"""

    resolved = resolve_path(root_path, output_dir)
    if resolved is None:
        raise ValueError("pilot_paper 完整结果包输出目录不能为空")
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("pilot_paper 完整结果包输出目录必须位于 outputs/ 下") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def collect_required_entries(root_path: Path, output_dir: Path, archive_path: Path) -> tuple[Path, ...]:
    """收集完整结果包应包含的受治理核对文件。"""

    entries: list[Path] = []
    for relative_dir in REQUIRED_OUTPUT_DIRS:
        directory = root_path / relative_dir
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.resolve() != archive_path.resolve() and path.suffix.lower() != ".zip":
                entries.append(path)
    for relative_path in PACKAGE_EXTRA_PATHS:
        path = root_path / relative_path
        if path.is_file():
            entries.append(path)
    for path in sorted(output_dir.glob("*.json")):
        if path.is_file():
            entries.append(path)
    unique_entries: list[Path] = []
    seen_paths: set[Path] = set()
    for entry in entries:
        resolved = entry.resolve()
        if resolved not in seen_paths:
            unique_entries.append(entry)
            seen_paths.add(resolved)
    return tuple(unique_entries)


def write_archive_with_progress(root_path: Path, archive_path: Path, entries: Iterable[Path], compression_method: int) -> None:
    """把完整结果写入 zip, 并按文件数和字节数显示打包进度。"""

    entry_list = tuple(entries)
    total_bytes = sum(path.stat().st_size for path in entry_list if path.is_file())
    progress = WorkProgress(
        "pilot_paper complete package archive",
        len(entry_list),
        total_bytes=total_bytes,
        emit_every_count=100,
    )
    progress.emit(0, copied_bytes=0, profile=f"archive={archive_path.name}", force=True)
    copied_bytes = 0
    with ZipFile(archive_path, mode="w", compression=compression_method) as archive:
        for index, entry in enumerate(entry_list, start=1):
            archive.write(entry, relative_or_absolute(entry, root_path))
            copied_bytes += entry.stat().st_size if entry.is_file() else 0
            progress.emit(
                index,
                copied_bytes=copied_bytes,
                profile=f"archive={archive_path.name} file={relative_or_absolute(entry, root_path)}",
            )
    progress.emit(
        len(entry_list),
        copied_bytes=copied_bytes,
        profile=f"archive={archive_path.name} done",
        force=True,
    )


def build_readiness_summary(
    root_path: Path,
    entries: Iterable[Path],
    materialization_report: dict[str, Any],
    paper_claim_scale: str,
) -> dict[str, Any]:
    """汇总完整结果包覆盖状态。"""

    common_protocol_summary = read_json(root_path / DEFAULT_COMMON_PROTOCOL_SUMMARY_PATH)
    existing_dirs = [relative_dir for relative_dir in REQUIRED_OUTPUT_DIRS if (root_path / relative_dir).exists()]
    missing_dirs = [relative_dir for relative_dir in REQUIRED_OUTPUT_DIRS if not (root_path / relative_dir).exists()]
    entry_list = tuple(entries)
    package_ready = len(missing_dirs) == 0 and bool(entry_list)
    run_claim_ready = bool(common_protocol_summary.get("paper_run_claim_ready", False))
    probe_claim_ready = bool(common_protocol_summary.get("probe_claim_ready", False))
    pilot_claim_ready = bool(common_protocol_summary.get("pilot_claim_ready", False))
    full_claim_ready = bool(common_protocol_summary.get("full_claim_ready", False))
    return {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_claim_scale": paper_claim_scale,
        "required_output_dir_count": len(REQUIRED_OUTPUT_DIRS),
        "existing_required_output_dir_count": len(existing_dirs),
        "missing_required_output_dir_count": len(missing_dirs),
        "existing_required_output_dirs": existing_dirs,
        "missing_required_output_dirs": missing_dirs,
        "archive_entry_count": len(entry_list),
        "archive_entry_digest": build_stable_digest([relative_or_absolute(path, root_path) for path in entry_list]),
        "materialization_report": materialization_report,
        "paper_run_complete_result_package_ready": package_ready,
        "paper_run_claim_ready": run_claim_ready,
        "paper_run_claim_type": common_protocol_summary.get("paper_run_claim_type", ""),
        "probe_paper_complete_result_package_ready": paper_claim_scale == "probe_paper" and package_ready,
        "pilot_paper_complete_result_package_ready": paper_claim_scale == "pilot_paper" and package_ready,
        "full_paper_complete_result_package_ready": paper_claim_scale == "full_paper" and package_ready,
        "probe_claim_ready": probe_claim_ready,
        "pilot_claim_ready": pilot_claim_ready,
        "full_claim_ready": full_claim_ready,
        "supports_paper_claim": package_ready and run_claim_ready,
    }


def write_pilot_paper_complete_result_package_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    drive_output_dir: str | None = None,
    archive_name: str = "pilot_paper_complete_result_package.zip",
    package_paths: Iterable[str | Path] = (),
    package_search_roots: Iterable[str | Path] | None = None,
    materialize_packages: bool = True,
    zip_compression: str = "stored",
) -> dict[str, Any]:
    """写出 pilot_paper 完整结果包, 并按需从 Drive 结果包物化 outputs/ 条目。"""

    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(root_path)
    resolved_drive_output_dir = paper_run.drive_dir("complete_result_package") if drive_output_dir is None else drive_output_dir
    resolved_package_search_roots = (
        (paper_run.drive_result_root,) if package_search_roots is None else tuple(package_search_roots)
    )
    output_path = ensure_output_dir_under_outputs(root_path, output_dir)
    archive_path = output_path / archive_name
    packages = expand_package_paths(root_path, package_paths, resolved_package_search_roots)
    materialization_report = (
        materialize_output_entries(root_path, packages)
        if packages and materialize_packages
        else {
            "input_package_count": len(packages),
            "input_package_paths": [relative_or_absolute(path, root_path) for path in packages],
            "materialized_output_entry_count": 0,
            "materialized_output_total_bytes": 0,
            "skipped_output_entry_count": 0,
            "materialized_output_entries_digest": build_stable_digest([]),
            "skipped_output_entries": [],
            "materialization_skipped": bool(packages) and not materialize_packages,
        }
    )
    compression_method = ZIP_COMPRESSION_METHODS.get(str(zip_compression).strip().lower())
    if compression_method is None:
        raise ValueError(f"未知完整结果包 zip 压缩方式: {zip_compression}")

    package_manifest_path = output_path / "pilot_paper_complete_package_input_manifest.json"
    summary_path = output_path / "pilot_paper_complete_package_summary.json"
    manifest_path = output_path / "manifest.local.json"
    for stale_path in (package_manifest_path, summary_path, manifest_path):
        if stale_path.exists():
            stale_path.unlink()

    entries = collect_required_entries(root_path, output_path, archive_path)
    summary = build_readiness_summary(root_path, entries, materialization_report, paper_run.run_name)
    package_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_claim_scale": paper_run.run_name,
        "entry_paths": [relative_or_absolute(entry, root_path) for entry in entries],
        "entry_count": len(entries),
        "entry_paths_digest": build_stable_digest([relative_or_absolute(entry, root_path) for entry in entries]),
    }
    write_json(package_manifest_path, package_manifest)
    write_json(summary_path, summary)
    manifest = build_artifact_manifest(
        artifact_id="pilot_paper_complete_result_package_manifest",
        artifact_type="local_manifest",
        input_paths=tuple([relative_or_absolute(path, root_path) for path in packages] + package_manifest["entry_paths"]),
        output_paths=(
            relative_or_absolute(archive_path, root_path),
            relative_or_absolute(package_manifest_path, root_path),
            relative_or_absolute(summary_path, root_path),
            relative_or_absolute(manifest_path, root_path),
        ),
        config={
            "archive_name": archive_name,
            "drive_output_dir": resolved_drive_output_dir,
            "required_output_dirs": list(REQUIRED_OUTPUT_DIRS),
            "package_search_roots": [str(value) for value in resolved_package_search_roots],
            "materialize_packages": materialize_packages,
            "zip_compression": str(zip_compression),
        },
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_pilot_paper_complete_result_package.py",
        metadata=summary,
    ).to_dict()
    write_json(manifest_path, manifest)
    entries = collect_required_entries(root_path, output_path, archive_path)
    summary = build_readiness_summary(root_path, entries, materialization_report, paper_run.run_name)
    package_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_claim_scale": paper_run.run_name,
        "entry_paths": [relative_or_absolute(entry, root_path) for entry in entries],
        "entry_count": len(entries),
        "entry_paths_digest": build_stable_digest([relative_or_absolute(entry, root_path) for entry in entries]),
    }
    write_json(package_manifest_path, package_manifest)
    write_json(summary_path, summary)
    write_archive_with_progress(root_path, archive_path, entries, compression_method)

    archive_digest = file_digest_with_progress(archive_path, "pilot_paper complete package digest")

    drive_archive_path = ""
    drive_archive_digest = ""
    if resolved_drive_output_dir:
        drive_dir = Path(resolved_drive_output_dir).expanduser()
        drive_dir.mkdir(parents=True, exist_ok=True)
        mirrored_path = drive_dir / archive_name
        copy_file_with_progress(archive_path, mirrored_path, "pilot_paper complete package drive copy")
        drive_archive_path = str(mirrored_path)
        drive_archive_digest = archive_digest

    record = PilotPaperCompletePackageRecord(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest=archive_digest,
        archive_entry_count=len(entries),
        drive_archive_path=drive_archive_path,
        drive_archive_digest=drive_archive_digest,
        metadata=summary,
    )
    final_summary = record.to_dict()
    write_json(summary_path, final_summary)
    manifest.setdefault("metadata", {})["archive_digest"] = record.archive_digest
    manifest.setdefault("metadata", {})["drive_archive_digest"] = record.drive_archive_digest
    manifest.setdefault("metadata", {})["archive_entry_count"] = record.archive_entry_count
    write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="打包 pilot_paper 完整结果包。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="本地输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--drive-output-dir", default=None, help="Google Drive 镜像目录。")
    parser.add_argument("--archive-name", default="pilot_paper_complete_result_package.zip", help="压缩包文件名。")
    parser.add_argument("--package-path", action="append", default=[], help="可重复传入的前序结果 zip 包。")
    parser.add_argument("--package-search-root", action="append", default=[], help="递归查找 zip 包的目录。")
    parser.add_argument("--skip-package-materialization", action="store_true", help="跳过从前序 zip 物化 outputs/ 条目。")
    parser.add_argument(
        "--zip-compression",
        choices=sorted(ZIP_COMPRESSION_METHODS),
        default="stored",
        help="完整结果包 zip 压缩方式; stored 对 PNG 和已有压缩包更快。",
    )
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    manifest = write_pilot_paper_complete_result_package_outputs(
        root=args.root,
        output_dir=args.output_dir,
        drive_output_dir=args.drive_output_dir,
        archive_name=args.archive_name,
        package_paths=args.package_path,
        package_search_roots=args.package_search_root,
        materialize_packages=not args.skip_package_materialization,
        zip_compression=args.zip_compression,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
