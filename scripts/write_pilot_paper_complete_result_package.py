"""写出当前论文运行层级的完整结果包。

该脚本只打包已经由受治理运行时与 artifact builder 产生的结果。输入结果包必须
通过 ``--package-path`` 逐个显式给出, 脚本不会扫描目录并猜测应使用哪个运行包。
归档范围严格限制到当前 ``paper_run_name`` 对应的输出子目录, 因而不会把
``probe_paper``、``pilot_paper`` 与 ``full_paper`` 的证据混入同一个归档。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.runtime import repository_environment
from experiments.runtime.repository_environment import resolve_code_version
from main.core.digest import build_stable_digest
from paper_experiments.analysis.result_closure_gate import build_source_file_sha256_map
from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    ClosurePackageSelectionError,
    normalize_clean_code_version,
)
from scripts.write_pilot_paper_result_records import WorkProgress, materialize_output_entries

CONSTRUCTION_UNIT_NAME = "pilot_paper_complete_result_package"
DEFAULT_OUTPUT_DIR = Path("outputs/pilot_paper_complete_result_package")
COMMON_PROTOCOL_SUMMARY_NAME = "pilot_paper_common_protocol_summary.json"
ZIP_COMPRESSION_METHODS = {
    "stored": ZIP_STORED,
    "deflated": ZIP_DEFLATED,
}

# 每个目录都必须在其下继续绑定 paper_run_name。该结构使打包器能够只遍历
# 当前运行层级, 而不需要依赖文件内容猜测某个结果属于哪个统计规模。
REQUIRED_OUTPUT_DIR_NAMES = (
    "image_only_dataset_runtime",
    "formal_mechanism_ablation",
    "dataset_level_quality",
    "external_baseline_method_faithful",
    "tree_ring_official_reference",
    "gaussian_shading_official_reference",
    "shallow_diffuse_official_reference",
    "t2smark_formal_reproduction",
    "external_baseline_results",
    "primary_baseline_formal_import",
    "external_baseline_comparison",
    "pilot_paper_fixed_fpr_results",
    "pilot_paper_fixed_fpr_common_protocol",
    "pilot_paper_result_analysis",
    "attack_matrix",
    "fixed_fpr_threshold_audit",
    "primary_baseline_method_faithful_adapter_protocol",
    "primary_baseline_evidence",
    "official_reference_fidelity_evidence",
    "paired_superiority_analysis",
    "paper_artifact_evidence_audit",
    "submission_readiness",
    "evidence_closure_entry_review",
    "paper_result_closure",
    "result_closure_gate",
)

# Prompt 文件由 build_package_extra_paths 按当前运行层级动态加入。这里不登记其他
# 运行层级的 Prompt 文件, 避免归档携带与当前 claim scale 无关的数据定义。
PACKAGE_EXTRA_PATHS = (
    "configs/prompt_source_registry.json",
    "configs/model_sd35.yaml",
    "configs/model_source_registry.json",
    "configs/colab_sd35_runtime_constraints.txt",
    "docs/builds/prompt_dataset_provenance.md",
    "docs/builds/real_scientific_operator_implementation.md",
    "docs/builds/algorithm_primitives_semantic_conditioned_latent_manifold_watermark.md",
    "paper_workflow/README.md",
    "paper_workflow/colab_utils/semantic_watermark_image_only.py",
    "paper_workflow/notebooks/semantic_watermark_image_only_run.ipynb",
    "scripts/write_pilot_paper_result_records.py",
    "scripts/write_pilot_paper_fixed_fpr_common_protocol_outputs.py",
    "scripts/write_primary_baseline_result_candidates.py",
    "scripts/write_primary_baseline_formal_import_protocol.py",
    "scripts/write_primary_baseline_method_faithful_adapter_protocol.py",
    "scripts/write_primary_baseline_evidence_outputs.py",
    "scripts/write_official_reference_fidelity_evidence_outputs.py",
    "scripts/write_paired_superiority_outputs.py",
    "scripts/write_external_baseline_comparison_outputs.py",
    "scripts/write_attack_matrix_outputs.py",
    "scripts/write_fixed_fpr_threshold_audit_outputs.py",
    "scripts/write_paper_artifact_evidence_audit_outputs.py",
    "scripts/write_submission_readiness_outputs.py",
    "scripts/write_evidence_closure_entry_review_outputs.py",
    "scripts/write_result_closure_gate_outputs.py",
    "scripts/write_pilot_paper_complete_result_package.py",
    "scripts/run_gpu_server_result_closure.py",
    "scripts/run_image_only_dataset_runtime.py",
    "scripts/run_runtime_rerun_ablations.py",
    "experiments/artifacts/dataset_level_quality_outputs.py",
    "experiments/artifacts/detection_score_curves.py",
    "paper_experiments/baselines/official_reference_fidelity_evidence.py",
    "paper_experiments/analysis/paired_superiority.py",
    "paper_experiments/analysis/paper_artifact_data_validation.py",
    "paper_experiments/analysis/fixed_fpr_threshold_audit.py",
    "paper_experiments/analysis/paper_evidence_audit.py",
    "paper_experiments/analysis/submission_readiness.py",
    "paper_experiments/analysis/evidence_closure_entry_review.py",
    "paper_experiments/analysis/result_closure_gate.py",
    "paper_experiments/runners/closure_package_selection.py",
    "paper_experiments/runners/paper_result_closure.py",
    "experiments/runners/semantic_watermark_runtime.py",
    "experiments/runners/image_only_dataset_runtime.py",
    "experiments/ablations/runtime_rerun.py",
    "experiments/protocol/method_runtime_config.py",
    "experiments/protocol/paper_run_config.py",
    "experiments/runtime/model_sources.py",
    "experiments/runtime/diffusion/sd3_pipeline_runtime.py",
    "experiments/runtime/diffusion/semantic_features.py",
    "experiments/runtime/diffusion/regeneration_attacks.py",
    "experiments/runtime/image_attacks.py",
    "paper_experiments/runners/model_snapshot_runtime.py",
    "main/methods/semantic/branch_risk.py",
    "main/methods/subspace/jacobian_nullspace.py",
    "main/methods/carrier/keyed_tensor.py",
    "main/methods/geometry/differentiable_attention.py",
    "main/methods/geometry/attention_alignment.py",
    "main/methods/detection/image_only.py",
    "scripts/write_pilot_paper_result_analysis_outputs.py",
)


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 对象, 缺失文件返回空对象并由 readiness 门禁统一阻断。"""

    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


@dataclass(frozen=True)
class CompletePackageArchiveReceipt:
    """记录最终 zip 字节摘要与 Drive 镜像信息。

    该 receipt 必须写在 zip 外部。这样最终摘要不需要写回已经完成的归档,
    从根本上避免“归档内 metadata”与“归档后被覆盖的本地 metadata”不一致。
    """

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
    """把 JSON 兼容对象转换为稳定文本。"""

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
        raise ValueError("完整结果包输出目录不能为空")
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("完整结果包输出目录必须位于 outputs/ 下") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def build_required_output_dirs(paper_run_name: str) -> tuple[str, ...]:
    """构造当前论文运行层级独占的受治理输出目录。"""

    normalized = str(paper_run_name).strip()
    if not normalized:
        raise ValueError("paper_run_name 不能为空")
    return tuple(f"outputs/{directory_name}/{normalized}" for directory_name in REQUIRED_OUTPUT_DIR_NAMES)


def build_package_extra_paths(prompt_file: str | Path) -> tuple[str, ...]:
    """加入当前运行层级 Prompt 文件与共享重建实现。"""

    return (Path(prompt_file).as_posix(), *PACKAGE_EXTRA_PATHS)


def resolve_explicit_package_paths(
    root_path: Path,
    package_paths: Iterable[str | Path],
) -> tuple[Path, ...]:
    """只解析调用者显式给出的 zip 包, 不扫描任何父目录。"""

    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw_path in package_paths:
        package_path = resolve_path(root_path, raw_path)
        if package_path is None or not package_path.is_file() or package_path.suffix.lower() != ".zip":
            raise FileNotFoundError(f"显式论文结果包不存在或不是 zip: {raw_path}")
        if package_path not in seen:
            resolved.append(package_path)
            seen.add(package_path)
    return tuple(resolved)


def collect_required_entries(
    root_path: Path,
    required_output_dirs: Iterable[str],
    package_extra_paths: Iterable[str],
    excluded_paths: Iterable[Path] = (),
) -> tuple[Path, ...]:
    """收集当前论文运行层级的受治理文件。

    遍历只从 ``outputs/<artifact>/<paper_run_name>`` 开始。相邻运行层级即使
    同时存在于本地 ``outputs/`` 中, 也不会进入返回值。
    """

    excluded = {path.resolve() for path in excluded_paths}
    entries: list[Path] = []
    for relative_dir in required_output_dirs:
        directory = root_path / relative_dir
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.resolve() not in excluded and path.suffix.lower() != ".zip":
                entries.append(path)
    for relative_path in package_extra_paths:
        path = root_path / relative_path
        if path.is_file() and path.resolve() not in excluded:
            entries.append(path)

    unique_entries: list[Path] = []
    seen_paths: set[Path] = set()
    for entry in entries:
        resolved = entry.resolve()
        if resolved not in seen_paths:
            unique_entries.append(entry)
            seen_paths.add(resolved)
    return tuple(unique_entries)


def build_entry_payload_digest(root_path: Path, entries: Iterable[Path]) -> str:
    """对归档前的文件路径与真实内容摘要构造稳定 payload 摘要。"""

    payload = [
        {
            "path": relative_or_absolute(path, root_path),
            "sha256": file_digest(path),
            "size_bytes": path.stat().st_size,
        }
        for path in entries
    ]
    return build_stable_digest(payload)


def write_archive_with_progress(
    root_path: Path,
    archive_path: Path,
    entries: Iterable[Path],
    compression_method: int,
) -> None:
    """把完整结果写入 zip, 并按文件数和字节数显示打包进度。"""

    entry_list = tuple(entries)
    total_bytes = sum(path.stat().st_size for path in entry_list if path.is_file())
    progress = WorkProgress(
        "paper run complete package archive",
        len(entry_list),
        total_bytes=total_bytes,
        emit_every_count=100,
    )
    progress.emit(0, copied_bytes=0, profile=f"archive={archive_path.name}", force=True)
    copied_bytes = 0
    with ZipFile(archive_path, mode="w", compression=compression_method) as archive:
        for index, entry in enumerate(entry_list, start=1):
            archive.write(entry, relative_or_absolute(entry, root_path))
            copied_bytes += entry.stat().st_size
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


def _same_float(value: Any, expected: float) -> bool:
    """用固定绝对容差比较有限浮点值。"""

    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(resolved) and math.isclose(
        resolved,
        float(expected),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def _is_sha256(value: Any) -> bool:
    """判断字段是否是64位十六进制 SHA-256。"""

    text = str(value).strip()
    return len(text) == 64 and all(character in "0123456789abcdefABCDEF" for character in text)


def build_closure_input_lock_status(
    root_path: Path,
    *,
    paper_run_name: str,
    target_fpr: float,
    explicit_packages: Iterable[Path],
) -> dict[str, Any]:
    """核验当前运行层级 closure input lock 与显式输入包。

    锁文件只证明“曾经选择过”某些包还不够。完整包生成前必须重新计算每个
    zip 的 SHA-256, 并确认显式 ``--package-path`` 集合与锁定集合完全相同。
    这一检查可以阻止锁生成后包内容被替换, 也可以阻止打包器静默消费另一批包。
    """

    lock_path = (
        root_path
        / "outputs"
        / "paper_result_closure"
        / paper_run_name
        / "closure_input_lock.json"
    )
    lock_manifest_path = lock_path.parent / "input_lock_manifest.local.json"
    lock_payload = read_json(lock_path)
    lock_manifest = read_json(lock_manifest_path)
    records_value = lock_payload.get("closure_input_packages", ())
    records = (
        [dict(row) for row in records_value if isinstance(row, dict)]
        if isinstance(records_value, list)
        else []
    )
    records_shape_ready = isinstance(records_value, list) and len(records) == len(records_value)
    expected_families = tuple(spec.package_family for spec in CLOSURE_PACKAGE_FAMILY_SPECS)
    observed_families = [str(row.get("package_family", "")) for row in records]
    family_ready = (
        records_shape_ready
        and len(records) == len(expected_families)
        and len(set(observed_families)) == len(expected_families)
        and set(observed_families) == set(expected_families)
        and lock_payload.get("closure_input_package_count") == len(expected_families)
    )
    scope_ready = (
        lock_payload.get("paper_run_name") == paper_run_name
        and _same_float(lock_payload.get("target_fpr"), target_fpr)
        and all(row.get("paper_run_name") == paper_run_name for row in records)
        and all(_same_float(row.get("target_fpr"), target_fpr) for row in records)
    )
    declared_common_code_version = lock_payload.get("common_code_version")
    try:
        common_code_version = normalize_clean_code_version(declared_common_code_version)
        record_code_versions = {
            normalize_clean_code_version(row.get("code_version")) for row in records
        }
        common_code_version_ready = record_code_versions == {common_code_version}
    except ClosurePackageSelectionError:
        common_code_version = ""
        common_code_version_ready = False
    declared_lock_digest = str(lock_payload.get("closure_input_lock_digest", ""))
    digest_payload = dict(lock_payload)
    digest_payload.pop("closure_input_lock_digest", None)
    digest_ready = _is_sha256(declared_lock_digest) and declared_lock_digest == build_stable_digest(digest_payload)

    explicit_paths = {path.resolve() for path in explicit_packages}
    locked_paths: set[Path] = set()
    package_digests_ready = bool(records)
    package_metadata_ready = bool(records)
    for row in records:
        raw_path = str(row.get("package_path", "")).strip()
        package_path = Path(raw_path).expanduser()
        if not raw_path or not package_path.is_absolute():
            package_digests_ready = False
            package_metadata_ready = False
            continue
        resolved_package_path = package_path.resolve()
        locked_paths.add(resolved_package_path)
        declared_digest = str(row.get("package_sha256", ""))
        if (
            not resolved_package_path.is_file()
            or not _is_sha256(declared_digest)
            or file_digest(resolved_package_path) != declared_digest.lower()
        ):
            package_digests_ready = False
        if not str(row.get("code_version", "")).strip() or not str(row.get("generated_at", "")).strip():
            package_metadata_ready = False
    explicit_paths_ready = (
        len(explicit_paths) == len(expected_families)
        and explicit_paths == locked_paths
    )
    lock_relative = relative_or_absolute(lock_path, root_path)
    lock_manifest_relative = relative_or_absolute(lock_manifest_path, root_path)
    manifest_output_paths_value = lock_manifest.get("output_paths", ())
    manifest_output_paths = (
        {
            str(path).replace("\\", "/")
            for path in manifest_output_paths_value
            if str(path).strip()
        }
        if isinstance(manifest_output_paths_value, list | tuple)
        else set()
    )
    manifest_input_paths_value = lock_manifest.get("input_paths", ())
    manifest_input_paths = (
        [str(path).replace("\\", "/") for path in manifest_input_paths_value]
        if isinstance(manifest_input_paths_value, list | tuple)
        else []
    )
    manifest_metadata = lock_manifest.get("metadata", {})
    expected_manifest_config_digest = build_stable_digest(
        {
            "paper_run_name": paper_run_name,
            "target_fpr": target_fpr,
            "common_code_version": common_code_version,
            "closure_input_packages": records,
        }
    )
    manifest_ready = (
        lock_manifest.get("artifact_id") == f"{paper_run_name}_closure_input_lock_manifest"
        and lock_manifest.get("artifact_type") == "local_manifest"
        and lock_relative in manifest_output_paths
        and lock_manifest_relative in manifest_output_paths
        and manifest_input_paths == [str(row.get("package_path", "")).replace("\\", "/") for row in records]
        and lock_manifest.get("config_digest") == expected_manifest_config_digest
        and isinstance(manifest_metadata, dict)
        and manifest_metadata.get("closure_input_lock_ready") is True
        and manifest_metadata.get("closure_input_package_count") == len(expected_families)
        and manifest_metadata.get("closure_input_packages") == records
        and manifest_metadata.get("closure_input_lock_digest") == declared_lock_digest
        and manifest_metadata.get("paper_run_name") == paper_run_name
        and _same_float(manifest_metadata.get("target_fpr"), target_fpr)
        and manifest_metadata.get("common_code_version") == common_code_version
    )
    lock_ready = all(
        (
            lock_path.is_file(),
            lock_manifest_path.is_file(),
            family_ready,
            scope_ready,
            digest_ready,
            explicit_paths_ready,
            package_digests_ready,
            package_metadata_ready,
            common_code_version_ready,
            manifest_ready,
        )
    )
    return {
        "closure_input_lock_path": lock_relative,
        "closure_input_lock_present": lock_path.is_file(),
        "closure_input_lock_manifest_path": lock_manifest_relative,
        "closure_input_lock_manifest_ready": manifest_ready,
        "closure_input_lock_digest": declared_lock_digest,
        "closure_input_lock_digest_ready": digest_ready,
        "closure_input_package_count": len(records),
        "closure_input_family_ready": family_ready,
        "closure_input_scope_ready": scope_ready,
        "closure_input_explicit_paths_ready": explicit_paths_ready,
        "closure_input_package_digests_ready": package_digests_ready,
        "closure_input_package_metadata_ready": package_metadata_ready,
        "common_code_version": common_code_version,
        "closure_input_common_code_version_ready": common_code_version_ready,
        "closure_input_lock_ready": lock_ready,
    }


def build_result_closure_gate_status(
    root_path: Path,
    *,
    paper_run_name: str,
    target_fpr: float,
    expected_code_version: str,
) -> dict[str, Any]:
    """重算 closure gate 的报告、输入字节和 manifest 关键绑定."""

    gate_dir = root_path / "outputs" / "result_closure_gate" / paper_run_name
    report_path = gate_dir / "result_closure_gate_report.json"
    manifest_path = gate_dir / "manifest.local.json"
    report = read_json(report_path)
    manifest = read_json(manifest_path)
    report_relative = relative_or_absolute(report_path, root_path)
    manifest_relative = relative_or_absolute(manifest_path, root_path)
    output_paths_value = manifest.get("output_paths", ())
    output_paths = (
        {
            str(path).replace("\\", "/")
            for path in output_paths_value
            if str(path).strip()
        }
        if isinstance(output_paths_value, list | tuple)
        else set()
    )
    manifest_metadata = manifest.get("metadata", {})
    closure_source_file_sha256 = report.get("closure_source_file_sha256", {})
    source_map_shape_ready = (
        isinstance(closure_source_file_sha256, dict)
        and bool(closure_source_file_sha256)
        and all(
            isinstance(path, str)
            and bool(path.strip())
            and _is_sha256(digest)
            for path, digest in closure_source_file_sha256.items()
        )
    )
    source_paths: list[Path] = []
    if source_map_shape_ready:
        for recorded_path in closure_source_file_sha256:
            candidate = Path(recorded_path).expanduser()
            source_paths.append(
                candidate.resolve()
                if candidate.is_absolute()
                else (root_path / candidate).resolve()
            )
    try:
        recomputed_closure_source_file_sha256 = (
            build_source_file_sha256_map(source_paths, root=root_path)
            if source_map_shape_ready
            else {}
        )
    except (FileNotFoundError, OSError, ValueError):
        recomputed_closure_source_file_sha256 = {}
    source_files_ready = recomputed_closure_source_file_sha256 == closure_source_file_sha256
    closure_source_file_digest = str(report.get("closure_source_file_digest", ""))
    closure_source_file_digest_ready = (
        _is_sha256(closure_source_file_digest)
        and closure_source_file_digest == build_stable_digest(closure_source_file_sha256)
    )
    report_digest = (
        str(manifest_metadata.get("report_digest", ""))
        if isinstance(manifest_metadata, dict)
        else ""
    )
    report_digest_ready = (
        _is_sha256(report_digest)
        and report_path.is_file()
        and report_digest == file_digest(report_path)
    )
    manifest_input_paths_value = manifest.get("input_paths", ())
    manifest_input_paths = (
        [str(path).replace("\\", "/") for path in manifest_input_paths_value]
        if isinstance(manifest_input_paths_value, list | tuple)
        else []
    )
    source_path_set = set(closure_source_file_sha256) if isinstance(closure_source_file_sha256, dict) else set()
    manifest_input_paths_ready = (
        len(manifest_input_paths) == len(source_path_set)
        and set(manifest_input_paths) == source_path_set
    )
    input_bundle_digest = (
        str(manifest_metadata.get("input_bundle_digest", ""))
        if isinstance(manifest_metadata, dict)
        else ""
    )
    source_artifact_digests = report.get("source_artifact_digests", {})
    expected_prompt_id_digest = str(report.get("expected_prompt_id_digest", ""))
    expected_manifest_config = {
        "paper_claim_scale": paper_run_name,
        "target_fpr": target_fpr,
        "expected_prompt_count": report.get("expected_prompt_count"),
        "expected_test_count": report.get("expected_test_count"),
        "expected_prompt_id_digest": expected_prompt_id_digest,
        "input_bundle_digest": input_bundle_digest,
        "report_digest": report_digest,
        "source_artifact_digests": source_artifact_digests,
        "closure_source_file_sha256": closure_source_file_sha256,
        "closure_source_file_digest": closure_source_file_digest,
    }
    manifest_config_ready = (
        _is_sha256(input_bundle_digest)
        and manifest.get("config_digest") == build_stable_digest(expected_manifest_config)
    )
    try:
        manifest_code_version = normalize_clean_code_version(manifest.get("code_version"))
        normalized_expected_code_version = normalize_clean_code_version(expected_code_version)
        manifest_code_version_ready = manifest_code_version == normalized_expected_code_version
    except ClosurePackageSelectionError:
        manifest_code_version_ready = False
        normalized_expected_code_version = ""
    try:
        current_repository_code_version = normalize_clean_code_version(
            resolve_code_version(root_path)
        )
        current_repository_code_version_ready = (
            current_repository_code_version == normalized_expected_code_version
        )
    except ClosurePackageSelectionError:
        current_repository_code_version = ""
        current_repository_code_version_ready = False
    manifest_ready = (
        manifest.get("artifact_id") == f"{paper_run_name}_result_closure_gate_manifest"
        and manifest.get("artifact_type") == "local_manifest"
        and output_paths == {report_relative, manifest_relative}
        and manifest_input_paths_ready
        and manifest_config_ready
        and manifest_code_version_ready
        and current_repository_code_version_ready
        and isinstance(manifest_metadata, dict)
        and manifest_metadata.get("paper_claim_scale") == paper_run_name
        and _same_float(manifest_metadata.get("target_fpr"), target_fpr)
        and manifest_metadata.get("result_closure_ready") is True
        and manifest_metadata.get("closure_decision") == "pass"
        and manifest_metadata.get("evidence_closure_allowed") is True
        and manifest_metadata.get("input_bundle_digest") == input_bundle_digest
        and manifest_metadata.get("report_digest") == report_digest
        and manifest_metadata.get("source_artifact_digests") == source_artifact_digests
        and manifest_metadata.get("closure_source_file_sha256") == closure_source_file_sha256
        and manifest_metadata.get("closure_source_file_digest") == closure_source_file_digest
        and manifest_metadata.get("expected_prompt_id_digest") == expected_prompt_id_digest
    )
    report_ready = (
        report.get("paper_claim_scale") == paper_run_name
        and _same_float(report.get("target_fpr"), target_fpr)
        and report.get("result_closure_ready") is True
        and report.get("closure_decision") == "pass"
        and report.get("evidence_closure_allowed") is True
        and report.get("blocked_check_count") == 0
        and report.get("supports_paper_claim") is True
        and source_map_shape_ready
        and source_files_ready
        and closure_source_file_digest_ready
        and report_digest_ready
        and _is_sha256(expected_prompt_id_digest)
    )
    return {
        "result_closure_gate_report_path": report_relative,
        "result_closure_gate_report_present": report_path.is_file(),
        "result_closure_gate_manifest_path": manifest_relative,
        "result_closure_gate_manifest_ready": manifest_ready,
        "result_closure_gate_report_digest": report_digest,
        "result_closure_gate_report_digest_ready": report_digest_ready,
        "result_closure_gate_source_file_digest": closure_source_file_digest,
        "result_closure_gate_source_file_digest_ready": closure_source_file_digest_ready,
        "result_closure_gate_source_files_ready": source_files_ready,
        "result_closure_gate_manifest_config_ready": manifest_config_ready,
        "result_closure_gate_code_version_ready": manifest_code_version_ready,
        "current_repository_code_version": current_repository_code_version,
        "current_repository_code_version_ready": current_repository_code_version_ready,
        "result_closure_ready": report_ready and manifest_ready,
        "closure_decision": report.get("closure_decision", ""),
    }


def build_readiness_summary(
    root_path: Path,
    entries: Iterable[Path],
    materialization_report: dict[str, Any],
    paper_claim_scale: str,
    required_output_dirs: Iterable[str],
    *,
    target_fpr: float,
    explicit_packages: Iterable[Path],
) -> dict[str, Any]:
    """汇总当前论文运行层级完整结果包的 fail-closed 状态。"""

    required_dirs = tuple(required_output_dirs)
    common_protocol_summary_path = (
        root_path
        / "outputs"
        / "pilot_paper_fixed_fpr_common_protocol"
        / paper_claim_scale
        / COMMON_PROTOCOL_SUMMARY_NAME
    )
    common_protocol_summary = read_json(common_protocol_summary_path)
    existing_dirs = [relative_dir for relative_dir in required_dirs if (root_path / relative_dir).is_dir()]
    missing_dirs = [relative_dir for relative_dir in required_dirs if not (root_path / relative_dir).is_dir()]
    manifestless_dirs = [
        relative_dir
        for relative_dir in existing_dirs
        if not any((root_path / relative_dir).rglob("*manifest*.json"))
    ]
    entry_list = tuple(entries)
    run_claim_ready = common_protocol_summary.get("paper_run_claim_ready") is True
    closure_input_lock_status = build_closure_input_lock_status(
        root_path,
        paper_run_name=paper_claim_scale,
        target_fpr=target_fpr,
        explicit_packages=explicit_packages,
    )
    result_closure_gate_status = build_result_closure_gate_status(
        root_path,
        paper_run_name=paper_claim_scale,
        target_fpr=target_fpr,
        expected_code_version=closure_input_lock_status["common_code_version"],
    )
    package_ready = (
        not missing_dirs
        and not manifestless_dirs
        and bool(entry_list)
        and run_claim_ready
        and closure_input_lock_status["closure_input_lock_ready"]
        and result_closure_gate_status["result_closure_ready"]
    )
    probe_claim_ready = common_protocol_summary.get("probe_claim_ready") is True
    pilot_claim_ready = common_protocol_summary.get("pilot_claim_ready") is True
    full_claim_ready = common_protocol_summary.get("full_claim_ready") is True
    entry_paths = [relative_or_absolute(path, root_path) for path in entry_list]
    return {
        "construction_unit_name": CONSTRUCTION_UNIT_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_claim_scale": paper_claim_scale,
        "required_output_dir_count": len(required_dirs),
        "existing_required_output_dir_count": len(existing_dirs),
        "missing_required_output_dir_count": len(missing_dirs),
        "existing_required_output_dirs": existing_dirs,
        "missing_required_output_dirs": missing_dirs,
        "manifestless_required_output_dir_count": len(manifestless_dirs),
        "manifestless_required_output_dirs": manifestless_dirs,
        "archive_entry_count": len(entry_list),
        "archive_entry_digest": build_stable_digest(entry_paths),
        "materialization_report": materialization_report,
        **closure_input_lock_status,
        **result_closure_gate_status,
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


def build_empty_materialization_report(packages: Iterable[Path], *, materialize_packages: bool) -> dict[str, Any]:
    """构造未执行物化时的稳定审计记录。"""

    package_list = tuple(packages)
    return {
        "input_package_count": len(package_list),
        "input_package_paths": [path.as_posix() for path in package_list],
        "materialized_output_entry_count": 0,
        "materialized_output_total_bytes": 0,
        "skipped_output_entry_count": 0,
        "duplicate_identical_entry_count": 0,
        "materialized_output_entries_digest": build_stable_digest([]),
        "skipped_output_entries": [],
        "materialization_skipped": bool(package_list) and not materialize_packages,
    }


def require_complete_package_readiness(summary: dict[str, Any]) -> None:
    """在创建 zip 或 Drive 目录前强制执行完整结果包门禁。"""

    if summary.get("paper_run_complete_result_package_ready") is True:
        return
    missing_dirs = ",".join(str(value) for value in summary.get("missing_required_output_dirs", ()))
    manifestless_dirs = ",".join(str(value) for value in summary.get("manifestless_required_output_dirs", ()))
    raise RuntimeError(
        "完整结果包 readiness 未通过, 拒绝创建 zip 或复制 Drive: "
        f"paper_run_claim_ready={summary.get('paper_run_claim_ready')};"
        f"closure_input_lock_ready={summary.get('closure_input_lock_ready')};"
        f"result_closure_ready={summary.get('result_closure_ready')};"
        f"missing_dirs={missing_dirs};manifestless_dirs={manifestless_dirs}"
    )


def _validate_archive_name(archive_name: str) -> str:
    """限制 archive name 为当前目录中的单个 zip 文件名。"""

    normalized = str(archive_name).strip()
    path = Path(normalized)
    if not normalized or path.name != normalized or path.suffix.lower() != ".zip":
        raise ValueError("archive_name 必须是无目录前缀的 zip 文件名")
    return normalized


def write_pilot_paper_complete_result_package_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    drive_output_dir: str | None = None,
    archive_name: str | None = None,
    package_paths: Iterable[str | Path] = (),
    materialize_packages: bool = True,
    zip_compression: str = "stored",
) -> dict[str, Any]:
    """写出当前论文运行层级的完整结果包与包外 archive receipt。"""

    root_path = Path(root).resolve()
    formal_execution_run_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    paper_run = build_paper_run_config(root_path)
    resolved_drive_output_dir = (
        paper_run.drive_dir("complete_result_package") if drive_output_dir is None else drive_output_dir
    )
    resolved_archive_name = _validate_archive_name(
        archive_name or f"{paper_run.run_name}_complete_result_package.zip"
    )
    compression_method = ZIP_COMPRESSION_METHODS.get(str(zip_compression).strip().lower())
    if compression_method is None:
        raise ValueError(f"未知完整结果包 zip 压缩方式: {zip_compression}")

    output_path = ensure_output_dir_under_outputs(root_path, output_dir)
    archive_path = output_path / resolved_archive_name
    receipt_path = output_path / f"{Path(resolved_archive_name).stem}_archive_receipt.json"
    package_manifest_path = output_path / f"{paper_run.run_name}_complete_package_input_manifest.json"
    summary_path = output_path / f"{paper_run.run_name}_complete_package_readiness_summary.json"
    manifest_path = output_path / f"{paper_run.run_name}_complete_package_manifest.local.json"
    for stale_path in (archive_path, receipt_path, package_manifest_path, summary_path, manifest_path):
        if stale_path.exists():
            stale_path.unlink()

    packages = resolve_explicit_package_paths(root_path, package_paths)
    materialization_report = (
        materialize_output_entries(root_path, packages)
        if packages and materialize_packages
        else build_empty_materialization_report(packages, materialize_packages=materialize_packages)
    )
    required_output_dirs = build_required_output_dirs(paper_run.run_name)
    package_extra_paths = build_package_extra_paths(paper_run.prompt_file)
    payload_entries = collect_required_entries(
        root_path,
        required_output_dirs,
        package_extra_paths,
        excluded_paths=(archive_path, receipt_path, package_manifest_path, summary_path, manifest_path),
    )
    internal_metadata_entries = (package_manifest_path, summary_path, manifest_path)
    archive_entries = (*payload_entries, *internal_metadata_entries)
    readiness_summary = build_readiness_summary(
        root_path,
        archive_entries,
        materialization_report,
        paper_run.run_name,
        required_output_dirs,
        target_fpr=paper_run.target_fpr,
        explicit_packages=packages,
    )
    formal_execution_package_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    repository_environment.validate_formal_execution_lock_pair(
        formal_execution_run_lock,
        formal_execution_package_lock,
        formal_execution_run_lock["formal_execution_commit"],
    )
    payload_entry_paths = [relative_or_absolute(path, root_path) for path in payload_entries]
    archive_entry_paths = [relative_or_absolute(path, root_path) for path in archive_entries]
    entry_payload_digest = build_entry_payload_digest(root_path, payload_entries)
    package_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_claim_scale": paper_run.run_name,
        "payload_entry_paths": payload_entry_paths,
        "payload_entry_count": len(payload_entry_paths),
        "entry_paths": archive_entry_paths,
        "entry_count": len(archive_entry_paths),
        "entry_paths_digest": build_stable_digest(archive_entry_paths),
        "entry_payload_digest": entry_payload_digest,
        "formal_execution_run_lock": formal_execution_run_lock,
        "formal_execution_package_lock": formal_execution_package_lock,
    }
    write_json(package_manifest_path, package_manifest)
    write_json(summary_path, readiness_summary)

    rebuild_arguments = [
        "python",
        "scripts/write_pilot_paper_complete_result_package.py",
        "--root",
        root_path.as_posix(),
        "--output-dir",
        relative_or_absolute(output_path, root_path),
        "--archive-name",
        resolved_archive_name,
        "--drive-output-dir",
        str(resolved_drive_output_dir),
        "--zip-compression",
        str(zip_compression),
    ]
    for package_path in packages:
        rebuild_arguments.extend(("--package-path", package_path.as_posix()))
    if not materialize_packages:
        rebuild_arguments.append("--skip-package-materialization")

    manifest = build_artifact_manifest(
        artifact_id="pilot_paper_complete_result_package_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(
            [relative_or_absolute(path, root_path) for path in packages]
            + payload_entry_paths
        ),
        output_paths=(
            relative_or_absolute(archive_path, root_path),
            relative_or_absolute(package_manifest_path, root_path),
            relative_or_absolute(summary_path, root_path),
            relative_or_absolute(manifest_path, root_path),
            relative_or_absolute(receipt_path, root_path),
        ),
        config={
            "archive_name": resolved_archive_name,
            "drive_output_dir": resolved_drive_output_dir,
            "required_output_dirs": list(required_output_dirs),
            "explicit_package_paths": [relative_or_absolute(path, root_path) for path in packages],
            "materialize_packages": materialize_packages,
            "zip_compression": str(zip_compression),
        },
        code_version=formal_execution_package_lock["formal_execution_commit"],
        rebuild_command=subprocess.list2cmdline(rebuild_arguments),
        metadata={
            **readiness_summary,
            "archive_payload_digest": entry_payload_digest,
            "archive_digest_scope": "final_zip_bytes_external_sidecar",
            "final_archive_digest_available_in_sidecar": True,
        },
    ).to_dict()
    manifest["formal_execution_run_lock"] = formal_execution_run_lock
    manifest["formal_execution_package_lock"] = formal_execution_package_lock
    write_json(manifest_path, manifest)

    # 门禁必须位于 ZipFile 构造与 Drive 目录创建之前。失败时保留 readiness
    # summary 供审计, 但绝不产生看似可用的归档或远端副本。
    require_complete_package_readiness(readiness_summary)
    repository_environment.verify_formal_execution_lock_code_version(
        formal_execution_package_lock,
        readiness_summary.get("common_code_version"),
    )
    final_gate_status = build_result_closure_gate_status(
        root_path,
        paper_run_name=paper_run.run_name,
        target_fpr=paper_run.target_fpr,
        expected_code_version=str(readiness_summary["common_code_version"]),
    )
    if final_gate_status.get("result_closure_ready") is not True:
        raise RuntimeError("归档前结果闭合门禁输入字节绑定已失效, 拒绝创建 zip 或复制 Drive")
    write_archive_with_progress(root_path, archive_path, archive_entries, compression_method)
    try:
        final_package_lock = (
            repository_environment.require_published_formal_execution_lock(root_path)
        )
        repository_environment.validate_formal_execution_lock_pair(
            formal_execution_package_lock,
            final_package_lock,
            formal_execution_package_lock["formal_execution_commit"],
        )
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    archive_digest = file_digest_with_progress(archive_path, "paper run complete package digest")

    drive_archive_path = ""
    drive_archive_digest = ""
    drive_dir: Path | None = None
    if resolved_drive_output_dir:
        drive_dir = Path(resolved_drive_output_dir).expanduser()
        drive_dir.mkdir(parents=True, exist_ok=True)
        mirrored_path = drive_dir / resolved_archive_name
        copy_file_with_progress(archive_path, mirrored_path, "paper run complete package drive copy")
        drive_archive_path = str(mirrored_path)
        drive_archive_digest = archive_digest

    receipt = CompletePackageArchiveReceipt(
        archive_path=relative_or_absolute(archive_path, root_path),
        archive_digest=archive_digest,
        archive_entry_count=len(archive_entries),
        drive_archive_path=drive_archive_path,
        drive_archive_digest=drive_archive_digest,
        metadata={
            **readiness_summary,
            "archive_payload_digest": entry_payload_digest,
            "archive_digest_scope": "final_zip_bytes_external_sidecar",
            "final_archive_digest_available_in_sidecar": True,
        },
    ).to_dict()
    write_json(receipt_path, receipt)
    if drive_dir is not None:
        copy_file_with_progress(
            receipt_path,
            drive_dir / receipt_path.name,
            "paper run complete package receipt drive copy",
        )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="打包当前论文运行层级的完整结果包。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="本地输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--drive-output-dir", default=None, help="Google Drive 镜像目录。")
    parser.add_argument("--archive-name", default=None, help="压缩包文件名; 默认按当前 paper_run_name 生成。")
    parser.add_argument("--package-path", action="append", default=[], help="可重复传入的显式前序结果 zip 包。")
    parser.add_argument("--skip-package-materialization", action="store_true", help="跳过从显式前序 zip 物化 outputs/ 条目。")
    parser.add_argument(
        "--zip-compression",
        choices=sorted(ZIP_COMPRESSION_METHODS),
        default="stored",
        help="完整结果包 zip 压缩方式; stored 对 PNG 和已有压缩数据更快。",
    )
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    receipt = write_pilot_paper_complete_result_package_outputs(
        root=args.root,
        output_dir=args.output_dir,
        drive_output_dir=args.drive_output_dir,
        archive_name=args.archive_name,
        package_paths=args.package_path,
        materialize_packages=not args.skip_package_materialization,
        zip_compression=args.zip_compression,
    )
    print(stable_json_text(receipt), end="")


if __name__ == "__main__":
    main()
