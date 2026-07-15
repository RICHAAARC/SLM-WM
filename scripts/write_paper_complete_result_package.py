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
from pathlib import PurePosixPath
import shutil
import sys
from typing import Any, Iterable
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runtime.dependency_profiles import (
    REQUIRED_DEPENDENCY_PROFILE_NAMES,
    load_dependency_profile_registry,
)
from experiments.runtime.repository_environment import resolve_code_version
from main.core.digest import build_stable_digest
from paper_experiments.analysis.paper_claim_decisions import (
    ClaimDecisionGovernanceError,
    validate_claim_decision_bundle,
)
from paper_experiments.analysis.result_closure_gate import build_source_file_sha256_map
from paper_experiments.analysis.result_analysis_payload import (
    build_governed_paper_payload_path_map,
)
from paper_experiments.runners.paper_claim_provenance import (
    require_exact9_randomization_aggregate_provenance,
)
from scripts.write_paper_result_records import WorkProgress

CONSTRUCTION_UNIT_NAME = "paper_complete_result_package"
DEFAULT_OUTPUT_DIR = Path("outputs/paper_complete_result_package")
COMMON_PROTOCOL_SUMMARY_NAME = "paper_common_protocol_summary.json"
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
    "paper_fixed_fpr_results",
    "paper_fixed_fpr_common_protocol",
    "paper_result_analysis",
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

# Prompt 文件由 build_package_extra_paths 按当前运行层级动态加入. 共享重建代码
# 只允许来自 scripts 及更内层, 使完整结果包能够脱离 Notebook 和 Colab 包装复核.
# 完整选择清单是三级运行共享的来源证据, 但这里只加入当前运行层级的实际 Prompt 文件,
# 不把另外两个运行层级的独立 Prompt 配置当作本次运行输入.
PACKAGE_EXTRA_PATHS = (
    "configs/prompt_source_registry.json",
    "configs/prompt_selection_manifest.jsonl",
    "configs/model_sd35.yaml",
    "configs/model_source_registry.json",
    "configs/attack_conditioned_quality_estimand.json",
    "configs/independent_semantic_quality_evaluator.json",
    "configs/paper_quality_claim_protocol.json",
    "configs/paper_profile_protocol_registry.json",
    "configs/keyed_prg_cross_platform_known_answer.json",
    "configs/dependency_profile_registry.json",
    "configs/dependency_profiles/workflow_orchestrator_direct.txt",
    "configs/dependency_profiles/workflow_orchestrator_lock.txt",
    "configs/dependency_profiles/sd35_method_runtime_gpu_direct.txt",
    "configs/dependency_profiles/sd35_method_runtime_gpu_lock.txt",
    "configs/dependency_profiles/t2smark_sd35_gpu_direct.txt",
    "configs/dependency_profiles/t2smark_sd35_gpu_lock.txt",
    "configs/dependency_profiles/tree_ring_official_py39_cu117_direct.txt",
    "configs/dependency_profiles/tree_ring_official_py39_cu117_lock.txt",
    "configs/dependency_profiles/gaussian_shading_official_py38_cu117_direct.txt",
    "configs/dependency_profiles/gaussian_shading_official_py38_cu117_lock.txt",
    "configs/dependency_profiles/shallow_diffuse_official_py39_cu117_direct.txt",
    "configs/dependency_profiles/shallow_diffuse_official_py39_cu117_lock.txt",
    "docs/builds/prompt_dataset_provenance.md",
    "docs/builds/real_scientific_operator_implementation.md",
    "docs/builds/algorithm_primitives_semantic_conditioned_latent_manifold_watermark.md",
    "docs/builds/single_model_branch_risk_parameter_sensitivity.md",
    "docs/builds/formal_dependency_environment.md",
    "scripts/write_paper_result_records.py",
    "scripts/write_paper_fixed_fpr_common_protocol_outputs.py",
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
    "scripts/write_paper_complete_result_package.py",
    "scripts/run_gpu_server_result_closure.py",
    "scripts/formal_workflow_entry.py",
    "scripts/formal_workflow_environment.py",
    "scripts/run_formal_workflow_host.py",
    "scripts/run_gpu_server_workflow.py",
    "scripts/run_gpu_method_qualification.py",
    "scripts/write_gpu_method_qualification_report.py",
    "scripts/paper_result_closure.py",
    "scripts/semantic_watermark_scientific_workflow.py",
    "scripts/run_semantic_watermark_scientific_session.py",
    "scripts/run_image_only_dataset_runtime.py",
    "scripts/run_runtime_rerun_ablations.py",
    "experiments/ablations/branch_risk_sensitivity.py",
    "experiments/ablations/branch_risk_sensitivity_runtime.py",
    "experiments/ablations/branch_risk_sensitivity_workload.py",
    "scripts/build_external_baseline_command_plan.py",
    "scripts/run_external_baseline_command_plan.py",
    "scripts/validate_external_baseline_evidence.py",
    "experiments/artifacts/dataset_level_quality_outputs.py",
    "experiments/artifacts/paired_quality_outputs.py",
    "experiments/artifacts/independent_semantic_quality_outputs.py",
    "experiments/artifacts/detection_score_curves.py",
    "paper_experiments/baselines/official_reference_fidelity_evidence.py",
    "paper_experiments/baselines/method_faithful_adapter_protocol.py",
    "paper_experiments/baselines/primary_reproduction.py",
    "paper_experiments/baselines/tree_ring_official_reference.py",
    "paper_experiments/baselines/gaussian_shading_official_reference.py",
    "paper_experiments/baselines/shallow_diffuse_official_reference.py",
    "paper_experiments/analysis/paired_superiority.py",
    "paper_experiments/analysis/paper_profile_protocol_isomorphism.py",
    "paper_experiments/analysis/paper_quality_decisions.py",
    "paper_experiments/analysis/randomization_dataset_quality.py",
    "paper_experiments/analysis/paper_artifact_data_validation.py",
    "paper_experiments/analysis/fixed_fpr_threshold_audit.py",
    "paper_experiments/analysis/paper_evidence_audit.py",
    "paper_experiments/analysis/submission_readiness.py",
    "paper_experiments/analysis/evidence_closure_entry_review.py",
    "paper_experiments/analysis/result_closure_gate.py",
    "paper_experiments/runners/closure_package_selection.py",
    "paper_experiments/runners/randomization_repeat_evidence.py",
    "paper_experiments/runners/randomization_aggregate_provenance.py",
    "paper_experiments/runners/randomization_aggregate_record_workspace.py",
    "paper_experiments/runners/randomization_prompt_source_contract.py",
    "paper_experiments/runners/randomization_detection_statistics.py",
    "paper_experiments/runners/randomization_paired_superiority.py",
    "paper_experiments/runners/randomization_dataset_quality.py",
    "paper_experiments/runners/randomization_ablation_necessity.py",
    "paper_experiments/runners/randomization_parameter_sensitivity.py",
    "paper_experiments/runners/isolated_scientific_workflow.py",
    "paper_experiments/runners/external_baseline_method_faithful.py",
    "paper_experiments/runners/t2smark_formal_reproduction.py",
    "paper_experiments/runners/tree_ring_official_reference.py",
    "paper_experiments/runners/gaussian_shading_official_reference.py",
    "paper_experiments/runners/shallow_diffuse_official_reference.py",
    "paper_experiments/runners/official_reference_dependency_environment.py",
    "paper_experiments/baselines/command_plan_builder.py",
    "paper_experiments/baselines/command_plan_execution.py",
    "paper_experiments/baselines/evidence_validation_cli.py",
    "experiments/runners/semantic_watermark_runtime.py",
    "experiments/runners/image_only_dataset_workload.py",
    "experiments/runners/image_only_dataset_runtime.py",
    "experiments/ablations/mechanism_ablation_workload.py",
    "experiments/ablations/runtime_rerun.py",
    "experiments/protocol/method_runtime_config.py",
    "experiments/protocol/attack_conditioned_quality.py",
    "experiments/protocol/independent_semantic_quality.py",
    "experiments/protocol/gpu_method_qualification.py",
    "experiments/protocol/paper_run_config.py",
    "experiments/protocol/prompt_sources.py",
    "experiments/protocol/prompts.py",
    "experiments/protocol/splits.py",
    "experiments/runtime/model_sources.py",
    "experiments/runtime/dependency_profiles.py",
    "experiments/runtime/dependency_preparation.py",
    "experiments/runtime/isolated_dependency_environment.py",
    "experiments/runtime/isolated_scientific_execution.py",
    "experiments/runtime/scientific_execution_binding.py",
    "experiments/runtime/package_input_manifest.py",
    "experiments/runtime/semantic_watermark_scientific_session.py",
    "experiments/runtime/repository_environment.py",
    "experiments/runtime/diffusion/sd3_pipeline_runtime.py",
    "experiments/runtime/diffusion/semantic_model_loader.py",
    "main/methods/semantic/runtime.py",
    "experiments/runtime/diffusion/regeneration_attacks.py",
    "experiments/runtime/image_attacks.py",
    "paper_experiments/runners/model_snapshot_runtime.py",
    "main/methods/semantic/branch_risk.py",
    "main/methods/subspace/jacobian_nullspace.py",
    "main/methods/subspace/semantic_projection.py",
    "main/methods/carrier/keyed_tensor.py",
    "main/methods/geometry/differentiable_attention.py",
    "main/methods/geometry/attention_alignment.py",
    "main/methods/detection/image_only.py",
    "scripts/write_paper_result_analysis_outputs.py",
    "scripts/prepare_dependency_profile.py",
    "scripts/prepare_isolated_dependency_environment.py",
    "scripts/materialize_dependency_lock_candidate.py",
    "scripts/write_dependency_lock_review_bundle.py",
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


def _validated_zip_member_name(member_name: str) -> str:
    """把受治理来源 ZIP 成员限制为规范相对路径."""

    pure_path = PurePosixPath(member_name)
    if (
        not member_name
        or "\\" in member_name
        or pure_path.is_absolute()
        or any(part in {"", ".", ".."} for part in pure_path.parts)
        or pure_path.as_posix() != member_name
    ):
        raise RuntimeError(f"来源包包含非规范成员路径: {member_name}")
    return member_name


def verify_materialized_package_members(
    root_path: Path,
    output_members: dict[str, dict[str, Any]],
) -> tuple[Path, ...]:
    """确认本地上游产物仍与受治理来源包的成员字节完全一致."""

    verified: list[Path] = []
    outputs_root = (root_path / "outputs").resolve()
    for member_name, record in sorted(output_members.items()):
        target = (root_path / member_name).resolve()
        try:
            target.relative_to(outputs_root)
        except ValueError as exc:
            raise RuntimeError("来源包成员逃逸 outputs 目录") from exc
        if (
            not target.is_file()
            or target.is_symlink()
            or target.stat().st_size != int(record["member_size_bytes"])
            or file_digest(target) != record["member_sha256"]
        ):
            raise RuntimeError(
                f"已物化上游产物与受治理来源包不一致: {member_name}"
            )
        verified.append(target)
    return tuple(verified)


def copy_package_sources(
    root_path: Path,
    output_path: Path,
    paper_run_name: str,
    package_records: Iterable[dict[str, Any]],
) -> tuple[tuple[Path, ...], tuple[dict[str, Any], ...]]:
    """把受治理原始 ZIP 复制到完整包内部的确定性审计路径."""

    source_dir = output_path / "source_packages" / paper_run_name
    source_dir.mkdir(parents=True, exist_ok=True)
    copied_paths: list[Path] = []
    enriched_records: list[dict[str, Any]] = []
    for record in package_records:
        source_path = Path(str(record["source_package_path"])).resolve()
        family = str(record["package_family"])
        copied_path = source_dir / f"{family}.zip"
        shutil.copyfile(source_path, copied_path)
        copied_digest = file_digest(copied_path)
        if copied_digest != record["source_package_sha256"]:
            copied_path.unlink(missing_ok=True)
            raise RuntimeError(f"来源包审计副本摘要不一致: {family}")
        copied_paths.append(copied_path)
        enriched_records.append(
            {
                **record,
                "archived_source_path": relative_or_absolute(
                    copied_path,
                    root_path,
                ),
                "archived_source_sha256": copied_digest,
                "archived_source_size_bytes": copied_path.stat().st_size,
            }
        )
    return tuple(copied_paths), tuple(enriched_records)


def collect_result_closure_source_entries(
    root_path: Path,
    *,
    paper_run_name: str,
    excluded_paths: Iterable[Path],
) -> tuple[tuple[Path, ...], dict[str, str], bool]:
    """读取 closure gate 显式声明的输入集合, 不扫描任何结果目录."""

    gate_dir = root_path / "outputs" / "result_closure_gate" / paper_run_name
    report_path = gate_dir / "result_closure_gate_report.json"
    manifest_path = gate_dir / "manifest.local.json"
    excluded = {path.resolve() for path in excluded_paths}
    report = read_json(report_path)
    source_map_value = report.get("closure_source_file_sha256")
    source_map = (
        dict(source_map_value)
        if isinstance(source_map_value, dict)
        else {}
    )
    entries: list[Path] = []
    required_payload_paths = set(
        build_governed_paper_payload_path_map(paper_run_name).values()
    )
    source_map_ready = bool(source_map) and required_payload_paths <= set(source_map)
    for raw_path, raw_digest in sorted(source_map.items()):
        candidate = Path(str(raw_path)).expanduser()
        path = (
            candidate.resolve()
            if candidate.is_absolute()
            else (root_path / candidate).resolve()
        )
        try:
            path.relative_to(root_path)
        except ValueError:
            source_map_ready = False
            continue
        if (
            path in excluded
            or not path.is_file()
            or path.is_symlink()
            or not _is_sha256(raw_digest)
            or file_digest(path) != str(raw_digest).lower()
        ):
            source_map_ready = False
            continue
        entries.append(path)
    for governance_path in (report_path, manifest_path):
        if governance_path.is_file() and governance_path.resolve() not in excluded:
            entries.append(governance_path)
        else:
            source_map_ready = False
    return tuple(entries), source_map, source_map_ready


def collect_exact_complete_payload_entries(
    root_path: Path,
    *,
    required_output_dirs: Iterable[str],
    package_extra_paths: Iterable[str],
    package_member_paths: Iterable[Path],
    gate_source_paths: Iterable[Path],
    source_package_paths: Iterable[Path],
    source_provenance_path: Path | None,
    excluded_paths: Iterable[Path],
) -> tuple[tuple[Path, ...], dict[str, Any]]:
    """合并聚合来源成员、结果门禁输入和固定重建依赖的精确集合."""

    excluded = {path.resolve() for path in excluded_paths}
    candidates = [
        *package_member_paths,
        *gate_source_paths,
        *source_package_paths,
    ]
    if source_provenance_path is not None:
        candidates.append(source_provenance_path)
    for relative_path in package_extra_paths:
        candidates.append(root_path / relative_path)

    entries_by_path: dict[Path, Path] = {}
    invalid_paths: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root_path)
        except ValueError:
            invalid_paths.append(resolved.as_posix())
            continue
        if resolved in excluded:
            invalid_paths.append(relative_or_absolute(resolved, root_path))
            continue
        if not resolved.is_file() or resolved.is_symlink():
            continue
        entries_by_path[resolved] = resolved
    entries = tuple(
        entries_by_path[path]
        for path in sorted(entries_by_path, key=lambda item: item.as_posix())
    )
    entry_paths = {
        relative_or_absolute(path, root_path)
        for path in entries
    }
    required_dirs = tuple(required_output_dirs)
    uncovered_dirs = [
        relative_dir
        for relative_dir in required_dirs
        if not any(
            entry_path.startswith(relative_dir.rstrip("/") + "/")
            for entry_path in entry_paths
        )
    ]
    manifestless_dirs = [
        relative_dir
        for relative_dir in required_dirs
        if not any(
            entry_path.startswith(relative_dir.rstrip("/") + "/")
            and "manifest" in Path(entry_path).name
            and Path(entry_path).suffix.lower() == ".json"
            for entry_path in entry_paths
        )
    ]
    status = {
        "exact_payload_source_ready": (
            not invalid_paths
            and not uncovered_dirs
            and not manifestless_dirs
            and bool(entries)
        ),
        "exact_payload_invalid_paths": invalid_paths,
        "exact_payload_uncovered_required_output_dirs": uncovered_dirs,
        "exact_payload_manifestless_required_output_dirs": manifestless_dirs,
        "exact_payload_entry_count": len(entries),
        "exact_payload_entry_paths_digest": build_stable_digest(sorted(entry_paths)),
    }
    return entries, status


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


def build_archive_entry_records(
    root_path: Path,
    entries: Iterable[Path],
) -> tuple[dict[str, Any], ...]:
    """在 ZIP 写入前冻结每个成员的路径,大小和 SHA-256."""

    records = tuple(
        {
            "path": relative_or_absolute(path, root_path),
            "size_bytes": path.stat().st_size,
            "sha256": file_digest(path),
        }
        for path in entries
    )
    member_paths = [str(record["path"]) for record in records]
    if len(member_paths) != len(set(member_paths)):
        raise RuntimeError("完整结果包成员路径不得重复")
    return records


def validate_written_archive_entries(
    archive_path: Path,
    expected_records: Iterable[dict[str, Any]],
) -> None:
    """重开刚写出的 ZIP,逐成员复验路径,大小和真实内容摘要."""

    expected = {
        str(record["path"]): {
            "size_bytes": int(record["size_bytes"]),
            "sha256": str(record["sha256"]).lower(),
        }
        for record in expected_records
    }
    with ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise RuntimeError("完整结果包写后成员路径集合与冻结清单不一致")
        for info in infos:
            expected_record = expected[info.filename]
            if info.is_dir() or int(info.file_size) != expected_record["size_bytes"]:
                raise RuntimeError(
                    f"完整结果包写后成员大小不一致: {info.filename}"
                )
            digest = hashlib.sha256()
            with archive.open(info) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_record["sha256"]:
                raise RuntimeError(
                    f"完整结果包写后成员摘要不一致: {info.filename}"
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
    try:
        claim_decision_bundle = validate_claim_decision_bundle(report)
        claim_decision_bundle_ready = True
    except ClaimDecisionGovernanceError:
        claim_decision_bundle = {}
        claim_decision_bundle_ready = False
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
    expected_test_prompt_id_digest = str(
        report.get("expected_test_prompt_id_digest", "")
    )
    expected_manifest_config = {
        "paper_claim_scale": paper_run_name,
        "target_fpr": target_fpr,
        "expected_prompt_count": report.get("expected_prompt_count"),
        "expected_test_count": report.get("expected_test_count"),
        "expected_prompt_id_digest": expected_prompt_id_digest,
        "expected_test_prompt_id_digest": expected_test_prompt_id_digest,
        "input_bundle_digest": input_bundle_digest,
        "report_digest": report_digest,
        "source_artifact_digests": source_artifact_digests,
        "closure_source_file_sha256": closure_source_file_sha256,
        "closure_source_file_digest": closure_source_file_digest,
    }
    manifest_config_ready = (
        _is_sha256(input_bundle_digest)
        and manifest.get("config") == expected_manifest_config
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
        and manifest_metadata.get("expected_test_prompt_id_digest")
        == expected_test_prompt_id_digest
    )
    report_ready = (
        report.get("paper_claim_scale") == paper_run_name
        and _same_float(report.get("target_fpr"), target_fpr)
        and report.get("result_closure_ready") is True
        and report.get("closure_decision") == "pass"
        and report.get("evidence_closure_allowed") is True
        and report.get("blocked_check_count") == 0
        and claim_decision_bundle_ready
        and source_map_shape_ready
        and source_files_ready
        and closure_source_file_digest_ready
        and report_digest_ready
        and _is_sha256(expected_prompt_id_digest)
        and _is_sha256(expected_test_prompt_id_digest)
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
        "claim_decision_bundle_ready": claim_decision_bundle_ready,
        **claim_decision_bundle,
    }


def build_readiness_summary(
    root_path: Path,
    entries: Iterable[Path],
    materialization_report: dict[str, Any],
    paper_claim_scale: str,
    required_output_dirs: Iterable[str],
    *,
    target_fpr: float,
    randomization_aggregate_status: dict[str, Any],
    source_provenance_status: dict[str, Any] | None = None,
    exact_payload_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """汇总当前论文运行层级完整结果包的 fail-closed 状态。"""

    required_dirs = tuple(required_output_dirs)
    common_protocol_summary_path = (
        root_path
        / "outputs"
        / "paper_fixed_fpr_common_protocol"
        / paper_claim_scale
        / COMMON_PROTOCOL_SUMMARY_NAME
    )
    common_protocol_summary = read_json(common_protocol_summary_path)
    existing_dirs = [relative_dir for relative_dir in required_dirs if (root_path / relative_dir).is_dir()]
    missing_dirs = [relative_dir for relative_dir in required_dirs if not (root_path / relative_dir).is_dir()]
    resolved_exact_payload_status = dict(exact_payload_status or {})
    manifestless_dirs = list(
        resolved_exact_payload_status.get(
            "exact_payload_manifestless_required_output_dirs",
            (),
        )
    )
    entry_list = tuple(entries)
    resolved_randomization_aggregate_status = dict(
        randomization_aggregate_status
    )
    result_closure_gate_status = build_result_closure_gate_status(
        root_path,
        paper_run_name=paper_claim_scale,
        target_fpr=target_fpr,
        expected_code_version=str(
            resolved_randomization_aggregate_status.get(
                "common_code_version",
                "",
            )
        ),
    )
    run_claim_ready = (
        result_closure_gate_status.get("registered_claim_set_supported") is True
    )
    dependency_lock_status = build_dependency_lock_status(
        root_path,
        archive_entries=entry_list,
    )
    package_ready = (
        not missing_dirs
        and not manifestless_dirs
        and bool(entry_list)
        and resolved_randomization_aggregate_status.get(
            "randomization_aggregate_ready"
        )
        is True
        and result_closure_gate_status["result_closure_ready"]
        and dependency_lock_status["dependency_hash_locks_ready"]
        and resolved_exact_payload_status.get("exact_payload_source_ready") is True
        and (source_provenance_status or {}).get(
            "complete_package_source_provenance_ready"
        )
        is True
    )
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
        **resolved_randomization_aggregate_status,
        **result_closure_gate_status,
        **dependency_lock_status,
        **resolved_exact_payload_status,
        **dict(source_provenance_status or {}),
        "paper_run_complete_result_package_ready": package_ready,
        "paper_run_claim_ready": run_claim_ready,
        "paper_run_claim_type": common_protocol_summary.get("paper_run_claim_type", ""),
        "supports_paper_claim": run_claim_ready,
    }


def build_dependency_lock_status(
    root_path: Path,
    *,
    archive_entries: Iterable[Path],
) -> dict[str, Any]:
    """要求六个 profile 的完整哈希锁有效且实际进入归档范围."""

    entry_paths = {
        relative_or_absolute(path, root_path)
        for path in archive_entries
        if path.is_file()
    }
    try:
        profiles = load_dependency_profile_registry(
            root_path / "configs" / "dependency_profile_registry.json"
        )
    except (FileNotFoundError, ValueError) as error:
        return {
            "dependency_profile_count": 0,
            "dependency_profile_records": [],
            "dependency_hash_lock_count": 0,
            "dependency_hash_lock_archive_entries_ready": False,
            "dependency_profile_inputs_archive_entries_ready": False,
            "dependency_hash_locks_ready": False,
            "dependency_hash_lock_failure_reason": (
                f"dependency_profile_registry_invalid:{type(error).__name__}"
            ),
        }

    records: list[dict[str, Any]] = []
    for profile_id in REQUIRED_DEPENDENCY_PROFILE_NAMES:
        profile = profiles[profile_id]
        lock_path = root_path / profile.complete_hash_lock_path
        records.append(
            {
                "profile_id": profile.profile_name,
                "profile_digest": profile.profile_digest,
                "direct_requirements_path": profile.direct_requirements_path,
                "direct_requirements_digest": profile.direct_requirements_digest,
                "direct_requirements_in_archive": (
                    profile.direct_requirements_path in entry_paths
                ),
                "complete_hash_lock_path": profile.complete_hash_lock_path,
                "complete_hash_lock_present": profile.complete_hash_lock_present,
                "complete_hash_lock_digest": profile.complete_hash_lock_digest,
                "complete_hash_lock_dependency_count": (
                    profile.complete_hash_lock_dependency_count
                ),
                "complete_hash_lock_file_present": lock_path.is_file(),
                "complete_hash_lock_in_archive": (
                    profile.complete_hash_lock_path in entry_paths
                ),
                "formal_ready": profile.formal_ready,
                "readiness_blockers": list(profile.readiness_blockers),
            }
        )
    lock_records_ready = all(
        record["formal_ready"] is True
        and record["complete_hash_lock_present"] is True
        and record["complete_hash_lock_file_present"] is True
        and isinstance(record["complete_hash_lock_digest"], str)
        and len(record["complete_hash_lock_digest"]) == 64
        and int(record["complete_hash_lock_dependency_count"]) > 0
        and record["readiness_blockers"] == []
        for record in records
    )
    archive_entries_ready = all(
        record["complete_hash_lock_in_archive"] is True
        and record["direct_requirements_in_archive"] is True
        for record in records
    ) and (
        "configs/dependency_profile_registry.json" in entry_paths
    )
    locks_ready = (
        tuple(profiles) == REQUIRED_DEPENDENCY_PROFILE_NAMES
        and len(records) == len(REQUIRED_DEPENDENCY_PROFILE_NAMES)
        and lock_records_ready
        and archive_entries_ready
    )
    return {
        "dependency_profile_count": len(records),
        "dependency_profile_records": records,
        "dependency_hash_lock_count": sum(
            1 for record in records if record["complete_hash_lock_present"] is True
        ),
        "dependency_hash_lock_archive_entries_ready": archive_entries_ready,
        "dependency_profile_inputs_archive_entries_ready": archive_entries_ready,
        "dependency_hash_locks_ready": locks_ready,
        "dependency_hash_lock_failure_reason": (
            "" if locks_ready else "dependency_hash_lock_closure_incomplete"
        ),
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
        "randomization_aggregate_ready="
        f"{summary.get('randomization_aggregate_ready')};"
        f"result_closure_ready={summary.get('result_closure_ready')};"
        f"dependency_hash_locks_ready={summary.get('dependency_hash_locks_ready')};"
        f"missing_dirs={missing_dirs};manifestless_dirs={manifestless_dirs}"
    )


def _validate_archive_name(archive_name: str) -> str:
    """限制 archive name 为当前目录中的单个 zip 文件名。"""

    normalized = str(archive_name).strip()
    path = Path(normalized)
    if not normalized or path.name != normalized or path.suffix.lower() != ".zip":
        raise ValueError("archive_name 必须是无目录前缀的 zip 文件名")
    return normalized


def write_paper_complete_result_package_outputs(
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """验证精确9重复聚合来源后写出完整结果包."""

    return require_exact9_randomization_aggregate_provenance()


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
    receipt = write_paper_complete_result_package_outputs(
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
