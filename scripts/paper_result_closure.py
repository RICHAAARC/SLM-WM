"""按当前论文运行层级执行 CPU 结果闭合."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable, List

from experiments.protocol.paper_run_config import build_paper_run_config, normalize_paper_run_name
from experiments.runtime import repository_environment
from experiments.runtime.archive_naming import utc_archive_token
from experiments.runtime.repository_environment import (
    FORMAL_GIT_COMMIT_PATTERN,
    resolve_code_version,
)
from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    build_closure_input_selection_report,
)


CommandHook = Callable[[List[str]], None]
ProgressHook = Callable[[int, int, str], None]
PAPER_RESULT_CLOSURE_COMMAND_COUNT = 18


# 这些目录由当前闭合运行独占.正式执行会在物化锁定包前清理相同 run 的
# 内容, 从而使后续 builder 只能看到本次锁定输入, 而不会消费历史运行残留.
CLOSURE_RAW_OUTPUT_DIR_TEMPLATES: tuple[str, ...] = (
    "outputs/image_only_dataset_runtime/{paper_run_name}",
    "outputs/formal_mechanism_ablation/{paper_run_name}",
    "outputs/dataset_level_quality/{paper_run_name}",
    "outputs/external_baseline_method_faithful/{paper_run_name}",
    "outputs/tree_ring_official_reference/{paper_run_name}",
    "outputs/gaussian_shading_official_reference/{paper_run_name}",
    "outputs/shallow_diffuse_official_reference/{paper_run_name}",
    "outputs/t2smark_formal_reproduction/{paper_run_name}",
)

CLOSURE_DERIVED_OUTPUT_DIR_TEMPLATES: tuple[str, ...] = (
    "outputs/official_reference_fidelity_evidence/{paper_run_name}",
    "outputs/attack_matrix/{paper_run_name}",
    "outputs/paired_superiority_analysis/{paper_run_name}",
    "outputs/primary_baseline_method_faithful_adapter_protocol/{paper_run_name}",
    "outputs/external_baseline_results/{paper_run_name}",
    "outputs/primary_baseline_formal_import/{paper_run_name}",
    "outputs/fixed_fpr_threshold_audit/{paper_run_name}",
    "outputs/primary_baseline_evidence/{paper_run_name}",
    "outputs/external_baseline_comparison/{paper_run_name}",
    "outputs/pilot_paper_fixed_fpr_results/{paper_run_name}",
    "outputs/pilot_paper_fixed_fpr_common_protocol/{paper_run_name}",
    "outputs/pilot_paper_result_analysis/{paper_run_name}",
    "outputs/paper_artifact_evidence_audit/{paper_run_name}",
    "outputs/submission_readiness/{paper_run_name}",
    "outputs/evidence_closure_entry_review/{paper_run_name}",
    "outputs/result_closure_gate/{paper_run_name}",
    "outputs/pilot_paper_complete_result_package/{paper_run_name}",
)


def _short_commit(root: str | Path = ".") -> str:
    """从完整提交身份显式截取7位归档名称摘要."""

    code_version = resolve_code_version(Path(root).resolve())
    commit = code_version[:-6] if code_version.endswith("-dirty") else code_version
    if FORMAL_GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        raise RuntimeError("无法从完整 Git 提交身份构造归档名称")
    return commit[:7]


def _complete_archive_name(paper_run_name: str, *, root: str | Path = ".") -> str:
    """根据运行层级,UTC 时间和代码提交构造唯一归档名称."""

    return (
        f"{paper_run_name}_complete_result_package_"
        f"{utc_archive_token()}_{_short_commit(root)}.zip"
    )


def _path_argument(path: str | Path) -> str:
    """把命令参数路径转换为跨平台稳定文本."""

    return Path(path).as_posix()


def _command(script_name: str, *arguments: str) -> list[str]:
    """构造一个可脱离 Notebook 执行的 repository command."""

    return [sys.executable, f"scripts/{script_name}", *arguments]


def _package_record_map(
    closure_input_packages: Iterable[Mapping[str, Any]],
    *,
    paper_run_name: str,
    target_fpr: float,
) -> dict[str, dict[str, Any]]:
    """校验并索引恰好10个已锁定输入包记录."""

    records = [dict(record) for record in closure_input_packages]
    expected_families = tuple(
        specification.package_family for specification in CLOSURE_PACKAGE_FAMILY_SPECS
    )
    observed_families = [str(record.get("package_family", "")) for record in records]
    if (
        len(records) != len(expected_families)
        or len(set(observed_families)) != len(expected_families)
        or set(observed_families) != set(expected_families)
    ):
        raise ValueError("结果闭合命令必须绑定恰好10个互异的输入包 family")
    indexed = {str(record["package_family"]): record for record in records}
    for family_name in expected_families:
        record = indexed[family_name]
        package_path = str(record.get("package_path", "")).strip()
        if not package_path or not Path(package_path).is_absolute():
            raise ValueError(f"闭合输入包必须使用绝对路径: {family_name}")
        if record.get("paper_run_name") != paper_run_name or not math.isclose(
            float(record.get("target_fpr", float("nan"))),
            float(target_fpr),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"闭合输入包与当前论文运行协议不一致: {family_name}")
    return indexed


def _ordered_package_paths(package_records: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """按受治理 family 顺序返回显式包路径."""

    return tuple(
        str(package_records[specification.package_family]["package_path"])
        for specification in CLOSURE_PACKAGE_FAMILY_SPECS
    )


def _repeat_argument(name: str, values: Iterable[str | Path]) -> list[str]:
    """把可重复命令参数展开为平坦列表."""

    arguments: list[str] = []
    for value in values:
        arguments.extend((name, _path_argument(value)))
    return arguments


def _run_output_path(artifact_root: str, paper_run_name: str, file_name: str = "") -> str:
    """构造当前论文运行层级的受治理输出路径."""

    path = Path("outputs") / artifact_root / paper_run_name
    return (path / file_name).as_posix() if file_name else path.as_posix()


def clean_paper_result_closure_outputs(
    *,
    root: str | Path,
    paper_run_name: str,
    selected_package_paths: Iterable[str | Path],
) -> tuple[str, ...]:
    """清理当前 run 的受管输入物化目录和派生目录.

    输入锁目录不在清理集合中.若锁定 zip 位于待清理目录内, 函数会拒绝继续,
    避免先冻结输入后又删除输入文件.其他论文运行层级和非受管输出不受影响.
    """

    root_path = Path(root).resolve()
    outputs_root = (root_path / "outputs").resolve()
    normalized_run_name = normalize_paper_run_name(paper_run_name)
    selected_paths = tuple(Path(path).expanduser().resolve() for path in selected_package_paths)
    templates = CLOSURE_RAW_OUTPUT_DIR_TEMPLATES + CLOSURE_DERIVED_OUTPUT_DIR_TEMPLATES
    managed_paths: list[Path] = []
    for template in templates:
        managed_path = (root_path / template.format(paper_run_name=normalized_run_name)).resolve()
        try:
            managed_path.relative_to(outputs_root)
        except ValueError as error:
            raise ValueError("结果闭合清理路径必须位于 outputs/ 下") from error
        if any(
            package_path == managed_path or managed_path in package_path.parents
            for package_path in selected_paths
        ):
            raise ValueError(
                f"锁定输入包位于受管清理目录内, 拒绝删除: {managed_path.as_posix()}"
            )
        managed_paths.append(managed_path)

    removed_paths: list[str] = []
    for managed_path in managed_paths:
        if managed_path.is_dir():
            shutil.rmtree(managed_path)
            removed_paths.append(managed_path.relative_to(root_path).as_posix())
        elif managed_path.exists():
            managed_path.unlink()
            removed_paths.append(managed_path.relative_to(root_path).as_posix())
    return tuple(removed_paths)


def build_paper_result_closure_commands(
    *,
    closure_input_packages: Sequence[Mapping[str, Any]],
    complete_drive_output_dir: str | Path,
    paper_run_name: str,
    target_fpr: float,
    archive_name: str,
    root: str | Path = ".",
) -> list[list[str]]:
    """构造只消费本次锁定输入的 run-scoped CPU 闭合 DAG."""

    normalized_run_name = normalize_paper_run_name(paper_run_name)
    root_path = Path(root).resolve()
    package_records = _package_record_map(
        closure_input_packages,
        paper_run_name=normalized_run_name,
        target_fpr=float(target_fpr),
    )
    package_paths = _ordered_package_paths(package_records)
    runtime_dir = _run_output_path("image_only_dataset_runtime", normalized_run_name)
    ablation_dir = _run_output_path("formal_mechanism_ablation", normalized_run_name)
    quality_dir = _run_output_path("dataset_level_quality", normalized_run_name)
    method_collection_dir = _run_output_path(
        "external_baseline_method_faithful", normalized_run_name
    )
    t2smark_dir = _run_output_path("t2smark_formal_reproduction", normalized_run_name)
    attack_dir = _run_output_path("attack_matrix", normalized_run_name)
    official_reference_fidelity_dir = _run_output_path(
        "official_reference_fidelity_evidence", normalized_run_name
    )
    paired_superiority_dir = _run_output_path(
        "paired_superiority_analysis", normalized_run_name
    )
    adapter_dir = _run_output_path(
        "primary_baseline_method_faithful_adapter_protocol", normalized_run_name
    )
    candidate_dir = _run_output_path("external_baseline_results", normalized_run_name)
    formal_import_dir = _run_output_path("primary_baseline_formal_import", normalized_run_name)
    threshold_audit_dir = _run_output_path("fixed_fpr_threshold_audit", normalized_run_name)
    primary_evidence_dir = _run_output_path("primary_baseline_evidence", normalized_run_name)
    comparison_dir = _run_output_path("external_baseline_comparison", normalized_run_name)
    result_dir = _run_output_path("pilot_paper_fixed_fpr_results", normalized_run_name)
    common_protocol_dir = _run_output_path(
        "pilot_paper_fixed_fpr_common_protocol", normalized_run_name
    )
    analysis_dir = _run_output_path("pilot_paper_result_analysis", normalized_run_name)
    evidence_audit_dir = _run_output_path(
        "paper_artifact_evidence_audit", normalized_run_name
    )
    submission_dir = _run_output_path("submission_readiness", normalized_run_name)
    entry_review_dir = _run_output_path("evidence_closure_entry_review", normalized_run_name)
    complete_output_dir = _run_output_path(
        "pilot_paper_complete_result_package", normalized_run_name
    )

    root_argument = root_path.as_posix()
    materialize_command = _command(
        "write_pilot_paper_result_records.py",
        "--root",
        root_argument,
        "--output-dir",
        result_dir,
        *_repeat_argument("--package-path", package_paths),
        "--materialize-only",
    )
    official_reference_fidelity_command = _command(
        "write_official_reference_fidelity_evidence_outputs.py",
        "--root",
        root_argument,
        "--output-dir",
        official_reference_fidelity_dir,
        "--require-pass",
    )
    attack_command = _command(
        "write_attack_matrix_outputs.py",
        "--root",
        root_argument,
        "--paper-run-name",
        normalized_run_name,
        "--dataset-runtime-dir",
        runtime_dir,
        "--output-dir",
        attack_dir,
        "--image-attack-evidence-records-path",
        f"{attack_dir}/formal_attack_detection_records.jsonl",
    )
    threshold_command = _command(
        "write_fixed_fpr_threshold_audit_outputs.py",
        "--root",
        root_argument,
        "--output-dir",
        threshold_audit_dir,
        "--method-faithful-collection-root",
        method_collection_dir,
        "--t2smark-output-dir",
        t2smark_dir,
        "--require-pass",
    )
    paired_superiority_command = _command(
        "write_paired_superiority_outputs.py",
        "--root",
        root_argument,
        "--output-dir",
        paired_superiority_dir,
        "--proposed-records-path",
        f"{runtime_dir}/image_only_detection_records.jsonl",
        "--method-faithful-root",
        method_collection_dir,
        "--t2smark-observations-path",
        f"{t2smark_dir}/t2smark_adapter/baseline_observations.json",
        "--threshold-audit-rows-path",
        f"{threshold_audit_dir}/threshold_audit_rows.csv",
        "--threshold-audit-report-path",
        f"{threshold_audit_dir}/threshold_audit_report.json",
        "--threshold-audit-manifest-path",
        f"{threshold_audit_dir}/manifest.local.json",
        "--require-pass",
    )
    adapter_command = _command(
        "write_primary_baseline_method_faithful_adapter_protocol.py",
        "--root",
        root_argument,
        "--output-dir",
        adapter_dir,
        "--collection-root",
        method_collection_dir,
    )
    candidate_command = _command(
        "write_primary_baseline_result_candidates.py",
        "--root",
        root_argument,
        "--output-dir",
        candidate_dir,
        "--attack-manifest-path",
        f"{attack_dir}/attack_manifest.json",
        "--method-faithful-collection-path",
        method_collection_dir,
        "--t2smark-candidate-records-path",
        f"{t2smark_dir}/t2smark_formal_import_candidate_records.jsonl",
    )
    formal_import_command = _command(
        "write_primary_baseline_formal_import_protocol.py",
        "--root",
        root_argument,
        "--output-dir",
        formal_import_dir,
        "--source-registry-path",
        "external_baseline/source_registry.json",
        "--attack-manifest-path",
        f"{attack_dir}/attack_manifest.json",
        "--attack-family-metrics-path",
        f"{attack_dir}/attack_family_metrics.csv",
        "--candidate-records-path",
        f"{candidate_dir}/baseline_result_records.jsonl",
    )
    primary_evidence_command = _command(
        "write_primary_baseline_evidence_outputs.py",
        "--root",
        root_argument,
        "--output-dir",
        primary_evidence_dir,
        "--source-registry-path",
        "external_baseline/source_registry.json",
        "--collection-root",
        method_collection_dir,
        "--t2smark-formal-output-dir",
        t2smark_dir,
    )
    comparison_command = _command(
        "write_external_baseline_comparison_outputs.py",
        "--root",
        root_argument,
        "--output-dir",
        comparison_dir,
        "--attack-manifest-path",
        f"{attack_dir}/attack_manifest.json",
        "--attack-family-metrics-path",
        f"{attack_dir}/attack_family_metrics.csv",
        "--attack-matrix-manifest-path",
        f"{attack_dir}/manifest.local.json",
        "--threshold-report-path",
        f"{threshold_audit_dir}/threshold_audit_report.json",
        "--baseline-result-records-path",
        f"{candidate_dir}/baseline_result_records.jsonl",
        "--baseline-source-registry-path",
        "external_baseline/source_registry.json",
        "--evidence-search-root",
        root_argument,
    )
    result_record_command = _command(
        "write_pilot_paper_result_records.py",
        "--root",
        root_argument,
        "--output-dir",
        result_dir,
        "--baseline-records-path",
        f"{comparison_dir}/baseline_result_records.jsonl",
        "--baseline-validation-report-path",
        f"{comparison_dir}/baseline_formal_import_validation_report.json",
        "--dataset-quality-metrics-path",
        f"{quality_dir}/dataset_quality_metrics.csv",
        "--image-only-runtime-dir",
        runtime_dir,
        "--require-existing-evidence",
    )
    common_protocol_command = _command(
        "write_pilot_paper_fixed_fpr_common_protocol_outputs.py",
        "--root",
        root_argument,
        "--output-dir",
        common_protocol_dir,
        "--candidate-records-path",
        f"{result_dir}/pilot_paper_result_records.jsonl",
        "--paired-superiority-summary-path",
        f"{paired_superiority_dir}/paired_superiority_summary.json",
        "--paired-superiority-manifest-path",
        f"{paired_superiority_dir}/manifest.local.json",
        "--require-existing-evidence",
    )
    analysis_command = _command(
        "write_pilot_paper_result_analysis_outputs.py",
        "--root",
        root_argument,
        "--output-dir",
        analysis_dir,
        "--result-records-path",
        f"{result_dir}/pilot_paper_result_records.jsonl",
        "--attack-detection-records-path",
        f"{attack_dir}/attack_detection_records.jsonl",
        "--paired-superiority-summary-path",
        f"{paired_superiority_dir}/paired_superiority_summary.json",
        "--paired-superiority-table-path",
        f"{paired_superiority_dir}/paired_superiority_table.csv",
        "--paired-superiority-manifest-path",
        f"{paired_superiority_dir}/manifest.local.json",
    )
    evidence_audit_command = _command(
        "write_paper_artifact_evidence_audit_outputs.py",
        "--root",
        root_argument,
        "--output-dir",
        evidence_audit_dir,
        "--threshold-report-path",
        f"{runtime_dir}/dataset_runtime_summary.json",
        "--threshold-manifest-path",
        f"{runtime_dir}/manifest.local.json",
        "--threshold-audit-report-path",
        f"{threshold_audit_dir}/threshold_audit_report.json",
        "--threshold-audit-manifest-path",
        f"{threshold_audit_dir}/manifest.local.json",
        "--attack-manifest-path",
        f"{attack_dir}/attack_manifest.json",
        "--attack-matrix-manifest-path",
        f"{attack_dir}/manifest.local.json",
        "--baseline-manifest-path",
        f"{comparison_dir}/manifest.local.json",
        "--baseline-runtime-report-path",
        f"{comparison_dir}/baseline_runtime_report.json",
        "--dataset-quality-manifest-path",
        f"{quality_dir}/manifest.local.json",
        "--dataset-quality-summary-path",
        f"{quality_dir}/dataset_quality_summary.json",
        "--ablation-manifest-path",
        f"{ablation_dir}/manifest.local.json",
        "--ablation-claim-summary-path",
        f"{ablation_dir}/ablation_claim_summary.json",
    )
    submission_command = _command(
        "write_submission_readiness_outputs.py",
        "--root",
        root_argument,
        "--output-dir",
        submission_dir,
        "--evidence-manifest-path",
        f"{evidence_audit_dir}/manifest.local.json",
        "--builder-report-path",
        f"{evidence_audit_dir}/artifact_builder_readiness_report.json",
        "--blocker-report-path",
        f"{evidence_audit_dir}/submission_blocker_report.json",
        "--gap-list-path",
        f"{evidence_audit_dir}/evidence_gap_list.csv",
    )
    entry_review_command = _command(
        "write_evidence_closure_entry_review_outputs.py",
        "--root",
        root_argument,
        "--output-dir",
        entry_review_dir,
        "--submission-readiness-report-path",
        f"{submission_dir}/readiness_blocker_report.json",
        "--required-evidence-inputs-path",
        f"{submission_dir}/required_evidence_inputs.csv",
        "--paper-blocker-report-path",
        f"{evidence_audit_dir}/submission_blocker_report.json",
        "--baseline-runtime-report-path",
        f"{comparison_dir}/baseline_runtime_report.json",
        "--dataset-quality-summary-path",
        f"{quality_dir}/dataset_quality_summary.json",
    )
    gate_command = _command(
        "write_result_closure_gate_outputs.py",
        "--root",
        root_argument,
        "--output-root",
        "outputs/result_closure_gate",
        "--attack-report-path",
        f"{attack_dir}/attack_manifest.json",
        "--attack-manifest-path",
        f"{attack_dir}/manifest.local.json",
        "--threshold-audit-report-path",
        f"{threshold_audit_dir}/threshold_audit_report.json",
        "--threshold-audit-rows-path",
        f"{threshold_audit_dir}/threshold_audit_rows.csv",
        "--threshold-audit-manifest-path",
        f"{threshold_audit_dir}/manifest.local.json",
        "--primary-baseline-evidence-summary-path",
        f"{primary_evidence_dir}/primary_baseline_evidence_summary.json",
        "--primary-baseline-evidence-manifest-path",
        f"{primary_evidence_dir}/manifest.local.json",
        "--baseline-report-path",
        f"{comparison_dir}/baseline_runtime_report.json",
        "--baseline-manifest-path",
        f"{comparison_dir}/manifest.local.json",
        "--result-records-path",
        f"{result_dir}/pilot_paper_result_records.jsonl",
        "--result-record-summary-path",
        f"{result_dir}/pilot_paper_result_record_summary.json",
        "--result-record-manifest-path",
        f"{result_dir}/manifest.local.json",
        "--common-protocol-summary-path",
        f"{common_protocol_dir}/pilot_paper_common_protocol_summary.json",
        "--common-protocol-schema-path",
        f"{common_protocol_dir}/pilot_paper_result_import_schema.json",
        "--common-protocol-manifest-path",
        f"{common_protocol_dir}/manifest.local.json",
        "--result-analysis-summary-path",
        f"{analysis_dir}/result_analysis_summary.json",
        "--result-analysis-manifest-path",
        f"{analysis_dir}/manifest.local.json",
        "--paired-outcomes-path",
        f"{paired_superiority_dir}/paired_outcomes.jsonl",
        "--paired-superiority-rows-path",
        f"{paired_superiority_dir}/paired_superiority_table.csv",
        "--paired-superiority-summary-path",
        f"{paired_superiority_dir}/paired_superiority_summary.json",
        "--paired-superiority-manifest-path",
        f"{paired_superiority_dir}/manifest.local.json",
        "--ablation-summary-path",
        f"{ablation_dir}/ablation_claim_summary.json",
        "--ablation-manifest-path",
        f"{ablation_dir}/manifest.local.json",
        "--dataset-quality-summary-path",
        f"{quality_dir}/dataset_quality_summary.json",
        "--dataset-quality-feature-records-path",
        f"{quality_dir}/dataset_quality_formal_feature_records.jsonl",
        "--dataset-quality-feature-report-path",
        f"{quality_dir}/dataset_quality_formal_feature_import_report.json",
        "--dataset-quality-metrics-path",
        f"{quality_dir}/dataset_quality_metrics.csv",
        "--dataset-quality-manifest-path",
        f"{quality_dir}/manifest.local.json",
        "--evidence-builder-report-path",
        f"{evidence_audit_dir}/artifact_builder_readiness_report.json",
        "--evidence-blocker-report-path",
        f"{evidence_audit_dir}/submission_blocker_report.json",
        "--artifact-data-validation-report-path",
        f"{evidence_audit_dir}/artifact_data_validation_report.json",
        "--evidence-audit-manifest-path",
        f"{evidence_audit_dir}/manifest.local.json",
        "--submission-readiness-report-path",
        f"{submission_dir}/readiness_blocker_report.json",
        "--submission-readiness-manifest-path",
        f"{submission_dir}/submission_readiness_manifest.local.json",
        "--entry-review-report-path",
        f"{entry_review_dir}/entry_review_report.json",
        "--entry-review-manifest-path",
        f"{entry_review_dir}/manifest.local.json",
        "--require-pass",
    )
    complete_command = _command(
        "write_pilot_paper_complete_result_package.py",
        "--root",
        root_argument,
        "--output-dir",
        complete_output_dir,
        "--drive-output-dir",
        _path_argument(complete_drive_output_dir),
        "--archive-name",
        archive_name,
        *_repeat_argument("--package-path", package_paths),
        "--skip-package-materialization",
        "--zip-compression",
        "stored",
    )
    return [
        materialize_command,
        official_reference_fidelity_command,
        attack_command,
        threshold_command,
        paired_superiority_command,
        adapter_command,
        candidate_command,
        formal_import_command,
        primary_evidence_command,
        comparison_command,
        result_record_command,
        common_protocol_command,
        analysis_command,
        evidence_audit_command,
        submission_command,
        entry_review_command,
        gate_command,
        complete_command,
    ]


def run_paper_result_closure_commands(
    *,
    package_search_root: str | Path,
    complete_drive_output_dir: str | Path,
    paper_run_name: str,
    target_fpr: float,
    root: str | Path = ".",
    before_command: CommandHook | None = None,
    progress_hook: ProgressHook | None = None,
) -> dict[str, Any]:
    """锁定输入,清理当前 run,执行闭合 DAG 并返回精确归档路径."""

    root_path = Path(root).resolve()
    formal_execution_run_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    normalized_run_name = normalize_paper_run_name(paper_run_name)
    paper_run = build_paper_run_config(root_path)
    if paper_run.run_name != normalized_run_name or not math.isclose(
        paper_run.target_fpr,
        float(target_fpr),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "显式 paper_run_name/target_fpr 必须与当前 build_paper_run_config 完全一致"
        )
    selection_report = build_closure_input_selection_report(
        package_search_root,
        paper_run_name=normalized_run_name,
        target_fpr=float(target_fpr),
        root=root_path,
        write_lock=True,
    )
    closure_input_packages = tuple(selection_report["closure_input_packages"])
    package_paths = tuple(selection_report["selected_package_paths"])
    removed_paths = clean_paper_result_closure_outputs(
        root=root_path,
        paper_run_name=normalized_run_name,
        selected_package_paths=package_paths,
    )
    archive_name = _complete_archive_name(normalized_run_name, root=root_path)
    commands = build_paper_result_closure_commands(
        closure_input_packages=closure_input_packages,
        complete_drive_output_dir=complete_drive_output_dir,
        paper_run_name=normalized_run_name,
        target_fpr=float(target_fpr),
        archive_name=archive_name,
        root=root_path,
    )
    for command_index, command in enumerate(commands, start=1):
        command_name = command[1] if len(command) > 1 else command[0]
        if before_command is not None:
            before_command(command)
        print("run_repository_command", subprocess.list2cmdline(command))
        subprocess.run(command, cwd=root_path, check=True)
        if progress_hook is not None:
            progress_hook(command_index, len(commands), command_name)

    local_archive_path = (
        root_path
        / _run_output_path("pilot_paper_complete_result_package", normalized_run_name)
        / archive_name
    ).resolve()
    drive_output_path = Path(complete_drive_output_dir).expanduser()
    if not drive_output_path.is_absolute():
        drive_output_path = (root_path / drive_output_path).resolve()
    drive_archive_path = (drive_output_path / archive_name).resolve()
    if not local_archive_path.is_file():
        raise FileNotFoundError(f"本次完整结果归档未生成: {local_archive_path.as_posix()}")
    if not drive_archive_path.is_file():
        raise FileNotFoundError(f"本次完整结果归档未写回: {drive_archive_path.as_posix()}")
    formal_execution_package_lock = (
        repository_environment.require_published_formal_execution_lock(root_path)
    )
    repository_environment.validate_formal_execution_lock_pair(
        formal_execution_run_lock,
        formal_execution_package_lock,
        formal_execution_run_lock["formal_execution_commit"],
    )
    return {
        "complete_archive_path": drive_archive_path.as_posix(),
        "local_complete_archive_path": local_archive_path.as_posix(),
        "drive_complete_archive_path": drive_archive_path.as_posix(),
        "complete_archive_name": archive_name,
        "command_count": len(commands),
        "paper_run_name": normalized_run_name,
        "target_fpr": float(target_fpr),
        "closure_input_lock_path": selection_report["closure_input_lock_path"],
        "closure_input_lock_digest": selection_report["closure_input_lock_digest"],
        "closure_input_packages": list(closure_input_packages),
        "removed_managed_output_paths": list(removed_paths),
        "formal_execution_run_lock": formal_execution_run_lock,
        "formal_execution_package_lock": formal_execution_package_lock,
    }


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本."""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
