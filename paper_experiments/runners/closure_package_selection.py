"""为 CPU 论文结果闭合选择并冻结精确的上游结果包."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping
import zlib
from zipfile import BadZipFile, ZipFile

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.protocol.formal_randomization import (
    formal_randomization_protocol_record,
    resolve_formal_randomization_repeat,
    validate_formal_randomization_repeat_records,
)
from experiments.protocol.paper_run_config import normalize_paper_run_name
from experiments.runtime.dependency_profiles import require_dependency_profile_ready
from experiments.runtime.repository_environment import (
    FormalExecutionLockError,
    normalize_formal_git_commit,
    resolve_code_version,
    validate_formal_execution_lock_record,
)
from experiments.runtime.scientific_execution_binding import (
    BOUND_MANIFEST_DIGEST_SCOPE,
    normalize_scientific_absolute_path,
    semantic_command_sequence_digest,
    scientific_manifest_payload_digest,
    validate_dependency_environment_report_snapshot,
    validate_scientific_command_context_snapshot,
    validate_semantic_watermark_dispatch_report,
    validate_semantic_watermark_dispatch_artifact_snapshot,
)
from paper_experiments.runners.external_source_runtime import (
    CUDA_INSPECTION_PROGRAM,
    stable_json_payload_digest,
)


LOCK_OUTPUT_ROOT = Path("outputs/paper_result_closure")
LOCK_FILENAME = "closure_input_lock.json"
LOCK_MANIFEST_FILENAME = "input_lock_manifest.local.json"
MAX_GOVERNANCE_MEMBER_BYTES = 32 * 1024 * 1024
SEMANTIC_WATERMARK_PACKAGE_FAMILIES = frozenset(
    {
        "image_only_dataset_runtime",
        "dataset_level_quality",
        "runtime_rerun_ablation",
    }
)
RANDOMIZATION_REPEAT_PACKAGE_FAMILIES = frozenset(
    {
        "image_only_dataset_runtime",
        "runtime_rerun_ablation",
        "dataset_level_quality",
        "method_faithful_tree_ring",
        "method_faithful_gaussian_shading",
        "method_faithful_shallow_diffuse",
        "official_reference_t2smark",
    }
)
CROSS_REPEAT_INVARIANT_PACKAGE_FAMILIES = frozenset(
    {
        "official_reference_tree_ring",
        "official_reference_gaussian_shading",
        "official_reference_shallow_diffuse",
    }
)
SEMANTIC_SESSION_IDENTITY_FIELDS = (
    "code_version",
    "formal_execution_run_lock_digest",
    "scientific_profile_id",
    "scientific_profile_digest",
    "scientific_direct_requirements_digest",
    "scientific_complete_hash_lock_digest",
    "scientific_complete_hash_lock_dependency_count",
    "scientific_python_executable_digest",
    "scientific_execution_report_digest",
    "scientific_command_sequence_digest",
    "scientific_dependency_evidence_digest",
)


class ClosurePackageSelectionError(ValueError):
    """表示闭合输入包未满足精确选择契约."""


@dataclass(frozen=True)
class JsonFieldSource:
    """描述 ZIP 内一个 JSON 字段的来源."""

    member_template: str
    field_path: tuple[str, ...]


@dataclass(frozen=True)
class JsonValueRequirement:
    """描述 ZIP 内治理字段必须满足的精确值."""

    source: JsonFieldSource
    expected_value: Any


@dataclass(frozen=True)
class BaselineRowsSource:
    """描述用于交叉核验 baseline 身份的 JSONL 记录."""

    member_template: str
    baseline_field_path: tuple[str, ...] = ("baseline_id",)


@dataclass(frozen=True)
class ScientificExecutionBindingSpec:
    """描述结果包内隔离科学执行绑定的精确成员身份."""

    artifact_role: str
    profile_id: str
    execution_route: str
    binding_member_template: str
    execution_report_member_template: str
    dependency_report_member_template: str
    dispatch_report_member_template: str
    summary_member_template: str
    manifest_member_template: str


@dataclass(frozen=True)
class DependencyEnvironmentEvidenceSpec:
    """描述官方参考包内依赖, 官方命令与设备报告的成员身份."""

    profile_id: str
    report_member_template: str
    command_result_member_template: str
    environment_report_member_template: str
    run_manifest_member_template: str


@dataclass(frozen=True)
class ClosurePackageFamilySpec:
    """描述一个闭合输入 family 的必要成员和身份来源."""

    package_family: str
    filename_pattern: str
    baseline_id: str | None
    allowed_output_prefix_templates: tuple[str, ...]
    allowed_output_member_templates: tuple[str, ...]
    required_member_templates: tuple[str, ...]
    manifest_member_template: str
    manifest_artifact_id_template: str
    generated_at_source: JsonFieldSource
    paper_run_sources: tuple[JsonFieldSource, ...]
    target_fpr_sources: tuple[JsonFieldSource, ...]
    baseline_sources: tuple[JsonFieldSource, ...]
    code_version_sources: tuple[JsonFieldSource, ...]
    randomization_repeat_sources: tuple[JsonFieldSource, ...]
    value_requirements: tuple[JsonValueRequirement, ...]
    baseline_rows_sources: tuple[BaselineRowsSource, ...] = ()
    package_input_manifest_template: str | None = None
    scientific_execution_binding: ScientificExecutionBindingSpec | None = None
    dependency_environment_evidence: DependencyEnvironmentEvidenceSpec | None = None


@dataclass(frozen=True)
class ClosurePackageCandidate:
    """保存一个已通过包内治理校验的闭合输入候选."""

    package_family: str
    package_path: Path
    package_sha256: str
    paper_run_name: str
    target_fpr: float
    code_version: str
    formal_execution_run_lock_digest: str
    formal_execution_package_lock_digest: str
    generated_at: str
    generated_at_utc: datetime
    randomization_scope: str
    randomization_repeat_id: str = ""
    generation_seed_index: int = -1
    generation_seed_offset: int = -1
    watermark_key_index: int = -1
    formal_randomization_protocol_digest: str = ""
    scientific_profile_id: str = ""
    scientific_profile_digest: str = ""
    scientific_direct_requirements_digest: str = ""
    scientific_complete_hash_lock_digest: str = ""
    scientific_complete_hash_lock_dependency_count: int = 0
    scientific_python_executable_digest: str = ""
    scientific_execution_report_digest: str = ""
    scientific_command_dispatch_report_digest: str = ""
    scientific_command_sequence_digest: str = ""
    scientific_execution_binding_digest: str = ""
    scientific_dependency_evidence_digest: str = ""

    def to_lock_record(self) -> dict[str, Any]:
        """转换为持久化锁文件中的稳定记录."""

        return {
            "package_family": self.package_family,
            "package_path": self.package_path.resolve().as_posix(),
            "package_sha256": self.package_sha256,
            "paper_run_name": self.paper_run_name,
            "target_fpr": self.target_fpr,
            "code_version": self.code_version,
            "formal_execution_run_lock_digest": (
                self.formal_execution_run_lock_digest
            ),
            "formal_execution_package_lock_digest": (
                self.formal_execution_package_lock_digest
            ),
            "generated_at": self.generated_at,
            "randomization_scope": self.randomization_scope,
            "randomization_repeat_id": self.randomization_repeat_id,
            "generation_seed_index": self.generation_seed_index,
            "generation_seed_offset": self.generation_seed_offset,
            "watermark_key_index": self.watermark_key_index,
            "formal_randomization_protocol_digest": (
                self.formal_randomization_protocol_digest
            ),
            "scientific_profile_id": self.scientific_profile_id,
            "scientific_profile_digest": self.scientific_profile_digest,
            "scientific_direct_requirements_digest": (
                self.scientific_direct_requirements_digest
            ),
            "scientific_complete_hash_lock_digest": (
                self.scientific_complete_hash_lock_digest
            ),
            "scientific_complete_hash_lock_dependency_count": (
                self.scientific_complete_hash_lock_dependency_count
            ),
            "scientific_python_executable_digest": (
                self.scientific_python_executable_digest
            ),
            "scientific_execution_report_digest": (
                self.scientific_execution_report_digest
            ),
            "scientific_command_dispatch_report_digest": (
                self.scientific_command_dispatch_report_digest
            ),
            "scientific_command_sequence_digest": (
                self.scientific_command_sequence_digest
            ),
            "scientific_execution_binding_digest": (
                self.scientific_execution_binding_digest
            ),
            "scientific_dependency_evidence_digest": (
                self.scientific_dependency_evidence_digest
            ),
        }


def _source(member_template: str, *field_path: str) -> JsonFieldSource:
    """构造字段来源, 减少 family spec 中的重复样板."""

    return JsonFieldSource(member_template=member_template, field_path=tuple(field_path))


def _require(member_template: str, field_name: str, expected_value: Any) -> JsonValueRequirement:
    """构造单层 JSON 字段精确值要求."""

    return JsonValueRequirement(
        source=_source(member_template, field_name),
        expected_value=expected_value,
    )


def _scientific_binding(
    prefix: str,
    *,
    artifact_role: str,
    profile_id: str,
    execution_route: str,
    summary: str,
    manifest: str,
) -> ScientificExecutionBindingSpec:
    """构造一个产物根目录内的标准隔离科学执行绑定契约."""

    return ScientificExecutionBindingSpec(
        artifact_role=artifact_role,
        profile_id=profile_id,
        execution_route=execution_route,
        binding_member_template=prefix + "scientific_execution_binding.json",
        execution_report_member_template=(
            prefix + "isolated_scientific_execution_report.json"
        ),
        dependency_report_member_template=(
            prefix + "isolated_dependency_environment_report.json"
        ),
        dispatch_report_member_template=(
            prefix + "scientific_command_dispatch_report.json"
        ),
        summary_member_template=summary,
        manifest_member_template=manifest,
    )


IMAGE_RUNTIME_PREFIX = "outputs/image_only_dataset_runtime/{paper_run}/"
IMAGE_RUNTIME_SUMMARY = IMAGE_RUNTIME_PREFIX + "dataset_runtime_summary.json"
IMAGE_RUNTIME_MANIFEST = IMAGE_RUNTIME_PREFIX + "manifest.local.json"
IMAGE_RUNTIME_PACKAGE_INPUT = (
    IMAGE_RUNTIME_PREFIX + "image_only_dataset_package_input_manifest.json"
)

ABLATION_PREFIX = "outputs/formal_mechanism_ablation/{paper_run}/"
ABLATION_SUMMARY = ABLATION_PREFIX + "ablation_claim_summary.json"
ABLATION_MANIFEST = ABLATION_PREFIX + "manifest.local.json"
ABLATION_PACKAGE_INPUT = (
    ABLATION_PREFIX + "mechanism_ablation_package_input_manifest.json"
)

QUALITY_PREFIX = "outputs/dataset_level_quality/{paper_run}/"
QUALITY_SUMMARY = QUALITY_PREFIX + "dataset_quality_summary.json"
QUALITY_MANIFEST = QUALITY_PREFIX + "manifest.local.json"
QUALITY_PACKAGE_INPUT = QUALITY_PREFIX + "dataset_quality_package_input_manifest.json"


def _method_faithful_spec(baseline_id: str) -> ClosurePackageFamilySpec:
    """构造一个 common-backbone baseline 的独占结果包契约."""

    run_prefix = (
        "outputs/external_baseline_method_faithful/{paper_run}/run_records/{baseline}/"
    )
    split_prefix = "outputs/external_baseline_method_faithful/{paper_run}/split_observations/"
    summary = run_prefix + "{baseline}_summary.json"
    run_manifest = run_prefix + "{baseline}_manifest.local.json"
    package_record_prefix = run_prefix + "package_records/"
    package_input = package_record_prefix + "{baseline}_package_input_manifest.json"
    archive_manifest = package_record_prefix + "{baseline}_archive_manifest.local.json"
    transfer_manifest = split_prefix + "{baseline}_baseline_transfer_manifest.json"
    scientific_binding = _scientific_binding(
        run_prefix,
        artifact_role="external_baseline_method_faithful",
        profile_id="sd35_method_runtime_gpu",
        execution_route="isolated_method_faithful_workflow",
        summary=summary,
        manifest=run_manifest,
    )
    split_members = (
        split_prefix + "{baseline}_baseline_observations.json",
        split_prefix + "{baseline}_baseline_command_results.json",
        transfer_manifest,
    )
    return ClosurePackageFamilySpec(
        package_family=f"method_faithful_{baseline_id}",
        filename_pattern=f"external_baseline_method_faithful_package_{baseline_id}_*.zip",
        baseline_id=baseline_id,
        allowed_output_prefix_templates=(run_prefix,),
        allowed_output_member_templates=split_members,
        required_member_templates=(
            summary,
            run_manifest,
            package_input,
            package_record_prefix + "{baseline}_archive_summary.json",
            archive_manifest,
            scientific_binding.binding_member_template,
            scientific_binding.execution_report_member_template,
            scientific_binding.dependency_report_member_template,
            scientific_binding.dispatch_report_member_template,
            *split_members,
        ),
        manifest_member_template=archive_manifest,
        manifest_artifact_id_template="{baseline}_method_faithful_archive_manifest",
        generated_at_source=_source(package_input, "generated_at"),
        paper_run_sources=(
            _source(transfer_manifest, "paper_run_name"),
            _source(run_manifest, "config", "prompt_set"),
        ),
        target_fpr_sources=(
            _source(transfer_manifest, "target_fpr"),
            _source(run_manifest, "config", "target_fpr"),
        ),
        baseline_sources=(
            _source(transfer_manifest, "baseline_id"),
            _source(run_manifest, "config", "primary_baseline_id"),
            _source(archive_manifest, "metadata", "baseline_id"),
        ),
        code_version_sources=(
            _source(archive_manifest, "code_version"),
            _source(transfer_manifest, "code_version"),
        ),
        randomization_repeat_sources=(
            _source(package_input, "randomization_repeat_identity"),
            _source(archive_manifest, "config", "randomization_repeat_identity"),
        ),
        value_requirements=(
            _require(summary, "run_decision", "pass"),
            _require(summary, "external_baseline_method_faithful_ready", True),
            _require(summary, "primary_baseline_adapter_ready", True),
            _require(transfer_manifest, "transfer_ready", True),
        ),
        package_input_manifest_template=package_input,
        scientific_execution_binding=scientific_binding,
    )


def _official_reference_spec(baseline_id: str) -> ClosurePackageFamilySpec:
    """构造一个官方参考环境补充参考包的身份契约."""

    prefix = f"outputs/{baseline_id}_official_reference/{{paper_run}}/"
    summary = prefix + f"{baseline_id}_official_reference_summary.json"
    run_manifest = prefix + "manifest.local.json"
    package_input = prefix + f"{baseline_id}_official_reference_package_input_manifest.json"
    archive_manifest = prefix + f"{baseline_id}_official_reference_archive_manifest.local.json"
    records = prefix + f"{baseline_id}_official_reference_records.jsonl"
    validation = prefix + f"{baseline_id}_official_reference_validation_report.json"
    dependency_evidence = DependencyEnvironmentEvidenceSpec(
        profile_id={
            "tree_ring": "tree_ring_official_py39_cu117",
            "gaussian_shading": "gaussian_shading_official_py38_cu117",
            "shallow_diffuse": "shallow_diffuse_official_py39_cu117",
        }[baseline_id],
        report_member_template=(
            prefix + f"{baseline_id}_dependency_environment_prepare_result.json"
        ),
        command_result_member_template=(
            prefix + f"{baseline_id}_official_command_result.json"
        ),
        environment_report_member_template=(
            prefix + f"{baseline_id}_official_reference_environment_report.json"
        ),
        run_manifest_member_template=run_manifest,
    )
    return ClosurePackageFamilySpec(
        package_family=f"official_reference_{baseline_id}",
        filename_pattern=f"external_baseline_official_reference_package_{baseline_id}_*.zip",
        baseline_id=baseline_id,
        allowed_output_prefix_templates=(prefix,),
        allowed_output_member_templates=(),
        required_member_templates=(
            summary,
            run_manifest,
            records,
            validation,
            dependency_evidence.report_member_template,
            dependency_evidence.command_result_member_template,
            dependency_evidence.environment_report_member_template,
            package_input,
            prefix + f"{baseline_id}_official_reference_archive_summary.json",
            archive_manifest,
        ),
        manifest_member_template=archive_manifest,
        manifest_artifact_id_template=f"{baseline_id}_official_reference_archive_manifest",
        generated_at_source=_source(package_input, "generated_at"),
        paper_run_sources=(_source(summary, "paper_claim_scale"),),
        target_fpr_sources=(_source(summary, "target_fpr"),),
        baseline_sources=(_source(summary, "baseline_id"),),
        code_version_sources=(
            _source(archive_manifest, "code_version"),
            _source(run_manifest, "code_version"),
        ),
        randomization_repeat_sources=(),
        value_requirements=(
            _require(summary, "run_decision", "pass"),
            _require(summary, f"{baseline_id}_official_reference_ready", True),
            _require(summary, "reference_import_ready", True),
            _require(summary, "model_source_ready", True),
            _require(summary, "model_snapshot_scope_ready", True),
            _require(summary, "openclip_source_ready", True),
            JsonValueRequirement(
                source=_source(run_manifest, "metadata", "run_decision"),
                expected_value="pass",
            ),
        ),
        baseline_rows_sources=(BaselineRowsSource(records),),
        package_input_manifest_template=package_input,
        dependency_environment_evidence=dependency_evidence,
    )


T2SMARK_PREFIX = "outputs/t2smark_formal_reproduction/{paper_run}/"
T2SMARK_SUMMARY = T2SMARK_PREFIX + "t2smark_formal_reproduction_summary.json"
T2SMARK_RUN_MANIFEST = T2SMARK_PREFIX + "t2smark_formal_reproduction_manifest.local.json"
T2SMARK_PACKAGE_INPUT = T2SMARK_PREFIX + "t2smark_formal_package_input_manifest.json"
T2SMARK_ARCHIVE_MANIFEST = T2SMARK_PREFIX + "t2smark_formal_archive_manifest.local.json"
T2SMARK_CANDIDATES = T2SMARK_PREFIX + "t2smark_formal_import_candidate_records.jsonl"


CLOSURE_PACKAGE_FAMILY_SPECS: tuple[ClosurePackageFamilySpec, ...] = (
    ClosurePackageFamilySpec(
        package_family="image_only_dataset_runtime",
        filename_pattern="image_only_dataset_runtime_package_*.zip",
        baseline_id=None,
        allowed_output_prefix_templates=(IMAGE_RUNTIME_PREFIX,),
        allowed_output_member_templates=(),
        required_member_templates=(
            IMAGE_RUNTIME_PREFIX + "runtime_results.jsonl",
            IMAGE_RUNTIME_PREFIX + "image_only_detection_records.jsonl",
            IMAGE_RUNTIME_PREFIX + "watermark_quality_image_registry.jsonl",
            IMAGE_RUNTIME_PREFIX + "frozen_evidence_protocol.json",
            IMAGE_RUNTIME_PREFIX + "test_detection_metrics.csv",
            IMAGE_RUNTIME_PREFIX + "score_distribution_table.csv",
            IMAGE_RUNTIME_PREFIX + "roc_curve_points.csv",
            IMAGE_RUNTIME_PREFIX + "det_curve_points.csv",
            IMAGE_RUNTIME_SUMMARY,
            IMAGE_RUNTIME_MANIFEST,
            IMAGE_RUNTIME_PACKAGE_INPUT,
            IMAGE_RUNTIME_PREFIX + "scientific_execution_binding.json",
            IMAGE_RUNTIME_PREFIX + "isolated_scientific_execution_report.json",
            IMAGE_RUNTIME_PREFIX + "isolated_dependency_environment_report.json",
            IMAGE_RUNTIME_PREFIX + "scientific_command_dispatch_report.json",
        ),
        manifest_member_template=IMAGE_RUNTIME_MANIFEST,
        manifest_artifact_id_template="{paper_run}_image_only_dataset_runtime_manifest",
        generated_at_source=_source(IMAGE_RUNTIME_SUMMARY, "generated_at"),
        paper_run_sources=(
            _source(IMAGE_RUNTIME_SUMMARY, "paper_run_name"),
            _source(IMAGE_RUNTIME_MANIFEST, "config", "paper_run", "run_name"),
            _source(IMAGE_RUNTIME_PACKAGE_INPUT, "paper_run_name"),
        ),
        target_fpr_sources=(
            _source(IMAGE_RUNTIME_SUMMARY, "target_fpr"),
            _source(IMAGE_RUNTIME_MANIFEST, "config", "paper_run", "target_fpr"),
            _source(IMAGE_RUNTIME_PACKAGE_INPUT, "target_fpr"),
        ),
        baseline_sources=(),
        code_version_sources=(_source(IMAGE_RUNTIME_MANIFEST, "code_version"),),
        randomization_repeat_sources=(
            _source(IMAGE_RUNTIME_PACKAGE_INPUT, "randomization_repeat_identity"),
            _source(IMAGE_RUNTIME_SUMMARY, "randomization_repeat_identity"),
            _source(IMAGE_RUNTIME_MANIFEST, "config", "paper_run"),
        ),
        value_requirements=(
            _require(IMAGE_RUNTIME_SUMMARY, "protocol_decision", "pass"),
            _require(IMAGE_RUNTIME_SUMMARY, "full_method_claim_ready", True),
            _require(
                IMAGE_RUNTIME_SUMMARY,
                "scientific_unit_provenance_ready",
                True,
            ),
            _require(IMAGE_RUNTIME_SUMMARY, "supports_paper_claim", True),
            _require(
                IMAGE_RUNTIME_PACKAGE_INPUT,
                "report_schema",
                "exact_package_input_manifest",
            ),
            _require(IMAGE_RUNTIME_PACKAGE_INPUT, "schema_version", 2),
            _require(
                IMAGE_RUNTIME_PACKAGE_INPUT,
                "package_family",
                "image_only_dataset_runtime",
            ),
            _require(IMAGE_RUNTIME_PACKAGE_INPUT, "decision", "pass"),
        ),
        package_input_manifest_template=IMAGE_RUNTIME_PACKAGE_INPUT,
        scientific_execution_binding=_scientific_binding(
            IMAGE_RUNTIME_PREFIX,
            artifact_role="image_only_dataset_runtime",
            profile_id="sd35_method_runtime_gpu",
            execution_route="semantic_watermark_session",
            summary=IMAGE_RUNTIME_SUMMARY,
            manifest=IMAGE_RUNTIME_MANIFEST,
        ),
    ),
    ClosurePackageFamilySpec(
        package_family="runtime_rerun_ablation",
        filename_pattern="runtime_rerun_ablation_package_*.zip",
        baseline_id=None,
        allowed_output_prefix_templates=(ABLATION_PREFIX,),
        allowed_output_member_templates=(),
        required_member_templates=(
            ABLATION_PREFIX + "runtime_rerun_records.jsonl",
            ABLATION_PREFIX + "formal_detection_records.jsonl",
            ABLATION_PREFIX + "per_ablation_frozen_protocols.json",
            ABLATION_PREFIX + "mechanism_ablation_metrics.csv",
            ABLATION_PREFIX + "mechanism_pairwise_delta.csv",
            ABLATION_PREFIX + "mechanism_necessity_statistics.csv",
            ABLATION_PREFIX + "mechanism_necessity_summary.json",
            ABLATION_SUMMARY,
            ABLATION_MANIFEST,
            ABLATION_PACKAGE_INPUT,
            ABLATION_PREFIX + "scientific_execution_binding.json",
            ABLATION_PREFIX + "isolated_scientific_execution_report.json",
            ABLATION_PREFIX + "isolated_dependency_environment_report.json",
            ABLATION_PREFIX + "scientific_command_dispatch_report.json",
        ),
        manifest_member_template=ABLATION_MANIFEST,
        manifest_artifact_id_template="formal_mechanism_ablation_manifest",
        generated_at_source=_source(ABLATION_SUMMARY, "generated_at"),
        paper_run_sources=(
            _source(ABLATION_SUMMARY, "paper_run_name"),
            _source(ABLATION_PACKAGE_INPUT, "paper_run_name"),
        ),
        target_fpr_sources=(
            _source(ABLATION_SUMMARY, "target_fpr"),
            _source(ABLATION_MANIFEST, "config", "target_fpr"),
            _source(ABLATION_PACKAGE_INPUT, "target_fpr"),
        ),
        baseline_sources=(),
        code_version_sources=(_source(ABLATION_MANIFEST, "code_version"),),
        randomization_repeat_sources=(
            _source(ABLATION_PACKAGE_INPUT, "randomization_repeat_identity"),
            _source(ABLATION_SUMMARY, "randomization_repeat_identity"),
            _source(ABLATION_MANIFEST, "config", "randomization_repeat_identity"),
        ),
        value_requirements=(
            _require(ABLATION_SUMMARY, "protocol_decision", "pass"),
            _require(ABLATION_SUMMARY, "ablation_claim_gate_ready", True),
            _require(
                ABLATION_SUMMARY,
                "ablation_necessity_statistics_ready",
                True,
            ),
            _require(
                ABLATION_SUMMARY,
                "scientific_unit_provenance_ready",
                True,
            ),
            _require(ABLATION_SUMMARY, "supports_paper_claim", True),
            _require(
                ABLATION_PACKAGE_INPUT,
                "report_schema",
                "exact_package_input_manifest",
            ),
            _require(ABLATION_PACKAGE_INPUT, "schema_version", 2),
            _require(
                ABLATION_PACKAGE_INPUT,
                "package_family",
                "runtime_rerun_ablation",
            ),
            _require(ABLATION_PACKAGE_INPUT, "decision", "pass"),
        ),
        package_input_manifest_template=ABLATION_PACKAGE_INPUT,
        scientific_execution_binding=_scientific_binding(
            ABLATION_PREFIX,
            artifact_role="runtime_rerun_ablation",
            profile_id="sd35_method_runtime_gpu",
            execution_route="semantic_watermark_ablation_session",
            summary=ABLATION_SUMMARY,
            manifest=ABLATION_MANIFEST,
        ),
    ),
    ClosurePackageFamilySpec(
        package_family="dataset_level_quality",
        filename_pattern="dataset_level_quality_package_*.zip",
        baseline_id=None,
        allowed_output_prefix_templates=(QUALITY_PREFIX,),
        allowed_output_member_templates=(),
        required_member_templates=(
            QUALITY_PREFIX + "dataset_quality_image_records.jsonl",
            QUALITY_PREFIX + "dataset_quality_image_resolution_records.jsonl",
            QUALITY_PREFIX + "dataset_quality_formal_feature_import_report.json",
            QUALITY_PREFIX + "dataset_quality_formal_feature_records.jsonl",
            QUALITY_PREFIX + "dataset_quality_metrics.csv",
            QUALITY_SUMMARY,
            QUALITY_MANIFEST,
            QUALITY_PACKAGE_INPUT,
            QUALITY_PREFIX + "scientific_execution_binding.json",
            QUALITY_PREFIX + "isolated_scientific_execution_report.json",
            QUALITY_PREFIX + "isolated_dependency_environment_report.json",
            QUALITY_PREFIX + "scientific_command_dispatch_report.json",
        ),
        manifest_member_template=QUALITY_MANIFEST,
        manifest_artifact_id_template="dataset_level_quality_manifest",
        generated_at_source=_source(QUALITY_SUMMARY, "generated_at"),
        paper_run_sources=(
            _source(QUALITY_SUMMARY, "paper_run_name"),
            _source(QUALITY_MANIFEST, "metadata", "paper_run_name"),
            _source(QUALITY_PACKAGE_INPUT, "paper_run_name"),
        ),
        target_fpr_sources=(
            _source(QUALITY_SUMMARY, "target_fpr"),
            _source(QUALITY_MANIFEST, "metadata", "target_fpr"),
            _source(QUALITY_PACKAGE_INPUT, "target_fpr"),
        ),
        baseline_sources=(),
        code_version_sources=(_source(QUALITY_MANIFEST, "code_version"),),
        randomization_repeat_sources=(
            _source(QUALITY_PACKAGE_INPUT, "randomization_repeat_identity"),
            _source(QUALITY_SUMMARY, "randomization_repeat_identity"),
            _source(QUALITY_MANIFEST, "config", "randomization_repeat_identity"),
        ),
        value_requirements=(
            _require(QUALITY_SUMMARY, "formal_feature_backend_ready", True),
            _require(QUALITY_SUMMARY, "formal_sample_scale_ready", True),
            _require(QUALITY_SUMMARY, "canonical_formal_feature_extractor_ready", True),
            _require(
                QUALITY_SUMMARY,
                "scientific_unit_provenance_ready",
                True,
            ),
            _require(
                QUALITY_SUMMARY,
                "scientific_unit_provenance_identity_ready",
                True,
            ),
            _require(QUALITY_SUMMARY, "formal_fid_kid_claim_gate_ready", True),
            _require(
                QUALITY_PACKAGE_INPUT,
                "report_schema",
                "exact_package_input_manifest",
            ),
            _require(QUALITY_PACKAGE_INPUT, "schema_version", 2),
            _require(
                QUALITY_PACKAGE_INPUT,
                "package_family",
                "dataset_level_quality",
            ),
            _require(QUALITY_PACKAGE_INPUT, "decision", "pass"),
        ),
        package_input_manifest_template=QUALITY_PACKAGE_INPUT,
        scientific_execution_binding=_scientific_binding(
            QUALITY_PREFIX,
            artifact_role="dataset_level_quality",
            profile_id="sd35_method_runtime_gpu",
            execution_route="semantic_watermark_session",
            summary=QUALITY_SUMMARY,
            manifest=QUALITY_MANIFEST,
        ),
    ),
    _method_faithful_spec("tree_ring"),
    _method_faithful_spec("gaussian_shading"),
    _method_faithful_spec("shallow_diffuse"),
    _official_reference_spec("tree_ring"),
    _official_reference_spec("gaussian_shading"),
    _official_reference_spec("shallow_diffuse"),
    ClosurePackageFamilySpec(
        package_family="official_reference_t2smark",
        filename_pattern="external_baseline_official_reference_package_t2smark_*.zip",
        baseline_id="t2smark",
        allowed_output_prefix_templates=(T2SMARK_PREFIX,),
        allowed_output_member_templates=(),
        required_member_templates=(
            T2SMARK_SUMMARY,
            T2SMARK_RUN_MANIFEST,
            T2SMARK_PREFIX + "t2smark_formal_import_validation_report.json",
            T2SMARK_CANDIDATES,
            T2SMARK_PREFIX + "t2smark_formal_strict_pair_quality_summary.json",
            T2SMARK_PREFIX + "t2smark_adapter/baseline_observations.json",
            T2SMARK_PACKAGE_INPUT,
            T2SMARK_PREFIX + "t2smark_formal_archive_summary.json",
            T2SMARK_ARCHIVE_MANIFEST,
            T2SMARK_PREFIX + "scientific_execution_binding.json",
            T2SMARK_PREFIX + "isolated_scientific_execution_report.json",
            T2SMARK_PREFIX + "isolated_dependency_environment_report.json",
            T2SMARK_PREFIX + "scientific_command_dispatch_report.json",
        ),
        manifest_member_template=T2SMARK_ARCHIVE_MANIFEST,
        manifest_artifact_id_template="t2smark_formal_archive_manifest",
        generated_at_source=_source(T2SMARK_PACKAGE_INPUT, "generated_at"),
        paper_run_sources=(
            _source(T2SMARK_SUMMARY, "paper_claim_scale"),
            _source(T2SMARK_RUN_MANIFEST, "config", "prompt_set"),
        ),
        target_fpr_sources=(
            _source(T2SMARK_SUMMARY, "target_fpr"),
            _source(T2SMARK_RUN_MANIFEST, "config", "target_fpr"),
        ),
        baseline_sources=(_source(T2SMARK_SUMMARY, "baseline_id"),),
        code_version_sources=(
            _source(T2SMARK_ARCHIVE_MANIFEST, "code_version"),
            _source(T2SMARK_RUN_MANIFEST, "code_version"),
        ),
        randomization_repeat_sources=(
            _source(T2SMARK_PACKAGE_INPUT, "randomization_repeat_identity"),
            _source(
                T2SMARK_ARCHIVE_MANIFEST,
                "config",
                "randomization_repeat_identity",
            ),
        ),
        value_requirements=(
            _require(T2SMARK_SUMMARY, "run_decision", "pass"),
            _require(T2SMARK_SUMMARY, "t2smark_formal_reproduction_ready", True),
            _require(T2SMARK_SUMMARY, "formal_import_validation_ready", True),
            _require(T2SMARK_SUMMARY, "t2smark_formal_attack_ready", True),
            _require(T2SMARK_SUMMARY, "t2smark_strict_pair_quality_ready", True),
        ),
        baseline_rows_sources=(BaselineRowsSource(T2SMARK_CANDIDATES),),
        package_input_manifest_template=T2SMARK_PACKAGE_INPUT,
        scientific_execution_binding=_scientific_binding(
            T2SMARK_PREFIX,
            artifact_role="t2smark_formal_reproduction",
            profile_id="t2smark_sd35_gpu",
            execution_route="isolated_t2smark_workflow",
            summary=T2SMARK_SUMMARY,
            manifest=T2SMARK_RUN_MANIFEST,
        ),
    ),
)


def _format_template(template: str, spec: ClosurePackageFamilySpec, paper_run_name: str) -> str:
    """把论文层级和 baseline 身份填入成员路径模板."""

    return template.format(
        paper_run=paper_run_name,
        baseline=spec.baseline_id or "",
    )


def _field_value(payload: Any, field_path: tuple[str, ...], *, member_name: str) -> Any:
    """从 JSON object 中读取嵌套字段, 缺失时给出稳定诊断."""

    current = payload
    for field_name in field_path:
        if not isinstance(current, dict) or field_name not in current:
            dotted_path = ".".join(field_path)
            raise ClosurePackageSelectionError(f"{member_name} 缺少治理字段 {dotted_path}")
        current = current[field_name]
    return current


def _read_json_object(
    archive: ZipFile,
    member_name: str,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """读取大小受限的 JSON object, 并在单个候选内复用解析结果."""

    if member_name in cache:
        return cache[member_name]
    try:
        info = archive.getinfo(member_name)
    except KeyError as error:
        raise ClosurePackageSelectionError(f"结果包缺少治理成员 {member_name}") from error
    if info.file_size <= 0 or info.file_size > MAX_GOVERNANCE_MEMBER_BYTES:
        raise ClosurePackageSelectionError(f"治理成员大小非法: {member_name}")
    try:
        payload = json.loads(archive.read(member_name).decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ClosurePackageSelectionError(f"治理成员不是有效 JSON: {member_name}") from error
    if not isinstance(payload, dict):
        raise ClosurePackageSelectionError(f"治理成员必须是 JSON object: {member_name}")
    cache[member_name] = payload
    return payload


def _read_source_value(
    archive: ZipFile,
    source: JsonFieldSource,
    spec: ClosurePackageFamilySpec,
    paper_run_name: str,
    cache: dict[str, dict[str, Any]],
) -> Any:
    """读取一个已经按 family 模板展开的 JSON 字段."""

    member_name = _format_template(source.member_template, spec, paper_run_name)
    payload = _read_json_object(archive, member_name, cache)
    return _field_value(payload, source.field_path, member_name=member_name)


def _validated_generated_at(value: Any) -> tuple[str, datetime]:
    """要求 generated_at 为带时区的 ISO-8601 时间."""

    if not isinstance(value, str) or not value.strip():
        raise ClosurePackageSelectionError("generated_at 必须是非空 ISO-8601 字符串")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ClosurePackageSelectionError("generated_at 不是有效 ISO-8601 时间") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ClosurePackageSelectionError("generated_at 必须携带时区")
    return value.strip(), parsed.astimezone(timezone.utc)


def normalize_clean_code_version(value: Any) -> str:
    """要求受治理代码版本是精确40位小写 Git 提交 SHA.

    正式结果包不再兼容短 SHA, 大小写转换, ``-dirty``, 不可用降级值或
    任意自由文本, 避免不同提交前缀被聚合为同一次论文证据闭合.
    """

    try:
        return normalize_formal_git_commit(value)
    except FormalExecutionLockError as exc:
        raise ClosurePackageSelectionError(
            "code_version 必须是精确40位小写 clean Git 提交 SHA"
        ) from exc


def _validate_package_execution_locks(
    manifest: dict[str, Any],
    *,
    package_family: str,
    code_version: str,
) -> tuple[str, str]:
    """逐个复验运行锁与打包锁, 并绑定包内完整代码提交身份.

    两个锁分别证明正式任务启动边界和 ZIP 打包边界的仓库状态. 此处调用
    统一 validator 严格复验 schema, 摘要和布尔门禁, 再要求两个锁的完整
    commit 与所有 ``code_version_sources`` 汇总出的提交完全一致.
    """

    validated_locks: dict[str, dict[str, Any]] = {}
    for field_name in (
        "formal_execution_run_lock",
        "formal_execution_package_lock",
    ):
        if field_name not in manifest:
            raise ClosurePackageSelectionError(
                f"{package_family} 的 manifest 缺少 {field_name}"
            )
        try:
            validated_lock = validate_formal_execution_lock_record(
                manifest[field_name]
            )
        except FormalExecutionLockError as exc:
            raise ClosurePackageSelectionError(
                f"{package_family} 的 {field_name} 未通过严格复验"
            ) from exc
        if not isinstance(validated_lock, dict):
            raise ClosurePackageSelectionError(
                f"{package_family} 的 {field_name} validator 返回类型非法"
            )
        validated_locks[field_name] = validated_lock

    run_lock = validated_locks["formal_execution_run_lock"]
    package_lock = validated_locks["formal_execution_package_lock"]
    run_commit = normalize_clean_code_version(run_lock.get("formal_execution_commit"))
    package_commit = normalize_clean_code_version(
        package_lock.get("formal_execution_commit")
    )
    if run_commit != package_commit:
        raise ClosurePackageSelectionError(
            f"{package_family} 的运行锁与打包锁 commit 不一致"
        )
    if run_commit != code_version:
        raise ClosurePackageSelectionError(
            f"{package_family} 的执行锁 commit 与 code_version 来源不一致"
        )
    return (
        str(run_lock["formal_execution_lock_digest"]),
        str(package_lock["formal_execution_lock_digest"]),
    )


def _archive_member_sha256(archive: ZipFile, member_name: str) -> str:
    """流式计算 ZIP 成员摘要, 支持图像等大型动态证据成员."""

    digest = hashlib.sha256()
    with archive.open(member_name) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_file_digest(payload: Mapping[str, Any]) -> str:
    """按 repository JSON 报告排版重建文件级 SHA-256."""

    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_sha256(value: Any, *, field_name: str) -> str:
    """要求治理摘要是规范的小写 SHA-256."""

    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ClosurePackageSelectionError(f"科学执行证据摘要非法: {field_name}")
    return value


def _validate_binding_formal_lock(
    value: Any,
    *,
    package_family: str,
    code_version: str,
) -> dict[str, Any]:
    """复验科学执行绑定携带的正式执行锁."""

    try:
        validated = validate_formal_execution_lock_record(value)
    except FormalExecutionLockError as exc:
        raise ClosurePackageSelectionError(
            f"{package_family} 的科学执行绑定锁未通过严格复验"
        ) from exc
    if validated.get("formal_execution_commit") != code_version:
        raise ClosurePackageSelectionError(
            f"{package_family} 的科学执行绑定锁与 code_version 不一致"
        )
    return validated


def _validate_scientific_command_identity(
    execution_report: Mapping[str, Any],
    *,
    contract: ScientificExecutionBindingSpec,
    spec: ClosurePackageFamilySpec,
    paper_run_name: str,
) -> None:
    """复验科学解释器实际执行的 argv、route 与上下文环境."""

    child_tail = execution_report.get("child_argv_tail")
    execution = execution_report.get("execution")
    if not isinstance(child_tail, list) or not isinstance(execution, dict):
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的科学命令身份结构无效"
        )
    try:
        context = validate_scientific_command_context_snapshot(
            execution_report,
            expected_profile_id=contract.profile_id,
        )
    except RuntimeError as exc:
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的科学命令上下文身份无效"
        ) from exc

    if contract.execution_route in {
        "semantic_watermark_session",
        "semantic_watermark_ablation_session",
    }:
        valid_tails = (
            [
                "-m",
                "experiments.runtime.semantic_watermark_scientific_session",
            ],
            [
                "-m",
                "experiments.runtime.semantic_watermark_scientific_session",
                "--run-formal-ablation",
            ],
        )
        if contract.execution_route == "semantic_watermark_ablation_session":
            valid_tails = (valid_tails[1],)
        if child_tail not in valid_tails:
            raise ClosurePackageSelectionError(
                f"{spec.package_family} 的主方法科学 route 不匹配"
            )
        return

    expected_workflow = (
        "external_baseline_method_faithful"
        if contract.execution_route == "isolated_method_faithful_workflow"
        else "official_reference_t2smark"
    )
    expected_prefix = (
        "outputs/external_baseline_method_faithful/"
        f"{paper_run_name}/run_records/{spec.baseline_id}/scientific_execution/"
        if expected_workflow == "external_baseline_method_faithful"
        else f"outputs/t2smark_formal_reproduction/{paper_run_name}/scientific_execution/"
    )
    expected_envelope_suffix = (
        expected_prefix + "scientific_workflow_result_envelope.json"
    )
    route_ready = (
        len(child_tail) == 8
        and child_tail[:5]
        == [
            "-m",
            "paper_experiments.runners.isolated_scientific_workflow",
            "--child-workflow",
            expected_workflow,
            "--root",
        ]
        and normalize_scientific_absolute_path(child_tail[5])
        == context["repository_root"]
        and child_tail[6] == "--result-envelope"
        and normalize_scientific_absolute_path(child_tail[7])
        == context["repository_root"] + "/" + expected_envelope_suffix
    )
    if not route_ready:
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的隔离 workflow route 不匹配"
        )


def _validate_scientific_execution_binding(
    archive: ZipFile,
    *,
    spec: ClosurePackageFamilySpec,
    paper_run_name: str,
    code_version: str,
    package_manifest: dict[str, Any],
    cache: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """从 ZIP 内离线复验科学产物、解释器环境与完整锁的绑定关系."""

    contract = spec.scientific_execution_binding
    if contract is None:
        return {
            "scientific_profile_id": "",
            "scientific_profile_digest": "",
            "scientific_direct_requirements_digest": "",
            "scientific_complete_hash_lock_digest": "",
            "scientific_complete_hash_lock_dependency_count": 0,
            "scientific_python_executable_digest": "",
            "scientific_execution_report_digest": "",
            "scientific_command_dispatch_report_digest": "",
            "scientific_command_sequence_digest": "",
            "scientific_execution_binding_digest": "",
            "scientific_dependency_evidence_digest": "",
        }

    member_names = {
        "binding": _format_template(
            contract.binding_member_template, spec, paper_run_name
        ),
        "execution": _format_template(
            contract.execution_report_member_template, spec, paper_run_name
        ),
        "dependency": _format_template(
            contract.dependency_report_member_template, spec, paper_run_name
        ),
        "dispatch": _format_template(
            contract.dispatch_report_member_template, spec, paper_run_name
        ),
        "summary": _format_template(
            contract.summary_member_template, spec, paper_run_name
        ),
        "manifest": _format_template(
            contract.manifest_member_template, spec, paper_run_name
        ),
    }
    binding = _read_json_object(archive, member_names["binding"], cache)
    expected_binding_values = {
        "report_schema": "scientific_execution_binding",
        "schema_version": 2,
        "artifact_role": contract.artifact_role,
        "paper_run_name": paper_run_name,
        "profile_id": contract.profile_id,
        "decision": "pass",
        "supports_paper_claim": False,
    }
    for field_name, expected_value in expected_binding_values.items():
        if binding.get(field_name) != expected_value:
            raise ClosurePackageSelectionError(
                f"{spec.package_family} 的科学执行绑定字段不一致: {field_name}"
            )

    path_digest_contracts = (
        (
            "scientific_execution_report_path",
            "scientific_execution_report_digest",
            "execution",
        ),
        (
            "dependency_environment_report_path",
            "dependency_environment_report_digest",
            "dependency",
        ),
        (
            "scientific_command_dispatch_report_path",
            "scientific_command_dispatch_report_digest",
            "dispatch",
        ),
        ("bound_summary_path", "bound_summary_digest", "summary"),
    )
    for path_field, digest_field, member_role in path_digest_contracts:
        member_name = member_names[member_role]
        if binding.get(path_field) != member_name:
            raise ClosurePackageSelectionError(
                f"{spec.package_family} 的科学执行绑定路径不一致: {path_field}"
            )
        declared_digest = _normalized_sha256(
            binding.get(digest_field),
            field_name=digest_field,
        )
        if _archive_member_sha256(archive, member_name) != declared_digest:
            raise ClosurePackageSelectionError(
                f"{spec.package_family} 的科学执行绑定摘要不匹配: {digest_field}"
            )
    if binding.get("bound_manifest_path") != member_names["manifest"]:
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的科学 manifest 绑定路径不一致"
        )
    bound_manifest = _read_json_object(
        archive,
        member_names["manifest"],
        cache,
    )
    if (
        binding.get("bound_manifest_digest_scope")
        != BOUND_MANIFEST_DIGEST_SCOPE
        or binding.get("bound_manifest_scientific_digest")
        != scientific_manifest_payload_digest(bound_manifest)
    ):
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的科学 manifest 绑定摘要不匹配"
        )

    profile_digest = _normalized_sha256(
        binding.get("profile_digest"), field_name="profile_digest"
    )
    direct_requirements_digest = _normalized_sha256(
        binding.get("direct_requirements_digest"),
        field_name="direct_requirements_digest",
    )
    complete_hash_lock_digest = _normalized_sha256(
        binding.get("complete_hash_lock_digest"),
        field_name="complete_hash_lock_digest",
    )
    binding_lock = _validate_binding_formal_lock(
        binding.get("formal_execution_lock"),
        package_family=spec.package_family,
        code_version=code_version,
    )
    if (
        binding.get("formal_execution_commit")
        != binding_lock.get("formal_execution_commit")
        or binding.get("formal_execution_lock_digest")
        != binding_lock.get("formal_execution_lock_digest")
        or binding_lock != package_manifest.get("formal_execution_run_lock")
    ):
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的科学执行绑定锁与运行 manifest 不一致"
        )
    bound_manifest_lock = _validate_binding_formal_lock(
        bound_manifest.get("formal_execution_run_lock"),
        package_family=spec.package_family,
        code_version=code_version,
    )
    if (
        bound_manifest_lock != binding_lock
        or normalize_clean_code_version(bound_manifest.get("code_version"))
        != code_version
    ):
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的科学 manifest 锁或 code_version 不一致"
        )

    execution = _read_json_object(archive, member_names["execution"], cache)
    required_execution_true_fields = (
        "execution_completed",
        "dependency_environment_report_valid",
        "python_executable_revalidated_before_child",
        "python_executable_revalidated_after_child",
        "dependency_environment_report_revalidated_before_child",
        "dependency_environment_report_revalidated_after_child",
        "formal_execution_lock_ready",
        "formal_execution_lock_revalidated_before_child",
        "formal_execution_lock_revalidated_after_child",
    )
    execution_result = execution.get("execution")
    if not all(
        (
            execution.get("report_schema")
            == "isolated_scientific_execution_report",
            execution.get("schema_version") == 1,
            execution.get("operation_kind") == "isolated_scientific_execution",
            execution.get("profile_id") == contract.profile_id,
            execution.get("profile_digest") == profile_digest,
            execution.get("direct_requirements_digest")
            == direct_requirements_digest,
            execution.get("complete_hash_lock_digest")
            == complete_hash_lock_digest,
            int(execution.get("complete_hash_lock_dependency_count", 0)) > 0,
            execution.get("dependency_environment_report_path")
            == PurePosixPath(member_names["dependency"]).name,
            execution.get("dependency_environment_report_digest")
            == binding.get("dependency_environment_report_digest"),
            execution.get("execution_report_path")
            == PurePosixPath(member_names["execution"]).name,
            execution.get("decision") == "pass",
            execution.get("failure_reasons") == [],
            execution.get("supports_paper_claim") is False,
            all(
                execution.get(field_name) is True
                for field_name in required_execution_true_fields
            ),
            isinstance(execution_result, dict),
            execution_result.get("attempted") is True
            if isinstance(execution_result, dict)
            else False,
            execution_result.get("return_code") == 0
            if isinstance(execution_result, dict)
            else False,
        )
    ):
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的隔离科学执行报告未通过离线复验"
        )
    _normalized_sha256(
        execution.get("python_executable_sha256"),
        field_name="python_executable_sha256",
    )
    execution_lock = _validate_binding_formal_lock(
        execution.get("formal_execution_lock"),
        package_family=spec.package_family,
        code_version=code_version,
    )
    if (
        execution_lock != binding_lock
        or execution.get("formal_execution_commit")
        != binding_lock.get("formal_execution_commit")
        or execution.get("formal_execution_lock_digest")
        != binding_lock.get("formal_execution_lock_digest")
    ):
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的科学执行报告锁与绑定不一致"
        )
    _validate_scientific_command_identity(
        execution,
        contract=contract,
        spec=spec,
        paper_run_name=paper_run_name,
    )

    dependency = _read_json_object(archive, member_names["dependency"], cache)
    if not all(
        (
            dependency.get("report_schema")
            == "isolated_dependency_environment_preparation_report",
            dependency.get("schema_version") == 1,
            dependency.get("operation_kind")
            == "formal_dependency_environment_preparation",
            dependency.get("profile_id") == contract.profile_id,
            dependency.get("profile_digest") == profile_digest,
            dependency.get("direct_requirements_digest")
            == direct_requirements_digest,
            dependency.get("complete_hash_lock_digest")
            == complete_hash_lock_digest,
            int(dependency.get("complete_hash_lock_dependency_count", 0)) > 0,
            dependency.get("formal_preparation_completed") is True,
            dependency.get("formal_ready") is True,
            dependency.get("formal_execution_lock_ready") is True,
            dependency.get("decision") == "pass",
            dependency.get("failure_reasons") == [],
            dependency.get("supports_paper_claim") is False,
            dependency.get("formal_execution_lock") == binding_lock,
            dependency.get("formal_execution_commit")
            == binding_lock.get("formal_execution_commit"),
            dependency.get("formal_execution_lock_digest")
            == binding_lock.get("formal_execution_lock_digest"),
        )
    ):
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的隔离依赖环境报告未通过离线复验"
        )
    dependency_python_digest = _normalized_sha256(
        dependency.get("python_executable_sha256"),
        field_name="dependency_python_executable_sha256",
    )
    if (
        dependency.get("python_executable_sha256_after_preparation")
        != dependency_python_digest
        or execution.get("python_executable_sha256") != dependency_python_digest
    ):
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的隔离解释器摘要证据不一致"
        )
    try:
        validate_dependency_environment_report_snapshot(
            dependency,
            expected_profile_id=contract.profile_id,
            expected_profile_digest=profile_digest,
            expected_direct_requirements_digest=direct_requirements_digest,
            expected_complete_hash_lock_digest=complete_hash_lock_digest,
            expected_complete_hash_lock_dependency_count=int(
                execution["complete_hash_lock_dependency_count"]
            ),
            expected_python_executable_path=execution[
                "python_executable_path"
            ],
            expected_python_executable_digest=dependency_python_digest,
            expected_formal_execution_lock=binding_lock,
            expected_working_directory=execution_result["working_directory"],
        )
    except RuntimeError as exc:
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的隔离依赖嵌套证据未通过复验"
        ) from exc

    dispatch = _read_json_object(archive, member_names["dispatch"], cache)
    if not all(
        (
            dispatch.get("report_schema")
            == "scientific_command_dispatch_report",
            dispatch.get("schema_version") == 1,
            dispatch.get("paper_run_name") == paper_run_name,
            dispatch.get("decision") == "pass",
            dispatch.get("failure_reasons", []) == [],
            dispatch.get("supports_paper_claim") is False,
        )
    ):
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的科学命令调度报告未通过离线复验"
        )
    if contract.execution_route in {
        "semantic_watermark_session",
        "semantic_watermark_ablation_session",
    }:
        try:
            validate_semantic_watermark_dispatch_report(
                dispatch,
                execution,
            )
            validate_semantic_watermark_dispatch_artifact_snapshot(
                dispatch,
                artifact_role=contract.artifact_role,
                summary_path=member_names["summary"],
                summary_sha256=_archive_member_sha256(
                    archive,
                    member_names["summary"],
                ),
                manifest_path=member_names["manifest"],
                manifest_payload=bound_manifest,
                manifest_sha256=None,
                formal_execution_lock=binding_lock,
            )
        except RuntimeError as exc:
            raise ClosurePackageSelectionError(
                f"{spec.package_family} 的主方法逐命令证据未闭合"
            ) from exc
    else:
        expected_workflow = (
            "external_baseline_method_faithful"
            if contract.execution_route == "isolated_method_faithful_workflow"
            else "official_reference_t2smark"
        )
        if not all(
            (
                dispatch.get("operation_kind")
                == "isolated_scientific_workflow",
                dispatch.get("workflow_name") == expected_workflow,
                dispatch.get("profile_id") == contract.profile_id,
                dispatch.get("formal_execution_lock") == binding_lock,
                dispatch.get("formal_execution_commit")
                == binding_lock.get("formal_execution_commit"),
                dispatch.get("formal_execution_lock_digest")
                == binding_lock.get("formal_execution_lock_digest"),
                dispatch.get("summary_path") == member_names["summary"],
                dispatch.get("summary_sha256")
                == _archive_member_sha256(archive, member_names["summary"]),
                dispatch.get("manifest_path") == member_names["manifest"],
                dispatch.get("manifest_sha256")
                == _archive_member_sha256(archive, member_names["manifest"]),
                dispatch.get("dependency_environment_report_digest")
                == binding.get("dependency_environment_report_digest"),
            )
        ):
            raise ClosurePackageSelectionError(
                f"{spec.package_family} 的隔离 workflow 调度证据未闭合"
            )

    return {
        "scientific_profile_id": contract.profile_id,
        "scientific_profile_digest": profile_digest,
        "scientific_direct_requirements_digest": direct_requirements_digest,
        "scientific_complete_hash_lock_digest": complete_hash_lock_digest,
        "scientific_complete_hash_lock_dependency_count": int(
            dependency["complete_hash_lock_dependency_count"]
        ),
        "scientific_python_executable_digest": dependency_python_digest,
        "scientific_execution_report_digest": _archive_member_sha256(
            archive, member_names["execution"]
        ),
        "scientific_command_dispatch_report_digest": _archive_member_sha256(
            archive, member_names["dispatch"]
        ),
        "scientific_command_sequence_digest": (
            semantic_command_sequence_digest(dispatch)
            if contract.execution_route
            in {
                "semantic_watermark_session",
                "semantic_watermark_ablation_session",
            }
            else ""
        ),
        "scientific_execution_binding_digest": _archive_member_sha256(
            archive, member_names["binding"]
        ),
        "scientific_dependency_evidence_digest": _archive_member_sha256(
            archive, member_names["dependency"]
        ),
    }


def _official_config_text(config: Mapping[str, Any], field_name: str) -> str:
    """读取官方命令使用的必需标量配置."""

    if field_name not in config:
        raise ClosurePackageSelectionError(
            f"官方参考运行 manifest 缺少命令配置: {field_name}"
        )
    value = config[field_name]
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ClosurePackageSelectionError(
            f"官方参考运行 manifest 命令配置类型无效: {field_name}"
        )
    text = str(value)
    if not text:
        raise ClosurePackageSelectionError(
            f"官方参考运行 manifest 命令配置为空: {field_name}"
        )
    return text


def _official_config_int(config: Mapping[str, Any], field_name: str) -> int:
    """读取官方命令使用的必需整数配置."""

    text = _official_config_text(config, field_name)
    try:
        return int(text)
    except ValueError as exc:
        raise ClosurePackageSelectionError(
            f"官方参考运行 manifest 整数配置无效: {field_name}"
        ) from exc


def _repository_child_absolute_path(repository_root: str, relative_path: Any) -> str:
    """把受治理仓库相对路径转换为跨主机可比较的绝对路径."""

    text = str(relative_path or "")
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != text
    ):
        raise ClosurePackageSelectionError("官方命令仓库路径必须是规范 POSIX 相对路径")
    return repository_root.rstrip("/") + "/" + path.as_posix()


def _expected_official_reference_command(
    *,
    baseline_id: str,
    config: Mapping[str, Any],
    repository_root: str,
    paper_run_name: str,
    python_executable: str,
) -> tuple[list[str], frozenset[int], str]:
    """从运行 manifest 重建官方入口的精确 argv 与 cwd 契约."""

    expected_source_dirs = {
        "tree_ring": "external_baseline/primary/tree_ring/source",
        "gaussian_shading": "external_baseline/primary/gaussian_shading/source",
        "shallow_diffuse": "external_baseline/primary/shallow_diffuse/source",
    }
    expected_output_dirs = {
        "tree_ring": "outputs/tree_ring_official_reference",
        "gaussian_shading": "outputs/gaussian_shading_official_reference",
        "shallow_diffuse": "outputs/shallow_diffuse_official_reference",
    }
    entrypoint_names = {
        "tree_ring": "run_tree_ring_watermark.py",
        "gaussian_shading": "run_gaussian_shading.py",
        "shallow_diffuse": "run_shallow_diffuse_t2i.py",
    }
    if baseline_id not in expected_source_dirs:
        raise ClosurePackageSelectionError("官方参考 baseline 身份无效")
    source_dir_value = _official_config_text(config, "source_dir")
    output_dir_value = _official_config_text(config, "output_dir")
    if (
        source_dir_value != expected_source_dirs[baseline_id]
        or output_dir_value != expected_output_dirs[baseline_id]
    ):
        raise ClosurePackageSelectionError("官方参考命令 source 或 output family 不一致")
    if config.get("dependency_profile_id") is None or config.get("require_cuda") is not True:
        raise ClosurePackageSelectionError("官方参考命令必须绑定固定 profile 并要求 CUDA")
    source_dir = _repository_child_absolute_path(repository_root, source_dir_value)
    entrypoint = source_dir + "/" + entrypoint_names[baseline_id]
    sample_count = max(1, _official_config_int(config, "sample_count"))

    if baseline_id == "tree_ring":
        start_index = _official_config_int(config, "start_index")
        command = [
            python_executable,
            entrypoint,
            "--run_name",
            _official_config_text(config, "run_name"),
            "--model_id",
            _official_config_text(config, "official_model_id"),
            "--dataset",
            _official_config_text(config, "dataset"),
            "--w_channel",
            "3",
            "--w_pattern",
            "ring",
            "--start",
            str(start_index),
            "--end",
            str(start_index + sample_count),
            "--reference_model",
            _official_config_text(config, "reference_model"),
            "--reference_model_pretrain",
            _official_config_text(config, "reference_model_checkpoint_path"),
            "--with_tracking",
        ]
        return command, frozenset({0, 1}), source_dir

    if baseline_id == "gaussian_shading":
        official_output_subdir = _official_config_text(
            config,
            "official_output_subdir",
        )
        output_dir = _repository_child_absolute_path(
            repository_root,
            output_dir_value + "/" + paper_run_name + "/" + official_output_subdir,
        )
        command = [
            python_executable,
            entrypoint,
            "--num",
            str(sample_count),
            "--fpr",
            _official_config_text(config, "fpr"),
            "--channel_copy",
            _official_config_text(config, "channel_copy"),
            "--hw_copy",
            _official_config_text(config, "hw_copy"),
            "--user_number",
            _official_config_text(config, "user_number"),
            "--gen_seed",
            _official_config_text(config, "gen_seed"),
            "--image_length",
            _official_config_text(config, "image_length"),
            "--guidance_scale",
            _official_config_text(config, "guidance_scale"),
            "--num_inference_steps",
            _official_config_text(config, "num_inference_steps"),
            "--num_inversion_steps",
            _official_config_text(config, "num_inversion_steps"),
            "--dataset_path",
            _official_config_text(config, "dataset_path"),
            "--model_path",
            _official_config_text(config, "official_model_id"),
            "--output_path",
            output_dir,
        ]
        use_chacha = config.get("use_chacha")
        if not isinstance(use_chacha, bool):
            raise ClosurePackageSelectionError("Gaussian Shading chacha 配置类型无效")
        if use_chacha:
            command.append("--chacha")
        command.extend(
            [
                "--reference_model",
                _official_config_text(config, "reference_model"),
                "--reference_model_pretrain",
                _official_config_text(config, "reference_model_checkpoint_path"),
            ]
        )
        output_value_index = command.index("--output_path") + 1
        return command, frozenset({0, 1, output_value_index}), source_dir

    start_index = _official_config_int(config, "start_index")
    command = [
        python_executable,
        entrypoint,
        "--run_name",
        _official_config_text(config, "run_name"),
        "--model_id",
        _official_config_text(config, "official_model_id"),
        "--dataset",
        _official_config_text(config, "dataset"),
        "--image_length",
        _official_config_text(config, "image_length"),
        "--guidance_scale",
        _official_config_text(config, "guidance_scale"),
        "--num_inference_steps",
        _official_config_text(config, "num_inference_steps"),
        "--w_seed",
        _official_config_text(config, "w_seed"),
        "--w_channel",
        _official_config_text(config, "w_channel"),
        "--w_pattern",
        _official_config_text(config, "w_pattern"),
        "--w_mask_shape",
        _official_config_text(config, "w_mask_shape"),
        "--w_radius",
        _official_config_text(config, "w_radius"),
        "--w_measurement",
        _official_config_text(config, "w_measurement"),
        "--w_injection",
        _official_config_text(config, "w_injection"),
        "--reference_model",
        _official_config_text(config, "reference_model"),
        "--reference_model_pretrain",
        _official_config_text(config, "reference_model_checkpoint_path"),
        "--edit_time_list",
        _official_config_text(config, "edit_time_list"),
        "--start",
        str(start_index),
        "--end",
        str(start_index + sample_count),
    ]
    working_directory = _repository_child_absolute_path(
        repository_root,
        output_dir_value + "/" + paper_run_name,
    )
    return command, frozenset({0, 1}), working_directory


def _official_command_matches(
    observed: Any,
    expected: list[str],
    *,
    path_indices: frozenset[int],
) -> bool:
    """逐 token 比较 argv, 仅对绝对路径执行跨平台规范化."""

    if (
        not isinstance(observed, list)
        or len(observed) != len(expected)
        or any(not isinstance(token, str) for token in observed)
    ):
        return False
    for index, expected_token in enumerate(expected):
        observed_token = observed[index]
        if index not in path_indices:
            if observed_token != expected_token:
                return False
            continue
        try:
            if normalize_scientific_absolute_path(observed_token) != (
                normalize_scientific_absolute_path(expected_token)
            ):
                return False
        except RuntimeError:
            return False
    return True


def _validate_official_reference_execution_evidence(
    *,
    spec: ClosurePackageFamilySpec,
    paper_run_name: str,
    command_result: Mapping[str, Any],
    environment_report: Mapping[str, Any],
    dependency_report: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> None:
    """复验官方命令 argv, cwd, 隔离解释器与 CUDA 探针身份."""

    baseline_id = str(spec.baseline_id or "")
    config = run_manifest.get("config")
    embedded_dependency = dependency_report.get(
        "isolated_dependency_environment_report"
    )
    if not isinstance(config, Mapping) or not isinstance(
        embedded_dependency,
        Mapping,
    ):
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 缺少官方命令配置或依赖快照"
        )
    dependency_python = str(
        dependency_report.get("dependency_python_executable", "")
    )
    dependency_python_digest = _normalized_sha256(
        embedded_dependency.get("python_executable_sha256"),
        field_name="official_dependency_python_sha256",
    )
    try:
        repository_root = normalize_scientific_absolute_path(
            embedded_dependency.get("working_directory")
        )
        if (
            normalize_scientific_absolute_path(dependency_python)
            != normalize_scientific_absolute_path(
                embedded_dependency.get("python_executable_path")
            )
        ):
            raise ClosurePackageSelectionError(
                f"{spec.package_family} 的外层依赖解释器与内嵌报告不一致"
            )
    except RuntimeError as exc:
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的依赖解释器路径无效"
        ) from exc
    expected_profile_id = spec.dependency_environment_evidence.profile_id
    if config.get("dependency_profile_id") != expected_profile_id:
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的命令配置未绑定固定依赖 profile"
        )

    dependency_field = f"{baseline_id}_official_reference_dependency_environment_report"
    device_field = f"{baseline_id}_official_reference_device_report"
    device_report = environment_report.get(device_field)
    if (
        environment_report.get(dependency_field) != dependency_report
        or not isinstance(device_report, Mapping)
    ):
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的环境报告未绑定依赖与 CUDA 报告"
        )
    device_digest = stable_json_payload_digest(device_report)
    device_count = device_report.get("device_count")
    device_command = device_report.get("command")
    expected_device_command = [
        dependency_python,
        "-c",
        CUDA_INSPECTION_PROGRAM,
        "1",
    ]
    device_ready = all(
        (
            device_report.get("decision") == "pass",
            device_report.get("failure_reasons") == [],
            device_report.get("return_code") == 0,
            device_report.get("stderr") == "",
            device_report.get("torch_available") is True,
            device_report.get("cuda_available") is True,
            device_report.get("device") == "cuda",
            device_report.get("supports_paper_claim") is False,
            isinstance(device_count, int),
            not isinstance(device_count, bool),
            int(device_count or 0) > 0,
            bool(str(device_report.get("gpu_name", "")).strip()),
            bool(str(device_report.get("torch_version", "")).strip()),
            bool(str(device_report.get("torch_cuda_version", "")).strip()),
            device_report.get("python_executable_sha256")
            == dependency_python_digest,
            device_command == expected_device_command,
        )
    )
    try:
        device_ready = device_ready and all(
            (
                normalize_scientific_absolute_path(
                    device_report.get("python_executable")
                )
                == normalize_scientific_absolute_path(dependency_python),
                normalize_scientific_absolute_path(
                    device_report.get("working_directory")
                )
                == repository_root,
            )
        )
    except RuntimeError:
        device_ready = False
    stdout_lines = [
        line
        for line in str(device_report.get("stdout", "")).splitlines()
        if line.strip()
    ]
    if len(stdout_lines) != 1:
        device_ready = False
    else:
        try:
            stdout_payload = json.loads(stdout_lines[0])
        except json.JSONDecodeError:
            device_ready = False
        else:
            expected_stdout = {
                field_name: device_report.get(field_name)
                for field_name in (
                    "python_executable",
                    "torch_available",
                    "cuda_available",
                    "device",
                    "torch_version",
                    "torch_cuda_version",
                    "device_count",
                    "gpu_name",
                )
            }
            device_ready = device_ready and stdout_payload == expected_stdout
    if not device_ready:
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的 CUDA 子进程核验身份无效"
        )

    expected_command, path_indices, expected_cwd = (
        _expected_official_reference_command(
            baseline_id=baseline_id,
            config=config,
            repository_root=repository_root,
            paper_run_name=paper_run_name,
            python_executable=dependency_python,
        )
    )
    command_ready = all(
        (
            command_result.get("report_schema")
            == "official_reference_command_execution_report",
            command_result.get("schema_version") == 1,
            command_result.get("baseline_id") == baseline_id,
            command_result.get("official_command_requested") is True,
            command_result.get("return_code") == 0,
            command_result.get("official_command_execution_evidence_ready")
            is True,
            command_result.get("failure_reasons") == [],
            command_result.get("supports_paper_claim") is False,
            command_result.get("dependency_python_executable_sha256")
            == dependency_python_digest,
            command_result.get("cuda_inspection_report_digest") == device_digest,
            _official_command_matches(
                command_result.get("official_command"),
                expected_command,
                path_indices=path_indices,
            ),
        )
    )
    try:
        command_ready = command_ready and all(
            (
                normalize_scientific_absolute_path(
                    command_result.get("dependency_python_executable")
                )
                == normalize_scientific_absolute_path(dependency_python),
                normalize_scientific_absolute_path(
                    command_result.get("official_command_working_directory")
                )
                == normalize_scientific_absolute_path(expected_cwd),
            )
        )
    except RuntimeError:
        command_ready = False
    if not command_ready:
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的官方命令 argv, cwd 或解释器身份无效"
        )


def _validate_dependency_environment_evidence(
    archive: ZipFile,
    *,
    spec: ClosurePackageFamilySpec,
    paper_run_name: str,
    code_version: str,
    package_manifest: dict[str, Any],
    cache: dict[str, dict[str, Any]],
) -> dict[str, str] | None:
    """复验官方参考包内嵌的隔离依赖环境与执行锁证据."""

    contract = spec.dependency_environment_evidence
    if contract is None:
        return None
    if spec.scientific_execution_binding is not None:
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 同时声明了两类科学环境证据"
        )
    member_name = _format_template(
        contract.report_member_template,
        spec,
        paper_run_name,
    )
    command_result_member = _format_template(
        contract.command_result_member_template,
        spec,
        paper_run_name,
    )
    environment_report_member = _format_template(
        contract.environment_report_member_template,
        spec,
        paper_run_name,
    )
    run_manifest_member = _format_template(
        contract.run_manifest_member_template,
        spec,
        paper_run_name,
    )
    report = _read_json_object(archive, member_name, cache)
    profile_digest = _normalized_sha256(
        report.get("dependency_profile_digest"),
        field_name="dependency_profile_digest",
    )
    complete_hash_lock_digest = _normalized_sha256(
        report.get("dependency_lock_digest"),
        field_name="dependency_lock_digest",
    )
    isolated_report_digest = _normalized_sha256(
        report.get("isolated_dependency_environment_report_digest"),
        field_name="isolated_dependency_environment_report_digest",
    )
    embedded = report.get("isolated_dependency_environment_report")
    if not isinstance(embedded, dict):
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 缺少内嵌隔离依赖环境报告"
        )
    if _stable_json_file_digest(embedded) != isolated_report_digest:
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的内嵌隔离依赖环境报告摘要不匹配"
        )
    formal_lock = _validate_binding_formal_lock(
        embedded.get("formal_execution_lock"),
        package_family=spec.package_family,
        code_version=code_version,
    )
    if formal_lock != package_manifest.get("formal_execution_run_lock"):
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的隔离依赖环境锁与运行 manifest 不一致"
        )
    expected_outer_values = {
        "dependency_environment_requested": True,
        "dependency_environment_ready": True,
        "dependency_environment_materialized": True,
        "dependency_environment_profile_id": contract.profile_id,
        "dependency_profile_id": contract.profile_id,
        "dependency_profile_ready": True,
        "dependency_lock_ready": True,
        "dependency_environment_report_valid": True,
        "dependency_installation_performed": True,
        "dependency_environment_failure_reason": "",
    }
    if any(
        report.get(field_name) != expected_value
        for field_name, expected_value in expected_outer_values.items()
    ):
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的官方参考隔离依赖外层门禁无效"
        )
    if not all(
        (
            embedded.get("report_schema")
            == "isolated_dependency_environment_preparation_report",
            embedded.get("schema_version") == 1,
            embedded.get("operation_kind")
            == "formal_dependency_environment_preparation",
            embedded.get("profile_id") == contract.profile_id,
            embedded.get("profile_digest") == profile_digest,
            embedded.get("complete_hash_lock_digest")
            == complete_hash_lock_digest,
            int(embedded.get("complete_hash_lock_dependency_count", 0)) > 0,
            embedded.get("provisioned") is True,
            embedded.get("formal_preparation_completed") is True,
            embedded.get("formal_ready") is True,
            embedded.get("formal_execution_lock_ready") is True,
            embedded.get("formal_execution_lock") == formal_lock,
            embedded.get("formal_execution_commit")
            == formal_lock.get("formal_execution_commit"),
            embedded.get("formal_execution_lock_digest")
            == formal_lock.get("formal_execution_lock_digest"),
            embedded.get("decision") == "pass",
            embedded.get("failure_reasons") == [],
            embedded.get("supports_paper_claim") is False,
        )
    ):
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的内嵌隔离依赖环境报告未通过复验"
        )
    direct_requirements_digest = _normalized_sha256(
        embedded.get("direct_requirements_digest"),
        field_name="direct_requirements_digest",
    )
    python_digest = _normalized_sha256(
        embedded.get("python_executable_sha256"),
        field_name="python_executable_sha256",
    )
    if embedded.get("python_executable_sha256_after_preparation") != python_digest:
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的隔离解释器准备前后摘要不一致"
        )
    try:
        validate_dependency_environment_report_snapshot(
            embedded,
            expected_profile_id=contract.profile_id,
            expected_profile_digest=profile_digest,
            expected_direct_requirements_digest=direct_requirements_digest,
            expected_complete_hash_lock_digest=complete_hash_lock_digest,
            expected_complete_hash_lock_dependency_count=int(
                embedded["complete_hash_lock_dependency_count"]
            ),
            expected_python_executable_path=embedded[
                "python_executable_path"
            ],
            expected_python_executable_digest=python_digest,
            expected_formal_execution_lock=formal_lock,
            expected_working_directory=embedded["working_directory"],
        )
    except RuntimeError as exc:
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的官方参考依赖嵌套证据未通过复验"
        ) from exc
    dependency_preparation = embedded.get("dependency_preparation_report")
    if not isinstance(dependency_preparation, dict) or not all(
        (
            dependency_preparation.get("profile_id") == contract.profile_id,
            dependency_preparation.get("profile_digest") == profile_digest,
            dependency_preparation.get("direct_requirements_digest")
            == direct_requirements_digest,
            dependency_preparation.get("complete_hash_lock_digest")
            == complete_hash_lock_digest,
            dependency_preparation.get("formal_ready") is True,
            dependency_preparation.get("formal_execution_lock_ready") is True,
            dependency_preparation.get("formal_execution_lock") == formal_lock,
            dependency_preparation.get("decision") == "pass",
            dependency_preparation.get("failure_reasons") == [],
            dependency_preparation.get("supports_paper_claim") is False,
        )
    ):
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的内层依赖准备报告未通过复验"
        )
    dependency_preparation_digest = _normalized_sha256(
        embedded.get("dependency_preparation_report_digest"),
        field_name="dependency_preparation_report_digest",
    )
    provision_report = embedded.get("provision_report")
    provision_report_digest = _normalized_sha256(
        embedded.get("provision_report_digest"),
        field_name="provision_report_digest",
    )
    if (
        _stable_json_file_digest(dependency_preparation)
        != dependency_preparation_digest
        or not isinstance(provision_report, dict)
        or _stable_json_file_digest(provision_report) != provision_report_digest
    ):
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 的内层依赖或 provision 摘要不匹配"
        )
    command_result = _read_json_object(
        archive,
        command_result_member,
        cache,
    )
    environment_report = _read_json_object(
        archive,
        environment_report_member,
        cache,
    )
    run_manifest = _read_json_object(
        archive,
        run_manifest_member,
        cache,
    )
    _validate_official_reference_execution_evidence(
        spec=spec,
        paper_run_name=paper_run_name,
        command_result=command_result,
        environment_report=environment_report,
        dependency_report=report,
        run_manifest=run_manifest,
    )
    combined_evidence_digest = stable_json_payload_digest(
        {
            "dependency_environment_report_sha256": _archive_member_sha256(
                archive,
                member_name,
            ),
            "official_command_result_sha256": _archive_member_sha256(
                archive,
                command_result_member,
            ),
            "official_environment_report_sha256": _archive_member_sha256(
                archive,
                environment_report_member,
            ),
        }
    )
    return {
        "scientific_profile_id": contract.profile_id,
        "scientific_profile_digest": profile_digest,
        "scientific_direct_requirements_digest": direct_requirements_digest,
        "scientific_complete_hash_lock_digest": complete_hash_lock_digest,
        "scientific_complete_hash_lock_dependency_count": int(
            embedded["complete_hash_lock_dependency_count"]
        ),
        "scientific_python_executable_digest": python_digest,
        "scientific_execution_report_digest": "",
        "scientific_command_dispatch_report_digest": "",
        "scientific_command_sequence_digest": "",
        "scientific_execution_binding_digest": "",
        "scientific_dependency_evidence_digest": combined_evidence_digest,
    }


def _validate_declared_archive_entries(
    archive: ZipFile,
    archive_member_names: set[str],
    spec: ClosurePackageFamilySpec,
    paper_run_name: str,
    cache: dict[str, dict[str, Any]],
) -> None:
    """在 package input manifest 提供逐成员摘要时核验精确成员集合.

    ``entry_paths`` 描述动态运行产物, ``entry_sha256`` 绑定其真实字节. 打包时
    后写入的 package input、archive summary 与 archive manifest 属于显式治理
    成员, 因此允许作为声明集合之外的固定成员; 其他额外成员一律拒绝.
    """

    template = spec.package_input_manifest_template
    if template is None:
        return
    package_input_member = _format_template(template, spec, paper_run_name)
    package_input = _read_json_object(archive, package_input_member, cache)
    has_paths = "entry_paths" in package_input
    has_digests = "entry_sha256" in package_input
    if not has_paths or not has_digests:
        raise ClosurePackageSelectionError(
            f"{package_input_member} 必须同时声明 entry_paths 与 entry_sha256"
        )
    paths_value = package_input["entry_paths"]
    digests_value = package_input["entry_sha256"]
    if not isinstance(paths_value, list) or not isinstance(digests_value, dict):
        raise ClosurePackageSelectionError(
            f"{package_input_member} 的成员清单或摘要映射类型非法"
        )
    declared_paths: list[str] = []
    for raw_path in paths_value:
        if not isinstance(raw_path, str):
            raise ClosurePackageSelectionError(
                f"{package_input_member} 的 entry_paths 必须只包含字符串"
            )
        pure_path = PurePosixPath(raw_path)
        if (
            not raw_path
            or "\\" in raw_path
            or pure_path.is_absolute()
            or any(part in {"", ".", ".."} for part in pure_path.parts)
            or pure_path.as_posix() != raw_path
        ):
            raise ClosurePackageSelectionError(
                f"{package_input_member} 声明了非规范成员路径: {raw_path}"
            )
        declared_paths.append(raw_path)
    declared_set = set(declared_paths)
    if len(declared_set) != len(declared_paths) or not declared_set:
        raise ClosurePackageSelectionError(
            f"{package_input_member} 的 entry_paths 为空或包含重复项"
        )
    if "entry_count" in package_input and package_input.get("entry_count") != len(
        declared_paths
    ):
        raise ClosurePackageSelectionError(
            f"{package_input_member} 的 entry_count 与精确成员集合不一致"
        )
    if set(digests_value) != declared_set:
        raise ClosurePackageSelectionError(
            f"{package_input_member} 的 entry_sha256 键必须与 entry_paths 完全一致"
        )

    required_members = {
        _format_template(template_value, spec, paper_run_name)
        for template_value in spec.required_member_templates
    }
    governance_members = required_members - declared_set
    expected_members = declared_set | governance_members
    if archive_member_names != expected_members:
        unexpected = sorted(archive_member_names - expected_members)
        missing = sorted(expected_members - archive_member_names)
        raise ClosurePackageSelectionError(
            f"{package_input_member} 与 ZIP 精确成员集合不一致: "
            f"unexpected={','.join(unexpected)};missing={','.join(missing)}"
        )
    for member_name in declared_paths:
        declared_digest = digests_value[member_name]
        if not isinstance(declared_digest, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", declared_digest.strip()
        ):
            raise ClosurePackageSelectionError(
                f"{package_input_member} 的成员摘要非法: {member_name}"
            )
        if _archive_member_sha256(archive, member_name) != declared_digest.strip().lower():
            raise ClosurePackageSelectionError(
                f"{package_input_member} 的成员摘要不匹配: {member_name}"
            )
    package_manifest_member = _format_template(
        spec.manifest_member_template,
        spec,
        paper_run_name,
    )
    package_manifest = _read_json_object(
        archive,
        package_manifest_member,
        cache,
    )
    lock_fields = (
        "formal_execution_run_lock",
        "formal_execution_package_lock",
    )
    if any(field_name in package_input for field_name in lock_fields) and any(
        package_input.get(field_name) != package_manifest.get(field_name)
        for field_name in lock_fields
    ):
        raise ClosurePackageSelectionError(
            f"{package_input_member} 的运行锁或打包锁与 package manifest 不一致"
        )


def _validate_archive_member_names(
    archive: ZipFile,
    spec: ClosurePackageFamilySpec,
    paper_run_name: str,
) -> set[str]:
    """拒绝路径穿越、目录、符号链接和 family 白名单外成员."""

    infos = archive.infolist()
    if not infos:
        raise ClosurePackageSelectionError("结果包不得为空 ZIP")
    allowed_prefixes = tuple(
        _format_template(template, spec, paper_run_name)
        for template in spec.allowed_output_prefix_templates
    )
    allowed_members = {
        _format_template(template, spec, paper_run_name)
        for template in spec.allowed_output_member_templates
    }
    normalized_names: set[str] = set()
    for info in infos:
        raw_name = info.filename
        if not raw_name or "\\" in raw_name or raw_name.startswith("/"):
            raise ClosurePackageSelectionError(f"ZIP 成员路径格式非法: {raw_name}")
        pure_path = PurePosixPath(raw_name)
        if (
            pure_path.is_absolute()
            or any(part in {"", ".", ".."} for part in pure_path.parts)
            or (pure_path.parts and pure_path.parts[0].endswith(":"))
            or pure_path.as_posix() != raw_name
        ):
            raise ClosurePackageSelectionError(f"ZIP 成员存在路径穿越或非规范路径: {raw_name}")
        if info.is_dir():
            raise ClosurePackageSelectionError(f"结果包不得包含目录成员: {raw_name}")
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if unix_mode and stat.S_ISLNK(unix_mode):
            raise ClosurePackageSelectionError(f"结果包不得包含符号链接: {raw_name}")
        if raw_name in normalized_names:
            raise ClosurePackageSelectionError(f"结果包包含重复成员: {raw_name}")
        normalized_names.add(raw_name)
        if not raw_name.startswith("outputs/"):
            raise ClosurePackageSelectionError(f"结果包包含 outputs 外成员: {raw_name}")
        if raw_name not in allowed_members and not any(
            raw_name.startswith(prefix) for prefix in allowed_prefixes
        ):
            raise ClosurePackageSelectionError(f"结果包包含 family 白名单外成员: {raw_name}")
    required_members = {
        _format_template(template, spec, paper_run_name)
        for template in spec.required_member_templates
    }
    missing_members = sorted(required_members - normalized_names)
    if missing_members:
        raise ClosurePackageSelectionError(
            f"{spec.package_family} 缺少必要成员: {','.join(missing_members)}"
        )
    return normalized_names


def _validate_baseline_rows(
    archive: ZipFile,
    rows_source: BaselineRowsSource,
    spec: ClosurePackageFamilySpec,
    paper_run_name: str,
) -> None:
    """逐行核验官方记录或 formal candidate 的 baseline 身份."""

    expected_baseline = spec.baseline_id
    if expected_baseline is None:
        return
    member_name = _format_template(rows_source.member_template, spec, paper_run_name)
    info = archive.getinfo(member_name)
    if info.file_size <= 0 or info.file_size > MAX_GOVERNANCE_MEMBER_BYTES:
        raise ClosurePackageSelectionError(f"baseline 身份记录大小非法: {member_name}")
    row_count = 0
    with archive.open(member_name) as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            row_count += 1
            try:
                row = json.loads(raw_line.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ClosurePackageSelectionError(
                    f"baseline 身份记录不是有效 JSONL: {member_name}"
                ) from error
            actual_baseline = _field_value(
                row,
                rows_source.baseline_field_path,
                member_name=member_name,
            )
            if actual_baseline != expected_baseline:
                raise ClosurePackageSelectionError(
                    f"{member_name} 的 baseline 身份不匹配: {actual_baseline}"
                )
    if row_count <= 0:
        raise ClosurePackageSelectionError(f"baseline 身份记录不得为空: {member_name}")


def _file_sha256(path: Path) -> str:
    """流式计算结果包 SHA-256, 避免把大型包整体读入内存."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_closure_package(
    package_path: str | Path,
    *,
    spec: ClosurePackageFamilySpec,
    paper_run_name: str,
    target_fpr: float,
    randomization_repeat_id: str | None = None,
) -> ClosurePackageCandidate:
    """按指定 family 的包内身份和证据契约校验单个 ZIP."""

    resolved_paper_run = normalize_paper_run_name(paper_run_name)
    expected_target_fpr = float(target_fpr)
    expected_repeat = resolve_formal_randomization_repeat(
        randomization_repeat_id
    )
    expected_randomization_protocol_digest = (
        formal_randomization_protocol_record()[
            "formal_randomization_protocol_digest"
        ]
    )
    if not math.isfinite(expected_target_fpr) or not 0.0 < expected_target_fpr < 1.0:
        raise ClosurePackageSelectionError("target_fpr 必须是位于 (0, 1) 的有限数值")
    path = Path(package_path).expanduser()
    if not path.is_file() or path.is_symlink():
        raise ClosurePackageSelectionError(f"闭合输入必须是普通 ZIP 文件: {path}")
    if path.suffix.lower() != ".zip" or path.stat().st_size <= 0:
        raise ClosurePackageSelectionError(f"闭合输入不是非空 ZIP 文件: {path}")
    try:
        with ZipFile(path) as archive:
            archive_member_names = _validate_archive_member_names(
                archive, spec, resolved_paper_run
            )
            damaged_member = archive.testzip()
            if damaged_member is not None:
                raise ClosurePackageSelectionError(f"ZIP 成员 CRC 校验失败: {damaged_member}")
            cache: dict[str, dict[str, Any]] = {}
            manifest_member = _format_template(
                spec.manifest_member_template,
                spec,
                resolved_paper_run,
            )
            manifest = _read_json_object(archive, manifest_member, cache)
            expected_artifact_id = _format_template(
                spec.manifest_artifact_id_template,
                spec,
                resolved_paper_run,
            )
            if manifest.get("artifact_id") != expected_artifact_id:
                raise ClosurePackageSelectionError(
                    f"{spec.package_family} 的 artifact_id 不匹配"
                )
            if manifest.get("artifact_type") != "local_manifest":
                raise ClosurePackageSelectionError(
                    f"{spec.package_family} 的 manifest 类型不匹配"
                )

            for source in spec.paper_run_sources:
                actual_run = _read_source_value(
                    archive,
                    source,
                    spec,
                    resolved_paper_run,
                    cache,
                )
                if actual_run != resolved_paper_run:
                    raise ClosurePackageSelectionError(
                        f"{spec.package_family} 的论文运行层级不匹配: {actual_run}"
                    )
            for source in spec.target_fpr_sources:
                actual_fpr = _read_source_value(
                    archive,
                    source,
                    spec,
                    resolved_paper_run,
                    cache,
                )
                try:
                    numeric_fpr = float(actual_fpr)
                except (TypeError, ValueError) as error:
                    raise ClosurePackageSelectionError(
                        f"{spec.package_family} 的 target_fpr 不是数值"
                    ) from error
                if not math.isfinite(numeric_fpr) or not math.isclose(
                    numeric_fpr,
                    expected_target_fpr,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ClosurePackageSelectionError(
                        f"{spec.package_family} 的 target_fpr 不匹配: {actual_fpr}"
                    )
            for source in spec.baseline_sources:
                actual_baseline = _read_source_value(
                    archive,
                    source,
                    spec,
                    resolved_paper_run,
                    cache,
                )
                if actual_baseline != spec.baseline_id:
                    raise ClosurePackageSelectionError(
                        f"{spec.package_family} 的 baseline 身份不匹配: {actual_baseline}"
                    )

            if spec.package_family in RANDOMIZATION_REPEAT_PACKAGE_FAMILIES:
                if len(spec.randomization_repeat_sources) < 2:
                    raise ClosurePackageSelectionError(
                        f"{spec.package_family} 缺少双来源 repeat 身份"
                    )
                repeat_identities = []
                for source in spec.randomization_repeat_sources:
                    raw_identity = _read_source_value(
                        archive,
                        source,
                        spec,
                        resolved_paper_run,
                        cache,
                    )
                    if not isinstance(raw_identity, dict):
                        raise ClosurePackageSelectionError(
                            f"{spec.package_family} 的 repeat 身份必须是 JSON object"
                        )
                    try:
                        normalized_identity = (
                            validate_formal_randomization_repeat_records(
                                [raw_identity],
                                require_exact_registry=False,
                            )[0]
                        )
                    except (TypeError, ValueError) as exc:
                        raise ClosurePackageSelectionError(
                            f"{spec.package_family} 的 repeat 身份未通过注册表校验"
                        ) from exc
                    if (
                        raw_identity.get(
                            "formal_randomization_protocol_digest"
                        )
                        != expected_randomization_protocol_digest
                    ):
                        raise ClosurePackageSelectionError(
                            f"{spec.package_family} 的随机化协议摘要不匹配"
                        )
                    repeat_identities.append(normalized_identity)
                if any(
                    identity != expected_repeat.to_dict()
                    for identity in repeat_identities
                ) or len(
                    {
                        json.dumps(identity, sort_keys=True)
                        for identity in repeat_identities
                    }
                ) != 1:
                    raise ClosurePackageSelectionError(
                        f"{spec.package_family} 未绑定活动随机化 repeat"
                    )
                randomization_evidence = {
                    "randomization_scope": "active_repeat_component",
                    **expected_repeat.to_dict(),
                    "formal_randomization_protocol_digest": (
                        expected_randomization_protocol_digest
                    ),
                }
            elif spec.package_family in CROSS_REPEAT_INVARIANT_PACKAGE_FAMILIES:
                if spec.randomization_repeat_sources:
                    raise ClosurePackageSelectionError(
                        f"{spec.package_family} 不得声明活动 repeat 来源"
                    )
                randomization_evidence = {
                    "randomization_scope": "cross_repeat_invariant",
                }
            else:
                raise ClosurePackageSelectionError(
                    f"未登记 package family 的随机化职责: {spec.package_family}"
                )

            code_versions = [
                _read_source_value(
                    archive,
                    source,
                    spec,
                    resolved_paper_run,
                    cache,
                )
                for source in spec.code_version_sources
            ]
            if not code_versions:
                raise ClosurePackageSelectionError(
                    f"{spec.package_family} 缺少有效 code_version"
                )
            try:
                normalized_code_versions = {
                    normalize_clean_code_version(version) for version in code_versions
                }
            except ClosurePackageSelectionError as error:
                raise ClosurePackageSelectionError(
                    f"{spec.package_family} 缺少有效 code_version: {error}"
                ) from error
            if len(normalized_code_versions) != 1:
                raise ClosurePackageSelectionError(
                    f"{spec.package_family} 的 code_version 来源不一致"
                )
            code_version = next(iter(normalized_code_versions))
            (
                formal_execution_run_lock_digest,
                formal_execution_package_lock_digest,
            ) = _validate_package_execution_locks(
                manifest,
                package_family=spec.package_family,
                code_version=code_version,
            )
            scientific_execution_evidence = _validate_scientific_execution_binding(
                archive,
                spec=spec,
                paper_run_name=resolved_paper_run,
                code_version=code_version,
                package_manifest=manifest,
                cache=cache,
            )
            dependency_environment_evidence = (
                _validate_dependency_environment_evidence(
                    archive,
                    spec=spec,
                    paper_run_name=resolved_paper_run,
                    code_version=code_version,
                    package_manifest=manifest,
                    cache=cache,
                )
            )
            if dependency_environment_evidence is not None:
                scientific_execution_evidence = dependency_environment_evidence

            for requirement in spec.value_requirements:
                actual_value = _read_source_value(
                    archive,
                    requirement.source,
                    spec,
                    resolved_paper_run,
                    cache,
                )
                if isinstance(requirement.expected_value, bool):
                    matches = actual_value is requirement.expected_value
                else:
                    matches = actual_value == requirement.expected_value
                if not matches:
                    member_name = _format_template(
                        requirement.source.member_template,
                        spec,
                        resolved_paper_run,
                    )
                    field_name = ".".join(requirement.source.field_path)
                    raise ClosurePackageSelectionError(
                        f"{member_name} 的 {field_name} 未通过闭合门禁"
                    )
            for rows_source in spec.baseline_rows_sources:
                _validate_baseline_rows(
                    archive,
                    rows_source,
                    spec,
                    resolved_paper_run,
                )
            _validate_declared_archive_entries(
                archive,
                archive_member_names,
                spec,
                resolved_paper_run,
                cache,
            )
            generated_value = _read_source_value(
                archive,
                spec.generated_at_source,
                spec,
                resolved_paper_run,
                cache,
            )
            generated_at, generated_at_utc = _validated_generated_at(generated_value)
    except (BadZipFile, EOFError, NotImplementedError, OSError, RuntimeError, zlib.error) as error:
        if isinstance(error, ClosurePackageSelectionError):
            raise
        raise ClosurePackageSelectionError(f"损坏或不可读取的 ZIP: {path}") from error

    return ClosurePackageCandidate(
        package_family=spec.package_family,
        package_path=path.resolve(),
        package_sha256=_file_sha256(path),
        paper_run_name=resolved_paper_run,
        target_fpr=expected_target_fpr,
        code_version=code_version,
        formal_execution_run_lock_digest=formal_execution_run_lock_digest,
        formal_execution_package_lock_digest=formal_execution_package_lock_digest,
        generated_at=generated_at,
        generated_at_utc=generated_at_utc,
        **randomization_evidence,
        **scientific_execution_evidence,
    )


def _select_latest_candidate(
    candidates: list[ClosurePackageCandidate],
    *,
    package_family: str,
) -> ClosurePackageCandidate:
    """按包内 generated_at 选择最新证据, 并拒绝同时间不同内容."""

    latest_time = max(candidate.generated_at_utc for candidate in candidates)
    latest = [candidate for candidate in candidates if candidate.generated_at_utc == latest_time]
    distinct_digests = {candidate.package_sha256 for candidate in latest}
    if len(distinct_digests) > 1:
        raise ClosurePackageSelectionError(
            f"{package_family} 存在 generated_at 相同但内容不同的歧义候选"
        )
    return min(latest, key=lambda candidate: candidate.package_path.as_posix())


def _validate_candidate_repository_profile(
    candidate: ClosurePackageCandidate,
    *,
    repository_root: Path,
) -> None:
    """把包内科学环境身份锚定到当前提交的正式依赖 registry 与完整锁."""

    if not candidate.scientific_profile_id:
        raise ClosurePackageSelectionError(
            f"{candidate.package_family} 未传播科学依赖 profile 身份"
        )
    registry_path = repository_root / "configs" / "dependency_profile_registry.json"
    try:
        profile = require_dependency_profile_ready(
            candidate.scientific_profile_id,
            registry_path,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise ClosurePackageSelectionError(
            f"{candidate.package_family} 对应的仓库依赖 profile 未正式就绪"
        ) from exc
    if not all(
        (
            profile.profile_name == candidate.scientific_profile_id,
            profile.profile_digest == candidate.scientific_profile_digest,
            profile.direct_requirements_digest
            == candidate.scientific_direct_requirements_digest,
            profile.complete_hash_lock_digest
            == candidate.scientific_complete_hash_lock_digest,
            profile.complete_hash_lock_dependency_count
            == candidate.scientific_complete_hash_lock_dependency_count,
            profile.complete_hash_lock_present is True,
            profile.formal_ready is True,
            profile.readiness_blockers == (),
        )
    ):
        raise ClosurePackageSelectionError(
            f"{candidate.package_family} 的包内依赖身份与仓库正式 profile 不一致"
        )


def _validate_semantic_watermark_session_group(
    candidates: list[ClosurePackageCandidate],
) -> None:
    """要求主方法、质量和消融三个包来自同一次完整科学会话."""

    selected = {
        candidate.package_family: candidate
        for candidate in candidates
        if candidate.package_family in SEMANTIC_WATERMARK_PACKAGE_FAMILIES
    }
    if set(selected) != SEMANTIC_WATERMARK_PACKAGE_FAMILIES:
        raise ClosurePackageSelectionError("主方法闭合包未覆盖完整科学会话角色")
    mismatched_fields = [
        field_name
        for field_name in SEMANTIC_SESSION_IDENTITY_FIELDS
        if len(
            {
                getattr(candidate, field_name)
                for candidate in selected.values()
            }
        )
        != 1
    ]
    if mismatched_fields:
        raise ClosurePackageSelectionError(
            "主方法三个结果包不属于同一科学会话: "
            + ",".join(mismatched_fields)
        )


def _stable_digest(payload: Any) -> str:
    """计算锁内容的确定性 SHA-256."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_closure_input_lock_payloads(
    lock_payload: dict[str, Any],
    lock_manifest: dict[str, Any],
    *,
    paper_run_name: str,
    target_fpr: float,
    randomization_repeat_id: str | None = None,
) -> dict[str, str]:
    """复验 CPU 结果闭合输入锁的基本身份与规范摘要.

    该函数属于通用跨产物治理写法.它不重新读取大型 ZIP, 只验证在下游
    records 与协议产物中必须稳定传播的 run,FPR,精确包集合,代码版本和
    锁摘要.最终完整打包器仍负责逐包字节摘要复核.
    """

    resolved_run_name = normalize_paper_run_name(paper_run_name)
    expected_target_fpr = float(target_fpr)
    expected_repeat = resolve_formal_randomization_repeat(
        randomization_repeat_id
    )
    expected_randomization_protocol_digest = (
        formal_randomization_protocol_record()[
            "formal_randomization_protocol_digest"
        ]
    )
    expected_families = {spec.package_family for spec in CLOSURE_PACKAGE_FAMILY_SPECS}
    expected_scientific_profiles = {
        spec.package_family: (
            spec.scientific_execution_binding.profile_id
            if spec.scientific_execution_binding is not None
            else spec.dependency_environment_evidence.profile_id
        )
        for spec in CLOSURE_PACKAGE_FAMILY_SPECS
        if spec.scientific_execution_binding is not None
        or spec.dependency_environment_evidence is not None
    }
    binding_families = {
        spec.package_family
        for spec in CLOSURE_PACKAGE_FAMILY_SPECS
        if spec.scientific_execution_binding is not None
    }
    semantic_session_families = {
        spec.package_family
        for spec in CLOSURE_PACKAGE_FAMILY_SPECS
        if spec.scientific_execution_binding is not None
        and spec.scientific_execution_binding.execution_route
        in {
            "semantic_watermark_session",
            "semantic_watermark_ablation_session",
        }
    }
    records = lock_payload.get("closure_input_packages")
    if not isinstance(records, list):
        raise ClosurePackageSelectionError("closure input lock 缺少包记录列表")
    actual_families = [str(record.get("package_family", "")) for record in records]
    common_code_version = normalize_clean_code_version(
        lock_payload.get("common_code_version")
    )
    declared_digest = str(lock_payload.get("closure_input_lock_digest", ""))
    digest_payload = dict(lock_payload)
    digest_payload.pop("closure_input_lock_digest", None)

    def scientific_record_ready(record: dict[str, Any]) -> bool:
        """复验锁记录是否保留该 family 必需的科学执行摘要."""

        package_family = str(record.get("package_family", ""))
        expected_profile_id = expected_scientific_profiles.get(package_family)
        fields = (
            "scientific_profile_digest",
            "scientific_direct_requirements_digest",
            "scientific_complete_hash_lock_digest",
            "scientific_python_executable_digest",
            "scientific_dependency_evidence_digest",
        )
        binding_report_fields = (
            "scientific_execution_report_digest",
            "scientific_command_dispatch_report_digest",
        )
        if expected_profile_id is None:
            return (
                record.get("scientific_profile_id") == ""
                and record.get("scientific_execution_binding_digest") == ""
                and record.get("scientific_complete_hash_lock_dependency_count")
                == 0
                and all(
                    record.get(field_name) == ""
                    for field_name in (*fields, *binding_report_fields)
                )
                and record.get("scientific_command_sequence_digest") == ""
            )
        base_ready = (
            record.get("scientific_profile_id") == expected_profile_id
            and isinstance(
                record.get("scientific_complete_hash_lock_dependency_count"),
                int,
            )
            and not isinstance(
                record.get("scientific_complete_hash_lock_dependency_count"),
                bool,
            )
            and record["scientific_complete_hash_lock_dependency_count"] > 0
            and all(
                isinstance(record.get(field_name), str)
                and re.fullmatch(r"[0-9a-f]{64}", record[field_name])
                is not None
                for field_name in fields
            )
        )
        binding_digest = record.get("scientific_execution_binding_digest")
        if package_family in binding_families:
            binding_ready = (
                isinstance(binding_digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", binding_digest) is not None
                and all(
                    isinstance(record.get(field_name), str)
                    and re.fullmatch(
                        r"[0-9a-f]{64}",
                        record[field_name],
                    )
                    is not None
                    for field_name in binding_report_fields
                )
            )
            sequence_digest = record.get("scientific_command_sequence_digest")
            sequence_ready = (
                isinstance(sequence_digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", sequence_digest) is not None
                if package_family in semantic_session_families
                else sequence_digest == ""
            )
            return base_ready and binding_ready and sequence_ready
        return (
            base_ready
            and binding_digest == ""
            and all(record.get(field_name) == "" for field_name in binding_report_fields)
            and record.get("scientific_command_sequence_digest") == ""
        )

    def randomization_record_ready(record: dict[str, Any]) -> bool:
        """区分活动 repeat 证据与跨 repeat 不变的忠实度证据."""

        package_family = str(record.get("package_family", ""))
        if package_family in RANDOMIZATION_REPEAT_PACKAGE_FAMILIES:
            expected_identity = {
                **expected_repeat.to_dict(),
                "formal_randomization_protocol_digest": (
                    expected_randomization_protocol_digest
                ),
            }
            return bool(
                record.get("randomization_scope")
                == "active_repeat_component"
                and all(
                    record.get(field_name) == expected_value
                    for field_name, expected_value in expected_identity.items()
                )
            )
        if package_family in CROSS_REPEAT_INVARIANT_PACKAGE_FAMILIES:
            return bool(
                record.get("randomization_scope")
                == "cross_repeat_invariant"
                and record.get("randomization_repeat_id") == ""
                and record.get("generation_seed_index") == -1
                and record.get("generation_seed_offset") == -1
                and record.get("watermark_key_index") == -1
                and record.get("formal_randomization_protocol_digest") == ""
            )
        return False

    package_rows_ready = all(
        isinstance(record, dict)
        and str(record.get("paper_run_name", "")) == resolved_run_name
        and math.isclose(
            float(record.get("target_fpr", float("nan"))),
            expected_target_fpr,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and normalize_clean_code_version(record.get("code_version"))
        == common_code_version
        and bool(str(record.get("package_path", "")))
        and bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(record.get("package_sha256", ""))))
        and bool(
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(record.get("formal_execution_run_lock_digest", "")),
            )
        )
        and bool(
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(record.get("formal_execution_package_lock_digest", "")),
            )
        )
        and scientific_record_ready(record)
        and randomization_record_ready(record)
        for record in records
    )
    semantic_session_records = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("package_family")
        in SEMANTIC_WATERMARK_PACKAGE_FAMILIES
    ]
    semantic_session_group_ready = (
        len(semantic_session_records) == 3
        and all(
            len(
                {
                    record.get(field_name)
                    for record in semantic_session_records
                }
            )
            == 1
            for field_name in SEMANTIC_SESSION_IDENTITY_FIELDS
        )
    )
    expected_run_lock_digests = {
        str(record.get("package_family", "")): str(
            record.get("formal_execution_run_lock_digest", "")
        )
        for record in records
        if isinstance(record, dict)
    }
    expected_package_lock_digests = {
        str(record.get("package_family", "")): str(
            record.get("formal_execution_package_lock_digest", "")
        )
        for record in records
        if isinstance(record, dict)
    }
    metadata = lock_manifest.get("metadata", {})
    output_paths = lock_manifest.get("output_paths", ())
    expected_output_suffixes = (
        f"{LOCK_OUTPUT_ROOT.as_posix()}/{resolved_run_name}/{LOCK_FILENAME}",
        f"{LOCK_OUTPUT_ROOT.as_posix()}/{resolved_run_name}/{LOCK_MANIFEST_FILENAME}",
    )
    ready = (
        str(lock_payload.get("paper_run_name", "")) == resolved_run_name
        and math.isclose(
            float(lock_payload.get("target_fpr", float("nan"))),
            expected_target_fpr,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and int(lock_payload.get("closure_input_package_count", -1))
        == len(CLOSURE_PACKAGE_FAMILY_SPECS)
        and len(records) == len(CLOSURE_PACKAGE_FAMILY_SPECS)
        and len(set(actual_families)) == len(actual_families)
        and set(actual_families) == expected_families
        and package_rows_ready
        and semantic_session_group_ready
        and lock_payload.get("formal_execution_run_lock_digests")
        == expected_run_lock_digests
        and lock_payload.get("formal_execution_package_lock_digests")
        == expected_package_lock_digests
        and lock_payload.get("randomization_repeat_identity")
        == {
            **expected_repeat.to_dict(),
            "formal_randomization_protocol_digest": (
                expected_randomization_protocol_digest
            ),
        }
        and lock_payload.get("repeat_component_input_ready") is True
        and lock_payload.get("randomization_aggregate_ready") is False
        and lock_payload.get("supports_paper_claim") is False
        and bool(re.fullmatch(r"[0-9a-fA-F]{64}", declared_digest))
        and _stable_digest(digest_payload) == declared_digest
        and str(lock_manifest.get("artifact_id", ""))
        == f"{resolved_run_name}_closure_input_lock_manifest"
        and isinstance(metadata, dict)
        and metadata.get("closure_input_lock_ready") is True
        and int(metadata.get("closure_input_package_count", -1)) == len(records)
        and metadata.get("closure_input_packages") == records
        and str(metadata.get("paper_run_name", "")) == resolved_run_name
        and math.isclose(
            float(metadata.get("target_fpr", float("nan"))),
            expected_target_fpr,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and str(metadata.get("closure_input_lock_digest", "")) == declared_digest
        and str(metadata.get("common_code_version", "")) == common_code_version
        and metadata.get("randomization_repeat_identity")
        == lock_payload.get("randomization_repeat_identity")
        and metadata.get("repeat_component_input_ready") is True
        and metadata.get("randomization_aggregate_ready") is False
        and metadata.get("supports_paper_claim") is False
        and all(
            any(str(path).replace("\\", "/").endswith(suffix) for path in output_paths)
            for suffix in expected_output_suffixes
        )
    )
    if not ready:
        raise ClosurePackageSelectionError("closure input lock 基本身份或摘要复验失败")
    return {
        "closure_input_lock_digest": declared_digest,
        "common_code_version": common_code_version,
        "randomization_repeat_id": expected_repeat.randomization_repeat_id,
    }


def load_validated_closure_input_lock(
    root: str | Path,
    *,
    paper_run_name: str,
    target_fpr: float,
    randomization_repeat_id: str | None = None,
) -> dict[str, Any]:
    """从当前论文运行目录读取并复验 closure input lock."""

    repository_root = Path(root).resolve()
    resolved_run_name = normalize_paper_run_name(paper_run_name)
    lock_dir = repository_root / LOCK_OUTPUT_ROOT / resolved_run_name
    lock_path = lock_dir / LOCK_FILENAME
    manifest_path = lock_dir / LOCK_MANIFEST_FILENAME
    if not lock_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("当前论文运行缺少 closure input lock 或独立 manifest")
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8-sig"))
    lock_manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(lock_payload, dict) or not isinstance(lock_manifest, dict):
        raise TypeError("closure input lock 与独立 manifest 必须是 JSON 对象")
    provenance = validate_closure_input_lock_payloads(
        lock_payload,
        lock_manifest,
        paper_run_name=resolved_run_name,
        target_fpr=target_fpr,
        randomization_repeat_id=randomization_repeat_id,
    )
    return {
        **provenance,
        "closure_input_lock": lock_payload,
        "closure_input_lock_manifest": lock_manifest,
        "closure_input_lock_path": lock_path,
        "closure_input_lock_manifest_path": manifest_path,
    }


def _write_closure_input_lock(
    *,
    repository_root: Path,
    paper_run_name: str,
    target_fpr: float,
    common_code_version: str,
    lock_payload: dict[str, Any],
    lock_records: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """把已通过选择的锁和独立 manifest 原子替换到 run-scoped 目录."""

    lock_output_dir = repository_root / LOCK_OUTPUT_ROOT / paper_run_name
    lock_path = lock_output_dir / LOCK_FILENAME
    lock_manifest_path = lock_output_dir / LOCK_MANIFEST_FILENAME
    lock_output_dir.mkdir(parents=True, exist_ok=True)
    lock_temporary_path = lock_path.with_suffix(".json.tmp")
    manifest_temporary_path = lock_manifest_path.with_suffix(".json.tmp")
    for temporary_path in (lock_temporary_path, manifest_temporary_path):
        if temporary_path.exists():
            temporary_path.unlink()
    lock_temporary_path.write_text(
        json.dumps(lock_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lock_manifest = build_artifact_manifest(
        artifact_id=f"{paper_run_name}_closure_input_lock_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(record["package_path"] for record in lock_records),
        output_paths=(
            lock_path.relative_to(repository_root).as_posix(),
            lock_manifest_path.relative_to(repository_root).as_posix(),
        ),
        config={
            "paper_run_name": paper_run_name,
            "target_fpr": target_fpr,
            "common_code_version": common_code_version,
            "closure_input_packages": lock_records,
            "randomization_repeat_identity": lock_payload[
                "randomization_repeat_identity"
            ],
            "repeat_component_input_ready": True,
            "randomization_aggregate_ready": False,
            "supports_paper_claim": False,
        },
        code_version=resolve_code_version(repository_root),
        rebuild_command=(
            "调用 paper_experiments.runners.closure_package_selection."
            "select_and_lock_closure_input_packages"
        ),
        metadata={
            "closure_input_lock_ready": True,
            "closure_input_package_count": len(lock_records),
            "closure_input_packages": lock_records,
            "closure_input_lock_digest": lock_payload["closure_input_lock_digest"],
            "paper_run_name": paper_run_name,
            "target_fpr": target_fpr,
            "common_code_version": common_code_version,
            "randomization_repeat_identity": lock_payload[
                "randomization_repeat_identity"
            ],
            "repeat_component_input_ready": True,
            "randomization_aggregate_ready": False,
            "supports_paper_claim": False,
        },
    ).to_dict()
    manifest_temporary_path.write_text(
        json.dumps(lock_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lock_temporary_path.replace(lock_path)
    manifest_temporary_path.replace(lock_manifest_path)
    return lock_path, lock_manifest_path


def build_closure_input_selection_report(
    package_search_root: str | Path,
    *,
    paper_run_name: str,
    target_fpr: float,
    randomization_repeat_id: str | None = None,
    root: str | Path = ".",
    write_lock: bool = False,
) -> dict[str, Any]:
    """解析10类精确包, 可选择在正式执行时写出锁和独立 manifest."""

    resolved_paper_run = normalize_paper_run_name(paper_run_name)
    expected_target_fpr = float(target_fpr)
    expected_repeat = resolve_formal_randomization_repeat(
        randomization_repeat_id
    )
    if not math.isfinite(expected_target_fpr) or not 0.0 < expected_target_fpr < 1.0:
        raise ClosurePackageSelectionError("target_fpr 必须是位于 (0, 1) 的有限数值")
    search_root = Path(package_search_root).expanduser()
    if not search_root.is_dir():
        raise ClosurePackageSelectionError("package_search_root 必须是存在的目录")
    repository_root = Path(root).resolve()

    selected_candidates: list[ClosurePackageCandidate] = []
    for spec in CLOSURE_PACKAGE_FAMILY_SPECS:
        matching_paths = sorted(
            (
                path
                for path in search_root.rglob("*")
                if fnmatch.fnmatchcase(path.name, spec.filename_pattern)
            ),
            key=lambda path: path.as_posix(),
        )
        valid_candidates: list[ClosurePackageCandidate] = []
        rejection_messages: list[str] = []
        for candidate_path in matching_paths:
            try:
                candidate = inspect_closure_package(
                    candidate_path,
                    spec=spec,
                    paper_run_name=resolved_paper_run,
                    target_fpr=expected_target_fpr,
                    randomization_repeat_id=(
                        expected_repeat.randomization_repeat_id
                    ),
                )
                _validate_candidate_repository_profile(
                    candidate,
                    repository_root=repository_root,
                )
                valid_candidates.append(candidate)
            except ClosurePackageSelectionError as error:
                rejection_messages.append(f"{candidate_path.as_posix()}={error}")
        if not valid_candidates:
            rejection_text = ";".join(rejection_messages[:5]) or "没有匹配文件"
            raise ClosurePackageSelectionError(
                f"{spec.package_family} 缺少有效闭合输入包: {rejection_text}"
            )
        selected_candidates.append(
            _select_latest_candidate(
                valid_candidates,
                package_family=spec.package_family,
            )
        )

    if len(selected_candidates) != len(CLOSURE_PACKAGE_FAMILY_SPECS) or len(
        {candidate.package_family for candidate in selected_candidates}
    ) != len(CLOSURE_PACKAGE_FAMILY_SPECS):
        raise ClosurePackageSelectionError("闭合输入必须恰好覆盖10个互异 package family")
    active_repeat_candidates = [
        candidate
        for candidate in selected_candidates
        if candidate.package_family in RANDOMIZATION_REPEAT_PACKAGE_FAMILIES
    ]
    invariant_candidates = [
        candidate
        for candidate in selected_candidates
        if candidate.package_family in CROSS_REPEAT_INVARIANT_PACKAGE_FAMILIES
    ]
    if (
        len(active_repeat_candidates)
        != len(RANDOMIZATION_REPEAT_PACKAGE_FAMILIES)
        or any(
            candidate.randomization_repeat_id
            != expected_repeat.randomization_repeat_id
            or candidate.randomization_scope != "active_repeat_component"
            for candidate in active_repeat_candidates
        )
        or len(invariant_candidates)
        != len(CROSS_REPEAT_INVARIANT_PACKAGE_FAMILIES)
        or any(
            candidate.randomization_scope != "cross_repeat_invariant"
            for candidate in invariant_candidates
        )
    ):
        raise ClosurePackageSelectionError(
            "闭合输入未精确覆盖活动 repeat 证据与跨 repeat 不变证据"
        )
    _validate_semantic_watermark_session_group(selected_candidates)
    common_code_versions = {
        normalize_clean_code_version(candidate.code_version)
        for candidate in selected_candidates
    }
    if len(common_code_versions) != 1:
        raise ClosurePackageSelectionError("10个闭合输入包必须共享同一 clean Git code_version")
    common_code_version = next(iter(common_code_versions))
    repository_code_version = normalize_clean_code_version(
        resolve_code_version(repository_root)
    )
    if common_code_version != repository_code_version:
        raise ClosurePackageSelectionError(
            "10个闭合输入包的 code_version 必须匹配当前 clean 仓库提交"
        )
    lock_records = [candidate.to_lock_record() for candidate in selected_candidates]
    formal_execution_run_lock_digests = {
        candidate.package_family: candidate.formal_execution_run_lock_digest
        for candidate in selected_candidates
    }
    formal_execution_package_lock_digests = {
        candidate.package_family: candidate.formal_execution_package_lock_digest
        for candidate in selected_candidates
    }
    lock_payload: dict[str, Any] = {
        "paper_run_name": resolved_paper_run,
        "target_fpr": expected_target_fpr,
        "common_code_version": common_code_version,
        "closure_input_package_count": len(lock_records),
        "closure_input_packages": lock_records,
        "formal_execution_run_lock_digests": formal_execution_run_lock_digests,
        "formal_execution_package_lock_digests": (
            formal_execution_package_lock_digests
        ),
        "randomization_repeat_identity": {
            **expected_repeat.to_dict(),
            "formal_randomization_protocol_digest": (
                formal_randomization_protocol_record()[
                    "formal_randomization_protocol_digest"
                ]
            ),
        },
        "repeat_component_input_ready": True,
        "randomization_aggregate_ready": False,
        "supports_paper_claim": False,
    }
    lock_payload["closure_input_lock_digest"] = _stable_digest(lock_payload)
    lock_path = repository_root / LOCK_OUTPUT_ROOT / resolved_paper_run / LOCK_FILENAME
    lock_manifest_path = (
        repository_root / LOCK_OUTPUT_ROOT / resolved_paper_run / LOCK_MANIFEST_FILENAME
    )
    if write_lock:
        lock_path, lock_manifest_path = _write_closure_input_lock(
            repository_root=repository_root,
            paper_run_name=resolved_paper_run,
            target_fpr=expected_target_fpr,
            common_code_version=common_code_version,
            lock_payload=lock_payload,
            lock_records=lock_records,
        )
    return {
        **lock_payload,
        "selected_package_paths": [record["package_path"] for record in lock_records],
        "closure_input_lock_path": lock_path.as_posix(),
        "closure_input_lock_manifest_path": lock_manifest_path.as_posix(),
        "closure_input_lock_written": bool(write_lock),
        "closure_input_selection_ready": True,
        "repeat_component_input_ready": True,
        "randomization_aggregate_ready": False,
        "supports_paper_claim": False,
    }


def select_and_lock_closure_input_packages(
    package_search_root: str | Path,
    *,
    paper_run_name: str,
    target_fpr: float,
    randomization_repeat_id: str | None = None,
    root: str | Path = ".",
) -> tuple[str, ...]:
    """正式选择并冻结10个上游包, 返回其显式绝对路径."""

    report = build_closure_input_selection_report(
        package_search_root,
        paper_run_name=paper_run_name,
        target_fpr=target_fpr,
        randomization_repeat_id=randomization_repeat_id,
        root=root,
        write_lock=True,
    )
    return tuple(str(path) for path in report["selected_package_paths"])
