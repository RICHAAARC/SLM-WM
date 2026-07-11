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
from typing import Any
import zlib
from zipfile import BadZipFile, ZipFile

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.protocol.paper_run_config import normalize_paper_run_name
from experiments.runtime.repository_environment import resolve_code_version


LOCK_OUTPUT_ROOT = Path("outputs/paper_result_closure")
LOCK_FILENAME = "closure_input_lock.json"
LOCK_MANIFEST_FILENAME = "input_lock_manifest.local.json"
MAX_GOVERNANCE_MEMBER_BYTES = 32 * 1024 * 1024
CLEAN_CODE_VERSION_PATTERN = re.compile(r"^[0-9a-fA-F]{7,40}$")


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
    value_requirements: tuple[JsonValueRequirement, ...]
    baseline_rows_sources: tuple[BaselineRowsSource, ...] = ()
    package_input_manifest_template: str | None = None


@dataclass(frozen=True)
class ClosurePackageCandidate:
    """保存一个已通过包内治理校验的闭合输入候选."""

    package_family: str
    package_path: Path
    package_sha256: str
    paper_run_name: str
    target_fpr: float
    code_version: str
    generated_at: str
    generated_at_utc: datetime

    def to_lock_record(self) -> dict[str, Any]:
        """转换为持久化锁文件中的稳定记录."""

        return {
            "package_family": self.package_family,
            "package_path": self.package_path.resolve().as_posix(),
            "package_sha256": self.package_sha256,
            "paper_run_name": self.paper_run_name,
            "target_fpr": self.target_fpr,
            "code_version": self.code_version,
            "generated_at": self.generated_at,
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


IMAGE_RUNTIME_PREFIX = "outputs/image_only_dataset_runtime/{paper_run}/"
IMAGE_RUNTIME_SUMMARY = IMAGE_RUNTIME_PREFIX + "dataset_runtime_summary.json"
IMAGE_RUNTIME_MANIFEST = IMAGE_RUNTIME_PREFIX + "manifest.local.json"

ABLATION_PREFIX = "outputs/formal_mechanism_ablation/{paper_run}/"
ABLATION_SUMMARY = ABLATION_PREFIX + "ablation_claim_summary.json"
ABLATION_MANIFEST = ABLATION_PREFIX + "manifest.local.json"

QUALITY_PREFIX = "outputs/dataset_level_quality/{paper_run}/"
QUALITY_SUMMARY = QUALITY_PREFIX + "dataset_quality_summary.json"
QUALITY_MANIFEST = QUALITY_PREFIX + "manifest.local.json"


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
        value_requirements=(
            _require(summary, "run_decision", "pass"),
            _require(summary, "external_baseline_method_faithful_ready", True),
            _require(summary, "primary_baseline_adapter_ready", True),
            _require(transfer_manifest, "transfer_ready", True),
        ),
        package_input_manifest_template=package_input,
    )


def _official_reference_spec(baseline_id: str) -> ClosurePackageFamilySpec:
    """构造一个官方原始环境补充参考包的身份契约."""

    prefix = f"outputs/{baseline_id}_official_reference/{{paper_run}}/"
    summary = prefix + f"{baseline_id}_official_reference_summary.json"
    run_manifest = prefix + "manifest.local.json"
    package_input = prefix + f"{baseline_id}_official_reference_package_input_manifest.json"
    archive_manifest = prefix + f"{baseline_id}_official_reference_archive_manifest.local.json"
    records = prefix + f"{baseline_id}_official_reference_records.jsonl"
    validation = prefix + f"{baseline_id}_official_reference_validation_report.json"
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
        value_requirements=(
            _require(summary, "run_decision", "pass"),
            _require(summary, f"{baseline_id}_official_reference_ready", True),
            _require(summary, "reference_import_ready", True),
            JsonValueRequirement(
                source=_source(run_manifest, "metadata", "run_decision"),
                expected_value="pass",
            ),
        ),
        baseline_rows_sources=(BaselineRowsSource(records),),
        package_input_manifest_template=package_input,
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
        ),
        manifest_member_template=IMAGE_RUNTIME_MANIFEST,
        manifest_artifact_id_template="{paper_run}_image_only_dataset_runtime_manifest",
        generated_at_source=_source(IMAGE_RUNTIME_SUMMARY, "generated_at"),
        paper_run_sources=(
            _source(IMAGE_RUNTIME_SUMMARY, "paper_run_name"),
            _source(IMAGE_RUNTIME_MANIFEST, "config", "paper_run", "run_name"),
        ),
        target_fpr_sources=(
            _source(IMAGE_RUNTIME_SUMMARY, "target_fpr"),
            _source(IMAGE_RUNTIME_MANIFEST, "config", "paper_run", "target_fpr"),
        ),
        baseline_sources=(),
        code_version_sources=(_source(IMAGE_RUNTIME_MANIFEST, "code_version"),),
        value_requirements=(
            _require(IMAGE_RUNTIME_SUMMARY, "protocol_decision", "pass"),
            _require(IMAGE_RUNTIME_SUMMARY, "full_method_claim_ready", True),
            _require(IMAGE_RUNTIME_SUMMARY, "supports_paper_claim", True),
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
            ABLATION_SUMMARY,
            ABLATION_MANIFEST,
        ),
        manifest_member_template=ABLATION_MANIFEST,
        manifest_artifact_id_template="formal_mechanism_ablation_manifest",
        generated_at_source=_source(ABLATION_SUMMARY, "generated_at"),
        paper_run_sources=(_source(ABLATION_SUMMARY, "paper_run_name"),),
        target_fpr_sources=(
            _source(ABLATION_SUMMARY, "target_fpr"),
            _source(ABLATION_MANIFEST, "config", "target_fpr"),
        ),
        baseline_sources=(),
        code_version_sources=(_source(ABLATION_MANIFEST, "code_version"),),
        value_requirements=(
            _require(ABLATION_SUMMARY, "protocol_decision", "pass"),
            _require(ABLATION_SUMMARY, "ablation_claim_gate_ready", True),
            _require(ABLATION_SUMMARY, "supports_paper_claim", True),
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
        ),
        manifest_member_template=QUALITY_MANIFEST,
        manifest_artifact_id_template="dataset_level_quality_manifest",
        generated_at_source=_source(QUALITY_SUMMARY, "generated_at"),
        paper_run_sources=(
            _source(QUALITY_SUMMARY, "paper_run_name"),
            _source(QUALITY_MANIFEST, "metadata", "paper_run_name"),
        ),
        target_fpr_sources=(
            _source(QUALITY_SUMMARY, "target_fpr"),
            _source(QUALITY_MANIFEST, "metadata", "target_fpr"),
        ),
        baseline_sources=(),
        code_version_sources=(_source(QUALITY_MANIFEST, "code_version"),),
        value_requirements=(
            _require(QUALITY_SUMMARY, "formal_feature_backend_ready", True),
            _require(QUALITY_SUMMARY, "formal_sample_scale_ready", True),
            _require(QUALITY_SUMMARY, "canonical_formal_feature_extractor_ready", True),
            _require(QUALITY_SUMMARY, "formal_fid_kid_claim_gate_ready", True),
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
        value_requirements=(
            _require(T2SMARK_SUMMARY, "run_decision", "pass"),
            _require(T2SMARK_SUMMARY, "t2smark_formal_reproduction_ready", True),
            _require(T2SMARK_SUMMARY, "formal_import_validation_ready", True),
            _require(T2SMARK_SUMMARY, "t2smark_formal_attack_ready", True),
            _require(T2SMARK_SUMMARY, "t2smark_strict_pair_quality_ready", True),
        ),
        baseline_rows_sources=(BaselineRowsSource(T2SMARK_CANDIDATES),),
        package_input_manifest_template=T2SMARK_PACKAGE_INPUT,
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
    """把受治理 Git 提交标识规范化为小写纯十六进制文本.

    正式结果包只接受7至40位的规范短提交或完整提交标识. 该边界会
    明确拒绝 ``-dirty``、不可用降级值和任意自由文本, 避免不同代码状态被
    聚合为同一次论文证据闭合.
    """

    if not isinstance(value, str):
        raise ClosurePackageSelectionError("code_version 必须是 clean Git 提交标识")
    normalized = value.strip()
    if CLEAN_CODE_VERSION_PATTERN.fullmatch(normalized) is None:
        raise ClosurePackageSelectionError("code_version 必须是7至40位纯十六进制 clean Git 提交标识")
    return normalized.lower()


def _archive_member_sha256(archive: ZipFile, member_name: str) -> str:
    """流式计算 ZIP 成员摘要, 支持图像等大型动态证据成员."""

    digest = hashlib.sha256()
    with archive.open(member_name) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if not has_paths and not has_digests:
        return
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
) -> ClosurePackageCandidate:
    """按指定 family 的包内身份和证据契约校验单个 ZIP."""

    resolved_paper_run = normalize_paper_run_name(paper_run_name)
    expected_target_fpr = float(target_fpr)
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
        generated_at=generated_at,
        generated_at_utc=generated_at_utc,
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
) -> dict[str, str]:
    """复验 CPU 结果闭合输入锁的基本身份与规范摘要.

    该函数属于通用跨产物治理写法.它不重新读取大型 ZIP, 只验证在下游
    records 与协议产物中必须稳定传播的 run,FPR,精确包集合,代码版本和
    锁摘要.最终完整打包器仍负责逐包字节摘要复核.
    """

    resolved_run_name = normalize_paper_run_name(paper_run_name)
    expected_target_fpr = float(target_fpr)
    expected_families = {spec.package_family for spec in CLOSURE_PACKAGE_FAMILY_SPECS}
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
        for record in records
    )
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
    }


def load_validated_closure_input_lock(
    root: str | Path,
    *,
    paper_run_name: str,
    target_fpr: float,
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
    root: str | Path = ".",
    write_lock: bool = False,
) -> dict[str, Any]:
    """解析10类精确包, 可选择在正式执行时写出锁和独立 manifest."""

    resolved_paper_run = normalize_paper_run_name(paper_run_name)
    expected_target_fpr = float(target_fpr)
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
                valid_candidates.append(
                    inspect_closure_package(
                        candidate_path,
                        spec=spec,
                        paper_run_name=resolved_paper_run,
                        target_fpr=expected_target_fpr,
                    )
                )
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
    common_code_versions = {
        normalize_clean_code_version(candidate.code_version)
        for candidate in selected_candidates
    }
    if len(common_code_versions) != 1:
        raise ClosurePackageSelectionError("10个闭合输入包必须共享同一 clean Git code_version")
    common_code_version = next(iter(common_code_versions))
    lock_records = [candidate.to_lock_record() for candidate in selected_candidates]
    lock_payload: dict[str, Any] = {
        "paper_run_name": resolved_paper_run,
        "target_fpr": expected_target_fpr,
        "common_code_version": common_code_version,
        "closure_input_package_count": len(lock_records),
        "closure_input_packages": lock_records,
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
    }


def select_and_lock_closure_input_packages(
    package_search_root: str | Path,
    *,
    paper_run_name: str,
    target_fpr: float,
    root: str | Path = ".",
) -> tuple[str, ...]:
    """正式选择并冻结10个上游包, 返回其显式绝对路径."""

    report = build_closure_input_selection_report(
        package_search_root,
        paper_run_name=paper_run_name,
        target_fpr=target_fpr,
        root=root,
        write_lock=True,
    )
    return tuple(str(path) for path in report["selected_package_paths"])
