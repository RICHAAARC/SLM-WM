"""从已验证的精确9+3聚合来源中读取跨重复原始记录.

该模块只提供受限的临时工作区与记录迭代接口. 公共入口只能接收
``RandomizationAggregateProvenance``. 工作区进入时会先在创建临时目录前
重新调用聚合包生产 validator, 再逐字段比较调用方对象与复验对象. 随后才把
聚合 ZIP 复制到内部临时目录, 并对每个单重复组件和每个 leaf ZIP 再次调用
生产 validator 或 inspector.

工作区不会接受任意输入路径或成员路径, 也不会向调用方返回临时文件路径.
调用方只能从冻结的成员登记中选择记录源, 从而避免把包内自由路径解释为
文件系统路径. 该层只暴露原始事实, 不计算论文统计量, 不写持久产物, 也不
支持论文结论.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, BinaryIO, Iterator, Mapping
from zipfile import BadZipFile, ZipFile

from experiments.protocol.formal_randomization import (
    formal_randomization_repeat_ids,
    resolve_formal_randomization_repeat,
)
from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    ClosurePackageCandidate,
    inspect_closure_package,
)
from paper_experiments.runners.randomization_aggregate_provenance import (
    RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES,
    RandomizationAggregateProvenance,
    validate_randomization_aggregate_provenance,
)
from paper_experiments.runners.randomization_repeat_evidence import (
    RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES,
    validate_randomization_repeat_evidence_package,
)


RECORD_FORMAT_JSONL = "jsonl_objects"
RECORD_FORMAT_JSON_ARRAY = "json_array_objects"
RECORD_FORMAT_JSON_OBJECT = "json_object"
RECORD_FORMAT_RAW_BYTES = "raw_bytes"

RECORD_GROUP_OBSERVATION = "observation"
RECORD_GROUP_ABLATION = "ablation"
RECORD_GROUP_SENSITIVITY = "sensitivity"
RECORD_GROUP_QUALITY = "quality"
RECORD_GROUP_THRESHOLD_BINDING = "threshold_binding"
RECORD_GROUP_REFERENCE = "reference"
RECORD_GROUP_PROMPT_RUNTIME = "prompt_runtime"
RECORD_GROUP_PROMPT_SOURCE = "prompt_source"
RECORD_GROUP_RUN_MANIFEST = "run_manifest"


class RandomizationAggregateRecordWorkspaceError(ValueError):
    """表示聚合原始记录工作区未满足不可绕过的来源边界."""


@dataclass(frozen=True)
class RandomizationAggregateRecordSource:
    """描述一个只能由工作区内部规范映射定位的原始记录成员."""

    randomization_scope: str
    randomization_repeat_id: str
    package_family: str
    record_group: str
    record_role: str
    record_format: str
    record_member: str
    record_sha256: str
    leaf_package_sha256: str
    randomization_repeat_component_sha256: str
    randomization_repeat_evidence_manifest_digest: str
    component_content_digest: str
    randomization_aggregate_package_sha256: str
    common_code_version: str
    randomization_aggregate_digest: str


@dataclass(frozen=True)
class RandomizationAggregateQualityFeaturePair:
    """绑定同一质量图像记录的 source/comparison 原始 feature."""

    randomization_repeat_id: str
    dataset_quality_record_id: str
    image_record_source: RandomizationAggregateRecordSource
    feature_record_source: RandomizationAggregateRecordSource
    image_record: Mapping[str, Any]
    source_feature_record: Mapping[str, Any]
    comparison_feature_record: Mapping[str, Any]


@dataclass(frozen=True)
class _RecordMemberSpec:
    """登记一个 leaf family 中可被正式统计层消费的规范成员."""

    package_family: str
    record_group: str
    record_role: str
    member_template: str
    record_format: str
    randomization_scope: str = "active_repeat_component"


_ACTIVE_RECORD_MEMBER_SPECS = (
    _RecordMemberSpec(
        package_family="image_only_dataset_runtime",
        record_group=RECORD_GROUP_RUN_MANIFEST,
        record_role="semantic_watermark_dataset_manifest",
        member_template=(
            "outputs/image_only_dataset_runtime/{paper_run}/"
            "manifest.local.json"
        ),
        record_format=RECORD_FORMAT_JSON_OBJECT,
    ),
    _RecordMemberSpec(
        package_family="image_only_dataset_runtime",
        record_group=RECORD_GROUP_PROMPT_RUNTIME,
        record_role="semantic_watermark_runtime_record",
        member_template=(
            "outputs/image_only_dataset_runtime/{paper_run}/"
            "runtime_results.jsonl"
        ),
        record_format=RECORD_FORMAT_JSONL,
    ),
    _RecordMemberSpec(
        package_family="image_only_dataset_runtime",
        record_group=RECORD_GROUP_PROMPT_SOURCE,
        record_role="governed_prompt_file_bytes",
        member_template=(
            "outputs/image_only_dataset_runtime/{paper_run}/"
            "prompt_source_snapshot/paper_main_{paper_run}_prompts.txt"
        ),
        record_format=RECORD_FORMAT_RAW_BYTES,
    ),
    _RecordMemberSpec(
        package_family="image_only_dataset_runtime",
        record_group=RECORD_GROUP_PROMPT_SOURCE,
        record_role="governed_prompt_selection_manifest_bytes",
        member_template=(
            "outputs/image_only_dataset_runtime/{paper_run}/"
            "prompt_source_snapshot/prompt_selection_manifest.jsonl"
        ),
        record_format=RECORD_FORMAT_RAW_BYTES,
    ),
    _RecordMemberSpec(
        package_family="image_only_dataset_runtime",
        record_group=RECORD_GROUP_PROMPT_SOURCE,
        record_role="governed_prompt_source_registry_bytes",
        member_template=(
            "outputs/image_only_dataset_runtime/{paper_run}/"
            "prompt_source_snapshot/prompt_source_registry.json"
        ),
        record_format=RECORD_FORMAT_RAW_BYTES,
    ),
    _RecordMemberSpec(
        package_family="image_only_dataset_runtime",
        record_group=RECORD_GROUP_OBSERVATION,
        record_role="semantic_watermark_detection_observation",
        member_template=(
            "outputs/image_only_dataset_runtime/{paper_run}/"
            "image_only_detection_records.jsonl"
        ),
        record_format=RECORD_FORMAT_JSONL,
    ),
    _RecordMemberSpec(
        package_family="image_only_dataset_runtime",
        record_group=RECORD_GROUP_THRESHOLD_BINDING,
        record_role="semantic_watermark_frozen_evidence_protocol",
        member_template=(
            "outputs/image_only_dataset_runtime/{paper_run}/"
            "frozen_evidence_protocol.json"
        ),
        record_format=RECORD_FORMAT_JSON_OBJECT,
    ),
    _RecordMemberSpec(
        package_family="method_faithful_tree_ring",
        record_group=RECORD_GROUP_RUN_MANIFEST,
        record_role="tree_ring_baseline_run_manifest",
        member_template=(
            "outputs/external_baseline_method_faithful/{paper_run}/"
            "run_records/tree_ring/tree_ring_manifest.local.json"
        ),
        record_format=RECORD_FORMAT_JSON_OBJECT,
    ),
    _RecordMemberSpec(
        package_family="method_faithful_tree_ring",
        record_group=RECORD_GROUP_OBSERVATION,
        record_role="tree_ring_baseline_observation",
        member_template=(
            "outputs/external_baseline_method_faithful/{paper_run}/"
            "split_observations/tree_ring_baseline_observations.json"
        ),
        record_format=RECORD_FORMAT_JSON_ARRAY,
    ),
    _RecordMemberSpec(
        package_family="method_faithful_tree_ring",
        record_group=RECORD_GROUP_THRESHOLD_BINDING,
        record_role="tree_ring_baseline_transfer_manifest",
        member_template=(
            "outputs/external_baseline_method_faithful/{paper_run}/"
            "split_observations/tree_ring_baseline_transfer_manifest.json"
        ),
        record_format=RECORD_FORMAT_JSON_OBJECT,
    ),
    _RecordMemberSpec(
        package_family="method_faithful_gaussian_shading",
        record_group=RECORD_GROUP_RUN_MANIFEST,
        record_role="gaussian_shading_baseline_run_manifest",
        member_template=(
            "outputs/external_baseline_method_faithful/{paper_run}/"
            "run_records/gaussian_shading/"
            "gaussian_shading_manifest.local.json"
        ),
        record_format=RECORD_FORMAT_JSON_OBJECT,
    ),
    _RecordMemberSpec(
        package_family="method_faithful_gaussian_shading",
        record_group=RECORD_GROUP_OBSERVATION,
        record_role="gaussian_shading_baseline_observation",
        member_template=(
            "outputs/external_baseline_method_faithful/{paper_run}/"
            "split_observations/gaussian_shading_baseline_observations.json"
        ),
        record_format=RECORD_FORMAT_JSON_ARRAY,
    ),
    _RecordMemberSpec(
        package_family="method_faithful_gaussian_shading",
        record_group=RECORD_GROUP_THRESHOLD_BINDING,
        record_role="gaussian_shading_baseline_transfer_manifest",
        member_template=(
            "outputs/external_baseline_method_faithful/{paper_run}/"
            "split_observations/gaussian_shading_baseline_transfer_manifest.json"
        ),
        record_format=RECORD_FORMAT_JSON_OBJECT,
    ),
    _RecordMemberSpec(
        package_family="method_faithful_shallow_diffuse",
        record_group=RECORD_GROUP_RUN_MANIFEST,
        record_role="shallow_diffuse_baseline_run_manifest",
        member_template=(
            "outputs/external_baseline_method_faithful/{paper_run}/"
            "run_records/shallow_diffuse/"
            "shallow_diffuse_manifest.local.json"
        ),
        record_format=RECORD_FORMAT_JSON_OBJECT,
    ),
    _RecordMemberSpec(
        package_family="method_faithful_shallow_diffuse",
        record_group=RECORD_GROUP_OBSERVATION,
        record_role="shallow_diffuse_baseline_observation",
        member_template=(
            "outputs/external_baseline_method_faithful/{paper_run}/"
            "split_observations/shallow_diffuse_baseline_observations.json"
        ),
        record_format=RECORD_FORMAT_JSON_ARRAY,
    ),
    _RecordMemberSpec(
        package_family="method_faithful_shallow_diffuse",
        record_group=RECORD_GROUP_THRESHOLD_BINDING,
        record_role="shallow_diffuse_baseline_transfer_manifest",
        member_template=(
            "outputs/external_baseline_method_faithful/{paper_run}/"
            "split_observations/shallow_diffuse_baseline_transfer_manifest.json"
        ),
        record_format=RECORD_FORMAT_JSON_OBJECT,
    ),
    _RecordMemberSpec(
        package_family="official_reference_t2smark",
        record_group=RECORD_GROUP_RUN_MANIFEST,
        record_role="t2smark_baseline_run_manifest",
        member_template=(
            "outputs/t2smark_formal_reproduction/{paper_run}/"
            "t2smark_formal_reproduction_manifest.local.json"
        ),
        record_format=RECORD_FORMAT_JSON_OBJECT,
    ),
    _RecordMemberSpec(
        package_family="official_reference_t2smark",
        record_group=RECORD_GROUP_OBSERVATION,
        record_role="t2smark_baseline_observation",
        member_template=(
            "outputs/t2smark_formal_reproduction/{paper_run}/"
            "t2smark_adapter/baseline_observations.json"
        ),
        record_format=RECORD_FORMAT_JSON_ARRAY,
    ),
    _RecordMemberSpec(
        package_family="official_reference_t2smark",
        record_group=RECORD_GROUP_THRESHOLD_BINDING,
        record_role="t2smark_formal_import_candidate_record",
        member_template=(
            "outputs/t2smark_formal_reproduction/{paper_run}/"
            "t2smark_formal_import_candidate_records.jsonl"
        ),
        record_format=RECORD_FORMAT_JSONL,
    ),
    _RecordMemberSpec(
        package_family="runtime_rerun_ablation",
        record_group=RECORD_GROUP_ABLATION,
        record_role="ablation_runtime_record",
        member_template=(
            "outputs/formal_mechanism_ablation/{paper_run}/"
            "runtime_rerun_records.jsonl"
        ),
        record_format=RECORD_FORMAT_JSONL,
    ),
    _RecordMemberSpec(
        package_family="runtime_rerun_ablation",
        record_group=RECORD_GROUP_ABLATION,
        record_role="ablation_detection_record",
        member_template=(
            "outputs/formal_mechanism_ablation/{paper_run}/"
            "formal_detection_records.jsonl"
        ),
        record_format=RECORD_FORMAT_JSONL,
    ),
    _RecordMemberSpec(
        package_family="runtime_rerun_ablation",
        record_group=RECORD_GROUP_ABLATION,
        record_role="ablation_frozen_protocol",
        member_template=(
            "outputs/formal_mechanism_ablation/{paper_run}/"
            "per_ablation_frozen_protocols.json"
        ),
        record_format=RECORD_FORMAT_JSON_OBJECT,
    ),
    _RecordMemberSpec(
        package_family="branch_risk_parameter_sensitivity",
        record_group=RECORD_GROUP_SENSITIVITY,
        record_role="parameter_sensitivity_runtime_record",
        member_template=(
            "outputs/formal_branch_risk_sensitivity/{paper_run}/"
            "parameter_sensitivity_records.jsonl"
        ),
        record_format=RECORD_FORMAT_JSONL,
    ),
    _RecordMemberSpec(
        package_family="branch_risk_parameter_sensitivity",
        record_group=RECORD_GROUP_SENSITIVITY,
        record_role="parameter_sensitivity_detection_record",
        member_template=(
            "outputs/formal_branch_risk_sensitivity/{paper_run}/"
            "formal_detection_records.jsonl"
        ),
        record_format=RECORD_FORMAT_JSONL,
    ),
    _RecordMemberSpec(
        package_family="branch_risk_parameter_sensitivity",
        record_group=RECORD_GROUP_SENSITIVITY,
        record_role="parameter_sensitivity_frozen_protocol",
        member_template=(
            "outputs/formal_branch_risk_sensitivity/{paper_run}/"
            "per_setting_frozen_protocols.json"
        ),
        record_format=RECORD_FORMAT_JSON_OBJECT,
    ),
    _RecordMemberSpec(
        package_family="branch_risk_parameter_sensitivity",
        record_group=RECORD_GROUP_RUN_MANIFEST,
        record_role="parameter_sensitivity_run_manifest",
        member_template=(
            "outputs/formal_branch_risk_sensitivity/{paper_run}/"
            "manifest.local.json"
        ),
        record_format=RECORD_FORMAT_JSON_OBJECT,
    ),
    _RecordMemberSpec(
        package_family="runtime_rerun_ablation",
        record_group=RECORD_GROUP_RUN_MANIFEST,
        record_role="ablation_run_manifest",
        member_template=(
            "outputs/formal_mechanism_ablation/{paper_run}/"
            "manifest.local.json"
        ),
        record_format=RECORD_FORMAT_JSON_OBJECT,
    ),
    _RecordMemberSpec(
        package_family="dataset_level_quality",
        record_group=RECORD_GROUP_QUALITY,
        record_role="quality_image_record",
        member_template=(
            "outputs/dataset_level_quality/{paper_run}/"
            "dataset_quality_image_records.jsonl"
        ),
        record_format=RECORD_FORMAT_JSONL,
    ),
    _RecordMemberSpec(
        package_family="dataset_level_quality",
        record_group=RECORD_GROUP_QUALITY,
        record_role="quality_image_resolution_record",
        member_template=(
            "outputs/dataset_level_quality/{paper_run}/"
            "dataset_quality_image_resolution_records.jsonl"
        ),
        record_format=RECORD_FORMAT_JSONL,
    ),
    _RecordMemberSpec(
        package_family="dataset_level_quality",
        record_group=RECORD_GROUP_QUALITY,
        record_role="quality_feature_record",
        member_template=(
            "outputs/dataset_level_quality/{paper_run}/"
            "dataset_quality_formal_feature_records.jsonl"
        ),
        record_format=RECORD_FORMAT_JSONL,
    ),
    _RecordMemberSpec(
        package_family="dataset_level_quality",
        record_group=RECORD_GROUP_QUALITY,
        record_role="attack_quality_image_record",
        member_template=(
            "outputs/dataset_level_quality/{paper_run}/"
            "attack_conditioned_quality/"
            "attack_conditioned_quality_image_records.jsonl"
        ),
        record_format=RECORD_FORMAT_JSONL,
    ),
    _RecordMemberSpec(
        package_family="dataset_level_quality",
        record_group=RECORD_GROUP_QUALITY,
        record_role="attack_quality_pair_record",
        member_template=(
            "outputs/dataset_level_quality/{paper_run}/"
            "attack_conditioned_quality/"
            "attack_conditioned_quality_pair_records.jsonl"
        ),
        record_format=RECORD_FORMAT_JSONL,
    ),
    _RecordMemberSpec(
        package_family="dataset_level_quality",
        record_group=RECORD_GROUP_QUALITY,
        record_role="attack_quality_inception_feature_record",
        member_template=(
            "outputs/dataset_level_quality/{paper_run}/"
            "attack_conditioned_quality/"
            "attack_conditioned_quality_inception_feature_records.jsonl"
        ),
        record_format=RECORD_FORMAT_JSONL,
    ),
    _RecordMemberSpec(
        package_family="dataset_level_quality",
        record_group=RECORD_GROUP_QUALITY,
        record_role="paired_quality_clip_feature_record",
        member_template=(
            "outputs/dataset_level_quality/{paper_run}/"
            "attack_conditioned_quality/"
            "paired_quality_clip_feature_records.jsonl"
        ),
        record_format=RECORD_FORMAT_JSONL,
    ),
    _RecordMemberSpec(
        package_family="dataset_level_quality",
        record_group=RECORD_GROUP_QUALITY,
        record_role="paired_quality_metric_record",
        member_template=(
            "outputs/dataset_level_quality/{paper_run}/"
            "attack_conditioned_quality/paired_quality_metric_records.jsonl"
        ),
        record_format=RECORD_FORMAT_JSONL,
    ),
)

_INVARIANT_RECORD_MEMBER_SPECS = tuple(
    _RecordMemberSpec(
        package_family=f"official_reference_{baseline_id}",
        record_group=RECORD_GROUP_REFERENCE,
        record_role=f"{baseline_id}_official_reference_observation",
        member_template=(
            f"outputs/{baseline_id}_official_reference/{{paper_run}}/"
            f"{baseline_id}_official_reference_records.jsonl"
        ),
        record_format=RECORD_FORMAT_JSONL,
        randomization_scope="cross_repeat_invariant",
    )
    for baseline_id in ("tree_ring", "gaussian_shading", "shallow_diffuse")
)

_RECORD_MEMBER_SPECS = (
    *_ACTIVE_RECORD_MEMBER_SPECS,
    *_INVARIANT_RECORD_MEMBER_SPECS,
)
_SPECIFICATIONS_BY_FAMILY = {
    specification.package_family: specification
    for specification in CLOSURE_PACKAGE_FAMILY_SPECS
}


def _opened_file_sha256(stream: BinaryIO) -> str:
    """在同一已打开句柄上计算完整摘要并复位到文件起点."""

    digest = hashlib.sha256()
    stream.seek(0)
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


@contextmanager
def _open_zip_bound_to_sha256(
    path: Path,
    *,
    expected_sha256: str,
    role: str,
) -> Iterator[ZipFile]:
    """用同一文件句柄完成 ZIP 全字节摘要绑定与成员读取."""

    try:
        with path.open("rb") as stream:
            actual_sha256 = _opened_file_sha256(stream)
            if actual_sha256 != expected_sha256:
                raise RandomizationAggregateRecordWorkspaceError(
                    f"{role} 与外层登记摘要不一致"
                )
            with ZipFile(stream) as archive:
                yield archive
    except (BadZipFile, OSError) as exc:
        if isinstance(exc, RandomizationAggregateRecordWorkspaceError):
            raise
        raise RandomizationAggregateRecordWorkspaceError(
            f"{role} 无法通过固定文件句柄读取"
        ) from exc


def _zip_member_sha256(archive: ZipFile, member_name: str) -> str:
    """流式计算 leaf ZIP 中一个规范成员的摘要."""

    digest = hashlib.sha256()
    with archive.open(member_name, "r") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_zip_member(
    archive: ZipFile,
    member_name: str,
    destination: Path,
    *,
    expected_sha256: str,
) -> None:
    """把固定登记成员复制到内部路径并立即核对字节摘要."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    try:
        with archive.open(member_name, "r") as source, destination.open(
            "xb"
        ) as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                target.write(chunk)
                digest.update(chunk)
    except (KeyError, OSError) as exc:
        destination.unlink(missing_ok=True)
        raise RandomizationAggregateRecordWorkspaceError(
            f"聚合来源缺少或无法复制规范成员: {member_name}"
        ) from exc
    if digest.hexdigest() != expected_sha256:
        destination.unlink(missing_ok=True)
        raise RandomizationAggregateRecordWorkspaceError(
            f"聚合来源规范成员摘要漂移: {member_name}"
        )


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """拒绝普通 JSON parser 会静默覆盖的重复字段."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RandomizationAggregateRecordWorkspaceError(
                f"原始记录包含重复 JSON 字段: {key}"
            )
        result[key] = value
    return result


def _decode_json(payload: bytes, *, source_role: str) -> Any:
    """用严格 UTF-8 与重复字段门禁读取 JSON 值."""

    try:
        return json.loads(
            payload.decode("utf-8-sig"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RandomizationAggregateRecordWorkspaceError(
            f"原始记录成员不是有效 JSON: {source_role}"
        ) from exc


def _plain_value(value: Any) -> Any:
    """把不可变 provenance 容器转换为可精确比较的普通值."""

    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_value(item) for item in value]
    if isinstance(value, Path):
        return value.resolve().as_posix()
    return value


def _freeze(value: Any) -> Any:
    """递归冻结暴露给统计层的原始记录."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _revalidate_provenance(
    source: RandomizationAggregateProvenance,
) -> RandomizationAggregateProvenance:
    """在创建临时目录前重新验证来源并逐字段比较 dataclass."""

    if not isinstance(source, RandomizationAggregateProvenance):
        raise TypeError(
            "聚合记录工作区只接受 RandomizationAggregateProvenance"
        )
    try:
        paper_run_name = str(source.payload["paper_run_name"])
        target_fpr = float(source.payload["target_fpr"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RandomizationAggregateRecordWorkspaceError(
            "调用方 provenance 缺少运行身份"
        ) from exc
    try:
        validated = validate_randomization_aggregate_provenance(
            source.package_path,
            paper_run_name=paper_run_name,
            target_fpr=target_fpr,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RandomizationAggregateRecordWorkspaceError(
            "聚合来源未通过消费时生产复验"
        ) from exc
    mismatched_fields = tuple(
        field.name
        for field in fields(RandomizationAggregateProvenance)
        if _plain_value(getattr(source, field.name))
        != _plain_value(getattr(validated, field.name))
    )
    if mismatched_fields:
        raise RandomizationAggregateRecordWorkspaceError(
            "调用方 provenance 与生产复验对象字段不一致: "
            + ",".join(mismatched_fields)
        )
    return validated


def _validated_repeat_manifest(
    archive: ZipFile,
    *,
    randomization_repeat_id: str,
) -> dict[str, Any]:
    """从已通过生产 validator 的组件中读取严格 manifest."""

    member_name = (
        f"randomization_repeat_evidence/{randomization_repeat_id}/"
        "randomization_repeat_evidence_manifest.json"
    )
    try:
        payload = _decode_json(
            archive.read(member_name),
            source_role=member_name,
        )
    except (KeyError, OSError) as exc:
        raise RandomizationAggregateRecordWorkspaceError(
            f"单重复组件缺少规范 manifest: {randomization_repeat_id}"
        ) from exc
    if not isinstance(payload, dict):
        raise RandomizationAggregateRecordWorkspaceError(
            f"单重复组件 manifest 不是 JSON object: {randomization_repeat_id}"
        )
    return payload


def _candidate_matches_repeat_leaf(
    candidate: ClosurePackageCandidate,
    *,
    leaf_record: Mapping[str, Any],
    randomization_repeat_id: str,
    code_version: str,
) -> bool:
    """核对生产 inspector 返回的活动 leaf 身份与外层锁."""

    repeat = resolve_formal_randomization_repeat(randomization_repeat_id)
    return all(
        (
            candidate.package_family == leaf_record.get("package_family"),
            candidate.package_sha256 == leaf_record.get("package_sha256"),
            candidate.code_version == code_version,
            candidate.formal_execution_run_lock_digest
            == leaf_record.get("formal_execution_run_lock_digest"),
            candidate.formal_execution_package_lock_digest
            == leaf_record.get("formal_execution_package_lock_digest"),
            candidate.randomization_scope == "active_repeat_component",
            candidate.randomization_repeat_id == randomization_repeat_id,
            candidate.generation_seed_index == repeat.generation_seed_index,
            candidate.generation_seed_offset == repeat.generation_seed_offset,
            candidate.watermark_key_index == repeat.watermark_key_index,
        )
    )


def _candidate_matches_invariant(
    candidate: ClosurePackageCandidate,
    *,
    invariant_record: Mapping[str, Any],
    code_version: str,
) -> bool:
    """核对生产 inspector 返回的不变包身份与外层锁."""

    return all(
        (
            candidate.package_family == invariant_record.get("package_family"),
            candidate.package_sha256 == invariant_record.get("package_sha256"),
            candidate.code_version == code_version,
            candidate.formal_execution_run_lock_digest
            == invariant_record.get("formal_execution_run_lock_digest"),
            candidate.formal_execution_package_lock_digest
            == invariant_record.get("formal_execution_package_lock_digest"),
            candidate.randomization_scope == "cross_repeat_invariant",
            candidate.randomization_repeat_id == "",
            candidate.generation_seed_index == -1,
            candidate.generation_seed_offset == -1,
            candidate.watermark_key_index == -1,
        )
    )


class RandomizationAggregateRecordWorkspace:
    """管理一次不可逃逸的跨重复原始记录读取生命周期."""

    def __init__(self, source: RandomizationAggregateProvenance) -> None:
        if not isinstance(source, RandomizationAggregateProvenance):
            raise TypeError(
                "聚合记录工作区只接受 RandomizationAggregateProvenance"
            )
        self._source = source
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._package_paths: dict[tuple[str, str, str], Path] = {}
        self._record_sources: tuple[RandomizationAggregateRecordSource, ...] = ()
        self._source_keys: set[tuple[str, str, str, str, str]] = set()
        self._active = False
        self._closed = False

    def __enter__(self) -> "RandomizationAggregateRecordWorkspace":
        """先复验 provenance, 再创建并填充内部临时目录."""

        if self._active or self._closed:
            raise RandomizationAggregateRecordWorkspaceError(
                "聚合记录工作区实例只能进入一次"
            )
        validated = _revalidate_provenance(self._source)
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="slm_wm_randomization_records_"
        )
        self._temporary_directory = temporary_directory
        try:
            self._prepare_workspace(
                validated,
                Path(temporary_directory.name),
            )
        except Exception:
            temporary_directory.cleanup()
            self._temporary_directory = None
            self._package_paths.clear()
            self._record_sources = ()
            self._source_keys.clear()
            self._closed = True
            raise
        self._active = True
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        """删除内部临时目录并使全部读取方法永久失效."""

        self._active = False
        self._package_paths.clear()
        self._record_sources = ()
        self._source_keys.clear()
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None
        self._closed = True

    def _require_active(self) -> None:
        """阻止在 context 生命周期外读取临时来源."""

        if not self._active:
            raise RandomizationAggregateRecordWorkspaceError(
                "聚合记录工作区只能在 with context 内使用"
            )

    @property
    def record_sources(self) -> tuple[RandomizationAggregateRecordSource, ...]:
        """返回不含文件系统路径的全部规范记录源描述符."""

        self._require_active()
        return self._record_sources

    @property
    def observation_sources(
        self,
    ) -> tuple[RandomizationAggregateRecordSource, ...]:
        """返回精确45个活动 repeat 主方法与 baseline observation 源."""

        self._require_active()
        return tuple(
            source
            for source in self._record_sources
            if source.record_group == RECORD_GROUP_OBSERVATION
        )

    @property
    def ablation_sources(
        self,
    ) -> tuple[RandomizationAggregateRecordSource, ...]:
        """返回逐重复消融记录与冻结协议源."""

        self._require_active()
        return tuple(
            source
            for source in self._record_sources
            if source.record_group == RECORD_GROUP_ABLATION
        )

    @property
    def sensitivity_sources(
        self,
    ) -> tuple[RandomizationAggregateRecordSource, ...]:
        """返回逐重复的参数敏感性运行、检测和冻结协议记录源。"""

        self._require_active()
        return tuple(
            source
            for source in self._record_sources
            if source.record_group == RECORD_GROUP_SENSITIVITY
        )

    @property
    def threshold_binding_sources(
        self,
    ) -> tuple[RandomizationAggregateRecordSource, ...]:
        """返回精确45个活动 repeat 阈值协议或声明来源."""

        self._require_active()
        return tuple(
            source
            for source in self._record_sources
            if source.record_group == RECORD_GROUP_THRESHOLD_BINDING
        )

    @property
    def prompt_runtime_sources(
        self,
    ) -> tuple[RandomizationAggregateRecordSource, ...]:
        """返回9个用于逐字节重建 Prompt exact-set 的 runtime 源."""

        self._require_active()
        return tuple(
            source
            for source in self._record_sources
            if source.record_group == RECORD_GROUP_PROMPT_RUNTIME
        )

    @property
    def prompt_source_sources(
        self,
    ) -> tuple[RandomizationAggregateRecordSource, ...]:
        """返回27个受治理 Prompt 文件、选择清单和来源注册表字节源."""

        self._require_active()
        return tuple(
            source
            for source in self._record_sources
            if source.record_group == RECORD_GROUP_PROMPT_SOURCE
        )

    @property
    def reference_sources(
        self,
    ) -> tuple[RandomizationAggregateRecordSource, ...]:
        """返回3个跨 repeat 官方参考忠实度记录源."""

        self._require_active()
        return tuple(
            source
            for source in self._record_sources
            if source.record_group == RECORD_GROUP_REFERENCE
        )

    @property
    def quality_sources(
        self,
    ) -> tuple[RandomizationAggregateRecordSource, ...]:
        """返回逐重复质量图像身份、解析与 feature 记录源."""

        self._require_active()
        return tuple(
            source
            for source in self._record_sources
            if source.record_group == RECORD_GROUP_QUALITY
        )

    @property
    def quality_feature_sources(
        self,
    ) -> tuple[RandomizationAggregateRecordSource, ...]:
        """只返回逐重复 Inception feature 记录源."""

        self._require_active()
        return tuple(
            source
            for source in self._record_sources
            if source.record_role == "quality_feature_record"
        )

    def iter_quality_feature_pairs(
        self,
        randomization_repeat_id: str,
    ) -> Iterator[RandomizationAggregateQualityFeaturePair]:
        """按质量记录身份联接一个 repeat 的图像与两类原始 feature."""

        self._require_active()
        image_source = self.find_source(
            randomization_repeat_id=randomization_repeat_id,
            package_family="dataset_level_quality",
            record_role="quality_image_record",
        )
        feature_source = self.find_source(
            randomization_repeat_id=randomization_repeat_id,
            package_family="dataset_level_quality",
            record_role="quality_feature_record",
        )
        image_records = tuple(self.iter_records(image_source))
        feature_records = tuple(self.iter_records(feature_source))
        image_by_id: dict[str, Mapping[str, Any]] = {}
        for record in image_records:
            record_id = str(record.get("dataset_quality_record_id", ""))
            if not record_id or record_id in image_by_id:
                raise RandomizationAggregateRecordWorkspaceError(
                    "质量图像记录身份缺失或重复"
                )
            image_by_id[record_id] = record
        feature_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
        for record in feature_records:
            record_id = str(record.get("dataset_quality_record_id", ""))
            image_role = str(record.get("dataset_quality_image_role", ""))
            key = (record_id, image_role)
            if (
                record_id not in image_by_id
                or image_role not in {"source", "comparison"}
                or key in feature_by_key
            ):
                raise RandomizationAggregateRecordWorkspaceError(
                    "质量 feature 与图像记录身份无法唯一联接"
                )
            feature_by_key[key] = record
        expected_keys = {
            (record_id, image_role)
            for record_id in image_by_id
            for image_role in ("source", "comparison")
        }
        if set(feature_by_key) != expected_keys:
            raise RandomizationAggregateRecordWorkspaceError(
                "质量 feature 未精确覆盖全部图像记录与角色"
            )
        return iter(
            tuple(
                RandomizationAggregateQualityFeaturePair(
                    randomization_repeat_id=randomization_repeat_id,
                    dataset_quality_record_id=record_id,
                    image_record_source=image_source,
                    feature_record_source=feature_source,
                    image_record=image_by_id[record_id],
                    source_feature_record=feature_by_key[
                        (record_id, "source")
                    ],
                    comparison_feature_record=feature_by_key[
                        (record_id, "comparison")
                    ],
                )
                for record_id in image_by_id
            )
        )

    def find_source(
        self,
        *,
        randomization_repeat_id: str,
        package_family: str,
        record_role: str,
    ) -> RandomizationAggregateRecordSource:
        """按登记身份查找来源, 不接受或推断任何成员路径."""

        self._require_active()
        matches = tuple(
            source
            for source in self._record_sources
            if source.randomization_repeat_id == randomization_repeat_id
            and source.package_family == package_family
            and source.record_role == record_role
        )
        if len(matches) != 1:
            raise RandomizationAggregateRecordWorkspaceError(
                "未找到唯一的规范聚合记录源"
            )
        return matches[0]

    def iter_records(
        self,
        source: RandomizationAggregateRecordSource,
    ) -> Iterator[Mapping[str, Any]]:
        """读取并冻结一个 JSONL 或 JSON array 记录成员."""

        self._require_active()
        self._require_registered_source(source)
        if source.record_format == RECORD_FORMAT_JSON_OBJECT:
            raise RandomizationAggregateRecordWorkspaceError(
                "JSON object 来源必须使用 read_object"
            )
        payload = self._read_member_bytes(source)
        if source.record_format == RECORD_FORMAT_JSONL:
            rows: list[Mapping[str, Any]] = []
            for line_index, raw_line in enumerate(payload.splitlines(), start=1):
                if not raw_line.strip():
                    continue
                row = _decode_json(
                    raw_line,
                    source_role=f"{source.record_role}:{line_index}",
                )
                if not isinstance(row, dict):
                    raise RandomizationAggregateRecordWorkspaceError(
                        "JSONL 原始记录行必须是 JSON object"
                    )
                rows.append(_freeze(row))
        elif source.record_format == RECORD_FORMAT_JSON_ARRAY:
            value = _decode_json(payload, source_role=source.record_role)
            if not isinstance(value, list) or any(
                not isinstance(row, dict) for row in value
            ):
                raise RandomizationAggregateRecordWorkspaceError(
                    "JSON array 原始记录必须只包含 object"
                )
            rows = [_freeze(row) for row in value]
        else:
            raise RandomizationAggregateRecordWorkspaceError(
                "记录源使用了未登记格式"
            )
        if not rows:
            raise RandomizationAggregateRecordWorkspaceError(
                "正式原始记录成员不得为空"
            )
        return iter(tuple(rows))

    def read_object(
        self,
        source: RandomizationAggregateRecordSource,
    ) -> Mapping[str, Any]:
        """读取并冻结一个登记的 JSON object, 主要用于消融冻结协议."""

        self._require_active()
        self._require_registered_source(source)
        if source.record_format != RECORD_FORMAT_JSON_OBJECT:
            raise RandomizationAggregateRecordWorkspaceError(
                "非 JSON object 来源必须使用 iter_records"
            )
        value = _decode_json(
            self._read_member_bytes(source),
            source_role=source.record_role,
        )
        if not isinstance(value, dict) or not value:
            raise RandomizationAggregateRecordWorkspaceError(
                "正式 JSON object 原始记录不得为空"
            )
        return _freeze(value)

    def read_bytes(
        self,
        source: RandomizationAggregateRecordSource,
    ) -> bytes:
        """读取一个登记的原始字节成员, 不暴露 leaf 临时路径."""

        self._require_active()
        self._require_registered_source(source)
        if source.record_format != RECORD_FORMAT_RAW_BYTES:
            raise RandomizationAggregateRecordWorkspaceError(
                "只有 raw bytes 来源允许使用 read_bytes"
            )
        payload = self._read_member_bytes(source)
        if not payload:
            raise RandomizationAggregateRecordWorkspaceError(
                "正式原始字节成员不得为空"
            )
        return payload

    def _require_registered_source(
        self,
        source: RandomizationAggregateRecordSource,
    ) -> None:
        """拒绝调用方自行构造的成员描述符."""

        if not isinstance(source, RandomizationAggregateRecordSource):
            raise TypeError("记录迭代只接受工作区来源描述符")
        key = self._source_key(source)
        if key not in self._source_keys or source not in self._record_sources:
            raise RandomizationAggregateRecordWorkspaceError(
                "记录来源不是当前工作区登记对象"
            )

    @staticmethod
    def _source_key(
        source: RandomizationAggregateRecordSource,
    ) -> tuple[str, str, str, str, str]:
        """构造工作区内部的不可歧义来源键."""

        return (
            source.randomization_scope,
            source.randomization_repeat_id,
            source.package_family,
            source.record_role,
            source.record_member,
        )

    def _read_member_bytes(
        self,
        source: RandomizationAggregateRecordSource,
    ) -> bytes:
        """从内部 leaf 路径读取规范成员并再次核对成员摘要."""

        package_key = (
            source.randomization_scope,
            source.randomization_repeat_id,
            source.package_family,
        )
        package_path = self._package_paths.get(package_key)
        if package_path is None:
            raise RandomizationAggregateRecordWorkspaceError(
                "记录源没有对应的已验证 leaf 包"
            )
        try:
            with _open_zip_bound_to_sha256(
                package_path,
                expected_sha256=source.leaf_package_sha256,
                role="临时 leaf 包",
            ) as archive:
                payload = archive.read(source.record_member)
        except KeyError as exc:
            raise RandomizationAggregateRecordWorkspaceError(
                "已验证 leaf 包的记录成员不可读取"
            ) from exc
        if hashlib.sha256(payload).hexdigest() != source.record_sha256:
            raise RandomizationAggregateRecordWorkspaceError(
                "临时 leaf 记录成员在读取时发生摘要漂移"
            )
        return payload

    def _prepare_workspace(
        self,
        validated: RandomizationAggregateProvenance,
        temporary_root: Path,
    ) -> None:
        """复制来源并完成9个组件、全部 leaf 与全部记录成员复验。"""

        aggregate_copy = temporary_root / "aggregate_source.zip"
        with validated.package_path.open("rb") as source, aggregate_copy.open(
            "xb"
        ) as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        paper_run_name = str(validated.payload["paper_run_name"])
        target_fpr = float(validated.payload["target_fpr"])
        code_version = validated.common_code_version
        repeat_records = {
            str(record["randomization_repeat_id"]): record
            for record in validated.randomization_repeat_components
        }
        invariant_records = {
            str(record["package_family"]): record
            for record in validated.invariant_packages
        }
        descriptors: list[RandomizationAggregateRecordSource] = []
        try:
            with _open_zip_bound_to_sha256(
                aggregate_copy,
                expected_sha256=validated.package_sha256,
                role="聚合包临时副本",
            ) as aggregate_archive:
                for repeat_id in formal_randomization_repeat_ids():
                    record = repeat_records[repeat_id]
                    component_path = (
                        temporary_root
                        / "repeat_components"
                        / repeat_id
                        / "component.zip"
                    )
                    _copy_zip_member(
                        aggregate_archive,
                        str(record["archive_member"]),
                        component_path,
                        expected_sha256=str(record["package_sha256"]),
                    )
                    report = validate_randomization_repeat_evidence_package(
                        component_path,
                        paper_run_name=paper_run_name,
                        target_fpr=target_fpr,
                        randomization_repeat_id=repeat_id,
                    )
                    if not all(
                        (
                            report.get("archive_sha256")
                            == record["package_sha256"],
                            report.get("randomization_repeat_id") == repeat_id,
                            report.get("code_version") == code_version,
                            report.get("repeat_component_ready") is True,
                            report.get("randomization_aggregate_ready") is False,
                            report.get("supports_paper_claim") is False,
                        )
                    ):
                        raise RandomizationAggregateRecordWorkspaceError(
                            f"单重复生产复验结果不一致: {repeat_id}"
                        )
                    with _open_zip_bound_to_sha256(
                        component_path,
                        expected_sha256=str(record["package_sha256"]),
                        role=f"单重复组件 {repeat_id}",
                    ) as component_archive:
                        descriptors.extend(
                            self._prepare_repeat_leaf_packages(
                                component_archive,
                                temporary_root=temporary_root,
                                randomization_repeat_id=repeat_id,
                                paper_run_name=paper_run_name,
                                target_fpr=target_fpr,
                                code_version=code_version,
                                aggregate_digest=(
                                    validated.randomization_aggregate_digest
                                ),
                                aggregate_package_sha256=(
                                    validated.package_sha256
                                ),
                                repeat_component_sha256=str(
                                    record["package_sha256"]
                                ),
                                repeat_manifest_digest=str(
                                    record[
                                        "randomization_repeat_evidence_manifest_digest"
                                    ]
                                ),
                                component_content_digest=str(
                                    record["component_content_digest"]
                                ),
                            )
                        )
                for package_family in (
                    RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES
                ):
                    record = invariant_records[package_family]
                    package_path = (
                        temporary_root
                        / "invariant_packages"
                        / f"{package_family}.zip"
                    )
                    _copy_zip_member(
                        aggregate_archive,
                        str(record["archive_member"]),
                        package_path,
                        expected_sha256=str(record["package_sha256"]),
                    )
                    specification = _SPECIFICATIONS_BY_FAMILY[package_family]
                    candidate = inspect_closure_package(
                        package_path,
                        spec=specification,
                        paper_run_name=paper_run_name,
                        target_fpr=target_fpr,
                        randomization_repeat_id=None,
                    )
                    if not _candidate_matches_invariant(
                        candidate,
                        invariant_record=record,
                        code_version=code_version,
                    ):
                        raise RandomizationAggregateRecordWorkspaceError(
                            f"不变包生产复验结果不一致: {package_family}"
                        )
                    with _open_zip_bound_to_sha256(
                        package_path,
                        expected_sha256=str(record["package_sha256"]),
                        role=f"不变 leaf {package_family}",
                    ) as package_archive:
                        descriptors.extend(
                            self._record_descriptors_for_package(
                                package_archive,
                                leaf_package_sha256=str(
                                    record["package_sha256"]
                                ),
                                randomization_scope=(
                                    "cross_repeat_invariant"
                                ),
                                randomization_repeat_id="",
                                package_family=package_family,
                                paper_run_name=paper_run_name,
                                code_version=code_version,
                                aggregate_digest=(
                                    validated.randomization_aggregate_digest
                                ),
                                aggregate_package_sha256=(
                                    validated.package_sha256
                                ),
                                repeat_component_sha256="",
                                repeat_manifest_digest="",
                                component_content_digest="",
                            )
                        )
                    package_key = (
                        "cross_repeat_invariant",
                        "",
                        package_family,
                    )
                    self._package_paths[package_key] = package_path
        except (BadZipFile, KeyError, OSError) as exc:
            if isinstance(exc, RandomizationAggregateRecordWorkspaceError):
                raise
            raise RandomizationAggregateRecordWorkspaceError(
                "聚合原始记录临时工作区无法建立"
            ) from exc
        keys = tuple(self._source_key(source) for source in descriptors)
        if len(keys) != len(set(keys)):
            raise RandomizationAggregateRecordWorkspaceError(
                "聚合原始记录登记包含重复来源"
            )
        self._record_sources = tuple(descriptors)
        self._source_keys = set(keys)

    def _prepare_repeat_leaf_packages(
        self,
        component_archive: ZipFile,
        *,
        temporary_root: Path,
        randomization_repeat_id: str,
        paper_run_name: str,
        target_fpr: float,
        code_version: str,
        aggregate_digest: str,
        aggregate_package_sha256: str,
        repeat_component_sha256: str,
        repeat_manifest_digest: str,
        component_content_digest: str,
    ) -> list[RandomizationAggregateRecordSource]:
        """复制并逐一调用生产 inspector 复验一个 repeat 的8个 leaf."""

        manifest = _validated_repeat_manifest(
            component_archive,
            randomization_repeat_id=randomization_repeat_id,
        )
        raw_leaf_records = manifest.get("leaf_packages")
        if not isinstance(raw_leaf_records, list):
            raise RandomizationAggregateRecordWorkspaceError(
                "单重复 manifest 缺少 leaf 记录"
            )
        leaf_records = tuple(
            dict(record) if isinstance(record, Mapping) else {}
            for record in raw_leaf_records
        )
        if tuple(
            str(record.get("package_family", "")) for record in leaf_records
        ) != RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES:
            raise RandomizationAggregateRecordWorkspaceError(
                "单重复 leaf family 顺序与权威登记不一致"
        )
        descriptors: list[RandomizationAggregateRecordSource] = []
        for leaf_record in leaf_records:
            package_family = str(leaf_record["package_family"])
            package_path = (
                temporary_root
                / "leaf_packages"
                / randomization_repeat_id
                / f"{package_family}.zip"
            )
            _copy_zip_member(
                component_archive,
                str(leaf_record["archive_member"]),
                package_path,
                expected_sha256=str(leaf_record["package_sha256"]),
            )
            specification = _SPECIFICATIONS_BY_FAMILY[package_family]
            candidate = inspect_closure_package(
                package_path,
                spec=specification,
                paper_run_name=paper_run_name,
                target_fpr=target_fpr,
                randomization_repeat_id=randomization_repeat_id,
            )
            if not _candidate_matches_repeat_leaf(
                candidate,
                leaf_record=leaf_record,
                randomization_repeat_id=randomization_repeat_id,
                code_version=code_version,
            ):
                raise RandomizationAggregateRecordWorkspaceError(
                    "活动 leaf 生产复验结果不一致: "
                    f"{randomization_repeat_id}/{package_family}"
                )
            with _open_zip_bound_to_sha256(
                package_path,
                expected_sha256=str(leaf_record["package_sha256"]),
                role=(
                    "活动 leaf "
                    f"{randomization_repeat_id}/{package_family}"
                ),
            ) as package_archive:
                descriptors.extend(
                    self._record_descriptors_for_package(
                        package_archive,
                        leaf_package_sha256=str(
                            leaf_record["package_sha256"]
                        ),
                        randomization_scope="active_repeat_component",
                        randomization_repeat_id=randomization_repeat_id,
                        package_family=package_family,
                        paper_run_name=paper_run_name,
                        code_version=code_version,
                        aggregate_digest=aggregate_digest,
                        aggregate_package_sha256=aggregate_package_sha256,
                        repeat_component_sha256=repeat_component_sha256,
                        repeat_manifest_digest=repeat_manifest_digest,
                        component_content_digest=component_content_digest,
                    )
                )
            package_key = (
                "active_repeat_component",
                randomization_repeat_id,
                package_family,
            )
            self._package_paths[package_key] = package_path
        return descriptors

    @staticmethod
    def _record_descriptors_for_package(
        archive: ZipFile,
        *,
        leaf_package_sha256: str,
        randomization_scope: str,
        randomization_repeat_id: str,
        package_family: str,
        paper_run_name: str,
        code_version: str,
        aggregate_digest: str,
        aggregate_package_sha256: str,
        repeat_component_sha256: str,
        repeat_manifest_digest: str,
        component_content_digest: str,
    ) -> list[RandomizationAggregateRecordSource]:
        """从固定模板构造不含临时路径的记录源描述符."""

        matching_specs = tuple(
            specification
            for specification in _RECORD_MEMBER_SPECS
            if specification.package_family == package_family
            and specification.randomization_scope == randomization_scope
        )
        if not matching_specs:
            return []
        descriptors: list[RandomizationAggregateRecordSource] = []
        try:
            member_names = set(archive.namelist())
            for specification in matching_specs:
                member_name = specification.member_template.format(
                    paper_run=paper_run_name
                )
                if member_name not in member_names:
                    raise RandomizationAggregateRecordWorkspaceError(
                        "已验证 leaf 缺少登记的统计记录成员: "
                        f"{package_family}/{specification.record_role}"
                    )
                descriptors.append(
                    RandomizationAggregateRecordSource(
                        randomization_scope=randomization_scope,
                        randomization_repeat_id=randomization_repeat_id,
                        package_family=package_family,
                        record_group=specification.record_group,
                        record_role=specification.record_role,
                        record_format=specification.record_format,
                        record_member=member_name,
                        record_sha256=_zip_member_sha256(
                            archive,
                            member_name,
                        ),
                        leaf_package_sha256=leaf_package_sha256,
                        randomization_repeat_component_sha256=(
                            repeat_component_sha256
                        ),
                        randomization_repeat_evidence_manifest_digest=(
                            repeat_manifest_digest
                        ),
                        component_content_digest=component_content_digest,
                        randomization_aggregate_package_sha256=(
                            aggregate_package_sha256
                        ),
                        common_code_version=code_version,
                        randomization_aggregate_digest=aggregate_digest,
                    )
                )
        except (KeyError, OSError) as exc:
            if isinstance(exc, RandomizationAggregateRecordWorkspaceError):
                raise
            raise RandomizationAggregateRecordWorkspaceError(
                f"已验证 leaf 的记录成员不可读取: {package_family}"
            ) from exc
        return descriptors


def open_randomization_aggregate_record_workspace(
    source: RandomizationAggregateProvenance,
) -> RandomizationAggregateRecordWorkspace:
    """返回只能在 ``with`` 中使用的聚合原始记录工作区."""

    return RandomizationAggregateRecordWorkspace(source)


__all__ = [
    "RECORD_FORMAT_JSONL",
    "RECORD_FORMAT_JSON_ARRAY",
    "RECORD_FORMAT_JSON_OBJECT",
    "RECORD_FORMAT_RAW_BYTES",
    "RECORD_GROUP_OBSERVATION",
    "RECORD_GROUP_ABLATION",
    "RECORD_GROUP_QUALITY",
    "RECORD_GROUP_THRESHOLD_BINDING",
    "RECORD_GROUP_REFERENCE",
    "RECORD_GROUP_PROMPT_RUNTIME",
    "RECORD_GROUP_PROMPT_SOURCE",
    "RECORD_GROUP_RUN_MANIFEST",
    "RandomizationAggregateRecordSource",
    "RandomizationAggregateQualityFeaturePair",
    "RandomizationAggregateRecordWorkspace",
    "RandomizationAggregateRecordWorkspaceError",
    "open_randomization_aggregate_record_workspace",
]
