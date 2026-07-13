"""把精确9重复组件与3个不变官方参考包封装为自包含来源包.

该模块只建立跨重复输入的版本化字节来源, 不计算论文统计量, 也不允许
直接支持论文结论. 聚合 ZIP 保存12个输入 ZIP 的原始字节, 并在写后重新
调用单重复生产 validator 与官方参考包生产 inspector. 后续统计层只能消费
本模块返回的不可变来源对象, 避免绕过输入身份、执行锁或代码版本校验.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import Any, Iterable, Mapping
from zipfile import BadZipFile, ZIP_STORED, ZipFile

from experiments.protocol.formal_randomization import (
    formal_randomization_protocol_record,
    formal_randomization_repeat_ids,
    formal_randomization_repeat_registry_digest,
    resolve_formal_randomization_repeat,
    validate_formal_randomization_repeat_records,
)
from experiments.protocol.paper_run_config import (
    RUN_DEFAULTS,
    normalize_paper_run_name,
    validate_frozen_paper_run_target_fpr,
)
from experiments.runtime.repository_environment import resolve_code_version
from paper_experiments.analysis.paper_claim_field_policy import (
    find_zip_paper_claim_violation,
)
from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    ClosurePackageCandidate,
    inspect_closure_package,
    normalize_clean_code_version,
    validate_closure_candidate_repository_profile,
)
from paper_experiments.runners.randomization_repeat_evidence import (
    validate_randomization_repeat_evidence_package,
)


RANDOMIZATION_AGGREGATE_PAYLOAD_SCHEMA = (
    "randomization_aggregate_provenance_payload"
)
RANDOMIZATION_AGGREGATE_MANIFEST_ARTIFACT_ID = (
    "randomization_aggregate_provenance_manifest"
)
RANDOMIZATION_AGGREGATE_SCHEMA_VERSION = 1
RANDOMIZATION_AGGREGATE_OUTPUT_ROOT = Path(
    "outputs/randomization_aggregate_provenance"
)
RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES = (
    "official_reference_tree_ring",
    "official_reference_gaussian_shading",
    "official_reference_shallow_diffuse",
)

# 该列表用于同步 ``docs/field_registry.md``. 列表只包含本模块新增字段,
# 已登记的通用 provenance 字段不重复列出.
RANDOMIZATION_AGGREGATE_FIELD_REGISTRY_SUGGESTIONS = (
    "randomization_aggregate_schema_version",
    "randomization_repeat_ids",
    "randomization_repeat_components",
    "invariant_packages",
    "randomization_aggregate_digest",
    "payload_member",
    "payload_sha256",
    "randomization_repeat_component_count",
    "invariant_package_count",
    "invariant_package_families",
)

_REPEAT_COMPONENT_RECORD_FIELDS = frozenset(
    {
        "randomization_repeat_id",
        "generation_seed_index",
        "generation_seed_offset",
        "watermark_key_index",
        "archive_member",
        "package_sha256",
        "code_version",
        "formal_randomization_protocol_digest",
        "randomization_repeat_evidence_manifest_digest",
        "component_content_digest",
        "leaf_package_set_digest",
    }
)
_INVARIANT_PACKAGE_RECORD_FIELDS = frozenset(
    {
        "package_family",
        "randomization_scope",
        "archive_member",
        "package_sha256",
        "code_version",
        "formal_execution_run_lock_digest",
        "formal_execution_package_lock_digest",
    }
)
_PAYLOAD_FIELDS = frozenset(
    {
        "report_schema",
        "randomization_aggregate_schema_version",
        "generated_at",
        "paper_run_name",
        "target_fpr",
        "formal_randomization_repeat_registry_digest",
        "randomization_repeat_ids",
        "randomization_repeat_components",
        "invariant_packages",
        "common_code_version",
        "formal_randomization_protocol_digest",
        "randomization_aggregate_digest",
        "randomization_aggregate_ready",
        "supports_paper_claim",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "artifact_id",
        "artifact_type",
        "randomization_aggregate_schema_version",
        "generated_at",
        "paper_run_name",
        "target_fpr",
        "formal_randomization_repeat_registry_digest",
        "randomization_repeat_ids",
        "randomization_repeat_components",
        "invariant_packages",
        "common_code_version",
        "formal_randomization_protocol_digest",
        "randomization_aggregate_digest",
        "randomization_aggregate_ready",
        "supports_paper_claim",
        "payload_member",
        "payload_sha256",
        "input_paths",
        "output_paths",
        "entry_sha256",
        "entry_paths_digest",
        "manifest_digest",
        "code_version",
        "rebuild_command",
        "config",
        "config_digest",
        "metadata",
    }
)


class RandomizationAggregateProvenanceError(ValueError):
    """表示跨重复来源包没有满足精确12输入契约."""


@dataclass(frozen=True)
class RandomizationAggregateProvenance:
    """保存已经独立复验且不可变的跨重复来源对象."""

    package_path: Path
    package_sha256: str
    payload_path: str
    payload_sha256: str
    manifest_path: str
    manifest_sha256: str
    payload: Mapping[str, Any]
    manifest: Mapping[str, Any]
    randomization_repeat_components: tuple[Mapping[str, Any], ...]
    invariant_packages: tuple[Mapping[str, Any], ...]
    common_code_version: str
    randomization_aggregate_digest: str


def _stable_digest(value: Any) -> str:
    """计算 JSON 兼容对象的稳定 SHA-256."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    """计算内存字节的 SHA-256."""

    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    """流式计算普通文件的 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_member_sha256(archive: ZipFile, member_name: str) -> str:
    """流式计算 ZIP 成员的 SHA-256."""

    digest = hashlib.sha256()
    with archive.open(member_name, "r") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """拒绝会被普通 JSON parser 静默覆盖的重复字段."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RandomizationAggregateProvenanceError(
                f"聚合来源 JSON 包含重复字段: {key}"
            )
        result[key] = value
    return result


def _read_json_object(payload: bytes, *, role: str) -> dict[str, Any]:
    """读取严格 JSON object, 同时拒绝重复字段."""

    try:
        value = json.loads(
            payload.decode("utf-8-sig"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RandomizationAggregateProvenanceError(
            f"聚合来源 {role} 不是有效 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise RandomizationAggregateProvenanceError(
            f"聚合来源 {role} 必须是 JSON object"
        )
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    """序列化规范人类可读 JSON 字节."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _deep_freeze(value: Any) -> Any:
    """递归冻结来源对象, 防止统计层修改已验证事实."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _aggregate_prefix(paper_run_name: str) -> str:
    """返回聚合归档内的规范职责前缀."""

    return f"randomization_aggregate_provenance/{paper_run_name}"


def _payload_member_name(paper_run_name: str) -> str:
    """返回聚合 payload 的规范成员路径."""

    return f"{_aggregate_prefix(paper_run_name)}/randomization_aggregate_payload.json"


def _manifest_member_name(paper_run_name: str) -> str:
    """返回聚合 manifest 的规范成员路径."""

    return f"{_aggregate_prefix(paper_run_name)}/randomization_aggregate_manifest.json"


def _repeat_component_member_name(
    _paper_run_name: str,
    randomization_repeat_id: str,
) -> str:
    """返回一个单重复组件的规范成员路径."""

    return f"repeat_components/{randomization_repeat_id}.zip"


def _invariant_package_member_name(
    _paper_run_name: str,
    package_family: str,
) -> str:
    """返回一个跨重复不变包的规范成员路径."""

    return f"invariant_packages/{package_family}.zip"


def _repeat_manifest_member_name(randomization_repeat_id: str) -> str:
    """返回单重复外层 manifest 的规范成员路径."""

    return (
        f"randomization_repeat_evidence/{randomization_repeat_id}/"
        "randomization_repeat_evidence_manifest.json"
    )


def _load_repeat_manifest(
    package_path: Path,
    *,
    randomization_repeat_id: str,
) -> dict[str, Any]:
    """读取已由生产 validator 复验过的单重复外层 manifest."""

    member_name = _repeat_manifest_member_name(randomization_repeat_id)
    try:
        with ZipFile(package_path) as archive:
            return _read_json_object(
                archive.read(member_name),
                role=f"repeat manifest {randomization_repeat_id}",
            )
    except (BadZipFile, KeyError, OSError) as exc:
        raise RandomizationAggregateProvenanceError(
            f"单重复组件缺少规范 manifest: {randomization_repeat_id}"
        ) from exc


def _require_regular_zip(path_value: str | Path, *, role: str) -> Path:
    """要求外部输入是非符号链接的普通 ZIP 文件."""

    unresolved = Path(path_value).expanduser()
    if unresolved.is_symlink():
        raise RandomizationAggregateProvenanceError(f"{role} 不得是符号链接")
    path = unresolved.resolve()
    if (
        not path.is_file()
        or path.suffix.lower() != ".zip"
        or path.stat().st_size <= 0
    ):
        raise RandomizationAggregateProvenanceError(
            f"{role} 必须是非空普通 ZIP 文件"
        )
    return path


def _validate_generated_at(value: Any) -> str:
    """要求生成时间是带时区的 ISO-8601 字符串."""

    if not isinstance(value, str) or not value.strip():
        raise RandomizationAggregateProvenanceError("聚合来源缺少 generated_at")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RandomizationAggregateProvenanceError(
            "聚合来源 generated_at 不是 ISO-8601 时间"
        ) from exc
    if timestamp.tzinfo is None:
        raise RandomizationAggregateProvenanceError(
            "聚合来源 generated_at 必须包含时区"
        )
    return value


def _validate_sha256(value: Any, *, role: str, length: int = 64) -> str:
    """要求摘要是指定长度的小写十六进制字符串."""

    normalized = str(value)
    if re.fullmatch(rf"[0-9a-f]{{{length}}}", normalized) is None:
        raise RandomizationAggregateProvenanceError(f"{role} 摘要格式无效")
    return normalized


def _specification_by_family() -> dict[str, Any]:
    """构造生产 inspector 使用的 family 规格索引."""

    return {
        specification.package_family: specification
        for specification in CLOSURE_PACKAGE_FAMILY_SPECS
        if specification.package_family
        in RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES
    }


def _repeat_component_record(
    package_path: Path,
    *,
    paper_run_name: str,
    target_fpr: float,
    randomization_repeat_id: str,
) -> dict[str, Any]:
    """调用生产 validator 并构造一个规范单重复来源记录."""

    try:
        report = validate_randomization_repeat_evidence_package(
            package_path,
            paper_run_name=paper_run_name,
            target_fpr=target_fpr,
            randomization_repeat_id=randomization_repeat_id,
        )
    except (OSError, ValueError) as exc:
        raise RandomizationAggregateProvenanceError(
            f"单重复组件未通过生产 validator: {randomization_repeat_id}"
        ) from exc
    manifest = _load_repeat_manifest(
        package_path,
        randomization_repeat_id=randomization_repeat_id,
    )
    repeat = resolve_formal_randomization_repeat(randomization_repeat_id)
    expected_protocol_digest = formal_randomization_protocol_record()[
        "formal_randomization_protocol_digest"
    ]
    if not all(
        (
            report.get("archive_sha256") == _file_sha256(package_path),
            report.get("randomization_repeat_id") == randomization_repeat_id,
            type(report.get("generation_seed_index")) is int,
            report.get("generation_seed_index") == repeat.generation_seed_index,
            type(report.get("generation_seed_offset")) is int,
            report.get("generation_seed_offset") == repeat.generation_seed_offset,
            type(report.get("watermark_key_index")) is int,
            report.get("watermark_key_index") == repeat.watermark_key_index,
            report.get("formal_randomization_protocol_digest")
            == expected_protocol_digest,
            report.get("repeat_component_ready") is True,
            report.get("randomization_aggregate_ready") is False,
            report.get("supports_paper_claim") is False,
            manifest.get("leaf_package_set_digest")
            == _stable_digest(manifest.get("leaf_packages")),
        )
    ):
        raise RandomizationAggregateProvenanceError(
            f"单重复组件来源记录不一致: {randomization_repeat_id}"
        )
    return {
        "randomization_repeat_id": repeat.randomization_repeat_id,
        "generation_seed_index": repeat.generation_seed_index,
        "generation_seed_offset": repeat.generation_seed_offset,
        "watermark_key_index": repeat.watermark_key_index,
        "archive_member": _repeat_component_member_name(
            paper_run_name,
            randomization_repeat_id,
        ),
        "package_sha256": str(report["archive_sha256"]),
        "code_version": str(report["code_version"]),
        "formal_randomization_protocol_digest": expected_protocol_digest,
        "randomization_repeat_evidence_manifest_digest": str(
            report["randomization_repeat_evidence_manifest_digest"]
        ),
        "component_content_digest": str(report["component_content_digest"]),
        "leaf_package_set_digest": str(manifest["leaf_package_set_digest"]),
    }


def _invariant_package_record(
    package_path: Path,
    *,
    package_family: str,
    paper_run_name: str,
    target_fpr: float,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """调用生产 inspector 并构造一个规范不变官方参考来源记录."""

    specification = _specification_by_family().get(package_family)
    if specification is None:
        raise RandomizationAggregateProvenanceError(
            f"未登记的不变官方参考包 family: {package_family}"
        )
    try:
        claim_violation = find_zip_paper_claim_violation(package_path)
    except (BadZipFile, OSError, UnicodeError, ValueError) as exc:
        raise RandomizationAggregateProvenanceError(
            f"不变官方参考包结构化成员不可复验: {package_family}"
        ) from exc
    if claim_violation is not None:
        raise RandomizationAggregateProvenanceError(
            "不变官方参考包包含正向论文结论字段: "
            f"{package_family}:{claim_violation.path}"
        )
    try:
        candidate: ClosurePackageCandidate = inspect_closure_package(
            package_path,
            spec=specification,
            paper_run_name=paper_run_name,
            target_fpr=target_fpr,
            randomization_repeat_id=None,
        )
    except (OSError, ValueError) as exc:
        raise RandomizationAggregateProvenanceError(
            f"不变官方参考包未通过生产 inspector: {package_family}"
        ) from exc
    if repository_root is not None:
        try:
            validate_closure_candidate_repository_profile(
                candidate,
                repository_root=repository_root,
            )
        except (OSError, ValueError) as exc:
            raise RandomizationAggregateProvenanceError(
                f"不变官方参考包未匹配当前仓库依赖 profile: {package_family}"
            ) from exc
    if not all(
        (
            candidate.package_family == package_family,
            candidate.package_sha256 == _file_sha256(package_path),
            candidate.randomization_scope == "cross_repeat_invariant",
            candidate.randomization_repeat_id == "",
            candidate.generation_seed_index == -1,
            candidate.generation_seed_offset == -1,
            candidate.watermark_key_index == -1,
        )
    ):
        raise RandomizationAggregateProvenanceError(
            f"不变官方参考包错误绑定了活动 repeat: {package_family}"
        )
    _validate_sha256(
        candidate.formal_execution_run_lock_digest,
        role=f"{package_family} run lock",
    )
    _validate_sha256(
        candidate.formal_execution_package_lock_digest,
        role=f"{package_family} package lock",
    )
    return {
        "package_family": package_family,
        "randomization_scope": "cross_repeat_invariant",
        "archive_member": _invariant_package_member_name(
            paper_run_name,
            package_family,
        ),
        "package_sha256": candidate.package_sha256,
        "code_version": candidate.code_version,
        "formal_execution_run_lock_digest": (
            candidate.formal_execution_run_lock_digest
        ),
        "formal_execution_package_lock_digest": (
            candidate.formal_execution_package_lock_digest
        ),
    }


def _aggregate_core(
    *,
    paper_run_name: str,
    target_fpr: float,
    repeat_components: list[dict[str, Any]],
    invariant_packages: list[dict[str, Any]],
    common_code_version: str,
) -> dict[str, Any]:
    """构造排除时间与自身摘要的规范聚合内容."""

    return {
        "randomization_aggregate_schema_version": (
            RANDOMIZATION_AGGREGATE_SCHEMA_VERSION
        ),
        "paper_run_name": paper_run_name,
        "target_fpr": float(target_fpr),
        "formal_randomization_repeat_registry_digest": (
            formal_randomization_repeat_registry_digest()
        ),
        "randomization_repeat_ids": list(formal_randomization_repeat_ids()),
        "randomization_repeat_components": repeat_components,
        "invariant_packages": invariant_packages,
        "common_code_version": common_code_version,
        "formal_randomization_protocol_digest": (
            formal_randomization_protocol_record()[
                "formal_randomization_protocol_digest"
            ]
        ),
        "randomization_aggregate_ready": True,
        "supports_paper_claim": False,
    }


def _build_rebuild_command(
    *,
    paper_run_name: str,
    target_fpr: float,
) -> str:
    """构造由自包含 aggregate ZIP 安全重建来源包的 CLI 模板."""

    arguments = [
        "python",
        "-m",
        "paper_experiments.runners.randomization_aggregate_provenance",
        "--paper-run-name",
        paper_run_name,
        "--target-fpr",
        str(float(target_fpr)),
        "--rebuild-source-aggregate-package-path",
        "{aggregate_package_path}",
    ]
    return " ".join(arguments)


def build_randomization_aggregate_payload(
    *,
    paper_run_name: str,
    target_fpr: float,
    repeat_components: list[dict[str, Any]],
    invariant_packages: list[dict[str, Any]],
    common_code_version: str,
    generated_at: str,
) -> dict[str, Any]:
    """从已校验来源记录构造版本化 aggregate payload."""

    resolved_run_name = normalize_paper_run_name(paper_run_name)
    resolved_target_fpr = validate_frozen_paper_run_target_fpr(
        resolved_run_name,
        target_fpr,
    )
    core = _aggregate_core(
        paper_run_name=resolved_run_name,
        target_fpr=resolved_target_fpr,
        repeat_components=repeat_components,
        invariant_packages=invariant_packages,
        common_code_version=common_code_version,
    )
    return {
        "report_schema": RANDOMIZATION_AGGREGATE_PAYLOAD_SCHEMA,
        "generated_at": _validate_generated_at(generated_at),
        **core,
        "randomization_aggregate_digest": _stable_digest(core),
    }


def _build_randomization_aggregate_manifest(
    payload: dict[str, Any],
    *,
    payload_bytes: bytes,
) -> dict[str, Any]:
    """构造绑定 payload 与全部输入成员摘要的外层 manifest."""

    paper_run_name = str(payload["paper_run_name"])
    repeat_records = list(payload["randomization_repeat_components"])
    invariant_records = list(payload["invariant_packages"])
    input_paths = [
        str(record["archive_member"])
        for record in repeat_records + invariant_records
    ]
    payload_member = _payload_member_name(paper_run_name)
    entry_sha256 = {
        str(record["archive_member"]): str(record["package_sha256"])
        for record in repeat_records + invariant_records
    }
    entry_sha256[payload_member] = _bytes_sha256(payload_bytes)
    config = {
        "paper_run_name": paper_run_name,
        "target_fpr": float(payload["target_fpr"]),
        "formal_randomization_repeat_registry_digest": payload[
            "formal_randomization_repeat_registry_digest"
        ],
        "formal_randomization_protocol_digest": payload[
            "formal_randomization_protocol_digest"
        ],
        "randomization_repeat_ids": payload["randomization_repeat_ids"],
        "invariant_package_families": list(
            RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES
        ),
        "rebuild_input_mode": "self_contained_aggregate_zip",
        "rebuild_working_directory": "repository_root",
        "rebuild_source_argument": "{aggregate_package_path}",
    }
    metadata = {
        "randomization_aggregate_schema_version": payload[
            "randomization_aggregate_schema_version"
        ],
        "randomization_aggregate_ready": payload[
            "randomization_aggregate_ready"
        ],
        "supports_paper_claim": payload["supports_paper_claim"],
        "randomization_aggregate_digest": payload[
            "randomization_aggregate_digest"
        ],
        "common_code_version": payload["common_code_version"],
        "formal_randomization_protocol_digest": payload[
            "formal_randomization_protocol_digest"
        ],
        "randomization_repeat_ids": payload["randomization_repeat_ids"],
        "randomization_repeat_component_count": len(repeat_records),
        "invariant_package_count": len(invariant_records),
    }
    manifest = {
        "artifact_id": RANDOMIZATION_AGGREGATE_MANIFEST_ARTIFACT_ID,
        "artifact_type": "local_manifest",
        **{
            field_name: payload[field_name]
            for field_name in (
                "randomization_aggregate_schema_version",
                "generated_at",
                "paper_run_name",
                "target_fpr",
                "formal_randomization_repeat_registry_digest",
                "randomization_repeat_ids",
                "randomization_repeat_components",
                "invariant_packages",
                "common_code_version",
                "formal_randomization_protocol_digest",
                "randomization_aggregate_digest",
                "randomization_aggregate_ready",
                "supports_paper_claim",
            )
        },
        "payload_member": payload_member,
        "payload_sha256": _bytes_sha256(payload_bytes),
        "input_paths": input_paths,
        "output_paths": [
            payload_member,
            _manifest_member_name(paper_run_name),
        ],
        "entry_sha256": entry_sha256,
        "entry_paths_digest": _stable_digest(sorted(entry_sha256)),
        "code_version": payload["common_code_version"],
        "rebuild_command": _build_rebuild_command(
            paper_run_name=paper_run_name,
            target_fpr=float(payload["target_fpr"]),
        ),
        "config": config,
        "config_digest": _stable_digest(config),
        "metadata": metadata,
    }
    manifest["manifest_digest"] = _stable_digest(manifest)
    return manifest


def _validate_record_shapes(
    payload: Mapping[str, Any],
    *,
    paper_run_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """校验 payload 中两类记录的精确顺序、字段和规范成员路径."""

    raw_repeat_records = payload.get("randomization_repeat_components")
    raw_invariant_records = payload.get("invariant_packages")
    if not isinstance(raw_repeat_records, list) or not isinstance(
        raw_invariant_records,
        list,
    ):
        raise RandomizationAggregateProvenanceError(
            "聚合来源缺少 repeat 或 invariant 记录"
        )
    repeat_records: list[dict[str, Any]] = []
    for repeat_id, raw_record in zip(
        formal_randomization_repeat_ids(),
        raw_repeat_records,
    ):
        if (
            not isinstance(raw_record, dict)
            or frozenset(raw_record) != _REPEAT_COMPONENT_RECORD_FIELDS
            or raw_record.get("randomization_repeat_id") != repeat_id
            or raw_record.get("archive_member")
            != _repeat_component_member_name(paper_run_name, repeat_id)
        ):
            raise RandomizationAggregateProvenanceError(
                f"聚合来源 repeat 记录不规范: {repeat_id}"
            )
        repeat_records.append(dict(raw_record))
    if len(raw_repeat_records) != len(formal_randomization_repeat_ids()):
        raise RandomizationAggregateProvenanceError(
            "聚合来源必须精确唯一覆盖权威9个 repeat"
        )
    try:
        validate_formal_randomization_repeat_records(
            repeat_records,
            require_exact_registry=True,
        )
    except (TypeError, ValueError) as exc:
        raise RandomizationAggregateProvenanceError(
            "聚合来源 repeat 身份未精确匹配权威注册表"
        ) from exc

    invariant_records: list[dict[str, Any]] = []
    for package_family, raw_record in zip(
        RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES,
        raw_invariant_records,
    ):
        if (
            not isinstance(raw_record, dict)
            or frozenset(raw_record) != _INVARIANT_PACKAGE_RECORD_FIELDS
            or raw_record.get("package_family") != package_family
            or raw_record.get("randomization_scope")
            != "cross_repeat_invariant"
            or raw_record.get("archive_member")
            != _invariant_package_member_name(paper_run_name, package_family)
        ):
            raise RandomizationAggregateProvenanceError(
                f"聚合来源 invariant 记录不规范: {package_family}"
            )
        invariant_records.append(dict(raw_record))
    if len(raw_invariant_records) != len(
        RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES
    ):
        raise RandomizationAggregateProvenanceError(
            "聚合来源必须精确覆盖3个跨 repeat 不变官方参考包"
        )
    return repeat_records, invariant_records


def _validate_archive_members(archive: ZipFile) -> list[str]:
    """拒绝重复、目录、符号链接和路径穿越成员."""

    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)) or any(info.is_dir() for info in infos):
        raise RandomizationAggregateProvenanceError(
            "聚合来源 ZIP 成员必须唯一且均为文件"
        )
    for info in infos:
        member = PurePosixPath(info.filename)
        if (
            not info.filename
            or "\\" in info.filename
            or "\x00" in info.filename
            or member.is_absolute()
            or ".." in member.parts
            or member.as_posix() != info.filename
            or stat.S_ISLNK(info.external_attr >> 16)
        ):
            raise RandomizationAggregateProvenanceError(
                "聚合来源 ZIP 包含不安全成员"
            )
    damaged_member = archive.testzip()
    if damaged_member is not None:
        raise RandomizationAggregateProvenanceError(
            f"聚合来源 ZIP CRC 失败: {damaged_member}"
        )
    return names


def _validate_payload_and_manifest(
    payload: dict[str, Any],
    manifest: dict[str, Any],
    *,
    paper_run_name: str,
    target_fpr: float,
    payload_bytes: bytes,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """复算 payload、manifest 与聚合摘要的全部静态绑定."""

    if frozenset(payload) != _PAYLOAD_FIELDS:
        raise RandomizationAggregateProvenanceError(
            "聚合 payload 字段集合不匹配版本化 schema"
        )
    if frozenset(manifest) != _MANIFEST_FIELDS:
        raise RandomizationAggregateProvenanceError(
            "聚合 manifest 字段集合不匹配版本化 schema"
        )
    _validate_generated_at(payload.get("generated_at"))
    expected_protocol_digest = formal_randomization_protocol_record()[
        "formal_randomization_protocol_digest"
    ]
    static_ready = all(
        (
            payload.get("report_schema")
            == RANDOMIZATION_AGGREGATE_PAYLOAD_SCHEMA,
            type(payload.get("randomization_aggregate_schema_version")) is int,
            payload.get("randomization_aggregate_schema_version")
            == RANDOMIZATION_AGGREGATE_SCHEMA_VERSION,
            payload.get("paper_run_name") == paper_run_name,
            isinstance(payload.get("target_fpr"), (int, float)),
            math.isclose(
                float(payload.get("target_fpr", math.nan)),
                target_fpr,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            payload.get("formal_randomization_repeat_registry_digest")
            == formal_randomization_repeat_registry_digest(),
            payload.get("randomization_repeat_ids")
            == list(formal_randomization_repeat_ids()),
            payload.get("formal_randomization_protocol_digest")
            == expected_protocol_digest,
            payload.get("randomization_aggregate_ready") is True,
            payload.get("supports_paper_claim") is False,
            manifest.get("artifact_id")
            == RANDOMIZATION_AGGREGATE_MANIFEST_ARTIFACT_ID,
            manifest.get("artifact_type") == "local_manifest",
            type(manifest.get("randomization_aggregate_schema_version")) is int,
        )
    )
    if not static_ready:
        raise RandomizationAggregateProvenanceError(
            "聚合来源运行身份、协议或结论边界不匹配"
        )
    repeat_records, invariant_records = _validate_record_shapes(
        payload,
        paper_run_name=paper_run_name,
    )
    code_versions = {
        str(record.get("code_version", ""))
        for record in repeat_records + invariant_records
    }
    common_code_version = str(payload.get("common_code_version", ""))
    if (
        len(code_versions) != 1
        or code_versions != {common_code_version}
        or re.fullmatch(r"[0-9a-f]{40}", common_code_version) is None
    ):
        raise RandomizationAggregateProvenanceError(
            "聚合来源12个输入必须共享 clean code version"
        )
    for record in repeat_records:
        if record.get("formal_randomization_protocol_digest") != (
            expected_protocol_digest
        ):
            raise RandomizationAggregateProvenanceError(
                "单重复组件随机化协议摘要不一致"
            )
        for field_name in (
            "package_sha256",
            "randomization_repeat_evidence_manifest_digest",
            "component_content_digest",
            "leaf_package_set_digest",
        ):
            _validate_sha256(record.get(field_name), role=field_name)
    for record in invariant_records:
        for field_name in (
            "package_sha256",
            "formal_execution_run_lock_digest",
            "formal_execution_package_lock_digest",
        ):
            _validate_sha256(record.get(field_name), role=field_name)
    package_digests = [
        str(record["package_sha256"])
        for record in repeat_records + invariant_records
    ]
    if len(package_digests) != len(set(package_digests)):
        raise RandomizationAggregateProvenanceError(
            "聚合来源不得重复使用同一输入 ZIP 字节"
        )
    core = _aggregate_core(
        paper_run_name=paper_run_name,
        target_fpr=target_fpr,
        repeat_components=repeat_records,
        invariant_packages=invariant_records,
        common_code_version=common_code_version,
    )
    if payload.get("randomization_aggregate_digest") != _stable_digest(core):
        raise RandomizationAggregateProvenanceError(
            "聚合来源内容摘要不匹配"
        )
    mirrored_fields = {
        "randomization_aggregate_schema_version",
        "generated_at",
        "paper_run_name",
        "target_fpr",
        "formal_randomization_repeat_registry_digest",
        "randomization_repeat_ids",
        "randomization_repeat_components",
        "invariant_packages",
        "common_code_version",
        "formal_randomization_protocol_digest",
        "randomization_aggregate_digest",
        "randomization_aggregate_ready",
        "supports_paper_claim",
    }
    if any(manifest.get(field) != payload.get(field) for field in mirrored_fields):
        raise RandomizationAggregateProvenanceError(
            "聚合 manifest 未精确镜像 payload 身份"
        )
    payload_member = _payload_member_name(paper_run_name)
    input_paths = [
        str(record["archive_member"])
        for record in repeat_records + invariant_records
    ]
    expected_entry_sha256 = {
        str(record["archive_member"]): str(record["package_sha256"])
        for record in repeat_records + invariant_records
    }
    expected_entry_sha256[payload_member] = _bytes_sha256(payload_bytes)
    expected_config = {
        "paper_run_name": paper_run_name,
        "target_fpr": target_fpr,
        "formal_randomization_repeat_registry_digest": (
            formal_randomization_repeat_registry_digest()
        ),
        "formal_randomization_protocol_digest": (
            formal_randomization_protocol_record()[
                "formal_randomization_protocol_digest"
            ]
        ),
        "randomization_repeat_ids": list(formal_randomization_repeat_ids()),
        "invariant_package_families": list(
            RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES
        ),
        "rebuild_input_mode": "self_contained_aggregate_zip",
        "rebuild_working_directory": "repository_root",
        "rebuild_source_argument": "{aggregate_package_path}",
    }
    expected_metadata = {
        "randomization_aggregate_schema_version": (
            RANDOMIZATION_AGGREGATE_SCHEMA_VERSION
        ),
        "randomization_aggregate_ready": True,
        "supports_paper_claim": False,
        "randomization_aggregate_digest": payload[
            "randomization_aggregate_digest"
        ],
        "common_code_version": common_code_version,
        "formal_randomization_protocol_digest": (
            formal_randomization_protocol_record()[
                "formal_randomization_protocol_digest"
            ]
        ),
        "randomization_repeat_ids": list(formal_randomization_repeat_ids()),
        "randomization_repeat_component_count": len(repeat_records),
        "invariant_package_count": len(invariant_records),
    }
    manifest_metadata = manifest.get("metadata")
    metadata_schema_version = (
        manifest_metadata.get("randomization_aggregate_schema_version")
        if isinstance(manifest_metadata, dict)
        else None
    )
    manifest_without_digest = dict(manifest)
    declared_manifest_digest = manifest_without_digest.pop(
        "manifest_digest",
        None,
    )
    manifest_ready = all(
        (
            manifest.get("payload_member") == payload_member,
            manifest.get("payload_sha256") == _bytes_sha256(payload_bytes),
            manifest.get("input_paths") == input_paths,
            manifest.get("output_paths")
            == [payload_member, _manifest_member_name(paper_run_name)],
            manifest.get("entry_sha256") == expected_entry_sha256,
            manifest.get("entry_paths_digest")
            == _stable_digest(sorted(expected_entry_sha256)),
            manifest.get("code_version") == common_code_version,
            manifest.get("rebuild_command")
            == _build_rebuild_command(
                paper_run_name=paper_run_name,
                target_fpr=target_fpr,
            ),
            manifest.get("config") == expected_config,
            manifest.get("config_digest") == _stable_digest(expected_config),
            manifest_metadata == expected_metadata,
            isinstance(manifest_metadata, dict),
            type(metadata_schema_version) is int,
            declared_manifest_digest == _stable_digest(manifest_without_digest),
        )
    )
    if not manifest_ready:
        raise RandomizationAggregateProvenanceError(
            "聚合 manifest 成员摘要或自身摘要不匹配"
        )
    return repeat_records, invariant_records


def _sidecar_paths(package_path: Path) -> tuple[Path, Path]:
    """返回写包器使用的确定性 payload 与 manifest 旁路路径."""

    return (
        package_path.with_suffix(".payload.json"),
        package_path.with_suffix(".manifest.json"),
    )


def validate_randomization_aggregate_provenance(
    package_path: str | Path,
    *,
    paper_run_name: str,
    target_fpr: float,
) -> RandomizationAggregateProvenance:
    """以聚合 ZIP 原始字节为权威来源执行完整写后或消费时复验."""

    path = _require_regular_zip(package_path, role="聚合来源包")
    resolved_run_name = normalize_paper_run_name(paper_run_name)
    expected_target_fpr = float(target_fpr)
    if (
        not math.isfinite(expected_target_fpr)
        or not 0.0 < expected_target_fpr < 1.0
    ):
        raise RandomizationAggregateProvenanceError(
            "target_fpr 必须是位于 (0, 1) 的有限数值"
        )
    payload_member = _payload_member_name(resolved_run_name)
    manifest_member = _manifest_member_name(resolved_run_name)
    try:
        with ZipFile(path) as archive:
            names = _validate_archive_members(archive)
            if payload_member not in names or manifest_member not in names:
                raise RandomizationAggregateProvenanceError(
                    "聚合来源 ZIP 缺少规范 payload 或 manifest"
                )
            payload_bytes = archive.read(payload_member)
            manifest_bytes = archive.read(manifest_member)
            payload = _read_json_object(payload_bytes, role="payload")
            manifest = _read_json_object(manifest_bytes, role="manifest")
            repeat_records, invariant_records = _validate_payload_and_manifest(
                payload,
                manifest,
                paper_run_name=resolved_run_name,
                target_fpr=expected_target_fpr,
                payload_bytes=payload_bytes,
            )
            expected_names = {
                payload_member,
                manifest_member,
                *(
                    str(record["archive_member"])
                    for record in repeat_records + invariant_records
                ),
            }
            if set(names) != expected_names:
                raise RandomizationAggregateProvenanceError(
                    "聚合来源 ZIP 未精确覆盖规范14个成员"
                )
            entry_sha256 = manifest["entry_sha256"]
            for member_name, expected_sha256 in entry_sha256.items():
                if _zip_member_sha256(archive, member_name) != expected_sha256:
                    raise RandomizationAggregateProvenanceError(
                        f"聚合来源成员字节摘要不匹配: {member_name}"
                    )
            with tempfile.TemporaryDirectory(
                prefix="slm_wm_randomization_aggregate_"
            ) as temporary_directory:
                temporary_root = Path(temporary_directory)
                rebuilt_repeat_records: list[dict[str, Any]] = []
                for record in repeat_records:
                    repeat_id = str(record["randomization_repeat_id"])
                    temporary_package = temporary_root / f"{repeat_id}.zip"
                    with archive.open(str(record["archive_member"]), "r") as source, (
                        temporary_package.open("wb")
                    ) as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
                    rebuilt_repeat_records.append(
                        _repeat_component_record(
                            temporary_package,
                            paper_run_name=resolved_run_name,
                            target_fpr=expected_target_fpr,
                            randomization_repeat_id=repeat_id,
                        )
                    )
                rebuilt_invariant_records: list[dict[str, Any]] = []
                for record in invariant_records:
                    package_family = str(record["package_family"])
                    temporary_package = temporary_root / f"{package_family}.zip"
                    with archive.open(str(record["archive_member"]), "r") as source, (
                        temporary_package.open("wb")
                    ) as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
                    rebuilt_invariant_records.append(
                        _invariant_package_record(
                            temporary_package,
                            package_family=package_family,
                            paper_run_name=resolved_run_name,
                            target_fpr=expected_target_fpr,
                        )
                    )
            if (
                rebuilt_repeat_records != repeat_records
                or rebuilt_invariant_records != invariant_records
            ):
                raise RandomizationAggregateProvenanceError(
                    "聚合来源嵌套输入的生产复验结果与 payload 不一致"
                )
    except (BadZipFile, EOFError, KeyError, OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, RandomizationAggregateProvenanceError):
            raise
        raise RandomizationAggregateProvenanceError(
            "聚合来源 ZIP 不可读取"
        ) from exc

    payload_sidecar, manifest_sidecar = _sidecar_paths(path)
    sidecar_presence = (payload_sidecar.exists(), manifest_sidecar.exists())
    if any(sidecar_presence) and not all(sidecar_presence):
        raise RandomizationAggregateProvenanceError(
            "聚合来源旁路 payload 与 manifest 必须同时存在"
        )
    if all(sidecar_presence):
        if payload_sidecar.is_symlink() or manifest_sidecar.is_symlink():
            raise RandomizationAggregateProvenanceError(
                "聚合来源旁路文件不得是符号链接"
            )
        if (
            payload_sidecar.read_bytes() != payload_bytes
            or manifest_sidecar.read_bytes() != manifest_bytes
        ):
            raise RandomizationAggregateProvenanceError(
                "聚合来源旁路文件与 ZIP 内权威字节不一致"
            )

    frozen_payload = _deep_freeze(payload)
    frozen_manifest = _deep_freeze(manifest)
    payload_path = (
        payload_sidecar.as_posix()
        if all(sidecar_presence)
        else f"{path.as_posix()}!/{payload_member}"
    )
    manifest_path = (
        manifest_sidecar.as_posix()
        if all(sidecar_presence)
        else f"{path.as_posix()}!/{manifest_member}"
    )
    return RandomizationAggregateProvenance(
        package_path=path,
        package_sha256=_file_sha256(path),
        payload_path=payload_path,
        payload_sha256=_bytes_sha256(payload_bytes),
        manifest_path=manifest_path,
        manifest_sha256=_bytes_sha256(manifest_bytes),
        payload=frozen_payload,
        manifest=frozen_manifest,
        randomization_repeat_components=tuple(
            _deep_freeze(record) for record in repeat_records
        ),
        invariant_packages=tuple(
            _deep_freeze(record) for record in invariant_records
        ),
        common_code_version=str(payload["common_code_version"]),
        randomization_aggregate_digest=str(
            payload["randomization_aggregate_digest"]
        ),
    )


def _resolve_output_directory(
    *,
    repository_root: Path,
    output_dir: str | Path | None,
    paper_run_name: str,
) -> Path:
    """解析 outputs 内目录并拒绝任何符号链接路径段."""

    requested = (
        repository_root
        / RANDOMIZATION_AGGREGATE_OUTPUT_ROOT
        / paper_run_name
        if output_dir is None
        else Path(output_dir).expanduser()
    )
    if not requested.is_absolute():
        requested = repository_root / requested
    current = requested
    while current != current.parent:
        if current.exists() and current.is_symlink():
            raise RandomizationAggregateProvenanceError(
                "聚合来源输出路径不得包含符号链接"
            )
        if current == repository_root:
            break
        current = current.parent
    resolved = requested.resolve()
    try:
        resolved.relative_to((repository_root / "outputs").resolve())
    except ValueError as exc:
        raise RandomizationAggregateProvenanceError(
            "聚合来源输出目录必须位于 outputs 下"
        ) from exc
    return resolved


def write_randomization_aggregate_provenance_package(
    repeat_component_paths: Mapping[str, str | Path],
    invariant_package_paths: Mapping[str, str | Path],
    *,
    paper_run_name: str,
    target_fpr: float,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
) -> RandomizationAggregateProvenance:
    """复验精确9+3输入并写出 payload、manifest 与自包含聚合 ZIP."""

    repository_root = Path(root).resolve()
    resolved_run_name = normalize_paper_run_name(paper_run_name)
    resolved_target_fpr = validate_frozen_paper_run_target_fpr(
        resolved_run_name,
        target_fpr,
    )
    try:
        repository_code_version = normalize_clean_code_version(
            resolve_code_version(repository_root)
        )
    except ValueError as exc:
        raise RandomizationAggregateProvenanceError(
            "aggregate writer 必须运行在可解析的 clean Git checkout"
        ) from exc
    expected_repeat_ids = formal_randomization_repeat_ids()
    if set(repeat_component_paths) != set(expected_repeat_ids) or len(
        repeat_component_paths
    ) != len(expected_repeat_ids):
        raise RandomizationAggregateProvenanceError(
            "repeat_component_paths 必须精确唯一覆盖权威9个 repeat"
        )
    if set(invariant_package_paths) != set(
        RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES
    ) or len(invariant_package_paths) != len(
        RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES
    ):
        raise RandomizationAggregateProvenanceError(
            "invariant_package_paths 必须精确覆盖3个不变官方参考包"
        )
    expected_target_fpr = resolved_target_fpr
    if (
        not math.isfinite(expected_target_fpr)
        or not 0.0 < expected_target_fpr < 1.0
    ):
        raise RandomizationAggregateProvenanceError(
            "target_fpr 必须是位于 (0, 1) 的有限数值"
        )
    resolved_repeat_paths = {
        repeat_id: _require_regular_zip(
            repeat_component_paths[repeat_id],
            role=f"repeat component {repeat_id}",
        )
        for repeat_id in expected_repeat_ids
    }
    resolved_invariant_paths = {
        package_family: _require_regular_zip(
            invariant_package_paths[package_family],
            role=f"invariant package {package_family}",
        )
        for package_family in RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES
    }
    all_paths = [
        *resolved_repeat_paths.values(),
        *resolved_invariant_paths.values(),
    ]
    if len({path.as_posix() for path in all_paths}) != len(all_paths):
        raise RandomizationAggregateProvenanceError(
            "聚合来源不得重复使用同一输入路径"
        )

    repeat_records = [
        _repeat_component_record(
            resolved_repeat_paths[repeat_id],
            paper_run_name=resolved_run_name,
            target_fpr=expected_target_fpr,
            randomization_repeat_id=repeat_id,
        )
        for repeat_id in expected_repeat_ids
    ]
    invariant_records = [
        _invariant_package_record(
            resolved_invariant_paths[package_family],
            package_family=package_family,
            paper_run_name=resolved_run_name,
            target_fpr=expected_target_fpr,
            repository_root=repository_root,
        )
        for package_family in RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES
    ]
    code_versions = {
        str(record["code_version"])
        for record in repeat_records + invariant_records
    }
    package_digests = {
        str(record["package_sha256"])
        for record in repeat_records + invariant_records
    }
    if len(code_versions) != 1 or re.fullmatch(
        r"[0-9a-f]{40}",
        next(iter(code_versions), ""),
    ) is None:
        raise RandomizationAggregateProvenanceError(
            "聚合来源12个输入必须共享 clean code version"
        )
    if code_versions != {repository_code_version}:
        raise RandomizationAggregateProvenanceError(
            "聚合来源 common_code_version 必须匹配当前 clean Git checkout"
        )
    if len(package_digests) != len(all_paths):
        raise RandomizationAggregateProvenanceError(
            "聚合来源不得重复使用同一输入 ZIP 字节"
        )
    common_code_version = next(iter(code_versions))
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = build_randomization_aggregate_payload(
        paper_run_name=resolved_run_name,
        target_fpr=expected_target_fpr,
        repeat_components=repeat_records,
        invariant_packages=invariant_records,
        common_code_version=common_code_version,
        generated_at=generated_at,
    )
    payload_bytes = _json_bytes(payload)
    manifest = _build_randomization_aggregate_manifest(
        payload,
        payload_bytes=payload_bytes,
    )
    manifest_bytes = _json_bytes(manifest)

    resolved_output_dir = _resolve_output_directory(
        repository_root=repository_root,
        output_dir=output_dir,
        paper_run_name=resolved_run_name,
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_token = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    ).lower()
    archive_path = resolved_output_dir / (
        f"randomization_aggregate_{timestamp_token}_{common_code_version[:7]}.zip"
    )
    payload_sidecar, manifest_sidecar = _sidecar_paths(archive_path)
    temporary_archive = archive_path.with_suffix(".zip.partial")
    temporary_payload = payload_sidecar.with_suffix(".json.partial")
    temporary_manifest = manifest_sidecar.with_suffix(".json.partial")
    temporary_paths = (
        temporary_archive,
        temporary_payload,
        temporary_manifest,
    )
    for temporary_path in temporary_paths:
        temporary_path.unlink(missing_ok=True)
    try:
        temporary_payload.write_bytes(payload_bytes)
        temporary_manifest.write_bytes(manifest_bytes)
        with ZipFile(
            temporary_archive,
            mode="w",
            compression=ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for repeat_id in expected_repeat_ids:
                archive.write(
                    resolved_repeat_paths[repeat_id],
                    _repeat_component_member_name(resolved_run_name, repeat_id),
                )
            for package_family in (
                RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES
            ):
                archive.write(
                    resolved_invariant_paths[package_family],
                    _invariant_package_member_name(
                        resolved_run_name,
                        package_family,
                    ),
                )
            archive.writestr(_payload_member_name(resolved_run_name), payload_bytes)
            archive.writestr(
                _manifest_member_name(resolved_run_name),
                manifest_bytes,
            )
        temporary_payload.replace(payload_sidecar)
        temporary_manifest.replace(manifest_sidecar)
        temporary_archive.replace(archive_path)
        provenance = validate_randomization_aggregate_provenance(
            archive_path,
            paper_run_name=resolved_run_name,
            target_fpr=expected_target_fpr,
        )
    except Exception:
        for path in (
            *temporary_paths,
            archive_path,
            payload_sidecar,
            manifest_sidecar,
        ):
            path.unlink(missing_ok=True)
        raise
    return provenance


def rebuild_randomization_aggregate_provenance_package(
    source_package_path: str | Path,
    *,
    paper_run_name: str,
    target_fpr: float,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
) -> RandomizationAggregateProvenance:
    """从已验证自包含 ZIP 安全提取12个原始输入并重新构造 aggregate."""

    resolved_run_name = normalize_paper_run_name(paper_run_name)
    resolved_target_fpr = validate_frozen_paper_run_target_fpr(
        resolved_run_name,
        target_fpr,
    )
    source = validate_randomization_aggregate_provenance(
        source_package_path,
        paper_run_name=resolved_run_name,
        target_fpr=resolved_target_fpr,
    )
    with tempfile.TemporaryDirectory(
        prefix="slm_wm_randomization_aggregate_rebuild_"
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        repeat_component_paths: dict[str, Path] = {}
        invariant_package_paths: dict[str, Path] = {}
        with ZipFile(source.package_path) as archive:
            for record in source.randomization_repeat_components:
                repeat_id = str(record["randomization_repeat_id"])
                destination = temporary_root / "repeat_components" / (
                    f"{repeat_id}.zip"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(str(record["archive_member"]), "r") as input_stream, (
                    destination.open("wb")
                ) as output_stream:
                    shutil.copyfileobj(
                        input_stream,
                        output_stream,
                        length=1024 * 1024,
                    )
                repeat_component_paths[repeat_id] = destination
            for record in source.invariant_packages:
                package_family = str(record["package_family"])
                destination = temporary_root / "invariant_packages" / (
                    f"{package_family}.zip"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(str(record["archive_member"]), "r") as input_stream, (
                    destination.open("wb")
                ) as output_stream:
                    shutil.copyfileobj(
                        input_stream,
                        output_stream,
                        length=1024 * 1024,
                    )
                invariant_package_paths[package_family] = destination

        return write_randomization_aggregate_provenance_package(
            repeat_component_paths,
            invariant_package_paths,
            paper_run_name=resolved_run_name,
            target_fpr=resolved_target_fpr,
            root=root,
            output_dir=output_dir,
        )


def parse_randomization_aggregate_input_paths(
    values: Iterable[str],
    *,
    expected_keys: tuple[str, ...],
    role: str,
) -> dict[str, Path]:
    """解析 ``身份=路径`` 参数并要求精确有序覆盖治理身份."""

    result: dict[str, Path] = {}
    for value in values:
        key, separator, raw_path = str(value).partition("=")
        normalized_key = key.strip()
        normalized_path = raw_path.strip()
        if not separator or not normalized_key or not normalized_path:
            raise ValueError(f"{role} 必须使用 身份=路径 格式")
        if normalized_key in result:
            raise ValueError(f"{role} 身份重复: {normalized_key}")
        result[normalized_key] = Path(normalized_path)
    if tuple(result) != expected_keys:
        missing = [key for key in expected_keys if key not in result]
        unexpected = [key for key in result if key not in expected_keys]
        raise ValueError(
            f"{role} 必须按规范顺序精确覆盖身份集合; "
            f"missing={missing}; unexpected={unexpected}"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    """构造可脱离外层 scripts 运行的 aggregate 重建 parser."""

    parser = argparse.ArgumentParser(
        description="复验精确9个 repeat 组件和3个不变包并写出自包含聚合来源包。"
    )
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument(
        "--paper-run-name",
        required=True,
        choices=tuple(RUN_DEFAULTS),
        help="论文运行层级。",
    )
    parser.add_argument(
        "--target-fpr",
        required=True,
        type=float,
        help="必须与论文运行层级冻结值一致的目标 FPR。",
    )
    parser.add_argument(
        "--repeat-component",
        action="append",
        default=[],
        metavar="REPEAT_ID=ZIP_PATH",
        help="按权威 repeat 顺序重复9次。",
    )
    parser.add_argument(
        "--invariant-package",
        action="append",
        default=[],
        metavar="PACKAGE_FAMILY=ZIP_PATH",
        help="按固定 family 顺序重复3次。",
    )
    parser.add_argument(
        "--rebuild-source-aggregate-package-path",
        default=None,
        help=(
            "从已验证自包含 aggregate ZIP 安全提取12个输入并重建; "
            "不得与显式 repeat/invariant 参数并用。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="可选输出目录, 必须位于仓库 outputs/ 下。",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """执行层内权威 CLI, 外层 script 只需转发该入口."""

    arguments = build_parser().parse_args(argv)
    paper_run_name = normalize_paper_run_name(arguments.paper_run_name)
    expected_target_fpr = validate_frozen_paper_run_target_fpr(
        paper_run_name,
        arguments.target_fpr,
    )
    if arguments.rebuild_source_aggregate_package_path is not None:
        if arguments.repeat_component or arguments.invariant_package:
            raise ValueError(
                "自包含 aggregate 重建不得同时接收显式输入包参数"
            )
        provenance = rebuild_randomization_aggregate_provenance_package(
            arguments.rebuild_source_aggregate_package_path,
            paper_run_name=paper_run_name,
            target_fpr=expected_target_fpr,
            root=arguments.root,
            output_dir=arguments.output_dir,
        )
    else:
        repeat_components = parse_randomization_aggregate_input_paths(
            arguments.repeat_component,
            expected_keys=formal_randomization_repeat_ids(),
            role="repeat component",
        )
        invariant_packages = parse_randomization_aggregate_input_paths(
            arguments.invariant_package,
            expected_keys=RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES,
            role="invariant package",
        )
        provenance = write_randomization_aggregate_provenance_package(
            repeat_components,
            invariant_packages,
            paper_run_name=paper_run_name,
            target_fpr=expected_target_fpr,
            root=arguments.root,
            output_dir=arguments.output_dir,
        )
    print(
        json.dumps(
            {
                "package_path": provenance.package_path.as_posix(),
                "package_sha256": provenance.package_sha256,
                "randomization_aggregate_digest": (
                    provenance.randomization_aggregate_digest
                ),
                "common_code_version": provenance.common_code_version,
                "randomization_aggregate_ready": True,
                "supports_paper_claim": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        end="",
    )


__all__ = [
    "RANDOMIZATION_AGGREGATE_FIELD_REGISTRY_SUGGESTIONS",
    "RANDOMIZATION_AGGREGATE_INVARIANT_PACKAGE_FAMILIES",
    "RANDOMIZATION_AGGREGATE_MANIFEST_ARTIFACT_ID",
    "RANDOMIZATION_AGGREGATE_OUTPUT_ROOT",
    "RANDOMIZATION_AGGREGATE_PAYLOAD_SCHEMA",
    "RANDOMIZATION_AGGREGATE_SCHEMA_VERSION",
    "RandomizationAggregateProvenance",
    "RandomizationAggregateProvenanceError",
    "build_randomization_aggregate_payload",
    "build_parser",
    "main",
    "parse_randomization_aggregate_input_paths",
    "rebuild_randomization_aggregate_provenance_package",
    "validate_randomization_aggregate_provenance",
    "write_randomization_aggregate_provenance_package",
]


if __name__ == "__main__":
    main()
