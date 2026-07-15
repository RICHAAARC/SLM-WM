"""把一个正式随机化重复的全部上游证据封装为自包含结果包。

该模块只负责单个 ``randomization_repeat_id`` 的输入闭合. 写包时保持上游
leaf ZIP 原始字节; 写后验证会在临时目录中逐包调用生产 selector 复验, 不向
持久化输出解压上游内容. 该模块不计算跨重复统计. 最终聚合层把这里生成的
结果包作为不可直接支持论文结论的原子输入, 并从全部 leaf ZIP 重建统计事实。
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any, Iterable, Mapping
from zipfile import BadZipFile, ZIP_STORED, ZipFile

from experiments.protocol.formal_randomization import (
    FORMAL_RANDOMIZATION_REPEAT_IDENTITY_FIELDS,
    formal_randomization_protocol_record,
    formal_randomization_repeat_registry_digest,
    resolve_formal_randomization_repeat,
    validate_formal_randomization_repeat_records,
)
from experiments.protocol.paper_run_config import (
    normalize_paper_run_name,
    validate_frozen_paper_run_target_fpr,
)
from paper_experiments.runners.closure_package_selection import (
    CLOSURE_PACKAGE_FAMILY_SPECS,
    ClosurePackageCandidate,
    inspect_closure_package,
    select_randomization_repeat_package_candidates,
)
from paper_experiments.analysis.paper_claim_field_policy import (
    find_zip_paper_claim_violation,
)


RANDOMIZATION_REPEAT_EVIDENCE_SCHEMA = "randomization_repeat_evidence_manifest"
RANDOMIZATION_REPEAT_EVIDENCE_SCHEMA_VERSION = 1
RANDOMIZATION_REPEAT_EVIDENCE_OUTPUT_ROOT = Path(
    "outputs/randomization_repeat_evidence"
)
RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES = (
    "image_only_dataset_runtime",
    "runtime_rerun_ablation",
    "branch_risk_parameter_sensitivity",
    "dataset_level_quality",
    "method_faithful_tree_ring",
    "method_faithful_gaussian_shading",
    "method_faithful_shallow_diffuse",
    "official_reference_t2smark",
)


class RandomizationRepeatEvidenceError(ValueError):
    """表示单重复证据包没有满足精确输入契约."""


def _stable_digest(value: Any) -> str:
    """计算 JSON 兼容对象的稳定 SHA-256."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    """流式计算普通文件的 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_member_sha256(archive: ZipFile, member_name: str) -> str:
    """流式计算 ZIP 成员的 SHA-256, 避免把大型 leaf 包整体载入内存."""

    digest = hashlib.sha256()
    with archive.open(member_name, "r") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repeat_identity(randomization_repeat_id: str) -> dict[str, Any]:
    """从权威注册表构造单重复身份记录."""

    repeat = resolve_formal_randomization_repeat(randomization_repeat_id)
    protocol = formal_randomization_protocol_record()
    return {
        **repeat.to_dict(),
        "formal_randomization_protocol_digest": protocol[
            "formal_randomization_protocol_digest"
        ],
    }


def _candidate_repeat_identity(candidate: ClosurePackageCandidate) -> dict[str, Any]:
    """提取选择器已经从 leaf 包双来源核验的重复身份."""

    return {
        "randomization_repeat_id": str(candidate.randomization_repeat_id),
        "generation_seed_index": int(candidate.generation_seed_index),
        "generation_seed_offset": int(candidate.generation_seed_offset),
        "watermark_key_index": int(candidate.watermark_key_index),
        "formal_randomization_protocol_digest": str(
            candidate.formal_randomization_protocol_digest
        ),
    }


def _leaf_member_name(
    *,
    randomization_repeat_id: str,
    package_family: str,
) -> str:
    """构造不会与其他重复或 family 冲突的 leaf ZIP 成员名."""

    return (
        f"randomization_repeat_evidence/{randomization_repeat_id}/"
        f"leaf_packages/{package_family}.zip"
    )


def _manifest_member_name(randomization_repeat_id: str) -> str:
    """返回单重复证据包内唯一 manifest 成员名."""

    return (
        f"randomization_repeat_evidence/{randomization_repeat_id}/"
        "randomization_repeat_evidence_manifest.json"
    )


def _canonical_candidates(
    candidates: Iterable[ClosurePackageCandidate],
    *,
    paper_run_name: str,
    target_fpr: float,
    randomization_repeat_id: str,
) -> tuple[ClosurePackageCandidate, ...]:
    """要求候选精确覆盖全部 leaf 包并共享同一重复和代码版本。"""

    materialized = tuple(candidates)
    actual_families = tuple(candidate.package_family for candidate in materialized)
    if actual_families != RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES:
        raise RandomizationRepeatEvidenceError(
            "单重复证据必须按稳定顺序精确覆盖全部随机化 leaf 包"
        )
    expected_identity = _repeat_identity(randomization_repeat_id)
    expected_run_name = normalize_paper_run_name(paper_run_name)
    expected_target_fpr = float(target_fpr)
    code_versions = set()
    for candidate in materialized:
        if candidate.paper_run_name != expected_run_name:
            raise RandomizationRepeatEvidenceError("leaf 包论文运行层级不一致")
        if not math.isclose(
            float(candidate.target_fpr),
            expected_target_fpr,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RandomizationRepeatEvidenceError("leaf 包 target_fpr 不一致")
        if str(candidate.randomization_scope) != "active_repeat_component":
            raise RandomizationRepeatEvidenceError(
                "单重复证据不得混入 cross-repeat invariant 官方参考包"
            )
        if _candidate_repeat_identity(candidate) != expected_identity:
            raise RandomizationRepeatEvidenceError("leaf 包随机化重复身份不一致")
        unresolved_package_path = Path(candidate.package_path).expanduser()
        if unresolved_package_path.is_symlink():
            raise RandomizationRepeatEvidenceError("leaf 包不得是符号链接")
        package_path = unresolved_package_path.resolve()
        if (
            not package_path.is_file()
            or package_path.suffix.lower() != ".zip"
        ):
            raise RandomizationRepeatEvidenceError("leaf 包必须是普通 ZIP 文件")
        if _file_sha256(package_path) != str(candidate.package_sha256).lower():
            raise RandomizationRepeatEvidenceError("leaf 包字节摘要已漂移")
        code_versions.add(str(candidate.code_version))
    if len(code_versions) != 1:
        raise RandomizationRepeatEvidenceError("全部 leaf 包必须共享同一代码版本")
    code_version = next(iter(code_versions))
    if re.fullmatch(r"[0-9a-f]{40}", code_version) is None:
        raise RandomizationRepeatEvidenceError("leaf 包代码版本必须是40位小写 Git commit")
    return materialized


def build_randomization_repeat_evidence_manifest(
    candidates: Iterable[ClosurePackageCandidate],
    *,
    paper_run_name: str,
    target_fpr: float,
    randomization_repeat_id: str,
    generated_at: str,
) -> dict[str, Any]:
    """构造精确绑定8个 leaf ZIP 的单重复 manifest."""

    resolved_run_name = normalize_paper_run_name(paper_run_name)
    resolved_target_fpr = validate_frozen_paper_run_target_fpr(
        resolved_run_name,
        target_fpr,
    )
    canonical = _canonical_candidates(
        candidates,
        paper_run_name=resolved_run_name,
        target_fpr=resolved_target_fpr,
        randomization_repeat_id=randomization_repeat_id,
    )
    repeat_identity = _repeat_identity(randomization_repeat_id)
    code_version = str(canonical[0].code_version)
    leaf_records = []
    for candidate in canonical:
        package_path = Path(candidate.package_path).resolve()
        leaf_records.append(
            {
                "package_family": candidate.package_family,
                "archive_member": _leaf_member_name(
                    randomization_repeat_id=randomization_repeat_id,
                    package_family=candidate.package_family,
                ),
                "package_sha256": str(candidate.package_sha256).lower(),
                "code_version": str(candidate.code_version),
                "formal_execution_run_lock_digest": str(
                    candidate.formal_execution_run_lock_digest
                ),
                "formal_execution_package_lock_digest": str(
                    candidate.formal_execution_package_lock_digest
                ),
            }
        )
    leaf_package_sha256_map = {
        str(record["package_family"]): str(record["package_sha256"])
        for record in leaf_records
    }
    content_payload = {
        "paper_run_name": resolved_run_name,
        "target_fpr": resolved_target_fpr,
        **repeat_identity,
        "formal_randomization_repeat_registry_digest": (
            formal_randomization_repeat_registry_digest()
        ),
        "code_version": code_version,
        "leaf_packages": leaf_records,
        "repeat_component_ready": True,
        "randomization_aggregate_ready": False,
        "supports_paper_claim": False,
    }
    payload = {
        "report_schema": RANDOMIZATION_REPEAT_EVIDENCE_SCHEMA,
        "schema_version": RANDOMIZATION_REPEAT_EVIDENCE_SCHEMA_VERSION,
        **content_payload,
        "generated_at": str(generated_at),
        "leaf_package_family_count": len(leaf_records),
        "leaf_package_families": list(RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES),
        "leaf_packages": leaf_records,
        "leaf_package_sha256_map": leaf_package_sha256_map,
        "leaf_package_set_digest": _stable_digest(leaf_records),
        "component_content_digest": _stable_digest(content_payload),
    }
    payload["randomization_repeat_evidence_manifest_digest"] = _stable_digest(
        payload
    )
    return payload


def _manifest_digest_ready(manifest: Mapping[str, Any]) -> bool:
    """复算 manifest 自身摘要, 拒绝自由修改治理字段."""

    declared = str(
        manifest.get("randomization_repeat_evidence_manifest_digest", "")
    )
    payload = dict(manifest)
    payload.pop("randomization_repeat_evidence_manifest_digest", None)
    return re.fullmatch(r"[0-9a-f]{64}", declared) is not None and (
        _stable_digest(payload) == declared
    )


def _validate_nested_leaf_packages(
    archive: ZipFile,
    leaf_records: list[dict[str, Any]],
    *,
    paper_run_name: str,
    target_fpr: float,
    randomization_repeat_id: str,
) -> None:
    """独立复验7个嵌套 leaf ZIP 的包内科学与治理契约."""

    specification_by_family = {
        specification.package_family: specification
        for specification in CLOSURE_PACKAGE_FAMILY_SPECS
        if specification.package_family
        in RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES
    }
    try:
        with tempfile.TemporaryDirectory(
            prefix="slm_wm_repeat_evidence_"
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            for record in leaf_records:
                family = str(record["package_family"])
                member_name = str(record["archive_member"])
                specification = specification_by_family[family]
                temporary_leaf = temporary_root / f"{family}.zip"
                with archive.open(member_name, "r") as source, (
                    temporary_leaf.open("wb")
                ) as destination:
                    shutil.copyfileobj(
                        source,
                        destination,
                        length=1024 * 1024,
                    )
                violation = find_zip_paper_claim_violation(temporary_leaf)
                if violation is not None:
                    raise RandomizationRepeatEvidenceError(
                        "单重复 leaf 包包含正向论文结论字段: "
                        f"{violation.path}"
                    )
                candidate = inspect_closure_package(
                    temporary_leaf,
                    spec=specification,
                    paper_run_name=paper_run_name,
                    target_fpr=target_fpr,
                    randomization_repeat_id=randomization_repeat_id,
                )
                if (
                    candidate.package_family != family
                    or candidate.package_sha256
                    != str(record["package_sha256"])
                    or candidate.code_version
                    != str(record["code_version"])
                    or candidate.formal_execution_run_lock_digest
                    != str(record["formal_execution_run_lock_digest"])
                    or candidate.formal_execution_package_lock_digest
                    != str(record["formal_execution_package_lock_digest"])
                ):
                    raise RandomizationRepeatEvidenceError(
                        f"嵌套 leaf ZIP 复验身份不一致: {family}"
                    )
    except (KeyError, OSError, ValueError) as exc:
        if isinstance(exc, RandomizationRepeatEvidenceError):
            raise
        raise RandomizationRepeatEvidenceError(
            "嵌套 leaf ZIP 未通过独立包内契约复验"
        ) from exc


def validate_randomization_repeat_evidence_package(
    archive_path: str | Path,
    *,
    paper_run_name: str,
    target_fpr: float,
    randomization_repeat_id: str,
) -> dict[str, Any]:
    """写后或消费时复验外层包、manifest 与8个 leaf ZIP 的字节绑定."""

    unresolved_path = Path(archive_path).expanduser()
    if unresolved_path.is_symlink():
        raise RandomizationRepeatEvidenceError("单重复证据包不得是符号链接")
    path = unresolved_path.resolve()
    if not path.is_file() or path.suffix.lower() != ".zip":
        raise RandomizationRepeatEvidenceError("单重复证据包必须是普通 ZIP 文件")
    expected_manifest_member = _manifest_member_name(randomization_repeat_id)
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or any(info.is_dir() for info in infos):
                raise RandomizationRepeatEvidenceError("单重复证据包成员必须唯一且均为文件")
            for info in infos:
                member = PurePosixPath(info.filename)
                if (
                    member.is_absolute()
                    or ".." in member.parts
                    or member.as_posix() != info.filename
                    or stat.S_ISLNK(info.external_attr >> 16)
                ):
                    raise RandomizationRepeatEvidenceError("单重复证据包包含不安全成员")
            damaged_member = archive.testzip()
            if damaged_member is not None:
                raise RandomizationRepeatEvidenceError(
                    f"单重复证据包 CRC 失败: {damaged_member}"
                )
            if expected_manifest_member not in names:
                raise RandomizationRepeatEvidenceError("单重复证据包缺少规范 manifest")
            try:
                manifest = json.loads(
                    archive.read(expected_manifest_member).decode("utf-8-sig")
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RandomizationRepeatEvidenceError("单重复 manifest 不是有效 JSON") from exc
            if not isinstance(manifest, dict):
                raise RandomizationRepeatEvidenceError("单重复 manifest 必须是 JSON object")
            expected_identity = _repeat_identity(randomization_repeat_id)
            expected_repeat_identity = {
                field_name: expected_identity[field_name]
                for field_name in FORMAL_RANDOMIZATION_REPEAT_IDENTITY_FIELDS
            }
            try:
                normalized_repeat_identity = (
                    validate_formal_randomization_repeat_records(
                        [manifest],
                        require_exact_registry=False,
                    )[0]
                )
            except (TypeError, ValueError) as exc:
                raise RandomizationRepeatEvidenceError(
                    "单重复 manifest 正式随机化身份无效"
                ) from exc
            identity_ready = (
                normalized_repeat_identity == expected_repeat_identity
                and manifest.get("formal_randomization_protocol_digest")
                == expected_identity["formal_randomization_protocol_digest"]
            )
            leaf_records = manifest.get("leaf_packages")
            if not isinstance(leaf_records, list):
                raise RandomizationRepeatEvidenceError("单重复 manifest 缺少 leaf 记录")
            families = tuple(
                str(record.get("package_family", ""))
                for record in leaf_records
                if isinstance(record, dict)
            )
            expected_leaf_names = {
                str(record.get("archive_member", ""))
                for record in leaf_records
                if isinstance(record, dict)
            }
            expected_names = expected_leaf_names | {expected_manifest_member}
            leaf_sha256_map = manifest.get("leaf_package_sha256_map")
            records_ready = (
                families == RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES
                and len(leaf_records) == len(RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES)
                and len(expected_leaf_names) == len(leaf_records)
                and all(
                    str(record.get("archive_member", ""))
                    == _leaf_member_name(
                        randomization_repeat_id=randomization_repeat_id,
                        package_family=str(record.get("package_family", "")),
                    )
                    for record in leaf_records
                )
                and set(names) == expected_names
                and isinstance(leaf_sha256_map, dict)
                and leaf_sha256_map
                == {
                    str(record["package_family"]): str(record["package_sha256"])
                    for record in leaf_records
                }
                and manifest.get("leaf_package_set_digest")
                == _stable_digest(leaf_records)
                and all(
                    re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(record.get("package_sha256", "")),
                    )
                    is not None
                    and _zip_member_sha256(
                        archive,
                        str(record.get("archive_member", "")),
                    )
                    == str(record.get("package_sha256", ""))
                    and str(record.get("code_version", ""))
                    == str(manifest.get("code_version", ""))
                    for record in leaf_records
                )
            )
            if records_ready:
                _validate_nested_leaf_packages(
                    archive,
                    leaf_records,
                    paper_run_name=paper_run_name,
                    target_fpr=target_fpr,
                    randomization_repeat_id=randomization_repeat_id,
                )
            ready = all(
                (
                    manifest.get("report_schema")
                    == RANDOMIZATION_REPEAT_EVIDENCE_SCHEMA,
                    type(manifest.get("schema_version")) is int,
                    manifest.get("schema_version")
                    == RANDOMIZATION_REPEAT_EVIDENCE_SCHEMA_VERSION,
                    manifest.get("paper_run_name")
                    == normalize_paper_run_name(paper_run_name),
                    math.isclose(
                        float(manifest.get("target_fpr", math.nan)),
                        float(target_fpr),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ),
                    identity_ready,
                    manifest.get(
                        "formal_randomization_repeat_registry_digest"
                    )
                    == formal_randomization_repeat_registry_digest(),
                    re.fullmatch(
                        r"[0-9a-f]{40}", str(manifest.get("code_version", ""))
                    )
                    is not None,
                    manifest.get("leaf_package_family_count")
                    == len(RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES),
                    manifest.get("leaf_package_families")
                    == list(RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES),
                    records_ready,
                    manifest.get("component_content_digest")
                    == _stable_digest(
                        {
                            "paper_run_name": manifest.get("paper_run_name"),
                            "target_fpr": manifest.get("target_fpr"),
                            **expected_identity,
                            "formal_randomization_repeat_registry_digest": (
                                formal_randomization_repeat_registry_digest()
                            ),
                            "code_version": manifest.get("code_version"),
                            "leaf_packages": leaf_records,
                            "repeat_component_ready": True,
                            "randomization_aggregate_ready": False,
                            "supports_paper_claim": False,
                        }
                    ),
                    manifest.get("repeat_component_ready") is True,
                    manifest.get("randomization_aggregate_ready") is False,
                    manifest.get("supports_paper_claim") is False,
                    _manifest_digest_ready(manifest),
                )
            )
            if not ready:
                raise RandomizationRepeatEvidenceError(
                    "单重复证据包身份、leaf 摘要或结论边界未通过"
                )
    except (BadZipFile, EOFError, OSError, RuntimeError) as exc:
        if isinstance(exc, RandomizationRepeatEvidenceError):
            raise
        raise RandomizationRepeatEvidenceError("单重复证据包不可读取") from exc
    return {
        "archive_path": path.as_posix(),
        "archive_sha256": _file_sha256(path),
        "archive_entry_count": len(RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES) + 1,
        "paper_run_name": normalize_paper_run_name(paper_run_name),
        **_repeat_identity(randomization_repeat_id),
        "code_version": str(manifest["code_version"]),
        "leaf_package_sha256_map": dict(manifest["leaf_package_sha256_map"]),
        "randomization_repeat_evidence_manifest_digest": str(
            manifest["randomization_repeat_evidence_manifest_digest"]
        ),
        "component_content_digest": str(manifest["component_content_digest"]),
        "repeat_component_ready": True,
        "randomization_aggregate_ready": False,
        "supports_paper_claim": False,
    }


def write_randomization_repeat_evidence_package(
    package_search_root: str | Path,
    *,
    paper_run_name: str,
    target_fpr: float,
    randomization_repeat_id: str,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """选择全部 leaf ZIP, 不解压地写出单重复自包含证据包。"""

    repository_root = Path(root).resolve()
    resolved_run_name = normalize_paper_run_name(paper_run_name)
    resolved_target_fpr = validate_frozen_paper_run_target_fpr(
        resolved_run_name,
        target_fpr,
    )
    candidates = select_randomization_repeat_package_candidates(
        package_search_root,
        paper_run_name=resolved_run_name,
        target_fpr=resolved_target_fpr,
        randomization_repeat_id=randomization_repeat_id,
        root=repository_root,
    )
    canonical = _canonical_candidates(
        candidates,
        paper_run_name=resolved_run_name,
        target_fpr=resolved_target_fpr,
        randomization_repeat_id=randomization_repeat_id,
    )
    resolved_output_dir = (
        repository_root
        / RANDOMIZATION_REPEAT_EVIDENCE_OUTPUT_ROOT
        / resolved_run_name
        / randomization_repeat_id
        if output_dir is None
        else Path(output_dir)
    )
    resolved_output_dir = (
        resolved_output_dir.expanduser().resolve()
        if resolved_output_dir.is_absolute()
        else (repository_root / resolved_output_dir).expanduser().resolve()
    )
    try:
        resolved_output_dir.relative_to((repository_root / "outputs").resolve())
    except ValueError as exc:
        raise RandomizationRepeatEvidenceError(
            "单重复证据输出目录必须位于 outputs 下"
        ) from exc
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    timestamp_token = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    ).lower()
    code_version = str(canonical[0].code_version)
    # 文件名保持职责语义, 同时控制长度以兼容 Windows 深层测试目录.
    archive_name = (
        f"repeat_evidence_{randomization_repeat_id}_{timestamp_token}_"
        f"{code_version[:7]}.zip"
    )
    archive_path = resolved_output_dir / archive_name
    temporary_path = archive_path.with_suffix(".zip.partial")
    temporary_path.unlink(missing_ok=True)
    manifest = build_randomization_repeat_evidence_manifest(
        canonical,
        paper_run_name=resolved_run_name,
        target_fpr=resolved_target_fpr,
        randomization_repeat_id=randomization_repeat_id,
        generated_at=generated_at,
    )
    try:
        with ZipFile(
            temporary_path,
            mode="w",
            compression=ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for candidate, leaf_record in zip(
                canonical,
                manifest["leaf_packages"],
            ):
                archive.write(
                    Path(candidate.package_path),
                    str(leaf_record["archive_member"]),
                )
            archive.writestr(
                _manifest_member_name(randomization_repeat_id),
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
        temporary_path.replace(archive_path)
        report = validate_randomization_repeat_evidence_package(
            archive_path,
            paper_run_name=resolved_run_name,
            target_fpr=resolved_target_fpr,
            randomization_repeat_id=randomization_repeat_id,
        )
    except Exception:
        temporary_path.unlink(missing_ok=True)
        archive_path.unlink(missing_ok=True)
        raise
    return report


__all__ = [
    "RANDOMIZATION_REPEAT_EVIDENCE_OUTPUT_ROOT",
    "RANDOMIZATION_REPEAT_EVIDENCE_SCHEMA",
    "RANDOMIZATION_REPEAT_EVIDENCE_SCHEMA_VERSION",
    "RANDOMIZATION_REPEAT_LEAF_PACKAGE_FAMILIES",
    "RandomizationRepeatEvidenceError",
    "build_randomization_repeat_evidence_manifest",
    "validate_randomization_repeat_evidence_package",
    "write_randomization_repeat_evidence_package",
]
