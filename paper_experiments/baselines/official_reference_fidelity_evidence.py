"""核验三个官方参考复现 family 的方法忠实度证据边界.

该模块只审计 Tree-Ring,Gaussian Shading 和 Shallow Diffuse 的官方原始环境复现.
这些结果用于补充方法忠实度证据, 不进入 SD3.5 common-backbone 主表优势比较.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from experiments.runtime.repository_environment import (
    FormalExecutionLockError,
    normalize_formal_git_commit,
)
from main.core.digest import build_stable_digest


OFFICIAL_REFERENCE_BASELINE_IDS = (
    "tree_ring",
    "gaussian_shading",
    "shallow_diffuse",
)
SUPPLEMENTAL_METHOD_FIDELITY_ROLE = "supplemental_method_fidelity_reference"
ARCHIVE_GOVERNANCE_SCOPE = "external_summary_records_final_archive_digest"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class OfficialReferenceFidelityEvidenceError(ValueError):
    """表示官方参考方法忠实度证据没有满足正式边界."""


@dataclass(frozen=True)
class OfficialReferenceFamilySpec:
    """描述一个官方参考 family 的固定文件名和身份."""

    baseline_id: str
    output_family: str
    summary_name: str
    records_name: str
    validation_name: str
    package_input_name: str
    archive_summary_name: str
    archive_manifest_name: str

    @property
    def official_ready_field(self) -> str:
        """返回该方法 summary 中的官方参考 ready 字段."""

        return f"{self.baseline_id}_official_reference_ready"

    @property
    def run_manifest_artifact_id(self) -> str:
        """返回该方法运行 manifest 的固定 artifact 身份."""

        return f"{self.baseline_id}_official_reference_manifest"

    @property
    def archive_manifest_artifact_id(self) -> str:
        """返回该方法归档 manifest 的固定 artifact 身份."""

        return f"{self.baseline_id}_official_reference_archive_manifest"

    @property
    def archive_name_prefix(self) -> str:
        """返回该方法归档文件的固定名称前缀."""

        return f"external_baseline_official_reference_package_{self.baseline_id}_"


@dataclass(frozen=True)
class OfficialReferenceSourceAudit:
    """保存一个官方参考 family 的证据记录和实际输入路径."""

    evidence_record: dict[str, Any]
    input_paths: tuple[Path, ...]


def _family_spec(baseline_id: str) -> OfficialReferenceFamilySpec:
    """按统一命名规则构造官方参考 family 描述."""

    return OfficialReferenceFamilySpec(
        baseline_id=baseline_id,
        output_family=f"outputs/{baseline_id}_official_reference",
        summary_name=f"{baseline_id}_official_reference_summary.json",
        records_name=f"{baseline_id}_official_reference_records.jsonl",
        validation_name=f"{baseline_id}_official_reference_validation_report.json",
        package_input_name=f"{baseline_id}_official_reference_package_input_manifest.json",
        archive_summary_name=f"{baseline_id}_official_reference_archive_summary.json",
        archive_manifest_name=f"{baseline_id}_official_reference_archive_manifest.local.json",
    )


OFFICIAL_REFERENCE_FAMILY_SPECS = tuple(
    _family_spec(baseline_id) for baseline_id in OFFICIAL_REFERENCE_BASELINE_IDS
)


def normalize_clean_code_version(value: Any) -> str:
    """要求官方参考证据使用精确40位小写 clean Git 提交 SHA."""

    try:
        return normalize_formal_git_commit(value)
    except FormalExecutionLockError as exc:
        raise OfficialReferenceFidelityEvidenceError(
            "code_version 必须是精确40位小写 clean Git 提交 SHA"
        ) from exc


def file_sha256(path: Path) -> str:
    """流式计算文件 SHA-256, 可复用于大型官方复现输出."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    """集中实现正式输入边界的 fail-closed 断言."""

    if not condition:
        raise OfficialReferenceFidelityEvidenceError(message)


def _read_json_object(path: Path, role: str) -> dict[str, Any]:
    """读取一个必须存在的 JSON object."""

    _require(path.is_file() and not path.is_symlink(), f"{role} 必须是普通文件: {path.as_posix()}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OfficialReferenceFidelityEvidenceError(f"{role} 不是有效 JSON") from error
    _require(isinstance(payload, dict), f"{role} 必须是 JSON object")
    return dict(payload)


def _read_jsonl_records(path: Path, role: str) -> list[dict[str, Any]]:
    """读取非空 JSONL 记录并拒绝非 object 行."""

    _require(path.is_file() and not path.is_symlink(), f"{role} 必须是普通文件: {path.as_posix()}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise OfficialReferenceFidelityEvidenceError(
                f"{role} 第{line_number}行不是有效 JSON"
            ) from error
        _require(isinstance(row, dict), f"{role} 第{line_number}行必须是 JSON object")
        rows.append(dict(row))
    _require(bool(rows), f"{role} 不得为空")
    return rows


def _relative_path(path: Path, root_path: Path) -> str:
    """返回受仓库根目录约束的规范 POSIX 相对路径."""

    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError as error:
        raise OfficialReferenceFidelityEvidenceError(
            f"正式证据路径越出仓库根目录: {path.as_posix()}"
        ) from error


def _normalized_declared_path(value: Any, role: str) -> str:
    """校验 package input manifest 中的仓库相对成员路径."""

    _require(isinstance(value, str), f"{role} 必须是字符串")
    raw_path = value
    pure_path = PurePosixPath(raw_path)
    _require(
        bool(raw_path)
        and "\\" not in raw_path
        and not pure_path.is_absolute()
        and all(part not in {"", ".", ".."} for part in pure_path.parts)
        and pure_path.as_posix() == raw_path,
        f"{role} 不是规范仓库相对路径: {raw_path}",
    )
    return raw_path


def _target_fpr_matches(value: Any, expected_target_fpr: float) -> bool:
    """判断输入 FPR 是否与当前论文运行协议精确一致."""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric_value) and math.isclose(
        numeric_value,
        expected_target_fpr,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def _require_timezone_timestamp(value: Any, role: str) -> None:
    """要求治理时间携带明确时区."""

    _require(isinstance(value, str) and bool(value.strip()), f"{role} 必须是 ISO-8601 时间")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise OfficialReferenceFidelityEvidenceError(f"{role} 不是有效 ISO-8601 时间") from error
    _require(parsed.tzinfo is not None and parsed.utcoffset() is not None, f"{role} 必须携带时区")


def _source_paths(source_dir: Path, spec: OfficialReferenceFamilySpec) -> dict[str, Path]:
    """集中构造审计必须读取的七类官方参考治理文件."""

    return {
        "summary": source_dir / spec.summary_name,
        "run_manifest": source_dir / "manifest.local.json",
        "records": source_dir / spec.records_name,
        "validation_report": source_dir / spec.validation_name,
        "package_input_manifest": source_dir / spec.package_input_name,
        "archive_summary": source_dir / spec.archive_summary_name,
        "archive_manifest": source_dir / spec.archive_manifest_name,
    }


def _validate_summary_and_records(
    *,
    spec: OfficialReferenceFamilySpec,
    paper_run_name: str,
    target_fpr: float,
    paths: Mapping[str, Path],
    root_path: Path,
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> str:
    """核验运行摘要,逐方法记录,validation 和运行 manifest."""

    baseline_id = spec.baseline_id
    _require(summary.get("baseline_id") == baseline_id, f"{baseline_id} summary 的 baseline 身份不一致")
    _require(summary.get("paper_claim_scale") == paper_run_name, f"{baseline_id} summary 的论文运行层级不一致")
    _require(_target_fpr_matches(summary.get("target_fpr"), target_fpr), f"{baseline_id} summary 的 target_fpr 不一致")
    _require(summary.get("run_decision") == "pass", f"{baseline_id} summary 的 run_decision 未通过")
    _require(summary.get(spec.official_ready_field) is True, f"{baseline_id} 官方参考 ready 未通过")
    _require(summary.get("reference_import_ready") is True, f"{baseline_id} reference import 未通过")
    _require(summary.get("main_table_eligible") is False, f"{baseline_id} 官方参考不得进入主表")
    _require(summary.get("supports_paper_claim") is False, f"{baseline_id} 上游官方参考不得直接声明主表结论")
    _require(
        int(summary.get("governed_reference_record_count", -1)) == len(records),
        f"{baseline_id} summary 的 governed record 数量不一致",
    )
    expected_summary_paths = {
        "summary_path": _relative_path(paths["summary"], root_path),
        "reference_records_path": _relative_path(paths["records"], root_path),
        "reference_validation_path": _relative_path(paths["validation_report"], root_path),
    }
    for field_name, expected_path in expected_summary_paths.items():
        _require(summary.get(field_name) == expected_path, f"{baseline_id} summary 的 {field_name} 未绑定当前文件")

    _require(
        all(row.get("baseline_id") == baseline_id for row in records),
        f"{baseline_id} records 包含其他 baseline 身份",
    )
    _require(
        all(row.get("supplemental_table_role") == SUPPLEMENTAL_METHOD_FIDELITY_ROLE for row in records),
        f"{baseline_id} records 的补充表职责不一致",
    )
    _require(
        all(row.get("main_table_eligible") is False for row in records),
        f"{baseline_id} records 不得进入主表",
    )
    _require(
        all(row.get("supports_paper_claim") is False for row in records),
        f"{baseline_id} records 不得直接支持主表论文结论",
    )

    validation_ready = (
        validation.get("reference_import_ready") is True
        and int(validation.get("input_record_count", -1)) == len(records)
        and int(validation.get("accepted_reference_record_count", -1)) == len(records)
        and int(validation.get("rejected_reference_record_count", -1)) == 0
        and int(validation.get("reference_issue_count", -1)) == 0
        and validation.get("issues") == []
        and validation.get("accepted_records") == list(records)
    )
    _require(
        validation_ready,
        f"{baseline_id} validation 存在拒绝记录, 问题或计数不一致",
    )
    summary_metadata = summary.get("metadata")
    _require(isinstance(summary_metadata, dict), f"{baseline_id} summary 缺少 metadata")
    _require(
        summary_metadata.get("validation") == validation,
        f"{baseline_id} summary 未绑定当前 validation 内容",
    )

    run_metadata = run_manifest.get("metadata")
    _require(isinstance(run_metadata, dict), f"{baseline_id} run manifest 缺少 metadata")
    _require(run_manifest.get("artifact_id") == spec.run_manifest_artifact_id, f"{baseline_id} run manifest 身份不一致")
    _require(run_manifest.get("artifact_type") == "local_manifest", f"{baseline_id} run manifest 类型不一致")
    _require(run_metadata.get("run_decision") == "pass", f"{baseline_id} run manifest 的决策未通过")
    _require(run_metadata.get(spec.official_ready_field) is True, f"{baseline_id} run manifest 的 ready 未通过")
    _require(run_metadata.get("main_table_eligible") is False, f"{baseline_id} run manifest 错误声明主表资格")
    _require(run_metadata.get("supports_paper_claim") is False, f"{baseline_id} run manifest 错误声明主表结论")
    output_paths = run_manifest.get("output_paths")
    _require(isinstance(output_paths, list), f"{baseline_id} run manifest 缺少 output_paths")
    required_outputs = {
        _relative_path(paths[role], root_path)
        for role in ("summary", "run_manifest", "records", "validation_report")
    }
    _require(required_outputs <= set(output_paths), f"{baseline_id} run manifest 未绑定必要输出")
    return normalize_clean_code_version(run_manifest.get("code_version"))


def _validate_package_entries(
    *,
    spec: OfficialReferenceFamilySpec,
    source_dir: Path,
    root_path: Path,
    paths: Mapping[str, Path],
    paper_run_name: str,
    target_fpr: float,
    package_input: Mapping[str, Any],
) -> tuple[list[str], dict[str, str], str]:
    """按 producer 语义核验动态 entry 清单和逐文件 SHA-256."""

    baseline_id = spec.baseline_id
    _require(package_input.get("baseline_id") == baseline_id, f"{baseline_id} package input 身份不一致")
    _require(package_input.get("paper_run_name") == paper_run_name, f"{baseline_id} package input 论文运行层级不一致")
    _require(_target_fpr_matches(package_input.get("target_fpr"), target_fpr), f"{baseline_id} package input target_fpr 不一致")
    _require(package_input.get("embedded_digest_scope") == ARCHIVE_GOVERNANCE_SCOPE, f"{baseline_id} package input 治理范围不一致")
    _require_timezone_timestamp(package_input.get("generated_at"), f"{baseline_id} package input generated_at")

    raw_entry_paths = package_input.get("entry_paths")
    raw_entry_sha256 = package_input.get("entry_sha256")
    _require(isinstance(raw_entry_paths, list), f"{baseline_id} package input 缺少 entry_paths")
    _require(isinstance(raw_entry_sha256, dict), f"{baseline_id} package input 缺少 entry_sha256")
    entry_paths = [
        _normalized_declared_path(value, f"{baseline_id} entry_paths")
        for value in raw_entry_paths
    ]
    _require(bool(entry_paths), f"{baseline_id} package input 的 entry_paths 不得为空")
    _require(len(entry_paths) == len(set(entry_paths)), f"{baseline_id} package input 的 entry_paths 包含重复项")
    _require(entry_paths == sorted(entry_paths), f"{baseline_id} package input 的 entry_paths 必须按 producer 顺序排序")
    _require(set(raw_entry_sha256) == set(entry_paths), f"{baseline_id} entry_sha256 键与 entry_paths 不一致")
    entry_sha256: dict[str, str] = {}
    for entry_path in entry_paths:
        declared_digest = raw_entry_sha256[entry_path]
        _require(
            isinstance(declared_digest, str)
            and SHA256_PATTERN.fullmatch(declared_digest.strip().lower()) is not None,
            f"{baseline_id} 声明了非法 SHA-256: {entry_path}",
        )
        entry_sha256[entry_path] = declared_digest.strip().lower()

    governance_paths = {
        _relative_path(paths[role], root_path)
        for role in ("package_input_manifest", "archive_summary", "archive_manifest")
    }
    governance_files = {
        paths[role].resolve()
        for role in ("package_input_manifest", "archive_summary", "archive_manifest")
    }
    _require(not governance_paths.intersection(entry_paths), f"{baseline_id} 动态 entry 不得包含后写入的归档治理文件")
    expected_entry_count = len(entry_paths) + len(governance_paths)
    _require(int(package_input.get("entry_count", -1)) == expected_entry_count, f"{baseline_id} package entry_count 不符合 producer 语义")

    actual_dynamic_paths: list[str] = []
    for path in sorted(source_dir.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() == ".zip"
            or path.resolve() in governance_files
        ):
            continue
        _require(not path.is_symlink(), f"{baseline_id} 动态 entry 不得是符号链接: {path.as_posix()}")
        actual_dynamic_paths.append(_relative_path(path, root_path))
    _require(actual_dynamic_paths == entry_paths, f"{baseline_id} package input 未精确覆盖已物化动态文件")

    required_dynamic_paths = {
        _relative_path(paths[role], root_path)
        for role in ("summary", "run_manifest", "records", "validation_report")
    }
    _require(required_dynamic_paths <= set(entry_paths), f"{baseline_id} package input 未绑定必要运行证据")
    for entry_path in entry_paths:
        resolved_entry = (root_path / Path(*PurePosixPath(entry_path).parts)).resolve()
        try:
            resolved_entry.relative_to(source_dir.resolve())
        except ValueError as error:
            raise OfficialReferenceFidelityEvidenceError(
                f"{baseline_id} package entry 越出当前 family: {entry_path}"
            ) from error
        _require(resolved_entry.is_file() and not resolved_entry.is_symlink(), f"{baseline_id} package entry 不存在: {entry_path}")
        _require(file_sha256(resolved_entry) == entry_sha256[entry_path], f"{baseline_id} package entry 摘要不匹配: {entry_path}")

    package_entry_digest = build_stable_digest(
        {"entry_paths": entry_paths, "entry_sha256": entry_sha256}
    )
    return entry_paths, entry_sha256, package_entry_digest


def _validate_archive_governance(
    *,
    spec: OfficialReferenceFamilySpec,
    source_dir: Path,
    root_path: Path,
    paths: Mapping[str, Path],
    paper_run_name: str,
    target_fpr: float,
    entry_paths: Sequence[str],
    package_input: Mapping[str, Any],
    archive_summary: Mapping[str, Any],
    archive_manifest: Mapping[str, Any],
    run_code_version: str,
) -> str:
    """核验后写入归档治理文件, 不把它们误当作动态 entry."""

    baseline_id = spec.baseline_id
    package_path_value = _normalized_declared_path(archive_summary.get("archive_path"), f"{baseline_id} archive_path")
    package_path = (root_path / Path(*PurePosixPath(package_path_value).parts)).resolve()
    _require(package_path.parent == source_dir.resolve(), f"{baseline_id} archive_path 未位于当前 run 目录")
    _require(
        package_path.name.startswith(spec.archive_name_prefix)
        and package_path.suffix.lower() == ".zip",
        f"{baseline_id} archive_path 名称不符合 family 契约",
    )
    _require(
        int(archive_summary.get("archive_entry_count", -1))
        == int(package_input.get("entry_count", -2)),
        f"{baseline_id} archive summary 的 entry 数量不一致",
    )
    archive_summary_metadata = archive_summary.get("metadata")
    _require(isinstance(archive_summary_metadata, dict), f"{baseline_id} archive summary 缺少 metadata")
    _require(
        archive_summary_metadata.get("embedded_digest_scope") == ARCHIVE_GOVERNANCE_SCOPE,
        f"{baseline_id} archive summary 治理范围不一致",
    )
    _require_timezone_timestamp(archive_summary_metadata.get("generated_at"), f"{baseline_id} archive summary generated_at")
    _require(
        Path(str(archive_summary.get("drive_archive_path", ""))).name == package_path.name,
        f"{baseline_id} Drive 归档名称与本地归档不一致",
    )

    archive_metadata = archive_manifest.get("metadata")
    _require(isinstance(archive_metadata, dict), f"{baseline_id} archive manifest 缺少 metadata")
    _require(archive_manifest.get("artifact_id") == spec.archive_manifest_artifact_id, f"{baseline_id} archive manifest 身份不一致")
    _require(archive_manifest.get("artifact_type") == "local_manifest", f"{baseline_id} archive manifest 类型不一致")
    _require(archive_metadata.get("embedded_digest_scope") == ARCHIVE_GOVERNANCE_SCOPE, f"{baseline_id} archive manifest 治理范围不一致")
    _require(archive_metadata.get("paper_run_name") == paper_run_name, f"{baseline_id} archive manifest 论文运行层级不一致")
    _require(_target_fpr_matches(archive_metadata.get("target_fpr"), target_fpr), f"{baseline_id} archive manifest target_fpr 不一致")
    _require(archive_metadata.get("baseline_id") == baseline_id, f"{baseline_id} archive manifest baseline 身份不一致")
    _require(archive_metadata.get("main_table_eligible") is False, f"{baseline_id} archive manifest 不得声明主表资格")

    package_input_path = _relative_path(paths["package_input_manifest"], root_path)
    _require(
        archive_manifest.get("input_paths") == [*entry_paths, package_input_path],
        f"{baseline_id} archive manifest input_paths 未按 producer 语义绑定",
    )
    expected_output_paths = [
        package_path_value,
        _relative_path(paths["archive_summary"], root_path),
        _relative_path(paths["archive_manifest"], root_path),
    ]
    _require(archive_manifest.get("output_paths") == expected_output_paths, f"{baseline_id} archive manifest output_paths 不一致")

    archive_digest = str(archive_summary.get("archive_digest", "") or "").strip().lower()
    drive_archive_digest = str(archive_summary.get("drive_archive_digest", "") or "").strip().lower()
    _require(bool(archive_digest) == bool(drive_archive_digest), f"{baseline_id} archive summary 的本地与 Drive 摘要状态不一致")
    if archive_digest:
        _require(SHA256_PATTERN.fullmatch(archive_digest) is not None, f"{baseline_id} archive_digest 非法")
        _require(SHA256_PATTERN.fullmatch(drive_archive_digest) is not None, f"{baseline_id} drive_archive_digest 非法")
        _require(archive_metadata.get("archive_digest") == archive_digest, f"{baseline_id} archive manifest 未绑定本地归档摘要")
        _require(archive_metadata.get("drive_archive_digest") == drive_archive_digest, f"{baseline_id} archive manifest 未绑定 Drive 归档摘要")
        if package_path.is_file():
            _require(file_sha256(package_path) == archive_digest, f"{baseline_id} 本地 ZIP 摘要不匹配")
    else:
        _require(not archive_metadata.get("archive_digest"), f"{baseline_id} 内嵌 archive manifest 不得提前声明归档摘要")
        _require(not archive_metadata.get("drive_archive_digest"), f"{baseline_id} 内嵌 archive manifest 不得提前声明 Drive 摘要")

    archive_code_version = normalize_clean_code_version(archive_manifest.get("code_version"))
    _require(archive_code_version == run_code_version, f"{baseline_id} run 与 archive code_version 不一致")
    return archive_code_version


def audit_official_reference_fidelity_source(
    *,
    root: str | Path,
    spec: OfficialReferenceFamilySpec,
    paper_run_name: str,
    target_fpr: float,
    source_dir: str | Path | None = None,
) -> OfficialReferenceSourceAudit:
    """读取并核验一个已物化官方参考 family 的完整治理证据."""

    root_path = Path(root).resolve()
    expected_source_dir = (root_path / spec.output_family / paper_run_name).resolve()
    resolved_source_dir = (
        expected_source_dir
        if source_dir is None
        else (Path(source_dir).resolve() if Path(source_dir).is_absolute() else (root_path / source_dir).resolve())
    )
    _require(resolved_source_dir == expected_source_dir, f"{spec.baseline_id} source_dir 必须使用当前 run 的正式 outputs family")
    _require(resolved_source_dir.is_dir() and not resolved_source_dir.is_symlink(), f"{spec.baseline_id} source_dir 不存在")
    paths = _source_paths(resolved_source_dir, spec)
    summary = _read_json_object(paths["summary"], f"{spec.baseline_id} summary")
    run_manifest = _read_json_object(paths["run_manifest"], f"{spec.baseline_id} run manifest")
    records = _read_jsonl_records(paths["records"], f"{spec.baseline_id} records")
    validation = _read_json_object(paths["validation_report"], f"{spec.baseline_id} validation report")
    package_input = _read_json_object(paths["package_input_manifest"], f"{spec.baseline_id} package input manifest")
    archive_summary = _read_json_object(paths["archive_summary"], f"{spec.baseline_id} archive summary")
    archive_manifest = _read_json_object(paths["archive_manifest"], f"{spec.baseline_id} archive manifest")

    run_code_version = _validate_summary_and_records(
        spec=spec,
        paper_run_name=paper_run_name,
        target_fpr=target_fpr,
        paths=paths,
        root_path=root_path,
        summary=summary,
        records=records,
        validation=validation,
        run_manifest=run_manifest,
    )
    entry_paths, _entry_sha256, package_entry_digest = _validate_package_entries(
        spec=spec,
        source_dir=resolved_source_dir,
        root_path=root_path,
        paths=paths,
        paper_run_name=paper_run_name,
        target_fpr=target_fpr,
        package_input=package_input,
    )
    archive_code_version = _validate_archive_governance(
        spec=spec,
        source_dir=resolved_source_dir,
        root_path=root_path,
        paths=paths,
        paper_run_name=paper_run_name,
        target_fpr=target_fpr,
        entry_paths=entry_paths,
        package_input=package_input,
        archive_summary=archive_summary,
        archive_manifest=archive_manifest,
        run_code_version=run_code_version,
    )

    source_paths = {role: _relative_path(path, root_path) for role, path in paths.items()}
    source_digests = {role: file_sha256(path) for role, path in paths.items()}
    record_payload: dict[str, Any] = {
        "baseline_id": spec.baseline_id,
        "paper_claim_scale": paper_run_name,
        "target_fpr": target_fpr,
        "supplemental_table_role": SUPPLEMENTAL_METHOD_FIDELITY_ROLE,
        "run_decision": "pass",
        "official_reference_ready": True,
        "reference_import_ready": True,
        "governed_reference_record_count": len(records),
        "records_nonempty_ready": True,
        "records_baseline_identity_ready": True,
        "validation_zero_rejection_ready": True,
        "run_manifest_ready": True,
        "package_input_exact_set_ready": True,
        "package_input_digests_ready": True,
        "package_governance_semantics_ready": True,
        "source_code_version_consistent_ready": run_code_version == archive_code_version,
        "code_version": run_code_version,
        "declared_package_entry_count": len(entry_paths),
        "official_reference_package_entry_digest": package_entry_digest,
        "official_reference_source_paths": source_paths,
        "official_reference_source_artifact_digests": source_digests,
        "main_table_eligible": False,
        "supports_main_table_superiority_claim": False,
        "supplemental_method_fidelity_evidence_ready": True,
        "official_reference_fidelity_evidence_ready": True,
    }
    record_digest = build_stable_digest(record_payload)
    evidence_record = {
        "official_reference_fidelity_record_id": (
            f"{spec.baseline_id}_official_reference_fidelity_{record_digest[:16]}"
        ),
        "official_reference_fidelity_record_digest": record_digest,
        **record_payload,
    }
    return OfficialReferenceSourceAudit(
        evidence_record=evidence_record,
        input_paths=tuple(paths.values()),
    )


def build_official_reference_fidelity_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """从逐方法证据记录构造精确三方法的补充忠实度摘要."""

    materialized_records = [dict(record) for record in records]
    expected_ids = list(OFFICIAL_REFERENCE_BASELINE_IDS)
    actual_ids = [str(record.get("baseline_id", "")) for record in materialized_records]
    id_counts = Counter(actual_ids)
    duplicate_ids = sorted(
        baseline_id for baseline_id, count in id_counts.items() if baseline_id and count > 1
    )
    missing_ids = sorted(set(expected_ids) - set(actual_ids))
    unexpected_ids = sorted(set(actual_ids) - set(expected_ids))
    exact_set_ready = (
        len(actual_ids) == len(expected_ids)
        and not duplicate_ids
        and not missing_ids
        and not unexpected_ids
    )
    code_versions: set[str] = set()
    for record in materialized_records:
        try:
            code_versions.add(normalize_clean_code_version(record.get("code_version")))
        except OfficialReferenceFidelityEvidenceError:
            code_versions.add("")
    common_code_version_ready = len(code_versions) == 1 and "" not in code_versions
    common_code_version = next(iter(code_versions)) if common_code_version_ready else ""
    ready_count = sum(
        record.get("official_reference_fidelity_evidence_ready") is True
        and record.get("supplemental_method_fidelity_evidence_ready") is True
        and record.get("supports_main_table_superiority_claim") is False
        and record.get("main_table_eligible") is False
        for record in materialized_records
    )
    evidence_ready = (
        exact_set_ready
        and common_code_version_ready
        and ready_count == len(expected_ids)
        and all(record.get("source_code_version_consistent_ready") is True for record in materialized_records)
    )
    source_digest_map = {
        str(record.get("baseline_id", "")): record.get(
            "official_reference_source_artifact_digests", {}
        )
        for record in materialized_records
        if str(record.get("baseline_id", ""))
    }
    return {
        "expected_official_reference_baseline_ids": expected_ids,
        "actual_official_reference_baseline_ids": actual_ids,
        "missing_official_reference_baseline_ids": missing_ids,
        "unexpected_official_reference_baseline_ids": unexpected_ids,
        "duplicate_official_reference_baseline_ids": duplicate_ids,
        "official_reference_exact_set_ready": exact_set_ready,
        "official_reference_fidelity_record_count": len(materialized_records),
        "official_reference_fidelity_ready_count": ready_count,
        "common_code_version": common_code_version,
        "common_code_version_ready": common_code_version_ready,
        "official_reference_source_artifact_digests": source_digest_map,
        "official_reference_fidelity_evidence_digest": build_stable_digest(
            materialized_records
        ),
        "main_table_eligible": False,
        "supports_main_table_superiority_claim": False,
        "supplemental_method_fidelity_evidence_ready": evidence_ready,
        "official_reference_fidelity_evidence_ready": evidence_ready,
    }


def audit_exact_official_reference_fidelity_evidence(
    *,
    root: str | Path,
    paper_run_name: str,
    target_fpr: float,
    source_dirs: Mapping[str, str | Path] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[Path, ...]]:
    """核验精确三个官方参考 family 并返回 records,summary 和输入路径."""

    configured_source_dirs = dict(source_dirs or {})
    _require(
        not configured_source_dirs
        or set(configured_source_dirs) == set(OFFICIAL_REFERENCE_BASELINE_IDS),
        "source_dirs 必须为空或精确提供三个官方参考 baseline",
    )
    audits = [
        audit_official_reference_fidelity_source(
            root=root,
            spec=spec,
            paper_run_name=paper_run_name,
            target_fpr=target_fpr,
            source_dir=configured_source_dirs.get(spec.baseline_id),
        )
        for spec in OFFICIAL_REFERENCE_FAMILY_SPECS
    ]
    records = [audit.evidence_record for audit in audits]
    summary = build_official_reference_fidelity_summary(records)
    _require(
        summary["official_reference_fidelity_evidence_ready"] is True,
        "三个官方参考方法忠实度证据未满足精确集合或代码版本门禁",
    )
    input_paths = tuple(path for audit in audits for path in audit.input_paths)
    return records, summary, input_paths
