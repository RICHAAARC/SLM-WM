"""构造由逐成员摘要约束的正式结果包输入清单.

该模块只负责冻结 ZIP 写入前的路径和文件字节, 不解释具体科学指标. 主方法运行,
质量评估和机制消融可以共享同一实现, 避免三个打包器维护不同的成员闭合规则.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from zipfile import ZipFile

from experiments.protocol.formal_randomization import (
    formal_randomization_protocol_record,
    validate_formal_randomization_repeat_records,
)


PACKAGE_INPUT_MANIFEST_SCHEMA = "exact_package_input_manifest"
PACKAGE_INPUT_MANIFEST_SCHEMA_VERSION = 2

SCIENTIFIC_BINDING_EVIDENCE_FIELDS = (
    "scientific_execution_report_path",
    "dependency_environment_report_path",
    "scientific_command_dispatch_report_path",
)


def _file_sha256(path: Path) -> str:
    """流式计算一个正式包成员的 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repository_file(
    raw_path: object,
    *,
    repository_root: Path,
    source_dir: Path,
    field_name: str,
) -> Path:
    """把治理清单中的仓库相对路径解析为同一产物目录内的普通文件."""

    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{field_name} 必须是非空仓库相对路径")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise ValueError(f"{field_name} 不得使用绝对路径")
    resolved = (repository_root / candidate).resolve()
    try:
        resolved.relative_to(source_dir)
    except ValueError as exc:
        raise ValueError(f"{field_name} 必须位于当前正式产物目录内") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"{field_name} 指向的正式产物不是普通文件: {resolved}")
    return resolved


def collect_exact_package_entries(
    *,
    repository_root: str | Path,
    source_dir: str | Path,
    artifact_manifest: Mapping[str, Any],
    scientific_binding_path: str | Path,
) -> tuple[Path, ...]:
    """从 artifact manifest 和科学执行绑定解析精确打包输入.

    该函数不扫描目录. artifact manifest 负责声明科学产物, scientific binding
    负责声明执行报告、依赖报告和调度报告. 因此同目录遗留日志或同前缀文件不会
    被自动吸收到正式结果包中, 三类主方法打包器可以复用相同闭合规则.
    """

    root = Path(repository_root).resolve()
    resolved_source_dir = Path(source_dir).resolve()
    resolved_binding_path = Path(scientific_binding_path).resolve()
    try:
        resolved_binding_path.relative_to(resolved_source_dir)
    except ValueError as exc:
        raise ValueError("scientific binding 必须位于当前正式产物目录内") from exc
    if not resolved_binding_path.is_file() or resolved_binding_path.is_symlink():
        raise FileNotFoundError("scientific binding 必须是当前产物目录内的普通文件")

    output_paths = artifact_manifest.get("output_paths")
    if not isinstance(output_paths, list) or not output_paths:
        raise ValueError("artifact manifest 必须声明非空 output_paths")
    entries = [
        _resolve_repository_file(
            raw_path,
            repository_root=root,
            source_dir=resolved_source_dir,
            field_name="artifact_manifest.output_paths",
        )
        for raw_path in output_paths
    ]

    binding = json.loads(resolved_binding_path.read_text(encoding="utf-8-sig"))
    if not isinstance(binding, dict):
        raise TypeError("scientific binding 必须是 JSON object")
    entries.append(resolved_binding_path)
    for field_name in SCIENTIFIC_BINDING_EVIDENCE_FIELDS:
        entries.append(
            _resolve_repository_file(
                binding.get(field_name),
                repository_root=root,
                source_dir=resolved_source_dir,
                field_name=f"scientific_binding.{field_name}",
            )
        )

    unique_entries = tuple(sorted(set(entries), key=lambda path: path.as_posix()))
    if len(unique_entries) != len(entries):
        raise ValueError("artifact manifest 与 scientific binding 声明了重复打包成员")
    return unique_entries


def write_exact_package_input_manifest(
    manifest_path: str | Path,
    *,
    repository_root: str | Path,
    package_family: str,
    paper_run_name: str,
    target_fpr: float,
    randomization_repeat_identity: Mapping[str, Any],
    formal_randomization_protocol_digest: str,
    entries: Iterable[str | Path],
    formal_execution_run_lock: Mapping[str, Any],
    formal_execution_package_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """冻结待归档文件的精确路径集合和逐成员摘要.

    返回清单不把自身列入 ``entry_paths``. ZIP 打包器应在写出清单后把清单作为
    独立治理成员追加到归档, closure selector 再核验声明集合与实际集合完全相等.
    """

    root = Path(repository_root).resolve()
    resolved_manifest_path = Path(manifest_path).resolve()
    resolved_entries = tuple(sorted((Path(path).resolve() for path in entries)))
    if not resolved_entries:
        raise RuntimeError("正式结果包输入成员集合不得为空")
    if len(resolved_entries) != len(set(resolved_entries)):
        raise RuntimeError("正式结果包输入成员集合不得包含重复路径")

    entry_paths: list[str] = []
    entry_sha256: dict[str, str] = {}
    for path in resolved_entries:
        try:
            relative_path = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("正式结果包成员必须位于当前仓库根目录内") from exc
        if path == resolved_manifest_path:
            raise ValueError("package input manifest 不得把自身声明为输入成员")
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"正式结果包成员必须是普通文件: {path}")
        entry_paths.append(relative_path)
        entry_sha256[relative_path] = _file_sha256(path)

    repeat_fields = (
        "randomization_repeat_id",
        "generation_seed_index",
        "generation_seed_offset",
        "watermark_key_index",
    )
    repeat_identity = {
        field_name: randomization_repeat_identity.get(field_name)
        for field_name in repeat_fields
    }
    if (
        not isinstance(repeat_identity["randomization_repeat_id"], str)
        or not repeat_identity["randomization_repeat_id"]
        or any(
            isinstance(repeat_identity[field_name], bool)
            or not isinstance(repeat_identity[field_name], int)
            or repeat_identity[field_name] < 0
            for field_name in repeat_fields[1:]
        )
    ):
        raise ValueError("正式结果包缺少有效随机化 repeat 身份")
    if (
        not isinstance(formal_randomization_protocol_digest, str)
        or len(formal_randomization_protocol_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in formal_randomization_protocol_digest
        )
    ):
        raise ValueError("正式结果包随机化协议摘要必须是小写 SHA-256")
    try:
        normalized_repeat_identity = (
            validate_formal_randomization_repeat_records(
                [repeat_identity],
                require_exact_registry=False,
            )[0]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("正式结果包 repeat 身份未匹配权威注册表") from exc
    if (
        normalized_repeat_identity != repeat_identity
        or formal_randomization_protocol_digest
        != formal_randomization_protocol_record()[
            "formal_randomization_protocol_digest"
        ]
    ):
        raise ValueError("正式结果包随机化身份或协议摘要未匹配权威注册表")

    payload = {
        "report_schema": PACKAGE_INPUT_MANIFEST_SCHEMA,
        "schema_version": PACKAGE_INPUT_MANIFEST_SCHEMA_VERSION,
        "package_family": str(package_family),
        "paper_run_name": str(paper_run_name),
        "target_fpr": float(target_fpr),
        **repeat_identity,
        "formal_randomization_protocol_digest": (
            formal_randomization_protocol_digest
        ),
        "randomization_repeat_identity": {
            **repeat_identity,
            "formal_randomization_protocol_digest": (
                formal_randomization_protocol_digest
            ),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(entry_paths),
        "entry_paths": entry_paths,
        "entry_sha256": entry_sha256,
        "formal_execution_run_lock": dict(formal_execution_run_lock),
        "formal_execution_package_lock": dict(formal_execution_package_lock),
        "decision": "pass",
        "supports_paper_claim": False,
    }
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def validate_exact_package_archive(
    archive_path: str | Path,
    *,
    repository_root: str | Path,
    package_input_manifest_path: str | Path,
) -> None:
    """在 ZIP 写出后复验精确成员集合和逐成员 SHA-256.

    该写后校验用于发现 manifest 冻结后、ZIP 读取文件期间发生的字节漂移.
    三个主方法 producer 应在返回归档路径前调用它, closure selector 仍会在消费
    环节独立重复验证, 两层校验分别覆盖生产时和闭合时的信任边界.
    """

    root = Path(repository_root).resolve()
    resolved_manifest_path = Path(package_input_manifest_path).resolve()
    manifest = json.loads(
        resolved_manifest_path.read_text(encoding="utf-8-sig")
    )
    if not isinstance(manifest, dict):
        raise RuntimeError("package input manifest 必须是 JSON object")
    entry_paths = manifest.get("entry_paths")
    entry_sha256 = manifest.get("entry_sha256")
    if not isinstance(entry_paths, list) or not isinstance(entry_sha256, dict):
        raise RuntimeError("package input manifest 缺少精确成员或摘要映射")
    if (
        not entry_paths
        or len(entry_paths) != len(set(entry_paths))
        or manifest.get("entry_count") != len(entry_paths)
        or set(entry_sha256) != set(entry_paths)
    ):
        raise RuntimeError("package input manifest 的精确成员集合不自洽")
    manifest_member = resolved_manifest_path.relative_to(root).as_posix()
    expected_members = set(entry_paths) | {manifest_member}

    with ZipFile(Path(archive_path)) as archive:
        infos = archive.infolist()
        member_names = [info.filename for info in infos]
        if (
            len(member_names) != len(set(member_names))
            or set(member_names) != expected_members
            or any(info.is_dir() for info in infos)
        ):
            raise RuntimeError("正式结果包写后成员集合与 package input 不一致")
        if archive.read(manifest_member) != resolved_manifest_path.read_bytes():
            raise RuntimeError("正式结果包内 package input 字节与本地清单不一致")
        for member_name in entry_paths:
            declared_digest = str(entry_sha256[member_name]).lower()
            actual_digest = hashlib.sha256(archive.read(member_name)).hexdigest()
            if actual_digest != declared_digest:
                raise RuntimeError(
                    f"正式结果包写后成员摘要与 package input 不一致: {member_name}"
                )
