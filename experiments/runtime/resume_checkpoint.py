"""为长耗时科学工作负载提供摘要绑定的外部续跑检查点.

该模块不感知 Colab 或 Google Drive. 调用方可以把检查点根目录指向挂载盘、
服务器持久磁盘或其他本地文件系统. 每个完成单元先复制数据文件, 最后原子写入
检查点 manifest; 恢复时只接受 manifest 完整、执行锁一致且逐文件摘要匹配的
快照. 所有检查点均明确标记为中间状态, 不能直接支持论文主张.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any, Callable, Iterable, Mapping

from experiments.runtime.repository_environment import (
    require_published_formal_execution_lock,
)
from main.core.digest import build_stable_digest


CHECKPOINT_ROOT_ENVIRONMENT_KEY = "SLM_WM_RESUME_CHECKPOINT_DIR"
CHECKPOINT_MANIFEST_SCHEMA = "semantic_resume_checkpoint_manifest"
CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1
CHECKPOINT_MANIFEST_FILE_NAME = "checkpoint_manifest.json"


def _file_sha256(path: Path) -> str:
    """流式计算检查点文件摘要."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json(value: Any) -> str:
    """生成可重复比较的 JSON 文本."""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    """在同一目录写入临时文件并通过 rename 发布完整文本."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".partial")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def _atomic_copy(source: Path, destination: Path, expected_digest: str) -> None:
    """复制一个文件并在发布前复验临时副本摘要."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".partial")
    with source.open("rb") as source_stream, temporary_path.open("wb") as target_stream:
        shutil.copyfileobj(source_stream, target_stream, length=8 * 1024 * 1024)
    if _file_sha256(temporary_path) != expected_digest:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"检查点临时副本摘要不一致: {destination}")
    os.replace(temporary_path, destination)


def _resolve_checkpoint_root(checkpoint_root: str | Path | None) -> Path | None:
    """解析显式参数或环境变量提供的持久化根目录."""

    raw_value = (
        str(checkpoint_root)
        if checkpoint_root is not None
        else os.environ.get(CHECKPOINT_ROOT_ENVIRONMENT_KEY, "")
    )
    if not raw_value.strip():
        return None
    resolved = Path(raw_value).expanduser().resolve()
    if resolved.exists() and (not resolved.is_dir() or resolved.is_symlink()):
        raise ValueError("续跑检查点根路径必须是普通目录")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _require_path_component(value: str, field_name: str) -> str:
    """要求外部目录身份只能使用一个普通路径分量."""

    normalized = str(value).strip()
    candidate = PurePosixPath(normalized.replace("\\", "/"))
    if (
        not normalized
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.parts[0] in {"", ".", ".."}
    ):
        raise ValueError(f"{field_name} 必须是单一语义名称")
    return normalized


def _relative_repository_file(path: Path, repository_root: Path) -> str:
    """要求检查点来源是仓库 outputs 下的普通文件."""

    resolved = path.resolve()
    outputs_root = (repository_root / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("续跑检查点来源必须位于 outputs 目录") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"续跑检查点来源不是普通文件: {resolved}")
    return resolved.relative_to(repository_root).as_posix()


def _entry_records(
    repository_root: Path,
    paths: Iterable[str | Path],
) -> tuple[dict[str, Any], ...]:
    """冻结检查点来源文件的仓库相对路径、大小和摘要."""

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in paths:
        candidate = Path(raw_path)
        path = candidate.resolve() if candidate.is_absolute() else (repository_root / candidate).resolve()
        relative_path = _relative_repository_file(path, repository_root)
        if relative_path in seen:
            raise ValueError(f"续跑检查点包含重复路径: {relative_path}")
        seen.add(relative_path)
        records.append(
            {
                "path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not records:
        raise ValueError("续跑检查点不得为空")
    ordered_records = sorted(records, key=lambda record: str(record["path"]))
    for index, record in enumerate(ordered_records):
        # 扁平 payload 名称避免 Windows 长路径, 原仓库路径仍由 manifest 完整绑定.
        record["payload_name_intermediate"] = f"{index:08d}.bin"
    return tuple(ordered_records)


def persist_checkpoint_files(
    *,
    repository_root: str | Path,
    artifact_role: str,
    paper_run_name: str,
    checkpoint_kind: str,
    checkpoint_id: str,
    paths: Iterable[str | Path],
    checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    """把一组完整文件发布为内容摘要绑定的中间检查点."""

    root = Path(repository_root).resolve()
    persistent_root = _resolve_checkpoint_root(checkpoint_root)
    if persistent_root is None:
        return {
            "checkpoint_persistence_configured": False,
            "checkpoint_persisted": False,
            "supports_paper_claim": False,
        }
    normalized_role = _require_path_component(artifact_role, "artifact_role")
    normalized_run = _require_path_component(paper_run_name, "paper_run_name")
    normalized_kind = _require_path_component(checkpoint_kind, "checkpoint_kind")
    normalized_id = _require_path_component(checkpoint_id, "checkpoint_id")

    records = _entry_records(root, paths)
    execution_lock = require_published_formal_execution_lock(root)
    content_digest = build_stable_digest(list(records))
    snapshot_name = f"{normalized_id}_{content_digest[:16]}"
    snapshot_dir = (
        persistent_root
        / normalized_run
        / normalized_role
        / normalized_kind
        / snapshot_name
    )
    payload_root = snapshot_dir / "payload"
    for record in records:
        source = root / str(record["path"])
        destination = payload_root / str(record["payload_name_intermediate"])
        _atomic_copy(source, destination, str(record["sha256"]))

    payload = {
        "report_schema": CHECKPOINT_MANIFEST_SCHEMA,
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "artifact_role": normalized_role,
        "paper_run_name": normalized_run,
        "checkpoint_kind": normalized_kind,
        "checkpoint_id": normalized_id,
        "checkpoint_content_digest": content_digest,
        "entry_count": len(records),
        "entry_records": list(records),
        "formal_execution_lock": execution_lock,
        "checkpoint_decision": "complete",
        "evidence_eligibility": "intermediate_state_only",
        "supports_paper_claim": False,
    }
    digest_payload = dict(payload)
    payload["checkpoint_manifest_digest"] = build_stable_digest(digest_payload)
    manifest_path = snapshot_dir / CHECKPOINT_MANIFEST_FILE_NAME
    _atomic_write_text(manifest_path, _stable_json(payload))
    return {
        "checkpoint_persistence_configured": True,
        "checkpoint_persisted": True,
        "checkpoint_manifest_path": manifest_path.as_posix(),
        "checkpoint_manifest_digest": payload["checkpoint_manifest_digest"],
        "checkpoint_content_digest": content_digest,
        "entry_count": len(records),
        "supports_paper_claim": False,
    }


def persist_completed_unit_from_manifest(
    manifest_path: str | Path,
    *,
    repository_root: str | Path,
    artifact_role: str,
    paper_run_name: str,
    checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    """按运行 manifest 的精确 output_paths 保存一个完成单元."""

    root = Path(repository_root).resolve()
    resolved_manifest = Path(manifest_path).resolve()
    manifest = json.loads(resolved_manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, Mapping):
        raise TypeError("完成单元 manifest 必须是 JSON object")
    output_paths = manifest.get("output_paths")
    if not isinstance(output_paths, list) or not output_paths:
        raise ValueError("完成单元 manifest 缺少 output_paths")
    relative_manifest = _relative_repository_file(resolved_manifest, root)
    if relative_manifest not in output_paths:
        raise ValueError("完成单元 manifest 必须把自身列入 output_paths")
    return persist_checkpoint_files(
        repository_root=root,
        artifact_role=artifact_role,
        paper_run_name=paper_run_name,
        checkpoint_kind="completed_scientific_units",
        checkpoint_id=resolved_manifest.parent.name,
        paths=output_paths,
        checkpoint_root=checkpoint_root,
    )


def persist_progress_checkpoint(
    progress_path: str | Path,
    *,
    repository_root: str | Path,
    artifact_role: str,
    paper_run_name: str,
    checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    """保存不具备论文证据资格的运行进度文件."""

    persistent_root = _resolve_checkpoint_root(checkpoint_root)
    publication = persist_checkpoint_files(
        repository_root=repository_root,
        artifact_role=artifact_role,
        paper_run_name=paper_run_name,
        checkpoint_kind="progress_records",
        checkpoint_id=Path(progress_path).stem,
        paths=(progress_path,),
        checkpoint_root=persistent_root,
    )
    if persistent_root is None:
        return publication
    published_manifest = Path(publication["checkpoint_manifest_path"]).resolve()
    progress_root = (
        persistent_root
        / _require_path_component(paper_run_name, "paper_run_name")
        / _require_path_component(artifact_role, "artifact_role")
        / "progress_records"
    )
    for manifest_path in progress_root.rglob(CHECKPOINT_MANIFEST_FILE_NAME):
        if manifest_path.resolve() != published_manifest:
            manifest_path.unlink(missing_ok=True)
    return publication


def _validated_checkpoint_manifest(
    manifest_path: Path,
    *,
    expected_role: str,
    expected_run: str,
    execution_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """复验一个外部检查点 manifest 的身份、锁和自摘要."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError("续跑检查点 manifest 必须是 JSON object")
    declared_digest = str(payload.get("checkpoint_manifest_digest", ""))
    digest_payload = dict(payload)
    digest_payload.pop("checkpoint_manifest_digest", None)
    ready = all(
        (
            payload.get("report_schema") == CHECKPOINT_MANIFEST_SCHEMA,
            payload.get("schema_version") == CHECKPOINT_MANIFEST_SCHEMA_VERSION,
            payload.get("artifact_role") == expected_role,
            payload.get("paper_run_name") == expected_run,
            payload.get("formal_execution_lock") == dict(execution_lock),
            payload.get("checkpoint_decision") == "complete",
            payload.get("evidence_eligibility") == "intermediate_state_only",
            payload.get("supports_paper_claim") is False,
            declared_digest == build_stable_digest(digest_payload),
        )
    )
    records = payload.get("entry_records")
    if (
        not ready
        or not isinstance(records, list)
        or not records
        or payload.get("entry_count") != len(records)
        or payload.get("checkpoint_content_digest") != build_stable_digest(records)
    ):
        raise RuntimeError(f"续跑检查点 manifest 未通过治理校验: {manifest_path}")
    checkpoint_kind = payload.get("checkpoint_kind")
    entry_paths = [
        str(record.get("path", ""))
        for record in records
        if isinstance(record, dict)
    ]
    local_manifest_count = sum(
        Path(path).name.endswith("manifest.local.json") for path in entry_paths
    )
    kind_ready = (
        checkpoint_kind == "progress_records"
        and len(entry_paths) == 1
        and Path(entry_paths[0]).name.endswith("progress.json")
        and local_manifest_count == 0
    ) or (
        checkpoint_kind in {
            "completed_scientific_units",
            "completed_workflow",
        }
        and local_manifest_count == 1
    ) or (
        checkpoint_kind == "feature_batches"
        and sum(Path(path).name == "feature_checkpoint_context.json" for path in entry_paths)
        == 1
        and sum(
            Path(path).name.startswith("feature_batch_")
            and Path(path).suffix == ".jsonl"
            for path in entry_paths
        )
        == 1
        and local_manifest_count == 0
    )
    if not kind_ready:
        raise RuntimeError(f"续跑检查点 kind 与成员职责不一致: {manifest_path}")
    return payload


def restore_role_checkpoints(
    *,
    repository_root: str | Path,
    artifact_role: str,
    paper_run_name: str,
    allowed_output_prefix: str,
    checkpoint_root: str | Path | None = None,
    before_restore: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """先完整验证, 再恢复某个角色的全部完整快照.

    全部 manifest、路径和 payload 摘要通过后才调用 ``before_restore``. 外层可在
    该回调中清理当前角色的旧文件, 从而既不混入 stale 文件, 也不会因损坏的
    外部 checkpoint 提前删除本地有效结果.
    """

    root = Path(repository_root).resolve()
    persistent_root = _resolve_checkpoint_root(checkpoint_root)
    if persistent_root is None:
        return {
            "checkpoint_persistence_configured": False,
            "restored_manifest_count": 0,
            "restored_file_count": 0,
            "supports_paper_claim": False,
        }
    role_root = persistent_root / paper_run_name / artifact_role
    if not role_root.is_dir():
        return {
            "checkpoint_persistence_configured": True,
            "restored_manifest_count": 0,
            "restored_file_count": 0,
            "supports_paper_claim": False,
        }
    execution_lock = require_published_formal_execution_lock(root)
    manifest_paths = sorted(role_root.rglob(CHECKPOINT_MANIFEST_FILE_NAME))
    if not manifest_paths:
        return {
            "checkpoint_persistence_configured": True,
            "restored_manifest_count": 0,
            "restored_file_count": 0,
            "supports_paper_claim": False,
        }
    restored_records: dict[str, dict[str, Any]] = {}
    snapshot_by_path: dict[str, Path] = {}
    validated_manifest_count = 0
    normalized_prefix = str(allowed_output_prefix).rstrip("/") + "/"
    for manifest_path in manifest_paths:
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise RuntimeError(f"续跑检查点 manifest 不是普通文件: {manifest_path}")
        payload = _validated_checkpoint_manifest(
            manifest_path,
            expected_role=artifact_role,
            expected_run=paper_run_name,
            execution_lock=execution_lock,
        )
        payload_root = manifest_path.parent / "payload"
        for raw_record in payload["entry_records"]:
            if not isinstance(raw_record, dict):
                raise TypeError("续跑检查点 entry record 必须是 JSON object")
            relative_path = str(raw_record.get("path", ""))
            payload_name = str(raw_record.get("payload_name_intermediate", ""))
            if (
                not payload_name.endswith(".bin")
                or Path(payload_name).name != payload_name
            ):
                raise RuntimeError("续跑检查点 payload 名称无效")
            pure_path = PurePosixPath(relative_path)
            if (
                not relative_path.startswith(normalized_prefix)
                or "\\" in relative_path
                or pure_path.is_absolute()
                or any(part in {"", ".", ".."} for part in pure_path.parts)
                or pure_path.as_posix() != relative_path
            ):
                raise RuntimeError(f"续跑检查点包含角色目录外路径: {relative_path}")
            source = (payload_root / payload_name).resolve()
            try:
                source.relative_to(payload_root.resolve())
            except ValueError as exc:
                raise RuntimeError("续跑检查点 payload 路径越界") from exc
            expected_size = int(raw_record.get("size_bytes", -1))
            expected_digest = str(raw_record.get("sha256", ""))
            if (
                not source.is_file()
                or source.is_symlink()
                or source.stat().st_size != expected_size
                or _file_sha256(source) != expected_digest
            ):
                raise RuntimeError(f"续跑检查点 payload 摘要不一致: {relative_path}")
            existing = restored_records.get(relative_path)
            if existing is not None and existing != raw_record:
                raise RuntimeError(f"续跑检查点对同一路径声明了冲突字节: {relative_path}")
            restored_records[relative_path] = dict(raw_record)
            snapshot_by_path[relative_path] = payload_root
        validated_manifest_count += 1

    ordered_paths = sorted(
        restored_records,
        key=lambda path: (Path(path).name.endswith("manifest.local.json"), path),
    )
    if before_restore is not None:
        before_restore()
    for relative_path in ordered_paths:
        record = restored_records[relative_path]
        source = snapshot_by_path[relative_path] / str(
            record["payload_name_intermediate"]
        )
        destination = (root / relative_path).resolve()
        try:
            destination.relative_to((root / "outputs").resolve())
        except ValueError as exc:
            raise RuntimeError("续跑检查点恢复目标逃逸 outputs 目录") from exc
        _atomic_copy(source, destination, str(record["sha256"]))
    return {
        "checkpoint_persistence_configured": True,
        "restored_manifest_count": validated_manifest_count,
        "restored_file_count": len(ordered_paths),
        "restored_entry_digest": build_stable_digest(ordered_paths),
        "supports_paper_claim": False,
    }


def clear_progress_checkpoints(
    *,
    artifact_role: str,
    paper_run_name: str,
    checkpoint_root: str | Path | None = None,
) -> None:
    """在正式产物闭合后移除可误导入口判断的进度快照."""

    persistent_root = _resolve_checkpoint_root(checkpoint_root)
    if persistent_root is None:
        return
    progress_root = (
        persistent_root
        / paper_run_name
        / artifact_role
        / "progress_records"
    )
    if not progress_root.is_dir():
        return
    for manifest_path in progress_root.rglob(CHECKPOINT_MANIFEST_FILE_NAME):
        manifest_path.unlink(missing_ok=True)
