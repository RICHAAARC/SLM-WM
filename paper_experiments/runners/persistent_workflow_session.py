"""为外部 GPU workflow 提供统一的持久化、恢复与重入边界.

该模块位于完整论文实验层, 因而 Notebook 和独立服务器脚本都可以复用它.
它只保存 runner 已经写入 ``outputs/`` 的普通文件, 不实现 baseline 算法, 也不
把 checkpoint 升级为论文证据. 正式证据仍必须经过各 workflow 的 archive
打包器和完整闭合门禁.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
from typing import Any, Callable, Mapping, Sequence
import uuid

from experiments.protocol.paper_run_config import PaperRunConfig, build_paper_run_config
from experiments.runtime.dependency_profiles import require_dependency_profile_ready
from experiments.runtime.repository_environment import (
    require_published_formal_execution_lock,
    validate_formal_execution_lock_record,
)
from experiments.runtime.resume_checkpoint import (
    persist_checkpoint_files,
    restore_role_checkpoints,
)
from experiments.runtime.scientific_execution_binding import (
    validate_scientific_execution_binding,
)
from main.core.digest import build_stable_digest


CHECKPOINT_SCHEMA = "external_workflow_persistence_checkpoint"
CHECKPOINT_POINTER_SCHEMA = "external_workflow_persistence_pointer"
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_DIRECTORY_NAME = "workflow_checkpoints"
CHECKPOINT_POINTER_FILE_NAME = "current_checkpoint.json"
CHECKPOINT_MANIFEST_FILE_NAME = "checkpoint_manifest.json"
CHECKPOINT_PAYLOAD_DIRECTORY_NAME = "payload"
DEFAULT_CHECKPOINT_INTERVAL_SECONDS = 60.0
CHECKPOINT_INTERVAL_ENVIRONMENT_KEY = (
    "SLM_WM_WORKFLOW_CHECKPOINT_INTERVAL_SECONDS"
)


class WorkflowCheckpointError(RuntimeError):
    """表示持久化 checkpoint 的身份、内容或边界无效."""


@dataclass(frozen=True)
class PersistentWorkflowRoute:
    """描述一条外部 workflow 的唯一输出范围和完成门禁."""

    route_name: str
    checkpoint_role_name: str
    workflow_name: str
    scientific_profile_id: str
    output_family: str
    summary_relative_path: str
    manifest_relative_path: str
    ready_field: str
    baseline_id: str
    summary_baseline_field: str
    summary_paper_run_field: str
    required_relative_paths: tuple[str, ...]
    scientific_artifact_role: str | None = None

    def output_root(self, repository_root: Path, paper_run_name: str) -> Path:
        """返回当前论文运行层级的正式输出根目录."""

        return (
            repository_root
            / "outputs"
            / self.output_family
            / paper_run_name
        ).resolve()

    def summary_path(self, repository_root: Path, paper_run_name: str) -> Path:
        """返回当前路由的正式运行摘要路径."""

        return self.output_root(repository_root, paper_run_name) / self.summary_relative_path

    def manifest_path(self, repository_root: Path, paper_run_name: str) -> Path:
        """返回当前路由的正式运行 manifest 路径."""

        return self.output_root(repository_root, paper_run_name) / self.manifest_relative_path

    def allows_relative_path(self, relative_path: Path) -> bool:
        """限制 checkpoint 只能保存当前路由拥有的运行文件.

        method-faithful 的三个 baseline 共用一个输出 family, 因而必须额外限制
        ``run_records`` 和 ``split_observations`` 的 baseline 身份. 其他4条路由
        各自拥有独立输出 family, 可以保存其论文运行层级下的全部非归档文件.
        """

        if relative_path.is_absolute() or ".." in relative_path.parts:
            return False
        if not relative_path.parts or _is_package_artifact(relative_path):
            return False
        if self.workflow_name != "external_baseline_method_faithful":
            return True
        if (
            len(relative_path.parts) >= 3
            and relative_path.parts[0] == "run_records"
            and relative_path.parts[1] == self.baseline_id
        ):
            return "package_records" not in relative_path.parts
        split_names = {
            f"{self.baseline_id}_baseline_observations.json",
            f"{self.baseline_id}_baseline_command_results.json",
            f"{self.baseline_id}_baseline_transfer_manifest.json",
        }
        return (
            len(relative_path.parts) == 2
            and relative_path.parts[0] == "split_observations"
            and relative_path.name in split_names
        )


def _method_faithful_route(baseline_id: str) -> PersistentWorkflowRoute:
    """构造一个单 baseline 的 method-faithful 持久化路由."""

    return PersistentWorkflowRoute(
        route_name=f"external_baseline_{baseline_id}",
        checkpoint_role_name={
            "tree_ring": "tree_method",
            "gaussian_shading": "gaussian_method",
            "shallow_diffuse": "shallow_method",
        }[baseline_id],
        workflow_name="external_baseline_method_faithful",
        scientific_profile_id="sd35_method_runtime_gpu",
        output_family="external_baseline_method_faithful",
        summary_relative_path=f"run_records/{baseline_id}/{baseline_id}_summary.json",
        manifest_relative_path=(
            f"run_records/{baseline_id}/{baseline_id}_manifest.local.json"
        ),
        ready_field="external_baseline_method_faithful_ready",
        baseline_id=baseline_id,
        summary_baseline_field="primary_baseline_id",
        summary_paper_run_field="paper_run_name",
        required_relative_paths=(
            f"run_records/{baseline_id}/scientific_execution_binding.json",
            f"split_observations/{baseline_id}_baseline_observations.json",
            f"split_observations/{baseline_id}_baseline_command_results.json",
            f"split_observations/{baseline_id}_baseline_transfer_manifest.json",
        ),
        scientific_artifact_role="external_baseline_method_faithful",
    )


PERSISTENT_WORKFLOW_ROUTES = {
    route.route_name: route
    for route in (
        _method_faithful_route("tree_ring"),
        _method_faithful_route("gaussian_shading"),
        _method_faithful_route("shallow_diffuse"),
        PersistentWorkflowRoute(
            route_name="official_reference_t2smark",
            checkpoint_role_name="t2smark_official",
            workflow_name="official_reference_t2smark",
            scientific_profile_id="t2smark_sd35_gpu",
            output_family="t2smark_formal_reproduction",
            summary_relative_path="t2smark_formal_reproduction_summary.json",
            manifest_relative_path="t2smark_formal_reproduction_manifest.local.json",
            ready_field="t2smark_formal_reproduction_ready",
            baseline_id="t2smark",
            summary_baseline_field="baseline_id",
            summary_paper_run_field="paper_claim_scale",
            required_relative_paths=(
                "scientific_execution_binding.json",
                "t2smark_formal_import_candidate_records.jsonl",
                "t2smark_formal_import_validation_report.json",
            ),
            scientific_artifact_role="t2smark_formal_reproduction",
        ),
        PersistentWorkflowRoute(
            route_name="official_reference_tree_ring",
            checkpoint_role_name="tree_official",
            workflow_name="official_reference_tree_ring",
            scientific_profile_id="tree_ring_official_py39_cu117",
            output_family="tree_ring_official_reference",
            summary_relative_path="tree_ring_official_reference_summary.json",
            manifest_relative_path="manifest.local.json",
            ready_field="tree_ring_official_reference_ready",
            baseline_id="tree_ring",
            summary_baseline_field="baseline_id",
            summary_paper_run_field="paper_claim_scale",
            required_relative_paths=(
                "tree_ring_official_command_result.json",
                "tree_ring_dependency_environment_prepare_result.json",
                "tree_ring_official_reference_environment_report.json",
                "tree_ring_official_reference_records.jsonl",
                "tree_ring_official_reference_validation_report.json",
            ),
        ),
        PersistentWorkflowRoute(
            route_name="official_reference_gaussian_shading",
            checkpoint_role_name="gaussian_official",
            workflow_name="official_reference_gaussian_shading",
            scientific_profile_id="gaussian_shading_official_py38_cu117",
            output_family="gaussian_shading_official_reference",
            summary_relative_path="gaussian_shading_official_reference_summary.json",
            manifest_relative_path="manifest.local.json",
            ready_field="gaussian_shading_official_reference_ready",
            baseline_id="gaussian_shading",
            summary_baseline_field="baseline_id",
            summary_paper_run_field="paper_claim_scale",
            required_relative_paths=(
                "gaussian_shading_official_command_result.json",
                "gaussian_shading_dependency_environment_prepare_result.json",
                "gaussian_shading_official_reference_environment_report.json",
                "gaussian_shading_official_reference_records.jsonl",
                "gaussian_shading_official_reference_validation_report.json",
            ),
        ),
        PersistentWorkflowRoute(
            route_name="official_reference_shallow_diffuse",
            checkpoint_role_name="shallow_official",
            workflow_name="official_reference_shallow_diffuse",
            scientific_profile_id="shallow_diffuse_official_py39_cu117",
            output_family="shallow_diffuse_official_reference",
            summary_relative_path="shallow_diffuse_official_reference_summary.json",
            manifest_relative_path="manifest.local.json",
            ready_field="shallow_diffuse_official_reference_ready",
            baseline_id="shallow_diffuse",
            summary_baseline_field="baseline_id",
            summary_paper_run_field="paper_claim_scale",
            required_relative_paths=(
                "shallow_diffuse_official_command_result.json",
                "shallow_diffuse_dependency_environment_prepare_result.json",
                "shallow_diffuse_official_reference_environment_report.json",
                "shallow_diffuse_official_reference_records.jsonl",
                "shallow_diffuse_official_reference_validation_report.json",
            ),
        ),
    )
}


def resolve_persistent_workflow_route(
    workflow_name: str,
    *,
    baseline_id: str | None = None,
) -> PersistentWorkflowRoute:
    """把 Notebook 共享名称或服务器公开名称解析为7条唯一路由."""

    if workflow_name == "external_baseline_method_faithful":
        resolved_baseline_id = (
            baseline_id or os.environ.get("SLM_WM_PRIMARY_BASELINE_ID", "")
        ).strip()
        route_name = f"external_baseline_{resolved_baseline_id}"
    else:
        route_name = workflow_name
    try:
        return PERSISTENT_WORKFLOW_ROUTES[route_name]
    except KeyError as exc:
        raise ValueError(f"不支持持久化的外部 workflow: {route_name}") from exc


def _is_package_artifact(relative_path: Path) -> bool:
    """排除正式归档及其打包记录, 保持 checkpoint 与证据包分离."""

    name = relative_path.name
    return (
        relative_path.suffix.lower() == ".zip"
        or "package_records" in relative_path.parts
        or name.endswith("_archive_summary.json")
        or name.endswith("_archive_manifest.local.json")
        or name.endswith("_package_input_manifest.json")
    )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    """读取必需 JSON object, 避免各边界重复实现解析规则."""

    if not path.is_file():
        raise WorkflowCheckpointError(f"{label} 不存在: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowCheckpointError(f"{label} 不是有效 JSON") from exc
    if not isinstance(payload, dict):
        raise WorkflowCheckpointError(f"{label} 必须是 JSON object")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """通过同目录临时文件原子替换 JSON 指针."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _file_sha256(path: Path) -> str:
    """计算普通文件的 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """计算排除自指摘要字段后的稳定 checkpoint 摘要."""

    payload = {
        key: value
        for key, value in manifest.items()
        if key
        not in {
            "checkpoint_digest_intermediate",
            "checkpoint_generation_intermediate",
        }
    }
    return build_stable_digest(payload)


def _configuration_environment() -> dict[str, str]:
    """收集会改变科学结果的 SLM 配置, 排除路径和持久化控制字段.

    环境值只进入稳定摘要, checkpoint 不保存原值. 这样既能阻止参数漂移后的
    错误重入, 又不会把访问令牌或本地路径复制到 Drive.
    """

    excluded_names = {
        "SLM_WM_FORMAL_EXECUTION_COMMIT",
        "SLM_WM_FORMAL_EXECUTION_LOCK_DIGEST",
        CHECKPOINT_INTERVAL_ENVIRONMENT_KEY,
        "SLM_WM_WORKFLOW_PERSISTENT_OUTPUT_DIR",
        "SLM_WM_RESUME_CHECKPOINT_DIR",
        "SLM_WM_DRIVE_RESULT_ROOT",
        "SLM_WM_WORKSPACE_DIR",
    }
    result: dict[str, str] = {}
    for name, value in sorted(os.environ.items()):
        if not name.startswith("SLM_WM_") or name in excluded_names:
            continue
        if name.endswith("_DRIVE_OUTPUT_DIR") or name.endswith("_DRIVE_DIR"):
            continue
        if any(token in name for token in ("TOKEN", "SECRET", "PASSWORD")):
            continue
        result[name] = value
    return result


def _paper_run_identity(paper_run: PaperRunConfig) -> dict[str, Any]:
    """移除纯存储路径后返回会影响科学协议的论文运行身份."""

    identity = paper_run.to_dict()
    identity.pop("drive_result_root", None)
    return identity


class PersistentWorkflowSession:
    """管理单条 workflow 的不可变 checkpoint generation."""

    def __init__(
        self,
        *,
        repository_root: str | Path,
        persistent_output_dir: str | Path,
        route: PersistentWorkflowRoute,
        paper_run: PaperRunConfig,
        formal_execution_lock: Mapping[str, Any],
        interval_seconds: float = DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.persistent_output_dir = Path(persistent_output_dir).expanduser().resolve()
        self.route = route
        self.paper_run = paper_run
        self.formal_execution_lock = validate_formal_execution_lock_record(
            formal_execution_lock
        )
        scientific_profile = require_dependency_profile_ready(
            route.scientific_profile_id,
            self.repository_root / "configs" / "dependency_profile_registry.json",
        )
        self.scientific_profile_digest = scientific_profile.profile_digest
        self.scientific_complete_hash_lock_digest = (
            scientific_profile.complete_hash_lock_digest
        )
        if interval_seconds < 0:
            raise ValueError("checkpoint 间隔不得为负数")
        self.interval_seconds = float(interval_seconds)
        configuration_environment = _configuration_environment()
        self.configuration_environment_keys = tuple(configuration_environment)
        self.configuration_identity_digest = build_stable_digest(
            {
                "route_name": route.route_name,
                "workflow_name": route.workflow_name,
                "scientific_profile_id": route.scientific_profile_id,
                "scientific_profile_digest": self.scientific_profile_digest,
                "scientific_complete_hash_lock_digest": (
                    self.scientific_complete_hash_lock_digest
                ),
                "baseline_id": route.baseline_id,
                "paper_run": _paper_run_identity(paper_run),
                "formal_execution_lock": self.formal_execution_lock,
                "configuration_environment_digest": build_stable_digest(
                    configuration_environment
                ),
            }
        )
        self.checkpoint_directory = (
            self.persistent_output_dir
            / CHECKPOINT_DIRECTORY_NAME
            / paper_run.run_name
            / route.route_name
            / self.configuration_identity_digest[:16]
        )
        output_root = route.output_root(self.repository_root, paper_run.run_name)
        try:
            self.checkpoint_directory.resolve().relative_to(output_root)
        except ValueError:
            pass
        else:
            raise ValueError("checkpoint 目录不得位于当前 workflow 输出目录内")
        self._snapshot_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_error: BaseException | None = None

    @property
    def pointer_path(self) -> Path:
        """返回当前身份的原子 generation 指针."""

        return self.checkpoint_directory / CHECKPOINT_POINTER_FILE_NAME

    @property
    def generations_directory(self) -> Path:
        """返回不可变 generation 根目录."""

        return self.checkpoint_directory / "generations"

    @property
    def completed_checkpoint_artifact_role(self) -> str:
        """返回同时绑定路由、profile、配置和执行锁的完成单元角色.

        底层 ``resume_checkpoint`` 以 artifact role 隔离恢复集合. 在角色中加入
        当前配置身份摘要后, 不同 commit、依赖 profile 或科学参数不会被同一次
        恢复扫描混合.
        """

        return (
            f"{self.route.checkpoint_role_name}_"
            f"{self.configuration_identity_digest[:12]}"
        )

    @property
    def shared_checkpoint_root(self) -> Path:
        """返回内层通用 checkpoint 原语使用的持久化根目录."""

        return self.persistent_output_dir / "cp"

    def _iter_source_files(self) -> tuple[tuple[Path, Path], ...]:
        """枚举属于当前路由且可安全复制的普通文件."""

        output_root = self.route.output_root(
            self.repository_root,
            self.paper_run.run_name,
        )
        if not output_root.exists():
            return ()
        source_files: list[tuple[Path, Path]] = []
        for source_path in sorted(output_root.rglob("*")):
            if source_path.is_symlink():
                raise WorkflowCheckpointError("workflow 输出不得包含符号链接")
            if not source_path.is_file():
                continue
            relative_path = source_path.relative_to(output_root)
            if self.route.allows_relative_path(relative_path):
                source_files.append((source_path, relative_path))
        return tuple(source_files)

    def persist_validated_completed_unit(self) -> dict[str, Any]:
        """通过内层通用原语保存已通过完成门禁的整条 workflow.

        该快照仍声明 ``supports_paper_claim=false``. 它只允许断线后的计算重入,
        不能替代各 workflow 的正式 archive、records 或精确9+3聚合来源.
        """

        source_paths = tuple(
            source_path for source_path, _ in self._iter_source_files()
        )
        return persist_checkpoint_files(
            repository_root=self.repository_root,
            artifact_role=self.completed_checkpoint_artifact_role,
            paper_run_name=self.paper_run.run_name,
            checkpoint_kind="completed_workflow",
            checkpoint_id="workflow",
            paths=source_paths,
            checkpoint_root=self.shared_checkpoint_root,
        )

    def restore_validated_completed_unit(self) -> dict[str, Any]:
        """通过内层通用原语恢复同一身份的完成单元."""

        completed_role_root = (
            self.shared_checkpoint_root
            / self.paper_run.run_name
            / self.completed_checkpoint_artifact_role
        )
        if not completed_role_root.is_dir():
            return {
                "checkpoint_persistence_configured": True,
                "restored_manifest_count": 0,
                "restored_file_count": 0,
                "supports_paper_claim": False,
            }
        output_prefix = self.route.output_root(
            self.repository_root,
            self.paper_run.run_name,
        ).relative_to(self.repository_root).as_posix()
        return restore_role_checkpoints(
            repository_root=self.repository_root,
            artifact_role=self.completed_checkpoint_artifact_role,
            paper_run_name=self.paper_run.run_name,
            allowed_output_prefix=output_prefix,
            checkpoint_root=self.shared_checkpoint_root,
            before_restore=self._clear_route_outputs,
        )

    def _copy_stable_file(
        self,
        source_path: Path,
        target_path: Path,
    ) -> tuple[int, str] | None:
        """只保留复制期间未变化的文件, 避免读取 runner 正在改写的半文件."""

        before = source_path.stat()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with source_path.open("rb") as source, target_path.open("wb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                target.write(chunk)
                digest.update(chunk)
        after = source_path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or target_path.stat().st_size != after.st_size
        ):
            target_path.unlink(missing_ok=True)
            return None
        return after.st_size, digest.hexdigest()

    def snapshot(self, checkpoint_state: str) -> dict[str, Any]:
        """写出一个不可变 generation, 再原子更新 current 指针."""

        if checkpoint_state not in {"running", "interrupted", "completed"}:
            raise ValueError("未知 checkpoint 状态")
        with self._snapshot_lock:
            self.generations_directory.mkdir(parents=True, exist_ok=True)
            build_directory = self.generations_directory / f".new-{uuid.uuid4().hex[:16]}"
            payload_directory = build_directory / CHECKPOINT_PAYLOAD_DIRECTORY_NAME
            payload_directory.mkdir(parents=True, exist_ok=False)
            file_records: list[dict[str, Any]] = []
            unstable_files: list[str] = []
            try:
                for file_index, (source_path, _output_relative_path) in enumerate(
                    self._iter_source_files()
                ):
                    repository_relative_path = source_path.relative_to(
                        self.repository_root
                    )
                    payload_name = f"{file_index:08d}.bin"
                    target_path = payload_directory / payload_name
                    copied = self._copy_stable_file(source_path, target_path)
                    if copied is None:
                        unstable_files.append(repository_relative_path.as_posix())
                        continue
                    size_bytes, digest = copied
                    file_records.append(
                        {
                            "path": repository_relative_path.as_posix(),
                            "payload_name_intermediate": payload_name,
                            "size_bytes": size_bytes,
                            "sha256": digest,
                        }
                    )
                manifest: dict[str, Any] = {
                    "report_schema": CHECKPOINT_SCHEMA,
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "route_name": self.route.route_name,
                    "workflow_name": self.route.workflow_name,
                    "scientific_profile_id": self.route.scientific_profile_id,
                    "scientific_profile_digest": self.scientific_profile_digest,
                    "scientific_complete_hash_lock_digest": (
                        self.scientific_complete_hash_lock_digest
                    ),
                    "baseline_id": self.route.baseline_id,
                    "paper_run_name": self.paper_run.run_name,
                    "formal_execution_lock": self.formal_execution_lock,
                    "configuration_identity_digest": self.configuration_identity_digest,
                    "configuration_environment_keys_intermediate": list(
                        self.configuration_environment_keys
                    ),
                    "checkpoint_state_intermediate": checkpoint_state,
                    "workflow_completed_intermediate": checkpoint_state == "completed",
                    "checkpoint_file_records_intermediate": file_records,
                    "unstable_files_intermediate": unstable_files,
                    "decision": "diagnostic",
                    "supports_paper_claim": False,
                }
                checkpoint_digest = _checkpoint_manifest_digest(manifest)
                manifest["checkpoint_digest_intermediate"] = checkpoint_digest
                manifest["checkpoint_generation_intermediate"] = checkpoint_digest
                (build_directory / CHECKPOINT_MANIFEST_FILE_NAME).write_text(
                    json.dumps(
                        manifest,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                generation_directory = (
                    self.generations_directory / checkpoint_digest[:16]
                )
                if generation_directory.exists():
                    shutil.rmtree(build_directory)
                else:
                    os.replace(build_directory, generation_directory)
                pointer = {
                    "report_schema": CHECKPOINT_POINTER_SCHEMA,
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "route_name": self.route.route_name,
                    "workflow_name": self.route.workflow_name,
                    "paper_run_name": self.paper_run.run_name,
                    "configuration_identity_digest": self.configuration_identity_digest,
                    "checkpoint_state_intermediate": checkpoint_state,
                    "workflow_completed_intermediate": checkpoint_state == "completed",
                    "checkpoint_digest_intermediate": checkpoint_digest,
                    "checkpoint_generation_intermediate": checkpoint_digest,
                    "decision": "diagnostic",
                    "supports_paper_claim": False,
                }
                _write_json_atomic(self.pointer_path, pointer)
                return manifest
            except BaseException:
                if build_directory.exists():
                    shutil.rmtree(build_directory)
                raise

    def _load_current_generation(self) -> tuple[dict[str, Any], Path] | None:
        """读取 current 指针并完整复验 generation 内容."""

        if not self.pointer_path.is_file():
            return None
        pointer = _read_json_object(self.pointer_path, "checkpoint 指针")
        pointer_fields = {
            "report_schema",
            "schema_version",
            "route_name",
            "workflow_name",
            "paper_run_name",
            "configuration_identity_digest",
            "checkpoint_state_intermediate",
            "workflow_completed_intermediate",
            "checkpoint_digest_intermediate",
            "checkpoint_generation_intermediate",
            "decision",
            "supports_paper_claim",
        }
        if set(pointer) != pointer_fields:
            raise WorkflowCheckpointError("checkpoint 指针字段集合无效")
        expected_pointer = {
            "report_schema": CHECKPOINT_POINTER_SCHEMA,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "route_name": self.route.route_name,
            "workflow_name": self.route.workflow_name,
            "paper_run_name": self.paper_run.run_name,
            "configuration_identity_digest": self.configuration_identity_digest,
            "decision": "diagnostic",
            "supports_paper_claim": False,
        }
        if any(pointer.get(key) != value for key, value in expected_pointer.items()):
            raise WorkflowCheckpointError("checkpoint 指针身份不匹配")
        generation = pointer.get("checkpoint_generation_intermediate")
        digest = pointer.get("checkpoint_digest_intermediate")
        if (
            not isinstance(generation, str)
            or len(generation) != 64
            or generation != digest
        ):
            raise WorkflowCheckpointError("checkpoint 指针摘要无效")
        generation_directory = self.generations_directory / generation[:16]
        manifest = _read_json_object(
            generation_directory / CHECKPOINT_MANIFEST_FILE_NAME,
            "checkpoint manifest",
        )
        manifest_fields = {
            "report_schema",
            "schema_version",
            "route_name",
            "workflow_name",
            "scientific_profile_id",
            "scientific_profile_digest",
            "scientific_complete_hash_lock_digest",
            "baseline_id",
            "paper_run_name",
            "formal_execution_lock",
            "configuration_identity_digest",
            "configuration_environment_keys_intermediate",
            "checkpoint_state_intermediate",
            "workflow_completed_intermediate",
            "checkpoint_file_records_intermediate",
            "unstable_files_intermediate",
            "checkpoint_digest_intermediate",
            "checkpoint_generation_intermediate",
            "decision",
            "supports_paper_claim",
        }
        manifest_identity = {
            "report_schema": CHECKPOINT_SCHEMA,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "route_name": self.route.route_name,
            "workflow_name": self.route.workflow_name,
            "scientific_profile_id": self.route.scientific_profile_id,
            "scientific_profile_digest": self.scientific_profile_digest,
            "scientific_complete_hash_lock_digest": (
                self.scientific_complete_hash_lock_digest
            ),
            "baseline_id": self.route.baseline_id,
            "paper_run_name": self.paper_run.run_name,
            "decision": "diagnostic",
            "supports_paper_claim": False,
        }
        if (
            set(manifest) != manifest_fields
            or any(
                manifest.get(key) != value
                for key, value in manifest_identity.items()
            )
            or manifest.get("checkpoint_generation_intermediate") != generation
            or manifest.get("checkpoint_digest_intermediate") != generation
            or _checkpoint_manifest_digest(manifest) != generation
            or manifest.get("formal_execution_lock") != self.formal_execution_lock
            or manifest.get("configuration_identity_digest")
            != self.configuration_identity_digest
            or manifest.get("supports_paper_claim") is not False
            or pointer.get("checkpoint_state_intermediate")
            != manifest.get("checkpoint_state_intermediate")
            or pointer.get("workflow_completed_intermediate")
            != manifest.get("workflow_completed_intermediate")
        ):
            raise WorkflowCheckpointError("checkpoint manifest 身份或摘要无效")
        checkpoint_state = manifest.get("checkpoint_state_intermediate")
        workflow_completed = manifest.get("workflow_completed_intermediate")
        if (
            checkpoint_state not in {"running", "interrupted", "completed"}
            or not isinstance(workflow_completed, bool)
            or workflow_completed != (checkpoint_state == "completed")
        ):
            raise WorkflowCheckpointError("checkpoint 状态组合无效")
        if manifest.get("configuration_environment_keys_intermediate") != list(
            self.configuration_environment_keys
        ):
            raise WorkflowCheckpointError("checkpoint 配置环境键集合不一致")
        unstable_files = manifest.get("unstable_files_intermediate")
        if not isinstance(unstable_files, list) or any(
            not isinstance(path, str) for path in unstable_files
        ):
            raise WorkflowCheckpointError("checkpoint 不稳定文件记录无效")
        records = manifest.get("checkpoint_file_records_intermediate")
        if not isinstance(records, list):
            raise WorkflowCheckpointError("checkpoint 文件记录必须是列表")
        payload_directory = generation_directory / CHECKPOINT_PAYLOAD_DIRECTORY_NAME
        seen_paths: set[str] = set()
        output_root = self.route.output_root(
            self.repository_root,
            self.paper_run.run_name,
        )
        for record in records:
            if not isinstance(record, dict) or set(record) != {
                "path",
                "payload_name_intermediate",
                "size_bytes",
                "sha256",
            }:
                raise WorkflowCheckpointError("checkpoint 文件记录必须是 object")
            path_value = record.get("path")
            if not isinstance(path_value, str) or path_value in seen_paths:
                raise WorkflowCheckpointError("checkpoint 文件路径无效或重复")
            size_bytes = record.get("size_bytes")
            sha256 = record.get("sha256")
            payload_name = record.get("payload_name_intermediate")
            if (
                not isinstance(size_bytes, int)
                or isinstance(size_bytes, bool)
                or size_bytes < 0
                or not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
                or not isinstance(payload_name, str)
                or not payload_name.endswith(".bin")
                or Path(payload_name).name != payload_name
            ):
                raise WorkflowCheckpointError("checkpoint 文件大小或摘要无效")
            seen_paths.add(path_value)
            repository_relative_path = Path(path_value)
            if repository_relative_path.is_absolute() or ".." in repository_relative_path.parts:
                raise WorkflowCheckpointError("checkpoint 文件路径越界")
            destination = (self.repository_root / repository_relative_path).resolve()
            try:
                output_relative_path = destination.relative_to(output_root)
            except ValueError as exc:
                raise WorkflowCheckpointError("checkpoint 文件不属于当前输出 family") from exc
            if not self.route.allows_relative_path(output_relative_path):
                raise WorkflowCheckpointError("checkpoint 文件不属于当前路由")
            payload_path = payload_directory / payload_name
            if payload_path.is_symlink() or not payload_path.is_file():
                raise WorkflowCheckpointError("checkpoint payload 文件缺失或类型无效")
            if (
                payload_path.stat().st_size != record.get("size_bytes")
                or _file_sha256(payload_path) != record.get("sha256")
            ):
                raise WorkflowCheckpointError("checkpoint payload 摘要失配")
        actual_payload_paths = {
            path.name
            for path in payload_directory.rglob("*")
            if path.is_file()
        }
        declared_payload_paths = {
            str(record["payload_name_intermediate"]) for record in records
        }
        if (
            len(declared_payload_paths) != len(records)
            or actual_payload_paths != declared_payload_paths
        ):
            raise WorkflowCheckpointError("checkpoint payload 文件集合与 manifest 不一致")
        return manifest, payload_directory

    def _clear_route_outputs(self) -> None:
        """只清理当前路由拥有的本地文件, 避免混合旧会话内容."""

        output_root = self.route.output_root(
            self.repository_root,
            self.paper_run.run_name,
        )
        if self.route.workflow_name != "external_baseline_method_faithful":
            if output_root.is_symlink():
                output_root.unlink()
            elif output_root.exists():
                shutil.rmtree(output_root)
            return
        run_directory = output_root / "run_records" / self.route.baseline_id
        if run_directory.is_symlink():
            run_directory.unlink()
        elif run_directory.exists():
            shutil.rmtree(run_directory)
        split_directory = output_root / "split_observations"
        for name in (
            f"{self.route.baseline_id}_baseline_observations.json",
            f"{self.route.baseline_id}_baseline_command_results.json",
            f"{self.route.baseline_id}_baseline_transfer_manifest.json",
        ):
            path = split_directory / name
            if path.is_file() or path.is_symlink():
                path.unlink()

    def restore_latest(self) -> dict[str, Any] | None:
        """验证并恢复 current generation, 不把其标记为正式证据."""

        current = self._load_current_generation()
        if current is None:
            return None
        manifest, payload_directory = current
        self._clear_route_outputs()
        records = manifest["checkpoint_file_records_intermediate"]
        for record in records:
            repository_relative_path = Path(record["path"])
            source_path = payload_directory / record["payload_name_intermediate"]
            destination_path = self.repository_root / repository_relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = destination_path.with_name(
                f".{destination_path.name}.{uuid.uuid4().hex}.restore"
            )
            shutil.copyfile(source_path, temporary_path)
            if (
                temporary_path.stat().st_size != record["size_bytes"]
                or _file_sha256(temporary_path) != record["sha256"]
            ):
                temporary_path.unlink(missing_ok=True)
                raise WorkflowCheckpointError("checkpoint 恢复后的文件摘要失配")
            os.replace(temporary_path, destination_path)
        return manifest

    def validate_completed_output(
        self,
        returned_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """验证完成态摘要、运行锁和必需文件, 决定是否允许跳过重算."""

        summary = _read_json_object(
            self.route.summary_path(self.repository_root, self.paper_run.run_name),
            "workflow summary",
        )
        manifest = _read_json_object(
            self.route.manifest_path(self.repository_root, self.paper_run.run_name),
            "workflow manifest",
        )
        if returned_summary is not None and dict(returned_summary) != summary:
            raise WorkflowCheckpointError("runner 返回摘要与持久化摘要不一致")
        if not all(
            (
                summary.get("run_decision") == "pass",
                summary.get(self.route.ready_field) is True,
                summary.get(self.route.summary_baseline_field)
                == self.route.baseline_id,
                summary.get(self.route.summary_paper_run_field)
                == self.paper_run.run_name,
                manifest.get("formal_execution_run_lock")
                == self.formal_execution_lock,
                manifest.get("code_version")
                == self.formal_execution_lock["formal_execution_commit"],
            )
        ):
            raise WorkflowCheckpointError("workflow 完成态门禁未通过")
        output_root = self.route.output_root(
            self.repository_root,
            self.paper_run.run_name,
        )
        required_paths = (
            self.route.summary_relative_path,
            self.route.manifest_relative_path,
            *self.route.required_relative_paths,
        )
        if any(not (output_root / relative_path).is_file() for relative_path in required_paths):
            raise WorkflowCheckpointError("workflow 完成态必需文件不完整")
        if self.route.scientific_artifact_role is not None:
            binding_path = output_root / self.route.required_relative_paths[0]
            validate_scientific_execution_binding(
                binding_path,
                expected_artifact_role=self.route.scientific_artifact_role,
                expected_paper_run_name=self.paper_run.run_name,
                repository_root=self.repository_root,
            )
        return summary

    def start_periodic_snapshot(self) -> None:
        """立即保存一次运行态, 再启动后台定时镜像."""

        self.snapshot("running")
        if self.interval_seconds == 0:
            return

        def worker() -> None:
            while not self._stop_event.wait(self.interval_seconds):
                try:
                    self.snapshot("running")
                except BaseException as exc:
                    self._thread_error = exc
                    self._stop_event.set()
                    return

        self._thread = threading.Thread(
            target=worker,
            name=f"{self.route.route_name}-checkpoint",
            daemon=True,
        )
        self._thread.start()

    def stop_periodic_snapshot(self, checkpoint_state: str) -> dict[str, Any]:
        """停止后台镜像并写出最后一个明确状态."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        if self._thread_error is not None:
            raise WorkflowCheckpointError("后台 checkpoint 镜像失败") from self._thread_error
        return self.snapshot(checkpoint_state)


def _resolve_interval_seconds(value: float | None) -> float:
    """解析显式参数或环境变量中的 checkpoint 周期."""

    if value is not None:
        return float(value)
    return float(
        os.environ.get(
            CHECKPOINT_INTERVAL_ENVIRONMENT_KEY,
            str(DEFAULT_CHECKPOINT_INTERVAL_SECONDS),
        )
    )


def build_persistent_workflow_session(
    *,
    root: str | Path,
    workflow_name: str,
    persistent_output_dir: str | Path,
    baseline_id: str | None = None,
    checkpoint_interval_seconds: float | None = None,
    paper_run: PaperRunConfig | None = None,
    formal_execution_lock: Mapping[str, Any] | None = None,
) -> PersistentWorkflowSession:
    """构造 Notebook 与服务器共享的持久化会话."""

    root_path = Path(root).resolve()
    return PersistentWorkflowSession(
        repository_root=root_path,
        persistent_output_dir=persistent_output_dir,
        route=resolve_persistent_workflow_route(
            workflow_name,
            baseline_id=baseline_id,
        ),
        paper_run=paper_run or build_paper_run_config(root_path),
        formal_execution_lock=(
            formal_execution_lock
            or require_published_formal_execution_lock(root_path)
        ),
        interval_seconds=_resolve_interval_seconds(checkpoint_interval_seconds),
    )


def run_persistent_workflow(
    *,
    root: str | Path,
    workflow_name: str,
    runner: Callable[[], Mapping[str, Any]],
    persistent_output_dir: str | Path | None,
    baseline_id: str | None = None,
    checkpoint_interval_seconds: float | None = None,
) -> Mapping[str, Any]:
    """恢复、运行并保存一条外部 workflow.

    完成态 checkpoint 通过全部本地门禁后可以直接重入. 运行态或中断态只恢复
    已稳定落盘的文件, 随后仍调用原科学 runner; runner 是否复用某个科学单元
    由其自身协议决定, checkpoint 本身不伪造逐样本完成记录.
    """

    if persistent_output_dir is None:
        return runner()
    session = build_persistent_workflow_session(
        root=root,
        workflow_name=workflow_name,
        persistent_output_dir=persistent_output_dir,
        baseline_id=baseline_id,
        checkpoint_interval_seconds=checkpoint_interval_seconds,
    )
    completed_restore = session.restore_validated_completed_unit()
    if int(completed_restore.get("restored_manifest_count", 0)) > 0:
        return session.validate_completed_output()

    session.restore_latest()
    session.start_periodic_snapshot()
    try:
        summary = runner()
        if not isinstance(summary, Mapping):
            raise TypeError("workflow runner 必须返回 summary mapping")
        validated_summary = session.validate_completed_output(summary)
    except BaseException as original_error:
        try:
            session.stop_periodic_snapshot("interrupted")
        except BaseException as persistence_error:
            if hasattr(original_error, "add_note"):
                original_error.add_note(
                    f"中断态 checkpoint 写出失败: {type(persistence_error).__name__}"
                )
        raise
    session.stop_periodic_snapshot("completed")
    session.persist_validated_completed_unit()
    return validated_summary


def restore_completed_persistent_workflow(
    *,
    root: str | Path,
    workflow_name: str,
    persistent_output_dir: str | Path | None,
    baseline_id: str | None = None,
) -> Mapping[str, Any] | None:
    """供独立打包入口恢复已完成结果; 无 checkpoint 时不改动本地输出."""

    if persistent_output_dir is None:
        return None
    persistent_path = Path(persistent_output_dir).expanduser().resolve()
    if not any(
        (persistent_path / directory_name).is_dir()
        for directory_name in (CHECKPOINT_DIRECTORY_NAME, "cp")
    ):
        return None
    session = build_persistent_workflow_session(
        root=root,
        workflow_name=workflow_name,
        persistent_output_dir=persistent_path,
        baseline_id=baseline_id,
        checkpoint_interval_seconds=0,
    )
    completed_restore = session.restore_validated_completed_unit()
    if int(completed_restore.get("restored_manifest_count", 0)) == 0:
        return None
    return session.validate_completed_output()


def persistent_route_names() -> Sequence[str]:
    """返回受共享持久化协议覆盖的7条公开路由."""

    return tuple(PERSISTENT_WORKFLOW_ROUTES)
