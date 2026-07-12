"""管理 SD3.5 方法忠实外部基线的原子科学完成单元.

长耗时外部基线会跨多个 Colab 会话运行. 该模块把一次 Prompt 源图生成或一次
Prompt 攻击评估保存为独立原子记录, 并把记录绑定到代码锁,依赖锁,外部方法
源码身份,Prompt,配置,随机种子和实际 CUDA 设备. 聚合层只读取完整且重新
验证通过的单元, 因而中断后的会话只需计算缺失单元.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from experiments.runtime import repository_environment
from experiments.runtime.scientific_unit_provenance import (
    SCIENTIFIC_UNIT_PROVENANCE_AGGREGATE_FIELDS,
    aggregate_scientific_unit_provenance,
    build_scientific_unit_provenance,
    validate_scientific_unit_provenance,
)
from main.core.digest import build_stable_digest


METHOD_FAITHFUL_UNIT_SCHEMA = "method_faithful_prompt_scientific_unit"
METHOD_FAITHFUL_UNIT_SCHEMA_VERSION = 1
METHOD_FAITHFUL_SOURCE_IDENTITY_SCHEMA = "method_faithful_governed_source_identity"
METHOD_FAITHFUL_SOURCE_IDENTITY_SCHEMA_VERSION = 1
METHOD_FAITHFUL_RUN_IDENTITY_SCHEMA = "method_faithful_atomic_run_identity"
METHOD_FAITHFUL_RUN_IDENTITY_SCHEMA_VERSION = 1
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ATOMIC_TEMP_PATTERN = re.compile(r"^\..+\.\d+\.tmp$")
_BASELINE_IDS = ("tree_ring", "gaussian_shading", "shallow_diffuse")
_METHOD_IMPLEMENTATION_PATHS = {
    baseline_id: (
        f"external_baseline/primary/{baseline_id}/adapter/run_slm_eval.py",
        f"external_baseline/primary/{baseline_id}/adapter/method_faithful_sd35.py",
    )
    for baseline_id in _BASELINE_IDS
}
_COMMON_IMPLEMENTATION_PATHS = (
    "external_baseline/primary/sd35_method_faithful_common.py",
    "external_baseline/primary/sd35_method_faithful_units.py",
)


def _stable_json_text(value: Any) -> str:
    """以稳定字段顺序序列化 JSON."""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _read_json(path: Path) -> Any:
    """读取 UTF-8 JSON 文件."""

    return json.loads(path.read_text(encoding="utf-8-sig"))


def _file_sha256(path: Path) -> str:
    """计算文件内容的 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_irreversible_random_material_digest(*values: Any) -> str:
    """计算秘密随机材料的长度分隔 SHA-256, 不持久化材料原文."""

    digest = hashlib.sha256()
    for value in values:
        if value is None:
            encoded = b"<none>"
        elif isinstance(value, bytes):
            encoded = value
        elif isinstance(value, str):
            encoded = value.encode("utf-8")
        elif hasattr(value, "detach"):
            encoded = value.detach().cpu().contiguous().numpy().tobytes()
        else:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    """先同步临时文件再原子替换目标, 避免中断留下半条完成记录."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(_stable_json_text(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _remove_abandoned_atomic_temps(directory: Path) -> None:
    """仅清理由本 helper 命名且未形成完成记录的普通临时文件."""

    resolved_directory = directory.resolve()
    for path in directory.iterdir():
        if _ATOMIC_TEMP_PATTERN.fullmatch(path.name) is None:
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("方法忠实原子临时路径必须是 artifact 目录内的普通文件")
        if path.resolve().parent != resolved_directory:
            raise ValueError("方法忠实原子临时路径逃逸 artifact 目录")
        path.unlink()


def _required_text(value: Any, field_name: str) -> str:
    """解析必需非空文本."""

    resolved = str(value or "").strip()
    if not resolved:
        raise ValueError(f"方法忠实完成单元缺少 {field_name}")
    return resolved


def _required_sha256(value: Any, field_name: str) -> str:
    """解析必需 SHA-256 文本."""

    resolved = _required_text(value, field_name)
    if _SHA256_PATTERN.fullmatch(resolved) is None:
        raise ValueError(f"方法忠实完成单元的 {field_name} 不是 SHA-256")
    return resolved


def _repository_root() -> Path:
    """返回包含外部基线登记表的仓库根目录."""

    return Path(__file__).resolve().parents[2]


def _relative_repository_path(path: Path, root_path: Path) -> str:
    """要求实现文件位于仓库内并返回 POSIX 相对路径."""

    return path.resolve().relative_to(root_path.resolve()).as_posix()


def repository_relative_method_faithful_path(
    context: "MethodFaithfulUnitContext",
    path: str | Path,
) -> str:
    """把仓库内运行文件规范化为可跨 workspace 搬迁的相对 POSIX 路径."""

    candidate = Path(path)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (context.root_path / candidate).resolve()
    )
    return resolved.relative_to(context.root_path).as_posix()


def resolve_method_faithful_output_path(
    context: "MethodFaithfulUnitContext",
    path: str | Path,
) -> Path:
    """在当前 checkout 中解析单元记录的可迁移 outputs 相对路径."""

    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError("方法忠实完成单元不得使用绝对 outputs 路径")
    resolved = (context.root_path / candidate).resolve()
    resolved.relative_to(context.artifact_root)
    return resolved


def _canonicalize_unit_data_paths(value: Any, root_path: Path, field_name: str = "") -> Any:
    """递归规范化 unit_data 中以 _path 结尾的仓库文件字段."""

    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_unit_data_paths(item, root_path, str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _canonicalize_unit_data_paths(item, root_path, field_name)
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            _canonicalize_unit_data_paths(item, root_path, field_name)
            for item in value
        ]
    if field_name.endswith("_path") and isinstance(value, str) and value:
        candidate = Path(value)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (root_path / candidate).resolve()
        )
        return resolved.relative_to(root_path).as_posix()
    return value


def _validate_unit_data_paths(value: Any, root_path: Path, field_name: str = "") -> None:
    """拒绝 unit_data 中的绝对路径,父目录逃逸或平台相关分隔符."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_unit_data_paths(item, root_path, str(key))
        for key, item in value.items():
            key_text = str(key)
            if not (
                key_text.endswith("_path")
                and isinstance(item, str)
                and item
            ):
                continue
            resolved = (root_path / item).resolve()
            resolved.relative_to((root_path / "outputs").resolve())
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"方法忠实完成单元引用的 outputs 文件不存在: {item}"
                )
            digest_field = key_text.removesuffix("_path") + "_digest"
            if digest_field in value and _file_sha256(resolved) != _required_sha256(
                value[digest_field],
                digest_field,
            ):
                raise ValueError("方法忠实完成单元 unit_data 文件摘要不匹配")
        return
    if isinstance(value, list):
        for item in value:
            _validate_unit_data_paths(item, root_path, field_name)
        return
    if not (field_name.endswith("_path") and isinstance(value, str) and value):
        return
    candidate = Path(value)
    if candidate.is_absolute() or "\\" in value:
        raise ValueError("方法忠实完成单元文件路径必须为相对 POSIX 路径")
    resolved = (root_path / candidate).resolve()
    resolved.relative_to(root_path)


def build_method_faithful_source_identity(
    baseline_id: str,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """绑定登记提交与实际执行适配实现的精确摘要."""

    resolved_baseline_id = str(baseline_id)
    if resolved_baseline_id not in _BASELINE_IDS:
        raise ValueError(f"不支持的方法忠实 baseline: {resolved_baseline_id}")
    root_path = Path(root).resolve() if root is not None else _repository_root()
    registry_path = root_path / "external_baseline/source_registry.json"
    registry = _read_json(registry_path)
    matches = [
        dict(item)
        for item in registry.get("baseline_sources", ())
        if str(item.get("baseline_id", "")) == resolved_baseline_id
    ]
    if len(matches) != 1:
        raise ValueError("外部 baseline 源码登记身份必须唯一")
    registry_item = matches[0]
    official_commit = _required_text(
        registry_item.get("official_repository_commit"),
        "official_repository_commit",
    )
    if _COMMIT_PATTERN.fullmatch(official_commit) is None:
        raise ValueError("外部 baseline 登记提交不是40位 Git commit")
    implementation_paths = (
        *_METHOD_IMPLEMENTATION_PATHS[resolved_baseline_id],
        *_COMMON_IMPLEMENTATION_PATHS,
    )
    implementation_sha256 = {
        relative_path: _file_sha256(root_path / relative_path)
        for relative_path in implementation_paths
    }
    payload = {
        "report_schema": METHOD_FAITHFUL_SOURCE_IDENTITY_SCHEMA,
        "schema_version": METHOD_FAITHFUL_SOURCE_IDENTITY_SCHEMA_VERSION,
        "baseline_id": resolved_baseline_id,
        "official_repository_url": _required_text(
            registry_item.get("official_repository_url"),
            "official_repository_url",
        ),
        "official_repository_commit": official_commit,
        "source_registry_path": _relative_repository_path(registry_path, root_path),
        "source_registry_sha256": _file_sha256(registry_path),
        "source_registry_item_digest": build_stable_digest(registry_item),
        "adapter_implementation_sha256": implementation_sha256,
        "supports_paper_claim": False,
    }
    payload["method_faithful_source_identity_digest"] = build_stable_digest(payload)
    return validate_method_faithful_source_identity(
        payload,
        expected_baseline_id=resolved_baseline_id,
    )


def validate_method_faithful_source_identity(
    record: Mapping[str, Any],
    *,
    expected_baseline_id: str | None = None,
) -> dict[str, Any]:
    """验证外部方法登记提交与适配实现摘要的自一致性."""

    payload = dict(record)
    if payload.get("report_schema") != METHOD_FAITHFUL_SOURCE_IDENTITY_SCHEMA:
        raise ValueError("方法忠实源码身份 schema 不匹配")
    if payload.get("schema_version") != METHOD_FAITHFUL_SOURCE_IDENTITY_SCHEMA_VERSION:
        raise ValueError("方法忠实源码身份 schema_version 不匹配")
    baseline_id = _required_text(payload.get("baseline_id"), "baseline_id")
    if baseline_id not in _BASELINE_IDS:
        raise ValueError("方法忠实源码身份包含未登记 baseline")
    if expected_baseline_id is not None and baseline_id != expected_baseline_id:
        raise ValueError("方法忠实源码身份 baseline 不匹配")
    if payload.get("supports_paper_claim") is not False:
        raise ValueError("方法忠实源码身份记录自身不得支持论文主张")
    commit = _required_text(
        payload.get("official_repository_commit"),
        "official_repository_commit",
    )
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError("方法忠实源码身份登记提交无效")
    for field_name in (
        "source_registry_sha256",
        "source_registry_item_digest",
    ):
        _required_sha256(payload.get(field_name), field_name)
    implementation_sha256 = payload.get("adapter_implementation_sha256")
    if not isinstance(implementation_sha256, Mapping) or not implementation_sha256:
        raise TypeError("方法忠实源码身份缺少适配实现摘要")
    for relative_path, digest in implementation_sha256.items():
        _required_text(relative_path, "adapter_implementation_path")
        _required_sha256(digest, "adapter_implementation_sha256")
    identity_digest = _required_sha256(
        payload.get("method_faithful_source_identity_digest"),
        "method_faithful_source_identity_digest",
    )
    digest_payload = {
        key: value
        for key, value in payload.items()
        if key != "method_faithful_source_identity_digest"
    }
    if build_stable_digest(digest_payload) != identity_digest:
        raise ValueError("方法忠实源码身份自摘要不匹配")
    return payload


def _prompt_identity(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    """构造不依赖绝对文件路径的 Prompt 身份."""

    prompt_text = _required_text(
        row.get("prompt_text") or row.get("prompt") or row.get("caption") or row.get("text"),
        "prompt_text",
    )
    prompt_id = _required_text(row.get("prompt_id") or f"prompt_{index:05d}", "prompt_id")
    prompt_digest = str(row.get("prompt_digest") or build_stable_digest(prompt_text))
    _required_sha256(prompt_digest, "prompt_digest")
    return {
        "prompt_id": prompt_id,
        "prompt_index": int(row.get("prompt_index", index - 1)),
        "prompt_set": _required_text(row.get("prompt_set") or "unspecified", "prompt_set"),
        "split": _required_text(row.get("split") or "test", "split"),
        "prompt_text": prompt_text,
        "prompt_digest": prompt_digest,
    }


@dataclass(frozen=True)
class MethodFaithfulUnitContext:
    """保存同一次 adapter 运行共享的代码,依赖和配置身份."""

    baseline_id: str
    root_path: Path
    artifact_root: Path
    unit_record_dir: Path
    run_identity_path: Path
    run_config: dict[str, Any]
    run_config_digest: str
    stable_execution_identity: dict[str, Any]
    stable_execution_identity_digest: str
    source_identity: dict[str, Any]
    runtime_environment: dict[str, Any]
    execution_device: str
    torch_module: Any


@dataclass(frozen=True)
class MethodFaithfulUnitSpec:
    """描述一个预期完成单元的稳定身份和记录路径."""

    unit_kind: str
    unit_id: str
    unit_config: dict[str, Any]
    unit_config_digest: str
    prompt_identity: dict[str, Any]
    random_identity_random: dict[str, Any]
    record_path: Path


def build_method_faithful_unit_context(
    *,
    baseline_id: str,
    artifact_root: str | Path,
    run_config: Mapping[str, Any],
    execution_device: str,
    torch_module: Any,
    root: str | Path | None = None,
) -> MethodFaithfulUnitContext:
    """创建真实运行上下文并验证当前代码锁和隔离依赖锁."""

    root_path = Path(root).resolve() if root is not None else _repository_root()
    resolved_artifact_root = Path(artifact_root).resolve()
    resolved_artifact_root.relative_to((root_path / "outputs").resolve())
    resolved_artifact_root.mkdir(parents=True, exist_ok=True)
    _remove_abandoned_atomic_temps(resolved_artifact_root)
    execution_lock = repository_environment.require_published_formal_execution_lock(root_path)
    runtime_environment = repository_environment.build_runtime_environment_report(
        "sd35_method_runtime_gpu",
        verified_formal_execution_lock=execution_lock,
    )
    if runtime_environment.get("dependency_environment_ready") is not True:
        raise RuntimeError("方法忠实完成单元的隔离依赖环境未通过完整锁门禁")
    if runtime_environment.get("isolated_scientific_context_ready") is not True:
        raise RuntimeError("方法忠实完成单元不在受治理隔离解释器中")
    isolated_context = runtime_environment.get("isolated_scientific_context")
    package_versions = runtime_environment.get("package_versions")
    if not isinstance(isolated_context, Mapping) or not isinstance(package_versions, Mapping):
        raise TypeError("方法忠实完成单元缺少隔离解释器或依赖版本身份")
    stable_execution_identity = {
        "dependency_profile_id": _required_text(
            runtime_environment.get("dependency_profile_id"),
            "dependency_profile_id",
        ),
        "dependency_profile_digest": _required_sha256(
            runtime_environment.get("dependency_profile_digest"),
            "dependency_profile_digest",
        ),
        "direct_requirements_digest": _required_sha256(
            runtime_environment.get("direct_requirements_digest"),
            "direct_requirements_digest",
        ),
        "complete_hash_lock_digest": _required_sha256(
            runtime_environment.get("complete_hash_lock_digest"),
            "complete_hash_lock_digest",
        ),
        "formal_execution_commit": _required_text(
            runtime_environment.get("formal_execution_commit"),
            "formal_execution_commit",
        ),
        "formal_execution_lock_digest": _required_sha256(
            runtime_environment.get("formal_execution_lock_digest"),
            "formal_execution_lock_digest",
        ),
        "python_version": _required_text(
            runtime_environment.get("python_version"),
            "python_version",
        ),
        "python_executable_sha256": _required_sha256(
            isolated_context.get("current_python_executable_sha256"),
            "python_executable_sha256",
        ),
        "torch_version": _required_text(package_versions.get("torch"), "torch_version"),
        "torch_cuda_version": _required_text(
            runtime_environment.get("cuda_version"),
            "torch_cuda_version",
        ),
    }
    if _COMMIT_PATTERN.fullmatch(stable_execution_identity["formal_execution_commit"]) is None:
        raise ValueError("方法忠实完成单元代码锁不是40位 Git commit")
    stable_execution_identity_digest = build_stable_digest(stable_execution_identity)
    normalized_run_config = dict(run_config)
    source_identity = build_method_faithful_source_identity(baseline_id, root=root_path)
    run_config_digest = build_stable_digest(
        {
            "baseline_id": baseline_id,
            "run_config": normalized_run_config,
            "stable_execution_identity": stable_execution_identity,
            "source_identity_digest": source_identity[
                "method_faithful_source_identity_digest"
            ],
        }
    )
    unit_record_dir = resolved_artifact_root / "completed_scientific_units"
    unit_record_dir.mkdir(parents=True, exist_ok=True)
    _remove_abandoned_atomic_temps(unit_record_dir)
    run_identity_path = resolved_artifact_root / "method_faithful_run_identity.json"
    run_identity = {
        "report_schema": METHOD_FAITHFUL_RUN_IDENTITY_SCHEMA,
        "schema_version": METHOD_FAITHFUL_RUN_IDENTITY_SCHEMA_VERSION,
        "baseline_id": str(baseline_id),
        "run_config": normalized_run_config,
        "run_config_digest": run_config_digest,
        "stable_scientific_execution_identity": stable_execution_identity,
        "stable_scientific_execution_identity_digest": stable_execution_identity_digest,
        "method_faithful_source_identity": source_identity,
        "method_faithful_source_identity_digest": source_identity[
            "method_faithful_source_identity_digest"
        ],
        "supports_paper_claim": False,
    }
    run_identity["method_faithful_run_identity_digest"] = build_stable_digest(
        run_identity
    )
    if run_identity_path.is_file():
        if _read_json(run_identity_path) != run_identity:
            raise ValueError("已有方法忠实原子运行身份与当前代码,依赖锁或配置不一致")
    else:
        if any(unit_record_dir.glob("*.json")):
            raise ValueError("已有方法忠实完成单元缺少原子运行身份记录")
        _atomic_write_json(run_identity_path, run_identity)
    return MethodFaithfulUnitContext(
        baseline_id=str(baseline_id),
        root_path=root_path,
        artifact_root=resolved_artifact_root,
        unit_record_dir=unit_record_dir,
        run_identity_path=run_identity_path,
        run_config=normalized_run_config,
        run_config_digest=run_config_digest,
        stable_execution_identity=stable_execution_identity,
        stable_execution_identity_digest=stable_execution_identity_digest,
        source_identity=source_identity,
        runtime_environment=runtime_environment,
        execution_device=str(execution_device),
        torch_module=torch_module,
    )


def build_method_faithful_unit_spec(
    context: MethodFaithfulUnitContext,
    *,
    unit_kind: str,
    row: Mapping[str, Any],
    index: int,
    random_identity_random: Mapping[str, Any],
    unit_parameters: Mapping[str, Any],
) -> MethodFaithfulUnitSpec:
    """按 Prompt,算子参数和随机性构造唯一完成单元身份."""

    prompt_identity = _prompt_identity(row, index)
    random_identity = dict(random_identity_random)
    invalid_random_fields = sorted(
        field_name
        for field_name in random_identity
        if not (
            field_name.endswith("_random")
            or field_name.endswith("_digest_random")
        )
    )
    if invalid_random_fields:
        raise ValueError(
            "方法忠实单元随机字段命名无效: " + ",".join(invalid_random_fields)
        )
    unit_config = {
        "baseline_id": context.baseline_id,
        "unit_kind": _required_text(unit_kind, "unit_kind"),
        "run_config_digest": context.run_config_digest,
        "prompt_identity": prompt_identity,
        "random_identity_random": random_identity,
        "unit_parameters": dict(unit_parameters),
    }
    unit_config_digest = build_stable_digest(unit_config)
    unit_id = (
        f"{context.baseline_id}__{unit_kind}__"
        f"{int(index):05d}__{unit_config_digest[:16]}"
    )
    record_path = context.unit_record_dir / (
        f"{unit_kind[:16]}__{int(index):05d}__{unit_config_digest[:16]}.json"
    )
    return MethodFaithfulUnitSpec(
        unit_kind=str(unit_kind),
        unit_id=unit_id,
        unit_config=unit_config,
        unit_config_digest=unit_config_digest,
        prompt_identity=prompt_identity,
        random_identity_random=random_identity,
        record_path=record_path,
    )


def _artifact_records(
    context: MethodFaithfulUnitContext,
    artifact_paths: Iterable[str | Path],
) -> list[dict[str, Any]]:
    """绑定单元实际写出的所有图像文件."""

    records: list[dict[str, Any]] = []
    for value in artifact_paths:
        path = Path(value).resolve()
        path.relative_to(context.artifact_root)
        if not path.is_file():
            raise FileNotFoundError(f"方法忠实单元产物不存在: {path}")
        records.append(
            {
                "artifact_path": path.relative_to(context.root_path).as_posix(),
                "artifact_sha256": _file_sha256(path),
                "artifact_size": path.stat().st_size,
            }
        )
    records.sort(key=lambda row: str(row["artifact_path"]))
    if len(records) != len({str(row["artifact_path"]) for row in records}):
        raise ValueError("方法忠实单元产物路径不得重复")
    return records


def write_completed_method_faithful_unit(
    context: MethodFaithfulUnitContext,
    spec: MethodFaithfulUnitSpec,
    *,
    unit_data: Mapping[str, Any],
    artifact_paths: Iterable[str | Path],
) -> dict[str, Any]:
    """在科学计算完成的同一进程原子写出完成记录."""

    if spec.record_path.exists():
        raise FileExistsError("方法忠实完成单元已经存在, 不得覆盖")
    provenance = build_scientific_unit_provenance(
        scientific_unit_id=spec.unit_id,
        scientific_unit_config_digest=spec.unit_config_digest,
        runtime_environment=context.runtime_environment,
        execution_device_name=context.execution_device,
        torch_module=context.torch_module,
        random_identity_random=spec.random_identity_random,
    )
    canonical_unit_data = _canonicalize_unit_data_paths(
        dict(unit_data),
        context.root_path,
    )
    _validate_unit_data_paths(canonical_unit_data, context.root_path)
    payload = {
        "report_schema": METHOD_FAITHFUL_UNIT_SCHEMA,
        "schema_version": METHOD_FAITHFUL_UNIT_SCHEMA_VERSION,
        "baseline_id": context.baseline_id,
        "unit_kind": spec.unit_kind,
        "scientific_unit_id": spec.unit_id,
        "scientific_unit_config": spec.unit_config,
        "scientific_unit_config_digest": spec.unit_config_digest,
        "prompt_identity": spec.prompt_identity,
        "run_config": context.run_config,
        "run_config_digest": context.run_config_digest,
        "stable_scientific_execution_identity": context.stable_execution_identity,
        "stable_scientific_execution_identity_digest": context.stable_execution_identity_digest,
        "method_faithful_source_identity": context.source_identity,
        "method_faithful_source_identity_digest": context.source_identity[
            "method_faithful_source_identity_digest"
        ],
        "scientific_unit_provenance": provenance,
        "unit_artifacts": _artifact_records(context, artifact_paths),
        "unit_data": canonical_unit_data,
        "unit_complete": True,
        "supports_paper_claim": False,
    }
    payload["method_faithful_unit_digest"] = build_stable_digest(payload)
    _atomic_write_json(spec.record_path, payload)
    return validate_completed_method_faithful_unit(
        spec.record_path,
        expected_spec=spec,
        expected_context=context,
    )


def validate_completed_method_faithful_unit(
    record: str | Path | Mapping[str, Any],
    *,
    expected_spec: MethodFaithfulUnitSpec | None = None,
    expected_context: MethodFaithfulUnitContext | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """验证完成记录,自摘要,科学来源和所有图像字节摘要."""

    payload = (
        _read_json(Path(record))
        if isinstance(record, (str, Path))
        else dict(record)
    )
    if payload.get("report_schema") != METHOD_FAITHFUL_UNIT_SCHEMA:
        raise ValueError("方法忠实完成单元 schema 不匹配")
    if payload.get("schema_version") != METHOD_FAITHFUL_UNIT_SCHEMA_VERSION:
        raise ValueError("方法忠实完成单元 schema_version 不匹配")
    if payload.get("unit_complete") is not True:
        raise ValueError("方法忠实完成单元未标记完整")
    if payload.get("supports_paper_claim") is not False:
        raise ValueError("方法忠实完成单元自身不得支持论文主张")
    baseline_id = _required_text(payload.get("baseline_id"), "baseline_id")
    unit_id = _required_text(payload.get("scientific_unit_id"), "scientific_unit_id")
    config_digest = _required_sha256(
        payload.get("scientific_unit_config_digest"),
        "scientific_unit_config_digest",
    )
    if build_stable_digest(payload.get("scientific_unit_config")) != config_digest:
        raise ValueError("方法忠实完成单元配置摘要不匹配")
    run_config_digest = _required_sha256(
        payload.get("run_config_digest"),
        "run_config_digest",
    )
    if expected_spec is not None:
        if unit_id != expected_spec.unit_id or config_digest != expected_spec.unit_config_digest:
            raise ValueError("已有方法忠实单元身份与本次配置不一致")
        if payload.get("prompt_identity") != expected_spec.prompt_identity:
            raise ValueError("已有方法忠实单元 Prompt 身份与本次配置不一致")
        if payload.get("unit_kind") != expected_spec.unit_kind:
            raise ValueError("已有方法忠实单元算子身份与本次配置不一致")
    if expected_context is not None:
        if baseline_id != expected_context.baseline_id:
            raise ValueError("已有方法忠实单元 baseline 身份不一致")
        if run_config_digest != expected_context.run_config_digest:
            raise ValueError("已有方法忠实单元运行配置身份不一致")
        if payload.get("run_config") != expected_context.run_config:
            raise ValueError("已有方法忠实单元运行配置内容不一致")
        if (
            payload.get("stable_scientific_execution_identity")
            != expected_context.stable_execution_identity
        ):
            raise ValueError("已有方法忠实单元稳定代码或依赖锁身份不一致")
        if (
            payload.get("stable_scientific_execution_identity_digest")
            != expected_context.stable_execution_identity_digest
        ):
            raise ValueError("已有方法忠实单元稳定执行身份摘要不一致")
        if payload.get("method_faithful_source_identity") != expected_context.source_identity:
            raise ValueError("已有方法忠实单元源码身份不一致")
    source_identity = validate_method_faithful_source_identity(
        payload.get("method_faithful_source_identity", {}),
        expected_baseline_id=baseline_id,
    )
    if (
        source_identity["method_faithful_source_identity_digest"]
        != payload.get("method_faithful_source_identity_digest")
    ):
        raise ValueError("方法忠实完成单元源码身份摘要引用不匹配")
    provenance = validate_scientific_unit_provenance(
        payload.get("scientific_unit_provenance", {}),
        expected_unit_id=unit_id,
        expected_config_digest=config_digest,
    )
    stable_execution_identity = payload.get("stable_scientific_execution_identity")
    if not isinstance(stable_execution_identity, Mapping):
        raise TypeError("方法忠实完成单元缺少稳定代码与依赖锁身份")
    stable_execution_identity = dict(stable_execution_identity)
    stable_execution_identity_digest = _required_sha256(
        payload.get("stable_scientific_execution_identity_digest"),
        "stable_scientific_execution_identity_digest",
    )
    if build_stable_digest(stable_execution_identity) != stable_execution_identity_digest:
        raise ValueError("方法忠实完成单元稳定执行身份自摘要不匹配")
    provenance_environment = provenance["scientific_execution_environment"]
    for field_name, expected_value in stable_execution_identity.items():
        if provenance_environment.get(field_name) != expected_value:
            raise ValueError(
                f"方法忠实完成单元稳定执行身份与逐单元来源的 {field_name} 不一致"
            )
    root_path = (
        expected_context.root_path
        if expected_context is not None
        else (Path(root).resolve() if root is not None else _repository_root())
    )
    unit_data = payload.get("unit_data")
    if not isinstance(unit_data, Mapping):
        raise TypeError("方法忠实完成单元缺少 unit_data")
    _validate_unit_data_paths(unit_data, root_path)
    artifacts = payload.get("unit_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("方法忠实完成单元必须绑定至少一个真实产物")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise TypeError("方法忠实完成单元产物记录必须为 object")
        path = (root_path / _required_text(artifact.get("artifact_path"), "artifact_path")).resolve()
        if expected_context is not None:
            path.relative_to(expected_context.artifact_root)
        if not path.is_file():
            raise FileNotFoundError(f"方法忠实完成单元绑定产物不存在: {path}")
        if _file_sha256(path) != _required_sha256(
            artifact.get("artifact_sha256"),
            "artifact_sha256",
        ):
            raise ValueError("方法忠实完成单元产物摘要不匹配")
        if path.stat().st_size != int(artifact.get("artifact_size", -1)):
            raise ValueError("方法忠实完成单元产物大小不匹配")
    unit_digest = _required_sha256(
        payload.get("method_faithful_unit_digest"),
        "method_faithful_unit_digest",
    )
    digest_payload = {
        key: value
        for key, value in payload.items()
        if key != "method_faithful_unit_digest"
    }
    if build_stable_digest(digest_payload) != unit_digest:
        raise ValueError("方法忠实完成单元自摘要不匹配")
    return payload


def load_completed_method_faithful_unit(
    context: MethodFaithfulUnitContext,
    spec: MethodFaithfulUnitSpec,
) -> dict[str, Any] | None:
    """读取已有完整单元; 缺失返回 None, 损坏或身份不符直接闭锁."""

    if not spec.record_path.exists():
        return None
    return validate_completed_method_faithful_unit(
        spec.record_path,
        expected_spec=spec,
        expected_context=context,
    )


def apply_frozen_threshold(
    observations_without_threshold: Iterable[Mapping[str, Any]],
    *,
    threshold: float,
    threshold_source: str,
) -> list[dict[str, Any]]:
    """在所有 calibration 源单元齐备后确定性应用冻结阈值."""

    observations: list[dict[str, Any]] = []
    for row in observations_without_threshold:
        updated = dict(row)
        updated["threshold"] = float(threshold)
        updated["threshold_source"] = str(threshold_source)
        updated["detection_decision"] = bool(float(updated["score"]) >= float(threshold))
        updated["final_decision"] = updated["detection_decision"]
        updated["baseline_observation_digest"] = build_stable_digest(updated)
        observations.append(updated)
    return observations


def threshold_independent_observation(
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """移除冻结阈值前不得成立的判定字段, 仅保存真实连续分数."""

    return {
        key: value
        for key, value in observation.items()
        if key
        not in {
            "threshold",
            "threshold_source",
            "detection_decision",
            "final_decision",
            "baseline_observation_digest",
        }
    }


def _validate_method_faithful_unit_exact_set(
    records: Iterable[Mapping[str, Any]],
    run_config: Mapping[str, Any],
) -> dict[str, Any]:
    """集中验证全 Prompt 源图与仅 test 攻击的精确笛卡尔积."""

    source_prompt_identities: list[dict[str, Any]] = []
    raw_observations: list[dict[str, Any]] = []
    observed_attack_keys: list[tuple[str, str, str]] = []
    for record in records:
        unit_data = record.get("unit_data")
        if not isinstance(unit_data, Mapping):
            raise TypeError("方法忠实完成单元缺少 unit_data")
        unit_kind = str(record.get("unit_kind", ""))
        if unit_kind == "source_pair":
            source_observations = unit_data.get("observations_without_threshold")
            if not isinstance(source_observations, list) or len(source_observations) != 2:
                raise ValueError("Prompt 源图单元必须包含阴性和阳性两条分数")
            roles = [str(row.get("sample_role", "")) for row in source_observations]
            if roles != ["clean_negative", "positive_source"]:
                raise ValueError("Prompt 源图单元角色顺序必须为 clean_negative,positive_source")
            prompt_identity = dict(record.get("prompt_identity", {}))
            prompt_id = str(prompt_identity.get("prompt_id", ""))
            if any(str(row.get("prompt_id", "")) != prompt_id for row in source_observations):
                raise ValueError("Prompt 源图单元 observation 的 Prompt 身份不一致")
            source_prompt_identities.append(prompt_identity)
            raw_observations.extend(dict(row) for row in source_observations)
            continue
        if not unit_kind.startswith("formal_attack_"):
            raise ValueError("方法忠实完成单元包含未知算子职责")
        attack_observation = unit_data.get("observation_without_threshold")
        if not isinstance(attack_observation, Mapping):
            raise ValueError("Prompt 攻击单元必须包含一条攻击分数")
        prompt_identity = dict(record.get("prompt_identity", {}))
        if prompt_identity.get("split") != "test":
            raise ValueError("正式攻击完成单元只能绑定 test Prompt")
        attack_row = dict(attack_observation)
        prompt_id = str(prompt_identity.get("prompt_id", ""))
        if str(attack_row.get("prompt_id", "")) != prompt_id:
            raise ValueError("Prompt 攻击单元 observation 的 Prompt 身份不一致")
        sample_role = str(attack_row.get("sample_role", ""))
        if sample_role not in {"attacked_negative", "attacked_positive"}:
            raise ValueError("Prompt 攻击单元角色必须为 attacked_negative 或 attacked_positive")
        raw_observations.append(attack_row)
        observed_attack_keys.append(
            (prompt_id, str(attack_row.get("attack_name", "")), sample_role)
        )

    prompt_ids = [str(row.get("prompt_id", "")) for row in source_prompt_identities]
    if any(not prompt_id for prompt_id in prompt_ids) or len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("方法忠实源图单元 Prompt id 必须非空且唯一")
    expected_prompt_count = int(run_config.get("prompt_count", -1))
    if len(source_prompt_identities) != expected_prompt_count:
        raise ValueError("方法忠实源图单元未覆盖正式 Prompt exact set")
    test_prompt_ids = {
        str(row["prompt_id"])
        for row in source_prompt_identities
        if str(row.get("split", "")) == "test"
    }
    if int(run_config.get("test_prompt_count", -1)) != len(test_prompt_ids):
        raise ValueError("方法忠实运行配置的 test Prompt 计数不一致")
    attack_names = tuple(str(name) for name in run_config.get("attack_families", ()))
    if len(attack_names) != len(set(attack_names)):
        raise ValueError("方法忠实运行配置的正式攻击名称不得重复")
    expected_attack_keys = {
        (prompt_id, attack_name, sample_role)
        for prompt_id in test_prompt_ids
        for attack_name in attack_names
        for sample_role in ("attacked_negative", "attacked_positive")
    }
    if len(observed_attack_keys) != len(set(observed_attack_keys)):
        raise ValueError("方法忠实攻击单元包含重复 Prompt,攻击或角色")
    if set(observed_attack_keys) != expected_attack_keys:
        raise ValueError("方法忠实攻击单元未覆盖 test Prompt 攻击 exact set")
    return {
        "raw_observations": raw_observations,
        "source_prompt_count": len(source_prompt_identities),
        "test_prompt_count": len(test_prompt_ids),
        "expected_formal_attack_unit_count": len(expected_attack_keys),
        "actual_formal_attack_unit_count": len(observed_attack_keys),
    }


def aggregate_method_faithful_unit_records(
    context: MethodFaithfulUnitContext,
    records: Iterable[Mapping[str, Any]],
    *,
    expected_specs: Iterable[MethodFaithfulUnitSpec],
) -> dict[str, Any]:
    """验证 exact-set 单元并汇总跨会话科学来源集合."""

    specs = tuple(expected_specs)
    rows = tuple(records)
    if len(rows) != len(specs):
        raise ValueError("方法忠实完成单元数量与正式计划不一致")
    validated = [
        validate_completed_method_faithful_unit(
            row,
            expected_spec=spec,
            expected_context=context,
        )
        for row, spec in zip(rows, specs, strict=True)
    ]
    exact_set = _validate_method_faithful_unit_exact_set(
        validated,
        context.run_config,
    )
    expected_paths = {spec.record_path.resolve() for spec in specs}
    actual_paths = {path.resolve() for path in context.unit_record_dir.glob("*.json")}
    if actual_paths != expected_paths:
        raise ValueError("方法忠实完成单元目录包含缺失或额外记录")
    referenced_artifact_paths = {
        (context.root_path / str(artifact["artifact_path"])).resolve()
        for row in validated
        for artifact in row["unit_artifacts"]
    }
    actual_image_paths = {
        path.resolve()
        for path in (context.artifact_root / "images").rglob("*")
        if path.is_file()
    }
    if actual_image_paths != referenced_artifact_paths:
        raise ValueError("方法忠实图像目录包含未绑定旧文件或缺失单元产物")
    allowed_derived_names = {
        f"{context.baseline_id}_image_pairs.json",
        "attacked_image_manifest.json",
    }
    allowed_artifact_files = {
        *referenced_artifact_paths,
        *expected_paths,
        context.run_identity_path.resolve(),
        *(
            (context.artifact_root / name).resolve()
            for name in allowed_derived_names
            if (context.artifact_root / name).is_file()
        ),
    }
    actual_artifact_files = {
        path.resolve()
        for path in context.artifact_root.rglob("*")
        if path.is_file()
    }
    if actual_artifact_files != allowed_artifact_files:
        raise ValueError("方法忠实 artifact 目录包含不可重建旧文件")
    provenance_aggregate = aggregate_scientific_unit_provenance(
        (row["scientific_unit_provenance"] for row in validated),
        expected_reference_count=len(specs),
    )
    if provenance_aggregate["scientific_unit_provenance_ready"] is not True:
        raise RuntimeError("方法忠实完成单元科学来源集合未闭合")
    return {
        "method_faithful_scientific_unit_count": len(validated),
        "method_faithful_scientific_unit_record_paths": [
            spec.record_path.relative_to(context.root_path).as_posix() for spec in specs
        ],
        "method_faithful_scientific_unit_records_digest": build_stable_digest(validated),
        "method_faithful_scientific_unit_resume_ready": True,
        "method_faithful_run_identity_path": context.run_identity_path.relative_to(
            context.root_path
        ).as_posix(),
        "method_faithful_run_identity_sha256": _file_sha256(
            context.run_identity_path
        ),
        "method_faithful_source_prompt_unit_count": exact_set["source_prompt_count"],
        "method_faithful_formal_attack_unit_count": exact_set[
            "actual_formal_attack_unit_count"
        ],
        **provenance_aggregate,
    }


def validate_method_faithful_adapter_unit_evidence(
    *,
    manifest: Mapping[str, Any],
    observation_rows: Iterable[Mapping[str, Any]],
    root: str | Path | None = None,
) -> dict[str, Any]:
    """由 runner 重读原子单元并复算最终 observation 与聚合摘要."""

    root_path = Path(root).resolve() if root is not None else _repository_root()
    payload = dict(manifest)
    adapter_digest = _required_sha256(payload.get("adapter_digest"), "adapter_digest")
    if build_stable_digest(
        {key: value for key, value in payload.items() if key != "adapter_digest"}
    ) != adapter_digest:
        raise ValueError("adapter manifest 自摘要不匹配")
    baseline_id = _required_text(payload.get("baseline_id"), "baseline_id")
    current_source_identity = build_method_faithful_source_identity(
        baseline_id,
        root=root_path,
    )
    if payload.get("method_faithful_source_identity") != current_source_identity:
        raise ValueError("adapter manifest 的受治理源码身份与当前代码不一致")
    run_identity_path = (
        root_path
        / _required_text(
            payload.get("method_faithful_run_identity_path"),
            "method_faithful_run_identity_path",
        )
    ).resolve()
    if not run_identity_path.is_file():
        raise FileNotFoundError("adapter manifest 绑定的原子运行身份记录不存在")
    if _file_sha256(run_identity_path) != _required_sha256(
        payload.get("method_faithful_run_identity_sha256"),
        "method_faithful_run_identity_sha256",
    ):
        raise ValueError("adapter manifest 的原子运行身份文件摘要不一致")
    run_identity = _read_json(run_identity_path)
    expected_run_identity = {
        "report_schema": METHOD_FAITHFUL_RUN_IDENTITY_SCHEMA,
        "schema_version": METHOD_FAITHFUL_RUN_IDENTITY_SCHEMA_VERSION,
        "baseline_id": baseline_id,
        "run_config": payload.get("run_config"),
        "run_config_digest": payload.get("run_config_digest"),
        "stable_scientific_execution_identity": payload.get(
            "stable_scientific_execution_identity"
        ),
        "stable_scientific_execution_identity_digest": payload.get(
            "stable_scientific_execution_identity_digest"
        ),
        "method_faithful_source_identity": current_source_identity,
        "method_faithful_source_identity_digest": current_source_identity[
            "method_faithful_source_identity_digest"
        ],
        "supports_paper_claim": False,
    }
    expected_run_identity["method_faithful_run_identity_digest"] = build_stable_digest(
        expected_run_identity
    )
    if run_identity != expected_run_identity:
        raise ValueError("adapter manifest 的原子运行身份内容不一致")
    record_paths = payload.get("method_faithful_scientific_unit_record_paths")
    if not isinstance(record_paths, list) or not record_paths:
        raise ValueError("adapter manifest 缺少方法忠实完成单元路径")
    validated = [
        validate_completed_method_faithful_unit(root_path / str(path), root=root_path)
        for path in record_paths
    ]
    if any(
        record.get("method_faithful_source_identity") != current_source_identity
        for record in validated
    ):
        raise ValueError("完成单元源码身份与 adapter manifest 不一致")
    manifest_stable_identity = payload.get("stable_scientific_execution_identity")
    manifest_stable_identity_digest = payload.get(
        "stable_scientific_execution_identity_digest"
    )
    if any(
        record.get("stable_scientific_execution_identity")
        != manifest_stable_identity
        or record.get("stable_scientific_execution_identity_digest")
        != manifest_stable_identity_digest
        for record in validated
    ):
        raise ValueError("完成单元稳定代码或依赖锁身份与 adapter manifest 不一致")
    if any(
        record.get("run_config") != payload.get("run_config")
        or record.get("run_config_digest") != payload.get("run_config_digest")
        for record in validated
    ):
        raise ValueError("完成单元运行配置与 adapter manifest 不一致")
    if len(validated) != int(payload.get("method_faithful_scientific_unit_count", -1)):
        raise ValueError("adapter manifest 的方法忠实完成单元计数不一致")
    if build_stable_digest(validated) != payload.get(
        "method_faithful_scientific_unit_records_digest"
    ):
        raise ValueError("adapter manifest 的方法忠实完成单元集合摘要不一致")
    provenance_aggregate = aggregate_scientific_unit_provenance(
        (row["scientific_unit_provenance"] for row in validated),
        expected_reference_count=len(validated),
    )
    for field_name in SCIENTIFIC_UNIT_PROVENANCE_AGGREGATE_FIELDS:
        if payload.get(field_name) != provenance_aggregate[field_name]:
            raise ValueError(f"adapter manifest 的 {field_name} 聚合值不一致")
    run_config = payload.get("run_config")
    if not isinstance(run_config, Mapping):
        raise TypeError("adapter manifest 缺少方法忠实运行配置")
    exact_set = _validate_method_faithful_unit_exact_set(validated, run_config)
    if int(payload.get("test_prompt_count", -1)) != exact_set["test_prompt_count"]:
        raise ValueError("adapter manifest 的 test Prompt 计数不一致")
    if int(payload.get("expected_formal_attack_unit_count", -1)) != exact_set[
        "expected_formal_attack_unit_count"
    ]:
        raise ValueError("adapter manifest 的正式攻击单元期望计数不一致")
    threshold = float(payload["threshold"])
    threshold_source = _required_text(payload.get("threshold_source"), "threshold_source")
    reconstructed = apply_frozen_threshold(
        exact_set["raw_observations"],
        threshold=threshold,
        threshold_source=threshold_source,
    )
    actual_observations = [dict(row) for row in observation_rows]
    if reconstructed != actual_observations:
        raise ValueError("最终 observation 无法由原子完成单元确定性重建")
    return {
        "method_faithful_scientific_unit_count": len(validated),
        "method_faithful_scientific_unit_records_digest": build_stable_digest(validated),
        "method_faithful_scientific_unit_resume_ready": True,
        "method_faithful_source_prompt_unit_count": exact_set["source_prompt_count"],
        "method_faithful_formal_attack_unit_count": exact_set[
            "actual_formal_attack_unit_count"
        ],
        **provenance_aggregate,
    }
