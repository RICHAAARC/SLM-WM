"""读取并校验不可变模型与数据资源登记表.

该模块把 Hugging Face 仓库标识和精确提交 revision 收敛到单一登记表.运行路径
只能使用40位小写十六进制提交, 不能使用 ``main``、分支名或短提交, 从而保证远程
仓库更新后仍可重建同一模型和数据输入.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


MODEL_SOURCE_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "model_source_registry.json"
)

_SOURCE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_EXACT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REPOSITORY_TYPES = frozenset({"dataset", "model"})


@dataclass(frozen=True)
class ModelSourceRequiredFile:
    """表示远程资源中必须逐字节匹配的单个运行输入文件."""

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """返回可写入运行证据的文件身份."""

        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class ModelSource:
    """表示一个由精确提交固定的远程模型或数据资源."""

    source_name: str
    provider: str
    repository_id: str
    repository_type: str
    revision: str
    source_url: str
    revision_url: str
    access_policy: str
    usage_roles: tuple[str, ...]
    required_files: tuple[ModelSourceRequiredFile, ...] = ()
    upstream_repository_id: str = ""
    upstream_access_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        """返回可直接写入运行记录的资源来源字段."""

        payload = {
            "source_name": self.source_name,
            "provider": self.provider,
            "repository_id": self.repository_id,
            "repository_type": self.repository_type,
            "revision": self.revision,
            "source_url": self.source_url,
            "revision_url": self.revision_url,
            "access_policy": self.access_policy,
            "usage_roles": list(self.usage_roles),
        }
        if self.required_files:
            payload["required_files"] = [item.to_dict() for item in self.required_files]
        if self.upstream_repository_id:
            payload["upstream_repository_id"] = self.upstream_repository_id
            payload["upstream_access_status"] = self.upstream_access_status
        return payload


def validate_exact_revision(revision: str, field_name: str = "revision") -> str:
    """校验 revision 是完整且不可变的40位 Git 提交."""

    if not isinstance(revision, str) or _EXACT_REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError(f"{field_name} 必须是40位小写十六进制 Git 提交")
    return revision


def _parse_required_files(source_name: str, payload: Any) -> tuple[ModelSourceRequiredFile, ...]:
    """校验可选文件清单, 防止精确 revision 下选择错误权重文件."""

    if payload is None:
        return ()
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"required_files 必须是非空 JSON 数组: {source_name}")
    required_files: list[ModelSourceRequiredFile] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"required_files[{index}] 必须是 JSON 对象: {source_name}")
        path = str(item.get("path", "")).strip().replace("\\", "/")
        path_parts = PurePosixPath(path).parts
        if (
            not path
            or path.startswith("/")
            or ".." in path_parts
            or PurePosixPath(path).as_posix() != path
            or any(character in path for character in "*?[]")
        ):
            raise ValueError(f"required_files[{index}].path 必须是仓库内 POSIX 相对路径: {source_name}")
        sha256 = str(item.get("sha256", "")).strip()
        if _SHA256_PATTERN.fullmatch(sha256) is None:
            raise ValueError(f"required_files[{index}].sha256 必须是64位小写 SHA-256: {source_name}")
        size_bytes = item.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
            raise ValueError(f"required_files[{index}].size_bytes 必须是正整数: {source_name}")
        required_files.append(
            ModelSourceRequiredFile(
                path=path,
                sha256=sha256,
                size_bytes=size_bytes,
            )
        )
    paths = tuple(item.path for item in required_files)
    if len(paths) != len(set(paths)):
        raise ValueError(f"required_files 包含重复路径: {source_name}")
    return tuple(required_files)


def _parse_source(source_name: str, payload: Any) -> ModelSource:
    """把单条 JSON 记录解析为经过统一校验的资源对象."""

    if _SOURCE_NAME_PATTERN.fullmatch(source_name) is None:
        raise ValueError(f"模型资源名称不符合 snake_case: {source_name}")
    if not isinstance(payload, dict):
        raise ValueError(f"模型资源记录必须是 JSON 对象: {source_name}")
    required_fields = (
        "provider",
        "repository_id",
        "repository_type",
        "revision",
        "source_url",
        "revision_url",
        "access_policy",
        "usage_roles",
    )
    missing_fields = tuple(field for field in required_fields if not payload.get(field))
    if missing_fields:
        raise ValueError(f"模型资源记录缺少字段 {missing_fields}: {source_name}")
    repository_id = str(payload["repository_id"])
    if repository_id.count("/") != 1 or any(character.isspace() for character in repository_id):
        raise ValueError(f"repository_id 必须是 owner/name 格式: {source_name}")
    repository_type = str(payload["repository_type"])
    if repository_type not in _REPOSITORY_TYPES:
        raise ValueError(f"repository_type 必须是 model 或 dataset: {source_name}")
    revision = validate_exact_revision(str(payload["revision"]), f"{source_name}.revision")
    revision_url = str(payload["revision_url"])
    if not revision_url.endswith(f"/tree/{revision}"):
        raise ValueError(f"revision_url 必须指向登记的精确提交: {source_name}")
    usage_roles_payload = payload["usage_roles"]
    if (
        not isinstance(usage_roles_payload, list)
        or not usage_roles_payload
        or any(not isinstance(role, str) or not role for role in usage_roles_payload)
        or len(usage_roles_payload) != len(set(usage_roles_payload))
    ):
        raise ValueError(f"usage_roles 必须是非空且无重复的字符串数组: {source_name}")
    return ModelSource(
        source_name=source_name,
        provider=str(payload["provider"]),
        repository_id=repository_id,
        repository_type=repository_type,
        revision=revision,
        source_url=str(payload["source_url"]),
        revision_url=revision_url,
        access_policy=str(payload["access_policy"]),
        usage_roles=tuple(usage_roles_payload),
        required_files=_parse_required_files(source_name, payload.get("required_files")),
        upstream_repository_id=str(payload.get("upstream_repository_id", "")),
        upstream_access_status=str(payload.get("upstream_access_status", "")),
    )


def load_model_source_registry(
    path: str | Path = MODEL_SOURCE_REGISTRY_PATH,
) -> dict[str, ModelSource]:
    """加载完整资源登记表, 并在返回前校验全部记录."""

    registry_path = Path(path)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"模型资源登记表不存在: {registry_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型资源登记表不是有效 JSON: {registry_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("模型资源登记表根节点必须是 JSON 对象")
    if payload.get("registry_schema") != "immutable_hugging_face_source_registry":
        raise ValueError("模型资源登记表 registry_schema 不匹配")
    if payload.get("schema_version") != 1:
        raise ValueError("模型资源登记表 schema_version 必须为 1")
    sources = payload.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("模型资源登记表必须包含非空 sources 对象")
    parsed = {name: _parse_source(name, record) for name, record in sources.items()}
    references = tuple((source.repository_id, source.revision) for source in parsed.values())
    if len(references) != len(set(references)):
        raise ValueError("模型资源登记表包含重复的 repository_id 与 revision")
    return parsed


def get_model_source(
    source_name: str,
    path: str | Path = MODEL_SOURCE_REGISTRY_PATH,
) -> ModelSource:
    """按稳定资源名称读取一条已校验记录."""

    sources = load_model_source_registry(path)
    try:
        return sources[source_name]
    except KeyError as exc:
        raise KeyError(f"模型资源未登记: {source_name}") from exc


def require_registered_model_reference(
    repository_id: str,
    revision: str,
    required_usage_role: str | None = None,
    path: str | Path = MODEL_SOURCE_REGISTRY_PATH,
) -> ModelSource:
    """确认模型仓库与精确 revision 组合存在于受治理登记表."""

    validate_exact_revision(revision)
    for source in load_model_source_registry(path).values():
        if (
            source.repository_type == "model"
            and source.repository_id == repository_id
            and source.revision == revision
            and (required_usage_role is None or required_usage_role in source.usage_roles)
        ):
            return source
    role_suffix = "" if required_usage_role is None else f" role={required_usage_role}"
    raise ValueError(f"模型仓库、revision 与用途组合未登记: {repository_id}@{revision}{role_suffix}")
