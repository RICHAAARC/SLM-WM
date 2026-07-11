"""构造并验证跨会话科学完成单元的真实运行来源记录.

长耗时实验可以由多个 Colab 会话共同完成. 因此, 最终汇总不能只记录最后
一个会话的设备环境. 该模块把每个已完成科学单元实际使用的代码锁、依赖锁、
Python、PyTorch、CUDA、GPU 和随机性身份绑定为可复算摘要, 并提供去重聚合.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any

from main.core.digest import build_stable_digest


SCIENTIFIC_UNIT_PROVENANCE_SCHEMA = "scientific_unit_runtime_provenance"
SCIENTIFIC_UNIT_PROVENANCE_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SCIENTIFIC_UNIT_PROVENANCE_AGGREGATE_FIELDS = (
    "scientific_unit_provenance_reference_count",
    "scientific_unit_provenance_record_count",
    "scientific_unit_provenance_records_digest",
    "scientific_unit_ids",
    "scientific_unit_config_digests",
    "scientific_execution_environment_digests",
    "scientific_dependency_profile_ids",
    "scientific_dependency_profile_digests",
    "scientific_complete_hash_lock_digests",
    "scientific_formal_execution_commits",
    "scientific_formal_execution_lock_digests",
    "scientific_torch_versions",
    "scientific_torch_cuda_versions",
    "scientific_execution_device_names",
    "scientific_cuda_device_names",
    "scientific_random_identity_digests_random",
    "scientific_unit_provenance_ready",
)


def _required_text(value: Any, field_name: str) -> str:
    """解析必需非空文本, 让来源 schema 的错误集中在单一边界."""

    resolved = str(value or "").strip()
    if not resolved:
        raise ValueError(f"科学完成单元来源缺少 {field_name}")
    return resolved


def _required_sha256(value: Any, field_name: str) -> str:
    """解析必需 SHA-256 字段."""

    resolved = _required_text(value, field_name)
    if not _SHA256_PATTERN.fullmatch(resolved):
        raise ValueError(f"科学完成单元来源的 {field_name} 不是 SHA-256")
    return resolved


def _random_identity(random_identity_random: Mapping[str, Any]) -> dict[str, Any]:
    """规范化随机性身份并执行字段命名治理."""

    resolved = dict(random_identity_random)
    if not resolved:
        raise ValueError("科学完成单元必须声明随机性身份")
    invalid_names = sorted(
        name
        for name in resolved
        if not (name.endswith("_random") or name.endswith("_digest_random"))
    )
    if invalid_names:
        raise ValueError(
            "随机性身份字段必须以 _random 或 _digest_random 结尾: "
            + ",".join(invalid_names)
        )
    return resolved


def _active_cuda_identity(torch_module: Any, execution_device_name: str) -> dict[str, Any]:
    """读取本次张量执行所绑定的实际 CUDA 设备身份."""

    if not execution_device_name.startswith("cuda"):
        raise ValueError("正式科学完成单元必须在 CUDA 设备执行")
    if not bool(torch_module.cuda.is_available()):
        raise RuntimeError("正式科学完成单元声明 CUDA, 但 PyTorch 未发现 CUDA")
    if ":" in execution_device_name:
        device_index = int(execution_device_name.rsplit(":", 1)[1])
    else:
        device_index = int(torch_module.cuda.current_device())
    device_count = int(torch_module.cuda.device_count())
    if device_index < 0 or device_index >= device_count:
        raise RuntimeError("科学完成单元的 CUDA 设备索引超出可见设备范围")
    capability = torch_module.cuda.get_device_capability(device_index)
    return {
        "execution_device_name": execution_device_name,
        "cuda_device_index": device_index,
        "cuda_device_name": str(torch_module.cuda.get_device_name(device_index)),
        "cuda_device_capability": [int(capability[0]), int(capability[1])],
    }


def build_scientific_unit_provenance(
    *,
    scientific_unit_id: str,
    scientific_unit_config_digest: str,
    runtime_environment: Mapping[str, Any],
    execution_device_name: str,
    torch_module: Any,
    random_identity_random: Mapping[str, Any],
) -> dict[str, Any]:
    """绑定一个实际完成单元的环境、配置和随机性身份.

    调用方应在科学计算完成的同一进程中调用该函数. 这样设备字段来自真正执行
    tensor 的 PyTorch runtime, 而不是由后续 CPU 汇总会话推测.
    """

    environment = dict(runtime_environment)
    if environment.get("dependency_environment_ready") is not True:
        raise RuntimeError("科学完成单元的依赖环境未通过完整锁门禁")
    if environment.get("formal_execution_lock_ready") is not True:
        raise RuntimeError("科学完成单元缺少正式代码执行锁")
    if environment.get("isolated_scientific_context_ready") is not True:
        raise RuntimeError("科学完成单元未在受治理隔离解释器中执行")

    package_versions = environment.get("package_versions")
    if not isinstance(package_versions, Mapping):
        raise TypeError("科学完成单元缺少依赖版本记录")
    isolated_context = environment.get("isolated_scientific_context")
    if not isinstance(isolated_context, Mapping):
        raise TypeError("科学完成单元缺少隔离解释器记录")
    formal_execution_commit = _required_text(
        environment.get("formal_execution_commit"),
        "formal_execution_commit",
    )
    if not _COMMIT_PATTERN.fullmatch(formal_execution_commit):
        raise ValueError("科学完成单元代码锁不是40位 Git commit")

    torch_version = _required_text(package_versions.get("torch"), "torch_version")
    actual_torch_version = _required_text(
        getattr(torch_module, "__version__", ""),
        "actual_torch_version",
    )
    if torch_version != actual_torch_version:
        raise RuntimeError("依赖报告与实际 PyTorch 版本不一致")
    torch_cuda_version = _required_text(
        getattr(getattr(torch_module, "version", None), "cuda", ""),
        "torch_cuda_version",
    )
    if str(environment.get("cuda_version") or "") != torch_cuda_version:
        raise RuntimeError("依赖报告与实际 PyTorch CUDA build 不一致")

    device_identity = _active_cuda_identity(torch_module, execution_device_name)
    reported_gpu_name = _required_text(environment.get("gpu_name"), "gpu_name")
    if device_identity["cuda_device_name"] != reported_gpu_name:
        raise RuntimeError("依赖报告与实际 CUDA 设备名称不一致")

    execution_environment = {
        "dependency_profile_id": _required_text(
            environment.get("dependency_profile_id"),
            "dependency_profile_id",
        ),
        "dependency_profile_digest": _required_sha256(
            environment.get("dependency_profile_digest"),
            "dependency_profile_digest",
        ),
        "direct_requirements_digest": _required_sha256(
            environment.get("direct_requirements_digest"),
            "direct_requirements_digest",
        ),
        "complete_hash_lock_digest": _required_sha256(
            environment.get("complete_hash_lock_digest"),
            "complete_hash_lock_digest",
        ),
        "formal_execution_commit": formal_execution_commit,
        "formal_execution_lock_digest": _required_sha256(
            environment.get("formal_execution_lock_digest"),
            "formal_execution_lock_digest",
        ),
        "dependency_environment_report_digest": _required_sha256(
            isolated_context.get("dependency_environment_report_actual_digest"),
            "dependency_environment_report_digest",
        ),
        "python_version": _required_text(
            environment.get("python_version"),
            "python_version",
        ),
        "python_executable_sha256": _required_sha256(
            isolated_context.get("current_python_executable_sha256"),
            "python_executable_sha256",
        ),
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda_version,
        "cuda_available": True,
        "visible_cuda_device_count": int(environment.get("device_count", 0)),
        **device_identity,
    }
    if execution_environment["visible_cuda_device_count"] <= 0:
        raise RuntimeError("科学完成单元没有可见 CUDA 设备")
    environment_digest = build_stable_digest(execution_environment)
    resolved_random_identity = _random_identity(random_identity_random)
    random_identity_digest = build_stable_digest(resolved_random_identity)
    payload = {
        "report_schema": SCIENTIFIC_UNIT_PROVENANCE_SCHEMA,
        "schema_version": SCIENTIFIC_UNIT_PROVENANCE_SCHEMA_VERSION,
        "scientific_unit_id": _required_text(scientific_unit_id, "scientific_unit_id"),
        "scientific_unit_config_digest": _required_sha256(
            scientific_unit_config_digest,
            "scientific_unit_config_digest",
        ),
        "scientific_execution_environment": execution_environment,
        "scientific_execution_environment_digest": environment_digest,
        "scientific_random_identity_random": resolved_random_identity,
        "scientific_random_identity_digest_random": random_identity_digest,
        "supports_paper_claim": False,
    }
    payload["scientific_unit_provenance_digest"] = build_stable_digest(payload)
    return validate_scientific_unit_provenance(payload)


def validate_scientific_unit_provenance(
    record: Mapping[str, Any],
    *,
    expected_unit_id: str | None = None,
    expected_config_digest: str | None = None,
) -> dict[str, Any]:
    """验证持久化完成单元来源记录的自摘要和关键科学身份."""

    payload = dict(record)
    if payload.get("report_schema") != SCIENTIFIC_UNIT_PROVENANCE_SCHEMA:
        raise ValueError("科学完成单元来源 schema 不匹配")
    if payload.get("schema_version") != SCIENTIFIC_UNIT_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("科学完成单元来源 schema_version 不匹配")
    if payload.get("supports_paper_claim") is not False:
        raise ValueError("科学完成单元来源记录自身不得直接支持论文主张")
    unit_id = _required_text(payload.get("scientific_unit_id"), "scientific_unit_id")
    config_digest = _required_sha256(
        payload.get("scientific_unit_config_digest"),
        "scientific_unit_config_digest",
    )
    if expected_unit_id is not None and unit_id != expected_unit_id:
        raise ValueError("科学完成单元标识与调用方预期不一致")
    if expected_config_digest is not None and config_digest != expected_config_digest:
        raise ValueError("科学完成单元配置摘要与调用方预期不一致")

    execution_environment = payload.get("scientific_execution_environment")
    if not isinstance(execution_environment, Mapping):
        raise TypeError("科学完成单元来源缺少执行环境")
    environment = dict(execution_environment)
    for field_name in (
        "dependency_profile_digest",
        "direct_requirements_digest",
        "complete_hash_lock_digest",
        "formal_execution_lock_digest",
        "dependency_environment_report_digest",
        "python_executable_sha256",
    ):
        _required_sha256(environment.get(field_name), field_name)
    commit = _required_text(
        environment.get("formal_execution_commit"),
        "formal_execution_commit",
    )
    if not _COMMIT_PATTERN.fullmatch(commit):
        raise ValueError("科学完成单元来源中的代码锁无效")
    for field_name in (
        "dependency_profile_id",
        "python_version",
        "torch_version",
        "torch_cuda_version",
        "execution_device_name",
        "cuda_device_name",
    ):
        _required_text(environment.get(field_name), field_name)
    if environment.get("cuda_available") is not True:
        raise ValueError("科学完成单元来源未声明 CUDA 可用")
    if not str(environment.get("execution_device_name")).startswith("cuda"):
        raise ValueError("科学完成单元来源不是 CUDA 执行设备")
    visible_device_count = environment.get("visible_cuda_device_count")
    if (
        isinstance(visible_device_count, bool)
        or not isinstance(visible_device_count, int)
        or visible_device_count <= 0
    ):
        raise ValueError("科学完成单元来源的可见 CUDA 设备数无效")
    device_index = environment.get("cuda_device_index")
    if (
        isinstance(device_index, bool)
        or not isinstance(device_index, int)
        or device_index < 0
        or device_index >= visible_device_count
    ):
        raise ValueError("科学完成单元来源的 CUDA 设备索引无效")
    capability = environment.get("cuda_device_capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in capability)
    ):
        raise ValueError("科学完成单元来源的 CUDA capability 无效")
    environment_digest = _required_sha256(
        payload.get("scientific_execution_environment_digest"),
        "scientific_execution_environment_digest",
    )
    if build_stable_digest(environment) != environment_digest:
        raise ValueError("科学完成单元执行环境摘要不匹配")

    random_identity = payload.get("scientific_random_identity_random")
    if not isinstance(random_identity, Mapping):
        raise TypeError("科学完成单元来源缺少随机性身份")
    resolved_random_identity = _random_identity(random_identity)
    random_digest = _required_sha256(
        payload.get("scientific_random_identity_digest_random"),
        "scientific_random_identity_digest_random",
    )
    if build_stable_digest(resolved_random_identity) != random_digest:
        raise ValueError("科学完成单元随机性身份摘要不匹配")

    provenance_digest = _required_sha256(
        payload.get("scientific_unit_provenance_digest"),
        "scientific_unit_provenance_digest",
    )
    digest_payload = {
        key: value
        for key, value in payload.items()
        if key != "scientific_unit_provenance_digest"
    }
    if build_stable_digest(digest_payload) != provenance_digest:
        raise ValueError("科学完成单元来源自摘要不匹配")
    return payload


def aggregate_scientific_unit_provenance(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_reference_count: int,
) -> dict[str, Any]:
    """按真实完成单元去重并汇总跨会话运行身份集合."""

    references = tuple(validate_scientific_unit_provenance(record) for record in records)
    by_unit_id: dict[str, dict[str, Any]] = {}
    for record in references:
        unit_id = str(record["scientific_unit_id"])
        existing = by_unit_id.get(unit_id)
        if existing is not None and existing != record:
            raise ValueError("同一科学完成单元包含冲突来源记录")
        by_unit_id[unit_id] = record
    unique_records = [by_unit_id[unit_id] for unit_id in sorted(by_unit_id)]
    environments = [record["scientific_execution_environment"] for record in unique_records]

    def values(field_name: str) -> list[Any]:
        return sorted({environment[field_name] for environment in environments})

    reference_count = len(references)
    return {
        "scientific_unit_provenance_reference_count": reference_count,
        "scientific_unit_provenance_record_count": len(unique_records),
        "scientific_unit_provenance_records_digest": build_stable_digest(unique_records),
        "scientific_unit_ids": [
            record["scientific_unit_id"] for record in unique_records
        ],
        "scientific_unit_config_digests": sorted(
            {record["scientific_unit_config_digest"] for record in unique_records}
        ),
        "scientific_execution_environment_digests": sorted(
            {
                record["scientific_execution_environment_digest"]
                for record in unique_records
            }
        ),
        "scientific_dependency_profile_ids": values("dependency_profile_id"),
        "scientific_dependency_profile_digests": values("dependency_profile_digest"),
        "scientific_complete_hash_lock_digests": values("complete_hash_lock_digest"),
        "scientific_formal_execution_commits": values("formal_execution_commit"),
        "scientific_formal_execution_lock_digests": values(
            "formal_execution_lock_digest"
        ),
        "scientific_torch_versions": values("torch_version"),
        "scientific_torch_cuda_versions": values("torch_cuda_version"),
        "scientific_execution_device_names": values("execution_device_name"),
        "scientific_cuda_device_names": values("cuda_device_name"),
        "scientific_random_identity_digests_random": sorted(
            {
                record["scientific_random_identity_digest_random"]
                for record in unique_records
            }
        ),
        "scientific_unit_provenance_ready": (
            expected_reference_count > 0
            and reference_count == expected_reference_count
            and bool(unique_records)
        ),
    }
