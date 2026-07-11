"""解析隔离依赖 profile 并执行正式运行前的哈希锁门禁.

直接依赖文件只表达人工审计后提交的精确输入, 不等同于已经解析完成的 wheel
闭包. 完整闭包必须在登记的 Linux x86_64 profile 环境中物化为逐项带
SHA-256 的锁文件. 父编排 profile 是 CPU 环境, 五个科学 profile 是 CUDA
环境. 本模块在锁文件缺失时保留 profile 查询能力, 但正式 readiness 必须关闭.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata as importlib_metadata
import json
from pathlib import Path, PurePosixPath
import platform
import re
from typing import Any


DEPENDENCY_PROFILE_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "dependency_profile_registry.json"
)

WORKFLOW_ORCHESTRATOR_PROFILE_ID = "workflow_orchestrator"
REQUIRED_DEPENDENCY_PROFILE_NAMES = (
    WORKFLOW_ORCHESTRATOR_PROFILE_ID,
    "sd35_method_runtime_gpu",
    "t2smark_sd35_gpu",
    "tree_ring_official_py39_cu117",
    "gaussian_shading_official_py38_cu117",
    "shallow_diffuse_official_py39_cu117",
)

_REGISTRY_SCHEMA = "isolated_dependency_profile_registry"
_REGISTRY_SCHEMA_VERSION = 1
_DIRECT_INPUT_CONTRACT = {
    "artifact_role": "committed_exact_direct_dependency_input",
    "requirement_format": "name==version",
    "exact_operator": "==",
}
_COMPLETE_HASH_LOCK_CONTRACT = {
    "artifact_role": "target_runtime_complete_wheel_hash_lock",
    "requirement_format": "name==version --hash=sha256:<digest>",
    "materialization_environment": "registered_linux_x86_64_profile",
    "repository_commit_required": True,
    "formal_readiness_requires_valid_lock": True,
}

_PROFILE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION_PATTERN = re.compile(r"^[0-9][A-Za-z0-9.!+_-]*$")
_PYTHON_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_CUDA_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+$")
_EXACT_REQUIREMENT_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[0-9][A-Za-z0-9.!+_-]*)$"
)
_HASH_TOKEN_PATTERN = re.compile(r"^--hash=sha256:(?P<digest>[0-9a-f]{64})$")
_CUDA_LOCAL_VERSION_PATTERN = re.compile(r"\+cu(?P<tag>[0-9]+)$")
_SUPPORTED_PYTORCH_PAIRS = frozenset(
    {
        ("2.11.0+cu128", "0.26.0+cu128"),
        ("2.5.0+cu124", "0.20.0+cu124"),
        ("1.13.0+cu117", "0.14.0+cu117"),
    }
)


@dataclass(frozen=True)
class ExactDependency:
    """表示一个只使用 ``==`` 固定的直接依赖."""

    package_name: str
    normalized_name: str
    version: str

    @property
    def specification(self) -> str:
        """返回标准化前仍可直接交给 pip 的精确规格."""

        return f"{self.package_name}=={self.version}"


@dataclass(frozen=True)
class LockedDependency:
    """表示完整锁中的一个精确依赖及其可接受 wheel 摘要."""

    dependency: ExactDependency
    sha256_digests: tuple[str, ...]


@dataclass(frozen=True)
class DependencyProfile:
    """表示隔离环境身份、直接输入和完整哈希锁状态."""

    profile_name: str
    execution_role: str
    python_implementation: str
    python_version: str
    operating_system: str
    machine: str
    accelerator_runtime: str
    cuda_version: str | None
    torch_version: str | None
    torchvision_version: str | None
    pytorch_index_url: str | None
    direct_requirements_path: str
    direct_requirements: tuple[str, ...]
    direct_requirements_digest: str
    complete_hash_lock_path: str
    complete_hash_lock_present: bool
    complete_hash_lock_digest: str | None
    complete_hash_lock_dependency_count: int
    locked_requirements: tuple[str, ...]
    profile_digest: str
    formal_ready: bool
    readiness_blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """返回不含机器绝对路径和时间戳的稳定摘要."""

        return {
            "profile_name": self.profile_name,
            "execution_role": self.execution_role,
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "operating_system": self.operating_system,
            "machine": self.machine,
            "accelerator_runtime": self.accelerator_runtime,
            "cuda_version": self.cuda_version,
            "torch_version": self.torch_version,
            "torchvision_version": self.torchvision_version,
            "pytorch_index_url": self.pytorch_index_url,
            "direct_requirements_path": self.direct_requirements_path,
            "direct_requirements": list(self.direct_requirements),
            "direct_requirements_digest": self.direct_requirements_digest,
            "complete_hash_lock_path": self.complete_hash_lock_path,
            "complete_hash_lock_present": self.complete_hash_lock_present,
            "complete_hash_lock_digest": self.complete_hash_lock_digest,
            "complete_hash_lock_dependency_count": self.complete_hash_lock_dependency_count,
            "locked_requirements": list(self.locked_requirements),
            "profile_digest": self.profile_digest,
            "formal_ready": self.formal_ready,
            "readiness_blockers": list(self.readiness_blockers),
        }


def _stable_digest(payload: Any) -> str:
    """对 JSON 兼容数据计算跨平台稳定摘要."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_package_name(package_name: str) -> str:
    """按 Python 包名称比较规则统一连字符、下划线和点号."""

    return re.sub(r"[-_.]+", "-", package_name).lower()


def parse_exact_requirement_spec(
    specification: str,
    *,
    field_name: str = "requirement",
) -> ExactDependency:
    """解析单个精确直接依赖, 拒绝范围、别名和未版本化输入."""

    if not isinstance(specification, str) or specification != specification.strip():
        raise ValueError(f"{field_name} 必须是不带首尾空白的字符串")
    match = _EXACT_REQUIREMENT_PATTERN.fullmatch(specification)
    if match is None:
        raise ValueError(f"{field_name} 必须使用 name==version 精确规格")
    package_name = match.group("name")
    version = match.group("version")
    if _VERSION_PATTERN.fullmatch(version) is None or version.lower() == "latest":
        raise ValueError(f"{field_name} 必须固定可解析的精确版本")
    return ExactDependency(
        package_name=package_name,
        normalized_name=_normalize_package_name(package_name),
        version=version,
    )


def _canonical_dependency_payload(
    dependencies: tuple[ExactDependency, ...],
) -> list[dict[str, str]]:
    """生成与文件换行和条目顺序无关的直接依赖身份."""

    return [
        {"package_name": dependency.normalized_name, "version": dependency.version}
        for dependency in sorted(dependencies, key=lambda item: item.normalized_name)
    ]


def direct_requirements_digest(dependencies: tuple[ExactDependency, ...]) -> str:
    """计算精确直接依赖集合的稳定摘要."""

    return _stable_digest(_canonical_dependency_payload(dependencies))


def load_exact_direct_requirements(path: str | Path) -> tuple[ExactDependency, ...]:
    """读取直接依赖文件并集中执行精确规格校验."""

    requirements_path = Path(path)
    try:
        lines = requirements_path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"直接依赖输入不存在: {requirements_path}") from exc

    dependencies: list[ExactDependency] = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        dependencies.append(
            parse_exact_requirement_spec(
                line,
                field_name=f"{requirements_path.name}:{line_number}",
            )
        )
    if not dependencies:
        raise ValueError(f"直接依赖输入不得为空: {requirements_path}")
    normalized_names = tuple(item.normalized_name for item in dependencies)
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError(f"直接依赖输入包含重复包名: {requirements_path}")
    return tuple(dependencies)


def _logical_lock_lines(path: Path) -> tuple[str, ...]:
    """合并 pip 哈希锁中的反斜杠续行并忽略说明注释."""

    try:
        source_lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"完整哈希锁不存在: {path}") from exc

    logical_lines: list[str] = []
    fragments: list[str] = []
    for raw_line in source_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if " #" in line:
            line = line.split(" #", maxsplit=1)[0].rstrip()
        continued = line.endswith("\\")
        fragment = line[:-1].strip() if continued else line
        if fragment:
            fragments.append(fragment)
        if not continued:
            if fragments:
                logical_lines.append(" ".join(fragments))
            fragments = []
    if fragments:
        raise ValueError(f"完整哈希锁包含未结束续行: {path}")
    if not logical_lines:
        raise ValueError(f"完整哈希锁不得为空: {path}")
    return tuple(logical_lines)


def _load_complete_hash_lock(path: Path) -> tuple[LockedDependency, ...]:
    """读取完整锁, 要求每个解析后依赖都携带至少一个 SHA-256."""

    locked_dependencies: list[LockedDependency] = []
    for line_number, line in enumerate(_logical_lock_lines(path), start=1):
        tokens = line.split()
        dependency = parse_exact_requirement_spec(
            tokens[0],
            field_name=f"{path.name}:{line_number}",
        )
        if len(tokens) < 2:
            raise ValueError(f"完整哈希锁条目缺少 SHA-256: {path.name}:{line_number}")
        digests: list[str] = []
        for token in tokens[1:]:
            match = _HASH_TOKEN_PATTERN.fullmatch(token)
            if match is None:
                raise ValueError(f"完整哈希锁只允许 sha256 hash 参数: {path.name}:{line_number}")
            digests.append(match.group("digest"))
        if len(digests) != len(set(digests)):
            raise ValueError(f"完整哈希锁条目包含重复 SHA-256: {path.name}:{line_number}")
        locked_dependencies.append(
            LockedDependency(
                dependency=dependency,
                sha256_digests=tuple(sorted(digests)),
            )
        )

    normalized_names = tuple(item.dependency.normalized_name for item in locked_dependencies)
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError(f"完整哈希锁包含重复包名: {path}")
    return tuple(locked_dependencies)


def _complete_hash_lock_digest(
    locked_dependencies: tuple[LockedDependency, ...],
) -> str:
    """计算忽略排版和注释的完整哈希锁稳定摘要."""

    payload = [
        {
            "package_name": item.dependency.normalized_name,
            "version": item.dependency.version,
            "sha256_digests": list(item.sha256_digests),
        }
        for item in sorted(
            locked_dependencies,
            key=lambda record: record.dependency.normalized_name,
        )
    ]
    return _stable_digest(payload)


def _validate_repository_relative_path(
    value: Any,
    *,
    field_name: str,
    expected_path: str,
) -> str:
    """限制 profile 文件位于受治理配置目录并绑定语义文件名."""

    if not isinstance(value, str) or value != expected_path:
        raise ValueError(f"{field_name} 必须是 {expected_path}")
    pure_path = PurePosixPath(value)
    if pure_path.is_absolute() or ".." in pure_path.parts or pure_path.as_posix() != value:
        raise ValueError(f"{field_name} 必须是仓库内 POSIX 相对路径")
    return value


def _require_exact_keys(
    payload: Any,
    *,
    required_keys: frozenset[str],
    field_name: str,
) -> dict[str, Any]:
    """统一校验固定 schema 对象的字段集合."""

    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} 必须是 JSON 对象")
    actual_keys = frozenset(payload)
    if actual_keys != required_keys:
        missing = sorted(required_keys - actual_keys)
        unexpected = sorted(actual_keys - required_keys)
        raise ValueError(
            f"{field_name} 字段不匹配, missing={missing}, unexpected={unexpected}"
        )
    return payload


def _cuda_tag(cuda_version: str) -> str:
    """把点分 CUDA 版本转换为 PyTorch local version 使用的标签."""

    return "cu" + cuda_version.replace(".", "")


def _validate_pytorch_identity(
    *,
    profile_name: str,
    cuda_version: str,
    torch_version: str,
    torchvision_version: str,
    pytorch_index_url: str,
    direct_dependencies: tuple[ExactDependency, ...],
) -> None:
    """验证 CUDA、索引地址、torch pair 与直接输入完全一致."""

    pair = (torch_version, torchvision_version)
    if pair not in _SUPPORTED_PYTORCH_PAIRS:
        raise ValueError(f"未登记的 torch/torchvision 精确组合: {profile_name}")
    expected_cuda_tag = _cuda_tag(cuda_version)
    for field_name, version in (("torch_version", torch_version), ("torchvision_version", torchvision_version)):
        match = _CUDA_LOCAL_VERSION_PATTERN.search(version)
        if match is None or f"cu{match.group('tag')}" != expected_cuda_tag:
            raise ValueError(f"{profile_name}.{field_name} 与 CUDA 版本不一致")
    if pytorch_index_url != f"https://download.pytorch.org/whl/{expected_cuda_tag}":
        raise ValueError(f"{profile_name}.pytorch.index_url 与 CUDA 版本不一致")

    direct_by_name = {item.normalized_name: item.version for item in direct_dependencies}
    if direct_by_name.get("torch") != torch_version:
        raise ValueError(f"{profile_name} 直接输入中的 torch 与 profile 不一致")
    if direct_by_name.get("torchvision") != torchvision_version:
        raise ValueError(f"{profile_name} 直接输入中的 torchvision 与 profile 不一致")


def _validate_lock_covers_direct_inputs(
    *,
    profile_name: str,
    direct_dependencies: tuple[ExactDependency, ...],
    locked_dependencies: tuple[LockedDependency, ...],
) -> None:
    """确认完整锁包含所有直接输入且没有改写其版本."""

    locked_by_name = {
        item.dependency.normalized_name: item.dependency.version
        for item in locked_dependencies
    }
    for dependency in direct_dependencies:
        locked_version = locked_by_name.get(dependency.normalized_name)
        if locked_version is None:
            raise ValueError(
                f"完整哈希锁缺少直接依赖 {dependency.normalized_name}: {profile_name}"
            )
        if locked_version != dependency.version:
            raise ValueError(
                f"完整哈希锁改写直接依赖版本 {dependency.normalized_name}: {profile_name}"
            )


def _parse_profile(
    profile_name: str,
    payload: Any,
    *,
    repository_root: Path,
) -> DependencyProfile:
    """把单条 profile 记录解析为经过统一门禁的不可变对象."""

    if _PROFILE_NAME_PATTERN.fullmatch(profile_name) is None:
        raise ValueError(f"依赖 profile 名称不符合 snake_case: {profile_name}")
    record = _require_exact_keys(
        payload,
        required_keys=frozenset(
            {
                "execution_role",
                "python",
                "platform",
                "accelerator",
                "pytorch",
                "direct_requirements_path",
                "direct_requirements_digest",
                "complete_hash_lock_path",
            }
        ),
        field_name=profile_name,
    )

    execution_role = record["execution_role"]
    if not isinstance(execution_role, str) or _PROFILE_NAME_PATTERN.fullmatch(execution_role) is None:
        raise ValueError(f"{profile_name}.execution_role 必须是语义明确的 snake_case")

    python_record = _require_exact_keys(
        record["python"],
        required_keys=frozenset({"implementation", "version"}),
        field_name=f"{profile_name}.python",
    )
    if python_record["implementation"] != "CPython":
        raise ValueError(f"{profile_name}.python.implementation 必须是 CPython")
    python_version = python_record["version"]
    if not isinstance(python_version, str) or _PYTHON_VERSION_PATTERN.fullmatch(python_version) is None:
        raise ValueError(f"{profile_name}.python.version 必须精确到 patch 版本")

    platform_record = _require_exact_keys(
        record["platform"],
        required_keys=frozenset({"operating_system", "machine"}),
        field_name=f"{profile_name}.platform",
    )
    if platform_record != {"operating_system": "linux", "machine": "x86_64"}:
        raise ValueError(f"{profile_name}.platform 必须精确登记为 linux/x86_64")

    accelerator_record = _require_exact_keys(
        record["accelerator"],
        required_keys=frozenset({"runtime", "cuda_version"}),
        field_name=f"{profile_name}.accelerator",
    )
    accelerator_runtime = accelerator_record["runtime"]
    if accelerator_runtime not in {"cpu", "cuda"}:
        raise ValueError(f"{profile_name}.accelerator.runtime 必须是 cpu 或 cuda")
    if profile_name == WORKFLOW_ORCHESTRATOR_PROFILE_ID:
        if accelerator_runtime != "cpu":
            raise ValueError("workflow_orchestrator 必须是 CPU 编排 profile")
    elif accelerator_runtime != "cuda":
        raise ValueError(f"科学 profile 必须使用 cuda: {profile_name}")
    cuda_version = accelerator_record["cuda_version"]
    if accelerator_runtime == "cuda":
        if (
            not isinstance(cuda_version, str)
            or _CUDA_VERSION_PATTERN.fullmatch(cuda_version) is None
        ):
            raise ValueError(f"{profile_name}.accelerator.cuda_version 必须是精确 CUDA 版本")
    elif cuda_version is not None:
        raise ValueError("CPU 编排 profile 的 cuda_version 必须为 null")

    pytorch_record = _require_exact_keys(
        record["pytorch"],
        required_keys=frozenset({"torch_version", "torchvision_version", "index_url"}),
        field_name=f"{profile_name}.pytorch",
    )
    torch_version = pytorch_record["torch_version"]
    torchvision_version = pytorch_record["torchvision_version"]
    pytorch_index_url = pytorch_record["index_url"]
    if accelerator_runtime == "cuda":
        if not all(
            isinstance(value, str) and value
            for value in (torch_version, torchvision_version, pytorch_index_url)
        ):
            raise ValueError(f"CUDA profile 必须精确登记 PyTorch identity: {profile_name}")
    elif any(
        value is not None
        for value in (torch_version, torchvision_version, pytorch_index_url)
    ):
        raise ValueError("CPU 编排 profile 的 PyTorch identity 必须全部为 null")

    direct_path_value = _validate_repository_relative_path(
        record["direct_requirements_path"],
        field_name=f"{profile_name}.direct_requirements_path",
        expected_path=f"configs/dependency_profiles/{profile_name}_direct.txt",
    )
    lock_path_value = _validate_repository_relative_path(
        record["complete_hash_lock_path"],
        field_name=f"{profile_name}.complete_hash_lock_path",
        expected_path=f"configs/dependency_profiles/{profile_name}_lock.txt",
    )
    direct_path = repository_root / PurePosixPath(direct_path_value)
    direct_dependencies = load_exact_direct_requirements(direct_path)
    computed_direct_digest = direct_requirements_digest(direct_dependencies)
    registered_direct_digest = record["direct_requirements_digest"]
    if registered_direct_digest != computed_direct_digest:
        raise ValueError(f"{profile_name}.direct_requirements_digest 与直接输入不一致")

    if accelerator_runtime == "cuda":
        assert isinstance(cuda_version, str)
        assert isinstance(torch_version, str)
        assert isinstance(torchvision_version, str)
        assert isinstance(pytorch_index_url, str)
        _validate_pytorch_identity(
            profile_name=profile_name,
            cuda_version=cuda_version,
            torch_version=torch_version,
            torchvision_version=torchvision_version,
            pytorch_index_url=pytorch_index_url,
            direct_dependencies=direct_dependencies,
        )
    else:
        direct_names = {item.normalized_name for item in direct_dependencies}
        forbidden_gpu_dependencies = sorted(direct_names & {"torch", "torchvision"})
        if forbidden_gpu_dependencies:
            raise ValueError(
                "CPU 编排 profile 不得登记 GPU PyTorch 直接依赖: "
                + ",".join(forbidden_gpu_dependencies)
            )

    lock_path = repository_root / PurePosixPath(lock_path_value)
    if lock_path.exists() and not lock_path.is_file():
        raise ValueError(f"完整哈希锁路径不是文件: {lock_path_value}")
    if lock_path.is_file():
        locked_dependencies = _load_complete_hash_lock(lock_path)
        _validate_lock_covers_direct_inputs(
            profile_name=profile_name,
            direct_dependencies=direct_dependencies,
            locked_dependencies=locked_dependencies,
        )
        lock_digest: str | None = _complete_hash_lock_digest(locked_dependencies)
        lock_dependency_count = len(locked_dependencies)
        locked_requirements = tuple(
            f"{item.dependency.normalized_name}=={item.dependency.version}"
            for item in sorted(
                locked_dependencies,
                key=lambda record: record.dependency.normalized_name,
            )
        )
        lock_present = True
        formal_ready = True
        blockers: tuple[str, ...] = ()
    else:
        lock_digest = None
        lock_dependency_count = 0
        locked_requirements = ()
        lock_present = False
        formal_ready = False
        blockers = ("complete_hash_lock_missing",)

    profile_identity = {
        "profile_name": profile_name,
        "execution_role": execution_role,
        "python": dict(python_record),
        "platform": dict(platform_record),
        "accelerator": dict(accelerator_record),
        "pytorch": dict(pytorch_record),
        "direct_requirements_path": direct_path_value,
        "direct_requirements_digest": computed_direct_digest,
        "complete_hash_lock_path": lock_path_value,
    }
    return DependencyProfile(
        profile_name=profile_name,
        execution_role=execution_role,
        python_implementation=str(python_record["implementation"]),
        python_version=python_version,
        operating_system=str(platform_record["operating_system"]),
        machine=str(platform_record["machine"]),
        accelerator_runtime=str(accelerator_runtime),
        cuda_version=cuda_version,
        torch_version=torch_version,
        torchvision_version=torchvision_version,
        pytorch_index_url=pytorch_index_url,
        direct_requirements_path=direct_path_value,
        direct_requirements=tuple(item.specification for item in direct_dependencies),
        direct_requirements_digest=computed_direct_digest,
        complete_hash_lock_path=lock_path_value,
        complete_hash_lock_present=lock_present,
        complete_hash_lock_digest=lock_digest,
        complete_hash_lock_dependency_count=lock_dependency_count,
        locked_requirements=locked_requirements,
        profile_digest=_stable_digest(profile_identity),
        formal_ready=formal_ready,
        readiness_blockers=blockers,
    )


def load_dependency_profile_registry(
    path: str | Path = DEPENDENCY_PROFILE_REGISTRY_PATH,
) -> dict[str, DependencyProfile]:
    """加载六个隔离 profile, 同时校验直接输入和可选完整哈希锁."""

    registry_path = Path(path)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"依赖 profile 登记表不存在: {registry_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"依赖 profile 登记表不是有效 JSON: {registry_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("依赖 profile 登记表根节点必须是 JSON 对象")
    if payload.get("registry_schema") != _REGISTRY_SCHEMA:
        raise ValueError("依赖 profile 登记表 registry_schema 不匹配")
    if payload.get("schema_version") != _REGISTRY_SCHEMA_VERSION:
        raise ValueError("依赖 profile 登记表 schema_version 必须为 1")
    if payload.get("direct_dependency_input_contract") != _DIRECT_INPUT_CONTRACT:
        raise ValueError("依赖 profile 登记表的直接输入契约不匹配")
    if payload.get("complete_hash_lock_contract") != _COMPLETE_HASH_LOCK_CONTRACT:
        raise ValueError("依赖 profile 登记表的完整哈希锁契约不匹配")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("依赖 profile 登记表必须包含 profiles 对象")
    if frozenset(profiles) != frozenset(REQUIRED_DEPENDENCY_PROFILE_NAMES):
        raise ValueError("依赖 profile 登记表必须且只能包含六个受治理 profile")

    repository_root = registry_path.resolve().parent.parent
    parsed = {
        profile_name: _parse_profile(
            profile_name,
            profiles[profile_name],
            repository_root=repository_root,
        )
        for profile_name in REQUIRED_DEPENDENCY_PROFILE_NAMES
    }
    direct_paths = tuple(item.direct_requirements_path for item in parsed.values())
    lock_paths = tuple(item.complete_hash_lock_path for item in parsed.values())
    if len(direct_paths) != len(set(direct_paths)) or len(lock_paths) != len(set(lock_paths)):
        raise ValueError("依赖 profile 登记表包含重复的输入或锁文件路径")
    return parsed


def get_dependency_profile(
    profile_name: str,
    path: str | Path = DEPENDENCY_PROFILE_REGISTRY_PATH,
) -> DependencyProfile:
    """按语义名称查询一条经过完整校验的依赖 profile."""

    profiles = load_dependency_profile_registry(path)
    try:
        return profiles[profile_name]
    except KeyError as exc:
        raise KeyError(f"依赖 profile 未登记: {profile_name}") from exc


def build_dependency_profile_summary(
    profile_name: str,
    path: str | Path = DEPENDENCY_PROFILE_REGISTRY_PATH,
) -> dict[str, Any]:
    """生成可直接进入环境报告的稳定 profile 摘要."""

    summary = get_dependency_profile(profile_name, path).to_dict()
    summary["summary_digest"] = _stable_digest(summary)
    return summary


def require_dependency_profile_ready(
    profile_name: str,
    path: str | Path = DEPENDENCY_PROFILE_REGISTRY_PATH,
) -> DependencyProfile:
    """要求 profile 已提交有效完整哈希锁, 否则阻断正式运行."""

    profile = get_dependency_profile(profile_name, path)
    if not profile.formal_ready:
        blockers = ",".join(profile.readiness_blockers)
        raise RuntimeError(f"依赖 profile 尚未达到正式 readiness: {profile_name}; blockers={blockers}")
    return profile


def _installed_distribution_version(package_name: str) -> str | None:
    """读取已安装 distribution 版本, 缺失时返回 ``None``."""

    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _import_torch_module() -> Any | None:
    """按需导入 torch, 使 registry 查询本身不依赖 GPU 包."""

    try:
        import torch
    except Exception:
        return None
    return torch


def _normalized_machine_name(machine: str) -> str:
    """统一常见机器架构别名, 同时保留未知架构供严格比较."""

    normalized = machine.strip().lower()
    if normalized in {"amd64", "x64", "x86-64"}:
        return "x86_64"
    if normalized in {"arm64", "armv8"}:
        return "aarch64"
    return normalized


def _torch_runtime_observation(torch_module: Any | None) -> dict[str, Any]:
    """读取 torch 模块版本、编译 CUDA 版本和 CUDA 可用状态."""

    if torch_module is None:
        return {
            "torch_module_available": False,
            "torch_module_version": None,
            "torch_cuda_version": None,
            "cuda_available": False,
        }
    torch_version = getattr(torch_module, "__version__", None)
    version_namespace = getattr(torch_module, "version", None)
    cuda_version = getattr(version_namespace, "cuda", None)
    cuda_namespace = getattr(torch_module, "cuda", None)
    try:
        cuda_available = bool(cuda_namespace is not None and cuda_namespace.is_available())
    except Exception:
        cuda_available = False
    return {
        "torch_module_available": True,
        "torch_module_version": None if torch_version is None else str(torch_version),
        "torch_cuda_version": None if cuda_version is None else str(cuda_version),
        "cuda_available": cuda_available,
    }


def inspect_dependency_profile_environment(
    profile_name: str,
    *,
    torch_module: Any | None = None,
    path: str | Path = DEPENDENCY_PROFILE_REGISTRY_PATH,
) -> dict[str, Any]:
    """核对当前解释器与依赖 profile, 返回稳定且可复用的 readiness 报告.

    该函数只读取解释器、distribution metadata 和 torch 设备状态, 不执行安装或
    网络访问. ``pass`` 同时要求完整哈希锁已就绪、运行平台精确匹配且完整锁中
    的全部直接与传递依赖版本一致. 五个科学 profile 还要求 CUDA 与 PyTorch
    identity 精确匹配; CPU 父编排 profile 不导入 torch, 也不执行 CUDA 门禁.
    """

    profile_record = get_dependency_profile(profile_name, path)
    if profile_record.accelerator_runtime == "cuda":
        resolved_torch_module = (
            torch_module if torch_module is not None else _import_torch_module()
        )
        torch_observation = _torch_runtime_observation(resolved_torch_module)
    else:
        torch_observation = {
            "torch_module_available": None,
            "torch_module_version": None,
            "torch_cuda_version": None,
            "cuda_available": None,
        }

    expected_dependencies: dict[str, str] = {}
    distribution_query_names: dict[str, str] = {}
    for specification in profile_record.direct_requirements:
        dependency = parse_exact_requirement_spec(specification)
        expected_dependencies[dependency.normalized_name] = dependency.version
        distribution_query_names[dependency.normalized_name] = dependency.package_name
    expected_locked_dependencies: dict[str, str] = {}
    for specification in profile_record.locked_requirements:
        dependency = parse_exact_requirement_spec(specification)
        expected_locked_dependencies[dependency.normalized_name] = dependency.version
        distribution_query_names.setdefault(
            dependency.normalized_name,
            dependency.package_name,
        )
    installed_version_cache = {
        normalized_name: _installed_distribution_version(package_name)
        for normalized_name, package_name in sorted(distribution_query_names.items())
    }
    installed_dependencies = {
        package_name: installed_version_cache[package_name]
        for package_name in expected_dependencies
    }
    installed_locked_dependencies = {
        package_name: installed_version_cache[package_name]
        for package_name in expected_locked_dependencies
    }

    expected_environment = {
        "python_implementation": profile_record.python_implementation,
        "python_version": profile_record.python_version,
        "operating_system": profile_record.operating_system,
        "machine": profile_record.machine,
        "accelerator_runtime": profile_record.accelerator_runtime,
        "cuda_version": profile_record.cuda_version,
        "torch_version": profile_record.torch_version,
        "torchvision_version": profile_record.torchvision_version,
        "direct_dependencies": dict(sorted(expected_dependencies.items())),
        "locked_dependencies": dict(sorted(expected_locked_dependencies.items())),
    }
    observed_environment = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "operating_system": platform.system().strip().lower(),
        "machine": _normalized_machine_name(platform.machine()),
        **torch_observation,
        "direct_dependencies": dict(sorted(installed_dependencies.items())),
        "locked_dependencies": dict(sorted(installed_locked_dependencies.items())),
    }

    mismatches: list[str] = []
    if observed_environment["python_implementation"] != profile_record.python_implementation:
        mismatches.append("python_implementation_mismatch")
    if observed_environment["python_version"] != profile_record.python_version:
        mismatches.append("python_version_mismatch")
    if observed_environment["operating_system"] != profile_record.operating_system:
        mismatches.append("operating_system_mismatch")
    if observed_environment["machine"] != profile_record.machine:
        mismatches.append("machine_mismatch")
    if profile_record.accelerator_runtime == "cuda":
        if not torch_observation["torch_module_available"]:
            mismatches.append("torch_module_unavailable")
        elif torch_observation["torch_module_version"] != profile_record.torch_version:
            mismatches.append("torch_module_version_mismatch")
        if not torch_observation["cuda_available"]:
            mismatches.append("cuda_unavailable")
        if torch_observation["torch_cuda_version"] != profile_record.cuda_version:
            mismatches.append("torch_cuda_version_mismatch")

    if profile_record.complete_hash_lock_present:
        for package_name, expected_version in sorted(expected_locked_dependencies.items()):
            installed_version = installed_locked_dependencies[package_name]
            if installed_version is None:
                mismatches.append(f"locked_dependency_missing:{package_name}")
            elif installed_version != expected_version:
                mismatches.append(f"locked_dependency_version_mismatch:{package_name}")
    else:
        for package_name, expected_version in sorted(expected_dependencies.items()):
            installed_version = installed_dependencies[package_name]
            if installed_version is None:
                mismatches.append(f"direct_dependency_missing:{package_name}")
            elif installed_version != expected_version:
                mismatches.append(f"direct_dependency_version_mismatch:{package_name}")

    readiness_blockers = tuple(
        dict.fromkeys((*profile_record.readiness_blockers, *mismatches))
    )
    environment_match = not mismatches
    decision = "pass" if profile_record.formal_ready and environment_match else "blocked"
    report = {
        "profile_name": profile_record.profile_name,
        "profile_digest": profile_record.profile_digest,
        "complete_hash_lock_digest": profile_record.complete_hash_lock_digest,
        "profile_formal_ready": profile_record.formal_ready,
        "expected_environment": expected_environment,
        "observed_environment": observed_environment,
        "environment_match": environment_match,
        "mismatches": mismatches,
        "readiness_blockers": list(readiness_blockers),
        "decision": decision,
    }
    report["inspection_digest"] = _stable_digest(report)
    return report
