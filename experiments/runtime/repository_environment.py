"""仓库实验运行环境与摘要工具。

该模块保存不依赖 Notebook 的通用运行环境能力, 供核心实验 runner、完整论文
实验 runner 与 Colab 包装层共同复用。这样可以避免正式实验逻辑为了读取 Git
版本、依赖版本或文件摘要而反向依赖 `paper_workflow/`。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

from main.core.digest import build_stable_digest
from experiments.runtime.dependency_profiles import (
    build_dependency_profile_summary,
    inspect_dependency_profile_environment,
)

FORMAL_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
FORMAL_EXECUTION_LOCK_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FORMAL_EXECUTION_LOCK_SCHEMA = "clean_detached_git_commit_v1"
FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY = "SLM_WM_FORMAL_EXECUTION_COMMIT"
FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY = (
    "SLM_WM_FORMAL_EXECUTION_LOCK_DIGEST"
)
FORMAL_EXECUTION_LOCK_FIELDS = frozenset(
    {
        "formal_execution_lock_schema",
        "formal_execution_commit",
        "formal_execution_head_detached",
        "formal_execution_worktree_clean",
        "formal_execution_lock_ready",
        "formal_execution_lock_digest",
    }
)


class FormalExecutionLockError(ValueError):
    """表示仓库状态不满足正式执行的 Git 身份锁契约."""


def normalize_formal_git_commit(value: Any) -> str:
    """要求提交身份是精确40位小写 Git SHA.

    该函数不执行大小写转换或空白裁剪. 正式证据必须传播唯一完整提交身份,
    不能把短 SHA, 分支名或经过宽松规范化的文本升级为正式代码版本.
    """

    if not isinstance(value, str) or FORMAL_GIT_COMMIT_PATTERN.fullmatch(value) is None:
        raise FormalExecutionLockError("正式 Git 提交必须是精确40位小写十六进制 SHA")
    return value


def _formal_execution_lock_payload(commit: str) -> dict[str, Any]:
    """构造执行锁唯一允许的摘要载荷."""

    return {
        "formal_execution_lock_schema": FORMAL_EXECUTION_LOCK_SCHEMA,
        "formal_execution_commit": commit,
        "formal_execution_head_detached": True,
        "formal_execution_worktree_clean": True,
        "formal_execution_lock_ready": True,
    }


def validate_formal_execution_lock_record(
    record: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """严格验证执行锁字段, 值类型与稳定摘要.

    该函数只验证记录自身的完整性. 需要确认记录仍对应当前仓库状态时,
    应调用 ``require_published_formal_execution_lock`` 实时读取 Git 状态.
    """

    if not isinstance(record, Mapping):
        raise FormalExecutionLockError("正式执行锁必须是字段映射")
    if set(record) != FORMAL_EXECUTION_LOCK_FIELDS:
        raise FormalExecutionLockError("正式执行锁字段集合不符合唯一 schema")

    commit = normalize_formal_git_commit(record["formal_execution_commit"])
    if record["formal_execution_lock_schema"] != FORMAL_EXECUTION_LOCK_SCHEMA:
        raise FormalExecutionLockError("正式执行锁 schema 不受支持")
    if record["formal_execution_head_detached"] is not True:
        raise FormalExecutionLockError("正式执行锁必须声明 detached HEAD")
    if record["formal_execution_worktree_clean"] is not True:
        raise FormalExecutionLockError("正式执行锁必须声明 clean 工作树")
    if record["formal_execution_lock_ready"] is not True:
        raise FormalExecutionLockError("正式执行锁必须处于 ready 状态")

    digest = record["formal_execution_lock_digest"]
    if (
        not isinstance(digest, str)
        or FORMAL_EXECUTION_LOCK_DIGEST_PATTERN.fullmatch(digest) is None
    ):
        raise FormalExecutionLockError("正式执行锁摘要必须是精确 SHA-256")
    payload = _formal_execution_lock_payload(commit)
    if digest != build_stable_digest(payload):
        raise FormalExecutionLockError("正式执行锁摘要与规范载荷不一致")
    return {**payload, "formal_execution_lock_digest": digest}


def publish_formal_execution_lock(
    record: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """验证后统一发布当前进程可继承的执行锁环境变量."""

    validated_record = validate_formal_execution_lock_record(record)
    os.environ[FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY] = validated_record[
        "formal_execution_commit"
    ]
    os.environ[FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY] = validated_record[
        "formal_execution_lock_digest"
    ]
    return validated_record


def verify_formal_execution_lock_code_version(
    lock_record: Mapping[str, Any] | Any,
    code_version: Any,
) -> dict[str, Any]:
    """验证产物代码版本与一个严格执行锁使用同一完整提交."""

    validated_record = validate_formal_execution_lock_record(lock_record)
    normalized_code_version = normalize_formal_git_commit(code_version)
    if normalized_code_version != validated_record["formal_execution_commit"]:
        raise FormalExecutionLockError("产物代码版本与正式执行锁提交不一致")
    return validated_record


def validate_formal_execution_lock_pair(
    run_lock: Mapping[str, Any] | Any,
    package_lock: Mapping[str, Any] | Any,
    code_version: Any,
) -> dict[str, Any]:
    """验证运行锁, 打包锁和产物代码版本绑定到同一提交.

    返回值是规范化后的运行锁. 两个输入锁在严格 schema 下具有相同提交时,
    其摘要也必须相同, 因而调用方可直接传播返回记录而不再重复拼装校验.
    """

    validated_run_lock = verify_formal_execution_lock_code_version(
        run_lock,
        code_version,
    )
    validated_package_lock = verify_formal_execution_lock_code_version(
        package_lock,
        code_version,
    )
    if validated_run_lock != validated_package_lock:
        raise FormalExecutionLockError("运行执行锁与打包执行锁不一致")
    return validated_run_lock


def _run_git(
    root_path: Path,
    arguments: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """在指定仓库执行无交互 Git 查询并保留完整返回状态."""

    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root_path,
            check=check,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FormalExecutionLockError("无法读取正式执行所需的 Git 仓库状态") from exc


def build_formal_execution_lock(
    root_path: str | Path,
    expected_commit: str,
) -> dict[str, Any]:
    """验证 clean detached HEAD 并构造可传播的稳定执行锁记录.

    该原语属于通用正式执行边界. 脚本与 Colab workflow 可以在启动 GPU 作业前
    调用它, 并把返回记录写入 manifest 或 job identity. 函数成功返回即表示当前
    HEAD 精确等于预期完整提交, 工作树无已跟踪或未跟踪改动且 HEAD 已 detached.
    """

    repository_root = Path(root_path).resolve()
    normalized_expected_commit = normalize_formal_git_commit(expected_commit)
    head_result = _run_git(
        repository_root,
        ["rev-parse", "--verify", "HEAD^{commit}"],
    )
    current_commit = normalize_formal_git_commit(head_result.stdout.strip())
    if current_commit != normalized_expected_commit:
        raise FormalExecutionLockError("当前 HEAD 与正式执行预期提交不一致")

    status_result = _run_git(
        repository_root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
    )
    if status_result.stdout:
        raise FormalExecutionLockError("正式执行要求 Git 工作树完全 clean")

    symbolic_result = _run_git(
        repository_root,
        ["symbolic-ref", "-q", "HEAD"],
        check=False,
    )
    if symbolic_result.returncode == 0:
        raise FormalExecutionLockError("正式执行要求 HEAD 处于 detached 状态")
    if symbolic_result.returncode != 1:
        raise FormalExecutionLockError("无法确认 HEAD 是否处于 detached 状态")

    payload = _formal_execution_lock_payload(current_commit)
    return validate_formal_execution_lock_record({
        **payload,
        "formal_execution_lock_digest": build_stable_digest(payload),
    })


def require_published_formal_execution_lock(
    root_path: str | Path,
) -> dict[str, Any]:
    """读取已发布身份并实时复验当前仓库的 clean detached HEAD.

    环境变量只用于传播预期提交和摘要, 不能单独证明仓库状态. 该函数会重新
    执行 Git 查询并重建规范锁, 因而发布后发生的 checkout 或工作树改动都会
    在正式 workflow 进入业务逻辑前被拒绝.
    """

    published_commit = normalize_formal_git_commit(
        os.environ.get(FORMAL_EXECUTION_COMMIT_ENVIRONMENT_KEY)
    )
    published_digest = os.environ.get(
        FORMAL_EXECUTION_LOCK_DIGEST_ENVIRONMENT_KEY,
        "",
    )
    if FORMAL_EXECUTION_LOCK_DIGEST_PATTERN.fullmatch(published_digest) is None:
        raise FormalExecutionLockError("已发布正式执行锁摘要必须是精确 SHA-256")

    verified_record = build_formal_execution_lock(root_path, published_commit)
    if verified_record["formal_execution_lock_digest"] != published_digest:
        raise FormalExecutionLockError("已发布正式执行锁摘要与当前仓库状态不一致")
    return validate_formal_execution_lock_record(verified_record)


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定、可读文本。"""

    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def json_line(value: Any) -> str:
    """把 JSON 兼容对象转为 JSONL 单行文本。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def resolve_code_version(root_path: Path) -> str:
    """读取当前完整 Git 提交标识, 不可用时返回稳定降级值.

    此函数属于通用工程写法: 任何实验产物 manifest 都需要记录代码版本,
    因此它必须位于 Notebook 无关的运行环境工具层。
    """

    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "git_version_unavailable"
    commit_id = commit_result.stdout.strip()
    if FORMAL_GIT_COMMIT_PATTERN.fullmatch(commit_id) is None:
        return "git_version_unavailable"
    return f"{commit_id}-dirty" if status_result.stdout.strip() else commit_id


def file_digest(path: Path) -> str:
    """计算文件内容 SHA-256 摘要。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(tensor: Any) -> str:
    """根据 tensor 数值生成稳定摘要。

    此处只保留摘要而不保存原始张量, 主要考虑是降低产物体积, 同时保留跨运行
    对齐检查所需的可审计指纹。
    """

    values = tensor.detach().float().cpu().reshape(-1).tolist()
    rounded_values = [round(float(value), 8) for value in values]
    return build_stable_digest(rounded_values)


def build_runtime_environment_report(
    dependency_profile_id: str,
    torch_module: Any | None = None,
    *,
    verified_formal_execution_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造绑定完整哈希锁 profile 的真实运行环境快照.

    该函数不会安装依赖. 调用方必须显式声明运行职责对应的 profile, 从而避免
    主方法、T2SMark 与 official-reference 环境被隐式合并为一套依赖组合.
    """

    if torch_module is None:
        try:
            import torch as imported_torch_module
        except Exception:
            imported_torch_module = None
        torch_module = imported_torch_module
    profile_summary = build_dependency_profile_summary(dependency_profile_id)
    dependency_inspection = inspect_dependency_profile_environment(
        dependency_profile_id,
        torch_module=torch_module,
    )
    observed_environment = dependency_inspection["observed_environment"]
    package_versions = observed_environment["direct_dependencies"]
    cuda_available = bool(observed_environment["cuda_available"])
    cuda_version = observed_environment["torch_cuda_version"]
    gpu_name = ""
    device_count = 0
    if torch_module is not None:
        device_count = int(torch_module.cuda.device_count()) if cuda_available else 0
        gpu_name = torch_module.cuda.get_device_name(0) if cuda_available and device_count else ""
    validated_execution_lock = (
        validate_formal_execution_lock_record(verified_formal_execution_lock)
        if verified_formal_execution_lock is not None
        else None
    )
    formal_execution_commit = (
        validated_execution_lock["formal_execution_commit"]
        if validated_execution_lock is not None
        else ""
    )
    formal_execution_lock_digest = (
        validated_execution_lock["formal_execution_lock_digest"]
        if validated_execution_lock is not None
        else ""
    )
    return {
        "dependency_mode": "committed_complete_hash_lock",
        "dependency_profile_id": dependency_profile_id,
        "dependency_profile_digest": profile_summary["profile_digest"],
        "dependency_profile_summary_digest": profile_summary["summary_digest"],
        "direct_requirements_path": profile_summary["direct_requirements_path"],
        "direct_requirements_digest": profile_summary["direct_requirements_digest"],
        "complete_hash_lock_path": profile_summary["complete_hash_lock_path"],
        "complete_hash_lock_digest": profile_summary["complete_hash_lock_digest"],
        "complete_hash_lock_present": profile_summary["complete_hash_lock_present"],
        "complete_hash_lock_dependency_count": profile_summary[
            "complete_hash_lock_dependency_count"
        ],
        "dependency_profile_formal_ready": profile_summary["formal_ready"],
        "dependency_environment_ready": dependency_inspection["decision"] == "pass",
        "dependency_readiness_blockers": dependency_inspection[
            "readiness_blockers"
        ],
        "dependency_environment_inspection": dependency_inspection,
        "python_version": observed_environment["python_version"],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "package_versions": package_versions,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "device_count": device_count,
        "gpu_name": gpu_name,
        "formal_execution_lock": validated_execution_lock or {},
        "formal_execution_commit": formal_execution_commit,
        "formal_execution_lock_digest": formal_execution_lock_digest,
        "formal_execution_lock_ready": validated_execution_lock is not None,
    }


def flatten_environment_versions(environment_report: dict[str, Any]) -> dict[str, str]:
    """把常用依赖版本提升为统一摘要字段, 便于 result metadata 直接读取。"""

    package_versions = environment_report["package_versions"]
    return {
        "torch_version": package_versions["torch"],
        "diffusers_version": package_versions["diffusers"],
        "transformers_version": package_versions["transformers"],
        "accelerate_version": package_versions["accelerate"],
        "huggingface_hub_version": package_versions["huggingface_hub"],
        "tokenizers_version": package_versions["tokenizers"],
        "safetensors_version": package_versions["safetensors"],
        "sentencepiece_version": package_versions["sentencepiece"],
        "protobuf_version": package_versions["protobuf"],
        "numpy_version": package_versions["numpy"],
        "pillow_version": package_versions["pillow"],
    }
