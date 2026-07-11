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
    REQUIRED_DEPENDENCY_PROFILE_NAMES,
    WORKFLOW_ORCHESTRATOR_PROFILE_ID,
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
ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY = (
    "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH"
)
ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY = (
    "SLM_WM_ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST"
)
ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_SCHEMA = (
    "isolated_dependency_environment_preparation_report"
)
ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_SCHEMA_VERSION = 1
ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_FILE_NAME = (
    "isolated_dependency_environment_report.json"
)
SCIENTIFIC_DEPENDENCY_PROFILE_IDS = frozenset(
    profile_id
    for profile_id in REQUIRED_DEPENDENCY_PROFILE_NAMES
    if profile_id != WORKFLOW_ORCHESTRATOR_PROFILE_ID
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


def _is_sha256(value: Any) -> bool:
    """判断一个值是否为规范小写 SHA-256 文本."""

    return (
        isinstance(value, str)
        and FORMAL_EXECUTION_LOCK_DIGEST_PATTERN.fullmatch(value) is not None
    )


def _isolated_context_report_skeleton(
    dependency_profile_id: str,
    *,
    required: bool,
) -> dict[str, Any]:
    """构造科学子解释器上下文检查的稳定报告骨架."""

    return {
        "required": required,
        "profile_id": dependency_profile_id,
        "dependency_environment_report_path": "",
        "dependency_environment_report_digest": "",
        "dependency_environment_report_actual_digest": "",
        "reported_profile_digest": "",
        "reported_complete_hash_lock_digest": "",
        "reported_formal_execution_lock_digest": "",
        "reported_python_executable": "",
        "reported_python_executable_sha256": "",
        "current_python_executable": str(Path(sys.executable).absolute()),
        "current_python_executable_sha256": "",
        "blockers": [],
        "decision": "blocked" if required else "not_required",
        "ready": not required,
    }


def _append_context_blocker(context: dict[str, Any], blocker: str) -> None:
    """按首次出现顺序加入稳定 blocker, 避免重复诊断."""

    if blocker not in context["blockers"]:
        context["blockers"].append(blocker)


def _load_isolated_environment_report(
    context: dict[str, Any],
    *,
    repository_root: Path,
    dependency_profile_id: str,
) -> dict[str, Any] | None:
    """读取并校验注入报告的路径、文件摘要和 JSON 外形."""

    path_value = os.environ.get(
        ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_PATH_ENVIRONMENT_KEY,
        "",
    )
    digest_value = os.environ.get(
        ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_DIGEST_ENVIRONMENT_KEY,
        "",
    )
    context["dependency_environment_report_path"] = path_value
    context["dependency_environment_report_digest"] = digest_value
    if not path_value or path_value != path_value.strip():
        _append_context_blocker(
            context,
            "isolated_context_environment_report_path_missing",
        )
        return None
    if not _is_sha256(digest_value):
        _append_context_blocker(
            context,
            "isolated_context_environment_report_digest_invalid",
        )
        return None

    report_path = Path(path_value)
    expected_path = (
        repository_root
        / "outputs"
        / "dependency_profiles"
        / dependency_profile_id
        / ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_FILE_NAME
    ).absolute()
    if not report_path.is_absolute() or report_path.absolute() != expected_path:
        _append_context_blocker(
            context,
            "isolated_context_environment_report_path_mismatch",
        )
        return None
    if not report_path.is_file():
        _append_context_blocker(
            context,
            "isolated_context_environment_report_missing",
        )
        return None
    actual_digest = file_digest(report_path)
    context["dependency_environment_report_actual_digest"] = actual_digest
    if actual_digest != digest_value:
        _append_context_blocker(
            context,
            "isolated_context_environment_report_digest_mismatch",
        )
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        _append_context_blocker(
            context,
            "isolated_context_environment_report_invalid_json",
        )
        return None
    if not isinstance(report, dict):
        _append_context_blocker(
            context,
            "isolated_context_environment_report_not_object",
        )
        return None
    return report


def _inspect_isolated_scientific_context(
    dependency_profile_id: str,
    *,
    profile_summary: Mapping[str, Any],
    validated_execution_lock: Mapping[str, Any] | None,
    repository_root: Path,
) -> dict[str, Any]:
    """严格核验科学进程继承的隔离环境、解释器和执行锁身份."""

    required = dependency_profile_id in SCIENTIFIC_DEPENDENCY_PROFILE_IDS
    context = _isolated_context_report_skeleton(
        dependency_profile_id,
        required=required,
    )
    if not required:
        return context

    report = _load_isolated_environment_report(
        context,
        repository_root=repository_root,
        dependency_profile_id=dependency_profile_id,
    )
    if report is None:
        return context

    context["reported_profile_digest"] = str(report.get("profile_digest", ""))
    context["reported_complete_hash_lock_digest"] = str(
        report.get("complete_hash_lock_digest", "")
    )
    context["reported_formal_execution_lock_digest"] = str(
        report.get("formal_execution_lock_digest", "")
    )
    context["reported_python_executable"] = str(
        report.get("python_executable_path", "")
    )
    context["reported_python_executable_sha256"] = str(
        report.get("python_executable_sha256", "")
    )

    report_contract = {
        "report_schema": ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_SCHEMA,
        "schema_version": ISOLATED_DEPENDENCY_ENVIRONMENT_REPORT_SCHEMA_VERSION,
        "operation_kind": "formal_dependency_environment_preparation",
        "profile_id": dependency_profile_id,
        "profile_digest": profile_summary["profile_digest"],
        "direct_requirements_digest": profile_summary["direct_requirements_digest"],
        "complete_hash_lock_digest": profile_summary["complete_hash_lock_digest"],
        "complete_hash_lock_dependency_count": profile_summary[
            "complete_hash_lock_dependency_count"
        ],
        "provisioned": True,
        "formal_preparation_completed": True,
        "formal_ready": True,
        "decision": "pass",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    schema_fields = {
        "report_schema",
        "schema_version",
        "operation_kind",
        "provisioned",
        "formal_preparation_completed",
        "formal_ready",
        "decision",
        "failure_reasons",
        "supports_paper_claim",
    }
    if any(report.get(field) != report_contract[field] for field in schema_fields):
        _append_context_blocker(
            context,
            "isolated_context_environment_report_not_ready",
        )
    profile_fields = {
        "profile_id",
        "profile_digest",
        "direct_requirements_digest",
    }
    if any(report.get(field) != report_contract[field] for field in profile_fields):
        _append_context_blocker(
            context,
            "isolated_context_profile_identity_mismatch",
        )
    lock_fields = {
        "complete_hash_lock_digest",
        "complete_hash_lock_dependency_count",
    }
    if (
        profile_summary.get("complete_hash_lock_present") is not True
        or profile_summary.get("formal_ready") is not True
        or not _is_sha256(profile_summary.get("complete_hash_lock_digest"))
        or any(report.get(field) != report_contract[field] for field in lock_fields)
    ):
        _append_context_blocker(
            context,
            "isolated_context_complete_hash_lock_mismatch",
        )

    if validated_execution_lock is None:
        _append_context_blocker(
            context,
            "isolated_context_formal_execution_lock_missing",
        )
    else:
        expected_execution_lock = dict(validated_execution_lock)
        if (
            report.get("formal_execution_lock") != expected_execution_lock
            or report.get("formal_execution_commit")
            != expected_execution_lock["formal_execution_commit"]
            or report.get("formal_execution_lock_digest")
            != expected_execution_lock["formal_execution_lock_digest"]
            or report.get("formal_execution_lock_ready") is not True
        ):
            _append_context_blocker(
                context,
                "isolated_context_formal_execution_lock_mismatch",
            )

    current_python = Path(sys.executable).absolute()
    reported_python_value = report.get("python_executable_path")
    reported_python = (
        Path(reported_python_value).absolute()
        if isinstance(reported_python_value, str) and reported_python_value
        else None
    )
    if reported_python is None or reported_python != current_python:
        _append_context_blocker(
            context,
            "isolated_context_python_executable_mismatch",
        )
    if not current_python.is_file():
        _append_context_blocker(
            context,
            "isolated_context_python_executable_missing",
        )
    else:
        current_python_digest = file_digest(current_python)
        context["current_python_executable_sha256"] = current_python_digest
        reported_python_digest = report.get("python_executable_sha256")
        if (
            not _is_sha256(reported_python_digest)
            or report.get("python_executable_sha256_after_preparation")
            != reported_python_digest
            or current_python_digest != reported_python_digest
        ):
            _append_context_blocker(
                context,
                "isolated_context_python_executable_digest_mismatch",
            )

    dependency_preparation_report = report.get("dependency_preparation_report")
    if not isinstance(dependency_preparation_report, dict):
        _append_context_blocker(
            context,
            "isolated_context_dependency_preparation_report_invalid",
        )
    else:
        nested_contract = {
            "profile_id": dependency_profile_id,
            "profile_digest": profile_summary["profile_digest"],
            "direct_requirements_digest": profile_summary[
                "direct_requirements_digest"
            ],
            "complete_hash_lock_digest": profile_summary[
                "complete_hash_lock_digest"
            ],
            "complete_hash_lock_dependency_count": profile_summary[
                "complete_hash_lock_dependency_count"
            ],
            "python_executable": str(current_python),
            "formal_execution_lock": (
                dict(validated_execution_lock)
                if validated_execution_lock is not None
                else None
            ),
            "formal_ready": True,
            "decision": "pass",
            "failure_reasons": [],
            "supports_paper_claim": False,
        }
        if any(
            dependency_preparation_report.get(field) != expected_value
            for field, expected_value in nested_contract.items()
        ):
            _append_context_blocker(
                context,
                "isolated_context_dependency_preparation_report_mismatch",
            )

    context["ready"] = not context["blockers"]
    context["decision"] = "pass" if context["ready"] else "blocked"
    return context


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
    repository_root: str | Path | None = None,
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
    resolved_repository_root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    isolated_scientific_context = _inspect_isolated_scientific_context(
        dependency_profile_id,
        profile_summary=profile_summary,
        validated_execution_lock=validated_execution_lock,
        repository_root=resolved_repository_root,
    )
    dependency_readiness_blockers = list(
        dict.fromkeys(
            (
                *dependency_inspection["readiness_blockers"],
                *isolated_scientific_context["blockers"],
            )
        )
    )
    dependency_environment_ready = (
        dependency_inspection["decision"] == "pass"
        and isolated_scientific_context["ready"] is True
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
        "dependency_environment_ready": dependency_environment_ready,
        "dependency_readiness_blockers": dependency_readiness_blockers,
        "dependency_environment_inspection": dependency_inspection,
        "isolated_scientific_context_required": isolated_scientific_context[
            "required"
        ],
        "isolated_scientific_context_ready": isolated_scientific_context["ready"],
        "isolated_scientific_context": isolated_scientific_context,
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
