"""生成 Colab Notebook 可读取的正式依赖 profile 报告."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.runtime.dependency_profiles import (
    build_dependency_profile_summary,
    inspect_dependency_profile_environment,
)


DEPENDENCY_PREPARATION_REPORT_ROOT = Path("outputs/dependency_profiles")
DEPENDENCY_PREPARATION_REPORT_NAME = "dependency_profile_report.json"
DEFAULT_WORKFLOW_DEPENDENCY_PROFILE_ID = "workflow_orchestrator"


def dependency_preparation_report_path(profile_id: str) -> str:
    """返回统一 CLI 为指定 profile 写出的仓库相对报告路径."""

    return (
        DEPENDENCY_PREPARATION_REPORT_ROOT
        / profile_id
        / DEPENDENCY_PREPARATION_REPORT_NAME
    ).as_posix()


def build_dependency_report(
    profile_id: str = DEFAULT_WORKFLOW_DEPENDENCY_PROFILE_ID,
    *,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """核验当前解释器是否精确满足一个受治理依赖 profile.

    该函数只读取 registry、完整哈希锁和当前环境状态. 依赖安装必须先由
    ``scripts/prepare_dependency_profile.py`` 完成, Notebook 不保存或拼装
    任何包规格与安装命令.
    """

    profile_summary = build_dependency_profile_summary(profile_id)
    inspection = inspect_dependency_profile_environment(
        profile_id,
        torch_module=torch_module,
    )
    observed_packages = inspection["observed_environment"]["direct_dependencies"]
    missing_packages = sorted(
        package_name
        for package_name, installed_version in observed_packages.items()
        if installed_version is None
    )
    decision = "pass" if inspection["decision"] == "pass" else "blocked"
    return {
        "dependency_decision": decision,
        "dependency_mode": "committed_complete_hash_lock",
        "dependency_profile_id": profile_id,
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
        "dependency_preparation_report_path": dependency_preparation_report_path(
            profile_id
        ),
        "dependency_count": len(observed_packages),
        "missing_dependency_count": len(missing_packages),
        "missing_dependencies": missing_packages,
        "package_versions": observed_packages,
        "environment_inspection": inspection,
        "unsupported_reasons": list(inspection["readiness_blockers"]),
        "supports_paper_claim": False,
    }


def build_notebook_dependency_report(profile_id: str) -> dict[str, Any]:
    """按 Notebook 声明的固定 profile 标识生成依赖报告."""

    return build_dependency_report(profile_id)
