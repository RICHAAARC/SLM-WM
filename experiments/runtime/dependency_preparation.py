"""从受治理依赖 profile 准备并核验当前 Python 运行环境.

该 CLI 不解析 Notebook 内的包清单, 也不把直接依赖输入解释为完整锁.
正式准备只接受 registry API 已确认就绪且已经提交到 Git 的完整哈希锁.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runtime.dependency_profiles import (  # noqa: E402
    DependencyProfile,
    WORKFLOW_ORCHESTRATOR_PROFILE_ID,
    build_dependency_profile_summary,
    get_dependency_profile,
    inspect_dependency_profile_environment,
    require_dependency_profile_ready,
)
from experiments.runtime.repository_environment import (  # noqa: E402
    require_published_formal_execution_lock,
)


REPORT_SCHEMA = "dependency_profile_preparation_report"
REPORT_SCHEMA_VERSION = 1
REPORT_RELATIVE_ROOT = Path("outputs/dependency_profiles")
REPORT_FILE_NAME = "dependency_profile_report.json"

CommandRunner = Callable[[Sequence[str], Path], Any]


def _run_command(command: Sequence[str], working_directory: Path) -> dict[str, Any]:
    """执行 argv 命令并返回可写入正式报告的完整进程结果."""

    completed = subprocess.run(
        list(command),
        cwd=working_directory,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    return {
        "return_code": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _normalize_command_result(result: Any) -> dict[str, Any]:
    """把测试 runner 与 subprocess 结果统一为进程证据字段."""

    if isinstance(result, int) and not isinstance(result, bool):
        return {"return_code": result, "stdout": "", "stderr": ""}
    if isinstance(result, subprocess.CompletedProcess):
        return {
            "return_code": int(result.returncode),
            "stdout": str(result.stdout or ""),
            "stderr": str(result.stderr or ""),
        }
    if isinstance(result, dict):
        return_code = result.get("return_code")
        if isinstance(return_code, bool) or not isinstance(return_code, int):
            raise ValueError("command runner 结果必须包含整数 return_code")
        return {
            "return_code": return_code,
            "stdout": str(result.get("stdout") or ""),
            "stderr": str(result.get("stderr") or ""),
        }
    raise TypeError("command runner 必须返回 int、CompletedProcess 或 dict")


def _report_path(repository_root: Path, profile_id: str) -> Path:
    """返回受治理报告路径, 并保证路径不会越过 ``outputs/``."""

    outputs_root = (repository_root / "outputs").resolve()
    report_path = (
        repository_root / REPORT_RELATIVE_ROOT / profile_id / REPORT_FILE_NAME
    ).resolve()
    try:
        report_path.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError(f"依赖 profile 报告必须位于 outputs/ 下: {report_path}") from exc
    return report_path


def _write_report(report_path: Path, report: dict[str, Any]) -> None:
    """以稳定 JSON 排版写入报告, 供后续 runner 和证据审计复用."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(
        (
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    )


def _inspect_committed_file(
    repository_root: Path,
    relative_path: str,
) -> dict[str, Any]:
    """确认一个受治理输入来自 ``HEAD`` 且工作树内容没有漂移."""

    posix_path = PurePosixPath(relative_path).as_posix()
    head_result = subprocess.run(
        ["git", "-C", str(repository_root), "cat-file", "-e", f"HEAD:{posix_path}"],
        check=False,
        capture_output=True,
        text=True,
    )
    worktree_result = subprocess.run(
        ["git", "-C", str(repository_root), "diff", "--quiet", "HEAD", "--", posix_path],
        check=False,
        capture_output=True,
        text=True,
    )
    working_tree_file_present = (repository_root / PurePosixPath(posix_path)).is_file()
    head_contains_file = head_result.returncode == 0
    worktree_matches_head = worktree_result.returncode == 0
    return {
        "path": posix_path,
        "working_tree_file_present": working_tree_file_present,
        "head_contains_file": head_contains_file,
        "worktree_matches_head": worktree_matches_head,
        "is_committed": (
            working_tree_file_present and head_contains_file and worktree_matches_head
        ),
    }


def _inspect_dependency_files_commit_state(
    repository_root: Path,
    profile: DependencyProfile,
) -> dict[str, Any]:
    """同时核验 registry、直接输入和完整哈希锁的提交状态."""

    registry_relative_path = "configs/dependency_profile_registry.json"
    files = {
        "registry": _inspect_committed_file(repository_root, registry_relative_path),
        "direct_requirements": _inspect_committed_file(
            repository_root,
            profile.direct_requirements_path,
        ),
        "complete_hash_lock": _inspect_committed_file(
            repository_root,
            profile.complete_hash_lock_path,
        ),
    }
    return {
        "files": files,
        "all_committed": all(record["is_committed"] for record in files.values()),
    }


_INTERPRETER_PLATFORM_MISMATCHES = frozenset(
    {
        "python_implementation_mismatch",
        "python_version_mismatch",
        "operating_system_mismatch",
        "machine_mismatch",
    }
)


def _build_report(
    profile: DependencyProfile,
    profile_summary: dict[str, Any],
    working_directory: Path,
) -> dict[str, Any]:
    """构造依赖准备报告骨架, 失败路径与成功路径共享同一 schema."""

    return {
        "report_schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "profile_id": profile.profile_name,
        "execution_role": profile.execution_role,
        "python_executable": sys.executable,
        "working_directory": str(working_directory),
        "profile_digest": profile.profile_digest,
        "profile_summary_digest": profile_summary["summary_digest"],
        "direct_requirements_path": profile.direct_requirements_path,
        "direct_requirements_digest": profile.direct_requirements_digest,
        "complete_hash_lock_path": profile.complete_hash_lock_path,
        "complete_hash_lock_digest": profile.complete_hash_lock_digest,
        "complete_hash_lock_dependency_count": profile.complete_hash_lock_dependency_count,
        "pytorch_index_url": profile.pytorch_index_url,
        "formal_ready": profile.formal_ready,
        "readiness_blockers": list(profile.readiness_blockers),
        "formal_execution_lock": {},
        "formal_execution_commit": None,
        "formal_execution_lock_digest": None,
        "formal_execution_lock_ready": False,
        "repository_commit_state": {
            "files": {},
            "all_committed": False,
        },
        "installation": {
            "attempted": False,
            "command": [],
            "working_directory": str(working_directory),
            "return_code": None,
            "stdout": "",
            "stderr": "",
        },
        "pip_check": {
            "compatibility_check_required": (
                profile.profile_name != WORKFLOW_ORCHESTRATOR_PROFILE_ID
            ),
            "attempted": False,
            "command": [],
            "working_directory": str(working_directory),
            "return_code": None,
            "stdout": "",
            "stderr": "",
            "decision": (
                "pending"
                if profile.profile_name != WORKFLOW_ORCHESTRATOR_PROFILE_ID
                else "not_applicable_to_orchestrator"
            ),
        },
        "runtime_comparison": None,
        "decision": "fail",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }


def prepare_dependency_profile(
    profile_id: str,
    *,
    repository_root: str | Path = ROOT,
    command_runner: CommandRunner = _run_command,
) -> tuple[dict[str, Any], Path]:
    """安装并核验一个正式依赖 profile, 同时持久化闭合报告.

    ``repository_root`` 和 ``command_runner`` 仅用于轻量功能测试与复用;
    命令行入口始终使用当前仓库根目录和当前 Python 解释器.
    """

    root = Path(repository_root).resolve()
    registry_path = root / "configs/dependency_profile_registry.json"
    profile = get_dependency_profile(profile_id, registry_path)
    profile_summary = build_dependency_profile_summary(profile_id, registry_path)
    report = _build_report(profile, profile_summary, root)
    report_path = _report_path(root, profile.profile_name)
    try:
        formal_execution_lock = require_published_formal_execution_lock(root)
    except ValueError as exc:
        report["failure_reasons"] = [
            f"formal_execution_lock_not_ready:{type(exc).__name__}"
        ]
        _write_report(report_path, report)
        return report, report_path
    report["formal_execution_lock"] = formal_execution_lock
    report["formal_execution_commit"] = formal_execution_lock[
        "formal_execution_commit"
    ]
    report["formal_execution_lock_digest"] = formal_execution_lock[
        "formal_execution_lock_digest"
    ]
    report["formal_execution_lock_ready"] = True
    repository_commit_state = _inspect_dependency_files_commit_state(
        root,
        profile,
    )
    report["repository_commit_state"] = repository_commit_state

    try:
        ready_profile = require_dependency_profile_ready(profile_id, registry_path)
    except RuntimeError:
        report["failure_reasons"] = list(profile.readiness_blockers)
        if not report["failure_reasons"]:
            report["failure_reasons"] = ["dependency_profile_not_ready"]
        _write_report(report_path, report)
        return report, report_path

    lock_path = (root / PurePosixPath(ready_profile.complete_hash_lock_path)).resolve()
    if not repository_commit_state["all_committed"]:
        report["failure_reasons"] = ["dependency_profile_inputs_not_committed"]
        _write_report(report_path, report)
        return report, report_path

    pre_install_inspection = inspect_dependency_profile_environment(
        profile_id,
        torch_module=object(),
        path=registry_path,
    )
    interpreter_platform_mismatches = [
        reason
        for reason in pre_install_inspection["mismatches"]
        if reason in _INTERPRETER_PLATFORM_MISMATCHES
    ]
    if interpreter_platform_mismatches:
        report["runtime_comparison"] = pre_install_inspection
        report["failure_reasons"] = interpreter_platform_mismatches
        _write_report(report_path, report)
        return report, report_path

    install_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--require-hashes",
        "--only-binary=:all:",
    ]
    if ready_profile.pytorch_index_url is not None:
        install_command.extend(
            ["--extra-index-url", ready_profile.pytorch_index_url]
        )
    install_command.extend(["-r", str(lock_path)])
    installation_result = _normalize_command_result(
        command_runner(install_command, root)
    )
    report["installation"] = {
        "attempted": True,
        "command": install_command,
        "working_directory": str(root),
        **installation_result,
    }
    if installation_result["return_code"] != 0:
        report["failure_reasons"] = ["pip_install_failed"]
        _write_report(report_path, report)
        return report, report_path

    if profile.profile_name != WORKFLOW_ORCHESTRATOR_PROFILE_ID:
        pip_check_command = [sys.executable, "-m", "pip", "check"]
        pip_check_result = _normalize_command_result(
            command_runner(pip_check_command, root)
        )
        report["pip_check"] = {
            "compatibility_check_required": True,
            "attempted": True,
            "command": pip_check_command,
            "working_directory": str(root),
            **pip_check_result,
            "decision": "pass" if pip_check_result["return_code"] == 0 else "fail",
        }
        if pip_check_result["return_code"] != 0:
            report["failure_reasons"] = ["pip_check_failed"]
            _write_report(report_path, report)
            return report, report_path

    runtime_comparison = inspect_dependency_profile_environment(
        profile_id,
        path=registry_path,
    )
    report["runtime_comparison"] = runtime_comparison

    failure_reasons: list[str] = list(runtime_comparison["mismatches"])
    report["failure_reasons"] = failure_reasons
    report["decision"] = "pass" if not failure_reasons else "fail"
    _write_report(report_path, report)
    return report, report_path


def build_parser() -> argparse.ArgumentParser:
    """构造依赖 profile 准备命令的参数解析器."""

    parser = argparse.ArgumentParser(
        description="使用已提交完整哈希锁准备并核验隔离依赖 profile."
    )
    parser.add_argument("--profile", required=True, help="registry 中登记的 profile id.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行 CLI 并用退出码表达依赖环境是否通过正式门禁."""

    args = build_parser().parse_args(argv)
    try:
        report, report_path = prepare_dependency_profile(args.profile)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "profile_id": args.profile,
                    "decision": "fail",
                    "failure_reasons": [f"profile_registry_error:{type(exc).__name__}"],
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    print(
        json.dumps(
            {
                "profile_id": report["profile_id"],
                "python_executable": report["python_executable"],
                "report_path": str(report_path),
                "decision": report["decision"],
                "failure_reasons": report["failure_reasons"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
