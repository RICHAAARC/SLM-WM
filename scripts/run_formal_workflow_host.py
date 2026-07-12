"""从任意 Linux x86_64 宿主启动精确的论文父编排环境.

该入口只依赖 Python 标准库和仓库内固定的 ``uv`` wheel 引导记录. 宿主
解释器不需要提供 ``pip``、``venv`` 或 ``ensurepip``. 入口先验证当前仓库是
请求提交对应的 clean detached checkout, 再创建 registry 指定的精确 CPython,
按已提交完整哈希锁准备 ``workflow_orchestrator``, 最后在该解释器内调用
``paper_workflow/cli/formal_workflow_entry.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.write_dependency_lock_review_bundle import (  # noqa: E402
    QUALIFICATION_TOOL_LOCK_RELATIVE_PATH,
    _download_qualification_tool_wheel,
    _materialize_qualification_uv_tool,
    _read_qualification_tool_lock,
    _remove_qualification_runtime_root,
    _sha256,
)
from experiments.runtime.dependency_profiles import (  # noqa: E402
    _complete_hash_lock_digest,
    _load_complete_hash_lock,
)


WORKFLOW_ORCHESTRATOR_PROFILE_ID = "workflow_orchestrator"
EXPECTED_ORCHESTRATOR_PYTHON_VERSION = "3.12.13"
FORMAL_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
GPU_WORKFLOW_NAMES = (
    "image_only_dataset",
    "mechanism_ablation",
    "external_baseline_tree_ring",
    "external_baseline_gaussian_shading",
    "external_baseline_shallow_diffuse",
    "official_reference_t2smark",
    "official_reference_tree_ring",
    "official_reference_gaussian_shading",
    "official_reference_shallow_diffuse",
)
DEFAULT_RUNTIME_ROOT = (
    Path(tempfile.gettempdir()) / "slm_wm_formal_workflow_orchestrator"
)
_SANITIZED_ENVIRONMENT_NAMES = frozenset(
    {
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "PIP_CONFIG_FILE",
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "VIRTUAL_ENV_DISABLE_PROMPT",
        "_CONDA_ROOT",
    }
)


class FormalWorkflowHostError(RuntimeError):
    """表示宿主引导或精确父环境门禁失败."""


def _normalized_machine(value: str) -> str:
    """把常见 AMD64 名称统一为 registry 使用的 x86_64."""

    normalized = value.strip().lower()
    return "x86_64" if normalized in {"amd64", "x64", "x86_64"} else normalized


def _sanitized_environment(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """移除可改变 Python、pip 或 uv 行为的宿主隐式状态."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in _SANITIZED_ENVIRONMENT_NAMES
        and not key.startswith("PIP_")
        and not key.startswith("UV_")
    }
    environment.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PYTHONNOUSERSITE": "1",
            "UV_NO_CONFIG": "1",
            "UV_NO_SYSTEM_CONFIG": "1",
        }
    )
    if overrides:
        environment.update({str(key): str(value) for key, value in overrides.items()})
    return environment


def _run_git(root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """在仓库根目录执行不经 shell 的 Git 查询."""

    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )


def validate_clean_detached_checkout(root: str | Path, repository_commit: str) -> Path:
    """要求 root 精确对应请求提交的 clean detached checkout."""

    root_path = Path(root).resolve()
    if FORMAL_COMMIT_PATTERN.fullmatch(repository_commit) is None:
        raise FormalWorkflowHostError("repository_commit 必须是40位小写 Git SHA")
    top_level = _run_git(root_path, ["rev-parse", "--show-toplevel"])
    if top_level.returncode != 0 or Path(top_level.stdout.strip()).resolve() != root_path:
        raise FormalWorkflowHostError("root 必须是当前 Git checkout 根目录")
    head = _run_git(root_path, ["rev-parse", "HEAD"])
    if head.returncode != 0 or head.stdout.strip() != repository_commit:
        raise FormalWorkflowHostError("当前 HEAD 与 repository_commit 不一致")
    symbolic = _run_git(root_path, ["symbolic-ref", "-q", "HEAD"])
    if symbolic.returncode == 0:
        raise FormalWorkflowHostError("正式运行要求 detached HEAD")
    if symbolic.returncode not in {1, 128}:
        raise FormalWorkflowHostError("无法核验 detached HEAD")
    status = _run_git(root_path, ["status", "--porcelain=v1", "--untracked-files=all"])
    if status.returncode != 0 or status.stdout:
        raise FormalWorkflowHostError("正式运行要求 clean Git 工作树")
    return root_path


def _orchestrator_profile(root: Path) -> dict[str, Any]:
    """使用标准库读取父 profile 的解释器与完整锁路径."""

    registry_path = root / "configs/dependency_profile_registry.json"
    _require_committed_file(root, registry_path)
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        profile = registry["profiles"][WORKFLOW_ORCHESTRATOR_PROFILE_ID]
        python_version = str(profile["python"]["version"])
        lock_relative_path = str(profile["complete_hash_lock_path"])
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FormalWorkflowHostError("workflow_orchestrator registry 无效") from exc
    try:
        profile_identity_valid = all(
            (
                python_version == EXPECTED_ORCHESTRATOR_PYTHON_VERSION,
                profile["python"].get("implementation") == "CPython",
                profile.get("execution_role") == "workflow_orchestration",
                profile.get("accelerator", {}).get("runtime") == "cpu",
                profile.get("platform", {}).get("operating_system") == "linux",
                profile.get("platform", {}).get("machine") == "x86_64",
            )
        )
    except AttributeError as exc:
        raise FormalWorkflowHostError("workflow_orchestrator registry 无效") from exc
    if not profile_identity_valid:
        raise FormalWorkflowHostError(
            "workflow_orchestrator 必须登记 Linux x86_64 CPython 3.12.13 CPU 父环境"
        )
    lock_source_path = root / lock_relative_path
    lock_path = lock_source_path.resolve()
    try:
        lock_path.relative_to((root / "configs/dependency_profiles").resolve())
    except ValueError as exc:
        raise FormalWorkflowHostError("workflow_orchestrator 完整锁路径越界") from exc
    if not lock_source_path.is_file():
        raise FormalWorkflowHostError("workflow_orchestrator 完整哈希锁不存在")
    _require_committed_file(root, lock_source_path)
    return {
        "python_version": python_version,
        "complete_hash_lock_path": lock_path,
        "complete_hash_lock_digest": _complete_hash_lock_digest(
            _load_complete_hash_lock(lock_path)
        ),
    }


def _require_committed_file(root: Path, path: Path) -> str:
    """要求一个引导输入是当前 HEAD 中同内容的普通文件."""

    root_path = root.resolve()
    resolved_path = path.resolve()
    try:
        relative_path = resolved_path.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise FormalWorkflowHostError("宿主引导输入路径越过仓库根目录") from exc
    if not path.is_file() or path.is_symlink():
        raise FormalWorkflowHostError("宿主引导输入必须是仓库普通文件")
    tracked = _run_git(root_path, ["cat-file", "-e", f"HEAD:{relative_path}"])
    matches_head = _run_git(
        root_path,
        ["diff", "--quiet", "HEAD", "--", relative_path],
    )
    if tracked.returncode != 0 or matches_head.returncode != 0:
        raise FormalWorkflowHostError("宿主引导输入必须来自当前 HEAD")
    return relative_path


def _execute_checked(
    command: Sequence[str],
    *,
    root: Path,
    environment: Mapping[str, str],
    operation: str,
) -> None:
    """执行一个宿主引导命令, 任一非零退出码立即闭锁."""

    completed = subprocess.run(
        list(command),
        cwd=root,
        env=dict(environment),
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise FormalWorkflowHostError(f"{operation} 失败, return_code={completed.returncode}")


def prepare_exact_orchestrator(
    *,
    root: Path,
    runtime_root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """从固定 uv wheel 创建并准备精确父编排解释器."""

    if platform.system().strip().lower() != "linux" or _normalized_machine(
        platform.machine()
    ) != "x86_64":
        raise FormalWorkflowHostError("正式 host launcher 只支持 Linux x86_64")
    profile = _orchestrator_profile(root)
    resolved_runtime_root = Path(runtime_root).expanduser().resolve()
    try:
        _remove_qualification_runtime_root(resolved_runtime_root)
    except (OSError, ValueError) as exc:
        raise FormalWorkflowHostError("无法清理父编排运行目录") from exc

    tool_lock_source_path = root / QUALIFICATION_TOOL_LOCK_RELATIVE_PATH
    tool_lock_path = tool_lock_source_path.resolve()
    try:
        _require_committed_file(root, tool_lock_source_path)
        _, wheel_url, wheel_digest = _read_qualification_tool_lock(tool_lock_path)
        _, uv_executable, _ = _materialize_qualification_uv_tool(
            wheel_url=wheel_url,
            expected_wheel_digest=wheel_digest,
            runtime_root=resolved_runtime_root,
            downloader=_download_qualification_tool_wheel,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise FormalWorkflowHostError("固定 uv wheel 引导失败") from exc

    managed_python_root = resolved_runtime_root / "managed_pythons"
    environment_root = resolved_runtime_root / WORKFLOW_ORCHESTRATOR_PROFILE_ID
    python_executable = environment_root / "bin/python"
    cache_root = resolved_runtime_root / "cache"
    uv_environment = _sanitized_environment(
        {
            "UV_CACHE_DIR": str(cache_root / "uv"),
            "UV_PYTHON_INSTALL_DIR": str(managed_python_root),
        }
    )
    python_version = str(profile["python_version"])
    commands = (
        ("uv_version", [str(uv_executable), "--version"], uv_environment),
        (
            "python_install",
            [
                str(uv_executable),
                "python",
                "install",
                python_version,
                "--install-dir",
                str(managed_python_root),
            ],
            uv_environment,
        ),
        (
            "orchestrator_venv",
            [
                str(uv_executable),
                "venv",
                "--clear",
                "--python",
                python_version,
                "--managed-python",
                str(environment_root),
            ],
            uv_environment,
        ),
        ("orchestrator_ensurepip", [str(python_executable), "-m", "ensurepip"], uv_environment),
        (
            "orchestrator_hash_install",
            [
                str(python_executable),
                "-m",
                "pip",
                "install",
                "--require-hashes",
                "--only-binary=:all:",
                "-r",
                str(profile["complete_hash_lock_path"]),
            ],
            _sanitized_environment({"PIP_CACHE_DIR": str(cache_root / "pip")}),
        ),
        ("orchestrator_pip_check", [str(python_executable), "-m", "pip", "check"], uv_environment),
        (
            "orchestrator_python_inspection",
            [
                str(python_executable),
                "-c",
                (
                    "import platform,sys;"
                    f"raise SystemExit(0 if platform.python_version() == '{python_version}' "
                    "and sys.flags.isolated == 1 else 1)"
                ),
            ],
            uv_environment,
        ),
    )
    for operation, command, environment in commands:
        if operation == "orchestrator_python_inspection":
            command = [command[0], "-I", *command[1:]]
        _execute_checked(command, root=root, environment=environment, operation=operation)
    if not python_executable.is_file():
        raise FormalWorkflowHostError("精确父编排 Python executable 不存在")
    return python_executable, {
        "profile_id": WORKFLOW_ORCHESTRATOR_PROFILE_ID,
        "python_version": python_version,
        "complete_hash_lock_digest": profile["complete_hash_lock_digest"],
        "python_executable": str(python_executable),
        "python_executable_sha256": _sha256(python_executable),
    }


def build_child_command(
    arguments: argparse.Namespace,
    python_executable: Path,
    root: Path,
    bootstrap_identity: Mapping[str, Any],
) -> list[str]:
    """构造精确父解释器中的唯一 Colab/服务器子入口命令."""

    command = [
        str(python_executable),
        "-I",
        str(root / "paper_workflow/cli/formal_workflow_entry.py"),
        arguments.operation,
        "--root",
        str(root),
        "--repository-commit",
        arguments.repository_commit,
        "--paper-run-name",
        arguments.paper_run_name,
        "--result-path",
        arguments.result_path,
        "--orchestrator-profile-id",
        str(bootstrap_identity["profile_id"]),
        "--orchestrator-python-version",
        str(bootstrap_identity["python_version"]),
        "--orchestrator-lock-digest",
        str(bootstrap_identity["complete_hash_lock_digest"]),
        "--orchestrator-python-executable",
        str(bootstrap_identity["python_executable"]),
        "--orchestrator-python-executable-sha256",
        str(bootstrap_identity["python_executable_sha256"]),
    ]
    if arguments.operation == "gpu":
        command.extend(["--workflow", arguments.workflow])
        if arguments.persistent_output_dir:
            command.extend(["--persistent-output-dir", arguments.persistent_output_dir])
    else:
        command.extend(
            [
                "--package-search-root",
                arguments.package_search_root,
                "--complete-output-dir",
                arguments.complete_output_dir,
            ]
        )
        if arguments.dry_run:
            command.append("--dry-run")
    return command


def launch_formal_workflow(arguments: argparse.Namespace) -> int:
    """验证 checkout、准备父环境并把控制权交给精确子解释器."""

    if sys.flags.isolated != 1:
        raise FormalWorkflowHostError("正式 host launcher 必须使用 python -I 调用")
    root = validate_clean_detached_checkout(arguments.root, arguments.repository_commit)
    runtime_root = Path(arguments.runtime_root) / arguments.repository_commit
    python_executable, bootstrap_identity = prepare_exact_orchestrator(
        root=root,
        runtime_root=runtime_root,
    )
    child_command = build_child_command(
        arguments,
        python_executable,
        root,
        bootstrap_identity,
    )
    child_environment = _sanitized_environment(
        {
            "PATH": str(python_executable.parent) + os.pathsep + os.environ.get("PATH", ""),
        }
    )
    completed = subprocess.run(
        child_command,
        cwd=root,
        env=child_environment,
        check=False,
        shell=False,
    )
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    """构造 GPU workflow 与 CPU 闭合共用的宿主入口参数."""

    parser = argparse.ArgumentParser(
        description="从固定 uv wheel 创建精确 workflow_orchestrator 并执行论文入口."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument(
        "--runtime-root",
        default=str(DEFAULT_RUNTIME_ROOT),
        help="仅保存临时 managed Python 和隔离父环境的本地目录.",
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    gpu = subparsers.add_parser("gpu")
    gpu.add_argument("--workflow", required=True, choices=GPU_WORKFLOW_NAMES)
    gpu.add_argument(
        "--paper-run-name",
        required=True,
        choices=("probe_paper", "pilot_paper", "full_paper"),
    )
    gpu.add_argument("--persistent-output-dir", default="")
    gpu.add_argument("--result-path", required=True)
    closure = subparsers.add_parser("closure")
    closure.add_argument(
        "--paper-run-name",
        required=True,
        choices=("probe_paper", "pilot_paper", "full_paper"),
    )
    closure.add_argument("--package-search-root", required=True)
    closure.add_argument("--complete-output-dir", required=True)
    closure.add_argument("--result-path", required=True)
    closure.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行宿主入口并用非零退出码表达任一闭锁原因."""

    arguments = build_parser().parse_args(argv)
    try:
        return launch_formal_workflow(arguments)
    except (FormalWorkflowHostError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "decision": "fail",
                    "failure_reasons": [type(exc).__name__],
                    "diagnostic_message": str(exc),
                    "supports_paper_claim": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
