"""外部 baseline runner 共用的源码登记与命令执行工具。"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any

from experiments.runtime.progress import (
    PROGRESS_EVENT_ENV_NAME,
    call_runner_with_progress_status,
    run_quiet_subprocess_with_progress,
)
from main.core.digest import build_stable_digest


DEFAULT_SOURCE_REGISTRY_PATH = "external_baseline/source_registry.json"


def load_baseline_registry_item(root_path: Path, baseline_id: str) -> dict[str, Any]:
    """从受治理登记表读取指定外部方法的固定源码 revision。"""

    registry_path = root_path / DEFAULT_SOURCE_REGISTRY_PATH
    registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    for item in registry.get("baseline_sources", ()):
        if item.get("baseline_id") == baseline_id:
            return dict(item)
    raise KeyError(f"baseline_registry_item_missing:{baseline_id}")


def normalize_repository_url(repository_url: str) -> str:
    """把 GitHub SSH 地址转换为无需 SSH key 的 HTTPS 地址。"""

    if repository_url.startswith("git@github.com:"):
        return "https://github.com/" + repository_url.split(":", 1)[1]
    return repository_url


def _normalized_repository_identity(repository_url: str) -> str:
    """把 Git 远端地址归一化为可比较的 HTTPS 仓库身份。"""

    normalized = normalize_repository_url(repository_url).rstrip("/")
    return normalized[:-4] if normalized.endswith(".git") else normalized


def _run_git_checked(source_dir: Path, command: list[str]) -> subprocess.CompletedProcess[str]:
    """在固定外部源码目录执行必须成功的 Git 命令。"""

    return subprocess.run(
        ["git", *command],
        cwd=source_dir,
        check=True,
        capture_output=True,
        text=True,
    )


def prepare_registered_source_checkout(
    root_path: Path,
    baseline_id: str,
    source_dir: Path,
) -> dict[str, Any]:
    """把外部源码缓存恢复到登记提交的 clean detached 工作树。

    ``source/`` 是由 runner 管理的第三方缓存, 不允许保存用户修改。每次正式运行
    都先恢复登记提交, 再由当前仓库代码应用确定性兼容补丁。
    """

    resolved_root = root_path.resolve()
    resolved_source = source_dir.resolve()
    resolved_source.relative_to(resolved_root)
    if not (resolved_source / ".git").exists():
        raise RuntimeError(f"{baseline_id} 外部源码目录不是可验证的 Git checkout")
    registry_item = load_baseline_registry_item(resolved_root, baseline_id)
    expected_commit = str(registry_item.get("official_repository_commit", ""))
    expected_url = str(registry_item.get("official_repository_url", ""))
    if len(expected_commit) != 40:
        raise RuntimeError(f"{baseline_id} 外部源码登记提交无效")
    actual_url = _run_git_checked(resolved_source, ["remote", "get-url", "origin"]).stdout.strip()
    if _normalized_repository_identity(actual_url) != _normalized_repository_identity(expected_url):
        raise RuntimeError(f"{baseline_id} 外部源码远端与登记仓库不一致")
    _run_git_checked(resolved_source, ["cat-file", "-e", f"{expected_commit}^{{commit}}"])
    _run_git_checked(resolved_source, ["checkout", "--detach", expected_commit])
    _run_git_checked(resolved_source, ["reset", "--hard", expected_commit])
    _run_git_checked(resolved_source, ["clean", "-fdx"])
    actual_commit = _run_git_checked(resolved_source, ["rev-parse", "HEAD"]).stdout.strip()
    status = _run_git_checked(
        resolved_source,
        ["status", "--porcelain=v1", "--untracked-files=all"],
    ).stdout.strip()
    if actual_commit != expected_commit or status:
        raise RuntimeError(f"{baseline_id} 外部源码无法恢复到登记 clean commit")
    return {
        "official_repository_url": _normalized_repository_identity(expected_url),
        "official_repository_commit": expected_commit,
        "source_head_commit": actual_commit,
        "source_remote_url": _normalized_repository_identity(actual_url),
        "source_base_worktree_clean": True,
        "source_identity_ready": True,
    }


def _file_sha256(path: Path) -> str:
    """计算固定源码文件摘要。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_registered_source_patch_evidence(
    root_path: Path,
    baseline_id: str,
    source_dir: Path,
    expected_modified_paths: tuple[str, ...],
) -> dict[str, Any]:
    """验证确定性补丁只修改允许文件并构造精确工作树摘要。"""

    resolved_source = source_dir.resolve()
    resolved_source.relative_to(root_path.resolve())
    registry_item = load_baseline_registry_item(root_path.resolve(), baseline_id)
    expected_commit = str(registry_item.get("official_repository_commit", ""))
    actual_commit = _run_git_checked(resolved_source, ["rev-parse", "HEAD"]).stdout.strip()
    if actual_commit != expected_commit:
        raise RuntimeError(f"{baseline_id} 补丁工作树 HEAD 与登记提交不一致")
    modified_paths = tuple(
        line.strip().replace("\\", "/")
        for line in _run_git_checked(
            resolved_source,
            ["diff", "--name-only", "HEAD", "--"],
        ).stdout.splitlines()
        if line.strip()
    )
    if set(modified_paths) != set(expected_modified_paths):
        raise RuntimeError(f"{baseline_id} 确定性源码补丁修改文件集合不一致")
    untracked = _run_git_checked(
        resolved_source,
        ["ls-files", "--others", "--exclude-standard"],
    ).stdout.strip()
    if untracked:
        raise RuntimeError(f"{baseline_id} 补丁工作树包含未跟踪文件")
    diff_bytes = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", *expected_modified_paths],
        cwd=resolved_source,
        check=True,
        capture_output=True,
    ).stdout
    patch_sha256 = hashlib.sha256(diff_bytes).hexdigest()
    patched_source_sha256 = {
        relative_path: _file_sha256(resolved_source / relative_path)
        for relative_path in expected_modified_paths
    }
    source_worktree_digest = build_stable_digest(
        {
            "official_repository_commit": expected_commit,
            "source_patch_sha256": patch_sha256,
            "patched_source_sha256": patched_source_sha256,
        }
    )
    return {
        "official_repository_commit": expected_commit,
        "source_modified_paths": list(modified_paths),
        "source_patch_sha256": patch_sha256,
        "patched_source_sha256": patched_source_sha256,
        "source_worktree_digest": source_worktree_digest,
        "source_worktree_exact": True,
    }


def ensure_cuda_if_requested(require_cuda: bool) -> dict[str, Any]:
    """在正式 GPU runner 边界集中核验 PyTorch 与 CUDA。"""

    try:
        import torch
    except Exception as error:  # pragma: no cover - 本地轻量测试不依赖 torch
        if require_cuda:
            raise RuntimeError("正式外部 baseline 运行要求可导入 PyTorch") from error
        return {"torch_available": False, "cuda_available": False, "device": "cpu"}
    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise RuntimeError("正式外部 baseline 运行要求 CUDA 可用")
    return {
        "torch_available": True,
        "cuda_available": cuda_available,
        "device": "cuda" if cuda_available else "cpu",
        "torch_version": str(torch.__version__),
    }


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    progress: object | None = None,
    progress_profile: str = "",
    child_progress_path: str | Path | None = None,
) -> dict[str, Any]:
    """执行显式 argv 命令并保留标准输出与错误输出。"""

    command_env = None
    if child_progress_path is not None:
        command_env = dict(os.environ)
        command_env[PROGRESS_EVENT_ENV_NAME] = str(child_progress_path)
    completed = run_quiet_subprocess_with_progress(
        command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        progress=progress,
        progress_profile=progress_profile or "operation=argv_command",
        env=command_env,
        heartbeat_seconds=15.0,
        child_progress_path=child_progress_path,
    )
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_command_with_progress_status(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    progress: object | None = None,
    progress_profile: str = "",
    child_progress_path: str | Path | None = None,
    command_runner: Any = run_command,
) -> dict[str, Any]:
    """兼容真实进度 runner 与只接受最小参数的测试替身。"""

    if child_progress_path is None:
        return call_runner_with_progress_status(
            command_runner,
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            progress=progress,
            progress_profile=progress_profile,
        )
    try:
        return command_runner(
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            progress=progress,
            progress_profile=progress_profile,
            child_progress_path=child_progress_path,
        )
    except TypeError:
        return command_runner(command, cwd=cwd, timeout_seconds=timeout_seconds)
