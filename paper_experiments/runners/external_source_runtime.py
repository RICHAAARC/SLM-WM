"""外部 baseline runner 共用的源码登记与命令执行工具。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from experiments.runtime.progress import (
    PROGRESS_EVENT_ENV_NAME,
    call_runner_with_progress_status,
    run_quiet_subprocess_with_progress,
)


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
