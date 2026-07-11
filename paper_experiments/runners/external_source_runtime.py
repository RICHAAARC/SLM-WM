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
CUDA_INSPECTION_PROGRAM = """\
import json
import sys

payload = {
    "python_executable": sys.executable,
    "torch_available": False,
    "cuda_available": False,
    "device": "cpu",
    "torch_version": "",
    "torch_cuda_version": "",
    "device_count": 0,
    "gpu_name": "",
}
exit_code = 0
try:
    import torch
except Exception as error:
    sys.stderr.write(type(error).__name__ + ":" + str(error))
    exit_code = 2
else:
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    payload.update(
        {
            "torch_available": True,
            "cuda_available": cuda_available,
            "device": "cuda" if cuda_available else "cpu",
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda or ""),
            "device_count": device_count,
            "gpu_name": (
                str(torch.cuda.get_device_name(0))
                if cuda_available and device_count > 0
                else ""
            ),
        }
    )
    if sys.argv[1] == "1" and not cuda_available:
        exit_code = 3
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
raise SystemExit(exit_code)
"""


def stable_json_payload_digest(value: Any) -> str:
    """计算跨报告引用使用的规范 JSON SHA-256."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bind_successful_official_command_execution_evidence(
    result: dict[str, Any],
    *,
    baseline_id: str,
    command: list[str],
    working_directory: str | Path,
    dependency_python_executable: str | Path,
    cuda_inspection_report: dict[str, Any],
) -> dict[str, Any]:
    """把成功官方命令绑定到隔离解释器, 工作目录与 CUDA 核验.

    该函数属于通用跨进程证据写法.官方 runner 仍负责构造各自 argv, 此处只
    统一证明实际命令的第一个参数, CUDA 探针解释器和依赖解释器完全一致.
    """

    resolved_working_directory = Path(working_directory).expanduser().absolute()
    requested_python = Path(str(dependency_python_executable)).expanduser()
    if not requested_python.is_absolute():
        requested_python = resolved_working_directory / requested_python
    requested_python = requested_python.absolute()
    reported_python = Path(
        str(cuda_inspection_report.get("python_executable", ""))
    ).expanduser()
    command_python = Path(str(command[0])).expanduser() if command else Path("")
    evidence_ready = all(
        (
            baseline_id in {"tree_ring", "gaussian_shading", "shallow_diffuse"},
            result.get("official_command_requested") is True,
            result.get("return_code") == 0,
            result.get("official_command") == command,
            bool(command),
            command_python.is_absolute(),
            command_python.absolute() == requested_python,
            reported_python.is_absolute(),
            reported_python.absolute() == requested_python,
            cuda_inspection_report.get("decision") == "pass",
            cuda_inspection_report.get("failure_reasons") == [],
            cuda_inspection_report.get("return_code") == 0,
            cuda_inspection_report.get("torch_available") is True,
            cuda_inspection_report.get("cuda_available") is True,
            cuda_inspection_report.get("device") == "cuda",
            cuda_inspection_report.get("supports_paper_claim") is False,
        )
    )
    device_count = cuda_inspection_report.get("device_count")
    evidence_ready = evidence_ready and (
        isinstance(device_count, int)
        and not isinstance(device_count, bool)
        and device_count > 0
        and bool(str(cuda_inspection_report.get("gpu_name", "")).strip())
        and bool(str(cuda_inspection_report.get("torch_version", "")).strip())
    )
    if not evidence_ready:
        raise RuntimeError("官方命令无法绑定到已通过的隔离 CUDA 解释器")
    bound = dict(result)
    bound.update(
        {
            "report_schema": "official_reference_command_execution_report",
            "schema_version": 1,
            "baseline_id": baseline_id,
            "official_command_working_directory": str(
                resolved_working_directory
            ),
            "dependency_python_executable": str(requested_python),
            "dependency_python_executable_sha256": str(
                cuda_inspection_report.get("python_executable_sha256", "")
            ),
            "cuda_inspection_report_digest": stable_json_payload_digest(
                cuda_inspection_report
            ),
            "official_command_execution_evidence_ready": True,
            "failure_reasons": [],
            "supports_paper_claim": False,
        }
    )
    return bound


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


def inspect_cuda_with_python_executable(
    python_executable: str | Path,
    *,
    require_cuda: bool,
    cwd: Path,
    timeout_seconds: int = 60,
    command_runner: Any | None = None,
) -> dict[str, Any]:
    """用指定隔离 Python 子进程核验 torch 与 CUDA 并保存完整进程证据.

    该函数属于通用父子进程隔离写法.父编排解释器只构造 argv 和解析 JSON,
    不导入 torch; official-reference runner 因而只依赖其已验证科学解释器.
    """

    requested_python = Path(str(python_executable)).expanduser()
    resolved_python = (
        requested_python.absolute()
        if requested_python.is_absolute()
        else (cwd / requested_python).absolute()
    ) if str(python_executable) else None
    command = (
        [
            str(resolved_python),
            "-c",
            CUDA_INSPECTION_PROGRAM,
            "1" if require_cuda else "0",
        ]
        if resolved_python is not None
        else []
    )
    report: dict[str, Any] = {
        "command": command,
        "working_directory": str(cwd.expanduser().absolute()),
        "return_code": None,
        "stdout": "",
        "stderr": "",
        "python_executable": "" if resolved_python is None else str(resolved_python),
        "python_executable_sha256": "",
        "torch_available": False,
        "cuda_available": False,
        "device": "cpu",
        "torch_version": "",
        "torch_cuda_version": "",
        "device_count": 0,
        "gpu_name": "",
        "decision": "fail",
        "failure_reasons": [],
        "supports_paper_claim": False,
    }
    if resolved_python is None or not resolved_python.is_file():
        report["failure_reasons"] = ["isolated_python_executable_missing"]
        return report
    report["python_executable_sha256"] = _file_sha256(resolved_python)

    runner = command_runner or run_command
    try:
        result = runner(
            command,
            cwd=cwd,
            timeout_seconds=int(timeout_seconds),
        )
    except Exception as error:
        report["stderr"] = f"{type(error).__name__}:{error}"
        report["failure_reasons"] = ["cuda_inspection_command_failed"]
        return report
    report.update(
        {
            "return_code": int(result.get("return_code", -1)),
            "stdout": str(result.get("stdout") or ""),
            "stderr": str(result.get("stderr") or ""),
        }
    )
    output_lines = [line for line in report["stdout"].splitlines() if line.strip()]
    if not output_lines:
        report["failure_reasons"] = ["cuda_inspection_output_missing"]
        return report
    try:
        payload = json.loads(output_lines[-1])
    except json.JSONDecodeError:
        report["failure_reasons"] = ["cuda_inspection_output_invalid"]
        return report
    if not isinstance(payload, dict):
        report["failure_reasons"] = ["cuda_inspection_output_invalid"]
        return report
    reported_python = Path(str(payload.get("python_executable", "")))
    if (
        not reported_python.is_absolute()
        or reported_python.absolute() != resolved_python
    ):
        report["failure_reasons"] = ["cuda_inspection_python_identity_mismatch"]
        return report
    runtime_fields = (
        "torch_available",
        "cuda_available",
        "device",
        "torch_version",
        "torch_cuda_version",
        "device_count",
        "gpu_name",
    )
    report.update({field_name: payload.get(field_name) for field_name in runtime_fields})
    if report["return_code"] != 0:
        report["failure_reasons"] = [
            "cuda_required_but_unavailable"
            if require_cuda and report.get("torch_available") is True
            else "torch_runtime_unavailable"
        ]
        return report
    if report.get("torch_available") is not True:
        report["failure_reasons"] = ["torch_runtime_unavailable"]
        return report
    if require_cuda and report.get("cuda_available") is not True:
        report["failure_reasons"] = ["cuda_required_but_unavailable"]
        return report
    report["decision"] = "pass"
    report["failure_reasons"] = []
    return report


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
