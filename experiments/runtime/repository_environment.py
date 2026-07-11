"""仓库实验运行环境与摘要工具。

该模块保存不依赖 Notebook 的通用运行环境能力, 供核心实验 runner、完整论文
实验 runner 与 Colab 包装层共同复用。这样可以避免正式实验逻辑为了读取 Git
版本、依赖版本或文件摘要而反向依赖 `paper_workflow/`。
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

from main.core.digest import build_stable_digest

COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND = (
    "%pip install -q --upgrade diffusers transformers accelerate safetensors sentencepiece protobuf huggingface_hub"
)
RUNTIME_ENVIRONMENT_PACKAGES = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
    "sentencepiece",
    "protobuf",
    "numpy",
    "pillow",
    "lpips",
    "torchvision",
)


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定、可读文本。"""

    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def json_line(value: Any) -> str:
    """把 JSON 兼容对象转为 JSONL 单行文本。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def resolve_code_version(root_path: Path) -> str:
    """读取当前 Git 提交标识, 不可用时返回稳定降级值。

    此函数属于通用工程写法: 任何实验产物 manifest 都需要记录代码版本,
    因此它必须位于 Notebook 无关的运行环境工具层。
    """

    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--short"],
            cwd=root_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "git_version_unavailable"
    commit_id = commit_result.stdout.strip()
    if not commit_id:
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


def read_package_version(package_name: str) -> str:
    """读取已安装 Python 包版本, 未安装时返回稳定的审计值。"""

    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "not_installed"


def build_runtime_environment_report(
    torch_module: Any | None = None,
    install_command: str = COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND,
) -> dict[str, Any]:
    """构造真实运行环境快照, 用于复现依赖组合与 GPU 条件。

    `install_command` 默认保持 Colab 动态升级命令, 但服务器 runner 可以传入
    自身的环境准备命令, 因而该函数不再绑定到 Colab helper。
    """

    package_versions = {package_name: read_package_version(package_name) for package_name in RUNTIME_ENVIRONMENT_PACKAGES}
    cuda_available = None
    cuda_version = None
    gpu_name = ""
    device_count = 0
    if torch_module is not None:
        package_versions["torch"] = str(getattr(torch_module, "__version__", package_versions["torch"]))
        cuda_available = bool(torch_module.cuda.is_available())
        cuda_version = getattr(torch_module.version, "cuda", None)
        device_count = int(torch_module.cuda.device_count()) if cuda_available else 0
        gpu_name = torch_module.cuda.get_device_name(0) if cuda_available and device_count else ""
    return {
        "dependency_mode": "colab_dynamic_upgrade",
        "manual_version_pins": False,
        "pip_install_command": install_command,
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "package_versions": package_versions,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "device_count": device_count,
        "gpu_name": gpu_name,
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
