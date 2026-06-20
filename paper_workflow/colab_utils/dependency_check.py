"""记录 Colab workflow 所需的轻量依赖状态。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from importlib.util import find_spec
from typing import Any

from paper_workflow.colab_utils.sd_runtime_cold_start import COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND

WORKFLOW_DEPENDENCIES = (
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "sentencepiece",
    "protobuf",
    "huggingface_hub",
)


@dataclass(frozen=True)
class DependencyStatus:
    """描述一个 Python 依赖是否可导入以及当前版本。"""

    dependency_name: str
    module_available: bool
    installed_version: str

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的依赖记录。"""
        return asdict(self)


def read_dependency_version(dependency_name: str) -> str:
    """读取依赖版本; 若当前环境没有该包则返回 unavailable。"""
    try:
        return importlib_metadata.version(dependency_name)
    except importlib_metadata.PackageNotFoundError:
        return "unavailable"


def inspect_dependency(dependency_name: str) -> DependencyStatus:
    """检查单个依赖是否存在。"""
    import_name = "google.protobuf" if dependency_name == "protobuf" else dependency_name
    return DependencyStatus(
        dependency_name=dependency_name,
        module_available=find_spec(import_name) is not None,
        installed_version=read_dependency_version(dependency_name),
    )


def build_dependency_report(
    dependency_names: tuple[str, ...] = WORKFLOW_DEPENDENCIES,
) -> dict[str, Any]:
    """生成 Colab workflow 依赖快照, 仅记录状态, 不主动安装。"""
    statuses = tuple(inspect_dependency(name) for name in dependency_names)
    missing = tuple(status.dependency_name for status in statuses if not status.module_available)
    return {
        "dependency_decision": "pass" if not missing else "unsupported",
        "dependency_mode": "colab_dynamic_upgrade",
        "dependency_count": len(statuses),
        "missing_dependency_count": len(missing),
        "unsupported_reasons": [f"missing:{name}" for name in missing],
        "pip_install_command": COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND,
        "package_versions": {
            status.dependency_name: status.installed_version for status in statuses
        },
        "dependencies": [status.to_dict() for status in statuses],
        "supports_paper_claim": False,
    }
