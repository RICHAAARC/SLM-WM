"""记录 Colab workflow 所需的轻量依赖状态。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from importlib.util import find_spec
from typing import Any

COLAB_DYNAMIC_DEPENDENCY_INSTALL_COMMAND = (
    "%pip install -q --upgrade diffusers transformers accelerate safetensors "
    "sentencepiece protobuf huggingface_hub"
)

WORKFLOW_DEPENDENCIES = (
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "sentencepiece",
    "protobuf",
    "huggingface_hub",
)
DEPENDENCY_IMPORT_NAMES = {
    "protobuf": "google.protobuf",
    "pillow": "PIL",
    "open_clip_torch": "open_clip",
    "scikit-learn": "sklearn",
    "torch-fidelity": "torch_fidelity",
}
NOTEBOOK_DEPENDENCY_PROFILES = {
    "semantic_watermark_image_only": WORKFLOW_DEPENDENCIES
    + (
        "numpy",
        "pillow",
        "scipy",
        "torchvision",
        "torch-fidelity",
    ),
    "sd35_runtime": WORKFLOW_DEPENDENCIES
    + (
        "numpy",
        "tokenizers",
        "scipy",
        "torchvision",
    ),
    "dataset_level_quality": (
        "torch",
        "torchvision",
        "scipy",
        "pillow",
    ),
    "external_baseline_method_faithful": WORKFLOW_DEPENDENCIES
    + (
        "open_clip_torch",
        "scikit-learn",
        "scipy",
        "pandas",
        "datasets",
        "tqdm",
    ),
    "official_reference_light": (
        "packaging",
        "huggingface_hub",
        "torch",
    ),
    "official_reference_t2smark": WORKFLOW_DEPENDENCIES
    + (
        "open_clip_torch",
        "torch",
    ),
}


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
    import_name = DEPENDENCY_IMPORT_NAMES.get(dependency_name, dependency_name)
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


def build_notebook_dependency_report(profile_name: str) -> dict[str, Any]:
    """按 Notebook 入口职责生成统一依赖诊断报告。

    Notebook 只传入语义化 profile 名称, 具体依赖清单集中维护在本模块。
    这样后续依赖增删只需修改 repository helper, 不需要同步修改多个
    Colab Notebook 入口。
    """

    if profile_name not in NOTEBOOK_DEPENDENCY_PROFILES:
        raise ValueError(f"unknown_notebook_dependency_profile:{profile_name}")
    report = build_dependency_report(NOTEBOOK_DEPENDENCY_PROFILES[profile_name])
    return {
        **report,
        "dependency_profile_name": profile_name,
    }
