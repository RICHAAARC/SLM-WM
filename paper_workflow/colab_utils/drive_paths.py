"""集中管理 Colab 与 Drive workflow 的路径约定。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_WORKFLOW_NAME = "colab_drive_workflow"
DEFAULT_LOCAL_OUTPUT_DIR = Path("outputs/colab_drive_workflow")
DEFAULT_DRIVE_ROOT = Path("outputs/colab_drive_workflow/drive_mirror")


@dataclass(frozen=True)
class DriveWorkflowPaths:
    """记录一次 Colab Drive workflow 所需的本地与 Drive 路径。"""

    repository_root: Path
    local_output_dir: Path
    drive_root: Path
    workflow_name: str = DEFAULT_WORKFLOW_NAME

    @property
    def drive_workflow_dir(self) -> Path:
        """返回 Drive 中保存 workflow manifest 与镜像文件的目录。"""
        return self.drive_root / self.workflow_name

    @property
    def drive_local_output_dir(self) -> Path:
        """返回 Drive 中保存 outputs 镜像文件的目录。"""
        return self.drive_workflow_dir / "local_outputs"

    def to_dict(self) -> dict[str, str]:
        """转为可写入 JSON 的路径摘要。"""
        return {
            "repository_root": self.repository_root.as_posix(),
            "local_output_dir": self.local_output_dir.as_posix(),
            "drive_root": self.drive_root.as_posix(),
            "drive_workflow_dir": self.drive_workflow_dir.as_posix(),
            "drive_local_output_dir": self.drive_local_output_dir.as_posix(),
            "workflow_name": self.workflow_name,
        }


def resolve_under_root(root: Path, path: str | Path) -> Path:
    """把相对路径解析到仓库根目录下, 绝对路径保持原语义。"""
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def ensure_output_path(root: Path, path: Path) -> Path:
    """确保持久化本地输出位于 outputs 目录下。"""
    outputs_root = (root / "outputs").resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("本地持久化输出必须位于 outputs/ 目录下") from exc
    return resolved_path


def build_drive_workflow_paths(
    root: str | Path = ".",
    local_output_dir: str | Path = DEFAULT_LOCAL_OUTPUT_DIR,
    drive_root: str | Path = DEFAULT_DRIVE_ROOT,
    workflow_name: str = DEFAULT_WORKFLOW_NAME,
) -> DriveWorkflowPaths:
    """根据仓库根目录和 Drive 根目录构造 workflow 路径对象。"""
    repository_root = Path(root).resolve()
    resolved_local_output_dir = ensure_output_path(
        repository_root,
        resolve_under_root(repository_root, local_output_dir),
    )
    resolved_drive_root = resolve_under_root(repository_root, Path(drive_root).expanduser())
    return DriveWorkflowPaths(
        repository_root=repository_root,
        local_output_dir=resolved_local_output_dir,
        drive_root=resolved_drive_root,
        workflow_name=workflow_name,
    )


def path_payload(paths: DriveWorkflowPaths) -> dict[str, Any]:
    """返回用于 manifest metadata 的路径摘要。"""
    return paths.to_dict()
