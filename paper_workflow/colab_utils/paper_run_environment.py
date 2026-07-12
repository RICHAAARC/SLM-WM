"""为 Colab 会话包装内层正式工作流环境配置."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paper_workflow.notebook_utils.notebook_runtime import (
    mark_notebook_runtime_start,
)
from scripts.formal_workflow_environment import (
    _resolve_paper_run_name,
    configure_formal_workflow_environment,
)


def configure_paper_run_environment(
    workflow_name: str,
    *,
    baseline_id: str = "",
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    """记录 Notebook 起点并复用可独立服务器执行的内层配置."""

    mark_notebook_runtime_start(
        workflow_name=workflow_name,
        baseline_id=baseline_id,
        source="configure_paper_run_environment",
    )
    return configure_formal_workflow_environment(
        workflow_name,
        baseline_id=baseline_id,
        repository_root=repository_root,
    )


__all__ = [
    "_resolve_paper_run_name",
    "configure_paper_run_environment",
]
