"""Colab Notebook 入口层通用 helper。

该模块只封装 Notebook 运行入口的重复样板: 统一归档命名、短提交读取、
Drive 输出目录解析和 workflow 打包调度。正式方法逻辑、攻击协议、
baseline 适配和论文结果闭合仍由各自 repository module 实现。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from experiments.runtime.archive_naming import (
    build_workflow_archive_name,
)
from experiments.protocol.paper_run_config import build_paper_run_config
from paper_workflow.notebook_utils.notebook_runtime import write_notebook_runtime_report

WORKFLOW_DRIVE_OUTPUT_ENV_KEYS = {
    "external_baseline_method_faithful": "SLM_WM_EXTERNAL_BASELINE_DRIVE_OUTPUT_DIR",
    "official_reference_tree_ring": "SLM_WM_TREE_RING_OFFICIAL_DRIVE_OUTPUT_DIR",
    "official_reference_gaussian_shading": "SLM_WM_GAUSSIAN_SHADING_OFFICIAL_DRIVE_OUTPUT_DIR",
    "official_reference_shallow_diffuse": "SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_DRIVE_OUTPUT_DIR",
    "official_reference_t2smark": "SLM_WM_T2SMARK_FORMAL_DRIVE_OUTPUT_DIR",
}
def _run_function_for_workflow(workflow_name: str) -> Callable[..., Any]:
    """延迟解析正式 workflow runner, 使 Notebook 只调用统一入口。"""

    if workflow_name == "official_reference_tree_ring":
        from paper_experiments.runners.tree_ring_official_reference import run_default_tree_ring_official_reference_plan

        return run_default_tree_ring_official_reference_plan
    if workflow_name == "official_reference_gaussian_shading":
        from paper_experiments.runners.gaussian_shading_official_reference import (
            run_default_gaussian_shading_official_reference_plan,
        )

        return run_default_gaussian_shading_official_reference_plan
    if workflow_name == "official_reference_shallow_diffuse":
        from paper_experiments.runners.shallow_diffuse_official_reference import (
            run_default_shallow_diffuse_official_reference_plan,
        )

        return run_default_shallow_diffuse_official_reference_plan
    raise ValueError(f"unknown_notebook_workflow:{workflow_name}")


def run_workflow(*, root: str | Path = ".", workflow_name: str) -> Any:
    """通过统一薄入口运行正式 repository workflow。"""

    if workflow_name in {
        "external_baseline_method_faithful",
        "official_reference_t2smark",
    }:
        from paper_experiments.runners.isolated_scientific_workflow import (
            run_isolated_scientific_workflow,
        )

        return run_isolated_scientific_workflow(
            root=root,
            workflow_name=workflow_name,
        )
    return _run_function_for_workflow(workflow_name)(root=root)


def resolve_drive_output_dir(workflow_name: str, drive_output_dir: str | None = None) -> str | None:
    """从显式参数或约定环境变量解析 Drive 输出目录。"""

    if drive_output_dir:
        return drive_output_dir
    env_key = WORKFLOW_DRIVE_OUTPUT_ENV_KEYS.get(workflow_name)
    if not env_key:
        return None
    return os.environ.get(env_key)


def _package_function_for_workflow(workflow_name: str) -> Callable[..., Any]:
    """延迟导入 workflow 打包函数, 避免 Notebook 入口加载无关重依赖。"""

    if workflow_name == "external_baseline_method_faithful":
        from paper_experiments.runners.external_baseline_method_faithful import (
            package_external_baseline_method_faithful_outputs,
        )

        return package_external_baseline_method_faithful_outputs
    if workflow_name == "official_reference_tree_ring":
        from paper_experiments.runners.tree_ring_official_reference import package_tree_ring_official_reference_outputs

        return package_tree_ring_official_reference_outputs
    if workflow_name == "official_reference_gaussian_shading":
        from paper_experiments.runners.gaussian_shading_official_reference import (
            package_gaussian_shading_official_reference_outputs,
        )

        return package_gaussian_shading_official_reference_outputs
    if workflow_name == "official_reference_shallow_diffuse":
        from paper_experiments.runners.shallow_diffuse_official_reference import (
            package_shallow_diffuse_official_reference_outputs,
        )

        return package_shallow_diffuse_official_reference_outputs
    raise ValueError(f"unknown_notebook_workflow:{workflow_name}")


def package_workflow_outputs(
    *,
    root: str | Path = ".",
    workflow_name: str,
    drive_output_dir: str | None = None,
    baseline_id: str | None = None,
) -> Any:
    """统一执行单一 workflow 的 archive 打包与 Drive 镜像。"""

    archive_name = build_workflow_archive_name(workflow_name, root=root, baseline_id=baseline_id)
    resolved_drive_output_dir = resolve_drive_output_dir(workflow_name, drive_output_dir)
    resolved_baseline_id = baseline_id or os.environ.get("SLM_WM_PRIMARY_BASELINE_ID", "")
    if workflow_name == "external_baseline_method_faithful" and not resolved_baseline_id:
        raise ValueError("external_baseline_method_faithful 打包必须提供单一 baseline_id")
    paper_run = build_paper_run_config(root)
    runtime_output_dir = (
        f"outputs/notebook_runtime_observation/{paper_run.run_name}/{workflow_name}"
    )
    if resolved_baseline_id:
        runtime_output_dir = f"{runtime_output_dir}/{resolved_baseline_id}"
    write_notebook_runtime_report(
        root=root,
        workflow_name=workflow_name,
        output_dir=runtime_output_dir,
        baseline_id=resolved_baseline_id,
        drive_output_dir=resolved_drive_output_dir,
        archive_name=archive_name,
    )
    package_kwargs: dict[str, Any] = {"root": root, "archive_name": archive_name}
    if resolved_drive_output_dir is not None:
        package_kwargs["drive_output_dir"] = resolved_drive_output_dir
    if workflow_name == "external_baseline_method_faithful":
        package_kwargs["baseline_id"] = resolved_baseline_id
    if workflow_name == "official_reference_t2smark":
        if resolved_drive_output_dir is None:
            raise ValueError("T2SMark 隔离打包必须提供 Drive 输出目录")
        from paper_experiments.runners.isolated_scientific_workflow import (
            package_isolated_scientific_workflow_outputs,
        )

        return package_isolated_scientific_workflow_outputs(
            root=root,
            workflow_name=workflow_name,
            drive_output_dir=resolved_drive_output_dir,
            archive_name=archive_name,
        )
    package_function = _package_function_for_workflow(workflow_name)
    return package_function(**package_kwargs)

