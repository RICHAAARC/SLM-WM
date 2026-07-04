"""Colab Notebook 入口层通用 helper。

该模块只封装 Notebook 运行入口的重复样板: 统一归档命名、短提交读取、
Drive 输出目录解析和 workflow 打包调度。正式方法逻辑、攻击协议、
baseline 适配和论文结果闭合仍由各自 repository module 实现。
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
from typing import Any, Callable

from paper_workflow.colab_utils.notebook_runtime import write_notebook_runtime_report

WORKFLOW_ARCHIVE_PREFIXES = {
    "real_sd_runtime_probe": "real_sd_runtime_probe_package",
    "minimal_diffusion_latent_injection": "minimal_latent_injection_package",
    "attention_geometry": "attention_geometry_package",
    "attention_latent_injection": "attention_latent_injection_package",
    "aligned_rescoring": "aligned_rescoring_package",
    "threshold_calibration": "threshold_calibration_package",
    "real_attack_evaluation": "real_attack_evaluation_package",
    "conventional_geometric_attack_evaluation": "conventional_geometric_attack_evaluation_package",
    "dataset_level_quality": "dataset_level_quality_package",
    "external_baseline_method_faithful": "external_baseline_method_faithful_package",
    "official_reference_tree_ring": "external_baseline_official_reference_package_tree_ring",
    "official_reference_gaussian_shading": "external_baseline_official_reference_package_gaussian_shading",
    "official_reference_shallow_diffuse": "external_baseline_official_reference_package_shallow_diffuse",
    "official_reference_t2smark": "external_baseline_official_reference_package_t2smark",
}
WORKFLOW_DRIVE_OUTPUT_ENV_KEYS = {
    "real_sd_runtime_probe": "SLM_WM_RUNTIME_DRIVE_OUTPUT_DIR",
    "minimal_diffusion_latent_injection": "SLM_WM_INJECTION_DRIVE_OUTPUT_DIR",
    "attention_geometry": "SLM_WM_DRIVE_OUTPUT_DIR",
    "attention_latent_injection": "SLM_WM_DRIVE_OUTPUT_DIR",
    "aligned_rescoring": "SLM_WM_DRIVE_OUTPUT_DIR",
    "threshold_calibration": "SLM_WM_THRESHOLD_CALIBRATION_DRIVE_DIR",
    "real_attack_evaluation": "SLM_WM_DRIVE_OUTPUT_DIR",
    "conventional_geometric_attack_evaluation": "SLM_WM_CONVENTIONAL_GEOMETRIC_ATTACK_DRIVE_DIR",
    "dataset_level_quality": "SLM_WM_DATASET_QUALITY_DRIVE_DIR",
    "external_baseline_method_faithful": "SLM_WM_EXTERNAL_BASELINE_DRIVE_OUTPUT_DIR",
    "official_reference_tree_ring": "SLM_WM_TREE_RING_OFFICIAL_DRIVE_OUTPUT_DIR",
    "official_reference_gaussian_shading": "SLM_WM_GAUSSIAN_SHADING_OFFICIAL_DRIVE_OUTPUT_DIR",
    "official_reference_shallow_diffuse": "SLM_WM_SHALLOW_DIFFUSE_OFFICIAL_DRIVE_OUTPUT_DIR",
    "official_reference_t2smark": "SLM_WM_T2SMARK_FULL_MAIN_DRIVE_OUTPUT_DIR",
}
WORKFLOW_LOCAL_OUTPUT_DIRS = {
    "real_sd_runtime_probe": "outputs/real_sd_runtime_probe",
    "minimal_diffusion_latent_injection": "outputs/minimal_diffusion_latent_injection",
    "attention_geometry": "outputs/real_attention_geometry",
    "attention_latent_injection": "outputs/attention_latent_injection",
    "aligned_rescoring": "outputs/aligned_rescoring",
    "threshold_calibration": "outputs/threshold_calibration",
    "real_attack_evaluation": "outputs/real_attack_evaluation",
    "conventional_geometric_attack_evaluation": "outputs/conventional_geometric_attack_evaluation",
    "dataset_level_quality": "outputs/dataset_level_quality",
    "external_baseline_method_faithful": "outputs/external_baseline_method_faithful",
    "official_reference_tree_ring": "outputs/tree_ring_official_reference",
    "official_reference_gaussian_shading": "outputs/gaussian_shading_official_reference",
    "official_reference_shallow_diffuse": "outputs/shallow_diffuse_official_reference",
    "official_reference_t2smark": "outputs/t2smark_full_main_reproduction",
}


def resolve_short_commit(root: str | Path = ".") -> str:
    """读取当前仓库短提交, 失败时返回稳定占位文本。"""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(root),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "git_unknown"
    return result.stdout.strip() or "git_unknown"


def utc_archive_token() -> str:
    """生成所有 Notebook 归档共用的 UTC 时间后缀。"""

    current_time = datetime.now(timezone.utc)
    return f"{current_time:%Y%m%d}t{current_time:%H%M%S}z"


def build_workflow_archive_name(
    workflow_name: str,
    *,
    root: str | Path = ".",
    baseline_id: str | None = None,
) -> str:
    """根据 workflow 语义生成统一归档文件名。

    命名统一为 `<语义前缀>[_<baseline>]_<utc>_<short_commit>.zip`。
    baseline 后缀只用于 method-faithful external baseline, 以便四个
    baseline 共用同一打包入口但仍能在 Drive 目录中直接区分产物。
    """

    if workflow_name not in WORKFLOW_ARCHIVE_PREFIXES:
        raise ValueError(f"unknown_notebook_workflow:{workflow_name}")
    prefix = WORKFLOW_ARCHIVE_PREFIXES[workflow_name]
    if workflow_name == "external_baseline_method_faithful":
        selected_baseline = baseline_id or os.environ.get("SLM_WM_PRIMARY_BASELINE_METHODS", "")
        normalized_baseline = selected_baseline.replace(",", "_").strip("_")
        if normalized_baseline:
            prefix = f"{prefix}_{normalized_baseline}"
    return f"{prefix}_{utc_archive_token()}_{resolve_short_commit(root)}.zip"


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

    if workflow_name == "real_sd_runtime_probe":
        from paper_workflow.colab_utils.sd_runtime_cold_start import package_probe_outputs

        return package_probe_outputs
    if workflow_name == "minimal_diffusion_latent_injection":
        from paper_workflow.colab_utils.minimal_latent_injection import package_injection_outputs

        return package_injection_outputs
    if workflow_name == "attention_geometry":
        from paper_workflow.colab_utils.attention_geometry_capture import package_attention_geometry_outputs

        return package_attention_geometry_outputs
    if workflow_name == "attention_latent_injection":
        from paper_workflow.colab_utils.attention_latent_injection import package_attention_latent_injection_outputs

        return package_attention_latent_injection_outputs
    if workflow_name == "aligned_rescoring":
        from paper_workflow.colab_utils.aligned_rescoring import package_aligned_rescoring_outputs

        return package_aligned_rescoring_outputs
    if workflow_name == "threshold_calibration":
        from paper_workflow.colab_utils.threshold_calibration import package_threshold_calibration_outputs

        return package_threshold_calibration_outputs
    if workflow_name == "real_attack_evaluation":
        from paper_workflow.colab_utils.real_attack_evaluation import package_real_attack_evaluation_outputs

        return package_real_attack_evaluation_outputs
    if workflow_name == "conventional_geometric_attack_evaluation":
        from paper_workflow.colab_utils.conventional_geometric_attack_evaluation import (
            package_conventional_geometric_attack_evaluation_outputs,
        )

        return package_conventional_geometric_attack_evaluation_outputs
    if workflow_name == "dataset_level_quality":
        from paper_workflow.colab_utils.dataset_level_quality import package_dataset_level_quality_outputs

        return package_dataset_level_quality_outputs
    if workflow_name == "external_baseline_method_faithful":
        from paper_workflow.colab_utils.external_baseline_method_faithful import (
            package_external_baseline_method_faithful_outputs,
        )

        return package_external_baseline_method_faithful_outputs
    if workflow_name == "official_reference_tree_ring":
        from paper_workflow.colab_utils.tree_ring_official_reference import package_tree_ring_official_reference_outputs

        return package_tree_ring_official_reference_outputs
    if workflow_name == "official_reference_gaussian_shading":
        from paper_workflow.colab_utils.gaussian_shading_official_reference import (
            package_gaussian_shading_official_reference_outputs,
        )

        return package_gaussian_shading_official_reference_outputs
    if workflow_name == "official_reference_shallow_diffuse":
        from paper_workflow.colab_utils.shallow_diffuse_official_reference import (
            package_shallow_diffuse_official_reference_outputs,
        )

        return package_shallow_diffuse_official_reference_outputs
    if workflow_name == "official_reference_t2smark":
        from paper_workflow.colab_utils.t2smark_full_main_reproduction import (
            package_t2smark_full_main_reproduction_outputs,
        )

        return package_t2smark_full_main_reproduction_outputs
    raise ValueError(f"unknown_notebook_workflow:{workflow_name}")


def package_workflow_outputs(
    *,
    root: str | Path = ".",
    workflow_name: str,
    drive_output_dir: str | None = None,
    baseline_id: str | None = None,
) -> Any:
    """统一执行单一 workflow 的 archive 打包与 Drive 镜像。"""

    package_function = _package_function_for_workflow(workflow_name)
    archive_name = build_workflow_archive_name(workflow_name, root=root, baseline_id=baseline_id)
    resolved_drive_output_dir = resolve_drive_output_dir(workflow_name, drive_output_dir)
    write_notebook_runtime_report(
        root=root,
        workflow_name=workflow_name,
        output_dir=WORKFLOW_LOCAL_OUTPUT_DIRS[workflow_name],
        baseline_id=baseline_id or os.environ.get("SLM_WM_PRIMARY_BASELINE_METHODS", ""),
        drive_output_dir=resolved_drive_output_dir,
        archive_name=archive_name,
    )
    package_kwargs: dict[str, Any] = {"root": root, "archive_name": archive_name}
    if resolved_drive_output_dir is not None:
        package_kwargs["drive_output_dir"] = resolved_drive_output_dir
    return package_function(**package_kwargs)


def package_runtime_method_precheck_outputs(root: str | Path = ".") -> dict[str, Any]:
    """统一打包运行时诊断 Notebook 的两个预检产物。"""

    runtime_archive = package_workflow_outputs(root=root, workflow_name="real_sd_runtime_probe")
    injection_archive = package_workflow_outputs(root=root, workflow_name="minimal_diffusion_latent_injection")
    return {
        "runtime_archive": runtime_archive.to_dict(),
        "injection_archive": injection_archive.to_dict(),
    }
