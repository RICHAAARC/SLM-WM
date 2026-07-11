"""服务器与 Notebook 共用的 workflow 归档命名工具。"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

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
    """生成归档共用的 UTC 时间后缀。"""

    current_time = datetime.now(timezone.utc)
    return f"{current_time:%Y%m%d}t{current_time:%H%M%S}z"


def build_workflow_archive_name(
    workflow_name: str,
    *,
    root: str | Path = ".",
    baseline_id: str | None = None,
) -> str:
    """根据 workflow 语义生成统一归档文件名。"""

    if workflow_name not in WORKFLOW_ARCHIVE_PREFIXES:
        raise ValueError(f"unknown_notebook_workflow:{workflow_name}")
    prefix = WORKFLOW_ARCHIVE_PREFIXES[workflow_name]
    if workflow_name == "external_baseline_method_faithful":
        normalized_baseline = (baseline_id or os.environ.get("SLM_WM_PRIMARY_BASELINE_ID", "")).strip()
        if normalized_baseline not in {"tree_ring", "gaussian_shading", "shallow_diffuse"}:
            raise ValueError("external_baseline_method_faithful 归档必须指定唯一受支持 baseline_id")
        prefix = f"{prefix}_{normalized_baseline}"
    return f"{prefix}_{utc_archive_token()}_{resolve_short_commit(root)}.zip"
