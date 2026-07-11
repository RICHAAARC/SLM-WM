"""为 Colab 注入归档目标目录并调用仓库 GPU 会话."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from scripts.semantic_watermark_scientific_workflow import (
    run_semantic_watermark_image_only_session as run_repository_scientific_session,
)


def run_semantic_watermark_image_only_session(
    root: str | Path = ".",
    *,
    run_formal_ablation: bool = False,
) -> dict[str, Any]:
    """把 Colab Drive 目录映射为显式归档目标后运行仓库会话."""

    destinations: dict[str, str] = {
        "image_only_dataset_runtime": os.environ[
            "SLM_WM_IMAGE_ONLY_RUNTIME_DRIVE_DIR"
        ],
        "dataset_level_quality": os.environ[
            "SLM_WM_DATASET_QUALITY_DRIVE_DIR"
        ],
    }
    if run_formal_ablation:
        destinations["runtime_rerun_ablation"] = os.environ[
            "SLM_WM_RUNTIME_RERUN_ABLATION_DRIVE_DIR"
        ]
    return run_repository_scientific_session(
        root,
        run_formal_ablation=run_formal_ablation,
        archive_destination_dirs=destinations,
        resume_checkpoint_dir=os.environ.get(
            "SLM_WM_RESUME_CHECKPOINT_DIR"
        ),
    )
