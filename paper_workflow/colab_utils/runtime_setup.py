"""生成 Colab Drive workflow 的运行环境摘要。"""

from __future__ import annotations

from datetime import datetime, timezone
import platform
import sys
from typing import Any

from paper_workflow.colab_utils.dependency_check import build_dependency_report


def build_runtime_setup_report() -> dict[str, Any]:
    """生成轻量运行环境报告, 不触发真实模型加载。"""
    dependency_report = build_dependency_report()
    return {
        "construction_unit_name": "colab_drive_workflow",
        "workflow_name": "colab_drive_workflow",
        "workflow_decision": dependency_report["dependency_decision"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "dependency_report": dependency_report,
        "supports_paper_claim": False,
    }
