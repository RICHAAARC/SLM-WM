"""Notebook 入口运行时间记录 helper。

该模块属于 Notebook 入口治理层: 它只记录远程执行入口从配置发布到
归档落盘之间的观测时间, 不参与方法算法、攻击协议或论文统计计算。
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

NOTEBOOK_RUNTIME_STARTED_AT_KEY = "SLM_WM_NOTEBOOK_RUNTIME_STARTED_AT"
NOTEBOOK_RUNTIME_STARTED_MONOTONIC_KEY = "SLM_WM_NOTEBOOK_RUNTIME_STARTED_MONOTONIC"
NOTEBOOK_RUNTIME_START_SOURCE_KEY = "SLM_WM_NOTEBOOK_RUNTIME_START_SOURCE"
NOTEBOOK_RUNTIME_WORKFLOW_NAME_KEY = "SLM_WM_NOTEBOOK_RUNTIME_WORKFLOW_NAME"
NOTEBOOK_RUNTIME_BASELINE_ID_KEY = "SLM_WM_NOTEBOOK_RUNTIME_BASELINE_ID"


def utc_now_text() -> str:
    """返回带 UTC 时区的稳定时间文本。"""

    return datetime.now(timezone.utc).isoformat()


def mark_notebook_runtime_start(
    *,
    workflow_name: str,
    baseline_id: str = "",
    source: str = "notebook_entrypoint",
    reset: bool = True,
) -> dict[str, str]:
    """记录当前 Notebook 入口的运行计时起点。

    该函数属于通用入口写法: Notebook 或环境配置 helper 只需要声明当前
    workflow 名称, 具体时间字段和环境变量名由本模块统一维护。`reset=True`
    用于避免 Colab 同一内核连续运行不同入口时沿用上一入口的计时起点。
    """

    if reset or NOTEBOOK_RUNTIME_STARTED_AT_KEY not in os.environ:
        os.environ[NOTEBOOK_RUNTIME_STARTED_AT_KEY] = utc_now_text()
        os.environ[NOTEBOOK_RUNTIME_STARTED_MONOTONIC_KEY] = f"{time.monotonic():.9f}"
    os.environ[NOTEBOOK_RUNTIME_START_SOURCE_KEY] = source
    os.environ[NOTEBOOK_RUNTIME_WORKFLOW_NAME_KEY] = workflow_name
    os.environ[NOTEBOOK_RUNTIME_BASELINE_ID_KEY] = baseline_id
    return {
        "notebook_runtime_started_at": os.environ[NOTEBOOK_RUNTIME_STARTED_AT_KEY],
        "notebook_runtime_start_source": os.environ[NOTEBOOK_RUNTIME_START_SOURCE_KEY],
        "notebook_runtime_workflow_name": os.environ[NOTEBOOK_RUNTIME_WORKFLOW_NAME_KEY],
        "notebook_runtime_baseline_id": os.environ[NOTEBOOK_RUNTIME_BASELINE_ID_KEY],
    }


def ensure_notebook_runtime_start(*, workflow_name: str, baseline_id: str = "") -> dict[str, str]:
    """验证 Notebook 入口已经显式初始化运行计时。"""

    if NOTEBOOK_RUNTIME_STARTED_AT_KEY in os.environ:
        return {
            "notebook_runtime_started_at": os.environ[NOTEBOOK_RUNTIME_STARTED_AT_KEY],
            "notebook_runtime_start_source": os.environ.get(NOTEBOOK_RUNTIME_START_SOURCE_KEY, ""),
            "notebook_runtime_workflow_name": os.environ.get(NOTEBOOK_RUNTIME_WORKFLOW_NAME_KEY, workflow_name),
            "notebook_runtime_baseline_id": os.environ.get(NOTEBOOK_RUNTIME_BASELINE_ID_KEY, baseline_id),
        }
    raise RuntimeError("Notebook 归档前必须通过运行环境入口初始化计时")


def _safe_float(text: str) -> float | None:
    """把环境变量文本安全转换为浮点数。"""

    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def build_notebook_runtime_report(
    *,
    workflow_name: str,
    baseline_id: str = "",
    drive_output_dir: str | None = None,
    archive_name: str | None = None,
) -> dict[str, Any]:
    """生成当前 Notebook 入口的运行时间报告。

    报告只描述入口运行观测边界, 不作为方法有效性或论文结论的证据来源。
    `elapsed_seconds` 的起点必须由 `configure_paper_run_environment` 或对应
    workflow 入口显式写入。归档函数只消费该记录, 不创建替代计时边界。
    """

    ensure_notebook_runtime_start(workflow_name=workflow_name, baseline_id=baseline_id)
    finished_at = utc_now_text()
    started_monotonic = _safe_float(os.environ.get(NOTEBOOK_RUNTIME_STARTED_MONOTONIC_KEY, ""))
    elapsed_seconds = None if started_monotonic is None else max(0.0, time.monotonic() - started_monotonic)
    report = {
        "construction_unit_name": "notebook_runtime_observation",
        "workflow_name": workflow_name,
        "paper_run_name": os.environ.get("SLM_WM_PAPER_RUN_NAME", ""),
        "baseline_id": baseline_id or os.environ.get(NOTEBOOK_RUNTIME_BASELINE_ID_KEY, ""),
        "notebook_runtime_started_at": os.environ.get(NOTEBOOK_RUNTIME_STARTED_AT_KEY, ""),
        "notebook_runtime_finished_at": finished_at,
        "notebook_runtime_elapsed_seconds": elapsed_seconds,
        "notebook_runtime_elapsed_minutes": None if elapsed_seconds is None else elapsed_seconds / 60.0,
        "notebook_runtime_start_source": os.environ.get(NOTEBOOK_RUNTIME_START_SOURCE_KEY, ""),
        "notebook_runtime_timing_boundary": "from_paper_run_environment_configuration_to_archive_packaging",
        "drive_output_dir": drive_output_dir or "",
        "archive_name": archive_name or "",
        "supports_paper_claim": False,
    }
    return report


def write_notebook_runtime_report(
    *,
    root: str | Path = ".",
    workflow_name: str,
    output_dir: str | Path,
    baseline_id: str = "",
    drive_output_dir: str | None = None,
    archive_name: str | None = None,
) -> Path:
    """把运行时间报告写入对应 workflow 的 outputs 子目录。"""

    root_path = Path(root)
    output_path = (root_path / output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "notebook_runtime_report.json"
    report = build_notebook_runtime_report(
        workflow_name=workflow_name,
        baseline_id=baseline_id,
        drive_output_dir=drive_output_dir,
        archive_name=archive_name,
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path
