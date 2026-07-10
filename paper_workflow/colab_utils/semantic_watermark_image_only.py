"""调度真实科学算子、仅图像检测和正式消融的 Colab 会话。

该模块只负责调用 repository script、读取续跑状态并把完成后的结果包镜像到
Google Drive。方法实现、统计协议和证据门禁仍分别位于 ``main/``、
``experiments/`` 和 ``scripts/``。这种分层属于通用工程写法: Notebook 保持为
薄入口, 长耗时任务的状态判断和结果复制集中在可测试的 Python 模块中。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    """读取一个 JSON 对象, 缺失或格式不符时返回空对象。"""

    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_repository_script(root_path: Path, script_path: str) -> None:
    """使用当前 Python 环境运行 repository script, 并保留实时控制台输出。"""

    subprocess.run(
        [sys.executable, script_path],
        cwd=root_path,
        env=os.environ.copy(),
        check=True,
    )


def _mirror_latest_archive(source_dir: Path, pattern: str, destination_dir: Path) -> str:
    """把最新结果包复制到 Drive 受治理目录并返回目标路径。

    复制前先创建目标目录。若同名文件已经存在, ``copy2`` 会覆盖该文件而不会
    生成第二套不确定名称, 因而可安全用于 Colab 会话重复执行。
    """

    matches = sorted(source_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"缺少待镜像结果包: {source_dir / pattern}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / matches[-1].name
    shutil.copy2(matches[-1], destination)
    return str(destination)


def run_semantic_watermark_image_only_session(
    root: str | Path = ".",
    *,
    run_formal_ablation: bool = False,
) -> dict[str, Any]:
    """执行一次可中断、可恢复的 Colab GPU 会话。

    数据集主流程优先运行。只有主流程、冻结阈值和正式 Inception 质量输出均已
    完成时, 才允许启动真实机制消融。每次会话的新增工作量由环境变量控制,
    已完成运行由 manifest、配置摘要和代码版本共同校验后复用。
    """

    root_path = Path(root).resolve()
    paper_run_name = os.environ.get("SLM_WM_PAPER_RUN_NAME", "pilot_paper")
    runtime_output_dir = root_path / "outputs" / "image_only_dataset_runtime" / paper_run_name
    quality_output_dir = root_path / "outputs" / "dataset_level_quality" / paper_run_name
    ablation_output_dir = root_path / "outputs" / "formal_mechanism_ablation" / paper_run_name

    _run_repository_script(root_path, "scripts/run_image_only_dataset_runtime.py")
    runtime_progress_path = runtime_output_dir / "dataset_runtime_progress.json"
    if runtime_progress_path.is_file():
        progress = _read_json(runtime_progress_path)
        return {
            "workflow_decision": "resume_required",
            "paper_run_name": paper_run_name,
            "active_workflow": "image_only_dataset_runtime",
            "runtime_progress": progress,
            "formal_ablation_requested": run_formal_ablation,
            "supports_paper_claim": False,
        }

    runtime_summary = _read_json(runtime_output_dir / "dataset_runtime_summary.json")
    quality_summary = _read_json(quality_output_dir / "dataset_quality_summary.json")
    if runtime_summary.get("protocol_decision") != "pass":
        raise RuntimeError("仅图像数据集运行未生成通过协议的正式摘要")
    if not quality_summary.get("formal_fid_kid_claim_gate_ready", False):
        raise RuntimeError("数据集运行完成, 但规范 Inception FID/KID 尚未闭合")

    mirrored_archives = {
        "image_only_dataset_runtime": _mirror_latest_archive(
            runtime_output_dir,
            "image_only_dataset_runtime_package_*.zip",
            Path(os.environ["SLM_WM_IMAGE_ONLY_RUNTIME_DRIVE_DIR"]),
        ),
        "dataset_level_quality": _mirror_latest_archive(
            quality_output_dir,
            "dataset_level_quality_package_*.zip",
            Path(os.environ["SLM_WM_DATASET_QUALITY_DRIVE_DIR"]),
        ),
    }
    if not run_formal_ablation:
        return {
            "workflow_decision": "dataset_complete",
            "paper_run_name": paper_run_name,
            "active_workflow": "image_only_dataset_runtime",
            "runtime_summary": runtime_summary,
            "quality_summary": quality_summary,
            "mirrored_archives": mirrored_archives,
            "formal_ablation_requested": False,
            "supports_paper_claim": False,
        }

    _run_repository_script(root_path, "scripts/run_runtime_rerun_ablations.py")
    ablation_progress_path = ablation_output_dir / "runtime_rerun_progress.json"
    if ablation_progress_path.is_file():
        return {
            "workflow_decision": "resume_required",
            "paper_run_name": paper_run_name,
            "active_workflow": "runtime_rerun_ablation",
            "runtime_summary": runtime_summary,
            "quality_summary": quality_summary,
            "ablation_progress": _read_json(ablation_progress_path),
            "mirrored_archives": mirrored_archives,
            "formal_ablation_requested": True,
            "supports_paper_claim": False,
        }

    ablation_summary = _read_json(ablation_output_dir / "ablation_claim_summary.json")
    if ablation_summary.get("protocol_decision") != "pass":
        raise RuntimeError("正式机制消融未生成通过协议的摘要")
    mirrored_archives["runtime_rerun_ablation"] = _mirror_latest_archive(
        ablation_output_dir,
        "runtime_rerun_ablation_package_*.zip",
        Path(os.environ["SLM_WM_RUNTIME_RERUN_ABLATION_DRIVE_DIR"]),
    )
    return {
        "workflow_decision": "complete",
        "paper_run_name": paper_run_name,
        "active_workflow": "runtime_rerun_ablation",
        "runtime_summary": runtime_summary,
        "quality_summary": quality_summary,
        "ablation_summary": ablation_summary,
        "mirrored_archives": mirrored_archives,
        "formal_ablation_requested": True,
        "supports_paper_claim": bool(
            runtime_summary.get("supports_paper_claim", False)
            and quality_summary.get("supports_paper_claim", False)
            and ablation_summary.get("supports_paper_claim", False)
        ),
    }
