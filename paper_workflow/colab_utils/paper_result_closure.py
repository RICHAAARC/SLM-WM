"""Colab 论文结果闭合命令调度 helper。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any

from paper_workflow.colab_utils.progress import progress_bar, update_progress


def _short_commit() -> str:
    """读取当前仓库短提交, 用于结果包命名。"""

    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _complete_archive_name(paper_run_name: str) -> str:
    """根据当前论文运行层级生成完整结果包名称。"""

    utc_suffix = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%sz")
    return f"{paper_run_name}_complete_result_package_{utc_suffix}_{_short_commit()}.zip"


def build_paper_result_closure_commands(
    *,
    package_search_root: str,
    complete_drive_output_dir: str,
    target_fpr: str,
    paper_run_name: str,
) -> list[list[str]]:
    """构造结果闭合所需 repository command 序列。

    该函数只调度已有脚本, 不直接写正式 records、tables、figures 或 reports。
    这样 Notebook 不需要拼接命令、结果包名称和输出路径。
    """

    complete_archive_name = _complete_archive_name(paper_run_name)
    return [
        [sys.executable, "scripts/write_pilot_paper_result_records.py", "--package-search-root", package_search_root, "--materialize-only"],
        [sys.executable, "scripts/write_attack_matrix_outputs.py"],
        [sys.executable, "scripts/write_primary_baseline_result_candidates.py", "--target-fpr-override", target_fpr],
        [sys.executable, "scripts/write_primary_baseline_formal_import_protocol.py"],
        [sys.executable, "scripts/write_external_baseline_comparison_outputs.py"],
        [sys.executable, "scripts/write_internal_ablation_outputs.py"],
        [sys.executable, "scripts/write_pilot_paper_result_records.py", "--require-existing-evidence"],
        [
            sys.executable,
            "scripts/write_pilot_paper_fixed_fpr_common_protocol_outputs.py",
            "--candidate-records-path",
            "outputs/pilot_paper_fixed_fpr_results/pilot_paper_result_records.jsonl",
            "--require-existing-evidence",
        ],
        [
            sys.executable,
            "scripts/write_pilot_paper_complete_result_package.py",
            "--package-search-root",
            package_search_root,
            "--drive-output-dir",
            complete_drive_output_dir,
            "--archive-name",
            complete_archive_name,
            "--skip-package-materialization",
            "--zip-compression",
            "stored",
        ],
    ]


def run_paper_result_closure_commands(
    *,
    package_search_root: str,
    complete_drive_output_dir: str,
    target_fpr: str,
    paper_run_name: str,
) -> dict[str, Any]:
    """运行论文结果闭合命令, 并返回最新 Drive 结果包路径。"""

    commands = build_paper_result_closure_commands(
        package_search_root=package_search_root,
        complete_drive_output_dir=complete_drive_output_dir,
        target_fpr=target_fpr,
        paper_run_name=paper_run_name,
    )
    with progress_bar(len(commands), desc="paper result closure commands", enabled=True) as command_progress:
        for command_index, command in enumerate(commands, start=1):
            command_name = command[1] if len(command) > 1 else command[0]
            print("run_repository_command", " ".join(command))
            subprocess.run(command, check=True)
            update_progress(
                command_progress,
                profile=f"command={command_name} index={command_index}/{len(commands)}",
            )

    archive_pattern = f"{paper_run_name}_complete_result_package_*.zip"
    complete_archives = sorted(Path(complete_drive_output_dir).glob(archive_pattern))
    if not complete_archives:
        raise FileNotFoundError("未找到写回 Google Drive 的当前论文运行层级完整结果包")
    return {
        "latest_complete_archive": str(complete_archives[-1]),
        "command_count": len(commands),
        "paper_run_name": paper_run_name,
        "target_fpr": target_fpr,
    }
