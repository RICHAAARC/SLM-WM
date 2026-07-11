"""Colab 论文结果闭合包装 helper。

正式命令计划和服务器可复用执行逻辑位于
`paper_experiments.runners.paper_result_closure`。本模块只追加 Colab 进度显示
和 Notebook runtime 报告, 使 Notebook 仅承担 Colab 启动职责。
"""

from __future__ import annotations

from typing import Any

from experiments.protocol.paper_run_config import build_paper_run_config, normalize_paper_run_name
from experiments.runtime.progress import progress_bar, update_progress
from paper_experiments.runners.paper_result_closure import (
    PAPER_RESULT_CLOSURE_COMMAND_COUNT,
    run_paper_result_closure_commands as run_repository_paper_result_closure_commands,
)
from paper_workflow.notebook_utils.notebook_runtime import write_notebook_runtime_report


def _archive_name_from_complete_package_command(command: list[str]) -> str:
    """从完整结果包命令中读取 archive name。"""

    return command[command.index("--archive-name") + 1] if "--archive-name" in command else ""


def _is_complete_result_package_command(command: list[str]) -> bool:
    """判断当前命令是否为完整结果包写出命令。"""

    command_name = command[1] if len(command) > 1 else command[0]
    return command_name.endswith("write_pilot_paper_complete_result_package.py")


def run_paper_result_closure_commands(
    *,
    package_search_root: str,
    complete_drive_output_dir: str,
    paper_run_name: str,
    root: str = ".",
) -> dict[str, Any]:
    """运行论文结果闭合命令, 并返回本次精确的 Drive 结果包路径。

    该函数属于 Colab 运行层包装: 它不构造正式结果, 只把完整论文实验层 runner
    与 Notebook runtime 报告、总体进度显示连接起来。
    """

    paper_run = build_paper_run_config(root)
    normalized_run_name = normalize_paper_run_name(paper_run_name)
    if paper_run.run_name != normalized_run_name:
        raise ValueError("Colab 闭合入口的 paper_run_name 必须与统一环境配置一致")

    with progress_bar(
        PAPER_RESULT_CLOSURE_COMMAND_COUNT,
        desc="paper result closure commands",
        enabled=True,
    ) as command_progress:

        def before_command(command: list[str]) -> None:
            if _is_complete_result_package_command(command):
                write_notebook_runtime_report(
                    root=root,
                    workflow_name="paper_result_closure",
                    output_dir=(
                        "outputs/pilot_paper_complete_result_package/"
                        f"{normalized_run_name}"
                    ),
                    drive_output_dir=complete_drive_output_dir,
                    archive_name=_archive_name_from_complete_package_command(command),
                )

        def progress_hook(command_index: int, command_count: int, command_name: str) -> None:
            update_progress(
                command_progress,
                profile=f"command={command_name} index={command_index}/{command_count}",
            )

        return run_repository_paper_result_closure_commands(
            package_search_root=package_search_root,
            complete_drive_output_dir=complete_drive_output_dir,
            paper_run_name=normalized_run_name,
            target_fpr=paper_run.target_fpr,
            root=root,
            before_command=before_command,
            progress_hook=progress_hook,
        )
