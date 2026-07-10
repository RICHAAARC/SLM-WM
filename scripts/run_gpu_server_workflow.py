"""在无 Notebook 的 CUDA 服务器上执行当前正式工作流。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

WORKFLOW_COMMANDS = {
    "image_only_dataset": [sys.executable, "scripts/run_image_only_dataset_runtime.py"],
    "mechanism_ablation": [sys.executable, "scripts/run_runtime_rerun_ablations.py"],
}


def run_workflow(
    workflow_name: str,
    paper_run_name: str,
    root: str | Path = ".",
) -> dict[str, Any]:
    """执行一个正式 GPU 工作流并返回可序列化进程结果。"""

    if workflow_name not in WORKFLOW_COMMANDS:
        raise ValueError(f"未知正式工作流: {workflow_name}")
    root_path = Path(root).resolve()
    environment = os.environ.copy()
    environment["SLM_WM_PAPER_RUN_NAME"] = paper_run_name
    completed = subprocess.run(
        WORKFLOW_COMMANDS[workflow_name],
        cwd=root_path,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    result = {
        "workflow_name": workflow_name,
        "paper_run_name": paper_run_name,
        "command": WORKFLOW_COMMANDS[workflow_name],
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    """构造服务器入口参数。"""

    parser = argparse.ArgumentParser(description="运行不依赖 Notebook 的正式 GPU 工作流。")
    parser.add_argument("--workflow", required=True, choices=tuple(WORKFLOW_COMMANDS))
    parser.add_argument(
        "--paper-run-name",
        required=True,
        choices=("probe_paper", "pilot_paper", "full_paper"),
    )
    parser.add_argument("--root", default=".")
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    print(
        json.dumps(
            run_workflow(args.workflow, args.paper_run_name, args.root),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
