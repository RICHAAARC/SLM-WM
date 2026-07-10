"""把本地 outputs 产物镜像到 Drive workflow 目录。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_workflow.colab_utils.drive_paths import DEFAULT_DRIVE_ROOT, DEFAULT_LOCAL_OUTPUT_DIR, DEFAULT_WORKFLOW_NAME
from paper_workflow.colab_utils.drive_workflow import run_local_output_sync
from paper_workflow.colab_utils.manifest_io import stable_json_text


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="同步本地 outputs 产物到 Drive workflow 目录。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--drive-root", default=str(DEFAULT_DRIVE_ROOT), help="Drive 或本地镜像根目录。")
    parser.add_argument("--local-output-dir", default=str(DEFAULT_LOCAL_OUTPUT_DIR), help="本地输出目录, 必须位于 outputs/ 下。")
    parser.add_argument("--workflow-name", default=DEFAULT_WORKFLOW_NAME, help="workflow 语义名称。")
    return parser


def main() -> None:
    """执行镜像并打印报告。"""
    args = build_parser().parse_args()
    report = run_local_output_sync(
        root=args.root,
        drive_root=args.drive_root,
        local_output_dir=args.local_output_dir,
        workflow_name=args.workflow_name,
    )
    print(stable_json_text(report), end="")


if __name__ == "__main__":
    main()
