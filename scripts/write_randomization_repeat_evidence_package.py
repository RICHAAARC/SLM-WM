"""写出单个正式随机化重复的7类自包含上游证据包."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.runners.randomization_repeat_evidence import (
    write_randomization_repeat_evidence_package,
)


def build_parser() -> argparse.ArgumentParser:
    """构造脱离 Notebook 可运行的命令行参数."""

    parser = argparse.ArgumentParser(
        description="选择7类 leaf ZIP 并写出单重复自包含证据包。"
    )
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument(
        "--package-search-root",
        required=True,
        help="包含7类当前重复上游结果包的搜索目录。",
    )
    parser.add_argument("--paper-run-name", required=True)
    parser.add_argument("--target-fpr", type=float, required=True)
    parser.add_argument("--randomization-repeat-id", required=True)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="可选输出目录; 必须位于仓库 outputs 下。",
    )
    return parser


def main() -> None:
    """执行单重复证据包生产并打印写后校验报告."""

    args = build_parser().parse_args()
    report = write_randomization_repeat_evidence_package(
        args.package_search_root,
        paper_run_name=args.paper_run_name,
        target_fpr=args.target_fpr,
        randomization_repeat_id=args.randomization_repeat_id,
        root=args.root,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
