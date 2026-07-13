"""在跨重复原始记录重算接入前阻断服务器论文闭合入口.

版本化精确9重复聚合包的构造与独立验证已经实现, 但聚合包本身不直接支持
论文结论. 在45个阈值与跨重复统计 Writer 接入不可变 aggregate 来源前, 本
入口不发布 dry-run 或正式结果.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NoReturn

from experiments.protocol.paper_run_config import RUN_DEFAULTS


AGGREGATE_RECOMPUTATION_REQUIRED_MESSAGE = (
    "汇总服务器论文闭合必须从已验证聚合包重算跨重复原始记录;"
    "当前正式统计 Writer 尚未接入"
)


def execute_server_result_closure(
    *,
    root: str | Path,
    paper_run_name: str,
    randomization_aggregate_package_path: str | Path,
    complete_output_dir: str | Path,
    repository_commit: str,
    dry_run: bool = False,
) -> NoReturn:
    """拒绝在跨重复原始记录重算未接入时启动论文闭合."""

    del (
        root,
        paper_run_name,
        randomization_aggregate_package_path,
        complete_output_dir,
        repository_commit,
        dry_run,
    )
    raise RuntimeError(AGGREGATE_RECOMPUTATION_REQUIRED_MESSAGE)


def build_parser() -> argparse.ArgumentParser:
    """构造保留精确运行身份的命令行参数解析器."""

    parser = argparse.ArgumentParser(
        description="从已验证精确9重复聚合包执行服务器论文结果闭合."
    )
    parser.add_argument("--root", default=".", help="仓库根目录.")
    parser.add_argument(
        "--repository-commit",
        required=True,
        help="汇总执行使用的精确40位小写 Git SHA.",
    )
    parser.add_argument(
        "--paper-run-name",
        required=True,
        choices=sorted(RUN_DEFAULTS),
        help="必须显式指定论文运行层级, 避免误用其他统计规模.",
    )
    parser.add_argument(
        "--randomization-aggregate-package-path",
        required=True,
        help="精确9重复聚合来源 ZIP 的显式路径.",
    )
    parser.add_argument(
        "--complete-output-dir",
        required=True,
        help="未来完整结果包输出目录.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="跨重复原始记录重算接入前该选项同样 fail-closed.",
    )
    return parser


def main() -> None:
    """解析参数并执行 fail-closed 服务器入口."""

    args = build_parser().parse_args()
    execute_server_result_closure(
        root=args.root,
        paper_run_name=args.paper_run_name,
        randomization_aggregate_package_path=(
            args.randomization_aggregate_package_path
        ),
        complete_output_dir=args.complete_output_dir,
        repository_commit=args.repository_commit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
