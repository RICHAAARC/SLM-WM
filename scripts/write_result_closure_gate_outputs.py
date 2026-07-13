"""提供跨重复论文结果闭合门禁的唯一脚本入口.

正式门禁只能消费已复验的精确9重复聚合包, 并必须从包内原始记录重新计算
全部统计. 当前聚合来源构造与验证已经闭合, 但跨重复原始记录 Writer 尚未
接入;因此本入口在读取结果输入或创建输出前统一拒绝, 不保留单重复或任意
路径拼接的兼容物化路径.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.runners.paper_claim_provenance import (
    require_exact9_randomization_aggregate_provenance,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/result_closure_gate")


def write_result_closure_gate_outputs(
    *,
    root: str | Path = ".",
    randomization_aggregate_package_path: str | Path | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    **kwargs: Any,
) -> dict[str, Any]:
    """在跨重复原始记录重算接入前拒绝正式结果闭合物化."""

    del root, randomization_aggregate_package_path, output_root, kwargs
    require_exact9_randomization_aggregate_provenance()


def build_parser() -> argparse.ArgumentParser:
    """构造只接受精确聚合来源的命令行参数解析器."""

    parser = argparse.ArgumentParser(
        description="从精确9重复聚合来源重建论文结果闭合门禁."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--randomization-aggregate-package-path",
        required=True,
    )
    return parser


def main() -> None:
    """解析唯一聚合来源并进入失败即关闭的正式 Writer 边界."""

    args = build_parser().parse_args()
    write_result_closure_gate_outputs(
        root=args.root,
        output_root=args.output_root,
        randomization_aggregate_package_path=(
            args.randomization_aggregate_package_path
        ),
    )


if __name__ == "__main__":
    main()
