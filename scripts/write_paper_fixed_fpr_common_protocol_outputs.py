"""阻断缺少精确9重复聚合来源的正式 fixed-FPR 共同协议写入."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.runners.paper_claim_provenance import (
    require_exact9_randomization_aggregate_provenance,
)


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本."""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_paper_fixed_fpr_common_protocol_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
    candidate_records_path: str | Path | None = None,
    paired_superiority_summary_path: str | Path | None = None,
    paired_superiority_manifest_path: str | Path | None = None,
    require_existing_evidence: bool = False,
) -> dict[str, Any]:
    """在读取输入或创建输出前要求版本化精确9重复聚合来源.

    该公开入口不接受单 repeat 包,调用方传入的包身份或声明的 ready 字段.
    当前 Writer 尚未消费不可变 aggregate 来源对象, 因而统一失败即关闭,
    防止已验证来源之外的历史文件或自报字段产出正式结论.
    """

    del (
        root,
        output_dir,
        candidate_records_path,
        paired_superiority_summary_path,
        paired_superiority_manifest_path,
        require_existing_evidence,
    )
    require_exact9_randomization_aggregate_provenance()


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器."""

    parser = argparse.ArgumentParser(
        description="写出精确9重复聚合后的 fixed-FPR 共同协议产物."
    )
    parser.add_argument("--root", default=".", help="仓库根目录.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录; 必须位于 outputs/ 下.",
    )
    parser.add_argument("--candidate-records-path", default=None)
    parser.add_argument("--paired-superiority-summary-path", default=None)
    parser.add_argument("--paired-superiority-manifest-path", default=None)
    parser.add_argument(
        "--require-existing-evidence",
        action="store_true",
        help="校验 evidence_paths 指向的文件存在.",
    )
    return parser


def main() -> None:
    """命令行入口."""

    args = build_parser().parse_args()
    manifest = write_paper_fixed_fpr_common_protocol_outputs(
        root=args.root,
        output_dir=args.output_dir,
        candidate_records_path=args.candidate_records_path,
        paired_superiority_summary_path=args.paired_superiority_summary_path,
        paired_superiority_manifest_path=args.paired_superiority_manifest_path,
        require_existing_evidence=args.require_existing_evidence,
    )
    print(stable_json_text(manifest), end="")


if __name__ == "__main__":
    main()
