"""校验外部 baseline 执行证据边界。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.baselines.evidence_validator import validate_external_baseline_evidence


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="校验外部 baseline 执行证据 manifest。")
    parser.add_argument("--baseline-execution-manifest", required=True)
    parser.add_argument("--require-formal-claim", action="store_true", help="要求 manifest 声明正式结果并绑定证据路径。")
    parser.add_argument("--out", default=None, help="可选报告输出路径, 必须由调用方放在 outputs/ 下。")
    parser.add_argument("--require-pass", action="store_true", help="报告未通过时返回非零退出码。")
    return parser


def main() -> None:
    """CLI 入口。"""

    args = build_parser().parse_args()
    report = validate_external_baseline_evidence(
        baseline_execution_manifest=args.baseline_execution_manifest,
        require_formal_claim=args.require_formal_claim,
    )
    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_pass and report["overall_decision"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
