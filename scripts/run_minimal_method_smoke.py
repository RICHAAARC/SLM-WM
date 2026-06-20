"""运行 minimal method package 可复用的核心 smoke。

该脚本只依赖 `main.methods` 暴露的最小方法入口, 默认把 smoke 报告打印到标准
输出。若显式要求写文件, 输出路径必须位于 `outputs/` 下。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.methods import build_core_method_smoke_bundle


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转为稳定、可读的 UTF-8 文本。"""
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def ensure_output_path_under_outputs(root_path: Path, output_path: Path) -> Path:
    """确保显式写出的 smoke 报告位于 `outputs/` 下。"""
    resolved_output_path = (root_path / output_path).resolve() if not output_path.is_absolute() else output_path.resolve()
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved_output_path.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("minimal method smoke 输出文件必须位于 outputs/ 下") from exc
    return resolved_output_path


def build_minimal_smoke_report() -> dict[str, Any]:
    """构造最小方法包 smoke 报告。"""
    bundle = build_core_method_smoke_bundle()
    return {
        "stage_name": "stage_02_core_method_smoke_test",
        "artifact_id": "minimal_method_smoke_report",
        "artifact_type": "stdout_or_local_report",
        "decision": "pass"
        if bundle.metrics["key_separation_margin"] > 0
        and not bundle.metrics["wrong_key_over_threshold"]
        and bundle.metrics["geometry_unreliable_rescue_blocked"]
        and bundle.metrics["attestation_layering_pass"]
        else "fail",
        "metrics": bundle.metrics,
        "scenario_count": len(bundle.scenarios),
        "metadata": {
            "minimal_method_dependency": "main.methods",
            "writes_persistent_output_by_default": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="运行 minimal method package smoke。")
    parser.add_argument("--root", default=".", help="仓库根目录。")
    parser.add_argument("--output", default=None, help="可选输出文件, 必须位于 outputs/ 下。")
    return parser


def main() -> None:
    """命令行入口。"""
    parser = build_parser()
    args = parser.parse_args()
    report = build_minimal_smoke_report()
    text = stable_json_text(report)
    if args.output:
        root_path = Path(args.root).resolve()
        output_path = ensure_output_path_under_outputs(root_path, Path(args.output))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
