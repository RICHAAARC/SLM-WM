"""写出主方法与四个外部 baseline 的统一 fixed-FPR 阈值审计。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.runners.paper_claim_provenance import (
    require_exact9_randomization_aggregate_provenance,
)
from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.protocol.splits import build_group_split_counts
from experiments.runtime.repository_environment import resolve_code_version
from paper_experiments.analysis.fixed_fpr_threshold_audit import (
    audit_baseline_fixed_fpr,
    audit_main_method_fixed_fpr,
    build_fixed_fpr_threshold_audit_report,
    build_fixed_fpr_threshold_manifest_config,
)
from paper_experiments.baselines.method_faithful_observation_collection import (
    file_sha256,
    load_method_faithful_observation_collection,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/fixed_fpr_threshold_audit")
DEFAULT_METHOD_FAITHFUL_OUTPUT_ROOT = Path("outputs/external_baseline_method_faithful")
DEFAULT_T2SMARK_OUTPUT_ROOT = Path("outputs/t2smark_formal_reproduction")


def _read_json(path: Path) -> dict[str, Any]:
    """读取必须存在的 JSON 对象。"""

    if not path.is_file():
        raise FileNotFoundError(f"fixed-FPR 阈值审计输入不存在: {path.as_posix()}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"fixed-FPR 阈值审计输入必须是 JSON 对象: {path.as_posix()}")
    return dict(payload)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """读取必须存在且非空的 JSONL observation。"""

    if not path.is_file():
        raise FileNotFoundError(f"fixed-FPR observation 不存在: {path.as_posix()}")
    rows = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"fixed-FPR observation 必须是非空 JSONL 对象序列: {path.as_posix()}")
    return tuple(dict(row) for row in rows)


def _read_json_array(path: Path) -> tuple[dict[str, Any], ...]:
    """读取必须存在且非空的 JSON observation 数组。"""

    if not path.is_file():
        raise FileNotFoundError(f"fixed-FPR observation 不存在: {path.as_posix()}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or not payload or any(not isinstance(row, dict) for row in payload):
        raise ValueError(f"fixed-FPR observation 必须是非空 JSON 对象数组: {path.as_posix()}")
    return tuple(dict(row) for row in payload)


def _read_candidate_threshold(
    path: Path,
    *,
    target_fpr: float,
) -> tuple[float, str]:
    """读取 T2SMark 候选记录共享的目标 FPR、阈值与摘要。"""

    rows = _read_jsonl(path)
    thresholds = {float(row["calibrated_detection_threshold"]) for row in rows}
    digests = {str(row.get("threshold_digest", "")) for row in rows}
    target_fprs = {float(row["target_fpr"]) for row in rows}
    if (
        len(thresholds) != 1
        or len(digests) != 1
        or target_fprs != {float(target_fpr)}
    ):
        raise ValueError("T2SMark formal 候选记录未共享当前协议的目标 FPR、冻结阈值与摘要")
    return next(iter(thresholds)), next(iter(digests))


def _write_json(path: Path, payload: Any) -> None:
    """写出稳定 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_rows_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """写出固定字段顺序的阈值审计表。"""

    materialized = tuple(rows)
    fieldnames = (
        "method_id",
        "threshold_source",
        "target_fpr",
        "calibration_clean_negative_count",
        "test_clean_negative_count",
        "calibrated_detection_threshold",
        "threshold_digest",
        "observation_source_sha256",
        "protocol_target_ready",
        "protocol_value_ready",
        "detection_decision_ready",
        "split_count_ready",
        "fixed_fpr_threshold_ready",
        "supports_paper_claim",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def write_fixed_fpr_threshold_audit_outputs(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """在精确9重复原始记录重算 Writer 就绪前拒绝正式结论物化."""

    require_exact9_randomization_aggregate_provenance()


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="写出五方法统一 fixed-FPR 阈值审计。")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录; 默认写入 outputs/fixed_fpr_threshold_audit/<paper_run_name>。",
    )
    parser.add_argument(
        "--method-faithful-collection-root",
        default=None,
        help="三个 common-backbone baseline 的当前运行层级 exact-set 根目录。",
    )
    parser.add_argument(
        "--t2smark-output-dir",
        default=None,
        help="T2SMark 当前运行层级正式复现目录。",
    )
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    report = write_fixed_fpr_threshold_audit_outputs(
        root=args.root,
        output_dir=args.output_dir,
        method_faithful_collection_root=args.method_faithful_collection_root,
        t2smark_output_dir=args.t2smark_output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_pass and not report["fixed_fpr_threshold_audit_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
