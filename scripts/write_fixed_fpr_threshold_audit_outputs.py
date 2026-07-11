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

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.protocol.splits import build_group_split_counts
from experiments.runtime.repository_environment import resolve_code_version
from paper_experiments.analysis.fixed_fpr_threshold_audit import (
    audit_baseline_fixed_fpr,
    audit_main_method_fixed_fpr,
    build_fixed_fpr_threshold_audit_report,
)
from paper_experiments.baselines.method_faithful_observation_collection import (
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


def write_fixed_fpr_threshold_audit_outputs(
    *,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
    method_faithful_collection_root: str | Path | None = None,
    t2smark_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """独立重算五个方法的正式阈值并写出受治理报告。"""

    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(root_path)
    split_counts = build_group_split_counts(paper_run.prompt_count)
    run_dir = root_path / "outputs" / "image_only_dataset_runtime" / paper_run.run_name
    main_observation_path = run_dir / "image_only_detection_records.jsonl"
    main_protocol_path = run_dir / "frozen_evidence_protocol.json"
    rows: list[dict[str, Any]] = [
        audit_main_method_fixed_fpr(
            _read_jsonl(main_observation_path),
            _read_json(main_protocol_path),
            target_fpr=paper_run.target_fpr,
            expected_calibration_negative_count=split_counts["calibration"],
            expected_test_negative_count=split_counts["test"],
        )
    ]

    collection_root = (
        DEFAULT_METHOD_FAITHFUL_OUTPUT_ROOT / paper_run.run_name
        if method_faithful_collection_root is None
        else Path(method_faithful_collection_root)
    )
    if not collection_root.is_absolute():
        collection_root = (root_path / collection_root).resolve()
    common_sources = load_method_faithful_observation_collection(
        collection_root,
        project_root=root_path,
    )
    for source in common_sources:
        manifest = source.transfer_manifest
        rows.append(
            audit_baseline_fixed_fpr(
                source.baseline_id,
                source.rows,
                target_fpr=paper_run.target_fpr,
                expected_calibration_negative_count=split_counts["calibration"],
                expected_test_negative_count=split_counts["test"],
                declared_threshold=float(manifest["threshold"]),
                declared_threshold_digest=str(manifest["threshold_digest"]),
            )
        )

    resolved_t2smark_dir = (
        DEFAULT_T2SMARK_OUTPUT_ROOT / paper_run.run_name
        if t2smark_output_dir is None
        else Path(t2smark_output_dir)
    )
    if not resolved_t2smark_dir.is_absolute():
        resolved_t2smark_dir = (root_path / resolved_t2smark_dir).resolve()
    t2_observation_path = resolved_t2smark_dir / "t2smark_adapter" / "baseline_observations.json"
    t2_candidate_path = resolved_t2smark_dir / "t2smark_formal_import_candidate_records.jsonl"
    t2_threshold, t2_threshold_digest = _read_candidate_threshold(
        t2_candidate_path,
        target_fpr=paper_run.target_fpr,
    )
    rows.append(
        audit_baseline_fixed_fpr(
            "t2smark",
            _read_json_array(t2_observation_path),
            target_fpr=paper_run.target_fpr,
            expected_calibration_negative_count=split_counts["calibration"],
            expected_test_negative_count=split_counts["test"],
            declared_threshold=t2_threshold,
            declared_threshold_digest=t2_threshold_digest,
        )
    )

    report = build_fixed_fpr_threshold_audit_report(
        rows,
        paper_run_name=paper_run.run_name,
        target_fpr=paper_run.target_fpr,
    )
    resolved_output_dir = (
        root_path / DEFAULT_OUTPUT_ROOT / paper_run.run_name
        if output_dir is None
        else Path(output_dir)
    )
    if not resolved_output_dir.is_absolute():
        resolved_output_dir = (root_path / resolved_output_dir).resolve()
    resolved_output_dir.relative_to((root_path / "outputs").resolve())
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = resolved_output_dir / "threshold_audit_rows.csv"
    report_path = resolved_output_dir / "threshold_audit_report.json"
    manifest_path = resolved_output_dir / "manifest.local.json"
    _write_rows_csv(rows_path, rows)
    _write_json(report_path, report)
    manifest = build_artifact_manifest(
        artifact_id="fixed_fpr_threshold_audit_manifest",
        artifact_type="local_manifest",
        input_paths=(
            main_observation_path.relative_to(root_path).as_posix(),
            main_protocol_path.relative_to(root_path).as_posix(),
            *(
                path.relative_to(root_path).as_posix()
                for source in common_sources
                for path in (
                    source.observations_path,
                    source.transfer_manifest_path,
                    source.prompt_plan_path,
                    source.adapter_manifest_path,
                    source.execution_manifest_path,
                )
            ),
            t2_observation_path.relative_to(root_path).as_posix(),
            t2_candidate_path.relative_to(root_path).as_posix(),
        ),
        output_paths=(
            rows_path.relative_to(root_path).as_posix(),
            report_path.relative_to(root_path).as_posix(),
            manifest_path.relative_to(root_path).as_posix(),
        ),
        config={"paper_run": paper_run.to_dict()},
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_fixed_fpr_threshold_audit_outputs.py --require-pass",
        metadata=report,
    ).to_dict()
    _write_json(manifest_path, manifest)
    return report


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
