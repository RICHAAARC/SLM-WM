"""写出当前论文运行层级的 CPU 结果闭合语义门禁。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.artifacts.artifact_manifest import build_artifact_manifest
from experiments.artifacts.dataset_level_quality_outputs import (
    canonical_prompt_ids_for_paper_run,
)
from experiments.protocol.paper_run_config import build_paper_run_config
from experiments.protocol.splits import build_group_split_counts
from experiments.runtime.repository_environment import resolve_code_version
from main.core.digest import build_stable_digest
from paper_experiments.analysis.result_closure_gate import (
    ResultClosureGateInput,
    build_result_closure_gate_checks,
    build_result_closure_gate_report,
    build_source_file_sha256_map,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/result_closure_gate")


def stable_json_text(value: Any) -> str:
    """把 JSON 兼容对象转换为稳定文本。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _resolve_path(root_path: Path, path: str | Path) -> Path:
    """把输入路径解析为绝对路径。"""

    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()


def _relative_or_absolute(path: Path, root_path: Path) -> str:
    """优先记录相对仓库根目录的受治理路径。"""

    try:
        return path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    """读取必须存在的 JSON 对象。"""

    if not path.is_file():
        raise FileNotFoundError(f"结果闭合门禁缺少 JSON 输入: {path.as_posix()}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"结果闭合门禁输入必须是 JSON 对象: {path.as_posix()}")
    return dict(payload)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """读取必须存在且非空的 JSONL 正式记录。"""

    if not path.is_file():
        raise FileNotFoundError(f"结果闭合门禁缺少 JSONL 输入: {path.as_posix()}")
    rows = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"结果闭合门禁要求非空 JSONL 对象序列: {path.as_posix()}")
    return tuple(dict(row) for row in rows)


def _read_csv(path: Path) -> tuple[dict[str, Any], ...]:
    """读取必须存在且非空的 CSV 审计行。"""

    if not path.is_file():
        raise FileNotFoundError(f"结果闭合门禁缺少 CSV 输入: {path.as_posix()}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = tuple(dict(row) for row in csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果闭合门禁要求非空 CSV 记录: {path.as_posix()}")
    return rows


def _ensure_run_output_dir(root_path: Path, output_root: str | Path, paper_run_name: str) -> Path:
    """构造 outputs/ 下按论文运行层级隔离的门禁目录。"""

    resolved_root = _resolve_path(root_path, output_root)
    outputs_root = (root_path / "outputs").resolve()
    try:
        resolved_root.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("结果闭合门禁输出根目录必须位于 outputs/ 下") from exc
    resolved = resolved_root / paper_run_name
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _per_run_path(
    root_path: Path,
    path: str | Path | None,
    *,
    artifact_root: str,
    paper_run_name: str,
    file_name: str,
) -> Path:
    """解析显式路径, 未提供时使用统一的当前 run 产物路径。"""

    if path is not None:
        return _resolve_path(root_path, path)
    return (root_path / "outputs" / artifact_root / paper_run_name / file_name).resolve()


def write_result_closure_gate_outputs(
    *,
    root: str | Path = ".",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    attack_report_path: str | Path | None = None,
    attack_manifest_path: str | Path | None = None,
    threshold_audit_report_path: str | Path | None = None,
    threshold_audit_rows_path: str | Path | None = None,
    threshold_audit_manifest_path: str | Path | None = None,
    primary_baseline_evidence_summary_path: str | Path | None = None,
    primary_baseline_evidence_manifest_path: str | Path | None = None,
    baseline_report_path: str | Path | None = None,
    baseline_manifest_path: str | Path | None = None,
    result_records_path: str | Path | None = None,
    result_record_summary_path: str | Path | None = None,
    result_record_manifest_path: str | Path | None = None,
    common_protocol_summary_path: str | Path | None = None,
    common_protocol_schema_path: str | Path | None = None,
    common_protocol_manifest_path: str | Path | None = None,
    result_analysis_summary_path: str | Path | None = None,
    result_analysis_manifest_path: str | Path | None = None,
    ablation_summary_path: str | Path | None = None,
    ablation_manifest_path: str | Path | None = None,
    dataset_quality_summary_path: str | Path | None = None,
    dataset_quality_feature_records_path: str | Path | None = None,
    dataset_quality_feature_report_path: str | Path | None = None,
    dataset_quality_metrics_path: str | Path | None = None,
    dataset_quality_manifest_path: str | Path | None = None,
    evidence_builder_report_path: str | Path | None = None,
    evidence_blocker_report_path: str | Path | None = None,
    evidence_audit_manifest_path: str | Path | None = None,
    submission_readiness_report_path: str | Path | None = None,
    submission_readiness_manifest_path: str | Path | None = None,
    entry_review_report_path: str | Path | None = None,
    entry_review_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """读取所有正式证据并写出当前 run 的结果闭合 report 与 manifest。"""

    root_path = Path(root).resolve()
    paper_run = build_paper_run_config(root_path)
    split_counts = build_group_split_counts(paper_run.prompt_count)
    output_dir = _ensure_run_output_dir(root_path, output_root, paper_run.run_name)
    requested_prompt_path = _resolve_path(root_path, paper_run.prompt_file)
    packaged_prompt_path = (ROOT / paper_run.prompt_file).resolve()
    canonical_prompt_path = (
        requested_prompt_path if requested_prompt_path.is_file() else packaged_prompt_path
    )

    resolved_paths = {
        "canonical_prompt_file": canonical_prompt_path,
        "attack_report": _per_run_path(
            root_path,
            attack_report_path,
            artifact_root="attack_matrix",
            paper_run_name=paper_run.run_name,
            file_name="attack_manifest.json",
        ),
        "attack_manifest": _per_run_path(
            root_path,
            attack_manifest_path,
            artifact_root="attack_matrix",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "threshold_audit_report": _per_run_path(
            root_path,
            threshold_audit_report_path,
            artifact_root="fixed_fpr_threshold_audit",
            paper_run_name=paper_run.run_name,
            file_name="threshold_audit_report.json",
        ),
        "threshold_audit_rows": _per_run_path(
            root_path,
            threshold_audit_rows_path,
            artifact_root="fixed_fpr_threshold_audit",
            paper_run_name=paper_run.run_name,
            file_name="threshold_audit_rows.csv",
        ),
        "threshold_audit_manifest": _per_run_path(
            root_path,
            threshold_audit_manifest_path,
            artifact_root="fixed_fpr_threshold_audit",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "primary_baseline_evidence_summary": _per_run_path(
            root_path,
            primary_baseline_evidence_summary_path,
            artifact_root="primary_baseline_evidence",
            paper_run_name=paper_run.run_name,
            file_name="primary_baseline_evidence_summary.json",
        ),
        "primary_baseline_evidence_manifest": _per_run_path(
            root_path,
            primary_baseline_evidence_manifest_path,
            artifact_root="primary_baseline_evidence",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "baseline_report": _per_run_path(
            root_path,
            baseline_report_path,
            artifact_root="external_baseline_comparison",
            paper_run_name=paper_run.run_name,
            file_name="baseline_runtime_report.json",
        ),
        "baseline_manifest": _per_run_path(
            root_path,
            baseline_manifest_path,
            artifact_root="external_baseline_comparison",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "result_records": _per_run_path(
            root_path,
            result_records_path,
            artifact_root="pilot_paper_fixed_fpr_results",
            paper_run_name=paper_run.run_name,
            file_name="pilot_paper_result_records.jsonl",
        ),
        "result_record_summary": _per_run_path(
            root_path,
            result_record_summary_path,
            artifact_root="pilot_paper_fixed_fpr_results",
            paper_run_name=paper_run.run_name,
            file_name="pilot_paper_result_record_summary.json",
        ),
        "result_record_manifest": _per_run_path(
            root_path,
            result_record_manifest_path,
            artifact_root="pilot_paper_fixed_fpr_results",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "common_protocol_summary": _per_run_path(
            root_path,
            common_protocol_summary_path,
            artifact_root="pilot_paper_fixed_fpr_common_protocol",
            paper_run_name=paper_run.run_name,
            file_name="pilot_paper_common_protocol_summary.json",
        ),
        "common_protocol_schema": _per_run_path(
            root_path,
            common_protocol_schema_path,
            artifact_root="pilot_paper_fixed_fpr_common_protocol",
            paper_run_name=paper_run.run_name,
            file_name="pilot_paper_result_import_schema.json",
        ),
        "common_protocol_manifest": _per_run_path(
            root_path,
            common_protocol_manifest_path,
            artifact_root="pilot_paper_fixed_fpr_common_protocol",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "result_analysis_summary": _per_run_path(
            root_path,
            result_analysis_summary_path,
            artifact_root="pilot_paper_result_analysis",
            paper_run_name=paper_run.run_name,
            file_name="result_analysis_summary.json",
        ),
        "result_analysis_manifest": _per_run_path(
            root_path,
            result_analysis_manifest_path,
            artifact_root="pilot_paper_result_analysis",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "ablation_summary": _per_run_path(
            root_path,
            ablation_summary_path,
            artifact_root="formal_mechanism_ablation",
            paper_run_name=paper_run.run_name,
            file_name="ablation_claim_summary.json",
        ),
        "ablation_manifest": _per_run_path(
            root_path,
            ablation_manifest_path,
            artifact_root="formal_mechanism_ablation",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "dataset_quality_summary": _per_run_path(
            root_path,
            dataset_quality_summary_path,
            artifact_root="dataset_level_quality",
            paper_run_name=paper_run.run_name,
            file_name="dataset_quality_summary.json",
        ),
        "dataset_quality_feature_report": _per_run_path(
            root_path,
            dataset_quality_feature_report_path,
            artifact_root="dataset_level_quality",
            paper_run_name=paper_run.run_name,
            file_name="dataset_quality_formal_feature_import_report.json",
        ),
        "dataset_quality_feature_records": _per_run_path(
            root_path,
            dataset_quality_feature_records_path,
            artifact_root="dataset_level_quality",
            paper_run_name=paper_run.run_name,
            file_name="dataset_quality_formal_feature_records.jsonl",
        ),
        "dataset_quality_metrics": _per_run_path(
            root_path,
            dataset_quality_metrics_path,
            artifact_root="dataset_level_quality",
            paper_run_name=paper_run.run_name,
            file_name="dataset_quality_metrics.csv",
        ),
        "dataset_quality_manifest": _per_run_path(
            root_path,
            dataset_quality_manifest_path,
            artifact_root="dataset_level_quality",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "evidence_builder_report": _per_run_path(
            root_path,
            evidence_builder_report_path,
            artifact_root="paper_artifact_evidence_audit",
            paper_run_name=paper_run.run_name,
            file_name="artifact_builder_readiness_report.json",
        ),
        "evidence_blocker_report": _per_run_path(
            root_path,
            evidence_blocker_report_path,
            artifact_root="paper_artifact_evidence_audit",
            paper_run_name=paper_run.run_name,
            file_name="submission_blocker_report.json",
        ),
        "evidence_audit_manifest": _per_run_path(
            root_path,
            evidence_audit_manifest_path,
            artifact_root="paper_artifact_evidence_audit",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
        "submission_readiness_report": _per_run_path(
            root_path,
            submission_readiness_report_path,
            artifact_root="submission_readiness",
            paper_run_name=paper_run.run_name,
            file_name="readiness_blocker_report.json",
        ),
        "submission_readiness_manifest": _per_run_path(
            root_path,
            submission_readiness_manifest_path,
            artifact_root="submission_readiness",
            paper_run_name=paper_run.run_name,
            file_name="submission_readiness_manifest.local.json",
        ),
        "entry_review_report": _per_run_path(
            root_path,
            entry_review_report_path,
            artifact_root="evidence_closure_entry_review",
            paper_run_name=paper_run.run_name,
            file_name="entry_review_report.json",
        ),
        "entry_review_manifest": _per_run_path(
            root_path,
            entry_review_manifest_path,
            artifact_root="evidence_closure_entry_review",
            paper_run_name=paper_run.run_name,
            file_name="manifest.local.json",
        ),
    }
    closure_source_file_sha256 = build_source_file_sha256_map(
        resolved_paths.values(),
        root=root_path,
    )
    closure_source_file_digest = build_stable_digest(closure_source_file_sha256)
    canonical_prompt_ids = canonical_prompt_ids_for_paper_run(
        root_path=root_path,
        prompt_set=paper_run.prompt_set,
        prompt_file=paper_run.prompt_file,
    )
    bundle = ResultClosureGateInput(
        expected_paper_claim_scale=paper_run.run_name,
        expected_target_fpr=paper_run.target_fpr,
        expected_prompt_count=paper_run.prompt_count,
        expected_test_count=int(split_counts["test"]),
        expected_prompt_id_digest=build_stable_digest(sorted(canonical_prompt_ids)),
        attack_report=_read_json(resolved_paths["attack_report"]),
        attack_manifest=_read_json(resolved_paths["attack_manifest"]),
        threshold_audit_report=_read_json(resolved_paths["threshold_audit_report"]),
        threshold_audit_rows=_read_csv(resolved_paths["threshold_audit_rows"]),
        threshold_audit_manifest=_read_json(resolved_paths["threshold_audit_manifest"]),
        primary_baseline_evidence_summary=_read_json(
            resolved_paths["primary_baseline_evidence_summary"]
        ),
        primary_baseline_evidence_manifest=_read_json(
            resolved_paths["primary_baseline_evidence_manifest"]
        ),
        baseline_report=_read_json(resolved_paths["baseline_report"]),
        baseline_manifest=_read_json(resolved_paths["baseline_manifest"]),
        result_records=_read_jsonl(resolved_paths["result_records"]),
        result_record_summary=_read_json(resolved_paths["result_record_summary"]),
        result_record_manifest=_read_json(resolved_paths["result_record_manifest"]),
        common_protocol_summary=_read_json(resolved_paths["common_protocol_summary"]),
        common_protocol_schema=_read_json(resolved_paths["common_protocol_schema"]),
        common_protocol_manifest=_read_json(resolved_paths["common_protocol_manifest"]),
        result_analysis_summary=_read_json(resolved_paths["result_analysis_summary"]),
        result_analysis_manifest=_read_json(resolved_paths["result_analysis_manifest"]),
        ablation_summary=_read_json(resolved_paths["ablation_summary"]),
        ablation_manifest=_read_json(resolved_paths["ablation_manifest"]),
        dataset_quality_summary=_read_json(resolved_paths["dataset_quality_summary"]),
        dataset_quality_feature_report=_read_json(
            resolved_paths["dataset_quality_feature_report"]
        ),
        dataset_quality_metrics=_read_csv(resolved_paths["dataset_quality_metrics"]),
        dataset_quality_feature_records_sha256=closure_source_file_sha256[
            _relative_or_absolute(
                resolved_paths["dataset_quality_feature_records"],
                root_path,
            )
        ],
        dataset_quality_manifest=_read_json(resolved_paths["dataset_quality_manifest"]),
        evidence_builder_report=_read_json(resolved_paths["evidence_builder_report"]),
        evidence_blocker_report=_read_json(resolved_paths["evidence_blocker_report"]),
        evidence_audit_manifest=_read_json(resolved_paths["evidence_audit_manifest"]),
        submission_readiness_report=_read_json(resolved_paths["submission_readiness_report"]),
        submission_readiness_manifest=_read_json(resolved_paths["submission_readiness_manifest"]),
        entry_review_report=_read_json(resolved_paths["entry_review_report"]),
        entry_review_manifest=_read_json(resolved_paths["entry_review_manifest"]),
    )
    checks = build_result_closure_gate_checks(bundle)
    report = build_result_closure_gate_report(bundle, checks)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["closure_source_file_sha256"] = closure_source_file_sha256
    report["closure_source_file_digest"] = closure_source_file_digest

    report_path = output_dir / "result_closure_gate_report.json"
    manifest_path = output_dir / "manifest.local.json"
    report_path.write_text(stable_json_text(report), encoding="utf-8")
    input_bundle_digest = build_stable_digest(bundle.to_dict())
    report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    manifest_config = {
        "paper_claim_scale": paper_run.run_name,
        "target_fpr": paper_run.target_fpr,
        "expected_prompt_count": paper_run.prompt_count,
        "expected_test_count": int(split_counts["test"]),
        "expected_prompt_id_digest": bundle.expected_prompt_id_digest,
        "input_bundle_digest": input_bundle_digest,
        "report_digest": report_digest,
        "source_artifact_digests": report["source_artifact_digests"],
        "closure_source_file_sha256": closure_source_file_sha256,
        "closure_source_file_digest": closure_source_file_digest,
    }
    manifest = build_artifact_manifest(
        artifact_id=f"{paper_run.run_name}_result_closure_gate_manifest",
        artifact_type="local_manifest",
        input_paths=tuple(_relative_or_absolute(path, root_path) for path in resolved_paths.values()),
        output_paths=(
            _relative_or_absolute(report_path, root_path),
            _relative_or_absolute(manifest_path, root_path),
        ),
        config=manifest_config,
        code_version=resolve_code_version(root_path),
        rebuild_command="python scripts/write_result_closure_gate_outputs.py --require-pass",
        metadata={
            "paper_claim_scale": report["paper_claim_scale"],
            "target_fpr": report["target_fpr"],
            "blocked_check_count": report["blocked_check_count"],
            "evidence_closure_allowed": report["evidence_closure_allowed"],
            "result_closure_ready": report["result_closure_ready"],
            "closure_decision": report["closure_decision"],
            "supports_paper_claim": report["supports_paper_claim"],
            "generated_at": report["generated_at"],
            "expected_prompt_id_digest": report["expected_prompt_id_digest"],
            "input_bundle_digest": input_bundle_digest,
            "report_digest": report_digest,
            "source_artifact_digests": report["source_artifact_digests"],
            "closure_source_file_sha256": closure_source_file_sha256,
            "closure_source_file_digest": closure_source_file_digest,
        },
    ).to_dict()
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="写出当前 run 的论文结果闭合语义门禁。")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--attack-report-path", default=None)
    parser.add_argument("--attack-manifest-path", default=None)
    parser.add_argument("--threshold-audit-report-path", default=None)
    parser.add_argument("--threshold-audit-rows-path", default=None)
    parser.add_argument("--threshold-audit-manifest-path", default=None)
    parser.add_argument("--primary-baseline-evidence-summary-path", default=None)
    parser.add_argument("--primary-baseline-evidence-manifest-path", default=None)
    parser.add_argument("--baseline-report-path", default=None)
    parser.add_argument("--baseline-manifest-path", default=None)
    parser.add_argument("--result-records-path", default=None)
    parser.add_argument("--result-record-summary-path", default=None)
    parser.add_argument("--result-record-manifest-path", default=None)
    parser.add_argument("--common-protocol-summary-path", default=None)
    parser.add_argument("--common-protocol-schema-path", default=None)
    parser.add_argument("--common-protocol-manifest-path", default=None)
    parser.add_argument("--result-analysis-summary-path", default=None)
    parser.add_argument("--result-analysis-manifest-path", default=None)
    parser.add_argument("--ablation-summary-path", default=None)
    parser.add_argument("--ablation-manifest-path", default=None)
    parser.add_argument("--dataset-quality-summary-path", default=None)
    parser.add_argument("--dataset-quality-feature-records-path", default=None)
    parser.add_argument("--dataset-quality-feature-report-path", default=None)
    parser.add_argument("--dataset-quality-metrics-path", default=None)
    parser.add_argument("--dataset-quality-manifest-path", default=None)
    parser.add_argument("--evidence-builder-report-path", default=None)
    parser.add_argument("--evidence-blocker-report-path", default=None)
    parser.add_argument("--evidence-audit-manifest-path", default=None)
    parser.add_argument("--submission-readiness-report-path", default=None)
    parser.add_argument("--submission-readiness-manifest-path", default=None)
    parser.add_argument("--entry-review-report-path", default=None)
    parser.add_argument("--entry-review-manifest-path", default=None)
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main() -> None:
    """命令行入口, `--require-pass` 在阻断状态下返回非零。"""

    args = build_parser().parse_args()
    report = write_result_closure_gate_outputs(
        root=args.root,
        output_root=args.output_root,
        attack_report_path=args.attack_report_path,
        attack_manifest_path=args.attack_manifest_path,
        threshold_audit_report_path=args.threshold_audit_report_path,
        threshold_audit_rows_path=args.threshold_audit_rows_path,
        threshold_audit_manifest_path=args.threshold_audit_manifest_path,
        primary_baseline_evidence_summary_path=args.primary_baseline_evidence_summary_path,
        primary_baseline_evidence_manifest_path=args.primary_baseline_evidence_manifest_path,
        baseline_report_path=args.baseline_report_path,
        baseline_manifest_path=args.baseline_manifest_path,
        result_records_path=args.result_records_path,
        result_record_summary_path=args.result_record_summary_path,
        result_record_manifest_path=args.result_record_manifest_path,
        common_protocol_summary_path=args.common_protocol_summary_path,
        common_protocol_schema_path=args.common_protocol_schema_path,
        common_protocol_manifest_path=args.common_protocol_manifest_path,
        result_analysis_summary_path=args.result_analysis_summary_path,
        result_analysis_manifest_path=args.result_analysis_manifest_path,
        ablation_summary_path=args.ablation_summary_path,
        ablation_manifest_path=args.ablation_manifest_path,
        dataset_quality_summary_path=args.dataset_quality_summary_path,
        dataset_quality_feature_records_path=args.dataset_quality_feature_records_path,
        dataset_quality_feature_report_path=args.dataset_quality_feature_report_path,
        dataset_quality_metrics_path=args.dataset_quality_metrics_path,
        dataset_quality_manifest_path=args.dataset_quality_manifest_path,
        evidence_builder_report_path=args.evidence_builder_report_path,
        evidence_blocker_report_path=args.evidence_blocker_report_path,
        evidence_audit_manifest_path=args.evidence_audit_manifest_path,
        submission_readiness_report_path=args.submission_readiness_report_path,
        submission_readiness_manifest_path=args.submission_readiness_manifest_path,
        entry_review_report_path=args.entry_review_report_path,
        entry_review_manifest_path=args.entry_review_manifest_path,
    )
    print(stable_json_text(report), end="")
    if args.require_pass and not report["result_closure_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
