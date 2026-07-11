"""论文投稿级证据闭合入口审计的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.write_evidence_closure_entry_review_outputs import write_evidence_closure_entry_review_outputs


def write_json(path: Path, value: dict[str, object]) -> None:
    """写出测试 JSON 输入。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_required_inputs(path: Path, *, include_blockers: bool = True) -> None:
    """写出仍需补齐的证据输入清单。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "required_input_id",
                "required_input_area",
                "required_input_severity",
                "required_action",
                "related_artifacts",
                "closes_claim_ids",
                "recommended_order",
                "input_ready",
                "supports_paper_claim",
            ],
        )
        writer.writeheader()
        if not include_blockers:
            return
        writer.writerow(
            {
                "required_input_id": "gap_baseline_results",
                "required_input_area": "baseline_comparison",
                "required_input_severity": "critical",
                "required_action": "补齐外部 baseline 结果。",
                "related_artifacts": "outputs/external_baseline_comparison/baseline_runtime_report.json",
                "closes_claim_ids": "claim_baseline_superiority",
                "recommended_order": 1,
                "input_ready": False,
                "supports_paper_claim": False,
            }
        )
        writer.writerow(
            {
                "required_input_id": "gap_paper_run_sample_scale",
                "required_input_area": "statistical_power",
                "required_input_severity": "critical",
                "required_action": "补齐 当前运行层级完整统计。",
                "related_artifacts": "outputs/threshold_calibration",
                "closes_claim_ids": "claim_submission_ready_package",
                "recommended_order": 2,
                "input_ready": False,
                "supports_paper_claim": False,
            }
        )


@pytest.mark.quick
def test_evidence_closure_entry_review_blocks_before_formal_evidence_is_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式证据仍缺失时, 入口审计应可重建但不允许进入证据闭合."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "pilot_paper")

    submission_report = tmp_path / "outputs" / "submission_readiness" / "readiness_blocker_report.json"
    required_inputs = tmp_path / "outputs" / "submission_readiness" / "required_evidence_inputs.csv"
    paper_blocker = tmp_path / "outputs" / "paper_artifact_evidence_audit" / "submission_blocker_report.json"
    baseline_report = tmp_path / "outputs" / "external_baseline_comparison" / "baseline_runtime_report.json"
    dataset_quality = tmp_path / "outputs" / "dataset_level_quality" / "dataset_quality_summary.json"
    write_json(
        submission_report,
        {
            "submission_ready": False,
            "readiness_decision": "blocked",
            "artifact_builder_ready": True,
            "release_dry_run_ready": True,
            "required_input_count": 2,
            "critical_required_input_count": 2,
            "primary_blockers": ["gap_baseline_results", "gap_paper_run_sample_scale"],
            "recommended_next_action": "先补齐正式证据。",
        },
    )
    write_required_inputs(required_inputs)
    write_json(paper_blocker, {"blocking_claim_count": 5})
    write_json(
        baseline_report,
        {
            "primary_baseline_results_ready": False,
            "formal_import_validation_ready": False,
            "accepted_formal_import_count": 0,
            "formal_evidence_path_resolution_ready": True,
        },
    )
    write_json(
        dataset_quality,
        {
            "formal_fid_kid_ready": False,
            "formal_sample_scale_ready": False,
            "formal_feature_backend_ready": False,
        },
    )

    manifest = write_evidence_closure_entry_review_outputs(
        root=tmp_path,
        submission_readiness_report_path=submission_report,
        required_evidence_inputs_path=required_inputs,
        paper_blocker_report_path=paper_blocker,
        baseline_runtime_report_path=baseline_report,
        dataset_quality_summary_path=dataset_quality,
    )

    output_dir = tmp_path / "outputs" / "evidence_closure_entry_review" / "pilot_paper"
    report = json.loads((output_dir / "entry_review_report.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((output_dir / "entry_review_checklist.csv").open(encoding="utf-8")))

    assert manifest["artifact_id"] == "evidence_closure_entry_review_manifest"
    assert report["entry_review_ready"] is True
    assert report["evidence_closure_allowed"] is False
    assert report["entry_review_decision"] == "blocked_before_evidence_closure"
    assert "formal_comparison_reference_ready" in report["blocked_review_item_ids"]
    assert "paper_run_sample_scale_ready" in report["blocked_review_item_ids"]
    assert "dataset_level_quality_ready" in report["blocked_review_item_ids"]
    assert {row["supports_paper_claim"] for row in rows} == {"False"}
    assert all(row["audit_note"] for row in rows)


@pytest.mark.quick
def test_evidence_closure_entry_review_automatically_allows_complete_governed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全部受治理证据通过时, 入口审计应无需人工批准地进入证据闭合."""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "pilot_paper")

    submission_report = tmp_path / "outputs" / "submission_readiness" / "readiness_blocker_report.json"
    required_inputs = tmp_path / "outputs" / "submission_readiness" / "required_evidence_inputs.csv"
    paper_blocker = tmp_path / "outputs" / "paper_artifact_evidence_audit" / "submission_blocker_report.json"
    baseline_report = tmp_path / "outputs" / "external_baseline_comparison" / "baseline_runtime_report.json"
    dataset_quality = tmp_path / "outputs" / "dataset_level_quality" / "dataset_quality_summary.json"
    write_json(
        submission_report,
        {
            "submission_ready": True,
            "readiness_decision": "ready",
            "artifact_builder_ready": True,
            "release_dry_run_ready": True,
            "required_input_count": 0,
            "critical_required_input_count": 0,
            "primary_blockers": [],
            "recommended_next_action": "进入证据闭合.",
        },
    )
    write_required_inputs(required_inputs, include_blockers=False)
    write_json(paper_blocker, {"blocking_claim_count": 0})
    write_json(
        baseline_report,
        {
            "comparison_table_supports_paper_claim": True,
            "supports_paper_claim": True,
            "primary_baseline_formal_ready": True,
            "primary_baseline_results_ready": True,
            "primary_baseline_formal_template_coverage_ready": True,
            "primary_baseline_formal_evidence_collection_ready": True,
            "formal_import_validation_ready": True,
            "formal_evidence_path_resolution_ready": True,
            "accepted_formal_import_count": 4,
        },
    )
    write_json(
        dataset_quality,
        {
            "formal_fid_kid_ready": True,
            "formal_sample_scale_ready": True,
            "formal_feature_backend_ready": True,
        },
    )

    write_evidence_closure_entry_review_outputs(
        root=tmp_path,
        submission_readiness_report_path=submission_report,
        required_evidence_inputs_path=required_inputs,
        paper_blocker_report_path=paper_blocker,
        baseline_runtime_report_path=baseline_report,
        dataset_quality_summary_path=dataset_quality,
    )

    output_dir = tmp_path / "outputs" / "evidence_closure_entry_review" / "pilot_paper"
    report = json.loads((output_dir / "entry_review_report.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((output_dir / "entry_review_checklist.csv").open(encoding="utf-8")))

    assert report["entry_review_ready"] is True
    assert report["evidence_closure_allowed"] is True
    assert report["entry_review_decision"] == "ready_for_evidence_closure"
    assert report["blocked_review_item_count"] == 0
    assert report["blocked_review_item_ids"] == []
    assert {row["review_status"] for row in rows} == {"ready"}
    assert all(row["audit_note"] for row in rows)
