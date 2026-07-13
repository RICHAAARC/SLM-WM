"""验证证据闭合入口的纯判定逻辑, 不绕过正式聚合 Writer."""

from __future__ import annotations

import pytest

from paper_experiments.analysis.evidence_closure_entry_review import (
    EvidenceClosureEntryInput,
    build_evidence_closure_entry_checklist,
    build_evidence_closure_entry_review_report,
)


pytestmark = pytest.mark.quick


def _bundle(*, ready: bool) -> EvidenceClosureEntryInput:
    """构造阻断态或完整态的受治理内存输入."""

    return EvidenceClosureEntryInput(
        submission_readiness_report={
            "submission_ready": ready,
            "readiness_decision": "ready" if ready else "blocked",
            "artifact_builder_ready": True,
            "release_dry_run_ready": True,
            "required_input_count": 0 if ready else 2,
            "critical_required_input_count": 0 if ready else 2,
            "primary_blockers": [] if ready else [
                "gap_baseline_results",
                "gap_paper_run_sample_scale",
            ],
            "recommended_next_action": (
                "进入证据闭合" if ready else "补齐正式证据"
            ),
        },
        required_evidence_rows=(
            ()
            if ready
            else (
                {"required_input_id": "gap_baseline_results"},
                {"required_input_id": "gap_paper_run_sample_scale"},
            )
        ),
        paper_blocker_report={"blocking_claim_count": 0 if ready else 5},
        baseline_runtime_report={
            "comparison_table_supports_paper_claim": ready,
            "supports_paper_claim": ready,
            "primary_baseline_formal_ready": ready,
            "primary_baseline_results_ready": ready,
            "primary_baseline_formal_template_coverage_ready": ready,
            "primary_baseline_formal_evidence_collection_ready": ready,
            "formal_import_validation_ready": ready,
            "formal_evidence_path_resolution_ready": True,
            "accepted_formal_import_count": 4 if ready else 0,
        },
        dataset_quality_summary={
            "formal_fid_kid_ready": ready,
            "formal_sample_scale_ready": ready,
            "formal_feature_backend_ready": ready,
        },
    )


def test_evidence_closure_entry_review_blocks_before_formal_evidence_is_ready() -> None:
    """缺少正式证据时, 纯判定必须保留完整阻断原因."""

    bundle = _bundle(ready=False)
    checklist = build_evidence_closure_entry_checklist(bundle)
    report = build_evidence_closure_entry_review_report(bundle, checklist)

    assert report["entry_review_ready"] is True
    assert report["evidence_closure_allowed"] is False
    assert report["entry_review_decision"] == "blocked_before_evidence_closure"
    assert "formal_comparison_reference_ready" in report[
        "blocked_review_item_ids"
    ]
    assert "paper_run_sample_scale_ready" in report[
        "blocked_review_item_ids"
    ]
    assert "dataset_level_quality_ready" in report[
        "blocked_review_item_ids"
    ]
    assert all(row["supports_paper_claim"] is False for row in checklist)
    assert all(row["audit_note"] for row in checklist)


def test_evidence_closure_entry_review_allows_complete_governed_evidence() -> None:
    """全部受治理事实通过时, 纯判定应给出唯一允许状态."""

    bundle = _bundle(ready=True)
    checklist = build_evidence_closure_entry_checklist(bundle)
    report = build_evidence_closure_entry_review_report(bundle, checklist)

    assert report["entry_review_ready"] is True
    assert report["evidence_closure_allowed"] is True
    assert report["entry_review_decision"] == "ready_for_evidence_closure"
    assert report["blocked_review_item_count"] == 0
    assert report["blocked_review_item_ids"] == []
    assert {row["review_status"] for row in checklist} == {"ready"}
    assert all(row["supports_paper_claim"] is False for row in checklist)
