"""论文图表证据审计链路的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from main.analysis.paper_evidence_audit import (
    AuditInputBundle,
    build_builder_readiness_report,
    build_claim_audit_rows,
    build_evidence_gap_rows,
    build_submission_blocker_report,
    build_table_readiness_rows,
    build_figure_readiness_rows,
)
from scripts.write_paper_artifact_evidence_audit_outputs import write_paper_artifact_evidence_audit_outputs


def make_audit_input_bundle() -> AuditInputBundle:
    """构造最小审计输入, 用于验证 claim、产物和缺口的边界判定。"""
    return AuditInputBundle(
        threshold_report={
            "raw_content_claim_ready": True,
            "full_method_claim_ready": False,
            "perceptual_metrics_ready": True,
            "threshold_degenerate": False,
            "supports_paper_claim": False,
        },
        threshold_manifest={"artifact_id": "threshold_calibration_manifest", "supports_paper_claim": False},
        attack_manifest={
            "full_method_claim_ready": False,
            "gpu_attack_unsupported_count": 1,
            "supports_paper_claim": False,
        },
        attack_matrix_manifest={"artifact_id": "attack_matrix_manifest", "supports_paper_claim": False},
        baseline_manifest={"artifact_id": "external_baseline_comparison_manifest", "supports_paper_claim": False},
        baseline_runtime_report={"baseline_results_ready": False, "supports_paper_claim": False},
        ablation_manifest={"artifact_id": "internal_ablation_evidence_manifest", "supports_paper_claim": False},
        ablation_claim_summary={"mechanism_coverage_ready": True, "supports_paper_claim": False},
        source_path_map={
            "threshold_report": "outputs/threshold_calibration/threshold_degeneracy_report.json",
            "attack_manifest": "outputs/attack_matrix/attack_manifest.json",
            "baseline_runtime_report": "outputs/external_baseline_comparison/baseline_runtime_report.json",
            "baseline_comparison_table": "outputs/external_baseline_comparison/baseline_comparison_table.csv",
            "ablation_claim_summary": "outputs/internal_ablation_evidence/ablation_claim_summary.json",
            "quality_metrics_summary": "outputs/threshold_calibration/quality_metrics_summary.csv",
        },
    )


@pytest.mark.quick
def test_claim_audit_reports_current_paper_evidence_boundary() -> None:
    """审计表应明确区分工程可重建证据、预览证据和不可支持的论文主张。"""
    bundle = make_audit_input_bundle()
    claim_rows = build_claim_audit_rows(bundle)
    table_rows = build_table_readiness_rows(bundle)
    figure_rows = build_figure_readiness_rows(bundle)
    gap_rows = build_evidence_gap_rows(bundle)
    builder_report = build_builder_readiness_report(claim_rows, table_rows, figure_rows)
    blocker_report = build_submission_blocker_report(claim_rows, gap_rows, builder_report)

    claims_by_id = {row["claim_id"]: row for row in claim_rows}
    gap_ids = {row["gap_id"] for row in gap_rows}

    assert len(claim_rows) >= 7
    assert claims_by_id["claim_baseline_superiority"]["claim_decision"] == "unsupported"
    assert claims_by_id["claim_attack_robustness_under_common_matrix"]["claim_decision"] == "preview_only"
    assert "gap_real_attacked_image_closed_loop" in gap_ids
    assert builder_report["artifact_builder_ready"] is True
    assert blocker_report["submission_ready"] is False
    assert blocker_report["critical_gap_count"] >= 4
    assert all(row["supports_paper_claim"] is False for row in claim_rows + table_rows + figure_rows + gap_rows)


def write_json(path: Path, value: dict) -> None:
    """以 UTF-8 写入稳定 JSON, 供脚本在临时目录中读取。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_minimal_upstream_artifacts(tmp_path: Path) -> None:
    """写入论文证据审计脚本所需的最小上游产物。"""
    write_json(
        tmp_path / "outputs" / "threshold_calibration" / "threshold_degeneracy_report.json",
        {
            "raw_content_claim_ready": True,
            "full_method_claim_ready": False,
            "perceptual_metrics_ready": True,
            "threshold_degenerate": False,
            "supports_paper_claim": False,
        },
    )
    write_json(
        tmp_path / "outputs" / "threshold_calibration" / "manifest.local.json",
        {"artifact_id": "threshold_calibration_manifest", "supports_paper_claim": False},
    )
    write_json(
        tmp_path / "outputs" / "attack_matrix" / "attack_manifest.json",
        {"full_method_claim_ready": False, "gpu_attack_unsupported_count": 1, "supports_paper_claim": False},
    )
    write_json(
        tmp_path / "outputs" / "attack_matrix" / "manifest.local.json",
        {"artifact_id": "attack_matrix_manifest", "supports_paper_claim": False},
    )
    write_json(
        tmp_path / "outputs" / "external_baseline_comparison" / "manifest.local.json",
        {"artifact_id": "external_baseline_comparison_manifest", "supports_paper_claim": False},
    )
    write_json(
        tmp_path / "outputs" / "external_baseline_comparison" / "baseline_runtime_report.json",
        {"baseline_results_ready": False, "supports_paper_claim": False},
    )
    write_json(
        tmp_path / "outputs" / "internal_ablation_evidence" / "manifest.local.json",
        {"artifact_id": "internal_ablation_evidence_manifest", "supports_paper_claim": False},
    )
    write_json(
        tmp_path / "outputs" / "internal_ablation_evidence" / "ablation_claim_summary.json",
        {"mechanism_coverage_ready": True, "supports_paper_claim": False},
    )


@pytest.mark.quick
def test_paper_artifact_evidence_outputs_are_rebuildable_and_claim_safe(tmp_path: Path) -> None:
    """脚本应从受治理输入重建审计产物, 且不得把当前结果标记为论文级主张。"""
    write_minimal_upstream_artifacts(tmp_path)

    manifest = write_paper_artifact_evidence_audit_outputs(root=tmp_path)
    output_dir = tmp_path / "outputs" / "paper_artifact_evidence_audit"
    expected_files = {
        "claim_audit_table.csv",
        "paper_table_readiness.csv",
        "paper_figure_readiness.csv",
        "evidence_gap_list.csv",
        "artifact_builder_readiness_report.json",
        "evidence_audit_dry_run.json",
        "submission_blocker_report.json",
        "manifest.local.json",
    }

    assert expected_files == {path.name for path in output_dir.iterdir()}
    assert manifest["artifact_id"] == "paper_artifact_evidence_audit_manifest"
    assert manifest["metadata"]["submission_ready"] is False
    assert manifest["metadata"]["paper_artifact_audit_ready"] is True

    claim_rows = list(csv.DictReader((output_dir / "claim_audit_table.csv").open(encoding="utf-8")))
    gap_rows = list(csv.DictReader((output_dir / "evidence_gap_list.csv").open(encoding="utf-8")))
    blocker_report = json.loads((output_dir / "submission_blocker_report.json").read_text(encoding="utf-8"))
    dry_run_report = json.loads((output_dir / "evidence_audit_dry_run.json").read_text(encoding="utf-8"))

    assert len(claim_rows) >= 7
    assert any(row["claim_id"] == "claim_baseline_superiority" and row["claim_decision"] == "unsupported" for row in claim_rows)
    assert any(row["gap_id"] == "gap_baseline_results" for row in gap_rows)
    assert all(row["supports_paper_claim"] == "False" for row in claim_rows + gap_rows)
    assert blocker_report["submission_ready"] is False
    assert blocker_report["supports_paper_claim"] is False
    assert dry_run_report["dry_run_decision"] == "pass"
