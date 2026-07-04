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
        baseline_small_sample_manifest={"artifact_id": "primary_baseline_small_sample_evidence_manifest", "supports_paper_claim": False},
        baseline_small_sample_summary={
            "small_sample_evidence_ready": True,
            "covered_primary_baseline_count": 4,
            "formal_full_paper_run_requested": False,
            "paper_claim_ready": False,
            "supports_paper_claim": False,
        },
        dataset_quality_manifest={"artifact_id": "dataset_level_quality_manifest", "supports_paper_claim": False},
        dataset_quality_summary={
            "dataset_level_quality_proxy_ready": True,
            "formal_fid_kid_ready": False,
            "supports_paper_claim": False,
        },
        ablation_manifest={"artifact_id": "internal_ablation_evidence_manifest", "supports_paper_claim": False},
        ablation_claim_summary={"mechanism_coverage_ready": True, "supports_paper_claim": False},
        source_path_map={
            "threshold_report": "outputs/threshold_calibration/threshold_degeneracy_report.json",
            "attack_manifest": "outputs/attack_matrix/attack_manifest.json",
            "baseline_runtime_report": "outputs/external_baseline_comparison/baseline_runtime_report.json",
            "baseline_comparison_table": "outputs/external_baseline_comparison/baseline_comparison_table.csv",
            "baseline_small_sample_summary": "outputs/primary_baseline_small_sample_evidence/primary_baseline_small_sample_evidence_summary.json",
            "baseline_small_sample_records": "outputs/primary_baseline_small_sample_evidence/primary_baseline_small_sample_evidence_records.jsonl",
            "baseline_small_sample_comparison_table": "outputs/primary_baseline_small_sample_evidence/primary_baseline_small_sample_comparison_table.csv",
            "dataset_quality_summary": "outputs/dataset_level_quality/dataset_quality_summary.json",
            "dataset_quality_metrics": "outputs/dataset_level_quality/dataset_quality_metrics.csv",
            "ablation_claim_summary": "outputs/internal_ablation_evidence/ablation_claim_summary.json",
            "quality_metrics_summary": "outputs/threshold_calibration/quality_metrics_summary.csv",
        },
    )


def make_real_attack_ready_bundle() -> AuditInputBundle:
    """构造真实攻击闭环已进入 manifest 的审计输入。"""
    bundle = make_audit_input_bundle()
    ready_attack_manifest = {
        **bundle.attack_manifest,
        "gpu_attack_unsupported_count": 0,
        "real_attacked_image_closed_loop_ready": True,
        "formal_attack_detection_ready": True,
        "real_attacked_image_count": 4,
        "required_regeneration_attack_count": 4,
        "measured_regeneration_attack_count": 4,
        "regeneration_attack_gpu_validation_ready": True,
    }
    return AuditInputBundle(
        threshold_report=bundle.threshold_report,
        threshold_manifest=bundle.threshold_manifest,
        attack_manifest=ready_attack_manifest,
        attack_matrix_manifest=bundle.attack_matrix_manifest,
        baseline_manifest=bundle.baseline_manifest,
        baseline_runtime_report=bundle.baseline_runtime_report,
        baseline_small_sample_manifest=bundle.baseline_small_sample_manifest,
        baseline_small_sample_summary=bundle.baseline_small_sample_summary,
        dataset_quality_manifest=bundle.dataset_quality_manifest,
        dataset_quality_summary=bundle.dataset_quality_summary,
        ablation_manifest=bundle.ablation_manifest,
        ablation_claim_summary=bundle.ablation_claim_summary,
        source_path_map=bundle.source_path_map,
    )


def make_boundary_ready_bundle() -> AuditInputBundle:
    """构造真实攻击与 fixed-FPR / rescue 边界均已闭合的小样本审计输入。"""
    bundle = make_real_attack_ready_bundle()
    ready_threshold_report = {
        **bundle.threshold_report,
        "fixed_fpr_and_rescue_boundary_ready": True,
        "fixed_fpr_boundary_ready": True,
        "rescue_boundary_ready": True,
    }
    return AuditInputBundle(
        threshold_report=ready_threshold_report,
        threshold_manifest=bundle.threshold_manifest,
        attack_manifest=bundle.attack_manifest,
        attack_matrix_manifest=bundle.attack_matrix_manifest,
        baseline_manifest=bundle.baseline_manifest,
        baseline_runtime_report=bundle.baseline_runtime_report,
        baseline_small_sample_manifest=bundle.baseline_small_sample_manifest,
        baseline_small_sample_summary=bundle.baseline_small_sample_summary,
        dataset_quality_manifest=bundle.dataset_quality_manifest,
        dataset_quality_summary=bundle.dataset_quality_summary,
        ablation_manifest=bundle.ablation_manifest,
        ablation_claim_summary=bundle.ablation_claim_summary,
        source_path_map=bundle.source_path_map,
    )


def make_ablation_claim_ready_bundle() -> AuditInputBundle:
    """构造内部强消融 standalone claim 已闭合的审计输入。"""
    bundle = make_boundary_ready_bundle()
    ready_ablation_summary = {
        **bundle.ablation_claim_summary,
        "mechanism_coverage_ready": True,
        "ablation_claim_gate_ready": True,
        "strong_ablation_standalone_claim_ready": True,
        "supports_paper_claim": True,
    }
    return AuditInputBundle(
        threshold_report=bundle.threshold_report,
        threshold_manifest=bundle.threshold_manifest,
        attack_manifest=bundle.attack_manifest,
        attack_matrix_manifest=bundle.attack_matrix_manifest,
        baseline_manifest=bundle.baseline_manifest,
        baseline_runtime_report=bundle.baseline_runtime_report,
        baseline_small_sample_manifest=bundle.baseline_small_sample_manifest,
        baseline_small_sample_summary=bundle.baseline_small_sample_summary,
        dataset_quality_manifest=bundle.dataset_quality_manifest,
        dataset_quality_summary=bundle.dataset_quality_summary,
        ablation_manifest=bundle.ablation_manifest,
        ablation_claim_summary=ready_ablation_summary,
        source_path_map=bundle.source_path_map,
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
    assert claims_by_id["claim_baseline_small_sample_evidence_boundary"]["primary_blocker"] == "not_full_paper_claim"
    assert claims_by_id["claim_attack_robustness_under_common_matrix"]["claim_decision"] == "preview_only"
    assert "gap_real_attacked_image_closed_loop" in gap_ids
    assert builder_report["artifact_builder_ready"] is True
    assert blocker_report["submission_ready"] is False
    assert blocker_report["critical_gap_count"] >= 4
    assert all(row["supports_paper_claim"] is False for row in claim_rows + table_rows + figure_rows + gap_rows)


@pytest.mark.quick
def test_real_attack_ready_manifest_removes_real_attack_gap_items() -> None:
    """真实攻击闭环 ready 后, 审计不应继续报告对应旧缺口。"""
    bundle = make_real_attack_ready_bundle()
    claim_rows = build_claim_audit_rows(bundle)
    table_rows = build_table_readiness_rows(bundle)
    figure_rows = build_figure_readiness_rows(bundle)
    gap_rows = build_evidence_gap_rows(bundle)
    builder_report = build_builder_readiness_report(claim_rows, table_rows, figure_rows)
    blocker_report = build_submission_blocker_report(claim_rows, gap_rows, builder_report)

    claims_by_id = {row["claim_id"]: row for row in claim_rows}
    table_by_id = {row["audit_item_id"]: row for row in table_rows}
    figure_by_id = {row["audit_item_id"]: row for row in figure_rows}
    gap_ids = {row["gap_id"] for row in gap_rows}

    assert "gap_real_attacked_image_closed_loop" not in gap_ids
    assert "gap_regeneration_attack_gpu_validation" not in gap_ids
    assert claims_by_id["claim_attack_robustness_under_common_matrix"]["primary_blocker"] == "record_level_proxy_boundary"
    assert table_by_id["table_attack_robustness"]["primary_blocker"] == "record_level_proxy_boundary"
    assert table_by_id["table_baseline_small_sample_evidence"]["primary_blocker"] == "not_full_paper_claim"
    assert figure_by_id["figure_attack_robustness"]["primary_blocker"] == "record_level_proxy_boundary"
    assert "真实攻击闭环" not in blocker_report["recommended_next_action"]
    assert "完整方法 fixed-FPR 重校准" in blocker_report["recommended_next_action"]


@pytest.mark.quick
def test_ready_fixed_fpr_and_rescue_boundary_removes_recalibration_gap() -> None:
    """真实攻击闭环和阈值边界均 ready 后, 审计不应继续要求重复重校准。"""
    bundle = make_boundary_ready_bundle()
    claim_rows = build_claim_audit_rows(bundle)
    table_rows = build_table_readiness_rows(bundle)
    figure_rows = build_figure_readiness_rows(bundle)
    gap_rows = build_evidence_gap_rows(bundle)
    builder_report = build_builder_readiness_report(claim_rows, table_rows, figure_rows)
    blocker_report = build_submission_blocker_report(claim_rows, gap_rows, builder_report)

    gap_ids = {row["gap_id"] for row in gap_rows}

    assert "gap_full_method_fixed_fpr_recalibration" not in gap_ids
    assert {"gap_baseline_results", "gap_full_main_sample_scale", "gap_dataset_level_fid_kid"}.issubset(gap_ids)
    assert "完整方法 fixed-FPR 重校准" not in blocker_report["recommended_next_action"]
    assert blocker_report["submission_ready"] is False


@pytest.mark.quick
def test_ablation_claim_gate_marks_ablation_claim_artifacts_ready() -> None:
    """内部强消融门禁 ready 后, claim、表格和图数据应显式支持 standalone claim。"""
    bundle = make_ablation_claim_ready_bundle()
    claim_rows = build_claim_audit_rows(bundle)
    table_rows = build_table_readiness_rows(bundle)
    figure_rows = build_figure_readiness_rows(bundle)

    claims_by_id = {row["claim_id"]: row for row in claim_rows}
    table_by_id = {row["audit_item_id"]: row for row in table_rows}
    figure_by_id = {row["audit_item_id"]: row for row in figure_rows}

    assert claims_by_id["claim_internal_mechanism_necessity"]["claim_decision"] == "paper_supported"
    assert claims_by_id["claim_internal_mechanism_necessity"]["supports_paper_claim"] is True
    assert table_by_id["table_internal_ablation"]["paper_ready"] is True
    assert table_by_id["table_internal_ablation"]["supports_paper_claim"] is True
    assert figure_by_id["figure_ablation_delta"]["paper_ready"] is True
    assert figure_by_id["figure_ablation_delta"]["supports_paper_claim"] is True


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
        tmp_path / "outputs" / "primary_baseline_small_sample_evidence" / "manifest.local.json",
        {"artifact_id": "primary_baseline_small_sample_evidence_manifest", "supports_paper_claim": False},
    )
    write_json(
        tmp_path / "outputs" / "primary_baseline_small_sample_evidence" / "primary_baseline_small_sample_evidence_summary.json",
        {
            "small_sample_evidence_ready": True,
            "covered_primary_baseline_count": 4,
            "formal_full_paper_run_requested": False,
            "paper_claim_ready": False,
            "supports_paper_claim": False,
        },
    )
    write_json(
        tmp_path / "outputs" / "dataset_level_quality" / "manifest.local.json",
        {"artifact_id": "dataset_level_quality_manifest", "supports_paper_claim": False},
    )
    write_json(
        tmp_path / "outputs" / "dataset_level_quality" / "dataset_quality_summary.json",
        {
            "dataset_level_quality_proxy_ready": True,
            "formal_fid_kid_ready": False,
            "supports_paper_claim": False,
        },
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
