"""论文图表证据审计链路的轻量功能测试。"""

from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import pytest

from paper_experiments.analysis.paper_evidence_audit import (
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
    """构造正式证据尚未闭合的最小审计输入。"""

    return AuditInputBundle(
        threshold_report={
            "paper_run_name": "pilot_paper",
            "prompt_count": 700,
            "runtime_result_count": 700,
            "split_counts": {"test": 340},
            "protocol_decision": "pass",
            "raw_content_claim_ready": True,
            "full_method_claim_ready": False,
            "perceptual_metrics_ready": True,
            "scientific_operator_gate_ready": False,
            "supports_paper_claim": False,
        },
        threshold_manifest={"artifact_id": "pilot_runtime_manifest", "supports_paper_claim": False},
        threshold_audit_report={
            "method_identity_ready": False,
            "all_method_thresholds_ready": False,
            "fixed_fpr_threshold_audit_ready": False,
            "supports_paper_claim": False,
        },
        threshold_audit_manifest={
            "artifact_id": "fixed_fpr_threshold_audit_manifest",
            "supports_paper_claim": False,
        },
        attack_manifest={
            "full_method_claim_ready": False,
            "real_attacked_image_closed_loop_ready": False,
            "formal_attack_detection_ready": False,
            "supports_paper_claim": False,
        },
        attack_matrix_manifest={"artifact_id": "pilot_runtime_manifest", "supports_paper_claim": False},
        baseline_manifest={"artifact_id": "external_baseline_comparison_manifest", "supports_paper_claim": False},
        baseline_runtime_report={"baseline_results_ready": False, "supports_paper_claim": False},
        dataset_quality_manifest={"artifact_id": "dataset_level_quality_manifest", "supports_paper_claim": False},
        dataset_quality_summary={
            "formal_fid_kid_ready": False,
            "canonical_formal_feature_extractor_ready": False,
            "formal_fid_kid_claim_gate_ready": False,
            "supports_paper_claim": False,
        },
        ablation_manifest={"artifact_id": "formal_mechanism_ablation_manifest", "supports_paper_claim": False},
        ablation_claim_summary={"mechanism_coverage_ready": True, "supports_paper_claim": False},
        source_path_map={
            "threshold_report": "outputs/image_only_dataset_runtime/pilot_paper/dataset_runtime_summary.json",
            "threshold_audit_report": "outputs/fixed_fpr_threshold_audit/pilot_paper/threshold_audit_report.json",
            "attack_manifest": "outputs/attack_matrix/pilot_paper/attack_manifest.json",
            "baseline_runtime_report": "outputs/external_baseline_comparison/pilot_paper/baseline_runtime_report.json",
            "baseline_comparison_table": "outputs/external_baseline_comparison/pilot_paper/baseline_comparison_table.csv",
            "dataset_quality_summary": "outputs/dataset_level_quality/pilot_paper/dataset_quality_summary.json",
            "dataset_quality_metrics": "outputs/dataset_level_quality/pilot_paper/dataset_quality_metrics.csv",
            "ablation_claim_summary": "outputs/formal_mechanism_ablation/pilot_paper/ablation_claim_summary.json",
            "quality_metrics_summary": "outputs/image_only_dataset_runtime/pilot_paper/runtime_results.jsonl",
        },
    )


def make_real_attack_ready_bundle() -> AuditInputBundle:
    """构造真实攻击与仅图像检测均已闭合的审计输入。"""

    bundle = make_audit_input_bundle()
    attack = {
        **bundle.attack_manifest,
        "real_attacked_image_closed_loop_ready": True,
        "formal_attack_detection_ready": True,
        "real_attacked_image_count": 4,
        "required_real_gpu_attack_count": 8,
        "measured_real_gpu_attack_count": 8,
        "real_gpu_attack_validation_ready": True,
        "detector_input_access_mode": "image_key_public_model_only",
        "generation_latent_trace_required": False,
    }
    return replace(bundle, attack_manifest=attack)


def make_boundary_ready_bundle() -> AuditInputBundle:
    """构造真实攻击与 fixed-FPR / rescue 均已闭合的审计输入。"""

    bundle = make_real_attack_ready_bundle()
    threshold = {
        **bundle.threshold_report,
        "fixed_fpr_and_rescue_boundary_ready": True,
        "fixed_fpr_boundary_ready": True,
        "rescue_boundary_ready": True,
    }
    threshold_audit = {
        **bundle.threshold_audit_report,
        "method_identity_ready": True,
        "all_method_thresholds_ready": True,
        "fixed_fpr_threshold_audit_ready": True,
        "supports_paper_claim": True,
    }
    return replace(
        bundle,
        threshold_report=threshold,
        threshold_audit_report=threshold_audit,
    )


def make_ablation_claim_ready_bundle() -> AuditInputBundle:
    """构造正式机制消融声明已闭合的审计输入。"""

    bundle = make_boundary_ready_bundle()
    summary = {
        **bundle.ablation_claim_summary,
        "mechanism_coverage_ready": True,
        "ablation_claim_gate_ready": True,
        "strong_ablation_standalone_claim_ready": True,
        "supports_paper_claim": True,
    }
    return replace(bundle, ablation_claim_summary=summary)


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
    assert builder_report["artifact_builder_ready"] is False
    assert builder_report["paper_artifact_claim_ready"] is False
    assert blocker_report["submission_ready"] is False
    assert blocker_report["critical_gap_count"] >= 3
    assert all(row["supports_paper_claim"] is False for row in claim_rows + table_rows + figure_rows + gap_rows)


@pytest.mark.quick
def test_baseline_row_availability_cannot_replace_formal_comparison_gate() -> None:
    """baseline 表中存在实测行不等于完整模板、证据路径和 claim 门禁已闭合。"""

    bundle = replace(
        make_audit_input_bundle(),
        baseline_runtime_report={
            "baseline_results_ready": True,
            "comparison_table_supports_paper_claim": False,
            "supports_paper_claim": False,
        },
    )

    claim_rows = build_claim_audit_rows(bundle)
    gap_rows = build_evidence_gap_rows(bundle)
    baseline_claim = next(row for row in claim_rows if row["claim_id"] == "claim_baseline_superiority")

    assert baseline_claim["paper_claim_supported"] is False
    assert any(row["gap_id"] == "gap_baseline_results" for row in gap_rows)


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
    assert claims_by_id["claim_attack_robustness_under_common_matrix"]["primary_blocker"] == ""
    assert table_by_id["table_attack_robustness"]["primary_blocker"] == ""
    assert figure_by_id["figure_attack_robustness"]["primary_blocker"] == ""
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
    assert {"gap_baseline_results", "gap_dataset_level_fid_kid", "gap_formal_mechanism_ablation"}.issubset(gap_ids)
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
    """写出当前正式路径要求的最小上游审计产物。"""

    runtime_dir = tmp_path / "outputs" / "image_only_dataset_runtime" / "pilot_paper"
    quality_dir = tmp_path / "outputs" / "dataset_level_quality" / "pilot_paper"
    ablation_dir = tmp_path / "outputs" / "formal_mechanism_ablation" / "pilot_paper"
    threshold_audit_dir = tmp_path / "outputs" / "fixed_fpr_threshold_audit" / "pilot_paper"
    attack_dir = tmp_path / "outputs" / "attack_matrix" / "pilot_paper"
    baseline_dir = tmp_path / "outputs" / "external_baseline_comparison" / "pilot_paper"
    bundle = make_audit_input_bundle()
    write_json(runtime_dir / "dataset_runtime_summary.json", make_audit_input_bundle().threshold_report)
    write_json(runtime_dir / "manifest.local.json", {"artifact_id": "pilot_runtime_manifest"})
    write_json(threshold_audit_dir / "threshold_audit_report.json", bundle.threshold_audit_report)
    write_json(threshold_audit_dir / "manifest.local.json", bundle.threshold_audit_manifest)
    write_json(attack_dir / "attack_manifest.json", bundle.attack_manifest)
    write_json(attack_dir / "manifest.local.json", bundle.attack_matrix_manifest)
    write_json(
        baseline_dir / "manifest.local.json",
        {"artifact_id": "external_baseline_comparison_manifest", "supports_paper_claim": False},
    )
    write_json(
        baseline_dir / "baseline_runtime_report.json",
        {"primary_baseline_results_ready": False, "supports_paper_claim": False},
    )
    write_json(quality_dir / "manifest.local.json", {"artifact_id": "dataset_level_quality_manifest"})
    write_json(quality_dir / "dataset_quality_summary.json", make_audit_input_bundle().dataset_quality_summary)
    write_json(ablation_dir / "manifest.local.json", {"artifact_id": "formal_mechanism_ablation_manifest"})
    write_json(ablation_dir / "ablation_claim_summary.json", make_audit_input_bundle().ablation_claim_summary)


@pytest.mark.quick
def test_paper_artifact_evidence_outputs_are_rebuildable_and_claim_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """脚本应从受治理输入重建审计产物, 且不得把当前结果标记为论文级主张。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "pilot_paper")
    write_minimal_upstream_artifacts(tmp_path)

    manifest = write_paper_artifact_evidence_audit_outputs(root=tmp_path)
    output_dir = tmp_path / "outputs" / "paper_artifact_evidence_audit" / "pilot_paper"
    expected_files = {
        "claim_audit_table.csv",
        "paper_table_readiness.csv",
        "paper_figure_readiness.csv",
        "evidence_gap_list.csv",
        "artifact_builder_readiness_report.json",
        "evidence_audit_dry_run.json",
        "submission_blocker_report.json",
        "artifact_data_validation_report.json",
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
    data_validation_report = json.loads(
        (output_dir / "artifact_data_validation_report.json").read_text(encoding="utf-8")
    )

    assert len(claim_rows) >= 7
    assert any(row["claim_id"] == "claim_baseline_superiority" and row["claim_decision"] == "unsupported" for row in claim_rows)
    assert any(row["gap_id"] == "gap_baseline_results" for row in gap_rows)
    assert all(row["supports_paper_claim"] == "False" for row in claim_rows + gap_rows)
    assert blocker_report["submission_ready"] is False
    assert blocker_report["supports_paper_claim"] is False
    assert dry_run_report["dry_run_decision"] == "fail"
    assert data_validation_report["artifact_data_validation_ready"] is False
    assert {
        "raw_image_only_detection_records_ready",
        "score_distribution_table_ready",
        "roc_curve_points_ready",
        "det_curve_points_ready",
    }.issubset(data_validation_report["blocked_artifact_data_ids"])
    assert len(data_validation_report["source_paths"]) == 11
    raw_detection_path = (
        "outputs/image_only_dataset_runtime/pilot_paper/"
        "image_only_detection_records.jsonl"
    )
    assert data_validation_report["source_paths"][
        "raw_image_only_detection_records_ready"
    ] == raw_detection_path
    assert manifest["metadata"]["raw_image_only_detection_records_ready"] is False
    assert manifest["metadata"]["raw_image_only_detection_records_sha256"] == ""
    assert raw_detection_path in manifest["input_paths"]
    assert set(data_validation_report["source_paths"].values()).issubset(
        set(manifest["input_paths"])
    )
