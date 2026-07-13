"""论文图表证据审计链路的轻量功能测试。"""

from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from experiments.ablations.necessity_statistics import (
    build_ablation_necessity_statistics,
)
from experiments.ablations.runtime_rerun import (
    FORMAL_RUNTIME_RERUN_ABLATION_IDS,
)
from experiments.protocol.paper_run_config import PaperRunPromptContract
from paper_experiments.analysis.paper_evidence_audit import (
    AuditInputBundle,
    build_builder_readiness_report,
    build_claim_audit_rows,
    build_evidence_gap_rows,
    build_submission_blocker_report,
    build_table_readiness_rows,
    build_figure_readiness_rows,
)


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
            "full_method_component_ready": False,
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
            "full_method_component_ready": False,
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
            "formal_fid_kid_component_ready": False,
            "supports_paper_claim": False,
        },
        ablation_manifest={"artifact_id": "formal_mechanism_ablation_manifest", "supports_paper_claim": False},
        ablation_component_summary={"mechanism_coverage_ready": True, "supports_paper_claim": False},
        source_path_map={
            "threshold_report": "outputs/image_only_dataset_runtime/pilot_paper/dataset_runtime_summary.json",
            "threshold_audit_report": "outputs/fixed_fpr_threshold_audit/pilot_paper/threshold_audit_report.json",
            "attack_manifest": "outputs/attack_matrix/pilot_paper/attack_manifest.json",
            "baseline_runtime_report": "outputs/external_baseline_comparison/pilot_paper/baseline_runtime_report.json",
            "baseline_comparison_table": "outputs/external_baseline_comparison/pilot_paper/baseline_comparison_table.csv",
            "dataset_quality_summary": "outputs/dataset_level_quality/pilot_paper/dataset_quality_summary.json",
            "dataset_quality_metrics": "outputs/dataset_level_quality/pilot_paper/dataset_quality_metrics.csv",
            "ablation_component_summary": "outputs/formal_mechanism_ablation/pilot_paper/ablation_component_summary.json",
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


def make_ablation_claim_ready_bundle(
    unsupported_ablation_id: str | None = None,
) -> AuditInputBundle:
    """构造正式机制消融声明已闭合的审计输入。"""

    bundle = make_boundary_ready_bundle()
    variant_ids = tuple(
        ablation_id
        for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
        if ablation_id != "complete_method"
    )
    records = [
        {
            "ablation_id": ablation_id,
            "prompt_id": f"prompt_{prompt_index:03d}",
            "split": "test",
            "formal_attack_coverage_ready": True,
            "attacked_positive_rate": (
                1.0
                if ablation_id in {"complete_method", unsupported_ablation_id}
                else 0.0
            ),
            "positive_source_positive": ablation_id
            in {"complete_method", unsupported_ablation_id},
            "paired_ssim": 0.95,
        }
        for ablation_id in FORMAL_RUNTIME_RERUN_ABLATION_IDS
        for prompt_index in range(34)
    ]
    necessity_rows, necessity_summary = build_ablation_necessity_statistics(
        records,
        expected_ablation_ids=variant_ids,
        expected_paired_prompt_count=34,
        bootstrap_resample_count=1000,
    )
    summary = {
        **bundle.ablation_component_summary,
        "mechanism_coverage_ready": True,
        "ablation_component_ready": True,
        **necessity_summary,
        "supports_paper_claim": True,
    }
    source_path_map = {
        **bundle.source_path_map,
        "mechanism_necessity_statistics": (
            "outputs/formal_mechanism_ablation/pilot_paper/"
            "mechanism_necessity_statistics.csv"
        ),
    }
    return replace(
        bundle,
        ablation_component_summary=summary,
        ablation_necessity_rows=tuple(necessity_rows),
        ablation_necessity_summary=necessity_summary,
        source_path_map=source_path_map,
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


@pytest.mark.quick
def test_measured_negative_ablation_is_honest_but_does_not_block_package() -> None:
    """完整统计中的负结论应标记 measured_not_supported, 不伪造必要性。"""

    unsupported_id = "without_attention_geometry"
    bundle = make_ablation_claim_ready_bundle(unsupported_id)
    claim_rows = build_claim_audit_rows(bundle)
    claims_by_id = {row["claim_id"]: row for row in claim_rows}

    assert claims_by_id["claim_internal_mechanism_necessity"]["claim_decision"] == (
        "measured_not_supported"
    )
    assert claims_by_id["claim_necessity_attention_geometry"]["claim_decision"] == (
        "measured_not_supported"
    )
    assert claims_by_id["claim_necessity_attention_geometry"][
        "paper_claim_supported"
    ] is False
    assert build_table_readiness_rows(bundle)[4]["paper_ready"] is True


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
    write_json(ablation_dir / "ablation_component_summary.json", make_audit_input_bundle().ablation_component_summary)


def write_test_prompt_contract(tmp_path: Path) -> PaperRunPromptContract:
    """为临时 writer 测试显式注入最小 Prompt 依赖。"""

    relative_path = Path("configs/paper_main_pilot_paper_prompts.txt")
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("a controlled prompt\n", encoding="utf-8")
    return PaperRunPromptContract(
        run_name="pilot_paper",
        prompt_file=relative_path.as_posix(),
        expected_prompt_count=1,
        prompt_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
