"""内部机制消融证据链路的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.ablations import (
    FORMAL_ABLATION_METRIC_STATUS,
    aggregate_ablation_by_attack_family,
    build_ablation_claim_summary,
    build_ablation_records,
    default_ablation_specs,
    filter_ablation_claim_input_records,
)
from scripts.write_internal_ablation_outputs import write_internal_ablation_outputs


REQUIRED_ABLATIONS = {
    "full_slm_wm",
    "global_null_space",
    "no_semantic_mask",
    "no_semantic_jvp",
    "no_risk_weight",
    "random_basis",
    "lf_only",
    "hf_only",
    "no_hf",
    "no_lf",
    "no_tail_truncation",
    "fft_sync_only",
    "image_registration_only",
    "no_attention_anchor",
    "no_rescue",
    "no_attestation",
    "geo_direct_positive_audit",
}
CORE_CLAIM_ABLATIONS = REQUIRED_ABLATIONS - {"geo_direct_positive_audit"}


def sample_attack_record(record_id: str, sample_role: str, attack_family: str, evidence_decision: bool) -> dict:
    """构造最小攻击记录 fixture。"""
    is_positive = sample_role == "positive_source"
    return {
        "attack_record_id": record_id,
        "attack_family": attack_family,
        "attack_name": "rotation" if attack_family == "geometric_transform" else "jpeg_compression",
        "resource_profile": "probe",
        "split": "test",
        "sample_role": sample_role,
        "metric_status": "measured_from_local_proxy",
        "unsupported_reason": "",
        "evidence_decision": evidence_decision,
        "score_retention": 0.92 if is_positive else 0.80,
        "lf_score_retention": 0.94 if is_positive else 0.78,
        "hf_score_retention": 0.89 if is_positive else 0.82,
        "quality_score_proxy": 0.91,
        "attention_consistency_proxy": 0.88,
        "geometry_reliable": True,
        "rescue_applied": is_positive,
        "raw_content_score_before": 0.90 if is_positive else 0.20,
        "raw_content_score_after": 0.64 if is_positive else 0.16,
        "aligned_content_score_after": 0.74 if is_positive else 0.18,
        "supports_paper_claim": False,
    }


def formal_attack_record(record_id: str, sample_role: str, attack_family: str, evidence_decision: bool) -> dict:
    """构造可进入正式消融 claim gate 的真实图像级攻击记录。"""
    return {
        **sample_attack_record(record_id, sample_role, attack_family, evidence_decision),
        "resource_profile": "full_main",
        "metric_status": FORMAL_ABLATION_METRIC_STATUS,
        "supports_paper_claim": True,
    }


def formal_attack_manifest() -> dict:
    """构造正式攻击闭环 ready 的最小 manifest。"""
    return {
        "attack_metrics_ready": True,
        "real_attacked_image_closed_loop_ready": True,
        "formal_attack_detection_ready": True,
        "regeneration_attack_gpu_validation_ready": True,
        "formal_proxy_replacement_incomplete_count": 0,
        "evaluation_boundary": {"calibrated_content_threshold": 0.50, "target_fpr": 0.05},
        "supports_paper_claim": True,
    }


@pytest.mark.quick
def test_default_ablation_specs_cover_required_mechanisms() -> None:
    """默认消融清单应覆盖完整方法、语义、子空间、载体、几何和 attestation。"""
    specs = default_ablation_specs()
    spec_by_id = {spec.ablation_id: spec for spec in specs}

    assert set(spec_by_id) == REQUIRED_ABLATIONS
    assert spec_by_id["no_attestation"].attestation_mode == "missing"
    assert spec_by_id["geo_direct_positive_audit"].formal_method_allowed is False
    assert {spec.mechanism_group for spec in specs} >= {
        "complete_method",
        "semantic_routing",
        "safe_subspace",
        "content_carrier",
        "geometry_recovery",
        "attestation",
        "audit_control",
    }


@pytest.mark.quick
def test_ablation_records_apply_real_mechanism_changes() -> None:
    """关键消融应改变对应字段, 而不是只改方法名称。"""
    records = (
        sample_attack_record("positive_rotation", "positive_source", "geometric_transform", True),
        sample_attack_record("clean_rotation", "clean_negative", "geometric_transform", False),
    )
    rows = build_ablation_records(records, default_ablation_specs(), threshold=0.50)
    by_key = {(row["ablation_id"], row["attack_record_id"]): row for row in rows}

    full_positive = by_key[("full_slm_wm", "positive_rotation")]
    assert full_positive["ablated_detection_decision"] is True
    assert full_positive["ablated_score_retention"] == records[0]["score_retention"]

    no_rescue_positive = by_key[("no_rescue", "positive_rotation")]
    assert no_rescue_positive["ablated_rescue_applied"] is False
    assert no_rescue_positive["ablated_aligned_content_score_after"] == no_rescue_positive["ablated_raw_content_score_after"]
    assert full_positive["ablated_aligned_content_score_after"] > full_positive["ablated_raw_content_score_after"]

    no_anchor_clean = by_key[("no_attention_anchor", "clean_rotation")]
    assert no_anchor_clean["ablated_geometry_reliable"] is False
    assert no_anchor_clean["ablated_attention_consistency_proxy"] < no_anchor_clean["baseline_attention_consistency_proxy"]

    no_attestation_positive = by_key[("no_attestation", "positive_rotation")]
    assert no_attestation_positive["ablated_evidence_decision"] is True
    assert no_attestation_positive["ablated_detection_decision"] is False
    assert no_attestation_positive["attestation_available"] is False

    geo_audit_clean = by_key[("geo_direct_positive_audit", "clean_rotation")]
    assert geo_audit_clean["formal_method_allowed"] is False
    assert geo_audit_clean["ablated_detection_decision"] is True


@pytest.mark.quick
def test_ablation_preserves_real_attack_metric_status() -> None:
    """消融证据应保留真实 attacked image watermark rescore formal record 的统计来源。"""
    real_record = {
        **formal_attack_record("real_img2img", "positive_source", "regeneration_attack", True),
        "attack_name": "img2img_regeneration",
    }

    rows = build_ablation_records((real_record,), default_ablation_specs(), threshold=0.50)
    family_rows = aggregate_ablation_by_attack_family(rows)
    summary = build_ablation_claim_summary(
        default_ablation_specs(),
        rows,
        [],
        formal_attack_manifest(),
        {"metadata": {"baseline_results_ready": True}},
    )

    assert {row["metric_status"] for row in rows} == {FORMAL_ABLATION_METRIC_STATUS}
    assert {row["metric_status"] for row in family_rows} == {FORMAL_ABLATION_METRIC_STATUS}
    assert summary["unsupported_reasons"] == []
    assert summary["ablation_claim_formal_input_ready"] is True


@pytest.mark.quick
def test_ablation_claim_input_filter_excludes_probe_proxy_records() -> None:
    """正式消融 claim 输入应排除 probe proxy, 避免 mixed/local proxy 口径污染。"""
    records = (
        formal_attack_record("formal_positive", "positive_source", "standard_distortion", True),
        formal_attack_record("formal_clean", "clean_negative", "standard_distortion", False),
        sample_attack_record("probe_positive", "positive_source", "standard_distortion", True),
        {**sample_attack_record("unsupported", "positive_source", "regeneration_attack", False), "metric_status": "unsupported"},
    )

    accepted, report = filter_ablation_claim_input_records(records)

    assert len(accepted) == 2
    assert report["ablation_claim_input_record_count"] == 2
    assert report["ablation_claim_excluded_record_count"] == 2
    assert report["ablation_claim_excluded_proxy_record_count"] == 1
    assert {row["attack_record_id"] for row in accepted} == {"formal_positive", "formal_clean"}


@pytest.mark.quick
def test_ablation_claim_summary_uses_current_paper_run_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """内部消融声明摘要应跟随当前论文运行层级, 不能固定为 pilot_paper。"""

    monkeypatch.setenv("SLM_WM_PAPER_RUN_NAME", "full_paper")
    monkeypatch.delenv("SLM_WM_PROMPT_SET", raising=False)
    monkeypatch.delenv("SLM_WM_PROMPT_FILE", raising=False)

    summary = build_ablation_claim_summary(
        default_ablation_specs(),
        (),
        (),
        {"attack_metrics_ready": False, "evaluation_boundary": {"target_fpr": 0.01}},
        {"metadata": {"baseline_results_ready": False}},
    )

    assert summary["paper_claim_scale"] == "full_paper"
    assert summary["result_protocol_name"] == "full_paper_fixed_fpr_common_protocol"


def write_input_artifacts(tmp_path: Path, *, formal: bool) -> tuple[Path, Path, Path, Path, Path]:
    """写出内部消融脚本所需的最小上游输入。"""
    attack_dir = tmp_path / "outputs" / "attack_matrix"
    threshold_dir = tmp_path / "outputs" / "threshold_calibration"
    baseline_dir = tmp_path / "outputs" / "external_baseline_comparison"
    attack_dir.mkdir(parents=True)
    threshold_dir.mkdir(parents=True)
    baseline_dir.mkdir(parents=True)

    records_path = attack_dir / "attack_detection_records.jsonl"
    attack_manifest_path = attack_dir / "attack_manifest.json"
    attack_matrix_manifest_path = attack_dir / "manifest.local.json"
    threshold_report_path = threshold_dir / "threshold_degeneracy_report.json"
    baseline_manifest_path = baseline_dir / "manifest.local.json"

    if formal:
        records = [
            formal_attack_record("positive_jpeg", "positive_source", "standard_distortion", True),
            formal_attack_record("clean_jpeg", "clean_negative", "standard_distortion", False),
            formal_attack_record("positive_rotation", "positive_source", "geometric_transform", True),
            formal_attack_record("clean_rotation", "clean_negative", "geometric_transform", False),
            {**formal_attack_record("positive_img2img", "positive_source", "regeneration_attack", True), "resource_profile": "full_extra", "attack_name": "img2img_regeneration"},
            {**formal_attack_record("clean_img2img", "clean_negative", "regeneration_attack", False), "resource_profile": "full_extra", "attack_name": "img2img_regeneration"},
            sample_attack_record("probe_positive_jpeg", "positive_source", "standard_distortion", True),
        ]
        attack_manifest = formal_attack_manifest()
        baseline_manifest = {"metadata": {"baseline_results_ready": True, "supports_paper_claim": True}}
    else:
        records = [
            sample_attack_record("positive_jpeg", "positive_source", "standard_distortion", True),
            sample_attack_record("clean_jpeg", "clean_negative", "standard_distortion", False),
            sample_attack_record("attacked_rotation", "attacked_negative", "geometric_transform", False),
            {
                **sample_attack_record("unsupported_gpu", "positive_source", "regeneration_attack", False),
                "metric_status": "unsupported",
                "unsupported_reason": "real_gpu_attack_required",
            },
        ]
        attack_manifest = {
            "attack_metrics_ready": True,
            "evaluation_boundary": {"calibrated_content_threshold": 0.50, "target_fpr": 0.05},
            "supports_paper_claim": False,
        }
        baseline_manifest = {"metadata": {"baseline_results_ready": False, "supports_paper_claim": False}}

    records_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    attack_manifest_path.write_text(json.dumps(attack_manifest, ensure_ascii=False), encoding="utf-8")
    attack_matrix_manifest_path.write_text(json.dumps({"artifact_id": "attack_matrix_manifest", "config_digest": "digest"}), encoding="utf-8")
    threshold_report_path.write_text(
        json.dumps({"calibrated_content_threshold": 0.50, "target_fpr": 0.05, "supports_paper_claim": formal}, ensure_ascii=False),
        encoding="utf-8",
    )
    baseline_manifest_path.write_text(json.dumps(baseline_manifest, ensure_ascii=False), encoding="utf-8")
    return records_path, attack_manifest_path, attack_matrix_manifest_path, threshold_report_path, baseline_manifest_path


@pytest.mark.quick
def test_internal_ablation_outputs_are_rebuildable_and_claim_safe_for_proxy_inputs(tmp_path: Path) -> None:
    """只有 probe proxy 输入时, 脚本应重建治理报告但不得支持论文主张。"""
    records_path, attack_manifest_path, attack_matrix_manifest_path, threshold_report_path, baseline_manifest_path = write_input_artifacts(
        tmp_path,
        formal=False,
    )

    manifest = write_internal_ablation_outputs(
        root=tmp_path,
        attack_records_path=records_path,
        attack_manifest_path=attack_manifest_path,
        attack_matrix_manifest_path=attack_matrix_manifest_path,
        threshold_report_path=threshold_report_path,
        baseline_manifest_path=baseline_manifest_path,
    )
    output_dir = tmp_path / "outputs" / "internal_ablation_evidence"
    records = [json.loads(line) for line in (output_dir / "ablation_records.jsonl").read_text(encoding="utf-8").splitlines()]
    mechanism_rows = list(csv.DictReader((output_dir / "mechanism_ablation_table.csv").open(encoding="utf-8")))
    claim_summary = json.loads((output_dir / "ablation_claim_summary.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "internal_ablation_evidence_manifest"
    assert records == []
    assert mechanism_rows == []
    assert claim_summary["ablation_claim_input_record_count"] == 0
    assert claim_summary["ablation_claim_excluded_record_count"] == 4
    assert claim_summary["ablation_claim_gate_ready"] is False
    assert claim_summary["supports_paper_claim"] is False


@pytest.mark.quick
def test_internal_ablation_outputs_build_standalone_claim_gate_from_formal_records(tmp_path: Path) -> None:
    """正式输入应生成强消融 standalone claim gate, 且排除 probe proxy 记录。"""
    records_path, attack_manifest_path, attack_matrix_manifest_path, threshold_report_path, baseline_manifest_path = write_input_artifacts(
        tmp_path,
        formal=True,
    )

    manifest = write_internal_ablation_outputs(
        root=tmp_path,
        attack_records_path=records_path,
        attack_manifest_path=attack_manifest_path,
        attack_matrix_manifest_path=attack_matrix_manifest_path,
        threshold_report_path=threshold_report_path,
        baseline_manifest_path=baseline_manifest_path,
    )
    output_dir = tmp_path / "outputs" / "internal_ablation_evidence"
    records = [json.loads(line) for line in (output_dir / "ablation_records.jsonl").read_text(encoding="utf-8").splitlines()]
    mechanism_rows = list(csv.DictReader((output_dir / "mechanism_ablation_table.csv").open(encoding="utf-8")))
    delta_rows = list(csv.DictReader((output_dir / "method_pairwise_delta_table.csv").open(encoding="utf-8")))
    family_rows = list(csv.DictReader((output_dir / "ablation_by_attack_family.csv").open(encoding="utf-8")))
    claim_input_report = json.loads((output_dir / "ablation_claim_input_report.json").read_text(encoding="utf-8"))
    claim_summary = json.loads((output_dir / "ablation_claim_summary.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "internal_ablation_evidence_manifest"
    assert len(records) == len(REQUIRED_ABLATIONS) * 6
    assert {row["ablation_id"] for row in mechanism_rows} == REQUIRED_ABLATIONS
    assert {row["metric_status"] for row in mechanism_rows} == {FORMAL_ABLATION_METRIC_STATUS}
    assert {row["metric_status"] for row in family_rows} == {FORMAL_ABLATION_METRIC_STATUS}
    assert {row["supports_paper_claim"] for row in mechanism_rows if row["ablation_id"] in CORE_CLAIM_ABLATIONS} == {"True"}
    assert next(row for row in mechanism_rows if row["ablation_id"] == "geo_direct_positive_audit")["supports_paper_claim"] == "False"
    assert any(row["ablation_id"] == "no_attestation" and row["supports_paper_claim"] == "True" for row in delta_rows)
    assert claim_input_report["ablation_claim_excluded_proxy_record_count"] == 1
    assert claim_summary["ablation_claim_gate_ready"] is True
    assert claim_summary["strong_ablation_standalone_claim_ready"] is True
    assert claim_summary["core_ablation_ready_count"] == len(CORE_CLAIM_ABLATIONS)
    assert claim_summary["supports_paper_claim"] is True
