"""fixed-FPR 阈值校准与指标产物的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.protocol.calibration import FixedFprCalibrationConfig, empirical_threshold_at_fpr
from scripts.write_threshold_calibration_outputs import write_threshold_calibration_outputs


def json_line(value: dict[str, object]) -> str:
    """将字典转为 JSONL 单行。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


@pytest.mark.quick
def test_fixed_fpr_threshold_uses_calibration_clean_negative_only() -> None:
    """fixed-FPR 阈值应只由 calibration clean negative 冻结。"""
    config = FixedFprCalibrationConfig(target_fpr=0.25)
    threshold = empirical_threshold_at_fpr([0.1, 0.2, 0.3, 0.4], config)

    assert threshold.threshold_value == 0.4
    assert threshold.calibration_negative_count == 4
    assert threshold.allowed_false_positive_count == 1
    assert threshold.observed_fpr == 0.25
    assert threshold.metadata["threshold_source"] == "calibration_clean_negative"
    assert threshold.supports_paper_claim is False


def rescue_record(
    record_id: str,
    split: str,
    sample_role: str,
    raw_score: float,
    aligned_score: float | None = None,
) -> dict[str, object]:
    """构造 full-rescue 检测记录。"""
    return {
        "aligned_detection_record_id": record_id,
        "content_detection_record_id": f"content_{record_id}",
        "prompt_id": f"prompt_{record_id}",
        "split": split,
        "sample_role": sample_role,
        "rescue_ablation_mode": "full_rescue",
        "raw_content_score": raw_score,
        "aligned_content_score": raw_score if aligned_score is None else aligned_score,
        "rescue_score_gain": 0.0 if aligned_score is None else aligned_score - raw_score,
        "fail_reason": "geometry_suspected",
        "geometry_reliable": True,
        "supports_paper_claim": False,
    }


def write_rescue_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """写出可被阈值校准脚本读取的最小 rescue 输入。"""
    rescue_dir = tmp_path / "outputs" / "geometric_rescue"
    rescue_dir.mkdir(parents=True)
    records_path = rescue_dir / "aligned_detection_records.jsonl"
    audit_path = rescue_dir / "geometry_rescue_audit.json"
    records = [
        rescue_record("cal_pos", "calibration", "positive_source", 0.90),
        rescue_record("cal_clean_a", "calibration", "clean_negative", 0.10),
        rescue_record("cal_clean_b", "calibration", "clean_negative", 0.20),
        rescue_record("test_pos", "test", "positive_source", 0.80),
        rescue_record("test_clean", "test", "clean_negative", 0.30),
        rescue_record("test_attacked", "test", "attacked_negative", 0.25, 0.35),
    ]
    records_path.write_text("".join(json_line(record) for record in records), encoding="utf-8")
    audit_path.write_text(
        json.dumps(
            {
                "attention_geometry_ready": True,
                "image_quality_metrics_ready": True,
                "attention_latent_injection_package_path": "",
                "supports_paper_claim": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return records_path, audit_path


def write_boundary_separation_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """写出 attacked negative 超标但 clean negative 仍满足 fixed-FPR 的输入。"""
    rescue_dir = tmp_path / "outputs" / "geometric_rescue"
    rescue_dir.mkdir(parents=True)
    records_path = rescue_dir / "aligned_detection_records.jsonl"
    audit_path = rescue_dir / "geometry_rescue_audit.json"
    records = [
        rescue_record("cal_pos", "calibration", "positive_source", 0.90),
        rescue_record("cal_clean_a", "calibration", "clean_negative", 0.10),
        rescue_record("cal_clean_b", "calibration", "clean_negative", 0.20),
        rescue_record("cal_clean_c", "calibration", "clean_negative", 0.30),
        rescue_record("cal_clean_d", "calibration", "clean_negative", 0.40),
        rescue_record("test_clean", "test", "clean_negative", 0.10),
        rescue_record("test_attacked", "test", "attacked_negative", 0.45),
    ]
    records_path.write_text("".join(json_line(record) for record in records), encoding="utf-8")
    audit_path.write_text(
        json.dumps(
            {
                "attention_geometry_ready": True,
                "image_quality_metrics_ready": True,
                "attention_latent_injection_package_path": "",
                "supports_paper_claim": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return records_path, audit_path


def write_aligned_rescoring_package(package_path: Path) -> None:
    """写出包含 LPIPS 与 CLIP pair-level 指标的最小 aligned rescoring 结果包。"""
    package_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "run_decision": "pass",
        "image_quality_metrics_ready": True,
        "perceptual_metrics_ready": True,
        "aligned_rescoring_record_count": 3,
        "real_aligned_rescore_count": 3,
    }
    quality_text = (
        "carrier_id,psnr,ssim,mse,mean_abs_error,lpips,lpips_status,clip_score,clip_score_clean,clip_score_aligned,"
        "clip_score_delta,clip_score_status,fid,fid_status,kid,kid_status\n"
        "carrier_a,30.0,0.99,0.001,0.02,0.12,measured,0.31,0.30,0.31,0.01,measured,"
        "unsupported,dataset_level_metric_not_computed_in_pair_run,"
        "unsupported,dataset_level_metric_not_computed_in_pair_run\n"
    )
    with ZipFile(package_path, "w") as archive:
        archive.writestr(
            "outputs/aligned_rescoring/aligned_rescoring_result.json",
            json.dumps(result, ensure_ascii=False),
        )
        archive.writestr("outputs/aligned_rescoring/aligned_rescoring_quality_metrics.csv", quality_text)


@pytest.mark.quick
def test_threshold_calibration_outputs_are_rebuildable_and_keep_fpr_scopes_separate(tmp_path: Path) -> None:
    """阈值、clean FPR、attacked FPR 和质量指标应可由 records 重建。"""
    records_path, audit_path = write_rescue_inputs(tmp_path)

    manifest = write_threshold_calibration_outputs(
        root=tmp_path,
        rescue_records_path=records_path,
        rescue_audit_path=audit_path,
        target_fpr=0.5,
    )
    output_dir = tmp_path / "outputs" / "threshold_calibration"
    threshold_report = json.loads((output_dir / "threshold_degeneracy_report.json").read_text(encoding="utf-8"))
    operating_rows = list(csv.DictReader((output_dir / "fixed_fpr_operating_points.csv").open(encoding="utf-8")))
    fpr_rows = list(csv.DictReader((output_dir / "rescue_fpr_audit.csv").open(encoding="utf-8")))
    quality_rows = list(csv.DictReader((output_dir / "quality_metrics_summary.csv").open(encoding="utf-8")))

    assert manifest["artifact_id"] == "threshold_calibration_manifest"
    assert threshold_report["metadata"]["threshold_source"] == "calibration_clean_negative"
    assert threshold_report["full_method_claim_ready"] is False
    assert operating_rows[0]["supports_paper_claim"] == "False"
    assert {row["decision_scope"] for row in fpr_rows} == {
        "raw_content_clean_negative",
        "evidence_clean_negative",
        "evidence_attacked_negative",
    }
    fpr_by_scope = {row["decision_scope"]: row for row in fpr_rows}
    assert fpr_by_scope["raw_content_clean_negative"]["statistical_boundary"] == "fixed_fpr_raw_clean_control"
    assert fpr_by_scope["raw_content_clean_negative"]["governs_fixed_fpr"] == "True"
    assert fpr_by_scope["evidence_clean_negative"]["statistical_boundary"] == "fixed_fpr_evidence_clean_control"
    assert fpr_by_scope["evidence_clean_negative"]["governs_fixed_fpr"] == "True"
    assert fpr_by_scope["evidence_attacked_negative"]["statistical_boundary"] == "attack_robustness_diagnostic"
    assert fpr_by_scope["evidence_attacked_negative"]["governs_fixed_fpr"] == "False"
    assert threshold_report["fixed_fpr_control_scope"] == "calibration_clean_negative"
    assert threshold_report["rescue_control_scope"] == "evidence_clean_negative"
    assert threshold_report["attacked_negative_governs_fixed_fpr"] is False
    assert threshold_report["rescue_changes_fpr_denominator"] is False
    assert any(row["quality_metric_name"] == "lpips" and row["metric_status"] == "unsupported" for row in quality_rows)


@pytest.mark.quick
def test_attacked_negative_fpr_is_diagnostic_not_fixed_fpr_denominator(tmp_path: Path) -> None:
    """attacked negative FPR 超标不应改变 fixed-FPR 的 clean negative 控制边界。"""
    records_path, audit_path = write_boundary_separation_inputs(tmp_path)

    write_threshold_calibration_outputs(
        root=tmp_path,
        rescue_records_path=records_path,
        rescue_audit_path=audit_path,
        target_fpr=0.5,
    )
    output_dir = tmp_path / "outputs" / "threshold_calibration"
    threshold_report = json.loads((output_dir / "threshold_degeneracy_report.json").read_text(encoding="utf-8"))
    fpr_rows = {
        row["decision_scope"]: row
        for row in csv.DictReader((output_dir / "rescue_fpr_audit.csv").open(encoding="utf-8"))
    }

    assert threshold_report["clean_fpr_exceeds_target"] is False
    assert threshold_report["attacked_fpr_diagnostic_exceeds_target"] is True
    assert threshold_report["evidence_fpr_exceeds_target"] is False
    assert threshold_report["fixed_fpr_and_rescue_boundary_ready"] is True
    assert fpr_rows["evidence_attacked_negative"]["fpr_exceeds_target"] == "True"
    assert fpr_rows["evidence_attacked_negative"]["governs_fixed_fpr"] == "False"


@pytest.mark.quick
def test_evidence_clean_fpr_governs_threshold_protocol_decision(tmp_path: Path) -> None:
    """rescue 后 clean negative FPR 超标时, 阈值协议必须失败。"""
    rescue_dir = tmp_path / "outputs" / "geometric_rescue"
    rescue_dir.mkdir(parents=True)
    records_path = rescue_dir / "aligned_detection_records.jsonl"
    audit_path = rescue_dir / "geometry_rescue_audit.json"
    records = [
        rescue_record("cal_pos", "calibration", "positive_source", 0.90),
        rescue_record("cal_clean_a", "calibration", "clean_negative", 0.10),
        rescue_record("cal_clean_b", "calibration", "clean_negative", 0.20),
        rescue_record("test_clean_rescued", "test", "clean_negative", 0.18, 0.25),
    ]
    records_path.write_text("".join(json_line(record) for record in records), encoding="utf-8")
    audit_path.write_text(
        json.dumps({"attention_geometry_ready": True, "image_quality_metrics_ready": True}, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = write_threshold_calibration_outputs(
        root=tmp_path,
        rescue_records_path=records_path,
        rescue_audit_path=audit_path,
        target_fpr=0.5,
    )
    threshold_report = json.loads(
        (tmp_path / "outputs" / "threshold_calibration" / "threshold_degeneracy_report.json").read_text(encoding="utf-8")
    )

    assert threshold_report["raw_content_claim_ready"] is True
    assert threshold_report["evidence_fpr_exceeds_target"] is True
    assert threshold_report["fixed_fpr_and_rescue_boundary_ready"] is False
    assert manifest["metadata"]["protocol_decision"] == "fail"


@pytest.mark.quick
def test_minimum_clean_negative_count_gates_fixed_fpr_readiness(tmp_path: Path) -> None:
    """pilot_paper 规模门控不足时, 即使经验 FPR 满足也不能标记为 ready。"""
    records_path, audit_path = write_rescue_inputs(tmp_path)

    manifest = write_threshold_calibration_outputs(
        root=tmp_path,
        rescue_records_path=records_path,
        rescue_audit_path=audit_path,
        target_fpr=0.5,
        minimum_clean_negative_count=100,
    )
    threshold_report = json.loads(
        (tmp_path / "outputs" / "threshold_calibration" / "threshold_degeneracy_report.json").read_text(encoding="utf-8")
    )

    assert threshold_report["minimum_clean_negative_count"] == 100
    assert threshold_report["calibration_negative_count_ready"] is False
    assert threshold_report["evidence_clean_negative_count_ready"] is False
    assert threshold_report["minimum_clean_negative_count_ready"] is False
    assert threshold_report["fixed_fpr_and_rescue_boundary_ready"] is False
    assert manifest["metadata"]["protocol_decision"] == "fail"


@pytest.mark.quick
def test_threshold_calibration_propagates_aligned_rescoring_pair_metrics(tmp_path: Path) -> None:
    """最新真实 aligned rescoring 质量指标应进入阈值校准摘要和 manifest。"""
    records_path, audit_path = write_rescue_inputs(tmp_path)
    package_path = tmp_path / "outputs" / "aligned_rescoring_package_20260620t17281781976491z_b37b14f.zip"
    write_aligned_rescoring_package(package_path)

    manifest = write_threshold_calibration_outputs(
        root=tmp_path,
        rescue_records_path=records_path,
        rescue_audit_path=audit_path,
        target_fpr=0.5,
        aligned_rescoring_package_path=package_path,
    )
    output_dir = tmp_path / "outputs" / "threshold_calibration"
    threshold_report = json.loads((output_dir / "threshold_degeneracy_report.json").read_text(encoding="utf-8"))
    quality_rows = {
        row["quality_metric_name"]: row
        for row in csv.DictReader((output_dir / "quality_metrics_summary.csv").open(encoding="utf-8"))
    }

    assert quality_rows["psnr"]["quality_metric_value"] == "30.0"
    assert quality_rows["psnr"]["quality_metric_source"] == "aligned_rescoring_package"
    assert quality_rows["lpips"]["quality_metric_value"] == "0.12"
    assert quality_rows["lpips"]["metric_status"] == "measured"
    assert quality_rows["clip_score_clean"]["quality_metric_value"] == "0.30"
    assert quality_rows["clip_score_aligned"]["quality_metric_value"] == "0.31"
    assert quality_rows["clip_score_delta"]["quality_metric_value"] == "0.01"
    assert quality_rows["fid"]["metric_status"] == "dataset_level_metric_not_computed_in_pair_run"
    assert threshold_report["aligned_rescoring_quality_metrics_ready"] is True
    assert threshold_report["real_aligned_rescore_count"] == 3
    assert manifest["metadata"]["aligned_rescoring_quality_metrics_ready"] is True
    assert "outputs/aligned_rescoring_package_20260620t17281781976491z_b37b14f.zip" in manifest["input_paths"]
