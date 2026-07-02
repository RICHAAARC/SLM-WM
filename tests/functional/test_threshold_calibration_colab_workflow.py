"""Threshold calibration Colab 工作流 helper 测试。"""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from main.methods.detection import RESCUE_ABLATION_MODES
from paper_workflow.colab_utils.threshold_calibration import (
    package_threshold_calibration_outputs,
    parse_optional_record_limit,
    run_default_threshold_calibration_from_drive_plan,
)


def json_line(value: dict[str, object]) -> str:
    """把字典转换为 JSONL 单行。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def content_record(record_id: str, split: str, sample_role: str, score: float) -> dict[str, object]:
    """构造最小内容检测记录。"""
    return {
        "content_detection_record_id": record_id,
        "prompt_id": "prompt_sample",
        "split": split,
        "content_score": score,
        "metadata": {"sample_role": sample_role, "supports_paper_claim": False},
        "supports_paper_claim": False,
    }


def write_attention_injection_package(package_path: Path) -> None:
    """写出可驱动几何恢复重建的最小 attention latent injection 包。"""
    package_path.parent.mkdir(parents=True, exist_ok=True)
    nested_geometry_path = package_path.parent / "nested_geometry.zip"
    geometry_record = {
        "geometry_evidence_record_id": "geometry_evidence_sample",
        "attention_graph_id": "attention_graph_sample",
        "capture_id": "capture_sample",
        "attention_relation_consistency": 0.92,
        "anchor_inlier_ratio": 0.8,
        "registration_confidence": 0.86,
        "recovered_sync_consistency": 0.88,
        "alignment_residual": 0.12,
        "geometry_reliable": True,
        "direct_positive_decision": False,
        "supports_paper_claim": False,
        "metadata": {"anchor_graph_digest": "digest_sample"},
    }
    with ZipFile(nested_geometry_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "outputs/attention_geometry/geometry_evidence_records.jsonl",
            json_line(geometry_record),
        )
        archive.writestr(
            "outputs/attention_geometry/geometry_evidence_summary.json",
            json.dumps({"attention_geometry_ready": True, "supports_paper_claim": False}, ensure_ascii=False),
        )
    carrier_record = {
        "carrier_id": "carrier_sample",
        "fallback_mode": "active_update",
        "attention_graph_id": "attention_graph_sample",
        "metadata": {"prompt_id": "prompt_sample", "supports_paper_claim": False},
        "supports_paper_claim": False,
    }
    with ZipFile(package_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "outputs/attention_latent_injection/attention_latent_injection_result.json",
            json.dumps(
                {
                    "run_decision": "pass",
                    "image_quality_metrics_ready": True,
                    "latent_update_count": 1,
                    "supports_paper_claim": False,
                },
                ensure_ascii=False,
            ),
        )
        archive.writestr(
            "outputs/attention_latent_update/attention_update_summary.json",
            json.dumps({"attention_geometry_ready": True, "supports_paper_claim": False}, ensure_ascii=False),
        )
        archive.writestr(
            "outputs/attention_latent_update/attention_carrier_records.jsonl",
            json_line(carrier_record),
        )
        archive.write(
            nested_geometry_path,
            "outputs/attention_latent_injection/input_packages/attention_geometry_package_sample.zip",
        )


def write_aligned_rescoring_package(package_path: Path, records: list[dict[str, object]] | None = None) -> None:
    """写出包含内容检测记录与 pair-level 质量指标的最小 aligned rescoring 包。"""
    package_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_records = records or [
        content_record("cal_pos", "calibration", "positive_source", 0.90),
        content_record("cal_clean_a", "calibration", "clean_negative", 0.10),
        content_record("cal_clean_b", "calibration", "clean_negative", 0.20),
        content_record("test_pos", "test", "positive_source", 0.82),
        content_record("test_clean", "test", "clean_negative", 0.15),
        content_record("test_attacked", "test", "attacked_negative", 0.25),
    ]
    quality_text = (
        "carrier_id,psnr,ssim,mse,mean_abs_error,lpips,lpips_status,clip_score,clip_score_clean,clip_score_aligned,"
        "clip_score_delta,clip_score_status,fid,fid_status,kid,kid_status\n"
        "carrier_sample,31.0,0.98,0.001,0.02,0.10,measured,0.32,0.31,0.32,0.01,measured,"
        "unsupported,dataset_level_metric_not_computed_in_pair_run,"
        "unsupported,dataset_level_metric_not_computed_in_pair_run\n"
    )
    with ZipFile(package_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "outputs/aligned_rescoring/aligned_rescoring_result.json",
            json.dumps(
                {
                    "run_decision": "pass",
                    "image_quality_metrics_ready": True,
                    "perceptual_metrics_ready": True,
                    "aligned_rescoring_record_count": 3,
                    "real_aligned_rescore_count": 3,
                },
                ensure_ascii=False,
            ),
        )
        archive.writestr("outputs/aligned_rescoring/aligned_rescoring_quality_metrics.csv", quality_text)
        archive.writestr(
            "outputs/content_carriers/content_detection_records.jsonl",
            "".join(json_line(record) for record in resolved_records),
        )


@pytest.mark.quick
def test_optional_record_limit_parser_treats_all_as_unbounded() -> None:
    """threshold calibration helper 应把 all 解析为不截断。"""
    assert parse_optional_record_limit("all") is None
    assert parse_optional_record_limit("unlimited") is None
    assert parse_optional_record_limit(None) is None
    assert parse_optional_record_limit("96") == 96


@pytest.mark.quick
def test_threshold_calibration_drive_workflow_generates_package_ready_outputs(tmp_path: Path) -> None:
    """Drive workflow helper 应从前序 zip 重建阈值产物并可镜像结果包。"""
    attention_dir = tmp_path / "drive" / "attention_latent_injection"
    aligned_dir = tmp_path / "drive" / "aligned_rescoring"
    write_attention_injection_package(attention_dir / "attention_latent_injection_package_20260621t000000z_sample.zip")
    write_aligned_rescoring_package(aligned_dir / "aligned_rescoring_package_20260621t000000z_sample.zip")

    summary = run_default_threshold_calibration_from_drive_plan(
        root=tmp_path,
        attention_injection_drive_dir=str(attention_dir),
        aligned_rescoring_drive_dir=str(aligned_dir),
        target_fpr=0.5,
        minimum_clean_negative_count=0,
    )
    record = package_threshold_calibration_outputs(root=tmp_path, drive_output_dir=str(tmp_path / "drive" / "threshold_calibration"))
    archive_path = tmp_path / record.archive_path

    assert summary["run_decision"] == "pass"
    assert summary["threshold_calibration_ready"] is True
    assert summary["geometric_rescue_ready"] is True
    assert summary["geometric_rescue_record_count"] > 0
    assert summary["metadata"]["fixed_fpr_control_scope"] == "calibration_clean_negative"
    assert summary["metadata"]["rescue_control_scope"] == "evidence_clean_negative"
    assert summary["metadata"]["attacked_negative_governs_fixed_fpr"] is False
    assert summary["metadata"]["evidence_fpr_exceeds_target"] is False
    assert summary["metadata"]["attacked_fpr_diagnostic_exceeds_target"] is True
    assert archive_path.exists()
    assert record.archive_digest == record.drive_archive_digest
    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "outputs/threshold_calibration/calibration_thresholds.json" in names
        assert "outputs/geometric_rescue/aligned_detection_records.jsonl" in names
        assert "outputs/content_carriers/content_detection_records.jsonl" in names
        assert "outputs/threshold_calibration/threshold_calibration_archive_summary.json" in names
        assert "outputs/threshold_calibration/threshold_calibration_archive_manifest.local.json" in names
        embedded_summary = json.loads(
            archive.read("outputs/threshold_calibration/threshold_calibration_archive_summary.json").decode("utf-8")
        )
        assert embedded_summary["metadata"]["archive_payload_digest"]
        assert embedded_summary["metadata"]["archive_digest_scope"] == "external_sidecar_after_archive_write"
        assert "archive_digest" not in embedded_summary


@pytest.mark.quick
def test_threshold_calibration_drive_workflow_uses_all_content_records_by_default(tmp_path: Path) -> None:
    """pilot_paper 阈值校准路径默认不应沿用 96 条诊断截断。"""
    attention_dir = tmp_path / "drive" / "attention_latent_injection"
    aligned_dir = tmp_path / "drive" / "aligned_rescoring"
    write_attention_injection_package(attention_dir / "attention_latent_injection_package_20260621t000000z_sample.zip")
    records: list[dict[str, object]] = []
    for index in range(40):
        records.extend(
            [
                content_record(f"cal_pos_{index}", "calibration", "positive_source", 0.90),
                content_record(f"cal_clean_{index}", "calibration", "clean_negative", 0.01 + index * 0.001),
                content_record(f"test_attacked_{index}", "test", "attacked_negative", 0.20),
            ]
        )
    write_aligned_rescoring_package(
        aligned_dir / "aligned_rescoring_package_20260621t000000z_sample.zip",
        records=records,
    )

    summary = run_default_threshold_calibration_from_drive_plan(
        root=tmp_path,
        attention_injection_drive_dir=str(attention_dir),
        aligned_rescoring_drive_dir=str(aligned_dir),
        target_fpr=0.5,
        minimum_clean_negative_count=0,
    )

    assert summary["run_decision"] == "pass"
    assert summary["metadata"]["max_content_records"] is None
    assert summary["geometric_rescue_record_count"] == len(records) * len(RESCUE_ABLATION_MODES)
