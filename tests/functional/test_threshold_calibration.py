"""fixed-FPR 阈值校准与指标产物的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

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


@pytest.mark.quick
def test_threshold_calibration_outputs_are_rebuildable_and_keep_fpr_scopes_separate(tmp_path: Path) -> None:
    """阈值、clean FPR、attacked FPR 和质量指标应可由 records 重建。"""
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
    assert any(row["quality_metric_name"] == "lpips" and row["metric_status"] == "unsupported" for row in quality_rows)
