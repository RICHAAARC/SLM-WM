"""攻击矩阵协议与产物重建的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.protocol.attacks import (
    AttackEvaluationBoundary,
    attack_config_digest,
    build_attack_detection_record,
    default_attack_configs,
)
from scripts.write_attack_matrix_outputs import write_attack_matrix_outputs


def json_line(value: dict[str, object]) -> str:
    """将字典转为 JSONL 单行。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def source_record(record_id: str, sample_role: str, raw_score: float, aligned_score: float) -> dict[str, object]:
    """构造 full rescue 源检测记录。"""
    return {
        "aligned_detection_record_id": f"aligned_{record_id}",
        "content_detection_record_id": f"content_{record_id}",
        "prompt_id": f"prompt_{record_id}",
        "split": "test",
        "sample_role": sample_role,
        "rescue_ablation_mode": "full_rescue",
        "raw_content_score": raw_score,
        "aligned_content_score": aligned_score,
        "fail_reason": "geometry_suspected",
        "geometry_reliable": True,
        "rescue_applied": False,
        "evidence_decision": raw_score >= 0.50,
        "supports_paper_claim": False,
    }


@pytest.mark.quick
def test_attack_config_digest_is_stable() -> None:
    """攻击配置摘要应保持稳定。"""
    config = default_attack_configs()[0]

    assert attack_config_digest(config) == attack_config_digest(config)
    assert len(attack_config_digest(config)) == 64


@pytest.mark.quick
def test_conventional_attack_reduces_retention_without_claim() -> None:
    """常规攻击代理应降低分数保持率, 且不支持论文主张。"""
    config = next(item for item in default_attack_configs() if item.attack_id == "jpeg_compression_main")
    boundary = AttackEvaluationBoundary(
        calibrated_content_threshold=0.50,
        target_fpr=0.05,
        rescue_margin_low=-0.05,
        allowed_fail_reasons=("geometry_suspected", "low_confidence"),
    )

    record = build_attack_detection_record(source_record("a", "positive_source", 0.82, 0.86), config, boundary).to_dict()

    assert record["attack_performed"] is True
    assert record["metric_status"] == "measured_from_local_proxy"
    assert record["score_retention"] < 1.0
    assert record["raw_content_score_after"] < record["raw_content_score_before"]
    assert record["attacked_image_available"] is False
    assert record["supports_paper_claim"] is False


@pytest.mark.quick
def test_regeneration_attack_requires_real_gpu_artifacts() -> None:
    """再扩散攻击在本地无真实 GPU 产物时必须保持 unsupported。"""
    config = next(item for item in default_attack_configs() if item.attack_name == "img2img_regeneration")
    boundary = AttackEvaluationBoundary(
        calibrated_content_threshold=0.50,
        target_fpr=0.05,
        rescue_margin_low=-0.05,
        allowed_fail_reasons=("geometry_suspected", "low_confidence"),
    )

    record = build_attack_detection_record(source_record("b", "positive_source", 0.82, 0.86), config, boundary).to_dict()

    assert record["attack_performed"] is False
    assert record["metric_status"] == "unsupported"
    assert record["unsupported_reason"] == "real_gpu_attack_required"
    assert record["attacked_image_digest_source"] == "not_generated"
    assert record["supports_paper_claim"] is False


@pytest.mark.quick
def test_attack_matrix_outputs_are_rebuildable(tmp_path: Path) -> None:
    """攻击矩阵产物应可由 rescue records 和 fixed-FPR 边界重建。"""
    rescue_dir = tmp_path / "outputs" / "geometric_rescue"
    calibration_dir = tmp_path / "outputs" / "threshold_calibration"
    rescue_dir.mkdir(parents=True)
    calibration_dir.mkdir(parents=True)
    records_path = rescue_dir / "aligned_detection_records.jsonl"
    records_path.write_text(
        "".join(
            json_line(record)
            for record in (
                source_record("pos", "positive_source", 0.82, 0.86),
                source_record("clean", "clean_negative", 0.20, 0.22),
                source_record("attacked", "attacked_negative", 0.48, 0.52),
            )
        ),
        encoding="utf-8",
    )
    rescue_manifest_path = rescue_dir / "manifest.local.json"
    rescue_manifest_path.write_text(json.dumps({"artifact_id": "geometric_rescue_manifest"}), encoding="utf-8")
    thresholds_path = calibration_dir / "calibration_thresholds.json"
    thresholds_path.write_text(
        json.dumps({"threshold_value": 0.50, "target_fpr": 0.05}, ensure_ascii=False),
        encoding="utf-8",
    )
    threshold_report_path = calibration_dir / "threshold_degeneracy_report.json"
    threshold_report_path.write_text(
        json.dumps(
            {
                "calibrated_content_threshold": 0.50,
                "target_fpr": 0.05,
                "rescue_margin_low": -0.05,
                "allowed_fail_reasons": ["geometry_suspected", "low_confidence"],
                "full_method_claim_ready": False,
                "supports_paper_claim": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calibration_manifest_path = calibration_dir / "manifest.local.json"
    calibration_manifest_path.write_text(json.dumps({"artifact_id": "threshold_calibration_manifest"}), encoding="utf-8")

    manifest = write_attack_matrix_outputs(
        root=tmp_path,
        rescue_records_path=records_path,
        rescue_manifest_path=rescue_manifest_path,
        calibration_thresholds_path=thresholds_path,
        threshold_report_path=threshold_report_path,
        calibration_manifest_path=calibration_manifest_path,
        max_source_records=None,
    )
    output_dir = tmp_path / "outputs" / "attack_matrix"
    attack_manifest = json.loads((output_dir / "attack_manifest.json").read_text(encoding="utf-8"))
    family_rows = list(csv.DictReader((output_dir / "attack_family_metrics.csv").open(encoding="utf-8")))
    registry_lines = (output_dir / "attacked_image_registry.jsonl").read_text(encoding="utf-8").splitlines()

    assert manifest["artifact_id"] == "attack_matrix_manifest"
    assert attack_manifest["attack_metrics_ready"] is True
    assert attack_manifest["gpu_attack_unsupported_count"] > 0
    assert (output_dir / "attacked_images").is_dir()
    assert len(registry_lines) == attack_manifest["attack_record_count"]
    assert any(row["metric_status"] == "unsupported" for row in family_rows)
    assert all(row["supports_paper_claim"] == "False" for row in family_rows)
