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


def formal_real_attack_record(
    attack_name: str,
    score_after: float,
    evidence_decision: bool,
    *,
    attack_family: str = "regeneration_attack",
    resource_profile: str = "full_extra",
    requires_gpu: bool = True,
    split: str = "calibration",
    sample_role: str = "attacked_negative",
) -> dict[str, object]:
    """构造真实 attacked image formal record fixture。"""
    record_digest = f"digest_{attack_name}"
    return {
        "attack_record_id": f"real_{attack_name}",
        "attack_record_digest": record_digest,
        "source_record_id": "aligned_real_source",
        "source_image_digest": f"source_digest_{attack_name}",
        "source_image_digest_source": "sha256_file",
        "attack_id": f"real_{attack_name}",
        "attack_family": attack_family,
        "attack_name": attack_name,
        "attack_strength": 0.35,
        "resource_profile": resource_profile,
        "requires_gpu": requires_gpu,
        "attack_parameters": {"fixture": True},
        "attack_config_digest": f"config_digest_{attack_name}",
        "attacked_image_digest": f"attacked_digest_{attack_name}",
        "attacked_image_digest_source": "sha256_file",
        "attacked_image_available": True,
        "attack_performed": True,
        "split": split,
        "sample_role": sample_role,
        "raw_content_score_before": 0.70,
        "raw_content_score_after": score_after,
        "aligned_content_score_before": 0.72,
        "aligned_content_score_after": score_after,
        "lf_score_retention": 0.60,
        "tail_score_retention": 0.62,
        "score_retention": 0.61,
        "quality_score": 0.82,
        "attention_consistency_proxy": 0.57,
        "geometry_reliable": True,
        "rescue_eligible": False,
        "rescue_applied": False,
        "evidence_decision": evidence_decision,
        "metric_status": "measured_from_real_attacked_image_watermark_rescore_formal_protocol",
        "unsupported_reason": "",
        "supports_paper_claim": False,
        "metadata": {
            "formal_boundary_ready": True,
            "source_image_path": f"outputs/aligned_rescoring/aligned_images/{attack_name}_source.png",
            "attacked_image_path": f"outputs/real_attack_evaluation/attacked_images/{attack_name}_attacked.png",
        },
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
                "fixed_fpr_control_scope": "calibration_clean_negative",
                "fixed_fpr_denominator_role": "clean_negative_only",
                "rescue_control_scope": "evidence_clean_negative",
                "rescue_changes_fpr_denominator": False,
                "attacked_negative_boundary_role": "attack_robustness_diagnostic_not_fpr_denominator",
                "attacked_negative_governs_fixed_fpr": False,
                "aligned_rescoring_package_path": "outputs/aligned_rescoring_package_20260620t17281781976491z_b37b14f.zip",
                "aligned_rescoring_package_digest": "abc123",
                "aligned_rescoring_quality_metrics_ready": True,
                "perceptual_metrics_ready": True,
                "aligned_rescoring_record_count": 3,
                "real_aligned_rescore_count": 3,
                "full_method_claim_ready": False,
                "supports_paper_claim": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calibration_manifest_path = calibration_dir / "manifest.local.json"
    calibration_manifest_path.write_text(
        json.dumps(
            {
                "artifact_id": "threshold_calibration_manifest",
                "metadata": {"aligned_rescoring_quality_metrics_ready": True},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

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
    assert attack_manifest["aligned_rescoring_quality_metrics_ready"] is True
    assert attack_manifest["real_aligned_rescore_count"] == 3
    assert attack_manifest["evaluation_boundary"]["fixed_fpr_control_scope"] == "calibration_clean_negative"
    assert attack_manifest["evaluation_boundary"]["rescue_control_scope"] == "evidence_clean_negative"
    assert attack_manifest["evaluation_boundary"]["attacked_negative_governs_fixed_fpr"] is False
    assert manifest["metadata"]["aligned_rescoring_quality_metrics_ready"] is True
    assert (output_dir / "attacked_images").is_dir()
    assert len(registry_lines) == attack_manifest["attack_record_count"]
    assert any(row["metric_status"] == "unsupported" for row in family_rows)
    assert all(row["supports_paper_claim"] == "False" for row in family_rows)


@pytest.mark.quick
def test_attack_matrix_ingests_real_attack_formal_records(tmp_path: Path) -> None:
    """真实 attacked image formal records 应进入正式攻击矩阵统计边界。"""
    rescue_dir = tmp_path / "outputs" / "geometric_rescue"
    calibration_dir = tmp_path / "outputs" / "threshold_calibration"
    real_attack_dir = tmp_path / "outputs" / "real_attack_evaluation"
    rescue_dir.mkdir(parents=True)
    calibration_dir.mkdir(parents=True)
    real_attack_dir.mkdir(parents=True)

    records_path = rescue_dir / "aligned_detection_records.jsonl"
    records_path.write_text(
        json_line(source_record("pos", "positive_source", 0.82, 0.86)),
        encoding="utf-8",
    )
    rescue_manifest_path = rescue_dir / "manifest.local.json"
    rescue_manifest_path.write_text(json.dumps({"artifact_id": "geometric_rescue_manifest"}), encoding="utf-8")
    thresholds_path = calibration_dir / "calibration_thresholds.json"
    thresholds_path.write_text(json.dumps({"threshold_value": 0.50, "target_fpr": 0.05}), encoding="utf-8")
    threshold_report_path = calibration_dir / "threshold_degeneracy_report.json"
    threshold_report_path.write_text(
        json.dumps(
            {
                "calibrated_content_threshold": 0.50,
                "target_fpr": 0.05,
                "rescue_margin_low": -0.05,
                "allowed_fail_reasons": ["geometry_suspected", "low_confidence"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calibration_manifest_path = calibration_dir / "manifest.local.json"
    calibration_manifest_path.write_text(json.dumps({"artifact_id": "threshold_calibration_manifest"}), encoding="utf-8")
    real_attack_records_path = real_attack_dir / "formal_attack_detection_records.jsonl"
    real_records = [
        formal_real_attack_record("img2img_regeneration", 0.44, False),
        formal_real_attack_record("ddim_inversion_regeneration", 0.46, False),
        formal_real_attack_record("sdedit_regeneration", 0.48, False),
        formal_real_attack_record("diffusion_purification", 0.54, True),
        formal_real_attack_record("global_editing_attack", 0.42, False, attack_family="global_editing_attack"),
        formal_real_attack_record("local_editing_attack", 0.45, False, attack_family="local_editing_attack"),
        formal_real_attack_record("visual_paraphrase_attack", 0.40, False, attack_family="visual_paraphrase_attack"),
        formal_real_attack_record("adversarial_removal_attack", 0.43, False, attack_family="adversarial_removal_attack"),
    ]
    real_attack_records_path.write_text("".join(json_line(record) for record in real_records), encoding="utf-8")

    write_attack_matrix_outputs(
        root=tmp_path,
        rescue_records_path=records_path,
        rescue_manifest_path=rescue_manifest_path,
        calibration_thresholds_path=thresholds_path,
        threshold_report_path=threshold_report_path,
        calibration_manifest_path=calibration_manifest_path,
        real_attack_records_path=real_attack_records_path,
        max_source_records=None,
    )

    output_dir = tmp_path / "outputs" / "attack_matrix"
    attack_manifest = json.loads((output_dir / "attack_manifest.json").read_text(encoding="utf-8"))
    family_rows = list(csv.DictReader((output_dir / "attack_family_metrics.csv").open(encoding="utf-8")))
    regeneration_rows = [row for row in family_rows if row["attack_family"] == "regeneration_attack"]

    assert attack_manifest["formal_real_attack_record_count"] == 8
    assert attack_manifest["real_attacked_image_count"] == 8
    assert attack_manifest["real_attacked_image_closed_loop_ready"] is True
    assert attack_manifest["formal_attack_detection_ready"] is True
    assert attack_manifest["regeneration_attack_gpu_validation_ready"] is True
    assert attack_manifest["gpu_attack_real_measurement_missing_count"] == 0
    assert attack_manifest["gpu_attack_unsupported_count"] == 0
    assert set(attack_manifest["real_regeneration_attack_names"]) == {
        "img2img_regeneration",
        "ddim_inversion_regeneration",
        "sdedit_regeneration",
        "diffusion_purification",
    }
    assert set(attack_manifest["real_gpu_attack_names"]) == {
        "img2img_regeneration",
        "ddim_inversion_regeneration",
        "sdedit_regeneration",
        "diffusion_purification",
        "global_editing_attack",
        "local_editing_attack",
        "visual_paraphrase_attack",
        "adversarial_removal_attack",
    }
    assert regeneration_rows
    assert all(
        row["metric_status"] == "measured_from_real_attacked_image_watermark_rescore_formal_protocol"
        for row in regeneration_rows
    )


@pytest.mark.quick
def test_attack_matrix_prefers_formal_image_attack_records_over_local_proxy(tmp_path: Path) -> None:
    """已有图像级 formal records 时, 攻击矩阵应移除同配置本地代理记录。"""
    rescue_dir = tmp_path / "outputs" / "geometric_rescue"
    calibration_dir = tmp_path / "outputs" / "threshold_calibration"
    conventional_dir = tmp_path / "outputs" / "conventional_geometric_attack_evaluation"
    rescue_dir.mkdir(parents=True)
    calibration_dir.mkdir(parents=True)
    conventional_dir.mkdir(parents=True)

    records_path = rescue_dir / "aligned_detection_records.jsonl"
    records_path.write_text(
        json_line(source_record("pos", "positive_source", 0.82, 0.86)),
        encoding="utf-8",
    )
    rescue_manifest_path = rescue_dir / "manifest.local.json"
    rescue_manifest_path.write_text(json.dumps({"artifact_id": "geometric_rescue_manifest"}), encoding="utf-8")
    thresholds_path = calibration_dir / "calibration_thresholds.json"
    thresholds_path.write_text(json.dumps({"threshold_value": 0.50, "target_fpr": 0.05}), encoding="utf-8")
    threshold_report_path = calibration_dir / "threshold_degeneracy_report.json"
    threshold_report_path.write_text(
        json.dumps(
            {
                "calibrated_content_threshold": 0.50,
                "target_fpr": 0.05,
                "rescue_margin_low": -0.05,
                "allowed_fail_reasons": ["geometry_suspected", "low_confidence"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calibration_manifest_path = calibration_dir / "manifest.local.json"
    calibration_manifest_path.write_text(json.dumps({"artifact_id": "threshold_calibration_manifest"}), encoding="utf-8")
    conventional_records_path = conventional_dir / "formal_attack_detection_records.jsonl"
    conventional_record = formal_real_attack_record(
        "jpeg_compression",
        0.43,
        False,
        attack_family="standard_distortion",
        resource_profile="full_main",
        requires_gpu=False,
        split="test",
        sample_role="positive_source",
    )
    conventional_records_path.write_text(json_line(conventional_record), encoding="utf-8")

    write_attack_matrix_outputs(
        root=tmp_path,
        rescue_records_path=records_path,
        rescue_manifest_path=rescue_manifest_path,
        calibration_thresholds_path=thresholds_path,
        threshold_report_path=threshold_report_path,
        calibration_manifest_path=calibration_manifest_path,
        conventional_geometric_records_path=conventional_records_path,
        max_source_records=None,
    )

    output_dir = tmp_path / "outputs" / "attack_matrix"
    evidence_records_path = tmp_path / "outputs" / "image_attack_evidence" / "formal_attack_detection_records.jsonl"
    attack_records = [json.loads(line) for line in (output_dir / "attack_detection_records.jsonl").read_text(encoding="utf-8").splitlines()]
    family_rows = list(csv.DictReader((output_dir / "attack_family_metrics.csv").open(encoding="utf-8")))
    attack_manifest = json.loads((output_dir / "attack_manifest.json").read_text(encoding="utf-8"))
    jpeg_full_main_records = [
        record
        for record in attack_records
        if record["attack_family"] == "standard_distortion"
        and record["attack_name"] == "jpeg_compression"
        and record["resource_profile"] == "full_main"
    ]
    jpeg_full_main_rows = [
        row
        for row in family_rows
        if row["attack_family"] == "standard_distortion"
        and row["attack_name"] == "jpeg_compression"
        and row["resource_profile"] == "full_main"
    ]

    assert evidence_records_path.exists()
    assert len(evidence_records_path.read_text(encoding="utf-8").splitlines()) == 1
    assert len(jpeg_full_main_records) == 1
    assert jpeg_full_main_records[0]["metric_status"] == "measured_from_real_attacked_image_watermark_rescore_formal_protocol"
    assert jpeg_full_main_rows[0]["metric_status"] == "measured_from_real_attacked_image_watermark_rescore_formal_protocol"
    assert jpeg_full_main_rows[0]["attack_record_count"] == "1"
    assert attack_manifest["formal_real_attack_record_count"] == 1
    assert attack_manifest["formal_proxy_replacement_complete_count"] == 1
    assert attack_manifest["formal_proxy_replacement_incomplete_count"] == 0
    assert attack_manifest["image_attack_evidence_records_path"] == "outputs/image_attack_evidence/formal_attack_detection_records.jsonl"


@pytest.mark.quick
def test_attack_matrix_keeps_proxy_records_when_formal_coverage_is_partial(tmp_path: Path) -> None:
    """真实 formal records 只覆盖部分样本角色时, 不应整体移除同攻击配置 proxy 记录。"""
    rescue_dir = tmp_path / "outputs" / "geometric_rescue"
    calibration_dir = tmp_path / "outputs" / "threshold_calibration"
    conventional_dir = tmp_path / "outputs" / "conventional_geometric_attack_evaluation"
    rescue_dir.mkdir(parents=True)
    calibration_dir.mkdir(parents=True)
    conventional_dir.mkdir(parents=True)

    records_path = rescue_dir / "aligned_detection_records.jsonl"
    records_path.write_text(
        "".join(
            json_line(record)
            for record in (
                source_record("pos", "positive_source", 0.82, 0.86),
                source_record("clean", "clean_negative", 0.20, 0.22),
            )
        ),
        encoding="utf-8",
    )
    rescue_manifest_path = rescue_dir / "manifest.local.json"
    rescue_manifest_path.write_text(json.dumps({"artifact_id": "geometric_rescue_manifest"}), encoding="utf-8")
    thresholds_path = calibration_dir / "calibration_thresholds.json"
    thresholds_path.write_text(json.dumps({"threshold_value": 0.50, "target_fpr": 0.05}), encoding="utf-8")
    threshold_report_path = calibration_dir / "threshold_degeneracy_report.json"
    threshold_report_path.write_text(
        json.dumps(
            {
                "calibrated_content_threshold": 0.50,
                "target_fpr": 0.05,
                "rescue_margin_low": -0.05,
                "allowed_fail_reasons": ["geometry_suspected", "low_confidence"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calibration_manifest_path = calibration_dir / "manifest.local.json"
    calibration_manifest_path.write_text(json.dumps({"artifact_id": "threshold_calibration_manifest"}), encoding="utf-8")
    conventional_records_path = conventional_dir / "formal_attack_detection_records.jsonl"
    conventional_record = formal_real_attack_record(
        "jpeg_compression",
        0.43,
        False,
        attack_family="standard_distortion",
        resource_profile="full_main",
        requires_gpu=False,
        split="test",
        sample_role="positive_source",
    )
    conventional_records_path.write_text(json_line(conventional_record), encoding="utf-8")

    write_attack_matrix_outputs(
        root=tmp_path,
        rescue_records_path=records_path,
        rescue_manifest_path=rescue_manifest_path,
        calibration_thresholds_path=thresholds_path,
        threshold_report_path=threshold_report_path,
        calibration_manifest_path=calibration_manifest_path,
        conventional_geometric_records_path=conventional_records_path,
        max_source_records=None,
    )

    output_dir = tmp_path / "outputs" / "attack_matrix"
    attack_records = [
        json.loads(line)
        for line in (output_dir / "attack_detection_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    attack_manifest = json.loads((output_dir / "attack_manifest.json").read_text(encoding="utf-8"))
    jpeg_full_main_records = [
        record
        for record in attack_records
        if record["attack_family"] == "standard_distortion"
        and record["attack_name"] == "jpeg_compression"
        and record["resource_profile"] == "full_main"
    ]

    assert len(jpeg_full_main_records) == 3
    assert attack_manifest["formal_proxy_replacement_complete_count"] == 0
    assert attack_manifest["formal_proxy_replacement_incomplete_count"] == 1


@pytest.mark.quick
def test_attack_matrix_formal_claim_roles_replace_proxy_even_with_attacked_negative_proxy(tmp_path: Path) -> None:
    """formal 覆盖 fixed-FPR claim 角色时, attacked_negative proxy 不应导致 mixed 口径。"""
    rescue_dir = tmp_path / "outputs" / "geometric_rescue"
    calibration_dir = tmp_path / "outputs" / "threshold_calibration"
    conventional_dir = tmp_path / "outputs" / "conventional_geometric_attack_evaluation"
    rescue_dir.mkdir(parents=True)
    calibration_dir.mkdir(parents=True)
    conventional_dir.mkdir(parents=True)

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
    thresholds_path.write_text(json.dumps({"threshold_value": 0.50, "target_fpr": 0.05}), encoding="utf-8")
    threshold_report_path = calibration_dir / "threshold_degeneracy_report.json"
    threshold_report_path.write_text(
        json.dumps(
            {
                "calibrated_content_threshold": 0.50,
                "target_fpr": 0.05,
                "rescue_margin_low": -0.05,
                "allowed_fail_reasons": ["geometry_suspected", "low_confidence"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calibration_manifest_path = calibration_dir / "manifest.local.json"
    calibration_manifest_path.write_text(json.dumps({"artifact_id": "threshold_calibration_manifest"}), encoding="utf-8")
    conventional_records_path = conventional_dir / "formal_attack_detection_records.jsonl"
    formal_positive = formal_real_attack_record(
        "jpeg_compression",
        0.71,
        True,
        attack_family="standard_distortion",
        resource_profile="full_main",
        requires_gpu=False,
        split="test",
        sample_role="positive_source",
    )
    formal_clean = formal_real_attack_record(
        "jpeg_compression",
        0.11,
        False,
        attack_family="standard_distortion",
        resource_profile="full_main",
        requires_gpu=False,
        split="test",
        sample_role="clean_negative",
    )
    formal_clean["attack_record_id"] = "real_jpeg_compression_clean"
    formal_clean["attack_record_digest"] = "digest_jpeg_compression_clean"
    conventional_records_path.write_text(
        "".join(json_line(record) for record in (formal_positive, formal_clean)),
        encoding="utf-8",
    )

    write_attack_matrix_outputs(
        root=tmp_path,
        rescue_records_path=records_path,
        rescue_manifest_path=rescue_manifest_path,
        calibration_thresholds_path=thresholds_path,
        threshold_report_path=threshold_report_path,
        calibration_manifest_path=calibration_manifest_path,
        conventional_geometric_records_path=conventional_records_path,
        max_source_records=None,
    )

    output_dir = tmp_path / "outputs" / "attack_matrix"
    attack_records = [
        json.loads(line)
        for line in (output_dir / "attack_detection_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    family_rows = list(csv.DictReader((output_dir / "attack_family_metrics.csv").open(encoding="utf-8")))
    attack_manifest = json.loads((output_dir / "attack_manifest.json").read_text(encoding="utf-8"))
    jpeg_full_main_records = [
        record
        for record in attack_records
        if record["attack_family"] == "standard_distortion"
        and record["attack_name"] == "jpeg_compression"
        and record["resource_profile"] == "full_main"
    ]
    jpeg_full_main_rows = [
        row
        for row in family_rows
        if row["attack_family"] == "standard_distortion"
        and row["attack_name"] == "jpeg_compression"
        and row["resource_profile"] == "full_main"
    ]

    assert len(jpeg_full_main_records) == 2
    assert {record["sample_role"] for record in jpeg_full_main_records} == {"positive_source", "clean_negative"}
    assert jpeg_full_main_rows[0]["metric_status"] == "measured_from_real_attacked_image_watermark_rescore_formal_protocol"
    assert jpeg_full_main_rows[0]["attack_record_count"] == "2"
    assert jpeg_full_main_rows[0]["positive_count"] == "1"
    assert jpeg_full_main_rows[0]["negative_count"] == "1"
    assert attack_manifest["formal_proxy_replacement_complete_count"] == 1
    assert attack_manifest["formal_proxy_replacement_incomplete_count"] == 0
    assert attack_manifest["formal_proxy_replacement_requires_complete_split_role_coverage"] is False
