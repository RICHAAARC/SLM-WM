"""外部 baseline 公平对比协议的轻量功能测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.baselines import default_baseline_specs
from scripts.write_external_baseline_comparison_outputs import write_external_baseline_comparison_outputs


@pytest.mark.quick
def test_default_baseline_specs_keep_missing_results_unsupported() -> None:
    """默认外部 baseline 只登记协议 adapter, 不伪造外部复现结果。"""
    specs = default_baseline_specs()

    assert len(specs) == 8
    assert {spec.comparison_group for spec in specs} == {"primary", "supplemental"}
    assert all(spec.baseline_adapter_ready for spec in specs)
    assert not any(spec.baseline_reproduced_result_ready for spec in specs)
    assert not any(spec.baseline_imported_result_ready for spec in specs)
    assert {spec.unsupported_reason for spec in specs} == {"external_baseline_result_missing"}


def write_input_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """写出外部 baseline 对比脚本需要的最小上游输入。"""
    attack_dir = tmp_path / "outputs" / "attack_matrix"
    threshold_dir = tmp_path / "outputs" / "threshold_calibration"
    attack_dir.mkdir(parents=True)
    threshold_dir.mkdir(parents=True)
    attack_manifest_path = attack_dir / "attack_manifest.json"
    attack_family_metrics_path = attack_dir / "attack_family_metrics.csv"
    attack_matrix_manifest_path = attack_dir / "manifest.local.json"
    threshold_report_path = threshold_dir / "threshold_degeneracy_report.json"

    attack_manifest_path.write_text(
        json.dumps(
            {
                "attack_metrics_ready": True,
                "evaluation_boundary": {
                    "target_fpr": 0.05,
                    "calibrated_content_threshold": 0.50,
                    "rescue_margin_low": -0.05,
                    "allowed_fail_reasons": ["geometry_suspected", "low_confidence"],
                },
                "supports_paper_claim": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with attack_family_metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "attack_family",
                "attack_name",
                "resource_profile",
                "metric_status",
                "attack_record_count",
                "supported_record_count",
                "unsupported_record_count",
                "positive_count",
                "negative_count",
                "true_positive_rate",
                "false_positive_rate",
                "clean_false_positive_rate",
                "attacked_false_positive_rate",
                "quality_score_proxy_mean",
                "score_retention_mean",
                "supports_paper_claim",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "attack_family": "standard_distortion",
                "attack_name": "jpeg_compression",
                "resource_profile": "probe",
                "metric_status": "measured_from_local_proxy",
                "attack_record_count": 6,
                "supported_record_count": 6,
                "unsupported_record_count": 0,
                "positive_count": 2,
                "negative_count": 4,
                "true_positive_rate": 0.5,
                "false_positive_rate": 0.25,
                "clean_false_positive_rate": 0.0,
                "attacked_false_positive_rate": 0.5,
                "quality_score_proxy_mean": 0.9,
                "score_retention_mean": 0.8,
                "supports_paper_claim": False,
            }
        )
        writer.writerow(
            {
                "attack_family": "regeneration_attack",
                "attack_name": "img2img_regeneration",
                "resource_profile": "full_extra",
                "metric_status": "unsupported",
                "attack_record_count": 6,
                "supported_record_count": 0,
                "unsupported_record_count": 6,
                "positive_count": 0,
                "negative_count": 0,
                "true_positive_rate": 0.0,
                "false_positive_rate": 0.0,
                "clean_false_positive_rate": 0.0,
                "attacked_false_positive_rate": 0.0,
                "quality_score_proxy_mean": 0.0,
                "score_retention_mean": 0.0,
                "supports_paper_claim": False,
            }
        )
    attack_matrix_manifest_path.write_text(
        json.dumps({"artifact_id": "attack_matrix_manifest", "config_digest": "digest"}, ensure_ascii=False),
        encoding="utf-8",
    )
    threshold_report_path.write_text(
        json.dumps({"target_fpr": 0.05, "threshold_degenerate": False, "supports_paper_claim": False}, ensure_ascii=False),
        encoding="utf-8",
    )
    return attack_manifest_path, attack_family_metrics_path, attack_matrix_manifest_path, threshold_report_path


@pytest.mark.quick
def test_external_baseline_outputs_are_rebuildable_and_claim_safe(tmp_path: Path) -> None:
    """外部 baseline 对比产物应由上游攻击矩阵与 baseline spec 重建, 且不支持论文主张。"""
    attack_manifest_path, attack_family_metrics_path, attack_matrix_manifest_path, threshold_report_path = write_input_artifacts(tmp_path)

    manifest = write_external_baseline_comparison_outputs(
        root=tmp_path,
        attack_manifest_path=attack_manifest_path,
        attack_family_metrics_path=attack_family_metrics_path,
        attack_matrix_manifest_path=attack_matrix_manifest_path,
        threshold_report_path=threshold_report_path,
    )
    output_dir = tmp_path / "outputs" / "external_baseline_comparison"
    observations = [
        json.loads(line)
        for line in (output_dir / "baseline_observations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline_rows = list(csv.DictReader((output_dir / "baseline_metrics.csv").open(encoding="utf-8")))
    comparison_rows = list(csv.DictReader((output_dir / "baseline_comparison_table.csv").open(encoding="utf-8")))
    runtime_report = json.loads((output_dir / "baseline_runtime_report.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "external_baseline_comparison_manifest"
    assert len(observations) == len(default_baseline_specs()) * 2
    assert len(baseline_rows) == len(default_baseline_specs())
    assert runtime_report["comparison_protocol_ready"] is True
    assert runtime_report["baseline_results_ready"] is False
    assert {row["metric_status"] for row in baseline_rows} == {"unsupported"}
    assert any(row["method_id"] == "slm_wm_current" for row in comparison_rows)
    assert all(row["supports_paper_claim"] == "False" for row in comparison_rows)
